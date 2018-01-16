/* Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "flusher.hpp"

struct flusher_qp_info * flusher_qp=NULL;

//-------------------------------- STATIC ------------------------------------
static int * flsign_h;
static gds_flusher_buf flread_d;
static gds_flusher_buf flack_d;
static int flusher_value=0;
static pthread_t flusher_thread;
static int gds_flusher_service = -1;
static int gds_gpu_has_flusher=-1;
static int local_gpu_id=0;

static const char * flusher_int_to_str(int flusher_int)
{
    if(flusher_int == GDS_FLUSHER_TYPE_CPU)
        return "CPU Thread";
    else if(flusher_int == GDS_FLUSHER_TYPE_NIC)
        return "NIC RDMA PUT";
    else
        return "Unknown";
}

static inline int gds_flusher_service_active() {
    if(gds_flusher_service == GDS_FLUSHER_TYPE_CPU || gds_flusher_service == GDS_FLUSHER_TYPE_NIC)
        return 1;
    else
        return 0;
}
#define CHECK_FLUSHER_SERVICE()                                                 \
    if(!gds_flusher_service_active())                                           \
    {                                                                           \
        gds_dbg("Flusher service not active (%d)\n", gds_flusher_service);      \
        goto out;                                                               \
    }

#define ROUND_TO(V,PS) ((((V) + (PS) - 1)/(PS)) * (PS))

static int gds_flusher_pin_buffer(gds_flusher_buf * fl_mem, size_t req_size, int type_mem)
{
    int ret=0;
    CUcontext gpu_ctx;
    CUdevice gpu_device;
    size_t size = ROUND_TO(req_size, GDS_GPU_PAGE_SIZE);

    gds_dbg("GPU%u: malloc req_size=%zu size=%zu\n", local_gpu_id, req_size, size);

    if (!fl_mem) {
        gds_err("invalid params\n");
        return EINVAL;
    }

    // NOTE: gpu_id's primary context is assumed to be the right one
    // breaks horribly with multiple contexts
    int num_gpus;
    do {
        CUresult err = cuDeviceGetCount(&num_gpus);
        if (CUDA_SUCCESS == err) {
                break;
        } else if (CUDA_ERROR_NOT_INITIALIZED == err) {
                gds_err("CUDA error %d in cuDeviceGetCount, calling cuInit\n", err);
                CUCHECK(cuInit(0));
                // try again
                continue;
        } else {
                gds_err("CUDA error %d in cuDeviceGetCount, returning EIO\n", err);
                return EIO;
        }
    } while(0);
    gds_dbg("num_gpus=%d\n", num_gpus);
    if (local_gpu_id >= num_gpus) {
            gds_err("invalid num_GPUs=%d while requesting GPU id %d\n", num_gpus, local_gpu_id);
            return EINVAL;
    }

    CUCHECK(cuDeviceGet(&gpu_device, local_gpu_id));
    gds_dbg("local_gpu_id=%d gpu_device=%d\n", local_gpu_id, gpu_device);
    // TODO: check for existing context before switching to the interop one
    CUCHECK(cuDevicePrimaryCtxRetain(&gpu_ctx, gpu_device));
    CUCHECK(cuCtxPushCurrent(gpu_ctx));
    assert(gpu_ctx != NULL);

    gds_mem_desc_t *desc = (gds_mem_desc_t *)calloc(1, sizeof(gds_mem_desc_t));
    if (!desc) {
        gds_err("error while allocating mem desc\n");
        ret = ENOMEM;
        goto out;
    }

    ret = gds_alloc_mapped_memory(desc, size, type_mem);
    if (ret) {
        gds_err("error %d while allocating mapped GPU buffers\n", ret);
        goto out;
    }

    fl_mem->buf_h = desc->h_ptr;
    fl_mem->buf_d = desc->d_ptr;
    fl_mem->desc = desc;

    out:
    if (ret)
        free(desc); // desc can be NULL

    CUCHECK(cuCtxPopCurrent(NULL));
    CUCHECK(cuDevicePrimaryCtxRelease(gpu_device));

    return ret;
}

static int gds_flusher_free_pinned_buffer(gds_flusher_buf * fl_mem)
{
    int ret = 0;
    CUcontext gpu_ctx;
    CUdevice gpu_device;

    gds_dbg("GPU%u: mfree\n", local_gpu_id);

    if (!fl_mem->desc) {
        gds_err("invalid handle\n");
        return EINVAL;
    }

    if (!fl_mem->buf_h) {
        gds_err("invalid host_addr\n");
        return EINVAL;
    }

    // NOTE: gpu_id's primary context is assumed to be the right one
    // breaks horribly with multiple contexts

    CUCHECK(cuDeviceGet(&gpu_device, local_gpu_id));
    CUCHECK(cuDevicePrimaryCtxRetain(&gpu_ctx, gpu_device));
    CUCHECK(cuCtxPushCurrent(gpu_ctx));
    assert(gpu_ctx != NULL);

    gds_mem_desc_t *desc = (gds_mem_desc_t *)fl_mem->desc;
    ret = gds_free_mapped_memory(desc);
    if (ret) {
        gds_err("error %d while freeing mapped GPU buffers\n", ret);
    }
    free(desc);

    CUCHECK(cuCtxPopCurrent(NULL));
    CUCHECK(cuDevicePrimaryCtxRelease(gpu_device));

    return ret;
}

static int gds_flusher_create_qp()
{
    struct ibv_port_attr port_attr;
    int qp_flags=0;
    int attr_flags=0;
    int ret=0;
    struct ibv_ah_attr ib_ah_attr;
    struct ibv_qp_attr attr;
    gds_qp_init_attr_t qp_init_attr;

    qp_flags |= GDS_CREATE_QP_FLUSHER;
    //qp_flags |= GDS_CREATE_QP_GPU_INVALIDATE_RX_CQ;
    qp_flags |= GDS_CREATE_QP_GPU_INVALIDATE_TX_CQ;

    // --------------------------------------------

    memset(&qp_init_attr, 0, sizeof(gds_qp_init_attr_t));

    qp_init_attr.send_cq                = 0;
    qp_init_attr.recv_cq                = 0;
    qp_init_attr.cap.max_send_wr        = 512;
    qp_init_attr.cap.max_recv_wr        = 512;
    qp_init_attr.cap.max_send_sge       = 1;
    qp_init_attr.cap.max_recv_sge       = 1;
    qp_init_attr.cap.max_inline_data    = 0; //The flusher must not inline data!!
    qp_init_attr.qp_type                = IBV_QPT_RC; //UD & RDMA_WRITE are not compatible!

    flusher_qp->loopback_qp = gds_create_qp(flusher_qp->pd, flusher_qp->context, &qp_init_attr, flusher_qp->gpu_id, qp_flags);
    if (!flusher_qp->loopback_qp) {
        gds_err("gds_set_loopback_qp returned NULL\n");
        ret=EINVAL;
        goto out;
    }

    // --------------------------------------------

    memset(&attr, 0, sizeof(struct ibv_qp_attr));
    attr.qp_state        = IBV_QPS_INIT;
    attr.pkey_index      = 0;
    attr.port_num        = flusher_qp->ib_port;
    attr.qkey            = GDS_FLUSHER_QKEY;
    attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;
    
    attr_flags           = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;

    if (ibv_modify_qp(flusher_qp->loopback_qp->qp, &attr, attr_flags)) {
        gds_err("Failed to modify QP to INIT\n");
        goto clean_qp;
    }
    
    if(ibv_query_port(flusher_qp->context, flusher_qp->ib_port, &port_attr))
    {
        fprintf(stderr, "Failed to modify QP to INIT\n");
        goto clean_qp;
    }

    flusher_qp->lid=port_attr.lid;
    flusher_qp->qpn=flusher_qp->loopback_qp->qp->qp_num;
    flusher_qp->psn=0; //lrand48() & 0xffffff;

    // --------------------------------------------

    memset(&attr, 0, sizeof(struct ibv_qp_attr));
    attr.qp_state               = IBV_QPS_RTR;
    attr.path_mtu               = port_attr.active_mtu;
    attr.dest_qp_num            = flusher_qp->qpn;
    attr.rq_psn                 = flusher_qp->psn;
    attr.ah_attr.dlid           = flusher_qp->lid;
    attr.max_dest_rd_atomic     = 1;
    attr.min_rnr_timer          = 12;
    attr.ah_attr.is_global      = 0;
    attr.ah_attr.sl             = 0;
    attr.ah_attr.src_path_bits  = 0;
    attr.ah_attr.port_num       = flusher_qp->ib_port;
    attr_flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MIN_RNR_TIMER | IBV_QP_MAX_DEST_RD_ATOMIC;    

    if (ibv_modify_qp(flusher_qp->loopback_qp->qp, &attr, attr_flags)) {
        gds_err("Failed to modify QP to RTR\n");
        goto clean_qp;
    }
    
    // --------------------------------------------

    memset(&attr, 0, sizeof(struct ibv_qp_attr));
    attr.qp_state       = IBV_QPS_RTS;
    attr.sq_psn         = 0;
    attr.timeout        = 20;
    attr.retry_cnt      = 7;
    attr.rnr_retry      = 7;
    attr.max_rd_atomic  = 1;
    attr_flags = IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC;

    if (ibv_modify_qp(flusher_qp->loopback_qp->qp, &attr, attr_flags)) {
        gds_err("Failed to modify QP to RTS\n");
        goto clean_qp;
    }

    out:
    return ret;

    clean_qp:
        gds_destroy_qp(flusher_qp->loopback_qp);
        return EINVAL;

}

static int gds_flusher_prepare_put_dmem(gds_send_request_t * send_info, struct gds_flusher_buf * src_fbuf, struct gds_flusher_buf * dst_fbuf)
{
    int ret=0;
    struct ibv_sge list;
    gds_send_wr p_ewr;
    gds_send_wr *bad_ewr;
    
    assert(send_info);
    assert(src_fbuf);
    assert(dst_fbuf);

    //send read word
    memset(&list, 0, sizeof(struct ibv_sge));
    list.addr   = (uintptr_t) (src_fbuf->buf_d);
    list.length = src_fbuf->size;
    list.lkey   = src_fbuf->mr->lkey;

    memset(&p_ewr, 0, sizeof(gds_send_wr));
    p_ewr.next                  = NULL;
    p_ewr.exp_send_flags        = 0; //IBV_EXP_SEND_SIGNALED;
    p_ewr.exp_opcode            = IBV_EXP_WR_RDMA_WRITE;
    p_ewr.wr_id                 = 1;
    p_ewr.num_sge               = 1;
    p_ewr.sg_list               = &list;
    p_ewr.wr.rdma.remote_addr   = dst_fbuf->buf_d;
    p_ewr.wr.rdma.rkey          = dst_fbuf->mr->rkey;

    ret = gds_prepare_send(flusher_qp->loopback_qp, &p_ewr, &bad_ewr, send_info);
    if (ret) {
        gds_err("error %d in gds_prepare_send\n", ret);
        goto out;
    }

    out:
    return ret;
}
//------------------------------- COMMON ------------------------------------
int gds_gpu_flusher_env()
{
    if (-1 == gds_gpu_has_flusher) {
        const char *env = getenv("GDS_GPU_HAS_FLUSHER");
        if (env)
            gds_gpu_has_flusher = !!atoi(env);
        else
            gds_gpu_has_flusher = 0;

        gds_warn("GDS_GPU_HAS_FLUSHER=%d\n", gds_gpu_has_flusher);
    }
    return gds_gpu_has_flusher;
}

int gds_service_flusher_env()
{
    if (-1 == gds_flusher_service) {
        const char *env = getenv("GDS_FLUSHER_SERVICE");
        if (env)
        {
            gds_flusher_service = atoi(env);
            if(gds_flusher_service != GDS_FLUSHER_TYPE_CPU && gds_flusher_service != GDS_FLUSHER_TYPE_NIC)
            {
                //gds_err("Erroneous value GDS_FLUSHER_SERVICE=%d. Service not activated\n", gds_flusher_service);
                gds_flusher_service=0;
            }
        }
        else
            gds_flusher_service = 0;

        gds_warn("GDS_FLUSHER_SERVICE=%d\n", gds_flusher_service);
    }
    return gds_flusher_service;
}

int gds_flusher_check_envs()
{
    gds_gpu_flusher_env();

    if(gds_gpu_has_flusher == 1)
    {
        gds_flusher_service=0;
        gds_warn("Using GPU native flusher\n");
    }
    else
    {
        gds_service_flusher_env();
        if(gds_flusher_service == 0)
            gds_warn("No flusher service nor GPU native flusher\n");
        else
            gds_warn("Using flusher service '%s' (%d)\n", flusher_int_to_str(gds_flusher_service), gds_flusher_service);
    }
}

int gds_flusher_init(struct ibv_pd *pd, struct ibv_context *context, int gpu_id)
{
    int ret = 0;
    unsigned int flag = 1;

    gds_flusher_check_envs();
    CHECK_FLUSHER_SERVICE();

    local_gpu_id=gpu_id;
    
    flread_d.size=1*sizeof(int);
    flack_d.size=1*sizeof(int);

    gds_dbg("gds_flusher_service=%d\n", gds_flusher_service);

    if(gds_flusher_service == GDS_FLUSHER_TYPE_CPU)
    {
        // ------------------ READ WORD ------------------
        ret = gds_flusher_pin_buffer(&flread_d, flread_d.size, GDS_MEMORY_GPU);
        if (ret) {
            gds_err("error %d while allocating mapped GPU buffers\n", ret);
            goto out;
        }
        CUCHECK(cuMemsetD32(flread_d.buf_d, 0, 1));
        gds_dbg("ReadWord pinned. size: %d buf_d=%p buf_h=%p\n", flread_d.size, flread_d.buf_d, flread_d.buf_h);

        // ------------------ ACK WORD ------------------
        ret = gds_flusher_pin_buffer(&flack_d, flack_d.size, GDS_MEMORY_GPU);
        if (ret) {
            gds_err("error %d while allocating mapped GPU buffers\n", ret);
            goto out;
        }
        CUCHECK(cuMemsetD32(flack_d.buf_d, 0, 1));
        gds_dbg("Ackword pinned. size: %d buf_d=%p buf_h=%p\n", flack_d.size, flack_d.buf_d, flack_d.buf_h);

        // ------------------ SIGNAL WORD ------------------
        CUCHECK(cuMemAllocHost((void**)&flsign_h, 1*sizeof(int)));
        memset(flsign_h, 0, sizeof(int));
        gds_dbg("SignalWord on Host Mem %p\n", flsign_h);

        // ------------------ THREAD ------------------
        gds_flusher_start_thread(&flusher_thread);
    }
    else if(gds_flusher_service == GDS_FLUSHER_TYPE_NIC)
    { 
        // ------------------ READ WORD ------------------
        ret = gds_flusher_pin_buffer(&flread_d, flread_d.size, GDS_MEMORY_GPU);
        if (ret) {
            gds_err("error %d while allocating mapped GPU buffers\n", ret);
            goto out;
        }
        CUCHECK(cuMemsetD32(flread_d.buf_d, 0, 1));
        gds_dbg("ReadWord pinned. size: %d buf_d=%p buf_h=%p\n", flread_d.size, flread_d.buf_d, flread_d.buf_h);

        // ------------------ ACK WORD ------------------
        ret = gds_flusher_pin_buffer(&flack_d, flack_d.size, GDS_MEMORY_GPU);
        if (ret) {
            gds_err("error %d while allocating mapped GPU buffers\n", ret);
            goto out;
        }
        CUCHECK(cuMemsetD32(flack_d.buf_d, 0, 1));
        gds_dbg("Ackword pinned. size: %d buf_d=%p buf_h=%p\n", flack_d.size, flack_d.buf_d, flack_d.buf_h);

        if(flusher_qp == NULL)
        {
            flusher_qp = (struct flusher_qp_info *) calloc (1, sizeof(struct flusher_qp_info));
            if(!flusher_qp) {
                gds_err("error %d while allocating mapped GPU buffers\n", ret);
                goto out;
            }

            flusher_qp->gpu_id = local_gpu_id;
            flusher_qp->ib_port=GDS_FLUSHER_PORT;
            flusher_qp->context=context;
            flusher_qp->pd=pd;
            
            ret = gds_flusher_create_qp();
            if (ret) {
                gds_err("error %d gds_flusher_create_qp\n", ret);
                goto out;
            }

            flread_d.mr = ibv_reg_mr(flusher_qp->pd, (void*)flread_d.buf_d, flread_d.size, IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE);
            if (!flread_d.mr) {
                gds_err("Couldn't register MR\n");
                ret=EINVAL;
                goto out;
            }
            
            gds_dbg("flread_d ibv_reg_mr addr:%p size:%zu flags=0x%08x, reg=%p lkey=%x, rkey=%x\n", 
                flread_d.buf_d, flread_d.size, IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE, flread_d.mr, flread_d.mr->lkey, flread_d.mr->rkey);

            flack_d.mr = ibv_reg_mr(flusher_qp->pd, (void*)flack_d.buf_d, flack_d.size, IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE);
            if (!flread_d.mr) {
                gds_err("Couldn't register MR\n");
                ret=EINVAL;
                goto out;
            }
            
            gds_dbg("flack_d ibv_reg_mr addr:%p size:%zu flags=0x%08x, reg=%p lkey=%x, rkey=%x\n", 
                flack_d.buf_d, flack_d.size, IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE, flack_d.mr, flack_d.mr->lkey, flack_d.mr->rkey);

        }
    }

    if(!ret)
        gds_warn("Flusher initialized\n");

    out:
    return ret;
}

int gds_flusher_destroy()
{
    int ret = 0;
    unsigned int flag = 1;

    CHECK_FLUSHER_SERVICE();
    
    gds_dbg("gds_flusher_service=%d\n", gds_flusher_service);

    if(gds_flusher_service == GDS_FLUSHER_TYPE_CPU)
    {
        ret=gds_flusher_stop_thread(flusher_thread);
        if(ret)
        {
            gds_err("gds_fill_poke error %d\n", ret);
            goto out;
        }

        ret=gds_flusher_free_pinned_buffer(&(flread_d));
        if (ret) {
            gds_err("error %d while freeing mapped GPU buffers\n", ret);
            goto out;
        }

        ret=gds_flusher_free_pinned_buffer(&(flack_d));
        if (ret) {
            gds_err("error %d while freeing mapped GPU buffers\n", ret);
            goto out;
        }

        CUCHECK(cuMemFreeHost(flsign_h));

        gds_dbg("Device words unpinned\n");
    }
    else if(gds_flusher_service == GDS_FLUSHER_TYPE_NIC)
    {
        if(!flusher_qp) {
            gds_err("error !flusher_qp\n");
            ret=EINVAL;
            goto out;
        }

        if(!(flusher_qp->loopback_qp)) {
            gds_err("error !loopback_qp\n");
            ret=EINVAL;
            goto out;
        }
        
        ret = ibv_destroy_qp(flusher_qp->loopback_qp->qp);
        if (ret) {
            gds_err("error %d in destroy_qp\n", ret);
            goto out;
        }

        assert(flusher_qp->loopback_qp->send_cq.cq);
        ret = ibv_destroy_cq(flusher_qp->loopback_qp->send_cq.cq);
        if (ret) {
            gds_err("error %d in destroy_cq send_cq\n", ret);
            goto out;
        }
        //send_cq == recv_cq

        if(flread_d.mr) {
            ret = ibv_dereg_mr(flread_d.mr);
            if (ret) {
                gds_err("error %d in ibv_dereg_mr\n", ret);
                goto out;
            }            
        }

        if(flack_d.mr) {
            ret = ibv_dereg_mr(flack_d.mr);
            if (ret) {
                gds_err("error %d in ibv_dereg_mr\n", ret);
                goto out;
            }            
        }

        free(flusher_qp->loopback_qp);
        free(flusher_qp);
        flusher_qp=NULL;

        gds_dbg("Flusher QP destroyed\n");

    }
    
    if(!ret) gds_warn("Flusher destroyed\n");

    out:
    return ret;
}

int gds_flusher_count_op()
{
    if(gds_flusher_service == GDS_FLUSHER_TYPE_CPU)
        return GDS_FLUSHER_OP_CPU;
    
    if(gds_flusher_service == GDS_FLUSHER_TYPE_NIC)
        return GDS_FLUSHER_OP_NIC;

    return 0;
}

void gds_flusher_set_flag(int * flags)
{
    //Enable GPU flusher if GPU has internal flusher (flusher service ignored)
    if(gds_gpu_has_flusher == 1)
        (*flags) |= GDS_WAIT_POST_FLUSH;

    gds_dbg("flags=%x\n", (*flags));
}

//Not actually used for now!
int gds_flusher_post_stream(CUstream stream)
{
    gds_descriptor_t desc[3];
    gds_send_request_t send_info;
    int k = 0, ret = 0;

    CHECK_FLUSHER_SERVICE();

    if(gds_flusher_service == GDS_FLUSHER_TYPE_CPU)
    {
        //write32 signal
        desc[k].tag             = GDS_TAG_WRITE_VALUE32;
        desc[k].write32.ptr     = (uint32_t *) flsign_h;
        desc[k].write32.value   = flusher_value+1;
        desc[k].write32.flags   = GDS_MEMORY_HOST;
        ++k;

        //wait32 ackword
        desc[k].tag = GDS_TAG_WAIT_VALUE32;
        desc[k].wait32.ptr          = (uint32_t *)flack_d.buf_d;
        desc[k].wait32.value        = flusher_value+1;
        desc[k].wait32.cond_flags   = GDS_WAIT_COND_EQ;
        desc[k].wait32.flags        = GDS_MEMORY_GPU;
        ++k;
    }
    else
    {
        //flusher NIC
        //write read word
        desc[k].tag             = GDS_TAG_WRITE_VALUE32;
        desc[k].write32.ptr     = (uint32_t *) flread_d.buf_d;
        desc[k].write32.value   = flusher_value+1;
        desc[k].write32.flags   = GDS_MEMORY_GPU;
        ++k;
        //write order respected?

       
        ret=gds_flusher_prepare_put_dmem(&send_info, &flread_d, &flack_d);
        if(ret)
        {
            gds_err("gds_flusher_prepare_put_dmem, err: %d\n", ret);
            goto out;
        }

        desc[k].tag = GDS_TAG_SEND;
        desc[k].send = &send_info;
        ++k;

        //wait32 ackword
        desc[k].tag                 = GDS_TAG_WAIT_VALUE32;
        desc[k].wait32.ptr          = (uint32_t *)flack_d.buf_d;
        desc[k].wait32.value        = flusher_value+1;
        desc[k].wait32.cond_flags   = GDS_WAIT_COND_EQ;
        desc[k].wait32.flags        = GDS_MEMORY_GPU;
        ++k;
    }

    ret = gds_stream_post_descriptors(stream, k, desc, 0);
    if (ret)
    {
        gds_err("gds_stream_post_descriptors, err: %d\n", ret);
        return EINVAL;
    }
    //not multithread safe!
    flusher_value++;

    out:
        return ret;
}

int gds_flusher_add_ops(CUstreamBatchMemOpParams *params, int &idx)
{
    int ret=0, tmp_idx=0;
    gds_send_request_t send_info;

    CHECK_FLUSHER_SERVICE();
    
    if(gds_flusher_service == GDS_FLUSHER_TYPE_CPU)
    {
        gds_dbg("gds_fill_poke flsign_h=%p, flusher_value+1=%d, idx=%d\n", flsign_h, flusher_value+1, idx);
        ret = gds_fill_poke(params+idx, (uint32_t*)flsign_h, flusher_value+1, GDS_MEMORY_GPU);
        if(ret)
        {
            gds_err("gds_fill_poke error %d\n", ret);
            goto out;
        }
        ++idx;

        gds_dbg("gds_fill_poll flack_d.buf_d=%p, flusher_value+1=%d, idx=%d\n", flack_d.buf_d, flusher_value+1, idx);
        ret = gds_fill_poll(params+idx, (uint32_t*)flack_d.buf_d, flusher_value+1, GDS_WAIT_COND_EQ, GDS_MEMORY_GPU);
        if(ret)
        {
            gds_err("gds_fill_poll error %d\n", ret);
            goto out;
        }
        ++idx;
    }
    else if(gds_flusher_service == GDS_FLUSHER_TYPE_NIC)
    {
        ret = gds_fill_poke(params+idx, (uint32_t*)flread_d.buf_d, flusher_value+1, GDS_MEMORY_GPU);
        if(ret)
        {
            gds_err("gds_fill_poke error %d\n", ret);
            goto out;
        }
        gds_dbg("gds_fill_poke done flread_d.buf_d=%p, flusher_value+1=%d, idx=%d\n", flread_d.buf_d, flusher_value+1, idx);

        ++idx;
    
        ret=gds_flusher_prepare_put_dmem(&send_info, &flread_d, &flack_d);
        if(ret)
        {
            gds_err("gds_flusher_prepare_put_dmem, err: %d\n", ret);
            goto out;
        }

        ret = gds_post_ops(send_info.commit.entries, send_info.commit.storage, params, idx);
        if (ret) {
            gds_err("error %d in gds_post_ops\n", ret);
            goto out;
        }    
        gds_dbg("gds_post_ops send_info.commit.entries=%p, flusher_value+1=%d, idx=%d\n", send_info.commit.entries, flusher_value+1, idx);
        
        ret = gds_fill_poll(params+idx, (uint32_t*)flack_d.buf_d, flusher_value+1, GDS_WAIT_COND_EQ, GDS_MEMORY_GPU);
        if(ret)
        {
            gds_err("gds_fill_poll error %d\n", ret);
            goto out;
        }
        gds_dbg("gds_fill_poll flack_d.buf_d=%p, flusher_value+1=%d, idx=%d\n", flack_d.buf_d, flusher_value+1, idx);
        ++idx;
    
    }

        gds_dbg("Final idx=%d\n", idx);

    ++flusher_value;

    out:
        return ret;

}

int gds_flusher_start_thread(pthread_t *fThread) //,  threadFunc, void *arg)
{
    int ret=0;

    CHECK_FLUSHER_SERVICE();

    if(fThread == NULL)
    {
        gds_warn("error input");
        return EINVAL;
    }
    
    gds_dbg("Create Thread\n");
        
    if(pthread_create(fThread, NULL, gds_flusher_func_thread, NULL) != 0) {
        gds_err("pthread_create, err: %d\n", ret);
        return EINVAL;
    }

    out:
    return ret;
}

int gds_flusher_stop_thread(pthread_t fThread)
{
    int ret=0;
    void * tret;

    CHECK_FLUSHER_SERVICE();

    ret=pthread_cancel(fThread);
    if(ret)
    {
        gds_err("pthread_cancel, ret: %d\n", ret);
        return EINVAL;
    }

    ret=pthread_join(fThread, &tret);
    if(ret)
    {
        gds_err("pthread_join, ret: %d, thread ret: %ld\n", ret, (long)tret);
        return EINVAL;
    }

    out:
    return ret;
}

#if 0
#include <sys/time.h>

#define TIMER_DEF(n)     struct timeval temp_1_##n={0,0}, temp_2_##n={0,0}
#define TIMER_START(n)   gettimeofday(&temp_1_##n, (struct timezone*)0)
#define TIMER_STOP(n)    gettimeofday(&temp_2_##n, (struct timezone*)0)
#define TIMER_ELAPSED(n) ((temp_2_##n.tv_sec-temp_1_##n.tv_sec)*1.e6+(temp_2_##n.tv_usec-temp_1_##n.tv_usec))
#endif

//-------------------------------- NVTX -----------------------------------------
#include "nvToolsExt.h"
const uint32_t colors[] = { 0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff };
const int num_colors = sizeof(colors)/sizeof(uint32_t);

#define PUSH_RANGE(name,cid) { \
    int color_id = cid; \
    color_id = color_id%num_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxRangePushEx(&eventAttrib); \
}
#define POP_RANGE nvtxRangePop();
//-------------------------------------------------------------------------------

typedef int64_t gds_us_t;
static inline gds_us_t gds_get_time_us()
{
        struct timespec ts;
        int ret = clock_gettime(CLOCK_MONOTONIC, &ts);
        if (ret) {
                fprintf(stderr, "error in gettime %d/%s\n", errno, strerror(errno));
                exit(EXIT_FAILURE);
        }
        return (gds_us_t)ts.tv_sec * 1000 * 1000 + (gds_us_t)ts.tv_nsec / 1000;
}

void * gds_flusher_func_thread(void * arg)
{
    int last_value=0;
    int tmp=0;
    gds_us_t start, end;
    gds_us_t delta1;
    gds_us_t delta2;
    gds_us_t delta3;
    gds_us_t delta4;
    int local_rank = atoi(getenv("OMPI_COMM_WORLD_LOCAL_RANK"));

    //Should be the default setting
    //pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, &old_state);
 
    while(1) //avoid mutex for kill condition
    {
        gds_dbg("Thread waiting on flsign_h=%p, current last_value=%d\n", flsign_h, last_value);
        
        PUSH_RANGE("THREAD", 1);
        //start = gds_get_time_us();
        while( (ACCESS_ONCE( *flsign_h )) <= last_value ) { pthread_testcancel(); }
        //end = gds_get_time_us();
        //delta1 = end - start;

        //start = gds_get_time_us();
        last_value = ACCESS_ONCE( *flsign_h );
        rmb();
        //end = gds_get_time_us();
        //delta2 = end - start;
        
        gds_dbg("Thread last_value=%d\n", last_value);

        //start = gds_get_time_us();
        tmp = ACCESS_ONCE ( *((int*)flread_d.buf_h) ); //Should be always 0!
        wmb();
        //end = gds_get_time_us();
        //delta3 = end - start;

        //start = gds_get_time_us();   
        //gds_dbg("Thread tmp=%d after wmb\n", tmp);
        ACCESS_ONCE ( *((int*)flack_d.buf_h) ) = last_value;
        wmb();
        //end = gds_get_time_us();
        //delta4 = end - start;

        POP_RANGE;
        //if(local_rank == 0)
        //    gds_warn("thread --> last_value: %d, while polling time: %.2f us, sign read time: %.2f us, read_d time: %.2f us, write ack time: %.2f us\n", last_value, (double)delta1, (double)delta2,  (double)delta3, (double)delta4);
    }

    return NULL;
}

//-----------------------------------------------------------------------------
/*
 * Local variables:
 *  c-indent-level: 8
 *  c-basic-offset: 8
 *  tab-width: 8
 *  indent-tabs-mode: nil
 * End:
 */
