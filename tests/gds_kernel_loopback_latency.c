/*
 * GPUDirect Async loopback latency benchmark
 * 
 *
 * based on OFED libibverbs ud_pingpong test.
 */
/*
 * Copyright (c) 2005 Topspin Communications.  All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * OpenIB.org BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#if HAVE_CONFIG_H
#  include <config.h>
#endif /* HAVE_CONFIG_H */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <netdb.h>
#include <malloc.h>
#include <getopt.h>
#include <arpa/inet.h>
#include <time.h>
#include <assert.h>
#include <errno.h>

#include <cuda.h>
#include <cudaProfiler.h>
#include <cuda_runtime_api.h>
#include <gdsync.h>

#include "pingpong.h"
#include "gpu.h"
#include "test_utils.h"

//-----------------------------------------------------------------------------

struct prof prof;
int prof_idx = 0;

//-----------------------------------------------------------------------------


#define dbg(FMT, ARGS...)  gpu_dbg(FMT, ## ARGS)

#define min(A,B) ((A)<(B)?(A):(B))

#define USE_CUDA_PROFILER 1

enum {
        PINGPONG_RECV_WRID = 1,
        PINGPONG_SEND_WRID = 2,
};

static int page_size;
int stream_cb_error = 0;

struct pingpong_context {
        struct ibv_context	*context;
        struct ibv_comp_channel *channel;
        struct ibv_pd		*pd;
        struct ibv_mr		*mr;
        struct ibv_cq		*tx_cq;
        struct ibv_cq		*rx_cq;
        struct ibv_qp		*qp;
        struct gds_qp		*gds_qp;
        struct ibv_ah		*ah;
        void			*buf;
        char			*txbuf;
        char                    *rxbuf;
        char                    *rx_flag;
        int			 size;
        int                      calc_size;
        int			 rx_depth;
        int			 pending;
        struct ibv_port_attr     portinfo;
        int                      gpu_id;
        int                      kernel_duration;
        int                      peersync;
        int                      peersync_gpu_cq;
        int                      consume_rx_cqe;
        int                      gpumem;
        int                      use_desc_apis;
        int                      skip_kernel_launch;
};

static int my_rank = 0, comm_size = 1;

struct pingpong_dest {
        int lid;
        int qpn;
        int psn;
        union ibv_gid gid;
};

static int pp_connect_ctx(struct pingpong_context *ctx, int port, int my_psn,
                int sl, struct pingpong_dest *dest, int sgid_idx)
{
        struct ibv_ah_attr ah_attr = {
                .is_global     = 0,
                .dlid          = dest->lid,
                .sl            = sl,
                .src_path_bits = 0,
                .port_num      = port
        };
        struct ibv_qp_attr attr = {
                .qp_state		= IBV_QPS_RTR
        };

        if (ibv_modify_qp(ctx->qp, &attr, IBV_QP_STATE)) {
                fprintf(stderr, "Failed to modify QP to RTR\n");
                return 1;
        }

        attr.qp_state	    = IBV_QPS_RTS;
        attr.sq_psn	    = my_psn;

        if (ibv_modify_qp(ctx->qp, &attr,
                                IBV_QP_STATE              |
                                IBV_QP_SQ_PSN)) {
                fprintf(stderr, "Failed to modify QP to RTS\n");
                return 1;
        }

        if (dest->gid.global.interface_id) {
                ah_attr.is_global = 1;
                ah_attr.grh.hop_limit = 1;
                ah_attr.grh.dgid = dest->gid;
                ah_attr.grh.sgid_index = sgid_idx;
        }

        ctx->ah = ibv_create_ah(ctx->pd, &ah_attr);
        if (!ctx->ah) {
                fprintf(stderr, "Failed to create AH\n");
                return 1;
        }

        return 0;
}

static struct pingpong_dest *pp_client_exch_dest(const char *servername, int port,
                const struct pingpong_dest *my_dest)
{
        struct pingpong_dest *rem_dest = NULL;

        fprintf(stderr, "%04x:%06x:%06x:%s\n", my_dest->lid, my_dest->qpn,
                        my_dest->psn, (char *)&my_dest->gid);
        rem_dest = malloc(sizeof *rem_dest);
        if (!rem_dest)
                goto out;
        memcpy(rem_dest, my_dest, sizeof(struct pingpong_dest));
        fprintf(stderr, "%04x:%06x:%06x\n", rem_dest->lid, rem_dest->qpn,
                        rem_dest->psn);

out:
        return rem_dest;
}

static inline unsigned long align_to(unsigned long val, unsigned long pow2)
{
        return (val + pow2 - 1) & ~(pow2 - 1);
}

static struct pingpong_context *pp_init_ctx(struct ibv_device *ib_dev, int size, int calc_size,
                int rx_depth, int port,
                int use_event,
                int gpu_id,
                int peersync,
                int peersync_gpu_cq,
                int peersync_gpu_dbrec,
                int consume_rx_cqe,
                int sched_mode,
                int use_gpumem,
                int use_desc_apis,
                int skip_kernel_launch)
{
        struct pingpong_context *ctx;

        if (gpu_id >=0 && gpu_init(gpu_id, sched_mode)) {
                fprintf(stderr, "error in GPU init.\n");
                return NULL;
        }

        ctx = malloc(sizeof *ctx);
        if (!ctx)
                return NULL;

        ctx->size     = size;
        ctx->calc_size = calc_size;
        ctx->rx_depth = rx_depth;
        ctx->gpu_id   = gpu_id;
        ctx->gpumem   = use_gpumem;
        ctx->use_desc_apis = use_desc_apis;
        ctx->skip_kernel_launch = skip_kernel_launch;

        size_t alloc_size = 3 * align_to(size + 40, page_size);
        if (ctx->gpumem) {
                ctx->buf = gpu_malloc(page_size, alloc_size);
                printf("allocated GPU buffer address at %p\n", ctx->buf);
        } else {
                printf("allocating CPU memory buf\n");
                ctx->buf = memalign(page_size, alloc_size);
                printf("allocated CPU buffer address at %p\n", ctx->buf);
        }

        if (!ctx->buf) {
                fprintf(stderr, "Couldn't allocate work buf.\n");
                goto clean_ctx;
        }
        printf("ctx buf=%p\n", ctx->buf);
        ctx->rxbuf = (char*)ctx->buf;
        ctx->txbuf = (char*)ctx->buf + align_to(size + 40, page_size);

        ctx->rx_flag =  memalign(page_size, alloc_size);
        if (!ctx->rx_flag) {
                fprintf(stderr, "Couldn't allocate rx_flag buf\n");  
                goto clean_ctx;
        }

        ctx->kernel_duration = 0;
        ctx->peersync = peersync;
        ctx->peersync_gpu_cq = peersync_gpu_cq;
        ctx->consume_rx_cqe = consume_rx_cqe;

        // must be ZERO!!! for rx_flag to work...
        if (ctx->gpumem)
                gpu_memset(ctx->buf, 0, alloc_size);
        else
                memset(ctx->buf, 0, alloc_size);

        memset(ctx->rx_flag, 0, alloc_size);

        if (!ctx->skip_kernel_launch) {
                // pipe-cleaner
                gpu_launch_kernel_on_stream(ctx->calc_size, ctx->peersync, gpu_stream_server);
                gpu_launch_kernel_on_stream(ctx->calc_size, ctx->peersync, gpu_stream_server);
                gpu_launch_kernel_on_stream(ctx->calc_size, ctx->peersync, gpu_stream_server);
                // client stream is not really used
                gpu_launch_kernel_on_stream(ctx->calc_size, ctx->peersync, gpu_stream_client);
                gpu_launch_kernel_on_stream(ctx->calc_size, ctx->peersync, gpu_stream_client);
                gpu_launch_kernel_on_stream(ctx->calc_size, ctx->peersync, gpu_stream_client);
                CUCHECK(cuCtxSynchronize());
        }

        ctx->context = ibv_open_device(ib_dev);
        if (!ctx->context) {
                fprintf(stderr, "Couldn't get context for %s\n",
                                ibv_get_device_name(ib_dev));
                goto clean_buffer;
        }

        if (use_event) {
                ctx->channel = ibv_create_comp_channel(ctx->context);
                if (!ctx->channel) {
                        fprintf(stderr, "Couldn't create completion channel\n");
                        goto clean_device;
                }
        } else
                ctx->channel = NULL;

        ctx->pd = ibv_alloc_pd(ctx->context);
        if (!ctx->pd) {
                fprintf(stderr, "Couldn't allocate PD\n");
                goto clean_comp_channel;
        }

        ctx->mr = ibv_reg_mr(ctx->pd, ctx->buf, alloc_size, IBV_ACCESS_LOCAL_WRITE);
        if (!ctx->mr) {
                fprintf(stderr, "Couldn't register MR\n");
                goto clean_pd;
        }

        int gds_flags = 0;
        if (peersync_gpu_cq)
                gds_flags |= GDS_CREATE_QP_RX_CQ_ON_GPU;
        if (peersync_gpu_dbrec)
                gds_flags |= GDS_CREATE_QP_WQ_DBREC_ON_GPU;

        gds_qp_init_attr_t attr = {
                .send_cq = 0,
                .recv_cq = 0,
                .cap     = {
                        .max_send_wr  = rx_depth,
                        .max_recv_wr  = rx_depth,
                        .max_send_sge = 1,
                        .max_recv_sge = 1
                },
                .qp_type = IBV_QPT_UD,
        };

        //why?
        if (my_rank == 1) {
                printf("sleeping 2s\n");
                sleep(2);
        }
        ctx->gds_qp = gds_create_qp(ctx->pd, ctx->context, &attr, gpu_id, gds_flags);

        if (!ctx->gds_qp) {
                fprintf(stderr, "Couldn't create QP (%d/%s)\n", errno, strerror(errno));
                goto clean_mr;
        }
        ctx->qp = ctx->gds_qp->qp;
        ctx->tx_cq = ctx->gds_qp->qp->send_cq;
        ctx->rx_cq = ctx->gds_qp->qp->recv_cq;

        {
                struct ibv_qp_attr attr = {
                        .qp_state        = IBV_QPS_INIT,
                        .pkey_index      = 0,
                        .port_num        = port,
                        .qkey            = 0x11111111
                };

                if (ibv_modify_qp(ctx->qp, &attr,
                                        IBV_QP_STATE              |
                                        IBV_QP_PKEY_INDEX         |
                                        IBV_QP_PORT               |
                                        IBV_QP_QKEY)) {
                        fprintf(stderr, "Failed to modify QP to INIT\n");
                        goto clean_qp;
                }
        }

        return ctx;

clean_qp:
        gds_destroy_qp(ctx->gds_qp);

clean_mr:
        ibv_dereg_mr(ctx->mr);

clean_pd:
        ibv_dealloc_pd(ctx->pd);

clean_comp_channel:
        if (ctx->channel)
                ibv_destroy_comp_channel(ctx->channel);

clean_device:
        ibv_close_device(ctx->context);

clean_buffer:
        if (ctx->gpumem)
                gpu_free(ctx->buf); 
        else 
                free(ctx->buf);

clean_ctx:
        if (ctx->gpu_id >= 0)
                gpu_finalize();
        free(ctx);

        return NULL;
}

int pp_close_ctx(struct pingpong_context *ctx)
{
        if (gds_destroy_qp(ctx->gds_qp)) {
                fprintf(stderr, "Couldn't destroy QP\n");
        }

        if (ibv_dereg_mr(ctx->mr)) {
                fprintf(stderr, "Couldn't deregister MR\n");
        }

        if (ibv_destroy_ah(ctx->ah)) {
                fprintf(stderr, "Couldn't destroy AH\n");
        }

        if (ibv_dealloc_pd(ctx->pd)) {
                fprintf(stderr, "Couldn't deallocate PD\n");
        }

        if (ctx->channel) {
                if (ibv_destroy_comp_channel(ctx->channel)) {
                        fprintf(stderr, "Couldn't destroy completion channel\n");
                }
        }

        if (ibv_close_device(ctx->context)) {
                fprintf(stderr, "Couldn't release context\n");
        }

        if (ctx->gpumem)
                gpu_free(ctx->buf); 
        else 
                free(ctx->buf);

        if (ctx->gpu_id >= 0)
                gpu_finalize();

        free(ctx);

        return 0;
}

static int block_server_stream(struct pingpong_context *ctx)
{
        gds_descriptor_t desc;
        desc.tag = GDS_TAG_WAIT_VALUE32;
        gds_prepare_wait_value32(&desc.wait32, (uint32_t *)ctx->rx_flag, 1, GDS_WAIT_COND_GEQ, GDS_MEMORY_HOST);

        gds_atomic_set_dword(desc.wait32.ptr, 0);
        gds_wmb();

        gpu_dbg("before gds_stream_post_descriptors\n");
        CUCHECK(gds_stream_post_descriptors(gpu_stream_server, 1, &desc, 0));
        gpu_dbg("after gds_stream_post_descriptors\n");
        return 0;
}

static int unblock_server_stream(struct pingpong_context *ctx)
{
        int retcode = 0;
        usleep(100);
        int ret = cuStreamQuery(gpu_stream_server);
        switch (ret) {
                case CUDA_ERROR_NOT_READY:
                        break;
                case CUDA_SUCCESS:
                        gpu_err("unexpected idle stream\n");
                        retcode = EINVAL;
                        break;
                default:
                        gpu_err("unexpected error %d in stream query\n", ret);
                        retcode = EINVAL;
                        break;
        }
        gds_atomic_set_dword((uint32_t *)ctx->rx_flag, 1);
        return retcode;
}

static int pp_post_recv(struct pingpong_context *ctx, int n)
{
        struct ibv_sge list = {
                .addr	= (uintptr_t) ctx->rxbuf,
                .length = ctx->size + 40,
                .lkey	= ctx->mr->lkey
        };
        struct ibv_recv_wr wr = {
                .wr_id	    = PINGPONG_RECV_WRID,
                .sg_list    = &list,
                .num_sge    = 1,
        };
        struct ibv_recv_wr *bad_wr;
        int i;
        gpu_dbg("posting %d recvs\n", n);
        for (i = 0; i < n; ++i)
                if (ibv_post_recv(ctx->qp, &wr, &bad_wr))
                        break;
        gpu_dbg("posted %d recvs\n", i);
        return i;
}

// will be needed when implementing the !peersync !use_desc_apis case
static int pp_post_send(struct pingpong_context *ctx, uint32_t qpn)
{
        struct ibv_sge list = {
                .addr	= (uintptr_t) ctx->txbuf,
                .length = ctx->size,
                .lkey	= ctx->mr->lkey
        };
        gds_send_wr ewr = {
                .wr_id	    = PINGPONG_SEND_WRID,
                .sg_list    = &list,
                .num_sge    = 1,
                .opcode     = IBV_WR_SEND,
                .send_flags = IBV_SEND_SIGNALED,
                .wr         = {
                        .ud = {
                                .ah          = ctx->ah,
                                .remote_qpn  = qpn,
                                .remote_qkey = 0x11111111
                        }
                }
        };
        gds_send_wr *bad_ewr;
        return gds_post_send(ctx->gds_qp, &ewr, &bad_ewr);
}

static int pp_post_gpu_send(struct pingpong_context *ctx, uint32_t qpn, CUstream *p_gpu_stream)
{
        struct ibv_sge list = {
                .addr	= (uintptr_t) ctx->txbuf,
                .length = ctx->size,
                .lkey	= ctx->mr->lkey
        };
        gds_send_wr ewr = {
                .wr_id	    = PINGPONG_SEND_WRID,
                .sg_list    = &list,
                .num_sge    = 1,
                .opcode     = IBV_WR_SEND,
                .send_flags = IBV_SEND_SIGNALED,
                .wr         = {
                        .ud = {
                                .ah          = ctx->ah,
                                .remote_qpn  = qpn,
                                .remote_qkey = 0x11111111
                        }
                }
        };
        gds_send_wr *bad_ewr;
        return gds_stream_queue_send(*p_gpu_stream, ctx->gds_qp, &ewr, &bad_ewr);
}

static int pp_prepare_gpu_send(struct pingpong_context *ctx, uint32_t qpn, gds_send_request_t *req)
{
        struct ibv_sge list = {
                .addr	= (uintptr_t) ctx->txbuf,
                .length = ctx->size,
                .lkey	= ctx->mr->lkey
        };
        gds_send_wr ewr = {
                .wr_id	    = PINGPONG_SEND_WRID,
                .sg_list    = &list,
                .num_sge    = 1,
                .opcode     = IBV_WR_SEND,
                .send_flags = IBV_SEND_SIGNALED,
                .wr         = {
                        .ud = {
                                .ah          = ctx->ah,
                                .remote_qpn  = qpn,
                                .remote_qkey = 0x11111111
                        }
                }
        };
        gds_send_wr *bad_ewr;
        return gds_prepare_send(ctx->gds_qp, &ewr, &bad_ewr, req);
}

typedef struct work_desc {
        gds_send_request_t send_rq;
        gds_wait_request_t wait_tx_rq;
        gds_wait_request_t wait_rx_rq;
#define N_WORK_DESCS 3
        gds_descriptor_t descs[N_WORK_DESCS];
} work_desc_t;

static void post_work_cb(CUstream hStream, CUresult status, void *userData)\
{
        int retcode;
        work_desc_t *wdesc = (work_desc_t *)userData;
        gpu_dbg("stream callback wdesc=%p\n", wdesc);
        assert(wdesc);
        NVTX_PUSH("work_cb", 1);
        if (status != CUDA_SUCCESS) {
                fprintf(stderr,"ERROR: CUresult %d in stream callback\n", status);
                goto out;
        }
        assert(sizeof(wdesc->descs)/sizeof(wdesc->descs[0]) == N_WORK_DESCS);
        retcode = gds_post_descriptors(sizeof(wdesc->descs)/sizeof(wdesc->descs[0]), wdesc->descs, 0);
        if (retcode) {
                fprintf(stderr,"ERROR: error %d returned by gds_post_descriptors, going on...\n", retcode);
                stream_cb_error = 1;
        }
out:
        free(wdesc);
        NVTX_POP();
}

static int pp_post_work(struct pingpong_context *ctx, int n_posts, int rcnt, uint32_t qpn, int is_client)
{
        int retcode = 0;
        int i, ret = 0;
        int posted_recv = 0;

        gpu_dbg("n_posts=%d rcnt=%d is_client=%d\n", n_posts, rcnt, is_client);

        if (n_posts <= 0) {
                gpu_dbg("nothing to do\n");
                return 0;
        }

        NVTX_PUSH("post recv", 1);
        posted_recv = pp_post_recv(ctx, n_posts);
        if (posted_recv < 0) {
                fprintf(stderr,"ERROR: can't post recv (%d) n_posts=%d is_client=%d\n", 
                                posted_recv, n_posts, is_client);
                exit(EXIT_FAILURE);
                return 0;
        } else if (posted_recv != n_posts) {
                fprintf(stderr,"ERROR: couldn't post all recvs (%d posted, %d requested)\n", posted_recv, n_posts);
                if (!posted_recv)
                        return 0;
        }
        NVTX_POP();

        PROF(&prof, prof_idx++);

        NVTX_PUSH("post send+wait", 1);
        for (i = 0; i < posted_recv; ++i) {
                if (ctx->use_desc_apis) {
                        work_desc_t *wdesc = calloc(1, sizeof(*wdesc));
                        int k = 0;
                        ret = pp_prepare_gpu_send(ctx, qpn, &wdesc->send_rq);
                        if (ret) {
                                retcode = -ret;
                                break;
                        }
                        assert(k < N_WORK_DESCS);
                        wdesc->descs[k].tag = GDS_TAG_SEND;
                        wdesc->descs[k].send = &wdesc->send_rq;
                        ++k;

                        ret = gds_prepare_wait_cq(ctx->gds_qp->send_cq, &wdesc->wait_tx_rq, 0);
                        if (ret) {
                                retcode = -ret;
                                break;
                        }
                        assert(k < N_WORK_DESCS);
                        wdesc->descs[k].tag = GDS_TAG_WAIT;
                        wdesc->descs[k].wait = &wdesc->wait_tx_rq;
                        ++k;

                        ret = gds_prepare_wait_cq(ctx->gds_qp->recv_cq, &wdesc->wait_rx_rq, 0);
                        if (ret) {
                                retcode = -ret;
                                break;
                        }
                        assert(k < N_WORK_DESCS);
                        wdesc->descs[k].tag = GDS_TAG_WAIT;
                        wdesc->descs[k].wait = &wdesc->wait_rx_rq;
                        ++k;

                        if (ctx->peersync) {
                                gpu_dbg("before gds_stream_post_descriptors\n");
                                ret = gds_stream_post_descriptors(gpu_stream_server, k, wdesc->descs, 0);
                                gpu_dbg("after gds_stream_post_descriptors\n");
                                free(wdesc);
                                if (ret) {
                                        retcode = -ret;
                                        break;
                                }
                        } else {
                                gpu_dbg("adding post_work_cb to stream=%p\n", gpu_stream_server);
                                CUCHECK(cuStreamAddCallback(gpu_stream_server, post_work_cb, wdesc, 0));
                        }
                } else if (ctx->peersync) {
                        ret = pp_post_gpu_send(ctx, qpn, &gpu_stream_server);
                        if (ret) {
                                gpu_err("error %d in pp_post_gpu_send, posted_recv=%d posted_so_far=%d is_client=%d \n",
                                                ret, posted_recv, i, is_client);
                                retcode = -ret;
                                break;
                        }

                        ret = gds_stream_wait_cq(gpu_stream_server, ctx->gds_qp->send_cq, 0);
                        if (ret) {
                                // TODO: rollback gpu send
                                gpu_err("error %d in gds_stream_wait_cq\n", ret);
                                retcode = -ret;
                                break;
                        }

                        ret = gds_stream_wait_cq(gpu_stream_server, ctx->gds_qp->recv_cq, ctx->consume_rx_cqe);
                        if (ret) {
                                // TODO: rollback gpu send and wait send_cq
                                gpu_err("error %d in gds_stream_wait_cq\n", ret);
                                retcode = -ret;
                                break;
                        }
                } else {
                        gpu_err("!peersync case only supported when using descriptor APIs\n");
                        retcode = -EINVAL;
                        break;
                }
                if (ctx->skip_kernel_launch) {
                        gpu_warn_once("NOT LAUNCHING ANY KERNEL AT ALL\n");
                } else {
                        gpu_launch_kernel_on_stream(ctx->calc_size, ctx->peersync, gpu_stream_server);
                }

        }
        PROF(&prof, prof_idx++);
        if (!retcode) {
                retcode = i;
                gpu_post_release_tracking_event(&gpu_stream_server);
        }
        NVTX_POP();

        return retcode;
}

static void usage(const char *argv0)
{
        printf("Usage:\n");
        printf("  %s            start a server and wait for connection\n", argv0);
        printf("  %s <host>     connect to server at <host>\n", argv0);
        printf("\n");
        printf("Options:\n");
        printf("  -p, --port=<port>      listen on/connect to port <port> (default 18515)\n");
        printf("  -d, --ib-dev=<dev>     use IB device <dev> (default first device found)\n");
        printf("  -i, --ib-port=<port>   use port <port> of IB device (default 1)\n");
        printf("  -s, --size=<size>      size of message to exchange (default 1024)\n");
        printf("  -r, --rx-depth=<dep>   number of receives to post at a time (default 500)\n");
        printf("  -n, --iters=<iters>    number of exchanges (default 1000)\n");
        printf("  -e, --events           sleep on CQ events (default poll)\n");
        printf("  -g, --gid-idx=<gid index> local port gid index\n");
        printf("  -S, --gpu-calc-size=<size>  size of GPU compute buffer (default 128KB)\n");
        printf("  -G, --gpu-id           use specified GPU (default 0)\n");
        printf("  -B, --batch-length=<n> max batch length (default 20)\n");
        printf("  -P, --peersync            disable GPUDirect PeerSync support (default enabled)\n");
        printf("  -C, --peersync-gpu-cq     enable GPUDirect PeerSync GPU CQ support (default disabled)\n");
        printf("  -D, --peersync-gpu-dbrec  enable QP DBREC on GPU memory (default disabled)\n");
        printf("  -U, --peersync-desc-apis  use batched descriptor APIs (default disabled)\n");
        printf("  -Q, --consume-rx-cqe      enable GPU consumes RX CQE support (default disabled)\n");
        printf("  -M, --gpu-sched-mode      set CUDA context sched mode, default (A)UTO, (S)PIN, (Y)IELD, (B)LOCKING\n");
        printf("  -E, --gpu-mem             allocate GPU instead of CPU memory buffers\n");
        printf("  -K, --skip-kernel-launch  no GPU kernel computations, only communications\n");
        printf("  -L, --hide-cpu-launch-latency try to prelaunch work on blocked stream then unblock\n");
}

int main(int argc, char *argv[])
{
        struct ibv_device      **dev_list;
        struct ibv_device	*ib_dev;
        struct pingpong_context *ctx;
        struct pingpong_dest     my_dest;
        struct pingpong_dest    *rem_dest;
        struct timeval           start, end;
        const char              *ib_devname = NULL;
        char                    *servername = NULL;
        int                      port = 18515;
        int                      ib_port = 1;
        int                      size = 1024;
        int                      calc_size = 128*1024;
        int                      rx_depth = 2*512;
        int                      iters = 1000;
        int                      use_event = 0;
        int                      routs;
        int                      nposted;
        int                      rcnt, scnt;
        int                      sl = 0;
        int			 gidx = -1;
        char			 gid[INET6_ADDRSTRLEN];
        int                      gpu_id = 0;
        int                      peersync = 1;
        int                      peersync_gpu_cq = 0;
        int                      peersync_gpu_dbrec = 0;
        int                      max_batch_len = 20;
        int                      consume_rx_cqe = 0;
        int                      sched_mode = CU_CTX_SCHED_AUTO;
        int                      ret = 0;
        int                      wait_key = -1;
        int                      use_gpumem = 0;
        int                      use_desc_apis = 0;
        int                      skip_kernel_launch = 0;
        int                      hide_cpu_launch_latency = 0;

        /*printf("sizeof(gds_send_request_t)=%zu\n", sizeof(gds_send_request_t));
        printf("sizeof(gds_mlx5_peer_commit)=%zu\n", sizeof(struct gds_mlx5_peer_commit));
        printf("sizeof(gds_mlx5_peer_op_wr)=%zu\n", sizeof(struct gds_mlx5_peer_op_wr));
        printf("sizeof(gds_wait_request_t)=%zu\n", sizeof(gds_wait_request_t));
        printf("sizeof(gds_mlx5_peer_peek)=%zu\n", sizeof(struct gds_mlx5_peer_peek));
        exit(0);*/
        fprintf(stdout, "libgdsync build version 0x%08x, major=%d minor=%d\n", GDS_API_VERSION, GDS_API_MAJOR_VERSION, GDS_API_MINOR_VERSION);

        int version;
        ret = gds_query_param(GDS_PARAM_VERSION, &version);
        if (ret) {
                fprintf(stderr, "error querying libgdsync version\n");
                exit(EXIT_FAILURE);
        }
        fprintf(stdout, "libgdsync queried version 0x%08x\n", version);
        if (!GDS_API_VERSION_COMPATIBLE(version)) {
                fprintf(stderr, "incompatible libgdsync version 0x%08x\n", version);
                exit(EXIT_FAILURE);
        }

        srand48(getpid() * time(NULL));

        while (1) {
                int c;

                static struct option long_options[] = {
                        { .name = "port",     .has_arg = 1, .val = 'p' },
                        { .name = "ib-dev",   .has_arg = 1, .val = 'd' },
                        { .name = "ib-port",  .has_arg = 1, .val = 'i' },
                        { .name = "size",     .has_arg = 1, .val = 's' },
                        { .name = "rx-depth", .has_arg = 1, .val = 'r' },
                        { .name = "iters",    .has_arg = 1, .val = 'n' },
                        { .name = "sl",       .has_arg = 1, .val = 'l' },
                        { .name = "events",   .has_arg = 0, .val = 'e' },
                        { .name = "gid-idx",  .has_arg = 1, .val = 'g' },
                        { .name = "gpu-id",          .has_arg = 1, .val = 'G' },
                        { .name = "peersync",        .has_arg = 0, .val = 'P' },
                        { .name = "peersync-gpu-cq", .has_arg = 0, .val = 'C' },
                        { .name = "peersync-gpu-dbrec", .has_arg = 1, .val = 'D' },
                        { .name = "peersync-desc-apis", .has_arg = 0, .val = 'U' },
                        { .name = "gpu-calc-size",   .has_arg = 1, .val = 'S' },
                        { .name = "batch-length",    .has_arg = 1, .val = 'B' },
                        { .name = "consume-rx-cqe",  .has_arg = 0, .val = 'Q' },
                        { .name = "gpu-sched-mode",  .has_arg = 1, .val = 'M' },
                        { .name = "gpu-mem",         .has_arg = 0, .val = 'E' },
                        { .name = "wait-key",        .has_arg = 1, .val = 'W' },
                        { .name = "skip-kernel-launch", .has_arg = 0, .val = 'K' },
                        { .name = "hide-cpu-launch-latency", .has_arg = 0, .val = 'L' },
                        { 0 }
                };

                c = getopt_long(argc, argv, "p:d:i:s:r:n:l:eg:G:S:B:PCDQM:W:EUKL", long_options, NULL);
                if (c == -1)
                        break;

                switch (c) {
                        case 'p':
                                port = strtol(optarg, NULL, 0);
                                if (port < 0 || port > 65535) {
                                        usage(argv[0]);
                                        ret = 1;
                                        exit(EXIT_FAILURE);
                                }
                                break;

                        case 'd':
                                ib_devname = strdup(optarg);
                                break;

                        case 'i':
                                ib_port = strtol(optarg, NULL, 0);
                                if (ib_port < 0) {
                                        usage(argv[0]);
                                        ret = 1;
                                        exit(EXIT_FAILURE);
                                }
                                break;

                        case 's':
                                size = strtol(optarg, NULL, 0);
                                printf("INFO: message size=%d\n", size);
                                break;

                        case 'S':
                                calc_size = strtol(optarg, NULL, 0);
                                printf("INFO: kernel calc size=%d\n", calc_size);
                                break;

                        case 'r':
                                rx_depth = strtol(optarg, NULL, 0);
                                break;

                        case 'n':
                                iters = strtol(optarg, NULL, 0);
                                break;

                        case 'l':
                                sl = strtol(optarg, NULL, 0);
                                break;

                        case 'e':
                                ++use_event;
                                break;

                        case 'g':
                                gidx = strtol(optarg, NULL, 0);
                                break;

                        case 'G':
                                gpu_id = strtol(optarg, NULL, 0);
                                printf("INFO: gpu id=%d\n", gpu_id);
                                break;

                        case 'B':
                                max_batch_len = strtol(optarg, NULL, 0);
                                printf("INFO: max_batch_len=%d\n", max_batch_len);
                                break;

                        case 'P':
                                peersync = !peersync;
                                printf("INFO: switching PeerSync %s\n", peersync?"ON":"OFF");
                                if (!peersync) {
                                        printf("WARNING: PeerSync OFF is approximated using CUDA stream callbacks\n");
                                }
                                break;

                        case 'Q':
                                consume_rx_cqe = !consume_rx_cqe;
                                printf("INFO: switching consume_rx_cqe %s\n", consume_rx_cqe?"ON":"OFF");
                                break;

                        case 'C':
                                peersync_gpu_cq = !peersync_gpu_cq;
                                printf("INFO: switching %s PeerSync GPU CQ\n", peersync_gpu_cq?"ON":"OFF");
                                break;

                        case 'D':
                                peersync_gpu_dbrec= !peersync_gpu_dbrec;
                                printf("INFO: switching %s PeerSync GPU QP DBREC\n", peersync_gpu_dbrec?"ON":"OFF");
                                break;

                        case 'M':
                                {
                                        char m = *optarg;
                                        printf("INFO: sched mode '%c'\n", m);
                                        switch (m) {
                                                case 'S': sched_mode = CU_CTX_SCHED_SPIN; break;
                                                case 'Y': sched_mode = CU_CTX_SCHED_YIELD; break;
                                                case 'B': sched_mode = CU_CTX_SCHED_BLOCKING_SYNC; break;
                                                case 'A': sched_mode = CU_CTX_SCHED_AUTO; break;
                                                default: printf("ERROR: unexpected value %c\n", m); exit(EXIT_FAILURE); break;
                                        }
                                }
                                break;

                        case 'W':
                                wait_key = strtol(optarg, NULL, 0);
                                printf("INFO: wait_key=%d\n", wait_key);
                                break;

                        case 'E':
                                use_gpumem = !use_gpumem;
                                printf("INFO: use_gpumem=%d\n", use_gpumem);
                                break;

                        case 'U':
                                use_desc_apis = 1;
                                printf("INFO: use_desc_apis=%d\n", use_desc_apis);
                                break;

                        case 'K':
                                skip_kernel_launch = 1;
                                printf("INFO: skip_kernel_launch=%d\n", skip_kernel_launch);
                                break;

                        case 'L':
                                hide_cpu_launch_latency = 1;
                                printf("INFO: hide_cpu_launch_latency=%d\n", hide_cpu_launch_latency);
                                break;

                        default:
                                usage(argv[0]);
                                return 1;
                }
        }

        if (!peersync && !use_desc_apis) {
                gpu_err("!peersync case only supported when using descriptor APIs, enabling them\n");
                use_desc_apis = 1;
                return 1;
        }

        assert(comm_size == 1);
        char *hostnames[1] = {"localhost"};

        if (my_rank == 0) {
                servername = hostnames[0];
                printf("[%d] pid=%d server:%s\n", my_rank, getpid(), servername);
        }

        const char *tags = NULL;
        tags = "wait trk|pollrxcq|polltxcq|postrecv|postwork| poketrk";
        prof_init(&prof, 100000, 100000, "100us", 60, 2, tags);
        prof_disable(&prof);

        page_size = sysconf(_SC_PAGESIZE);

        dev_list = ibv_get_device_list(NULL);
        if (!dev_list) {
                perror("Failed to get IB devices list");
                return 1;
        }

        if (!ib_devname) {
                const char *value = getenv("USE_HCA"); 
                if (value != NULL) {
                        printf("[%d] USE_HCA: <%s>\n", my_rank, value);
                        ib_devname = value;
                }
        } else {
                printf("[%d] requested IB device: <%s>\n", my_rank, ib_devname);
        }

        if (!ib_devname) {
                printf("picking 1st available device\n");
                ib_dev = *dev_list;
                if (!ib_dev) {
                        fprintf(stderr, "No IB devices found\n");
                        return 1;
                }
        } else {
                int i;
                for (i = 0; dev_list[i]; ++i)
                        if (!strcmp(ibv_get_device_name(dev_list[i]), ib_devname))
                                break;
                ib_dev = dev_list[i];
                if (!ib_dev) {
                        fprintf(stderr, "IB device %s not found\n", ib_devname);
                        return 1;
                }
        }

        {
                const char *env = getenv("USE_GPU");
                if (env) {
                        gpu_id = atoi(env);
                        printf("USE_GPU=%s(%d)\n", env, gpu_id);
                }
        }
        printf("use gpumem: %d\n", use_gpumem);
        ctx = pp_init_ctx(ib_dev, size, calc_size, rx_depth, ib_port, 0, gpu_id, peersync, peersync_gpu_cq, peersync_gpu_dbrec, consume_rx_cqe, sched_mode, use_gpumem, use_desc_apis, skip_kernel_launch);
        if (!ctx)
                return 1;

        //pre-posting
        int nrecv = pp_post_recv(ctx, max_batch_len);
        if (nrecv < max_batch_len) {
                fprintf(stderr, "Couldn't post receive (%d)\n", nrecv);
                return 1;
        }

        if (pp_get_port_info(ctx->context, ib_port, &ctx->portinfo)) {
                fprintf(stderr, "Couldn't get port info\n");
                return 1;
        }
        my_dest.lid = ctx->portinfo.lid;

        my_dest.qpn = ctx->qp->qp_num;
        my_dest.psn = lrand48() & 0xffffff;

        if (gidx >= 0) {
                if (ibv_query_gid(ctx->context, ib_port, gidx, &my_dest.gid)) {
                        fprintf(stderr, "Could not get local gid for gid index "
                                        "%d\n", gidx);
                        return 1;
                }
        } else
                memset(&my_dest.gid, 0, sizeof my_dest.gid);

        inet_ntop(AF_INET6, &my_dest.gid, gid, sizeof gid);
        printf("  local address:  LID 0x%04x, QPN 0x%06x, PSN 0x%06x: GID %s\n",
                        my_dest.lid, my_dest.qpn, my_dest.psn, gid);

        rem_dest = pp_client_exch_dest(servername, port, &my_dest);

        if (!rem_dest) {
                fprintf(stderr, "Could not exchange destination\n");
                ret = 1;
                goto out;
        }

        inet_ntop(AF_INET6, &rem_dest->gid, gid, sizeof gid);
        printf("  remote address: LID 0x%04x, QPN 0x%06x, PSN 0x%06x, GID %s\n",
                        rem_dest->lid, rem_dest->qpn, rem_dest->psn, gid);

        if (servername) {
                if (pp_connect_ctx(ctx, ib_port, my_dest.psn, sl, rem_dest, gidx))
                        return 1;
                //sleep(1);
        }

        if (hide_cpu_launch_latency) {
                printf("INFO: blocking stream ...\n");
                block_server_stream(ctx);
        }

        if (gettimeofday(&start, NULL)) {
                perror("gettimeofday");
                ret = 1;
                goto out;
        }

        // for performance reasons, multiple batches back-to-back are posted here
        rcnt = scnt = 0;
        nposted = 0;
        routs = 0;
        const int n_batches = 3;
        int last_batch_len = 0;
        int n_post = 0;
        int n_posted;
        int batch;

        for (batch=0; batch<n_batches; ++batch) {
                PROF(&prof, prof_idx++);
                PROF(&prof, prof_idx++);
                PROF(&prof, prof_idx++);
                PROF(&prof, prof_idx++);

                n_post = min(min(ctx->rx_depth/2, iters-nposted), max_batch_len);
                gpu_dbg("batch=%d n_post=%d\n", batch, n_post);
                n_posted = pp_post_work(ctx, n_post, 0, rem_dest->qpn, servername?1:0);
                PROF(&prof, prof_idx++);
                if (n_posted < 0) {
                        fprintf(stderr, "ERROR: got error %d\n", n_posted);
                        ret = 1;
                        goto out;
                }
                else if (n_posted != n_post) {
                        fprintf(stderr, "ERROR: Couldn't post work, got %d requested %d\n", n_posted, n_post);
                        ret = 1;
                        goto out;
                }
                routs += n_posted;
                nposted += n_posted;
                last_batch_len = n_posted;
                printf("[%d] batch %d: posted %d sequences\n", my_rank, batch, n_posted);
        }

        ctx->pending = PINGPONG_RECV_WRID;

        float pre_post_us = 0;

        if (gettimeofday(&end, NULL)) {
                perror("gettimeofday");
                ret = 1;
                goto out;
        }
        {
                float usec = (end.tv_sec - start.tv_sec) * 1000000 +
                        (end.tv_usec - start.tv_usec);
                printf("pre-posting took %.2f usec\n", usec);
                pre_post_us = usec;
        }

        if (hide_cpu_launch_latency) {
                printf("ignoring pre-posting time and unblocking the stream\n");
                pre_post_us = 0;
                if (unblock_server_stream(ctx)) {
                        exit(EXIT_FAILURE);
                }
        }

        if (!my_rank) {
                puts("");
                printf("batch info: rx+kernel+tx %d per batch\n", n_posted); // this is the last actually
                printf("pre-posted %d sequences in %d batches\n", nposted, 2);
                printf("GPU kernel calc buf size: %d\n", ctx->calc_size);
                printf("iters=%d tx/rx_depth=%d\n", iters, ctx->rx_depth);
                printf("\n");
                printf("testing....\n");
                fflush(stdout);
        }

        if (gettimeofday(&start, NULL)) {
                perror("gettimeofday");
                return 1;
        }
        prof_enable(&prof);
        prof_idx = 0;
        int got_error = 0;
        int iter = 0;
        while ((rcnt < iters && scnt < iters) && !got_error && !stream_cb_error) {
                ++iter;
                PROF(&prof, prof_idx++);

                int ret = gpu_wait_tracking_event(1000*1000);
                if (ret == ENOMEM) {
                        gpu_dbg("gpu_wait_tracking_event nothing to do (%d)\n", ret);
                } else if (ret == EAGAIN) {
                        gpu_err("gpu_wait_tracking_event timout (%d), retrying\n", ret);
                        prof_reset(&prof);
                        continue;
                } else if (ret) {
                        gpu_err("gpu_wait_tracking_event failed (%d)\n", ret);
                        got_error = ret;
                }

                PROF(&prof, prof_idx++);

                // don't call poll_cq on events which are still being polled by the GPU
                int n_rx_ev = 0;
                if (!ctx->consume_rx_cqe) {
                        struct ibv_wc wc[max_batch_len];
                        int ne = 0, i;

                        ne = gds_poll_cq(ctx->gds_qp->recv_cq, max_batch_len, wc);
                        if (ne < 0) {
                                fprintf(stderr, "poll RX CQ failed %d\n", ne);
                                return 1;
                        }
                        n_rx_ev += ne;
                        for (i = 0; i < ne; ++i) {
                                if (wc[i].status != IBV_WC_SUCCESS) {
                                        fprintf(stderr, "Failed status %s (%d) for wr_id %d\n",
                                                        ibv_wc_status_str(wc[i].status),
                                                        wc[i].status, (int) wc[i].wr_id);
                                        return 1;
                                }

                                switch ((int) wc[i].wr_id) {
                                        case PINGPONG_RECV_WRID:
                                                ++rcnt;
                                                break;
                                        default:
                                                fprintf(stderr, "Completion for unknown wr_id %d\n",
                                                                (int) wc[i].wr_id);
                                                return 1;
                                }
                        }
                } else {
                        n_rx_ev = last_batch_len;
                        rcnt += last_batch_len;
                }

                PROF(&prof, prof_idx++);
                int n_tx_ev = 0;
                {
                        struct ibv_wc wc[max_batch_len];
                        int ne, i;

                        ne = gds_poll_cq(ctx->gds_qp->send_cq, max_batch_len, wc);
                        if (ne < 0) {
                                fprintf(stderr, "poll TX CQ failed %d\n", ne);
                                return 1;
                        }
                        n_tx_ev += ne;
                        for (i = 0; i < ne; ++i) {
                                if (wc[i].status != IBV_WC_SUCCESS) {
                                        fprintf(stderr, "Failed status %s (%d) for wr_id %d\n",
                                                        ibv_wc_status_str(wc[i].status),
                                                        wc[i].status, (int) wc[i].wr_id);
                                        return 1;
                                }

                                switch ((int) wc[i].wr_id) {
                                        case PINGPONG_SEND_WRID:
                                                ++scnt;
                                                break;
                                        default:
                                                fprintf(stderr, "Completion for unknown wr_id %d\n",
                                                                (int) wc[i].wr_id);
                                                ret = 1;
                                                goto out;
                                }
                        }
                }

                PROF(&prof, prof_idx++);
                if (1 && (n_tx_ev || n_rx_ev)) {
                        //fprintf(stderr, "iter=%d n_rx_ev=%d, n_tx_ev=%d\n", iter, n_rx_ev, n_tx_ev); fflush(stdout);
                }
                if (n_tx_ev || n_rx_ev) {
                        // update counters
                        routs -= last_batch_len;
                        if (n_tx_ev != last_batch_len)
                                gpu_dbg("[%d] partially completed batch, got tx ev %d, batch len %d\n", iter, n_tx_ev, last_batch_len);
                        if (n_rx_ev != last_batch_len)
                                gpu_dbg("[%d] partially completed batch, got rx ev %d, batch len %d\n", iter, n_rx_ev, last_batch_len);
                        if (nposted < iters) {
                                // potentially submit new work
                                n_post = min(min(ctx->rx_depth/2, iters-nposted), max_batch_len);
                                int n = pp_post_work(ctx, n_post, nposted, rem_dest->qpn, servername?1:0);
                                if (n != n_post) {
                                        fprintf(stderr, "ERROR: post_work error (%d) rcnt=%d n_post=%d routs=%d\n", n, rcnt, n_post, routs);
                                        return 1;
                                }
                                last_batch_len = n;
                                routs += n;
                                nposted += n;
                        }
                } else {
                        PROF(&prof, prof_idx++);
                        PROF(&prof, prof_idx++);
                }
                PROF(&prof, prof_idx++);
                prof_update(&prof);
                prof_idx = 0;


                if (got_error || stream_cb_error) {
                        gpu_err("[%d] exiting due to error(s)\n", my_rank);
                        return 1;
                }

                if (wait_key >= 0) {
                        if (iter == wait_key) {
                                puts("press any key");
                                getchar();
                        }
                }
        }

        if (gettimeofday(&end, NULL)) {
                perror("gettimeofday");
                ret = 1;
        }

        {
                float usec = (end.tv_sec - start.tv_sec) * 1000000 +
                        (end.tv_usec - start.tv_usec) + pre_post_us;
                long long bytes = (long long) size * iters * 2;

                printf("[%d] %lld bytes in %.2f seconds = %.2f Mbit/sec\n",
                                my_rank, bytes, usec / 1000000., bytes * 8. / usec);
                printf("[%d] %d iters in %.2f seconds = %.2f usec/iter\n",
                                my_rank, iters, usec / 1000000., usec / iters);
        }

        if (prof_enabled(&prof)) {
                printf("dumping prof\n");
                prof_dump(&prof);
        }
        prof_destroy(&prof);

        return 0;

out:

        if (pp_close_ctx(ctx))
                ret = 1;

        ibv_free_device_list(dev_list);
        free(rem_dest);


        return ret;
}

/*
 * Local variables:
 *  c-indent-level: 8
 *  c-basic-offset: 8
 *  tab-width: 8
 * End:
 */
