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

#if HAVE_CONFIG_H
#  include <config.h>
#endif /* HAVE_CONFIG_H */

#include <unistd.h>
#include <string.h>
#include <assert.h>
#include <inttypes.h>

//#include <map>
//#include <algorithm>
//#include <string>
//using namespace std;

//#include <cuda.h>
//#include <gdrapi.h>

#include "gdsync.h"
#include "gdsync/tools.h"
#include "objs.hpp"
#include "utils.hpp"
#include "memmgr.hpp"
#include "utils.hpp"
#include "archutils.h"
#include "mlnxutils.h"
#include "transport.hpp"

//-----------------------------------------------------------------------------

static void gds_init_send_info(gds_send_request_t *info)
{
        gds_dbg("send_request=%p\n", info);
        memset(info, 0, sizeof(*info));

        gds_main_transport->init_send_info(info);
}

//-----------------------------------------------------------------------------

static void gds_init_wait_request(gds_wait_request_t *request, uint32_t offset)
{
        gds_dbg("wait_request=%p offset=%08x\n", request, offset);
        memset(request, 0, sizeof(*request));

        gds_main_transport->init_wait_request(request, offset);
}

//-----------------------------------------------------------------------------

static int gds_rollback_qp(struct gds_qp *qp, gds_send_request_t *send_info)
{
        assert(qp);
        assert(send_info);

        return gds_main_transport->rollback_qp(qp, send_info);
}

//-----------------------------------------------------------------------------

int gds_post_send(struct gds_qp *qp, gds_send_wr *p_ewr, gds_send_wr **bad_ewr)
{
        int ret = 0, ret_roll=0;
        gds_send_request_t send_info;
        ret = gds_prepare_send(qp, p_ewr, bad_ewr, &send_info);
        if (ret) {
                gds_err("error %d in gds_prepare_send\n", ret);
                goto out;
        }

        ret = gds_post_pokes_on_cpu(1, &send_info, NULL, 0);
        if (ret) {
                gds_err("error %d in gds_post_pokes_on_cpu\n", ret);
                ret_roll = gds_rollback_qp(qp, &send_info);
                if (ret_roll) {
                        gds_err("error %d in gds_rollback_qp\n", ret_roll);
                }

                goto out;
        }

out:
        return ret;
}

//-----------------------------------------------------------------------------

int gds_post_recv(struct gds_qp *qp, struct ibv_recv_wr *wr, struct ibv_recv_wr **bad_wr)
{
        int ret = 0;

        assert(qp);
        assert(wr);
        assert(bad_wr);
        gds_dbg("qp=%p wr=%p\n", qp, wr);
        ret = gds_main_transport->post_recv(qp, wr, bad_wr);
        if (ret) {
                gds_err("error %d in post_recv\n", ret);
                goto out;
        }

out:
        return ret;
}

//-----------------------------------------------------------------------------

int gds_prepare_send(struct gds_qp *gqp, gds_send_wr *p_ewr, 
                     gds_send_wr **bad_ewr, 
                     gds_send_request_t *request)
{
        int ret = 0;

        gds_init_send_info(request);
        assert(gqp);
        assert(gqp->qp);

        ret = gds_main_transport->prepare_send(gqp, p_ewr, bad_ewr, request);
        if (ret)
                gds_err("Error %d in prepare_send.\n", ret);

        return ret;
}

//-----------------------------------------------------------------------------

int gds_stream_queue_send(CUstream stream, struct gds_qp *qp, gds_send_wr *p_ewr, gds_send_wr **bad_ewr)
{
        int ret = 0, ret_roll = 0;
        gds_send_request_t send_info;
        gds_descriptor_t descs[1];

        assert(qp);
        assert(p_ewr);

        ret = gds_prepare_send(qp, p_ewr, bad_ewr, &send_info);
        if (ret) {
                gds_err("error %d in gds_prepare_send\n", ret);
                goto out;
        }

        descs[0].tag = GDS_TAG_SEND;
        descs[0].send = &send_info;

        ret=gds_stream_post_descriptors(stream, 1, descs, 0);
        if (ret) {
                gds_err("error %d in gds_stream_post_descriptors\n", ret);
                goto out;
        }

        out:
        return ret;
}

//-----------------------------------------------------------------------------

int gds_stream_post_send(CUstream stream, gds_send_request_t *request)
{
        int ret = 0;
        ret = gds_stream_post_send_all(stream, 1, request);
        if (ret) {
                gds_err("gds_stream_post_send_all (%d)\n", ret);
        }
        return ret;
}

//-----------------------------------------------------------------------------

int gds_stream_post_send_all(CUstream stream, int count, gds_send_request_t *request)
{
        int ret = 0, k = 0;
        gds_descriptor_t * descs = NULL;

        assert(request);
        assert(count);

        descs = (gds_descriptor_t *) calloc(count, sizeof(gds_descriptor_t));
        if(!descs)
        {
                gds_err("Calloc for %d elements\n", count);
                ret=ENOMEM;
                goto out;
        }

        for (k=0; k<count; k++) {
                descs[k].tag = GDS_TAG_SEND;
                descs[k].send = &request[k];
        }

        ret=gds_stream_post_descriptors(stream, count, descs, 0);
        if (ret) {
                gds_err("error %d in gds_stream_post_descriptors\n", ret);
                goto out;
        }

        out:
            if(descs) free(descs);
            return ret;
}

//-----------------------------------------------------------------------------

int gds_prepare_wait_cq(struct gds_cq *cq, gds_wait_request_t *request, int flags)
{
        if (flags != 0) {
                gds_err("invalid flags != 0\n");
                return EINVAL;
        }

        gds_init_wait_request(request, cq->curr_offset++);

        return gds_main_transport->prepare_wait_cq(cq, request, flags);
}

//-----------------------------------------------------------------------------

int gds_append_wait_cq(gds_wait_request_t *request, uint32_t *dw, uint32_t val)
{
        int ret = gds_transport_init();
        if (ret) {
                gds_err("error in gds_transport_init\n");
                goto out;
        }

        ret = gds_main_transport->append_wait_cq(request, dw, val);

out:
        return ret;
}

//-----------------------------------------------------------------------------

int gds_stream_post_wait_cq(CUstream stream, gds_wait_request_t *request)
{
	return gds_stream_post_wait_cq_multi(stream, 1, request, NULL, 0);
}

//-----------------------------------------------------------------------------

int gds_stream_post_wait_cq_all(CUstream stream, int count, gds_wait_request_t *requests)
{
	return gds_stream_post_wait_cq_multi(stream, count, requests, NULL, 0);
}

//-----------------------------------------------------------------------------

static int gds_abort_wait_cq(struct gds_cq *cq, gds_wait_request_t *request)
{
        assert(cq);
        assert(request);

        return gds_main_transport->abort_wait_cq(cq, request);
}

//-----------------------------------------------------------------------------

int gds_stream_wait_cq(CUstream stream, struct gds_cq *cq, int flags)
{
        int retcode = 0;
        int ret;
        gds_wait_request_t request;

        assert(cq);
        assert(stream);

        if (flags) {
                retcode = EINVAL;
                goto out;
        }

        ret = gds_prepare_wait_cq(cq, &request, flags);
        if (ret) {
                gds_err("error %d in gds_prepare_wait_cq\n", ret);
                goto out;
        }

        ret = gds_stream_post_wait_cq(stream, &request);
        if (ret) {
                gds_err("error %d in gds_stream_post_wait_cq_ex\n", ret);
                int retcode2 = gds_abort_wait_cq(cq, &request);
                if (retcode2) {
                        gds_err("nested error %d while aborting request\n", retcode2);
                }
                retcode = ret;
                goto out;
        }

out:
	return retcode;
}

//-----------------------------------------------------------------------------

int gds_post_wait_cq(struct gds_cq *cq, gds_wait_request_t *request, int flags)
{
        int retcode = 0;

        if (flags) {
                retcode = EINVAL;
                goto out;
        }

        retcode = gds_abort_wait_cq(cq, request);
out:
        return retcode;
}

//-----------------------------------------------------------------------------

int gds_prepare_wait_value32(gds_wait_value32_t *desc, uint32_t *ptr, uint32_t value, gds_wait_cond_flag_t cond_flags, int flags)
{
        int ret = 0;
        assert(desc);

        gds_dbg("desc=%p ptr=%p value=0x%08x cond_flags=0x%x flags=0x%x\n",
                desc, ptr, value, cond_flags, flags);

        if (flags & ~(GDS_WAIT_POST_FLUSH_REMOTE|GDS_MEMORY_MASK)) {
                gds_err("invalid flags\n");
                ret = EINVAL;
                goto out;
        }
        if (!is_valid(memtype_from_flags(flags))) {
                gds_err("invalid memory type in flags\n");
                ret = EINVAL;
                goto out;
        }
        if (!is_valid(cond_flags)) {
                gds_err("invalid cond flags\n");
                ret = EINVAL;
                goto out;
        }
        desc->ptr = ptr;
        desc->value = value;
        desc->flags = flags;
        desc->cond_flags = cond_flags;
out:
        return ret;
}

//-----------------------------------------------------------------------------

int gds_prepare_write_value32(gds_write_value32_t *desc, uint32_t *ptr, uint32_t value, int flags)
{
        int ret = 0;
        assert(desc);
        if (!is_valid(memtype_from_flags(flags))) {
                gds_err("invalid memory type in flags\n");
                ret = EINVAL;
                goto out;
        }
        if (flags & ~(GDS_WRITE_PRE_BARRIER_SYS|GDS_MEMORY_MASK)) {
                gds_err("invalid flags\n");
                ret = EINVAL;
                goto out;
        }
        desc->ptr = ptr;
        desc->value = value;
        desc->flags = flags;
out:
        return ret;
}

//-----------------------------------------------------------------------------

int gds_prepare_write_memory(gds_write_memory_t *desc, uint8_t *dest, const uint8_t *src, size_t count, int flags)
{
        int ret = 0;
        assert(desc);
        if (!is_valid(memtype_from_flags(flags))) {
                gds_err("invalid memory type in flags\n");
                ret = EINVAL;
                goto out;
        }
        if (flags & ~(GDS_WRITE_MEMORY_POST_BARRIER_SYS|GDS_WRITE_MEMORY_PRE_BARRIER_SYS|GDS_MEMORY_MASK)) {
                gds_err("invalid flags\n");
                ret = EINVAL;
                goto out;
        }
        desc->dest = dest;
        desc->src = src;
        desc->count = count;
        desc->flags = flags;
out:
        return ret;
}

//-----------------------------------------------------------------------------

static bool no_network_descs_after_entry(size_t n_descs, gds_descriptor_t *descs, size_t idx)
{
        bool ret = true;
        size_t i;
        for(i = idx+1; i < n_descs; ++i) {
                gds_descriptor_t *desc = descs + i;
                switch(desc->tag) {
                case GDS_TAG_SEND:
                case GDS_TAG_WAIT:
                        ret = false;
                        goto out;
                case GDS_TAG_WAIT_VALUE32:
                case GDS_TAG_WRITE_VALUE32:
                        break;
                default:
                        gds_err("invalid tag\n");
                        ret = EINVAL;
                        goto out;
                }
        }
out:
        return ret;
}

static int get_wait_info(size_t n_descs, gds_descriptor_t *descs, size_t &n_waits, size_t &last_wait)
{
        int ret = 0;
        size_t i;
        for(i = 0; i < n_descs; ++i) {
                gds_descriptor_t *desc = descs + i;
                switch(desc->tag) {
                case GDS_TAG_WAIT:
                        ++n_waits;
                        last_wait = i;
                        break;
                case GDS_TAG_SEND:
                case GDS_TAG_WAIT_VALUE32:
                case GDS_TAG_WRITE_VALUE32:
                case GDS_TAG_WRITE_MEMORY:
                        break;
                default:
                        gds_err("invalid tag\n");
                        ret = EINVAL;
                }
        }
        return ret;
}

static int calc_n_mem_ops(size_t n_descs, gds_descriptor_t *descs, size_t &n_mem_ops)
{
        int ret = 0;
        n_mem_ops = 0;
        size_t i;

        ret = gds_transport_init();
        if (ret) {
                gds_err("error in gds_transport_init\n");
                goto out;
        }

        for (i = 0; i < n_descs; ++i) {
                gds_descriptor_t *desc = descs + i;
                switch (desc->tag) {
                case GDS_TAG_SEND:
                        n_mem_ops += gds_main_transport->get_num_send_request_entries(desc->send) + 2; // extra space, ugly
                        break;
                case GDS_TAG_WAIT:
                        n_mem_ops += gds_main_transport->get_num_wait_request_entries(desc->wait) + 2; // ditto
                        break;
                case GDS_TAG_WAIT_VALUE32:
                case GDS_TAG_WRITE_VALUE32:
                case GDS_TAG_WRITE_MEMORY:
                        n_mem_ops += 2; // ditto
                        break;
                default:
                        gds_err("invalid tag\n");
                        ret = EINVAL;
                }
        }

out:
        return ret;
}

int gds_stream_post_descriptors(CUstream stream, size_t n_descs, gds_descriptor_t *descs, int flags)
{
        size_t i;
        int idx = 0;
        int ret = 0;
        int retcode = 0;
        size_t n_mem_ops = 0;
        size_t n_waits = 0;
        size_t last_wait = 0;
        bool move_flush = false;
        gds_peer *peer = NULL;
        gds_op_list_t params;

        ret = gds_transport_init();
        if (ret) {
                gds_err("error in gds_transport_init\n");
                goto out;
        }

        ret = calc_n_mem_ops(n_descs, descs, n_mem_ops);
        if (ret) {
                gds_err("error %d in calc_n_mem_ops\n", ret);
                goto out;
        }

        ret = get_wait_info(n_descs, descs, n_waits, last_wait);
        if (ret) {
                gds_err("error %d in get_wait_info\n", ret);
                goto out;
        }

        gds_dbg("n_descs=%zu n_waits=%zu n_mem_ops=%zu\n", n_descs, n_waits, n_mem_ops);

        // move flush to last wait in the whole batch
        if (n_waits && no_network_descs_after_entry(n_descs, descs, last_wait)) {
                gds_dbg("optimizing FLUSH to last wait i=%zu\n", last_wait);
                move_flush = true;
        }
        // alternatively, remove flush for wait is next op is a wait too

        peer = peer_from_stream(stream);
        if (!peer) {
                return EINVAL;
        }

        for (i = 0; i < n_descs; ++i) {
                gds_descriptor_t *desc = descs + i;
                switch (desc->tag) {
                case GDS_TAG_SEND: {
                        retcode = gds_main_transport->post_send_ops(peer, desc->send, params);
                        if (retcode) {
                                gds_err("error %d in gds_post_ops\n", retcode);
                                ret = retcode;
                                goto out;
                        }
                        break;
                }
                case GDS_TAG_WAIT: {
                        int flags = 0;
                        if (move_flush && i != last_wait) {
                                gds_dbg("discarding FLUSH!\n");
                                flags = GDS_POST_OPS_DISCARD_WAIT_FLUSH;
                        }
                        retcode = gds_main_transport->stream_post_wait_descriptor(peer, desc->wait, params, flags);
                        if (retcode) {
                                gds_err("error %d in stream_post_wait_descriptor\n", retcode);
                                ret = retcode;
                                goto out;
                        }
                        break;
                }
                case GDS_TAG_WAIT_VALUE32:
                        retcode = gds_fill_poll(peer, params, desc->wait32.ptr, desc->wait32.value, desc->wait32.cond_flags, desc->wait32.flags);
                        if (retcode) {
                                gds_err("error %d in gds_fill_poll\n", retcode);
                                ret = retcode;
                                goto out;
                        }
                        break;
                case GDS_TAG_WRITE_VALUE32:
                        retcode = gds_fill_poke(peer, params, desc->write32.ptr, desc->write32.value, desc->write32.flags);
                        if (retcode) {
                                gds_err("error %d in gds_fill_poke\n", retcode);
                                ret = retcode;
                                goto out;
                        }
                        break;
                case GDS_TAG_WRITE_MEMORY:
                        retcode = gds_fill_inlcpy(peer, params, desc->writemem.dest, desc->writemem.src, desc->writemem.count, desc->writemem.flags);
                        if (retcode) {
                                gds_err("error %d in gds_fill_inlcpy\n", retcode);
                                ret = retcode;
                                goto out;
                        }
                        break;
                default:
                        gds_err("invalid tag for %zu entry\n", i);
                        ret = EINVAL;
                        goto out;
                        break;
                }
        }
        retcode = gds_stream_batch_ops(peer, stream, params, 0);
        if (retcode) {
                gds_err("error %d in gds_stream_batch_ops\n", retcode);
                ret = retcode;
                goto out;
        }

out:
        return ret;
}

//-----------------------------------------------------------------------------

int gds_post_descriptors(size_t n_descs, gds_descriptor_t *descs, int flags)
{
        size_t i;
        int ret = 0;
        int retcode = 0;

        ret = gds_transport_init();
        if (ret) {
                gds_err("error in gds_transport_init\n");
                goto out;
        }

        for(i = 0; i < n_descs; ++i) {
                gds_descriptor_t *desc = descs + i;
                switch(desc->tag) {
                case GDS_TAG_SEND: {
                        gds_dbg("desc[%zu] SEND\n", i);
                        retcode = gds_main_transport->post_send_ops_on_cpu(desc->send, flags);
                        if (retcode) {
                                gds_err("error %d in gds_post_ops_on_cpu\n", retcode);
                                ret = retcode;
                                goto out;
                        }
                        break;
                }
                case GDS_TAG_WAIT: {
                        gds_dbg("desc[%zu] WAIT\n", i);
                        retcode = gds_main_transport->post_wait_descriptor(desc->wait, flags);
                        if (retcode) {
                                gds_err("error %d in post_wait_descriptor\n", retcode);
                                ret = retcode;
                                goto out;
                        }
                        break;
                }
                case GDS_TAG_WAIT_VALUE32: {
                        gds_dbg("desc[%zu] WAIT_VALUE32\n", i);
                        uint32_t *ptr = desc->wait32.ptr;
                        uint32_t value = desc->wait32.value;
                        bool flush = false;
                        if (desc->wait32.flags & GDS_WAIT_POST_FLUSH_REMOTE) {
                                gds_err("GDS_WAIT_POST_FLUSH_REMOTE flag is not supported yet\n");
                                flush = true;
                        }
                        gds_memory_type_t mem_type = (gds_memory_type_t)(desc->wait32.flags & GDS_MEMORY_MASK);
                        switch(mem_type) {
                        case GDS_MEMORY_GPU:
                                // dereferencing ptr may fail if ptr points to CUDA device memory
                        case GDS_MEMORY_HOST:
                        case GDS_MEMORY_IO:
                                break;
                        default:
                                gds_err("invalid memory type 0x%02x in WAIT_VALUE32\n", mem_type);
                                ret = EINVAL;
                                goto out;
                                break;
                        }
                        bool done = false;
                        do {
                                uint32_t data = gds_atomic_get(ptr);
                                switch(desc->wait32.cond_flags) {
                                case GDS_WAIT_COND_GEQ:
                                        done = ((int32_t)data - (int32_t)value >= 0);
                                        break;
                                case GDS_WAIT_COND_EQ:
                                        done = (data == value);
                                        break;
                                case GDS_WAIT_COND_AND:
                                        done = (data & value);
                                        break;
                                case GDS_WAIT_COND_NOR:
                                        done = ~(data | value);
                                        break;
                                default:
                                        gds_err("invalid condition flags 0x%02x in WAIT_VALUE32\n", desc->wait32.cond_flags);
                                        goto out;
                                        break;
                                }
                                if (done)
                                        break;
                                // TODO: more aggressive CPU relaxing needed here to avoid starving I/O fabric
                                arch_cpu_relax();
                        } while(true);
                        break;
                }
                case GDS_TAG_WRITE_VALUE32: {
                        gds_dbg("desc[%zu] WRITE_VALUE32\n", i);
                        uint32_t *ptr = desc->write32.ptr;
                        uint32_t value = desc->write32.value;
                        gds_memory_type_t mem_type = (gds_memory_type_t)(desc->write32.flags & GDS_MEMORY_MASK);
                        switch(mem_type) {
                        case GDS_MEMORY_GPU:
                                // dereferencing ptr may fail if ptr points to CUDA device memory
                        case GDS_MEMORY_HOST:
                        case GDS_MEMORY_IO:
                                break;
                        default:
                                gds_err("invalid memory type 0x%02x in WRITE_VALUE32\n", mem_type);
                                ret = EINVAL;
                                goto out;
                                break;
                        }
                        bool barrier = (desc->write32.flags & GDS_WRITE_PRE_BARRIER_SYS);
                        if (barrier)
                                wmb();
                        gds_atomic_set(ptr, value);
                        break;
                }
                case GDS_TAG_WRITE_MEMORY: {
                        void *dest = desc->writemem.dest;
                        const void *src = desc->writemem.src;
                        size_t nbytes = desc->writemem.count;
                        bool barrier = (desc->writemem.flags & GDS_WRITE_MEMORY_POST_BARRIER_SYS);
                        gds_memory_type_t mem_type = memtype_from_flags(desc->writemem.flags);
                        gds_dbg("desc[%zu] WRITE_MEMORY dest=%p src=%p len=%zu memtype=%02x\n", i, dest, src, nbytes, mem_type);
                        switch(mem_type) {
                        case GDS_MEMORY_GPU:
                        case GDS_MEMORY_HOST:
                                memcpy(dest, src, nbytes);
                                break;
                        case GDS_MEMORY_IO:
                                assert(nbytes % sizeof(uint64_t));
                                assert(((unsigned long)dest & 0x7) == 0);
                                gds_bf_copy((uint64_t*)dest, (uint64_t*)src, nbytes);
                                break;
                        default:
                                assert(!"invalid mem type");
                                break;
                        }
                        if (barrier)
                                wmb();
                        break;
                }
                default:
                        gds_err("invalid tag for %zu entry\n", i);
                        ret = EINVAL;
                        goto out;
                        break;
                }
        }
out:
        return ret;
}

//-----------------------------------------------------------------------------

int gds_poll_cq(struct gds_cq *gcq, int num_entries, struct ibv_wc *wc)
{
        if (!gcq)
                return EINVAL;

        if (num_entries < 0)
                return EINVAL;

        if (num_entries > 0 && !wc)
                return EINVAL;

        return gds_main_transport->poll_cq(gcq, num_entries, wc);
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
