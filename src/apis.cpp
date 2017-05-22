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
//#include <infiniband/verbs_exp.h>
//#include <gdrapi.h>

#include "gdsync.h"
#include "gdsync/tools.h"
#include "objs.hpp"
#include "utils.hpp"
#include "memmgr.hpp"
//#include "mem.hpp"


//-----------------------------------------------------------------------------

static void gds_init_ops(struct peer_op_wr *op, int count)
{
        int i = count;
        while (--i)
                op[i-1].next = &op[i];
        op[count-1].next = NULL;
}

//-----------------------------------------------------------------------------

static void gds_init_send_info(gds_send_request_t *info)
{
        gds_dbg("send_request=%p\n", info);
        memset(info, 0, sizeof(*info));

        info->commit.storage = info->wr;
        info->commit.entries = sizeof(info->wr)/sizeof(info->wr[0]);
        gds_init_ops(info->commit.storage, info->commit.entries);
}

//-----------------------------------------------------------------------------

static void gds_init_wait_request(gds_wait_request_t *request, uint32_t offset)
{
        gds_dbg("wait_request=%p offset=%08x\n", request, offset);
        memset(request, 0, sizeof(*request));
        request->peek.storage = request->wr;
        request->peek.entries = sizeof(request->wr)/sizeof(request->wr[0]);
        request->peek.whence = IBV_EXP_PEER_PEEK_ABSOLUTE;
        request->peek.offset = offset;
        gds_init_ops(request->peek.storage, request->peek.entries);
}

//-----------------------------------------------------------------------------

int gds_post_send(struct gds_qp *qp, struct ibv_send_wr *wr, struct ibv_send_wr **bad_wr)
{
        int ret = 0;
	gds_send_request_t send_info;
        gds_init_send_info(&send_info);

        gds_dbg("qp=%p wr=%p\n", qp, wr);
        assert(qp);
        assert(qp->qp);
        ret = ibv_post_send(qp->qp, wr, bad_wr);
        if (ret) {
                gds_err("error %d in gds_post_send\n", ret);
                goto out;
        }

        ret = ibv_exp_peer_commit_qp(qp->qp, &send_info.commit);
        if (ret) {
                gds_err("error %d in ibv_exp_peer_commit_qp\n", ret);
                goto out;
        }
        
        ret = gds_post_pokes_on_cpu(1, &send_info, NULL, 0);
        if (ret) {
                gds_err("error %d in gds_post_pokes_on_cpu\n", ret);
                goto out;
        }

out:
        return ret;
}

//-----------------------------------------------------------------------------

int gds_post_recv(struct gds_qp *qp, struct ibv_recv_wr *wr, struct ibv_recv_wr **bad_wr)
{
        int ret = 0;

        gds_dbg("qp=%p wr=%p\n", qp, wr);
        assert(qp);
        assert(qp->qp);
        ret = ibv_post_recv(qp->qp, wr, bad_wr);
        if (ret) {
                gds_err("error %d in ibv_post_recv\n", ret);
                goto out;
        }

out:
        return ret;
}

//-----------------------------------------------------------------------------

int gds_prepare_send(struct gds_qp *qp, struct ibv_exp_send_wr *p_ewr, 
                     struct ibv_exp_send_wr **bad_ewr, 
                     gds_send_request_t *request)
{
        int ret = 0;
        gds_init_send_info(request);
        assert(qp);
        assert(qp->qp);
        ret = ibv_exp_post_send(qp->qp, p_ewr, bad_ewr);
        if (ret) {

                if (ret == ENOMEM) {
                        // out of space error can happen too often to report
                        gds_dbg("ENOMEM error %d in ibv_exp_post_send\n", ret);
                } else {
                        gds_err("error %d in ibv_exp_post_send\n", ret);
                }
                goto out;
        }
        
        ret = ibv_exp_peer_commit_qp(qp->qp, &request->commit);
        if (ret) {
                gds_err("error %d in ibv_exp_peer_commit_qp\n", ret);
                //gds_wait_kernel();
                goto out;
        }
out:
        return ret;
}

//-----------------------------------------------------------------------------

int gds_stream_queue_send(CUstream stream, struct gds_qp *qp, struct ibv_exp_send_wr *p_ewr, struct ibv_exp_send_wr **bad_ewr)
{
        int ret = 0;
	gds_send_request_t send_info;

        ret = gds_prepare_send(qp, p_ewr, bad_ewr, &send_info);
        if (ret) {
                goto out;
        }

        ret = gds_post_pokes(stream, 1, &send_info, NULL, 0);
        if (ret) {
                goto out;
        }
out:
        return ret;
}

//-----------------------------------------------------------------------------

int gds_stream_post_send(CUstream stream, gds_send_request_t *request)
{
    int ret = 0;
    //struct ibv_exp_send_ex_info *info = (struct ibv_exp_send_ex_info *) request;
    ret = gds_post_pokes(stream, 1, request, NULL, 0);
    if (ret) {
            gds_err("gds_post_pokes (%d)\n", ret);
    }
    return ret;
}

//-----------------------------------------------------------------------------

int gds_stream_post_send_all(CUstream stream, int count, gds_send_request_t *request)
{
    int ret = 0;

    //struct ibv_exp_send_ex_info *info = (struct ibv_exp_send_ex_info *) request;

    ret = gds_post_pokes(stream, count, request, NULL, 0);
    if (ret) {
            gds_err("error in gds_post_pokes (%d)\n", ret);
    }
    return ret;
}

//-----------------------------------------------------------------------------

int gds_prepare_wait_cq(struct gds_cq *cq, gds_wait_request_t *request, int flags)
{
	int retcode = 0;

        if (flags != 0) {
                gds_err("invalid flags != 0\n");
                return EINVAL;
        }

        gds_init_wait_request(request, cq->curr_offset++);

        retcode = ibv_exp_peer_peek_cq(cq->cq, &request->peek);
        if (retcode == -ENOSPC) {
                // TODO: handle too few entries
                gds_err("not enough ops in peer_peek_cq\n");
                goto out;
        } else if (retcode) {
                gds_err("error %d in peer_peek_cq\n", retcode);
                goto out;
        }
        //gds_dump_wait_request(request, 1);
out:

	return retcode;
}

//-----------------------------------------------------------------------------

int gds_append_wait_cq(gds_wait_request_t *request, uint32_t *dw, uint32_t val)
{
	int ret = 0;
        unsigned MAX_NUM_ENTRIES = sizeof(request->wr)/sizeof(request->wr[0]);
        unsigned n = request->peek.entries;
        struct peer_op_wr *wr = request->peek.storage;

        if (n + 1 > MAX_NUM_ENTRIES) {
		gds_err("no space left to stuff a poke\n");
		ret = EINVAL;
		goto out;
	}

        // at least 1 op
        assert(n);
        assert(wr);

        for (; n; --n) wr = wr->next;
        assert(wr);

	wr->type = IBV_EXP_PEER_OP_STORE_DWORD;
	wr->wr.dword_va.data = val;
	wr->wr.dword_va.target_id = 0; // direct mapping, offset IS the address
	wr->wr.dword_va.offset = (ptrdiff_t)(dw-(uint32_t*)0);

        ++request->peek.entries;

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
        struct ibv_exp_peer_abort_peek abort_ctx;
        abort_ctx.peek_id = request->peek.peek_id;
        abort_ctx.comp_mask = 0;
        return ibv_exp_peer_abort_peek_cq(cq->cq, &abort_ctx);
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

        if (flags & ~(GDS_WAIT_POST_FLUSH|GDS_MEMORY_MASK)) {
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
        if (flags & ~(GDS_WRITE_PRE_BARRIER|GDS_MEMORY_MASK)) {
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

int gds_stream_post_poll_dword(CUstream stream, uint32_t *ptr, uint32_t magic, gds_wait_cond_flag_t cond_flags, int flags)
{
        int retcode = 0;
	CUstreamBatchMemOpParams param[1];
        retcode = gds_fill_poll(param, ptr, magic, cond_flags, flags);
        if (retcode) {
                gds_err("error in fill_poll\n");
                goto out;
        }
        retcode = gds_stream_batch_ops(stream, 1, param, 0);
        if (retcode) {
                gds_err("error in batch_ops\n");
                goto out;
        }
out:
        return retcode;
}

//-----------------------------------------------------------------------------

int gds_stream_post_poke_dword(CUstream stream, uint32_t *ptr, uint32_t value, int flags)
{
        int retcode = 0;
	CUstreamBatchMemOpParams param[1];
        retcode = gds_fill_poke(param, ptr, value, flags);
        if (retcode) {
                gds_err("error in fill_poke\n");
                goto out;
        }
        retcode = gds_stream_batch_ops(stream, 1, param, 0);
        if (retcode) {
                gds_err("error in batch_ops\n");
                goto out;
        }
out:
        return retcode;
}

//-----------------------------------------------------------------------------

int gds_stream_post_inline_copy(CUstream stream, void *ptr, void *src, size_t nbytes, int flags)
{
        int retcode = 0;
	CUstreamBatchMemOpParams param[1];

        retcode = gds_fill_inlcpy(param, ptr, src, nbytes, flags);
        if (retcode) {
                gds_err("error in fill_poke\n");
                goto out;
        }
        retcode = gds_stream_batch_ops(stream, 1, param, 0);
        if (retcode) {
                gds_err("error in batch_ops\n");
                goto out;
        }
out:
        return retcode;
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
                        break;
                default:
                        gds_err("invalid tag\n");
                        ret = EINVAL;
                }
        }
        return ret;
}

static size_t calc_n_mem_ops(size_t n_descs, gds_descriptor_t *descs)
{
        size_t n_mem_ops = 0;
        size_t i;
        for(i = 0; i < n_descs; ++i) {
                gds_descriptor_t *desc = descs + i;
                switch(desc->tag) {
                case GDS_TAG_SEND:
                        n_mem_ops += desc->send->commit.entries + 2; // extra space, ugly
                        break;
                case GDS_TAG_WAIT:
                        n_mem_ops += desc->wait->peek.entries + 2; // ditto
                        break;
                case GDS_TAG_WAIT_VALUE32:
                case GDS_TAG_WRITE_VALUE32:
                        n_mem_ops += 2; // ditto
                        break;
                default:
                        gds_err("invalid tag\n");
                }
        }
        return n_mem_ops;
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

        n_mem_ops = calc_n_mem_ops(n_descs, descs);
        get_wait_info(n_descs, descs, n_waits, last_wait);

        gds_dbg("n_descs=%zu n_waits=%zu n_mem_ops=%zu\n", n_descs, n_waits, n_mem_ops);

        // move flush to last wait in the whole batch
        if (n_waits && no_network_descs_after_entry(n_descs, descs, last_wait)) {
                gds_dbg("optimizing FLUSH to last wait i=%zu\n", last_wait);
                move_flush = true;
        }
        // alternatively, remove flush for wait is next op is a wait too

        CUstreamBatchMemOpParams params[n_mem_ops];

        for(i = 0; i < n_descs; ++i) {
                gds_descriptor_t *desc = descs + i;
                switch(desc->tag) {
                case GDS_TAG_SEND: {
                        gds_send_request_t *sreq = desc->send;
                        retcode = gds_post_ops(sreq->commit.entries, sreq->commit.storage, params, idx);
                        if (retcode) {
                                gds_err("error %d in gds_post_ops\n", retcode);
                                ret = retcode;
                                goto out;
                        }
                        // TODO: fix late checking
                        //assert(idx <= n_mem_ops);
                        if (idx >= n_mem_ops) {
                                gds_err("idx=%d is past allocation (%zu)\n", idx, n_mem_ops);
                                assert(!"corrupted heap");
                        }
                        break;
                }
                case GDS_TAG_WAIT: {
                        gds_wait_request_t *wreq = desc->wait;
                        int flags = 0;
                        if (move_flush && i != last_wait)
                                flags = GDS_POST_OPS_DISCARD_WAIT_FLUSH;
                        retcode = gds_post_ops(wreq->peek.entries, wreq->peek.storage, params, idx, flags);
                        if (retcode) {
                                gds_err("error %d in gds_post_ops\n", retcode);
                                ret = retcode;
                                goto out;
                        }
                        // TODO: fix late checking
                        assert(idx <= n_mem_ops);
                        break;
                }
                case GDS_TAG_WAIT_VALUE32:
                        retcode = gds_fill_poll(params+idx, desc->wait32.ptr, desc->wait32.value, desc->wait32.cond_flags, desc->wait32.flags);
                        if (retcode) {
                                gds_err("error %d in gds_fill_poll\n", retcode);
                                ret = retcode;
                                goto out;
                        }
                        ++idx;
                        break;
                case GDS_TAG_WRITE_VALUE32:
                        retcode = gds_fill_poke(params+idx, desc->write32.ptr, desc->write32.value, desc->write32.flags);
                        if (retcode) {
                                gds_err("error %d in gds_fill_poke\n", retcode);
                                ret = retcode;
                                goto out;
                        }
                        ++idx;
                        break;
                default:
                        assert(0);
                        break;
                }
        }
        retcode = gds_stream_batch_ops(stream, idx, params, 0);
        if (retcode) {
                gds_err("error in batch_ops\n");
                goto out;
        }

out:
        return ret;
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
