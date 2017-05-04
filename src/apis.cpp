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
int gds_rollback_qp(struct gds_qp *qp, gds_send_request_t * send_info, int flags)
{
    struct ibv_exp_rollback_ctx rollback;
    int ret=0;

    assert(qp);
    assert(qp->qp);
    assert(send_info);
    if(
        flags != IBV_EXP_ROLLBACK_ABORT_UNCOMMITED && 
        flags != IBV_EXP_ROLLBACK_ABORT_LATE
    )
    {
        gds_err("erroneous rollback flag input value\n");
        goto out;
    } 

    /* from ibv_exp_peer_commit call */
    rollback.rollback_id = send_info->commit.rollback_id;
    /* from ibv_exp_rollback_flags */
    rollback.flags = flags;
    /* Reserved for future expensions, must be 0 */
    rollback.comp_mask = 0;
    gds_warn("Need to rollback WQE %x\n", rollback.rollback_id);
    ret = ibv_exp_rollback_qp(qp->qp, &rollback);
    if(ret)
    {
        gds_err("error %d in ibv_exp_rollback_qp\n", ret);
    }

out:
    return ret;
}

int gds_post_send(struct gds_qp *qp, struct ibv_exp_send_wr *p_ewr, struct ibv_exp_send_wr **bad_ewr)
{
    int ret = 0;
    gds_send_request_t send_info;
    ret = gds_prepare_send(qp, p_ewr, bad_ewr, &send_info);
    if (ret) {
        gds_err("error %d in gds_prepare_send\n", ret);
        goto out;
    }

    ret = gds_post_pokes_on_cpu(1, &send_info, NULL, 0);
    if (ret) {
        gds_err("error %d in gds_post_pokes_on_cpu\n", ret);
        //the request has been committed here
        gds_rollback_qp(qp, &send_info, IBV_EXP_ROLLBACK_ABORT_LATE);
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
                        //request not commited yet!
                }
                goto out;
        }
        
        ret = ibv_exp_peer_commit_qp(qp->qp, &request->commit);
        if (ret) {
                gds_err("error %d in ibv_exp_peer_commit_qp\n", ret);
                //request not commited in case of error
                
                //gds_wait_kernel();
                goto out;
        }
out:
        return ret;
}

//-----------------------------------------------------------------------------

int gds_stream_queue_send_ex(CUstream stream, struct gds_qp *qp, struct ibv_exp_send_wr *p_ewr, struct ibv_exp_send_wr **bad_ewr, uint32_t *dw, uint32_t val)
{
        int ret = 0;
	gds_send_request_t send_info;

        ret = gds_prepare_send(qp, p_ewr, bad_ewr, &send_info);
        if (ret) {
                goto out;
        }

        ret = gds_post_pokes(stream, 1, &send_info, dw, val);
        if (ret) {
            //the request has been committed here
            gds_rollback_qp(qp, &send_info, IBV_EXP_ROLLBACK_ABORT_LATE);
            goto out;
        }
out:
        return ret;
}

//-----------------------------------------------------------------------------

int gds_stream_queue_send(CUstream stream, struct gds_qp *qp, struct ibv_exp_send_wr *p_ewr, struct ibv_exp_send_wr **bad_ewr)
{
        return gds_stream_queue_send_ex(stream, qp, p_ewr, bad_ewr, NULL, 0);
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

int gds_stream_post_send_ex(CUstream stream, gds_send_request_t *request, uint32_t *dw, uint32_t val)
{
    int ret = 0;

    //struct ibv_exp_send_ex_info *info = (struct ibv_exp_send_ex_info *) request;

    ret = gds_post_pokes(stream, 1, request, dw, val);
    if (ret) {
            gds_err("error in gds_post_pokes (%d)\n", ret);
    }
    return ret;
}

//-----------------------------------------------------------------------------

int gds_stream_queue_recv(CUstream stream, struct gds_qp *qp, struct ibv_recv_wr *p_ewr, struct ibv_recv_wr **bad_ewr)
{
        int ret = 0;
#if 0
	struct ibv_exp_recv_ex_info send_info;
        gds_dbg("calling gds_stream_queue_recv()\n");
        ret = ibv_exp_post_recv_ex(qp, p_ewr, bad_ewr, &recv_info);
        if (ret) {
                gds_err("ibv_exp_post_recv_ex (%d)\n", ret);
                //gds_wait_kernel();
                return ret;
        }
        ret = gds_post_pokes(stream, &send_info);
#else
        gds_err("queue_recv not implemented\n");
	ret = EINVAL;
#endif
        return ret;
}

//-----------------------------------------------------------------------------

int gds_prepare_wait_cq(struct gds_cq *cq, gds_wait_request_t *request, int flags)
{
	int retcode = 0;

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

int gds_stream_post_wait_cq_ex(CUstream stream, gds_wait_request_t *request, uint32_t *dw, uint32_t val)
{
	return gds_stream_post_wait_cq_multi(stream, 1, request, dw, val);
}

//-----------------------------------------------------------------------------

int gds_stream_post_wait_cq_all(CUstream stream, int count, gds_wait_request_t *requests)
{
	return gds_stream_post_wait_cq_multi(stream, count, requests, NULL, 0);
}

//-----------------------------------------------------------------------------

int gds_stream_wait_cq_ex(CUstream stream, struct gds_cq *cq, int flag, uint32_t *dw, uint32_t val)
{
        int retcode = 0;
        int ret;
        gds_wait_request_t request;

        assert(cq);
        assert(stream);

        ret = gds_prepare_wait_cq(cq, &request, flag);
        if (ret) {
                gds_err("error %d in gds_prepare_wait_cq\n", ret);
                goto out;
        }

	ret = gds_stream_post_wait_cq_ex(stream, &request, dw, val);
        if (ret) {
                gds_err("error %d in gds_stream_post_wait_cq_ex\n", ret);
                retcode = ret;
                goto out;
        }

out:
	return retcode;
}

//-----------------------------------------------------------------------------

int gds_stream_wait_cq(CUstream stream, struct gds_cq *cq, int flags)
{
        return gds_stream_wait_cq_ex(stream, cq, flags, NULL, 0);
}

//-----------------------------------------------------------------------------

int gds_post_wait_cq(struct gds_cq *cq, gds_wait_request_t *request, int flags)
{
        assert(cq);
        assert(request);
        struct ibv_exp_peer_abort_peek abort_ctx;
        abort_ctx.peek_id = request->peek.peek_id;
        abort_ctx.comp_mask = 0;
        return ibv_exp_peer_abort_peek_cq(cq->cq, &abort_ctx);
}

//-----------------------------------------------------------------------------

int gds_prepare_wait_value32(uint32_t *ptr, uint32_t value, int cond_flags, int flags, gds_value32_descriptor_t *desc)
{
        int ret = 0;
        assert(desc);
        desc->ptr = ptr;
        desc->value = value;
        desc->flags = flags;
        desc->cond_flags = cond_flags;
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
