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
#include <netinet/in.h>

#include "gdsync.h"
#include "gdsync/tools.h"
#include "objs.hpp"
#include "utils.hpp"
#include "memmgr.hpp"
#include "utils.hpp"
#include "archutils.h"
#include "mlnxutils.h"


//-----------------------------------------------------------------------------

static void gds_init_ops(struct gds_peer_op_wr *op, int count)
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
        request->peek.whence = GDS_PEER_PEEK_ABSOLUTE;
        request->peek.offset = offset;
        gds_init_ops(request->peek.storage, request->peek.entries);
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

static inline int set_datagram_seg(struct mlx5_wqe_datagram_seg *seg, gds_send_wr *wr)
{
        int ret = 0;
        mlx5dv_obj dv_obj;
        mlx5dv_ah dv_ah;

        memset(&dv_ah, 0, sizeof(mlx5dv_ah));
        dv_obj.ah.in = wr->wr.ud.ah;
        dv_obj.ah.out = &dv_ah;

        ret = mlx5dv_init_obj(&dv_obj, MLX5DV_OBJ_AH);
        if (ret || dv_ah.av == NULL) {
                gds_err("Error %d in mlx5dv_init_obj(..., MLX5DV_OBJ_AH)\n", ret);
                return ret;
        }

        memcpy(&seg->av, dv_ah.av, sizeof(struct mlx5_wqe_av));
        seg->av.dqp_dct = htobe32(wr->wr.ud.remote_qpn | MLX5_EXTENDED_UD_AV);
        seg->av.key.qkey.qkey = htobe32(wr->wr.ud.remote_qkey);
        return ret;
}

//-----------------------------------------------------------------------------

int gds_prepare_send(struct gds_qp *qp, gds_send_wr *p_ewr, 
                gds_send_wr **bad_ewr, 
                gds_send_request_t *request)
{
        int ret = 0;
        unsigned int idx;
        int size;
        void *seg;
        void *qend;
        int i;
        int nreq;

        struct mlx5_wqe_ctrl_seg *ctrl_seg;
        struct mlx5_wqe_data_seg *data_seg;

        struct gds_peer_op_wr *wr;

        if (p_ewr->opcode != IBV_WR_SEND) {
                gds_err("Unsupported opcode. Currently we support only IBV_WR_SEND\n");
                ret = -1;
                goto out;
        }

        gds_init_send_info(request);
        assert(qp);
        assert(qp->qp);

        // Emulating ibv_exp_post_send
        qend = (void *)((char *)qp->dv_qp.sq.buf + (qp->dv_qp.sq.wqe_cnt * qp->dv_qp.sq.stride));
        for (nreq = 0; p_ewr; ++nreq, p_ewr = p_ewr->next) {
                idx = qp->sq_cur_post & (qp->dv_qp.sq.wqe_cnt - 1);
                seg = (void *)((char *)qp->dv_qp.sq.buf + (idx * qp->dv_qp.sq.stride));

                ctrl_seg = (struct mlx5_wqe_ctrl_seg *)seg;
                size = sizeof(struct mlx5_wqe_ctrl_seg) / 16;
                seg = (void *)((char *)seg + sizeof(struct mlx5_wqe_ctrl_seg));

                switch (qp->qp->qp_type) {
                        case IBV_QPT_UD:
                                ret = set_datagram_seg((struct mlx5_wqe_datagram_seg *)seg, p_ewr);
                                if (ret)
                                        goto out;
                                size = sizeof(struct mlx5_wqe_datagram_seg) / 16;
                                seg = (void *)((char *)seg + sizeof(struct mlx5_wqe_datagram_seg));
                                if (seg == qend)
                                        seg = qp->dv_qp.sq.buf;
                                break;
                        case IBV_QPT_RC:
                                break;
                        default:
                                gds_err("Encountered unsupported qp_type. We currently support only IBV_QPT_UD\n");
                                ret = -2;
                                goto out;
                }

                data_seg = (struct mlx5_wqe_data_seg *)seg;
                for (i = 0; i < p_ewr->num_sge; ++i) {
                        if (data_seg == qend) {
                                seg = qp->dv_qp.sq.buf;
                                data_seg = (struct mlx5_wqe_data_seg *)seg;
                        }
                        if (p_ewr->sg_list[i].length) {
                                mlx5dv_set_data_seg(data_seg, p_ewr->sg_list[i].length, p_ewr->sg_list[i].lkey, p_ewr->sg_list[i].addr);
                                ++data_seg;
                                size += sizeof(struct mlx5_wqe_data_seg) / 16;
                        }
                }

                mlx5dv_set_ctrl_seg(ctrl_seg, qp->sq_cur_post & 0xffff, MLX5_OPCODE_SEND, 0, qp->qp->qp_num, MLX5_WQE_CTRL_CQ_UPDATE, size, 0, 0);

                qp->send_cq.wrid[idx] = p_ewr->wr_id;
                qp->sq_cur_post += (size * 16 + qp->dv_qp.sq.stride - 1) / qp->dv_qp.sq.stride;
        }

        if (!nreq)
                goto out;

        // Emulating ibv_exp_peer_commit_qp
        assert(request->commit.entries >= 3);

        request->commit.rollback_id = qp->peer_scur_post | ((uint64_t)qp->sq_cur_post << 32);
        qp->peer_scur_post = qp->sq_cur_post;

        wr = request->commit.storage;

        wr->type = GDS_PEER_OP_STORE_DWORD;
        wr->wr.dword_va.data = htonl(qp->sq_cur_post & 0xffff);
        wr->wr.dword_va.target_id = qp->peer_va_id_dbr;
        wr->wr.dword_va.offset = sizeof(uint32_t) * MLX5_SND_DBR;
        wr = wr->next;

        wr->type = GDS_PEER_OP_FENCE;
        wr->wr.fence.fence_flags = GDS_PEER_FENCE_OP_WRITE | GDS_PEER_FENCE_FROM_HCA | GDS_PEER_FENCE_MEM_SYS;
        wr = wr->next;

        wr->type = GDS_PEER_OP_STORE_QWORD;
        wr->wr.qword_va.data = *(__be64 *)ctrl_seg;
        wr->wr.qword_va.target_id = qp->peer_va_id_bf;
        wr->wr.qword_va.offset = qp->bf_offset;

        qp->bf_offset ^= qp->dv_qp.bf.size;
        request->commit.entries = 3;

out:
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

        ret = gds_stream_post_descriptors(stream, 1, descs, 0);
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
        if (!descs)
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
        int retcode = 0;

        gds_peer_attr *peer_attr = (gds_peer_attr *)cq->peer_attr;
        struct gds_peer_op_wr *wr;
        struct gds_peek_entry *peek;
        int n, cur_own;
        void *cqe;
        gds_cqe64 *cqe64;

        if (flags != 0) {
                gds_err("invalid flags != 0\n");
                return EINVAL;
        }

        gds_init_wait_request(request, cq->curr_offset++);

        // Emulating ibv_exp_peer_peek_cq
        assert(request->peek.entries >= 2);

        wr = request->peek.storage;
        n = request->peek.offset;

        cqe = (char *)cq->dv_cq.buf + (n & (cq->dv_cq.cqe_cnt - 1)) * cq->dv_cq.cqe_size;
        cur_own = n & cq->dv_cq.cqe_cnt;
        cqe64 = (gds_cqe64 *)((cq->dv_cq.cqe_size == 64) ? cqe : (char *)cqe + 64);

        if (cur_own) {
                wr->type = GDS_PEER_OP_POLL_AND_DWORD;
                wr->wr.dword_va.data = htonl(MLX5_CQE_OWNER_MASK);
        }
        else if (peer_attr->caps & GDS_PEER_OP_POLL_NOR_DWORD_CAP) {
                wr->type = GDS_PEER_OP_POLL_NOR_DWORD;
                wr->wr.dword_va.data = ~htonl(MLX5_CQE_OWNER_MASK);
        }
        else if (peer_attr->caps & GDS_PEER_OP_POLL_GEQ_DWORD_CAP) {
                wr->type = GDS_PEER_OP_POLL_GEQ_DWORD;
                wr->wr.dword_va.data = 0;
        }
        wr->wr.dword_va.target_id = cq->active_buf_va_id;
        wr->wr.dword_va.offset = (uintptr_t)&cqe64->wqe_counter - (uintptr_t)cq->dv_cq.buf;
        wr = wr->next;

        peek = cq->peer_peek_free;
        if (!peek) {
                return ENOMEM;
        }
        cq->peer_peek_free = GDS_PEEK_ENTRY(cq, peek->next);
        peek->busy = 1;
        mb();
        peek->next = GDS_PEEK_ENTRY_N(cq, cq->peer_peek_table[n & (cq->dv_cq.cqe_cnt - 1)]);
        cq->peer_peek_table[n & (cq->dv_cq.cqe_cnt - 1)] = peek;

        wr->type = GDS_PEER_OP_STORE_DWORD;
        wr->wr.dword_va.data = 0;
        wr->wr.dword_va.target_id = cq->peer_va_id;
        wr->wr.dword_va.offset = (uintptr_t)&peek->busy - (uintptr_t)cq->peer_buf.buf;

        request->peek.entries = 2;
        request->peek.peek_id = (uintptr_t)peek;

out:
        return retcode;
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
        ((struct gds_peek_entry *)request->peek.peek_id)->busy = 0;
        return 0;
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
                        case GDS_TAG_WRITE_MEMORY:
                                n_mem_ops += 2; // ditto
                                break;
                        default:
                                gds_err("invalid tag\n");
                                ret = EINVAL;
                }
        }
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

        for(i = 0; i < n_descs; ++i) {
                gds_descriptor_t *desc = descs + i;
                switch(desc->tag) {
                        case GDS_TAG_SEND: {
                                gds_send_request_t *sreq = desc->send;
                                retcode = gds_post_ops(peer, sreq->commit.entries, sreq->commit.storage, params);
                                if (retcode) {
                                        gds_err("error %d in gds_post_ops\n", retcode);
                                        ret = retcode;
                                        goto out;
                                }
                                break;
                        }
                        case GDS_TAG_WAIT: {
                                gds_wait_request_t *wreq = desc->wait;
                                int flags = 0;
                                if (move_flush && i != last_wait) {
                                        gds_dbg("discarding FLUSH!\n");
                                        flags = GDS_POST_OPS_DISCARD_WAIT_FLUSH;
                                }
                                retcode = gds_post_ops(peer, wreq->peek.entries, wreq->peek.storage, params, flags);
                                if (retcode) {
                                        gds_err("error %d in gds_post_ops\n", retcode);
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
        for(i = 0; i < n_descs; ++i) {
                gds_descriptor_t *desc = descs + i;
                switch(desc->tag) {
                        case GDS_TAG_SEND: {
                                gds_dbg("desc[%zu] SEND\n", i);
                                gds_send_request_t *sreq = desc->send;
                                break;
                        }
                        case GDS_TAG_WAIT: {
                                gds_dbg("desc[%zu] WAIT\n", i);
                                gds_wait_request_t *wreq = desc->wait;
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

/*
 * Local variables:
 *  c-indent-level: 8
 *  c-basic-offset: 8
 *  tab-width: 8
 *  indent-tabs-mode: nil
 * End:
 */
