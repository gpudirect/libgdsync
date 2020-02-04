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
#include <arpa/inet.h>

#include "gdsync.h"
#include "gdsync/mlx5.h"

#include "archutils.h"
#include "memmgr.hpp"
#include "mlx5.hpp"
#include "objs.hpp"
#include "utils.hpp"

//-----------------------------------------------------------------------------

int gds_mlx5_get_send_descs(gds_mlx5_send_info_t *mlx5_i, const gds_mlx5_send_request_t *request)
{
        int retcode = 0;
        size_t n_ops = request->commit.entries;
        gds_mlx5_peer_op_wr *op = request->commit.storage;
        size_t n = 0;

        memset(mlx5_i, 0, sizeof(*mlx5_i));

        for (; op && n < n_ops; op = op->next, ++n) {
                switch(op->type) {
                        case GDS_MLX5_PEER_OP_FENCE: {
                                gds_dbg("OP_FENCE: fence_flags=%" PRIu64 "\n", op->wr.fence.fence_flags);
                                uint32_t fence_op = (op->wr.fence.fence_flags & (GDS_PEER_FENCE_OP_READ|GDS_PEER_FENCE_OP_WRITE));
                                uint32_t fence_from = (op->wr.fence.fence_flags & (GDS_PEER_FENCE_FROM_CPU|GDS_PEER_FENCE_FROM_HCA));
                                uint32_t fence_mem = (op->wr.fence.fence_flags & (GDS_PEER_FENCE_MEM_SYS|GDS_PEER_FENCE_MEM_PEER));
                                if (fence_op == GDS_PEER_FENCE_OP_READ) {
                                        gds_dbg("nothing to do for read fences\n");
                                        break;
                                }
                                if (fence_from != GDS_PEER_FENCE_FROM_HCA) {
                                        gds_err("unexpected from fence\n");
                                        retcode = EINVAL;
                                        break;
                                }
                                if (fence_mem == GDS_PEER_FENCE_MEM_PEER) {
                                        gds_dbg("using light membar\n");
                                        mlx5_i->membar = 1;
                                }
                                else if (fence_mem == GDS_PEER_FENCE_MEM_SYS) {
                                        gds_dbg("using heavy membar\n");
                                        mlx5_i->membar_full = 1;
                                }
                                else {
                                        gds_err("unsupported fence combination\n");
                                        retcode = EINVAL;
                                        break;
                                }
                                break;
                        }
                        case GDS_MLX5_PEER_OP_STORE_DWORD: {
                                CUdeviceptr dev_ptr = range_from_id(op->wr.dword_va.target_id)->dptr + 
                                        op->wr.dword_va.offset;
                                uint32_t data = op->wr.dword_va.data;
                                gds_dbg("OP_STORE_DWORD dev_ptr=%" PRIx64 " data=%08x\n", (uint64_t)dev_ptr, data);
                                if (n != 0) {
                                        gds_err("store DWORD is not 1st op\n");
                                        retcode = EINVAL;
                                        break;
                                }
                                mlx5_i->dbrec_ptr = (uint32_t*)dev_ptr;
                                mlx5_i->dbrec_value = data;
                                break;
                        }
                        case GDS_MLX5_PEER_OP_STORE_QWORD: {
                                CUdeviceptr dev_ptr = range_from_id(op->wr.qword_va.target_id)->dptr +
                                        op->wr.qword_va.offset;
                                uint64_t data = op->wr.qword_va.data;
                                gds_dbg("OP_STORE_QWORD dev_ptr=%" PRIx64 " data=%" PRIx64 "\n", (uint64_t)dev_ptr, (uint64_t)data);
                                if (n != 2) {
                                        gds_err("store QWORD is not 3rd op\n");
                                        retcode = EINVAL;
                                        break;
                                }
                                mlx5_i->db_ptr = (uint64_t*)dev_ptr;
                                mlx5_i->db_value = data;
                                break;
                        }
                        case GDS_MLX5_PEER_OP_COPY_BLOCK: {
                                CUdeviceptr dev_ptr = range_from_id(op->wr.copy_op.target_id)->dptr +
                                        op->wr.copy_op.offset;
                                size_t len = op->wr.copy_op.len;
                                void *src = op->wr.copy_op.src;
                                gds_dbg("send inline detected\n");
                                if (len < 8 || len > 64) {
                                        gds_err("unexpected len %zu\n", len);
                                        retcode = EINVAL;
                                        break;
                                }
                                mlx5_i->db_ptr = (uint64_t*)dev_ptr;
                                mlx5_i->db_value = *(uint64_t*)src; 
                                break;
                        }
                        case GDS_MLX5_PEER_OP_POLL_AND_DWORD:
                        case GDS_MLX5_PEER_OP_POLL_GEQ_DWORD:
                        case GDS_MLX5_PEER_OP_POLL_NOR_DWORD: {
                                gds_err("unexpected polling op in send request\n");
                                retcode = EINVAL;
                                break;
                        }
                        default:
                                gds_err("undefined peer op type %d\n", op->type);
                                retcode = EINVAL;
                                break;
                }

                if (retcode) {
                        gds_err("error in fill func at entry n=%zu\n", n);
                        break;
                }
        }
        return retcode;
}

//-----------------------------------------------------------------------------

int gds_mlx5_get_send_info(int count, const gds_mlx5_send_request_t *requests, gds_mlx5_send_info_t *mlx5_infos)
{
        int retcode = 0;

        for (int j=0; j<count; j++) {
                gds_mlx5_send_info *mlx5_i = mlx5_infos + j;
                const gds_mlx5_send_request_t *request = requests + j;
                retcode = gds_mlx5_get_send_descs(mlx5_i, request);
                if (retcode) {
                        gds_err("error %d while retrieving descriptors for %dth request\n", retcode, j);
                        break;
                }
                gds_dbg("mlx5_i: dbrec={%p,%08x} db={%p,%" PRIx64 "}\n",
                                mlx5_i->dbrec_ptr, mlx5_i->dbrec_value, mlx5_i->db_ptr, mlx5_i->db_value);
        }

        return retcode;
}

//-----------------------------------------------------------------------------

int gds_mlx5_get_wait_descs(gds_mlx5_wait_info_t *mlx5_i, const gds_mlx5_wait_request_t *request)
{
        int retcode = 0;
        size_t n_ops = request->peek.entries;
        gds_mlx5_peer_op_wr *op = request->peek.storage;
        size_t n = 0;

        memset(mlx5_i, 0, sizeof(*mlx5_i));

        for (; op && n < n_ops; op = op->next, ++n) {
                switch(op->type) {
                        case GDS_MLX5_PEER_OP_FENCE: {
                                gds_dbg("OP_FENCE: fence_flags=%" PRIu64 "\n", op->wr.fence.fence_flags);
                                uint32_t fence_op = (op->wr.fence.fence_flags & (GDS_PEER_FENCE_OP_READ|GDS_PEER_FENCE_OP_WRITE));
                                uint32_t fence_from = (op->wr.fence.fence_flags & (GDS_PEER_FENCE_FROM_CPU|GDS_PEER_FENCE_FROM_HCA));
                                uint32_t fence_mem = (op->wr.fence.fence_flags & (GDS_PEER_FENCE_MEM_SYS|GDS_PEER_FENCE_MEM_PEER));
                                if (fence_op == GDS_PEER_FENCE_OP_READ) {
                                        gds_dbg("nothing to do for read fences\n");
                                        break;
                                }
                                if (fence_from != GDS_PEER_FENCE_FROM_HCA) {
                                        gds_err("unexpected from fence\n");
                                        retcode = EINVAL;
                                        break;
                                }
                                gds_err("unsupported fence combination\n");
                                retcode = EINVAL;
                                break;
                        }
                        case GDS_MLX5_PEER_OP_STORE_DWORD: {
                                CUdeviceptr dev_ptr = range_from_id(op->wr.dword_va.target_id)->dptr + 
                                        op->wr.dword_va.offset;
                                uint32_t data = op->wr.dword_va.data;
                                gds_dbg("OP_STORE_DWORD dev_ptr=%" PRIx64 " data=%08x\n", (uint64_t)dev_ptr, data);
                                if (n != 1) {
                                        gds_err("store DWORD is not 2nd op\n");
                                        retcode = EINVAL;
                                        break;
                                }
                                mlx5_i->flag_ptr = (uint32_t*)dev_ptr;
                                mlx5_i->flag_value = data;
                                break;
                        }
                        case GDS_MLX5_PEER_OP_STORE_QWORD: {
                                CUdeviceptr dev_ptr = range_from_id(op->wr.qword_va.target_id)->dptr +
                                        op->wr.qword_va.offset;
                                uint64_t data = op->wr.qword_va.data;
                                gds_dbg("OP_STORE_QWORD dev_ptr=%" PRIx64 " data=%" PRIx64 "\n", (uint64_t)dev_ptr, (uint64_t)data);
                                gds_err("unsupported QWORD op\n");
                                retcode = EINVAL;
                                break;
                        }
                        case GDS_MLX5_PEER_OP_COPY_BLOCK: {
                                CUdeviceptr dev_ptr = range_from_id(op->wr.copy_op.target_id)->dptr +
                                        op->wr.copy_op.offset;
                                size_t len = op->wr.copy_op.len;
                                void *src = op->wr.copy_op.src;
                                gds_err("unsupported COPY_BLOCK\n");
                                retcode = EINVAL;
                                break;
                        }
                        case GDS_MLX5_PEER_OP_POLL_AND_DWORD:
                        case GDS_MLX5_PEER_OP_POLL_GEQ_DWORD:
                        case GDS_MLX5_PEER_OP_POLL_NOR_DWORD: {
                                CUdeviceptr dev_ptr = range_from_id(op->wr.dword_va.target_id)->dptr + 
                                        op->wr.dword_va.offset;
                                uint32_t data = op->wr.dword_va.data;

                                gds_dbg("OP_POLL_DWORD dev_ptr=%" PRIx64 " data=%08x\n", (uint64_t)dev_ptr, data);

                                mlx5_i->cqe_ptr = (uint32_t *)dev_ptr;
                                mlx5_i->cqe_value = data;

                                switch(op->type) {
                                        case GDS_MLX5_PEER_OP_POLL_NOR_DWORD:
                                                // GPU SMs can always do NOR
                                                mlx5_i->cond = GDS_WAIT_COND_NOR;
                                                break;
                                        case GDS_MLX5_PEER_OP_POLL_GEQ_DWORD:
                                                mlx5_i->cond = GDS_WAIT_COND_GEQ;
                                                break;
                                        case GDS_MLX5_PEER_OP_POLL_AND_DWORD:
                                                mlx5_i->cond = GDS_WAIT_COND_AND;
                                                break;
                                        default:
                                                gds_err("unexpected op type\n");
                                                retcode = EINVAL;
                                                goto err;
                                }
                                break;
                        }
                        default:
                                gds_err("undefined peer op type %d\n", op->type);
                                retcode = EINVAL;
                                break;
                }
err:
                if (retcode) {
                        gds_err("error in fill func at entry n=%zu\n", n);
                        break;
                }
        }
        return retcode;
}

//-----------------------------------------------------------------------------

int gds_mlx5_get_wait_info(int count, const gds_wait_request_t *requests, gds_mlx5_wait_info_t *mlx5_infos)
{
        int retcode = 0;

        for (int j=0; j<count; j++) {
                gds_mlx5_wait_info *mlx5_i = mlx5_infos + j;
                const gds_wait_request_t *request = requests + j;
                gds_dbg("wait[%d] cqe_ptr=%p cqe_value=0x%08x flag_ptr=%p flag_value=0x%08x\n", 
                                j, mlx5_i->cqe_ptr, mlx5_i->cqe_value, mlx5_i->flag_ptr, mlx5_i->flag_value);
        }

        return retcode;
}

//-----------------------------------------------------------------------------

int gds_mlx5_get_dword_wait_info(uint32_t *ptr, uint32_t value, int flags, gds_mlx5_dword_wait_info_t *mlx5_info)
{
        int retcode = 0;
        CUdeviceptr dev_ptr = 0;

        assert(NULL != ptr);
        memset(mlx5_info, 0, sizeof(&mlx5_info));

        retcode = gds_map_mem(ptr, sizeof(*ptr), memtype_from_flags(flags), &dev_ptr);
        if (retcode) {
                gds_err("error %d while mapping addr %p\n", retcode, ptr);
                goto out;
        }

        gds_dbg("dev_ptr=%llx value=%08x\n", dev_ptr, value);
        mlx5_info->ptr = (uint32_t*)dev_ptr;
        mlx5_info->value = value;
out:
        return retcode;
}

//-----------------------------------------------------------------------------

int gds_mlx5_rollback_send(gds_mlx5_qp *gqp,
		       struct gds_mlx5_rollback_ctx *rollback)
{
	int diff;

	gqp->bf_offset = (rollback->rollback_id & GDS_MLX5_ROLLBACK_ID_PARITY_MASK) ?
					gqp->dv_qp.bf.size : 0;
	rollback->rollback_id &= GDS_MLX5_ROLLBACK_ID_PARITY_MASK - 1;

	if (rollback->flags & GDS_MLX5_ROLLBACK_ABORT_UNCOMMITED) {
		diff = (gqp->sq_cur_post & 0xffff)
		     - ntohl(gqp->dv_qp.dbrec[MLX5_SND_DBR]);
		if (diff < 0)
			diff += 0x10000;
		gqp->sq_cur_post -= diff;
	} else {
		if (!(rollback->flags & GDS_MLX5_ROLLBACK_ABORT_LATE)) {
			if (gqp->sq_cur_post !=
			    (rollback->rollback_id >> 32))
				return -ERANGE;
		}
		gqp->sq_cur_post = rollback->rollback_id & 0xffffffff;
	}
	return 0;
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

int gds_mlx5_post_send(gds_mlx5_qp *gqp, gds_send_wr *p_ewr, gds_send_wr **bad_ewr, gds_mlx5_peer_commit *commit)
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

        struct gds_mlx5_peer_op_wr *wr;

        gds_mlx5_cq *tx_cq = to_gds_mcq(gqp->send_cq);

        if (commit->entries < 3) {
                gds_err("not enough entries in gds_mlx5_peer_commit.\n");
                ret = EINVAL;
                goto out;
        }

        if (p_ewr->opcode != IBV_WR_SEND) {
                gds_err("Unsupported opcode. Currently we support only IBV_WR_SEND\n");
                ret = EINVAL;
                goto out;
        }

        if (p_ewr->send_flags != IBV_SEND_SIGNALED) {
                gds_err("Unsupported send_flags. Currently we support only IBV_SEND_SIGNALED\n");
                ret = EINVAL;
                goto out;
        }

        qend = (void *)((char *)gqp->dv_qp.sq.buf + (gqp->dv_qp.sq.wqe_cnt * gqp->dv_qp.sq.stride));
        for (nreq = 0; p_ewr; ++nreq, p_ewr = p_ewr->next) {
                idx = gqp->sq_cur_post & (gqp->dv_qp.sq.wqe_cnt - 1);
                seg = (void *)((char *)gqp->dv_qp.sq.buf + (idx * gqp->dv_qp.sq.stride));

                ctrl_seg = (struct mlx5_wqe_ctrl_seg *)seg;
                size = sizeof(struct mlx5_wqe_ctrl_seg) / 16;
                seg = (void *)((char *)seg + sizeof(struct mlx5_wqe_ctrl_seg));

                switch (gqp->qp->qp_type) {
                        case IBV_QPT_UD:
                                ret = set_datagram_seg((struct mlx5_wqe_datagram_seg *)seg, p_ewr);
                                if (ret)
                                        goto out;
                                size = sizeof(struct mlx5_wqe_datagram_seg) / 16;
                                seg = (void *)((char *)seg + sizeof(struct mlx5_wqe_datagram_seg));
                                if (seg == qend)
                                        seg = gqp->dv_qp.sq.buf;
                                break;
                        case IBV_QPT_RC:
                                break;
                        default:
                                gds_err("Encountered unsupported qp_type. We currently support only IBV_QPT_UD\n");
                                ret = EINVAL;
                                goto out;
                }

                data_seg = (struct mlx5_wqe_data_seg *)seg;
                for (i = 0; i < p_ewr->num_sge; ++i) {
                        if (data_seg == qend) {
                                seg = gqp->dv_qp.sq.buf;
                                data_seg = (struct mlx5_wqe_data_seg *)seg;
                        }
                        if (p_ewr->sg_list[i].length) {
                                mlx5dv_set_data_seg(data_seg, p_ewr->sg_list[i].length, p_ewr->sg_list[i].lkey, p_ewr->sg_list[i].addr);
                                ++data_seg;
                                size += sizeof(struct mlx5_wqe_data_seg) / 16;
                        }
                }

                mlx5dv_set_ctrl_seg(ctrl_seg, gqp->sq_cur_post & 0xffff, MLX5_OPCODE_SEND, 0, gqp->qp->qp_num, MLX5_WQE_CTRL_CQ_UPDATE, size, 0, 0);

                tx_cq->wrid[idx] = p_ewr->wr_id;
                gqp->sq_cur_post += (size * 16 + gqp->dv_qp.sq.stride - 1) / gqp->dv_qp.sq.stride;
        }

        commit->rollback_id = gqp->peer_scur_post | ((uint64_t)gqp->sq_cur_post << 32);
        gqp->peer_scur_post = gqp->sq_cur_post;

        wr = commit->storage;

        wr->type = GDS_MLX5_PEER_OP_STORE_DWORD;
        wr->wr.dword_va.data = htonl(gqp->sq_cur_post & 0xffff);
        wr->wr.dword_va.target_id = gqp->peer_va_id_dbr;
        wr->wr.dword_va.offset = sizeof(uint32_t) * MLX5_SND_DBR;
        wr = wr->next;

        wr->type = GDS_MLX5_PEER_OP_FENCE;
        wr->wr.fence.fence_flags = GDS_PEER_FENCE_OP_WRITE | GDS_PEER_FENCE_FROM_HCA | GDS_PEER_FENCE_MEM_SYS;
        wr = wr->next;

        wr->type = GDS_MLX5_PEER_OP_STORE_QWORD;
        wr->wr.qword_va.data = *(__be64 *)ctrl_seg;
        wr->wr.qword_va.target_id = gqp->peer_va_id_bf;
        wr->wr.qword_va.offset = gqp->bf_offset;

        gqp->bf_offset ^= gqp->dv_qp.bf.size;
        commit->entries = 3;

out:
        return ret;
}

//-----------------------------------------------------------------------------

int gds_mlx5_peer_peek_cq(gds_mlx5_cq *gcq, struct gds_mlx5_peer_peek *peek)
{
        int ret = 0;

        gds_peer_attr *peer_attr = (gds_peer_attr *)gcq->peer_attr;
        struct gds_mlx5_peer_op_wr *wr;
        int n, cur_own;
        void *cqe;
        gds_cqe64 *cqe64;
        struct gds_mlx5_peek_entry *tmp;

        if (peek->entries < 2) {
                gds_err("not enough entries in gds_mlx5_peek_entry.\n");
                ret = EINVAL;
                goto out;
        }

        wr = peek->storage;
        n = peek->offset;

        cqe = (char *)gcq->dv_cq.buf + (n & (gcq->dv_cq.cqe_cnt - 1)) * gcq->dv_cq.cqe_size;
        cur_own = n & gcq->dv_cq.cqe_cnt;
        cqe64 = (gds_cqe64 *)((gcq->dv_cq.cqe_size == 64) ? cqe : (char *)cqe + 64);

        if (cur_own) {
                wr->type = GDS_MLX5_PEER_OP_POLL_AND_DWORD;
                wr->wr.dword_va.data = htonl(MLX5_CQE_OWNER_MASK);
        }
        else if (peer_attr->caps & GDS_PEER_OP_POLL_NOR_DWORD_CAP) {
                wr->type = GDS_MLX5_PEER_OP_POLL_NOR_DWORD;
                wr->wr.dword_va.data = ~htonl(MLX5_CQE_OWNER_MASK);
        }
        else if (peer_attr->caps & GDS_PEER_OP_POLL_GEQ_DWORD_CAP) {
                wr->type = GDS_MLX5_PEER_OP_POLL_GEQ_DWORD;
                wr->wr.dword_va.data = 0;
        }
        wr->wr.dword_va.target_id = gcq->active_buf_va_id;
        wr->wr.dword_va.offset = (uintptr_t)&cqe64->wqe_counter - (uintptr_t)gcq->dv_cq.buf;
        wr = wr->next;

        tmp = gcq->peer_peek_free;
        if (!tmp) {
                ret = ENOMEM;
                goto out;
        }
        gcq->peer_peek_free = GDS_MLX5_PEEK_ENTRY(gcq, tmp->next);
        tmp->busy = 1;
        wmb();
        tmp->next = GDS_MLX5_PEEK_ENTRY_N(gcq, gcq->peer_peek_table[n & (gcq->dv_cq.cqe_cnt - 1)]);
        gcq->peer_peek_table[n & (gcq->dv_cq.cqe_cnt - 1)] = tmp;

        wr->type = GDS_MLX5_PEER_OP_STORE_DWORD;
        wr->wr.dword_va.data = 0;
        wr->wr.dword_va.target_id = gcq->peer_va_id;
        wr->wr.dword_va.offset = (uintptr_t)&tmp->busy - (uintptr_t)gcq->peer_buf->addr;

        peek->entries = 2;
        peek->peek_id = (uintptr_t)tmp;

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
