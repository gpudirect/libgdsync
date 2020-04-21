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
#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>

#include "gdsync.h"
#include "gdsync/mlx5.h"

#include "archutils.h"
#include "memmgr.hpp"
#include "mlx5.hpp"
#include "objs.hpp"
#include "utils.hpp"

//-----------------------------------------------------------------------------

#define MLX5_ATOMIC_SIZE 8

static const uint32_t mlx5_ib_opcode[] = {
	[IBV_WR_RDMA_WRITE]		= MLX5_OPCODE_RDMA_WRITE,
	[IBV_WR_RDMA_WRITE_WITH_IMM]	= MLX5_OPCODE_RDMA_WRITE_IMM,
	[IBV_WR_SEND]			= MLX5_OPCODE_SEND,
	[IBV_WR_SEND_WITH_IMM]		= MLX5_OPCODE_SEND_IMM,
	[IBV_WR_RDMA_READ]		= MLX5_OPCODE_RDMA_READ,
	[IBV_WR_ATOMIC_CMP_AND_SWP]	= MLX5_OPCODE_ATOMIC_CS,
	[IBV_WR_ATOMIC_FETCH_AND_ADD]	= MLX5_OPCODE_ATOMIC_FA,
	[IBV_WR_LOCAL_INV]		= MLX5_OPCODE_UMR,
	[IBV_WR_BIND_MW]		= MLX5_OPCODE_UMR,
	[IBV_WR_SEND_WITH_INV]		= MLX5_OPCODE_SEND_INVAL,
	[IBV_WR_TSO]			= MLX5_OPCODE_TSO,
	[IBV_WR_DRIVER1]		= MLX5_OPCODE_UMR,
};

struct mlx5_wqe_xrc_seg {
        __be32		xrc_srqn;
        uint8_t		rsvd[12];
};

struct mlx5_sg_copy_ptr {
        int	index;
        int	offset;
};

struct mlx5_wqe_eth_pad {
        uint8_t rsvd0[16];
};

struct mlx5_wqe_umr_data_seg {
        union {
                struct mlx5_wqe_umr_klm_seg	klm;
                uint8_t				reserved[64];
        };
};

struct mlx5_wqe_inline_seg {
	__be32		byte_count;
};

enum {
        MLX5_IPOIB_INLINE_MIN_HEADER_SIZE	= 4,
        MLX5_SOURCE_QPN_INLINE_MAX_HEADER_SIZE	= 18,
        MLX5_ETH_L2_INLINE_HEADER_SIZE	= 18,
        MLX5_ETH_L2_MIN_HEADER_SIZE	= 14,
};

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

static inline int mlx5_wq_overflow(gds_mlx5_qp *gqp, int nreq)
{
        gds_mlx5_cq *gcq = to_gds_mcq(gqp->send_cq);

	unsigned int cur;

	cur = gqp->sq_cur_post - gcq->cons_index;

	return cur + nreq >= gqp->dv_qp.sq.wqe_cnt;
}

//-----------------------------------------------------------------------------

static inline void *mlx5_get_send_wqe(gds_mlx5_qp *gqp, int n)
{
        return (void *)((uintptr_t)gqp->dv_qp.sq.buf + (n * gqp->dv_qp.sq.stride));
}

//-----------------------------------------------------------------------------

static inline __be32 send_ieth(gds_send_wr *wr)
{
	switch (wr->opcode) {
	case IBV_WR_SEND_WITH_IMM:
	case IBV_WR_RDMA_WRITE_WITH_IMM:
		return wr->imm_data;
	case IBV_WR_SEND_WITH_INV:
		return htobe32(wr->invalidate_rkey);
	default:
		return 0;
	}
}

//-----------------------------------------------------------------------------

static inline void set_raddr_seg(struct mlx5_wqe_raddr_seg *rseg,
                uint64_t remote_addr, uint32_t rkey)
{
        rseg->raddr    = htobe64(remote_addr);
        rseg->rkey     = htobe32(rkey);
        rseg->reserved = 0;
}

//-----------------------------------------------------------------------------

static inline void set_atomic_seg(struct mlx5_wqe_atomic_seg *aseg,
                enum ibv_wr_opcode opcode,
                uint64_t swap,
                uint64_t compare_add)
{
        if (opcode == IBV_WR_ATOMIC_CMP_AND_SWP) {
                aseg->swap_add = htobe64(swap);
                aseg->compare = htobe64(compare_add);
        } else {
                aseg->swap_add = htobe64(compare_add);
        }
}

//-----------------------------------------------------------------------------

#define ALIGN(x, log_a) ((((x) + (1 << (log_a)) - 1)) & ~((1 << (log_a)) - 1))

static inline __be16 get_klm_octo(int nentries)
{
        return htobe16(ALIGN(nentries, 3) / 2);
}

static void set_umr_data_seg(gds_mlx5_qp *qp, enum ibv_mw_type type,
                int32_t rkey,
                const struct ibv_mw_bind_info *bind_info,
                uint32_t qpn, void **seg, int *size)
{
        struct mlx5_wqe_umr_data_seg *data = (struct mlx5_wqe_umr_data_seg *)*seg;

        data->klm.byte_count = htobe32(bind_info->length);
        data->klm.mkey = htobe32(bind_info->mr->lkey);
        data->klm.address = htobe64(bind_info->addr);

        memset(&data->klm + 1, 0, sizeof(data->reserved) -
                        sizeof(data->klm));

        *seg = (void *)((uintptr_t)*seg + sizeof(*data));
        *size += (sizeof(*data) / 16);
}

static void set_umr_mkey_seg(gds_mlx5_qp *qp, enum ibv_mw_type type,
                int32_t rkey,
                const struct ibv_mw_bind_info *bind_info,
                uint32_t qpn, void **seg, int *size)
{
        struct mlx5_wqe_mkey_context_seg *mkey = (struct mlx5_wqe_mkey_context_seg *)*seg;

        mkey->qpn_mkey = htobe32((rkey & 0xFF) |
                        ((type == IBV_MW_TYPE_1 || !bind_info->length) ?
                         0xFFFFFF00 : qpn << 8));
        if (bind_info->length) {
                /* Local read is set in kernel */
                mkey->access_flags = 0;
                mkey->free = 0;
                if (bind_info->mw_access_flags & IBV_ACCESS_LOCAL_WRITE)
                        mkey->access_flags |=
                                MLX5_WQE_MKEY_CONTEXT_ACCESS_FLAGS_LOCAL_WRITE;
                if (bind_info->mw_access_flags & IBV_ACCESS_REMOTE_WRITE)
                        mkey->access_flags |=
                                MLX5_WQE_MKEY_CONTEXT_ACCESS_FLAGS_REMOTE_WRITE;
                if (bind_info->mw_access_flags & IBV_ACCESS_REMOTE_READ)
                        mkey->access_flags |=
                                MLX5_WQE_MKEY_CONTEXT_ACCESS_FLAGS_REMOTE_READ;
                if (bind_info->mw_access_flags & IBV_ACCESS_REMOTE_ATOMIC)
                        mkey->access_flags |=
                                MLX5_WQE_MKEY_CONTEXT_ACCESS_FLAGS_ATOMIC;
                if (bind_info->mw_access_flags & IBV_ACCESS_ZERO_BASED)
                        mkey->start_addr = 0;
                else
                        mkey->start_addr = htobe64(bind_info->addr);
                mkey->len = htobe64(bind_info->length);
        } else {
                mkey->free = MLX5_WQE_MKEY_CONTEXT_FREE;
        }

        *seg = (void *)((uintptr_t)*seg + sizeof(struct mlx5_wqe_mkey_context_seg));
        *size += (sizeof(struct mlx5_wqe_mkey_context_seg) / 16);
}

static inline void set_umr_control_seg(gds_mlx5_qp *qp, enum ibv_mw_type type,
                int32_t rkey,
                const struct ibv_mw_bind_info *bind_info,
                uint32_t qpn, void **seg, int *size)
{
        struct mlx5_wqe_umr_ctrl_seg *ctrl = (struct mlx5_wqe_umr_ctrl_seg *)*seg;

        ctrl->flags = MLX5_WQE_UMR_CTRL_FLAG_TRNSLATION_OFFSET |
                MLX5_WQE_UMR_CTRL_FLAG_INLINE;
        ctrl->mkey_mask = htobe64(MLX5_WQE_UMR_CTRL_MKEY_MASK_FREE |
                        MLX5_WQE_UMR_CTRL_MKEY_MASK_MKEY);
        ctrl->translation_offset = 0;
        memset(ctrl->rsvd0, 0, sizeof(ctrl->rsvd0));
        memset(ctrl->rsvd1, 0, sizeof(ctrl->rsvd1));

        if (type == IBV_MW_TYPE_2)
                ctrl->mkey_mask |= htobe64(MLX5_WQE_UMR_CTRL_MKEY_MASK_QPN);

        if (bind_info->length) {
                ctrl->klm_octowords = get_klm_octo(1);
                if (type == IBV_MW_TYPE_2)
                        ctrl->flags |=  MLX5_WQE_UMR_CTRL_FLAG_CHECK_FREE;
                ctrl->mkey_mask |= htobe64(MLX5_WQE_UMR_CTRL_MKEY_MASK_LEN	|
                                MLX5_WQE_UMR_CTRL_MKEY_MASK_START_ADDR |
                                MLX5_WQE_UMR_CTRL_MKEY_MASK_ACCESS_LOCAL_WRITE |
                                MLX5_WQE_UMR_CTRL_MKEY_MASK_ACCESS_REMOTE_READ |
                                MLX5_WQE_UMR_CTRL_MKEY_MASK_ACCESS_REMOTE_WRITE |
                                MLX5_WQE_UMR_CTRL_MKEY_MASK_ACCESS_ATOMIC);
        } else {
                ctrl->klm_octowords = get_klm_octo(0);
                if (type == IBV_MW_TYPE_2)
                        ctrl->flags |= MLX5_WQE_UMR_CTRL_FLAG_CHECK_QPN;
        }

        *seg = (void *)((uintptr_t)*seg + sizeof(struct mlx5_wqe_umr_ctrl_seg));
        *size += sizeof(struct mlx5_wqe_umr_ctrl_seg) / 16;
}

static inline int set_bind_wr(gds_mlx5_qp *gqp, enum ibv_mw_type type,
                int32_t rkey,
                const struct ibv_mw_bind_info *bind_info,
                uint32_t qpn, void **seg, int *size)
{
        void *qend = mlx5_get_send_wqe(gqp, gqp->dv_qp.sq.wqe_cnt);

        /* check that len > 2GB because KLM support only 2GB */
        if (bind_info->length > 1UL << 31)
                return EOPNOTSUPP;

        set_umr_control_seg(gqp, type, rkey, bind_info, qpn, seg, size);
        if (unlikely((*seg == qend)))
                *seg = mlx5_get_send_wqe(gqp, 0);

        set_umr_mkey_seg(gqp, type, rkey, bind_info, qpn, seg, size);
        if (!bind_info->length)
                return 0;

        if (unlikely((seg == qend)))
                *seg = mlx5_get_send_wqe(gqp, 0);

        set_umr_data_seg(gqp, type, rkey, bind_info, qpn, seg, size);
        return 0;
}

//-----------------------------------------------------------------------------

static inline int mlx5_post_send_underlay(gds_mlx5_qp *qp, gds_send_wr *wr,
                void **pseg, int *total_size,
                struct mlx5_sg_copy_ptr *sg_copy_ptr)
{
        struct mlx5_wqe_eth_seg *eseg;
        int inl_hdr_copy_size;
        void *seg = *pseg;
        int size = 0;

        if (unlikely(wr->opcode == IBV_WR_SEND_WITH_IMM))
                return EINVAL;

        memset(seg, 0, sizeof(struct mlx5_wqe_eth_pad));
        size += sizeof(struct mlx5_wqe_eth_pad);
        seg = (void *)((uintptr_t)seg + sizeof(struct mlx5_wqe_eth_pad));
        eseg = (struct mlx5_wqe_eth_seg *)seg;
        *((uint64_t *)eseg) = 0;
        eseg->rsvd2 = 0;

        if (wr->send_flags & IBV_SEND_IP_CSUM) {
                eseg->cs_flags |= MLX5_ETH_WQE_L3_CSUM | MLX5_ETH_WQE_L4_CSUM;
        }

        if (likely(wr->sg_list[0].length >= MLX5_SOURCE_QPN_INLINE_MAX_HEADER_SIZE))
                /* Copying the minimum required data unless inline mode is set */
                inl_hdr_copy_size = (wr->send_flags & IBV_SEND_INLINE) ?
                        MLX5_SOURCE_QPN_INLINE_MAX_HEADER_SIZE :
                        MLX5_IPOIB_INLINE_MIN_HEADER_SIZE;
        else {
                inl_hdr_copy_size = MLX5_IPOIB_INLINE_MIN_HEADER_SIZE;
                /* We expect at least 4 bytes as part of first entry to hold the IPoIB header */
                if (unlikely(wr->sg_list[0].length < inl_hdr_copy_size))
                        return EINVAL;
        }

        memcpy(eseg->inline_hdr_start, (void *)(uintptr_t)wr->sg_list[0].addr,
                        inl_hdr_copy_size);
        eseg->inline_hdr_sz = htobe16(inl_hdr_copy_size);
        size += sizeof(struct mlx5_wqe_eth_seg);
        seg = (void *)((uintptr_t)seg + sizeof(struct mlx5_wqe_eth_seg));

        /* If we copied all the sge into the inline-headers, then we need to
         * start copying from the next sge into the data-segment.
         */
        if (unlikely(wr->sg_list[0].length == inl_hdr_copy_size))
                sg_copy_ptr->index++;
        else
                sg_copy_ptr->offset = inl_hdr_copy_size;

        *pseg = seg;
        *total_size += (size / 16);
        return 0;
}

//-----------------------------------------------------------------------------

/* Copy tso header to eth segment with considering padding and WQE
 * wrap around in WQ buffer.
 */
static inline int set_tso_eth_seg(void **seg, void *hdr, uint16_t hdr_sz,
                uint16_t mss,
                gds_mlx5_qp *qp, int *size)
{
        struct mlx5_wqe_eth_seg *eseg = (struct mlx5_wqe_eth_seg *)*seg;
        int size_of_inl_hdr_start = sizeof(eseg->inline_hdr_start);
        uint64_t left, left_len, copy_sz;

        void *qend = mlx5_get_send_wqe(qp, qp->dv_qp.sq.wqe_cnt);

        if (unlikely(hdr_sz < MLX5_ETH_L2_MIN_HEADER_SIZE)) {
                gds_dbg("TSO header size should be at least %d\n",
                                MLX5_ETH_L2_MIN_HEADER_SIZE);
                return EINVAL;
        }

        left = hdr_sz;
        eseg->mss = htobe16(mss);
        eseg->inline_hdr_sz = htobe16(hdr_sz);

        /* Check if there is space till the end of queue, if yes,
         * copy all in one shot, otherwise copy till the end of queue,
         * rollback and then copy the left
         */
        left_len = (uintptr_t)qend - (uintptr_t)eseg->inline_hdr_start;
        copy_sz = MIN(left_len, left);

        memcpy(eseg->inline_hdr_start, hdr, copy_sz);

        /* The -1 is because there are already 16 bytes included in
         * eseg->inline_hdr[16]
         */
        *seg = (void *)((uintptr_t)seg + align(copy_sz - size_of_inl_hdr_start, 16) - 16);
        *size += align(copy_sz - size_of_inl_hdr_start, 16) / 16 - 1;

        /* The last wqe in the queue */
        if (unlikely(copy_sz < left)) {
                *seg = mlx5_get_send_wqe(qp, 0);
                left -= copy_sz;
                hdr = (void *)((uintptr_t)hdr + copy_sz);
                memcpy(*seg, hdr, left);
                *seg = (void *)((uintptr_t)*seg + align(left, 16));
                *size += align(left, 16) / 16;
        }

        return 0;
}

//-----------------------------------------------------------------------------

static inline int copy_eth_inline_headers(struct ibv_qp *ibqp,
                const void *list,
                size_t nelem,
                struct mlx5_wqe_eth_seg *eseg,
                struct mlx5_sg_copy_ptr *sg_copy_ptr,
                bool is_sge)
                __attribute__((always_inline));
static inline int copy_eth_inline_headers(struct ibv_qp *ibqp,
                const void *list,
                size_t nelem,
                struct mlx5_wqe_eth_seg *eseg,
                struct mlx5_sg_copy_ptr *sg_copy_ptr,
                bool is_sge)
{
        uint32_t inl_hdr_size = MLX5_ETH_L2_INLINE_HEADER_SIZE;
        size_t inl_hdr_copy_size = 0;
        int j = 0;
        size_t length;
        void *addr;

        if (unlikely(nelem < 1)) {
                gds_dbg("illegal num_sge: %zu, minimum is 1\n", nelem);
                return EINVAL;
        }

        if (is_sge) {
                addr = (void *)(uintptr_t)((struct ibv_sge *)list)[0].addr;
                length = (size_t)((struct ibv_sge *)list)[0].length;
        } else {
                addr = ((struct ibv_data_buf *)list)[0].addr;
                length = ((struct ibv_data_buf *)list)[0].length;
        }

        if (likely(length >= MLX5_ETH_L2_INLINE_HEADER_SIZE)) {
                inl_hdr_copy_size = inl_hdr_size;
                memcpy(eseg->inline_hdr_start, addr, inl_hdr_copy_size);
        } else {
                uint32_t inl_hdr_size_left = inl_hdr_size;

                for (j = 0; j < nelem && inl_hdr_size_left > 0; ++j) {
                        if (is_sge) {
                                addr = (void *)(uintptr_t)((struct ibv_sge *)list)[j].addr;
                                length = (size_t)((struct ibv_sge *)list)[j].length;
                        } else {
                                addr = ((struct ibv_data_buf *)list)[j].addr;
                                length = ((struct ibv_data_buf *)list)[j].length;
                        }

                        inl_hdr_copy_size = MIN(length, inl_hdr_size_left);
                        memcpy(eseg->inline_hdr_start +
                                        (MLX5_ETH_L2_INLINE_HEADER_SIZE - inl_hdr_size_left),
                                        addr, inl_hdr_copy_size);
                        inl_hdr_size_left -= inl_hdr_copy_size;
                }
                if (unlikely(inl_hdr_size_left)) {
                        gds_dbg("Ethernet headers < 16 bytes\n");
                        return EINVAL;
                }
                if (j)
                        --j;
        }

        eseg->inline_hdr_sz = htobe16(inl_hdr_size);

        /* If we copied all the sge into the inline-headers, then we need to
         * start copying from the next sge into the data-segment.
         */
        if (unlikely(length == inl_hdr_copy_size)) {
                ++j;
                inl_hdr_copy_size = 0;
        }

        sg_copy_ptr->index = j;
        sg_copy_ptr->offset = inl_hdr_copy_size;

        return 0;
}

//-----------------------------------------------------------------------------

static inline int set_data_inl_seg(gds_mlx5_qp *qp, gds_send_wr *wr,
                void *wqe, int *sz,
                struct mlx5_sg_copy_ptr *sg_copy_ptr)
{
        struct mlx5_wqe_inline_seg *seg;
        void *addr;
        int len;
        int i;
        int inl = 0;
        int copy;
        int offset = sg_copy_ptr->offset;
        void *qend = mlx5_get_send_wqe(qp, qp->dv_qp.sq.wqe_cnt);
        
        seg = (struct mlx5_wqe_inline_seg *)wqe;
        wqe = (void *)((uintptr_t)wqe + sizeof *seg);
        for (i = sg_copy_ptr->index; i < wr->num_sge; ++i) {
                addr = (void *) (unsigned long)(wr->sg_list[i].addr + offset);
                len  = wr->sg_list[i].length - offset;
                inl += len;
                offset = 0;

                if (unlikely((void *)((uintptr_t)wqe + len) > qend)) {
                        copy = (uintptr_t)qend - (uintptr_t)wqe;
                        memcpy(wqe, addr, copy);
                        addr = (void *)((uintptr_t)addr + copy);
                        len -= copy;
                        wqe = mlx5_get_send_wqe(qp, 0);
                }
                memcpy(wqe, addr, len);
                wqe = (void *)((uintptr_t)wqe + len);
        }

        if (likely(inl)) {
                seg->byte_count = htobe32(inl | MLX5_INLINE_SEG);
                *sz = align(inl + sizeof seg->byte_count, 16) / 16;
        } else
                *sz = 0;

        return 0;
}

//-----------------------------------------------------------------------------

static inline void set_data_ptr_seg(struct mlx5_wqe_data_seg *dseg, struct ibv_sge *sg,
                int offset)
{
        dseg->byte_count = htobe32(sg->length - offset);
        dseg->lkey       = htobe32(sg->lkey);
        dseg->addr       = htobe64(sg->addr + offset);
}

static inline void set_data_ptr_seg_atomic(struct mlx5_wqe_data_seg *dseg,
                struct ibv_sge *sg)
{
        dseg->byte_count = htobe32(MLX5_ATOMIC_SIZE);
        dseg->lkey       = htobe32(sg->lkey);
        dseg->addr       = htobe64(sg->addr);
}

//-----------------------------------------------------------------------------

int gds_mlx5_post_send(gds_mlx5_qp *gqp, gds_send_wr *p_ewr, gds_send_wr **bad_wr, gds_mlx5_peer_commit *commit)
{
        int ret = 0;
        unsigned int idx;
        int size;
        void *seg;
        void *qend = mlx5_get_send_wqe(gqp, gqp->dv_qp.sq.wqe_cnt);
        int i;
        int nreq;
	int inl = 0;
	uint8_t opmod = 0;
	uint32_t mlx5_opcode;

        struct mlx5_wqe_ctrl_seg *ctrl;
	struct mlx5_wqe_xrc_seg *xrc;
	struct mlx5_wqe_eth_seg *eseg;
	struct mlx5_wqe_data_seg *dpseg;
	struct mlx5_sg_copy_ptr sg_copy_ptr = {.index = 0, .offset = 0};

	uint8_t fence;
	uint8_t next_fence;

        struct gds_mlx5_peer_op_wr *wr;

        gds_mlx5_cq *tx_cq = to_gds_mcq(gqp->send_cq);

        if (commit->entries < 3) {
                gds_err("not enough entries in gds_mlx5_peer_commit.\n");
                ret = EINVAL;
                goto out;
        }

	next_fence = gqp->fm_cache;

	for (nreq = 0; p_ewr; ++nreq, p_ewr = p_ewr->next) {
                if (unlikely(p_ewr->opcode < 0 ||
                        p_ewr->opcode >= sizeof mlx5_ib_opcode / sizeof mlx5_ib_opcode[0])) {
                        gds_dbg("bad opcode %d\n", p_ewr->opcode);
                        ret = EINVAL;
                        *bad_wr = p_ewr;
                        goto out;
                }

                if (unlikely(mlx5_wq_overflow(gqp, nreq))) {
                        gds_dbg("work queue overflow\n");
                        ret = ENOMEM;
                        *bad_wr = p_ewr;
                        goto out;
                }

                if (p_ewr->send_flags & IBV_SEND_FENCE)
                        fence = MLX5_WQE_CTRL_FENCE;
                else
                        fence = next_fence;
                next_fence = 0;
                idx = gqp->sq_cur_post & (gqp->dv_qp.sq.wqe_cnt - 1);
                seg = mlx5_get_send_wqe(gqp, idx);
                ctrl = (struct mlx5_wqe_ctrl_seg *)seg;
                *(uint32_t *)((uintptr_t)seg + 8) = 0;
                ctrl->imm = send_ieth(p_ewr);
                ctrl->fm_ce_se = gqp->sq_signal_bits | fence |
                        (p_ewr->send_flags & IBV_SEND_SIGNALED ?
                         MLX5_WQE_CTRL_CQ_UPDATE : 0) |
                        (p_ewr->send_flags & IBV_SEND_SOLICITED ?
                         MLX5_WQE_CTRL_SOLICITED : 0);

                seg = (void *)((uintptr_t)seg + sizeof *ctrl);
                size = sizeof *ctrl / 16;

                switch (gqp->qp->qp_type) {
                case IBV_QPT_XRC_SEND:
                        if (unlikely(p_ewr->opcode != IBV_WR_BIND_MW &&
                                                p_ewr->opcode != IBV_WR_LOCAL_INV)) {
                                xrc = (struct mlx5_wqe_xrc_seg *)seg;
                                xrc->xrc_srqn = htobe32(p_ewr->qp_type.xrc.remote_srqn);
                                seg = (void *)((uintptr_t)seg + sizeof(*xrc));
                                size += sizeof(*xrc) / 16;
                        }
                        /* fall through */
                case IBV_QPT_RC:
                        switch (p_ewr->opcode) {
                        case IBV_WR_RDMA_READ:
                        case IBV_WR_RDMA_WRITE:
                        case IBV_WR_RDMA_WRITE_WITH_IMM:
                                set_raddr_seg((struct mlx5_wqe_raddr_seg *)seg, 
                                                p_ewr->wr.rdma.remote_addr,
                                                p_ewr->wr.rdma.rkey);
                                seg  = (void *)((uintptr_t)seg + sizeof(struct mlx5_wqe_raddr_seg));
                                size += sizeof(struct mlx5_wqe_raddr_seg) / 16;
                                break;

                        case IBV_WR_ATOMIC_CMP_AND_SWP:
                        case IBV_WR_ATOMIC_FETCH_AND_ADD:
                                set_raddr_seg((struct mlx5_wqe_raddr_seg *)seg, 
                                                p_ewr->wr.atomic.remote_addr,
                                                p_ewr->wr.atomic.rkey);
                                seg = (void *)((uintptr_t)seg + sizeof(struct mlx5_wqe_raddr_seg));

                                set_atomic_seg((struct mlx5_wqe_atomic_seg *)seg, 
                                                p_ewr->opcode,
                                                p_ewr->wr.atomic.swap,
                                                p_ewr->wr.atomic.compare_add);
                                seg = (void *)((uintptr_t)seg + sizeof(struct mlx5_wqe_atomic_seg));

                                size += (sizeof(struct mlx5_wqe_raddr_seg) +
                                                sizeof(struct mlx5_wqe_atomic_seg)) / 16;
                                break;

                        case IBV_WR_BIND_MW:
                                next_fence = MLX5_WQE_CTRL_INITIATOR_SMALL_FENCE;
                                ctrl->imm = htobe32(p_ewr->bind_mw.mw->rkey);
                                ret = set_bind_wr(gqp, p_ewr->bind_mw.mw->type,
                                                p_ewr->bind_mw.rkey,
                                                &p_ewr->bind_mw.bind_info,
                                                gqp->qp->qp_num, &seg, &size);
                                if (ret) {
                                        *bad_wr = p_ewr;
                                        goto out;
                                }
                                break;
                        case IBV_WR_LOCAL_INV: {
                                struct ibv_mw_bind_info	bind_info = {};

                                next_fence = MLX5_WQE_CTRL_INITIATOR_SMALL_FENCE;
                                ctrl->imm = htobe32(p_ewr->invalidate_rkey);
                                ret = set_bind_wr(gqp, IBV_MW_TYPE_2, 0,
                                                &bind_info, gqp->qp->qp_num,
                                                &seg, &size);
                                if (ret) {
                                        *bad_wr = p_ewr;
                                        goto out;
                                }
                                break;
			}

                        default:
                               break;
                        }
                        break;

                case IBV_QPT_UC:
                        switch (p_ewr->opcode) {
                        case IBV_WR_RDMA_WRITE:
                        case IBV_WR_RDMA_WRITE_WITH_IMM:
                                set_raddr_seg((struct mlx5_wqe_raddr_seg *)seg, 
                                                p_ewr->wr.rdma.remote_addr,
                                                p_ewr->wr.rdma.rkey);
                                seg = (void *)((uintptr_t)seg + sizeof(struct mlx5_wqe_raddr_seg));
                                size += sizeof(struct mlx5_wqe_raddr_seg) / 16;
                                break;
                        case IBV_WR_BIND_MW:
                                next_fence = MLX5_WQE_CTRL_INITIATOR_SMALL_FENCE;
                                ctrl->imm = htobe32(p_ewr->bind_mw.mw->rkey);
                                ret = set_bind_wr(gqp, p_ewr->bind_mw.mw->type,
                                                p_ewr->bind_mw.rkey,
                                                &p_ewr->bind_mw.bind_info,
                                                gqp->qp->qp_num, &seg, &size);
                                if (ret) {
                                        *bad_wr = p_ewr;
                                        goto out;
                                }
                                break;
                        case IBV_WR_LOCAL_INV: {
                                struct ibv_mw_bind_info	bind_info = {};

                                next_fence = MLX5_WQE_CTRL_INITIATOR_SMALL_FENCE;
                                ctrl->imm = htobe32(p_ewr->invalidate_rkey);
                                ret = set_bind_wr(gqp, IBV_MW_TYPE_2, 0,
                                                &bind_info, gqp->qp->qp_num,
                                                &seg, &size);
                                if (ret) {
                                        *bad_wr = p_ewr;
                                        goto out;
                                }
                                break;
                        }

                        default:
                               break;
                        }
                        break;

                case IBV_QPT_UD:
                        set_datagram_seg((struct mlx5_wqe_datagram_seg *)seg, p_ewr);
                        seg = (void *)((uintptr_t)seg + sizeof(struct mlx5_wqe_datagram_seg));
                        size += sizeof(struct mlx5_wqe_datagram_seg) / 16;
                        if (unlikely((seg == qend)))
                                seg = mlx5_get_send_wqe(gqp, 0);
                        break;

		case IBV_QPT_RAW_PACKET:
                        memset(seg, 0, sizeof(struct mlx5_wqe_eth_seg));
                        eseg = (struct mlx5_wqe_eth_seg *)seg;

                        if (p_ewr->send_flags & IBV_SEND_IP_CSUM) {
                                eseg->cs_flags |= MLX5_ETH_WQE_L3_CSUM | MLX5_ETH_WQE_L4_CSUM;
                        }

                        if (p_ewr->opcode == IBV_WR_TSO) {
                                ret = set_tso_eth_seg(&seg, p_ewr->tso.hdr,
                                                p_ewr->tso.hdr_sz,
                                                p_ewr->tso.mss, gqp, &size);
                                if (unlikely(ret)) {
                                        *bad_wr = p_ewr;
                                        goto out;
                                }

                                /* For TSO WR we always copy at least MLX5_ETH_L2_MIN_HEADER_SIZE
                                 * bytes of inline header which is included in struct mlx5_wqe_eth_seg.
                                 * If additional bytes are copied, 'seg' and 'size' are adjusted
                                 * inside set_tso_eth_seg().
                                 */

                                seg = (void *)((uintptr_t)seg + sizeof(struct mlx5_wqe_eth_seg));
                                size += sizeof(struct mlx5_wqe_eth_seg) / 16;
                        } else {
                                uint32_t inl_hdr_size = MLX5_ETH_L2_INLINE_HEADER_SIZE;

                                ret = copy_eth_inline_headers(gqp->qp, p_ewr->sg_list,
                                                p_ewr->num_sge, (struct mlx5_wqe_eth_seg *)seg,
                                                &sg_copy_ptr, 1);
                                if (unlikely(ret)) {
                                        *bad_wr = p_ewr;
                                        gds_dbg("copy_eth_inline_headers failed, err: %d\n", ret);
                                        goto out;
                                }

                                /* The eth segment size depends on the device's min inline
                                 * header requirement which can be 0 or 18. The basic eth segment
                                 * always includes room for first 2 inline header bytes (even if
                                 * copy size is 0) so the additional seg size is adjusted accordingly.
                                 */

                                seg = (void *)((uintptr_t)seg + (offsetof(struct mlx5_wqe_eth_seg, inline_hdr) +
                                                inl_hdr_size) & ~0xf);
                                size += (offsetof(struct mlx5_wqe_eth_seg, inline_hdr) +
                                                inl_hdr_size) >> 4;
                        }
                        break;

                default:
                        break;
                }

                if (p_ewr->send_flags & IBV_SEND_INLINE && p_ewr->num_sge) {
                        int uninitialized_var(sz);

                        ret = set_data_inl_seg(gqp, p_ewr, seg, &sz, &sg_copy_ptr);
                        if (unlikely(ret)) {
                                *bad_wr = p_ewr;
                                gds_dbg("inline layout failed, err %d\n", ret);
                                goto out;
                        }
                        inl = 1;
                        size += sz;
                } else {
                        dpseg = (struct mlx5_wqe_data_seg *)seg;
                        for (i = sg_copy_ptr.index; i < p_ewr->num_sge; ++i) {
                                if (unlikely(dpseg == qend)) {
                                        seg = mlx5_get_send_wqe(gqp, 0);
                                        dpseg = (struct mlx5_wqe_data_seg *)seg;
                                }
                                if (likely(p_ewr->sg_list[i].length)) {
                                        if (unlikely(p_ewr->opcode ==
                                                                IBV_WR_ATOMIC_CMP_AND_SWP ||
                                                                p_ewr->opcode ==
                                                                IBV_WR_ATOMIC_FETCH_AND_ADD))
                                                set_data_ptr_seg_atomic(dpseg, p_ewr->sg_list + i);
                                        else {
                                                set_data_ptr_seg(dpseg, p_ewr->sg_list + i,
                                                                sg_copy_ptr.offset);
                                        }
                                        sg_copy_ptr.offset = 0;
                                        ++dpseg;
                                        size += sizeof(struct mlx5_wqe_data_seg) / 16;
                                }
                        }
                }

                mlx5_opcode = mlx5_ib_opcode[p_ewr->opcode];
                ctrl->opmod_idx_opcode = htobe32(((gqp->sq_cur_post & 0xffff) << 8) |
                                mlx5_opcode			 |
                                (opmod << 24));
                ctrl->qpn_ds = htobe32(size | (gqp->qp->qp_num << 8));

                tx_cq->wrid[idx] = p_ewr->wr_id;
                gqp->sq_cur_post += (size * 16 + gqp->dv_qp.sq.stride - 1) / gqp->dv_qp.sq.stride;
        }

        #if 0
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
        #endif

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
        wr->wr.qword_va.data = *(__be64 *)ctrl;
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
