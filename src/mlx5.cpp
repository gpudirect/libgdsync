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
#include "mlnxutils.h"
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

int gds_mlx5_rollback_send(gds_mlx5_qp_t *mqp,
		       struct gds_mlx5_rollback_ctx *rollback)
{
	int diff;

	mqp->bf_offset = (rollback->rollback_id & GDS_MLX5_ROLLBACK_ID_PARITY_MASK) ?
					mqp->dvqp.bf.size : 0;
	rollback->rollback_id &= GDS_MLX5_ROLLBACK_ID_PARITY_MASK - 1;

	if (rollback->flags & GDS_MLX5_ROLLBACK_ABORT_UNCOMMITED) {
		diff = (mqp->sq_cur_post & 0xffff)
		     - ntohl(mqp->dvqp.dbrec[MLX5_SND_DBR]);
		if (diff < 0)
			diff += 0x10000;
		mqp->sq_cur_post -= diff;
	} else {
		if (!(rollback->flags & GDS_MLX5_ROLLBACK_ABORT_LATE)) {
			if (mqp->sq_cur_post !=
			    (rollback->rollback_id >> 32))
				return -ERANGE;
		}
		mqp->sq_cur_post = rollback->rollback_id & 0xffffffff;
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

static inline int mlx5_wq_overflow(gds_mlx5_qp_t *mqp, int nreq)
{
	unsigned cur;

	cur = mqp->wq->head - mqp->wq->tail;

	return cur + nreq >= mqp->dvqp.sq.wqe_cnt;
}

//-----------------------------------------------------------------------------

static inline void *mlx5_get_send_wqe(gds_mlx5_qp_t *mqp, int n)
{
        return (void *)((uintptr_t)mqp->dvqp.sq.buf + (n * mqp->dvqp.sq.stride));
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

static void set_umr_data_seg(gds_mlx5_qp_t *mqp, enum ibv_mw_type type,
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

static void set_umr_mkey_seg(gds_mlx5_qp_t *mqp, enum ibv_mw_type type,
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

static inline void set_umr_control_seg(gds_mlx5_qp_t *mqp, enum ibv_mw_type type,
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

static inline int set_bind_wr(gds_mlx5_qp_t *mqp, enum ibv_mw_type type,
                int32_t rkey,
                const struct ibv_mw_bind_info *bind_info,
                uint32_t qpn, void **seg, int *size)
{
        void *qend = mlx5_get_send_wqe(mqp, mqp->dvqp.sq.wqe_cnt);

        /* check that len > 2GB because KLM support only 2GB */
        if (bind_info->length > 1UL << 31)
                return EOPNOTSUPP;

        set_umr_control_seg(mqp, type, rkey, bind_info, qpn, seg, size);
        if (unlikely((*seg == qend)))
                *seg = mlx5_get_send_wqe(mqp, 0);

        set_umr_mkey_seg(mqp, type, rkey, bind_info, qpn, seg, size);
        if (!bind_info->length)
                return 0;

        if (unlikely((seg == qend)))
                *seg = mlx5_get_send_wqe(mqp, 0);

        set_umr_data_seg(mqp, type, rkey, bind_info, qpn, seg, size);
        return 0;
}

//-----------------------------------------------------------------------------

static inline int mlx5_post_send_underlay(gds_mlx5_qp_t *mqp, gds_send_wr *wr,
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
                gds_mlx5_qp_t *mqp, int *size)
{
        struct mlx5_wqe_eth_seg *eseg = (struct mlx5_wqe_eth_seg *)*seg;
        int size_of_inl_hdr_start = sizeof(eseg->inline_hdr_start);
        uint64_t left, left_len, copy_sz;

        void *qend = mlx5_get_send_wqe(mqp, mqp->dvqp.sq.wqe_cnt);

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
                *seg = mlx5_get_send_wqe(mqp, 0);
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

static inline int set_data_inl_seg(gds_mlx5_qp_t *mqp, gds_send_wr *wr,
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
        void *qend = mlx5_get_send_wqe(mqp, mqp->dvqp.sq.wqe_cnt);
        
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
                        wqe = mlx5_get_send_wqe(mqp, 0);
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

int gds_mlx5_post_send(gds_mlx5_qp_t *mqp, gds_send_wr *p_ewr, gds_send_wr **bad_wr, gds_mlx5_peer_commit *commit)
{
        int ret = 0;
        unsigned int idx;
        int size;
        void *seg;
        void *qend = mlx5_get_send_wqe(mqp, mqp->dvqp.sq.wqe_cnt);
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

        if (commit->entries < 3) {
                gds_err("not enough entries in gds_mlx5_peer_commit.\n");
                ret = EINVAL;
                goto out;
        }

	next_fence = mqp->fm_cache;

	for (nreq = 0; p_ewr; ++nreq, p_ewr = p_ewr->next) {
                if (unlikely(p_ewr->opcode < 0 ||
                        p_ewr->opcode >= sizeof mlx5_ib_opcode / sizeof mlx5_ib_opcode[0])) {
                        gds_dbg("bad opcode %d\n", p_ewr->opcode);
                        ret = EINVAL;
                        *bad_wr = p_ewr;
                        goto out;
                }

                if (unlikely(mlx5_wq_overflow(mqp, nreq))) {
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
                idx = mqp->sq_cur_post & (mqp->dvqp.sq.wqe_cnt - 1);
                seg = mlx5_get_send_wqe(mqp, idx);
                ctrl = (struct mlx5_wqe_ctrl_seg *)seg;
                *(uint32_t *)((uintptr_t)seg + 8) = 0;
                ctrl->imm = send_ieth(p_ewr);
                ctrl->fm_ce_se = mqp->sq_signal_bits | fence |
                        (p_ewr->send_flags & IBV_SEND_SIGNALED ?
                         MLX5_WQE_CTRL_CQ_UPDATE : 0) |
                        (p_ewr->send_flags & IBV_SEND_SOLICITED ?
                         MLX5_WQE_CTRL_SOLICITED : 0);

                seg = (void *)((uintptr_t)seg + sizeof *ctrl);
                size = sizeof *ctrl / 16;

                switch (mqp->gqp.ibqp->qp_type) {
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
                                ret = set_bind_wr(mqp, p_ewr->bind_mw.mw->type,
                                                p_ewr->bind_mw.rkey,
                                                &p_ewr->bind_mw.bind_info,
                                                mqp->gqp.ibqp->qp_num, &seg, &size);
                                if (ret) {
                                        *bad_wr = p_ewr;
                                        goto out;
                                }
                                break;
                        case IBV_WR_LOCAL_INV: {
                                struct ibv_mw_bind_info	bind_info = {};

                                next_fence = MLX5_WQE_CTRL_INITIATOR_SMALL_FENCE;
                                ctrl->imm = htobe32(p_ewr->invalidate_rkey);
                                ret = set_bind_wr(mqp, IBV_MW_TYPE_2, 0,
                                                &bind_info, mqp->gqp.ibqp->qp_num,
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
                                ret = set_bind_wr(mqp, p_ewr->bind_mw.mw->type,
                                                p_ewr->bind_mw.rkey,
                                                &p_ewr->bind_mw.bind_info,
                                                mqp->gqp.ibqp->qp_num, &seg, &size);
                                if (ret) {
                                        *bad_wr = p_ewr;
                                        goto out;
                                }
                                break;
                        case IBV_WR_LOCAL_INV: {
                                struct ibv_mw_bind_info	bind_info = {};

                                next_fence = MLX5_WQE_CTRL_INITIATOR_SMALL_FENCE;
                                ctrl->imm = htobe32(p_ewr->invalidate_rkey);
                                ret = set_bind_wr(mqp, IBV_MW_TYPE_2, 0,
                                                &bind_info, mqp->gqp.ibqp->qp_num,
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
                                seg = mlx5_get_send_wqe(mqp, 0);
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
                                                p_ewr->tso.mss, mqp, &size);
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

                                ret = copy_eth_inline_headers(mqp->gqp.ibqp, p_ewr->sg_list,
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

                        ret = set_data_inl_seg(mqp, p_ewr, seg, &sz, &sg_copy_ptr);
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
                                        seg = mlx5_get_send_wqe(mqp, 0);
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
                ctrl->opmod_idx_opcode = htobe32(((mqp->sq_cur_post & 0xffff) << 8) |
                                mlx5_opcode			 |
                                (opmod << 24));
                ctrl->qpn_ds = htobe32(size | (mqp->gqp.ibqp->qp_num << 8));

                mqp->wq->wrid[idx] = p_ewr->wr_id;
                mqp->wq->wqe_head[idx] = mqp->wq->head + nreq;
                mqp->sq_cur_post += (size * 16 + mqp->dvqp.sq.stride - 1) / mqp->dvqp.sq.stride;
        }

out:
	mqp->fm_cache = next_fence;

        if (likely(nreq > 0)) {
                mqp->wq->head += nreq;

                commit->rollback_id = mqp->qp_peer->scur_post | ((uint64_t)mqp->sq_cur_post << 32);
                mqp->qp_peer->scur_post = mqp->sq_cur_post;

                wr = commit->storage;

                wr->type = GDS_MLX5_PEER_OP_STORE_DWORD;
                wr->wr.dword_va.data = htonl(mqp->sq_cur_post & 0xffff);
                wr->wr.dword_va.target_id = mqp->qp_peer->dbr.va_id;
                wr->wr.dword_va.offset = sizeof(uint32_t) * MLX5_SND_DBR;
                wr = wr->next;

                wr->type = GDS_MLX5_PEER_OP_FENCE;
                wr->wr.fence.fence_flags = GDS_PEER_FENCE_OP_WRITE | GDS_PEER_FENCE_FROM_HCA | GDS_PEER_FENCE_MEM_SYS;
                wr = wr->next;

                wr->type = GDS_MLX5_PEER_OP_STORE_QWORD;
                wr->wr.qword_va.data = *(__be64 *)ctrl;
                wr->wr.qword_va.target_id = mqp->qp_peer->bf.va_id;
                wr->wr.qword_va.offset = mqp->bf_offset;

                mqp->bf_offset ^= mqp->dvqp.bf.size;
                commit->entries = 3;
        }

        return ret;
}

//-----------------------------------------------------------------------------

int gds_mlx5_peer_peek_cq(gds_mlx5_cq_t *mcq, struct gds_mlx5_peer_peek *peek)
{
        int ret = 0;

        gds_peer_attr *peer_attr = (gds_peer_attr *)mcq->peer_attr;
        struct gds_mlx5_peer_op_wr *wr;
        int n, cur_own;
        void *cqe;
        struct mlx5_cqe64 *cqe64;
        struct gds_mlx5_peek_entry *tmp;

        if (peek->entries < 2) {
                gds_err("not enough entries in gds_mlx5_peek_entry.\n");
                ret = EINVAL;
                goto out;
        }

        wr = peek->storage;
        n = peek->offset;

        cqe = (char *)mcq->dvcq.buf + (n & (mcq->dvcq.cqe_cnt - 1)) * mcq->dvcq.cqe_size;
        cur_own = n & mcq->dvcq.cqe_cnt;
        cqe64 = (struct mlx5_cqe64 *)((mcq->dvcq.cqe_size == 64) ? cqe : (char *)cqe + 64);

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
        wr->wr.dword_va.target_id = mcq->active_buf_va_id;
        wr->wr.dword_va.offset = (uintptr_t)&cqe64->wqe_counter - (uintptr_t)mcq->dvcq.buf;
        wr = wr->next;

        tmp = mcq->peer_peek_free;
        if (!tmp) {
                ret = ENOMEM;
                goto out;
        }
        mcq->peer_peek_free = GDS_MLX5_PEEK_ENTRY(mcq, tmp->next);
        tmp->busy = 1;
        wmb();
        tmp->next = GDS_MLX5_PEEK_ENTRY_N(mcq, mcq->peer_peek_table[n & (mcq->dvcq.cqe_cnt - 1)]);
        mcq->peer_peek_table[n & (mcq->dvcq.cqe_cnt - 1)] = tmp;

        wr->type = GDS_MLX5_PEER_OP_STORE_DWORD;
        wr->wr.dword_va.data = 0;
        wr->wr.dword_va.target_id = mcq->peer_va_id;
        wr->wr.dword_va.offset = (uintptr_t)&tmp->busy - (uintptr_t)mcq->peer_buf->addr;

        peek->entries = 2;
        peek->peek_id = (uintptr_t)tmp;

out:
        return ret;
}
//-----------------------------------------------------------------------------

int gds_mlx5_create_cq(struct ibv_cq *ibcq, gds_peer_attr *peer_attr, gds_mlx5_cq_t **out_mcq)
{
        int ret = 0;

        gds_mlx5_cq_t *mcq = NULL;
        gds_cq_t *gcq;
        mlx5dv_obj dv_obj;

        struct gds_buf_alloc_attr ba_attr;

        mcq = (gds_mlx5_cq_t *)calloc(1, sizeof(gds_mlx5_cq_t));
        if (!mcq) {
                gds_err("cannot allocate memory\n");
                ret = ENOMEM;
                goto err;
        }

        dv_obj.cq.in = ibcq;
        dv_obj.cq.out = &mcq->dvcq;
        ret = mlx5dv_init_obj(&dv_obj, MLX5DV_OBJ_CQ);
        if (ret) {
                gds_err("error %d in mlx5dv_init_obj MLX5DV_OBJ_CQ\n", ret);
                goto err;
        }

        mcq->peer_attr = peer_attr;

        mcq->active_buf_va_id = peer_attr->register_va(
                mcq->dvcq.buf,
                (uint64_t)mcq->dvcq.cqe_cnt * (uint64_t)mcq->dvcq.cqe_size,
                peer_attr->peer_id,
                NULL
        );
        if (!mcq->active_buf_va_id) {
                gds_err("error in peer_attr->register_va\n");
                ret = EINVAL;
                goto err;
        }

        mcq->peer_peek_table = (struct gds_mlx5_peek_entry **)malloc(sizeof(struct gds_mlx5_peek_entry *) * mcq->dvcq.cqe_cnt);
        if (!mcq->peer_peek_table) {
                gds_err("error %d in malloc peer_peek_table\n", errno);
                ret = ENOMEM;
                goto err;
        }
        memset(mcq->peer_peek_table, 0, sizeof(struct gds_peek_entry *) * mcq->dvcq.cqe_cnt);
        mcq->peer_dir = GDS_PEER_DIRECTION_FROM_PEER | GDS_PEER_DIRECTION_TO_CPU;

        ba_attr = {
                .length         = sizeof(struct gds_mlx5_peek_entry) * mcq->dvcq.cqe_cnt,
                .dir            = mcq->peer_dir,
                .peer_id        = peer_attr->peer_id,
                .alignment      = (uint32_t)sysconf(_SC_PAGESIZE),
                .comp_mask      = 0
        };
        mcq->peer_buf = peer_attr->buf_alloc(&ba_attr);
        if (!mcq->peer_buf) {
                gds_err("error %d in buf_alloc\n", errno);
                ret = ENOMEM;
                goto err;
        }

        mcq->peer_va_id = peer_attr->register_va(mcq->peer_buf->addr, mcq->peer_buf->length, peer_attr->peer_id, mcq->peer_buf);
        if (!mcq->peer_va_id) {
                gds_err("error %d in register_va\n", errno);
                ret = EINVAL;
                goto err;
        }

        memset(mcq->peer_buf->addr, 0, mcq->peer_buf->length);

        mcq->peer_peek_free = (struct gds_mlx5_peek_entry *)mcq->peer_buf->addr;
        for (int i = 0; i < mcq->dvcq.cqe_cnt - 1; ++i)
                mcq->peer_peek_free[i].next = i + 1;
        mcq->peer_peek_free[mcq->dvcq.cqe_size - 1].next = GDS_MLX5_LAST_PEEK_ENTRY;

        mcq->gcq.ibcq = ibcq;
        mcq->gcq.dtype = GDS_DRIVER_TYPE_MLX5;
        *out_mcq = mcq;

        return 0;

err:
        if (mcq) {
                if (mcq->peer_va_id)
                        peer_attr->unregister_va(mcq->peer_va_id, peer_attr->peer_id);

                if (mcq->peer_buf)
                        peer_attr->buf_release(mcq->peer_buf);

                if (mcq->peer_peek_table)
                        free(mcq->peer_peek_table);

                if (mcq->active_buf_va_id)
                        peer_attr->unregister_va(mcq->active_buf_va_id, peer_attr->peer_id);

                free(mcq);
        }

        return ret;
}

//-----------------------------------------------------------------------------

void gds_mlx5_destroy_cq(gds_mlx5_cq_t *mcq)
{
        int status = 0;
        if (mcq->peer_peek_table) {
                free(mcq->peer_peek_table);
                mcq->peer_peek_table = NULL;
        }

        if (mcq->wq)
                mcq->wq = NULL;

        if (mcq->peer_attr) {
                gds_peer_attr *peer_attr = mcq->peer_attr;
                if (mcq->active_buf_va_id) {
                        peer_attr->unregister_va(mcq->active_buf_va_id, peer_attr->peer_id);
                        mcq->active_buf_va_id = 0;
                }
                if (mcq->peer_va_id) {
                        peer_attr->unregister_va(mcq->peer_va_id, peer_attr->peer_id);
                        mcq->peer_va_id = 0;
                }
                if (mcq->peer_buf) {
                        mcq->peer_attr->buf_release(mcq->peer_buf);
                        mcq->peer_buf = NULL;
                }
        }

        if (mcq->gcq.ibcq) {
                status = ibv_destroy_cq(mcq->gcq.ibcq);
                if (status) {
                        gds_err("error %d in ibv_destroy\n", status);
                        return;
                }
                mcq->gcq.ibcq = NULL;
        }

        free(mcq);
}

//-----------------------------------------------------------------------------

static gds_mlx5_wq_t *mlx5_create_wq(uint32_t wqe_cnt)
{
        gds_mlx5_wq_t *wq = (gds_mlx5_wq_t *)calloc(1, sizeof(gds_mlx5_wq_t));
        if (!wq) {
                gds_err("error in calloc wq\n");
                goto err;
        }

        wq->wrid = (uint64_t *)malloc(wqe_cnt * sizeof(uint64_t));
        if (!wq->wrid) {
                gds_err("error in calloc wq->wrid\n");
                goto err;
        }

        wq->wqe_head = (uint64_t *)malloc(wqe_cnt * sizeof(uint64_t));
        if (!wq->wqe_head) {
                gds_err("error in calloc wq->wqe_head\n");
                goto err;
        }

        wq->wqe_cnt = wqe_cnt;

        return wq;
err:
        if (wq) {
                if (wq->wrid)
                        free(wq->wrid);
                if (wq->wqe_head)
                        free(wq->wqe_head);
                free(wq);
        }
        return NULL;
}

static void mlx5_destroy_wq(gds_mlx5_wq_t *wq)
{
        if (wq) {
                if (wq->wrid) {
                        free(wq->wrid);
                        wq->wrid = NULL;
                }
                if (wq->wqe_head) {
                        free(wq->wqe_head);
                        wq->wqe_head = NULL;
                }
                free(wq);
        }
}

//-----------------------------------------------------------------------------

int gds_mlx5_create_qp(struct ibv_qp *ibqp, gds_qp_init_attr_t *qp_attr, gds_mlx5_cq_t *tx_mcq, gds_mlx5_cq_t *rx_mcq, gds_mlx5_qp_peer_t *qp_peer, gds_mlx5_qp_t **out_mqp)
{
        int ret = 0;

        gds_mlx5_qp_t *mqp = NULL;
        gds_qp_t *gqp;

        gds_mlx5_wq_t *wq = NULL;

        gds_peer_attr *peer_attr = qp_peer->peer_attr;

        bool register_peer_dbr = false;

        mlx5dv_obj dv_obj;

        mqp = (gds_mlx5_qp_t *)calloc(1, sizeof(gds_mlx5_qp_t));
        if (!mqp) {
                gds_err("cannot allocate memory\n");
                ret = ENOMEM;
                goto err;
        }

        gqp = &mqp->gqp;
        gqp->dtype = GDS_DRIVER_TYPE_MLX5;
        gqp->ibqp = ibqp;

        tx_mcq->gcq.ibcq = ibqp->send_cq;
        tx_mcq->gcq.curr_offset = 0;
        tx_mcq->gcq.ctype = GDS_CQ_TYPE_SQ;
        gqp->send_cq = &tx_mcq->gcq;

        rx_mcq->gcq.ibcq = ibqp->recv_cq;
        rx_mcq->gcq.curr_offset = 0;
        rx_mcq->gcq.ctype = GDS_CQ_TYPE_RQ;
        gqp->recv_cq = &rx_mcq->gcq;

        if (qp_attr->sq_sig_all)
                mqp->sq_signal_bits = MLX5_WQE_CTRL_CQ_UPDATE;
        else
                mqp->sq_signal_bits = 0;

        dv_obj = {
                .qp = {
                        .in     = gqp->ibqp,
                        .out    = &mqp->dvqp
                }
        };
        ret = mlx5dv_init_obj(&dv_obj, MLX5DV_OBJ_QP);
        if (ret != 0) {
                gds_err("error in mlx5dv_init_obj MLX5DV_OBJ_QP\n");
                goto err;
        }

        if (!qp_peer->dbr.va_id) {
                qp_peer->dbr.va_id = peer_attr->register_va(
                        mqp->dvqp.dbrec,
                        qp_peer->dbr.size,
                        peer_attr->peer_id,
                        NULL
                );
                if (!qp_peer->dbr.va_id) {
                        gds_err("error in register_va\n");
                        goto err;
                }
                register_peer_dbr = true;
        }

        qp_peer->bf.va_id = peer_attr->register_va(
                (uint32_t *)mqp->dvqp.bf.reg,
                mqp->dvqp.bf.size,
                peer_attr->peer_id,
                GDS_PEER_IOMEMORY
        );
        if (!qp_peer->bf.va_id) {
                gds_err("error in register_va\n");
                goto err;
        }

        wq = mlx5_create_wq(mqp->dvqp.sq.wqe_cnt);
        if (!wq) {
                gds_err("error in mlx5_create_wq\n");
                ret = ENOMEM;
                goto err;
        }

        mqp->qp_peer = qp_peer;
        mqp->wq = wq;
        tx_mcq->wq = wq;
        *out_mqp = mqp;
        return 0;

err:
        if (register_peer_dbr)
                peer_attr->unregister_va(qp_peer->dbr.va_id, peer_attr->peer_id);

        if (qp_peer->bf.va_id)
                peer_attr->unregister_va(qp_peer->bf.va_id, peer_attr->peer_id);

        if (wq)
                mlx5_destroy_wq(wq);

        if (mqp)
                free(mqp);

        return ret;
}

//-----------------------------------------------------------------------------

void gds_mlx5_destroy_qp(gds_mlx5_qp_t *mqp)
{
        int status;
        gds_mlx5_qp_peer_t *qp_peer = mqp->qp_peer;
        
        if (qp_peer) {
                gds_peer_attr *peer_attr = qp_peer->peer_attr;
                if (qp_peer->dbr.va_id) {
                        peer_attr->unregister_va(qp_peer->dbr.va_id, peer_attr->peer_id);
                        qp_peer->dbr.va_id = 0;
                }

                if (qp_peer->bf.va_id) {
                        peer_attr->unregister_va(qp_peer->bf.va_id, peer_attr->peer_id);
                        qp_peer->bf.va_id = 0;
                }
        }

        if (mqp->gqp.ibqp) {
                status = ibv_destroy_qp(mqp->gqp.ibqp);
                if (status)
                        gds_err("error %d in ibv_destroy_qp\n", status);
        }

        if (mqp->gqp.send_cq) {
                gds_destroy_cq(mqp->gqp.send_cq);
                mqp->gqp.send_cq = NULL;
        }

        if (mqp->gqp.recv_cq) {
                gds_destroy_cq(mqp->gqp.recv_cq);
                mqp->gqp.recv_cq = NULL;
        }

        if (mqp->wq) {
                mlx5_destroy_wq(mqp->wq);
                mqp->wq = NULL;
        }

        free(mqp);
}

//-----------------------------------------------------------------------------

static void *pd_mem_alloc(struct ibv_pd *pd, void *pd_context, size_t size,
                        size_t alignment, uint64_t resource_type)
{
        assert(pd_context);

        gds_peer_attr *peer_attr = (gds_peer_attr *)pd_context;
        gds_buf_alloc_attr buf_attr = {
                .length         = size,
                .dir            = GDS_PEER_DIRECTION_FROM_PEER | GDS_PEER_DIRECTION_TO_HCA,
                .peer_id        = peer_attr->peer_id,
                .alignment      = (uint32_t)alignment,
                .comp_mask      = peer_attr->comp_mask
        };
        gds_peer *peer = peer_from_id(peer_attr->peer_id);
        gds_mlx5_qp_peer_t *qp_peer;
        uint64_t range_id;
        gds_buf *buf = NULL;
        void *ptr = NULL;

        gds_dbg("pd_mem_alloc: pd=%p, pd_context=%p, size=%zu, alignment=%zu, resource_type=0x%lx\n",
                pd, pd_context, size, alignment, resource_type);

        assert(peer->obj);
        qp_peer = (gds_mlx5_qp_peer_t *)peer->obj;

        switch (resource_type) {
                case MLX5DV_RES_TYPE_QP:
                        break;
                case MLX5DV_RES_TYPE_DBR:
                        buf = peer_attr->buf_alloc(&buf_attr);
                        qp_peer->dbr.size = size;
                        break;
                default:
                        gds_err("request allocation with unsupported resource_type\n");
                        break;
        }

        if (!buf) {
                int err;
                gds_dbg("alloc on host\n");
                return IBV_ALLOCATOR_USE_DEFAULT;
        }
        else {
                gds_dbg("alloc on GPU\n");
                ptr = buf->addr;
        }

        if ((range_id = peer_attr->register_va(ptr, size, peer_attr->peer_id, buf)) == 0) {
                gds_err("error in register_va\n");
                peer_attr->buf_release(buf);
                return IBV_ALLOCATOR_USE_DEFAULT;
        }

        if (resource_type == MLX5DV_RES_TYPE_DBR) {
                qp_peer->dbr.va_id = range_id;
                qp_peer->dbr.gbuf = buf;
        }
        
        return ptr;
}

static void pd_mem_free(struct ibv_pd *pd, void *pd_context, void *ptr,
                        uint64_t resource_type)
{
        gds_dbg("pd_mem_free: pd=%p, pd_context=%p, ptr=%p, resource_type=0x%lx\n",
                pd, pd_context, ptr, resource_type);

        assert(pd_context);

        gds_peer_attr *peer_attr = (gds_peer_attr *)pd_context;
        gds_peer *peer = peer_from_id(peer_attr->peer_id);

        assert(peer->obj);
        gds_mlx5_qp_peer_t *qp_peer = (gds_mlx5_qp_peer_t *)peer->obj;

        if (qp_peer->dbr.gbuf) {
                if (qp_peer->dbr.va_id) {
                        peer_attr->unregister_va(qp_peer->dbr.va_id, peer_attr->peer_id);
                        qp_peer->dbr.va_id = 0;
                }
                peer_attr->buf_release(qp_peer->dbr.gbuf);
                qp_peer->dbr.gbuf = NULL;
        }
}

int gds_mlx5_alloc_parent_domain(struct ibv_pd *p_pd, struct ibv_context *ibctx, gds_peer_attr *peer_attr, struct ibv_pd **out_pd, gds_mlx5_qp_peer_t **out_qp_peer)
{
        int ret = 0;

        struct ibv_parent_domain_init_attr pd_init_attr;
        struct ibv_pd *pd = NULL;
        gds_peer *peer = peer_from_id(peer_attr->peer_id);

        gds_mlx5_qp_peer_t *qp_peer = (gds_mlx5_qp_peer_t *)calloc(1, sizeof(gds_mlx5_qp_peer_t));
        if (!qp_peer) {
                gds_err("cannot allocate memory\n");
                ret = ENOMEM;
                goto err;
        }

        qp_peer->peer_attr = peer_attr;
        peer->obj = qp_peer;

        memset(&pd_init_attr, 0, sizeof(ibv_parent_domain_init_attr));
        pd_init_attr.pd = p_pd;
        pd_init_attr.comp_mask = IBV_PARENT_DOMAIN_INIT_ATTR_ALLOCATORS | IBV_PARENT_DOMAIN_INIT_ATTR_PD_CONTEXT;
        pd_init_attr.alloc = pd_mem_alloc;
        pd_init_attr.free = pd_mem_free;
        pd_init_attr.pd_context = peer_attr;

        pd = ibv_alloc_parent_domain(ibctx, &pd_init_attr);
        if (!pd) {
                gds_err("error in ibv_alloc_parent_domain\n");
                ret = EINVAL;
                goto err;
        }

        *out_pd = pd;
        *out_qp_peer = qp_peer;
        return 0;

err:
        if (qp_peer)
                free(qp_peer);
        return ret;
}

//-----------------------------------------------------------------------------

/*
   A) plain+membar:
   WR32
   MEMBAR
   WR32
   WR32

   B) plain:
   WR32
   WR32+PREBARRIER
   WR32

   C) sim64+membar:
   WR32
   MEMBAR
   INLCPY 8B

   D) sim64:
   INLCPY 4B + POSTBARRIER
   INLCPY 8B

   E) inlcpy+membar:
   WR32
   MEMBAR
   INLCPY XB

   F) inlcpy:
   INLCPY 4B + POSTBARRIER
   INLCPY 128B
 */

int gds_mlx5_post_ops(gds_peer *peer, size_t n_ops, struct gds_mlx5_peer_op_wr *op, gds_op_list_t &ops, int post_flags)
{
        int retcode = 0;
        size_t n = 0;
        bool prev_was_fence = false;
        bool use_inlcpy_for_dword = false;
        CUstreamBatchMemOpParams param;

        gds_dbg("n_ops=%zu\n", n_ops);

        if (!peer->has_memops) {
                gds_err("CUDA MemOps are required\n");
                return EINVAL;
        }

        // divert the request to the same engine handling 64bits
        // to avoid out-of-order execution
        // caveat: can't use membar if inlcpy is used for 4B writes (to simulate 8B writes)
        if (peer->has_inlcpy) {
                if (!peer->has_membar)
                        use_inlcpy_for_dword = true; // F
        }
        if (gds_simulate_write64()) {
                if (!peer->has_membar) {
                        gds_warn_once("enabling use_inlcpy_for_dword\n");
                        use_inlcpy_for_dword = true; // D
                }
        }

        for (; op && n < n_ops; op = op->next, ++n) {
                gds_dbg("op[%zu] type:%08x\n", n, op->type);
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
                                else {
                                        if (!peer->has_membar) {
                                                if (use_inlcpy_for_dword) {
                                                        assert(ops.size() > 0);
                                                        gds_dbg("patching previous param\n");
                                                        gds_enable_barrier_for_inlcpy(&ops.back());
                                                }
                                                else {
                                                        gds_dbg("recording fence event\n");
                                                        prev_was_fence = true;
                                                }
                                        }
                                        else {
                                                if (fence_from != GDS_PEER_FENCE_FROM_HCA) {
                                                        gds_err("unexpected from fence\n");
                                                        retcode = EINVAL;
                                                        break;
                                                }
                                                int flags = 0;
                                                if (fence_mem == GDS_PEER_FENCE_MEM_PEER) {
                                                        gds_dbg("using light membar\n");
                                                        flags = GDS_MEMBAR_DEFAULT | GDS_MEMBAR_MLX5;
                                                }
                                                else if (fence_mem == GDS_PEER_FENCE_MEM_SYS) {
                                                        gds_dbg("using heavy membar\n");
                                                        flags = GDS_MEMBAR_SYS | GDS_MEMBAR_MLX5;
                                                }
                                                else {
                                                        gds_err("unsupported fence combination\n");
                                                        retcode = EINVAL;
                                                        break;
                                                }
                                                retcode = gds_fill_membar(peer, ops, flags);
                                        }
                                }
                                break;
                        }
                        case GDS_MLX5_PEER_OP_STORE_DWORD: {
                                CUdeviceptr dev_ptr = range_from_id(op->wr.dword_va.target_id)->dptr + 
                                        op->wr.dword_va.offset;
                                uint32_t data = op->wr.dword_va.data;
                                int flags = 0;
                                gds_dbg("OP_STORE_DWORD dev_ptr=%llx data=%" PRIx32 "\n", dev_ptr, data);
                                if (use_inlcpy_for_dword) { // F || D
                                        // membar may be out of order WRT inlcpy
                                        if (peer->has_membar) {
                                                gds_err("invalid feature combination, inlcpy + membar\n");
                                                retcode = EINVAL;
                                                break;
                                        }
                                        // tail flush is set when following fence is met
                                        //  flags |= GDS_IMMCOPY_POST_TAIL_FLUSH;
                                        retcode = gds_fill_inlcpy(peer, ops, dev_ptr, &data, sizeof(data), flags);
                                }
                                else {  // A || B || C || E
                                        // can't guarantee ordering of write32+inlcpy unless
                                        // a membar is there
                                        // TODO: fix driver when !weak
                                        if (peer->has_inlcpy && !peer->has_membar) {
                                                gds_err("invalid feature combination, inlcpy needs membar\n");
                                                retcode = EINVAL;
                                                break;
                                        }
                                        if (prev_was_fence) {
                                                gds_dbg("using PRE_BARRIER as fence\n");
                                                flags |= GDS_WRITE_PRE_BARRIER;
                                                prev_was_fence = false;
                                        }
                                        retcode = gds_fill_poke(peer, ops, dev_ptr, data, flags);
                                }
                                break;
                        }
                        case GDS_MLX5_PEER_OP_STORE_QWORD: {
                                CUdeviceptr dev_ptr = range_from_id(op->wr.qword_va.target_id)->dptr +
                                        op->wr.qword_va.offset;
                                uint64_t data = op->wr.qword_va.data;
                                int flags = 0;
                                gds_dbg("OP_STORE_QWORD dev_ptr=%llx data=%" PRIx64 "\n", dev_ptr, data);
                                // C || D
                                if (gds_simulate_write64()) {
                                        // simulate 64-bit poke by inline copy
                                        if (!peer->has_membar) {
                                                gds_err("invalid feature combination, inlcpy needs membar\n");
                                                retcode = EINVAL;
                                                break;
                                        }

                                        // tail flush is never useful here
                                        //flags |= GDS_IMMCOPY_POST_TAIL_FLUSH;
                                        retcode = gds_fill_inlcpy(peer, ops, dev_ptr, &data, sizeof(data), flags);
                                }
                                else if (peer->has_write64) {
                                        retcode = gds_fill_poke64(peer, ops, dev_ptr, data, flags);
                                }
                                else {
                                        uint32_t datalo = gds_qword_lo(op->wr.qword_va.data);
                                        uint32_t datahi = gds_qword_hi(op->wr.qword_va.data);

                                        if (prev_was_fence) {
                                                gds_dbg("enabling PRE_BARRIER\n");
                                                flags |= GDS_WRITE_PRE_BARRIER;
                                                prev_was_fence = false;
                                        }
                                        retcode = gds_fill_poke(peer, ops, dev_ptr, datalo, flags);

                                        // get rid of the barrier, if there
                                        flags &= ~GDS_WRITE_PRE_BARRIER;

                                        // advance to next DWORD
                                        dev_ptr += sizeof(uint32_t);
                                        retcode = gds_fill_poke(peer, ops, dev_ptr, datahi, flags);
                                }

                                break;
                        }
                        case GDS_MLX5_PEER_OP_COPY_BLOCK: {
                                CUdeviceptr dev_ptr = range_from_id(op->wr.copy_op.target_id)->dptr +
                                        op->wr.copy_op.offset;
                                size_t len = op->wr.copy_op.len;
                                void *src = op->wr.copy_op.src;
                                int flags = 0;
                                gds_dbg("OP_COPY_BLOCK dev_ptr=%llx src=%p len=%zu\n", dev_ptr, src, len);
                                // catching any other size here
                                if (!peer->has_inlcpy) {
                                        gds_err("inline copy is not supported\n");
                                        retcode = EINVAL;
                                        break;
                                }
                                // IB Verbs bug
                                assert(len <= GDS_GPU_MAX_INLINE_SIZE);
                                //if (desc->need_flush) {
                                //        flags |= GDS_IMMCOPY_POST_TAIL_FLUSH;
                                //}
                                retcode = gds_fill_inlcpy(peer, ops, dev_ptr, src, len, flags);
                                break;
                        }
                        case GDS_MLX5_PEER_OP_POLL_AND_DWORD:
                        case GDS_MLX5_PEER_OP_POLL_GEQ_DWORD:
                        case GDS_MLX5_PEER_OP_POLL_NOR_DWORD: {
                                int poll_cond;
                                CUdeviceptr dev_ptr = range_from_id(op->wr.dword_va.target_id)->dptr + 
                                        op->wr.dword_va.offset;
                                uint32_t data = op->wr.dword_va.data;
                                // TODO: properly handle a following fence instead of blidly flushing
                                int flags = 0;
                                if (!(post_flags & GDS_POST_OPS_DISCARD_WAIT_FLUSH))
                                        flags |= GDS_WAIT_POST_FLUSH_REMOTE;

                                gds_dbg("OP_WAIT_DWORD dev_ptr=%llx data=%" PRIx32 " type=%" PRIx32 "\n", dev_ptr, data, (uint32_t)op->type);

                                switch(op->type) {
                                        case GDS_MLX5_PEER_OP_POLL_NOR_DWORD:
                                                poll_cond = GDS_WAIT_COND_NOR;
                                                break;
                                        case GDS_MLX5_PEER_OP_POLL_GEQ_DWORD:
                                                poll_cond = GDS_WAIT_COND_GEQ;
                                                break;
                                        case GDS_MLX5_PEER_OP_POLL_AND_DWORD:
                                                poll_cond = GDS_WAIT_COND_AND;
                                                break;
                                        default:
                                                assert(!"cannot happen");
                                                retcode = EINVAL;
                                                goto out;
                                }
                                retcode = gds_fill_poll(peer, ops, dev_ptr, data, poll_cond, flags);
                                break;
                        }
                        default:
                                gds_err("undefined peer op type %d\n", op->type);
                                retcode = EINVAL;
                                break;
                }
                if (retcode) {
                        gds_err("error in fill func at entry n=%zu\n", n);
                        goto out;
                }
        }

        assert(n_ops == n);

out:
        return retcode;
}

//-----------------------------------------------------------------------------

int gds_mlx5_post_ops_on_cpu(size_t n_ops, struct gds_mlx5_peer_op_wr *op, int post_flags)
{
        int retcode = 0;
        size_t n = 0;
        gds_dbg("n_ops=%zu op=%p post_flags=0x%x\n", n_ops, op, post_flags);
        for (; op && n < n_ops; op = op->next, ++n) {
                gds_dbg("op[%zu]=%p\n", n, op);
                switch(op->type) {
                        case GDS_MLX5_PEER_OP_FENCE: {
                                gds_dbg("FENCE flags=%" PRIu64 "\n", op->wr.fence.fence_flags);
                                uint32_t fence_op = (op->wr.fence.fence_flags & (GDS_PEER_FENCE_OP_READ|GDS_PEER_FENCE_OP_WRITE));
                                uint32_t fence_from = (op->wr.fence.fence_flags & (GDS_PEER_FENCE_FROM_CPU|GDS_PEER_FENCE_FROM_HCA));
                                uint32_t fence_mem = (op->wr.fence.fence_flags & (GDS_PEER_FENCE_MEM_SYS|GDS_PEER_FENCE_MEM_PEER));

                                if (fence_op == GDS_PEER_FENCE_OP_READ) {
                                        gds_warnc(1, "nothing to do for read fences\n");
                                        break;
                                }
                                else {
                                        if (fence_from != GDS_PEER_FENCE_FROM_HCA) {
                                                gds_err("unexpected from %08x fence, expected FROM_HCA\n", fence_from);
                                                retcode = EINVAL;
                                                break;
                                        }
                                        if (fence_mem == GDS_PEER_FENCE_MEM_PEER) {
                                                gds_dbg("using light membar\n");
                                                wmb();
                                        }
                                        else if (fence_mem == GDS_PEER_FENCE_MEM_SYS) {
                                                gds_dbg("using heavy membar\n");
                                                wmb();
                                        }
                                        else {
                                                gds_err("unsupported fence combination\n");
                                                retcode = EINVAL;
                                                break;
                                        }
                                }
                                break;
                        }
                        case GDS_MLX5_PEER_OP_STORE_DWORD: {
                                uint32_t *ptr = (uint32_t*)((ptrdiff_t)range_from_id(op->wr.dword_va.target_id)->va + op->wr.dword_va.offset);
                                uint32_t data = op->wr.dword_va.data;
                                // A || B || C || E
                                gds_dbg("STORE_DWORD ptr=%p data=%08" PRIx32 "\n", ptr, data);
                                gds_atomic_set(ptr, data);
                                break;
                        }
                        case GDS_MLX5_PEER_OP_STORE_QWORD: {
                                uint64_t *ptr = (uint64_t*)((ptrdiff_t)range_from_id(op->wr.qword_va.target_id)->va + op->wr.qword_va.offset);
                                uint64_t data = op->wr.qword_va.data;
                                gds_dbg("STORE_QWORD ptr=%p data=%016" PRIx64 "\n", ptr, data);
                                gds_atomic_set(ptr, data);
                                break;
                        }
                        case GDS_MLX5_PEER_OP_COPY_BLOCK: {
                                uint64_t *ptr = (uint64_t*)((ptrdiff_t)range_from_id(op->wr.copy_op.target_id)->va + op->wr.copy_op.offset);
                                uint64_t *src = (uint64_t*)op->wr.copy_op.src;
                                size_t n_bytes = op->wr.copy_op.len;
                                gds_dbg("COPY_BLOCK ptr=%p src=%p len=%zu\n", ptr, src, n_bytes);
                                gds_bf_copy(ptr, src, n_bytes);
                                break;
                        }
                        case GDS_MLX5_PEER_OP_POLL_AND_DWORD:
                        case GDS_MLX5_PEER_OP_POLL_GEQ_DWORD:
                        case GDS_MLX5_PEER_OP_POLL_NOR_DWORD: {
                                int poll_cond;
                                uint32_t *ptr = (uint32_t*)((ptrdiff_t)range_from_id(op->wr.dword_va.target_id)->va + op->wr.dword_va.offset);
                                uint32_t value = op->wr.dword_va.data;
                                bool flush = true;
                                if (post_flags & GDS_POST_OPS_DISCARD_WAIT_FLUSH)
                                        flush = false;
                                gds_dbg("WAIT_32 dev_ptr=%p data=%" PRIx32 " type=%" PRIx32 "\n", ptr, value, (uint32_t)op->type);
                                bool done = false;
                                do {
                                        uint32_t data = gds_atomic_get(ptr);
                                        switch(op->type) {
                                                case GDS_MLX5_PEER_OP_POLL_NOR_DWORD:
                                                        done = (0 != ~(data | value));
                                                        break;
                                                case GDS_MLX5_PEER_OP_POLL_GEQ_DWORD:
                                                        done = ((int32_t)data - (int32_t)value >= 0);
                                                        break;
                                                case GDS_MLX5_PEER_OP_POLL_AND_DWORD:
                                                        done = (0 != (data & value));
                                                        break;
                                                default:
                                                        gds_err("invalid op type %02x\n", op->type);
                                                        retcode = EINVAL;
                                                        goto out;
                                        }
                                        if (done)
                                                break;
                                        // TODO: more aggressive CPU relaxing needed here to avoid starving I/O fabric
                                        arch_cpu_relax();
                                } while(true);
                                break;
                        }
                        default:
                                gds_err("undefined peer op type %d\n", op->type);
                                retcode = EINVAL;
                                break;
                }
                if (retcode) {
                        gds_err("error %d at entry n=%zu\n", retcode, n);
                        goto out;
                }
        }

out:
        return retcode;
}

//-----------------------------------------------------------------------------

void gds_mlx5_dump_ops(struct gds_mlx5_peer_op_wr *op, size_t count)
{
        size_t n = 0;
        for (; op; op = op->next, ++n) {
                gds_dbg("op[%zu] type:%d\n", n, op->type);
                switch(op->type) {
                        case GDS_MLX5_PEER_OP_FENCE: {
                                gds_dbg("FENCE flags=%" PRIu64 "\n", op->wr.fence.fence_flags);
                                break;
                        }
                        case GDS_MLX5_PEER_OP_STORE_DWORD: {
                                CUdeviceptr dev_ptr = range_from_id(op->wr.dword_va.target_id)->dptr + 
                                        op->wr.dword_va.offset;
                                gds_dbg("STORE_QWORD data:%x target_id:%" PRIx64 " offset:%zu dev_ptr=%llx\n",
                                                op->wr.dword_va.data, op->wr.dword_va.target_id,
                                                op->wr.dword_va.offset, dev_ptr);
                                break;
                        }
                        case GDS_MLX5_PEER_OP_STORE_QWORD: {
                                CUdeviceptr dev_ptr = range_from_id(op->wr.qword_va.target_id)->dptr +
                                        op->wr.qword_va.offset;
                                gds_dbg("STORE_QWORD data:%" PRIx64 " target_id:%" PRIx64 " offset:%zu dev_ptr=%llx\n",
                                                op->wr.qword_va.data, op->wr.qword_va.target_id,
                                                op->wr.qword_va.offset, dev_ptr);
                                break;
                        }
                        case GDS_MLX5_PEER_OP_COPY_BLOCK: {
                                CUdeviceptr dev_ptr = range_from_id(op->wr.copy_op.target_id)->dptr +
                                        op->wr.copy_op.offset;
                                gds_dbg("COPY_BLOCK src:%p len:%zu target_id:%" PRIx64 " offset:%zu dev_ptr=%llx\n",
                                                op->wr.copy_op.src, op->wr.copy_op.len,
                                                op->wr.copy_op.target_id, op->wr.copy_op.offset,
                                                dev_ptr);
                                break;
                        }
                        case GDS_MLX5_PEER_OP_POLL_AND_DWORD:
                        case GDS_MLX5_PEER_OP_POLL_NOR_DWORD: {
                                CUdeviceptr dev_ptr = range_from_id(op->wr.dword_va.target_id)->dptr + 
                                        op->wr.dword_va.offset;
                                gds_dbg("%s data:%08x target_id:%" PRIx64 " offset:%zu dev_ptr=%llx\n", 
                                                (op->type==GDS_MLX5_PEER_OP_POLL_AND_DWORD) ? "POLL_AND_DW" : "POLL_NOR_SDW",
                                                op->wr.dword_va.data, 
                                                op->wr.dword_va.target_id, 
                                                op->wr.dword_va.offset, 
                                                dev_ptr);
                                break;
                        }
                        default:
                                gds_err("undefined peer op type %d\n", op->type);
                                break;
                }
        }

        assert(count == n);
}

//-----------------------------------------------------------------------------

int gds_mlx5_poll_cq(gds_mlx5_cq_t *mcq, int ne, struct ibv_wc *wc)
{
        unsigned int idx;
        int cnt;
        int p_ne;

        void *cqe;
        struct mlx5_cqe64 *cqe64;

        uint16_t wqe_ctr;
        int wqe_ctr_idx;

        assert(mcq->gcq.ctype == GDS_CQ_TYPE_SQ);

        for (cnt = 0; cnt < ne; ++cnt) {
                idx = mcq->cons_index & (mcq->dvcq.cqe_cnt - 1);
                while (mcq->peer_peek_table[idx]) {
                        struct gds_mlx5_peek_entry *tmp;
                        if (*(volatile uint32_t *)&mcq->peer_peek_table[idx]->busy) {
                                return cnt;
                        }
                        tmp = mcq->peer_peek_table[idx];
                        mcq->peer_peek_table[idx] = GDS_MLX5_PEEK_ENTRY(mcq, tmp->next);
                        tmp->next = GDS_MLX5_PEEK_ENTRY_N(mcq, mcq->peer_peek_free);
                        mcq->peer_peek_free = tmp;
                }
                cqe = (void *)((uintptr_t)mcq->dvcq.buf + mcq->cons_index * mcq->dvcq.cqe_size);
                cqe64 = (mcq->dvcq.cqe_size == 64) ? (struct mlx5_cqe64 *)cqe : (struct mlx5_cqe64 *)((uintptr_t)cqe + 64);

                wqe_ctr = be16toh(cqe64->wqe_counter);
                wqe_ctr_idx = wqe_ctr & (mcq->wq->wqe_cnt - 1);

                p_ne = ibv_poll_cq(mcq->gcq.ibcq, 1, wc + cnt);
                if (p_ne <= 0)
                        return p_ne;

                wc[cnt].wr_id = mcq->wq->wrid[wqe_ctr_idx];

                mcq->wq->tail = mcq->wq->wqe_head[wqe_ctr_idx] + 1;

                ++mcq->cons_index;
        }
        return cnt;
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
