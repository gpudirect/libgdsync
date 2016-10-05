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

#pragma once

/*
 * Dependencies & Verbs adaptation layer
 */

#include <infiniband/verbs.h>
#include <infiniband/verbs_exp.h>
#include <infiniband/peer_ops.h>

#define ibv_peer_commit			 ibv_exp_peer_commit
#define ibv_peer_commit_qp		 ibv_exp_peer_commit_qp

#define ibv_create_qp_ex		 ibv_exp_create_qp
#define ibv_qp_init_attr_ex		 ibv_exp_qp_init_attr
#define ibv_create_cq_attr_ex		 ibv_exp_cq_init_attr

#define IBV_QP_INIT_ATTR_PD		 IBV_EXP_QP_INIT_ATTR_PD
#define IBV_QP_INIT_ATTR_PEER_DIRECT	 IBV_EXP_QP_INIT_ATTR_PEER_DIRECT

#define IBV_CREATE_CQ_ATTR_PEER_DIRECT	 IBV_EXP_CQ_INIT_ATTR_PEER_DIRECT

#define IBV_PEER_OP_FENCE		 IBV_EXP_PEER_OP_FENCE
#define IBV_PEER_OP_STORE_DWORD		 IBV_EXP_PEER_OP_STORE_DWORD
#define IBV_PEER_OP_STORE_QWORD		 IBV_EXP_PEER_OP_STORE_QWORD
#define IBV_PEER_OP_POLL_AND_DWORD	 IBV_EXP_PEER_OP_POLL_AND_DWORD
#define IBV_PEER_OP_POLL_NOR_DWORD	 IBV_EXP_PEER_OP_POLL_NOR_DWORD
#define IBV_PEER_OP_POLL_GEQ_DWORD	 IBV_EXP_PEER_OP_POLL_GEQ_DWORD
#define IBV_PEER_OP_COPY_BLOCK           IBV_EXP_PEER_OP_COPY_BLOCK

#define IBV_PEER_OP_FENCE_CAP		 IBV_EXP_PEER_OP_FENCE_CAP
#define IBV_PEER_OP_STORE_DWORD_CAP	 IBV_EXP_PEER_OP_STORE_DWORD_CAP
#define IBV_PEER_OP_STORE_QWORD_CAP	 IBV_EXP_PEER_OP_STORE_QWORD_CAP
#define IBV_PEER_OP_COPY_BLOCK_CAP       IBV_EXP_PEER_OP_COPY_BLOCK_CAP
#define IBV_PEER_OP_POLL_AND_DWORD_CAP	 IBV_EXP_PEER_OP_POLL_AND_DWORD_CAP
#define IBV_PEER_OP_POLL_NOR_DWORD_CAP	 IBV_EXP_PEER_OP_POLL_NOR_DWORD_CAP

#define IBV_PEER_FENCE_OP_READ           IBV_EXP_PEER_FENCE_OP_READ      
#define IBV_PEER_FENCE_OP_WRITE          IBV_EXP_PEER_FENCE_OP_WRITE
#define IBV_PEER_FENCE_FROM_CPU          IBV_EXP_PEER_FENCE_FROM_CPU
#define IBV_PEER_FENCE_FROM_HCA          IBV_EXP_PEER_FENCE_FROM_HCA
#define IBV_PEER_FENCE_MEM_SYS           IBV_EXP_PEER_FENCE_MEM_SYS      
#define IBV_PEER_FENCE_MEM_PEER          IBV_EXP_PEER_FENCE_MEM_PEER

#define ibv_peer_direct_attr		 ibv_exp_peer_direct_attr
#define ibv_peer_direction		 ibv_exp_peer_direction
#define ibv_peer_op			 ibv_exp_peer_op

#define IBV_ROLLBACK_ABORT_UNCOMMITED    IBV_EXP_ROLLBACK_ABORT_UNCOMMITED
#define IBV_ROLLBACK_ABORT_LATE		 IBV_EXP_ROLLBACK_ABORT_LATE

#define ibv_rollback_ctx		 ibv_exp_rollback_ctx
#define ibv_rollback_qp			 ibv_exp_rollback_qp
#define ibv_peer_peek			 ibv_exp_peer_peek
#define ibv_peer_peek_cq		 ibv_exp_peer_peek_cq
#define ibv_peer_abort_peek		 ibv_exp_peer_abort_peek
#define ibv_peer_abort_peek_cq		 ibv_exp_peer_abort_peek_cq

#define IBV_PEER_DIRECTION_FROM_CPU	 IBV_EXP_PEER_DIRECTION_FROM_CPU
#define IBV_PEER_DIRECTION_FROM_HCA	 IBV_EXP_PEER_DIRECTION_FROM_HCA
#define IBV_PEER_DIRECTION_FROM_PEER     IBV_EXP_PEER_DIRECTION_FROM_PEER
#define IBV_PEER_DIRECTION_TO_CPU        IBV_EXP_PEER_DIRECTION_TO_CPU 
#define IBV_PEER_DIRECTION_TO_HCA        IBV_EXP_PEER_DIRECTION_TO_HCA 
#define IBV_PEER_DIRECTION_TO_PEER       IBV_EXP_PEER_DIRECTION_TO_PEER

#define ibv_peer_buf			 ibv_exp_peer_buf
#define ibv_peer_buf_alloc_attr		 ibv_exp_peer_buf_alloc_attr

#define ibv_create_cq_ex_(ctx, attr, n, ch) \
		ibv_exp_create_cq(ctx, n, NULL, ch, 0, attr)

#include <cuda.h>
#include <gdrapi.h>

#ifdef  __cplusplus
# define GDS_BEGIN_DECLS  extern "C" {
# define GDS_END_DECLS    }
#else
# define GDS_BEGIN_DECLS
# define GDS_END_DECLS
#endif

GDS_BEGIN_DECLS

#include <gdsync/core.h>
#include <gdsync/tools.h>
#include <gdsync/mlx5.h>

GDS_END_DECLS
