/* Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "objs.hpp"

//-----------------------------------------------------------------------------

#define GDS_MLX5_ROLLBACK_ID_PARITY_MASK (1ULL << 63)

typedef struct gds_mlx5_cq {
        struct ibv_cq          *cq;
        uint32_t                curr_offset;
        uint32_t                cons_index;
        gds_cq_type_t           type;
        struct mlx5dv_cq        dv_cq;
        uint64_t               *wrid;
        uint64_t                active_buf_va_id;
        gds_peer_attr          *peer_attr;
        uint64_t                peer_va_id;
        uint32_t                peer_dir;
        struct gds_buf         *peer_buf;
        struct gds_mlx5_peek_entry **peer_peek_table;
        struct gds_mlx5_peek_entry  *peer_peek_free;
} gds_mlx5_cq;

static inline gds_mlx5_cq *to_gds_mcq(struct gds_cq *cq) {
        return (gds_mlx5_cq *)cq;
}

static inline struct gds_cq *to_gds_cq(gds_mlx5_cq *mcq) {
        return (struct gds_cq *)mcq;
}

int gds_mlx5_rollback_send(struct gds_qp *qp, struct gds_mlx5_rollback_ctx *rollback);

int gds_mlx5_post_send(struct gds_qp *qp, gds_send_wr *p_ewr, gds_send_wr **bad_ewr, gds_mlx5_peer_commit *commit);

int gds_mlx5_peer_peek_cq(struct gds_cq *cq, struct gds_mlx5_peer_peek *peek);

//-----------------------------------------------------------------------------

/*
 * Local variables:
 *  c-indent-level: 8
 *  c-basic-offset: 8
 *  tab-width: 8
 *  indent-tabs-mode: nil
 * End:
 */
