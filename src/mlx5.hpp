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
#include "utils.hpp"

//-----------------------------------------------------------------------------

#define GDS_MLX5_ROLLBACK_ID_PARITY_MASK (1ULL << 63)
#define GDS_MLX5_LAST_PEEK_ENTRY (-1U)
#define GDS_MLX5_PEEK_ENTRY(cq, n) \
        (n == GDS_MLX5_LAST_PEEK_ENTRY ? NULL : \
         ((struct gds_mlx5_peek_entry *)cq->peer_buf->addr) + n)
#define GDS_MLX5_PEEK_ENTRY_N(cq, pe) \
        (pe == NULL ? GDS_MLX5_LAST_PEEK_ENTRY : \
         ((pe - (struct gds_mlx5_peek_entry *)cq->peer_buf->addr)))

struct gds_mlx5_peek_entry {
        uint32_t busy;
        uint32_t next;
};

enum {
        GDS_MLX5_PEER_PEEK_ABSOLUTE,
        GDS_MLX5_PEER_PEEK_RELATIVE
};

struct gds_mlx5_peer_peek {
        /* IN/OUT - linked list of empty/filled descriptors */
        struct gds_mlx5_peer_op_wr *storage;
        /* IN/OUT - number of allocated/filled descriptors */
        uint32_t entries;
        /* IN - Which CQ entry does the peer want to peek for
         * completion. According to "whence" directive entry
         * chosen as follows:
         * IBV_EXP_PEER_PEEK_ABSOLUTE -
         *  "offset" is absolute index of entry wrapped to 32-bit
         * IBV_EXP_PEER_PEEK_RELATIVE -
         *      "offset" is relative to current poll_cq location.
         */
        uint32_t whence;
        uint32_t offset;
        /* OUT - identifier used in ibv_exp_peer_ack_peek_cq to advance CQ */
        uint64_t peek_id;
        uint32_t comp_mask; /* Reserved for future expensions, must be 0 */
};

enum gds_mlx5_peer_op {
        GDS_MLX5_PEER_OP_RESERVED1   = 1,

        GDS_MLX5_PEER_OP_FENCE       = 0,

        GDS_MLX5_PEER_OP_STORE_DWORD = 4,
        GDS_MLX5_PEER_OP_STORE_QWORD = 2,
        GDS_MLX5_PEER_OP_COPY_BLOCK  = 3,

        GDS_MLX5_PEER_OP_POLL_AND_DWORD  = 12, 
        GDS_MLX5_PEER_OP_POLL_NOR_DWORD  = 13, 
        GDS_MLX5_PEER_OP_POLL_GEQ_DWORD  = 14, 
};

enum gds_peer_op_caps {
        GDS_PEER_OP_FENCE_CAP   = (1 << GDS_MLX5_PEER_OP_FENCE),
        GDS_PEER_OP_STORE_DWORD_CAP = (1 << GDS_MLX5_PEER_OP_STORE_DWORD),
        GDS_PEER_OP_STORE_QWORD_CAP = (1 << GDS_MLX5_PEER_OP_STORE_QWORD),
        GDS_PEER_OP_COPY_BLOCK_CAP  = (1 << GDS_MLX5_PEER_OP_COPY_BLOCK),
        GDS_PEER_OP_POLL_AND_DWORD_CAP
                = (1 << GDS_MLX5_PEER_OP_POLL_AND_DWORD),
        GDS_PEER_OP_POLL_NOR_DWORD_CAP
                = (1 << GDS_MLX5_PEER_OP_POLL_NOR_DWORD),
        GDS_PEER_OP_POLL_GEQ_DWORD_CAP
                = (1 << GDS_MLX5_PEER_OP_POLL_GEQ_DWORD),
};

struct gds_mlx5_peer_op_wr {
        struct gds_mlx5_peer_op_wr *next;
        enum gds_mlx5_peer_op type;
        union {
                struct {
                        uint64_t fence_flags; /* from gds_peer_fence */
                } fence;

                struct {
                        uint32_t  data;
                        uint64_t  target_id;
                        size_t    offset;
                } dword_va; /* Use for all operations targeting dword */

                struct {
                        uint64_t  data;
                        uint64_t  target_id;
                        size_t    offset;
                } qword_va; /* Use for all operations targeting qword */

                struct {
                        void     *src;
                        uint64_t  target_id;
                        size_t    offset;
                        size_t    len;
                } copy_op;
        } wr;
        uint32_t comp_mask; /* Reserved for future expensions, must be 0 */
};

struct gds_mlx5_peer_commit {
        /* IN/OUT - linked list of empty/filled descriptors */
        struct gds_mlx5_peer_op_wr *storage;
        /* IN/OUT - number of allocated/filled descriptors */
        uint32_t entries;
        /* OUT - identifier used in gds_rollback_qp to rollback WQEs set */
        uint64_t rollback_id;
        uint32_t comp_mask; /* Reserved for future expensions, must be 0 */
};

enum gds_mlx5_rollback_flags {
        /* Abort all WQEs which were not committed to HW yet.
         * rollback_id is ignored. **/
        GDS_MLX5_ROLLBACK_ABORT_UNCOMMITED = (1 << 0),
        /* Abort the request even if there are following requests
         * being aborted as well. **/
        GDS_MLX5_ROLLBACK_ABORT_LATE = (1 << 1),
};

struct gds_mlx5_rollback_ctx {
        uint64_t rollback_id; /* from ibv_exp_peer_commit call */
        uint32_t flags; /* from ibv_exp_rollback_flags */
        uint32_t comp_mask; /* Reserved for future expensions, must be 0 */
};

typedef struct gds_mlx5_send_request {
        struct gds_mlx5_peer_commit commit;
        struct gds_mlx5_peer_op_wr wr[GDS_SEND_INFO_MAX_OPS];
} gds_mlx5_send_request_t;

static inline gds_mlx5_send_request_t *to_gds_msreq(gds_send_request_t *req) {
        return (gds_mlx5_send_request_t *)req;
}

typedef struct gds_mlx5_wait_request {
        struct gds_mlx5_peer_peek peek;
        struct gds_mlx5_peer_op_wr wr[GDS_WAIT_INFO_MAX_OPS];
} gds_mlx5_wait_request_t;

static inline gds_mlx5_wait_request_t *to_gds_mwreq(gds_wait_request_t *req) {
        return (gds_mlx5_wait_request_t *)req;
}

typedef struct gds_mlx5_cq {
        gds_cq_t                gcq;
        uint32_t                cons_index;
        struct mlx5dv_cq        dvcq;
        uint64_t               *wrid;
        uint64_t                active_buf_va_id;
        gds_peer_attr          *peer_attr;
        uint64_t                peer_va_id;
        uint32_t                peer_dir;
        struct gds_buf         *peer_buf;
        struct gds_mlx5_peek_entry **peer_peek_table;
        struct gds_mlx5_peek_entry  *peer_peek_free;
} gds_mlx5_cq_t;

typedef struct gds_mlx5_qp_peer {
        gds_peer_attr *peer_attr;
        uint32_t scur_post;

        struct {
                uint64_t va_id;
                size_t size;
        } dbr;

        struct {
                uint64_t va_id;
        } bf;
} gds_mlx5_qp_peer_t;

typedef struct gds_mlx5_qp {
        gds_qp_t gqp;

        struct mlx5dv_qp dvqp;

        unsigned int sq_cur_post;
        uint8_t	sq_signal_bits;

        unsigned int bf_offset;

        uint8_t fm_cache;

        gds_mlx5_qp_peer_t *qp_peer;
} gds_mlx5_qp_t;

static inline gds_mlx5_cq_t *to_gds_mcq(struct gds_cq *gcq) {
        return container_of(gcq, gds_mlx5_cq_t, gcq);
}

static inline gds_mlx5_qp_t *to_gds_mqp(struct gds_qp *gqp) {
        return container_of(gqp, gds_mlx5_qp_t, gqp);
}

int gds_mlx5_rollback_send(gds_mlx5_qp_t *mqp, struct gds_mlx5_rollback_ctx *rollback);
int gds_mlx5_post_send(gds_mlx5_qp_t *mqp, gds_send_wr *p_ewr, gds_send_wr **bad_ewr, gds_mlx5_peer_commit *commit);
int gds_mlx5_peer_peek_cq(gds_mlx5_cq_t *mcq, struct gds_mlx5_peer_peek *peek);

int gds_mlx5_create_cq(struct ibv_cq *ibcq, gds_peer_attr *peer_attr, gds_mlx5_cq_t **out_mcq);
void gds_mlx5_destroy_cq(gds_mlx5_cq_t *mcq);

int gds_mlx5_alloc_parent_domain(struct ibv_pd *p_pd, struct ibv_context *ibctx, gds_peer_attr *peer_attr, struct ibv_pd **out_pd, gds_mlx5_qp_peer_t **out_qp_peer);

int gds_mlx5_create_qp(struct ibv_qp *ibqp, gds_qp_init_attr_t *qp_attr, gds_mlx5_cq_t *tx_mcq, gds_mlx5_cq_t *rx_mcq, gds_mlx5_qp_peer_t *qp_peer, gds_mlx5_qp_t **out_mqp);
void gds_mlx5_destroy_qp(gds_mlx5_qp_t *mqp);

int gds_mlx5_post_ops_on_cpu(size_t n_ops, struct gds_mlx5_peer_op_wr *op, int post_flags = 0);
int gds_mlx5_post_ops(gds_peer *peer, size_t n_ops, struct gds_mlx5_peer_op_wr *op, gds_op_list_t &ops, int post_flags = 0);

void gds_mlx5_dump_ops(struct gds_mlx5_peer_op_wr *op, size_t count);

//-----------------------------------------------------------------------------

/*
 * Local variables:
 *  c-indent-level: 8
 *  c-basic-offset: 8
 *  tab-width: 8
 *  indent-tabs-mode: nil
 * End:
 */
