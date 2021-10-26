/* Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include <unistd.h>
#include <string.h>
#include <assert.h>

#include <infiniband/verbs.h>

#include <gdsync.h>
#include <gdsync/mlx5.h>

#include "objs.hpp"
#include "utils.hpp"

#define GDS_MLX5_DV_DBR_BUF_SIZE                8
#define GDS_MLX5_DV_LOG_MAX_MSG_SIZE            30
#define GDS_MLX5_DV_ATOMIC_MODE                 0x3     // Up to 8 bytes with Remote Micro Application atomics enabled
#define GDS_MLX5_DV_ATOMIC_LIKE_WRITE_EN        0x1     // Enable atomic with RDMA WRITE
#define GDS_MLX5_DV_WQ_SIGNATURE                0x0     // Disable wq signature
#define GDS_MLX5_DV_COUNTER_SET_ID              0x0     // Do not connect to any counter set
#define GDS_MLX5_DV_LAG_TX_PORT_AFFINITY        0x0     // Let the device decide
#define GDS_MLX5_DV_LOG_ACK_REQ_FREQ            0x0     // ACK every packet
#define GDS_MLX5_DV_UAR_ALLOC_TYPE_BF           0x0     // Allocate a BF buffer

#define GDS_MLX5_DV_ROLLBACK_ID_PARITY_MASK (1ULL << 63)
#define GDS_MLX5_DV_LAST_PEEK_ENTRY (-1U)
#define GDS_MLX5_DV_PEEK_ENTRY(mcq, n) \
        (n == GDS_MLX5_DV_LAST_PEEK_ENTRY ? NULL : \
         ((struct gds_mlx5_dv_peek_entry *)mcq->cq_peer->pdata.gbuf->addr) + n)
#define GDS_MLX5_DV_PEEK_ENTRY_N(mcq, pe) \
        (pe == NULL ? GDS_MLX5_DV_LAST_PEEK_ENTRY : \
         ((pe - (struct gds_mlx5_dv_peek_entry *)mcq->cq_peer->pdata.gbuf->addr)))

enum {
        GDS_MLX5_DV_QPC_ST_RC   = 0x0,
        GDS_MLX5_DV_QPC_ST_UC   = 0x1,
        GDS_MLX5_DV_QPC_ST_UD   = 0x2
};

enum {
        GDS_MLX5_DV_QPC_RQ_TYPE_REGULAR = 0x0,
        GDS_MLX5_DV_QPC_RQ_TYPE_SRQ     = 0x1
};

enum {
        GDS_MLX5_DV_SEND_WQE_BB    = 64,
        GDS_MLX5_DV_SEND_WQE_SHIFT = 6,
        GDS_MLX5_DV_RECV_WQE_BB    = 64,
        GDS_MLX5_DV_RECV_WQE_SHIFT = 6,
};

typedef struct gds_mlx5_dv_peek_entry {
        uint32_t busy;
        uint32_t next;
} gds_mlx5_dv_peek_entry_t;

typedef struct gds_mlx5_dv_cq_peer {
        gds_peer_attr *peer_attr;

        struct {
                uint64_t va_id;
                size_t size;
                gds_buf *gbuf;
        } buf;

        struct {
                uint64_t va_id;
                size_t size;
                gds_buf *gbuf;
        } dbr;

        struct {
                uint64_t                        va_id;
                uint32_t                        dir;
                gds_buf                        *gbuf;
                gds_mlx5_dv_peek_entry_t      **peek_table;
                gds_mlx5_dv_peek_entry_t       *peek_free;
        } pdata;
} gds_mlx5_dv_cq_peer_t;

typedef struct gds_mlx5_dv_wq {
        uint64_t       *wrid;
        void           *buf;    // SQ and RQ point to different regions.
        __be32         *dbrec;
        unsigned int    cnt;
        uint64_t        head;
        uint64_t        tail;
} gds_mlx5_dv_wq_t;

typedef struct gds_mlx5_dv_cq {
        gds_cq_t                        gcq;
        uint32_t                        cons_index;
        struct mlx5dv_cq                dvcq;
        gds_mlx5_dv_wq_t               *wq;
        gds_mlx5_dv_cq_peer_t          *cq_peer;
} gds_mlx5_dv_cq_t;

typedef enum gds_mlx5_dv_qp_type {
        GDS_MLX5_DV_QP_TYPE_UNKNOWN = 0,
        GDS_MLX5_DV_QP_TYPE_RC,
        GDS_MLX5_DV_QP_TYPE_UD
} gds_mlx5_dv_qp_type_t;

typedef struct gds_mlx5_dv_qp {
        gds_qp_t                        gqp;
        gds_mlx5_dv_qp_type_t           qp_type;

        struct mlx5dv_devx_obj         *devx_qp;

        gds_buf                        *wq_buf;
        struct mlx5dv_devx_umem        *wq_umem;
        uint64_t                        wq_va_id;

        off_t                           sq_buf_offset;
        off_t                           rq_buf_offset;

        gds_mlx5_dv_wq_t                sq_wq;
        gds_mlx5_dv_wq_t                rq_wq;

        gds_buf                        *dbr_buf;
        struct mlx5dv_devx_umem        *dbr_umem;
        uint64_t                        dbr_va_id;

        struct mlx5dv_devx_uar         *bf_uar;
        size_t                          bf_size;        // Half of UAR reg size
        uint64_t                        bf_va_id;

        uint8_t                         port_num;
        struct ibv_port_attr            port_attr;

        gds_peer_attr                  *peer_attr;

        struct ibv_pd                  *parent_domain;

        struct mlx5_wqe_ctrl_seg       *peer_ctrl_seg;
} gds_mlx5_dv_qp_t;

typedef enum gds_mlx5_dv_peer_op {
        GDS_MLX5_DV_PEER_OP_RESERVED1   = 1,

        GDS_MLX5_DV_PEER_OP_FENCE       = 0,

        GDS_MLX5_DV_PEER_OP_STORE_DWORD = 4,
        GDS_MLX5_DV_PEER_OP_STORE_QWORD = 2,
        GDS_MLX5_DV_PEER_OP_COPY_BLOCK  = 3,

        GDS_MLX5_DV_PEER_OP_POLL_AND_DWORD      = 12,
        GDS_MLX5_DV_PEER_OP_POLL_NOR_DWORD      = 13,
        GDS_MLX5_DV_PEER_OP_POLL_GEQ_DWORD      = 14,
} gds_mlx5_dv_peer_op_t;

typedef enum gds_mlx5_dv_peer_fence {
        GDS_MLX5_DV_PEER_FENCE_OP_READ  = (1 << 0),
        GDS_MLX5_DV_PEER_FENCE_OP_WRITE = (1 << 1),
        GDS_MLX5_DV_PEER_FENCE_FROM_CPU = (1 << 2),
        GDS_MLX5_DV_PEER_FENCE_FROM_HCA = (1 << 3),
        GDS_MLX5_DV_PEER_FENCE_MEM_SYS  = (1 << 4),
        GDS_MLX5_DV_PEER_FENCE_MEM_PEER = (1 << 5),
} gds_mlx5_dv_peer_fence_t;

typedef struct gds_mlx5_dv_peer_op_wr {
        struct gds_mlx5_dv_peer_op_wr *next;
        gds_mlx5_dv_peer_op_t type;
        union {
                struct {
                        uint64_t fence_flags; /* from gds_mlx5_dv_peer_fence_t */
                } fence;

                struct {
                        uint32_t        data;
                        uint64_t        target_id;
                        size_t          offset;
                } dword_va; /* Use for all operations targeting dword */

                struct {
                        uint64_t        data;
                        uint64_t        target_id;
                        size_t          offset;
                } qword_va; /* Use for all operations targeting qword */

                struct {
                        void           *src;
                        uint64_t        target_id;
                        size_t          offset;
                        size_t          len;
                } copy_op;
        } wr;
        uint32_t comp_mask; /* Reserved for future expensions, must be 0 */
} gds_mlx5_dv_peer_op_wr_t;

typedef struct gds_mlx5_dv_peer_commit {
        /* IN/OUT - linked list of empty/filled descriptors */
        gds_mlx5_dv_peer_op_wr_t *storage;
        /* IN/OUT - number of allocated/filled descriptors */
        uint32_t entries;
        /* OUT - identifier used in gds_mlx5_dv_rollback_qp to rollback WQEs set */
        uint64_t rollback_id;
        uint32_t comp_mask; /* Reserved for future expensions, must be 0 */
} gds_mlx5_dv_peer_commit_t;

typedef struct gds_mlx5_dv_send_request {
        gds_mlx5_dv_peer_commit_t commit;
        gds_mlx5_dv_peer_op_wr_t wr[GDS_SEND_INFO_MAX_OPS];
        uint8_t pad1[32];
} gds_mlx5_dv_send_request_t;
static_assert(sizeof(gds_mlx5_dv_send_request_t) % 64 == 0, "gds_mlx5_dv_send_request_t must be 64-byte aligned.");
static_assert(sizeof(gds_mlx5_dv_send_request_t) <= sizeof(gds_send_request_t), "The size of gds_mlx5_dv_send_request_t must be less than or equal to that of gds_send_request_t.");

//-----------------------------------------------------------------------------

static inline gds_mlx5_dv_cq_t *to_gds_mdv_cq(gds_cq_t *gcq) {
        return container_of(gcq, gds_mlx5_dv_cq_t, gcq);
}

static inline gds_mlx5_dv_qp_t *to_gds_mdv_qp(gds_qp_t *gqp) {
        return container_of(gqp, gds_mlx5_dv_qp_t, gqp);
}

static inline gds_mlx5_dv_send_request_t *to_gds_mdv_send_request(gds_send_request_t *gsreq) {
        return (gds_mlx5_dv_send_request_t *)(gsreq);
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
