/****
 * Copyright (c) 2016, NVIDIA Corporation.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *    * Redistributions of source code must retain the above copyright notice,
 *      this list of conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *    * Neither the name of the NVIDIA Corporation nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 ****/

#pragma once

#ifndef __GDSYNC_H__
#error "don't include directly this header, use gdsync.h always"
#endif

#define GDS_API_MAJOR_VERSION    2U
#define GDS_API_MINOR_VERSION    2U
#define GDS_API_VERSION          ((GDS_API_MAJOR_VERSION << 16) | GDS_API_MINOR_VERSION)
#define GDS_API_VERSION_COMPATIBLE(v) \
    ( ((((v) & 0xffff0000U) >> 16) == GDS_API_MAJOR_VERSION) &&   \
      ((((v) & 0x0000ffffU) >> 0 ) >= GDS_API_MINOR_VERSION) )

#define IBV_EXP_SEND_GET_INFO (1 << 28)

typedef enum gds_param {
    GDS_PARAM_VERSION,
    GDS_NUM_PARAMS
} gds_param_t;

int gds_query_param(gds_param_t param, int *value);

enum gds_create_qp_flags {
    GDS_CREATE_QP_DEFAULT      = 0,
    GDS_CREATE_QP_WQ_ON_GPU    = 1<<0,
    GDS_CREATE_QP_TX_CQ_ON_GPU = 1<<1,
    GDS_CREATE_QP_RX_CQ_ON_GPU = 1<<2,
    GDS_CREATE_QP_WQ_DBREC_ON_GPU = 1<<5,
};

typedef struct ibv_exp_qp_init_attr gds_qp_init_attr_t;
typedef struct ibv_exp_send_wr gds_send_wr;

struct gds_cq {
        struct ibv_cq *cq;
        uint32_t curr_offset;
};

struct gds_qp {
        struct ibv_qp *qp;
        struct gds_cq send_cq;
        struct gds_cq recv_cq;
        struct ibv_exp_res_domain * res_domain;
        struct ibv_context *dev_context;

        void* swq; //send work queue pointer
        size_t swq_cnt; //counter tracking swq location
        size_t swq_size; //size of the swq (Blocks)
        size_t swq_stride; //size of Blocks
};

/* \brief: Create a peer-enabled QP attached to the specified GPU id.
 *
 * Peer QPs require dedicated send and recv CQs, e.g. cannot (easily)
 * use SRQ.
 */

struct gds_qp *gds_create_qp(struct ibv_pd *pd, struct ibv_context *context,
                             gds_qp_init_attr_t *qp_init_attr,
                             int gpu_id, int flags);

/* \brief: Destroy a peer-enabled QP
 *
 * The associated CQs are destroyed as well.
 */
int gds_destroy_qp(struct gds_qp *qp);

/* \brief: CPU-synchronous post send for peer QPs
 *
 * Notes:
 * - this API might have higher overhead than ibv_post_send. 
 * - It is provided for convenience only.
 */
int gds_post_send(struct gds_qp *qp, gds_send_wr *wr, gds_send_wr **bad_wr);

/* \brief: CPU-synchronous post recv for peer QPs
 *
 * Notes:
 * - there is no GPU-synchronous version of this because there is not a use case for it.
 */
int gds_post_recv(struct gds_qp *qp, struct ibv_recv_wr *wr, struct ibv_recv_wr **bad_wr);

int gds_stream_wait_cq(CUstream stream, struct gds_cq *cq, int flags);

/* \brief: GPU stream-synchronous send for peer QPs
 *
 * Notes:
 * - execution of the send operation happens in CUDA stream order
 */
int gds_stream_queue_send(CUstream stream, struct gds_qp *qp, gds_send_wr *p_ewr, gds_send_wr **bad_ewr);


// batched submission APIs

typedef enum gds_memory_type {
        GDS_MEMORY_GPU  = 1, /*< use this flag for both cudaMalloc/cuMemAlloc and cudaMallocHost/cuMemHostAlloc */
        GDS_MEMORY_HOST = 2,
        GDS_MEMORY_IO   = 4,
	GDS_MEMORY_MASK = 0x7
} gds_memory_type_t;

// Note: those flags below must not overlap with gds_memory_type_t
typedef enum gds_wait_flags {
        GDS_WAIT_POST_FLUSH_REMOTE = 1<<3, /*< add a trailing flush of the ingress GPUDirect RDMA data path on the GPU owning the stream */
        GDS_WAIT_POST_FLUSH = GDS_WAIT_POST_FLUSH_REMOTE /*< alias for backward compatibility */
} gds_wait_flags_t;

typedef enum gds_write_flags {
        GDS_WRITE_PRE_BARRIER_SYS = 1<<4, /*< add a heading memory barrier to the write value operation */
        GDS_WRITE_PRE_BARRIER = GDS_WRITE_PRE_BARRIER_SYS /*< alias for backward compatibility */
} gds_write_flags_t;

typedef enum gds_write_memory_flags {
        GDS_WRITE_MEMORY_POST_BARRIER_SYS = 1<<4, /*< add a trailing memory barrier to the memory write operation */
        GDS_WRITE_MEMORY_PRE_BARRIER_SYS  = 1<<5 /*< add a heading memory barrier to the memory write operation, for convenience only as not a native capability */
} gds_write_memory_flags_t;

typedef enum gds_membar_flags {
        GDS_MEMBAR_FLUSH_REMOTE = 1<<4,
        GDS_MEMBAR_DEFAULT      = 1<<5,
        GDS_MEMBAR_SYS          = 1<<6,
        GDS_MEMBAR_MLX5         = 1<<7 /*< modify the scope of the barrier, for internal use only */
} gds_membar_flags_t;

enum {
        GDS_SEND_INFO_MAX_OPS = 32,
        GDS_WAIT_INFO_MAX_OPS = 32
};

typedef enum gds_update_send_info_type {
        GDS_SYNC     = 1<<1,
        GDS_ASYNC    = 1<<2
} gds_update_send_info_type_t;



/**
 * Represents a posted send operation on a particular QP
 */

#define GDS_SEND_MAX_SGE 16

struct ptr_to_sge{
    uintptr_t ptr_to_size;
    uintptr_t ptr_to_lkey;
    uintptr_t ptr_to_addr;
    int offset;
};

struct gds_swr_info{
    size_t num_sge;
    struct ptr_to_sge sge_list[GDS_SEND_MAX_SGE];
    size_t wr_id;
};

typedef struct gds_send_request_info {
    struct gds_swr_info swr_info;
    //Size info
    uintptr_t ptr_to_size_wqe_h;
    CUdeviceptr ptr_to_size_wqe_d;
    uintptr_t ptr_to_size_new_h;
    CUdeviceptr ptr_to_size_new_d;

    //lkey info
    uintptr_t ptr_to_lkey_wqe_h;
    CUdeviceptr ptr_to_lkey_wqe_d;
    uintptr_t ptr_to_lkey_new_h;
    CUdeviceptr ptr_to_lkey_new_d;

    //Addr info
    uintptr_t ptr_to_addr_wqe_h;
    CUdeviceptr ptr_to_addr_wqe_d;
    uintptr_t ptr_to_addr_new_h;
    CUdeviceptr ptr_to_addr_new_d;

} gds_send_request_info_t;

typedef struct gds_send_request {
        struct ibv_exp_peer_commit commit;
        struct peer_op_wr wr[GDS_SEND_INFO_MAX_OPS];
        struct gds_send_request_info gds_sinfo;
} gds_send_request_t;

int gds_prepare_send(struct gds_qp *qp, gds_send_wr *p_ewr, gds_send_wr **bad_ewr, gds_send_request_t *request);
int gds_prepare_send_info(gds_send_request_t *request,
                        void * ptr_to_size_src, int ptr_to_size_flags,
                        void * ptr_to_lkey_src, int ptr_to_lkey_flags,
                        void * ptr_to_addr_from, int ptr_to_addr_flags);
int gds_update_send_info(gds_send_request_t *request, int send_flags, CUstream stream);
int gds_stream_post_send(CUstream stream, gds_send_request_t *request);
int gds_stream_post_send_all(CUstream stream, int count, gds_send_request_t *request);


/**
 * Represents a wait operation on a particular CQ
 */

typedef struct gds_wait_request {
        struct ibv_exp_peer_peek peek;
        struct peer_op_wr wr[GDS_WAIT_INFO_MAX_OPS];
} gds_wait_request_t;

/**
 * Initializes a wait request out of the next heading CQE, which is kept in
 * cq->curr_offset.
 *
 * flags: must be 0
 */
int gds_prepare_wait_cq(struct gds_cq *cq, gds_wait_request_t *request, int flags);

/**
 * Issues the descriptors contained in request on the CUDA stream
 *
 */
int gds_stream_post_wait_cq(CUstream stream, gds_wait_request_t *request);

/**
 * Issues the descriptors contained in the array of requests on the CUDA stream.
 * This has potentially less overhead than submitting each request individually.
 *
 */
int gds_stream_post_wait_cq_all(CUstream stream, int count, gds_wait_request_t *request);

/**
 * \brief CPU-synchronously enable polling on request
 *
 * Unblock calls to ibv_poll_cq. CPU will do what is necessary to make the corresponding
 * CQE poll-able.
 *
 */
int gds_post_wait_cq(struct gds_cq *cq, gds_wait_request_t *request, int flags);



/**
 * Represents the condition operation for wait operations on memory words
 */

typedef enum gds_wait_cond_flag {
        GDS_WAIT_COND_GEQ = 0, // must match verbs_exp enum
        GDS_WAIT_COND_EQ,
        GDS_WAIT_COND_AND,
        GDS_WAIT_COND_NOR
} gds_wait_cond_flag_t;

/**
 * Represents a wait operation on a 32-bits memory word
 */

typedef struct gds_wait_value32 { 
        uint32_t  *ptr;
        uint32_t   value;
        gds_wait_cond_flag_t cond_flags;
        int        flags; // takes gds_memory_type_t | gds_wait_flags_t
} gds_wait_value32_t;

/**
 * flags: gds_memory_type_t | gds_wait_flags_t
 */
int gds_prepare_wait_value32(gds_wait_value32_t *desc, uint32_t *ptr, uint32_t value, gds_wait_cond_flag_t cond_flags, int flags);



/**
 * Represents a write operation on a 32-bits memory word
 */

typedef struct gds_write_value32 { 
        uint32_t  *ptr;
        uint32_t   value;
        int        flags; // takes gds_memory_type_t | gds_write_flags_t
} gds_write_value32_t;

/**
 * flags:  gds_memory_type_t | gds_write_flags_t
 */
int gds_prepare_write_value32(gds_write_value32_t *desc, uint32_t *ptr, uint32_t value, int flags);



/**
 * Represents a staged copy operation
 * the src buffer can be reused after the API call
 */

typedef struct gds_write_memory { 
        uint8_t       *dest;
        const uint8_t *src;
        size_t         count;
        int            flags; // takes gds_memory_type_t | gds_write_memory_flags_t
} gds_write_memory_t;

/**
 * flags:  gds_memory_type_t | gds_write_memory_flags_t
 */
int gds_prepare_write_memory(gds_write_memory_t *desc, uint8_t *dest, const uint8_t *src, size_t count, int flags);



typedef enum gds_tag { 
        GDS_TAG_SEND,
        GDS_TAG_WAIT,
        GDS_TAG_WAIT_VALUE32,
        GDS_TAG_WRITE_VALUE32,
        GDS_TAG_WRITE_MEMORY
} gds_tag_t;

typedef struct gds_descriptor {
        gds_tag_t tag; /**< selector for union below */
        union {
                gds_send_request_t  *send;
                gds_wait_request_t  *wait;
                gds_wait_value32_t   wait32;
                gds_write_value32_t  write32;
                gds_write_memory_t   writemem;
        };
} gds_descriptor_t;

/**
 * \brief: post descriptors for peer QPs synchronized to the specified CUDA stream
 *
 * \param flags - must be 0
 *
 * \return
 * 0 on success or one standard errno error
 *
 */
int gds_stream_post_descriptors(CUstream stream, size_t n_descs, gds_descriptor_t *descs, int flags);

/**
 * \brief: CPU-synchronous post descriptors for peer QPs
 *
 *
 * \param flags - must be 0
 *
 * \return
 * 0 on success or one standard errno error
 * 
 *
 * Notes:
 * - This API might have higher overhead than issuing multiple ibv_post_send. 
 * - It is provided for convenience only.
 * - It might fail if trying to access CUDA device memory pointers
 */
int gds_post_descriptors(size_t n_descs, gds_descriptor_t *descs, int flags);

/**
 * \brief: TODO
 *
 *
 * \param flags - TODO
 *
 * \return
 * 0 on success or one standard errno error
 *
 *
 * Notes:
 * - TODO.
 */
int gds_report_post(struct gds_qp *gqp  /*, struct gds_send_wr* wr*/);

/**
 * \brief: TODO
 *
 *
 * \param flags - TODO
 *
 * \return
 * 0 on success or one standard errno error
 *
 *
 * Notes:
 * - TODO.
 */
int gds_query_last_info(struct gds_qp* qp, struct gds_swr_info* gds_info);

/*
 * Local variables:
 *  c-indent-level: 8
 *  c-basic-offset: 8
 *  tab-width: 8
 *  indent-tabs-mode: nil
 * End:
 */
