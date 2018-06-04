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

#ifndef __STDC_FORMAT_MACROS
#warning "__STDC_FORMAT_MACROS should be defined to pull definition of PRIx64, etc"
#endif
#include <inttypes.h> // to pull PRIx64

// internal assert function

void gds_assert(const char *cond, const char *file, unsigned line, const char *function);

#define GDS_ASSERT2(COND)                                               \
        do {                                                            \
                if (!(COND))                                            \
                        gds_assert(#COND, __FILE__, __LINE__, __FUNCTION__); \
        }                                                               \
        while(0)

#define GDS_ASSERT(COND) GDS_ASSERT2(COND)


// CUDA error checking

#define __CUCHECK(stmt, cond_str)					\
	do {								\
		CUresult result = (stmt);				\
		if (CUDA_SUCCESS != result) {				\
			const char *err_str = NULL;			\
			cuGetErrorString(result, &err_str);		\
			fprintf(stderr, "Assertion \"%s != cudaSuccess\" failed at %s:%d error=%d(%s)\n", \
                                cond_str, __FILE__, __LINE__, result, err_str); \
			exit(EXIT_FAILURE);                             \
		}							\
        } while (0)

#define CUCHECK(stmt) __CUCHECK(stmt, #stmt)

template <typename T>
static inline void gds_atomic_set(T *ptr, T value)
{
        *(volatile T*)ptr = value;
}

template <typename T>
static inline T gds_atomic_get(T *ptr)
{
        return *(volatile T*)ptr;
}

#define ROUND_UP(V,SIZE) (((V)+(SIZE)-1)/(SIZE)*(SIZE))

//-----------------------------------------------------------------------------

//static inline size_t host_page_size() { return sysconf(_SC_PAGESIZE); }
//#define GDS_HOST_PAGE_SIZE host_page_size()
#define GDS_HOST_PAGE_BITS 12
#define GDS_HOST_PAGE_SIZE (1ULL<<GDS_HOST_PAGE_BITS)
#define GDS_HOST_PAGE_OFF  (GDS_HOST_PAGE_SIZE-1)
#define GDS_HOST_PAGE_MASK (~(GDS_HOST_PAGE_OFF))

#define GDS_GPU_PAGE_BITS 16
#define GDS_GPU_PAGE_SIZE (1ULL<<GDS_GPU_PAGE_BITS)
#define GDS_GPU_PAGE_OFF  (GDS_GPU_PAGE_SIZE-1)
#define GDS_GPU_PAGE_MASK (~(GDS_GPU_PAGE_OFF))

//-----------------------------------------------------------------------------
// tracing support

enum gds_msg_level {
    GDS_MSG_DEBUG = 1,
    GDS_MSG_INFO,
    GDS_MSG_WARN,
    GDS_MSG_ERROR
};

#define gds_stream stderr
//#define gds_stream stdout

int gds_dbg_enabled();
#define gds_msg(LVL, LVLSTR, FMT, ARGS...)   do {			\
		fprintf(gds_stream, "[%d] GDS " LVLSTR " %s() " FMT, getpid(), __FUNCTION__ ,##ARGS); \
		fflush(gds_stream);                                     \
	} while(0)

#define gds_dbg(FMT, ARGS...)  do { if (gds_dbg_enabled()) gds_msg(GDS_MSG_DEBUG, "DBG ", FMT, ## ARGS); } while(0)
#define gds_dbgc(CNT, FMT, ARGS...) do { static int __cnt = 0; if (__cnt++ < CNT) gds_dbg(FMT, ## ARGS); } while(0)

#define gds_info(FMT, ARGS...) gds_msg(GDS_MSG_INFO,  "INFO ", FMT, ## ARGS)
#define gds_infoc(CNT, FMT, ARGS...) do { static int __cnt = 0; if (__cnt++ < CNT) gds_info(FMT, ## ARGS); } while(0)

#define gds_warn(FMT, ARGS...) gds_msg(GDS_MSG_WARN,  "WARN ", FMT, ## ARGS)
#define gds_warnc(CNT, FMT, ARGS...) do { static int __cnt = 0; if (__cnt++ < CNT) gds_warn(FMT, ## ARGS); } while(0)
#define gds_warn_once(FMT, ARGS...) gds_warnc(1, FMT, ## ARGS)

#define gds_err(FMT, ARGS...)  gds_msg(GDS_MSG_ERROR, "ERR  ", FMT, ##ARGS)


//-----------------------------------------------------------------------------

static inline int gds_curesult_to_errno(CUresult result)
{
        int retcode = 0;
        switch (result) {
        case CUDA_SUCCESS:             retcode = 0; break;
        case CUDA_ERROR_NOT_SUPPORTED: retcode = EPERM; break;
        case CUDA_ERROR_INVALID_VALUE: retcode = EINVAL; break;
        case CUDA_ERROR_OUT_OF_MEMORY: retcode = ENOMEM; break;
        // TODO: add missing cases
        default: retcode = EIO; break;
        }
        return retcode;
}

static inline gds_memory_type_t memtype_from_flags(int flags) {
        gds_memory_type_t ret = (gds_memory_type_t)(flags & GDS_MEMORY_MASK);
        return ret;
}

static inline bool is_valid(gds_memory_type_t type)
{
        bool ret = true;
        if (ret < GDS_MEMORY_GPU || ret > GDS_MEMORY_IO) {
                ret = false;
        }
        return ret;
}

static inline bool is_valid(gds_wait_cond_flag_t cond)
{
        bool ret = true;

        if ((int)cond < GDS_WAIT_COND_GEQ || (int)cond > GDS_WAIT_COND_NOR) {
                gds_dbg("cond flag=0x%x\n", cond);
                ret = false;
        }
        return ret;
}

static inline uint32_t gds_qword_lo(uint64_t v) {
        return (uint32_t)(v);
}

static inline uint32_t gds_qword_hi(uint64_t v) {
        return (uint32_t)(v >> 32);
}

//-----------------------------------------------------------------------------

typedef enum gds_alloc_cq_flags {
        GDS_ALLOC_CQ_DEFAULT = 0, // default on Host memory
        GDS_ALLOC_CQ_ON_GPU  = 1<<0,
        GDS_ALLOC_CQ_MASK    = 1<<0
} gds_alloc_cq_flags_t;

typedef enum gds_alloc_qp_flags {
        GDS_ALLOC_WQ_DEFAULT    = 0, // default on Host memory
        GDS_ALLOC_WQ_ON_GPU     = 1,
        GDS_ALLOC_WQ_MASK       = 1<<0,

        GDS_ALLOC_DBREC_DEFAULT = 0, // default on Host memory
        GDS_ALLOC_DBREC_ON_GPU  = 1<<4,
        GDS_ALLOC_DBREC_MASK    = 1<<4        
} gds_alloc_qp_flags_t;

#include <vector>

typedef std::vector<CUstreamBatchMemOpParams> gds_op_list_t;

struct gds_cq *gds_create_cq(struct ibv_context *context, int cqe, void *cq_context, struct ibv_comp_channel *channel, int comp_vector, int gpu_id, gds_alloc_cq_flags_t flags);
int gds_post_pokes(CUstream stream, int count, gds_send_request_t *info, uint32_t *dw, uint32_t val);
int gds_post_pokes_on_cpu(int count, gds_send_request_t *info, uint32_t *dw, uint32_t val);
int gds_stream_post_wait_cq_multi(CUstream stream, int count, gds_wait_request_t *request, uint32_t *dw, uint32_t val);
void gds_dump_wait_request(gds_wait_request_t *request, size_t count);
void gds_dump_param(CUstreamBatchMemOpParams *param);
void gds_dump_params(gds_op_list_t &params);

struct gds_peer;

int gds_fill_membar(gds_peer *peer, gds_op_list_t &param, int flags);
int gds_fill_inlcpy(gds_peer *peer, gds_op_list_t &param, void *ptr, const void *data, size_t n_bytes, int flags);
int gds_fill_poke(gds_peer *peer, gds_op_list_t &param, uint32_t *ptr, uint32_t value, int flags);
int gds_fill_poke64(gds_peer *peer, gds_op_list_t &param, uint64_t *ptr, uint64_t value, int flags);
int gds_fill_poll(gds_peer *peer, gds_op_list_t &param, uint32_t *ptr, uint32_t magic, int cond_flag, int flags);

int gds_stream_batch_ops(gds_peer *peer, CUstream stream, gds_op_list_t &params, int flags);

enum gds_post_ops_flags {
        GDS_POST_OPS_DISCARD_WAIT_FLUSH = 1<<0
};

struct gds_peer;
int gds_post_ops(gds_peer *peer, size_t n_ops, struct peer_op_wr *op, gds_op_list_t &params, int post_flags = 0);
int gds_post_ops_on_cpu(size_t n_descs, struct peer_op_wr *op, int post_flags = 0);
gds_peer *peer_from_stream(CUstream stream);

//-----------------------------------------------------------------------------

/*
 * Local variables:
 *  c-indent-level: 8
 *  c-basic-offset: 8
 *  tab-width: 8
 *  indent-tabs-mode: nil
 * End:
 */
