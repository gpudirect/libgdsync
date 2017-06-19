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

#include <cuda.h>
#include <infiniband/verbs_exp.h>

#define __ASSERT(cond, cond_str)                                        \
        do {                                                            \
                if (!(cond)) {                                          \
		fprintf(stderr, "Assertion \"%s\" failed at %s:%d\n", cond_str, __FILE__, __LINE__); \
		exit(EXIT_FAILURE);                                     \
	}                                                               \
} while(0)

#define ASSERT(x) __ASSERT((x), #x)


#define __GDSCHECK(stmt, cond_str)					\
	do {								\
		int result = (stmt);				\
		if (0 != result) {				\
			const char *err_str = strerror(result);               \
			fprintf(stderr, "Assertion \"%s != cudaSuccess\" failed at %s:%d error=%d(%s)\n", cond_str, __FILE__, __LINE__, result, err_str); \
			exit(EXIT_FAILURE);                             \
		}							\
        } while (0)

#define GDSCHECK(stmt) __GDSCHECK(stmt, #stmt)

//----

#define __CUCHECK(stmt, cond_str)					\
	do {								\
		CUresult result = (stmt);				\
		if (CUDA_SUCCESS != result) {				\
			const char *err_str = NULL;			\
			cuGetErrorString(result, &err_str);		\
			fprintf(stderr, "Assertion \"%s != cudaSuccess\" failed at %s:%d error=%d(%s)\n", cond_str, __FILE__, __LINE__, result, err_str); \
			exit(EXIT_FAILURE);                             \
		}							\
        } while (0)

#define CUCHECK(stmt) __CUCHECK(stmt, #stmt)

//----

#define __CUDACHECK(stmt, cond_str)					\
	do {								\
		cudaError_t result = (stmt);				\
		if (cudaSuccess != result) {				\
			fprintf(stderr, "Assertion \"%s != cudaSuccess\" failed at %s:%d error=%d(%s)\n", cond_str, __FILE__, __LINE__, result, cudaGetErrorString(result)); \
			exit(EXIT_FAILURE);				\
		}							\
        } while (0)

#define CUDACHECK(stmt) __CUDACHECK(stmt, #stmt)

enum gpu_msg_level {
    GPU_MSG_DEBUG = 1,
    GPU_MSG_INFO,
    GPU_MSG_WARN,
    GPU_MSG_ERROR
};


#define gpu_msg(LVL, LVLSTR, FMT, ARGS...)   fprintf(stderr, LVLSTR "[%s] " FMT, __FUNCTION__ ,##ARGS)

#if 0
#define gpu_dbg(FMT, ARGS...)  do {} while(0)
#else
#define gpu_dbg(FMT, ARGS...)  do { if (gpu_dbg_enabled()) gpu_msg(GPU_MSG_DEBUG, "DBG:  ", FMT, ## ARGS); } while(0)
#endif
#define gpu_dbgc(CNT, FMT, ARGS...) do { static int __cnt = 0; if (__cnt++ < CNT) gpu_dbg(FMT, ## ARGS); } while(0)
#define gpu_info(FMT, ARGS...) gpu_msg(GPU_MSG_INFO,  "INFO: ", FMT, ## ARGS)
#define gpu_infoc(CNT, FMT, ARGS...) do { static int __cnt = 0; if (__cnt++ < CNT) gpu_info(FMT, ## ARGS); } while(0)
#define gpu_warn(FMT, ARGS...) gpu_msg(GPU_MSG_WARN,  "WARN: ", FMT, ## ARGS)
#define gpu_err(FMT, ARGS...)  gpu_msg(GPU_MSG_ERROR, "ERR:  ", FMT, ##ARGS)

// oversubscribe SM by factor 2
static const int over_sub_factor = 2;
extern CUstream gpu_stream;
extern CUstream gpu_stream_server;
extern CUstream gpu_stream_client;
extern int gpu_num_sm;

BEGIN_C_DECLS

int gpu_dbg_enabled();

/* sched_mode=(CU_CTX_SCHED_SPIN,CU_CTX_SCHED_YIELD,CU_CTX_SCHED_BLOCKING_SYNC, CU_CTX_SCHED_AUTO) */
int gpu_init(int gpu_id, int sched_mode);
int gpu_finalize();
void *gpu_malloc(size_t page_size, size_t min_size);
int gpu_free(void *ptr);
int gpu_memset(void *ptr, const unsigned char c, size_t size);
int gpu_register_host_mem(void *ptr, size_t size);

int gpu_launch_kernel(size_t size, int is_peersync);
int gpu_launch_kernel_on_stream(size_t size, int is_peersync, CUstream s);
void gpu_post_release_tracking_event();
int gpu_wait_tracking_event(int tmout_us);

int gpu_launch_void_kernel();
int gpu_launch_dummy_kernel();

END_C_DECLS

/*
 * Local variables:
 *  c-indent-level: 8
 *  c-basic-offset: 8
 *  tab-width: 8
 * End:
 */
