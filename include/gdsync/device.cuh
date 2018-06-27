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

#include <gdsync.h> // for gds_poll_cond_flag_t

#ifndef GDS_DEVICE_ASSERT
#define GDS_DEVICE_ASSERT(C) { /* C */ }
#endif

#ifdef  __cplusplus

namespace gdsync {

    static const clock_t large_timeout = 1ULL<<32;
    enum { 
        ERROR_TIMEOUT = 11, // same as EAGAIN
        ERROR_INVALID = 22, //         EINVAL
    };

    //typedef enum wait_cond { WAIT_GEQ, WAIT_EQ, WAIT_AND, WAIT_NOR } wait_cond_t;
    typedef gds_wait_cond_flag_t wait_cond_t;

    struct sem32 {
        typedef uint32_t T;
        T sem;
        T value;

        __host__ __device__ inline volatile T *access_once() {
            return (volatile T *)&sem;
        }
    };
    typedef struct sem32 sem32_t;

    // indirect 32-bit semaphore
    struct isem32 {
        typedef uint32_t T;
        typedef  int32_t Tsigned;
        T *ptr;
        T value;

        __host__ __device__ inline volatile T *access_once() {
            return (volatile T *)ptr;
        }
        __host__ __device__ isem32() : ptr(NULL), value(0) {}
    };
    typedef struct isem32 isem32_t;

    struct isem64 {
        typedef uint64_t T;
        T *ptr;
        T value;

        __host__ __device__ inline volatile T *access_once() {
            return (volatile T *)ptr;
        }
        __host__ __device__ isem64() : ptr(NULL), value(0) {}
    };
    typedef struct isem64 isem64_t;

#if defined(__CUDACC__)
    namespace device {

        // NOTE: fences must be added by caller
        template<typename S> __device__ inline void release(S &sem) {
            //printf("[%d:%d] release %p=%08x\n", blockIdx.x, threadIdx.x, sem.access_once(), sem.value);
            GDS_DEVICE_ASSERT(0 != sem.access_once());
            *sem.access_once() = sem.value;
        }

        template<typename S> __device__ inline int wait(S &sem, wait_cond_t cond) {
            int ret = 0;
            switch(cond) {
            case GDS_WAIT_COND_EQ:  ret = wait_eq(sem);  break;
            case GDS_WAIT_COND_GEQ: ret = wait_geq(sem); break;
            case GDS_WAIT_COND_AND: ret = wait_and(sem); break;
            case GDS_WAIT_COND_NOR: ret = wait_nor(sem); break;
            default: ret = ERROR_INVALID; break;
            }
            return ret;
        }

        template <typename S> __device__ inline int wait_eq(S &sem) {
            int ret = ERROR_TIMEOUT;
            volatile clock_t tmout = clock() + large_timeout;
            do {
                if (*sem.access_once() == sem.value) {
                    ret = 0;
                    break;
                }
                __threadfence_block();
            } while(clock() < tmout);
            return ret;
        }

        template <typename S> __device__ inline int wait_geq(S &sem) {
            int ret = ERROR_TIMEOUT;
            volatile clock_t tmout = clock() + large_timeout;
            do {
                //printf("ptr=%p\n", sem.access_once());
                typedef typename S::Tsigned Ts;
                if ((Ts)*sem.access_once() - (Ts)sem.value >= 0) {
                    ret = 0;
                    break;
                }
                __threadfence_block();
            } while(clock() < tmout);
            return ret;
        }

        template <typename S> __device__ static inline int wait_and(S &sem) {
            int ret = ERROR_TIMEOUT;
            volatile clock_t tmout = clock() + large_timeout;
            do {
                if (*sem.access_once() & sem.value) {
                    ret = 0;
                    break;
                }
                __threadfence_block();
            } while(clock() < tmout);
            return ret;
        }

        template <typename S> __device__ static inline int wait_nor(S &sem) {
            int ret = ERROR_TIMEOUT;
            volatile clock_t tmout = clock() + large_timeout;
            do {
                if (~(*sem.access_once() | sem.value)) {
                    ret = 0;
                    break;
                }
                __threadfence_block();
            } while(clock() < tmout);
            return ret;
        }

    } // namespace device
#endif

} // namespace gdsync

#endif // __cplusplus
