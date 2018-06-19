/* Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#include <gdsync.h>
#include "flusher.hpp"
#include "utils.hpp"
#include "objs.hpp"
#include "task_queue.hpp"

//-----------------------------------------------------------------------------

template <class T>
struct gds_semaphore
{
        T *ptr;
        T value;
        gds_wait_cond_flag cond;
};

typedef gds_semaphore<uint32_t> sem32_t;

//-----------------------------------------------------------------------------

struct cpu_flusher {
        sem32_t m_wait;
        sem32_t m_flag;
        uint32_t *m_mapped_gpu_ptr;

        cpu_flusher(sem32_t wait, sem32_t flag, uint32_t *mapped_gpu_ptr) {
                assert(is_valid(wait.cond));
                assert(wait.ptr);
                assert(is_valid(flag.cond));
                assert(flag.ptr);
                assert(mapped_gpu_ptr);
                m_wait = wait;
                m_flag = flag;
                m_mapped_gpu_ptr = mapped_gpu_ptr;
        }
        ~cpu_flusher() {}

        bool run() {
                bool done = false;
                if (test()) {
                        flush();
                        signal();
                        delete this;
                        done = true;
                }
                return done;
        }

protected:
        void flush() {
                gds_atomic_get(m_mapped_gpu_ptr);
        }
        void signal() {
                gds_atomic_set(m_flag.ptr, m_flag.value);
        }
        bool test() {
                bool done = false;
                uint32_t data = gds_atomic_get(m_wait.ptr);
                uint32_t value = m_wait.value;
                switch(m_wait.cond) {
                case GDS_WAIT_COND_GEQ:
                        done = ((int32_t)data - (int32_t)value >= 0);
                        break;
                case GDS_WAIT_COND_EQ:
                        done = (data == value);
                        break;
                case GDS_WAIT_COND_AND:
                        done = (data & value);
                        break;
                case GDS_WAIT_COND_NOR:
                        done = ~(data | value);
                        break;
                default:
                        assert(!"cannot happen");
                }
                return done;
        }
};

//-----------------------------------------------------------------------------

static sem32_t get_gpu_flag()
{
        return sem32_t{NULL, 0, GDS_WAIT_COND_AND};
}

//-----------------------------------------------------------------------------

int flusher::post_ops(gds_peer *peer, gds_op_list_t &ops, CUdeviceptr ptr, uint32_t value, gds_wait_cond_flag cond)
{
        assert(is_valid(cond));
        assert(ptr);
        // original dword on which the CPU polls
        sem32_t wait = {reinterpret_cast<uint32_t*>(ptr), value, cond};
        int retcode = 0;
        cpu_flusher *cf;
        // flag on which the GPU polls
        sem32_t flag = get_gpu_flag();
        CUstreamBatchMemOpParams param;
        // inject GPU polling
        retcode = gds_fill_poll_raw(peer, param, reinterpret_cast<CUdeviceptr>(flag.ptr), flag.value, flag.cond);
        if (retcode)
                goto out;
        ops.push_back(param);
        // reuse the flag as the target dword for flushing
        cf = new (std::nothrow) cpu_flusher(wait, flag, flag.ptr);
        if (!cf) {
                gds_err("error while allocating flusher\n");
                retcode = ENOMEM;
                goto out;
        }
        // enqueue the CPU task which will poll over the original dword
        peer->tq->queue(std::bind(&cpu_flusher::run, cf));
out:
        return retcode;
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
