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

#include "gdsync/device.cuh"
#include "objs.hpp"
#include "utils.hpp"
#include "gdsync_kernels.hpp"

using namespace gdsync;

__host__
int gds_launch_1QPSend_2CQWait(gds_peer *peer, CUstream stream, gds_op_list_t &params)
{
        int ret = 0;
        param_1snd2wait p;
        void *krn_params[] = { &p };
        CUstreamBatchMemOpParams *param = NULL;
        gds_dbg("stream=%p #params=%zu cufunction=%p\n", stream, params.size(), peer->kernels.krn1snd2wait);

        if (params.size() != 7) {
                gds_dbg("unexpected %d params\n", params.size());
                ret = EINVAL;
                goto out;
        }

        // parameter marshaling
        param = &params.at(0);
        GDS_ASSERT(param->operation == CU_STREAM_MEM_OP_WRITE_VALUE_32);
        p.sem0.ptr = reinterpret_cast<uint32_t*>(param->writeValue.address);
        p.sem0.value = param->writeValue.value;
        gds_dbg("p.sem0 %p %x\n", p.sem0.ptr, p.sem0.value);

        param = &params.at(1);
        GDS_ASSERT(param->operation == CU_STREAM_MEM_OP_MEMORY_BARRIER);

        param = &params.at(2);
        switch(param->operation) {
        case CU_STREAM_MEM_OP_WRITE_MEMORY:
                // TODO: implement memset kernel
                // hack! need to assert mlx5 and size==64
                // converting write memory into write_64 of 1st qword
                p.sem1.ptr = reinterpret_cast<uint64_t*>(param->writeMemory.dst);
                p.sem1.value = *reinterpret_cast<uint64_t*>(param->writeMemory.src);
                break;
        case CU_STREAM_MEM_OP_WRITE_VALUE_64:
                p.sem1.ptr = reinterpret_cast<uint64_t*>(param->writeValue.address);
                p.sem1.value = param->writeValue.value64;
                break;
        default:
                gds_err("unexpected operation %d\n", param->operation);
                ret = EINVAL;
                goto out;
        }
        gds_dbg("p.sem1 %p %lx\n", p.sem1.ptr, p.sem1.value);

        param = &params.at(3);
        GDS_ASSERT(param->operation == CU_STREAM_MEM_OP_WAIT_VALUE_32);
        p.semw[0].ptr = reinterpret_cast<uint32_t*>(param->waitValue.address);
        p.semw[0].value = param->waitValue.value;
        p.condw[0] = gds_cuwait_flags_to_wait_cond(param->waitValue.flags);
        gds_dbg("p.semw[0]:%p value:0x%lx cond:0x%x\n", p.semw[0].ptr, p.semw[0].value, p.condw[0]);

        param = &params.at(4);
        GDS_ASSERT(param->operation == CU_STREAM_MEM_OP_WRITE_VALUE_32);
        p.sem23[0].ptr = reinterpret_cast<uint32_t*>(param->writeValue.address);
        p.sem23[0].value = param->writeValue.value;
        gds_dbg("p.sem23[0] %p %x\n", p.sem23[0].ptr, p.sem23[0].value);

        param = &params.at(5);
        GDS_ASSERT(param->operation == CU_STREAM_MEM_OP_WAIT_VALUE_32);
        p.semw[1].ptr = reinterpret_cast<uint32_t*>(param->waitValue.address);
        p.semw[1].value = param->waitValue.value;
        p.condw[0] = gds_cuwait_flags_to_wait_cond(param->waitValue.flags);
        gds_dbg("p.semw[1]:%p value:0x%lx cond:0x%x\n", p.semw[1].ptr, p.semw[1].value, p.condw[1]);

        param = &params.at(6);
        GDS_ASSERT(param->operation == CU_STREAM_MEM_OP_WRITE_VALUE_32);
        p.sem23[1].ptr = reinterpret_cast<uint32_t*>(param->writeValue.address);
        p.sem23[1].value = param->writeValue.value;
        gds_dbg("p.sem23[1] %p %x\n", p.sem23[1].ptr, p.sem23[1].value);

        CUCHECK(cuLaunchKernel(peer->kernels.krn1snd2wait,
                               1, 1, 1,    // 1x1x1 blocks
                               2*32, 1, 1, // 1x1x1 threads
                               0,          // shared mem
                               stream,     // stream
                               krn_params, // params
                               0 ));       // extra

        //CUCHECK(cuStreamSynchronize(stream));
out:
        return ret;
}

/*
 * Local variables:
 *  c-indent-level: 8
 *  c-basic-offset: 8
 *  tab-width: 8
 *  indent-tabs-mode: nil
 * End:
 */
