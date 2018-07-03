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

#include "gdsync/device.cuh"
#include "objs.hpp"
#include "utils.hpp"

using namespace gdsync;

#define gds_htonl(x) ( ((x & 0xFF) << 24) | ((x & 0xFF00) << 8) | ((x & 0xFF0000) >> 8) | ((x & 0xFF000000) >> 24) )
#define gds_ntohl(x) ( ((x & 0xFF) >> 24) | ((x & 0xFF00) >> 8) | ((x & 0xFF0000) << 8) | ((x & 0xFF000000) << 24) )

#define gds_ntohll(x) ( ((uint64_t)(gds_ntohl((int)((x << 32) >> 32))) << 32) |  (uint32_t)gds_ntohl(((int)(x >> 32))))
#define gds_htonll(x) ( ((uint64_t) gds_htonl((x) & 0xFFFFFFFFUL)) << 32) | gds_htonl((uint32_t)((x) >> 32))

/* Updates send parameters */
extern "C" __global__ void krnsetsndpar(
		CUdeviceptr ptr_to_size_wqe, CUdeviceptr ptr_to_size_new,
		CUdeviceptr ptr_to_lkey_wqe, CUdeviceptr ptr_to_lkey_new,
		CUdeviceptr ptr_to_addr_wqe, CUdeviceptr ptr_to_addr_new)
{
	//According to https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#hardware-implementation
	//The NVIDIA GPU architecture uses a little-endian representation.

	//Thread 0 set new size
	if(threadIdx.x == 0 && ptr_to_size_new != 0)
	{
		((uint32_t*)ptr_to_size_wqe)[0] = gds_htonl(((uint32_t*)ptr_to_size_new)[0]);
		//printf("Inside kernel th0, ptr_to_size_wqe=0x%08x, ptr_to_size_new=0x%08x\n", ((uint32_t*)ptr_to_size_wqe)[0], ((uint32_t*)ptr_to_size_new)[0]);
	}

	//Thread 1 set new addr
	if(threadIdx.x == 1 && ptr_to_addr_new != 0)
	{
		((uint64_t*)ptr_to_addr_wqe)[0] = gds_htonll(((uint64_t*)ptr_to_addr_new)[0]);
		//printf("Inside kernel th1, ptr_to_addr_wqe=0x%llx, ptr_to_addr_new=%llx\n", ((uint64_t*)ptr_to_addr_wqe)[0], ((uint64_t*)ptr_to_addr_new)[0]);
	}

	//Thread 2 set new lkey
	if(threadIdx.x == 2 && ptr_to_lkey_new != 0)
	{
		((uint32_t*)ptr_to_lkey_wqe)[0] = gds_htonl(((uint32_t*)ptr_to_lkey_new)[0]);
		//printf("Inside kernel th0, ptr_to_size_wqe=0x%08x, ptr_to_size_new=0x%08x\n", ((uint32_t*)ptr_to_size_wqe)[0], ((uint32_t*)ptr_to_size_new)[0]);
	}
	
	//We're updating host mem, so we need a global fence
	__threadfence_system();
}
