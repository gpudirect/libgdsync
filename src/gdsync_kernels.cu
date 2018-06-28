#include <cuda_runtime_api.h>

#include "gdsync/device.cuh"
#include "objs.hpp"
#include "utils.hpp"

using namespace gdsync;

#if 0
__global__ static void gds_1QPSend_2CQWait(isem32_t sem0, isem64_t sem1, isem32_t semw[2], wait_cond_t condw[2], isem32_t sem23[2])
{
    if (threadIdx.x==0) {
        device::release(sem0);
        __threadfence_system();
        device::release(sem1);
    }
    __syncthreads();
    if (threadIdx.x<2) {
        device::wait(semw[threadIdx.x], condw[threadIdx.x]);
        device::release(sem23[threadIdx.x]);
    }
}

int gds_launch_1QPSend_2CQWait(gds_peer *peer, CUstream stream, gds_op_list_t &params)
{
        isem32_t sem0;
        isem64_t sem1;
        isem32_t semw[2];
        wait_cond_t condw[2];
        isem32_t sem23[2];
        CUstreamBatchMemOpParams *param = NULL;

        GDS_ASSERT(params.size() == 7);

        param = &params.at(0);
        GDS_ASSERT(param->operation == CU_STREAM_MEM_OP_WRITE_VALUE_32);
        sem0.ptr = reinterpret_cast<uint32_t*>(param->writeValue.address);
        sem0.value = param->writeValue.value;

        param = &params.at(1);
        GDS_ASSERT(param->operation == CU_STREAM_MEM_OP_MEMORY_BARRIER);

        // hack! need to assert mlx5 and size==64
        // converting write memory into write_64 of 1st qword
        // todo: implement memset_t
        param = &params.at(2);
        GDS_ASSERT(param->operation == CU_STREAM_MEM_OP_WRITE_MEMORY);
        sem1.ptr = reinterpret_cast<uint64_t*>(param->writeMemory.dst);
        sem1.value = *reinterpret_cast<uint64_t*>(param->writeMemory.src);

        param = &params.at(3);
        GDS_ASSERT(param->operation == CU_STREAM_MEM_OP_WAIT_VALUE_32);
        semw[0].ptr = reinterpret_cast<uint32_t*>(param->waitValue.address);
        semw[0].value = param->waitValue.value;

        param = &params.at(4);
        GDS_ASSERT(param->operation == CU_STREAM_MEM_OP_WRITE_VALUE_32);
        sem23[0].ptr = reinterpret_cast<uint32_t*>(param->writeValue.address);
        sem23[0].value = param->writeValue.value;

        param = &params.at(5);
        GDS_ASSERT(param->operation == CU_STREAM_MEM_OP_WAIT_VALUE_32);
        semw[1].ptr = reinterpret_cast<uint32_t*>(param->waitValue.address);
        semw[1].value = param->waitValue.value;

        param = &params.at(6);
        GDS_ASSERT(param->operation == CU_STREAM_MEM_OP_WRITE_VALUE_32);
        sem23[1].ptr = reinterpret_cast<uint32_t*>(param->writeValue.address);
        sem23[1].value = param->writeValue.value;

        void *krn_params[] = {
                reinterpret_cast<void *>(&sem0),
                reinterpret_cast<void *>(&sem1),
                reinterpret_cast<void *>(&semw[0]),
                reinterpret_cast<void *>(&condw[0]),
                reinterpret_cast<void *>(&sem23[0])
        };
#if 0
        // this will introduce a dependency on CUDA RT lib
        CUDACHECK(cudaLaunchKernel(gds_1QPSend_2CQWait,
                                 dim3(1,1,1), // gridDim
                                 dim3(32,1,1) // blockDim
                                 krn_params,  // args
                                 0,           // sharedMem
                                 stream));
#endif
        return 0;
}

#endif

#if 0
extern "C" __global__ void gds_1QPSendInline64_2CQWait(isem32_t sem0, memset_t sem1, isem32_t semw[2], wait_cond_t condw[2], isem32_t sem23[2])
{
    if (threadIdx.x==0) {
        device::release(sem0);
        __threadfence_system();
        device::release(sem1);
    }
    __syncthreads();
    if (threadIdx.x<2) {
        wait(semw[threadIdx.x], condw[threadIdx.x]);
        device::release(sem23[threadIdx.x]);
    }
}
#endif

#define gds_htonl(x) ( ((x & 0xFF) << 24) | ((x & 0xFF00) << 8) | ((x & 0xFF0000) >> 8) | ((x & 0xFF000000) >> 24) )
#define gds_ntohl(x) ( ((x & 0xFF) >> 24) | ((x & 0xFF00) >> 8) | ((x & 0xFF0000) << 8) | ((x & 0xFF000000) << 24) )

#define gds_ntohll(x) ( ((uint64_t)(gds_ntohl((int)((x << 32) >> 32))) << 32) |  (uint32_t)gds_ntohl(((int)(x >> 32))))
#define gds_htonll(x) ( ((uint64_t) gds_htonl((x) & 0xFFFFFFFFUL)) << 32) | gds_htonl((uint32_t)((x) >> 32))

__global__ static void gds_update_send_params(
		CUdeviceptr ptr_to_size_wqe, CUdeviceptr ptr_to_size_new,
		CUdeviceptr ptr_to_addr_wqe, CUdeviceptr ptr_to_addr_new)
{
	//According to https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#hardware-implementation
	//The NVIDIA GPU architecture uses a little-endian representation.

	//Thread 0 set new size
	if(threadIdx.x == 0 && ptr_to_size_new != 0)
		((uint32_t*)ptr_to_size_wqe)[0] = gds_htonl(((uint32_t*)ptr_to_size_new)[0]);

	//Thread 1 set new addr
	if(threadIdx.x == 1 && ptr_to_addr_new != 0)
		((uint64_t*)ptr_to_addr_wqe)[0] = gds_htonll(((uint64_t*)ptr_to_addr_new)[0]);
	
}

int gds_launch_update_send_params(
		CUdeviceptr ptr_to_size_wqe, CUdeviceptr ptr_to_size_new,
		CUdeviceptr ptr_to_addr_wqe, CUdeviceptr ptr_to_addr_new,
		CUstream stream)
{
	gds_dbg("Launching gds_update_send_params with ptr_to_size_wqe=%x, ptr_to_size_new=%x, ptr_to_addr_wqe=%x, ptr_to_addr_new=%x\n", 
			ptr_to_size_wqe, ptr_to_size_new, ptr_to_addr_wqe, ptr_to_addr_new);

        gds_update_send_params<<<1,2,0,stream>>>(ptr_to_size_wqe, ptr_to_size_new, ptr_to_addr_wqe, ptr_to_addr_new);

        return 0;
}

#if 0
int gds_launch_update_send_params(
		CUdeviceptr ptr_to_size_wqe, CUdeviceptr ptr_to_size_new,
		CUdeviceptr ptr_to_addr_wqe, CUdeviceptr ptr_to_addr_new,
		CUstream stream)
{
	CUdeviceptr argBuffer[4];
	size_t argBufferSize=4;

	argBuffer[0]=ptr_to_size_wqe;
	argBuffer[1]=ptr_to_size_new;
	argBuffer[2]=ptr_to_addr_wqe;
	argBuffer[3]=ptr_to_addr_new;
	
	void *config[] = {
              CU_LAUNCH_PARAM_BUFFER_POINTER, argBuffer,
              CU_LAUNCH_PARAM_BUFFER_SIZE,    &argBufferSize,
              CU_LAUNCH_PARAM_END
          };

        CUCHECK(cuLaunchKernel(gds_update_send_info, 1, 0, 0, 2, 0, 0, 0, stream, NULL, config));

        return 0;
}
#endif