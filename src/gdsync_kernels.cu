//#include <cuda_runtime_api.h>

#include "gdsync/device.cuh"
#include "objs.hpp"
#include "utils.hpp"

using namespace gdsync;

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
