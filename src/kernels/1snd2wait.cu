#include "gdsync/device.cuh"
#include "objs.hpp"
#include "utils.hpp"

using namespace gdsync;

extern "C" __global__ void krn1snd2wait(isem32_t sem0, isem64_t sem1, isem32_t semw[2], wait_cond_t condw[2], isem32_t sem23[2])
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
