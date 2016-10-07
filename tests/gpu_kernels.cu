#include <sys/time.h>
#include <assert.h>

#include "gpu.h"

//---------------------------
// kernel stuff

__global__ void void_kernel()
{
}

int gpu_launch_void_kernel()
{
        const int nblocks = 1;
        const int nthreads = 1;
	void_kernel<<<nblocks, nthreads, 0, gpu_stream>>>();
        CUDACHECK(cudaGetLastError());
        return 0;
}

//----------

__global__ void dummy_kernel(int p0, float p1, float *p2)
{
        //const uint tid = threadIdx.x;
        //const uint bid = blockIdx.x;
        //const uint block_size = blockDim.x;
        //const uint grid_size = gridDim.x;
        //const uint gid = tid + bid*block_size;
        //const uint n_threads = block_size*grid_size;
        __syncthreads();
}

int gpu_launch_dummy_kernel(void)
{
        const int nblocks = over_sub_factor * gpu_num_sm;
        const int nthreads = 32;
        int p0 = 100;
        float p1 = 1.1f;
        float *p2 = NULL;
	dummy_kernel<<<nblocks, nthreads, 0, gpu_stream>>>(p0, p1, p2);
        CUDACHECK(cudaGetLastError());
        return 0;
}

//----------

__global__ void calc_kernel(int n, float c, float *in, float *out)
{
        const uint tid = threadIdx.x;
        const uint bid = blockIdx.x;
        const uint block_size = blockDim.x;
        const uint grid_size = gridDim.x;
        const uint gid = tid + bid*block_size;
        const uint n_threads = block_size*grid_size;
        for (int i=gid; i<n; i += n_threads)
                out[i] = in[i] * c;
}

int gpu_launch_calc_kernel(size_t size)
{
        const int nblocks = over_sub_factor * gpu_num_sm;
        const int nthreads = 32*2;
        int n = size / sizeof(float);
        static float *in = NULL;
        static float *out = NULL;
        if (!in) {
                in = (float*)gpu_malloc(4096, size);
                out = (float*)gpu_malloc(4096, size);
        }
	calc_kernel<<<nblocks, nthreads, 0, gpu_stream>>>(n, 1.0f, in, out);
        CUDACHECK(cudaGetLastError());
        return 0;
}

//----------

/*
 * Local variables:
 *  c-indent-level: 8
 *  c-basic-offset: 8
 *  tab-width: 8
 *  indent-tabs-mode: nil
 * End:
 */
