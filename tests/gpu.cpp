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

#include <unistd.h>
#include <assert.h>

#include <cuda_runtime_api.h>
#include <infiniband/verbs_exp.h>

#include "gdrapi.h"
#include "gdsync.h"
#include "gpu.h"

//-----------------------------------------------------------------------------
// tracing support

static int gpu_dbg_is_enabled = -1;
int gpu_dbg_enabled()
{
        if (-1 == gpu_dbg_is_enabled) {        
                const char *env = getenv("GPU_ENABLE_DEBUG");
                if (env) {
                        int en = atoi(env);
                        gpu_dbg_is_enabled = !!en;
                        printf("GPU_ENABLE_DEBUG=%s\n", env);
                } else
                        gpu_dbg_is_enabled = 0;
        }
        return gpu_dbg_is_enabled;
}

//--------------

static CUdevice gpu_device;
static CUcontext gpu_ctx;
static int gpu_blocking_sync_mode = 0;
int gpu_num_sm = 0;
int gpu_clock_rate = 0;
CUstream gpu_stream;
CUstream gpu_stream_server;
CUstream gpu_stream_client;
#define num_tracking_events 4
static int next_acquire = 0, next_release = 0, next_wait = 0;
static CUevent gpu_tracking_event[num_tracking_events];

int gpu_launch_calc_kernel_on_stream(size_t size, CUstream s);
int gpu_launch_void_kernel_on_stream(CUstream s);

int gpu_init(int gpu_id, int sched_mode)
{
	int ret = 0;

	printf("initializing CUDA\n");
	CUCHECK(cuInit(0));
	
	int deviceCount = 0;
	CUCHECK(cuDeviceGetCount(&deviceCount));

	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0) {
		printf("There are no available device(s) that support CUDA\n");
		ret = 1;
		goto out;
	} else
		printf("There are %d devices supporting CUDA, picking N.%d\n", deviceCount, gpu_id);

        if (getenv("USE_GPU")) {
                gpu_id = atoi(getenv("USE_GPU"));
                printf("overriding gpu_id with USE_GPU=%d\n", gpu_id);
        }

	if (gpu_id >= deviceCount) {
		printf("ERROR: requested GPU gpu_id beyond available\n");
		ret = 1;
		goto out;
	}
        gpu_blocking_sync_mode = (CU_CTX_SCHED_BLOCKING_SYNC == sched_mode) ? 1 : 0;

	int i;
	for (i=0; i<deviceCount; ++i) {
		CUCHECK(cuDeviceGet(&gpu_device, i));
		char name[128];
		CUCHECK(cuDeviceGetName(name, sizeof(name), gpu_device));
		int pciBusID, pciDeviceID;
		cuDeviceGetAttribute(&pciBusID, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, gpu_device);
		cuDeviceGetAttribute(&pciDeviceID, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, gpu_device);
		//printf("  Device PCI Bus ID / PCI location ID:           %d / %d\n", pciBusID, pciDeviceID);
		printf("GPU id:%d dev:%d name:%s pci %d:%d\n", i, gpu_device, name, pciBusID, pciDeviceID);
	}

	CUCHECK(cuDeviceGet(&gpu_device, gpu_id));

	printf("creating CUDA Primary Ctx on device:%d id:%d\n", gpu_device, gpu_id);
        CUCHECK(cuDevicePrimaryCtxRetain(&gpu_ctx, gpu_device));

	printf("making it the current CUDA Ctx\n");
	CUCHECK(cuCtxSetCurrent(gpu_ctx));

        // TODO: add a check for canMapHost

        //CUCHECK(cuDeviceGetProperties(&prop, gpu_device));
        cuDeviceGetAttribute(&gpu_num_sm, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, gpu_device);
        printf("num SMs per GPU:%d\n", gpu_num_sm);
        cuDeviceGetAttribute(&gpu_clock_rate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, gpu_device);
        printf("clock rate:%d\n", gpu_clock_rate);

	CUCHECK(cuStreamCreate(&gpu_stream, 0));
        printf("created main test CUDA stream %p\n", gpu_stream);
	CUCHECK(cuStreamCreate(&gpu_stream_server, 0));
        printf("created stream server CUDA stream %p\n", gpu_stream_server);
	CUCHECK(cuStreamCreate(&gpu_stream_client, 0));
        printf("created stream cliebt CUDA stream %p\n", gpu_stream_client);

        {
                int n;
                int ev_flags = CU_EVENT_DISABLE_TIMING;
                if (CU_CTX_SCHED_BLOCKING_SYNC == sched_mode) {
                        printf("creating events with blocking sync behavior\n");
                        ev_flags |= CU_EVENT_BLOCKING_SYNC;
                }
                for (n=0; n<num_tracking_events; ++n) {
                        CUCHECK(cuEventCreate(&gpu_tracking_event[n], ev_flags));
                        printf("created %d tracking event %p\n", n, gpu_tracking_event[n]);
                }
        }        

        //  pipe cleaner
        gpu_launch_void_kernel_on_stream(gpu_stream);
        cuStreamSynchronize(gpu_stream);

out:
        if (ret) {
                if (gpu_ctx)
                        CUCHECK(cuDevicePrimaryCtxRelease(gpu_device));
        }

	return ret;
}

int gpu_finalize()
{
	gpu_dbg("destroying current CUDA Ctx\n");
        CUCHECK(cuCtxSynchronize());
        int n;
        for (n=0; n<num_tracking_events; ++n)
                CUCHECK(cuEventDestroy(gpu_tracking_event[n]));
	CUCHECK(cuStreamDestroy(gpu_stream));
	CUCHECK(cuStreamDestroy(gpu_stream_server));
	CUCHECK(cuStreamDestroy(gpu_stream_client));
        CUCHECK(cuDevicePrimaryCtxRelease(gpu_device));

	return 0;
}

void *gpu_malloc(size_t page_size, size_t min_size)
{
	size_t n_pages = (min_size + page_size - 1)/page_size;
	size_t size = n_pages * page_size;

	printf("cuMemAlloc() of a %lu bytes GPU buffer\n", size);
	CUdeviceptr d_A;
	CUCHECK(cuMemAlloc(&d_A, size));
	CUCHECK(cuMemsetD8(d_A, 0, size));

	printf("allocated GPU buffer address at %016llx\n", d_A);
	return (void*)d_A;
}

int gpu_free(void *ptr)
{
	CUCHECK(cuMemFree((CUdeviceptr)ptr));
	return 0;
}

int gpu_memset(void *ptr, const unsigned char c, size_t size)
{
	printf("poisoning GPU buffer, filled with '0x%02x' !!!\n", c);
	CUCHECK(cuMemsetD8((CUdeviceptr)ptr, c, size));
	return 0;
}

int gpu_register_host_mem(void *ptr, size_t size)
{
        CUCHECK(cuMemHostRegister(ptr, size, CU_MEMHOSTREGISTER_DEVICEMAP | CU_MEMHOSTREGISTER_PORTABLE));

        return 0;
}

int gpu_launch_kernel(size_t size, int is_peersync)
{
        return gpu_launch_kernel_on_stream(size, is_peersync, gpu_stream);
}

int gpu_launch_kernel_on_stream(size_t size, int is_peersync, CUstream s)
{
        int ret = 0;
        if (0 == size)
                gpu_launch_void_kernel_on_stream(s);
        else
                gpu_launch_calc_kernel_on_stream(size, s);
        assert(cudaSuccess == cudaGetLastError());
        return ret;
}

void gpu_post_acquire_tracking_event()
{
        int n = (next_acquire++)%num_tracking_events;
        gpu_dbg("calling stream wait event on %d tracking event %p\n", n, gpu_tracking_event[n]);
        CUCHECK(cuStreamWaitEvent(gpu_stream, gpu_tracking_event[n], 0));
}

void gpu_post_release_tracking_event()
{
        int n = (next_release++)%num_tracking_events;
        gpu_dbg("recording %d tracking event %p\n", n, gpu_tracking_event[n]);
        CUCHECK(cuEventRecord(gpu_tracking_event[n], gpu_stream));
}

//#define MYCLOCK CLOCK_REALTIME
//#define MYCLOCK CLOCK_RAW_MONOTONIC
#define MYCLOCK CLOCK_MONOTONIC

int gpu_wait_tracking_event(int tmout_us)
{
        int ret = 0;
        int n = (next_wait)%num_tracking_events;

        if (next_wait >= next_release) {
                return ENOMEM;
        }
        if (tmout_us < 0)
                return EINVAL;

        if (gpu_blocking_sync_mode) {
                CUCHECK(cuEventSynchronize(gpu_tracking_event[n]));
                ++next_wait;
        } else {
                struct timespec ts;
                clock_gettime(MYCLOCK, &ts);
                uint64_t now = ts.tv_nsec/1000 + ts.tv_sec*1000000;
                uint64_t tmout = now + tmout_us;
                while (1) {
                        CUresult retcode = cuEventQuery(gpu_tracking_event[n]);
                        if (retcode == CUDA_SUCCESS) {
                                //gpu_dbg("event signaled\n");
                                ++next_wait;
                                break;
                        } else if (retcode == CUDA_ERROR_NOT_READY) {
                                // event has not been signaled yet
                                //sleep(1);
                        } else {
                                printf("cuEventQuery error (%d)\n", retcode);
                                ret = EFAULT;
                                break;
                        }
                        // time-out check
                        uint64_t now = ts.tv_nsec/1000 + ts.tv_sec*1000000;
                        if (((int64_t)tmout-(int64_t)now) < (long)0) {
                                gpu_info("timeout reached!! enabling debug tracing...\n");
                                gpu_dbg_is_enabled = 1;
                                ret = EAGAIN;
                                break;
                        }
                }
        }

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
