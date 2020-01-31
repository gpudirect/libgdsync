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

#if HAVE_CONFIG_H
#  include <config.h>
#endif /* HAVE_CONFIG_H */

#include <unistd.h>
#include <string.h>
#include <assert.h>
#include <inttypes.h>

#include <map>
#include <algorithm>
#include <string>
using namespace std;

#include <cuda.h>
#include <gdrapi.h>

#include "gdsync.h"
#include "gdsync/tools.h"
#include "objs.hpp"
#include "utils.hpp"
#include "mem.hpp"
#include "memmgr.hpp"

#ifndef GDS_GPU_PAGE_SIZE
#define GDR_GPU_PAGE_SHIFT   GPU_PAGE_SHIFT 
#define GDR_GPU_PAGE_SIZE    GPU_PAGE_SIZE  
#define GDR_GPU_PAGE_OFFSET  GPU_PAGE_OFFSET
#define GDR_GPU_PAGE_MASK    GPU_PAGE_MASK  
#endif

//-----------------------------------------------------------------------------

// BUG: not multi-thread safe
static gdr_t gdr = 0;

//-----------------------------------------------------------------------------

static int gds_map_gdr_memory(gds_mem_desc_t *desc, CUdeviceptr d_buf, size_t size, int flags)
{
        gdr_mh_t mh;
        gdr_info_t info;
        void *h_buf = NULL;
        void *bar_ptr  = NULL;
        size_t buf_size = size;
        int ret = 0;
        int retcode = 0;
        ptrdiff_t off = 0;

        assert(desc);
        assert(d_buf);
        assert(size);

        if (!gdr) {
                gdr = gdr_open();
                if (!gdr) {
                        gds_err("can't initialize GDRCopy library\n");
                        exit(EXIT_FAILURE);
                }
        }

        unsigned int flag = 1;
        CUCHECK(cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, d_buf));

        // pin it via GDRCopy
        ret = gdr_pin_buffer(gdr, d_buf, buf_size, 0, 0, &mh);
        if (ret) {
                gds_err("cannot pin buffer addr=%p retcode=%d(%s)\n", (void*)d_buf, ret, strerror(ret));
                retcode = ret;
                goto out;
        }

        ret = gdr_map(gdr, mh, &bar_ptr, buf_size);
        if (ret) {
                retcode = ret;
                goto out;
        }

        ret = gdr_get_info(gdr, mh, &info);
        if (ret) {
                retcode = ret;
                goto out;
        }
        // remember that mappings start on a 64KB boundary, so let's
        // calculate the offset from the head of the mapping to the
        // beginning of the buffer
        off = d_buf - info.va;
        h_buf = (void *)((char *)bar_ptr + off);
        if (off < 0) {
                gds_warn("unexpected offset %td\n", off);
                exit(EXIT_FAILURE);
        }
        desc->d_ptr = d_buf;
        desc->h_ptr = h_buf;
        desc->bar_ptr = bar_ptr;
        desc->flags = flags;
        desc->alloc_size = buf_size;
        desc->mh = mh;
        gds_dbg("d_ptr=%lx h_ptr=%p bar_ptr=%p flags=0x%08x alloc_size=%zd mh=%x\n",
                        (unsigned long)desc->d_ptr, desc->h_ptr, desc->bar_ptr, desc->flags, desc->alloc_size, desc->mh);
out:
        if (ret) {
                if (mh) {
                        if (bar_ptr)
                                gdr_unmap(gdr, mh, bar_ptr, buf_size);
                        gdr_unpin_buffer(gdr, mh);
                }
        }
        return ret;
}

//-----------------------------------------------------------------------------

static int gds_unmap_gdr_memory(gds_mem_desc_t *desc)
{
        int ret = 0;
        assert(desc);
        if (!gdr) {
                gds_err("GDRCopy library is not initialized\n");
                exit(EXIT_FAILURE);
        }
        if (!desc->d_ptr || !desc->h_ptr || !desc->alloc_size || !desc->mh || !desc->bar_ptr) {
                gds_err("invalid desc\n");
                return EINVAL;
        }
        gds_dbg("d_ptr=%lx h_ptr=%p alloc_size=%zd mh=%x\n",
                        (unsigned long)desc->d_ptr, desc->h_ptr, desc->alloc_size, desc->mh);
        gdr_unmap(gdr, desc->mh, desc->bar_ptr, desc->alloc_size);
        gdr_unpin_buffer(gdr, desc->mh);
        return ret;
}

//-----------------------------------------------------------------------------

static int gds_alloc_gdr_memory(gds_mem_desc_t *desc, size_t size, int flags)
{
        CUdeviceptr d_buf = 0;
        CUdeviceptr d_buf_aligned = 0;
        size_t buf_size = size + GDS_GPU_PAGE_SIZE - 1;
        int ret = 0;

        assert(desc);

        CUCHECK(cuMemAlloc(&d_buf, buf_size));

        d_buf_aligned = (d_buf + GDS_GPU_PAGE_SIZE - 1) & GDS_GPU_PAGE_MASK;

        gds_dbg("allocated GPU polling buffer d_buf=0x%llx req_size=%zu d_buf_aligned=0x%llx buf_size=%zu\n", d_buf, size, d_buf_aligned, buf_size);

        ret = gds_map_gdr_memory(desc, d_buf_aligned, size, flags);
        if (ret) {
                gds_err("error %d while mapping gdr memory\n", ret);
                CUCHECK(cuMemFree(d_buf));
                return ret;
        }

        desc->original_d_ptr = d_buf;
        return ret;
}

//-----------------------------------------------------------------------------

static int gds_free_gdr_memory(gds_mem_desc_t *desc)
{
        int ret = 0;
        assert(desc);
        if (!desc->d_ptr || !desc->h_ptr || !desc->alloc_size || !desc->mh || !desc->bar_ptr) {
                gds_err("invalid desc\n");
                return EINVAL;
        }
        gds_dbg("d_ptr=%lx h_ptr=%p alloc_size=%zd mh=%x\n",
                        (unsigned long)desc->d_ptr, desc->h_ptr, desc->alloc_size, desc->mh);

        ret = gds_unmap_gdr_memory(desc);
        if (ret) {
                gds_err("error %d while unmapping gdr, going on anyway\n", ret);
        }

        CUCHECK(cuMemFree(desc->original_d_ptr));
        return ret;
}

//-----------------------------------------------------------------------------

// WAR for Pascal
#ifdef USE_STATIC_MEM
const size_t s_size = PAGE_SIZE*128;
static char s_buf[s_size];
static unsigned s_buf_i = 0;
#endif

static int gds_alloc_pinned_memory(gds_mem_desc_t *desc, size_t size, int flags)
{
        int ret;
#ifdef USE_STATIC_MEM
        unsigned long s_addr = (unsigned long)(s_buf + s_buf_i);
        unsigned long s_off  = s_addr & (PAGE_SIZE-1);
        if (s_off)
                s_addr += PAGE_SIZE - s_off;
        *memptr = (void *)s_addr;
        gpu_info("s_buf_i=%d off=%lu memptr=%p\n", s_buf_i, s_off, *memptr);

        s_buf_i += s_off + size;

        if (s_buf_i >= s_size) {
                gds_err("can't alloc static memory\n");
                return ENOMEM;
        }
#else
        assert(desc);
        desc->h_ptr = NULL;
        ret = posix_memalign(&desc->h_ptr, GDS_HOST_PAGE_SIZE, size);
        if (ret) {
                goto out;
        }
        desc->d_ptr = 0;
        ret = gds_register_mem(desc->h_ptr, size, GDS_MEMORY_HOST, &desc->d_ptr);
        if (ret) {
                goto out;
        }
        desc->bar_ptr = NULL;
        desc->flags = flags;
        desc->alloc_size = size;
        desc->mh = 0;        
        gds_dbg("d_ptr=%lx h_ptr=%p flags=0x%08x alloc_size=%zd\n",
                        (unsigned long)desc->d_ptr, desc->h_ptr, desc->flags, desc->alloc_size);
out:
        if (ret) {
                if (desc->h_ptr) {
                        if (desc->d_ptr)
                                gds_unregister_mem(desc->h_ptr, desc->alloc_size);
                        free(desc->h_ptr);
                }
        }
#endif
        return ret;
}

//-----------------------------------------------------------------------------

static int gds_free_pinned_memory(gds_mem_desc_t *desc)
{
        int ret;
        assert(desc);
        if (!desc->d_ptr || !desc->h_ptr) {
                gds_err("invalid desc\n");
                return EINVAL;
        }
#ifdef USE_STATIC_MEM
        // BUG: TBD
#else
        gds_dbg("d_ptr=%lx h_ptr=%p flags=0x%08x alloc_size=%zd\n",
                        (unsigned long)desc->d_ptr, desc->h_ptr, desc->flags, desc->alloc_size);
        ret = gds_unregister_mem(desc->h_ptr, desc->alloc_size);
        free(desc->h_ptr);
        desc->h_ptr = NULL;
        desc->d_ptr = 0;
        desc->alloc_size = 0;
#endif
        return ret;
}

//-----------------------------------------------------------------------------

int gds_alloc_mapped_memory(gds_mem_desc_t *desc, size_t size, int flags)
{
        int ret = 0;
        if (!size) {
                gds_warn("silently ignoring zero size alloc!\n");
                return 0;
        }
        if (!desc) {
                gds_err("NULL desc!\n");
                return EINVAL;
        }
        switch(flags & GDS_MEMORY_MASK) {
                case GDS_MEMORY_GPU:
                        ret = gds_alloc_gdr_memory(desc, size, flags);
                        break;
                case GDS_MEMORY_HOST:
                        ret = gds_alloc_pinned_memory(desc, size, flags);
                        break;
                default:
                        gds_err("invalid flags\n");
                        ret = EINVAL;
                        break;
        }
        return ret;
}

//-----------------------------------------------------------------------------

int gds_free_mapped_memory(gds_mem_desc_t *desc)
{
        int ret = 0;
        if (!desc) {
                gds_err("NULL desc!\n");
                return EINVAL;
        }
        switch(desc->flags & GDS_MEMORY_MASK) {
                case GDS_MEMORY_GPU:
                        ret = gds_free_gdr_memory(desc);
                        break;
                case GDS_MEMORY_HOST:
                        ret = gds_free_pinned_memory(desc);
                        break;
                default:
                        ret = EINVAL;
                        break;
        }
        return ret;
}

//-----------------------------------------------------------------------------

#define ROUND_TO(V,PS) ((((V) + (PS) - 1)/(PS)) * (PS))

// allocate GPU memory with a GDR mapping (CPU can dereference it)
int gds_peer_malloc_ex(int peer_id, uint64_t peer_data, void **host_addr, CUdeviceptr *peer_addr, size_t req_size, void **phandle, gds_memory_type_t mem_type, bool has_cpu_mapping)
{
        int ret = 0;
        // assume GPUs are the only peers!!!
        int gpu_id = peer_id;
        CUcontext gpu_ctx;
        CUdevice gpu_device;
        size_t size = ROUND_TO(req_size, GDS_GPU_PAGE_SIZE);

        gds_dbg("GPU%u: malloc req_size=%zu size=%zu\n", gpu_id, req_size, size);

        if (!phandle || !host_addr || !peer_addr) {
                gds_err("invalid params\n");
                return EINVAL;
        }

        // NOTE: gpu_id's primary context is assumed to be the right one
        // breaks horribly with multiple contexts
        int num_gpus;
        do {
                CUresult err = cuDeviceGetCount(&num_gpus);
                if (CUDA_SUCCESS == err) {
                        break;
                } else if (CUDA_ERROR_NOT_INITIALIZED == err) {
                        gds_err("CUDA error %d in cuDeviceGetCount, calling cuInit\n", err);
                        CUCHECK(cuInit(0));
                        // try again
                        continue;
                } else {
                        gds_err("CUDA error %d in cuDeviceGetCount, returning EIO\n", err);
                        return EIO;
                }
        } while(0);
        gds_dbg("num_gpus=%d\n", num_gpus);
        if (gpu_id >= num_gpus) {
                gds_err("invalid num_GPUs=%d while requesting GPU id %d\n", num_gpus, gpu_id);
                return EINVAL;
        }

        CUCHECK(cuDeviceGet(&gpu_device, gpu_id));
        gds_dbg("gpu_id=%d gpu_device=%d\n", gpu_id, gpu_device);
        // TODO: check for existing context before switching to the interop one
        CUCHECK(cuDevicePrimaryCtxRetain(&gpu_ctx, gpu_device));
        CUCHECK(cuCtxPushCurrent(gpu_ctx));
        assert(gpu_ctx != NULL);

        gds_mem_desc_t *desc = (gds_mem_desc_t *)calloc(1, sizeof(gds_mem_desc_t));
        if (!desc) {
                gds_err("error while allocating mem desc\n");
                ret = ENOMEM;
                goto out;
        }

        ret = gds_alloc_mapped_memory(desc, size, mem_type);
        if (ret) {
                gds_err("error %d while allocating mapped GPU buffers\n", ret);
                goto out;
        }

        *host_addr = desc->h_ptr;
        *peer_addr = desc->d_ptr;
        *phandle = desc;

out:
        if (ret)
                free(desc); // desc can be NULL

        CUCHECK(cuCtxPopCurrent(NULL));
        CUCHECK(cuDevicePrimaryCtxRelease(gpu_device));

        return ret;
}

//-----------------------------------------------------------------------------

int gds_peer_malloc(int peer_id, uint64_t peer_data, void **host_addr, CUdeviceptr *peer_addr, size_t req_size, void **phandle, gds_memory_type_t mem_type)
{
        return gds_peer_malloc_ex(peer_id, peer_data, host_addr, peer_addr, req_size, phandle, mem_type, true);
}

//-----------------------------------------------------------------------------

int gds_peer_mfree(int peer_id, uint64_t peer_data, void *host_addr, void *handle)
{
        int ret = 0;
        // assume GPUs are the only peers!!!
        int gpu_id = peer_id;
        CUcontext gpu_ctx;
        CUdevice gpu_device;

        gds_dbg("GPU%u: mfree\n", gpu_id);

        if (!handle) {
                gds_err("invalid handle\n");
                return EINVAL;
        }

        if (!host_addr) {
                gds_err("invalid host_addr\n");
                return EINVAL;
        }

        // NOTE: gpu_id's primary context is assumed to be the right one
        // breaks horribly with multiple contexts

        CUCHECK(cuDeviceGet(&gpu_device, gpu_id));
        CUCHECK(cuDevicePrimaryCtxRetain(&gpu_ctx, gpu_device));
        CUCHECK(cuCtxPushCurrent(gpu_ctx));
        assert(gpu_ctx != NULL);

        gds_mem_desc_t *desc = (gds_mem_desc_t *)handle;
        ret = gds_free_mapped_memory(desc);
        if (ret) {
                gds_err("error %d while freeing mapped GPU buffers\n", ret);
        }
        free(desc);

        CUCHECK(cuCtxPopCurrent(NULL));
        CUCHECK(cuDevicePrimaryCtxRelease(gpu_device));

        return ret;
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
