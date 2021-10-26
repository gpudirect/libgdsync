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

//#include <map>
#include <algorithm>
#include <string>
using namespace std;

#include <cuda.h>
#include <gdrapi.h>

#include "gdsync.h"
#include "gdsync/tools.h"
#include "objs.hpp"
#include "utils.hpp"
#include "memmgr.hpp"
#include "mem.hpp"

//-----------------------------------------------------------------------------

gds_buf *gds_peer::alloc(size_t sz, uint32_t alignment, gds_memory_type_t mem_type)
{
        // TODO: support alignment
        // TODO: handle exception here
        gds_buf *buf = new gds_buf(this, sz, mem_type);
        if (!buf)
                return buf;
        int ret = gds_peer_malloc(gpu_id, 0, &buf->addr, &buf->peer_addr, buf->length, mem_type, &buf->handle);
        if (ret) {
                delete buf;
                buf = NULL;
                gds_err("error allocating GPU mapped memory\n");
        }
        return buf;
}

gds_buf *gds_peer::buf_alloc_cq(size_t length, uint32_t dir, uint32_t alignment, int flags)
{
        gds_buf *buf = NULL;
        switch (dir) {
        case (GDS_PEER_DIRECTION_FROM_HCA|GDS_PEER_DIRECTION_TO_PEER|GDS_PEER_DIRECTION_TO_CPU):
		// CQ dbrec
		if (GDS_ALLOC_CQ_DBREC_ON_GPU == (flags & GDS_ALLOC_CQ_DBREC_MASK)) {
			gds_dbg("allocating CQ DBREC on GPU mem\n");
			buf = alloc(length, alignment, GDS_MEMORY_GPU);
		}

		// CQ buf
		if (GDS_ALLOC_CQ_ON_GPU == (flags & GDS_ALLOC_CQ_MASK)) {
			gds_dbg("allocating CQ buf on GPU mem\n");
			buf = alloc(length, alignment, GDS_MEMORY_GPU);
		}
		break;
        case (GDS_PEER_DIRECTION_FROM_PEER|GDS_PEER_DIRECTION_TO_CPU):
                // CQ peer buf, helper buffer
                // on SYSMEM for the near future
                // GPU does a store to the 'busy' field as part of the peek_cq task
                // CPU polls on that field
                gds_dbg("allocating CQ peer buf on Host mem\n");
		buf = alloc(length, alignment, GDS_MEMORY_HOST);
                break;
        case (GDS_PEER_DIRECTION_FROM_PEER|GDS_PEER_DIRECTION_TO_HCA):
                gds_dbg("allocating CQ dbrec on Host mem\n");
		buf = alloc(length, alignment, GDS_MEMORY_HOST);
                break;
        default:
                gds_err("unexpected dir 0x%x\n", dir);
                break;
        }
        return buf;
}

gds_buf *gds_peer::buf_alloc_wq(size_t length, uint32_t dir, uint32_t alignment, int flags)
{
        gds_buf *buf = NULL;
        switch (dir) {
        case GDS_PEER_DIRECTION_FROM_PEER|GDS_PEER_DIRECTION_TO_HCA:
		// dbrec
		if (GDS_ALLOC_WQ_DBREC_ON_GPU == (flags & GDS_ALLOC_WQ_DBREC_MASK)) {
			gds_dbg("allocating WQ DBREC on GPU mem\n");
			buf = alloc(length, alignment, GDS_MEMORY_GPU);
		} else {
			gds_dbg("allocating WQ DBREC on Host mem\n");
		}

		// WQ
		if (GDS_ALLOC_WQ_ON_GPU == (flags & GDS_ALLOC_WQ_MASK)) {
			gds_dbg("allocating WQ buf on GPU mem\n");
			buf = alloc(length, alignment, GDS_MEMORY_GPU);
		} else {
			gds_dbg("allocating WQ buf on Host mem\n");
		}
		break;
        default:
                gds_err("unexpected dir=%08x\n", dir);
                break;
        }
        return buf;
}

gds_buf *gds_peer::buf_alloc(obj_type type, size_t length, uint32_t dir, uint32_t alignment, int flags)
{
        gds_buf *buf = NULL;
        gds_dbg("type=%d dir=%08x flags=%08x\n", type, dir, flags);
        switch (type) {
        case CQ:
                buf = buf_alloc_cq(length, dir, alignment, flags);
                break;
        case WQ:
                buf = buf_alloc_wq(length, dir, alignment, flags);
                break;
        default:
                gds_err("unexpected obj type=%d\n", type);
                break;
        }

        return buf;
}

void gds_peer::free(gds_buf *buf)
{
        int ret = gds_peer_mfree(gpu_id, 0, buf->addr, buf->handle);
        if (ret) {
                gds_err("error freeing GPU mapped memory\n");
        }
        delete buf;
}

// buf is a GPU mem buffer, which has a CPU mapping thanks to GDRcopy
gds_range *gds_peer::range_from_buf(gds_buf *buf, void *start, size_t length)
{
        gds_range *range = new gds_range;
        gds_dbg("buf=%p\n", buf);
        assert((ptrdiff_t)start >= (ptrdiff_t)buf->addr && 
               (ptrdiff_t)start + length <= (ptrdiff_t)buf->addr + buf->length);
        range->va = start; // CPU mapping
        range->dptr = buf->peer_addr + ((ptrdiff_t)start - (ptrdiff_t)buf->addr);
        range->size = length;
        range->buf = buf;
        range->type = GDS_MEMORY_GPU;
        return range;
}

gds_range *gds_peer::register_range(void *start, size_t length, int flags)
{
        int ret = 0;
        gds_range *range = NULL;
        gds_dbg("start=%p length=%zu\n", start, length);
        gds_memory_type_t mem_type = memtype_from_flags(flags);
        CUdeviceptr dev_ptr = 0;

        ret = gds_register_mem(start, length, mem_type, &dev_ptr);
        if (ret) {
                gds_err("got %d while registering range [%p,%p]\n", ret, start, (char*)start+length-1);
                goto out;
        }
        range = new gds_range;
        assert(range);
        range->va = start;
        range->dptr = dev_ptr;
        range->size = length;
        range->buf = NULL;
        range->type = mem_type;
out:
        gds_dbg("range=%p\n", range);
        return range;
}

void gds_peer::unregister(gds_range *range)
{
        assert(range);
        gds_dbg("unregistering range %p %zu buf=%p\n", range->va, range->size, range->buf);
        if (!range->buf) {
                // both HOST and IO mem
                gds_unregister_mem(range->va, range->size);
        } else {
                //GPU mem does not need deregistration
        }
        delete range;
}
