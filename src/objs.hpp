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

#pragma once

#include "task_queue.hpp"

static const size_t max_gpus = 16;

enum gds_peer_fence {
        GDS_PEER_FENCE_OP_READ      = (1 << 0), 
        GDS_PEER_FENCE_OP_WRITE     = (1 << 1), 
        GDS_PEER_FENCE_FROM_CPU     = (1 << 2), 
        GDS_PEER_FENCE_FROM_HCA     = (1 << 3), 
        GDS_PEER_FENCE_MEM_SYS      = (1 << 4), 
        GDS_PEER_FENCE_MEM_PEER     = (1 << 5), 
};

enum gds_peer_direction {
        GDS_PEER_DIRECTION_FROM_CPU  = (1 << 0),
        GDS_PEER_DIRECTION_FROM_HCA  = (1 << 1),
        GDS_PEER_DIRECTION_FROM_PEER = (1 << 2),
        GDS_PEER_DIRECTION_TO_CPU    = (1 << 3),
        GDS_PEER_DIRECTION_TO_HCA    = (1 << 4),
        GDS_PEER_DIRECTION_TO_PEER   = (1 << 5),
};  

#define GDS_PEER_IOMEMORY ((struct gds_buf *)-1UL)

struct gds_peer;

struct gds_buf_alloc_attr {
        size_t      length;
        uint32_t    dir;
        uint64_t    peer_id;
        uint32_t    alignment;
        uint32_t    comp_mask;
};

struct gds_buf {
        void       *addr;
        size_t      length;
        uint32_t    comp_mask;
        gds_peer   *peer;
        CUdeviceptr peer_addr;
        void       *handle;

        gds_buf(gds_peer *p, size_t sz): peer(p), peer_addr(0), handle(NULL) {
                addr = NULL;
                length = sz;
                comp_mask = 0;
        }
};

typedef struct {
        uint64_t        peer_id;
        struct gds_buf *(*buf_alloc)(struct gds_buf_alloc_attr *attr);
        int             (*buf_release)(struct gds_buf *pb);
        uint64_t        (*register_va)(void *start, size_t length, uint64_t peer_id, struct gds_buf *pb);
        int             (*unregister_va)(uint64_t target_id, uint64_t peer_id);
        uint64_t        caps;
        size_t          peer_dma_op_map_len;
        uint32_t        comp_mask;
        uint32_t        version;
} gds_peer_attr;


struct gds_range {
        void *va;
        CUdeviceptr dptr;
        size_t size;
        struct gds_buf *buf;
        gds_memory_type_t type;
};

static inline uint64_t range_to_id(gds_range *range)
{
        assert(range);
        return reinterpret_cast<uint64_t>(range);
}

static inline gds_range *range_from_id(uint64_t id)
{
        assert(id);
        return reinterpret_cast<gds_range *>(id);
}

class task_queue;

struct gds_peer {
        int gpu_id;
        CUdevice gpu_dev;
        CUcontext gpu_ctx;
        bool has_memops;
        bool has_remote_flush;
        bool has_write64;
        bool has_wait_nor;
        bool has_inlcpy;
        bool has_membar;
        bool has_weak;
        unsigned max_batch_size;
        gds_peer_attr attr;
        task_queue *tq;

        void *obj;
        enum obj_type { NONE, CQ, WQ, N_IBV_OBJS } alloc_type;
        // This field works as a ugly run-time parameters passing
        // mechanism, as it carries tracking info during the QP creation
        // phase, so no more than one outstanding call per peer is
        // supported.  In practice, before calling ibv_exp_create_cq(), we
        // patch this field with the appropriate value
        int alloc_flags; // out of gds_flags_t

        // register peer memory
        gds_range *range_from_buf(gds_buf *buf, void *start, size_t length);
        // register any other kind of memory
        enum range_type { HOST_MEM, IO_MEM };
        gds_range *register_range(void *start, size_t length, int flags);
        // unregister all kinds of memory
        void unregister(gds_range *range);

        gds_buf *alloc(size_t length, uint32_t alignment, gds_memory_type_t mem_type);
        gds_buf *buf_alloc_cq(size_t length, uint32_t dir, uint32_t alignment, int flags);
        gds_buf *buf_alloc_wq(size_t length, uint32_t dir, uint32_t alignment, int flags);
        gds_buf *buf_alloc(obj_type type, size_t length, uint32_t dir, uint32_t alignment, int flags);
        void free(gds_buf *buf);
};

static inline uint64_t peer_to_id(gds_peer *peer)
{
        assert(peer);
        return reinterpret_cast<uint64_t>(peer);
}

static inline gds_peer *peer_from_id(uint64_t id)
{
        assert(id);
        return reinterpret_cast<gds_peer *>(id);
}

/*
 * Local variables:
 *  c-indent-level: 8
 *  c-basic-offset: 8
 *  tab-width: 8
 * End:
 */
