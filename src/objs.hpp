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

static const size_t max_gpus = 16;

/**
 * Compatible with enum ibv_exp_peer_op
 */
typedef enum gds_peer_op {
    GDS_PEER_OP_RESERVED1   = 1,

    GDS_PEER_OP_FENCE       = 0,

    GDS_PEER_OP_STORE_DWORD = 4, 
    GDS_PEER_OP_STORE_QWORD = 2,
    GDS_PEER_OP_COPY_BLOCK  = 3,
    
    GDS_PEER_OP_POLL_AND_DWORD  = 12,
    GDS_PEER_OP_POLL_NOR_DWORD  = 13,
    GDS_PEER_OP_POLL_GEQ_DWORD  = 14,
} gds_peer_op_t;      
    
/**
 * Compatible with enum ibv_exp_peer_op_caps
 */
enum gds_peer_op_caps {
        GDS_PEER_OP_FENCE_CAP   = (1 << GDS_PEER_OP_FENCE),
        GDS_PEER_OP_STORE_DWORD_CAP = (1 << GDS_PEER_OP_STORE_DWORD),
        GDS_PEER_OP_STORE_QWORD_CAP = (1 << GDS_PEER_OP_STORE_QWORD),
        GDS_PEER_OP_COPY_BLOCK_CAP  = (1 << GDS_PEER_OP_COPY_BLOCK),
        GDS_PEER_OP_POLL_AND_DWORD_CAP
                = (1 << GDS_PEER_OP_POLL_AND_DWORD),
        GDS_PEER_OP_POLL_NOR_DWORD_CAP
                = (1 << GDS_PEER_OP_POLL_NOR_DWORD),
        GDS_PEER_OP_POLL_GEQ_DWORD_CAP
                = (1 << GDS_PEER_OP_POLL_GEQ_DWORD),
};


/**
 * Compatible with enum ibv_exp_peer_fence
 */
typedef enum gds_peer_fence {
        GDS_PEER_FENCE_OP_READ      = (1 << 0), 
        GDS_PEER_FENCE_OP_WRITE     = (1 << 1), 
        GDS_PEER_FENCE_FROM_CPU     = (1 << 2), 
        GDS_PEER_FENCE_FROM_HCA     = (1 << 3), 
        GDS_PEER_FENCE_MEM_SYS      = (1 << 4), 
        GDS_PEER_FENCE_MEM_PEER     = (1 << 5), 
} gds_peer_fence_t;

/**
 * Indicate HW entities supposed to access memory buffer:
 * GDS_PEER_DIRECTION_FROM_X means X writes to the buffer
 * GDS_PEER_DIRECTION_TO_Y means Y read from the buffer
 *
 * Compatible with enum ibv_exp_peer_direction
 */
enum gds_peer_direction {
        GDS_PEER_DIRECTION_FROM_CPU  = (1 << 0),
        GDS_PEER_DIRECTION_FROM_HCA  = (1 << 1),
        GDS_PEER_DIRECTION_FROM_PEER = (1 << 2),
        GDS_PEER_DIRECTION_TO_CPU    = (1 << 3),
        GDS_PEER_DIRECTION_TO_HCA    = (1 << 4),
        GDS_PEER_DIRECTION_TO_PEER   = (1 << 5),
};

/**
 * Compatible with enum ibv_exp_peer_direct_attr_mask
 */
enum gds_peer_direct_attr_mask {
        GDS_PEER_DIRECT_VERSION = (1 << 0) /* Must be set */
};

/**
 * Compatible with IBV_EXP_PEER_IOMEMORY
 */
#define GDS_PEER_IOMEMORY ((struct gds_buf *)-1UL)

/**
 * Compatible with struct ibv_exp_peer_buf_alloc_attr
 */
typedef struct gds_peer_buf_alloc_attr {
        size_t length;
        /* Bitmask from enum gds_peer_direction */
        uint32_t dir;
        /* The ID of the peer device which will be
         *      * accessing the allocated buffer
         *           */
        uint64_t peer_id;
        /* Data alignment */
        uint32_t alignment;
        /* Reserved for future extensions, must be 0 */
        uint32_t comp_mask;
} gds_peer_buf_alloc_attr_t;


/**
 * Compatible with struct ibv_exp_peer_buf
 */
typedef struct gds_peer_buf {
        void *addr;
        size_t length;
        /* Reserved for future extensions, must be 0 */
        uint32_t comp_mask;
} gds_peer_buf_t;

/**
 * Compatible with struct ibv_exp_peer_direct_attr
 */
typedef struct {
        /* Unique ID per peer device.
         * Used to identify specific HW devices where relevant.
         */
        uint64_t peer_id;
        /* buf_alloc callback should return gds_peer_buf_t with buffer
         * of at least attr->length.
         * @attr: description of desired buffer
         *
         * Buffer should be mapped in the application address space
         * for read/write (depends on attr->dir value).
         * attr->dir value is supposed to indicate the expected directions
         * of access to the buffer, to allow optimization by the peer driver.
         * If NULL returned then buffer will be allocated in system memory
         * by ibverbs driver.
         */
        gds_peer_buf_t *(*buf_alloc)(gds_peer_buf_alloc_attr_t *attr);
        /* If buffer was allocated by buf_alloc then buf_release will be
         * called to release it.
         * @pb: struct returned by buf_alloc
         *
         * buf_release is responsible to release everything allocated by
         * buf_alloc.
         * Return 0 on succes.
         */
        int (*buf_release)(gds_peer_buf_t *pb);
        /* register_va callback should register virtual address from the
         * application as an area the peer is allowed to access.
         * @start: pointer to beginning of region in virtual space
         * @length: length of region
         * @peer_id: the ID of the peer device which will be accessing
         * the region.
         * @pb: if registering a buffer that was returned from buf_alloc(),
         * pb is the struct that was returned. If registering io memory area,
         * pb is GDS_PEER_IOMEMORY. Otherwise - NULL
         *
         * Return id of registered address on success, 0 on failure.
         */
        uint64_t (*register_va)(void *start, size_t length, uint64_t peer_id,
                        gds_peer_buf_t *pb);
        /* If virtual address was registered with register_va then
         * unregister_va will be called to unregister it.
         * @target_id: id returned by register_va
         * @peer_id: the ID of the peer device passed to register_va
         *
         * Return 0 on success.
         */
        int (*unregister_va)(uint64_t target_id, uint64_t peer_id);
        /* Bitmask from gds_peer_op_caps */
        uint64_t caps;
        /* Maximal length of DMA operation the peer can do in copy-block */
        size_t peer_dma_op_map_len;
        /* From gds_peer_direct_attr_mask */
        uint32_t comp_mask;
        /* Feature version, must be 1 */
        uint32_t version;
} gds_peer_attr;

struct gds_peer;

struct gds_buf: gds_peer_buf_t {
        gds_peer   *peer;
        CUdeviceptr peer_addr;
        gds_memory_type_t mem_type;
        void       *handle;

        gds_buf(gds_peer *p, size_t sz, gds_memory_type_t mem_type): peer(p), peer_addr(0), handle(NULL), mem_type(mem_type) {
                addr = NULL;
                length = sz;
                comp_mask = 0;
        }
};

struct gds_range {
        void *va;
        CUdeviceptr dptr;
        size_t size;
        gds_buf *buf;
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
        void *opaque;

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

typedef struct gds_peer_op_wr {
        struct gds_peer_op_wr *next;
        gds_peer_op_t type;
        union {
                struct {
                        uint64_t fence_flags; /* from gds_peer_fence_t */
                } fence;

                struct {
                        uint32_t        data;
                        uint64_t        target_id;
                        size_t          offset;
                } dword_va; /* Use for all operations targeting dword */

                struct {
                        uint64_t        data;
                        uint64_t        target_id;
                        size_t          offset;
                } qword_va; /* Use for all operations targeting qword */

                struct {
                        void           *src;
                        uint64_t        target_id;
                        size_t          offset;
                        size_t          len;
                } copy_op;
        } wr;
        uint32_t comp_mask; /* Reserved for future expensions, must be 0 */
} gds_peer_op_wr_t;


/*
 * Local variables:
 *  c-indent-level: 8
 *  c-basic-offset: 8
 *  tab-width: 8
 * End:
 */
