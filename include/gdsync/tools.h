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
#ifndef __GDSYNC_H__
#error "don't include directly this header, use gdsync.h always"
#endif

// low-level APIs
// for testing purposes mainly

typedef enum gds_poll_cond_flag {
        GDS_POLL_COND_GEQ = 0, // must match verbs_exp enum
        GDS_POLL_COND_EQ,
        GDS_POLL_COND_AND,
        GDS_POLL_COND_NOR
} gds_poll_cond_flag_t;

typedef enum gds_memory_type {
        GDS_MEMORY_GPU  = 1,
        GDS_MEMORY_HOST = 2,
        GDS_MEMORY_IO   = 4,
	GDS_MEMORY_MASK = 0x7
} gds_poll_memory_type_t;

typedef enum gds_poll_flags {
	GDS_POLL_POST_FLUSH = 1<<3,
} gds_poll_flags_t;

typedef enum gds_poke_flags {
	GDS_POKE_POST_PRE_BARRIER = 1<<4,
} gds_poke_flags_t;

typedef enum gds_immcopy_flags {
	GDS_IMMCOPY_POST_TAIL_FLUSH = 1<<4,
} gds_immcopy_flags_t;

typedef enum gds_membar_flags {
	GDS_MEMBAR_FLUSH_REMOTE = 1<<4,
	GDS_MEMBAR_DEFAULT      = 1<<5,
	GDS_MEMBAR_SYS          = 1<<6,
} gds_membar_flags_t;

typedef struct gds_mem_desc {
    CUdeviceptr d_ptr;
    void       *h_ptr;
    void       *bar_ptr;
    int         flags;
    size_t      alloc_size;
    gdr_mh_t    mh;
} gds_mem_desc_t;
int gds_alloc_mapped_memory(gds_mem_desc_t *desc, size_t size, int flags);
int gds_free_mapped_memory(gds_mem_desc_t *desc);

// flags is combination of gds_memory_type and gds_poll_flags
int gds_stream_post_poll_dword(CUstream stream, uint32_t *ptr, uint32_t magic, int cond_flag, int flags);
int gds_stream_post_poke_dword(CUstream stream, uint32_t *ptr, uint32_t value, int flags);
int gds_stream_post_inline_copy(CUstream stream, void *ptr, void *src, size_t nbytes, int flags);
int gds_stream_post_polls_and_pokes(CUstream stream, 
				    size_t n_polls, uint32_t *ptrs[], uint32_t magics[], int cond_flags[], int poll_flags[], 
				    size_t n_pokes, uint32_t *poke_ptrs[], uint32_t poke_values[], int poke_flags[]);
int gds_stream_post_polls_and_immediate_copies(CUstream stream, 
						size_t n_polls, uint32_t *ptrs[], uint32_t magics[], int cond_flags[], int poll_flags[], 
						size_t n_imms, void *imm_ptrs[], void *imm_datas[], size_t imm_bytes[], int imm_flags[]);
