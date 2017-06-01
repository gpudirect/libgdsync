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
#error "gdsync.h must be included first"
#endif

// low-level APIs
// for testing purposes mainly

GDS_BEGIN_DECLS

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
int gds_stream_post_poll_dword(CUstream stream, uint32_t *ptr, uint32_t magic, gds_wait_cond_flag_t cond_flag, int flags);
int gds_stream_post_poke_dword(CUstream stream, uint32_t *ptr, uint32_t value, int flags);
int gds_stream_post_inline_copy(CUstream stream, void *ptr, void *src, size_t nbytes, int flags);
int gds_stream_post_polls_and_pokes(CUstream stream, 
				    size_t n_polls, uint32_t *ptrs[], uint32_t magics[], gds_wait_cond_flag_t cond_flags[], int poll_flags[], 
				    size_t n_pokes, uint32_t *poke_ptrs[], uint32_t poke_values[], int poke_flags[]);
int gds_stream_post_polls_and_immediate_copies(CUstream stream, 
						size_t n_polls, uint32_t *ptrs[], uint32_t magics[], gds_wait_cond_flag_t cond_flags[], int poll_flags[], 
						size_t n_imms, void *imm_ptrs[], void *imm_datas[], size_t imm_bytes[], int imm_flags[]);

GDS_END_DECLS
