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

GDS_BEGIN_DECLS

typedef struct gds_mlx5_dword_wait_info {
        uint32_t *ptr;
        uint32_t  value;	
} gds_mlx5_dword_wait_info_t;

int gds_mlx5_get_dword_wait_info(uint32_t *ptr, uint32_t value, int flags, gds_mlx5_dword_wait_info_t *mlx5_info);

typedef struct gds_mlx5_send_info {
        unsigned membar:1;
        unsigned membar_full:1;
        uint32_t *dbrec_ptr;
        uint32_t  dbrec_value;
        uint64_t *db_ptr;
        uint64_t  db_value;
} gds_mlx5_send_info_t;

int gds_mlx5_get_send_info(int count, const gds_send_request_t *requests, gds_mlx5_send_info_t *mlx5_infos);

typedef struct gds_mlx5_wait_info {
        gds_wait_cond_flag_t cond;
        uint32_t *cqe_ptr;
        uint32_t  cqe_value;
        uint32_t *flag_ptr;
        uint32_t  flag_value;
} gds_mlx5_wait_info_t;

int gds_mlx5_get_wait_info(int count, const gds_wait_request_t *requests, gds_mlx5_wait_info_t *mlx5_infos);

GDS_END_DECLS
