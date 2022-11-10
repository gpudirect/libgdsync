/*
 * Copyright (c) 2012 Mellanox Technologies, Inc.  All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * OpenIB.org BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma once

#include "gdsync.h"
#include "gdsync/mlx5.h"

// from libmlx5/src/doorbell.h

/*
 * Avoid using memcpy() to copy to BlueFlame page, since memcpy()
 * implementations may use move-string-buffer assembler instructions,
 * which do not guarantee order of copying.
 */
#if defined(__x86_64__)
#define COPY_64B_NT(dst, src)		\
	__asm__ __volatile__ (		\
	" movdqa   (%1),%%xmm0\n"	\
	" movdqa 16(%1),%%xmm1\n"	\
	" movdqa 32(%1),%%xmm2\n"	\
	" movdqa 48(%1),%%xmm3\n"	\
	" movntdq %%xmm0,   (%0)\n"	\
	" movntdq %%xmm1, 16(%0)\n"	\
	" movntdq %%xmm2, 32(%0)\n"	\
	" movntdq %%xmm3, 48(%0)\n"	\
	: : "r" (dst), "r" (src) : "memory");	\
	dst += 8;			\
	src += 8
#else
#define COPY_64B_NT(dst, src)	\
	*dst++ = *src++;	\
	*dst++ = *src++;	\
	*dst++ = *src++;	\
	*dst++ = *src++;	\
	*dst++ = *src++;	\
	*dst++ = *src++;	\
	*dst++ = *src++;	\
	*dst++ = *src++

#endif
// no WQ wrap-around check!!!
static inline void gds_bf_copy(uint64_t *dest, uint64_t *src, size_t n_bytes)
{
        assert(n_bytes % sizeof(uint64_t) == 0);
        assert(n_bytes < 128);
	while (n_bytes > 0) {
		COPY_64B_NT(dest, src);
		n_bytes -= 8 * sizeof(*dest);
	}
}

int gds_mlx5_get_send_descs(gds_mlx5_send_info_t *mlx5_i, const size_t n_ops, const gds_peer_op_wr_t *op);
int gds_mlx5_get_wait_descs(gds_mlx5_wait_info_t *mlx5_i, gds_peer_op_wr_t *op, size_t n_ops);


/*
 * Local variables:
 *  c-indent-level: 8
 *  c-basic-offset: 8
 *  tab-width: 8
 *  indent-tabs-mode: nil
 * End:
 */
