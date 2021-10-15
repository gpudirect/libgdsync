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

#include "gdsync.h"
#include "gdsync/mlx5.h"
#include "utils.hpp"
#include "memmgr.hpp"
//#include "mem.hpp"
#include "objs.hpp"
#include "utils.hpp"
#include "transport.hpp"

#if 0
                union { uint64_t qw; uint32_t dw[2]; } db_val;
                db_val.qw = 0;
                db_val.dw[0] = desc->data32;
                db_val.dw[1] = desc->data32;
                mlx5_i->db_value = db_val.qw;
#endif


//-----------------------------------------------------------------------------

int gds_mlx5_get_send_info(int count, const gds_send_request_t *requests, gds_mlx5_send_info_t *mlx5_infos)
{
        int retcode = gds_transport_init();
        if (retcode) {
                gds_err("error in gds_transport_init\n");
                goto out;
        }

	for (int j=0; j<count; j++) {
                gds_mlx5_send_info *mlx5_i = mlx5_infos + j;
                const gds_send_request_t *request = requests + j;
                retcode = gds_main_transport->get_send_descs(mlx5_i, request);
                if (retcode) {
                        gds_err("error %d while retrieving descriptors for %dth request\n", retcode, j);
                        break;
                }
                gds_dbg("mlx5_i: dbrec={%p,%08x} db={%p,%" PRIx64 "}\n",
                        mlx5_i->dbrec_ptr, mlx5_i->dbrec_value, mlx5_i->db_ptr, mlx5_i->db_value);
	}

out:
	return retcode;
}

//-----------------------------------------------------------------------------

int gds_mlx5_get_wait_descs(gds_mlx5_wait_info_t *mlx5_i, const gds_wait_request_t *request)
{
        return gds_main_transport->get_wait_descs(mlx5_i, request);
}

//-----------------------------------------------------------------------------

int gds_mlx5_get_wait_info(int count, const gds_wait_request_t *requests, gds_mlx5_wait_info_t *mlx5_infos)
{
        int retcode = 0;

	for (int j=0; j<count; j++) {
                gds_mlx5_wait_info *mlx5_i = mlx5_infos + j;
                const gds_wait_request_t *request = requests + j;
                retcode = gds_mlx5_get_wait_descs(mlx5_i, request);
                if (retcode) {
                        gds_err("error %d while retrieving descriptors for %dth request\n", retcode, j);
                        break;
                }
                gds_dbg("wait[%d] cqe_ptr=%p cqe_value=0x%08x flag_ptr=%p flag_value=0x%08x\n", 
                        j, mlx5_i->cqe_ptr, mlx5_i->cqe_value, mlx5_i->flag_ptr, mlx5_i->flag_value);
        }

        return retcode;
}

//-----------------------------------------------------------------------------

int gds_mlx5_get_dword_wait_info(uint32_t *ptr, uint32_t value, int flags, gds_mlx5_dword_wait_info_t *mlx5_info)
{
        int retcode = 0;
        CUdeviceptr dev_ptr = 0;

        assert(NULL != ptr);
        //assert((((unsigned long)ptr) & 0x3) == 0);
        memset(mlx5_info, 0, sizeof(&mlx5_info));

        retcode = gds_map_mem(ptr, sizeof(*ptr), memtype_from_flags(flags), &dev_ptr);
        if (retcode) {
                gds_err("error %d while mapping addr %p\n", retcode, ptr);
                goto out;
        }

        gds_dbg("dev_ptr=%llx value=%08x\n", dev_ptr, value);
        mlx5_info->ptr = (uint32_t*)dev_ptr;
        mlx5_info->value = value;
out:
        return retcode;
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
