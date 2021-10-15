/* Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include <assert.h>
#include <gdsync.h>
#include <gdsync/mlx5.h>

typedef struct gds_transport {
        int  (*create_qp)(struct ibv_pd *pd, struct ibv_context *context, gds_qp_init_attr_t *qp_attr, gds_peer *peer, gds_peer_attr *peer_attr, int flags, gds_qp_t **gqp);
        int  (*destroy_qp)(gds_qp_t *gqp);
        int  (*rollback_qp)(gds_qp_t *gqp, gds_send_request_t *request);

        void (*init_send_info)(gds_send_request_t *request);
        int  (*post_send_ops)(gds_peer *peer, gds_send_request_t *request, gds_op_list_t &ops);
        int  (*post_send_ops_on_cpu)(gds_send_request_t *request, int flags);
        int  (*prepare_send)(gds_qp_t *gqp, gds_send_wr *p_ewr, gds_send_wr **bad_ewr, gds_send_request_t *request);
        int  (*get_send_descs)(gds_mlx5_send_info_t *mlx5_i, const gds_send_request_t *_request);
        uint32_t (*get_num_send_request_entries)(gds_send_request_t *request);

        void (*init_wait_request)(gds_wait_request_t *request, uint32_t offset);
        void (*dump_wait_request)(gds_wait_request_t *request, size_t idx);
        int  (*stream_post_wait_descriptor)(gds_peer *peer, gds_wait_request_t *request, gds_op_list_t &params, int flags);
        int  (*post_wait_descriptor)(gds_wait_request_t *request, int flags);
        int  (*get_wait_descs)(gds_mlx5_wait_info_t *mlx5_i, const gds_wait_request_t *request);
        uint32_t (*get_num_wait_request_entries)(gds_wait_request_t *request);

        int  (*prepare_wait_cq)(gds_cq_t *gcq, gds_wait_request_t *request, int flags);
        int  (*append_wait_cq)(gds_wait_request_t *request, uint32_t *dw, uint32_t val);
        int  (*abort_wait_cq)(gds_cq_t *gcq, gds_wait_request_t *request);
} gds_transport_t;

extern gds_transport_t *gds_main_transport;

int gds_transport_mlx5_exp_init(gds_transport_t **transport);

static inline int gds_transport_init()
{
        int status = 0;
        if (!gds_main_transport) {
                gds_transport_t *t = NULL;
                status = gds_transport_mlx5_exp_init(&t);
                if (status) {
                        gds_err("error in gds_transport_mlx5_exp_init\n");
                        goto out;
                }
                assert(t);
                gds_main_transport = t;
        }
out:
        return status;
}

/*
 * Local variables:
 *  c-indent-level: 8
 *  c-basic-offset: 8
 *  tab-width: 8
 * End:
 */
