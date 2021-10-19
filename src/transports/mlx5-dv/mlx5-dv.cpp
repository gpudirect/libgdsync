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

#include <unistd.h>
#include <string.h>
#include <assert.h>
#include <arpa/inet.h>
#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>
#include <time.h>

#include <map>

#include "gdsync.h"
#include "gdsync/mlx5.h"
#include "gdsync/tools.h"

#include "archutils.h"
#include "mem.hpp"
#include "memmgr.hpp"
#include "mlx5-dv.hpp"
#include "mlnxutils.h"
#include "objs.hpp"
#include "utils.hpp"
#include "transport.hpp"
#include "mlx5_ifc.h"

//-----------------------------------------------------------------------------

/**
 * Create a CQ using DirectVerbs.
 * @params pd parent_domain with GPU memory allocation support.
 */
static int gds_mlx5_dv_create_cq(
        struct ibv_context *context, int cqe,
        void *cq_context, struct ibv_comp_channel *channel,
        int comp_vector, struct ibv_pd *pd,
        gds_peer_attr *peer_attr, int alloc_flags,
        gds_mlx5_dv_cq_t **out_mcq
)
{
        int ret = 0;

        bool register_peer_buf = false;
        bool register_peer_dbr = false;

        struct ibv_cq_ex *ibcq_ex = NULL;
        struct ibv_cq *ibcq = NULL;
        gds_mlx5_dv_cq_t *mcq = NULL;
        gds_cq_t *gcq;

        gds_mlx5_dv_cq_peer_t *mcq_peer = NULL;

        mlx5dv_obj dv_obj;

        gds_peer *peer = NULL;

        struct ibv_cq_init_attr_ex cq_attr = {
                .cqe            = (uint32_t)cqe,
                .cq_context     = cq_context,
                .channel        = channel,
                .comp_vector    = (uint32_t)comp_vector,
                .wc_flags       = IBV_WC_STANDARD_FLAGS,
                .comp_mask      = (uint32_t)IBV_CQ_INIT_ATTR_MASK_PD,
                .flags          = 0,
                .parent_domain  = pd
        };

        assert(peer_attr);

        mcq = (gds_mlx5_dv_cq_t *)calloc(1, sizeof(gds_mlx5_dv_cq_t));
        if (!mcq) {
                gds_err("cannot allocate memory\n");
                ret = ENOMEM;
                goto err;
        }

        mcq_peer = (gds_mlx5_dv_cq_peer_t *)calloc(1, sizeof(gds_mlx5_dv_cq_peer_t));
        if (!mcq_peer) {
                gds_err("cannot allocate memory\n");
                ret = ENOMEM;
                goto err;
        }
        mcq_peer->peer_attr = peer_attr;

        mcq->cq_peer = mcq_peer;

        peer = peer_from_id(peer_attr->peer_id);

        // Setup peer allocation
        peer->alloc_type = gds_peer::CQ;
        peer->alloc_flags = alloc_flags;
        // mcq_peer will be filled if we do allocation on device.
        // pd_mem_alloc is responsible for the registration.
        peer->opaque = mcq_peer;

        ibcq_ex = mlx5dv_create_cq(context, &cq_attr, NULL);
        if (!ibcq_ex) {
                gds_err("error in mlx5dv_create_cq\n");
                ret = EINVAL;
                goto err;
        }
        ibcq = ibv_cq_ex_to_cq(ibcq_ex);

        dv_obj.cq.in = ibcq;
        dv_obj.cq.out = &mcq->dvcq;
        ret = mlx5dv_init_obj(&dv_obj, MLX5DV_OBJ_CQ);
        if (ret) {
                gds_err("error %d in mlx5dv_init_obj MLX5DV_OBJ_CQ\n", ret);
                ret = EINVAL;
                goto err;
        }

        // If va_id is not set, pd_mem_alloc did not allocate the buffer on device.
        // Hence, we register the buffer to the device here.
        if (!mcq_peer->buf.va_id) {
                if (alloc_flags & GDS_ALLOC_CQ_ON_GPU)
                        gds_err("Cannot alloc CQ buf on GPU. Falling back to host.\n");
                mcq_peer->buf.va_id = peer_attr->register_va(
                        mcq->dvcq.buf,
                        (uint64_t)mcq->dvcq.cqe_cnt * (uint64_t)mcq->dvcq.cqe_size,
                        peer_attr->peer_id,
                        NULL
                );
                if (!mcq_peer->buf.va_id) {
                        gds_err("error in peer_attr->register_va\n");
                        ret = EINVAL;
                        goto err;
                }
                register_peer_buf = true;
        }

        if (!mcq_peer->dbr.va_id) {
                if (alloc_flags & GDS_ALLOC_CQ_DBREC_ON_GPU)
                        gds_err("Cannot alloc CQ DBREC on GPU. Falling back to host.\n");
                mcq_peer->dbr.va_id = peer_attr->register_va(
                        mcq->dvcq.dbrec,
                        GDS_MLX5_DV_DBR_BUF_SIZE,
                        peer_attr->peer_id,
                        NULL
                );
                if (!mcq_peer->dbr.va_id) {
                        gds_err("error in dbr register_va\n");
                        ret = EINVAL;
                        goto err;
                }
                register_peer_dbr = true;
        }

        mcq_peer->pdata.peek_table = (struct gds_mlx5_dv_peek_entry **)malloc(sizeof(struct gds_mlx5_dv_peek_entry *) * mcq->dvcq.cqe_cnt);
        if (!mcq_peer->pdata.peek_table) {
                gds_err("error %d in malloc peer_peek_table\n", errno);
                ret = ENOMEM;
                goto err;
        }
        memset(mcq_peer->pdata.peek_table, 0, sizeof(struct gds_peek_entry *) * mcq->dvcq.cqe_cnt);
        mcq_peer->pdata.dir = GDS_PEER_DIRECTION_FROM_PEER | GDS_PEER_DIRECTION_TO_CPU;

        mcq_peer->pdata.gbuf = peer->buf_alloc(
                peer->alloc_type, 
                sizeof(struct gds_mlx5_dv_peek_entry) * mcq->dvcq.cqe_cnt,
                mcq_peer->pdata.dir,
                (uint32_t)sysconf(_SC_PAGESIZE),
                peer->alloc_flags
        );
        if (!mcq_peer->pdata.gbuf) {
                gds_err("error %d in buf_alloc\n", errno);
                ret = ENOMEM;
                goto err;
        }

        mcq_peer->pdata.va_id = peer_attr->register_va(
                mcq_peer->pdata.gbuf->addr, 
                mcq_peer->pdata.gbuf->length, 
                peer_attr->peer_id, 
                mcq_peer->pdata.gbuf
        );
        if (!mcq_peer->pdata.va_id) {
                gds_err("error %d in register_va\n", errno);
                ret = EINVAL;
                goto err;
        }

        memset(mcq_peer->pdata.gbuf->addr, 0, mcq_peer->pdata.gbuf->length);

        mcq_peer->pdata.peek_free = (struct gds_mlx5_dv_peek_entry *)mcq_peer->pdata.gbuf->addr;
        for (int i = 0; i < mcq->dvcq.cqe_cnt - 1; ++i)
                mcq_peer->pdata.peek_free[i].next = i + 1;
        mcq_peer->pdata.peek_free[mcq->dvcq.cqe_size - 1].next = GDS_MLX5_DV_LAST_PEEK_ENTRY;

        mcq->gcq.cq = ibcq;
        *out_mcq = mcq;

        return 0;

err:
        if (mcq_peer) {
                if (mcq_peer->pdata.va_id)
                        peer_attr->unregister_va(mcq_peer->pdata.va_id, peer_attr->peer_id);

                if (mcq_peer->pdata.gbuf)
                        peer_attr->buf_release(mcq_peer->pdata.gbuf);

                if (mcq_peer->pdata.peek_table)
                        free(mcq_peer->pdata.peek_table);

                if (register_peer_buf)
                        peer_attr->unregister_va(mcq_peer->buf.va_id, peer_attr->peer_id);

                if (register_peer_dbr)
                        peer_attr->unregister_va(mcq_peer->dbr.va_id, peer_attr->peer_id);
        }

        if (ibcq)
                ibv_destroy_cq(ibcq);

        if (mcq_peer)
                free(mcq_peer);

        if (mcq)
                free(mcq);

        return ret;
}

//-----------------------------------------------------------------------------

static void gds_mlx5_dv_destroy_cq(gds_mlx5_dv_cq_t *mcq)
{
        int status = 0;
        gds_mlx5_dv_cq_peer_t *mcq_peer = mcq->cq_peer;

        if (mcq->wq)
                mcq->wq = NULL;

        if (mcq_peer && mcq_peer->pdata.peek_table) {
                free(mcq_peer->pdata.peek_table);
                mcq_peer->pdata.peek_table = NULL;
        }

        if (mcq_peer && mcq_peer->peer_attr) {
                gds_peer_attr *peer_attr = mcq_peer->peer_attr;
                gds_peer *peer = peer_from_id(peer_attr->peer_id);

                // This may be used by ibv_destroy_cq, which eventually calls pd_mem_free.
                peer->alloc_type = gds_peer::CQ;
                peer->opaque = mcq_peer;

                // gbuf has value iff pd_mem_alloc handled the allocation.
                // In that case, leave the deallocation to pd_mem_free.
                if (mcq_peer->buf.va_id && mcq_peer->buf.gbuf == NULL) {
                        peer_attr->unregister_va(mcq_peer->buf.va_id, peer_attr->peer_id);
                        mcq_peer->buf.va_id = 0;
                }
                if (mcq_peer->dbr.va_id && mcq_peer->dbr.gbuf == NULL) {
                        peer_attr->unregister_va(mcq_peer->dbr.va_id, peer_attr->peer_id);
                        mcq_peer->dbr.va_id = 0;
                }
                if (mcq_peer->pdata.va_id) {
                        peer_attr->unregister_va(mcq_peer->pdata.va_id, peer_attr->peer_id);
                        mcq_peer->pdata.va_id = 0;
                }
                if (mcq_peer->pdata.gbuf) {
                        peer_attr->buf_release(mcq_peer->pdata.gbuf);
                        mcq_peer->pdata.gbuf = NULL;
                }
        }

        if (mcq->gcq.cq) {
                status = ibv_destroy_cq(mcq->gcq.cq);
                if (status) {
                        gds_err("error %d in ibv_destroy\n", status);
                        return;
                }
                mcq->gcq.cq = NULL;
        }

        if (mcq_peer) {
                free(mcq_peer);
                mcq->cq_peer = NULL;
        }

        free(mcq);
}

//-----------------------------------------------------------------------------

static void *pd_mem_alloc(struct ibv_pd *pd, void *pd_context, size_t size,
                        size_t alignment, uint64_t resource_type)
{
        assert(pd_context);

        gds_peer_attr *peer_attr = (gds_peer_attr *)pd_context;
        gds_peer *peer = peer_from_id(peer_attr->peer_id);
        uint32_t dir = 0;
        uint64_t range_id;
        gds_buf *buf = NULL;
        void *ptr = NULL;

        gds_dbg("pd_mem_alloc: pd=%p, pd_context=%p, size=%zu, alignment=%zu, resource_type=0x%lx\n",
                pd, pd_context, size, alignment, resource_type);

        // Prevent incorrect setting of alloc type
        assert(!((resource_type == MLX5DV_RES_TYPE_QP || resource_type == MLX5DV_RES_TYPE_SRQ) && peer->alloc_type != gds_peer::WQ));
        assert(!(resource_type == MLX5DV_RES_TYPE_CQ && peer->alloc_type != gds_peer::CQ));
        assert(!(resource_type == MLX5DV_RES_TYPE_DBR && peer->alloc_type != gds_peer::WQ && peer->alloc_type != gds_peer::CQ));

        if (peer->alloc_type == gds_peer::WQ)
                dir = GDS_PEER_DIRECTION_FROM_PEER | GDS_PEER_DIRECTION_TO_HCA;
        else if (peer->alloc_type == gds_peer::CQ)
                dir = GDS_PEER_DIRECTION_FROM_HCA | GDS_PEER_DIRECTION_TO_PEER | GDS_PEER_DIRECTION_TO_CPU;
        else {
                gds_dbg("encountered unsupported alloc_type\n");
                return IBV_ALLOCATOR_USE_DEFAULT;
        }

        if (resource_type == MLX5DV_RES_TYPE_QP || resource_type == MLX5DV_RES_TYPE_SRQ || resource_type == MLX5DV_RES_TYPE_DBR || resource_type == MLX5DV_RES_TYPE_CQ) {
                buf = peer->buf_alloc(peer->alloc_type, size, dir, (uint32_t)alignment, peer->alloc_flags);
        }
        else
                gds_dbg("request allocation with unsupported resource_type\n");

        if (!buf) {
                gds_dbg("alloc on host\n");
                return IBV_ALLOCATOR_USE_DEFAULT;
        }
        else {
                gds_dbg("alloc on GPU\n");
                ptr = buf->addr;
        }

        if ((range_id = peer_attr->register_va(ptr, size, peer_attr->peer_id, buf)) == 0) {
                gds_err("error in register_va\n");
                peer->free(buf);
                return IBV_ALLOCATOR_USE_DEFAULT;
        }

        // peer->opaque should be set
        assert(peer->opaque);

        if (peer->alloc_type == gds_peer::WQ) {
                gds_mlx5_dv_qp_peer_t *mqp_peer = (gds_mlx5_dv_qp_peer_t *)peer->opaque;
                if (resource_type == MLX5DV_RES_TYPE_QP || resource_type == MLX5DV_RES_TYPE_SRQ) {
                        // BUG: Will be overrided if use with IBV_QPT_RAW_PACKET or MLX5_QP_FLAGS_USE_UNDERLAY
                        mqp_peer->wq.va_id = range_id;
                        mqp_peer->wq.size = size;
                        mqp_peer->wq.gbuf = buf;
                }
                else if (resource_type == MLX5DV_RES_TYPE_DBR) {
                        mqp_peer->dbr.va_id = range_id;
                        mqp_peer->dbr.size = size;
                        mqp_peer->dbr.gbuf = buf;
                }
                else
                        gds_err("Unsupported resource_type\n");
        }
        else if (peer->alloc_type == gds_peer::CQ) {
                gds_mlx5_dv_cq_peer_t *mcq_peer = (gds_mlx5_dv_cq_peer_t *)peer->opaque;
                if (resource_type == MLX5DV_RES_TYPE_CQ) {
                        mcq_peer->buf.va_id = range_id;
                        mcq_peer->buf.size = size;
                        mcq_peer->buf.gbuf = buf;
                }
                else if (resource_type == MLX5DV_RES_TYPE_DBR) {
                        mcq_peer->dbr.va_id = range_id;
                        mcq_peer->dbr.size = size;
                        mcq_peer->dbr.gbuf = buf;
                }
                else
                        gds_err("Unsupported resource_type\n");
        }
        else
                gds_err("Unsupported peer->alloc_type\n");

        return ptr;
}

//-----------------------------------------------------------------------------

static void pd_mem_free(struct ibv_pd *pd, void *pd_context, void *ptr,
                        uint64_t resource_type)
{
        gds_dbg("pd_mem_free: pd=%p, pd_context=%p, ptr=%p, resource_type=0x%lx\n",
                pd, pd_context, ptr, resource_type);

        assert(pd_context);

        gds_peer_attr *peer_attr = (gds_peer_attr *)pd_context;
        gds_peer *peer = peer_from_id(peer_attr->peer_id);

        // Prevent incorrect setting of alloc type
        assert(!(resource_type == MLX5DV_RES_TYPE_QP && peer->alloc_type != gds_peer::WQ));
        assert(!(resource_type == MLX5DV_RES_TYPE_CQ && peer->alloc_type != gds_peer::CQ));
        assert(!(resource_type == MLX5DV_RES_TYPE_DBR && peer->alloc_type != gds_peer::WQ && peer->alloc_type != gds_peer::CQ));

        assert(peer->opaque);

        if (peer->alloc_type == gds_peer::WQ) {
                gds_mlx5_dv_qp_peer_t *mqp_peer = (gds_mlx5_dv_qp_peer_t *)peer->opaque;
                if (resource_type == MLX5DV_RES_TYPE_QP && mqp_peer->wq.gbuf) {
                        if (mqp_peer->wq.va_id) {
                                peer_attr->unregister_va(mqp_peer->wq.va_id, peer_attr->peer_id);
                                mqp_peer->wq.va_id = 0;
                        }
                        peer->free(mqp_peer->wq.gbuf);
                        mqp_peer->wq.gbuf = NULL;
                }
                else if (resource_type == MLX5DV_RES_TYPE_DBR && mqp_peer->dbr.gbuf) {
                        if (mqp_peer->dbr.va_id) {
                                peer_attr->unregister_va(mqp_peer->dbr.va_id, peer_attr->peer_id);
                                mqp_peer->dbr.va_id = 0;
                        }
                        peer->free(mqp_peer->dbr.gbuf);
                        mqp_peer->dbr.gbuf = NULL;
                }
        }
        else if (peer->alloc_type == gds_peer::CQ) {
                gds_mlx5_dv_cq_peer_t *mcq_peer = (gds_mlx5_dv_cq_peer_t *)peer->opaque;
                if (resource_type == MLX5DV_RES_TYPE_CQ && mcq_peer->buf.gbuf) {
                        if (mcq_peer->buf.va_id) {
                                peer_attr->unregister_va(mcq_peer->buf.va_id, peer_attr->peer_id);
                                mcq_peer->buf.va_id = 0;
                        }
                        peer->free(mcq_peer->buf.gbuf);
                        mcq_peer->buf.gbuf = NULL;
                }
                else if (resource_type == MLX5DV_RES_TYPE_DBR && mcq_peer->dbr.gbuf) {
                        if (mcq_peer->dbr.va_id) {
                                peer_attr->unregister_va(mcq_peer->dbr.va_id, peer_attr->peer_id);
                                mcq_peer->dbr.va_id = 0;
                        }
                        peer->free(mcq_peer->dbr.gbuf);
                        mcq_peer->dbr.gbuf = NULL;
                }
        }
}

//-----------------------------------------------------------------------------

static int gds_mlx5_dv_alloc_parent_domain(struct ibv_pd *p_pd, struct ibv_context *ibctx, gds_peer_attr *peer_attr, struct ibv_pd **out_pd)
{
        int ret = 0;

        struct ibv_parent_domain_init_attr pd_init_attr;
        struct ibv_pd *pd = NULL;
        gds_peer *peer = peer_from_id(peer_attr->peer_id);

        memset(&pd_init_attr, 0, sizeof(ibv_parent_domain_init_attr));
        pd_init_attr.pd = p_pd;
        pd_init_attr.comp_mask = IBV_PARENT_DOMAIN_INIT_ATTR_ALLOCATORS | IBV_PARENT_DOMAIN_INIT_ATTR_PD_CONTEXT;
        pd_init_attr.alloc = pd_mem_alloc;
        pd_init_attr.free = pd_mem_free;
        pd_init_attr.pd_context = peer_attr;

        pd = ibv_alloc_parent_domain(ibctx, &pd_init_attr);
        if (!pd) {
                gds_err("error in ibv_alloc_parent_domain\n");
                ret = EINVAL;
                goto out;
        }

        *out_pd = pd;

out:
        return ret;
}

//-----------------------------------------------------------------------------

int gds_mlx5_dv_create_qp(
        struct ibv_pd *pd, struct ibv_context *context, gds_qp_init_attr_t *qp_attr, 
        gds_peer *peer, gds_peer_attr *peer_attr, int flags, gds_qp_t **gqp
)
{
        int status = 0;

        gds_mlx5_dv_qp_t *mdqp = NULL;
        struct ibv_qp *ibqp = NULL;

        struct ibv_pd *parent_domain = NULL;

        gds_mlx5_dv_cq_t *tx_mcq = NULL;
        gds_mlx5_dv_cq_t *rx_mcq = NULL;

        uint32_t alignment;

        struct mlx5dv_devx_uar *uar = NULL;
        uint64_t uar_range_id = 0;
        uint8_t log_bf_reg_size;
        size_t bf_reg_size;

        int max_tx;
        int max_rx;

        size_t wqe_size;
        struct mlx5dv_devx_umem *wq_umem = NULL;
        size_t wq_buf_size;
        gds_buf *wq_buf = NULL;
        uint64_t wq_buf_range_id = 0;

        struct mlx5dv_devx_umem *dbr_umem = NULL;
        size_t dbr_buf_size;
        gds_buf *dbr_buf = NULL;
        uint64_t dbr_buf_range_id = 0;

        mlx5dv_obj dv_obj;
        struct mlx5dv_pd dvpd = {0,};
        uint64_t dv_obj_type = 0;

        uint8_t cmd_in[DEVX_ST_SZ_BYTES(create_qp_in)] = {0,};
        uint8_t cmd_out[DEVX_ST_SZ_BYTES(create_qp_out)] = {0,};

        uint8_t cmd_cap_in[DEVX_ST_SZ_BYTES(query_hca_cap_in)] = {0,};
        uint8_t cmd_cap_out[DEVX_ST_SZ_BYTES(query_hca_cap_out)] = {0,};

        void *qpc;

        gds_mlx5_dv_qp_peer_t *mqp_peer = NULL;

        struct mlx5dv_devx_obj *devx_obj = NULL;

        uint32_t qpn;
        uint32_t st_val;

        gds_mlx5_dv_qp_type_t gmlx_qpt = GDS_MLX5_DV_QP_TYPE_UNKNOWN;

        assert(pd);
        assert(context);
        assert(qp_attr);
        assert(peer);
        assert(peer_attr);
        
        if (qp_attr->qp_type == IBV_QPT_RC) {
                gmlx_qpt = GDS_MLX5_DV_QP_TYPE_RC;
                st_val = GDS_MLX5_DV_QPC_ST_RC;
        }
        else if (qp_attr->qp_type == IBV_QPT_UD) {
                gmlx_qpt = GDS_MLX5_DV_QP_TYPE_UD;
                st_val = GDS_MLX5_DV_QPC_ST_UD;
        }

        if (gmlx_qpt == GDS_MLX5_DV_QP_TYPE_UNKNOWN) {
                gds_err("The requested QP type is not supported.\n");
                status = EINVAL;
                goto out;
        }

        if (gmlx_qpt == GDS_MLX5_DV_QP_TYPE_RC) {
                if (qp_attr->cap.max_send_sge != 1 || qp_attr->cap.max_recv_sge != 1) {
                        gds_err("Both cap.max_send_sge and cap.max_recv_sge must be 1.\n");
                        status = EINVAL;
                        goto out;
                }
        }

        mdqp = (gds_mlx5_dv_qp_t *)calloc(1, sizeof(gds_mlx5_dv_qp_t));
        if (!mdqp) {
                gds_err("cannot allocate mdqp\n");
                status = ENOMEM;
                goto out;
        }

        ibqp = (struct ibv_qp *)calloc(1, sizeof(struct ibv_qp));
        if (!ibqp) {
                gds_err("cannot allocate ibqp\n");
                status = ENOMEM;
                goto out;
        }

        status = gds_mlx5_dv_alloc_parent_domain(pd, context, peer_attr, &parent_domain);
        if (status) {
                gds_err("Error in gds_mlx5_dv_alloc_parent_domain\n");
                goto out;
        }

        status = gds_mlx5_dv_create_cq(
                context, qp_attr->cap.max_send_wr, NULL, NULL, 0, parent_domain, peer_attr, 
                (gds_alloc_cq_flags_t)((flags & GDS_CREATE_QP_TX_CQ_ON_GPU) ? (GDS_ALLOC_CQ_ON_GPU | GDS_ALLOC_CQ_DBREC_ON_GPU) : (GDS_ALLOC_CQ_DEFAULT | GDS_ALLOC_CQ_DBREC_DEFAULT)),
                &tx_mcq
        );
        if (status) {
                gds_err("Error in creating tx cq\n");
                goto out;
        }

        status = gds_mlx5_dv_create_cq(
                context, qp_attr->cap.max_recv_wr, NULL, NULL, 0, parent_domain, peer_attr, 
                (gds_alloc_cq_flags_t)((flags & GDS_CREATE_QP_RX_CQ_ON_GPU) ? (GDS_ALLOC_CQ_ON_GPU | GDS_ALLOC_CQ_DBREC_ON_GPU) : (GDS_ALLOC_CQ_DEFAULT | GDS_ALLOC_CQ_DBREC_DEFAULT)),
                &rx_mcq
        );
        if (status) {
                gds_err("Error in creating rx cq\n");
                goto out;
        }

	DEVX_SET(query_hca_cap_in, cmd_cap_in, opcode, MLX5_CMD_OP_QUERY_HCA_CAP);
	DEVX_SET(query_hca_cap_in, cmd_cap_in, op_mod,
                MLX5_SET_HCA_CAP_OP_MOD_GENERAL_DEVICE |
                HCA_CAP_OPMOD_GET_CUR
        );

	status = mlx5dv_devx_general_cmd(context, cmd_cap_in, sizeof(cmd_cap_in), cmd_cap_out, sizeof(cmd_cap_out));
	if (status) {
		gds_err("Error in mlx5dv_devx_general_cmd for HCA CAP.\n");
                goto out;
	}

        log_bf_reg_size = DEVX_GET(query_hca_cap_out, cmd_cap_out, capability.cmd_hca_cap.log_bf_reg_size);

        // The size of 1st + 2nd half (as when we use alternating DB)
        bf_reg_size = 1LLU << log_bf_reg_size;

        // Allocate UAR. This will be used as a DB/BF register).
        uar = mlx5dv_devx_alloc_uar(context, GDS_MLX5_DV_UAR_ALLOC_TYPE_BF);
        if (!uar) {
                gds_err("Error in mlx5dv_devx_uar\n");
                status = ENOMEM;
                goto out;
        }

        uar_range_id = peer_attr->register_va(
                uar->reg_addr,
                bf_reg_size,
                peer_attr->peer_id,
                GDS_PEER_IOMEMORY
        );
        if (!uar_range_id) {
                gds_err("Error in peer_attr->register_va for BF\n");
                status = EINVAL;
                goto out;
        }

        // In GPUVerbs, we use at most 4 16-byte elements.
        wqe_size = MLX5_SEND_WQE_BB;      // 64 bytes
        max_tx = GDS_ROUND_UP_POW2_OR_0(qp_attr->cap.max_send_wr);
        max_rx = GDS_ROUND_UP_POW2_OR_0(qp_attr->cap.max_recv_wr);
        wq_buf_size = (max_tx + max_rx) * wqe_size;

        // Allocate WQ buffer.
        alignment = (uint32_t)((flags & GDS_ALLOC_WQ_ON_GPU) ? GDS_GPU_PAGE_SIZE : sysconf(_SC_PAGESIZE));
        wq_buf = peer->alloc(wq_buf_size, alignment, (flags & GDS_ALLOC_WQ_ON_GPU) ? GDS_MEMORY_GPU : GDS_MEMORY_HOST);
        if (!wq_buf) {
                gds_err("Error in peer->alloc of wq_buf.\n");
                status = ENOMEM;
                goto out;
        }

        wq_umem = mlx5dv_devx_umem_reg(context, wq_buf->addr, wq_buf_size, IBV_ACCESS_LOCAL_WRITE);
        if (!wq_umem) {
                gds_err("Error in mlx5dv_devx_umem_reg for WQ\n");
                status = ENOMEM;
                goto out;
        }
        
        wq_buf_range_id = peer_attr->register_va(wq_buf->addr, wq_buf_size, peer_attr->peer_id, wq_buf);
        if (!wq_buf_range_id) {
                gds_err("Error in peer_attr->register_va for WQ\n");
                status = ENOMEM;
                goto out;
        }

        // Allocate DBR buffer.
        alignment = (uint32_t)((flags & GDS_ALLOC_WQ_DBREC_ON_GPU) ? GDS_GPU_PAGE_SIZE : sysconf(_SC_PAGESIZE));
        dbr_buf_size = GDS_MLX5_DV_DBR_BUF_SIZE;

        dbr_buf = peer->alloc(dbr_buf_size, alignment, (flags & GDS_ALLOC_WQ_DBREC_ON_GPU) ? GDS_MEMORY_GPU : GDS_MEMORY_HOST);
        if (!dbr_buf) {
                gds_err("Error in peer->alloc of dbr_buf.\n");
                status = ENOMEM;
                goto out;
        }

        dbr_umem = mlx5dv_devx_umem_reg(context, dbr_buf->addr, dbr_buf_size, IBV_ACCESS_LOCAL_WRITE);
        if (!dbr_umem) {
                gds_err("Error in mlx5dv_devx_umem_reg for DBR\n");
                status = ENOMEM;
                goto out;
        }
        
        dbr_buf_range_id = peer_attr->register_va(dbr_buf->addr, dbr_buf_size, peer_attr->peer_id, dbr_buf);
        if (!dbr_buf_range_id) {
                gds_err("Error in peer_attr->register_va for DBR\n");
                status = ENOMEM;
                goto out;
        }

        // Query more PD info with Direct-Verbs.
        dv_obj.pd.in = pd;
        dv_obj.pd.out = &dvpd;
        dv_obj_type = MLX5DV_OBJ_PD;
        status = mlx5dv_init_obj(&dv_obj, dv_obj_type);
        if (status) {
                gds_err("Error in mlx5dv_init_obj\n");
                goto out;
        }

        DEVX_SET(create_qp_in, cmd_in, opcode, MLX5_CMD_OP_CREATE_QP);
        DEVX_SET(create_qp_in, cmd_in, wq_umem_id, wq_umem->umem_id);   // WQ buffer

        qpc = DEVX_ADDR_OF(create_qp_in, cmd_in, qpc);
        DEVX_SET(qpc, qpc, st, st_val);
        DEVX_SET(qpc, qpc, pm_state, MLX5_QPC_PM_STATE_MIGRATED);
        DEVX_SET(qpc, qpc, pd, dvpd.pdn);
        DEVX_SET(qpc, qpc, uar_page, uar->page_id);     // BF register
        DEVX_SET(qpc, qpc, rq_type, GDS_MLX5_DV_QPC_RQ_TYPE_REGULAR);
        DEVX_SET(qpc, qpc, srqn_rmpn_xrqn, 0);
        DEVX_SET(qpc, qpc, cqn_snd, tx_mcq->dvcq.cqn);
        DEVX_SET(qpc, qpc, cqn_rcv, rx_mcq->dvcq.cqn);
        DEVX_SET(qpc, qpc, log_sq_size, GDS_ILOG2_OR0(max_tx));
        if (gmlx_qpt == GDS_MLX5_DV_QP_TYPE_RC)
                DEVX_SET(qpc, qpc, cs_req, 0);  // Disable CS Request
        DEVX_SET(qpc, qpc, cs_res, 0);  // Disable CS Respond
        if (gmlx_qpt == GDS_MLX5_DV_QP_TYPE_UD)
                DEVX_SET(qpc, qpc, cgs, 0);     // GRH is always scattered to the beginning of the receive buffer.
        DEVX_SET(qpc, qpc, dbr_umem_valid, 0x1); // Enable dbr_umem_id
        DEVX_SET64(qpc, qpc, dbr_addr, 0); // Offset 0 of dbr_umem_id (behavior changed because of dbr_umem_valid)
        DEVX_SET(qpc, qpc, dbr_umem_id, dbr_umem->umem_id); // DBR buffer
        DEVX_SET(qpc, qpc, user_index, 0);
        DEVX_SET(qpc, qpc, page_offset, 0);
        DEVX_SET(qpc, qpc, log_rq_size, GDS_ILOG2_OR0(max_rx));

        devx_obj = mlx5dv_devx_obj_create(context, cmd_in, sizeof(cmd_in), cmd_out, sizeof(cmd_out));
        if (!devx_obj) {
                gds_err("Error in mlx5dv_devx_obj_create for qp\n");
                status = EIO;
                goto out;
        }

        qpn = DEVX_GET(create_qp_out, cmd_out, qpn);

        mdqp->devx_qp = devx_obj;
        mdqp->qp_type = gmlx_qpt;

        mdqp->qp_peer = mqp_peer;

        mdqp->wq_buf = wq_buf;
        mdqp->wq_umem = wq_umem;
        mdqp->wq_va_id = wq_buf_range_id;

        mdqp->dbr_buf = dbr_buf;
        mdqp->dbr_umem = dbr_umem;
        mdqp->dbr_va_id = dbr_buf_range_id;

        mdqp->bf_uar = uar;
        mdqp->bf_size = bf_reg_size / 2;
        mdqp->bf_va_id = uar_range_id;

        mdqp->sq_cnt = max_tx;
        mdqp->rq_cnt = max_rx;

        mdqp->rq_buf_offset = 0;
        // Assume 1 recv sge if RC and no wq_sig
        mdqp->sq_buf_offset = (gmlx_qpt == GDS_MLX5_DV_QP_TYPE_RC) ? MAX(max_rx * sizeof(struct mlx5_wqe_data_seg), GDS_MLX5_DV_SEND_WQE_BB) : 0;

        mdqp->peer_attr = peer_attr;

        mdqp->gqp.send_cq = &tx_mcq->gcq;
        mdqp->gqp.recv_cq = &rx_mcq->gcq;

        mdqp->parent_domain = parent_domain;

        ibqp->context = context;
        ibqp->pd = pd;
        ibqp->send_cq = tx_mcq->gcq.cq;
        ibqp->recv_cq = rx_mcq->gcq.cq;
        ibqp->qp_num = qpn;
        ibqp->state = IBV_QPS_RESET;
        ibqp->qp_type = qp_attr->qp_type;

        mdqp->gqp.qp = ibqp;

        *gqp = &mdqp->gqp;

out:
        // Failed. Cleaning up.
        if (status) {
                if (devx_obj)
                        mlx5dv_devx_obj_destroy(devx_obj);
                
                if (dbr_buf_range_id)
                        peer_attr->unregister_va(dbr_buf_range_id, peer_attr->peer_id);

                if (dbr_umem)
                        mlx5dv_devx_umem_dereg(dbr_umem);

                if (dbr_buf)
                        peer->free(dbr_buf);

                if (wq_buf_range_id)
                        peer_attr->unregister_va(wq_buf_range_id, peer_attr->peer_id);

                if (wq_umem)
                        mlx5dv_devx_umem_dereg(wq_umem);

                if (wq_buf)
                        peer->free(wq_buf);

                if (uar_range_id)
                        peer_attr->unregister_va(uar_range_id, peer_attr->peer_id);

                if (uar)
                        mlx5dv_devx_free_uar(uar);

                if (mqp_peer)
                        free(mqp_peer);

                if (rx_mcq)
                        gds_mlx5_dv_destroy_cq(rx_mcq);

                if (tx_mcq)
                        gds_mlx5_dv_destroy_cq(tx_mcq);

                if (parent_domain)
                        ibv_dealloc_pd(parent_domain);

                if (ibqp)
                        free(ibqp);

                if (mdqp)
                        free(mdqp);
        }
        return status;
}

//-----------------------------------------------------------------------------

static int gds_mlx5_dv_modify_qp_rst2init(gds_mlx5_dv_qp_t *mdqp, struct ibv_qp_attr *attr, int attr_mask)
{
        int status = 0;

        uint8_t cmd_in[DEVX_ST_SZ_BYTES(rst2init_qp_in)] = {0,};
        uint8_t cmd_out[DEVX_ST_SZ_BYTES(rst2init_qp_out)] = {0,};

        void *qpc;

        assert(attr->qp_state == IBV_QPS_INIT);
        if (mdqp->gqp.qp->state != IBV_QPS_RESET) {
                gds_err("Incorrect current QP state.\n");
                status = EINVAL;
                goto out;
        }

        if (!(attr_mask & IBV_QP_PORT)) {
                gds_err("IBV_QP_PORT is required.\n");
                status = EINVAL;
                goto out;
        }

        if (!(attr_mask & IBV_QP_PKEY_INDEX)) {
                gds_err("IBV_QP_PKEY_INDEX is required.\n");
                status = EINVAL;
                goto out;
        }

        status = ibv_query_port(mdqp->gqp.qp->context, attr->port_num, &mdqp->port_attr);
        if (status) {
                gds_err("Error in ibv_query_port port_num=%d\n", attr->port_num);
                goto out;
        }

        DEVX_SET(rst2init_qp_in, cmd_in, opcode, MLX5_CMD_OP_RST2INIT_QP);
        DEVX_SET(rst2init_qp_in, cmd_in, qpn, mdqp->gqp.qp->qp_num);

        qpc = DEVX_ADDR_OF(rst2init_qp_in, cmd_in, qpc);
        DEVX_SET(qpc, qpc, pm_state, MLX5_QPC_PM_STATE_MIGRATED);
        DEVX_SET(qpc, qpc, primary_address_path.vhca_port_num, attr->port_num);
        DEVX_SET(qpc, qpc, primary_address_path.pkey_index, attr->pkey_index);

        if (attr_mask & IBV_QP_ACCESS_FLAGS) {
                DEVX_SET(qpc, qpc, rwe, !!(attr->qp_access_flags & IBV_ACCESS_REMOTE_WRITE));
                DEVX_SET(qpc, qpc, rre, !!(attr->qp_access_flags & IBV_ACCESS_REMOTE_READ));
                DEVX_SET(qpc, qpc, rae, !!(attr->qp_access_flags & IBV_ACCESS_REMOTE_ATOMIC));
                if (attr->qp_access_flags & IBV_ACCESS_REMOTE_ATOMIC) {
                        DEVX_SET(qpc, qpc, atomic_mode, GDS_MLX5_DV_ATOMIC_MODE); 
                        DEVX_SET(qpc, qpc, atomic_like_write_en, GDS_MLX5_DV_ATOMIC_LIKE_WRITE_EN);
                }
        }

        DEVX_SET(qpc, qpc, wq_signature, GDS_MLX5_DV_WQ_SIGNATURE);
        DEVX_SET(qpc, qpc, counter_set_id, GDS_MLX5_DV_COUNTER_SET_ID);
        DEVX_SET(qpc, qpc, lag_tx_port_affinity, GDS_MLX5_DV_LAG_TX_PORT_AFFINITY);

        status = mlx5dv_devx_obj_modify(mdqp->devx_qp, cmd_in, sizeof(cmd_in), cmd_out, sizeof(cmd_out));
        if (status) {
                gds_err("Error in mlx5dv_devx_obj_modify for RST2INIT_QP with syndrome %x\n", DEVX_GET(rst2init_qp_out, cmd_out, syndrome));
                goto out;
        }

        mdqp->gqp.qp->state = IBV_QPS_INIT;
        mdqp->port_num = attr->port_num;

out:
        return status;
}

//-----------------------------------------------------------------------------

static int gds_mlx5_dv_modify_qp_init2rtr(gds_mlx5_dv_qp_t *mdqp, struct ibv_qp_attr *attr, int attr_mask)
{
        int status = 0;

        uint8_t cmd_in[DEVX_ST_SZ_BYTES(init2rtr_qp_in)] = {0,};
        uint8_t cmd_out[DEVX_ST_SZ_BYTES(init2rtr_qp_out)] = {0,};

        void *qpc;

        assert(attr->qp_state == IBV_QPS_RTR);
        if (mdqp->gqp.qp->state != IBV_QPS_INIT) {
                gds_err("Incorrect current QP state.\n");
                status = EINVAL;
                goto out;
        }

        if (mdqp->qp_type == GDS_MLX5_DV_QP_TYPE_RC) {
                if (!(attr_mask & IBV_QP_DEST_QPN)) {
                        gds_err("IBV_QP_DEST_QPN is required.\n");
                        status = EINVAL;
                        goto out;
                }
        }

        if (!(attr_mask & IBV_QP_PATH_MTU)) {
                gds_err("IBV_QP_PATH_MTU is required.\n");
                status = EINVAL;
                goto out;
        }

        if (!(attr_mask & IBV_QP_AV)) {
                gds_err("IBV_QP_AV is required.\n");
                status = EINVAL;
                goto out;
        }

        if (mdqp->port_attr.link_layer != IBV_LINK_LAYER_INFINIBAND) {
                gds_err("We support infiniband link layer only.\n");
                status = ENOTSUP;
                goto out;
        }

        if (attr->ah_attr.is_global && attr->ah_attr.grh.flow_label != 0) {
                gds_err("Flow label is not supported.\n");
                status = ENOTSUP;
                goto out;
        }

        DEVX_SET(init2rtr_qp_in, cmd_in, opcode, MLX5_CMD_OP_INIT2RTR_QP);
        DEVX_SET(init2rtr_qp_in, cmd_in, op_mod, 0x0);   // Request INIT2RTR
        DEVX_SET(init2rtr_qp_in, cmd_in, opt_param_mask, 0x0);  // Don't pass optional params
        DEVX_SET(init2rtr_qp_in, cmd_in, qpn, mdqp->gqp.qp->qp_num);

        qpc = DEVX_ADDR_OF(init2rtr_qp_in, cmd_in, qpc);
        DEVX_SET(qpc, qpc, mtu, attr->path_mtu);
        DEVX_SET(qpc, qpc, log_msg_max, GDS_MLX5_DV_LOG_MAX_MSG_SIZE);

        if (attr_mask & IBV_QP_DEST_QPN)
                DEVX_SET(qpc, qpc, remote_qpn, attr->dest_qp_num);

        DEVX_SET(qpc, qpc, primary_address_path.grh, attr->ah_attr.is_global);
        DEVX_SET(qpc, qpc, primary_address_path.rlid, attr->ah_attr.dlid);
        DEVX_SET(qpc, qpc, primary_address_path.mlid, attr->ah_attr.src_path_bits & 0x7f);
        DEVX_SET(qpc, qpc, primary_address_path.sl, attr->ah_attr.sl);

        if (attr->ah_attr.is_global) {
                DEVX_SET(qpc, qpc, primary_address_path.hop_limit, attr->ah_attr.grh.hop_limit);
                DEVX_SET(qpc, qpc, primary_address_path.rgid_rip, attr->ah_attr.grh.hop_limit);
                memcpy(
                        DEVX_ADDR_OF(qpc, qpc, primary_address_path.rgid_rip),
                        &attr->ah_attr.grh.dgid,
                        DEVX_FLD_SZ_BYTES(qpc, primary_address_path.rgid_rip)
                );
                DEVX_SET(qpc, qpc, primary_address_path.tclass, attr->ah_attr.grh.traffic_class);
        }

        if (attr_mask & IBV_QP_MAX_DEST_RD_ATOMIC)
                DEVX_SET(qpc, qpc, log_rra_max, GDS_ILOG2_OR0(attr->max_dest_rd_atomic));

        if (attr_mask & IBV_QP_MIN_RNR_TIMER)
                DEVX_SET(qpc, qpc, min_rnr_nak, attr->min_rnr_timer);

        status = mlx5dv_devx_obj_modify(mdqp->devx_qp, cmd_in, sizeof(cmd_in), cmd_out, sizeof(cmd_out));
        if (status) {
                gds_err("Error in mlx5dv_devx_obj_modify for INIT2RTR_QP with syndrome %x\n", DEVX_GET(init2rtr_qp_out, cmd_out, syndrome));
                goto out;
        }

        mdqp->sl = attr->ah_attr.sl;

        mdqp->gqp.qp->state = IBV_QPS_RTR;

out:
        return status;
}

//-----------------------------------------------------------------------------

static int gds_mlx5_dv_modify_qp_rtr2rts(gds_mlx5_dv_qp_t *mdqp, struct ibv_qp_attr *attr, int attr_mask)
{
        int status = 0;

        uint8_t cmd_in[DEVX_ST_SZ_BYTES(rtr2rts_qp_in)] = {0,};
        uint8_t cmd_out[DEVX_ST_SZ_BYTES(rtr2rts_qp_out)] = {0,};

        void *qpc;

        assert(attr->qp_state == IBV_QPS_RTS);
        if (mdqp->gqp.qp->state != IBV_QPS_RTR) {
                gds_err("Incorrect current QP state.\n");
                status = EINVAL;
                goto out;
        }

        if (!(attr_mask & IBV_QP_MAX_QP_RD_ATOMIC)) {
                gds_err("IBV_QP_MAX_QP_RD_ATOMIC is required.\n");
                status = EINVAL;
                goto out;
        }

        if (!(attr_mask & IBV_QP_RETRY_CNT)) {
                gds_err("IBV_QP_RETRY_CNT is required.\n");
                status = EINVAL;
                goto out;
        }

        if (!(attr_mask & IBV_QP_RNR_RETRY)) {
                gds_err("IBV_QP_RNR_RETRY is required.\n");
                status = EINVAL;
                goto out;
        }

        if (!(attr_mask & IBV_QP_TIMEOUT)) {
                gds_err("IBV_QP_TIMEOUT is required.\n");
                status = EINVAL;
                goto out;
        }

        if (!(attr_mask & IBV_QP_SQ_PSN)) {
                gds_err("IBV_QP_SQ_PSN is required.\n");
                status = EINVAL;
                goto out;
        }

        DEVX_SET(rtr2rts_qp_in, cmd_in, opcode, MLX5_CMD_OP_RTR2RTS_QP);
        DEVX_SET(rtr2rts_qp_in, cmd_in, opt_param_mask, 0x0);  // Don't pass optional params
        DEVX_SET(rtr2rts_qp_in, cmd_in, qpn, mdqp->gqp.qp->qp_num);

        qpc = DEVX_ADDR_OF(rtr2rts_qp_in, cmd_in, qpc);
        DEVX_SET(qpc, qpc, log_sra_max, GDS_ILOG2_OR0(attr->max_rd_atomic));
        DEVX_SET(qpc, qpc, retry_count, attr->retry_cnt);
        DEVX_SET(qpc, qpc, rnr_retry, attr->rnr_retry);
        DEVX_SET(qpc, qpc, next_send_psn, attr->sq_psn);
        DEVX_SET(qpc, qpc, log_ack_req_freq, GDS_MLX5_DV_LOG_ACK_REQ_FREQ);
        DEVX_SET(qpc, qpc, primary_address_path.ack_timeout, attr->timeout);

        status = mlx5dv_devx_obj_modify(mdqp->devx_qp, cmd_in, sizeof(cmd_in), cmd_out, sizeof(cmd_out));
        if (status) {
                gds_err("Error in mlx5dv_devx_obj_modify for RTR2RTS_QP with syndrome %x\n", DEVX_GET(rtr2rts_qp_out, cmd_out, syndrome));
                goto out;
        }

        mdqp->gqp.qp->state = IBV_QPS_RTS;

out:
        return status;
}

//-----------------------------------------------------------------------------

int gds_mlx5_dv_modify_qp(gds_qp_t *gqp, struct ibv_qp_attr *attr, int attr_mask)
{
        int status = 0;

        gds_mlx5_dv_qp_t *mdqp;

        assert(gqp);
        assert(attr);

        mdqp = to_gds_mdv_qp(gqp);

        assert(mdqp->gqp.qp);

        if (!(attr_mask & IBV_QP_STATE)) {
                gds_err("IBV_QP_STATE is required.\n");
                status = EINVAL;
                goto out;
        }

        switch (attr->qp_state) {
                case IBV_QPS_INIT:
                        status = gds_mlx5_dv_modify_qp_rst2init(mdqp, attr, attr_mask);
                        break;
                case IBV_QPS_RTR:
                        status = gds_mlx5_dv_modify_qp_init2rtr(mdqp, attr, attr_mask);
                        break;
                case IBV_QPS_RTS:
                        status = gds_mlx5_dv_modify_qp_rtr2rts(mdqp, attr, attr_mask);
                        break;
                default:
                        gds_err("Encountered unsupported qp_state.\n");
                        status = EINVAL;
        }

out:
        return status;
}

//-----------------------------------------------------------------------------

int gds_mlx5_dv_destroy_qp(gds_qp_t *gqp)
{
        int status = 0;

        gds_mlx5_dv_qp_t *mdqp;

        gds_peer *peer = NULL;
        gds_mlx5_dv_qp_peer_t *mqp_peer;

        if (!gqp)
                return status;

        mdqp = to_gds_mdv_qp(gqp);

        assert(mdqp->devx_qp);

        mqp_peer = mdqp->qp_peer;

        if (mqp_peer) {
                gds_peer_attr *peer_attr = mqp_peer->peer_attr;
                gds_peer *peer = peer_from_id(peer_attr->peer_id);

                // This may be used by ibv_destroy_qp, which eventually calls pd_mem_free.
                peer->alloc_type = gds_peer::WQ;
                peer->opaque = mqp_peer;
        }

        status = mlx5dv_devx_obj_destroy(mdqp->devx_qp);
        if (status)
                gds_err("Error in mlx5dv_devx_obj_destroy for QP.\n");

        if (mdqp->gqp.send_cq) {
                gds_mlx5_dv_destroy_cq(to_gds_mdv_cq(mdqp->gqp.send_cq));
                mdqp->gqp.send_cq = NULL;
        }

        if (mdqp->gqp.recv_cq) {
                gds_mlx5_dv_destroy_cq(to_gds_mdv_cq(mdqp->gqp.recv_cq));
                mdqp->gqp.recv_cq = NULL;
        }

        if (mdqp->dbr_umem) {
                status = mlx5dv_devx_umem_dereg(mdqp->dbr_umem);
                if (status)
                        gds_err("Error in mlx5dv_devx_umem_dereg of dbr_umem.\n");
        }

        if (mdqp->wq_umem) {
                status = mlx5dv_devx_umem_dereg(mdqp->wq_umem);
                if (status)
                        gds_err("Error in mlx5dv_devx_umem_dereg of wq_umem.\n");
        }

        if (mdqp->dbr_buf || mdqp->wq_buf || mdqp->bf_va_id) {
                assert(mdqp->peer_attr);

                peer = peer_from_id(mdqp->peer_attr->peer_id);

                if (mdqp->bf_va_id)
                        mdqp->peer_attr->unregister_va(mdqp->bf_va_id, mdqp->peer_attr->peer_id);

                if (mdqp->dbr_buf) {
                        mdqp->peer_attr->unregister_va(mdqp->dbr_va_id, mdqp->peer_attr->peer_id);
                        peer->free(mdqp->dbr_buf);
                }

                if (mdqp->wq_buf) {
                        mdqp->peer_attr->unregister_va(mdqp->wq_va_id, mdqp->peer_attr->peer_id);
                        peer->free(mdqp->wq_buf);
                }
        }

        if (mdqp->bf_uar)
                mlx5dv_devx_free_uar(mdqp->bf_uar);

        if (mdqp->parent_domain)
                ibv_dealloc_pd(mdqp->parent_domain);

        if (mqp_peer)
                free(mqp_peer);

        if (mdqp->gqp.qp)
                free(mdqp->gqp.qp);
        
        free(mdqp);
}

//-----------------------------------------------------------------------------

int gds_transport_mlx5_dv_init(gds_transport_t **transport)
{
        int status = 0;

        gds_transport_t *t = (gds_transport_t *)calloc(1, sizeof(gds_transport_t));
        if (!t) {
                status = ENOMEM;
                goto out;
        }

        t->create_qp = gds_mlx5_dv_create_qp;
        t->destroy_qp = gds_mlx5_dv_destroy_qp;
        t->modify_qp = gds_mlx5_dv_modify_qp;
        #if 0
        t->rollback_qp = gds_mlx5_exp_rollback_qp;

        t->init_send_info = gds_mlx5_exp_init_send_info;
        t->post_send_ops = gds_mlx5_exp_post_send_ops;
        t->post_send_ops_on_cpu = gds_mlx5_exp_post_send_ops_on_cpu;
        t->prepare_send = gds_mlx5_exp_prepare_send;
        t->get_send_descs = gds_mlx5_exp_get_send_descs;
        t->get_num_send_request_entries = gds_mlx5_exp_get_num_send_request_entries;

        t->init_wait_request = gds_mlx5_exp_init_wait_request;
        t->dump_wait_request = gds_mlx5_exp_dump_wait_request;
        t->stream_post_wait_descriptor = gds_mlx5_exp_stream_post_wait_descriptor;
        t->post_wait_descriptor = gds_mlx5_exp_post_wait_descriptor;
        t->get_wait_descs = gds_mlx5_exp_get_wait_descs;
        t->get_num_wait_request_entries = gds_mlx5_exp_get_num_wait_request_entries;

        t->prepare_wait_cq = gds_mlx5_exp_prepare_wait_cq;
        t->append_wait_cq = gds_mlx5_exp_append_wait_cq;
        t->abort_wait_cq = gds_mlx5_exp_abort_wait_cq;
        #endif

        *transport = t;

out:
        return status;
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