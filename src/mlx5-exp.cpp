#include <string.h>
#include <infiniband/verbs.h>
#include <infiniband/verbs_exp.h>

#include "mlx5-exp.hpp"
#include "utils.hpp"

static ibv_exp_res_domain *gds_mlx5_exp_create_res_domain(struct ibv_context *context)
{
        if (!context) {
                gds_err("invalid context");
                return NULL;
        }

        ibv_exp_res_domain_init_attr res_domain_attr;
        memset(&res_domain_attr, 0, sizeof(res_domain_attr));

        res_domain_attr.comp_mask |= IBV_EXP_RES_DOMAIN_THREAD_MODEL;
        res_domain_attr.thread_model = IBV_EXP_THREAD_SINGLE;

        ibv_exp_res_domain *res_domain = ibv_exp_create_res_domain(context, &res_domain_attr);
        if (!res_domain) {
                gds_warn("Can't create resource domain\n");
        }

        return res_domain;
}

//-----------------------------------------------------------------------------

gds_mlx5_exp_cq_t *gds_mlx5_exp_create_cq(
        struct ibv_context *context, int cqe,
        void *cq_context, struct ibv_comp_channel *channel,
        int comp_vector, gds_peer *peer, gds_peer_attr *peer_attr, gds_alloc_cq_flags_t flags,
        struct ibv_exp_res_domain *res_domain)
{
        gds_mlx5_exp_cq_t *gmexpcq = NULL;
        ibv_exp_cq_init_attr attr;
        int ret = 0;

        assert(context);
        assert(peer);
        assert(peer_attr);

        gmexpcq = (gds_mlx5_exp_cq_t *)calloc(1, sizeof(gds_mlx5_exp_cq_t));
        if (!gmexpcq) {
            gds_err("cannot allocate memory\n");
            return NULL;
        }

        peer->alloc_type = gds_peer::CQ;
        peer->alloc_flags = flags;

        attr.comp_mask = IBV_EXP_CQ_INIT_ATTR_PEER_DIRECT;
        attr.flags = 0; // see ibv_exp_cq_create_flags
        static_assert(sizeof(gds_peer_attr) == sizeof(struct ibv_exp_peer_direct_attr));
        attr.peer_direct_attrs = (struct ibv_exp_peer_direct_attr *)(peer_attr);
        if (res_domain) {
            gds_dbg("using peer->res_domain %p for CQ\n", res_domain);
            attr.res_domain = res_domain;
            attr.comp_mask |= IBV_EXP_CQ_INIT_ATTR_RES_DOMAIN;
            gmexpcq->res_domain = res_domain;
        }
        
        int old_errno = errno;
        gmexpcq->gcq.cq = ibv_exp_create_cq(context, cqe, cq_context, channel, comp_vector, &attr);
        if (!gmexpcq->gcq.cq) {
            gds_err("error %d in ibv_exp_create_cq, old errno %d\n", errno, old_errno);
            return NULL;
        }

        gmexpcq->gcq.dtype = GDS_DRIVER_TYPE_MLX5_EXP;

        return gmexpcq;
}

//-----------------------------------------------------------------------------

gds_mlx5_exp_qp_t *gds_mlx5_exp_create_qp(
        struct ibv_pd *pd, struct ibv_context *context, gds_qp_init_attr_t *qp_attr, 
        gds_peer *peer, gds_peer_attr *peer_attr, int flags)
{
        int ret = 0;
        gds_mlx5_exp_qp_t *gmexpqp = NULL;
        struct ibv_qp *qp = NULL;
        gds_mlx5_exp_cq_t *rx_gmexpcq = NULL, *tx_gmexpcq = NULL;
        struct ibv_exp_qp_init_attr exp_qp_attr = {0,};
        int old_errno = errno;

        assert(pd);
        assert(context);
        assert(qp_attr);
        assert(peer);
        assert(peer_attr);

        gmexpqp = (gds_mlx5_exp_qp_t *)calloc(1, sizeof(gds_mlx5_exp_qp_t));
        if (!gmexpqp) {
            gds_err("cannot allocate memory\n");
            return NULL;
        }
        gmexpqp->gqp.dtype = GDS_DRIVER_TYPE_MLX5_EXP;

        gmexpqp->gqp.dev_context = context;

        gmexpqp->res_domain = gds_mlx5_exp_create_res_domain(context);
        if (gmexpqp->res_domain)
            gds_dbg("using res_domain %p\n", gmexpqp->res_domain);
        else
            gds_warn("NOT using res_domain\n");

        tx_gmexpcq = gds_mlx5_exp_create_cq(
                context, qp_attr->cap.max_send_wr, NULL, NULL, 0, peer, peer_attr,
                (flags & GDS_CREATE_QP_TX_CQ_ON_GPU) ? GDS_ALLOC_CQ_ON_GPU : GDS_ALLOC_CQ_DEFAULT, 
                gmexpqp->res_domain
        );
        if (!tx_gmexpcq) {
                ret = errno;
                gds_err("error %d while creating TX CQ, old_errno=%d\n", ret, old_errno);
                goto err;
        }

        rx_gmexpcq = gds_mlx5_exp_create_cq(
                context, qp_attr->cap.max_recv_wr, NULL, NULL, 0, peer, peer_attr,
                (flags & GDS_CREATE_QP_RX_CQ_ON_GPU) ? GDS_ALLOC_CQ_ON_GPU : GDS_ALLOC_CQ_DEFAULT, 
                gmexpqp->res_domain
        );
        if (!rx_gmexpcq) {
                ret = errno;
                gds_err("error %d while creating RX CQ\n", ret);
                goto err;
        }
        
        // peer registration
        peer->alloc_type = gds_peer::WQ;
        peer->alloc_flags = GDS_ALLOC_WQ_DEFAULT | GDS_ALLOC_DBREC_DEFAULT;
        if (flags & GDS_CREATE_QP_WQ_ON_GPU) {
                gds_err("error, QP WQ on GPU is not supported yet\n");
                goto err;
        }
        if (flags & GDS_CREATE_QP_WQ_DBREC_ON_GPU) {
                gds_warn("QP WQ DBREC on GPU\n");
                peer->alloc_flags |= GDS_ALLOC_DBREC_ON_GPU;
        }        

        exp_qp_attr.send_cq = tx_gmexpcq->gcq.cq;
        exp_qp_attr.recv_cq = rx_gmexpcq->gcq.cq;
        exp_qp_attr.pd = pd;
        exp_qp_attr.comp_mask = IBV_EXP_QP_INIT_ATTR_PD | IBV_EXP_QP_INIT_ATTR_PEER_DIRECT;
        static_assert(sizeof(gds_peer_attr) == sizeof(struct ibv_exp_peer_direct_attr));
        exp_qp_attr.peer_direct_attrs = (struct ibv_exp_peer_direct_attr *)peer_attr;
        exp_qp_attr.qp_type = qp_attr->qp_type;

        assert(sizeof(exp_qp_attr.cap) == sizeof(qp_attr->cap));

        memcpy(&exp_qp_attr.cap, &qp_attr->cap, sizeof(qp_attr->cap));

        qp = ibv_exp_create_qp(context, &exp_qp_attr);
        if (!qp) {
                ret = EINVAL;
                gds_err("error in ibv_exp_create_qp\n");
                goto err;
        }

        tx_gmexpcq->gcq.cq = qp->send_cq;
        rx_gmexpcq->gcq.cq = qp->recv_cq;

        gmexpqp->gqp.qp = qp;
        gmexpqp->gqp.send_cq = &tx_gmexpcq->gcq;
        gmexpqp->gqp.recv_cq = &rx_gmexpcq->gcq;

        gds_dbg("created gds_mlx5_exp_qp=%p\n", gmexpqp);

        return gmexpqp;

err:
        gds_dbg("destroying QP\n");
        gds_mlx5_exp_destroy_qp(gmexpqp);

        return NULL;
}

//-----------------------------------------------------------------------------

int gds_mlx5_exp_destroy_qp(gds_mlx5_exp_qp_t *gmexpqp)
{
        int retcode = 0;
        int ret;
        
        if (!gmexpqp) 
                return retcode;
        
        assert(gmexpqp->gqp.dtype == GDS_DRIVER_TYPE_MLX5_EXP);

        if (gmexpqp->gqp.qp) {
                ret = ibv_destroy_qp(gmexpqp->gqp.qp);
                if (ret) {
                        gds_err("error %d in destroy_qp\n", ret);
                        retcode = ret;
                }            
        }

        if (gmexpqp->gqp.send_cq) {
                ret = gds_destroy_cq(gmexpqp->gqp.send_cq);
                if (ret) {
                        gds_err("error %d in destroy_cq send_cq\n", ret);
                        retcode = ret;
                }
        }

        if (gmexpqp->gqp.recv_cq) {
                ret = gds_destroy_cq(gmexpqp->gqp.recv_cq);
                if (ret) {
                        gds_err("error %d in destroy_cq recv_cq\n", ret);
                        retcode = ret;
                }
        }

        if (gmexpqp->res_domain) {
            struct ibv_exp_destroy_res_domain_attr attr = {0,}; //IBV_EXP_DESTROY_RES_DOMAIN_RESERVED
            ret = ibv_exp_destroy_res_domain(gmexpqp->gqp.dev_context, gmexpqp->res_domain, &attr);
            if (ret) {
                    gds_err("ibv_exp_destroy_res_domain error %d: %s\n", ret, strerror(ret));
                    retcode = ret;
            }            
        }

        free(gmexpqp);

        return retcode;
}

//-----------------------------------------------------------------------------

int gds_mlx5_exp_destroy_cq(gds_mlx5_exp_cq_t *gmexpcq)
{
        int retcode = 0;
        int ret;
        
        if (!gmexpcq) 
                return retcode;
        
        assert(gmexpcq->gcq.dtype == GDS_DRIVER_TYPE_MLX5_EXP);

        if (gmexpcq->gcq.cq) {
                ret = ibv_destroy_cq(gmexpcq->gcq.cq);
                if (ret) {
                        gds_err("error %d in destroy_cq\n", ret);
                        retcode = ret;
                }            
        }

        // res_domain will be destroyed in gds_mlx5_exp_destroy_qp.

        free(gmexpcq);

        return retcode;
}

//-----------------------------------------------------------------------------

int gds_mlx5_exp_prepare_send(gds_mlx5_exp_qp_t *gmexpqp, gds_send_wr *p_ewr, 
                     gds_send_wr **bad_ewr, 
                     gds_mlx5_exp_send_request_t *request)
{
        int ret = 0;
        ret = ibv_post_send(gmexpqp->gqp.qp, p_ewr, bad_ewr);
        if (ret) {

                if (ret == ENOMEM) {
                        // out of space error can happen too often to report
                        gds_dbg("ENOMEM error %d in ibv_post_send\n", ret);
                } else {
                        gds_err("error %d in ibv_post_send\n", ret);
                }
                goto out;
        }
        
        ret = ibv_exp_peer_commit_qp(gmexpqp->gqp.qp, &request->commit);
        if (ret) {
                gds_err("error %d in ibv_exp_peer_commit_qp\n", ret);
                goto out;
        }
out:
        return ret;
}

//-----------------------------------------------------------------------------

void gds_mlx5_exp_init_send_info(gds_mlx5_exp_send_request_t *info)
{
        gds_dbg("send_request=%p\n", info);

        info->commit.storage = info->wr;
        info->commit.entries = sizeof(info->wr)/sizeof(info->wr[0]);
        gds_init_ops(info->commit.storage, info->commit.entries);
}

//-----------------------------------------------------------------------------

int gds_mlx5_exp_post_send_ops(gds_peer *peer, gds_mlx5_exp_send_request_t *info, gds_op_list_t &ops)
{
        return gds_post_ops(peer, info->commit.entries, info->commit.storage, ops, 0);
}

//-----------------------------------------------------------------------------

int gds_mlx5_exp_post_send_ops_on_cpu(gds_mlx5_exp_send_request_t *info, int flags)
{
        return gds_post_ops_on_cpu(info->commit.entries, info->commit.storage, flags);
}

//-----------------------------------------------------------------------------

void gds_mlx5_exp_init_wait_request(gds_mlx5_exp_wait_request_t *request, uint32_t offset)
{
        gds_dbg("wait_request=%p offset=%08x\n", request, offset);
        request->peek.storage = request->wr;
        request->peek.entries = sizeof(request->wr)/sizeof(request->wr[0]);
        request->peek.whence = IBV_EXP_PEER_PEEK_ABSOLUTE;
        request->peek.offset = offset;
        gds_init_ops(request->peek.storage, request->peek.entries);
}

//-----------------------------------------------------------------------------

static void gds_mlx5_exp_dump_ops(struct peer_op_wr *op, size_t count)
{
        size_t n = 0;
        for (; op; op = op->next, ++n) {
                gds_dbg("op[%zu] type:%d\n", n, op->type);
                switch(op->type) {
                case IBV_EXP_PEER_OP_FENCE: {
                        gds_dbg("FENCE flags=%" PRIu64 "\n", op->wr.fence.fence_flags);
                        break;
                }
                case IBV_EXP_PEER_OP_STORE_DWORD: {
                        CUdeviceptr dev_ptr = range_from_id(op->wr.dword_va.target_id)->dptr + 
                                op->wr.dword_va.offset;
                        gds_dbg("STORE_QWORD data:%x target_id:%" PRIx64 " offset:%zu dev_ptr=%llx\n",
                                op->wr.dword_va.data, op->wr.dword_va.target_id,
                                op->wr.dword_va.offset, dev_ptr);
                        break;
                }
                case IBV_EXP_PEER_OP_STORE_QWORD: {
                        CUdeviceptr dev_ptr = range_from_id(op->wr.qword_va.target_id)->dptr +
                                op->wr.qword_va.offset;
                        gds_dbg("STORE_QWORD data:%" PRIx64 " target_id:%" PRIx64 " offset:%zu dev_ptr=%llx\n",
                                op->wr.qword_va.data, op->wr.qword_va.target_id,
                                op->wr.qword_va.offset, dev_ptr);
                        break;
                }
                case IBV_EXP_PEER_OP_COPY_BLOCK: {
                        CUdeviceptr dev_ptr = range_from_id(op->wr.copy_op.target_id)->dptr +
                                op->wr.copy_op.offset;
                        gds_dbg("COPY_BLOCK src:%p len:%zu target_id:%" PRIx64 " offset:%zu dev_ptr=%llx\n",
                                op->wr.copy_op.src, op->wr.copy_op.len,
                                op->wr.copy_op.target_id, op->wr.copy_op.offset,
                                dev_ptr);
                        break;
                }
                case IBV_EXP_PEER_OP_POLL_AND_DWORD:
                case IBV_EXP_PEER_OP_POLL_NOR_DWORD: {
                        CUdeviceptr dev_ptr = range_from_id(op->wr.dword_va.target_id)->dptr + 
                                op->wr.dword_va.offset;
                        gds_dbg("%s data:%08x target_id:%" PRIx64 " offset:%zu dev_ptr=%llx\n", 
                                (op->type==IBV_EXP_PEER_OP_POLL_AND_DWORD) ? "POLL_AND_DW" : "POLL_NOR_SDW",
                                op->wr.dword_va.data, 
                                op->wr.dword_va.target_id, 
                                op->wr.dword_va.offset, 
                                dev_ptr);
                        break;
                }
                default:
                        gds_err("undefined peer op type %d\n", op->type);
                        break;
                }
        }

        assert(count == n);
}

//-----------------------------------------------------------------------------

void gds_mlx5_exp_dump_wait_request(gds_mlx5_exp_wait_request_t *request, size_t idx)
{
        struct ibv_exp_peer_peek *peek = &request->peek;
        gds_dbg("req[%zu] entries:%u whence:%u offset:%u peek_id:%" PRIx64 " comp_mask:%08x\n", 
                idx, peek->entries, peek->whence, peek->offset, 
                peek->peek_id, peek->comp_mask);
        gds_mlx5_exp_dump_ops(peek->storage, peek->entries);
}

//-----------------------------------------------------------------------------

int gds_mlx5_exp_prepare_wait_cq(gds_mlx5_exp_cq_t *mexpcq, gds_mlx5_exp_wait_request_t *request, int flags)
{
        int retcode = 0;

        retcode = ibv_exp_peer_peek_cq(mexpcq->gcq.cq, &request->peek);
        if (retcode == ENOSPC) {
                // TODO: handle too few entries
                gds_err("not enough ops in peer_peek_cq\n");
                goto out;
        } else if (retcode) {
                gds_err("error %d in peer_peek_cq\n", retcode);
                goto out;
        }
        //gds_dump_wait_request(request, 1);
        out:
	       return retcode;
}

//-----------------------------------------------------------------------------

int gds_mlx5_exp_append_wait_cq(gds_mlx5_exp_wait_request_t *request, uint32_t *dw, uint32_t val)
{
        int ret = 0;
        unsigned MAX_NUM_ENTRIES = sizeof(request->wr) / sizeof(request->wr[0]);
        unsigned n = request->peek.entries;
        struct peer_op_wr *wr = request->peek.storage;

        if (n + 1 > MAX_NUM_ENTRIES) {
            gds_err("no space left to stuff a poke\n");
            ret = ENOMEM;
            goto out;
        }

        // at least 1 op
        assert(n);
        assert(wr);

        for (; n; --n) 
                wr = wr->next;

        assert(wr);

        wr->type = IBV_EXP_PEER_OP_STORE_DWORD;
        wr->wr.dword_va.data = val;
        wr->wr.dword_va.target_id = 0; // direct mapping, offset IS the address
        wr->wr.dword_va.offset = (ptrdiff_t)(dw-(uint32_t*)0);

        ++request->peek.entries;

out:
        return ret;
}

//-----------------------------------------------------------------------------

int gds_mlx5_exp_abort_wait_cq(gds_mlx5_exp_cq_t *gmexpcq, gds_mlx5_exp_wait_request_t *request)
{
        struct ibv_exp_peer_abort_peek abort_ctx;
        abort_ctx.peek_id = request->peek.peek_id;
        abort_ctx.comp_mask = 0;
        return ibv_exp_peer_abort_peek_cq(gmexpcq->gcq.cq, &abort_ctx);
}

//-----------------------------------------------------------------------------

int gds_mlx5_exp_stream_post_wait_descriptor(gds_peer *peer, gds_mlx5_exp_wait_request_t *request, gds_op_list_t &params, int flags)
{
        int ret = 0;

        ret = gds_post_ops(peer, request->peek.entries, request->peek.storage, params, flags);
        if (ret)
                gds_err("error %d in gds_post_ops\n", ret);

        return ret;
}

//-----------------------------------------------------------------------------

int gds_mlx5_exp_post_wait_descriptor(gds_mlx5_exp_wait_request_t *request, int flags)
{
        int ret = 0;

        ret = gds_post_ops_on_cpu(request->peek.entries, request->peek.storage, flags);
        if (ret)
                gds_err("error %d in gds_post_ops_on_cpu\n", ret);

        return ret;
}

//-----------------------------------------------------------------------------

int gds_mlx5_exp_get_wait_descs(gds_mlx5_wait_info_t *mlx5_i, const gds_mlx5_exp_wait_request_t *request)
{
        int retcode = 0;
        size_t n_ops = request->peek.entries;
        peer_op_wr *op = request->peek.storage;
        size_t n = 0;

        memset(mlx5_i, 0, sizeof(*mlx5_i));

        for (; op && n < n_ops; op = op->next, ++n) {
                switch(op->type) {
                case IBV_EXP_PEER_OP_FENCE: {
                        gds_dbg("OP_FENCE: fence_flags=%" PRIu64 "\n", op->wr.fence.fence_flags);
                        uint32_t fence_op = (op->wr.fence.fence_flags & (IBV_EXP_PEER_FENCE_OP_READ|IBV_EXP_PEER_FENCE_OP_WRITE));
                        uint32_t fence_from = (op->wr.fence.fence_flags & (IBV_EXP_PEER_FENCE_FROM_CPU|IBV_EXP_PEER_FENCE_FROM_HCA));
                        uint32_t fence_mem = (op->wr.fence.fence_flags & (IBV_EXP_PEER_FENCE_MEM_SYS|IBV_EXP_PEER_FENCE_MEM_PEER));
                        if (fence_op == IBV_EXP_PEER_FENCE_OP_READ) {
                                gds_dbg("nothing to do for read fences\n");
                                break;
                        }
                        if (fence_from != IBV_EXP_PEER_FENCE_FROM_HCA) {
                                gds_err("unexpected from fence\n");
                                retcode = EINVAL;
                                break;
                        }
                        gds_err("unsupported fence combination\n");
                        retcode = EINVAL;
                        break;
                }
                case IBV_EXP_PEER_OP_STORE_DWORD: {
                        CUdeviceptr dev_ptr = range_from_id(op->wr.dword_va.target_id)->dptr + 
                                op->wr.dword_va.offset;
                        uint32_t data = op->wr.dword_va.data;
                        gds_dbg("OP_STORE_DWORD dev_ptr=%" PRIx64 " data=%08x\n", (uint64_t)dev_ptr, data);
                        if (n != 1) {
                                gds_err("store DWORD is not 2nd op\n");
                                retcode = EINVAL;
                                break;
                        }
                        mlx5_i->flag_ptr = (uint32_t*)dev_ptr;
                        mlx5_i->flag_value = data;
                        break;
                }
                case IBV_EXP_PEER_OP_STORE_QWORD: {
                        CUdeviceptr dev_ptr = range_from_id(op->wr.qword_va.target_id)->dptr +
                                op->wr.qword_va.offset;
                        uint64_t data = op->wr.qword_va.data;
                        gds_dbg("OP_STORE_QWORD dev_ptr=%" PRIx64 " data=%" PRIx64 "\n", (uint64_t)dev_ptr, (uint64_t)data);
                        gds_err("unsupported QWORD op\n");
                        retcode = EINVAL;
                        break;
                }
                case IBV_EXP_PEER_OP_COPY_BLOCK: {
                        CUdeviceptr dev_ptr = range_from_id(op->wr.copy_op.target_id)->dptr +
                                op->wr.copy_op.offset;
                        size_t len = op->wr.copy_op.len;
                        void *src = op->wr.copy_op.src;
                        gds_err("unsupported COPY_BLOCK\n");
                        retcode = EINVAL;
                        break;
                }
                case IBV_EXP_PEER_OP_POLL_AND_DWORD:
                case IBV_EXP_PEER_OP_POLL_GEQ_DWORD:
                case IBV_EXP_PEER_OP_POLL_NOR_DWORD: {
                        CUdeviceptr dev_ptr = range_from_id(op->wr.dword_va.target_id)->dptr + 
                                op->wr.dword_va.offset;
                        uint32_t data = op->wr.dword_va.data;

                        gds_dbg("OP_POLL_DWORD dev_ptr=%" PRIx64 " data=%08x\n", (uint64_t)dev_ptr, data);

                        mlx5_i->cqe_ptr = (uint32_t *)dev_ptr;
                        mlx5_i->cqe_value = data;

                        switch(op->type) {
                        case IBV_EXP_PEER_OP_POLL_NOR_DWORD:
                                // GPU SMs can always do NOR
                                mlx5_i->cond = GDS_WAIT_COND_NOR;
                                break;
                        case IBV_EXP_PEER_OP_POLL_GEQ_DWORD:
                                mlx5_i->cond = GDS_WAIT_COND_GEQ;
                                break;
                        case IBV_EXP_PEER_OP_POLL_AND_DWORD:
                                mlx5_i->cond = GDS_WAIT_COND_AND;
                                break;
                        default:
                                gds_err("unexpected op type\n");
                                retcode = EINVAL;
                                goto err;
                        }
                        break;
                }
                default:
                        gds_err("undefined peer op type %d\n", op->type);
                        retcode = EINVAL;
                        break;
                }
        err:
                if (retcode) {
                        gds_err("error in fill func at entry n=%zu\n", n);
                        break;
                }
        }
        return retcode;
}

//-----------------------------------------------------------------------------

int gds_mlx5_exp_rollback_qp(gds_mlx5_exp_qp_t *gmexpqp, gds_mlx5_exp_send_request_t *send_info)
{
        struct ibv_exp_rollback_ctx rollback;
        int ret = 0;
        enum ibv_exp_rollback_flags flag = IBV_EXP_ROLLBACK_ABORT_LATE;

        assert(gmexpqp);
        assert(gmexpqp->gqp.qp);
        assert(send_info);

        /* from ibv_exp_peer_commit call */
        rollback.rollback_id = send_info->commit.rollback_id;
        /* from ibv_exp_rollback_flag */
        rollback.flags = flag;
        /* Reserved for future expensions, must be 0 */
        rollback.comp_mask = 0;
        gds_warn("Need to rollback WQE %lx\n", rollback.rollback_id);
        ret = ibv_exp_rollback_qp(gmexpqp->gqp.qp, &rollback);
        if (ret)
                gds_err("error %d in ibv_exp_rollback_qp\n", ret);

out:
        return ret;
}
