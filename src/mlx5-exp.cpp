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

static struct gds_mlx5_exp_cq_t *
gds_mlx5_exp_create_cq(struct ibv_context *context, int cqe,
                        void *cq_context, struct ibv_comp_channel *channel,
                        int comp_vector, int gpu_id, gds_alloc_cq_flags_t flags,
                        struct ibv_exp_res_domain *res_domain)
{
        struct gds_mlx5_exp_cq_t *gmexpcq = NULL;
        ibv_exp_cq_init_attr attr;
        gds_peer *peer = NULL;
        gds_peer_attr *peer_attr = NULL;
        int ret = 0;

        if (!context)
        {
            gds_dbg("Invalid input context\n");
            return NULL;
        }

        gmexpcq = (gds_mlx5_exp_cq_t *)calloc(1, sizeof(gds_mlx5_exp_cq_t));
        if (!gmexpcq) {
            gds_err("cannot allocate memory\n");
            return NULL;
        }

        //Here we need to recover peer and peer_attr pointers to set alloc_type and alloc_flags
        //before ibv_exp_create_cq
        ret = gds_register_peer_by_ordinal(gpu_id, &peer, &peer_attr);
        if (ret) {
            gds_err("error %d while registering GPU peer\n", ret);
            return NULL;
        }
        assert(peer);
        assert(peer_attr);

        peer->alloc_type = gds_peer::CQ;
        peer->alloc_flags = flags;

        attr.comp_mask = IBV_EXP_CQ_INIT_ATTR_PEER_DIRECT;
        attr.flags = 0; // see ibv_exp_cq_create_flags
        attr.peer_direct_attrs = peer_attr;
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

struct gds_mlx5_exp_qp_t *gds_mlx5_exp_create_qp(struct ibv_pd *pd, struct ibv_context *context,
                                gds_qp_init_attr_t *qp_attr, int gpu_id, int flags)
{
        int ret = 0;
        gds_mlx5_exp_qp_t *gmexpqp = NULL;
        struct ibv_qp *qp = NULL;
        gds_mlx5_exp_cq_t *rx_gmexpcq = NULL, *tx_gmexpcq = NULL;
        gds_peer *peer = NULL;
        gds_peer_attr *peer_attr = NULL;
        struct ibv_qp_init_attr exp_qp_attr = {0,};
        int old_errno = errno;

        assert(pd);
        assert(context);
        assert(qp_attr);

        gmexpqp = (gds_mlx5_exp_qp_t *)calloc(1, sizeof(struct gds_qp));
        if (!gqp) {
            gds_err("cannot allocate memory\n");
            return NULL;
        }
        gmexpqp->gqp.dtype = GDS_DRIVER_TYPE_MLX5_EXP;

        gmexpqp->gqp.dev_context = context;

        // peer registration
        gds_dbg("before gds_register_peer_ex\n");
        ret = gds_register_peer_by_ordinal(gpu_id, &peer, &peer_attr);
        if (ret) {
            gds_err("error %d in gds_register_peer_ex\n", ret);
            goto err;
        }

        gmexpqp->res_domain = gds_create_res_domain(context);
        if (gmexpqp->res_domain)
            gds_dbg("using res_domain %p\n", gmexpqp->res_domain);
        else
            gds_warn("NOT using res_domain\n");

        tx_gmexpcq = gds_mlx5_exp_create_cq(context, qp_attr->cap.max_send_wr, NULL, NULL, 0, gpu_id, 
                              (flags & GDS_CREATE_QP_TX_CQ_ON_GPU) ? GDS_ALLOC_CQ_ON_GPU : GDS_ALLOC_CQ_DEFAULT, 
                              gmexpqp->res_domain);
        if (!tx_gmexpcq) {
                ret = errno;
                gds_err("error %d while creating TX CQ, old_errno=%d\n", ret, old_errno);
                goto err;
        }

        rx_gmexpcq = gds_mlx5_exp_create_cq(context, qp_attr->cap.max_recv_wr, NULL, NULL, 0, gpu_id, 
                              (flags & GDS_CREATE_QP_RX_CQ_ON_GPU) ? GDS_ALLOC_CQ_ON_GPU : GDS_ALLOC_CQ_DEFAULT, 
                              gmexpqp->res_domain);
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

        exp_qp_attr = {
                .send_cq = tx_gmexpcq->gcq.cq,
                .recv_cq = rx_gmexpcq->gcq.cq,
                .pd = pd,
                .comp_mask = IBV_EXP_QP_INIT_ATTR_PD | IBV_EXP_QP_INIT_ATTR_PEER_DIRECT,
                .peer_direct_attrs = peer_attr,
                .qp_type = qp_attr->qp_type
        };

        assert(sizeof(exp_qp_attr.cap) == sizeof(qp_attr->cap));

        memcpy(&exp_qp_attr.cap, &qp_attr->cap, sizeof(qp_attr->cap));

        qp = ibv_exp_create_qp(context, qp_attr);
        if (!qp) {
                ret = EINVAL;
                gds_err("error in ibv_exp_create_qp\n");
                goto err;
        }

        gmexpqp->gqp.qp = qp;
        gmexpqp->gqp.send_cq = tx_gmexpcq->gcq;
        gmexpqp->gqp.recv_cq = rx_gmexpcq->gcq;

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

