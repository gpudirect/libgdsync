#include <infiniband/verbs.h>
#include <infiniband/verbs_exp.h>

#include <gdsync.h>

typedef struct gds_mlx5_exp_cq {
        gds_cq_t                gcq;
        ibv_exp_res_domain     *res_domain;
} gds_mlx5_exp_cq_t;

typedef struct gds_mlx5_exp_qp {
        gds_qp_t                gqp;
        ibv_exp_res_domain     *res_domain;
} gds_mlx5_exp_qp_t;

static inline gds_mlx5_exp_cq_t *to_gds_mexp_cq(gds_cq_t *gcq) {
        assert(gcq->dtype == GDS_DRIVER_TYPE_MLX5_EXP);
        return container_of(gcq, gds_mlx5_exp_cq_t, gcq);
}

static inline gds_mlx5_exp_qp_t *to_gds_mexp_qp(gds_qp_t *gqp) {
        assert(gqp->dtype == GDS_DRIVER_TYPE_MLX5_EXP);
        return container_of(gqp, gds_mlx5_exp_qp_t, gqp);
}

gds_mlx5_exp_cq_t *gds_mlx5_exp_create_cq(
        struct ibv_context *context, int cqe,
        void *cq_context, struct ibv_comp_channel *channel,
        int comp_vector, gds_peer *peer, gds_peer_attr *peer_attr, gds_alloc_cq_flags_t flags,
        struct ibv_exp_res_domain *res_domain);

gds_mlx5_exp_qp_t *gds_mlx5_exp_create_qp(
        struct ibv_pd *pd, struct ibv_context *context, gds_qp_init_attr_t *qp_attr, 
        gds_peer *peer, gds_peer_attr *peer_attr, int flags);

int gds_mlx5_exp_destroy_cq(gds_mlx5_exp_cq_t *gmexpcq);
int gds_mlx5_exp_destroy_qp(gds_mlx5_exp_qp_t *gmexpqp);
