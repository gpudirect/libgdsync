typedef struct gds_mlx5_exp_cq {
        gds_cq_t                gcq;
        ibv_exp_res_domain     *res_domain
} gds_mlx5_exp_cq_t;

typedef struct gds_mlx5_exp_qp {
        gds_qp_t                gqp;
        ibv_exp_res_domain     *res_domain
} gds_mlx5_exp_qp_t;

static inline gds_mlx5_exp_cq_t *to_gds_mexp_cq(gds_cq_t *gcq) {
        assert(gcq->dtype == GDS_DRIVER_TYPE_MLX5_EXP);
        return container_of(gcq, gds_mlx5_exp_cq_t, gcq);
}

static inline gds_mlx5_exp_qp_t *to_gds_mexp_qp(gds_qp_t *gqp) {
        assert(gcq->dtype == GDS_DRIVER_TYPE_MLX5_EXP);
        return container_of(gqp, gds_mlx5_exp_qp_t, gqp);
}

