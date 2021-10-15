#include <unistd.h>
#include <string.h>
#include <assert.h>

#include <infiniband/verbs.h>
#include <infiniband/verbs_exp.h>
#include <infiniband/peer_ops.h>

#include <gdsync.h>
#include <gdsync/mlx5.h>

#include "objs.hpp"
#include "utils.hpp"

typedef struct gds_mlx5_exp_cq {
        gds_cq_t                gcq;
        ibv_exp_res_domain     *res_domain;
} gds_mlx5_exp_cq_t;

typedef struct gds_mlx5_exp_qp {
        gds_qp_t                gqp;
        ibv_exp_res_domain     *res_domain;
} gds_mlx5_exp_qp_t;

typedef struct gds_mlx5_exp_send_request {
        gds_driver_type_t dtype;
        uint8_t pad0[4];
        struct ibv_exp_peer_commit commit;
        struct peer_op_wr wr[GDS_SEND_INFO_MAX_OPS];
        uint8_t pad1[24];
} gds_mlx5_exp_send_request_t;
static_assert(sizeof(gds_mlx5_exp_send_request_t) % 64 == 0, "gds_mlx5_exp_send_request_t must be 64-byte aligned.");
static_assert(sizeof(gds_mlx5_exp_send_request_t) <= sizeof(gds_send_request_t), "The size of gds_mlx5_exp_send_request_t must be less than or equal to that of gds_send_request_t.");
static_assert(offsetof(gds_mlx5_exp_send_request_t, dtype) == offsetof(gds_send_request_t, dtype), "dtype of gds_mlx5_exp_send_request_t and gds_send_request_t must be at the same offset.");

typedef struct gds_mlx5_exp_wait_request {
        gds_driver_type_t dtype;
        uint8_t pad0[4];
        struct ibv_exp_peer_peek peek;
        struct peer_op_wr wr[GDS_WAIT_INFO_MAX_OPS];
        uint8_t pad1[16];
} gds_mlx5_exp_wait_request_t;
static_assert(sizeof(gds_mlx5_exp_wait_request_t) % 64 == 0, "gds_mlx5_exp_wait_request_t must be 64-byte aligned.");
static_assert(sizeof(gds_mlx5_exp_wait_request_t) <= sizeof(gds_wait_request_t), "The size of gds_mlx5_exp_wait_request_t must be less than or equal to that of gds_wait_request_t.");
static_assert(offsetof(gds_mlx5_exp_wait_request_t, dtype) == offsetof(gds_wait_request_t, dtype), "dtype of gds_mlx5_exp_wait_request_t and gds_wait_request_t must be at the same offset.");

static inline gds_mlx5_exp_cq_t *to_gds_mexp_cq(gds_cq_t *gcq) {
        assert(gcq->dtype == GDS_DRIVER_TYPE_MLX5_EXP);
        return container_of(gcq, gds_mlx5_exp_cq_t, gcq);
}

static inline gds_mlx5_exp_qp_t *to_gds_mexp_qp(gds_qp_t *gqp) {
        assert(gqp->dtype == GDS_DRIVER_TYPE_MLX5_EXP);
        return container_of(gqp, gds_mlx5_exp_qp_t, gqp);
}

static inline gds_mlx5_exp_send_request_t *to_gds_mexp_send_request(gds_send_request_t *gsreq) {
        assert(gsreq->dtype == GDS_DRIVER_TYPE_MLX5_EXP);
        return (gds_mlx5_exp_send_request_t *)(gsreq);
}

static inline const gds_mlx5_exp_send_request_t *to_gds_mexp_send_request(const gds_send_request_t *gsreq) {
        return (const gds_mlx5_exp_send_request_t *)to_gds_mexp_send_request((const gds_send_request_t *)gsreq);
}

static inline gds_mlx5_exp_wait_request_t *to_gds_mexp_wait_request(gds_wait_request_t *gwreq) {
        assert(gwreq->dtype == GDS_DRIVER_TYPE_MLX5_EXP);
        return (gds_mlx5_exp_wait_request_t *)(gwreq);
}

static inline const gds_mlx5_exp_wait_request_t *to_gds_mexp_wait_request(const gds_wait_request_t *gwreq) {
        return (const gds_mlx5_exp_wait_request_t *)to_gds_mexp_wait_request((const gds_wait_request_t *)gwreq);
}

