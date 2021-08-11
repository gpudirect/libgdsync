#include <unistd.h>
#include <string.h>
#include <assert.h>

#include <infiniband/verbs.h>
#include <infiniband/verbs_exp.h>

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

static inline gds_mlx5_exp_wait_request_t *to_gds_mexp_wait_request(gds_wait_request_t *gwreq) {
        assert(gwreq->dtype == GDS_DRIVER_TYPE_MLX5_EXP);
        return (gds_mlx5_exp_wait_request_t *)(gwreq);
}

static inline const gds_mlx5_exp_wait_request_t *to_gds_mexp_wait_request(const gds_wait_request_t *gwreq) {
        return (const gds_mlx5_exp_wait_request_t *)to_gds_mexp_wait_request((const gds_wait_request_t *)gwreq);
}

static inline uint32_t gds_mlx5_exp_get_num_wait_request_entries(gds_mlx5_exp_wait_request_t *gmexp_request) {
        return gmexp_request->peek.entries;
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

int gds_mlx5_exp_prepare_send(gds_mlx5_exp_qp_t *gmexpqp, gds_send_wr *p_ewr, 
                     gds_send_wr **bad_ewr, 
                     gds_send_request_t *request);

void gds_mlx5_exp_init_wait_request(gds_mlx5_exp_wait_request_t *request, uint32_t offset);
void gds_mlx5_exp_dump_wait_request(gds_mlx5_exp_wait_request_t *request, size_t count);
int gds_mlx5_exp_prepare_wait_cq(gds_mlx5_exp_cq_t *mexpcq, gds_mlx5_exp_wait_request_t *request, int flags);
int gds_mlx5_exp_append_wait_cq(gds_mlx5_exp_wait_request_t *request, uint32_t *dw, uint32_t val);
int gds_mlx5_exp_abort_wait_cq(gds_mlx5_exp_cq_t *gmexpcq, gds_mlx5_exp_wait_request_t *request);
int gds_mlx5_exp_stream_post_wait_descriptor(gds_peer *peer, gds_mlx5_exp_wait_request_t *request, gds_op_list_t &params, int flags);
int gds_mlx5_exp_post_wait_descriptor(gds_mlx5_exp_wait_request_t *request, int flags);
int gds_mlx5_exp_get_wait_descs(gds_mlx5_wait_info_t *mlx5_i, const gds_mlx5_exp_wait_request_t *request);
