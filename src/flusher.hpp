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
#include <inttypes.h>

#include <map>
#include <algorithm>
#include <string>
using namespace std;

#include <cuda.h>
//#include <cuda_runtime.h>
#include <infiniband/verbs_exp.h>
#include <gdrapi.h>

#include <pthread.h>
#include <sys/socket.h>
#include <arpa/inet.h>

#include "gdsync.h"
#include "gdsync/tools.h"
#include "objs.hpp"
#include "utils.hpp"
#include "memmgr.hpp"
#include "archutils.h"

typedef enum gds_flusher_type {
    GDS_FLUSHER_NONE=0,
    GDS_FLUSHER_NATIVE,
    GDS_FLUSHER_CPU,
    GDS_FLUSHER_NIC
} gds_flusher_type_t;

#define GDS_FLUSHER_OP_NATIVE 0
#define GDS_FLUSHER_OP_CPU 2
#define GDS_FLUSHER_OP_NIC 5

#define GDS_FLUSHER_PORT 1
#define GDS_FLUSHER_QKEY 0 //0x11111111

typedef struct gds_flusher_buf {
    CUdeviceptr buf_d;
    void * buf_h;
    int size;
    gdr_mh_t mh;
    gds_mem_desc_t *desc;
    struct ibv_mr * mr;
} gds_flusher_buf;
typedef gds_flusher_buf * gds_flusher_buf_t;

typedef struct flusher_qp_info {
    struct gds_qp *loopback_qp;
    struct ibv_pd *pd;
    struct ibv_context *context;
    int gpu_id;
    struct ibv_ah * ah;
    char gid_string[INET6_ADDRSTRLEN];
    union ibv_gid gid_bin;
    int lid;
    int qpn;
    int psn;
    int ib_port;
} flusher_qp_info;
typedef flusher_qp_info * flusher_qp_info_t;

//-----------------------------------------------------------------------------
bool gds_use_native_flusher();
int gds_flusher_setup();
int gds_flusher_get_envars();
int gds_flusher_init(struct ibv_pd *pd, struct ibv_context *context, int gpu_id);
int gds_flusher_destroy();
int gds_flusher_count_op();
int gds_flusher_post_stream(CUstream stream);
int gds_flusher_add_ops(CUstreamBatchMemOpParams *params, int &idx);

int gds_flusher_start_thread(pthread_t *fThread);
int gds_flusher_stop_thread(pthread_t fThread);
void * gds_flusher_func_thread(void *);
//-----------------------------------------------------------------------------
/*
 * Local variables:
 *  c-indent-level: 8
 *  c-basic-offset: 8
 *  tab-width: 8
 *  indent-tabs-mode: nil
 * End:
 */
