#if HAVE_CONFIG_H
#  include <config.h>
#endif /* HAVE_CONFIG_H */

#include "gdsync/device.cuh"
#include "objs.hpp"
#include "utils.hpp"

using namespace gdsync;

int gds_launch_1QPSend_2CQWait(gds_peer *peer, CUstream stream, gds_op_list_t &params)
{
        int ret = 0;
        isem32_t sem0;
        isem64_t sem1;
        isem32_t semw[2];
        wait_cond_t condw[2] = {GDS_WAIT_COND_GEQ, GDS_WAIT_COND_GEQ};
        isem32_t sem23[2];
        CUstreamBatchMemOpParams *param = NULL;

        gds_dbg("stream=%p #params=%zu cufunction=%p\n", stream, params.size(), peer->kernels.krn1snd2wait);
        do {
                if (params.size() != 7) {
                        gds_dbg("unexpected %d params\n", params.size());
                        ret = EINVAL;
                        break;
                }

                // marshal parameters
#if 0
                param = &params.at(0);
                GDS_ASSERT(param->operation == CU_STREAM_MEM_OP_WRITE_VALUE_32);
                sem0.ptr = reinterpret_cast<uint32_t*>(param->writeValue.address);
                sem0.value = param->writeValue.value;
                gds_dbg("sem0 %p %x\n", sem0.ptr, sem0.value);

                param = &params.at(1);
                GDS_ASSERT(param->operation == CU_STREAM_MEM_OP_MEMORY_BARRIER);

                // hack! need to assert mlx5 and size==64
                // converting write memory into write_64 of 1st qword
                // todo: implement memset_t
                param = &params.at(2);
                GDS_ASSERT(param->operation == CU_STREAM_MEM_OP_WRITE_MEMORY);
                sem1.ptr = reinterpret_cast<uint64_t*>(param->writeMemory.dst);
                sem1.value = *reinterpret_cast<uint64_t*>(param->writeMemory.src);
                gds_dbg("sem1 %p %lx\n", sem1.ptr, sem1.value);

                param = &params.at(3);
                GDS_ASSERT(param->operation == CU_STREAM_MEM_OP_WAIT_VALUE_32);
                semw[0].ptr = reinterpret_cast<uint32_t*>(param->waitValue.address);
                semw[0].value = param->waitValue.value;
                condw[0] = gds_cuwait_flags_to_wait_cond(param->waitValue.flags);
                gds_dbg("semw[0] %p 0x%lx 0x%x\n", semw[0].ptr, semw[0].value, condw[0]);

                param = &params.at(4);
                GDS_ASSERT(param->operation == CU_STREAM_MEM_OP_WRITE_VALUE_32);
                sem23[0].ptr = reinterpret_cast<uint32_t*>(param->writeValue.address);
                sem23[0].value = param->writeValue.value;
                gds_dbg("sem23[0] %p %x\n", sem23[0].ptr, sem23[0].value);

                param = &params.at(5);
                GDS_ASSERT(param->operation == CU_STREAM_MEM_OP_WAIT_VALUE_32);
                semw[1].ptr = reinterpret_cast<uint32_t*>(param->waitValue.address);
                semw[1].value = param->waitValue.value;
                condw[0] = gds_cuwait_flags_to_wait_cond(param->waitValue.flags);
                gds_dbg("semw[1] %p 0x%lx 0x%x\n", semw[1].ptr, semw[1].value, condw[1]);

                param = &params.at(6);
                GDS_ASSERT(param->operation == CU_STREAM_MEM_OP_WRITE_VALUE_32);
                sem23[1].ptr = reinterpret_cast<uint32_t*>(param->writeValue.address);
                sem23[1].value = param->writeValue.value;
                gds_dbg("sem23[1] %p %x\n", sem23[1].ptr, sem23[1].value);
#endif
                void *krn_params[] = {
                        reinterpret_cast<void *>(&sem0),
                        reinterpret_cast<void *>(&sem1),
                        reinterpret_cast<void *>(&semw[0]),
                        reinterpret_cast<void *>(&condw[0]),
                        reinterpret_cast<void *>(&sem23[0])
                };

                CUCHECK(cuLaunchKernel(peer->kernels.krn1snd2wait,
                                       1, 1, 1,    // 1x1x1 blocks
                                       1, 1, 1,    // 1x1x1 threads
                                       0,          // shared mem
                                       stream,     // stream
                                       krn_params, // params
                                       0 ));       // extra
                CUCHECK(cuStreamSynchronize(stream));
        } while(0);

        return ret;
}

int gds_launch_update_send_params(
                gds_peer *peer,
                CUdeviceptr ptr_to_size_wqe, CUdeviceptr ptr_to_size_new,
                CUdeviceptr ptr_to_lkey_wqe, CUdeviceptr ptr_to_lkey_new,
                CUdeviceptr ptr_to_addr_wqe, CUdeviceptr ptr_to_addr_new,
                CUstream stream)
{
        //gds_dbg("Launching gds_update_send_params with ptr_to_size_wqe=%x, ptr_to_size_new=%x, ptr_to_lkey_wqe=%x, ptr_to_lkey_new=%x, ptr_to_addr_wqe=%x, ptr_to_addr_new=%x\n", 
        //        ptr_to_size_wqe, ptr_to_size_new, ptr_to_lkey_wqe, ptr_to_lkey_new, ptr_to_addr_wqe, ptr_to_addr_new);

        void *krn_params[] = {
                reinterpret_cast<void *>(&ptr_to_size_wqe),
                reinterpret_cast<void *>(&ptr_to_size_new),
                reinterpret_cast<void *>(&ptr_to_lkey_wqe),
                reinterpret_cast<void *>(&ptr_to_lkey_new),
                reinterpret_cast<void *>(&ptr_to_addr_wqe),
                reinterpret_cast<void *>(&ptr_to_addr_new)
        };

        CUCHECK(cuLaunchKernel(peer->kernels.krnsetsndpar,
                                1, 1, 1,    // 1x1x1 blocks
                                3, 1, 1,    // 1x1x1 threads
                                0,          // shared mem
                                stream,     // stream
                                krn_params, // params
                                0 ));       // extra

        return 0;
}

/*
 * Local variables:
 *  c-indent-level: 8
 *  c-basic-offset: 8
 *  tab-width: 8
 *  indent-tabs-mode: nil
 * End:
 */
