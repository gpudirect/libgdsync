#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <netdb.h>
#include <malloc.h>
#include <getopt.h>
#include <arpa/inet.h>
#include <time.h>
#include <assert.h>
#include <limits.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <infiniband/verbs_exp.h>
#include <gdsync.h>
#include <gdsync/tools.h>
#include <gdsync/device.cuh>
#include <gdrapi.h>

#include "config.h"
#include "test_utils.h"
#include "gpu.h"

#define CHUNK_SIZE 16

int poll_dword_geq(uint32_t *ptr, uint32_t payload, gds_us_t tm)
{
        gds_us_t start = gds_get_time_us();
        int ret = 0;
        while(1) {
                uint32_t value = gds_atomic_read_dword(ptr);
                gpu_dbg("val=%x\n", value);
                if (value >= payload) {
                        ret = 0;
                        break;
                }
                // time-out check
                if ((gds_get_time_us()-start) > tm) {
                        gpu_dbg("timeout %ld us reached!!\n", tm);
                        ret = EWOULDBLOCK;
                        break;
                }
                //arch_cpu_relax();
        }
        return ret;
}

int main(int argc, char *argv[])
{
        int ret = 0;
        int gpu_id = 0;
        int num_iters = 1000;
        // this seems to minimize polling time
        size_t page_size = sysconf(_SC_PAGESIZE);
        size_t size = 1024*64;
        int use_gpu_buf = 0;
        int use_flush = 0;
        int use_membar = 0;
        int use_nor = 0;

        while(1) {            
                int c;
                c = getopt(argc, argv, "d:n:hfgmN");
                if (c == -1)
                        break;

                switch(c) {
                case 'd':
                        gpu_id = strtol(optarg, NULL, 0);
                        break;
                case 'm':
                        use_membar = !use_membar;
                        break;
                case 'n':
                        num_iters = strtol(optarg, NULL, 0);
                        break;
                case 'f':
                        use_flush = 1;
                        printf("INFO enabling flush\n");
                        break;
                case 'g':
                        use_gpu_buf = 1;
                        printf("INFO polling on GPU buffer\n");
                        break;
                case 'N':
                        use_nor = 1;
                        printf("INFO polling using NOR\n");
                        break;
                case 'h':
                        printf("Usage:\n"
                               " %s [options]\n"
                               "Options:\n"
                               " -d id  use gpu ordinal id\n"
                               " -n n   iterate n times\n"
                               " -f     issue a GPU RDMA flush following each poll\n"
                               " -g     allocate all memory on GPU\n"
                               " -m     issue memory barrier between signal and data stores\n"
                               " -N     poll memory using NOR condition (requires Volta)\n"
                               " -h     this help\n", argv[0]);
                        exit(EXIT_SUCCESS);
                        break;
                default:
                        printf("ERROR: invalid option\n");
                        exit(EXIT_FAILURE);
                }
        }

	if (gpu_init(gpu_id, CU_CTX_SCHED_AUTO)) {
		fprintf(stderr, "error in GPU init.\n");
		exit(EXIT_FAILURE);
	}

        puts("");
        printf("number iterations %d\n", num_iters);
        printf("use poll flush %d\n", use_flush);
        printf("use poke membar %d\n", use_membar);
        printf("buffer size %zd\n", size);
        printf("poll on %s buffer\n", use_gpu_buf?"GPU":"CPU");
        printf("write on %s buffer\n", use_gpu_buf?"GPU":"CPU");
        puts("");
        

        int mem_type = use_gpu_buf ? GDS_MEMORY_GPU : GDS_MEMORY_HOST;

        gds_mem_desc_t desc =  {0,};
        ret = gds_alloc_mapped_memory(&desc, size, mem_type);
        if (ret) {
                gpu_fail("error (%d) while allocating mem\n", ret);
        }
        CUdeviceptr d_buf = desc.d_ptr;
        void *h_buf = desc.h_ptr;
        printf("allocated d_buf=%p h_buf=%p\n", (void*)d_buf, h_buf);
        memset(h_buf, 0, size);

        gds_mem_desc_t desc_data =  {0,};
        ret = gds_alloc_mapped_memory(&desc_data, size, mem_type);
        if (ret) {
                gpu_fail("error (%d) while allocating mem\n", ret);
        }
        CUdeviceptr d_data = desc_data.d_ptr;
        uint32_t *h_data = (uint32_t*)desc_data.h_ptr;
        printf("allocated d_data=%p h_data=%p\n", (void*)d_data, h_data);
        memset(h_data, 0, size);

        int i;
        int value;
        int poll_flags = mem_type;
        if (use_flush)
                poll_flags |= GDS_WAIT_POST_FLUSH;


        printf("starting test (dot==1000 iterations)...\n");
        perf_start();

        int n_errors = 0;
        int round;
        int print_dots = 0;
        for (i = 0, value = 1; i < num_iters; ++i, ++value) {
                uint32_t bit = 0;
                for (round = 0; round < 2; ++round) {
                        ASSERT(value <= INT_MAX);
                        uint32_t *h_signal = (uint32_t*)h_buf +  ((0) % (size/sizeof(uint32_t)));
                        uint32_t *d_signal = (uint32_t*)d_buf +  ((0) % (size/sizeof(uint32_t)));
                        uint32_t   *signal = (mem_type == GDS_MEMORY_GPU ? d_signal : h_signal);

                        uint32_t *h_done   = (uint32_t*)h_buf +  ((1) % (size/sizeof(uint32_t)));
                        uint32_t *d_done   = (uint32_t*)d_buf +  ((1) % (size/sizeof(uint32_t)));
                        uint32_t   *done   = (mem_type == GDS_MEMORY_GPU ? d_done : h_done);

                        uint32_t *h_dbg    = (uint32_t*)h_buf +  ((2) % (size/sizeof(uint32_t)));
                        uint32_t *d_dbg    = (uint32_t*)d_buf +  ((2) % (size/sizeof(uint32_t)));
                        uint32_t   *dbg    = (mem_type == GDS_MEMORY_GPU ? d_dbg : h_dbg);

                        // CHUNK_SIZE contiguous blocks of dwords
                        ASSERT(size >= CHUNK_SIZE*sizeof(uint32_t));
                        uint32_t *h_vals   = (uint32_t*)h_data + ((i*CHUNK_SIZE) % (size/sizeof(uint32_t)));
                        uint32_t *d_vals   = (uint32_t*)d_data + ((i*CHUNK_SIZE) % (size/sizeof(uint32_t)));
                        uint32_t   *vals   = (mem_type == GDS_MEMORY_GPU ? d_vals : h_vals);

                        int ii;
                        //uint32_t src_data[CHUNK_SIZE] = {1, 2, 3};
                        uint32_t src_data[CHUNK_SIZE];
                        for (ii=0; ii<CHUNK_SIZE; ++ii) src_data[ii] = 1+ii;

                        if (0 == round) {
                                enum { n_descs = 10+CHUNK_SIZE };
                                gds_descriptor_t descs[n_descs];
                                int k = 0;

                                descs[k].tag = GDS_TAG_WRITE_VALUE32;
                                GDSCHECK(gds_prepare_write_value32(&descs[k].write32, dbg, 0xA000|i, mem_type));
                                ++k;

                                // wait for CPU signal
                                descs[k].tag = GDS_TAG_WAIT_VALUE32;
                                if (use_nor) {
                                        // sweep over the 32 bits of a dword
                                        bit = 1U<<(value & 31);
                                        uint32_t msk = ~bit;
                                        gpu_dbg("signal=%08x msk=%08x\n", bit, msk);
                                        gds_atomic_set_dword(h_signal, bit);
                                        // can fail if GPU does not support CU_STREAM_WAIT_VALUE_NOR
                                        GDSCHECK(gds_prepare_wait_value32(&descs[k].wait32, signal, msk, GDS_WAIT_COND_NOR, poll_flags));
                                }
                                else {
                                        // while (d_signal != 0);
                                        GDSCHECK(gds_prepare_wait_value32(&descs[k].wait32, signal, value, GDS_WAIT_COND_EQ, poll_flags));
                                }
                                ++k;

                                descs[k].tag = GDS_TAG_WRITE_VALUE32;
                                GDSCHECK(gds_prepare_write_value32(&descs[k].write32, dbg, 0xB000|i, mem_type));
                                ++k;

                                // d_vals[0] = 0
                                descs[k].tag = GDS_TAG_WRITE_VALUE32;
                                GDSCHECK(gds_prepare_write_value32(&descs[k].write32, vals, 0, mem_type | (use_membar ? GDS_WRITE_PRE_BARRIER : 0)));
                                ++k;

                                // overwrite d_vals[0...CHUNK_SIZE-1]={1,2,...}
                                // if CPU sees d_vals[0]==0, something went wrong with WRITE_MEMORY below
                                //
                                // NOTE: pre-barrier needed to fence previous write to 'vals'
#if HAVE_DECL_CU_STREAM_MEM_OP_WRITE_MEMORY
                                descs[k].tag = GDS_TAG_WRITE_MEMORY;
                                GDSCHECK(gds_prepare_write_memory(&descs[k].writemem, (uint8_t*)vals, (uint8_t*)src_data, sizeof(src_data), mem_type | GDS_WRITE_MEMORY_PRE_BARRIER_SYS));
                                ++k;
#else
                                for (ii=0; ii<CHUNK_SIZE; ++ii) {
                                        descs[k].tag = GDS_TAG_WRITE_VALUE32;
                                        GDSCHECK(gds_prepare_write_value32(&descs[k].write32, vals+ii, src_data[ii], mem_type | (ii==0 ? GDS_WRITE_PRE_BARRIER : 0)));
                                        ++k;
                                        ASSERT(k < n_descs);
                                }
#endif
                                // while (d_vals[0] != 0);
                                // will be updated to 0 by CPU
                                descs[k].tag = GDS_TAG_WAIT_VALUE32;
                                GDSCHECK(gds_prepare_wait_value32(&descs[k].wait32, vals, 0, GDS_WAIT_COND_EQ, poll_flags));
                                ++k;

                                // done = 1;
                                // to signal CPU
                                descs[k].tag = GDS_TAG_WRITE_VALUE32;
                                GDSCHECK(gds_prepare_write_value32(&descs[k].write32, done, value, mem_type));
                                ++k;

#if 0        
                                //puts("sleeping 1s");
                                //sleep(1);
#endif

                                GDSCHECK(gds_stream_post_descriptors(gpu_stream, k, descs, 0));
                        }
                        else {
                                int retcode;
                                // verify that the stream is still stuck at the 1st wait value
                                usleep(100);
                                ASSERT(cuStreamQuery(gpu_stream) == CUDA_ERROR_NOT_READY);
                                // verify that memory words still have their expected initial value
                                if (use_nor) {
                                        ASSERT(gds_atomic_read_dword(h_signal) & bit);
                                }
                                else {
                                        ASSERT(gds_atomic_read_dword(h_signal) == (value-1));
                                }
                                ASSERT(gds_atomic_read_dword(h_done) == (value-1));
                                
                                gpu_dbg("%d:       dbg @%p:%08x\n", i, h_dbg, gds_atomic_read_dword(h_dbg));
                                gpu_dbg("%d:       sig @%p:%08x\n", i, h_signal, gds_atomic_read_dword(h_signal));
                                gpu_dbg("%d:      done @%p:%08x\n", i, h_done, gds_atomic_read_dword(h_done));
                                if (use_nor) {
                                        // unset the bit
                                        gpu_dbg("%d: write sig @%p=%08x\n", i, h_signal, ~bit);
                                        gds_atomic_set_dword(h_signal, ~bit);
                                }
                                else {
                                        gpu_dbg("%d: write sig @%p=%08x\n", i, h_signal, value);
                                        gds_atomic_set_dword(h_signal, value);
                                }
                                gpu_dbg("%d:       sig @%p:%08x\n", i, h_signal, gds_atomic_read_dword(h_signal));
                                // enough for the GPU to wake up and observe the updated values in the prints below
                                //usleep(100);
                                gpu_dbg("%d:       dbg @%p:%08x\n", i, h_dbg, gds_atomic_read_dword(h_dbg));
                                gpu_dbg("%d:      done @%p:%08x\n", i, h_done, gds_atomic_read_dword(h_done));
                                gpu_dbg("%d: poll done @%p==%08x\n", i, h_done, value);
                                gds_us_t tmout = 1000; //usecs
                                retcode = poll_dword_geq(h_done, value, tmout);
                                if (retcode == EWOULDBLOCK) {
                                        // expected behaviour
                                        gpu_dbg("%d: unblocking stream by writing %p <= %08x\n", i, h_vals, 0);
                                        gds_atomic_set_dword(h_vals, 0);
                                        retcode = poll_dword_geq(h_done, value, tmout);
                                        if (retcode) {
                                                gpu_fail("error %d while polling done flag", retcode);
                                        }
                                }
                                else if (retcode) {
                                        gpu_fail("error %d while polling done\n", retcode);
                                }
                                else {
                                        gpu_err("%d: stream order violation\n", i);
                                        gpu_err("*done=%08x expected!=%08x &vals[0]=%08x\n", gds_atomic_read_dword(h_done), value, gds_atomic_read_dword(vals));
                                        ++n_errors;
                                }
                        }
                        if (i % 1000 == 0) {
                                print_dots = 1;
                                printf("."); fflush(stdout);
                        }
                }

        }

        if (print_dots)
                puts("");

        if (n_errors) {
                gpu_fail("detected n_errors=%d\n", n_errors);
        }
        else {
                ASSERT(cuStreamQuery(gpu_stream) == CUDA_SUCCESS);
                CUCHECK(cuStreamSynchronize(gpu_stream));
        }
        perf_stop();

err:
        gpu_dbg("calling gds_free_mapped_memory\n");
        ret = gds_free_mapped_memory(&desc);
        if (ret) {
                gpu_fail("error (%d) while freeing mem\n", ret);
        }

        ret = gds_free_mapped_memory(&desc_data);
        if (ret) {
                gpu_fail("error (%d) while freeing mem\n", ret);
        }

	gpu_finalize();

        printf(">>> SUCCESS\n");
        return ret;
}

/*
 * Local variables:
 *  c-indent-level: 8
 *  c-basic-offset: 8
 *  tab-width: 8
 * End:
 */
