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

#include <infiniband/verbs_exp.h>
#include <gdsync.h>
#include <gdsync/tools.h>
#include <gdsync/device.cuh>
#include <gdrapi.h>

#include "test_utils.h"
#include "gpu.h"

int poll_dword_geq(uint32_t *ptr, uint32_t payload, gds_us_t tm)
{
        gds_us_t start = gds_get_time_us();
        int ret = 0;
        while(1) {
                uint32_t value = ACCESS_ONCE(*ptr);
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
        int num_iters = 50;
        // this seems to minimize polling time
        int sleep_us = 10;
	size_t page_size = sysconf(_SC_PAGESIZE);
	size_t size = 1024*64;
        int use_gpu_buf = 0;
        int use_flush = 0;
        int use_combined = 0;
        int use_membar = 0;
        int wait_key = -1;
        CUstream gpu_stream;

        int n_bg_streams = 0;

        size_t n_pokes = 1;

        while(1) {            
                int c;
                c = getopt(argc, argv, "cd:p:n:s:hfgP:mW:");
                if (c == -1)
                        break;

                switch(c) {
                case 'd':
                        gpu_id = strtol(optarg, NULL, 0);
                        break;
                case 'W':
                        wait_key = strtol(optarg, NULL, 0);
                        break;
                case 'p':
                        n_bg_streams = strtol(optarg, NULL, 0);
                        break;
                case 'c':
                        // merge poll and multiple pokes
                        use_combined = 1;
                        break;
                case 'P':
                        // multiple pokes
                        n_pokes = strtol(optarg, NULL, 0);
                        break;
                case 'm':
                        use_membar = 1;
                        break;
                case 'n':
                        num_iters = strtol(optarg, NULL, 0);
                        break;
                case 's':
                        sleep_us = strtol(optarg, NULL, 0);
                        break;
                case 'f':
                        use_flush = 1;
                        printf("INFO enabling flush\n");
                        break;
                case 'g':
                        use_gpu_buf = 1;
                        printf("INFO polling on GPU buffer\n");
                        break;
                case 'h':
                        printf(" %s [-n <iters>][-s <sleep us>][-p # bg streams][-P # pokes][ckhfgomW]\n", argv[0]);
                        exit(EXIT_SUCCESS);
                        break;
                default:
                        printf("ERROR: invalid option\n");
                        exit(EXIT_FAILURE);
                }
        }

        CUstream bg_streams[n_bg_streams];
        memset(bg_streams, 0, sizeof(bg_streams));

	if (gpu_init(gpu_id, CU_CTX_SCHED_AUTO)) {
		fprintf(stderr, "error in GPU init.\n");
		exit(EXIT_FAILURE);
	}

        CUCHECK(cuStreamCreate(&gpu_stream, 0));

        puts("");
        printf("number iterations %d\n", num_iters);
        printf("num dwords per poke %zu\n", n_pokes);
        printf("use poll flush %d\n", use_flush);
        printf("use poke membar %d\n", use_membar);
        printf("use %d background streams\n", n_bg_streams);
        printf("sleep %dus\n", sleep_us);
        printf("buffer size %zd\n", size);
        printf("poll on %s buffer\n", use_gpu_buf?"GPU":"CPU");
        printf("write on %s buffer\n", use_gpu_buf?"GPU":"CPU");
        puts("");

        int mem_type = use_gpu_buf ? GDS_MEMORY_GPU : GDS_MEMORY_HOST;

        gds_mem_desc_t desc =  {0,};
        ret = gds_alloc_mapped_memory(&desc, size, mem_type);
        if (ret) {
                gpu_err("error (%d) while allocating mem\n", ret);
                goto out;
        }
        CUdeviceptr d_buf = desc.d_ptr;
        void *h_buf = desc.h_ptr;
        printf("allocated d_buf=%p h_buf=%p\n", (void*)d_buf, h_buf);
        memset(h_buf, 0, size);

        gds_mem_desc_t desc_data =  {0,};
        ret = gds_alloc_mapped_memory(&desc_data, size, mem_type);
        if (ret) {
                gpu_err("error (%d) while allocating mem\n", ret);
                goto out;
        }
        CUdeviceptr d_data = desc_data.d_ptr;
        uint32_t *h_data = desc_data.h_ptr;
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
#define CHUNK_SIZE 3
        int round;
        for (i = 0, value = 1; i < num_iters; ++i, ++value) {
                for (round = 0; round < 2; ++round) {
                        ASSERT(value <= INT_MAX);

                        uint32_t *h_signal = (uint32_t*)h_buf +  ((0) % (size/sizeof(uint32_t)));
                        uint32_t *d_signal = (uint32_t*)d_buf +  ((0) % (size/sizeof(uint32_t)));

                        uint32_t *h_done   = (uint32_t*)h_buf +  ((1) % (size/sizeof(uint32_t)));
                        uint32_t *d_done   = (uint32_t*)d_buf +  ((1) % (size/sizeof(uint32_t)));

                        uint32_t *h_dbg    = (uint32_t*)h_buf +  ((2) % (size/sizeof(uint32_t)));
                        uint32_t *d_dbg    = (uint32_t*)d_buf +  ((2) % (size/sizeof(uint32_t)));

                        uint32_t *h_vals   = (uint32_t*)h_data + ((i*CHUNK_SIZE) % (size/sizeof(uint32_t)));
                        uint32_t *d_vals   = (uint32_t*)d_data + ((i*CHUNK_SIZE) % (size/sizeof(uint32_t)));

                        uint32_t src_data[CHUNK_SIZE] = {1, 2, 3};

                        if (0 == round) {
                                gds_descriptor_t descs[10];
                                int k = 0;

                                descs[k].tag = GDS_TAG_WRITE_VALUE32;
                                descs[k].write32.ptr = use_gpu_buf ? d_dbg : h_dbg;
                                descs[k].write32.value = 0xA000|i;
                                descs[k].write32.flags = mem_type;
                                ++k;

                                //printf("%d: wait at %p for 0x%x\n", i, d_signal, value);
                                // wait for CPU signal
                                descs[k].tag = GDS_TAG_WAIT_VALUE32;
                                descs[k].wait32.ptr   = use_gpu_buf ? d_signal : h_signal;
                                descs[k].wait32.value = value;
                                descs[k].wait32.cond_flags = GDS_WAIT_COND_EQ;
                                descs[k].wait32.flags = poll_flags;
                                ++k;

                                descs[k].tag = GDS_TAG_WRITE_VALUE32;
                                descs[k].write32.ptr = use_gpu_buf ? d_dbg : h_dbg;
                                descs[k].write32.value = 0xB000|i;
                                descs[k].write32.flags = mem_type;
                                ++k;

#if 1
                                // d_vals[0] = 0
                                descs[k].tag = GDS_TAG_WRITE_VALUE32;
                                descs[k].write32.ptr = use_gpu_buf ? d_vals : h_vals;
                                descs[k].write32.value = 0;
                                descs[k].write32.flags = mem_type;
                                ++k;

                                // d_vals[0-2] = {1,2,3}
#if HAS_WRITE_MEMORY
                                descs[k].tag = GDS_TAG_WRITE_MEMORY;
                                descs[k].writemem.ptr = use_gpu_buf ? d_vals : h_vals;
                                descs[k].writemem.src = src_data;
                                descs[k].writemem.length = sizeof(src_data);
                                descs[k].writemem.flags = mem_type;
#else
                                descs[k].tag = GDS_TAG_WRITE_VALUE32;
                                descs[k].write32.ptr = use_gpu_buf ? d_vals : h_vals;
                                descs[k].write32.value = src_data[0];
                                descs[k].write32.flags = mem_type;
#endif
                                ++k;

                                // while (d_vals[0] != 1);
                                descs[k].tag = GDS_TAG_WAIT_VALUE32;
                                descs[k].wait32.ptr   = use_gpu_buf ? d_vals : h_vals;
                                descs[k].wait32.value = 0;
                                descs[k].wait32.cond_flags = GDS_WAIT_COND_EQ;
                                descs[k].wait32.flags = poll_flags;
                                ++k;
#endif
                                //printf("%d: write at %p 0x%x\n", i, d_done, value);
                                // signal CPU
                                descs[k].tag = GDS_TAG_WRITE_VALUE32;
                                descs[k].write32.ptr = use_gpu_buf ? d_done : h_done;
                                descs[k].write32.value = value;
                                descs[k].write32.flags = mem_type;
                                ++k;
        
                                ret = gds_stream_post_descriptors(gpu_stream, k, descs, 0);
                                if (ret)
                                        exit(EXIT_FAILURE);
                        }
                        else {
                                ASSERT(ACCESS_ONCE(*h_signal) == (value-1));
                                ASSERT(ACCESS_ONCE(*h_done) == (value-1));
                                
                                gpu_dbg("%d:       dbg @%p:%08x\n", i, h_dbg, ACCESS_ONCE(*h_dbg));
                                gpu_dbg("%d: set   sig @%p=%08x\n", i, h_signal, value);
                                gds_atomic_set_dword(h_signal, value);
                                gds_wmb();
                                gpu_dbg("%d:       sig @%p:%08x\n", i, h_signal, ACCESS_ONCE(*h_signal));
                                usleep(10);
                                gpu_dbg("%d:       dbg @%p:%08x\n", i, h_dbg, ACCESS_ONCE(*h_dbg));
                                gpu_dbg("%d:      done @%p:%08x\n", i, h_done, ACCESS_ONCE(*h_done));
                                gpu_dbg("%d: poll done @%p==%08x\n", i, h_done, value);
                                gds_us_t tmout = 1000; //usecs
                                int retcode = poll_dword_geq(h_done, value, tmout);
                                if (retcode == EWOULDBLOCK) {
                                        // normal behaviour
                                        gpu_dbg("%d: unblocking stream by writing %p <= %08x\n", i, h_vals, 0);
                                        gds_atomic_set_dword(h_vals, 0);
                                        retcode = poll_dword_geq(h_done, value, tmout);
                                        if (retcode) {
                                                gpu_err("got error %d, exiting\n", retcode);
                                                exit(EXIT_FAILURE);
                                        }
                                }
                                else if (retcode) {
                                        gpu_err("got error %d, exiting\n", retcode);
                                        exit(EXIT_FAILURE);
                                }
                                else {
                                        gpu_err("stream order violation\n");
                                        ++n_errors;
                                }
                        }
                        if (i % 1000 == 0) {
                                printf("."); fflush(stdout);
                        }
                }

        }

        if (n_errors) {
                gpu_err("detected n_errors=%d\n", n_errors);
        } else {
                //if (cuStreamQuery(gpu_stream) != CUDA_SUCCESS) {
                //        gpu_err("stream must be idle at this point, iter:%d\n", i);
                //        exit(EXIT_FAILURE);
                //}
                gpu_dbg("calling Stream Synchronize\n");
                CUCHECK(cuStreamSynchronize(gpu_stream));
        }
        perf_stop();

err:
        gpu_dbg("calling gds_free_mapped_memory\n");
        ret = gds_free_mapped_memory(&desc);
        if (ret) {
                gpu_err("error (%d) while freeing mem\n", ret);
                goto out;
        }

        ret = gds_free_mapped_memory(&desc_data);
        if (ret) {
                gpu_err("error (%d) while freeing mem\n", ret);
                goto out;
        }

out:
	gpu_finalize();

        printf("test finished!\n");
        return ret;
}

/*
 * Local variables:
 *  c-indent-level: 8
 *  c-basic-offset: 8
 *  tab-width: 8
 * End:
 */
