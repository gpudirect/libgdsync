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
#include <gdrapi.h>

#ifdef USE_PROF
#include "prof.h"
#endif
struct prof prof;
int prof_idx = 0;

#include "test_utils.h"
#include "gpu.h"


int main(int argc, char *argv[])
{
        int ret = 0;
	int gpu_id = 0;
        int num_iters = 1000;
        // this seems to minimize polling time
        int sleep_us = 10;
	size_t page_size = sysconf(_SC_PAGESIZE);
	size_t size = 1024*4;
        int use_gpu_buf = 0;
        int use_flush = 0;
        int use_combined = 0;
        int use_membar = 0;
        int wait_key = -1;
        CUstream gpu_stream;

        int use_bg_poll = 0;
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
                        use_bg_poll = 1;
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

        //if (use_combined && use_pokes) {
        //        fprintf(stderr, "error, incompatible switches\n");
	//	exit(EXIT_FAILURE);
        //}
        const char *tags = "postpoll|que poke|   sleep|  set dw|pollpoke|str sync";
        if ( /*prof_init(&prof, 1000, 1000, "1ms", 50, 1, tags)*/
                prof_init(&prof, 100, 100, "100ns", 25*4*2, 5, tags)) {
                fprintf(stderr, "error in prof_init init.\n");
		exit(EXIT_FAILURE);
	}

	if (gpu_init(gpu_id, CU_CTX_SCHED_AUTO)) {
		fprintf(stderr, "error in GPU init.\n");
		exit(EXIT_FAILURE);
	}

        CUCHECK(cuStreamCreate(&gpu_stream, 0));

        puts("");
        printf("number iterations %d\n", num_iters);
        printf("use poll flush %d\n", use_flush);
        printf("use poke membar %d\n", use_membar);
        printf("use %d background streams\n", n_bg_streams);
        printf("sleep %dus\n", sleep_us);
        printf("buffer size %zd\n", size);
        printf("poll on %s buffer\n", use_gpu_buf?"GPU":"CPU");
        printf("write on CPU buffer\n");
        puts("");

        gds_mem_desc_t desc =  {0,};
        ret = gds_alloc_mapped_memory(&desc, size, use_gpu_buf?GDS_MEMORY_GPU:GDS_MEMORY_HOST);
        if (ret) {
                gpu_err("error (%d) while allocating mem\n", ret);
                goto out;
        }
        CUdeviceptr d_buf = desc.d_ptr;
        void *h_buf = desc.h_ptr;
        printf("allocated d_buf=%lx h_buf=%p\n", (unsigned long)d_buf, h_buf);
        memset(h_buf, 0, size);

        gds_mem_desc_t desc_data =  {0,};
        ret = gds_alloc_mapped_memory(&desc_data, size, GDS_MEMORY_HOST);
        if (ret) {
                gpu_err("error (%d) while allocating mem\n", ret);
                goto out;
        }
        CUdeviceptr d_data = desc_data.d_ptr;
        void *h_data = desc_data.h_ptr;
        memset(h_data, 0, size);

        int i;
        int value;
        int poll_flags = GDS_MEMORY_HOST;
        if (use_gpu_buf)
                poll_flags = GDS_MEMORY_GPU;
        if (use_flush)
                poll_flags |= GDS_POLL_POST_FLUSH;

        uint32_t *h_bg_buf = NULL;
        if (use_bg_poll) {
                printf("launching background %dx poll\n", n_bg_streams);
                ASSERT(!posix_memalign((void*)&h_bg_buf, page_size, size)); 
                memset(h_bg_buf, 0, size);
                for (i=0; i<n_bg_streams; ++i) {
                        CUCHECK(cuStreamCreate(&bg_streams[i], 0));
                        //gpu_post_poll_dword_on_stream(str[i], h_bg_buf+i, 1, GDS_POLL_COND_GEQ, GDS_MEMORY_HOST);
                        gds_stream_post_poll_dword(bg_streams[i], h_bg_buf+i, 1, GDS_POLL_COND_GEQ, GDS_MEMORY_HOST);
                }
        }
        printf("starting test...\n");

	for (i = 0, value = 1; i < num_iters; ++i, ++value) {
                ASSERT(value <= INT_MAX);
                uint32_t *h_ptr = (uint32_t*)h_buf + (i % (size/sizeof(uint32_t)));
                uint32_t *d_ptr = (uint32_t*)d_buf + (i % (size/sizeof(uint32_t)));
                if (!use_gpu_buf) // use UVA host pointer as polling target
                        d_ptr = h_ptr;
                // post a sema.acquire on SYS/VIDMEM
                gpu_dbg("GEQ h_ptr=%p d_ptr=%p *h_ptr=%08x i=%d value=%d\n", h_ptr, d_ptr, *h_ptr, i, value);
                if (!use_gpu_buf) {
                        int c;
                        if (wait_key>=0 && i==wait_key) {
                                puts("press any key");
                                c = getchar();
                        }
                }
		PROF(&prof, prof_idx++);

                assert(n_pokes < size/sizeof(uint32_t));
                uint32_t *poke_ptrs[n_pokes];
                uint32_t  poke_values[n_pokes];
                int       poke_flags[n_pokes];
                static int j = 1;
                int k;
                for (k=0; k<n_pokes; ++k) {
                        poke_ptrs[k] = h_data+k;
                        poke_values[k] = 0xd4d00000|(j<<8)|k;
                        poke_flags[k] = GDS_MEMORY_HOST;
                        poke_flags[k] = (use_membar && k==n_pokes-1)?GDS_POKE_POST_PRE_BARRIER:0;
                        ACCESS_ONCE(*poke_ptrs[k]) = 0;
                        ++j;
                }
                uint32_t *poll_ptrs[1] = { d_ptr };
                uint32_t  poll_values[1] = { value };
                int       poll_cond_flags[1] = { GDS_POLL_COND_GEQ };
                int       poll_flagses[1] = { poll_flags };

                if (use_combined) {
                        gds_stream_post_polls_and_pokes(gpu_stream, 
                                                        1, poll_ptrs, poll_values, poll_cond_flags, poll_flagses, 
                                                        n_pokes, poke_ptrs, poke_values, poke_flags);
                        PROF(&prof, prof_idx++);
                } else {
                        gds_stream_post_poll_dword(gpu_stream,
                                                   d_ptr, value, GDS_POLL_COND_GEQ, poll_flags);
                        PROF(&prof, prof_idx++);
                        gds_stream_post_polls_and_pokes(gpu_stream, 
                                                        0, 0, 0, 0, 0, 
                                                        n_pokes, poke_ptrs, poke_values, poke_flags);
                }
		PROF(&prof, prof_idx++);

                if (use_gpu_buf) {
                        int c;
                        if (wait_key>=0 && i==wait_key) {
                                puts("press any key");
                                c = getchar();
                        }
                }
                // CPU waits some time here to make sure the previous commands
                // were actually fetched by the GPU
                gpu_dbg("sleeping %d us\n", sleep_us);
                gds_busy_wait_us(sleep_us);
                // stream should still be blocked in the wait operation
                if (cuStreamQuery(gpu_stream) == CUDA_SUCCESS) {
                        gpu_err("error, stream must NOT be idle at this point, iter:%d\n", i);
                        exit(EXIT_FAILURE);
                }
		PROF(&prof, prof_idx++);
                // CPU writes to SYS/VIDMEM, triggering the GPU acquire, 
                // which should trigger execution past the sema acquire
                gpu_dbg("writing h_ptr=%p value=%08x\n", h_ptr, value);
		gds_atomic_set_dword(h_ptr, value);
		PROF(&prof, prof_idx++);
                // CPU polling on zero-copy SYSMEM
                //if (use_pokes || use_combined)
                //        ret = gpu_poll_pokes();
                //else
                //        ret = gpu_poll_poke();
                gds_us_t tout = 100;
                gds_us_t start = gds_get_time_us();
                gds_us_t tmout = start + tout;
                while(1) {
                        uint32_t value = ACCESS_ONCE(*poke_ptrs[n_pokes-1]);
                        gpu_dbg("h_poke[%zu]=%x\n", n_pokes-1, value);
                        if (value) 
                                break;
                        // time-out check
                        if ((gds_get_time_us()-start) > (long)tmout) {
                                gpu_warn("timeout %ldus reached!!\n", tout);
                                goto err;
                        }
                        //arch_cpu_relax();
                }
		PROF(&prof, prof_idx++);
                // CUDA synchronize
		//gpu_wait_kernel();
                CUCHECK(cuStreamSynchronize(gpu_stream));
		PROF(&prof, prof_idx++);
		prof_update(&prof);
		prof_idx = 0;
	}
        prof_dump(&prof);
err:
        if (use_bg_poll) {
                printf("signaling %d background polling stream(s)\n", n_bg_streams);
                int s;
                for (s=0; s<n_bg_streams; ++s) {
                        if (cuStreamQuery(bg_streams[s]) == CUDA_SUCCESS) {
                                printf("error: bg stream[%d] is idle!\n", s);
                        }
                        gds_atomic_set_dword(h_bg_buf+s, 1);
                }
                gds_busy_wait_us(100); // wait for poll to trigger
                for (s=0; s<n_bg_streams; ++s) {
                        CUCHECK(cuStreamSynchronize(bg_streams[s]));
                        CUCHECK(cuStreamDestroy(bg_streams[s]));
                }
                free(h_bg_buf);
        }

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
        return ret;
}

/*
 * Local variables:
 *  c-indent-level: 8
 *  c-basic-offset: 8
 *  tab-width: 8
 * End:
 */
