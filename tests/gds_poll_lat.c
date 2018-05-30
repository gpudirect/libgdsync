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
#include <gdrapi.h>

#include "test_utils.h"
#include "gpu.h"

struct prof prof;
int prof_idx = 0;

int main(int argc, char *argv[])
{
        int ret = 0;
	int gpu_id = 0;
        int num_iters = 1000;
        // this seems to minimize polling time
        int sleep_us = 10;
	size_t page_size = sysconf(_SC_PAGESIZE);
	size_t size = 1024*64;
        int use_gpu_buf = 0;
        int use_flush = 0;
        int use_combined = 0;
        int use_membar = 0;
        int use_wrmem = 0;
        int wait_key = -1;
        //CUstream gpu_stream;

        int n_bg_streams = 0;

        size_t n_pokes = 1;

        while(1) {            
                int c;
                c = getopt(argc, argv, "cd:p:n:s:hfgP:mW:w");
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
                        gpu_info("enabling flush\n");
                        break;
                case 'g':
                        use_gpu_buf = 1;
                        gpu_info("polling on GPU buffer\n");
                        break;
                case 'w':
                        use_wrmem = 1;
                        gpu_info("enabling use of WRITE_MEMORY\n");
                        break;
                case '?':
                case 'h':
                        printf(" %s [-n <iters>][-s <sleep us>][-p # bg streams][-P # pokes][ckhfgomW]\n", argv[0]);
                        exit(EXIT_SUCCESS);
                        break;
                default:
                        gpu_err("invalid option '%c'\n", c);
                        exit(EXIT_FAILURE);
                }
        }

        if (n_pokes < 1) {
                gpu_err("n_pokes must be 1 at least\n");
                exit(EXIT_FAILURE);
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
                gpu_err("error in prof_init init.\n");
		exit(EXIT_FAILURE);
	}

	if (gpu_init(gpu_id, CU_CTX_SCHED_AUTO)) {
		gpu_err("error in GPU init.\n");
		exit(EXIT_FAILURE);
	}

        //CUCHECK(cuStreamCreate(&gpu_stream, 0));

        gpu_info("number iterations %d\n", num_iters);
        gpu_info("num dwords per poke %zu\n", n_pokes);
        gpu_info("use poll flush %d\n", use_flush);
        gpu_info("use poke membar %d\n", use_membar);
        gpu_info("use %d background streams\n", n_bg_streams);
        gpu_info("sleep %dus\n", sleep_us);
        gpu_info("buffer size %zd\n", size);
        gpu_info("poll on %s buffer\n", use_gpu_buf?"GPU":"CPU");
        gpu_info("write on %s buffer\n", use_gpu_buf?"GPU":"CPU");

        gds_mem_desc_t desc =  {0,};
        ret = gds_alloc_mapped_memory(&desc, size, use_gpu_buf?GDS_MEMORY_GPU:GDS_MEMORY_HOST);
        if (ret) {
                gpu_err("error (%d) while allocating mem\n", ret);
                goto out;
        }
        CUdeviceptr d_buf = desc.d_ptr;
        void *h_buf = desc.h_ptr;
        gpu_info("allocated d_buf=%p h_buf=%p\n", (void*)d_buf, h_buf);
        memset(h_buf, 0, size);

        gds_mem_desc_t desc_data =  {0,};
        ret = gds_alloc_mapped_memory(&desc_data, size, use_gpu_buf?GDS_MEMORY_GPU:GDS_MEMORY_HOST);
        if (ret) {
                gpu_err("error (%d) while allocating mem\n", ret);
                goto out;
        }
        CUdeviceptr d_data = desc_data.d_ptr;
        uint32_t *h_data = desc_data.h_ptr;
        gpu_info("allocated d_data=%p h_data=%p\n", (void*)d_data, h_data);
        memset(h_data, 0, size);

        int i;
        int value;
        int poll_flags = GDS_MEMORY_HOST;
        if (use_gpu_buf)
                poll_flags = GDS_MEMORY_GPU;
        if (use_flush)
                poll_flags |= GDS_WAIT_POST_FLUSH;

        uint32_t *h_bg_buf = NULL;
        if (n_bg_streams) {
                gpu_info("launching background %dx poll\n", n_bg_streams);
                ASSERT(!posix_memalign((void*)&h_bg_buf, page_size, size)); 
                memset(h_bg_buf, 0, size);
                for (i=0; i<n_bg_streams; ++i) {
                        CUCHECK(cuStreamCreate(&bg_streams[i], 0));
                        //gpu_post_poll_dword_on_stream(str[i], h_bg_buf+i, 1, GDS_WAIT_COND_GEQ, GDS_MEMORY_HOST);
                        //GDSCHECK(gds_stream_post_poll_dword(bg_streams[i], h_bg_buf+i, 1, GDS_WAIT_COND_GEQ, GDS_MEMORY_HOST));
                        gds_descriptor_t desc;
                        desc.tag = GDS_TAG_WAIT_VALUE32;
                        GDSCHECK(gds_prepare_wait_value32(&desc.wait32, h_bg_buf+i, 1, GDS_WAIT_COND_GEQ, GDS_MEMORY_HOST));
                        GDSCHECK(gds_stream_post_descriptors(bg_streams[i], 1, &desc, 0));
                }
        }

        gpu_info("starting test...\n");
        perf_start();
        gds_us_t delta_t = 0;
        int warmup = 5;
	for (i = 0, value = 1; i < num_iters; ++i, ++value) {
                ASSERT(value <= INT_MAX);
                uint32_t *h_ptr = (uint32_t*)h_buf + (i % (size/sizeof(uint32_t)));
                uint32_t *d_ptr = (uint32_t*)d_buf + (i % (size/sizeof(uint32_t)));

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
                gds_descriptor_t descs[n_pokes+1];
                uint32_t *poke_hptrs[n_pokes];
                static int j = 1;
                int k;

                descs[0].tag = GDS_TAG_WAIT_VALUE32;
                GDSCHECK(gds_prepare_wait_value32(&descs[0].wait32, use_gpu_buf ? d_ptr : h_ptr, value, GDS_WAIT_COND_GEQ, poll_flags));

                for (k=0; k<n_pokes; ++k) {
                        size_t off = ((k+i*n_pokes) % (size/sizeof(uint32_t)));
                        int dflags = use_gpu_buf ? GDS_MEMORY_GPU : GDS_MEMORY_HOST;
                        uint32_t *ptr = (use_gpu_buf ? (uint32_t*)(d_data) : h_data) + off;
                        if (use_membar && (k==n_pokes-1))
                                dflags |= GDS_WRITE_PRE_BARRIER;

                        if (!use_wrmem) {
                                gpu_dbg("poke %d WRITE_VALUE32\n", k);
                                descs[1+k].tag = GDS_TAG_WRITE_VALUE32;
                                GDSCHECK(gds_prepare_write_value32(&descs[1+k].write32,
                                                                   ptr,
                                                                   0xd4d00000|(j<<8)|k,
                                                                   dflags));
                        } else {
                                gpu_dbg("poke %d WRITE_MEMORY\n", k);
                                descs[1+k].tag = GDS_TAG_WRITE_MEMORY;
                                uint32_t word = 0xd4d00000|(j<<8)|k;
                                GDSCHECK(gds_prepare_write_memory(&descs[1+k].writemem,
                                                                  (uint8_t*)ptr,
                                                                  (uint8_t*)&word,
                                                                  sizeof(word),
                                                                  dflags));
                        }

                        poke_hptrs[k] =  h_data  + off;
                        ACCESS_ONCE(*poke_hptrs[k]) = 0;
                        ++j;
                        //printf("%d %d %p\n", i, k, poke_dptrs[k]);
                }

                if (use_combined) {
                        GDSCHECK(gds_stream_post_descriptors(gpu_stream, 1+n_pokes, descs, 0));
                        PROF(&prof, prof_idx++);
                } else {
                        // splitting submission into two chunks
                        GDSCHECK(gds_stream_post_descriptors(gpu_stream, 1, descs, 0));
                        PROF(&prof, prof_idx++);
                        GDSCHECK(gds_stream_post_descriptors(gpu_stream, n_pokes, descs+1, 0));
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
                gds_us_t end = start;
                while(1) {
                        uint32_t value = ACCESS_ONCE(*poke_hptrs[n_pokes-1]);
                        gpu_dbg("h_poke[%zu]=%x\n", n_pokes-1, value);
                        if (value) 
                                break;
                        end = gds_get_time_us();
                        if (end - start > tout) {
                                gpu_warn("timeout %ldus reached!!\n", tout);
                                goto err;
                        }
                        gds_cpu_relax();
                }
		PROF(&prof, prof_idx++);
                // CUDA synchronize
		//gpu_wait_kernel();
                CUCHECK(cuStreamSynchronize(gpu_stream));
		PROF(&prof, prof_idx++);
		prof_update(&prof);
		prof_idx = 0;
                
                if (i > warmup) {
                        delta_t += end - start;
                }
	}
        gpu_info("test finished!\n");

        if (num_iters > warmup) {
                double avg_wait_us = (double)delta_t / (double)(num_iters - warmup);
                printf("sleep time: %d   average wait time: %fus\n", sleep_us, avg_wait_us);
        }

        perf_stop();
        prof_dump(&prof);
err:
        if (n_bg_streams) {
                gpu_info("signaling %d background polling stream(s)\n", n_bg_streams);
                int s;
                for (s=0; s<n_bg_streams; ++s) {
                        if (cuStreamQuery(bg_streams[s]) == CUDA_SUCCESS) {
                                gpu_info("error: bg stream[%d] is idle!\n", s);
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
        gpu_info("calling gds_free_mapped_memory\n"); //fflush(stdout); sleep(1);
        GDSCHECK(gds_free_mapped_memory(&desc));
        GDSCHECK(ret = gds_free_mapped_memory(&desc_data));

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
