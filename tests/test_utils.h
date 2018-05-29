#pragma once

#include "gpu.h"

#ifdef USE_PROF
#include "prof.h"
#else
struct prof { };
#define PROF(P, H) do { } while(0)
static inline int prof_init(struct prof *p, int unit_scale, int scale_factor, const char* unit_scale_str, int nbins, int merge_bins, const char *tags) {return 0;}
static inline int prof_destroy(struct prof *p) {return 0;}
static inline void prof_dump(struct prof *p) {}
static inline void prof_update(struct prof *p) {}
static inline void prof_enable(struct prof *p) {}
static inline int  prof_enabled(struct prof *p) { return 0; }
static inline void prof_disable(struct prof *p) {}
static inline void prof_reset(struct prof *p) {}
#endif

#if defined(USE_PERF)
#include "perf.h"
#else
static int perf_start()
{
        gpu_warn("Performance instrumentation is disabled\n");
        return 0;
}
static int perf_stop()
{
        return 0;
}
#endif

typedef int64_t gds_us_t;
static inline gds_us_t gds_get_time_us()
{
        struct timespec ts;
        int ret = clock_gettime(CLOCK_MONOTONIC, &ts);
        if (ret) {
                gpu_err("error in gettime %d/%s\n", errno, strerror(errno));
                exit(EXIT_FAILURE);
        }
        return (gds_us_t)ts.tv_sec * 1000 * 1000 + (gds_us_t)ts.tv_nsec / 1000;
}

#if defined(__x86_64__) || defined (__i386__)

static inline void gds_cpu_relax(void)
{
        asm volatile("pause": : :"memory");
}

#define gds_wmb()   asm volatile("sfence" ::: "memory")

#elif defined(__powerpc__)
static void gds_cpu_relax(void) __attribute__((unused)) ;
static void gds_cpu_relax(void)
{
}

static void gds_wmb(void) __attribute__((unused)) ;
static void gds_wmb(void) 
{
	asm volatile("sync") ; 
}
#else
#error "platform not supported"
#endif

static inline void gds_busy_wait_us(gds_us_t tmout)
{
        gds_us_t start = gds_get_time_us();
        gds_us_t tm = start + tmout;
        do {
                gds_cpu_relax();
        } while ((tm-gds_get_time_us()) > 0);
}

#ifndef ACCESS_ONCE
#define ACCESS_ONCE(x) (*(volatile typeof(x) *)&(x))
#endif

static inline void gds_atomic_set_dword(uint32_t *ptr, uint32_t value)
{
        ACCESS_ONCE(*ptr) = value;
        //gds_wmb();
}

static inline uint32_t gds_atomic_read_dword(uint32_t *ptr)
{
        return ACCESS_ONCE(*ptr);
}

static inline int gds_poll_dword(uint32_t *ptr, uint32_t payload, gds_us_t tm, int (*pred)(uint32_t a, uint32_t b))
{
        gds_us_t start = gds_get_time_us();
        int ret = 0;
        while(1) {
                uint32_t value = gds_atomic_read_dword(ptr);
                gpu_dbg("value=%x\n", value);
                if (pred(value, payload)) {
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

static inline int gds_geq_dword(uint32_t a, uint32_t b)
{
        return ((int32_t)a - (int32_t)b >= 0);
}

static int gds_poll_dword_geq(uint32_t *ptr, uint32_t payload, gds_us_t tm)
{
        return gds_poll_dword(ptr, payload, tm, gds_geq_dword);
}

static inline int gds_neq_dword(uint32_t a, uint32_t b)
{
        return (int32_t)a != (int32_t)b;
}

static int gds_poll_dword_neq(uint32_t *ptr, uint32_t payload, gds_us_t tm)
{
        return gds_poll_dword(ptr, payload, tm, gds_neq_dword);
}

/*
 * Local variables:
 *  c-indent-level: 8
 *  c-basic-offset: 8
 *  tab-width: 8
 * End:
 */
