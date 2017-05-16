#pragma once

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
        printf("Performance instrumentation is disabled\n");
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
                fprintf(stderr, "error in gettime %d/%s\n", errno, strerror(errno));
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

void gds_atomic_set_dword(uint32_t *ptr, uint32_t value)
{
        ACCESS_ONCE(*ptr) = value;
        gds_wmb();
}
