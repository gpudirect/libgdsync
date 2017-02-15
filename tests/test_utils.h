#pragma once

#ifndef USE_PROF
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

#ifndef ACCESS_ONCE
#define ACCESS_ONCE(x) (*(volatile typeof(x) *)&(x))
#endif

typedef int64_t us_t;
static inline us_t gds_get_time_us()
{
        struct timespec ts;
        int ret = clock_gettime(CLOCK_MONOTONIC, &ts);
        if (ret) {
                fprintf(stderr, "error in gettime %d/%s\n", errno, strerror(errno));
                exit(EXIT_FAILURE);
        }
        return (us_t)ts.tv_sec * 1000 * 1000 + (us_t)ts.tv_nsec / 1000;
}

static inline void gds_cpu_relax(void)
{
        asm volatile("pause\n": : :"memory");
}

static inline void gds_busy_wait_us(us_t tmout)
{
        us_t start = gds_get_time_us();
        us_t tm = start + tmout;
        do {
                gds_cpu_relax();
        } while ((tm-gds_get_time_us()) > 0);
}

#define wmb()   asm volatile("sfence" ::: "memory")

void gds_atomic_set_dword(uint32_t *ptr, uint32_t value)
{
        ACCESS_ONCE(*ptr) = value;
        wmb();
}
