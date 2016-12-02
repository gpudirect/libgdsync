// TODO: ADD LICENCE HERE

#pragma once

// freely inspired from Linux kernel x86/inclide/asm/processor.h et al

#if defined(__x86_64__) || defined (__i386__)

static inline void arch_cpu_relax(void)
{
        asm volatile("pause\n": : :"memory");
}

#define mb()    asm volatile("mfence":::"memory")
#define rmb()   asm volatile("lfence":::"memory")
#define wmb()   asm volatile("sfence" ::: "memory")

#elif defined(__powerpc__)

static void arch_cpu_relax(void) __attribute__((unused)) ;
static void arch_cpu_relax(void)
{
}

static void wmb(void) __attribute__((unused)) ;
static void wmb(void) 
{
	asm volatile("sync") ; 
}
static void rmb(void) __attribute__((unused)) ;
static void rmb(void) 
{
	asm volatile("sync") ; 
}

#else
#error "platform not supported"
#endif

/*
 * Local variables:
 *  c-indent-level: 8
 *  c-basic-offset: 8
 *  tab-width: 8
 *  indent-tabs-mode: nil
 * End:
 */
