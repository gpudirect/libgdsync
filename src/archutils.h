#pragma once

// from Linux kernel

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
