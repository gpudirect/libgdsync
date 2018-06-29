#pragma once

#include <cstring>
#include <cassert>

struct kernel_cubin {
    const char *arch;
    int major;
    int minor;
    const unsigned char *cubin;
    //const int *p_cubin_size;
};

enum { max_kernel_cubins = 10 };

struct kernel_desc {
    const char *name;
    int n_cubins;
    struct kernel_cubin cubins[max_kernel_cubins];
    const unsigned char *fatbin;
};

int register_kernel(struct kernel_desc *desc);

#ifndef MERGE
#define MERGE_2(MAJOR,MINOR) MAJOR##MINOR
#define MERGE(MAJOR,MINOR) MERGE_2(MAJOR,MINOR)
#endif

#ifndef STRINGIFY
#define STRINGIFY2(STR) #STR
#define STRINGIFY(STR) STRINGIFY2(STR)
#endif

#define DEFINE_KERNEL_BY_ARCH2(KRN,ARCH) KRN##_##ARCH
#define DEFINE_KERNEL_BY_ARCH(KRN,ARCH) DEFINE_KERNEL_BY_ARCH2(KRN,ARCH)

#define DECLARE_CUBIN_KERNEL(KRN, ARCH)                                 \
    extern const char DEFINE_KERNEL_BY_ARCH(KRN, MERGE(sm,ARCH))[];     \

#define DECLARE_FATBIN_KERNEL(KRN)                                      \
    extern const char DEFINE_KERNEL_BY_ARCH(KRN, fatbin)[];             \

#define MAJOR_FROM_ARCH(ARCH)                   \
    ((ARCH)/10)

#define MINOR_FROM_ARCH(ARCH)                   \
    ((ARCH)%10)

#define DEFINE_KERNEL_ARCH_DESC_ENTRY(KRN, ARCH)                        \
    {                                                                   \
        STRINGIFY(ARCH),                                                \
        MAJOR_FROM_ARCH(ARCH),                                          \
        MINOR_FROM_ARCH(ARCH),                                          \
        DEFINE_KERNEL_BY_ARCH(KRN,MERGE(sm,ARCH))                       \
    }

#define DEFINE_KERNEL_FATBIN_DESC_ENTRY(KRN)      \
    { NULL, 0, 0, DEFINE_KERNEL_BY_ARCH(KRN,fatbin) }


#define KERNEL_DESC_BEGIN(KRN)                       \
static kernel_desc MERGE(KRN,desc);                  \
struct MERGE(KRN,ctor) {                             \
    MERGE(KRN,ctor)() {                              \
        MERGE(KRN,desc).name=STRINGIFY(KRN);         \
        MERGE(KRN,desc).n_cubins=0;                  \
        memset(MERGE(KRN,desc).cubins, 0, sizeof(MERGE(KRN,desc).cubins));    \
        MERGE(KRN,desc).fatbin=NULL;

//    desc.cubins[desc.n_cubins] = (struct kernel_cubin) \ //DEFINE_KERNEL_ARCH_DESC_ENTRY(KRN, ARCH);
#define KERNEL_CUBIN(KRN, ARCH)                                         \
    MERGE(KRN,desc).cubins[MERGE(KRN,desc).n_cubins] = (struct kernel_cubin) \
    {                                                                   \
        STRINGIFY(ARCH),                                                \
        MAJOR_FROM_ARCH(ARCH),                                          \
        MINOR_FROM_ARCH(ARCH),                                          \
        DEFINE_KERNEL_BY_ARCH(KRN,MERGE(sm,ARCH))                       \
    };                                                                  \
    ++MERGE(KRN,desc).n_cubins;                                         \
    assert(MERGE(KRN,desc).n_cubins <= max_kernel_cubins);

#define KERNEL_FATBIN(KRN)                                              \
    MERGE(KRN,desc).fatbin = DEFINE_KERNEL_BY_ARCH(KRN, fatbin);

#define KERNEL_DESC_END(KRN)                               \
    register_kernel(&MERGE(KRN,desc));                     \
    }                                                      \
} MERGE(KRN,ctor_inst);
