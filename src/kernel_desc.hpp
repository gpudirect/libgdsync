/* Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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
static struct kernel_desc MERGE(KRN,_desc);          \
void * MERGE(KRN,force_link)(void) { }               \
struct MERGE(KRN,ctor) {                             \
    MERGE(KRN,ctor)() {                              \
        MERGE(KRN,_desc).name=STRINGIFY(KRN);        \
        MERGE(KRN,_desc).n_cubins=0;                 \
        memset(MERGE(KRN,_desc).cubins, 0, sizeof(MERGE(KRN,_desc).cubins));    \
        MERGE(KRN,_desc).fatbin=NULL;

//    desc.cubins[desc.n_cubins] = (struct kernel_cubin) \ //DEFINE_KERNEL_ARCH_DESC_ENTRY(KRN, ARCH);
#define KERNEL_CUBIN(KRN, ARCH)                                         \
        MERGE(KRN,_desc).cubins[MERGE(KRN,_desc).n_cubins] = (struct kernel_cubin) \
        {                                                               \
            STRINGIFY(ARCH),                                            \
            MAJOR_FROM_ARCH(ARCH),                                      \
            MINOR_FROM_ARCH(ARCH),                                      \
            DEFINE_KERNEL_BY_ARCH(KRN,MERGE(sm,ARCH))                   \
        };                                                              \
        ++MERGE(KRN,_desc).n_cubins;                                    \
        assert(MERGE(KRN,_desc).n_cubins <= max_kernel_cubins);

#define KERNEL_FATBIN(KRN)                                              \
        MERGE(KRN,_desc).fatbin = DEFINE_KERNEL_BY_ARCH(KRN, fatbin);

#define KERNEL_DESC_END(KRN)                               \
        register_kernel(&MERGE(KRN,_desc));                \
    }                                                      \
} MERGE(KRN,ctor_inst);

#define KERNEL_FORCE_LINK(KRN) \
    extern void *MERGE(KRN,force_link)(void); \
    volatile void* c = MERGE(KRN,force_link)(); \


/*
 * Local variables:
 *  c-indent-level: 8
 *  c-basic-offset: 8
 *  tab-width: 8
 *  indent-tabs-mode: nil
 * End:
 */
