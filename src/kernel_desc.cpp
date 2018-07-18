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

#if HAVE_CONFIG_H
#  include <config.h>
#endif /* HAVE_CONFIG_H */

#include <string>
#include <cassert>
#include <map>
#include <cuda.h>
#include "gdsync.h"
#include "objs.hpp"
#include "utils.hpp"
#include "kernel_desc.hpp"

using namespace std;

#include <time.h>

static double microseconds(struct timespec* ts)
{
    return (double)ts->tv_sec*1000000.0 + (double)ts->tv_nsec / 1000.0;
}

typedef map<string, struct kernel_desc*> kernel_list_t;

static kernel_list_t &kernel_list()
{
    static kernel_list_t kl;
    return kl;
}

int register_kernel(struct kernel_desc *desc)
{
    int ret = 0;
    gds_dbg("registering kernel:%s desc=%p\n", desc->name, desc);
    auto p = kernel_list().insert(kernel_list_t::value_type(string(desc->name),desc));
    if (!p.second) {
        gds_err("cannot insert kernel %s in tracking list\n", desc->name);
        ret = -1;
    }
    return ret;
}

static CUmodule load_module(const unsigned char *bin_image)
{
    CUmodule module;
    struct timespec start, end;
    assert(0 == clock_gettime(CLOCK_MONOTONIC, &start));
    CUCHECK( cuModuleLoadData(&module, bin_image) );
    assert(0 == clock_gettime(CLOCK_MONOTONIC, &end));
    double usecs = microseconds(&end) - microseconds(&start);
    gds_dbg("loading module took %.2f usecs\n", usecs);
    return module;
}

static CUmodule load_fatmodule(const unsigned char *bin_image)
{
    CUmodule module;
    struct timespec start, end;
    assert(0 == clock_gettime(CLOCK_MONOTONIC, &start));
    CUCHECK( cuModuleLoadFatBinary(&module, bin_image) );
    assert(0 == clock_gettime(CLOCK_MONOTONIC, &end));
    double usecs = microseconds(&end) - microseconds(&start);
    gds_dbg("loading module took %.2f usecs\n", usecs);
    return module;
}

// leaking module
// TODO: track and eventually free modules

CUfunction gds_load_kernel(int arch_major, int arch_minor, const char *kernel_name, bool force_fatbin)
{
    CUmodule module = NULL;
    CUfunction function = NULL;

    gds_dbg("searching for kernel:%s for sm:%d.%d\n", kernel_name, arch_major, arch_minor);

    auto it = kernel_list().find(kernel_name);
    if (it != kernel_list().end()) {
        kernel_desc *desc = (*it).second;
        if (!force_fatbin) {
            for (int i=0; i<desc->n_cubins; ++i) {
                kernel_cubin *adesc = &desc->cubins[i];
                //cout << "kernel sm:" << adesc->major << adesc->minor << endl;
                if (adesc->major == arch_major && adesc->minor == arch_minor) {
                    //cout << "loading cubin for sm" << adesc->arch << endl;
                    module = load_module(adesc->cubin);
                    break;
                }
            }
        }
        if (!module) {
            //cout << "loading fatbin" << endl;
            module = load_fatmodule(desc->fatbin);
        }
    }

    if (module) {
        gds_dbg("module=%p kernel_name=%s\n", module, kernel_name);
        struct timespec start, end;
        assert(0 == clock_gettime(CLOCK_MONOTONIC, &start));
        CUCHECK( cuModuleGetFunction(&function, module, kernel_name) );
        assert(0 == clock_gettime(CLOCK_MONOTONIC, &end));
        double usecs = microseconds(&end) - microseconds(&start);
        gds_dbg("loading function %s took %.2f usecs\n", kernel_name, usecs);
    } else {
        gds_err("cannot find module matching kernel %s\n", kernel_name);
    }

    return function;
}

void foobar() {
    KERNEL_FORCE_LINK(krn1snd2wait);
    KERNEL_FORCE_LINK_2(krnsetsndpar);
}
