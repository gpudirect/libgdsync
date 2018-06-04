/* Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
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

// work around missing std::this_thread::sleep_until() in gcc-4.4.7 (RHL 6.x)
#ifndef _GLIBCXX_USE_NANOSLEEP
#define _GLIBCXX_USE_NANOSLEEP
#endif

#include <cstdio>
//#include <cstdlib>
#include <chrono>

#include "wq.hpp"

#if 0
#define msg(FMT, ARGS...)   do {                           \
                printf("%s() " FMT, __FUNCTION__ ,##ARGS); \
        } while(0)
#else
#define msg(FMT, ARGS...)   do { } while(0)
#endif

bool simple_fun()
{
        static int j = 0;
        msg("%s j=%d\n", __FUNCTION__, j);
        return ++j < 1000 ? true : false;
}

struct fast_printer {
        fast_printer() : m_k(0) {}

        bool run() {
                msg("%p fast_printer k=%d\n", this, m_k);
                return ++m_k < 2000 ? true : false;
        }
        int m_k;
};

struct sticky_printer {
        sticky_printer() : m_s(0) {}

        bool run() {
                msg("%p sticky_printer s=%d\n", this, m_s++);
                return true;
        }
        int m_s;
};

struct slow_printer {
        slow_printer(unsigned num_iterations=100) : m_i(0), m_ni(num_iterations), m_store_time(true) {}

        bool run() {
                auto now = std::chrono::system_clock::now();
                if (m_store_time) {
                        msg("storing now\n");
                        m_last_time = now;
                        m_store_time = false;
                }
        
                if (now > m_last_time + std::chrono::milliseconds(1)) {
                        m_last_time = now;
                        msg("%p slow_printer i=%d\n", this, m_i);
                        std::this_thread::sleep_for(std::chrono::microseconds(1));
                        if ( ++m_i < m_ni ) {
                                return true;
                        } else {
                                msg("slow_printer done!");
                                return false;
                        }
                }
        }
        int m_i;
        int m_ni;
        bool m_store_time;
        std::chrono::system_clock::time_point m_last_time;
};


main()
{
        {
                work_queue wq;
                int i = 0;
                
                slow_printer s;
                wq.queue(std::bind(&slow_printer::run, &s));

                wq.queue(&simple_fun);
                fast_printer f;
                wq.queue(std::bind(&fast_printer::run, &f));

                std::this_thread::sleep_for(std::chrono::microseconds(10));

                do {
                        sticky_printer sf;
                        auto fp = std::bind(&sticky_printer::run, &sf);
                        auto h = wq.queue(fp);
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                        wq.dequeue(h);
                } while (++i<10);
        }
        fflush(stdout);
        printf("SUCCESS\n");
}

/*
 * Local variables:
 *  c-indent-level: 8
 *  c-basic-offset: 8
 *  tab-width: 8
 * End:
 */
