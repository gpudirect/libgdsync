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

#include <thread>
#include <condition_variable>
#include <mutex>
#include <list>
#include <functional>
#include <cstdio>

#define TQDBG(FMT, ARGS...)   do { } while(0)
//#define TQDBG(FMT, ARGS...)   do {                                    \
//        printf("%s() " FMT "\n", __FUNCTION__ ,##ARGS);               \
//    } while(0)

// single threaded task queue
struct task_queue {
    task_queue() {
        TQDBG("CTOR");
        m_state = idle;
    }

    ~task_queue() {
        TQDBG("DTOR");
        kill_worker();
    }

    typedef std::function<bool()> task_t;
    typedef std::list<task_t> task_list_t;
    typedef task_list_t::iterator task_handle_t;

    template <typename Task>
    task_handle_t queue(Task t) {
        lock_t lock(m_mtx);
        task_handle_t handle = m_tasks.insert(m_tasks.end(), t);
        if (m_state == idle) {
            m_worker = std::thread(std::bind(&task_queue::worker_thread, this));
            while (m_state == idle) {
                m_cond.wait(lock);
            }
        }
        m_cond.notify_one();
        return handle;
    }

    // WARNING: dequeuing can race with self-dequeing task
    void dequeue(task_handle_t handle) {
        TQDBG("dequing task");
        std::lock_guard<std::mutex> g(m_mtx);
        m_tasks.erase(handle);
    }

private:
    void kill_worker() {
        TQDBG("killing worker thread");
        {
            lock_t g(m_mtx);
            if (m_state == running) {
                TQDBG("setting queue state to draining");
                m_state = draining;
                m_cond.notify_one();
            }
        }
        TQDBG("joining worker thread");
        m_worker.join();
    }

    void worker_thread() {
        lock_t lock(m_mtx);
        TQDBG("state<-running");
        m_state = running;
        m_cond.notify_all();
        while(true) {
            if (m_state == draining) {
                TQDBG("state<-done");
                m_state = done;
                break;
            }
            if (m_tasks.empty())
                m_cond.wait(lock);
            auto f = m_tasks.begin();
            while (f != m_tasks.end()) {
                auto fnext = std::next(f);
                task_t t = *f;
                lock.unlock();
                bool keep = t();
                lock.lock();
                if (!keep) {
                    TQDBG("removing task");
                    m_tasks.erase(f);
                    TQDBG("done");
                }
                f = fnext;
            }
        }
    }

    enum { idle, running, draining, done } m_state;

    // single threaded queue
    std::thread m_worker;
    std::mutex m_mtx;
    typedef std::unique_lock<std::mutex> lock_t;
    task_list_t m_tasks;
    std::condition_variable m_cond;
};

#undef TQDBG

/*
 * Local variables:
 *  c-indent-level: 8
 *  c-basic-offset: 8
 *  tab-width: 8
 * End:
 */
