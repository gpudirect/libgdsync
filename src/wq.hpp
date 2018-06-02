#pragma once

#include <thread>
#include <condition_variable>
#include <mutex>
#include <list>
#include <functional>


// single threaded work queue
struct work_queue {
    work_queue() {
        m_state = idle;
    }

    ~work_queue() {
        kill_worker();
    }

    template <typename Task>
    void queue(Task t) {
        lock_t lock(m_mtx);
        m_tasks.push_back(t);
        if (m_state == idle) {
            m_worker = std::thread(std::bind(&work_queue::worker_thread, this));
            while (m_state == idle) {
                m_cond.wait(lock);
            }
        }
        m_cond.notify_one();
    }

    template <typename Task>
    void dequeue(Task &t) {
        std::lock_guard<std::mutex> g(m_mtx);
        m_tasks.remove(t);
    }

private:
    void kill_worker() {
        puts("signaling worker thread");
        {
            lock_t g(m_mtx);
            if (m_state == running) {
                puts("state<-draining");
                m_state = draining;
                m_cond.notify_one();
            }
        }
        puts("joining");
        m_worker.join();
    }

    void worker_thread() {
        lock_t lock(m_mtx);
        puts("state<-running");
        m_state = running;
        m_cond.notify_all();
        while(true) {
            if (m_state == draining) {
                puts("state<-done");
                m_state = done;
                break;
            }
            if (m_tasks.empty())
                m_cond.wait(lock);
            auto f = m_tasks.begin();
            while (f != m_tasks.end()) {
                lock.unlock();
                task_t t = *f;
                bool keep = t();
                lock.lock();
                if (!keep) {
                    printf("removing task\n");
                    f = m_tasks.erase(f);
                    printf("done\n");
                } else {
                    ++f;
                }
            }
        }
    }

    enum { idle, running, draining, done } m_state;

    // single threaded queue
    std::thread m_worker;
    std::mutex m_mtx;
    typedef std::unique_lock<std::mutex> lock_t;
    typedef std::function<bool()> task_t;
    std::list<task_t> m_tasks;
    std::condition_variable m_cond;
};

/*
 * Local variables:
 *  c-indent-level: 8
 *  c-basic-offset: 8
 *  tab-width: 8
 * End:
 */
