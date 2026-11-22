// C++20, standard library only. Correctness-oriented CPU attention example.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "copy.hpp"
#ifdef __APPLE__
#include <pthread/qos.h>
#endif

inline int requested_qos() {
#ifdef __APPLE__
    qos_class_t value; int priority=0;
    if (pthread_get_qos_class_np(pthread_self(), &value, &priority)==0)
        return static_cast<int>(value);
#endif
    return -1;
}

template<class Run>
void warm_up(Run&& run) {
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(250);
    std::size_t calls=0;
    do { run(); ++calls; }
    while (calls<20 || std::chrono::steady_clock::now()<deadline);
}

// Fixed background-worker count; the calling thread waits rather than computes.
// run() calls must not overlap; task functions must not throw.
class Pool {
    std::mutex mutex_;
    std::condition_variable start_, done_;
    bool stop_ = false;
    std::size_t generation_ = 0, remaining_ = 0, tasks_ = 0;
    std::atomic<std::size_t> next_{0};
    std::function<void(std::size_t, std::size_t)> job_;
    std::vector<std::jthread> threads_;
    std::vector<int> qos_;
public:
    explicit Pool(std::size_t workers) : qos_(workers,-1) {
        if (!workers) throw std::invalid_argument("workers must be positive");
        try {
            for (std::size_t w = 0; w < workers; ++w) {
                threads_.emplace_back([this, w] {
                    qos_[w]=requested_qos();
                    std::size_t seen = 0;
                    for (;;) {
                        std::unique_lock lock(mutex_);
                        start_.wait(lock, [&] { return stop_ || generation_ != seen; });
                        if (stop_) return;
                        seen = generation_;
                        lock.unlock();
                        for (;;) {
                            const auto task = next_.fetch_add(1, std::memory_order_relaxed);
                            if (task >= tasks_) break;
                            job_(task, w);
                        }
                        lock.lock();
                        if (--remaining_ == 0) done_.notify_one();
                    }
                });
            }
        } catch (...) {
            { std::lock_guard lock(mutex_); stop_ = true; }
            start_.notify_all();
            threads_.clear();
            throw;
        }
    }
    std::size_t size() const { return threads_.size(); }
    void record_qos(std::ostream& out) const {
        out << "worker_requested_qos:";
        for (int value : qos_) out << ' ' << value;
        out << '\n';
    }
    void run(std::size_t tasks,
             const std::function<void(std::size_t, std::size_t)>& job) {
        std::unique_lock lock(mutex_);
        tasks_ = tasks;
        job_ = job;
        next_ = 0;
        remaining_ = threads_.size();
        ++generation_;
        start_.notify_all();
        done_.wait(lock, [&] { return remaining_ == 0; });
    }
    ~Pool() {
        { std::lock_guard lock(mutex_); stop_ = true; }
        start_.notify_all();
        threads_.clear(); // Join while mutexes and the job remain alive.
    }
};

struct Attention {
    std::string name;
    std::size_t heads = 8, queries, keys, dim = 64, tile = 32;
    std::vector<float> q, k, v, output;
    std::vector<std::vector<float>> scratch;
    Attention(std::string label, std::size_t nq, std::size_t nk)
        : name(std::move(label)), queries(nq), keys(nk),
          q(heads*nq*dim), k(heads*nk*dim), v(heads*nk*dim),
          output(heads*nq*dim), scratch(8, std::vector<float>(nk)) {
        // Portable, deterministic inputs. Finite, moderate-magnitude values.
        auto fill = [](auto& a, int phase) {
            for (std::size_t i=0; i<a.size(); ++i)
                a[i] = float(int((i*17+phase)%101)-50) / 50.0f;
        };
        fill(q, 3); fill(k, 7); fill(v, 11);
    }
    std::size_t blocks() const { return (queries+tile-1)/tile; }
    std::size_t tasks() const { return heads*blocks(); }
    void task(std::size_t id, std::size_t worker) {
        const auto head = id / blocks(), block = id % blocks();
        auto& scores = scratch[worker];
        for (std::size_t row=block*tile; row<std::min(queries,(block+1)*tile); ++row) {
            for (std::size_t j=0; j<keys; ++j) {
                double sum=0;
                for (std::size_t d=0; d<dim; ++d)
                    sum += double(q[(head*queries+row)*dim+d])*k[(head*keys+j)*dim+d];
                scores[j] = float(sum/std::sqrt(double(dim)));
            }
            const float max_score = *std::max_element(scores.begin(),scores.end());
            double denom=0;
            for (auto& value : scores) {
                value = float(std::exp(double(value-max_score)));
                denom += value;
            }
            for (std::size_t d=0; d<dim; ++d) {
                double sum=0;
                for (std::size_t j=0; j<keys; ++j)
                    sum += (double(scores[j])/denom)*v[(head*keys+j)*dim+d];
                output[(head*queries+row)*dim+d] = float(sum);
            }
        }
    }
    void run_inline() {
        for (std::size_t t=0; t<tasks(); ++t) task(t,0);
    }
    void run(Pool& pool) {
        if (pool.size()>scratch.size())
            throw std::invalid_argument("pool width exceeds score-scratch slots");
        pool.run(tasks(),[this](auto task_id, auto worker) { task(task_id,worker); });
    }
};

int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "Usage: attention_benchmark OUTPUT_DIRECTORY\n"; return 1; }
    const std::string dir=argv[1];
    std::ofstream csv(dir+"/attention-samples.csv");
    if (!csv) throw std::runtime_error("Cannot open output file");
    csv << "workload,workers,round,sample,ms\n" << std::setprecision(12);
    std::ofstream meta(dir+"/attention-environment.txt");
    meta << "Compiler: " << __VERSION__ << "\nC++: " << __cplusplus
         << "\nhardware_concurrency hint: " << std::thread::hardware_concurrency()
         << "\nB=1 H=8 Dk=Dv=64; query tile=32; unmasked; no dropout\n"
         << "10 shuffled rounds, 100 samples/block, minimum 250 ms and 20 warmup calls/block\n"
         << "Timing: inline attention OR persistent-pool dispatch + attention + completion wait\n";

    meta << "caller_requested_qos: " << requested_qos() << '\n';

    // Copy correctness: empty, boundary, partial tail and worker budgets.
    for (auto bytes : {std::size_t{0},std::size_t{131072},std::size_t{131073},std::size_t{1048577}}) {
        std::vector<std::byte> src(bytes), dst(bytes);
        for (std::size_t i=0; i<bytes; ++i) src[i]=std::byte(i%251);
        for (auto w : {0u,1u,4u,8u}) {
            std::fill(dst.begin(),dst.end(),std::byte{0});
            copy_chunked(dst.data(),src.data(),bytes,w);
            if (src != dst) throw std::runtime_error("copy mismatch");
        }
    }
    std::vector<Attention> cases;
    cases.emplace_back("short-context",1,16);
    cases.emplace_back("long-context",1,2048);
    cases.emplace_back("prefill",128,128);
    std::mt19937 rng(20261122);
    for (auto& attention : cases) {
        // An independent NumPy check reads these outputs after the run.
        attention.run_inline();
        const auto reference=attention.output;
        std::vector<std::size_t> order{0,1,2,4,8}; // 0 means inline, no pool.
        for (int round=0; round<10; ++round) {
            std::shuffle(order.begin(),order.end(),rng);
            for (const auto workers : order) {
                std::unique_ptr<Pool> pool;
                if (workers) pool = std::make_unique<Pool>(workers);
                auto run = [&] {
                    if (pool) attention.run(*pool);
                    else attention.run_inline();
                };
                // Pool creation and warm-up are outside the measured interval.
                warm_up(run);
                meta << "round=" << round << " workload=" << attention.name
                     << " workers=" << workers << " caller_requested_qos=" << requested_qos() << '\n';
                if (pool) pool->record_qos(meta);
                for (int sample=0; sample<100; ++sample) {
                    const auto start=std::chrono::steady_clock::now();
                    run();
                    const auto end=std::chrono::steady_clock::now();
                    const double ms=std::chrono::duration<double,std::milli>(end-start).count();
                    if (attention.output != reference) throw std::runtime_error("parallel mismatch");
                    csv << attention.name << ',' << workers << ',' << round << ',' << sample << ',' << ms << '\n';
                }
            }
            std::cout << attention.name << " round " << round+1 << "/10 complete" << std::endl;
        }
        std::ofstream out(dir+"/attention-"+attention.name+"-output.txt");
        out << std::setprecision(9);
        for (auto value : reference) out << value << '\n';
        std::cout << attention.name << ": passed all worker-count checks\n";
    }
}
