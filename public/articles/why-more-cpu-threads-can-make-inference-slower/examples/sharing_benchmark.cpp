// Controlled dispatch probe plus fixed attention jobs. Standard C++20 only.
#include "sharing_reference.hpp"

int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "Usage: sharing_benchmark OUTPUT_DIRECTORY\n"; return 1; }
    const std::string dir = argv[1];
    std::ofstream csv(dir+"/sharing-samples.csv");
    if (!csv) throw std::runtime_error("Cannot open samples file");
    csv << "workload,workers,round,sample,us\n" << std::setprecision(12);
    std::ofstream meta(dir+"/sharing-environment.txt");
    meta << "Compiler: " << __VERSION__ << "\nC++: " << __cplusplus
         << "\nhardware_concurrency hint: " << std::thread::hardware_concurrency()
         << "\n10 shuffled rounds; 100 observations/block; minimum 250 ms and 20 warmup calls/block\n"
         << "Empty dispatch: zero tasks, all configured workers acknowledge completion\n"
         << "Attention: B=1 H=8 Lq=1 Dk=Dv=64; Lk=16,64,2048; 8 tasks\n"
         << "workers=0: inline attention, no pool\n"
         << "Timing: dispatch/control OR complete attention; creation and I/O excluded\n";

    meta << "caller_requested_qos: " << requested_qos() << '\n';
    std::vector<Attention> jobs;
    jobs.emplace_back("keys-16",1,16);
    jobs.emplace_back("keys-64",1,64);
    jobs.emplace_back("keys-2048",1,2048);
    std::vector<std::vector<float>> expected;
    for (auto& job : jobs) {
        // This probe tests up to ten workers; the earlier benchmark used eight.
        job.scratch.resize(10, std::vector<float>(job.keys));
        job.run_inline();
        expected.push_back(job.output);
    }
    // Record all 27 configurations; no configuration is selected after timing.
    std::vector<std::pair<int,std::size_t>> order;
    for (auto w : {1u,2u,4u,6u,8u,10u}) order.emplace_back(-1,w);
    for (int j=0; j<3; ++j)
        for (auto w : {0u,1u,2u,4u,6u,8u,10u}) order.emplace_back(j,w);
    std::mt19937 rng(20261123);
    const std::function<void(std::size_t,std::size_t)> empty = [](auto,auto) {};
    for (int round=0; round<10; ++round) {
        std::shuffle(order.begin(),order.end(),rng);
        for (auto [j,workers] : order) {
            std::unique_ptr<Pool> pool;
            if (workers) pool=std::make_unique<Pool>(workers);
            auto run = [&] {
                if (j<0) pool->run(0,empty);
                else if (pool) jobs[j].run(*pool);
                else jobs[j].run_inline();
            };
            warm_up(run);
            meta << "round=" << round << " workload=" << (j<0 ? "empty-dispatch" : jobs[j].name)
                 << " workers=" << workers << " caller_requested_qos=" << requested_qos() << '\n';
            if (pool) pool->record_qos(meta);
            for (int sample=0; sample<100; ++sample) {
                const auto start=std::chrono::steady_clock::now();
                run();
                const auto end=std::chrono::steady_clock::now();
                const double us=std::chrono::duration<double,std::micro>(end-start).count();
                if (j>=0 && jobs[j].output!=expected[j])
                    throw std::runtime_error("Attention mismatch");
                csv << (j<0 ? "empty-dispatch" : jobs[j].name) << ','
                    << workers << ',' << round << ',' << sample << ',' << us << '\n';
            }
        }
        std::cout << "sharing round " << round+1 << "/10 complete" << std::endl;
    }
    for (int j=0; j<3; ++j) {
        std::ofstream out(dir+"/sharing-"+jobs[j].name+"-output.txt");
        out << std::setprecision(9);
        for (auto x : expected[j]) out << x << '\n';
    }
    std::cout << "27,000 observations completed; all attention outputs matched inline.\n";
}
