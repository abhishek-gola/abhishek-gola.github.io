#include "sharing_reference.hpp"
int main() {
    // Reject an unsafe caller modification before entering the worker pool.
    {
        Attention limited("guard",1,16);
        Pool oversized(10);
        bool rejected=false;
        try { limited.run(oversized); }
        catch (const std::invalid_argument&) { rejected=true; }
        if (!rejected) throw std::runtime_error("Missing scratch-width guard");
    }
    for (auto bytes : {std::size_t{0},std::size_t{131072},std::size_t{131073},std::size_t{1048577}}) {
        std::vector<std::byte> src(bytes),dst(bytes);
        for (std::size_t i=0; i<bytes; ++i) src[i]=std::byte(i%251);
        for (auto workers : {0u,1u,4u,8u}) {
            std::fill(dst.begin(),dst.end(),std::byte{255});
            copy_chunked(dst.data(),src.data(),bytes,workers);
            if (dst!=src) throw std::runtime_error("Copy mismatch");
        }
    }
    for (auto keys : {16u,64u,2048u}) {
        Attention job("safety",1,keys);
        job.scratch.resize(10,std::vector<float>(keys));
        job.run_inline();
        auto expected=job.output;
        for (auto workers : {1u,2u,4u,6u,8u,10u}) {
            Pool pool(workers);
            for (int repeat=0; repeat<20; ++repeat) {
                job.run(pool);
                if (job.output!=expected) throw std::runtime_error("Mismatch");
                pool.run(0,[](auto,auto) {});
                std::vector<int> visits(257,0);
                pool.run(visits.size(),[&](auto task,auto) { ++visits[task]; });
                if (!std::all_of(visits.begin(),visits.end(),[](int x) { return x==1; }))
                    throw std::runtime_error("Task count mismatch");
            }
        }
    }
    std::cout << "Copy, scratch guard, task-count and output checks passed.\n";
}
