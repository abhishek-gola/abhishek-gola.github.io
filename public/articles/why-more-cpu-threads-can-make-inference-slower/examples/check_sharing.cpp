#include "sharing_reference.hpp"
int main() {
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
            }
        }
    }
    std::cout << "All worker widths passed AddressSanitizer and output checks.\n";
}
