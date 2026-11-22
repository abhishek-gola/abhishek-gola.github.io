#pragma once
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <thread>
#include <vector>

// Valid, non-overlapping contiguous buffers; no concurrent external writes.
inline void copy_chunked(std::byte* dst, const std::byte* src,
                         std::size_t bytes, std::size_t workers) {
    constexpr std::size_t chunk = 64 * 1024;
    if (bytes == 0) return;
    if (bytes <= 2 * chunk || workers <= 1) {
        std::memcpy(dst, src, bytes);
        return;
    }
    const std::size_t tasks = bytes / chunk + (bytes % chunk != 0);
    workers = std::min(workers, tasks);
    std::vector<std::jthread> threads;
    threads.reserve(workers);
    for (std::size_t w = 0; w < workers; ++w) {
        threads.emplace_back([=] {
            for (std::size_t i = w; i < tasks; i += workers) {
                const std::size_t offset = i * chunk;
                std::memcpy(dst + offset, src + offset,
                            std::min(chunk, bytes - offset));
            }
        });
    }
    // jthread destructors join before this function returns.
}
