The article has five figures: copy partitioning, a conceptual cost model, a measured sharing-cost probe, a three-workload attention sweep and a serving-topology diagram. Current measurements use a time-based warm-up. Initial published runs remain available for comparison.

## Implementation sources and reported patch results

Abhishek Gola authored [OpenCV PR #28821](https://github.com/opencv/opencv/pull/28821), merged into 5.x as [d95badefa42cc24b6feaba3f4e7cdf469727a6ae](https://github.com/opencv/opencv/commit/d95badefa42cc24b6feaba3f4e7cdf469727a6ae).

- **Concatenation**, `modules/dnn/src/layers/concat2_layer.cpp`: the old parallel range was indexed by input. Its `nstripes = totalSize > 1000000 ? ninputs : 1` requested a single work unit at or below 1,000,000 bytes, resulting in serial execution. The revision partitions each input slice into chunks of up to 64 KiB and enters its explicit parallel branch above 64 KiB. Chunks stop at slice boundaries, so inner-axis concatenation can expose many small tasks.
- **Contiguous copying**, `reshapeAndCopyFirst` in `layers_common.cpp`: distinct contiguous buffers use explicit chunk scheduling above 128 KiB, with chunks of up to 64 KiB. Smaller buffers retain `copyTo`.
- **Other paths:** split changes to `(slice, output)` tasks and removes its old size-based single-work-unit setting. Transpose introduces an outer-range parallel loop without a byte-size gate at the call site. Those changes are outside the article's copy example.

The [maintainer's report](https://github.com/opencv/opencv/pull/28821#issuecomment-4294666478) labels the host “Ryzen x2700” and says “default cmake options.” It contains 106 rows, including 37 CPU rows. The thread count is not specified. The label is retained rather than interpreted as a particular Ryzen model.

The article selects YOLOv5 and YOLOv8 for relevance to the contribution, plus AlexNet to include a regression. [All 37 CPU rows](/articles/why-more-cpu-threads-can-make-inference-slower/data/reported-patch-cpu-results.csv) and [source metadata](/articles/why-more-cpu-threads-can-make-inference-slower/data/reported-patch-source.json) are supplied. The values are reported geometric-mean milliseconds for the combined patch. They do not isolate a particular layer change or constitute a thread-count sweep.

## Standalone C++ and correctness checks

[examples/copy.hpp](/articles/why-more-cpu-threads-can-make-inference-slower/examples/copy.hpp) uses standard C++20 `std::jthread` and `std::memcpy`. Each worker handles disjoint chunk indices. It creates threads per call, with automatic joining described in the [N4861 draft](https://timsong-cpp.github.io/cppwp/n4861/thread.jthread.class). Its inherited 128 KiB gate illustrates the branch; a useful cutoff for fresh threads would require measurement.

[examples/check_sharing.cpp](/articles/why-more-cpu-threads-can-make-inference-slower/examples/check_sharing.cpp) covers:

- Copy sizes zero, exactly 128 KiB, 128 KiB + 1 byte and 1 MiB + 1 byte, with zero, one, four and eight requested workers. Source bytes use `i % 251`, so common wrong-offset copies produce different results. The destination begins with a different sentinel value.
- Rejection of a pool wider than the attention kernel's score-scratch allocation. `Attention::run(Pool&)` checks this before dispatch.
- Exactly-once execution of 257 task indices, empty dispatch completion and attention agreement with inline execution across one, two, four, six, eight and ten workers.

Separate AddressSanitizer and ThreadSanitizer builds passed these checks. AddressSanitizer checks memory accesses; ThreadSanitizer detects races in exercised execution paths. Neither is a proof for arbitrary callers or inputs. [Validation details and source hashes](/articles/why-more-cpu-threads-can-make-inference-slower/data/validation.json) record the tested code. Sanitizer binaries are never used for performance measurements.

## Attention kernel

The kernel evaluates [scaled dot-product attention](https://arxiv.org/html/1706.03762v7#S3.SS2.SSS1), with batch one, eight heads, `Dk = Dv = 64`, unmasked inputs and no dropout. Inputs and outputs use FP32 storage; dot products and output sums accumulate in double precision. Stable softmax subtracts the maximum score over the complete key dimension.

Each task owns a query tile within a head, and each worker owns a score buffer. Key tiling requires consistent normalization statistics, as in [online-softmax attention](https://arxiv.org/html/2205.14135v2#S3.SS1). The reference uses scalar loops, including strided value reads and per-term normalization in the output loop. It retains that loop order across these comparisons; optimized blocking, vectorization and normalization hoisting can change its timings.

The pool uses condition variables, an atomic task index and a completion acknowledgment from every configured worker. Inline mode executes the same task kernel on the caller. Pool mode leaves the caller waiting. Inputs, outputs and score scratch are preallocated and reused; no BLAS pool is involved.

## Current measurement procedure

Both current runs were collected on 22 November 2026 on an Apple M5 with four performance and six efficiency cores, 16 GB memory, macOS 26.6.2 (25G83), arm64. Apple LLVM 21.0.0 compiled the timed executables with `-O3 -std=c++20 -pthread`.

Each configuration has ten blocks of one hundred timed calls. Before **every block**, warm-up runs for at least **250 milliseconds and twenty calls**. This replaces the initial runs' call-count-only warm-up. All observations are retained. Warm-up reduces startup exposure but does not guarantee stationary clocks, thermal state or scheduling.

Pools are recreated outside timing for each block, then reused through warm-up and measurement. That design exposes variation between pool instances and placements. `steady_clock` measures each call. Pool creation, tensor/scratch allocation, output validation and CSV writes are outside timing. Dispatch and completion are inside pool timing; inline timing covers the direct kernel. Pool bookkeeping is included.

The desktop is shared, with default scheduling and power management. No QoS class is assigned by the benchmark. `pthread_get_qos_class_np` records requested classes: the caller reported `USER_INTERACTIVE` (33), and workers reported `DEFAULT` (21), in both current runs. These values are defined in the macOS SDK's `sys/qos.h`. Worker inheritance is not assumed. Requested QoS does not identify the physical core used; effective priority, frequency and core placement were not traced.

### Figure 3: sharing-cost probe

[examples/sharing_benchmark.cpp](/articles/why-more-cpu-threads-can-make-inference-slower/examples/sharing_benchmark.cpp) uses the pool and kernel snapshot in [examples/sharing_reference.hpp](/articles/why-more-cpu-threads-can-make-inference-slower/examples/sharing_reference.hpp). The summarizer verifies it against the definitions in the attention source. Ten score buffers support the probe's maximum width of ten.

The 27 configurations comprise:

- Empty dispatch at one, two, four, six, eight and ten workers.
- Attention with 16, 64 or 2,048 keys, each executed inline and at those six pool widths. All have one query per head and eight tasks.

All configurations are shuffled together within each round, using seed `20261123`. The complete run has 27,000 timed observations.

Empty dispatch calls this pool with zero tasks. Workers still receive notification, check the task index and acknowledge completion. It measures this implementation's coordination path without attention arithmetic. Schedulers that skip empty work have a different path. The attention panel measures a complete operation; the empty-dispatch time cannot be subtracted from it to identify compute time because work changes scheduling, cache state and participation.

Figure 3 shows pooled medians for empty dispatch and 64-key attention. Faint dots show ten block medians. Its dashed baseline is the measured inline median. Lines connect tested widths only. The other shapes remain in the report and raw samples.

- [Raw samples](/articles/why-more-cpu-threads-can-make-inference-slower/data/sharing-samples.csv), [report](/articles/why-more-cpu-threads-can-make-inference-slower/data/sharing-report.json), [environment and QoS records](/articles/why-more-cpu-threads-can-make-inference-slower/data/sharing-environment.txt).
- [Independent numerical check and summarizer](/articles/why-more-cpu-threads-can-make-inference-slower/examples/summarize_sharing.py).
- Reference vectors: [16 keys](/articles/why-more-cpu-threads-can-make-inference-slower/data/sharing-keys-16-output.txt), [64 keys](/articles/why-more-cpu-threads-can-make-inference-slower/data/sharing-keys-64-output.txt), [2,048 keys](/articles/why-more-cpu-threads-can-make-inference-slower/data/sharing-keys-2048-output.txt).

### Figure 4: attention sweep

[examples/attention_benchmark.cpp](/articles/why-more-cpu-threads-can-make-inference-slower/examples/attention_benchmark.cpp) tests these shapes:

| Case | Queries | Keys | Query tile | Tasks |
| --- | ---: | ---: | ---: | ---: |
| Short context | 1 | 16 | 32 | 8 |
| Long context | 1 | 2,048 | 32 | 8 |
| Prefill-shaped | 128 | 128 | 32 | 32 |

Each shape runs inline and in pools of one, two, four and eight workers. Configuration order is shuffled within each workload, with seed `20261122`; workload order is fixed. The run has 15,000 timed observations. CSV `workers=0` means inline execution with no pool.

Figure 4 bars show pooled medians, diamonds show pooled p95 and faint dots show block medians. The short-context panel uses microseconds; the other panels use milliseconds, with separate scales.

- [Raw samples](/articles/why-more-cpu-threads-can-make-inference-slower/data/attention-samples.csv), [report](/articles/why-more-cpu-threads-can-make-inference-slower/data/attention-report.json), [environment and QoS records](/articles/why-more-cpu-threads-can-make-inference-slower/data/attention-environment.txt).
- [Independent numerical check and summarizer](/articles/why-more-cpu-threads-can-make-inference-slower/examples/summarize_attention.py).
- Reference vectors: [short context](/articles/why-more-cpu-threads-can-make-inference-slower/data/attention-short-context-output.txt), [long context](/articles/why-more-cpu-threads-can-make-inference-slower/data/attention-long-context-output.txt), [prefill-shaped](/articles/why-more-cpu-threads-can-make-inference-slower/data/attention-prefill-output.txt).

Both C++ programs compare every timed attention result against inline output after timing. Both Python summarizers independently regenerate the inputs and evaluate float64 attention with NumPy reductions. All outputs must be finite and match at `rtol=1e-5, atol=1e-6`. They validate sample grids and numerical agreement, and record source hashes. The plotting script independently checks medians and p95 against raw samples.

## Block effects and repeated runs

Pooled p95 combines within-block variation with shifts between blocks. Since each block is ten percent of a configuration's samples, one uniformly slow block can determine its 95th percentile. Pool recreation, the caller's state and system activity can move a whole block. These data alone do not identify a cause.

In the initial attention run, short-context inline round zero had a median near 11.8 μs, versus approximately 4.9 μs later. The initial one-worker pool's first block was near 19.8 μs, versus roughly 7 μs later. Those blocks largely determine the corresponding pooled p95 values. A steady slow block does not establish whether a core type, clock state or another mechanism caused it.

Time-based warm-up does not eliminate the issue. In the current sharing run, one 64-key, one-worker block has a median near 65.2 μs while the other block medians range from about 28.0 to 30.9 μs; pooled p95 is about 65.9 μs. The block dots and retained samples make that state shift visible rather than filtering it out. Pooled p95 is not a steady-state per-call latency guarantee or a confidence interval.

The sharing probe repeats the short- and long-context shapes in the attention sweep. In the initial runs, 16-key inline medians were about 4.9 and 6.3 μs, while 2,048-key inline medians were about 1.30 and 1.54 ms. Qualitative scaling repeated, but absolute times and ratios changed. Current reruns preserve that distinction between replication of a qualitative result and precise deployment estimates.

The [initial summaries](/articles/why-more-cpu-threads-can-make-inference-slower/data/initial-summaries.json) and [initial published package](/articles/why-more-cpu-threads-can-make-inference-slower/data/initial-publication.zip) retain all earlier reports, raw samples and the exact source used for those timings. Current reports identify the revised sources. Samples from different runs are never pooled. The article's observed ratio range covers the two initial runs and two current reruns; it is an observed range, not a statistical uncertainty interval.

## Other figures and runtime references

Figure 1 illustrates code-derived partitioning: 1 MiB gives sixteen 64 KiB chunks; 128 KiB + 1 byte gives three, with a one-byte tail. Tasks and thread counts are separate quantities.

Figure 2 uses arbitrary functions, unrelated to measured data:

```text
divisible_work(P) = 0.4 / P
coordination(P)   = 0.025 × (P − 1)
combined_cost(P)  = divisible_work(P) + coordination(P)
```

The broader cost equation includes serial and memory-contention terms. Subtracting two approximate costs with unchanged serial work yields the article's inequality. The conceptual turning point has no hardware significance.

Figure 5 depicts independent serving workers with their own pools. Shared-pool implementations can behave differently.

The [STREAM reference](https://www.cs.virginia.edu/stream/ref.html) explains read-plus-write effective bandwidth accounting. [ONNX Runtime threading](https://onnxruntime.ai/docs/performance/tune-performance/threading.html) and [profiling](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html) pages supply general examples. The source runtime's [GCD thread-setting documentation](https://docs.opencv.org/4.13.0/db/de0/group__core__utils.html) concerns its own interface, rather than a limit on Apple's GCD API. Backend behavior must be checked for the actual runtime version and build.

## Reproduction

Run these commands from the **unzipped package directory**, `why-more-cpu-threads-can-make-inference-slower`. Use Python 3.9 or later and a C++20 compiler/library with `std::jthread`. On macOS or Linux:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
build_dir=$(mktemp -d)

# Correctness diagnostics: run separately from timing.
c++ -O1 -g -fsanitize=address -std=c++20 -pthread examples/check_sharing.cpp -o "$build_dir/check_asan"
"$build_dir/check_asan"
c++ -O1 -g -fsanitize=thread -std=c++20 -pthread examples/check_sharing.cpp -o "$build_dir/check_tsan"
"$build_dir/check_tsan"

# Optional: these replace the CURRENT saved measurements with new runs.
c++ -O3 -std=c++20 -pthread examples/attention_benchmark.cpp -o "$build_dir/attention"
"$build_dir/attention" data
c++ -O3 -std=c++20 -pthread examples/sharing_benchmark.cpp -o "$build_dir/sharing"
"$build_dir/sharing" data

.venv/bin/python examples/summarize_attention.py
.venv/bin/python examples/summarize_sharing.py
.venv/bin/python figures.py
```

To retain the supplied measurements, omit the two optimized compilation/execution pairs. Run only the summarizers and figure script to regenerate plots from saved data. Sanitizer support varies by platform; the supplied validation was performed on this Mac. QoS recording is macOS-specific and reports unavailable elsewhere.

After collecting a new run, record its host, compiler flags and conditions in the corresponding environment file and update article values from the new reports. The original archive remains separate. PNG and editable SVG versions of all five figures are generated. Keep the whole folder together for relative links, or use the published article and hosted supporting files.
