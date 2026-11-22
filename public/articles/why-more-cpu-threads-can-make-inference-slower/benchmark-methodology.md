# Sources, benchmark methods and reproduction

The article combines code-derived examples, an arbitrary cost model and new reference-attention measurements. Figure 1 explains copy partitioning, Figure 2 is conceptual, Figure 3 measures sharing costs on this device, Figure 4 contains the earlier attention sweep and Figure 5 illustrates serving topology.

## Original implementation and historical results

Abhishek Gola authored [OpenCV PR #28821](https://github.com/opencv/opencv/pull/28821), merged into 5.x as [d95badefa42cc24b6feaba3f4e7cdf469727a6ae](https://github.com/opencv/opencv/commit/d95badefa42cc24b6feaba3f4e7cdf469727a6ae). The article keeps the project name out of its prose while retaining attribution in references. Implementation links are pinned to the merge commit.

- **Concatenation**, `modules/dnn/src/layers/concat2_layer.cpp`: the old range was indexed by input, with `nstripes = totalSize > 1000000 ? ninputs : 1`. At or below 1,000,000 bytes, the one-stripe path executes serially. The revision uses chunks of up to 64 × 1024 bytes per input slice and enters its explicit parallel branch when `totalSize > CHUNK && totalChunks > 0`. For valid nonempty inputs, the positive chunk-count guard adds no useful prose qualification. Eligibility therefore expands to the interval above 64 KiB and at or below 1,000,000 bytes. Chunks cannot cross slice boundaries, so inner-axis concatenation may expose many tiny tasks.
- **Contiguous copying**, `modules/dnn/src/layers/layers_common.cpp`, `reshapeAndCopyFirst`: for distinct buffers, the revision asserts continuity and explicitly schedules chunks only when `bytes > 2 * CHUNK_BYTES`, where `CHUNK_BYTES = 64 * 1024`. Otherwise it retains `copyTo`. This cutoff is strictly above 128 KiB. It says nothing about all underlying copy routines’ internal threading.
- **Scope:** split uses `(slice, output)` tasks and removes its former size-based one-stripe setting. Transpose introduces an outer-range parallel loop without an explicit byte-size cutoff at the call site. Scheduler behavior can still constrain execution. These paths require separate analysis.

The article describes why finer partitioning can expose more parallel work. It does not claim the new gates were proved optimal. A 1 MiB contiguous buffer gives 16 chunks; a 128 KiB + 1 byte buffer gives three, with a one-byte tail. These are arithmetic consequences of the branch and ceiling division.

The historical table transcribes three CPU rows from the maintainer’s Ryzen x2700 baseline-versus-patch report in the [PR discussion](https://github.com/opencv/opencv/pull/28821). Units are milliseconds, reported as geometric means, with default CMake options. AlexNet is 11.633 → 11.967, YOLOv5 is 37.554 → 31.078, and YOLOv8 is 50.080 → 45.075. These are reported combined-patch results, including a regression, rather than newly collected measurements, a copy-only ablation or a thread sweep. The PR description’s cross-engine table is a different comparison and is not used as before/after evidence.

## Standalone copy code

The article’s code matches [examples/copy.hpp](examples/copy.hpp), apart from its include guard. It is a standalone C++20 adaptation using `std::jthread`, static chunk-index assignment and `std::memcpy`. Thread lifetimes and automatic joining are documented in the [C++20-era N4861 draft](https://timsong-cpp.github.io/cppwp/n4861/thread.jthread.class).

This version creates threads per call. The article warns about that before the code listing and retains the inherited 128 KiB cutoff only to explain the branch. A cutoff for newly created threads would need separate measurement. The copy example’s performance is not used in Figures 3 or 4.

Buffers must be valid, non-overlapping and contiguous, with external concurrent access controlled. Zero bytes return immediately; budgets of zero or one use the serial route. The companion checks zero bytes, exactly 128 KiB, 128 KiB + 1 byte and 1 MiB + 1 byte at budgets of zero, one, four and eight.

## Figure 3: dedicated sharing-cost probe

### What is measured

[examples/sharing_benchmark.cpp](examples/sharing_benchmark.cpp) measures an empty-dispatch control and three fixed attention shapes. It uses [examples/sharing_reference.hpp](examples/sharing_reference.hpp), a snapshot of the pool and attention classes from the earlier benchmark. The summarizer verifies that the snapshot matches the original definitions. This leaves the earlier run and its source hashes unchanged.

The probe tests widths one, two, four, six, eight and ten. Its attention jobs use batch one, eight heads, one query per head, key/value dimensions of 64 and key counts 16, 64 or 2,048. Each shape exposes eight tasks. Ten score-scratch buffers are preallocated before warm-up, extending the earlier example's eight-buffer capacity for the larger worker range. Worker participation is determined by dynamic task assignment.

The empty-dispatch control calls `Pool::run` with zero tasks. This implementation still notifies all configured workers, performs atomic task-index checks and waits for every worker's completion acknowledgment. That is the measured notification/completion path, with no attention arithmetic and no thread creation inside the timer. It is specific to this pool: a scheduler that short-circuits empty work or activates fewer workers has a different control path.

The attention measurements time the complete operation, including pool dispatch and completion. The inline reference calls the same attention task kernel sequentially. An empty-dispatch latency cannot be subtracted from an attention latency to identify compute time: the two operations can have different worker timing, cache state, scheduling and contention. Figure 3 therefore uses separate panels rather than an additive cost breakdown.

### Procedure and validation

All twenty-seven configurations are specified before timing: six empty-dispatch widths, plus seven execution configurations for each of the three attention shapes, including inline. Their order is shuffled across all workloads within each of ten rounds, with seed `20261123`. Every block gets twenty warm-up calls and one hundred individually timed observations. There are one thousand observations per configuration and twenty-seven thousand overall. Timing uses `steady_clock` with microsecond reporting. Pool creation, tensor/scratch allocation, validation and CSV output remain outside the timer.

The host is the same Apple M5, four performance plus six efficiency cores, 16 GB memory, macOS 26.6.2, built with Apple LLVM 21.0.0 and `-O3 -std=c++20 -pthread`. Default scheduling and power management remain in effect on a shared desktop. Per-core placement and CPU frequency are not recorded. This dedicated run is separate from the earlier fifteen-thousand-sample attention sweep, so its samples are not pooled with that run.

Before the retained run, [examples/check_sharing.cpp](examples/check_sharing.cpp) exercised all three shapes at every worker width, with twenty repetitions each, under AddressSanitizer. It checked attention output agreement and the empty-dispatch path. The check passed. An initial probe attempt exposed the eight-slot scratch limit at ten workers; after extending the probe's scratch allocation, the complete benchmark was rerun from the start. Only the completed corrected run is included.

Every timed attention result is checked against inline output after stopping the timer. [examples/summarize_sharing.py](examples/summarize_sharing.py) also regenerates Q/K/V and independently evaluates attention using float64 NumPy reductions. All outputs must be finite and match at `rtol=1e-5, atol=1e-6`. It validates all twenty-seven round/sample grids and summarizes medians, estimated p95 and block medians.

- [Raw samples](data/sharing-samples.csv).
- [Summary, validation results and source hashes](data/sharing-report.json).
- [Compiler, host and procedure](data/sharing-environment.txt).
- Reference outputs: [16 keys](data/sharing-keys-16-output.txt), [64 keys](data/sharing-keys-64-output.txt), [2,048 keys](data/sharing-keys-2048-output.txt).

Figure 3 plots pooled medians for empty dispatch and the 64-key attention job. Faint dots show ten block medians per pool width. The dashed line in the attention panel is its measured inline median. Connected lines guide the eye between tested widths; the faint dots are observed block summaries, not confidence intervals. The figure script recomputes median/p95 from the raw CSV and verifies the summary before plotting. The article's numbers are rounded from that report. The other two attention shapes remain in the raw data and report.

This probe demonstrates a dispatch-cost increase and a workload-dependent turnaround in this pool on this device. It does not identify the best width for a production runtime or separate the causes of every latency difference. The broader kernel and host limitations listed for Figure 4 also apply.

## Figure 4: measured attention example

### Operation and numerical checks

The attention definition follows [Attention Is All You Need, section 3.2.1](https://arxiv.org/html/1706.03762v7#S3.SS2.SSS1). Each query row computes scaled dot products over every key, stable softmax and the weighted value sum. Tasks own disjoint output rows and each worker has separate score scratch. Key tiling needs consistent normalization statistics; [FlashAttention](https://arxiv.org/html/2205.14135v2#S3.SS1) describes a correct online-softmax approach.

All cases use batch one, eight heads, `Dk = Dv = 64`, finite deterministic inputs, unmasked attention and no dropout. Storage is FP32 with double-precision dot-product/output accumulation. Inputs, outputs and per-worker scratch are preallocated and reused. Scratch holds one row per worker, rather than a full score matrix.

| Case | Query rows | Keys | Query tile | Tasks |
| --- | ---: | ---: | ---: | ---: |
| Short context | 1 | 16 | 32 | 8 |
| Long context | 1 | 2,048 | 32 | 8 |
| Prefill-shaped | 128 | 128 | 32 | 32 |

[examples/attention_benchmark.cpp](examples/attention_benchmark.cpp) contains the complete kernel and persistent pool. Inline mode calls the same task kernel sequentially on the caller. Pool mode uses exactly one, two, four or eight standard-library background workers, condition variables and an atomic task index. The caller waits. There are no BLAS worker pools underneath the kernel. Tasks are dynamically assigned; available tasks do not guarantee every worker participates in useful computation on each call.

C++ checks exact output agreement against inline execution after every measured call. [examples/summarize_attention.py](examples/summarize_attention.py) independently regenerates Q/K/V and evaluates float64 attention with NumPy reductions. All outputs must be finite and match at `rtol=1e-5, atol=1e-6`. These checks establish numerical agreement for the included shapes, not optimized-kernel equivalence across arbitrary inputs.

### Sampling, timing and host

The saved run was collected on 22 November 2026. For each workload, ten rounds shuffle `[inline, pool-1, pool-2, pool-4, pool-8]` with a seeded generator. Every block gets twenty warm-up calls and one hundred measured calls. That is one thousand observations per configuration and fifteen thousand overall. Workload order is fixed; configuration order is shuffled within each workload.

`workers=0` in the CSV explicitly encodes inline execution with no pool. Inline timing covers direct kernel execution. Pool timing covers dispatch, attention and completion using `std::chrono::steady_clock`. Thread construction/destruction and tensor/scratch allocation are outside the timer. Pool dispatch includes its function-wrapper bookkeeping. Output comparisons and CSV writes also occur after timing. Q/K/V projection, preprocessing, the remaining model and serving queueing are excluded.

The host was an Apple M5 with ten cores (four performance, six efficiency), 16 GB memory and macOS 26.6.2 (25G83), arm64. The chip and core split were verified using `system_profiler`. Build flags were `-O3 -std=c++20 -pthread`, using Apple LLVM 21.0.0. The standard library reported a hardware-concurrency hint of ten.

The desktop was shared, with default OS scheduling and power management. Worker-to-core placement and CPU frequency were not recorded. Core heterogeneity can affect scaling, but the data do not attribute a particular difference to performance or efficiency cores. macOS scheduling policy is left to the OS; no claim of hard per-core pinning is made. The ten blocks are within one run, not ten isolated machine-level replications.

### Data and reporting

- [Raw samples](data/attention-samples.csv), with workload, worker encoding, round and sample indices.
- [Summary report](data/attention-report.json), with pooled median/p95, round medians, numerical errors and source hashes.
- [Environment](data/attention-environment.txt).
- Saved reference outputs: [short context](data/attention-short-context-output.txt), [long context](data/attention-long-context-output.txt), [prefill-shaped](data/attention-prefill-output.txt).

Figure 4 recomputes medians and p95 from the CSV and checks them against the report. Gray bars show inline execution; blue bars show pools. Diamonds show estimated p95, and panels have separate scales. Values are descriptive summaries with no confidence intervals. The larger sample count improves percentile resolution but does not remove correlation, scheduling noise or the need for repeated deployment-level experiments.

Ratios in the article use unrounded medians and are rounded to one decimal place. The inline baseline and one-worker pool are distinct. The kernel is a scalar teaching example; a production attention implementation can have different blocking, vectorization, precision and scheduling costs. The unmasked prefill-shaped case is not a full causal prefill pipeline.

## Excluded earlier runtime benchmark

An earlier draft contained a separate OpenCV 4.12.0 experiment with requested budgets one and four on the GCD backend. The [4.12.0 source](https://github.com/opencv/opencv/blob/4.12.0/modules/core/src/parallel.cpp) stores the setting but does not configure a four-worker GCD cap. A value of one bypasses its native parallel branch; `getNumThreads` reports CPU count for GCD rather than measuring active workers.

That older experiment is excluded from the article package because it does not establish a controlled one-versus-four-worker comparison. The replacement C++ attention benchmark has explicit pool widths. The original experiment remains in the speaking-demo workspace.

The main article’s backend implementation link is pinned to the contribution’s merge commit; the excluded experiment’s link intentionally stays at 4.12.0. The [4.13.0 API documentation](https://docs.opencv.org/4.13.0/db/de0/group__core__utils.html) describes GCD support as nonpositive settings only. Backend details must be checked against the actual build and version, rather than inferred from a setter or getter name.

## Conceptual figures and general references

Figure 2 uses arbitrary functions for hypothetical worker count `P` from one to sixteen:

```text
divisible_work(P) = 0.4 / P
coordination(P)   = 0.025 × (P − 1)
combined_cost(P)  = divisible_work(P) + coordination(P)
```

The constants are unrelated to the measured attention data. The model assumes perfectly divisible work and linear coordination cost, omitting serial and extra memory-contention terms from the article’s broader bookkeeping equation. Its continuous curves and turning point have no hardware significance. The article’s inequality follows by subtracting two approximate costs with unchanged workload and serial work.

Figure 5 depicts independent serving workers with their own pools. Shared-pool designs can differ, and configured capacity does not establish simultaneous execution.

The [STREAM reference](https://www.cs.virginia.edu/stream/ref.html) supports `2N/t` effective-copy-bandwidth accounting and distinguishes requested traffic from hardware traffic. No STREAM results are reported. [ONNX Runtime threading](https://onnxruntime.ai/docs/performance/tune-performance/threading.html) and [profiling](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html) pages supply general documentation examples, separate from the original contribution and the standard-library benchmark. The Python timing helper is an independent serial measurement utility with illustrative defaults.

## Reproduction and publication packaging

Use a C++20 compiler/library supporting `std::jthread`, and Python 3.9 or later. On macOS or Linux, from this directory:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# Optional: collect a NEW run, replacing the saved attention measurements.
# Omit the next three commands to retain the article's saved run.
build_dir=$(mktemp -d)
c++ -O3 -std=c++20 -pthread examples/attention_benchmark.cpp -o "$build_dir/attention_benchmark"
"$build_dir/attention_benchmark" data

.venv/bin/python examples/summarize_attention.py
.venv/bin/python figures.py
```

To reproduce the sharing-cost probe separately, retaining the earlier attention measurements:

```sh
build_dir=$(mktemp -d)
c++ -O1 -g -fsanitize=address -std=c++20 -pthread examples/check_sharing.cpp -o "$build_dir/check_sharing"
"$build_dir/check_sharing"
c++ -O3 -std=c++20 -pthread examples/sharing_benchmark.cpp -o "$build_dir/sharing_benchmark"
"$build_dir/sharing_benchmark" data
.venv/bin/python examples/summarize_sharing.py
.venv/bin/python figures.py
```

The sanitizer executable checks correctness only; timings come from the optimized executable without sanitizer instrumentation. The sharing probe writes its own `sharing-*` data files. To regenerate all figures from saved data, run both summarizers and `figures.py`, without either C++ timing command.

A new benchmark run replaces the CSV, reference vectors and basic metadata. Add the host OS/hardware and build flags to the corresponding environment text file, then update the article’s table and interpretation from the new report. Keep new measurements separate from old numeric claims.

The figures script writes PNG/SVG versions of all five figures. Dependencies are pinned in `requirements.txt`; SVG text remains editable. The Markdown uses local images and links, so distribute the entire folder or ZIP to retain them. For a standalone blog platform, upload the assets and supporting package and replace local links with that platform’s final URLs. The website version resolves supporting links for browser use.
