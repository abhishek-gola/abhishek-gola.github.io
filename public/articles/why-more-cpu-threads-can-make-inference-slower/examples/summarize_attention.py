"""Validate attention against NumPy and summarize the saved C++ measurements."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'


def values(shape, phase):
    i = np.arange(np.prod(shape), dtype=np.int64)
    return (((i * 17 + phase) % 101 - 50).astype(np.float32) / np.float32(50)).reshape(shape)


checks = []
for name, nq, nk in [('short-context', 1, 16), ('long-context', 1, 2048), ('prefill', 128, 128)]:
    q = values((8, nq, 64), 3).astype(np.float64)
    k = values((8, nk, 64), 7).astype(np.float64)
    v = values((8, nk, 64), 11).astype(np.float64)
    scores = np.einsum('hqd,hkd->hqk', q, k, optimize=False) / np.sqrt(64)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    expected = np.einsum('hqk,hkd->hqd', weights / weights.sum(axis=-1, keepdims=True), v, optimize=False)
    actual = np.loadtxt(DATA / f'attention-{name}-output.txt').reshape(expected.shape)
    assert np.isfinite(expected).all() and np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    checks.append({'workload': name, 'max_abs_error': float(np.abs(actual-expected).max())})

with (DATA / 'attention-samples.csv').open() as f:
    rows = list(csv.DictReader(f))
summaries = []
for name in ['short-context', 'long-context', 'prefill']:
    for workers in [0, 1, 2, 4, 8]:
        selected = [r for r in rows if r['workload'] == name and int(r['workers']) == workers]
        assert len(selected) == 1000
        samples = np.array([float(r['ms']) for r in selected])
        assert np.isfinite(samples).all() and (samples > 0).all()
        assert {(int(r['round']), int(r['sample'])) for r in selected} == {(r,s) for r in range(10) for s in range(100)}
        summaries.append({'workload': name, 'workers': workers, 'n': len(samples),
                          'median_ms': float(np.median(samples)),
                          'p95_ms': float(np.percentile(samples,95)),
                          'round_median_ms': [float(np.median([float(r['ms']) for r in selected if int(r['round'])==rnd])) for rnd in range(10)]})
report = {'scope': 'inline attention OR persistent-pool dispatch + attention + completion wait',
          'inline_encoding': 'workers=0 means execution on caller, no pool',
          'rounds': 10, 'samples_per_block': 100, 'warmups_per_block': 20,
          'reference_rtol': 1e-5, 'reference_atol': 1e-6, 'correctness': checks,
          'benchmark_sha256': hashlib.sha256((ROOT/'examples/attention_benchmark.cpp').read_bytes()).hexdigest(),
          'copy_header_sha256': hashlib.sha256((ROOT/'examples/copy.hpp').read_bytes()).hexdigest(),
          'summaries': summaries}
(DATA/'attention-report.json').write_text(json.dumps(report,indent=2)+'\n')
for row in summaries:
    print(f"{row['workload']:14} workers={row['workers']} median={row['median_ms']:.6f} ms p95={row['p95_ms']:.6f} ms")
print('Independent NumPy reference checks passed:', checks)
