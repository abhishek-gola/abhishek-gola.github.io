"""Validate the sharing probe's outputs and summarize its measured samples."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
source = (ROOT/'examples/attention_benchmark.cpp').read_text()
snapshot = (ROOT/'examples/sharing_reference.hpp').read_text()
assert snapshot.endswith(source[:source.index('int main(')])


def inputs(shape, phase):
    i = np.arange(np.prod(shape), dtype=np.int64)
    return (((i*17+phase)%101-50).astype(np.float32)/np.float32(50)).reshape(shape).astype(np.float64)


checks = []
for keys in [16,64,2048]:
    q = inputs((8,1,64),3)
    k = inputs((8,keys,64),7)
    v = inputs((8,keys,64),11)
    scores = np.einsum('hqd,hkd->hqk',q,k,optimize=False)/np.sqrt(64)
    weights = np.exp(scores-scores.max(axis=-1,keepdims=True))
    expected = np.einsum('hqk,hkd->hqd',weights/weights.sum(axis=-1,keepdims=True),v,optimize=False)
    actual = np.loadtxt(DATA/f'sharing-keys-{keys}-output.txt').reshape(expected.shape)
    assert np.isfinite(expected).all() and np.isfinite(actual).all()
    np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-6)
    checks.append({'keys':keys,'max_abs_error':float(np.abs(actual-expected).max())})

with (DATA/'sharing-samples.csv').open() as f:
    samples = list(csv.DictReader(f))
assert len(samples)==27000
summaries = []
for workload in ['empty-dispatch','keys-16','keys-64','keys-2048']:
    widths = [1,2,4,6,8,10] if workload=='empty-dispatch' else [0,1,2,4,6,8,10]
    for workers in widths:
        rows=[r for r in samples if r['workload']==workload and int(r['workers'])==workers]
        assert len(rows)==1000
        assert {(int(r['round']),int(r['sample'])) for r in rows}=={(a,b) for a in range(10) for b in range(100)}
        values=np.array([float(r['us']) for r in rows])
        assert np.isfinite(values).all() and (values>0).all()
        summaries.append({'workload':workload,'workers':workers,'n':len(rows),
                          'median_us':float(np.median(values)),
                          'p95_us':float(np.percentile(values,95)),
                          'round_median_us':[float(np.median([float(r['us']) for r in rows if int(r['round'])==a])) for a in range(10)]})
report={'rounds':10,'observations_per_block':100,'minimum_warmup_calls':20,'minimum_warmup_ms_per_block':250,
        'scope':'empty-dispatch control and total attention latency, measured separately',
        'interpretation':'empty dispatch is a separate control, not an additive component inferred from attention latency',
        'source_sha256':{name:hashlib.sha256((ROOT/'examples'/name).read_bytes()).hexdigest() for name in ['sharing_benchmark.cpp','sharing_reference.hpp','check_sharing.cpp']},
        'reference_rtol':1e-5,'reference_atol':1e-6,'correctness':checks,'summaries':summaries}
(DATA/'sharing-report.json').write_text(json.dumps(report,indent=2)+'\n')
for row in summaries:
    print(f"{row['workload']:15} workers={row['workers']:2} median={row['median_us']:.3f} us p95={row['p95_us']:.3f} us")
print('Independent NumPy checks passed:',checks)
