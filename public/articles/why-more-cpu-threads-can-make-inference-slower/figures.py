"""Generate code-derived diagrams and a clearly labeled illustrative cost model."""
from pathlib import Path
import os
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'thread-article-mpl'))
os.environ.setdefault('XDG_CACHE_HOME', str(Path(tempfile.gettempdir()) / 'thread-article-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'assets'
ASSETS.mkdir(exist_ok=True)
BLUE, ORANGE, INK = '#1765b4', '#b9501c', '#17283c'
MUTED, LINE, LIGHT = '#53667b', '#d6e0ea', '#eef5fc'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 14,
                     'text.color': INK, 'axes.labelcolor': INK,
                     'axes.edgecolor': LINE, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'svg.fonttype': 'none'})


def save(fig, name):
    for ext in ('png', 'svg'):
        fig.savefig(ASSETS / f'{name}.{ext}', dpi=160, facecolor='white',
                    bbox_inches='tight', pad_inches=.18)
    plt.close(fig)


def rect(ax, x, y, w, h, label, size=14):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=LIGHT, edgecolor=LINE, lw=1.3))
    ax.text(x+w/2, y+h/2, label, ha='center', va='center', fontsize=size)


def arrow(ax, start, end):
    ax.annotate('', xy=end, xytext=start,
                arrowprops={'arrowstyle': '-|>', 'color': MUTED, 'lw': 1.7})


fig, ax = plt.subplots(figsize=(5.5, 7.3))
ax.set(xlim=(0, 5.5), ylim=(0, 7.3)); ax.axis('off')
ax.text(.1, 6.92, 'Two routes for contiguous copies', fontsize=17, weight='bold')
ax.text(.1, 6.22, '128 KiB buffer', fontsize=16, weight='bold')
rect(ax, .1, 5.12, 5.2, .7, 'Existing direct-copy routine')
ax.text(2.7, 4.69, 'No explicit chunk scheduling', ha='center', color=BLUE)
ax.plot([.1, 5.3], [4.22, 4.22], color=LINE)
ax.text(.1, 3.67, '1 MiB buffer', fontsize=16, weight='bold')
for i in range(16):
    rect(ax, .15+(i%4)*1.3, 2.94-(i//4)*.52, 1.16, .41, str(i+1), size=14)
ax.text(2.7, .97, '16 chunks × 64 KiB', ha='center', color=BLUE, fontsize=16)
ax.text(2.7, .38, 'Chunks are tasks, not threads.', ha='center', fontsize=14)
save(fig, '01-copy-granularity')

# Arbitrary illustrative assumptions, NOT fitted to hardware or measured data.
workers = np.linspace(1, 16, 300)
divisible_work = .4 / workers
coordination = .025 * (workers - 1)
fig, ax = plt.subplots(figsize=(5.5, 6.1))
fig.subplots_adjust(left=.15, right=.97, bottom=.16, top=.67)
fig.text(.04, .96, 'The cost of sharing work', fontsize=19, weight='bold')
fig.text(.04, .89, 'Illustrative model • not measured', fontsize=14, color=MUTED)
ax.plot(workers, divisible_work, color=BLUE, lw=2.7, ls='--', label='Divisible work')
ax.plot(workers, coordination, color=ORANGE, lw=2.7, ls=':', label='Coordination')
ax.plot(workers, divisible_work+coordination, color=INK, lw=3, label='Combined cost')
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(.02, .85),
           frameon=False, fontsize=14, labelspacing=.35)
ax.set(xlim=(1, 16), ylim=(0, .46), xlabel='Worker count (illustrative)',
       ylabel='Execution cost (arbitrary units)')
ax.set_xticks([1, 4, 8, 12, 16]); ax.set_yticks([])
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='x', color=LINE, alpha=.6)
fig.text(.04, .035, 'No real-world optimum is implied.', fontsize=14, color=MUTED)
save(fig, '02-parallel-cost-model')

fig, ax = plt.subplots(figsize=(5.5, 7.8))
ax.set(xlim=(0, 5.5), ylim=(0, 7.8)); ax.axis('off')
ax.text(.1, 7.4, 'Independent workers, shared CPU', fontsize=17, weight='bold')
ax.text(.1, 6.87, 'Incoming requests', fontsize=15)
arrow(ax, (.35, 6.64), (.35, 6.25))
ax.plot([.35, .35], [2.05, 6.3], color=MUTED, lw=1.5)
ax.plot([5.05, 5.05], [1.4, 6.25], color=MUTED, lw=1.5)
for i in range(4):
    y = 5.65 - i*1.25
    rect(ax, .95, y, 3.45, .95, f'Worker {i+1}\nOwn pool: 8 threads', size=15)
    arrow(ax, (.35, y+.475), (.92, y+.475))
    arrow(ax, (4.43, y+.475), (5.05, y+.475))
arrow(ax, (5.05, 1.4), (5.05, 1.02))
rect(ax, .15, .15, 5.1, .85, 'Same cores, caches\nand memory bandwidth', size=15)
save(fig, '03-request-concurrency')
print('Generated three PNG figures and three SVG sources.')

# Actual attention measurements: saved data, not the model in Figure 2.
import csv
import json
report = json.loads((ROOT / 'data/attention-report.json').read_text())
with (ROOT / 'data/attention-samples.csv').open() as f:
    samples = list(csv.DictReader(f))
fig, axes = plt.subplots(3, 1, figsize=(5.8, 10.5))
fig.subplots_adjust(left=.18, right=.96, top=.84, bottom=.10, hspace=.83)
fig.text(.04, .965, 'Attention: measured CPU latency', fontsize=18, weight='bold')
fig.text(.04, .925, 'Inline vs pools • 1,000 calls per setting', fontsize=13)
from matplotlib.lines import Line2D
fig.legend(handles=[Rectangle((0,0),1,1,color=BLUE,label='Median'),
                    Line2D([0],[0],color=ORANGE,marker='D',linestyle='None',label='Estimated p95')],
           loc='upper left',bbox_to_anchor=(.02,.91),ncol=2,frameon=False,fontsize=14)
for ax, (name, title) in zip(axes, [
        ('short-context','Short context: 1 query × 16 keys'),
        ('long-context','Long context: 1 query × 2,048 keys'),
        ('prefill','Prefill-shaped: 128 queries × 128 keys')]):
    rows = [r for r in report['summaries'] if r['workload']==name]
    assert [r['workers'] for r in rows]==[0,1,2,4,8]
    for row in rows:
        measured=[float(r['ms']) for r in samples if r['workload']==name and int(r['workers'])==row['workers']]
        assert len(measured)==1000
        np.testing.assert_allclose([np.median(measured),np.percentile(measured,95)],
                                   [row['median_ms'],row['p95_ms']],rtol=1e-12)
    x=np.arange(5)
    ax.bar(x,[r['median_ms'] for r in rows],color=[MUTED,BLUE,BLUE,BLUE,BLUE],width=.58,zorder=3)
    ax.scatter(x,[r['p95_ms'] for r in rows],color=ORANGE,marker='D',s=45,zorder=4)
    ax.set_title(title,fontsize=14,weight='bold',loc='left',pad=12)
    ax.set_xticks(x,['Inline','1','2','4','8'])
    ax.set_ylabel('Latency (ms)',fontsize=14)
    ax.set_xlabel('Execution: inline or pool workers',fontsize=12)
    ax.set_ylim(bottom=0)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',color=LINE,zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=13)
fig.text(.04,.035,'Gray: inline • blue: pool • separate scales',fontsize=13,color=MUTED)
save(fig,'04-attention-benchmark')
print('Generated measured attention benchmark graph.')

# Dedicated sharing-cost probe: independently measured control and attention.
sharing = json.loads((ROOT/'data/sharing-report.json').read_text())
with (ROOT/'data/sharing-samples.csv').open() as f:
    sharing_samples = list(csv.DictReader(f))
assert len(sharing_samples)==27000
for row in sharing['summaries']:
    vals=[float(r['us']) for r in sharing_samples
          if r['workload']==row['workload'] and int(r['workers'])==row['workers']]
    assert len(vals)==1000
    np.testing.assert_allclose([np.median(vals),np.percentile(vals,95)],
                              [row['median_us'],row['p95_us']],rtol=1e-12)
widths=[1,2,4,6,8,10]
lookup={(r['workload'],r['workers']):r for r in sharing['summaries']}
fig,axes=plt.subplots(2,1,figsize=(5.8,8.9))
fig.subplots_adjust(left=.16,right=.97,top=.82,bottom=.10,hspace=.67)
fig.text(.04,.965,'Measured cost of sharing work',fontsize=18,weight='bold')
fig.text(.04,.92,'Apple M5 • 1,000 calls per setting',fontsize=14,color=MUTED)
for ax,name,title,color in [
    (axes[0],'empty-dispatch','Empty dispatch: coordination only',ORANGE),
    (axes[1],'keys-64','Fixed attention: 1 query × 64 keys',BLUE)]:
    medians=[lookup[name,w]['median_us'] for w in widths]
    for w in widths:
        dots=lookup[name,w]['round_median_us']
        ax.scatter(w+np.linspace(-.1,.1,len(dots)),dots,
                   color=color,alpha=.22,s=20,zorder=2)
    ax.plot(widths,medians,color=color,marker='o',lw=2.4,markersize=6,
            label='Measured median',zorder=3)
    ax.set_title(title,fontsize=14,weight='bold',loc='left',pad=15)
    ax.set(xlabel='Pool workers',ylabel='Latency (μs)',xlim=(.7,10.3))
    ax.set_xticks(widths)
    ax.set_ylim(bottom=0)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',color=LINE)
    ax.tick_params(labelsize=13)
inline=lookup['keys-64',0]['median_us']
axes[1].axhline(inline,color=MUTED,ls='--',lw=1.7,label='Inline median')
axes[1].legend(loc='upper left',bbox_to_anchor=(-.04,-.29),
               ncol=2,frameon=False,fontsize=12.5,handlelength=1.5,columnspacing=.8)
fig.text(.04,.025,'Faint dots: medians of ten measured blocks',fontsize=13,color=MUTED)
save(fig,'05-measured-sharing-cost')
print('Generated measured sharing-cost companion figure.')
