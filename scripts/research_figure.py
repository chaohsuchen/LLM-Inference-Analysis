"""Rebuild the portfolio figure from the published cell summaries (no GPU needed)."""
import csv
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    with (ROOT / 'docs/results/cells.csv').open() as f:
        rows = list(csv.DictReader(f))

    def values(workload, config, concurrency, metric):
        return [float(r[metric]) for r in rows if r['workload'] == workload
                and r['config'] == config and int(r['concurrency']) == concurrency]

    def med(w, cfg, c, metric):
        return statistics.median(values(w, cfg, c, metric))

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'svg.hashsalt': 'llm-inference-analysis',
                         'axes.labelcolor': '#283449', 'text.color': '#18283e'})
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8))
    fig.patch.set_facecolor('#f5f7fb')
    fig.suptitle('Faster LLM serving depends on the workload', x=.055, y=.97,
                 ha='left', fontsize=24, weight='bold')
    fig.text(.055, .865, 'RTX 4090  /  Qwen2.5-1.5B  /  20,736 measured requests  /  0 failures',
             fontsize=12, color='#52637b')
    blue, teal, coral = '#3559c7', '#087f8c', '#d85f46'
    metric = 'output_tokens_per_s'
    ax = axes[0]
    for cfg, label, color in [('serial', 'One active sequence', coral), ('batch', 'Up to 32 active sequences', blue)]:
        cs = [1, 8, 32]
        vs = [values('unique', cfg, c, metric) for c in cs]
        ys = [statistics.median(v) for v in vs]
        ax.errorbar(cs, ys, yerr=[[y-min(v) for y,v in zip(ys,vs)],
                                 [max(v)-y for y,v in zip(ys,vs)]],
                    color=color, marker='o', lw=2.5, capsize=4, label=label)
    ratio = med('unique','batch',32,metric)/med('unique','serial',32,metric)
    ax.set_title(f'01  Batching: {ratio:.1f}× throughput', loc='left', weight='bold', pad=18)
    ax.set(xlabel='Concurrent client requests', ylabel='Output tokens / second', xticks=[1,8,32], ylim=(0,5200))
    ax.legend(frameon=False, fontsize=9, loc='upper left')
    ax.text(0,-.28,'Unique prompts; benefit grows with load.',transform=ax.transAxes,fontsize=10)
    ax = axes[1]
    labels=['Unique prompts','Shared policy']
    for offset,cfg,label,color in [(-.19,'batch','Cache off',blue),(.19,'cache','Cache on',teal)]:
        ys=[med(w,cfg,32,metric) for w in ['unique','shared']]
        bars=ax.bar([i+offset for i in range(2)],ys,width=.36,color=color,label=label)
        ax.bar_label(bars,fmt='%.0f',padding=4,fontsize=9)
    ratio=med('shared','cache',32,metric)/med('shared','batch',32,metric)
    ax.set_title(f'02  Prefix reuse: {ratio:.2f}× throughput',loc='left',weight='bold',pad=18)
    ax.set(xticks=[0,1],xticklabels=labels,ylabel='Output tokens / second',ylim=(0,5600))
    ax.legend(frameon=False,fontsize=9,loc='upper right')
    ax.text(0,-.28,'Concurrency 32; unique prompts are the control.',transform=ax.transAxes,fontsize=10)
    ax=axes[2]
    configs=['batch','chunk2048','chunk512']
    for metric,label,color in [('all_ttft_s_p50','Typical request (p50)',teal),('short_ttft_s_p95','Short-request tail (p95)',coral)]:
        vs=[values('mixed',cfg,32,metric) for cfg in configs]
        ys=[1000*statistics.median(v) for v in vs]
        ax.errorbar(range(3),ys,yerr=[[y-1000*min(v) for y,v in zip(ys,vs)],
                                    [1000*max(v)-y for y,v in zip(ys,vs)]],
                    label=label,color=color,marker='o',lw=2.5,capsize=4)
    ax.set_title('03  Lower median, worse tail',loc='left',weight='bold',pad=18)
    ax.set(xticks=range(3),xticklabels=['Batch','Chunk 2048','Chunk 512'],
           ylabel='Time to first content (ms) ↓',ylim=(0,1450))
    ax.legend(frameon=False,fontsize=9,loc='upper left')
    ax.text(0,-.28,'Mixed prompts, concurrency 32; budget changes too.',transform=ax.transAxes,fontsize=10)
    for ax in axes:
        ax.set_facecolor('#f5f7fb');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.text(.055,.035,'Points/bars: medians of 3 run summaries. Whiskers: observed min–max, not confidence intervals. Synthetic inputs; fixed 128-token outputs.',fontsize=10,color='#52637b')
    fig.subplots_adjust(left=.055,right=.985,top=.73,bottom=.26,wspace=.35)
    dest=ROOT/'docs/figures';dest.mkdir(exist_ok=True)
    for ext in ['png','svg']:
        fig.savefig(dest/f'research_highlights.{ext}',dpi=180,facecolor=fig.get_facecolor(),
                    metadata={'Date': None} if ext == 'svg' else None)
        if ext == 'svg':
            path = dest / 'research_highlights.svg'
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)


if __name__ == '__main__':
    main()
