import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from result_io import parse_paths, load_json
args = parse_paths('Plot baseline latency and eligible CPU operation shares')
R, O = args.results_dir, args.output_dir
baselines = load_json(R/'summary.json')
ops = load_json(R/'operation-summary.json')['runs']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig = plt.figure(figsize=(12, 9), facecolor='white')
grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.25], hspace=0.55, wspace=0.27)
ax1, ax2, ax3 = fig.add_subplot(grid[0,0]), fig.add_subplot(grid[0,1]), fig.add_subplot(grid[1,:])
colors = {'CPU':'#275DAD', 'Metal':'#148271'}
for backend in ('CPU', 'Metal'):
    data = sorted([r for r in baselines if r['run'] in (backend.lower()+'-baseline',backend.lower()+'-8k')], key=lambda r:r['prompt_tokens'])
    x = np.arange(len(data))
    for ax, metric, low, high in ((ax1,'prefill_seconds','prefill_min_s','prefill_max_s'), (ax2,'decode_ms','decode_min_ms','decode_max_ms')):
        y = np.array([r[metric] for r in data])
        err = np.array([[r[metric]-r[low] for r in data], [r[high]-r[metric] for r in data]])
        ax.errorbar(x,y,yerr=err,marker='o',capsize=4,lw=2,color=colors[backend],label=backend)
        for xi, yi in zip(x,y):
            ax.annotate(f'{yi:.2f}', (xi,yi), xytext=(0,8 if backend=='CPU' else -18), textcoords='offset points',ha='center',fontsize=9,color=colors[backend])
        ax.set_xticks(x,['128','512','2,048','8,192'])
        ax.set_xlabel('Prompt tokens before 32 decode steps')
        ax.grid(axis='y',alpha=.16)
ax1.set_yscale('log'); ax1.set_ylim(.012,60)
ax1.set_ylabel('Prefill seconds · log scale');ax1.set_title('Prompt processing',loc='left',fontweight='bold')
ax2.set_ylim(0,18);ax2.set_ylabel('Decode ms / token');ax2.set_title('Cached single-token processing',loc='left',fontweight='bold')
ax1.legend(frameon=False)
groups = ['MLP projections','Attention/DeltaNet projections','Vocabulary output head','Attention QK/softmax/AV','DeltaNet recurrence','Copies/gather/state movement','Vector/norm/activation/other','Convolution']
labels = ['MLP matrices','Attention / DeltaNet matrices','Vocabulary head','Attention core','DeltaNet recurrence','Copies / gather / state','Vector / norm / activation','Convolution']
palette = ['#275DAD','#5B8BC3','#D0782B','#148271','#61A78B','#8E73AC','#B9A6CA','#CDBB9B']
left = np.zeros(2)
for g,label,c in zip(groups,labels,palette):
    vals=[]
    for phase in ('prefill','decode'):
        rows=[r for r in ops if r['prompt_tokens']==2048 and r['phase']==phase and r.get('timing_eligible_for_report', True)]
        vals.append(100*sum(r['groups_us'].get(g,0) for r in rows)/sum(r['timed_interval_us'] for r in rows))
    ax3.barh([1,0],vals,left=left,color=c,label=label,height=.45)
    for i, v in enumerate(vals):
        if v>=6:
            ax3.text(left[i]+v/2,1-i,f'{v:.0f}%',ha='center',va='center',color='white' if c not in ['#B9A6CA','#CDBB9B'] else '#222',fontsize=10,fontweight='bold')
    left+=vals
ax3.set_yticks([1,0],['2K prefill','2K decode']);ax3.set_xlim(0,100);ax3.set_ylim(-.6,1.6)
ax3.set_xlabel('Share of timed CPU operation intervals (%)')
ax3.set_title('CPU work at 2K: matrix operations dominate both phases',loc='left',fontweight='bold',pad=16)
ax3.legend(ncol=4,frameon=False,loc='upper center',bbox_to_anchor=(.5,-.28),fontsize=9,columnspacing=1.5)
fig.suptitle('Qwen3.5-2B in llama.cpp',x=.08,y=.98,ha='left',fontsize=22,fontweight='bold')
fig.text(.08,.935,'Apple M5 Pro · Q4_K_M mixed quantization · six CPU threads · one sequence',fontsize=11,color='#555')
fig.text(.08,.028,'Top: uninstrumented medians of 3 runs; whiskers show min–max. Bottom: 2 CPU traces, 32 decode steps each.\nCPU operation shares do not describe Metal kernels or predict accelerator area. Anomalous 8K timing shares omitted.',fontsize=9,color='#555')
fig.subplots_adjust(top=.86,bottom=.21,left=.1,right=.97)
fig.savefig(O/'profiling-summary.png',dpi=180)
fig.savefig(O/'profiling-summary.pdf')

print("Output directory:",O)
