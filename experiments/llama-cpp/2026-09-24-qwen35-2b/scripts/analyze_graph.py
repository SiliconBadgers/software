#!/usr/bin/env python3
"""Validate captured GGML graphs and derive hardware-independent work inventories."""
import argparse, collections, csv, html, json, math, pathlib, re, subprocess
VIEWS = {'NONE','VIEW','RESHAPE','PERMUTE','TRANSPOSE'}
COLORS = {'matrix':'#dbeafe','recurrent':'#ede9fe','convolution':'#ccfbf1','vector':'#fef3c7','memory':'#e2e8f0','metadata':'#f1f5f9'}

def category(n):
    op=n['op']
    if op in VIEWS: return 'metadata'
    if op in {'MUL_MAT','MUL_MAT_ID'}: return 'matrix'
    if op == 'GATED_DELTA_NET': return 'recurrent'
    if op == 'SSM_CONV': return 'convolution'
    if op in {'CPY','CONT','SET_ROWS','GET_ROWS','CONCAT'}: return 'memory'
    return 'vector'

def process(path):
    graph=json.loads(path.read_text()); ts={t['id']:t for t in graph['tensors']}; order=graph['scheduled_nodes']; pos={n:i for i,n in enumerate(order)}
    assert len(ts)==len(graph['tensors']) and len(set(order))==len(order)
    assert graph['metadata']['finite_logits']
    visiting=set(); done=set()
    def validate(i):
        assert i in ts and i not in visiting, ('missing tensor or cycle',i)
        if i in done: return
        visiting.add(i); t=ts[i]
        assert t['elements']==math.prod(t['shape'])
        for s in t['sources']: validate(s['tensor'])
        if t['view_source']>=0: validate(t['view_source'])
        visiting.remove(i); done.add(i)
    for i in ts: validate(i)
    for i in order:
        for s in ts[i]['sources']:
            j=s['tensor']
            if j in pos: assert pos[j]<pos[i], ('non-topological scheduled dependency',i,j)
    layers={}
    def layer(i):
        if i in layers: return layers[i]
        t=ts[i]; name=t['name']
        match=re.search(r'^blk\.(\d+)\.',name) or re.search(r'-(\d+)(?:\s|$)',name)
        value=int(match.group(1)) if match else max((layer(s['tensor']) for s in t['sources']),default=-1)
        layers[i]=value; return value
    for i in ts: layer(i)
    rows=[]; counts=collections.Counter(); work=collections.Counter(); cats=collections.Counter(); unresolved=collections.Counter()
    for index,i in enumerate(order):
        t=ts[i]; op=t['op']; counts[t['op_desc']]+=1; cats[category(t)]+=1
        macs=None
        if op=='MUL_MAT': macs=t['elements']*ts[t['sources'][0]['tensor']]['shape'][0]
        elif op=='SSM_CONV': macs=t['elements']*ts[t['sources'][1]['tensor']]['shape'][0]
        if macs is not None: work[op]+=macs
        elif category(t) not in {'metadata','memory'}: unresolved[t['op_desc']]+=1
        rows.append({'index':index,'tensor_id':i,'name':t['name'],'op':t['op_desc'],'category':category(t),'layer':layers[i],
            'dtype':t['dtype'],'shape':'x'.join(map(str,t['shape'])),'logical_output_bytes':t['logical_bytes'],
            'macs':macs if macs is not None else '', 'arithmetic_ops_2_per_mac':2*macs if macs is not None else '',
            'source_tensor_ids':';'.join(str(s['tensor']) for s in t['sources']), 'view_source':t['view_source']})
    target=path.with_suffix('')
    with target.with_suffix('.ops.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    summary={'metadata':graph['metadata'],'scheduled_nodes':len(order),'dependency_tensors':len(ts),
        'compute_and_memory_nodes':sum(v for k,v in cats.items() if k!='metadata'), 'category_counts':dict(cats),
        'operation_counts':dict(counts),'macs_by_supported_op':dict(work),'arithmetic_ops_by_supported_op':{k:2*v for k,v in work.items()},
        'arithmetic_not_yet_modeled':dict(unresolved),
        'limitations':['MAC counts cover MUL_MAT and SSM_CONV only; they are not total model FLOPs.',
            'Vector functions, reductions and fused GATED_DELTA_NET need separate hardware cost models.',
            'Logical tensor bytes and spans are not DRAM traffic; views alias storage and kernels reuse data.',
            'Dependencies are tensor dataflow. Stateful aliases and backend fusion require additional scheduling constraints.',
            'CPU-selected GGML graph with flash attention disabled; a target accelerator may lower or fuse it differently.'],
        'validation':{'acyclic':True,'source_ids_valid':True,'schedule_topological':True,'finite_logits':True}}
    target.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    def dot(selected,title):
        lines=['digraph G {','rankdir=TB;','graph [bgcolor="white",fontname="sans",label='+json.dumps(title)+',labelloc=t];','node [shape=box,style="rounded,filled",fontname="sans",fontsize=10];','edge [color="#94a3b8"];']
        for i in sorted(selected):
            t=ts[i]; label=f'{i}: {t["name"]}\n{t["op_desc"]} | {t["dtype"]} | '+ 'x'.join(map(str,t['shape']))
            lines.append(f'n{i} [label={json.dumps(label)},fillcolor="{COLORS[category(t)]}"];')
        for i in sorted(selected):
            for s in ts[i]['sources']:
                if s['tensor'] in selected: lines.append(f'n{s["tensor"]} -> n{i} [label="{s["slot"]}"];')
            v=ts[i]['view_source']
            if v in selected and v not in {s['tensor'] for s in ts[i]['sources']}: lines.append(f'n{v} -> n{i} [style=dashed,label="alias"];')
        lines.append('}');return '\n'.join(lines)+'\n'
    target.with_suffix('.dot').write_text(dot(set(ts),f'{path.parent.name} {path.stem}: complete tensor dataflow'))
    for il in (0,3):
        selected={i for i in order if layers[i]==il}
        selected.update(s['tensor'] for i in list(selected) for s in ts[i]['sources'])
        dest=path.parent/f'{path.stem}.layer{il}.dot'
        dest.write_text(dot(selected,f'{path.parent.name} {path.stem}: layer {il} (plus direct boundary inputs)'))
        subprocess.run(['dot','-Tsvg',str(dest),'-o',str(dest.with_suffix('.svg'))],check=True,timeout=45)
    return summary

def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=pathlib.Path);args=parser.parse_args()
    summaries={}
    for phase in ['prefill','decode']:
        summaries[phase]=process(args.directory/(phase+'.json'))
    report=['# Qwen3.5-2B graph inventory','',f'Workload: {args.directory.name}. Graphs captured during successful CPU inference.','',
        '| Phase | Input tokens | Past tokens | Scheduled nodes | Matrix MACs | Conv MACs |','|---|---:|---:|---:|---:|---:|']
    for phase,s in summaries.items():
        m=s['metadata'];w=s['macs_by_supported_op']
        report.append(f'| {phase} | {m["input_tokens"]} | {m["past_tokens"]} | {s["scheduled_nodes"]:,} | {w.get("MUL_MAT",0):,} | {w.get("SSM_CONV",0):,} |')
    report+=['','MAC = multiply-accumulate; the work inventory counts two arithmetic operations per MAC. These counts exclude vector/reduction work and fused recurrent operations. No accelerator latency or tokens/s is estimated yet.','', '## Operation counts','', '| Operation | Prefill | Decode |','|---|---:|---:|']
    for op in summaries['prefill']['operation_counts']:
        report.append(f'| {op} | {summaries["prefill"]["operation_counts"].get(op,0)} | {summaries["decode"]["operation_counts"].get(op,0)} |')
    report+=['','## Interpretation','']+['- '+x for x in summaries['prefill']['limitations']]
    report+=['','Raw JSON contains all dependencies, data types, shapes, byte strides, alias offsets, operation parameters, and scheduler order. DOT files contain full dataflow; layer 0 and layer 3 SVGs show representative recurrent and full-attention layers.','']
    (args.directory/'REPORT.md').write_text('\n'.join(report))
    table=''.join(f'<tr><td>{html.escape(op)}</td><td>{summaries["prefill"]["operation_counts"].get(op,0)}</td><td>{summaries["decode"]["operation_counts"].get(op,0)}</td></tr>' for op in summaries['prefill']['operation_counts'])
    links=''.join(f'<a href="{p}.{ext}">{p} {label}</a> ' for p in ['prefill','decode'] for ext,label in [('json','raw graph'),('dot','full DOT'),('ops.csv','operations'),('summary.json','summary')])
    text='''<!doctype html><meta charset="utf-8"><title>Qwen3.5-2B graph inventory</title><style>body{font:16px system-ui;color:#182b43;background:#f6f8fc;margin:30px auto;max-width:1200px;padding:0 24px}h1{font-size:32px}a{color:#155ab6;margin-right:16px;line-height:2.2}table{border-collapse:collapse;background:white}td,th{padding:7px 26px;text-align:left;border-bottom:1px solid #e2e8f0}.panel{background:white;padding:20px;border-radius:12px;margin:24px 0}.diagram{overflow:auto;height:650px;border:1px solid #dbe3ee;background:white}.diagram img{max-width:none;height:auto}select{padding:8px;font:inherit}.note{color:#52657d;line-height:1.6}</style>'''
    text+=f'<h1>Qwen3.5-2B / {html.escape(args.directory.name)}</h1><p>Captured llama.cpp computational graphs · BF16 weights · text inference · one sequence</p>'
    text+='<div class="panel"><strong>Verified:</strong> successful prefill and decode with finite logits; tensor dependencies validated. <p class="note">1,441 scheduled nodes per graph. The model has 18 recurrent layers and 6 full-attention layers. Matrix/convolution work is inventoried; accelerator performance awaits your hardware specification.</p>'+links+'</div>'
    text+='<div class="panel"><h2>Representative layer graphs</h2><p><select id="phase"><option value="prefill">Prefill</option><option value="decode">Decode</option></select> <select id="layer"><option value="0">Layer 0 · recurrent</option><option value="3">Layer 3 · full attention</option></select> <a id="open" href="prefill.layer0.svg">Open full-size diagram</a></p><p class="note">Scroll to explore. Blue: matrix; purple: recurrence; teal: convolution; yellow: vector; gray: memory/views. Shapes use GGML innermost dimension first. Boundary inputs are included.</p><div class="diagram"><img id="diagram" src="prefill.layer0.svg" alt="Captured tensor dependency graph"></div></div>'
    text+='<div class="panel"><h2>Operation inventory</h2><table><tr><th>Operation</th><th>Prefill</th><th>Decode</th></tr>'+table+'</table></div>'
    text+='<p class="note">Graph scope excludes vision and MTP. A CPU graph is an input to accelerator modeling; hardware mapping, fusion, memory traffic, and execution time remain to be specified. See REPORT.md for arithmetic coverage and limitations.</p>'
    text+='''<script>const phase=document.querySelector('#phase'),layer=document.querySelector('#layer');function update(){let p=phase.value+'.layer'+layer.value+'.svg';document.querySelector('#diagram').src=p;document.querySelector('#open').href=p}phase.onchange=update;layer.onchange=update;</script>'''
    (args.directory/'index.html').write_text(text)
    print(json.dumps({k:{'nodes':v['scheduled_nodes'],'matrix_macs':v['macs_by_supported_op']['MUL_MAT'],'validated':v['validation']} for k,v in summaries.items()},indent=2))
if __name__=='__main__': main()
