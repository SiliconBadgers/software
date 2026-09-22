import json,statistics,pathlib,csv
from result_io import parse_paths, exists, open_result
args=parse_paths('Summarize raw uninstrumented and diagnostic phase measurements')
R=args.results_dir;O=args.output_dir
summaries=[]
for name in ['cpu-baseline','metal-baseline','cpu-8k','metal-8k','cpu-profile','cpu-control','cpu-8k-control','cpu-8k-profile','cpu-8k-profile-repeat']:
 p=R/(name+'.jsonl')
 if not exists(p):continue
 
 with open_result(p) as stream: rows=[json.loads(l) for l in stream if l.strip()];rows=[r for r in rows if r['kind']=='measurement']
 for n in sorted(set(r['prompt_tokens'] for r in rows)):
  rs=[r for r in rows if r['prompt_tokens']==n];pre=[r['prefill_us']/1e6 for r in rs];dec=[statistics.mean(r['decode_us'])/1e3 for r in rs]
  summaries.append({'run':name,'prompt_tokens':n,'repetitions':len(rs),'decode_steps_per_rep':len(rs[0]['decode_us']),'prefill_seconds':statistics.median(pre),'prefill_tps':n/statistics.median(pre),'decode_ms':statistics.median(dec),'decode_tps':1000/statistics.median(dec),'prefill_min_s':min(pre),'prefill_max_s':max(pre),'decode_min_ms':min(dec),'decode_max_ms':max(dec),'prefill_samples_seconds':pre,'decode_run_means_ms':dec})
if not summaries: raise ValueError('No measurement files found in '+str(R))
(O/'summary.json').write_text(json.dumps(summaries,indent=2))
with (O/'summary.csv').open('w') as f:
 if summaries:
  w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
print(json.dumps(summaries,indent=2))

print("Output directory:",O)
