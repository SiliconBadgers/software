#!/usr/bin/env python3
"""Pack the four real captures with shape expressions; fail on unknown scaling."""
import json,pathlib
R=pathlib.Path(__file__).resolve().parents[1]
g=[json.loads((R/'graphs'/w/(p+'.json')).read_text()) for w,p in [('pp128','prefill'),('pp512','prefill'),('pp128','decode'),('pp512','decode')]]
def expr(v):
 a,b,c,d=v
 if a==b==c==d:return a
 if v==[256,512,256,768]:return 'K'
 if c==a and d==b and b==4*a:return ['C',a/512]
 if c==d:
  slope=(b-a)/384;off=a-128*slope
  if abs(slope+off-c)<1e-8:return ['T',slope,off]
  return ['P',slope,off,c]
 raise ValueError(('unrecognized shape',v))
ts=[]
for i,t in enumerate(g[0]['tensors']):
 other=[x['tensors'][i] for x in g]
 assert all(x['name']==t['name'] and x['op_desc']==t['op_desc'] and x['sources']==t['sources'] for x in other)
 ts.append({'id':i,'name':t['name'],'op':t['op_desc'],'dtype':t['dtype'],'shape':[expr([x['shape'][j] for x in other]) for j in range(4)],'src':[s['tensor'] for s in t['sources']],'view':t['view_source'],'weight':t['weight_buffer']})
weights={t['name']:t for t in ts if t['weight'] and t['view']<0 and t['op']=='NONE'}
data={'version':1,'llamaCommit':json.loads((R/'manifest.json').read_text())['llama_cpp_revision'],'model':'Qwen3.5-2B','parameters':1881825088,'tensors':ts,'order':g[0]['scheduled_nodes'],'weights':list(weights),'captureValidation':{'prefill128MACs':177021648896,'prefill512MACs':709782142976,'decode128MACs':1887567872,'decode512MACs':1900150784},'scope':'Text trunk, no tokenizer, vision, MTP, host or sampling.'}
(R/'explorer'/'graph-data.js').write_text('/* Generated from verified llama.cpp captures. */\n(function(r){const data='+json.dumps(data,separators=(',',':'))+';if(typeof module!=="undefined")module.exports=data;else r.QWEN_GRAPH=data;})(globalThis);\n')
print('Exported',len(ts),'tensor shape expressions; four captures matched.')
