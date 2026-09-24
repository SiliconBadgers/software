/* Analytical design-space model. See MODEL.md for equations and limits. */
(function(root){
'use strict';
const VIEWS=new Set(['NONE','VIEW','RESHAPE','PERMUTE','TRANSPOSE']);
const MEMORY=new Set(['GET_ROWS','SET_ROWS','CPY','CONT','CONCAT']);
const GiB=2**30,MiB=2**20;
const DEFAULTS={
 frequency:300,precision:'w8a16',batch:1,prompt:512,context:2048,
 matrixCount:4,rows:16,cols:16,matrixEfficiency:.75,
 matrixCaps:['gemm','gemv','attention','requant'],matrixTypes:['w8a16','w4a8','a16a16','a8a8'],
 rateW8A16:1,rateW4A8:2,rateA16:0.5,rateA8:1,
 vectorCount:2,vectorLanes:32,vectorEfficiency:.7,specialCycles:8,
 vectorCaps:['elementwise','reduce','exp','rope','quant','recurrent','conv'],
 recurrentCount:0,recurrentLanes:128,recurrentEfficiency:.7,recurrentScratch:64,
 dmaCount:1,dmaBytes:64,
 rvCount:1,rvIPC:1,rvCycles:4,rvSpecialCycles:40,rvFrequency:300,
 l1MiB:8,l1Banks:32,l1BytesPerBank:16,l1Efficiency:.65,stateReserve:.25,
 hbmGiB:16,hbmGBs:460,hbmEfficiency:.7,kvBits:16,stateBits:32,
 tileN:64,tileK:256,weightGroup:64,scaleBytes:2,launchCycles:40,overlap:true,
 dspPerPE:1,dspPerVectorLane:.5,dspPerRecurrentLane:1,dspBudget:9024,shellReserve:.2,
 costMatrix:1,costVector:8,costRecurrent:10,costSRAM:128,costCapability:40,costRiscV:200
};
function clone(v){return JSON.parse(JSON.stringify(v));}
function defaults(){return clone(DEFAULTS);}
function prod(a){return a.reduce((x,y)=>x*y,1);}
function round(n,m){return Math.ceil(n/m)*m;}
function shapeOf(t,phase,c){const T=phase==='prefill'?c.prompt:1,K=round(phase==='prefill'?c.prompt:c.context+1,256),C=round(Math.max(c.prompt,c.context+1),256);return t.shape.map(x=>typeof x==='number'?x:x==='K'?K:x==='C'?C:x[0]==='C'?C*x[1]:x[0]==='P'?(phase==='prefill'?T*x[1]+x[2]:x[3]):T*x[1]+x[2]);}
function precision(c){return c.precision==='w4a8'?{w:4,a:8,rate:c.rateW4A8,aa:'a8a8',aaRate:c.rateA8}:{w:8,a:16,rate:c.rateW8A16,aa:'a16a16',aaRate:c.rateA16};}
function validation(c){
 const errors=[];
 for(const k of ['matrixCaps','matrixTypes','vectorCaps'])if(!Array.isArray(c[k])||c[k].some(x=>typeof x!=='string'))errors.push(k+' must be a list of capabilities.');
 if(typeof c.overlap!=='boolean')errors.push('overlap must be a boolean.');
 for(const k of Object.keys(DEFAULTS)){if(typeof DEFAULTS[k]==='number'&&(!Number.isFinite(c[k])||c[k]<0))errors.push(k+' must be a nonnegative number.');}
 for(const k of ['frequency','batch','prompt','context','rows','cols','vectorLanes','recurrentLanes','l1MiB','l1Banks','l1BytesPerBank','hbmGiB','hbmGBs','tileN','tileK','rvFrequency','rvIPC','rvCycles','weightGroup'])if(!(c[k]>0))errors.push(k+' must exceed zero.');
 for(const k of ['matrixCount','vectorCount','recurrentCount','dmaCount','rvCount','batch','prompt','context','rows','cols','vectorLanes','recurrentLanes','l1Banks','tileN','tileK'])if(!Number.isInteger(c[k]))errors.push(k+' must be an integer.');
 for(const k of ['matrixEfficiency','vectorEfficiency','recurrentEfficiency','l1Efficiency','hbmEfficiency'])if(!(c[k]>0&&c[k]<=1))errors.push(k+' must be in (0, 1].');
 for(const k of ['stateReserve','shellReserve'])if(c[k]<0||c[k]>=1)errors.push(k+' must be in [0, 1).');
 if(!['w8a16','w4a8'].includes(c.precision))errors.push('Unsupported precision.');
 if(![8,16,32].includes(c.kvBits)||![16,32].includes(c.stateBits))errors.push('Unsupported state/cache storage precision.');
 return errors;
}
function resources(c){
 const pe=c.matrixCount*c.rows*c.cols,vl=c.vectorCount*c.vectorLanes,rl=c.recurrentCount*c.recurrentLanes;
 const dsp=pe*c.dspPerPE+vl*c.dspPerVectorLane+rl*c.dspPerRecurrentLane;
 const caps=(c.matrixCount?c.matrixCaps.length+c.matrixTypes.length:0)+(c.vectorCount?c.vectorCaps.length:0);
 const scratch=c.recurrentCount*c.recurrentScratch/1024;
 return {pe,vl,rl,dsp,dspLimit:c.dspBudget*(1-c.shellReserve),cost:pe*c.costMatrix+vl*c.costVector+rl*c.costRecurrent+(c.l1MiB+scratch)*c.costSRAM+caps*c.costCapability+c.rvCount*c.costRiscV+c.dmaCount*40};
}
function matrixTime(M,N,K,groups,c,p,isWeight){
 const rate=isWeight?p.rate:p.aaRate,units=c.matrixCount;
 if(!(units>0&&rate>0))return Infinity;
 const tilesM=Math.ceil(M/c.rows),tilesN=Math.ceil(N/c.cols),tiles=tilesM*tilesN*groups;
 let cycles;
 if(N<c.cols&&c.matrixCaps.includes('gemv')){
  const active=Math.min(units,Math.max(1,tilesM*groups));
  cycles=Math.ceil(M*N*K*groups/(active*c.rows*c.cols*rate)) + c.rows+c.cols;
 }else cycles=Math.ceil(tiles/units)*(K/rate+c.rows+c.cols-2);
 return cycles/(c.frequency*1e6*c.matrixEfficiency);
}
function tilePlan(M,N,K,c,p,isWeight){
 let m=Math.min(M,c.rows*4),n=Math.min(N,c.tileN),k=Math.min(K,c.tileK);
 const capacity=c.l1MiB*MiB*(1-c.stateReserve)/Math.max(1,c.matrixCount);
 const w=(isWeight?p.w:p.a)/8,a=p.a/8;
 const size=()=>2*(m*k*w+k*n*a+m*n*4);
 while(size()>capacity&&(m>1||n>1||k>1)){
  if(n>=m&&n>1)n=Math.max(1,Math.floor(n/2));else if(m>1)m=Math.max(1,Math.floor(m/2));else k=Math.max(1,Math.floor(k/2));
 }
 return {m,n,k,bytes:size(),fits:size()<=capacity};
}
function modelPhase(graph,c,phase,detail=false){
 const p=precision(c),B=c.batch,T=phase==='prefill'?c.prompt:1,f=c.frequency*1e6;
 const shapes=graph.tensors.map(t=>shapeOf(t,phase,c));
 const statePerSequence=18*(128*128*16+3*6144)*c.stateBits/8;
 const stateResident=Math.min(1,c.l1MiB*MiB*c.stateReserve/(statePerSequence*B));
 function bytes(id){const t=graph.tensors[id],s=shapes[id],e=prod(s);let bit=p.a;
  if(t.weight){bit=s[1]>1?p.w:32;return e*bit/8+(s[1]>1?Math.ceil(e/c.weightGroup)*c.scaleBytes:0);}
  if(t.dtype==='i32'||t.dtype==='i64')bit=t.dtype==='i64'?64:32;
  else if(/cache_[kv]_/.test(t.name))bit=c.kvBits;
  else if(/cache_[sr]_/.test(t.name)||/state|conv_states/.test(t.name))bit=c.stateBits;
  else if(t.name==='result_output')bit=32;
  return e*bit/8*B;
 }
 const vectorRate=c.vectorCount*c.vectorLanes*f*c.vectorEfficiency;
 const rvRate=c.rvCount*c.rvIPC*c.rvFrequency*1e6;
 const has=(...caps)=>c.vectorCount>0&&caps.every(x=>c.vectorCaps.includes(x));
 function scalar(ops,special=0){return rvRate>0?(ops*c.rvCycles+special*c.rvSpecialCycles)/rvRate:Infinity;}
 function vectorOrRV(ops,special,caps){return has(...caps)?{pool:'Vector',time:(ops+special*c.specialCycles)/vectorRate}:{pool:'RISC-V',time:scalar(ops,special)};}
 const sums={compute:0,l1:0,hbm:0,launch:0},pools={},ops={},rows=[],unmapped=new Set();
 let latency=0,hbmBytes=0,l1Bytes=0,spillBytes=0,macs=0,arithmetic=0,weightRead=0,peak=0,activeNodes=0,rvTime=0,matUtilNumerator=0,matTime=0;
 const weightNames=new Set();let weights=0;
 for(const t of graph.tensors){if(t.weight&&t.view<0&&t.op==='NONE'&&!weightNames.has(t.name)){weightNames.add(t.name);weights+=bytes(t.id);}}
 for(const id of graph.order){
  const t=graph.tensors[id],s=shapes[id],E=prod(s)*B,op=t.op,src=t.src;
  if(VIEWS.has(op)||E===0)continue;
  activeNodes++;
  let read=src.reduce((n,i)=>n+bytes(i),0),write=bytes(id),hb=0,local=0,spill=0,work=0,special=0,mm=0,outputStorage,map={pool:'Vector',time:0},extraRV=0;
  if(op==='MUL_MAT'){
   const a=shapes[src[0]],M=s[0],N=s[1]*(graph.tensors[src[0]].weight?B:1),K=a[0],groups=s[2]*s[3]*(graph.tensors[src[0]].weight?1:B),isWeight=graph.tensors[src[0]].weight;
   mm=M*N*K*groups;work=2*mm;macs+=mm;
   const type=isWeight?c.precision:p.aa,cap=isWeight?'gemm':'attention';
   if(c.matrixCount>0&&c.matrixCaps.includes(cap)&&c.matrixTypes.includes(type))map={pool:'Matrix',time:matrixTime(M,N,K,groups,c,p,isWeight)};
   else map=vectorOrRV(work,0,['gemm','elementwise']);
   const tile=tilePlan(M,N,K,c,p,isWeight),nr=Math.ceil(N/tile.n),mr=Math.ceil(M/tile.m);
   if(!tile.fits)unmapped.add('L1 too small for minimum matrix tile');
   read=(M*K*(isWeight?p.w:p.a)/8*nr+K*N*p.a/8*mr)*groups;
   if(isWeight){hb=bytes(src[0])*nr;weightRead+=hb;}
   else if(/cache_[kv]_/.test(graph.tensors[src[0]].name))hb=bytes(src[0]);
   if(map.pool==='Matrix'){matTime+=map.time;matUtilNumerator+=mm;}
   // Requantization/scale conversion is an explicit epilogue when not fused.
   if(!(map.pool==='Matrix'&&c.matrixCaps.includes('requant'))){
    const ep=vectorOrRV(3*E,0,['quant','elementwise']);map.time+=ep.time;
    if(ep.pool==='RISC-V')extraRV+=ep.time;
    local+=E*(4+p.a/8);map.epilogue=ep.pool;
   }
  }else if(op==='GATED_DELTA_NET'){
   const S=shapes[src[2]][0],H=shapes[src[2]][1],state=S*S*H*B;
   work=(7*S*S+3*S)*H*T*B;special=H*T*B;
   write=(S*H*T*B*p.a/8+state*c.stateBits/8);
   outputStorage=write;
   hb=state*c.stateBits/8*(phase==='prefill'?1:2)*(1-stateResident);
   const input=(3*S+2)*H*T*B*p.a/8;
   if(c.recurrentCount>0){
    const active=Math.min(c.recurrentCount,H*B);
    map={pool:'Recurrent',time:(work+special*c.specialCycles)/(active*c.recurrentLanes*f*c.recurrentEfficiency)};
    const fits=c.recurrentScratch*1024>=S*S*c.stateBits/8;
    read=input+state*c.stateBits/8*(fits?1:5*T);
   }else{
    map=vectorOrRV(work,special,['recurrent','elementwise','reduce','exp']);
    read=input+5*state*T*c.stateBits/8;
    write+=2*state*T*c.stateBits/8;
   }
  }else if(op==='SSM_CONV'){
   const K=shapes[src[1]][0];mm=E*K;macs+=mm;work=2*mm;
   if(c.matrixCount>0&&c.matrixCaps.includes('conv')&&c.matrixTypes.includes(c.precision))map={pool:'Matrix',time:work/(2*c.matrixCount*c.rows*c.cols*p.rate*f*c.matrixEfficiency)*4};
   else map=vectorOrRV(work,0,['conv','elementwise']);
   hb=bytes(src[1]);
  }else if(MEMORY.has(op)){
   if(op==='GET_ROWS'){
    read=write+bytes(src[1]);
    if(graph.tensors[src[0]].weight)hb=write;
   }else if(op==='CPY'||op==='CONT'||op==='SET_ROWS')read=bytes(src[0]);
   if(op==='SET_ROWS')write=bytes(src[0]);
   if(op==='SET_ROWS'&&src.some(i=>/cache_[kv]_/.test(graph.tensors[i].name)))hb=bytes(src[0]);
   map=c.dmaCount>0?{pool:'DMA',time:(read+write)/(c.dmaCount*c.dmaBytes*f)}:vectorOrRV((read+write)/4,0,['elementwise']);
  }else{
   let caps=['elementwise'];work=E;
   if(op==='RMS_NORM'){work=3*E;special=E/Math.max(1,s[0]);caps.push('reduce');}
   else if(op==='SOFT_MAX'){work=5*E;special=E;caps.push('reduce','exp');}
   else if(op==='SILU'||op==='SIGMOID'){work=3*E;special=E;caps.push('exp');}
   else if(op==='SOFTPLUS'){work=2*E;special=2*E;caps.push('exp');}
   else if(op==='SWIGLU'){work=4*E;special=E;caps.push('exp');}
   else if(op==='ROPE'){work=3*E;special=E/2;caps.push('rope');}
   else if(!['MUL','SCALE','ADD'].includes(op))unmapped.add(op+' has no cost formula');
   map=vectorOrRV(work,special,caps);
  }
  arithmetic+=work;
  // An explicit capacity-spill approximation, not a cache simulator.
  const working=op==='MUL_MAT'?0:Math.min(read,src.reduce((n,i)=>n+bytes(i),0))+(outputStorage===undefined?write:outputStorage);
  peak=Math.max(peak,working);
  if(op!=='GATED_DELTA_NET')spill=2*Math.max(0,working-c.l1MiB*MiB*(1-c.stateReserve));
  hb+=spill;
  local+=read+write+hb; // DMA ingress/egress also uses the shared SRAM ports.
  const ct=map.time,lt=local/(c.l1Banks*c.l1BytesPerBank*f*c.l1Efficiency),ht=hb/(c.hbmGBs*1e9*c.hbmEfficiency),launch=c.launchCycles/f;
  const time=(c.overlap?Math.max(ct,lt,ht):ct+lt+ht)+launch;
  if(!Number.isFinite(time))unmapped.add(op+' has no available implementation');
  const dominant=ht>=ct&&ht>=lt?'HBM':lt>=ct?'Shared L1':map.pool;
  latency+=time;hbmBytes+=hb;l1Bytes+=local;spillBytes+=spill;
  sums.compute+=ct;sums.l1+=lt;sums.hbm+=ht;sums.launch+=launch;
  pools[map.pool]=(pools[map.pool]||0)+ct;
  if(map.pool==='RISC-V')rvTime+=ct;else rvTime+=extraRV;
  if(extraRV){pools['RISC-V']=(pools['RISC-V']||0)+extraRV;pools[map.pool]-=extraRV;}
  if(!ops[op])ops[op]={op,count:0,seconds:0,hbm:0,l1:0,pools:new Set(),fallback:0};
  const agg=ops[op];agg.count++;agg.seconds+=time;agg.hbm+=hb;agg.l1+=local;agg.pools.add(map.pool);if(map.epilogue)agg.pools.add(map.epilogue);if(map.pool==='RISC-V'||extraRV)agg.fallback++;
  if(detail)rows.push({id,name:t.name,op,pool:map.pool,epilogue:map.epilogue||'',elements:E,macs:mm,seconds:time,compute:ct,l1:lt,hbm:ht,hbmBytes:hb,l1Bytes:local,dominant,working});
 }
 const kvBytes=6*2*512*round(Math.max(c.prompt,c.context+1),256)*B*c.kvBits/8;
 const requiredHBM=weights+kvBytes+statePerSequence*B+peak;
 const lowerBound=Math.max(sums.hbm,sums.l1,...Object.values(pools));
 return {phase,seconds:latency,perSequenceTPS:1/latency,aggregateTPS:(phase==='prefill'?c.prompt:1)*B/latency,
  hbmBytes,l1Bytes,spillBytes,weightRead,weights,requiredHBM,kvBytes,stateBytes:statePerSequence*B,stateResident,peakWorkingSet:peak,
  sums,pools,rvTime,macs,arithmetic,activeNodes,lowerBound,
  matrixUtilization:matTime?matUtilNumerator/(matTime*c.matrixCount*c.rows*c.cols*f*p.rate):0,
  unmapped:[...unmapped],ops:Object.values(ops).map(o=>({...o,pools:[...o.pools]})).sort((a,b)=>b.seconds-a.seconds),rows};
}
function evaluate(graph,c,detail=false){
 const errors=validation(c);if(errors.length)return {valid:false,errors,config:clone(c)};
 const r=resources(c),decode=modelPhase(graph,c,'decode',detail),prefill=modelPhase(graph,c,'prefill',detail);
 const issues=[...decode.unmapped,...prefill.unmapped];
 if(r.dsp>r.dspLimit)issues.push('Estimated DSP demand exceeds the configured usable budget.');
 if(Math.max(decode.requiredHBM,prefill.requiredHBM)>c.hbmGiB*GiB)issues.push('Weights, state, KV and peak working set exceed HBM capacity.');
 return {valid:issues.length===0,errors:[...new Set(issues)],config:clone(c),resource:r,decode,prefill};
}
function objectives(r,kind){
 if(kind==='latency-throughput')return [r.decode.seconds,-r.decode.aggregateTPS];
 if(kind==='prefill-cost')return [r.resource.cost,r.prefill.seconds];
 if(kind==='throughput-cost')return [r.resource.cost,-r.decode.aggregateTPS];
 return [r.resource.cost,r.decode.seconds];
}
function frontier(results,kind='latency-cost'){
 const valid=results.filter(r=>r.valid).map(r=>({r,v:objectives(r,kind)})).sort((a,b)=>a.v[0]-b.v[0]||a.v[1]-b.v[1]);
 const out=[];let best=Infinity;
 for(const x of valid){if(x.v[1]<best-1e-12){out.push(x.r);best=x.v[1];}}
 return out;
}
const api={defaults,clone,validation,resources,evaluate,modelPhase,shapeOf,precision,frontier,objectives,GiB,MiB};
if(typeof module!=='undefined')module.exports=api;else root.QwenModel=api;
})(globalThis);
