'use strict';
const E=QwenModel,G=QWEN_GRAPH,$=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)];
const QA=new URLSearchParams(location.search).has('qa');
let config=E.defaults(),activeTab='units',result,candidates=[],running=false,runId=0,sweepSnapshot='',debounce;
try{const saved=QA?null:JSON.parse(localStorage.getItem('qwen-design-v1'));if(saved){const v={...E.defaults(),...saved};if(!E.validation(v).length)config=v;}}catch(e){}
const esc=s=>String(s).replace(/[&<>"']/g,x=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[x]));
const num=(n,d=1)=>Number.isFinite(n)?(n!==0&&Math.abs(n)<.1?n.toPrecision(2):n.toLocaleString(undefined,{maximumFractionDigits:d})):'—';
const fmtMs=s=>s>=1?num(s,2)+' s':num(s*1000,2)+' ms';
const compact=n=>Math.abs(n)>=1e9?num(n/1e9,2)+'G':Math.abs(n)>=1e6?num(n/1e6,1)+'M':Math.abs(n)>=1e3?num(n/1e3,1)+'k':num(n,2);
const matrixCaps={gemm:'GEMM',gemv:'GEMV lane reuse',attention:'Attention A × A',conv:'Depthwise conv',requant:'Fused requant'};
const vectorCaps={elementwise:'Elementwise',reduce:'Reductions / rsqrt',exp:'Exp / log / sigmoid',rope:'RoPE / sin / cos',quant:'Requantization',recurrent:'Recurrence loop',conv:'Depthwise conv',gemm:'Software GEMM'};
function toast(s){$('#toast').textContent=s;$('#toast').hidden=false;setTimeout(()=>$('#toast').hidden=true,5000);}
function save(){if(QA)return;try{localStorage.setItem('qwen-design-v1',JSON.stringify(config));}catch(e){}}
function field(key,label,unit='',step=1,scale=1,min=0,max=''){
 return `<div class="field"><label for="p-${key}">${label}</label><div class="value"><input id="p-${key}" data-key="${key}" data-scale="${scale}" type="number" value="${numInput(config[key]*scale)}" min="${min}" ${max!==''?'max="'+max+'"':''} step="${step}"><span class="unit">${unit}</span></div></div>`;
}
function numInput(n){return Math.round(n*1e6)/1e6;}
function checks(key,options){return `<div class="caps">${Object.entries(options).map(([v,label])=>`<label class="check"><input type="checkbox" data-array="${key}" value="${v}" ${config[key].includes(v)?'checked':''}>${label}</label>`).join('')}</div>`;}
function section(title,body){return `<section class="control-section"><h3>${title}</h3>${body}</section>`;}
function controls(){
 $('#quick').innerHTML=field('frequency','Accelerator clock','MHz',25,1,1)+`<div class="field"><label for="precision">Matrix precision</label><select id="precision"><option value="w8a16" ${config.precision==='w8a16'?'selected':''}>W8 / A16 · INT32 acc</option><option value="w4a8" ${config.precision==='w4a8'?'selected':''}>W4 / A8 · INT32 acc</option></select></div>`;
 let s='';
 if(activeTab==='units'){
  s+=section('Matrix units',field('matrixCount','Units','',1)+field('rows','Array rows','',8,1,1)+field('cols','Array columns','',8,1,1)+field('matrixEfficiency','Compute efficiency','%',5,100,1,100)+checks('matrixCaps',matrixCaps)+`<details><summary>Supported types & MACs / PE / cycle</summary>${checks('matrixTypes',{w8a16:'W8 × A16',w4a8:'W4 × A8',a16a16:'A16 × A16',a8a8:'A8 × A8'})}${field('rateW8A16','W8 × A16','MAC',.25)}${field('rateW4A8','W4 × A8','MAC',.25)}${field('rateA16','A16 × A16','MAC',.25)}${field('rateA8','A8 × A8','MAC',.25)}<p class="small">Packing rates are assumptions. DSP cost uses physical PEs, independent of the active precision.</p></details>`);
  s+=section('Vector units',field('vectorCount','Units')+field('vectorLanes','Lanes / unit','',8,1,1)+field('vectorEfficiency','Lane efficiency','%',5,100,1,100)+field('specialCycles','Special function cost','cycles',1)+checks('vectorCaps',vectorCaps));
  s+=section('Dedicated recurrence',field('recurrentCount','Units')+field('recurrentLanes','Arithmetic lanes / unit','',16,1,1)+field('recurrentScratch','Scratch / unit','KiB',16)+field('recurrentEfficiency','Lane efficiency','%',5,100,1,100)+`<p class="small">A unit implements the whole gated-delta update, including exp. Set count to zero to use vector or RISC-V.</p>`);
  s+=section('DMA / movement',field('dmaCount','Engines')+field('dmaBytes','Bytes / engine / cycle','B',16,1,1));
  s+=section('RISC-V fallback',field('rvCount','Cores')+field('rvFrequency','Core clock','MHz',25,1,1)+field('rvIPC','Instructions / cycle','IPC',.25,1,.01)+field('rvCycles','Instructions / basic op','',1,1,.1)+field('rvSpecialCycles','Instructions / special','',5)+`<p class="small">Software fallback, no tokenization. Zero cores makes unsupported operations unmapped.</p>`);
 }else if(activeTab==='memory'){
  s+=section('Shared L1 SRAM',field('l1MiB','Capacity','MiB',1,1,.01)+field('l1Banks','Banks','',1,1,1)+field('l1BytesPerBank','Bytes / bank / cycle','B',4,1,1)+field('l1Efficiency','Bank / arbitration efficiency','%',5,100,1,100)+field('stateReserve','Budget for resident state','%',5,100,0,95));
  s+=section('HBM',field('hbmGiB','Capacity','GiB',1,1,.01)+field('hbmGBs','Peak bandwidth','GB/s',10,1,.01)+field('hbmEfficiency','Sustained efficiency','%',5,100,1,100)+`<p class="small">Default: your 460 GB/s baseline. AWS advertises up to 460 GiB/s (493.9 GB/s); overhead and access patterns reduce sustained bandwidth.</p>`);
  s+=section('Storage & tiling',field('kvBits','KV storage','bits',8,1,8,32)+field('stateBits','Recurrent state','bits',16,1,16,32)+field('tileN','Max tokens / weight tile','',16,1,1)+field('tileK','Reduction tile','',64,1,1)+field('weightGroup','Weights / quant group','',32,1,1)+field('scaleBytes','Scale bytes / group','B',1));
 }else if(activeTab==='workload'){
  s+=section('Workload',field('batch','Independent sequences','',1,1,1)+field('prompt','Prefill prompt','tokens',128,1,2)+field('context','Decode past context','tokens',256,1,1)+`<p class="small">Prefill starts from an empty context and emits one logit row per sequence. Decode is one step at the specified past context. Attention length rounds to 256, matching the captured graph.</p>`);
  s+=section('Execution assumptions',field('launchCycles','Launch overhead / operation','cycles',10)+`<label class="check"><input id="overlap" type="checkbox" ${config.overlap?'checked':''}>Overlap compute, L1 and HBM within each operation</label><p class="small">Off means add all three service times. Operations themselves follow the captured serial order. This bounds transfer overlap, not every source of uncertainty.</p>`);
 }else{
  s+=section('FPGA resource assumptions',field('dspBudget','Device DSP ceiling','',64,1,1)+field('shellReserve','DSP reserve for shell','%',5,100,0,95)+field('dspPerPE','DSPs / matrix PE','',.25)+field('dspPerVectorLane','DSPs / vector lane','',.25)+field('dspPerRecurrentLane','DSPs / recurrence lane','',.25)+`<p class="small">9,024 is the whole-device DSP count. The 20% reserve and per-lane costs are placeholders, not an AWS shell resource report. LUT/BRAM/URAM and timing closure are not checked.</p>`);
  s+=section('Editable cost proxy',field('costMatrix','Cost / matrix PE','',.25)+field('costVector','Cost / vector lane','',1)+field('costRecurrent','Cost / recurrence lane','',1)+field('costSRAM','Cost / MiB SRAM','',16)+field('costCapability','Cost / enabled capability','',10)+field('costRiscV','Cost / RISC-V core','',50)+`<p class="small">Relative units, not silicon area or dollars. Calibration can change the Pareto frontier. Capability/type overhead is counted once per unit family; DMA costs 40 units per engine.</p>`);
 }
 $('#controlBody').innerHTML=s;
 bindInputs();
}
function bindInputs(){
 $$('.controls [data-key]').forEach(el=>el.oninput=()=>{config[el.dataset.key]=Number(el.value)/Number(el.dataset.scale);queueUpdate();});
 $$('.controls [data-array]').forEach(el=>el.onchange=()=>{const k=el.dataset.array;config[k]=config[k].filter(x=>x!==el.value);if(el.checked)config[k].push(el.value);queueUpdate();});
 $('#precision').onchange=e=>{config.precision=e.target.value;queueUpdate();};
 if($('#overlap'))$('#overlap').onchange=e=>{config.overlap=e.target.checked;queueUpdate();};
}
function queueUpdate(){clearTimeout(debounce);debounce=setTimeout(update,180);}
function metric(label,value,unit,note){return `<div class="metric"><div class="metric-label">${label}</div><div class="metric-value">${value}<small>${unit}</small></div><div class="metric-note">${note}</div></div>`;}
function update(){
 result=E.evaluate(G,config,true);save();
 $('#currentLabel').textContent=`${config.matrixCount} matrix · ${config.vectorCount} vector · ${config.recurrentCount} recurrence\n${config.frequency} MHz · batch ${config.batch}`;
 if(!result.decode){$('#warnings').innerHTML='<div class="notice error">'+result.errors.map(esc).join('<br>')+'</div>';$('#metrics').innerHTML=metric('Invalid parameters','—','','Fix the highlighted assumptions.');for(const sel of ['#paretoChart','#batchChart'])$(sel).innerHTML='<div class="empty">Correct invalid parameters to model this design.</div>';for(const sel of ['#frontierTable','#batchTable','#serviceBars','#resourceSummary','#operationTable'])$(sel).innerHTML='';return;}
 let notices=[];
 if(!result.valid)notices.push(`<div class="notice error"><strong>Outside this model’s feasibility checks.</strong> ${result.errors.map(esc).join(' ')} Timings below are conditional; this design is excluded from the frontier.</div>`);
 if(result.decode.rvTime>result.decode.seconds*.1)notices.push(`<div class="notice"><strong>RISC-V fallback is significant.</strong> ${num(result.decode.rvTime*1000,1)} ms of scalar compute demand per decode step. Check the operation mapping below.</div>`);
 $('#warnings').innerHTML=notices.join('');
 const d=result.decode,p=result.prefill;
 $('#metrics').innerHTML=metric('Decode latency',num(d.seconds*1000,1),'ms / step',`${num(1/d.seconds,1)} tokens/s per sequence`)+metric('Decode throughput',num(d.aggregateTPS,1),'tokens/s',`Across ${config.batch} independent sequence${config.batch===1?'':'s'}`)+metric('Prefill latency',num(p.seconds*1000,0),'ms',`${num(p.aggregateTPS,0)} prompt tokens/s · ${config.prompt} tokens/seq`)+metric('Design cost',num(result.resource.cost,0),'proxy units',`${num(result.resource.dsp,0)} assumed DSPs / ${num(result.resource.dspLimit,0)} budget`);
 $('#fabricUnits').innerHTML=[['Matrix',config.matrixCount],['Vector',config.vectorCount],['GDN',config.recurrentCount],['RISC-V',config.rvCount]].map(([n,c])=>`<span class="${c?'':'off'}">${n} × ${c}</span>`).join('');
 $('#fabricMemory').textContent=`${config.l1MiB} MiB · ${num(config.l1Banks*config.l1BytesPerBank*config.frequency/1000*config.l1Efficiency,1)} effective GB/s`;
 $('#fabricHBM').textContent=`${config.hbmGiB} GiB · ${num(config.hbmGBs*config.hbmEfficiency,0)} GB/s`;
 drawPareto();drawBatch();breakdown();
 if(candidates.length&&JSON.stringify(config)!==sweepSnapshot&&!running)$('#sweepStatus').textContent=`${candidates.length} designs in the saved sweep snapshot. Current design shown separately.`;
}
function svgFrame(w,h){return `<svg viewBox="0 0 ${w} ${h}" role="img" xmlns="http://www.w3.org/2000/svg">`;}
function chartScale(values,lo,hi,log=false){let min=Math.min(...values.filter(Number.isFinite)),max=Math.max(...values.filter(Number.isFinite));if(!Number.isFinite(min)){min=1;max=10;}if(log){min=Math.max(1e-6,min);max=Math.max(min*1.2,max);}if(max===min){max=min+Math.max(1,Math.abs(min)*.15);min=Math.max(log?1e-6:0,min*.85);}let a=log?Math.log10(min):min,b=log?Math.log10(max):max,pad=(b-a)*.09;a-=pad;b+=pad;if(!log)a=Math.max(0,a);const f=v=>lo+((log?Math.log10(Math.max(1e-9,v)):v)-a)/(b-a)*(hi-lo);f.ticks=Array.from({length:5},(_,i)=>log?10**(a+(b-a)*i/4):a+(b-a)*i/4);return f;}
function color(r){return r.config.recurrentCount>0?'var(--orange)':r.config.matrixCount===0||r.config.vectorCount===0?'var(--yellow)':'var(--accent)';}
function viewValues(r,kind){if(kind==='latency-throughput')return [r.decode.seconds*1000,r.decode.aggregateTPS];if(kind==='throughput-cost')return [r.resource.cost,r.decode.aggregateTPS];if(kind==='prefill-cost')return [r.resource.cost,r.prefill.seconds*1000];return [r.resource.cost,r.decode.seconds*1000];}
function description(r){return `${r.config.matrixCount}× ${r.config.rows}×${r.config.cols} matrix · ${r.config.vectorCount}×${r.config.vectorLanes} vector · ${r.config.recurrentCount} GDN\n${r.config.precision.toUpperCase()} · ${r.config.frequency} MHz · batch ${r.config.batch}\n${fmtMs(r.decode.seconds)} / decode step · ${num(r.decode.aggregateTPS,1)} tokens/s\nCost ${num(r.resource.cost,0)} · ${num(r.resource.dsp,0)} assumed DSPs`;}
let plotted=[];
function drawPareto(){
 const kind=$('#objective').value,log=$('#logScale').checked;
 const valid=candidates.filter(r=>r.valid&&Number.isFinite(r.decode.seconds));
 const all=result&&result.decode&&result.valid?[...valid,result]:valid;
 plotted=all;
 if(!all.length){$('#paretoChart').innerHTML='<div class="empty"><strong>No feasible designs to plot.</strong><small>Enable an implementation path or adjust resource limits.</small></div>';$('#frontierTable').innerHTML='';return;}
 const W=780,H=335,L=76,R=22,T=16,B=58;
 const vals=all.map(r=>viewValues(r,kind)),sx=chartScale(vals.map(v=>v[0]),L,W-R,log&&kind==='latency-throughput'),sy=chartScale(vals.map(v=>v[1]),H-B,T,log);
 let s=svgFrame(W,H)+`<title>Design tradeoffs. Points are selectable.</title>`;
 for(const t of sx.ticks)s+=`<line x1="${sx(t)}" y1="${T}" x2="${sx(t)}" y2="${H-B}" stroke="var(--border)"/><text x="${sx(t)}" y="${H-B+23}" text-anchor="middle">${compact(t)}</text>`;
 for(const t of sy.ticks)s+=`<line x1="${L}" y1="${sy(t)}" x2="${W-R}" y2="${sy(t)}" stroke="var(--border)"/><text x="${L-11}" y="${sy(t)+4}" text-anchor="end">${compact(t)}</text>`;
 const front=E.frontier(all,kind);
 if(front.length>1)s+=`<polyline points="${front.map(r=>{const v=viewValues(r,kind);return sx(v[0])+','+sy(v[1]);}).join(' ')}" stroke="var(--accent)" stroke-width="1.7" stroke-dasharray="5 4" fill="none" opacity=".9"/>`;
 all.forEach((r,i)=>{const v=vals[i],current=r===result;s+=`<circle class="design-point" data-point="${i}" tabindex="0" role="button" aria-label="${esc(description(r))}" cx="${sx(v[0])}" cy="${sy(v[1])}" r="${current?6:4}" fill="${current?'var(--text)':color(r)}" stroke="var(--surface)" stroke-width="${current?3:1}" opacity="${current?1:.7}"/>`;});
 const xName=kind==='latency-throughput'?'Decode latency · ms / step':'Design cost · relative proxy units',yName=kind.includes('throughput')?'Aggregate tokens/s':kind==='prefill-cost'?'Prefill latency · ms':'Decode latency · ms';
 s+=`<text x="${(L+W-R)/2}" y="${H-9}" text-anchor="middle">${xName}${log&&kind==='latency-throughput'?' (log)':''}</text><text transform="translate(18 ${(T+H-B)/2}) rotate(-90)" text-anchor="middle">${yName}${log?' (log)':''}</text></svg>`;
 $('#paretoChart').innerHTML=s;
 $$('.design-point').forEach(el=>{const r=all[+el.dataset.point];el.onpointerenter=e=>showTooltip(e,description(r)+'\nClick to load design');el.onpointermove=positionTooltip;el.onpointerleave=()=>$('#tooltip').hidden=true;el.onclick=()=>loadDesign(r.config);el.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();loadDesign(r.config);}};});
 const shown=front;
 $('#frontierTable').innerHTML=`<div class="table-heading"><span>${front.length} nondominated design${front.length===1?'':'s'} · ${valid.length} / ${candidates.length} sweep candidates feasible</span><span>${kind==='latency-throughput'?'Lower latency':'Lower cost'}, ${kind.includes('throughput')?'higher throughput':'lower latency'}</span></div><div class="table-wrap"><table><thead><tr><th>Matrix / vector / GDN</th><th>Mode</th><th>Batch</th><th>Cost</th><th>Decode ms</th><th>Total tok/s</th></tr></thead><tbody>${shown.map((r,i)=>`<tr class="clickable" data-front="${i}" tabindex="0"><td>${r.config.matrixCount}×${r.config.rows}×${r.config.cols} / ${r.config.vectorCount}×${r.config.vectorLanes} / ${r.config.recurrentCount}</td><td>${r.config.precision==='w4a8'?'W4/A8':'W8/A16'}</td><td>${r.config.batch}</td><td>${num(r.resource.cost,0)}</td><td>${num(r.decode.seconds*1000,1)}</td><td>${num(r.decode.aggregateTPS,1)}</td></tr>`).join('')}</tbody></table></div>`;
 $$('[data-front]').forEach(el=>{el.onclick=()=>loadDesign(shown[+el.dataset.front].config);el.onkeydown=e=>{if(e.key==='Enter')el.click();};});
}
function showTooltip(e,text){const el=$('#tooltip');el.textContent=text;el.style.whiteSpace='pre-line';el.hidden=false;positionTooltip(e);}
function positionTooltip(e){const el=$('#tooltip');el.style.left=Math.min(e.clientX+16,innerWidth-el.offsetWidth-15)+'px';el.style.top=Math.min(e.clientY+16,innerHeight-el.offsetHeight-15)+'px';}
function loadDesign(c){config=E.clone(c);controls();update();$('#tooltip').hidden=true;toast('Design loaded. The sweep snapshot is retained for comparison.');}
function drawBatch(){
 if(!result.decode)return;
 const batches=[1,2,4,8,16,32];if(!batches.includes(config.batch))batches.push(config.batch);batches.sort((a,b)=>a-b);
 const data=batches.map(b=>({b,r:E.evaluate(G,{...config,batch:b})})),good=data.filter(x=>x.r.valid);
 const W=510,H=250,L=57,R=20,T=12,B=44;
 if(!good.length){$('#batchChart').innerHTML='<div class="empty">No feasible batch configurations.</div>';$('#batchTable').innerHTML='';return;}
 const sx=chartScale(batches,L,W-R,true),sy=chartScale(good.flatMap(x=>[x.r.decode.aggregateTPS,1/x.r.decode.seconds]),H-B,T,false);
 let s=svgFrame(W,H)+'<title>Total and per-sequence decode throughput by batch size</title>';
 for(const t of sy.ticks)s+=`<line x1="${L}" x2="${W-R}" y1="${sy(t)}" y2="${sy(t)}" stroke="var(--border)"/><text x="${L-8}" y="${sy(t)+4}" text-anchor="end">${compact(t)}</text>`;
 for(const b of batches)s+=`<text x="${sx(b)}" y="${H-B+22}" text-anchor="middle">${b}</text>`;
 for(const [key,col] of [['aggregateTPS','var(--accent)'],['perSequenceTPS','var(--orange)']]){
  s+=`<polyline points="${good.map(x=>sx(x.b)+','+sy(x.r.decode[key])).join(' ')}" fill="none" stroke="${col}" stroke-width="2"/>`;
  for(const x of good)s+=`<circle cx="${sx(x.b)}" cy="${sy(x.r.decode[key])}" r="3.5" fill="${col}"/>`;
 }
 s+=`<text x="${(L+W-R)/2}" y="${H-3}" text-anchor="middle">Independent sequences · batch</text><text transform="translate(13 100) rotate(-90)" text-anchor="middle">Tokens/s</text></svg>`;
 $('#batchChart').innerHTML=s;
 $('#batchTable').innerHTML=`<table><thead><tr><th>Batch</th><th>Step ms</th><th>Per seq/s</th><th>Total/s</th></tr></thead><tbody>${data.map(x=>`<tr class="${x.b===config.batch?'selected-row':''}"><td>${x.b}</td><td>${x.r.valid?num(x.r.decode.seconds*1000,1):'infeasible'}</td><td>${x.r.valid?num(1/x.r.decode.seconds,1):'—'}</td><td>${x.r.valid?num(x.r.decode.aggregateTPS,1):'—'}</td></tr>`).join('')}</tbody></table>`;
}
function breakdown(){
 if(!result||!result.decode)return;
 const phase=$('#phase').value,d=result[phase],top=Math.max(d.sums.compute,d.sums.l1,d.sums.hbm,d.sums.launch);
 $('#breakdownSubtitle').textContent=`${d.activeNodes} nonzero operations · ${fmtMs(d.seconds)} estimated · service demands overlap and are not additive`;
 $('#serviceBars').innerHTML=[['Compute',d.sums.compute],['Shared L1',d.sums.l1],['HBM',d.sums.hbm],['Launch',d.sums.launch]].map(([name,time])=>`<div class="service-item"><div class="service-top"><span>${name}</span><span>${fmtMs(time)}</span></div><div class="bar-track"><div class="bar-fill" style="width:${Number.isFinite(time/top)?100*time/top:0}%"></div></div></div>`).join('');
 $('#resourceSummary').innerHTML=[['HBM traffic',num(d.hbmBytes/1e9,2)+' GB'],['L1 traffic',num(d.l1Bytes/1e9,2)+' GB'],['Spill proxy',num(d.spillBytes/2**20,1)+' MiB'],['Resident recurrent state',num(d.stateResident*100,0)+'%'],['HBM footprint',num(d.requiredHBM/2**30,2)+' GiB'],['RISC-V demand',fmtMs(d.rvTime)]].map(([k,v])=>`<span>${k} <b>${v}</b></span>`).join('');
 $('#operationTable').innerHTML=`<table><thead><tr><th>Operation</th><th>Count</th><th>Implementation</th><th>Latency ms</th><th>HBM MiB</th><th>L1 MiB</th></tr></thead><tbody>${d.ops.map(o=>`<tr><td>${o.op}</td><td>${o.count}</td><td class="${o.fallback?'warn':''}">${o.pools.join(' + ')}</td><td>${num(o.seconds*1000,2)}</td><td>${num(o.hbm/2**20,2)}</td><td>${num(o.l1/2**20,2)}</td></tr>`).join('')}</tbody></table>`;
}
const sweepDefs=[['matrix','Matrix unit counts','0,1,2,4,8'],['arrays','Array shapes (rows × columns)','16x16,32x32'],['vector','Vector unit counts','0,1,2,4'],['lanes','Vector lanes / unit','32'],['recurrent','Recurrence unit counts','0,1'],['frequencies','Clock frequencies · MHz','300'],['batches','Batch sizes','1']];
function sweepControls(){
 $('#sweepControls').innerHTML=sweepDefs.map(([id,label,value])=>`<div class="sweep-field"><label for="s-${id}">${label}</label><input type="text" id="s-${id}" value="${value}"></div>`).join('')+`<div class="sweep-field"><label>Precision modes</label><label class="check"><input id="s-w8" type="checkbox" checked>W8 / A16</label><label class="check"><input id="s-w4" type="checkbox" checked>W4 / A8</label></div><div class="sweep-field wide"><label>Capability variants (in addition to current set)</label><label class="check"><input id="s-nosfu" type="checkbox">Remove vector exp/log/sigmoid support</label><label class="check"><input id="s-nogemv" type="checkbox">Remove matrix GEMV lane reuse</label></div>`;
 $$('#sweepControls input').forEach(el=>el.oninput=updateCount);updateCount();
}
function sweepGrid(countOnly=false){
 const values=id=>{const out=[...new Set($('#s-'+id).value.split(',').map(x=>x.trim()))];if(out.some(x=>!/^\d+$/.test(x)))throw Error('Sweep ranges must be comma-separated integers.');return out.map(Number);};
 const arrays=$('#s-arrays').value.toLowerCase().split(',').map(x=>x.trim().split('x').map(Number));
 if(arrays.some(a=>a.length!==2||a.some(n=>!Number.isInteger(n)||n<1||n>1024)))throw Error('Use array shapes such as 16x16,32x32 (dimensions 1–1024).');
 const dims=[values('matrix'),arrays,values('vector'),values('lanes'),values('recurrent'),values('frequencies'),values('batches'),[$('#s-w8').checked?'w8a16':null,$('#s-w4').checked?'w4a8':null].filter(Boolean),['current',...($('#s-nosfu').checked?['no-sfu']:[]),...($('#s-nogemv').checked?['no-gemv']:[])]];
 if(dims.some(a=>!a.length))throw Error('Select at least one precision.');
 const count=dims.reduce((n,a)=>n*a.length,1);if(countOnly)return count;if(count>4000)throw Error('This grid has '+count+' candidates. Keep it at or below 4,000.');
 const out=[];function visit(i,a){if(i<dims.length){for(const v of dims[i])visit(i+1,[...a,v]);return;}const [matrixCount,shape,vectorCount,vectorLanes,recurrentCount,frequency,batch,precision,variant]=a;const c={...E.clone(config),matrixCount,rows:shape[0],cols:shape[1],vectorCount,vectorLanes,recurrentCount,frequency,batch,precision};if(variant==='no-sfu')c.vectorCaps=c.vectorCaps.filter(x=>x!=='exp');if(variant==='no-gemv')c.matrixCaps=c.matrixCaps.filter(x=>x!=='gemv');out.push(c);}visit(0,[]);return out;
}
function updateCount(){try{$('#candidateCount').textContent=sweepGrid(true).toLocaleString()+' candidates';}catch(e){$('#candidateCount').textContent='Check ranges';}}
async function runSweep(){
 if(running){runId++;running=false;$('#runSweep').textContent='Run sweep';$('#sweepStatus').textContent='Sweep cancelled; completed candidates retained.';drawPareto();return;}
 let grid;try{grid=sweepGrid();}catch(e){toast(e.message);return;}
 if(E.validation(config).length){toast('Fix current parameter errors first.');return;}
 const id=++runId;running=true;candidates=[];sweepSnapshot=JSON.stringify(config);$('#runSweep').textContent='Stop sweep';
 for(let i=0;i<grid.length;i+=6){if(id!==runId)return;for(const c of grid.slice(i,i+6))candidates.push(E.evaluate(G,c));$('#sweepStatus').textContent=`Evaluating ${Math.min(i+6,grid.length)} / ${grid.length} designs…`;if(i%60===0)drawPareto();await new Promise(r=>setTimeout(r,0));}
 running=false;$('#runSweep').textContent='Run sweep';const valid=candidates.filter(x=>x.valid).length;$('#sweepStatus').textContent=`${grid.length} designs evaluated · ${valid} feasible under configured checks · click a point to inspect`;drawPareto();
}
function download(name,content,type){const url=URL.createObjectURL(new Blob([content],{type})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$('#exportConfig').onclick=()=>download('qwen-design.json',JSON.stringify({schemaVersion:1,model:G.model,config,summary:result&&result.decode?{decodeSeconds:result.decode.seconds,prefillSeconds:result.prefill.seconds,cost:result.resource.cost,valid:result.valid}:null},null,2),'application/json');
$('#importConfig').onclick=()=>$('#fileInput').click();
$('#fileInput').onchange=async e=>{try{const doc=JSON.parse(await e.target.files[0].text()),source=doc.config||doc,next=E.defaults();for(const k of Object.keys(next)){if(Object.prototype.hasOwnProperty.call(source,k))next[k]=source[k];}for(const k of ['matrixCaps','matrixTypes','vectorCaps'])if(!Array.isArray(next[k])||next[k].some(v=>typeof v!=='string'))throw Error('Invalid capability list.');const errors=E.validation(next);if(errors.length)throw Error(errors.join(' '));loadDesign(next);}catch(err){toast('Import failed: '+err.message);}e.target.value='';};
$('#exportSweep').onclick=()=>{if(!candidates.length){toast('Run a sweep first.');return;}const head=['valid','matrix_units','rows','cols','vector_units','vector_lanes','recurrence_units','MHz','precision','batch','cost_proxy','assumed_DSP','decode_ms','decode_total_tps','decode_per_sequence_tps','prefill_ms','prefill_tps','vector_capabilities','matrix_capabilities','reason'];const rows=candidates.map(r=>[r.valid,r.config.matrixCount,r.config.rows,r.config.cols,r.config.vectorCount,r.config.vectorLanes,r.config.recurrentCount,r.config.frequency,r.config.precision,r.config.batch,r.resource?.cost,r.resource?.dsp,r.decode?.seconds*1000,r.decode?.aggregateTPS,r.decode?.perSequenceTPS,r.prefill?.seconds*1000,r.prefill?.aggregateTPS,r.config.vectorCaps.join(';'),r.config.matrixCaps.join(';'),r.errors.join(';')]);download('qwen-design-sweep.csv',[head,...rows].map(row=>row.map(v=>'"'+String(v??'').replaceAll('"','""')+'"').join(',')).join('\n'),'text/csv');};
$('#reset').onclick=()=>{config=E.defaults();controls();update();toast('Default assumptions restored.');};
$$('[data-tab]').forEach(el=>el.onclick=()=>{activeTab=el.dataset.tab;$$('[data-tab]').forEach(x=>x.classList.toggle('active',x===el));controls();});
$('#objective').onchange=drawPareto;$('#logScale').onchange=drawPareto;$('#phase').onchange=breakdown;$('#runSweep').onclick=runSweep;
$('#version').textContent=`Graph source: llama.cpp ${G.llamaCommit.slice(0,12)} · Model v1.0`;
controls();sweepControls();update();

if(!QA)runSweep();
