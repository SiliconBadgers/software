#!/usr/bin/env node
'use strict';

const fs=require('node:fs');
const path=require('node:path');
const crypto=require('node:crypto');
const E=require('../explorer/engine.js');
const G=require('../explorer/graph-data.js');
const Q=require('../explorer/scheduler.js');

function sha256(file){return crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');}
function key(parts){return parts.join('|');}

function auditCaptures(captures){
 const files=[],classes=new Map();let totalOperations=0;
 for(const capture of captures){
  const file=path.resolve(capture.path),document=JSON.parse(fs.readFileSync(file,'utf8'));
  const tensors=document.tensors,byId=new Map(tensors.map(tensor=>[tensor.id,tensor]));
  const operations=tensors.filter(tensor=>tensor.op==='GATED_DELTA_NET').map(tensor=>{
   const sources=tensor.sources.map(source=>byId.get(source.tensor));
   const v=sources[2],state=sources[5];
   return {
    id:tensor.id,outputDtype:tensor.dtype,sourceDtypes:sources.map(source=>source.dtype),
    stateDimension:v.shape[0],heads:v.shape[1],tokens:v.shape[2],sequences:v.shape[3],
    stateShape:state.shape,outputShape:tensor.shape,snapshotSlots:tensor.op_params_i32[0],
   };
  });
  totalOperations+=operations.length;
  for(const operation of operations){
   const signature=JSON.stringify({
    outputDtype:operation.outputDtype,sourceDtypes:operation.sourceDtypes,
    stateDimension:operation.stateDimension,heads:operation.heads,sequences:operation.sequences,
    snapshotSlots:operation.snapshotSlots,
   });
   if(!classes.has(signature))classes.set(signature,{signature,count:0,captures:new Set(),example:operation});
   const group=classes.get(signature);group.count++;group.captures.add(capture.name);
  }
  files.push({name:capture.name,phase:capture.phase,path:capture.path,sha256:sha256(file),operationCount:operations.length,operations});
 }
 const classRows=[...classes.values()].map(group=>({...group,captures:[...group.captures]}));
 return {
  files,totalOperations,classCount:classRows.length,classes:classRows,
  validation:{
   eighteenPerCapture:files.every(file=>file.operationCount===18),
   allInputsAndOutputsF32:files.every(file=>file.operations.every(operation=>operation.outputDtype==='f32'&&operation.sourceDtypes.every(dtype=>dtype==='f32'))),
   stateShape128x128x16:files.every(file=>file.operations.every(operation=>operation.stateDimension===128&&operation.heads===16&&operation.stateShape[0]===128&&operation.stateShape[1]===128&&operation.stateShape[2]===16)),
  },
 };
}

function phaseSummary(modeled,schedule){
 const gdnRows=modeled.rows.filter(row=>row.op==='GATED_DELTA_NET');
 const work=modeled.aggregateTPS*modeled.seconds;
 return {
  latencyMs:schedule.seconds*1000,serialLatencyMs:modeled.seconds*1000,throughput:work/schedule.seconds,
  criticalPathMs:schedule.bounds.criticalPath*1000,aggregateResourceBoundMs:schedule.bounds.aggregate*1000,
  recurrentMatrixMacs:modeled.recurrentMatrixMacs,
  gdn:{
   pools:[...new Set(gdnRows.flatMap(row=>[row.pool,row.secondaryPool]).filter(Boolean))],
   computeMs:gdnRows.reduce((sum,row)=>sum+row.compute,0)*1000,
   matrixMs:gdnRows.reduce((sum,row)=>sum+row.recurrentMatrixSeconds,0)*1000,
   tailMs:gdnRows.reduce((sum,row)=>sum+row.secondaryTime,0)*1000,
   l1ServiceMs:gdnRows.reduce((sum,row)=>sum+row.l1,0)*1000,
   hbmServiceMs:gdnRows.reduce((sum,row)=>sum+row.hbm,0)*1000,
  },
 };
}

function mappingVariants(spec){
 const variants=[{mapping:'vector',recurrentMatrix:false,recurrentCount:0,recurrentMatrixPenalty:null}];
 for(const penalty of spec.penalties)variants.push({mapping:'matrix',recurrentMatrix:true,recurrentCount:0,recurrentMatrixPenalty:penalty});
 for(const count of spec.dedicatedCounts)variants.push({mapping:'dedicated',recurrentMatrix:false,recurrentCount:count,recurrentMatrixPenalty:null});
 return variants;
}

function summarizeComparisons(rows,spec){
 const valid=rows.filter(row=>row.valid),groups=new Map();
 for(const row of valid){
  const groupKey=key([row.workload,row.precision,row.memoryProfile,row.matrixCount]);
  if(!groups.has(groupKey))groups.set(groupKey,[]);groups.get(groupKey).push(row);
 }
 const penaltyComparisons=spec.penalties.map(penalty=>{
  const phases={};
  for(const phase of ['prefill','decode']){
   let total=0,beatsVector=0,beatsFastestDedicated=0;
   const byMatrixCount=Object.fromEntries(spec.matrixCounts.map(count=>[count,{total:0,beatsVector:0}]));
   const beatsDedicatedByCount=Object.fromEntries(spec.dedicatedCounts.map(count=>[count,{total:0,beats:0}]));
   for(const group of groups.values()){
    const matrix=group.find(row=>row.mapping==='matrix'&&row.recurrentMatrixPenalty===penalty);
    const vector=group.find(row=>row.mapping==='vector');
    const dedicated=group.filter(row=>row.mapping==='dedicated').sort((a,b)=>a[phase].latencyMs-b[phase].latencyMs)[0];
    if(!(matrix&&vector&&dedicated))continue;total++;
    const matrixClass=byMatrixCount[matrix.matrixCount];matrixClass.total++;
    if(matrix[phase].latencyMs<=vector[phase].latencyMs+1e-12){beatsVector++;matrixClass.beatsVector++;}
    if(matrix[phase].latencyMs<=dedicated[phase].latencyMs+1e-12)beatsFastestDedicated++;
    for(const count of spec.dedicatedCounts){
     const peer=group.find(row=>row.mapping==='dedicated'&&row.recurrentCount===count);
     if(!peer)continue;
     beatsDedicatedByCount[count].total++;
     if(matrix[phase].latencyMs<=peer[phase].latencyMs+1e-12)beatsDedicatedByCount[count].beats++;
    }
   }
   phases[phase]={total,beatsVector,beatsFastestDedicated,byMatrixCount,beatsDedicatedByCount};
  }
  return {penalty,phases};
 });
 const breakEven=[];
 for(const [groupKey,group] of groups){
  const [workload,precision,memoryProfile,matrixCount]=groupKey.split('|');
  for(const phase of ['prefill','decode']){
   const dedicated=group.filter(row=>row.mapping==='dedicated').sort((a,b)=>a[phase].latencyMs-b[phase].latencyMs)[0];
   const qualifying=group.filter(row=>row.mapping==='matrix'&&row[phase].latencyMs<=dedicated[phase].latencyMs+1e-12).sort((a,b)=>a.recurrentMatrixPenalty-b.recurrentMatrixPenalty);
   breakEven.push({workload,precision,memoryProfile,matrixCount:Number(matrixCount),phase,fastestDedicatedCount:dedicated.recurrentCount,largestSampledMatrixPenaltyAtOrBelowDedicated:qualifying.at(-1)?.recurrentMatrixPenalty??null});
  }
 }
 const frontierGroups=new Map();
 for(const row of valid){
  const groupKey=key([row.workload,row.precision,row.memoryProfile]);
  if(!frontierGroups.has(groupKey))frontierGroups.set(groupKey,[]);frontierGroups.get(groupKey).push(row);
 }
 const objectives=row=>[row.prefill.latencyMs,row.decode.latencyMs,row.resource.dsp,row.resource.cost];
 const dominates=(left,right)=>{
  const a=objectives(left),b=objectives(right);
  return a.every((value,index)=>value<=b[index]+1e-12)&&a.some((value,index)=>value<b[index]-1e-12);
 };
 const frontiers=[];
 const frontierMappingCounts={vector:0,matrix:0,dedicated:0};
 for(const [groupKey,group] of frontierGroups){
  const [workload,precision,memoryProfile]=groupKey.split('|');
  const frontier=group.filter(candidate=>!group.some(other=>other!==candidate&&dominates(other,candidate))).map(row=>({
   matrixCount:row.matrixCount,mapping:row.mapping,recurrentMatrixPenalty:row.recurrentMatrixPenalty,recurrentCount:row.recurrentCount,
   prefillLatencyMs:row.prefill.latencyMs,decodeLatencyMs:row.decode.latencyMs,dsp:row.resource.dsp,costProxy:row.resource.cost,
  }));
  for(const point of frontier)frontierMappingCounts[point.mapping]++;
  frontiers.push({workload,precision,memoryProfile,objectives:['prefillLatencyMs','decodeLatencyMs','dsp','costProxy'],points:frontier});
 }
 return {evaluated:rows.length,feasible:valid.length,penaltyComparisons,breakEven,frontierMappingCounts,frontiers};
}

function runStudy(spec){
 if(spec.schemaVersion!==1)throw new Error('Unsupported recurrence study schema.');
 const baseline={...E.defaults(),...(spec.configOverrides||{})},errors=E.validation(baseline);
 if(errors.length)throw new Error(`Invalid baseline: ${errors.join(' ')}`);
 const variants=mappingVariants(spec),rows=[];
 for(const workload of spec.workloads)for(const precision of spec.precisions)for(const memory of spec.memoryProfiles)for(const matrixCount of spec.matrixCounts)for(const variant of variants){
  const config={...E.clone(baseline),...workload.config,...memory.config,precision,matrixCount,...variant};
  if(variant.recurrentMatrixPenalty===null)config.recurrentMatrixPenalty=baseline.recurrentMatrixPenalty;
  const result=E.evaluate(G,config,true);
  const identity={workload:workload.name,precision,memoryProfile:memory.name,matrixCount,mapping:variant.mapping,recurrentMatrixPenalty:variant.recurrentMatrixPenalty,recurrentCount:variant.recurrentCount};
  if(!result.valid){rows.push({...identity,valid:false,errors:result.errors,resource:result.resource});continue;}
  rows.push({...identity,valid:true,resource:result.resource,prefill:phaseSummary(result.prefill,Q.schedulePhase(G,result.prefill)),decode:phaseSummary(result.decode,Q.schedulePhase(G,result.decode))});
 }
 return {baseline,rows,summary:summarizeComparisons(rows,spec)};
}

function csvCell(value){return `"${String(value??'').replaceAll('"','""')}"`;}
function toCsv(rows){
 const heads=['workload','precision','memory_profile','matrix_count','mapping','recurrence_matrix_penalty','dedicated_recurrence_count','valid','errors','dsp','cost_proxy',...['prefill','decode'].flatMap(phase=>[`${phase}_latency_ms`,`${phase}_throughput`,`${phase}_gdn_compute_ms`,`${phase}_gdn_matrix_ms`,`${phase}_gdn_tail_ms`,`${phase}_gdn_l1_ms`,`${phase}_gdn_hbm_ms`,`${phase}_gdn_pools`])];
 const body=rows.map(row=>[
  row.workload,row.precision,row.memoryProfile,row.matrixCount,row.mapping,row.recurrentMatrixPenalty,row.recurrentCount,row.valid,(row.errors||[]).join(';'),row.resource?.dsp,row.resource?.cost,
  ...['prefill','decode'].flatMap(phase=>[row[phase]?.latencyMs,row[phase]?.throughput,row[phase]?.gdn.computeMs,row[phase]?.gdn.matrixMs,row[phase]?.gdn.tailMs,row[phase]?.gdn.l1ServiceMs,row[phase]?.gdn.hbmServiceMs,row[phase]?.gdn.pools.join(';')]),
 ]);
 return [heads,...body].map(row=>row.map(csvCell).join(',')).join('\n')+'\n';
}

function parseArgs(argv){
 const args={};for(let i=0;i<argv.length;i+=2){if(!argv[i].startsWith('--')||argv[i+1]===undefined)throw new Error('Usage: sweep_recurrence_penalty.cjs --spec SPEC --json OUT.json --csv OUT.csv');args[argv[i].slice(2)]=argv[i+1];}
 for(const required of ['spec','json','csv'])if(!args[required])throw new Error(`Missing --${required}.`);return args;
}

function main(){
 const args=parseArgs(process.argv.slice(2)),specPath=path.resolve(args.spec),jsonPath=path.resolve(args.json),csvPath=path.resolve(args.csv);
 const spec=JSON.parse(fs.readFileSync(specPath,'utf8')),captureAudit=auditCaptures(spec.captures),study=runStudy(spec);
 if(!Object.values(captureAudit.validation).every(Boolean))throw new Error(`Capture audit failed: ${JSON.stringify(captureAudit.validation)}`);
 const document={
  schemaVersion:1,study:spec.study,
  qualification:'Class-wide analytical sensitivity study over captured F32 gated-delta recurrence. Penalties are assumptions, not measured multiplier throughput, synthesis results, or an architecture recommendation.',
  source:{llamaCommit:G.llamaCommit,model:G.model,engineSha256:sha256(path.resolve(__dirname,'../explorer/engine.js')),graphSha256:sha256(path.resolve(__dirname,'../explorer/graph-data.js')),schedulerSha256:sha256(path.resolve(__dirname,'../explorer/scheduler.js')),sweepScriptSha256:sha256(__filename),specSha256:sha256(specPath)},
  arithmeticModel:{stateDimension:128,heads:16,perHeadToken:{matrixMacs:'3*S*S (two state-vector products plus one outer-product update)',nonMatrixBasicOps:'S*S + 3*S',specialFunctions:'one exponential'},penaltyDefinition:'W8 x A16-equivalent PE cycles charged per 32-bit recurrence MAC.'},
  dimensions:{workloads:spec.workloads,precisions:spec.precisions,memoryProfiles:spec.memoryProfiles,matrixCounts:spec.matrixCounts,penalties:spec.penalties,dedicatedCounts:spec.dedicatedCounts},
  captureAudit,baseline:study.baseline,summary:study.summary,rows:study.rows,
 };
 fs.mkdirSync(path.dirname(jsonPath),{recursive:true});fs.mkdirSync(path.dirname(csvPath),{recursive:true});
 fs.writeFileSync(jsonPath,JSON.stringify(document,null,2)+'\n');fs.writeFileSync(csvPath,toCsv(study.rows));
 console.log(`PASS audited ${captureAudit.totalOperations} captured recurrence operations in ${captureAudit.classCount} arithmetic class(es).`);
 console.log(`PASS evaluated ${study.summary.evaluated} recurrence mappings (${study.summary.feasible} feasible).`);
}

module.exports={auditCaptures,mappingVariants,runStudy,summarizeComparisons,toCsv};
if(require.main===module)main();
