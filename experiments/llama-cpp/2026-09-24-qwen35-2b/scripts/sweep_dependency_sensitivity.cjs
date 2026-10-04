#!/usr/bin/env node
'use strict';

const fs=require('node:fs');
const path=require('node:path');
const crypto=require('node:crypto');
const E=require('../explorer/engine.js');
const G=require('../explorer/graph-data.js');
const Q=require('../explorer/scheduler.js');
const S=require('./sweep_sensitivity.cjs');

function sha256(file){return crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');}

function summarize(modeled,schedule){
 const summary=S.summarizePhase(modeled);
 const work=modeled.aggregateTPS*modeled.seconds;
 return {
  ...summary,
  serialLatencyMs:modeled.seconds*1000,
  latencyMs:schedule.seconds*1000,
  throughput:work/schedule.seconds,
  schedule:{
   mode:schedule.mode,
   speedup:modeled.seconds/schedule.seconds,
   criticalPathMs:schedule.bounds.criticalPath*1000,
   aggregateResourceBoundMs:schedule.bounds.aggregate*1000,
   resourceDemandMs:Object.fromEntries(Object.entries(schedule.bounds.resourceDemands).map(([key,value])=>[key,value*1000])),
  },
 };
}

function runStudy(spec){
 if(spec.schemaVersion!==1)throw new Error('Unsupported sensitivity spec schema.');
 if(!(spec.plateauThreshold>0&&spec.plateauThreshold<1))throw new Error('plateauThreshold must be between zero and one.');
 const baseline={...E.defaults(),...(spec.configOverrides||{})};
 const baselineErrors=E.validation(baseline);
 if(baselineErrors.length)throw new Error(`Invalid baseline: ${baselineErrors.join(' ')}`);
 const sweeps=[];
 for(const definition of spec.sweeps){
  if(!Object.hasOwn(baseline,definition.parameter))throw new Error(`Unknown parameter: ${definition.parameter}`);
  const values=[...new Set(definition.values)].sort((a,b)=>a-b);
  const rows=values.map(value=>{
   const result=E.evaluate(G,{...E.clone(baseline),[definition.parameter]:value},true);
   if(!result.valid)return {value,valid:false,errors:result.errors};
   const prefillSchedule=Q.schedulePhase(G,result.prefill);
   const decodeSchedule=Q.schedulePhase(G,result.decode);
   return {value,valid:true,resource:result.resource,prefill:summarize(result.prefill,prefillSchedule),decode:summarize(result.decode,decodeSchedule)};
  });
  sweeps.push({...definition,values,rows,knees:{
   prefill:S.findSustainedDoublingKnee(rows,'prefill',spec.plateauThreshold),
   decode:S.findSustainedDoublingKnee(rows,'decode',spec.plateauThreshold),
  }});
 }
 return {baseline,sweeps};
}

function csvCell(value){return `"${String(value??'').replaceAll('"','""')}"`;}
function toCsv(study){
 const heads=['parameter','unit','value','valid','errors','cost_proxy','assumed_dsp',...['prefill','decode'].flatMap(phase=>[
  `${phase}_serial_latency_ms`,`${phase}_scheduled_latency_ms`,`${phase}_scheduled_throughput`,`${phase}_speedup`,
  `${phase}_critical_path_ms`,`${phase}_aggregate_resource_bound_ms`,`${phase}_compute_service_ms`,
  `${phase}_l1_service_ms`,`${phase}_hbm_service_ms`,`${phase}_dominant_service`,
 ])];
 const rows=[];
 for(const sweep of study.sweeps)for(const row of sweep.rows){
  const values=[sweep.parameter,sweep.unit||'',row.value,row.valid,(row.errors||[]).join(';'),row.resource?.cost,row.resource?.dsp];
  for(const phase of ['prefill','decode']){
   const p=row[phase];values.push(p?.serialLatencyMs,p?.latencyMs,p?.throughput,p?.schedule.speedup,p?.schedule.criticalPathMs,p?.schedule.aggregateResourceBoundMs,p?.serviceMs.compute,p?.serviceMs.l1,p?.serviceMs.hbm,p?.dominant.label);
  }
  rows.push(values);
 }
 return [heads,...rows].map(row=>row.map(csvCell).join(',')).join('\n')+'\n';
}

function parseArgs(argv){
 const args={};
 for(let i=0;i<argv.length;i+=2){if(!argv[i].startsWith('--')||argv[i+1]===undefined)throw new Error('Usage: sweep_dependency_sensitivity.cjs --spec SPEC --json OUT.json --csv OUT.csv');args[argv[i].slice(2)]=argv[i+1];}
 for(const required of ['spec','json','csv'])if(!args[required])throw new Error(`Missing --${required}.`);
 return args;
}

function main(){
 const args=parseArgs(process.argv.slice(2)),specPath=path.resolve(args.spec),jsonPath=path.resolve(args.json),csvPath=path.resolve(args.csv);
 const spec=JSON.parse(fs.readFileSync(specPath,'utf8')),study=runStudy(spec);
 const enginePath=path.resolve(__dirname,'../explorer/engine.js'),graphPath=path.resolve(__dirname,'../explorer/graph-data.js'),schedulerPath=path.resolve(__dirname,'../explorer/scheduler.js');
 const document={
  schemaVersion:1,study:spec.study,
  qualification:'Uncalibrated analytical list schedule over captured source dependencies; not measured hardware performance, a cycle-accurate simulation, or an architecture optimum.',
  source:{llamaCommit:G.llamaCommit,model:G.model,engineSha256:sha256(enginePath),graphSha256:sha256(graphPath),schedulerSha256:sha256(schedulerPath),specSha256:sha256(specPath)},
  scheduler:{mode:'dependency-resource',policy:'Captured-order deterministic earliest-start list scheduling. Each operation uses its modeled compute pool, shared-L1 service, and HBM service concurrently; each resource is exclusive for its modeled service duration. Dispatch is serialized. This is a coarse analytical contention abstraction, not a physical bank/interconnect simulator.'},
  plateauDefinition:{threshold:spec.plateauThreshold,rule:'First positive sampled value for which every available exact-doubling scheduled-latency improvement is below the threshold. A lower-bound flag means the true knee may be below the sampled range. Null means the sampled range did not establish a sustained knee.'},
  baseline:study.baseline,sweeps:study.sweeps,
 };
 fs.mkdirSync(path.dirname(jsonPath),{recursive:true});fs.mkdirSync(path.dirname(csvPath),{recursive:true});
 fs.writeFileSync(jsonPath,JSON.stringify(document,null,2)+'\n');fs.writeFileSync(csvPath,toCsv(study));
 console.log(`PASS wrote ${study.sweeps.length} dependency-aware sensitivity sweeps to ${jsonPath}`);
 for(const sweep of study.sweeps)console.log(`${sweep.parameter}: prefill knee=${sweep.knees.prefill.value??'not established'}, decode knee=${sweep.knees.decode.value??'not established'}`);
}

module.exports={runStudy,toCsv,summarize};
if(require.main===module)main();
