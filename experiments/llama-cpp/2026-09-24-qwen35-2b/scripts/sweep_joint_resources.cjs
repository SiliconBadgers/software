#!/usr/bin/env node
'use strict';

const fs=require('node:fs');
const path=require('node:path');
const crypto=require('node:crypto');
const E=require('../explorer/engine.js');
const G=require('../explorer/graph-data.js');
const Q=require('../explorer/scheduler.js');

function sha256(file){return crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');}

function cartesian(dimensions){
 const entries=Object.entries(dimensions),rows=[];
 function visit(index,current){
  if(index===entries.length){rows.push({...current});return;}
  const [parameter,values]=entries[index];
  for(const value of values)visit(index+1,{...current,[parameter]:value});
 }
 visit(0,{});return rows;
}

function summarize(modeled,schedule){
 const work=modeled.aggregateTPS*modeled.seconds;
 return {
  latencyMs:schedule.seconds*1000,
  throughput:work/schedule.seconds,
  serialLatencyMs:modeled.seconds*1000,
  speedup:modeled.seconds/schedule.seconds,
  criticalPathMs:schedule.bounds.criticalPath*1000,
  aggregateResourceBoundMs:schedule.bounds.aggregate*1000,
  resourceDemandMs:Object.fromEntries(Object.entries(schedule.bounds.resourceDemands).map(([key,value])=>[key,value*1000])),
 };
}

function objectiveVector(row){return [row.prefill.latencyMs,row.decode.latencyMs,row.resource.dsp,row.config.l1Banks,row.config.hbmGBs];}
function dominates(a,b){
 const av=objectiveVector(a),bv=objectiveVector(b);
 return av.every((value,index)=>value<=bv[index]+1e-12)&&av.some((value,index)=>value<bv[index]-1e-12);
}
function paretoFrontier(rows){return rows.filter(row=>!rows.some(other=>other!==row&&dominates(other,row)));}

function resourceDominates(a,b){
 const av=[a.resource.dsp,a.config.l1Banks,a.config.hbmGBs],bv=[b.resource.dsp,b.config.l1Banks,b.config.hbmGBs];
 return av.every((value,index)=>value<=bv[index]+1e-12)&&av.some((value,index)=>value<bv[index]-1e-12);
}

function jointLatencyChampion(rows){
 const minPrefill=Math.min(...rows.map(row=>row.prefill.latencyMs));
 const minDecode=Math.min(...rows.map(row=>row.decode.latencyMs));
 return rows.find(row=>Math.abs(row.prefill.latencyMs-minPrefill)<1e-9&&Math.abs(row.decode.latencyMs-minDecode)<1e-9)||null;
}

function throughputTiers(rows,champion,fractions){
 if(!champion)return [];
 return fractions.map(fraction=>{
  const targets={prefill:champion.prefill.throughput*fraction,decode:champion.decode.throughput*fraction};
  const qualified=rows.filter(row=>row.prefill.throughput>=targets.prefill-1e-12&&row.decode.throughput>=targets.decode-1e-12);
  const survivors=qualified.filter(row=>!qualified.some(other=>other!==row&&resourceDominates(other,row)));
  return {fraction,targets,qualified:qualified.length,survivors:survivors.map(row=>({config:row.config,dsp:row.resource.dsp,prefillThroughput:row.prefill.throughput,decodeThroughput:row.decode.throughput}))};
 });
}

function runStudy(spec){
 if(spec.schemaVersion!==1)throw new Error('Unsupported joint-sweep schema.');
 const baseline={...E.defaults(),...(spec.configOverrides||{})},errors=E.validation(baseline);
 if(errors.length)throw new Error(`Invalid baseline: ${errors.join(' ')}`);
 for(const [parameter,values] of Object.entries(spec.dimensions||{})){
  if(!Object.hasOwn(baseline,parameter))throw new Error(`Unknown parameter: ${parameter}`);
  if(!Array.isArray(values)||!values.length||values.some(value=>!Number.isFinite(value)))throw new Error(`Dimension ${parameter} requires finite numeric values.`);
 }
 const rows=cartesian(spec.dimensions).map(config=>{
  const combined={...E.clone(baseline),...config};
  if(spec.constraints?.maxHbmGBs!==undefined&&combined.hbmGBs>spec.constraints.maxHbmGBs)return {config,valid:false,errors:['HBM bandwidth exceeds study constraint.']};
  const result=E.evaluate(G,combined,true);
  if(!result.valid)return {config,valid:false,errors:result.errors,resource:result.resource};
  const prefill=Q.schedulePhase(G,result.prefill),decode=Q.schedulePhase(G,result.decode);
  return {config,valid:true,resource:result.resource,prefill:summarize(result.prefill,prefill),decode:summarize(result.decode,decode)};
 });
  const feasible=rows.filter(row=>row.valid),frontier=paretoFrontier(feasible),champion=jointLatencyChampion(feasible);
  return {
  baseline,rows,
  summary:{evaluated:rows.length,feasible:feasible.length,paretoCount:frontier.length,paretoConfigs:frontier.map(row=>row.config),jointLatencyChampion:champion?.config||null,throughputTiers:throughputTiers(feasible,champion,spec.analysis?.throughputFractions||[])},
 };
}

function csvCell(value){return `"${String(value??'').replaceAll('"','""')}"`;}
function toCsv(study){
 const dimensionNames=Object.keys(study.dimensions);
 const heads=[...dimensionNames,'valid','pareto','errors','dsp','dsp_limit','cost_proxy','prefill_ms','prefill_tps','prefill_critical_path_ms','prefill_resource_bound_ms','decode_ms','decode_tps','decode_critical_path_ms','decode_resource_bound_ms'];
 const feasible=study.rows.filter(row=>row.valid),frontier=new Set(paretoFrontier(feasible));
 const rows=study.rows.map(row=>[
  ...dimensionNames.map(name=>row.config[name]),row.valid,frontier.has(row),(row.errors||[]).join(';'),row.resource?.dsp,row.resource?.dspLimit,row.resource?.cost,
  row.prefill?.latencyMs,row.prefill?.throughput,row.prefill?.criticalPathMs,row.prefill?.aggregateResourceBoundMs,
  row.decode?.latencyMs,row.decode?.throughput,row.decode?.criticalPathMs,row.decode?.aggregateResourceBoundMs,
 ]);
 return [heads,...rows].map(row=>row.map(csvCell).join(',')).join('\n')+'\n';
}

function parseArgs(argv){
 const args={};for(let i=0;i<argv.length;i+=2){if(!argv[i].startsWith('--')||argv[i+1]===undefined)throw new Error('Usage: sweep_joint_resources.cjs --spec SPEC --json OUT.json --csv OUT.csv');args[argv[i].slice(2)]=argv[i+1];}
 for(const required of ['spec','json','csv'])if(!args[required])throw new Error(`Missing --${required}.`);return args;
}

function main(){
 const args=parseArgs(process.argv.slice(2)),specPath=path.resolve(args.spec),jsonPath=path.resolve(args.json),csvPath=path.resolve(args.csv);
 const spec=JSON.parse(fs.readFileSync(specPath,'utf8')),study=runStudy(spec);
 const document={
  schemaVersion:1,study:spec.study,
  qualification:'Uncalibrated joint analytical tradeoff study; not measured hardware performance, physical feasibility, or a calibrated cost optimum.',
  objective:{pareto:['prefill scheduled latency','decode scheduled latency','modeled DSP demand','modeled L1 bank count','peak HBM bandwidth'],note:'The Pareto set is mathematical only over these explicit modeled quantities. No area, power, timing, physical banking or accuracy objective is implied.'},
  source:{llamaCommit:G.llamaCommit,model:G.model,engineSha256:sha256(path.resolve(__dirname,'../explorer/engine.js')),graphSha256:sha256(path.resolve(__dirname,'../explorer/graph-data.js')),schedulerSha256:sha256(path.resolve(__dirname,'../explorer/scheduler.js')),sweepScriptSha256:sha256(__filename),specSha256:sha256(specPath)},
  constraints:spec.constraints||{},dimensions:spec.dimensions,baseline:study.baseline,summary:study.summary,rows:study.rows,
 };
 fs.mkdirSync(path.dirname(jsonPath),{recursive:true});fs.mkdirSync(path.dirname(csvPath),{recursive:true});
 fs.writeFileSync(jsonPath,JSON.stringify(document,null,2)+'\n');fs.writeFileSync(csvPath,toCsv({...study,dimensions:spec.dimensions}));
 console.log(`PASS evaluated ${study.summary.evaluated} joint configurations (${study.summary.feasible} feasible, ${study.summary.paretoCount} Pareto).`);
 console.log(`Joint latency champion: ${JSON.stringify(study.summary.jointLatencyChampion)}`);
}

module.exports={cartesian,dominates,jointLatencyChampion,objectiveVector,paretoFrontier,resourceDominates,runStudy,throughputTiers,toCsv};
if(require.main===module)main();
