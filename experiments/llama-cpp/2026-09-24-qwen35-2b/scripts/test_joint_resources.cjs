#!/usr/bin/env node
'use strict';

const assert=require('node:assert/strict');
const J=require('./sweep_joint_resources.cjs');

let tests=0;
function test(name,fn){fn();console.log('PASS',name);tests++;}
const spec={
 schemaVersion:1,constraints:{maxHbmGBs:460},analysis:{throughputFractions:[0.5]},
 configOverrides:{batch:1,prompt:512,context:2048,precision:'w8a16',vectorCount:4,recurrentCount:4,dmaCount:1,l1MiB:8},
 dimensions:{matrixCount:[1,2],l1Banks:[32,64],hbmGBs:[100,200]},
};

test('cartesian product is complete and unique',()=>{
 const rows=J.cartesian(spec.dimensions);
 assert.equal(rows.length,8);assert.equal(new Set(rows.map(JSON.stringify)).size,8);
});

test('joint study respects dimensions and constraints',()=>{
 const study=J.runStudy(spec);
 assert.equal(study.summary.evaluated,8);assert.equal(study.summary.feasible,8);
 assert(study.rows.every(row=>row.config.hbmGBs<=460));
 assert(study.rows.every(row=>row.resource.dsp<=row.resource.dspLimit));
});

test('more sampled core resources do not worsen scheduled latency',()=>{
 const study=J.runStudy(spec),rows=study.rows.filter(row=>row.valid);
 for(const row of rows)for(const other of rows){
  const noLess=other.config.matrixCount>=row.config.matrixCount&&other.config.l1Banks>=row.config.l1Banks&&other.config.hbmGBs>=row.config.hbmGBs;
  if(noLess){assert(other.prefill.latencyMs<=row.prefill.latencyMs+1e-9);assert(other.decode.latencyMs<=row.decode.latencyMs+1e-9);}
 }
});

test('reported Pareto points are not dominated',()=>{
 const study=J.runStudy(spec),frontier=J.paretoFrontier(study.rows.filter(row=>row.valid));
 for(const row of frontier)assert(!study.rows.some(other=>other.valid&&other!==row&&J.dominates(other,row)));
});

test('CSV exposes objectives and frontier membership',()=>{
 const study=J.runStudy(spec),csv=J.toCsv({...study,dimensions:spec.dimensions});
 assert(csv.includes('prefill_ms'));assert(csv.includes('decode_ms'));assert(csv.includes('pareto'));
});

test('throughput tiers retain only resource-nondominated survivors',()=>{
 const study=J.runStudy(spec),tier=study.summary.throughputTiers[0];
 assert.equal(tier.fraction,0.5);assert(tier.survivors.length>0);
 for(const survivor of tier.survivors){
  assert(survivor.prefillThroughput>=tier.targets.prefill-1e-12);
  assert(survivor.decodeThroughput>=tier.targets.decode-1e-12);
 }
});

console.log(`${tests} joint-resource checks passed.`);
