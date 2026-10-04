#!/usr/bin/env node
'use strict';

const assert=require('node:assert/strict');
const E=require('../explorer/engine.js');
const G=require('../explorer/graph-data.js');
const Q=require('../explorer/scheduler.js');

let tests=0;
function test(name,fn){fn();console.log('PASS',name);tests++;}
function row(id,pool,seconds,services={}){return {id,name:`node-${id}`,op:'TEST',pool,seconds,compute:services.compute??seconds,l1:services.l1??0,hbm:services.hbm??0};}
function phase(rows){return {rows,activeNodes:rows.length,seconds:rows.reduce((sum,item)=>sum+item.seconds,0)};}

test('serial mode exactly preserves engine phase latency',()=>{
 const result=E.evaluate(G,E.defaults(),true);
 for(const name of ['prefill','decode']){
  const schedule=Q.schedulePhase(G,result[name],{mode:'serial'});
  assert(Math.abs(schedule.seconds-result[name].seconds)<1e-12);
 }
});

test('independent operations on different resources overlap',()=>{
 const graph={tensors:[{id:0,src:[]},{id:1,src:[]},{id:2,src:[0,1]}],order:[0,1,2]};
 const schedule=Q.schedulePhase(graph,phase([row(0,'Matrix',5),row(1,'Vector',4),row(2,'Matrix',1)]));
 assert.equal(schedule.seconds,6);
 assert.equal(schedule.bounds.criticalPath,6);
});

test('independent operations sharing a resource serialize',()=>{
 const graph={tensors:[{id:0,src:[]},{id:1,src:[]}],order:[0,1]};
 const schedule=Q.schedulePhase(graph,phase([row(0,'Matrix',5),row(1,'Matrix',4)]));
 assert.equal(schedule.seconds,9);
 assert.equal(schedule.bounds.aggregate,9);
});
test('secondary compute pools are reserved explicitly',()=>{
 const graph={tensors:[{id:0,src:[]},{id:1,src:[]}],order:[0,1]};
 const modeled=phase([
  row(0,'Matrix',4),
  {...row(1,'Matrix',3),secondaryPool:'Vector',secondaryTime:2},
 ]);
 const scheduled=Q.schedulePhase(graph,modeled);
 assert.equal(scheduled.bounds.resourceDemands.Vector,2);
 assert.equal(scheduled.calendars.Vector.length,1);
 assert.equal(scheduled.calendars.Vector[0].end-scheduled.calendars.Vector[0].start,2);
});

test('dependencies pass through inactive view nodes',()=>{
 const graph={tensors:[{id:0,src:[]},{id:1,src:[0]},{id:2,src:[1]}],order:[0,1,2]};
 const schedule=Q.schedulePhase(graph,phase([row(0,'Matrix',5),row(2,'Vector',1)]));
 assert.deepEqual(schedule.operations.find(operation=>operation.id===2).dependencies,[0]);
 assert.equal(schedule.seconds,6);
});

test('in-place cache writes precede later alias reads',()=>{
 const graph={tensors:[
  {id:0,op:'NONE',src:[],view:-1},
  {id:1,op:'SET_ROWS',src:[2,3,0],view:0},
  {id:2,op:'NONE',src:[],view:-1},
  {id:3,op:'NONE',src:[],view:-1},
  {id:4,op:'VIEW',src:[0],view:0},
  {id:5,op:'MUL',src:[4],view:-1},
 ],order:[1,5]};
 const schedule=Q.schedulePhase(graph,phase([row(1,'DMA',5),row(5,'Vector',1)]));
 assert.deepEqual(schedule.operations.find(operation=>operation.id===5).dependencies,[1]);
 assert.equal(schedule.seconds,6);
});

test('real graph schedule is deterministic and satisfies bounds',()=>{
 const modeled=E.evaluate(G,E.defaults(),true);
 const decodeDependencies=Q.activeDependencies(G,modeled.decode.rows);
 assert(decodeDependencies.get(1787).includes(1774));
 assert(decodeDependencies.get(1789).includes(1780));
 for(const name of ['prefill','decode']){
  const first=Q.schedulePhase(G,modeled[name]);
  const second=Q.schedulePhase(G,modeled[name]);
  assert.equal(first.seconds,second.seconds);
  assert(first.seconds>=first.bounds.criticalPath-1e-12);
  assert(first.seconds>=first.bounds.aggregate-1e-12);
  assert(first.seconds<=modeled[name].seconds+1e-12);
  assert.equal(first.operations.length,modeled[name].activeNodes);
 }
});

console.log(`${tests} scheduler checks passed.`);
