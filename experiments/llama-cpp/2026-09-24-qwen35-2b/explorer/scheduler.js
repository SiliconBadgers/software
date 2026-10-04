/* Deterministic dependency/resource scheduler layered over engine.js detail rows. */
(function(root){
'use strict';

const EPS=1e-12;

function activeDependencies(graph,rows){
 const active=new Set(rows.map(row=>row.id)),memo=new Map();
 function visit(id,trail=new Set()){
  if(memo.has(id))return memo.get(id);
  if(trail.has(id))throw new Error(`Dependency cycle while expanding tensor ${id}.`);
  const tensor=graph.tensors[id];
  if(!tensor)throw new Error(`Missing tensor ${id}.`);
  const next=new Set(trail);next.add(id);
  const deps=new Set();
  for(const source of tensor.src||[]){
   if(active.has(source))deps.add(source);
   else for(const dependency of visit(source,next))deps.add(dependency);
  }
  memo.set(id,deps);return deps;
 }
 const dependencies=new Map(rows.map(row=>[row.id,new Set(visit(row.id))]));
 function aliasRoot(id){
  const seen=new Set();
  while(graph.tensors[id]&&graph.tensors[id].view>=0&&graph.tensors[id].view!==id){
   if(seen.has(id))throw new Error(`Alias cycle while resolving tensor ${id}.`);
   seen.add(id);id=graph.tensors[id].view;
  }
  return id;
 }
 const mutableRoots=new Set();
 for(const tensor of graph.tensors){
  if((tensor.op==='CPY'||tensor.op==='SET_ROWS')&&tensor.view>=0)mutableRoots.add(aliasRoot(tensor.view));
 }
 const state=new Map([...mutableRoots].map(root=>[root,{writer:null,readers:new Set()}]));
 for(const id of graph.order){
  if(!active.has(id))continue;
  const tensor=graph.tensors[id],reads=new Set();
  for(const source of tensor.src||[]){const root=aliasRoot(source);if(mutableRoots.has(root))reads.add(root);}
  const writes=new Set();
  if((tensor.op==='CPY'||tensor.op==='SET_ROWS')&&tensor.view>=0)writes.add(aliasRoot(tensor.view));
  for(const root of reads){const slot=state.get(root);if(slot.writer!==null)dependencies.get(id).add(slot.writer);}
  for(const root of writes){
   const slot=state.get(root);if(slot.writer!==null)dependencies.get(id).add(slot.writer);
   for(const reader of slot.readers)dependencies.get(id).add(reader);
  }
  for(const root of reads)if(!writes.has(root))state.get(root).readers.add(id);
  for(const root of writes){const slot=state.get(root);slot.writer=id;slot.readers.clear();}
 }
 return new Map([...dependencies].map(([id,deps])=>[id,[...deps].filter(dependency=>dependency!==id).sort((a,b)=>a-b)]));
}

function firstAvailable(intervals,earliest,duration){
 if(!(duration>EPS))return earliest;
 let start=earliest;
 for(const interval of intervals){
  if(start+duration<=interval.start+EPS)break;
  if(start<interval.end-EPS&&start+duration>interval.start+EPS)start=interval.end;
 }
 return start;
}

function commonStart(calendars,requirements,earliest){
 let candidate=earliest;
 for(let iteration=0;iteration<100000;iteration++){
  let next=candidate;
  for(const [resource,duration] of Object.entries(requirements)){
   next=Math.max(next,firstAvailable(calendars[resource]||[],candidate,duration));
  }
  if(Math.abs(next-candidate)<=EPS)return candidate;
  candidate=next;
 }
 throw new Error('Unable to find a common resource interval.');
}

function reserve(calendars,resource,start,duration,id){
 if(!(duration>EPS))return;
 if(!calendars[resource])calendars[resource]=[];
 calendars[resource].push({start,end:start+duration,id});
 calendars[resource].sort((a,b)=>a.start-b.start||a.end-b.end||a.id-b.id);
}

function rowRequirements(row){
 const requirements={};
 if(row.compute>EPS)requirements[row.pool]=row.compute;
 if(row.secondaryTime>EPS&&row.secondaryPool&&row.secondaryPool!==row.pool){
  requirements[row.secondaryPool]=(requirements[row.secondaryPool]||0)+row.secondaryTime;
 }
 if(row.l1>EPS)requirements['Shared L1']=row.l1;
 if(row.hbm>EPS)requirements.HBM=row.hbm;
 return requirements;
}

function dependencyCriticalPath(rows,dependencies){
 const completion=new Map();let bound=0;
 for(const row of rows){
  const ready=Math.max(0,...dependencies.get(row.id).map(id=>completion.get(id)||0));
  const end=ready+row.seconds;completion.set(row.id,end);bound=Math.max(bound,end);
 }
 return bound;
}

function aggregateBounds(rows){
 const demands={Launch:0};
 for(const row of rows){
  demands.Launch+=(row.seconds-Math.max(row.compute,row.l1,row.hbm));
  for(const [resource,duration] of Object.entries(rowRequirements(row))){
   demands[resource]=(demands[resource]||0)+duration;
  }
 }
 return {demands,maximum:Math.max(0,...Object.values(demands))};
}

function validateSchedule(schedule){
 const byId=new Map(schedule.operations.map(operation=>[operation.id,operation]));
 for(const operation of schedule.operations){
  for(const dependency of operation.dependencies){
   const predecessor=byId.get(dependency);
   if(!predecessor)throw new Error(`Operation ${operation.id} has unscheduled dependency ${dependency}.`);
   if(predecessor.end>operation.launchStart+EPS)throw new Error(`Dependency ${dependency} completes after operation ${operation.id} launches.`);
  }
 }
 for(const [resource,intervals] of Object.entries(schedule.calendars)){
  for(let i=1;i<intervals.length;i++){
   if(intervals[i-1].end>intervals[i].start+EPS)throw new Error(`${resource} is overcommitted.`);
  }
 }
 if(schedule.seconds+EPS<schedule.bounds.criticalPath)throw new Error('Schedule beats its dependency critical-path bound.');
 if(schedule.seconds+EPS<schedule.bounds.aggregate)throw new Error('Schedule beats its aggregate resource-demand bound.');
 return true;
}

function serialSchedule(graph,phase){
 const dependencies=activeDependencies(graph,phase.rows),operations=[];let cursor=0;
 for(const row of phase.rows){
  const start=cursor,end=start+row.seconds;
  operations.push({id:row.id,name:row.name,op:row.op,pool:row.pool,dependencies:dependencies.get(row.id),launchStart:start,serviceStart:start,end});
  cursor=end;
 }
 return {mode:'serial',seconds:cursor,operations,calendars:{},bounds:{criticalPath:dependencyCriticalPath(phase.rows,dependencies),aggregate:aggregateBounds(phase.rows).maximum}};
}

function dependencySchedule(graph,phase){
 if(!Array.isArray(phase.rows)||phase.rows.length!==phase.activeNodes)throw new Error('Dependency scheduling requires engine detail rows for every active operation.');
 const rowsById=new Map(phase.rows.map(row=>[row.id,row]));
 const orderIndex=new Map(graph.order.map((id,index)=>[id,index]));
 const dependencies=activeDependencies(graph,phase.rows);
 const indegree=new Map(),dependents=new Map();
 for(const row of phase.rows){indegree.set(row.id,dependencies.get(row.id).length);dependents.set(row.id,[]);}
 for(const [id,deps] of dependencies)for(const dependency of deps){
  if(!dependents.has(dependency))throw new Error(`Active dependency ${dependency} is missing.`);
  dependents.get(dependency).push(id);
 }
 const ready=phase.rows.filter(row=>indegree.get(row.id)===0).map(row=>row.id);
 const sortReady=()=>ready.sort((a,b)=>(orderIndex.get(a)??Infinity)-(orderIndex.get(b)??Infinity)||a-b);
 sortReady();
 const calendars={Launch:[]},completion=new Map(),operations=[];
 while(ready.length){
  const id=ready.shift(),row=rowsById.get(id),deps=dependencies.get(id);
  const dependencyReady=Math.max(0,...deps.map(dependency=>completion.get(dependency)));
  const launch=Math.max(0,row.seconds-Math.max(row.compute,row.l1,row.hbm));
  const launchStart=firstAvailable(calendars.Launch,dependencyReady,launch);
  reserve(calendars,'Launch',launchStart,launch,id);
  const requirements=rowRequirements(row);
  const serviceStart=commonStart(calendars,requirements,launchStart+launch);
  for(const [resource,duration] of Object.entries(requirements))reserve(calendars,resource,serviceStart,duration,id);
  const end=serviceStart+Math.max(0,...Object.values(requirements));
  completion.set(id,end);
  operations.push({id,name:row.name,op:row.op,pool:row.pool,dependencies:deps,launchStart,serviceStart,end});
  for(const dependent of dependents.get(id)){
   indegree.set(dependent,indegree.get(dependent)-1);
   if(indegree.get(dependent)===0)ready.push(dependent);
  }
  sortReady();
 }
 if(operations.length!==phase.rows.length)throw new Error(`Dependency graph scheduled ${operations.length} of ${phase.rows.length} active operations.`);
 operations.sort((a,b)=>a.launchStart-b.launchStart||a.serviceStart-b.serviceStart||(orderIndex.get(a.id)??Infinity)-(orderIndex.get(b.id)??Infinity));
 const aggregate=aggregateBounds(phase.rows);
 const schedule={
  mode:'dependency-resource',
  seconds:Math.max(0,...operations.map(operation=>operation.end)),
  operations,calendars,
  bounds:{criticalPath:dependencyCriticalPath(phase.rows,dependencies),aggregate:aggregate.maximum,resourceDemands:aggregate.demands},
 };
 validateSchedule(schedule);return schedule;
}

function schedulePhase(graph,phase,options={}){
 const mode=options.mode||'dependency-resource';
 if(mode==='serial')return serialSchedule(graph,phase);
 if(mode==='dependency-resource')return dependencySchedule(graph,phase);
 throw new Error(`Unsupported scheduler mode: ${mode}`);
}

const api={activeDependencies,aggregateBounds,dependencyCriticalPath,schedulePhase,validateSchedule};
if(typeof module!=='undefined')module.exports=api;else root.QwenScheduler=api;
})(globalThis);
