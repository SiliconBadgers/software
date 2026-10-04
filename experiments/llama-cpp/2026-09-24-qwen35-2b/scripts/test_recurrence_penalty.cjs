const assert=require('node:assert/strict');
const path=require('node:path');
const R=require('./sweep_recurrence_penalty.cjs');

let tests=0;function test(name,fn){fn();console.log('PASS',name);tests++;}
const captures=[
 {name:'pp128-prefill',phase:'prefill',path:path.resolve(__dirname,'../graphs/pp128/prefill.json')},
 {name:'pp128-decode',phase:'decode',path:path.resolve(__dirname,'../graphs/pp128/decode.json')},
 {name:'pp512-prefill',phase:'prefill',path:path.resolve(__dirname,'../graphs/pp512/prefill.json')},
 {name:'pp512-decode',phase:'decode',path:path.resolve(__dirname,'../graphs/pp512/decode.json')},
];
test('capture audit covers every recurrence instance and dtype class',()=>{
 const audit=R.auditCaptures(captures);
 assert.equal(audit.totalOperations,72);assert.equal(audit.classCount,1);
 assert.deepEqual(audit.validation,{eighteenPerCapture:true,allInputsAndOutputsF32:true,stateShape128x128x16:true});
});
test('mapping grid spans vector matrix-penalty and dedicated classes',()=>{
 const rows=R.mappingVariants({penalties:[4,16],dedicatedCounts:[1,4]});
 assert.deepEqual(rows.map(row=>row.mapping),['vector','matrix','matrix','dedicated','dedicated']);
});
test('miniature study is complete and matrix penalty is monotonic',()=>{
 const spec={schemaVersion:1,captures,configOverrides:{prompt:128,context:128,vectorCount:4},workloads:[{name:'captured',config:{prompt:128,context:128,batch:1}}],precisions:['w8a16'],memoryProfiles:[{name:'default',config:{l1Banks:32,hbmGBs:460}}],matrixCounts:[4],penalties:[4,16],dedicatedCounts:[1,4]};
 const study=R.runStudy(spec);assert.equal(study.rows.length,5);assert.equal(study.summary.evaluated,5);assert.equal(study.summary.feasible,5);
 const matrix=study.rows.filter(row=>row.mapping==='matrix').sort((a,b)=>a.recurrentMatrixPenalty-b.recurrentMatrixPenalty);
 assert(matrix[1].prefill.gdn.computeMs>matrix[0].prefill.gdn.computeMs);
 assert(matrix[1].decode.gdn.computeMs>matrix[0].decode.gdn.computeMs);
 assert.equal(study.summary.penaltyComparisons[0].phases.prefill.byMatrixCount[4].total,1);
 assert.equal(study.summary.penaltyComparisons[0].phases.prefill.beatsDedicatedByCount[1].total,1);
 assert.equal(study.summary.frontiers.length,1);
 assert(study.summary.frontiers[0].points.length>0);
 assert(R.toCsv(study.rows).includes('recurrence_matrix_penalty'));
});
console.log(`${tests} checks passed.`);
