'use strict';

const assert = require('node:assert/strict');
const E = require('../explorer/engine.js');
const G = require('../explorer/graph-data.js');
const S = require('./sweep_sensitivity.cjs');

let tests = 0;
function test(name, fn) { fn(); console.log('PASS', name); tests++; }

test('sustained doubling knee rejects a temporary plateau', () => {
  const rows = [
    [1, 100], [2, 99], [4, 70], [8, 69], [16, 68],
  ].map(([value, latencyMs]) => ({
    value, valid: true,
    decode: {latencyMs, throughput: 1000 / latencyMs, dominant: {label: 'Matrix'}},
  }));
  assert.equal(S.findSustainedDoublingKnee(rows, 'decode', 0.02).value, 4);
});

test('dominant modeled-operation times sum to phase latency', () => {
  const result = E.evaluate(G, E.defaults(), true);
  const summary = S.dominantSummary(result.decode);
  const total = Object.values(summary.seconds).reduce((a, b) => a + b, 0);
  assert(Math.abs(total - result.decode.seconds) < 1e-12);
});

test('one-dimensional study preserves defaults and emits diagnostics', () => {
  const before = E.defaults();
  const study = S.runStudy({
    schemaVersion: 1,
    plateauThreshold: 0.02,
    configOverrides: {batch: 1, prompt: 512, context: 2048, precision: 'w8a16'},
    sweeps: [{parameter: 'hbmGBs', unit: 'GB/s', values: [100, 200]}],
  });
  assert.deepEqual(E.defaults(), before);
  assert.equal(study.sweeps[0].rows.length, 2);
  assert(study.sweeps[0].rows.every(row => row.valid));
  assert(study.sweeps[0].rows[1].decode.latencyMs <= study.sweeps[0].rows[0].decode.latencyMs);
  assert(Number.isFinite(study.sweeps[0].rows[0].decode.serviceMs.hbm));
});

test('CSV includes fixed-workload latency and service fields', () => {
  const study = S.runStudy({
    schemaVersion: 1,
    plateauThreshold: 0.02,
    sweeps: [{parameter: 'matrixCount', unit: 'units', values: [1]}],
  });
  const csv = S.toCsv(study);
  assert(csv.includes('decode_latency_ms'));
  assert(csv.includes('decode_hbm_service_ms'));
  assert(csv.includes('matrixCount'));
});

console.log(`${tests} sensitivity checks passed.`);
