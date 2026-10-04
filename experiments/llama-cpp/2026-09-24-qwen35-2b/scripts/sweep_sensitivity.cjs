#!/usr/bin/env node
'use strict';

const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const E = require('../explorer/engine.js');
const G = require('../explorer/graph-data.js');

const DOMINANT_KEYS = ['Matrix', 'Vector', 'Recurrent', 'DMA', 'RISC-V', 'Shared L1', 'HBM'];

function sha256(file) {
  return crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
}

function dominantSummary(phase) {
  const by = Object.fromEntries(DOMINANT_KEYS.map(key => [key, 0]));
  by.Other = 0;
  for (const row of phase.rows) {
    const key = Object.hasOwn(by, row.dominant) ? row.dominant : 'Other';
    by[key] += row.seconds;
  }
  const entries = Object.entries(by).sort((a, b) => b[1] - a[1]);
  return {
    label: entries[0][0],
    share: phase.seconds ? entries[0][1] / phase.seconds : 0,
    seconds: by,
  };
}

function summarizePhase(phase) {
  const dominant = dominantSummary(phase);
  return {
    latencyMs: phase.seconds * 1000,
    throughput: phase.aggregateTPS,
    serviceMs: {
      compute: phase.sums.compute * 1000,
      l1: phase.sums.l1 * 1000,
      hbm: phase.sums.hbm * 1000,
      launch: phase.sums.launch * 1000,
    },
    dominant: {
      label: dominant.label,
      share: dominant.share,
      modeledOperationMs: Object.fromEntries(
        Object.entries(dominant.seconds).map(([key, seconds]) => [key, seconds * 1000]),
      ),
    },
    hbmBytes: phase.hbmBytes,
    l1Bytes: phase.l1Bytes,
    spillBytes: phase.spillBytes,
    stateResident: phase.stateResident,
    lowerBoundMs: phase.lowerBound * 1000,
  };
}

function findSustainedDoublingKnee(rows, phase, threshold) {
  const feasible = rows.filter(row => row.valid && row.value > 0);
  const comparisons = [];
  for (const row of feasible) {
    const doubled = feasible.find(other => Math.abs(other.value - 2 * row.value) < 1e-12);
    if (!doubled) continue;
    const latency = row[phase].latencyMs;
    const nextLatency = doubled[phase].latencyMs;
    comparisons.push({
      value: row.value,
      doubledValue: doubled.value,
      latencyImprovement: (latency - nextLatency) / latency,
      throughputImprovement: (doubled[phase].throughput - row[phase].throughput) / row[phase].throughput,
      dominantBefore: row[phase].dominant.label,
      dominantAfter: doubled[phase].dominant.label,
    });
  }
  for (let i = 0; i < comparisons.length; i++) {
    const remaining = comparisons.slice(i);
    if (remaining.every(item => item.latencyImprovement < threshold)) {
      return {
        value: comparisons[i].value,
        atLowerSampleBound: i === 0,
        threshold,
        comparisons,
      };
    }
  }
  return {value: null, atLowerSampleBound: false, threshold, comparisons};
}

function runStudy(spec) {
  if (spec.schemaVersion !== 1) throw new Error('Unsupported sensitivity spec schema.');
  if (!(spec.plateauThreshold > 0 && spec.plateauThreshold < 1)) {
    throw new Error('plateauThreshold must be between zero and one.');
  }
  const baseline = {...E.defaults(), ...(spec.configOverrides || {})};
  const baselineErrors = E.validation(baseline);
  if (baselineErrors.length) throw new Error(`Invalid baseline: ${baselineErrors.join(' ')}`);
  const sweeps = [];
  for (const definition of spec.sweeps) {
    if (!Object.hasOwn(baseline, definition.parameter)) {
      throw new Error(`Unknown parameter: ${definition.parameter}`);
    }
    if (!Array.isArray(definition.values) || !definition.values.length || definition.values.some(v => !Number.isFinite(v))) {
      throw new Error(`Sweep ${definition.parameter} needs finite numeric values.`);
    }
    const values = [...new Set(definition.values)].sort((a, b) => a - b);
    const rows = values.map(value => {
      const result = E.evaluate(G, {...E.clone(baseline), [definition.parameter]: value}, true);
      if (!result.valid) return {value, valid: false, errors: result.errors};
      return {
        value,
        valid: true,
        resource: result.resource,
        prefill: summarizePhase(result.prefill),
        decode: summarizePhase(result.decode),
      };
    });
    sweeps.push({
      ...definition,
      values,
      rows,
      knees: {
        prefill: findSustainedDoublingKnee(rows, 'prefill', spec.plateauThreshold),
        decode: findSustainedDoublingKnee(rows, 'decode', spec.plateauThreshold),
      },
    });
  }
  return {baseline, sweeps};
}

function csvCell(value) {
  return `"${String(value ?? '').replaceAll('"', '""')}"`;
}

function toCsv(study) {
  const heads = [
    'parameter', 'unit', 'value', 'valid', 'errors', 'cost_proxy', 'assumed_dsp',
    ...['prefill', 'decode'].flatMap(phase => [
      `${phase}_latency_ms`, `${phase}_throughput`, `${phase}_compute_service_ms`,
      `${phase}_l1_service_ms`, `${phase}_hbm_service_ms`, `${phase}_launch_ms`,
      `${phase}_dominant`, `${phase}_dominant_share`, `${phase}_lower_bound_ms`,
      `${phase}_hbm_bytes`, `${phase}_l1_bytes`, `${phase}_spill_bytes`, `${phase}_state_resident`,
    ]),
  ];
  const rows = [];
  for (const sweep of study.sweeps) {
    for (const row of sweep.rows) {
      const values = [
        sweep.parameter, sweep.unit || '', row.value, row.valid, (row.errors || []).join(';'),
        row.resource?.cost, row.resource?.dsp,
      ];
      for (const phase of ['prefill', 'decode']) {
        const p = row[phase];
        values.push(
          p?.latencyMs, p?.throughput, p?.serviceMs.compute, p?.serviceMs.l1,
          p?.serviceMs.hbm, p?.serviceMs.launch, p?.dominant.label, p?.dominant.share,
          p?.lowerBoundMs, p?.hbmBytes, p?.l1Bytes, p?.spillBytes, p?.stateResident,
        );
      }
      rows.push(values);
    }
  }
  return [heads, ...rows].map(row => row.map(csvCell).join(',')).join('\n') + '\n';
}

function parseArgs(argv) {
  const args = {};
  for (let i = 0; i < argv.length; i += 2) {
    if (!argv[i].startsWith('--') || argv[i + 1] === undefined) {
      throw new Error('Usage: sweep_sensitivity.cjs --spec SPEC --json OUT.json --csv OUT.csv');
    }
    args[argv[i].slice(2)] = argv[i + 1];
  }
  for (const required of ['spec', 'json', 'csv']) if (!args[required]) throw new Error(`Missing --${required}.`);
  return args;
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const specPath = path.resolve(args.spec);
  const jsonPath = path.resolve(args.json);
  const csvPath = path.resolve(args.csv);
  const spec = JSON.parse(fs.readFileSync(specPath, 'utf8'));
  const study = runStudy(spec);
  const enginePath = path.resolve(__dirname, '../explorer/engine.js');
  const graphPath = path.resolve(__dirname, '../explorer/graph-data.js');
  const document = {
    schemaVersion: 1,
    study: spec.study,
    qualification: 'Uncalibrated analytical sensitivity study; not measured hardware performance or an architecture optimum.',
    source: {
      llamaCommit: G.llamaCommit,
      model: G.model,
      engineSha256: sha256(enginePath),
      graphSha256: sha256(graphPath),
      specSha256: sha256(specPath),
    },
    plateauDefinition: {
      threshold: spec.plateauThreshold,
      rule: 'First positive sampled value for which every available exact-doubling latency improvement is below the threshold. A lower-bound flag means the true knee may be below the sampled range. Null means the sampled range did not establish a sustained knee.',
    },
    baseline: study.baseline,
    sweeps: study.sweeps,
  };
  fs.mkdirSync(path.dirname(jsonPath), {recursive: true});
  fs.mkdirSync(path.dirname(csvPath), {recursive: true});
  fs.writeFileSync(jsonPath, JSON.stringify(document, null, 2) + '\n');
  fs.writeFileSync(csvPath, toCsv(study));
  console.log(`PASS wrote ${study.sweeps.length} sensitivity sweeps to ${jsonPath}`);
  for (const sweep of study.sweeps) {
    console.log(`${sweep.parameter}: prefill knee=${sweep.knees.prefill.value ?? 'not established'}, decode knee=${sweep.knees.decode.value ?? 'not established'}`);
  }
}

module.exports = {dominantSummary, summarizePhase, findSustainedDoublingKnee, runStudy, toCsv};
if (require.main === module) main();
