const $ = selector => document.querySelector(selector);
const $$ = selector => [...document.querySelectorAll(selector)];
const escape = value => String(value).replace(/[&<>"']/g, char => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[char]));
const number = (value, digits = 2) => Number.isFinite(value) ? value.toLocaleString(undefined, {maximumFractionDigits: digits}) : 'Unavailable';
const milliseconds = value => value >= 1 ? `${number(value)} s` : value < 0.001 ? `${number(value * 1e6)} µs` : `${number(value * 1000)} ms`;
const repo = 'https://github.com/SiliconBadgers/software';
let registry, build, study, component, config, result, sweep = [], evaluationTimer;
let capturedGraph;
const captureCache = new Map();
const colors = {Matrix: '#5076df', Vector: '#2c9b7b', Recurrent: '#b879ce', DMA: '#d28a34', 'RISC-V': '#d95664', 'Shared L1': '#2c9b7b', HBM: '#d28a34'};

function notice(message, error = false) {
  $('#notice').textContent = message;
  $('#notice').className = message ? `notice ${error ? 'error' : ''}` : '';
}

function download(filename, text, type = 'application/json') {
  const url = URL.createObjectURL(new Blob([text], {type}));
  const anchor = document.createElement('a');
  anchor.href = url; anchor.download = filename; anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function changeView() {
  const view = location.hash.slice(1) || 'model';
  const selected = ['model', 'graph', 'execution', 'work', 'sources'].includes(view) ? view : 'model';
  $$('[data-panel]').forEach(panel => panel.hidden = panel.dataset.panel !== selected);
  $$('[data-view]').forEach(link => {
    link.classList.toggle('active', link.dataset.view === selected);
    if (link.dataset.view === selected) link.setAttribute('aria-current', 'page');
    else link.removeAttribute('aria-current');
  });
  document.body.dataset.view = selected;
  if (study && selected === 'graph') renderGraph();
  if (study && selected === 'execution') renderExecution();
}

async function selectStudy() {
  component = registry.components.find(item => item.id === $('#study').value);
  const module = await import(`./${component.module}`);
  study = module.createStudy();
  for (const method of ['defaults', 'validate', 'evaluate', 'layerMap']) {
    if (typeof study[method] !== 'function') throw Error(`Study ${component.id} needs ${method}().`);
  }
  config = study.defaults();
  try {
    const saved = JSON.parse(localStorage.getItem(`sb-software-${component.id}-v1`));
    if (saved) {
      const candidate = {...config, ...saved};
      if (!study.validate(candidate).length) config = candidate;
    }
  } catch { /* Missing or invalid saved settings use this study's defaults. */ }
  $('#study-credit').innerHTML = `${escape(component.contributors.join(', '))} · <a href="${repo}/pull/${component.pullRequest}">PR #${component.pullRequest}</a>`;
  $('#model-scope').textContent = study.scope;
  $('#sweep-axis').innerHTML = study.controls.filter(control => control.type === 'number').map(control => `<option value="${control.key}" ${control.key === 'matrixCount' ? 'selected' : ''}>${escape(control.label)}</option>`).join('');
  $('#execution-layer').innerHTML = '<option value="all">All operations</option><option value="-1">Embedding</option>' + Array.from({length: 24}, (_, layer) => `<option value="${layer}">Layer ${layer} · ${(layer + 1) % 4 ? 'DeltaNet' : 'Full attention'}</option>`).join('') + '<option value="24">Output head</option>';
  clearSweep(); renderControls(); evaluate(); changeView();
}

function renderControls() {
  const groups = [...new Set(study.controls.map(control => control.group))];
  $('#controls').innerHTML = groups.map(group => `<details class="control-group" open><summary>${escape(group)}</summary>${study.controls.filter(control => control.group === group).map(control => {
    const id = `control-${control.key}`;
    if (control.type === 'boolean') return `<label class="check"><input id="${id}" data-key="${control.key}" type="checkbox" ${config[control.key] ? 'checked' : ''}>${escape(control.label)}</label>`;
    const input = control.type === 'select' ? `<select id="${id}" data-key="${control.key}">${control.options.map(([value, label]) => `<option value="${value}" ${config[control.key] === value ? 'selected' : ''}>${escape(label)}</option>`).join('')}</select>` : `<input id="${id}" data-key="${control.key}" type="number" min="${control.min}" step="${control.step}" value="${config[control.key]}">`;
    return `<label class="control" for="${id}"><span>${escape(control.label)}</span>${input}</label>`;
  }).join('')}</details>`).join('');
  $('#config-json').value = JSON.stringify(config, null, 2);
  $$('#controls [data-key]').forEach(input => input.addEventListener('input', () => {
    config[input.dataset.key] = input.type === 'checkbox' ? input.checked : input.type === 'number' ? Number(input.value) : input.value;
    $('#config-json').value = JSON.stringify(config, null, 2);
    clearTimeout(evaluationTimer); evaluationTimer = setTimeout(evaluate, 120);
  }));
}

function evaluate() {
  const errors = study.validate(config);
  if (errors.length) {
    result = null; notice(errors.join(' '), true); clearResults(); return;
  }
  result = study.evaluate(config);
  notice(result.valid ? '' : result.errors.join(' '), !result.valid);
  try { localStorage.setItem(`sb-software-${component.id}-v1`, JSON.stringify(config)); } catch { /* The study still runs when storage is unavailable. */ }
  renderMetrics(); renderCosts(); renderExecution(); renderTensors();
  const previous = sweep[0]?.base;
  if (previous && JSON.stringify(previous) !== JSON.stringify(config)) {
    $('#comparison-table').dataset.stale = 'true';
    $('#comparison-chart').setAttribute('aria-label', 'Saved comparison; current configuration has changed');
    $('#comparison-note').textContent = 'Saved comparison: settings changed after this run. Run again to compare the current configuration.';
  }
}

function clearResults() {
  for (const id of ['metrics', 'operation-table', 'timeline', 'execution-summary', 'execution-detail', 'tensor-table', 'tensor-detail']) $(`#${id}`).replaceChildren();
  $('#export-timeline').disabled = true;
}

const metric = (label, value, detail) => `<div class="metric"><span>${escape(label)}</span><strong>${escape(value)}</strong><small>${escape(detail)}</small></div>`;
function renderMetrics() {
  if (!result?.decode) return clearResults();
  $('#metrics').innerHTML = metric('Prefill', milliseconds(result.prefill.seconds), `${number(config.prompt, 0)} tokens per sequence`) + metric('Decode', milliseconds(result.decode.seconds), `${number(result.decode.aggregateTPS)} total tokens/s`) + metric('External memory', `${number(Math.max(result.prefill.requiredHBM, result.decode.requiredHBM) / 2**30)} GiB`, 'Estimated allocated footprint') + metric('Resource cost', number(result.resource.cost, 0), 'Relative cost proxy');
}

function renderCosts() {
  if (!result?.decode) return;
  const phase = result[$('#cost-phase').value];
  $('#operation-table').innerHTML = `<table><thead><tr><th>Operation</th><th>Count</th><th>Compute</th><th>Latency</th><th>External traffic</th><th>SRAM traffic</th></tr></thead><tbody>${phase.ops.map(op => `<tr><td>${escape(op.op)}</td><td>${op.count}</td><td>${escape(op.pools.join(' + '))}</td><td>${milliseconds(op.seconds)}</td><td>${number(op.hbm / 2**20)} MiB</td><td>${number(op.l1 / 2**20)} MiB</td></tr>`).join('')}</tbody></table>`;
}

function clearSweep() {
  sweep = []; $('#comparison-chart').innerHTML = '<p class="empty">Choose a parameter and run a comparison.</p>';
  $('#comparison-table').replaceChildren(); $('#export-sweep').disabled = true;
}

async function runSweep() {
  const values = $('#sweep-values').value.split(',').map(value => Number(value.trim()));
  if (!values.length || values.length > 32 || values.some(value => !Number.isFinite(value)) || $('#sweep-values').value.split(',').some(value => !value.trim())) return notice('Enter 1–32 comma-separated numbers.', true);
  if (study.validate(config).length) return notice('Correct the configuration before running a comparison.', true);
  const axis = $('#sweep-axis').value, base = JSON.parse(JSON.stringify(config));
  const next = [];
  $('#run-sweep').disabled = true;
  try {
    for (const value of values) {
      const candidate = {...base, [axis]: value};
      const outcome = study.evaluate(candidate);
      next.push({axis, value, base, config: candidate, result: outcome});
      await new Promise(resolve => setTimeout(resolve, 0));
    }
    sweep = next; renderSweep(); $('#export-sweep').disabled = false; notice('');
  } finally { $('#run-sweep').disabled = false; }
}

function renderSweep() {
  if (!sweep.length) return;
  const phase = $('#sweep-phase').value;
  const max = Math.max(...sweep.filter(item => item.result.valid).map(item => item.result[phase].seconds), 0.001);
  $('#comparison-chart').innerHTML = `<p id="comparison-note" class="muted">${escape(sweep[0].axis)} · ${phase} · all other settings held fixed</p><div class="comparison-bars">${sweep.map((item, index) => `<button class="comparison-row" data-design="${index}" ${item.result.decode ? '' : 'disabled'}><span>${item.value}</span><i style="width:${item.result.valid ? Math.max(0.5, item.result[phase].seconds / max * 100) : 0}%"></i><strong>${item.result.valid ? milliseconds(item.result[phase].seconds) : 'Infeasible'}</strong></button>`).join('')}</div>`;
  $('#comparison-table').innerHTML = `<table><thead><tr><th>${escape(sweep[0].axis)}</th><th>Prefill</th><th>Decode</th><th>Total tokens/s</th><th>Cost proxy</th><th>Status</th></tr></thead><tbody>${sweep.map(item => `<tr><td>${item.value}</td><td>${item.result.prefill ? milliseconds(item.result.prefill.seconds) : 'Unavailable'}</td><td>${item.result.decode ? milliseconds(item.result.decode.seconds) : 'Unavailable'}</td><td>${number(item.result.decode?.aggregateTPS)}</td><td>${number(item.result.resource?.cost, 0)}</td><td title="${escape(item.result.errors.join(' '))}">${item.result.valid ? 'Within configured limits' : 'Infeasible'}</td></tr>`).join('')}</tbody></table>`;
  $$('[data-design]').forEach(button => button.addEventListener('click', () => { config = JSON.parse(JSON.stringify(sweep[Number(button.dataset.design)].config)); renderControls(); evaluate(); }));
}

async function renderGraph() {
  const capture = $('#capture').value, phase = $('#graph-phase').value, layer = $('#graph-layer').value;
  $('#capture-image').src = `assets/graphs/${capture}/${phase}.layer${layer}.svg`;
  $('#raw-graph').href = `assets/graphs/${capture}/${phase}.json`;
  $('#graph-caption').textContent = `${capture} · ${phase} · layer ${layer} · original rendered capture`;
  const key = `${capture}/${phase}`;
  try {
    if (!captureCache.has(key)) {
      const response = await fetch(`assets/graphs/${key}.json`);
      if (!response.ok) throw Error('Capture data could not load.');
      captureCache.set(key, await response.json());
    }
    if (key !== `${$('#capture').value}/${$('#graph-phase').value}`) return;
    capturedGraph = captureCache.get(key);
    renderTensors();
  } catch (error) { $('#tensor-table').textContent = error.message; }
}

function renderTensors() {
  if (!study || !capturedGraph) return;
  const search = $('#tensor-search').value.toLowerCase();
  const tensors = new Map(capturedGraph.tensors.map(tensor => [tensor.id, tensor]));
  const layers = new Map(), layer = Number($('#graph-layer').value);
  let current = -1;
  for (const id of capturedGraph.scheduled_nodes) {
    const tensor = tensors.get(id), match = tensor.name.match(/-(\d+)\b|_l(\d+)\b/);
    if (match) current = Number(match[1] ?? match[2]);
    layers.set(id, current);
    if (tensor.name === 'l_out-23') current = 24;
  }
  const nodes = capturedGraph.scheduled_nodes.map(id => tensors.get(id)).filter(node => layers.get(node.id) === layer && `${node.name} ${node.op_desc}`.toLowerCase().includes(search));
  $('#tensor-table').innerHTML = `<table><thead><tr><th>Tensor</th><th>Operation</th><th>Capture shape</th><th>Type</th></tr></thead><tbody>${nodes.map(node => `<tr><td><button class="tensor-button" data-tensor="${node.id}">${escape(node.name)}</button></td><td>${escape(node.op_desc)}</td><td>${node.shape.join(' × ')}</td><td>${escape(node.dtype)}</td></tr>`).join('')}</tbody></table>`;
  $('#tensor-detail').replaceChildren();
  $$('[data-tensor]').forEach(button => button.addEventListener('click', () => {
    const node = tensors.get(Number(button.dataset.tensor));
    const inputs = node.sources.map(source => tensors.get(source.tensor));
    $('#tensor-detail').innerHTML = `<div class="detail"><h3>${escape(node.name)}</h3><p>${escape(node.op_desc)} · ${escape(node.dtype)}</p><h4>Direct inputs</h4><ul>${inputs.map(input => `<li>${escape(input.name)} · ${escape(input.op_desc)} · ${escape(input.dtype)}</li>`).join('') || '<li>No direct inputs</li>'}</ul><p class="muted">Source edges include bookkeeping operations. In-place state ordering is not expanded in this view.</p></div>`;
  }));
}

function timelineRows() {
  if (!result?.decode) return [];
  const phase = result[$('#execution-phase').value];
  let time = 0;
  return phase.rows.map(row => { const start = time; time += row.seconds; return {...row, start, end: time}; });
}

function renderExecution() {
  $('#export-timeline').disabled = !result?.decode;
  if (!result?.decode) return;
  const phase = result[$('#execution-phase').value];
  $('#execution-summary').innerHTML = metric('Total latency', milliseconds(phase.seconds), `${phase.activeNodes} operations`) + metric('Compute service', milliseconds(phase.sums.compute), 'Demand within the serial model') + metric('SRAM service', milliseconds(phase.sums.l1), 'Can overlap within each operation') + metric('External memory service', milliseconds(phase.sums.hbm), 'Can overlap within each operation');
  const layers = study.layerMap(), selected = $('#execution-layer').value;
  const rows = timelineRows().filter(row => selected === 'all' || layers.get(row.id) === Number(selected));
  if (!rows.length || !rows.every(row => Number.isFinite(row.seconds))) { $('#timeline').innerHTML = '<p class="empty">No finite timeline for this selection.</p>'; return; }
  const width = 1100, laneHeight = 34, labelWidth = 185;
  const pools = [...new Set(rows.map(row => row.pool))], start = rows[0].start, end = rows.at(-1).end;
  const scale = value => labelWidth + (value - start) / Math.max(1e-12, end - start) * (width - labelWidth - 20);
  let svg = `<svg viewBox="0 0 ${width} ${pools.length * laneHeight + 65}" role="img" aria-label="Estimated serial execution by compute pool">`;
  for (let tick = 0; tick <= 4; tick++) {
    const time = start + (end - start) * tick / 4, x = scale(time);
    svg += `<line x1="${x}" x2="${x}" y1="10" y2="${pools.length * laneHeight + 16}" class="grid-line"/><text x="${x}" y="${pools.length * laneHeight + 43}" text-anchor="${tick === 4 ? 'end' : 'start'}">${milliseconds(time)}</text>`;
  }
  pools.forEach((pool, lane) => { svg += `<text x="8" y="${lane * laneHeight + 31}">${escape(pool)}</text>`; });
  rows.forEach(row => {
    const x = scale(row.start), barWidth = Math.max(0.25, scale(row.end) - x), y = pools.indexOf(row.pool) * laneHeight + 15;
    svg += `<rect x="${x}" y="${y}" width="${barWidth}" height="22" fill="${colors[row.pool] || '#687588'}"><title>${escape(row.name)} · ${escape(row.op)} · ${milliseconds(row.seconds)}</title></rect>`;
  });
  $('#timeline').innerHTML = svg + '</svg><p class="muted">Bars show whole-operation durations on the assigned compute pool, including memory and launch costs. They are not compute utilization measurements.</p>';
  $('#execution-detail').innerHTML = `<div class="table-wrap"><table><thead><tr><th>Operation</th><th>Compute pool</th><th>Start</th><th>Duration</th><th>Compute</th><th>SRAM</th><th>External memory</th></tr></thead><tbody>${rows.map(row => `<tr><td>${escape(row.name)}<small>${escape(row.op)}</small></td><td>${escape(row.pool)}</td><td>${milliseconds(row.start)}</td><td>${milliseconds(row.seconds)}</td><td>${milliseconds(row.compute)}</td><td>${milliseconds(row.l1)}</td><td>${milliseconds(row.hbm)}</td></tr>`).join('')}</tbody></table></div>`;
}

function renderWork(statuses = build.work) {
  $('#work-list').innerHTML = registry.work.map(item => {
    const status = statuses?.find(status => status.number === item.pullRequest);
    const label = status?.merged_at ? 'Merged' : status?.state === 'open' ? 'Under review' : status?.state === 'closed' ? 'Closed' : 'See PR';
    const registered = registry.components.some(component => component.pullRequest === item.pullRequest);
    return `<article class="panel"><div class="card-top"><span class="tag ${label === 'Merged' ? 'good' : ''}">${label}</span><span class="muted">${escape(item.contributor)}</span></div><h2>${escape(item.title)}</h2><p>${escape(item.description)}</p><p class="muted">${registered ? 'Available in this workspace.' : 'Source and results are accessible through the PR; a workspace component is not registered.'}</p><a href="${repo}/pull/${item.pullRequest}">PR #${item.pullRequest} ↗</a></article>`;
  }).join('');
}

async function refreshWork() {
  renderWork();
  $('#work-status').textContent = `PR status recorded at build ${build.sha.slice(0, 8)}. Checking GitHub…`;
  try {
    const response = await fetch('https://api.github.com/repos/SiliconBadgers/software/pulls?state=all&per_page=100', {signal: AbortSignal.timeout(8000)});
    if (!response.ok) throw Error('GitHub status unavailable');
    renderWork(await response.json()); $('#work-status').textContent = 'PR status checked against GitHub just now.';
  } catch { $('#work-status').textContent = `Showing PR status recorded when this site was built (${build.sha.slice(0, 8)}). Live status is unavailable.`; }
}

function renderSources() {
  $('#source-list').innerHTML = registry.sources.map(source => `<article class="panel"><h2>${escape(source.title)}</h2><p>${escape(source.description)}</p><a href="${repo}/tree/${build.sha}/${source.path}">Browse source at this revision ↗</a></article>`).join('');
  $('#build-info').innerHTML = `<h2>This edition</h2><p>${escape(build.ref)} · <a href="${repo}/commit/${build.sha}">${build.sha.slice(0, 12)}</a></p><p>UI: SiliconBadgers software workspace. Original model and captured graphs: Zeb Taylor's import, <a href="${repo}/pull/6">PR #6</a>. The original model and experiment files are preserved.</p><p><a href="source-manifest.json">Source file hashes</a> · <a href="${repo}/blob/${build.sha}/web/README.md">Build and component instructions</a> · <a href="${repo}/blob/${build.sha}/experiments/llama-cpp/2026-09-24-qwen35-2b/explorer/MODEL.md">Model equations</a></p>`;
  $('#build-label').textContent = `${build.preview ? 'Preview' : 'Main'} · ${build.sha.slice(0, 8)}`;
  $('#preview-banner').hidden = !build.preview;
  $('#preview-banner').textContent = `Branch preview: ${build.ref}. This edition is separate from the main workspace.`;
}

async function start() {
  $('#configuration-panel').open = !matchMedia('(max-width: 760px)').matches;
  [registry, build] = await Promise.all(['registry.json', 'build-info.json'].map(async path => {
    const response = await fetch(path); if (!response.ok) throw Error(`Cannot load ${path}.`); return response.json();
  }));
  $('#study').innerHTML = registry.components.map(component => `<option value="${component.id}">${escape(component.title)}</option>`).join('');
  $('#study').addEventListener('change', () => selectStudy().catch(error => notice(error.message, true)));
  addEventListener('hashchange', changeView);
  $('#reset').onclick = () => { config = study.defaults(); renderControls(); evaluate(); };
  $('#apply-json').onclick = () => { try { const next = JSON.parse($('#config-json').value); const errors = study.validate(next); if (errors.length) throw Error(errors.join(' ')); config = next; renderControls(); evaluate(); } catch (error) { notice(error.message, true); } };
  $('#run-sweep').onclick = () => runSweep().catch(error => notice(error.message, true));
  $('#sweep-phase').onchange = renderSweep;
  $('#cost-phase').onchange = renderCosts;
  for (const id of ['capture', 'graph-phase', 'graph-layer']) $(`#${id}`).onchange = renderGraph;
  $('#tensor-search').oninput = renderTensors;
  for (const id of ['execution-phase', 'execution-layer']) $(`#${id}`).onchange = renderExecution;
  $('#export').onclick = () => download('siliconbadgers-design.json', JSON.stringify({schemaVersion: 1, component: component.id, sourceRevision: build.sha, config}, null, 2));
  $('#import').onclick = () => $('#import-file').click();
  $('#import-file').onchange = async event => {
    try {
      const file = event.target.files[0]; if (!file) return;
      const data = JSON.parse(await file.text());
      if (data.schemaVersion !== 1 || data.component !== component.id) throw Error('Select the matching study before importing this design.');
      const next = {...study.defaults(), ...data.config}, errors = study.validate(next);
      if (errors.length) throw Error(errors.join(' ')); config = next; renderControls(); evaluate();
    } catch (error) { notice(error.message, true); }
    finally { event.target.value = ''; }
  };
  $('#export-sweep').onclick = () => download('siliconbadgers-comparison.csv', 'parameter,value,prefill_ms,decode_ms,total_tokens_s,cost_proxy,valid,source_revision\n' + sweep.map(item => [item.axis, item.value, item.result.prefill?.seconds * 1000, item.result.decode?.seconds * 1000, item.result.decode?.aggregateTPS, item.result.resource?.cost, item.result.valid, build.sha].join(',')).join('\n'), 'text/csv');
  $('#export-timeline').onclick = () => download('siliconbadgers-timeline.json', JSON.stringify({component: component.id, sourceRevision: build.sha, phase: $('#execution-phase').value, model: 'serial', config, rows: timelineRows()}, null, 2));
  renderSources(); refreshWork(); await selectStudy();
  globalThis.SB_WORKSPACE_READY = true;
}
start().catch(error => notice(`The workspace could not load: ${error.message}`, true));
