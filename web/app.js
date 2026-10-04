import { drawDependencies } from "./graph.js";
import { initializeStudies } from "./studies.js";
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const escape = (value) =>
  String(value).replace(
    /[&<>"']/g,
    (char) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        char
      ],
  );
const number = (value, digits = 2) =>
  Number.isFinite(value)
    ? value.toLocaleString(undefined, { maximumFractionDigits: digits })
    : "Unavailable";
const milliseconds = (value) =>
  value >= 1
    ? `${number(value)} s`
    : value < 0.001
      ? `${number(value * 1e6)} µs`
      : `${number(value * 1000)} ms`;
const repo = "https://github.com/SiliconBadgers/software";
let registry,
  build,
  study,
  component,
  config,
  result,
  sweep = [],
  evaluationTimer;
let capturedGraph,
  dependencies,
  evaluationVersion = 0,
  selectionVersion = 0,
  navigationVersion = 0,
  expandedCaptures = [];
const captureCache = new Map();
const colors = {
  Matrix: "#81a5ff",
  Vector: "#87daca",
  Recurrent: "#d8a0ed",
  DMA: "#f1be75",
  "RISC-V": "#d95664",
  "Shared L1": "#87daca",
  HBM: "#f1be75",
};

function notice(message, error = false) {
  $("#notice").textContent = message;
  $("#notice").className = message ? `notice ${error ? "error" : ""}` : "";
}

function download(filename, text, type = "application/json") {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function changeView() {
  const previous = document.body.dataset.view;
  const [view, section] = (location.hash.slice(1) || "model").split("/");
  const selected = [
    "model",
    "graph",
    "execution",
    "work",
    "previews",
    "sources",
  ].includes(view)
    ? view
    : "model";
  $$("[data-panel]").forEach(
    (panel) => (panel.hidden = panel.dataset.panel !== selected),
  );
  $$("nav [data-view]").forEach((link) => {
    link.classList.toggle("active", link.dataset.view === selected);
    if (link.dataset.view === selected)
      link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  document.body.dataset.view = selected;
  if (selected !== previous) window.scrollTo(0, 0);
  if (study && selected === "graph") renderGraph();
  if (study && selected === "execution") renderExecution();
  if (build && selected === "previews") refreshPreviews();
  if (selected === "work" && section === "recurrence")
    $("#saved-study-panel").scrollIntoView({ block: "start" });
}

async function selectStudy() {
  const selection = ++selectionVersion;
  ++evaluationVersion;
  clearTimeout(evaluationTimer);
  const selectedComponent = registry.components.find(
    (item) => item.id === $("#study").value,
  );
  const module = await import(`./${selectedComponent.module}`);
  const selectedStudy = await module.createStudy();
  if (selection !== selectionVersion) return;
  component = selectedComponent;
  study = selectedStudy;
  for (const method of ["defaults", "validate", "evaluate", "layerMap"]) {
    if (typeof study[method] !== "function")
      throw Error(`Study ${component.id} needs ${method}().`);
  }
  config = study.defaults();
  try {
    const saved = JSON.parse(
      localStorage.getItem(`sb-software-${component.id}-v1`),
    );
    if (saved) {
      const candidate = { ...config, ...saved };
      if (!study.validate(candidate).length) config = candidate;
    }
  } catch {
    /* Missing or invalid saved settings use this study's defaults. */
  }
  try {
    localStorage.setItem("sb-software-engine-v1", component.id);
  } catch {
    /* Engine selection remains usable when browser storage is unavailable. */
  }
  $("#study-credit").innerHTML =
    `${escape(component.contributors.join(", "))} · ${(component.pullRequests || [component.pullRequest]).map((n) => `<a href="${repo}/pull/${n}">PR #${n}</a>`).join(" / ")}`;
  $("#model-scope").textContent = study.scope;
  $('[data-preset="long"]').textContent =
    component.id === "qwen-python" ? "8K capture" : "8K / 20K";
  $("#sweep-axis").innerHTML = study.controls
    .filter((control) => control.type === "number")
    .map(
      (control) =>
        `<option value="${control.key}" ${["matrixCount", "matrix.count"].includes(control.key) ? "selected" : ""}>${escape(control.label)}</option>`,
    )
    .join("");
  $("#execution-layer").innerHTML =
    '<option value="all">All operations</option><option value="-1">Embedding</option>' +
    Array.from(
      { length: 24 },
      (_, layer) =>
        `<option value="${layer}">Layer ${layer} · ${(layer + 1) % 4 ? "DeltaNet" : "Full attention"}</option>`,
    ).join("") +
    '<option value="24">Output head</option>';
  result = null;
  clearResults();
  clearSweep();
  renderControls();
  await evaluate();
  changeView();
}

function renderControls() {
  const groups = [...new Set(study.controls.map((control) => control.group))];
  $("#controls").innerHTML = groups
    .map(
      (group) =>
        `<details class="control-group" open><summary>${escape(group)}</summary>${study.controls
          .filter((control) => control.group === group)
          .map((control) => {
            const id = `control-${control.key}`;
            if (control.type === "boolean")
              return `<label class="check"><input id="${id}" data-key="${control.key}" type="checkbox" ${config[control.key] ? "checked" : ""}>${escape(control.label)}</label>`;
            const input =
              control.type === "select"
                ? `<select id="${id}" data-key="${control.key}">${control.options.map(([value, label]) => `<option value="${value}" ${config[control.key] === value ? "selected" : ""}>${escape(label)}</option>`).join("")}</select>`
                : `<input id="${id}" data-key="${control.key}" type="number" min="${control.min}" step="${control.step}" value="${config[control.key]}">`;
            return `<label class="control" for="${id}"><span>${escape(control.label)}</span>${input}</label>`;
          })
          .join("")}</details>`,
    )
    .join("");
  $("#config-json").value = JSON.stringify(config, null, 2);
  $$("#controls [data-key]").forEach((input) =>
    input.addEventListener("input", () => {
      config[input.dataset.key] =
        input.type === "checkbox"
          ? input.checked
          : input.type === "number"
            ? Number(input.value)
            : input.value;
      $("#config-json").value = JSON.stringify(config, null, 2);
      clearTimeout(evaluationTimer);
      evaluationTimer = setTimeout(evaluate, 120);
    }),
  );
}

async function evaluate() {
  const version = ++evaluationVersion;
  const errors = study.validate(config);
  if (errors.length) {
    result = null;
    notice(errors.join(" "), true);
    clearResults();
    return;
  }
  const currentStudy = study,
    snapshot = structuredClone(config);
  const asynchronous = component.id === "qwen-python";
  if (asynchronous) {
    notice("Calculating with the Python engine…");
    $("#export-timeline").disabled = true;
  }
  let outcome;
  try {
    outcome = await currentStudy.evaluate(snapshot);
  } catch (error) {
    if (version === evaluationVersion) {
      result = null;
      clearResults();
      notice(error.message, true);
    }
    return;
  }
  if (
    version !== evaluationVersion ||
    currentStudy !== study ||
    JSON.stringify(snapshot) !== JSON.stringify(config)
  )
    return;
  result = outcome;
  notice(result.valid ? "" : result.errors.join(" "), !result.valid);
  try {
    localStorage.setItem(
      `sb-software-${component.id}-v1`,
      JSON.stringify(config),
    );
  } catch {
    /* The study still runs when storage is unavailable. */
  }
  renderMetrics();
  renderOverview();
  renderCosts();
  renderExecution();
  renderTensors();
  const previous = sweep[0]?.base;
  if (previous && JSON.stringify(previous) !== JSON.stringify(config)) {
    $("#comparison-table").dataset.stale = "true";
    $("#comparison-chart").setAttribute(
      "aria-label",
      "Saved comparison; current configuration has changed",
    );
    $("#comparison-note").textContent =
      "Saved comparison: settings changed after this run. Run again to compare the current configuration.";
  }
}

function clearResults() {
  for (const id of [
    "metrics",
    "operation-table",
    "timeline",
    "execution-summary",
    "execution-detail",
    "tensor-table",
    "tensor-detail",
    "pool-breakdown",
    "schedule-comparison",
    "active-workload",
  ])
    $(`#${id}`).replaceChildren();
  $("#export-timeline").disabled = true;
}

const metric = (label, value, detail) =>
  `<div class="metric"><span>${escape(label)}</span><strong>${escape(value)}</strong><small>${escape(detail)}</small></div>`;
function renderMetrics() {
  if (!result?.decode) return clearResults();
  $("#active-workload").textContent =
    component.id === "qwen-python"
      ? `${config._capture} · ${config["precision.weights"] === "captured" ? "Captured mixed Q4_K_M weights" : config["precision.weights"]} · Python equations`
      : `Qwen3.5-2B · ${config.precision.toUpperCase()} estimate · BF16 capture topology`;
  $("#capture-shortcut").hidden = component.id !== "qwen-python";
  $("#metrics").innerHTML =
    metric(
      "Prefill",
      milliseconds(result.prefill.seconds),
      `${number(result.promptTokens ?? config.prompt, 0)} tokens per sequence`,
    ) +
    metric(
      "Decode",
      milliseconds(result.decode.seconds),
      `${number(result.decode.aggregateTPS)} total tokens/s`,
    ) +
    metric(
      "External memory",
      `${number(Math.max(result.prefill.requiredHBM, result.decode.requiredHBM) / 2 ** 30)} GiB`,
      "Estimated allocated footprint",
    ) +
    metric(
      "Resource cost",
      number(result.resource.cost, 0),
      "Relative cost proxy",
    );
}

function renderCosts() {
  if (!result?.decode) return;
  const phase = result[$("#cost-phase").value];
  $("#operation-table").innerHTML =
    `<table><thead><tr><th>Operation</th><th>Count</th><th>Compute</th><th>Latency</th><th>External traffic</th><th>SRAM traffic</th></tr></thead><tbody>${phase.ops.map((op) => `<tr><td>${escape(op.op)}</td><td>${op.count}</td><td>${escape(op.pools.join(" + "))}</td><td>${milliseconds(op.seconds)}</td><td>${number(op.hbm / 2 ** 20)} MiB</td><td>${number(op.l1 / 2 ** 20)} MiB</td></tr>`).join("")}</tbody></table>`;
}

function clearSweep() {
  sweep = [];
  $("#comparison-chart").innerHTML =
    '<p class="empty">Choose a parameter and run a comparison.</p>';
  $("#comparison-table").replaceChildren();
  $("#export-sweep").disabled = true;
}

async function runSweep() {
  const values = $("#sweep-values")
    .value.split(",")
    .map((value) => Number(value.trim()));
  if (
    !values.length ||
    values.length > 32 ||
    values.some((value) => !Number.isFinite(value)) ||
    $("#sweep-values")
      .value.split(",")
      .some((value) => !value.trim())
  )
    return notice("Enter 1–32 comma-separated numbers.", true);
  if (study.validate(config).length)
    return notice(
      "Correct the configuration before running a comparison.",
      true,
    );
  const axis = $("#sweep-axis").value,
    base = JSON.parse(JSON.stringify(config)),
    sweepStudy = study;
  const next = [];
  $("#run-sweep").disabled = true;
  try {
    for (const value of values) {
      const candidate = { ...base, [axis]: value };
      const outcome = await sweepStudy.evaluate(candidate);
      if (sweepStudy !== study) return;
      next.push({ axis, value, base, config: candidate, result: outcome });
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
    sweep = next;
    renderSweep();
    $("#export-sweep").disabled = false;
    notice("");
  } finally {
    $("#run-sweep").disabled = false;
  }
}

function renderSweep() {
  if (!sweep.length) return;
  const phase = $("#sweep-phase").value;
  const max = Math.max(
    ...sweep
      .filter((item) => item.result.valid)
      .map((item) => item.result[phase].seconds),
    0.001,
  );
  $("#comparison-chart").innerHTML =
    `<p id="comparison-note" class="muted">${escape(sweep[0].axis)} · ${phase} · all other settings held fixed</p><div class="comparison-bars">${sweep.map((item, index) => `<button class="comparison-row" data-design="${index}" ${item.result.decode ? "" : "disabled"}><span>${item.value}</span><i style="width:${item.result.valid ? Math.max(0.5, (item.result[phase].seconds / max) * 100) : 0}%"></i><strong>${item.result.valid ? milliseconds(item.result[phase].seconds) : "Infeasible"}</strong></button>`).join("")}</div>`;
  $("#comparison-table").innerHTML =
    `<table><thead><tr><th>${escape(sweep[0].axis)}</th><th>Prefill</th><th>Decode</th><th>Total tokens/s</th><th>Cost proxy</th><th>Status</th></tr></thead><tbody>${sweep.map((item) => `<tr><td>${item.value}</td><td>${item.result.prefill ? milliseconds(item.result.prefill.seconds) : "Unavailable"}</td><td>${item.result.decode ? milliseconds(item.result.decode.seconds) : "Unavailable"}</td><td>${number(item.result.decode?.aggregateTPS)}</td><td>${number(item.result.resource?.cost, 0)}</td><td title="${escape(item.result.errors.join(" "))}">${item.result.valid ? "Within configured limits" : "Infeasible"}</td></tr>`).join("")}</tbody></table>`;
  $$("[data-design]").forEach((button) =>
    button.addEventListener("click", () => {
      config = JSON.parse(
        JSON.stringify(sweep[Number(button.dataset.design)].config),
      );
      renderControls();
      evaluate();
    }),
  );
}

async function renderGraph() {
  const capture = $("#capture").value,
    phase = $("#graph-phase").value,
    layer = $("#graph-layer").value;
  const original = $("#graph-style").value === "original";
  const expanded = capture.startsWith("raghav:");
  const id = expanded ? capture.slice(7) : capture;
  const folder = expanded ? `assets/captures/${id}` : `assets/graphs/${id}`;
  $("#capture-image").src = `${folder}/${phase}.layer${layer}.svg`;
  $("#capture-image").hidden = !original;
  $("#dependency-graph").hidden = original;
  $("#grouping-control").hidden = original || expanded;
  $("#raw-graph").href = `${folder}/${phase}.${expanded ? "json.gz" : "json"}`;
  $("#graph-scope").textContent = expanded
    ? "Raghav Iyer’s recorded mixed-format Q4_K_M graph. Flash attention and prompt length follow the selected capture. Graph inspection is independent of the active engine’s configuration."
    : "Zeb Taylor’s recorded BF16-weight CPU graph, with flash attention off. Hardware controls change estimates without changing this capture.";
  $("#graph-caption").textContent =
    `${id} · ${phase} · layer ${layer} · ${original ? "original rendered capture" : expanded ? "Raghav Iyer’s active dependencies" : "Adrian Luo’s dependency map"}`;
  const key = `${capture}/${phase}`;

  try {
    if (!captureCache.has(key)) {
      const response = await fetch(
        `${folder}/${phase}.${expanded ? "json.gz" : "json"}`,
      );
      if (!response.ok) throw Error("Capture data could not load.");
      const graph = expanded
        ? await new Response(
            response.body.pipeThrough(new DecompressionStream("gzip")),
          ).json()
        : await response.json();
      const dependencyResponse = await fetch(
        expanded
          ? `${folder}/${phase}.dependencies.json`
          : `assets/dependencies/${capture}-${phase}.json`,
      );
      if (!dependencyResponse.ok) throw Error("Dependency map could not load.");
      captureCache.set(key, [graph, await dependencyResponse.json()]);
    }
    if (key !== `${$("#capture").value}/${$("#graph-phase").value}`) return;
    [capturedGraph, dependencies] = captureCache.get(key);
    $("#dependency-detail").replaceChildren();
    const counts = drawDependencies(
      dependencies,
      layer,
      !expanded && $("#graph-grouping").value === "fusion",
      $("#dependency-graph"),
      $("#dependency-detail"),
    );
    $("#dependency-summary").textContent =
      `${counts.nodes} active operations · ${counts.groups} displayed nodes · ${counts.stateEdges} state-ordering edges. Select a node for shapes and dependencies.`;
    renderTensors();
  } catch (error) {
    $("#tensor-table").textContent = error.message;
  }
}

function renderTensors() {
  if (!study || !capturedGraph) return;
  const search = $("#tensor-search").value.toLowerCase();
  const tensors = new Map(
    capturedGraph.tensors.map((tensor) => [tensor.id, tensor]),
  );
  const layers = new Map(),
    layer = Number($("#graph-layer").value);
  let current = -1;
  for (const id of capturedGraph.scheduled_nodes) {
    const tensor = tensors.get(id),
      match = tensor.name.match(/-(\d+)\b|_l(\d+)\b/);
    if (match) current = Number(match[1] ?? match[2]);
    layers.set(id, current);
    if (tensor.name === "l_out-23") current = 24;
  }
  const nodes = capturedGraph.scheduled_nodes
    .map((id) => tensors.get(id))
    .filter(
      (node) =>
        layers.get(node.id) === layer &&
        `${node.name} ${node.op_desc}`.toLowerCase().includes(search),
    );
  $("#tensor-table").innerHTML =
    `<table><thead><tr><th>Tensor</th><th>Operation</th><th>Capture shape</th><th>Type</th></tr></thead><tbody>${nodes.map((node) => `<tr><td><button class="tensor-button" data-tensor="${node.id}">${escape(node.name)}</button></td><td>${escape(node.op_desc)}</td><td>${node.shape.join(" × ")}</td><td>${escape(node.dtype)}</td></tr>`).join("")}</tbody></table>`;
  $("#tensor-detail").replaceChildren();
  $$("[data-tensor]").forEach((button) =>
    button.addEventListener("click", () => {
      const node = tensors.get(Number(button.dataset.tensor));
      const inputs = node.sources.map((source) => tensors.get(source.tensor));
      $("#tensor-detail").innerHTML =
        `<div class="detail"><h3>${escape(node.name)}</h3><p>${escape(node.op_desc)} · ${escape(node.dtype)}</p><h4>Direct inputs</h4><ul>${inputs.map((input) => `<li>${escape(input.name)} · ${escape(input.op_desc)} · ${escape(input.dtype)}</li>`).join("") || "<li>No direct inputs</li>"}</ul><h4>Active dependencies</h4><ul>${
          (dependencies?.edges || [])
            .filter((edge) => edge.dst_name === node.name)
            .map(
              (edge) =>
                `<li>${escape(edge.src_name)} · ${escape(edge.kind)}${edge.state_buffer ? " · " + escape(edge.state_buffer) : ""}</li>`,
            )
            .join("") ||
          "<li>No active predecessors in the dependency map.</li>"
        }</ul></div>`;
    }),
  );
}

function timelineRows() {
  if (!result?.decode) return [];
  const phase = result[$("#execution-phase").value];
  const byId = new Map(phase.rows.map((row) => [row.id, row]));
  return phase.execution.operations.map((operation) => ({
    ...byId.get(operation.id),
    ...operation,
    start: operation.launchStart,
    duration: operation.end - operation.launchStart,
  }));
}

function renderExecution() {
  $("#export-timeline").disabled = !result?.decode;
  if (!result?.decode) return;
  const phase = result[$("#execution-phase").value],
    schedule = phase.execution;
  $("#execution-scope").textContent =
    schedule.mode === "python-pools"
      ? "Raghav Iyer’s compute-pool schedule reserves whole-operation intervals, including auxiliary compute work. SRAM and external memory impose aggregate service-time floors; their contention is not scheduled as intervals. Total estimated latency can therefore extend beyond the last compute bar."
      : schedule.mode === "dependency-resource"
        ? "Eric Wang’s dependency scheduler reserves compute, SRAM, external memory and launch intervals. State read/write ordering is preserved. Each compute pool works on one operation at a time. These are modeled reservations, not measured hardware utilization."
        : "Serial baseline: operations run in captured order, with optional compute/memory overlap inside each operation. Switch Operation scheduling in the setup panel to inspect shared-resource reservations.";
  $("#execution-summary").innerHTML =
    metric(
      "Total latency",
      milliseconds(phase.seconds),
      `${phase.activeNodes} operations`,
    ) +
    metric(
      "Dependency path",
      milliseconds(schedule.bounds.criticalPath),
      "Operation costs along the longest path",
    ) +
    metric(
      "SRAM service",
      milliseconds(phase.sums.l1),
      "Total modeled demand",
    ) +
    metric(
      "External memory service",
      milliseconds(phase.sums.hbm),
      "Total modeled demand",
    );
  const layers = study.layerMap(),
    selected = $("#execution-layer").value;
  const rows = timelineRows().filter(
    (row) => selected === "all" || layers.get(row.id) === Number(selected),
  );
  if (!rows.length || !rows.every((row) => Number.isFinite(row.end))) {
    $("#timeline").innerHTML =
      '<p class="empty">No finite timeline for this selection.</p>';
    return;
  }
  const selectedIds = new Set(rows.map((r) => r.id));
  const reservations = ["dependency-resource", "python-pools"].includes(
    schedule.mode,
  )
    ? Object.fromEntries(
        Object.entries(schedule.calendars)
          .map(([pool, intervals]) => [
            pool,
            intervals.filter((i) => selectedIds.has(i.id)),
          ])
          .filter(([, intervals]) => intervals.length),
      )
    : Object.fromEntries(
        [...new Set(rows.map((r) => r.pool))].map((pool) => [
          pool,
          rows.filter((r) => r.pool === pool),
        ]),
      );
  const pools = Object.keys(reservations),
    start = Math.min(...rows.map((r) => r.start)),
    end = Math.max(
      ...rows.map((r) => r.end),
      selected === "all" ? phase.seconds : 0,
    );
  const width = 1100,
    laneHeight = 40,
    labelWidth = 160;
  const scale = (value) =>
    labelWidth +
    ((value - start) / Math.max(1e-12, end - start)) *
      (width - labelWidth - 20);
  const byId = new Map(rows.map((r) => [r.id, r]));
  let svg = `<svg viewBox="0 0 ${width} ${pools.length * laneHeight + 65}" role="img" aria-label="${schedule.mode === "serial" ? "Serial operation durations" : "Modeled resource reservations"}">`;
  for (let tick = 0; tick <= 4; tick++) {
    const time = start + ((end - start) * tick) / 4,
      x = scale(time);
    svg += `<line x1="${x}" x2="${x}" y1="10" y2="${pools.length * laneHeight + 16}" class="grid-line"/><text x="${x}" y="${pools.length * laneHeight + 43}" text-anchor="${tick === 4 ? "end" : "start"}">${milliseconds(time)}</text>`;
  }
  pools.forEach((pool, lane) => {
    svg += `<text x="8" y="${lane * laneHeight + 33}">${escape(pool)}</text>`;
    for (const interval of reservations[pool]) {
      const row = byId.get(interval.id),
        x = scale(interval.start),
        y = lane * laneHeight + 16;
      svg += `<rect x="${x}" y="${y}" width="${Math.max(0.4, scale(interval.end) - x)}" height="24" rx="2" fill="${colors[pool] || "#889bb5"}" fill-opacity=".85"><title>${escape(row.name)} · ${escape(row.op)} · ${milliseconds(interval.end - interval.start)}</title></rect>`;
    }
  });
  $("#timeline").innerHTML =
    svg +
    '</svg><p class="muted">' +
    (schedule.mode === "python-pools"
      ? "Whole-operation compute-pool reservations, including auxiliary work. Memory service is represented by aggregate bounds."
      : schedule.mode === "serial"
        ? "Bars include launch and memory time on the assigned compute pool."
        : "Bars show the intervals reserved by the analytical scheduler. Empty space indicates an unreserved interval.") +
    "</p>";
  $("#execution-detail").innerHTML =
    `<div class="table-wrap"><table><thead><tr><th>Operation</th><th>Pool</th><th>Launch</th><th>End</th><th>Compute</th><th>SRAM</th><th>External memory</th></tr></thead><tbody>${rows.map((row) => `<tr><td>${escape(row.name)}<small>${escape(row.op)}</small></td><td>${escape(row.pool)}</td><td>${milliseconds(row.start)}</td><td>${milliseconds(row.end)}</td><td>${milliseconds(row.compute)}</td><td>${milliseconds(row.l1)}</td><td>${milliseconds(row.hbm)}</td></tr>`).join("")}</tbody></table></div>`;
}

function renderOverview() {
  if (!result?.decode) return;
  const phase = result[$("#mix-phase").value],
    totals = new Map();
  for (const row of phase.rows)
    totals.set(row.pool, (totals.get(row.pool) || 0) + row.seconds);
  const total = phase.serialSeconds;
  $("#pool-breakdown").innerHTML = [...totals]
    .sort((a, b) => b[1] - a[1])
    .map(
      ([pool, seconds]) =>
        `<div class="pool-row"><span>${escape(pool)}</span><div class="track"><i style="background:${colors[pool] || "#889bb5"};width:${(seconds / total) * 100}%"></i></div><strong>${number((seconds / total) * 100, 1)}%</strong></div>`,
    )
    .join("");
  const parallel = phase.parallel?.seconds;
  $("#schedule-comparison").innerHTML =
    `<div class="schedule-values"><div><span>Serial</span><strong>${milliseconds(total)}</strong></div><span class="arrow">→</span><div><span>${component.id === "qwen-python" ? "Compute pools" : "Shared resources"}</span><strong>${parallel ? milliseconds(parallel) : "Unavailable"}</strong></div></div><div class="schedule-note">${parallel ? `<b>${number(total / parallel, 2)}×</b> modeled ratio · same configuration and operation costs.` : "Enable within-operation overlap to compare scheduling."}</div>`;
}

function renderWork(statuses = build.work) {
  $("#work-list").innerHTML = registry.work
    .map((item, index) => {
      const status = statuses?.find(
        (status) => status.number === item.pullRequest,
      );
      const label = status?.merged_at
        ? "Merged"
        : status?.state === "open"
          ? "Under review"
          : status?.state === "closed"
            ? "Closed"
            : "See PR";
      const registered =
        item.view &&
        (!item.component ||
          registry.components.some(
            (component) => component.id === item.component,
          ));
      return `<article class="panel study-card"><div class="card-top"><span class="tag ${label === "Merged" ? "good" : ""}">${label}</span><span class="muted">${escape(item.contributor)}</span></div><h2>${escape(item.title)}</h2><p>${escape(item.description)}</p><div class="card-actions">${registered ? `<button data-work-item="${index}">${escape(item.actionLabel || "Open study")} →</button>` : ""}${item.source ? `<a href="${repo}/tree/${build.sha}/${item.source}">Browse source ↗</a>` : ""}<a href="${repo}/pull/${item.pullRequest}">PR #${item.pullRequest} ↗</a></div></article>`;
    })
    .join("");
}

async function refreshPreviews() {
  if ($("#refresh-previews").disabled) return;
  $("#refresh-previews").disabled = true;
  const root = new URL(
    build.preview && location.pathname.includes("/previews/") ? "../../" : "./",
    location.href,
  );
  try {
    const response = await fetch(new URL("editions.json", root), {
      cache: "no-store",
      signal: AbortSignal.timeout(8000),
    });
    if (!response.ok) throw Error("Published editions could not be loaded.");
    const catalog = await response.json();
    if (catalog.schemaVersion !== 1 || !Array.isArray(catalog.editions))
      throw Error("The edition list is unavailable.");
    const editions = catalog.editions.filter(
      (edition) =>
        typeof edition.ref === "string" &&
        /^[a-f0-9]{40}$/.test(edition.sha) &&
        /^(\.\/|previews\/[a-z0-9-]+\/)$/.test(edition.path),
    );
    const previews = editions.filter((edition) => edition.preview);
    $("#preview-status").textContent =
      `${previews.length} published branch ${previews.length === 1 ? "preview" : "previews"}. Preview URLs remain available after branches are merged or deleted.`;
    $("#preview-list").innerHTML = editions
      .map((edition) => {
        const current = edition.sha === build.sha && edition.ref === build.ref;
        return `<article class="panel edition-card"><div class="card-top"><span class="tag ${edition.preview ? "" : "good"}">${edition.preview ? "Branch preview" : "Main"}</span>${current ? '<span class="muted">You are here</span>' : ""}</div><h2>${escape(edition.preview ? edition.ref : "Shared workspace")}</h2><p class="muted">${edition.preview ? "Published branch snapshot" : "Latest published main revision"} · <a href="${repo}/commit/${edition.sha}">${edition.sha.slice(0, 8)}</a></p><div class="card-actions"><a class="button-link" href="${new URL(edition.path, root).href}">${edition.preview ? "Open preview" : "Open main"} →</a><a href="${repo}/tree/${edition.sha}">Browse source ↗</a></div></article>`;
      })
      .join("");
  } catch (error) {
    $("#preview-status").textContent =
      `${error.message} Use Refresh to try again. Preview URLs are also listed in each publication run’s summary.`;
  } finally {
    $("#refresh-previews").disabled = false;
  }
}

async function refreshWork() {
  renderWork();
  $("#work-status").textContent =
    `PR status recorded at build ${build.sha.slice(0, 8)}. Checking GitHub…`;
  try {
    const response = await fetch(
      "https://api.github.com/repos/SiliconBadgers/software/pulls?state=all&per_page=100",
      { signal: AbortSignal.timeout(8000) },
    );
    if (!response.ok) throw Error("GitHub status unavailable");
    renderWork(await response.json());
    $("#work-status").textContent =
      "PR status checked against GitHub just now.";
  } catch {
    $("#work-status").textContent =
      `Showing PR status recorded when this site was built (${build.sha.slice(0, 8)}). Live status is unavailable.`;
  }
}

function renderSources() {
  $("#source-list").innerHTML = registry.sources
    .map(
      (source) =>
        `<article class="panel"><h2>${escape(source.title)}</h2><p>${escape(source.description)}</p><a href="${repo}/tree/${build.sha}/${source.path}">Browse source at this revision ↗</a></article>`,
    )
    .join("");
  $("#build-info").innerHTML =
    `<h2>This edition</h2><p>${escape(build.ref)} · <a href="${repo}/commit/${build.sha}">${build.sha.slice(0, 12)}</a></p><p>UI: SiliconBadgers Model Lab. Dependency maps: Adrian Luo, <a href="${repo}/pull/10">PR #10</a>. Resource scheduling: Eric Wang, <a href="${repo}/pull/9">PR #9</a>. Expanded profiling and Python model: Raghav Iyer, <a href="${repo}/pull/7">PR #7</a>. Original model and captured graphs: Zeb Taylor's import, <a href="${repo}/pull/6">PR #6</a>. The original model and experiment files are preserved.</p><p><a href="source-manifest.json">Source file hashes</a> · <a href="${repo}/blob/${build.sha}/web/README.md">Build and component instructions</a> · <a href="${repo}/blob/${build.sha}/experiments/llama-cpp/2026-09-24-qwen35-2b/explorer/MODEL.md">Model equations</a></p>`;
  $("#build-label").textContent =
    `${build.preview ? "Preview" : "Main"} · ${build.sha.slice(0, 8)}`;
  $("#preview-banner").hidden = !build.preview;
  $("#preview-banner").textContent =
    `Branch preview: ${build.ref}. This edition is separate from the main workspace.`;
}

async function start() {
  $("#configuration-panel").open = !matchMedia("(max-width: 760px)").matches;
  [registry, build] = await Promise.all(
    ["registry.json", "build-info.json"].map(async (path) => {
      const response = await fetch(path);
      if (!response.ok) throw Error(`Cannot load ${path}.`);
      return response.json();
    }),
  );
  const catalog = await fetch("assets/captures/index.json");
  if (catalog.ok) {
    expandedCaptures = await catalog.json();
    $("#capture").innerHTML =
      '<optgroup label="Zeb · BF16 / flash attention off"><option value="pp128">128 tokens</option><option value="pp512" selected>512 tokens</option></optgroup><optgroup label="Raghav · mixed Q4_K_M">' +
      expandedCaptures
        .map(
          (c) => `<option value="raghav:${c.id}">${escape(c.label)}</option>`,
        )
        .join("") +
      "</optgroup>";
  }
  $("#study").innerHTML = registry.components
    .map(
      (component) =>
        `<option value="${component.id}">${escape(component.title)}</option>`,
    )
    .join("");
  try {
    const savedEngine = localStorage.getItem("sb-software-engine-v1");
    if (registry.components.some((item) => item.id === savedEngine))
      $("#study").value = savedEngine;
  } catch {
    /* A fresh or storage-restricted browser opens the default engine. */
  }
  $("#study").addEventListener("change", () =>
    selectStudy().catch((error) => notice(error.message, true)),
  );
  addEventListener("hashchange", changeView);
  $("#capture-shortcut").onclick = () => {
    $("#capture").value = `raghav:${config._capture}`;
    location.hash = "graph";
    renderGraph();
  };
  $("#reset").onclick = () => {
    config = study.defaults();
    renderControls();
    evaluate();
  };
  $("#apply-json").onclick = () => {
    try {
      const next = JSON.parse($("#config-json").value);
      const errors = study.validate(next);
      if (errors.length) throw Error(errors.join(" "));
      config = next;
      renderControls();
      evaluate();
    } catch (error) {
      notice(error.message, true);
    }
  };
  $("#run-sweep").onclick = () =>
    runSweep().catch((error) => notice(error.message, true));
  $("#sweep-phase").onchange = renderSweep;
  $("#cost-phase").onchange = renderCosts;
  $("#mix-phase").onchange = renderOverview;
  for (const id of [
    "capture",
    "graph-phase",
    "graph-layer",
    "graph-style",
    "graph-grouping",
  ])
    $(`#${id}`).onchange = renderGraph;
  $("#tensor-search").oninput = renderTensors;
  for (const id of ["execution-phase", "execution-layer"])
    $(`#${id}`).onchange = renderExecution;
  $("#export").onclick = () =>
    download(
      "siliconbadgers-design.json",
      JSON.stringify(
        {
          schemaVersion: 1,
          component: component.id,
          sourceRevision: build.sha,
          config,
        },
        null,
        2,
      ),
    );
  $("#import").onclick = () => $("#import-file").click();
  $("#import-file").onchange = async (event) => {
    try {
      const file = event.target.files[0];
      if (!file) return;
      const data = JSON.parse(await file.text());
      if (data.schemaVersion !== 1 || data.component !== component.id)
        throw Error("Select the matching study before importing this design.");
      const next = { ...study.defaults(), ...data.config },
        errors = study.validate(next);
      if (errors.length) throw Error(errors.join(" "));
      config = next;
      renderControls();
      evaluate();
    } catch (error) {
      notice(error.message, true);
    } finally {
      event.target.value = "";
    }
  };
  $("#export-sweep").onclick = () =>
    download(
      "siliconbadgers-comparison.csv",
      "parameter,value,prefill_ms,decode_ms,total_tokens_s,cost_proxy,valid,source_revision\n" +
        sweep
          .map((item) =>
            [
              item.axis,
              item.value,
              item.result.prefill?.seconds * 1000,
              item.result.decode?.seconds * 1000,
              item.result.decode?.aggregateTPS,
              item.result.resource?.cost,
              item.result.valid,
              build.sha,
            ].join(","),
          )
          .join("\n"),
      "text/csv",
    );
  $("#export-timeline").onclick = () =>
    download(
      "siliconbadgers-timeline.json",
      JSON.stringify(
        {
          component: component.id,
          sourceRevision: build.sha,
          phase: $("#execution-phase").value,
          model: result[$("#execution-phase").value].execution.mode,
          config,
          latencySeconds: result[$("#execution-phase").value].seconds,
          rows: timelineRows(),
          native: result.native,
          reservations: result[$("#execution-phase").value].execution.calendars,
        },
        null,
        2,
      ),
    );
  $$("[data-preset]").forEach(
    (button) =>
      (button.onclick = () => {
        const preset = button.dataset.preset;
        const overrides =
          study.presets?.[preset] ??
          (preset === "vector"
            ? { vectorCount: 8 }
            : preset === "shared"
              ? { recurrentCount: 0, vectorCount: 8 }
              : preset === "long"
                ? { prompt: 8192, context: 20000 }
                : {});
        config = { ...study.defaults(), ...overrides };
        renderControls();
        evaluate();
        $$("[data-preset]").forEach((other) =>
          other.classList.toggle("active", other === button),
        );
      }),
  );
  $("#work-list").addEventListener("click", async (event) => {
    const button = event.target.closest("[data-work-item]");
    if (!button) return;
    const item = registry.work[Number(button.dataset.workItem)];
    const navigation = ++navigationVersion;
    button.disabled = true;
    try {
      if (
        item.component &&
        (item.component !== component?.id ||
          item.component !== $("#study").value)
      ) {
        $("#study").value = item.component;
        await selectStudy();
      }
      if (navigation !== navigationVersion) return;
      if (item.capture) {
        $("#capture").value = item.capture;
        $("#graph-grouping").value = "fusion";
        $("#graph-style").value = "dependencies";
      }
      if (location.hash === `#${item.view}`) changeView();
      else location.hash = item.view;
    } catch (error) {
      notice(error.message, true);
    } finally {
      button.disabled = false;
    }
  });
  $("#refresh-previews").onclick = refreshPreviews;
  renderSources();
  refreshWork();
  await selectStudy();
  initializeStudies(async (design) => {
    if (component.id !== "qwen-baseline") {
      $("#study").value = "qwen-baseline";
      await selectStudy();
    }
    config = structuredClone(design);
    renderControls();
    await evaluate();
    location.hash = "model";
  }).catch((error) => {
    $("#saved-study-chart").textContent = error.message;
  });
  globalThis.SB_WORKSPACE_READY = true;
}
start().catch((error) =>
  notice(`The workspace could not load: ${error.message}`, true),
);
