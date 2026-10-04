// Explore Eric Wang's saved recurrence sensitivity study with its original assumptions.
const esc = (value) =>
  String(value).replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
export async function initializeStudies(loadDesign) {
  const response = await fetch("assets/studies/recurrence.json");
  if (!response.ok) throw Error("Saved recurrence study could not load.");
  const data = await response.json(),
    dimensions = data.dimensions;
  const choices = [
    [
      "saved-workload",
      "Workload",
      dimensions.workloads.map((w) => w.name),
      "target-batch1",
    ],
    ["saved-precision", "Format", dimensions.precisions, "w8a16"],
    [
      "saved-memory",
      "Memory profile",
      dimensions.memoryProfiles.map((m) => m.name),
      "middle",
    ],
    ["saved-matrix", "Matrix units", dimensions.matrixCounts, 4],
    ["saved-phase", "Phase", ["decode", "prefill"], "decode"],
  ];
  document.querySelector("#saved-study-controls").innerHTML = choices
    .map(
      ([id, label, values, selected]) =>
        `<label>${label}<select id="${id}">${values.map((value) => `<option value="${value}" ${String(value) === String(selected) ? "selected" : ""}>${esc(value)}</option>`).join("")}</select></label>`,
    )
    .join("");
  function render() {
    const get = (id) => document.querySelector("#" + id).value;
    const phase = get("saved-phase");
    const rows = data.rows.filter(
      (row) =>
        row.workload === get("saved-workload") &&
        row.precision === get("saved-precision") &&
        row.memoryProfile === get("saved-memory") &&
        row.matrixCount === Number(get("saved-matrix")),
    );
    const label = (row) =>
      row.mapping === "vector"
        ? "Vector fallback"
        : row.mapping === "matrix"
          ? `Matrix fallback · ${row.recurrentMatrixPenalty}× penalty`
          : `Dedicated · ${row.recurrentCount} unit${row.recurrentCount === 1 ? "" : "s"}`;
    const max = Math.max(...rows.map((r) => r[phase].latencyMs));
    document.querySelector("#saved-study-chart").innerHTML = rows
      .map(
        (row) =>
          `<div class="saved-bar"><span>${esc(label(row))}</span><div class="track"><i style="width:${(row[phase].latencyMs / max) * 100}%;background:${row.mapping === "matrix" ? "var(--blue)" : row.mapping === "dedicated" ? "var(--pink)" : "var(--accent)"}"></i></div><strong>${row[phase].latencyMs.toFixed(2)} ms</strong></div>`,
      )
      .join("");
    document.querySelector("#saved-study-table").innerHTML =
      `<table><thead><tr><th>Recurrence mapping</th><th>Prefill · ms</th><th>Decode · ms</th><th>DSP estimate</th><th>Cost proxy</th><th></th></tr></thead><tbody>${rows.map((row, index) => `<tr><td>${esc(label(row))}</td><td>${row.prefill.latencyMs.toFixed(2)}</td><td>${row.decode.latencyMs.toFixed(2)}</td><td>${row.resource.dsp.toLocaleString()}</td><td>${row.resource.cost.toLocaleString()}</td><td><button data-saved-design="${index}">Load design</button></td></tr>`).join("")}</tbody></table>`;
    document.querySelectorAll("[data-saved-design]").forEach(
      (button) =>
        (button.onclick = () => {
          const row = rows[Number(button.dataset.savedDesign)],
            workload = dimensions.workloads.find(
              (w) => w.name === row.workload,
            ),
            memory = dimensions.memoryProfiles.find(
              (m) => m.name === row.memoryProfile,
            );
          loadDesign({
            ...data.baseline,
            ...workload.config,
            ...memory.config,
            precision: row.precision,
            matrixCount: row.matrixCount,
            recurrentCount: row.recurrentCount,
            recurrentMatrix: row.mapping === "matrix",
            recurrentMatrixPenalty:
              row.recurrentMatrixPenalty ??
              data.baseline.recurrentMatrixPenalty,
            executionMode: "dependency-resource",
          });
        }),
    );
  }
  document
    .querySelectorAll("#saved-study-controls select")
    .forEach((input) => (input.onchange = render));
  render();
}
