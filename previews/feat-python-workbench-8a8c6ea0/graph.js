// Display Adrian Luo's committed dependency tables without changing the captures.
const esc = (value) =>
  String(value).replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const palette = {
  matrix: "#81a5ff",
  weight_matrix: "#81a5ff",
  act_matrix: "#81a5ff",
  conv: "#87daca",
  vector: "#87daca",
  recurrent: "#d8a0ed",
  memory: "#f1be75",
};
export function drawDependencies(data, layer, grouped, target, detail) {
  const nodes = data.nodes.filter((n) => n.layer === String(layer));
  const byName = new Map(nodes.map((n) => [n.name, n]));
  const groups = new Map();
  for (const node of nodes) {
    const id = grouped ? node.fusion_group : node.name;
    if (!groups.has(id)) groups.set(id, { id, nodes: [], rank: 0 });
    groups.get(id).nodes.push(node);
  }
  const idOf = (name) =>
    !byName.has(name)
      ? undefined
      : grouped
        ? byName.get(name).fusion_group
        : name;
  const seen = new Set(),
    edges = [];
  for (const edge of data.edges) {
    const src = idOf(edge.src_name),
      dst = idOf(edge.dst_name);
    if (!src || !dst || src === dst) continue;
    const key = `${src}/${dst}/${edge.kind}`;
    if (!seen.has(key)) {
      edges.push({ ...edge, src, dst });
      seen.add(key);
    }
  }
  // Rank by data dependencies. State ordering is drawn separately, including backwards edges.
  for (const group of groups.values()) {
    for (const edge of edges.filter(
      (e) => e.dst === group.id && e.kind === "data",
    )) {
      const predecessor = groups.get(edge.src);
      if (
        Number(predecessor.nodes[0].sched_pos) <
        Number(group.nodes[0].sched_pos)
      )
        group.rank = Math.max(group.rank, predecessor.rank + 1);
    }
  }
  const ranks = new Map();
  for (const group of groups.values()) {
    if (!ranks.has(group.rank)) ranks.set(group.rank, []);
    ranks.get(group.rank).push(group);
  }
  const maxColumns = Math.max(
    1,
    ...[...ranks.values()].map((row) => row.length),
  );
  const width = Math.max(680, maxColumns * 206 + 100),
    height = (Math.max(0, ...ranks.keys()) + 1) * 94 + 45;
  for (const [rank, row] of ranks)
    row.forEach((g, index) => {
      g.x = (width - row.length * 206) / 2 + index * 206 + 11;
      g.y = rank * 94 + 20;
    });
  let svg = `<svg viewBox="0 0 ${width} ${height}" width="${width}" role="img" aria-label="Layer ${layer} dependencies${grouped ? " grouped by fusion candidate" : ""}"><defs><marker id="data-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#667a99"/></marker><marker id="state-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#c28db0"/></marker></defs>`;
  for (const edge of edges) {
    const a = groups.get(edge.src),
      b = groups.get(edge.dst),
      state = edge.kind !== "data";
    const x1 = a.x + 91,
      y1 = a.y + 58,
      x2 = b.x + 91,
      y2 = b.y;
    const path =
      y2 > y1
        ? `M${x1},${y1} C${x1},${(y1 + y2) / 2} ${x2},${(y1 + y2) / 2} ${x2},${y2}`
        : `M${a.x + 182},${a.y + 29} C${width - 12},${a.y + 29} ${width - 12},${b.y + 29} ${b.x + 182},${b.y + 29}`;
    svg += `<path d="${path}" class="graph-edge ${state ? "state" : ""}" marker-end="url(#${state ? "state" : "data"}-arrow)"><title>${esc(edge.src_name)} → ${esc(edge.dst_name)} · ${esc(edge.kind)}${edge.state_buffer ? " · " + esc(edge.state_buffer) : ""}</title></path>`;
  }
  const indexed = [...groups.values()];
  indexed.forEach((g, index) => {
    const anchor =
      g.nodes.find(
        (n) => n.category.includes("matrix") || n.category === "recurrent",
      ) || g.nodes[0];
    const color = palette[anchor.category] || palette.vector;
    const title = anchor.name.replace(/-\d+$/, "");
    const ops = [...new Set(g.nodes.map((n) => n.op))].join(" · ");
    svg += `<g class="graph-node" tabindex="0" role="button" aria-label="${esc(g.nodes.map((n) => n.name).join(", "))}" data-node-group="${index}"><rect x="${g.x}" y="${g.y}" width="182" height="58" rx="6" fill="#172333" stroke="${color}" stroke-opacity=".65"/><rect x="${g.x}" y="${g.y + 10}" width="3" height="38" rx="1" fill="${color}"/><text x="${g.x + 12}" y="${g.y + 23}" font-size="12">${esc(title.length > 23 ? title.slice(0, 21) + "…" : title)}</text><text class="node-sub" x="${g.x + 12}" y="${g.y + 43}">${esc(ops.length > 25 ? ops.slice(0, 23) + "…" : ops)}${g.nodes.length > 1 ? " ×" + g.nodes.length : ""}</text><title>${esc(g.nodes.map((n) => `${n.name}: ${n.op} [${n.shape_ggml}]`).join("\n"))}</title></g>`;
  });
  target.innerHTML = svg + "</svg>";
  target.querySelectorAll("[data-node-group]").forEach((element) => {
    const show = () => {
      const g = indexed[Number(element.dataset.nodeGroup)],
        names = new Set(g.nodes.map((n) => n.name));
      const incoming = data.edges.filter(
        (e) => names.has(e.dst_name) && !names.has(e.src_name),
      );
      detail.innerHTML = `<div class="detail"><h3>${esc(g.nodes[0].fusion_group)} · ${g.nodes.length} operation${g.nodes.length > 1 ? "s" : ""}</h3><p>${grouped ? "Structural fusion candidate. Grouping does not apply a kernel or reduce the timing estimate." : "Captured operation."}</p><div class="table-wrap"><table><thead><tr><th>Name</th><th>Operation</th><th>Shape</th><th>Type</th></tr></thead><tbody>${g.nodes.map((n) => `<tr><td>${esc(n.name)}</td><td>${esc(n.op)}</td><td>${esc(n.shape_ggml)}</td><td>${esc(n.dtype)}</td></tr>`).join("")}</tbody></table></div><h4>Incoming dependencies</h4><ul>${incoming.map((e) => `<li>${esc(e.src_name)} → ${esc(e.dst_name)} · ${esc(e.kind)}${e.state_buffer ? " · " + esc(e.state_buffer) : ""}</li>`).join("") || "<li>Inputs from outside this active layer.</li>"}</ul></div>`;
    };
    element.onclick = show;
    element.onkeydown = (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        show();
      }
    };
  });
  return {
    nodes: nodes.length,
    groups: groups.size,
    stateEdges: edges.filter((e) => e.kind !== "data").length,
  };
}
