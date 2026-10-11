const flatten = (object, prefix = "", out = {}) => {
  for (const [key, value] of Object.entries(object)) {
    const path = prefix ? `${prefix}.${key}` : key;
    if (value && typeof value === "object" && !Array.isArray(value))
      flatten(value, path, out);
    else out[path] = value;
  }
  return out;
};
let worker,
  sequence = 0;
const pending = new Map();
function request(config) {
  if (!worker) {
    worker = new Worker(new URL("./worker.js", import.meta.url));
    worker.onmessage = ({ data }) => {
      const task = pending.get(data.id);
      if (!task) return;
      clearTimeout(task.timer);
      pending.delete(data.id);
      data.error ? task.reject(Error(data.error)) : task.resolve(data.result);
    };
    worker.onerror = (event) => {
      for (const task of pending.values()) {
        clearTimeout(task.timer);
        task.reject(Error(event.message || "Python runtime could not start."));
      }
      pending.clear();
      worker.terminate();
      worker = null;
    };
  }
  return new Promise((resolve, reject) => {
    const id = ++sequence;
    const timer = setTimeout(() => {
      pending.delete(id);
      reject(
        Error(
          "Python evaluation timed out. Try a smaller capture or reload the engine.",
        ),
      );
    }, 120000);
    pending.set(id, { resolve, reject, timer });
    worker.postMessage({ id, capture: config._capture, config });
  });
}
export async function createStudy() {
  const [defaults, captures] = await Promise.all(
    ["assets/python/defaults.json", "assets/captures/index.json"].map(
      async (path) => {
        const response = await fetch(path);
        if (!response.ok) throw Error(`Cannot load ${path}.`);
        return response.json();
      },
    ),
  );
  const base = { ...flatten(defaults), _capture: "pp512-fa-on" };
  let layers = new Map();
  const numeric = (key, label, group, min = 0, step = 1) => ({
    key,
    label,
    group,
    type: "number",
    min,
    step,
  });
  const choice = (key, label, group, options) => ({
    key,
    label,
    group,
    type: "select",
    options,
  });
  const boolean = (key, label, group) => ({
    key,
    label,
    group,
    type: "boolean",
  });
  const controls = [
    choice(
      "_capture",
      "Captured workload",
      "Workload",
      captures.map((c) => [c.id, c.label]),
    ),
    numeric("batch", "Independent sequences", "Workload", 1),
    numeric("clock_mhz", "Clock · MHz", "Compute", 1, 25),
    numeric("matrix.count", "Matrix units", "Compute"),
    numeric("matrix.rows", "Array rows", "Compute", 1, 8),
    numeric("matrix.cols", "Array columns", "Compute", 1, 8),
    numeric("vector.count", "Vector units", "Compute"),
    numeric("vector.lanes", "Vector lanes", "Compute", 1, 8),
    numeric("recurrent.count", "Dedicated recurrence units", "Compute"),
    numeric(
      "recurrent.scratch_kib",
      "Local state buffer · KiB",
      "Compute",
      0,
      16,
    ),
    choice("recurrent.lowering", "Recurrence method", "Compute", [
      ["auto", "Lowest estimated cost"],
      ["serial", "Token by token"],
      ["chunked", "Process chunks"],
      ["matrix", "Token by token on matrix arrays"],
    ]),
    numeric(
      "recurrent.matrix_penalty",
      "Matrix recurrence penalty · x",
      "Compute",
      0.25,
      1,
    ),
    choice("precision.weights", "Weight format", "Compute", [
      ["captured", "Captured mixed formats"],
      ["uniform_int4", "Uniform INT4"],
      ["uniform_int8", "Uniform INT8"],
    ]),
    numeric("l1.mib", "SRAM · MiB", "Memory", 0.01, 0.25),
    numeric("l1.banks", "SRAM banks", "Memory", 1, 8),
    numeric("hbm.gbs", "External bandwidth · GB/s", "Memory", 0.01, 10),
    numeric("hbm.gib", "External capacity · GiB", "Memory", 0.01),
    choice("memory.weight_path", "Weight path", "Memory", [
      ["via_l1", "Through shared SRAM"],
      ["direct", "Direct to array"],
    ]),
    choice("memory.residency", "Activation residency", "Memory", [
      ["belady", "Track tensor residency"],
      ["per_op", "Per-operation estimate"],
    ]),
    choice("memory.fusion", "Fusion model", "Memory", [
      ["chains", "Single-consumer chains"],
      ["groups", "Anchored dependency-map groups"],
      ["none", "No fusion"],
    ]),
    choice("schedule.mode", "Operation scheduling", "Execution", [
      ["pools", "Compute-pool schedule"],
      ["serial", "Serial baseline"],
    ]),
    boolean(
      "schedule.within_op_overlap",
      "Overlap compute and memory within an operation",
      "Execution",
    ),
  ];
  return {
    controls,
    defaults: () => structuredClone(base),
    validate: (config) => {
      const errors = [];
      for (const key of Object.keys(base))
        if (!(key in config)) errors.push(`Missing ${key}.`);
      for (const key of Object.keys(config))
        if (!(key in base)) errors.push(`Unknown parameter ${key}.`);
      for (const control of controls) {
        const value = config[control.key];
        if (
          control.type === "number" &&
          (!Number.isFinite(value) || value < control.min)
        )
          errors.push(`${control.label} must be at least ${control.min}.`);
        if (
          control.type === "select" &&
          !control.options.some(([option]) => option === value)
        )
          errors.push(`Choose a supported ${control.label.toLowerCase()}.`);
      }
      return errors;
    },
    evaluate: async (config) => {
      const raw = await request(config);
      if (!raw.phases) return raw;
      const normalized = {
        valid: raw.valid,
        errors: raw.errors,
        resource: raw.resources,
        native: raw,
        promptTokens: raw.prompt_tokens,
      };
      layers = new Map();
      for (const name of ["prefill", "decode"]) {
        const phase = raw.phases[name];
        let currentLayer = -1;
        const rows = phase.rows.map((row) => {
          const match = row.name.match(/-(\d+)\b|_l(\d+)\b|blk\.(\d+)\./);
          if (match) currentLayer = Number(match[1] ?? match[2] ?? match[3]);
          layers.set(row.id, currentLayer);
          if (row.name === "l_out-23") currentLayer = 24;
          return {
            ...row,
            seconds: row.duration_s,
            compute: row.compute_s,
            l1: row.l1_s,
            hbm: row.hbm_s,
          };
        });
        let cursor = 0;
        const serial = config["schedule.mode"] === "serial";
        const operations = rows.map((row, index) => {
          const interval = phase.pool_schedule.operations[index];
          const launchStart = serial ? cursor : interval.start;
          cursor += row.seconds;
          return {
            ...row,
            launchStart,
            serviceStart: launchStart,
            start: launchStart,
            end: serial ? cursor : interval.end,
          };
        });
        const calendars = serial
          ? {}
          : Object.fromEntries(
              Object.entries(phase.pool_schedule.reservations).map(
                ([pool, intervals]) => [
                  pool,
                  intervals
                    .map((interval) => ({
                      ...interval,
                      id: rows[interval.node].id,
                    }))
                    .sort((a, b) => a.start - b.start),
                ],
              ),
            );
        normalized[name] = {
          ...phase,
          rows,
          serialSeconds: phase.bounds.serial,
          parallel: { seconds: phase.bounds.list },
          aggregateTPS: phase.tokens_per_s,
          requiredHBM: phase.footprint.hbm_required,
          activeNodes: phase.nodes,
          sums: {
            compute: rows.reduce((s, r) => s + r.compute, 0),
            l1: phase.l1_service_s,
            hbm: phase.hbm_service_s,
          },
          ops: Object.entries(phase.by_op)
            .map(([op, info]) => ({
              op,
              count: info.count,
              pools: [
                ...new Set(rows.filter((r) => r.op === op).map((r) => r.pool)),
              ],
              seconds: info.serial_s,
              hbm: info.hbm_bytes,
              l1: info.l1_bytes,
            }))
            .sort((a, b) => b.seconds - a.seconds),
          execution: {
            mode: serial ? "serial" : "python-pools",
            seconds: phase.seconds,
            operations,
            calendars,
            bounds: { criticalPath: phase.bounds.critical_path },
          },
        };
      }
      return normalized;
    },
    layerMap: () => layers,
    presets: {
      baseline: {},
      vector: { "vector.count": 8 },
      shared: { "recurrent.count": 0, "vector.count": 8 },
      long: { _capture: "pp8192-fa-on" },
    },
    scope:
      "Qwen3.5-2B captured workloads, mixed Q4_K_M weights, with flash attention on/off and prompt-specific graphs. The original Python engine models tiling, activation residency, fusion, local state storage and scheduling. Hardware rates and DSP costs are assumptions. Its compute-pool schedule uses aggregate memory-service bounds, rather than explicit SRAM/HBM interval reservations. CPU traces and numerical validation are separate from these estimates.",
  };
}
