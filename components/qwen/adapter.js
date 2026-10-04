// Adapter for the preserved model imported through software PR #6.
export function createStudy() {
  const engine = globalThis.QwenModel;
  const graph = globalThis.QWEN_GRAPH;
  const number = (key, label, group, min, step = 1) => ({key, label, group, type: 'number', min, step});
  const controls = [
    number('frequency', 'Clock · MHz', 'Compute', 1, 25),
    number('matrixCount', 'Matrix units', 'Compute', 0),
    number('rows', 'Array rows', 'Compute', 1, 8),
    number('cols', 'Array columns', 'Compute', 1, 8),
    number('vectorCount', 'Vector units', 'Compute', 0),
    number('vectorLanes', 'Vector lanes', 'Compute', 1, 8),
    number('recurrentCount', 'Dedicated recurrence units', 'Compute', 0),
    {key: 'precision', label: 'Matrix format', group: 'Compute', type: 'select', options: [['w8a16', 'W8 / A16'], ['w4a8', 'W4 / A8']]},
    number('l1MiB', 'SRAM · MiB', 'Memory', 0.01, 0.25),
    number('l1Banks', 'SRAM banks', 'Memory', 1, 8),
    number('hbmGBs', 'External bandwidth · GB/s', 'Memory', 0.01, 10),
    number('hbmGiB', 'External capacity · GiB', 'Memory', 0.01),
    number('prompt', 'Prompt · tokens', 'Workload', 2, 128),
    number('context', 'Decode context · tokens', 'Workload', 1, 128),
    number('batch', 'Independent sequences', 'Workload', 1),
    {key: 'overlap', label: 'Overlap compute and memory within an operation', group: 'Workload', type: 'boolean'}
  ];
  return {
    controls,
    defaults: () => engine.defaults(),
    validate: config => engine.validation(config),
    evaluate: config => engine.evaluate(graph, config, true),
    graph,
    scope: 'Qwen3.5-2B text trunk: 18 DeltaNet layers and 6 full-attention layers. Shapes derive from the 128/512-token captures; other lengths and batches are extrapolations. Precision, resource costs and timing are assumptions, with no accelerator measurements or accuracy validation.',
    layerMap: () => {
      const layers = new Map();
      let current = -1;
      for (const id of graph.order) {
        const match = graph.tensors[id].name.match(/-(\d+)\b|_l(\d+)\b/);
        if (match) current = Number(match[1] ?? match[2]);
        layers.set(id, current);
        if (graph.tensors[id].name === 'l_out-23') current = 24;
      }
      return layers;
    }
  };
}
