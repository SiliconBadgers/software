// The original sbengine runs in a Web Worker. No server or model weights are used.
const base = new URL("../../", self.location.href);
const runtime = "https://cdn.jsdelivr.net/pyodide/v0.29.5/full/";
let python, ready;
async function fetchFile(relative) {
  const response = await fetch(new URL(relative, base));
  if (!response.ok)
    throw Error(`Cannot load ${relative} (${response.status}).`);
  return new Uint8Array(await response.arrayBuffer());
}
async function initialize() {
  importScripts(runtime + "pyodide.js");
  python = await loadPyodide({ indexURL: runtime });
  const response = await fetch(new URL("assets/python/manifest.json", base));
  if (!response.ok) throw Error("Python asset manifest could not load.");
  const manifest = await response.json();
  for (const file of manifest.files) {
    const path = "/" + file.virtual;
    python.FS.mkdirTree(path.slice(0, path.lastIndexOf("/")));
    python.FS.writeFile(path, await fetchFile(file.published));
  }
  await python.runPythonAsync(
    'import sys\nsys.path.insert(0, "/engine")\nfrom bridge import run\n',
  );
}
async function execute(message) {
  ready ||= initialize();
  try {
    await ready;
  } catch (error) {
    ready = undefined;
    throw error;
  }
  const folder = `/captures/${message.capture}`;
  python.FS.mkdirTree(folder);
  for (const phase of ["prefill", "decode"]) {
    const path = `${folder}/${phase}.json.gz`;
    if (!python.FS.analyzePath(path).exists)
      python.FS.writeFile(
        path,
        await fetchFile(`assets/captures/${message.capture}/${phase}.json.gz`),
      );
  }
  python.globals.set("sb_request", JSON.stringify(message));
  return JSON.parse(await python.runPythonAsync("run(sb_request)"));
}
let queue = Promise.resolve();
self.onmessage = ({ data }) => {
  queue = queue.then(async () => {
    try {
      self.postMessage({ id: data.id, result: await execute(data) });
    } catch (error) {
      self.postMessage({ id: data.id, error: error.message });
    }
  });
};
