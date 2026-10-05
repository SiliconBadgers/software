/* Check the browser's original Python engine against native evaluation. */
const assert = require("node:assert/strict");
const { execFileSync } = require("node:child_process");
const fs = require("node:fs");
const { chromium } = require("playwright");
const nativeCode = `
import copy,json,sys
from pathlib import Path
sys.path.insert(0,'engine')
from sbengine.config import DEFAULTS,set_path
from sbengine.graph import load_workload
from sbengine.model import evaluate
request=json.loads(sys.argv[1]);cfg=copy.deepcopy(DEFAULTS)
for key,value in request['config'].items():
    if not key.startswith('_'):set_path(cfg,key,value)
w=load_workload(Path('results/2026-09-27_1839_windows-2/graphs')/request['config']['_capture'])
print(json.dumps(evaluate(w,cfg,detail=True)))
`;
const close = (a, b, label) =>
  assert(
    Math.abs(a - b) <= Math.max(1e-10, Math.abs(b) * 1e-10),
    `${label}: ${a} versus ${b}`,
  );
(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({
      viewport: { width: 1440, height: 1000 },
    }),
    errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  try {
    await page.goto(process.env.SB_WORKSPACE_URL || "http://127.0.0.1:8767/");
    await page.waitForFunction(() => window.SB_WORKSPACE_READY);
    await page.locator("#study").selectOption("qwen-python");
    await page.waitForFunction(
      () =>
        document.querySelector("#control-clock_mhz") &&
        !document.querySelector("#notice").textContent.includes("Calculating"),
      {},
      { timeout: 120000 },
    );
    assert.match(await page.locator("#metrics").innerText(), /512 tokens/);
    for (const changes of [
      {},
      { "memory.weight_path": "direct", "vector.count": 8 },
      { _capture: "pp8192-fa-on" },
      {
        _capture: "pp512-fa-off",
        "schedule.mode": "serial",
        "memory.fusion": "none",
      },
    ]) {
      const config = JSON.parse(
        await page.locator("#config-json").inputValue(),
      );
      Object.assign(config, changes);
      if ((await page.locator(".advanced").getAttribute("open")) === null)
        await page.locator(".advanced summary").click();
      await page.locator("#config-json").fill(JSON.stringify(config));
      await page.locator("#apply-json").click();
      await page.waitForFunction(
        () =>
          !document
            .querySelector("#notice")
            .textContent.includes("Calculating"),
        {},
        { timeout: 120000 },
      );
      assert.equal(await page.locator("#notice").innerText(), "");
      await page.locator("nav [data-view=execution]").click();
      await page.locator("#execution-layer").selectOption("all");
      const [download] = await Promise.all([
        page.waitForEvent("download"),
        page.locator("#export-timeline").click(),
      ]);
      const actual = JSON.parse(fs.readFileSync(await download.path(), "utf8"));
      const expected = JSON.parse(
        execFileSync(
          process.env.SB_NATIVE_PYTHON || "python3",
          ["-c", nativeCode, JSON.stringify({ config })],
          { encoding: "utf8" },
        ),
      );
      const phase = expected.phases.decode;
      assert.equal(
        actual.model,
        config["schedule.mode"] === "serial" ? "serial" : "python-pools",
      );
      assert.equal(actual.rows.length, phase.nodes);
      close(
        actual.rows.reduce((s, r) => s + r.seconds, 0),
        phase.bounds.serial,
        "Operation cost sum",
      );
      if (actual.model === "python-pools") {
        for (let i = 0; i < phase.rows.length; i++) {
          close(
            actual.rows[i].start,
            phase.pool_schedule.operations[i].start,
            "Pool start",
          );
          close(
            actual.rows[i].end,
            phase.pool_schedule.operations[i].end,
            "Pool end",
          );
        }
        assert.equal(
          Object.keys(actual.reservations).includes("HBM"),
          false,
          "Memory modeled as aggregate floor",
        );
      }
      const metricText = await page.locator("#metrics").innerText();
      // Export carries full precision so parity does not depend on rounded text.
      const nativeBrowser = actual.native;
      assert(nativeBrowser, "Native report included for reproduction");
      for (const name of ["prefill", "decode"]) {
        close(
          nativeBrowser.phases[name].seconds,
          expected.phases[name].seconds,
          `${name} latency`,
        );
        close(
          nativeBrowser.phases[name].bytes.l1,
          expected.phases[name].bytes.l1,
          `${name} L1 bytes`,
        );
        close(
          nativeBrowser.phases[name].footprint.hbm_required,
          expected.phases[name].footprint.hbm_required,
          `${name} footprint`,
        );
      }
      await page.locator("nav [data-view=model]").click();
    }
    await page.locator("#sweep-axis").selectOption("matrix.count");
    await page.locator("#sweep-values").fill("1,2,4");
    await page.locator("#run-sweep").click();
    await page.waitForFunction(
      () =>
        document.querySelectorAll("#comparison-table tbody tr").length === 3,
      {},
      { timeout: 120000 },
    );
    await page.locator("nav [data-view=graph]").click();
    await page.locator("#capture").selectOption("raghav:pp512-fa-on");
    await page.locator("#graph-layer").selectOption("3");
    await page.waitForFunction(
      () =>
        document
          .querySelector("#graph-caption")
          .textContent.includes("Raghav") &&
        document
          .querySelector("#tensor-table")
          .textContent.includes("FLASH_ATTN_EXT"),
    );
    assert((await page.locator(".graph-node").count()) > 10);
    assert(await page.locator("#grouping-control").isHidden());
    await page.locator("#graph-style").selectOption("original");
    await page.waitForFunction(
      () => document.querySelector("#capture-image").naturalWidth > 0,
    );
    await page.locator("nav [data-view=model]").click();
    await page.locator("#control-clock_mhz").fill("0");
    await page.waitForFunction(
      () => document.querySelector("#metrics").children.length === 0,
    );
    assert(await page.locator("#export-timeline").isDisabled());
    await page.locator("nav [data-view=work]").click();
    await page.getByRole("button", { name: "Open JS model" }).click();
    await page.waitForFunction(
      () =>
        document.querySelector("#control-frequency") &&
        document.querySelector("#metrics").children.length > 0,
    );
    assert.equal(await page.locator("#study").inputValue(), "qwen-baseline");
    await page.locator("nav [data-view=work]").click();
    await page.getByRole("button", { name: "Open Python model" }).click();
    await page.waitForFunction(
      () =>
        document.querySelector("#control-clock_mhz") &&
        !document.querySelector("#notice").textContent.includes("Calculating"),
    );
    await page.locator("#control-clock_mhz").fill("225");
    await page.waitForFunction(
      () =>
        JSON.parse(localStorage.getItem("sb-software-qwen-python-v1"))[
          "clock_mhz"
        ] === 225,
    );
    await page.reload();
    await page.waitForFunction(
      () => window.SB_WORKSPACE_READY,
      {},
      { timeout: 120000 },
    );
    assert.equal(
      await page.locator("#study").inputValue(),
      "qwen-python",
      "Reload keeps the selected engine",
    );
    assert.equal(
      await page.locator("#control-clock_mhz").inputValue(),
      "225",
      "Reload keeps its valid configuration",
    );
    await page.locator("nav [data-view=graph]").click();
    await page.locator("#capture").selectOption("raghav:pp512-fa-on");
    await page.locator("nav [data-view=work]").click();
    await page.getByRole("button", { name: "Explore dependencies" }).click();
    await page.waitForFunction(
      () =>
        document.querySelector("#capture").value === "pp512" &&
        document.querySelector("#graph-caption").textContent.includes("Adrian"),
    );
    assert.equal(await page.locator("#study").inputValue(), "qwen-baseline");
    assert(
      await page.locator("#grouping-control").isVisible(),
      "Adrian's card opens his grouped BF16 dependency map",
    );
    await page.locator("nav [data-view=work]").click();
    await page
      .getByRole("button", { name: "Compare recurrence designs" })
      .click();
    await page.waitForFunction(() => {
      const target = document.querySelector("#saved-study-panel");
      return (
        location.hash === "#work/recurrence" &&
        document.body.dataset.view === "work" &&
        target?.getBoundingClientRect().top < 50
      );
    });
    assert.equal(await page.locator("body").getAttribute("data-view"), "work");
    assert(
      (await page.locator("#saved-study-panel").boundingBox()).y < 50,
      "Eric's card goes directly to the saved experiment",
    );
    await page.setViewportSize({ width: 390, height: 844 });
    assert.equal(
      await page.evaluate(() => document.documentElement.scrollWidth),
      390,
    );
    assert.deepEqual(errors, []);
    console.log(
      "PASS Python/native parity, captured workloads, sweeps, graphs, export, study navigation and remembered engine",
    );
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
