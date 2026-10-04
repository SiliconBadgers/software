/* Functional browser checks for the deployed static application. */
const assert = require("node:assert/strict");
const { chromium } = require("playwright");

(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({
    viewport: { width: 1440, height: 1000 },
  });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  try {
    await page.goto(process.env.SB_WORKSPACE_URL || "http://127.0.0.1:8767/");
    await page.waitForFunction(() => window.SB_WORKSPACE_READY);
    const before = await page.locator("#metrics").innerText();
    await page.locator("#control-hbmGBs").fill("10");
    await page.waitForFunction(
      (previous) => document.querySelector("#metrics").innerText !== previous,
      before,
    );
    assert.notEqual(
      await page.locator("#metrics").innerText(),
      before,
      "Hardware controls change results",
    );
    await page.locator("#reset").click();
    await page.locator("#run-sweep").click();
    await page.waitForFunction(
      () =>
        document.querySelectorAll("#comparison-table tbody tr").length === 4,
    );
    assert.equal(
      await page.locator("#comparison-chart [data-design]").count(),
      4,
    );
    await page.locator("#sweep-values").fill("1,broken");
    await page.locator("#run-sweep").click();
    assert.match(await page.locator("#notice").innerText(), /comma-separated/);
    await page.locator("nav [data-view=graph]").click();
    await page.waitForFunction(
      () => document.querySelectorAll("#tensor-table tbody tr").length > 0,
    );
    await page.waitForFunction(
      () => document.querySelector("#capture-image").naturalWidth > 0,
    );
    assert(
      (await page.locator(".graph-node").count()) > 0,
      "Committed dependency maps rendered",
    );
    await page.locator(".graph-node").first().click();
    assert.match(
      await page.locator("#dependency-detail").innerText(),
      /Incoming dependencies/,
    );
    const grouped = await page.locator(".graph-node").count();
    await page.locator("#graph-grouping").selectOption("operations");
    await page.waitForFunction(
      (count) => document.querySelectorAll(".graph-node").length > count,
      grouped,
    );
    await page.locator("#graph-layer").selectOption("3");
    await page.locator("#tensor-search").fill("soft");
    assert.match(await page.locator("#tensor-table").innerText(), /SOFT_MAX/);
    await page.locator("[data-tensor]").first().click();
    assert.match(
      await page.locator("#tensor-detail").innerText(),
      /Direct inputs/,
    );
    await page.locator("nav [data-view=execution]").click();
    await page.locator("#execution-layer").selectOption("3");
    assert((await page.locator("#execution-detail tbody tr").count()) > 20);
    const [download] = await Promise.all([
      page.waitForEvent("download"),
      page.locator("#export-timeline").click(),
    ]);
    const fs = require("node:fs");
    const timeline = JSON.parse(fs.readFileSync(await download.path(), "utf8"));
    assert.equal(timeline.model, "serial");
    assert(
      timeline.rows.every(
        (row, index) =>
          index === 0 ||
          Math.abs(timeline.rows[index - 1].end - row.start) < 1e-10,
      ),
    );
    assert(
      Math.abs(
        timeline.rows.at(-1).end -
          timeline.rows.reduce((sum, row) => sum + row.seconds, 0),
      ) < 1e-10,
    );
    await page
      .locator("#control-executionMode")
      .selectOption("dependency-resource");
    await page.waitForFunction(() =>
      document
        .querySelector("#execution-scope")
        .textContent.includes("Eric Wang"),
    );
    const [scheduledDownload] = await Promise.all([
      page.waitForEvent("download"),
      page.locator("#export-timeline").click(),
    ]);
    const scheduled = JSON.parse(
      fs.readFileSync(await scheduledDownload.path(), "utf8"),
    );
    assert.equal(scheduled.model, "dependency-resource");
    assert(
      scheduled.reservations.HBM.length > 0 &&
        scheduled.reservations["Shared L1"].length > 0,
    );
    for (const intervals of Object.values(scheduled.reservations)) {
      assert(
        intervals.every(
          (row, index) =>
            index === 0 || intervals[index - 1].end <= row.start + 1e-12,
        ),
        "No resource overlap",
      );
    }
    await page.locator("nav [data-view=work]").click();
    await page.locator("aside").waitFor({ state: "hidden" });
    assert.equal(await page.locator("#work-list article").count(), 5);
    assert(
      await page.locator("aside").isHidden(),
      "Study browsing uses the content width",
    );
    await page.waitForFunction(
      () => document.querySelectorAll("[data-saved-design]").length === 12,
    );
    await page.locator('[data-saved-design="0"]').click();
    await page.waitForFunction(
      () =>
        document.querySelector("#control-executionMode").value ===
        "dependency-resource",
    );
    assert.equal(
      await page.locator("#control-hbmGBs").inputValue(),
      "300",
      "Saved design loads its study memory profile",
    );
    await page.waitForFunction(() => document.body.dataset.view === "model");
    assert.equal(
      await page.evaluate(() => window.scrollY),
      0,
      "Opening a loaded design starts at its results",
    );
    await page.locator("nav [data-view=sources]").click();
    assert.match(await page.locator("#build-info").innerText(), /PR #6/);
    await page.locator("nav [data-view=previews]").click();
    await page.locator("#preview-list article").first().waitFor();
    assert(await page.locator("aside").isHidden());
    const editionLinks = await page
      .locator("#preview-list .button-link")
      .evaluateAll((links) => links.map((link) => link.href));
    for (const link of editionLinks) {
      const response = await page.request.get(
        new URL("build-info.json", link).href,
      );
      assert.equal(
        response.status(),
        200,
        "Listed edition links reach a built workspace",
      );
    }
    await page.route("**/editions.json", (route) =>
      route.fulfill({ status: 503, body: "Unavailable" }),
    );
    await page.locator("#refresh-previews").click();
    await page.waitForFunction(() =>
      document
        .querySelector("#preview-status")
        .textContent.includes("Use Refresh"),
    );
    assert(
      (await page.locator("#preview-list article").count()) > 0,
      "Existing edition links survive a refresh failure",
    );
    await page.unroute("**/editions.json");
    await page.locator("#refresh-previews").click();
    await page.waitForFunction(
      () =>
        !document
          .querySelector("#preview-status")
          .textContent.includes("Use Refresh"),
    );
    await page.locator("nav [data-view=model]").click();
    await page.locator("#control-frequency").fill("0");
    await page.waitForFunction(
      () => document.querySelector("#metrics").children.length === 0,
    );
    assert(await page.locator("#export-timeline").isDisabled());
    await page.locator("#reset").click();
    await page.setViewportSize({ width: 390, height: 844 });
    await page.reload();
    await page.waitForFunction(() => window.SB_WORKSPACE_READY);
    assert.equal(
      await page.evaluate(() => document.documentElement.scrollWidth),
      390,
      "No mobile page overflow",
    );
    assert.equal(
      await page.locator("#configuration-panel").getAttribute("open"),
      null,
    );
    for (const view of ["work", "previews", "sources"]) {
      await page.locator(`nav [data-view=${view}]`).click();
      await page.waitForFunction(
        (view) => document.body.dataset.view === view,
        view,
      );
      assert(await page.locator("aside").isHidden());
      assert.equal(
        await page.evaluate(() => document.documentElement.scrollWidth),
        390,
      );
    }
    assert.deepEqual(errors, [], "No client exceptions");
    console.log(
      "PASS workspace navigation, controls, comparison, graphs, timeline export, invalid input and mobile layout",
    );
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
