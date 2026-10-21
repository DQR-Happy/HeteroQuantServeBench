import { test, expect } from '@playwright/test';

test.use({ actionTimeout: 15000 });

test('real model DMAT exposes measured heatmap, filters and exact raw artifacts', async ({
  page,
}) => {
  const run = process.env.HQSB_DMAT_RUN;
  test.skip(!run, 'Requires a real completed model memory capture');
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  const response = await page.request.get(`/api/console/v1/runs/${run}`);
  const record = await response.json();
  expect(record.state).toBe('completed');
  const receipt = record.metrics.diagnostic;
  expect(receipt.replay_outputs_match_source).toBe(true);
  expect(receipt.replay_environment_matches_source).toBe(true);
  expect(receipt.panel.counts.events).toBeGreaterThan(0);
  expect(receipt.panel.counts.dropped_events).toBe(0);
  await page.goto(`/diagnostics?source=${record.config.source_run_id}&run=${run}`);
  const heatmap = page.getByRole('img', { name: '真实访存密度热力图', exact: true }).first();
  await expect(heatmap).toBeVisible();
  await expect(heatmap.locator('canvas').first()).toBeVisible();
  await expect(
    page.getByRole('heading', { name: '真实访存密度图 · DMAT', exact: true }),
  ).toBeVisible();
  await page.getByRole('combobox', { name: 'DMAT 访问类型', exact: true }).press('ArrowDown');
  await page.getByText('读取', { exact: true }).last().click();
  await expect(page.getByRole('link', { name: '下载全部访存 SVG', exact: true })).toBeVisible();
  const raw = await page.request.get(
    `/api/console/v1/diagnostics/jobs/${run}/artifacts/${receipt.panel.memory_launches[0].trace}`,
    { headers: { Range: 'bytes=0-7' } },
  );
  expect(raw.ok()).toBe(true);
  expect(['HQDMAT1\n', 'HQDMAT2\n']).toContain((await raw.body()).subarray(0, 8).toString());
  const svg = await page.request.get(
    `/api/console/v1/diagnostics/jobs/${run}/artifacts/${receipt.panel.memory_launches[0].svg}`,
  );
  expect(svg.ok()).toBe(true);
  expect(await svg.text()).toMatch(/GPU globaltimer|thread_entry_clock64/);
  if (receipt.panel.memory_launches[0].probe.event_bytes === 48) {
    expect(receipt.panel.memory_launches[0].plot.unknown_events).toBe(0);
    for (const role of ['Q', 'K', 'V', 'O']) {
      expect(
        receipt.panel.memory_launches[0].plot.objects.some(
          (o: { role: string }) => o.role === role,
        ),
      ).toBe(true);
    }
    const rebin = page.waitForResponse(
      (r) => r.url().includes(`/memory/0?clock=GPU_globaltimer`) && r.ok(),
    );
    await page.getByRole('combobox', { name: 'DMAT 时间口径', exact: true }).press('ArrowDown');
    await page.getByText('GPU 全局时间 · ns（论文图 15）', { exact: true }).last().click();
    const response = await rebin;
    const fullView = await response.json();
    expect(fullView.plot.time_unit).toBe('ns');
    const main = page.getByRole('region', { name: '真实访存 DMAT', exact: true });
    await expect(main).toContainText('GPU 全局计时器');
    const chart = main.getByRole('img', { name: '真实访存密度热力图', exact: true });
    await chart.scrollIntoViewIfNeeded();
    await expect(chart).toHaveAttribute('data-rendered', 'true');
    const bounds = (await chart.boundingBox())!;
    const beforeScroll = await page.evaluate(() => window.scrollY);
    const windowRequests: string[] = [];
    page.on('request', (request) => {
      if (request.url().includes('/memory/0?') && request.url().includes('&start='))
        windowRequests.push(request.url());
    });
    await page.mouse.move(bounds.x + bounds.width * 0.6, bounds.y + bounds.height * 0.45);
    await page.mouse.wheel(0, 240);
    await expect.poll(() => page.evaluate(() => window.scrollY)).toBeGreaterThan(beforeScroll);
    expect(windowRequests).toHaveLength(0);
    const focusButton = main.getByRole('button', { name: '聚焦查看', exact: true });
    await focusButton.click();
    const dialog = page.getByRole('dialog', { name: 'DMAT 聚焦查看', exact: true });
    await expect(dialog).toBeVisible();
    const area = dialog.getByRole('group', { name: 'DMAT 图表操作区', exact: true });
    await expect(area).toBeFocused();
    const focusedChart = dialog.getByRole('img', { name: '真实访存密度热力图', exact: true });
    await expect(focusedChart).toHaveAttribute('data-rendered', 'true');
    expect((await focusedChart.boundingBox())!.height).toBeGreaterThan(400);
    const refine = dialog.getByRole('button', { name: '按此时间范围重新分箱', exact: true });
    await expect(refine).toBeDisabled();
    await dialog.getByRole('button', { name: '地址放大', exact: true }).click();
    const rowStart = Number(await focusedChart.getAttribute('data-address-start'));
    expect(rowStart).toBeGreaterThan(0);
    const rowEnd = Number(await focusedChart.getAttribute('data-address-end'));
    const pageScroll = await page.evaluate(() => window.scrollY);
    const focusedBounds = (await focusedChart.boundingBox())!;
    await page.mouse.move(
      focusedBounds.x + focusedBounds.width * 0.6,
      focusedBounds.y + focusedBounds.height * 0.5,
    );
    await page.mouse.wheel(0, 100);
    await expect
      .poll(async () => Number(await focusedChart.getAttribute('data-address-start')))
      .toBeGreaterThan(rowStart);
    expect(await page.evaluate(() => window.scrollY)).toBe(pageScroll);
    expect(
      Number(await focusedChart.getAttribute('data-address-end')) -
        Number(await focusedChart.getAttribute('data-address-start')),
    ).toBe(rowEnd - rowStart);
    // Pointer dragging and the keyboard offer independent, non-wheel panning.
    const beforeDrag = Number(await focusedChart.getAttribute('data-address-start'));
    await page.mouse.move(
      focusedBounds.x + focusedBounds.width * 0.65,
      focusedBounds.y + focusedBounds.height * 0.6,
    );
    await page.mouse.down();
    await page.mouse.move(
      focusedBounds.x + focusedBounds.width * 0.65,
      focusedBounds.y + focusedBounds.height * 0.4,
      { steps: 6 },
    );
    await page.mouse.up();
    await expect
      .poll(async () => Number(await focusedChart.getAttribute('data-address-start')))
      .toBeGreaterThan(beforeDrag);
    await area.focus();
    await area.press('ArrowUp');
    await expect
      .poll(async () => Number(await focusedChart.getAttribute('data-address-start')))
      .toBeLessThan(beforeDrag);
    await area.press('Home');
    await expect(focusedChart).toHaveAttribute('data-address-start', '0');
    await dialog.getByRole('combobox', { name: 'DMAT 滚轮操作', exact: true }).press('ArrowDown');
    await page.getByText('缩放时间', { exact: true }).last().click();
    await page.mouse.move(
      focusedBounds.x + focusedBounds.width * 0.6,
      focusedBounds.y + focusedBounds.height * 0.5,
    );
    await page.mouse.wheel(0, -100);
    await expect
      .poll(async () => Number(await focusedChart.getAttribute('data-time-start')))
      .toBeGreaterThan(0);
    await expect(focusedChart).toHaveAttribute('data-address-start', '0');
    await dialog.getByRole('button', { name: '适应当前数据', exact: true }).click();
    await dialog.getByRole('button', { name: '时间放大', exact: true }).click();
    // Move the horizontal slider itself; displayed bins follow its new range.
    const beforeSlider = Number(await focusedChart.getAttribute('data-time-start'));
    await page.mouse.move(
      focusedBounds.x + focusedBounds.width * 0.55,
      focusedBounds.y + focusedBounds.height - 15,
    );
    await page.mouse.down();
    await page.mouse.move(
      focusedBounds.x + focusedBounds.width * 0.65,
      focusedBounds.y + focusedBounds.height - 15,
      { steps: 5 },
    );
    await page.mouse.up();
    await expect
      .poll(async () => Number(await focusedChart.getAttribute('data-time-start')))
      .toBeGreaterThan(beforeSlider);
    await expect(refine).toBeEnabled();
    expect(windowRequests).toHaveLength(0);
    await dialog.getByRole('button', { name: '地址放大', exact: true }).click();
    await page.screenshot({ path: '../../reports/console/dmat-focus/focused-desktop.png' });
    const zoomed = page.waitForResponse(
      (r) =>
        r.url().includes('/memory/0?clock=GPU_globaltimer') &&
        r.url().includes('&start=') &&
        r.ok(),
    );
    await refine.click();
    const detail = await (await zoomed).json();
    expect(detail.derived_view.raw_integrity).toBe('verified_against_sealed_manifest');
    expect(detail.plot.time_end).toBeLessThan(fullView.plot.time_end);
    expect(detail.plot.time_bin).toBeLessThanOrEqual(fullView.plot.time_bin);
    await expect(focusedChart).toHaveAttribute('data-time-start', '0');
    await expect(dialog.getByRole('status', { name: 'DMAT 当前视窗', exact: true })).toContainText(
      detail.plot.time_start.toLocaleString(),
    );
    const restored = page.waitForResponse(
      (r) =>
        r.url().includes('/memory/0?clock=GPU_globaltimer') &&
        !r.url().includes('&start=') &&
        r.ok(),
    );
    await dialog.getByRole('button', { name: '完整时间范围', exact: true }).click();
    await restored;
    await expect(focusedChart).toHaveAttribute('data-time-start', '0');
    await page.setViewportSize({ width: 1100, height: 800 });
    await expect(dialog.getByRole('button', { name: '返回页面', exact: true })).toBeInViewport();
    await page.screenshot({ path: '../../reports/console/dmat-focus/focused-compact.png' });
    await area.focus();
    await area.press('Escape');
    await expect(dialog).toBeHidden();
    await expect(focusButton).toBeFocused();
    await page.setViewportSize({ width: 1512, height: 982 });
    const baseline = process.env.HQSB_DMAT_BASELINE;
    if (baseline) {
      await page.getByRole('combobox', { name: '诊断对照基线', exact: true }).fill(baseline);
      await page
        .locator('.ant-select-dropdown:visible .ant-select-item-option')
        .filter({ hasText: baseline })
        .click();
      await page.getByRole('button', { name: '比较机制变化', exact: true }).click();
      await expect(
        page.getByRole('heading', { name: '访存结构对照 · 统一时间和密度尺度', exact: true }),
      ).toBeVisible();
      const pair = page.getByRole('region', { name: 'DMAT 机制对照', exact: true });
      await expect(pair.locator('[data-rendered="true"]')).toHaveCount(2);
      await pair.screenshot({ path: '../../reports/console/dmat-v2/model-comparison.png' });
      const candidateFocus = pair.getByRole('button', {
        name: '聚焦查看 · 启动 1 · 候选',
        exact: true,
      });
      await candidateFocus.click();
      const candidateDialog = page.getByRole('dialog', {
        name: 'DMAT 聚焦查看 · 启动 1 · 候选',
        exact: true,
      });
      await expect(candidateDialog).toContainText('对照图沿用共同分箱与颜色尺度');
      await candidateDialog.getByRole('button', { name: '地址放大', exact: true }).click();
      await expect(
        candidateDialog.getByRole('button', { name: '按此时间范围重新分箱', exact: true }),
      ).toHaveCount(0);
      await candidateDialog.getByRole('button', { name: '返回页面', exact: true }).click();
      await expect(candidateFocus).toBeFocused();
    }
  }
  await expect(page.getByLabel('访问令牌', { exact: true })).toHaveCount(0);
  await heatmap.scrollIntoViewIfNeeded();
  await page.screenshot({
    path: '../../reports/console/dmat-v2/browser-desktop.png',
    fullPage: true,
  });
  await page.setViewportSize({ width: 1100, height: 800 });
  await expect(heatmap).toBeVisible();
  await page.screenshot({
    path: '../../reports/console/dmat-v2/browser-compact.png',
    fullPage: true,
  });
  await page.getByRole('combobox', { name: '诊断工具', exact: true }).press('ArrowDown');
  await page.getByText('ERETrace · warp / SM 观测', { exact: true }).click();
  await page.getByRole('combobox', { name: 'ERETrace 观测模式', exact: true }).press('ArrowDown');
  await page.getByText('真实访存 / DMAT', { exact: true }).last().click();
  await expect(
    page.getByRole('spinbutton', { name: 'DMAT 每线程容量', exact: true }),
  ).toBeVisible();
  await expect(page.getByRole('button', { name: '开始独立诊断', exact: true })).toBeEnabled();
  expect(errors).toEqual([]);
});
