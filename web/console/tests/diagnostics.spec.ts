import { test, expect } from '@playwright/test';
test.setTimeout(60000);

test('diagnostic page keeps prerequisites, real timeline and native artifacts visible', async ({
  page,
}) => {
  const run = process.env.HQSB_NSYS_RUN;
  test.skip(!run, 'Requires a real completed Nsight capture on the execution node');
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  const response = await page.request.get(`/api/console/v1/runs/${run}`);
  const record = await response.json();
  expect(record.kind).toBe('diagnostic');
  expect(record.metrics.diagnostic.panel.counts.kernel).toBeGreaterThan(0);
  await page.goto(`/diagnostics?run=${run}`);
  await expect(page.getByRole('heading', { name: '执行诊断', exact: true })).toBeVisible();
  await expect(page.getByLabel('访问令牌', { exact: true })).toHaveCount(0);
  const start = page.getByRole('button', { name: '开始独立诊断', exact: true });
  await expect(start).toBeDisabled();
  await expect(page.getByRole('img', { name: 'GPU 执行时间线', exact: true })).toBeVisible();
  await expect(page.getByText('哈希一致', { exact: false })).toBeVisible();
  await expect(page.getByRole('link', { name: 'capture.nsys-rep', exact: true })).toBeVisible();
  const artifact = await page.request.get(
    `/api/console/v1/diagnostics/jobs/${run}/artifacts/capture.nsys-rep`,
  );
  expect(artifact.ok()).toBeTruthy();
  expect((await artifact.body()).length).toBeGreaterThan(1000);
  const privateInput = await page.request.get(
    `/api/console/v1/diagnostics/jobs/${run}/artifacts/input.json`,
  );
  expect(privateInput.status()).toBe(404);
  await page.getByRole('combobox', { name: '诊断工具', exact: true }).press('ArrowDown');
  await page.getByText('ncu · Kernel 计数器', { exact: true }).click();
  await expect(page.getByRole('textbox', { name: 'Kernel 过滤', exact: true })).toBeVisible();
  await page.getByRole('button', { name: '选择此 Kernel', exact: true }).first().click();
  await expect(page.getByRole('textbox', { name: 'Kernel 过滤', exact: true })).not.toHaveValue(
    '.*',
  );
  await page.screenshot({
    path: '../../reports/console/nsight/diagnostics-ui.png',
    fullPage: true,
  });
  expect(errors).toEqual([]);
});

test('a saved counter permission failure remains actionable', async ({ page }) => {
  const run = process.env.HQSB_NCU_PERMISSION_RUN;
  test.skip(!run, 'Requires a saved actual GPU counter permission error');
  await page.goto(`/diagnostics?run=${run}`);
  await expect(page.getByRole('heading', { name: '执行诊断', exact: true })).toBeVisible();
  await expect(page.getByText(/GPU_COUNTER_PERMISSION_REQUIRED/)).toBeVisible();
  await expect(page.getByRole('link', { name: 'profiler.log', exact: true })).toBeVisible();
  await expect(page.getByRole('link', { name: '源性能与质量记录', exact: true })).toBeVisible();
  await expect(page.getByRole('link', { name: 'capture.ncu-rep', exact: true })).toHaveCount(0);
});

test('two real engine captures expose mechanism changes', async ({ page }) => {
  const baseline = process.env.HQSB_NSYS_RUN;
  const candidate = process.env.HQSB_NSYS_OTHER_RUN;
  test.skip(!baseline || !candidate, 'Requires two real captures of the same workload');
  await page.goto(`/diagnostics?run=${candidate}`);
  await page.getByRole('combobox', { name: '诊断对照基线', exact: true }).press('ArrowDown');
  await page
    .locator('.ant-select-dropdown:visible .ant-select-item-option-content')
    .filter({ hasText: baseline! })
    .click();
  const response = page.waitForResponse(
    (r) => r.url().endsWith('/diagnostics/comparisons') && r.request().method() === 'POST',
  );
  await page.getByRole('button', { name: '比较机制变化', exact: true }).click();
  expect((await (await response).json()).comparable).toBe(true);
  await expect(page.getByRole('columnheader', { name: '基线累计 ms', exact: true })).toBeVisible();
  await expect(
    page.getByText('采集口径一致，以下展示诊断窗口中的机制变化。', { exact: true }),
  ).toBeVisible();
});
