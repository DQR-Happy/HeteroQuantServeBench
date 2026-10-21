import { test, expect } from '@playwright/test';

test.beforeEach(async ({ page }) => {
  await expect
    .poll(
      async () => {
        const response = await page.request.get('/api/console/v1/overview');
        return (await response.json()).pending;
      },
      { timeout: 120000 },
    )
    .toBe(0);
});

test('real ERETrace capture exposes warp evidence and complete raw files', async ({ page }) => {
  const run = process.env.HQSB_ERE_RUN;
  test.skip(!run, 'Requires an actual completed ERETrace capture');
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  const response = await page.request.get(`/api/console/v1/runs/${run}`);
  const record = await response.json();
  expect(record.state).toBe('completed');
  const panel = record.metrics.diagnostic.panel;
  expect(panel.counts.observed_records).toBeGreaterThan(0);
  await page.goto(`/diagnostics?source=${record.config.source_run_id}&run=${run}`);
  await expect(page.getByRole('img', { name: 'SM warp 记录分布', exact: true })).toBeVisible();
  await expect(page.getByRole('columnheader', { name: 'P95 cycles', exact: true })).toBeVisible();
  if (panel.trace_bytes < 1024) {
    await expect(page.getByText(`记录文件 ${panel.trace_bytes} B`, { exact: true })).toBeVisible();
  }
  await expect(page.getByRole('link', { name: 'capture.ncu-rep', exact: true })).toHaveCount(0);
  await expect(page.getByLabel('访问令牌', { exact: true })).toHaveCount(0);
  const raw = await page.request.get(
    `/api/console/v1/diagnostics/jobs/${run}/artifacts/${panel.decoded_artifacts[0].path}`,
  );
  expect(raw.ok()).toBeTruthy();
  expect((await raw.body()).length).toBeGreaterThan(56);
  await page.getByRole('combobox', { name: '诊断工具', exact: true }).press('ArrowDown');
  await page.getByText('ERETrace · warp / SM 观测', { exact: true }).click();
  await expect(
    page.getByRole('textbox', { name: 'ERETrace Kernel 名称', exact: true }),
  ).toBeVisible();
  await expect(page.getByRole('button', { name: '开始独立诊断', exact: true })).toBeEnabled();
  await page.keyboard.press('Escape');
  await page.getByRole('heading', { name: '执行诊断', exact: true }).click();
  await page.screenshot({
    path: '../../reports/console/eretrace/diagnostics-desktop.png',
    fullPage: true,
  });
  await page.setViewportSize({ width: 1100, height: 800 });
  await expect(page.getByRole('img', { name: 'SM warp 记录分布', exact: true })).toBeVisible();
  await page.screenshot({
    path: '../../reports/console/eretrace/diagnostics-compact.png',
    fullPage: true,
  });
  expect(errors).toEqual([]);
});

test('ERETrace explains concurrency prerequisites without changing source options', async ({
  page,
}) => {
  const source = process.env.HQSB_ERE_CONCURRENT_SOURCE;
  test.skip(!source, 'Requires an existing concurrency > 1 source');
  await page.goto(`/diagnostics?source=${source}`);
  await page.getByRole('combobox', { name: '诊断工具', exact: true }).press('ArrowDown');
  await page.getByText('ERETrace · warp / SM 观测', { exact: true }).click();
  const button = page.getByRole('button', { name: '开始独立诊断', exact: true });
  await expect(button).toBeDisabled();
  await button.hover({ force: true });
  await expect(
    page.getByText('请在原生引擎公共评测中将请求并发设为 1，完成评测后选择新记录。', {
      exact: true,
    }),
  ).toBeVisible();
  await expect(page.getByRole('link', { name: /配置并重新评测/ })).toBeVisible();
  await expect(page.getByRole('link', { name: /配置并重新评测/ })).toHaveAttribute(
    'href',
    /\/engines\/benchmark\?deployment=/,
  );
});
