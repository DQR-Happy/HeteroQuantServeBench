import { expect, test } from '@playwright/test';
const api = '/api/console/v1';

test('L10 resident model works and managed prerequisite is accessible', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto('/application/run?deployment=qwen-reference');
  const start = page.getByRole('button', { name: '运行 L10 服务对照', exact: true });
  await expect(start).toBeEnabled();
  await page.getByRole('combobox', { name: 'L10 副本生命周期', exact: true }).focus();
  await page.getByRole('combobox', { name: 'L10 副本生命周期', exact: true }).press('ArrowDown');
  await page.getByText('实验托管 / 加载和回收', { exact: true }).click();
  await expect(start).toBeDisabled();
  await page.locator('[data-disabled-action]').filter({ has: start }).focus();
  await expect(
    page.getByText(
      '托管副本要求设备上没有其他驻留模型，实验只接管自己加载的模型。请先在设备页卸载。',
      { exact: true },
    ),
  ).toBeVisible();
  await expect(page.getByRole('link', { name: '卸载驻留模型 →', exact: true })).toHaveAttribute(
    'href',
    '/devices#deployment-qwen-reference',
  );
  await page.screenshot({
    path: '../../reports/console/l10-application/prerequisite-desktop.png',
    fullPage: true,
  });
  expect(errors).toEqual([]);
});

test('L10 load editor rejects invalid rows without losing the working page', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto('/application/run?deployment=qwen-reference');
  await page
    .getByText('编辑请求 JSON（prompt 或 messages、tenant、arrival_ms、max_output_tokens）', {
      exact: true,
    })
    .click();
  await page.getByLabel('L10 请求 JSON').fill('[null]');
  const validation = page.waitForResponse(
    (r) => r.url().endsWith('/application/validate') && r.request().method() === 'POST',
  );
  await page.getByRole('button', { name: '应用负载 JSON', exact: true }).click();
  expect((await validation).status()).toBe(422);
  await expect(page.getByRole('alert').filter({ hasText: 'model_attributes_type' })).toBeVisible();
  await expect(page.getByRole('heading', { name: '固定多请求负载', exact: true })).toBeVisible();
  await page.getByRole('button', { name: '恢复典型负载', exact: true }).click();
  await expect(page.getByLabel('L10 请求 JSON')).toHaveValue(/Once upon a time/);
  await expect(page.getByRole('button', { name: '运行 L10 服务对照', exact: true })).toBeEnabled();
  expect(errors).toEqual([]);
});

test('real L10 result separates service delivery from computed tokens', async ({ page }) => {
  const rows = (await (await page.request.get(api + '/application/jobs')).json()).items;
  const run = rows.find(
    (r: any) => r.state === 'completed' && r.config.application_plan.candidate.result_cache,
  );
  expect(run).toBeTruthy();
  await page.goto('/runs/' + run.id);
  await expect(page).toHaveURL(/\/application\/results\?/);
  await expect(
    page.getByRole('heading', { name: '服务与应用对照结果', exact: true }),
  ).toBeVisible();
  await expect(page.getByText('固定负载逐 Token 输出一致', { exact: true })).toBeVisible();
  await expect(page.getByText('服务交付 Token/s（含缓存）', { exact: true })).toBeVisible();
  await expect(page.getByText('实际计算输出 Token', { exact: true })).toBeVisible();
  await page.getByRole('combobox', { name: 'L10 查看方案', exact: true }).focus();
  await page.getByRole('combobox', { name: 'L10 查看方案', exact: true }).press('ArrowDown');
  await page.getByTitle('候选 B', { exact: true }).click();
  await expect(page.getByText('未调用', { exact: true }).first()).toBeVisible();
  await expect(page.getByText(/结果缓存命中 2 次/)).toBeVisible();
  await expect(
    page.getByRole('link', { name: '查看封存核验后的完整 JSON →', exact: true }),
  ).toHaveAttribute('href', api + '/runs/' + run.id + '/application');
  await page.screenshot({
    path: '../../reports/console/l10-application/results-desktop.png',
    fullPage: true,
  });
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1),
  ).toBeTruthy();
  await page.screenshot({
    path: '../../reports/console/l10-application/results-mobile.png',
    fullPage: true,
  });
});
