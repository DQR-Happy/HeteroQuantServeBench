import { expect, test } from '@playwright/test';

test('L8 history opens without login and exposes actual batches and evidence', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  const listing = await page.request.get('/api/console/v1/scheduling-benchmarks');
  expect(listing.ok()).toBeTruthy();
  const runs = (await listing.json()).items as { id: string; state: string; quality: string }[];
  const run = runs.find((r) => r.state === 'completed' && r.quality === 'exact_tokens_pass');
  test.skip(!run, 'Needs retained real-model scheduling evidence');
  await page.goto(`/scheduling?experiment=${run!.id}`);
  await expect(page.getByRole('heading', { name: '多请求调度实验', exact: true })).toBeVisible();
  await expect(page.getByLabel('访问令牌', { exact: true })).toHaveCount(0);
  await expect(page.getByRole('columnheader', { name: 'Token/s', exact: true })).toBeVisible();
  await expect(page.getByRole('columnheader', { name: '真实 B 最大值' })).toBeVisible();
  await page.getByRole('button', { name: '查看原始记录', exact: true }).click();
  await expect(page.locator('pre').filter({ hasText: 'input_shape' })).toBeVisible();
  await page.getByLabel('原始记录类型').first().click();
  await page.getByTitle('kv.release', { exact: true }).click();
  await expect(page.locator('pre').filter({ hasText: 'kv.release' })).toBeVisible();
  await page.screenshot({ path: '../../reports/console/l8/scheduling.png', fullPage: true });
  expect(errors).toEqual([]);
});
