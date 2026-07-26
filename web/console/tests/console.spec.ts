import { test, expect } from '@playwright/test';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
const token = readFileSync(
  process.env.HQSB_CONSOLE_TOKEN_FILE ?? path.resolve('../../.console/access-token'),
  'utf8',
).trim();
test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await page.getByLabel('访问令牌', { exact: true }).fill(token);
  await page.getByRole('button', { name: '进入工作台' }).click();
  await expect(page.getByRole('heading', { name: /从一个 token/ })).toBeVisible();
});

test('every route renders real API data without browser errors', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.screenshot({ path: '../../reports/console/overview.png', fullPage: true });
  for (const [route, title] of [
    ['/devices', '设备与部署'],
    ['/runs', '运行记录'],
    ['/compare', '运行对比'],
    ['/quantization', '量化与质量'],
    ['/kernels', '算子与配置目录'],
    ['/efficiency', '性能与能效'],
    ['/evidence', '证据中心'],
    ['/experiments', '实验地图'],
    ['/showcase', '技术展示路线'],
    ['/settings', '设置与使用指引'],
  ]) {
    await page.goto(route);
    await expect(page.getByRole('heading', { name: title, exact: true })).toBeVisible();
    await expect(page.getByText('数据读取失败', { exact: true })).toHaveCount(0);
  }
  expect(errors).toEqual([]);
});

test('evidence drawer shows original verdict and SHA256', async ({ page }) => {
  await page.goto('/evidence');
  await page.getByLabel('搜索实验').fill('E05-02');
  await page.getByRole('button', { name: 'E05-02', exact: true }).click();
  const sha = page.getByText(/SHA-256 [a-f0-9]{64}/);
  await expect(sha).toBeVisible();
  const downloadReady = page.waitForEvent('download');
  await page.getByRole('link', { name: '下载原件' }).click();
  const download = await downloadReady;
  const hash = createHash('sha256')
    .update(readFileSync((await download.path())!))
    .digest('hex');
  await expect(sha).toContainText(hash);
  await page.screenshot({ path: '../../reports/console/evidence.png', fullPage: true });
});

test('real prompt streams, metrics appear, and refresh retains final output', async ({ page }) => {
  await page.goto('/playground');
  await expect(page.getByText('已就绪', { exact: true })).toBeVisible();
  await page.getByLabel('推理输入').fill('请用两句话解释什么是 KV Cache。');
  await page.getByLabel('最大输出 token').fill('32');
  await page.getByRole('button', { name: '开始推理' }).click();
  await expect(page.locator('.output-text')).not.toBeEmpty();
  await expect(page.getByText('已完成', { exact: true })).toBeVisible({ timeout: 120000 });
  await expect(page.locator('.observation-column .stat-value').first()).not.toContainText('—');
  await page.screenshot({ path: '../../reports/console/playground.png', fullPage: true });
  const output = await page.locator('.output-text').textContent();
  await page.getByRole('link', { name: /查看完整运行/ }).click();
  await expect(page.getByRole('heading', { name: '运行详情', exact: true })).toBeVisible();
  await page.reload();
  await expect(page.locator('.detail-output')).toHaveText(output!);
  await page.getByRole('tab', { name: '指标与原始结果' }).click();
  await expect(page.getByText(/generated_token_ids/)).toBeVisible();
});

test('narrow viewport and logout are usable', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/playground');
  await expect(page.getByRole('heading', { name: '推理工作台' })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(
    true,
  );
  await page.screenshot({ path: '../../reports/console/mobile.png', fullPage: true });
  await page.getByRole('button', { name: '退出会话' }).click();
  await expect(page.getByLabel('访问令牌', { exact: true })).toBeVisible();
});
