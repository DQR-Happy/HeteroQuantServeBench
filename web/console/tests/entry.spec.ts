import { test, expect } from '@playwright/test';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const preview = path.resolve('public/preview.html');
const offlineReady = existsSync(preview);
for (const entry of ['index.html', 'dist/index.html']) {
  test(`opening ${entry} as a file shows an explanation and working preview link`, async ({
    page,
  }) => {
    await page.goto(pathToFileURL(path.resolve(entry)).href);
    await expect(page.getByRole('heading', { name: '此入口不能作为独立 HTML 运行' })).toBeVisible();
    await expect(page.getByText(/浏览器不会编译 TSX/)).toBeVisible();
    if (offlineReady) {
      await page.getByRole('link', { name: '打开可双击查看的离线界面预览' }).click();
      await expect(page.getByRole('heading', { name: 'HQSB 离线界面预览' })).toBeVisible();
    }
  });
}

test('self-contained offline preview switches actual screenshots without HTTP or API requests', async ({
  page,
}) => {
  test.skip(!offlineReady, 'Generate the private screenshot preview with build_preview.py first');
  const requests: string[] = [];
  page.on('request', (request) => {
    if (/^https?:/.test(request.url())) requests.push(request.url());
  });
  await page.goto(pathToFileURL(preview).href);
  await expect(page.getByText(/不连接后端、不执行推理/)).toBeVisible();
  for (const name of [
    '工作空间',
    '运行目录',
    '推理配置',
    '优化实验',
    'KV 布局',
    '分配与回收',
    'CPU 页样本',
    '算子关联',
    'A/B 对照',
    '手机界面',
  ]) {
    await page.getByRole('link', { name, exact: true }).click();
    await expect(page.getByRole('heading', { name, exact: true })).toBeVisible();
    const screenshot = page.getByRole('img', { name: `HQSB ${name}界面` });
    await expect(screenshot).toBeVisible();
    await expect
      .poll(() => screenshot.evaluate((img) => (img as HTMLImageElement).naturalWidth))
      .toBeGreaterThan(0);
  }
  expect(requests).toEqual([]);
  await page.getByRole('link', { name: '工作空间', exact: true }).click();
  await page.screenshot({ path: '../../reports/console/v04/offline-preview.png', fullPage: true });
});

test('HTTP entry loads the React workspace or an explicitly enabled login screen', async ({
  page,
}) => {
  await page.goto('/');
  const response = await page.request.get('/api/console/v1/session');
  if (response.status() === 401) {
    await expect(page.getByRole('heading', { name: '连接工作台', exact: true })).toBeVisible();
    await expect(page.getByLabel('访问令牌', { exact: true })).toBeVisible();
  } else {
    expect(response.ok()).toBeTruthy();
    await expect(page.getByRole('heading', { name: '工作空间', exact: true })).toBeVisible();
    await expect(page.getByLabel('访问令牌', { exact: true })).toHaveCount(0);
  }
  await expect(page.getByRole('heading', { name: '此入口不能作为独立 HTML 运行' })).toHaveCount(0);
});
