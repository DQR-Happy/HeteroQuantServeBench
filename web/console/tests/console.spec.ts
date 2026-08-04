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
  await expect(page.locator('.deployment-row').first()).toBeVisible();
  await page.screenshot({ path: '../../reports/console/overview.png', fullPage: true });
  for (const [route, title] of [
    ['/devices', '设备与部署'],
    ['/memory-flow', '内存与数据流'],
    ['/runs', '运行记录'],
    ['/research', '深度分析'],
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
  await page.getByRole('tab', { name: '全链路观测', exact: true }).click();
  await expect(page.getByRole('heading', { name: '阶段瀑布 · 工作进程主机时间' })).toBeVisible();
  await expect(page.getByRole('img', { name: '真实请求阶段时间线' })).toBeVisible();
  await page.getByRole('tab', { name: '优化假设', exact: true }).click();
  await expect(page.getByText('以下是证据驱动的候选假设，尚未证明优化有效。')).toBeVisible();
  await expect(page.getByText('数据读取失败', { exact: true })).toHaveCount(0);
});

test('memory attribution and historical research keep observation scopes visible', async ({
  page,
}) => {
  await page.goto('/devices');
  await expect(
    page.getByRole('heading', { name: '内存归因与生命周期', exact: true }),
  ).toBeVisible();
  await expect(page.getByText('系统现在可用', { exact: true })).toBeVisible();
  await expect(page.getByText('Console API', { exact: true })).toBeVisible();
  await expect(page.getByRole('heading', { name: '张量内存账本', exact: true })).toBeVisible();
  await expect(page.getByRole('tab', { name: '权重与 buffer', exact: true })).toBeVisible();
  await expect(page.getByText('数据读取失败', { exact: true })).toHaveCount(0);
  await page.goto('/research');
  await expect(
    page.getByRole('heading', { name: '阶段 → 框架算子 → GPU kernel', exact: true }),
  ).toBeVisible();
  await expect(page.getByLabel('选择样本与阶段')).toBeVisible();
  await page.getByRole('tab', { name: '量化执行真实性', exact: true }).click();
  await expect(
    page.getByRole('heading', { name: '量化：文件体积、执行路径与质量', exact: true }),
  ).toBeVisible();
  await expect(page.getByLabel('选择历史量化方案')).toBeVisible();
  await page.getByRole('tab', { name: '当前采集能力', exact: true }).click();
  await expect(page.getByRole('heading', { name: '采集能力与边界', exact: true })).toBeVisible();
  await page.goto('/quantization');
  await expect(page.getByRole('button', { name: '创建量化候选', exact: true })).toBeVisible();
  await expect(
    page.getByText('生成的是存储制品，质量尚未评估；不会自动切换当前 FP16 推理路径。'),
  ).toBeVisible();
  await expect(page.getByText('数据读取失败', { exact: true })).toHaveCount(0);
});

test('operator capture exposes real trace events and a downloadable Chrome trace', async ({
  page,
}) => {
  await page.goto('/playground');
  await expect(page.getByText('已就绪', { exact: true })).toBeVisible();
  await page.getByLabel('推理输入').fill('用一句话解释内存带宽。');
  await page.getByLabel('最大输出 token').fill('9');
  // Ant Select's visual value covers its readonly input; exercise the
  // accessible keyboard interaction instead of clicking that hidden surface.
  await page.getByLabel('观测模式').press('Enter');
  await page.getByText('算子诊断 · PyTorch Profiler', { exact: true }).click();
  await expect(page.getByText('诊断采集有开销', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: '开始推理' }).click();
  await expect(page.getByText('已完成', { exact: true })).toBeVisible({ timeout: 120000 });
  await page.getByRole('link', { name: /查看完整运行/ }).click();
  await page.getByRole('tab', { name: '全链路观测', exact: true }).click();
  await page.getByRole('tab', { name: '算子耗时', exact: true }).click();
  await expect(page.getByLabel('搜索采集算子')).toBeVisible();
  await expect(page.getByRole('cell', { name: /aten::/ }).first()).toBeVisible();
  await page.getByRole('tab', { name: '原始事件查询', exact: true }).click();
  await expect(page.getByLabel('搜索 trace 事件')).toBeVisible();
  await expect(page.getByText('数据读取失败', { exact: true })).toHaveCount(0);
  const downloadReady = page.waitForEvent('download');
  await page.getByRole('link', { name: /下载 Chrome trace/ }).click();
  const download = await downloadReady;
  const trace = JSON.parse(readFileSync((await download.path())!, 'utf8')) as {
    traceEvents: unknown[];
  };
  expect(trace.traceEvents.length).toBeGreaterThan(0);
  await page.screenshot({ path: '../../reports/console/observation-v02.png', fullPage: true });
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
