import { test, expect, type Page } from '@playwright/test';
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { enterWorkspace, checkSessionExit } from './session';

const reportFile =
  process.env.HQSB_OBSERVATION_REPORT ??
  path.resolve('../../reports/observation/console-validation-final-code.json');
const hasArchive = existsSync(reportFile);
const report: { selected_runs: Record<string, string> } = hasArchive
  ? JSON.parse(readFileSync(reportFile, 'utf8'))
  : { selected_runs: {} };
test.skip(!hasArchive, 'Requires a real validate_console.py archive via HQSB_OBSERVATION_REPORT');
test.use({ actionTimeout: 15000 });
const memory = report.selected_runs.memory;
const basic = report.selected_runs.basic;
async function login(page: Page) {
  await enterWorkspace(page);
  await expect(page.getByRole('heading', { name: '工作空间', exact: true })).toBeVisible();
}
test.beforeEach(async ({ page }) => {
  await login(page);
});

test('all workspaces render archived real API data and shared navigation', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.screenshot({
    path: '../../reports/console/v04/workspace-desktop.png',
    fullPage: true,
  });
  for (const [route, title] of [
    ['/playground', '推理工作台'],
    ['/runs', '运行记录'],
    ['/observations', '机制观测'],
    ['/memory-flow', '内存与数据流'],
    ['/compare', '运行对比'],
    ['/research', '深度分析'],
    ['/quantization', '量化与质量'],
    ['/kernels', '算子与配置目录'],
    ['/efficiency', '性能与能效'],
    ['/evidence', '证据中心'],
    ['/experiments', '实验地图'],
    ['/devices', '设备与部署'],
    ['/showcase', '技术展示路线'],
    ['/settings', '设置与使用指引'],
  ]) {
    await page.goto(route);
    await expect(page.getByRole('heading', { name: title, exact: true })).toBeVisible();
    await expect(page.getByText('数据读取失败', { exact: true })).toHaveCount(0);
  }
  await page.goto('/playground');
  await expect(page.getByLabel('推理输入')).toBeVisible();
  await expect(page.getByRole('heading', { name: '尚未提交请求', exact: true })).toBeVisible();
  await page.screenshot({
    path: '../../reports/console/v04/playground-desktop.png',
    fullPage: true,
  });
  expect(errors).toEqual([]);
});

test('mechanism investigation retains run identity, filters, layout and native sequence', async ({
  page,
}) => {
  await page.goto(`/observations?run=${memory}`);
  await expect(page.getByText('采集 complete · supervisor_acknowledged')).toBeVisible();
  await page.getByLabel('搜索观测指标').fill('cpu_minor_fault');
  await expect(
    page.getByRole('cell', { name: 'cpu_minor_fault_delta', exact: true }),
  ).toBeVisible();
  await page.getByLabel('搜索观测指标').fill('no_such_metric');
  await expect(page.getByText('当前条件没有匹配指标')).toBeVisible();
  await page.getByRole('tab', { name: 'KV 增长与布局', exact: true }).click();
  await expect(page.getByRole('cell', { name: 'layer.0.key', exact: true })).toBeVisible();
  await expect(page).toHaveURL(new RegExp(`run=${memory}.*view=kv`));
  await page.screenshot({ path: '../../reports/console/v04/kv-desktop.png', fullPage: true });
  await page.reload();
  await expect(page.getByRole('tab', { name: 'KV 增长与布局', exact: true })).toHaveAttribute(
    'aria-selected',
    'true',
  );
  await page.getByRole('tab', { name: '权重布局', exact: true }).click();
  await page
    .getByRole('tabpanel', { name: '权重布局', exact: true })
    .getByLabel('搜索 Tensor 名称')
    .fill('embed_tokens');
  await expect(
    page.getByRole('cell', { name: 'model.embed_tokens.weight', exact: true }),
  ).toBeVisible();
  await page.getByRole('tab', { name: '申请与回收', exact: true }).click();
  await page.getByRole('tab', { name: '分配代次', exact: true }).click();
  await expect(page.getByText(/预览 500 \/ /)).toBeVisible();
  await page.getByRole('tab', { name: 'Tensor 关联', exact: true }).click();
  await expect(
    page.getByRole('cell', { name: 'snapshot_correlated', exact: true }).first(),
  ).toBeVisible();
  await page.screenshot({
    path: '../../reports/console/v04/allocator-desktop.png',
    fullPage: true,
  });
  await page.getByRole('tab', { name: 'CPU 页样本', exact: true }).click();
  await expect(page.getByText('本次没有 CPU 页样本。请使用页诊断模式采集。')).toBeVisible();
  await page.getByRole('tab', { name: '原始证据', exact: true }).click();
  await page.getByText('按来源序号查询原始记录', { exact: true }).click();
  await expect(page.getByRole('cell', { name: 'boundary', exact: true }).first()).toBeVisible();
  await page.getByRole('button', { name: '下一页', exact: true }).click();
  await expect(page.getByText('第 2 页', { exact: false })).toBeVisible();
  await page.getByRole('button', { name: '上一页', exact: true }).click();
  await expect(page.getByText('第 1 页', { exact: false })).toBeVisible();
  await page.getByRole('tab', { name: '缺失与边界', exact: true }).click();
  await expect(page.getByText('暂不可测', { exact: true })).toBeVisible();
});

test('comparison is independent of latency gate, retains hold and clears stale selections', async ({
  page,
}) => {
  await page.goto(`/compare?left=${memory}&right=${basic}`);
  await page.getByRole('button', { name: '检查可比性与机制差异' }).click();
  await expect(page.getByText('对照受阻：控制条件不一致', { exact: true })).toBeVisible();
  await expect(page.getByText('决策 hold', { exact: true })).toBeVisible();
  await expect(page.getByText('质量 not_evaluated', { exact: true })).toBeVisible();
  await page.screenshot({ path: '../../reports/console/v04/compare-desktop.png', fullPage: true });
  await page.getByRole('combobox', { name: '机制对照变量' }).press('Enter');
  await page.getByText('控制变量：Attention 实现', { exact: true }).click();
  await expect(page.getByText('决策 hold', { exact: true })).toHaveCount(0);
  await page.goto(`/compare?left=${memory}`);
  await expect(page.getByRole('button', { name: '检查可比性与机制差异' })).toBeDisabled();
});

test('same-capture correlation presents tool IDs and coverage, not temporal inference', async ({
  page,
}) => {
  await page.goto(`/observations?run=${report.selected_runs.operators}&view=links`);
  await expect(
    page.getByRole('cell', { name: 'tool_external_id', exact: true }).first(),
  ).toBeVisible();
  await expect(page.getByText(/有界预览 200 条/)).toBeVisible();
  await page.screenshot({
    path: '../../reports/console/v04/correlation-desktop.png',
    fullPage: true,
  });
});

test('mobile and narrow desktop preserve main actions, readable content and signout', async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole('button', { name: '打开导航' }).click();
  await expect(page.getByRole('dialog')).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(page.getByRole('dialog')).toHaveCount(0);
  await expect(page.getByRole('button', { name: '打开导航' })).toBeFocused();
  await page.getByRole('button', { name: '打开导航' }).click();
  await page.getByRole('link', { name: '机制观测', exact: true }).click();
  await expect(page).toHaveURL(/\/observations$/);
  await expect(page.getByRole('dialog')).toHaveCount(0);
  for (const route of [
    '/playground',
    `/observations?run=${memory}&view=kv`,
    `/compare?left=${memory}&right=${basic}`,
  ]) {
    await page.goto(route);
    await expect(page.locator('h1')).toBeVisible();
    await expect
      .poll(() => page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth))
      .toBeLessThanOrEqual(1);
  }
  await page.screenshot({ path: '../../reports/console/v04/compare-mobile.png', fullPage: true });
  await page.setViewportSize({ width: 720, height: 450 });
  await page.goto(`/observations?run=${memory}&view=metrics`);
  await expect(page.getByLabel('搜索观测指标')).toBeVisible();
  await expect
    .poll(() => page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth))
    .toBeLessThanOrEqual(1);
  await checkSessionExit(page);
});

test('sealed integrity failure is explicit and never replaced by an empty or successful result', async ({
  page,
}) => {
  // Isolated response injection; does not modify the archived evidence.
  await page.route(`**/runs/${memory}/observations`, (route) =>
    route.fulfill({
      status: 409,
      contentType: 'application/json',
      body: JSON.stringify({ detail: 'Observation integrity check failed: FIXTURE_HASH_MISMATCH' }),
    }),
  );
  await page.goto(`/observations?run=${memory}`);
  await expect(page.getByText('数据读取失败', { exact: true })).toBeVisible();
  await expect(page.getByText(/FIXTURE_HASH_MISMATCH/)).toBeVisible();
  await expect(page.getByText('采集 complete · supervisor_acknowledged')).toHaveCount(0);
  await page.screenshot({ path: '../../reports/console/v04/integrity-error.png', fullPage: true });
});

test('candidate selection uses requested modes from run metadata and distinguishes decode steps', async ({
  page,
}) => {
  await page.goto(`/observations?run=${memory}&view=kv`);
  const select = page.getByRole('combobox', { name: '选择布局采样边界' });
  await select.press('Enter');
  await page.getByText('decode_step · step 2', { exact: true }).click();
  await expect(
    page.getByRole('cell', { name: '[1, 8, 25, 128] [25600, 3200, 128, 1]', exact: true }).first(),
  ).toBeVisible();
  const candidate = page.getByRole('combobox', { name: '选择观测运行' });
  await candidate.press('Enter');
  await page.getByText(new RegExp(`${report.selected_runs.pages} · pages ·`)).click();
  await expect(page).toHaveURL(new RegExp(`run=${report.selected_runs.pages}`));
  await expect(page.getByText('pages', { exact: true })).toBeVisible();
  await expect(page.getByRole('tab', { name: '内存与缺页', exact: true })).toHaveAttribute(
    'aria-selected',
    'true',
  );
  await page.getByRole('tab', { name: 'CPU 页样本', exact: true }).click();
  await expect(
    page.getByRole('cell', { name: 'sampled_cpu_pages_not_gpu_mappings', exact: true }).first(),
  ).toBeVisible();
  await page.screenshot({ path: '../../reports/console/v04/pages-desktop.png', fullPage: true });
});
