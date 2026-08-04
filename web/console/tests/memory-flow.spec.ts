import { test, expect, type Page } from '@playwright/test';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import type { Dataflow, ObservedRun } from '../src/features/memory-flow/types';

const token = readFileSync(
  process.env.HQSB_CONSOLE_TOKEN_FILE ?? path.resolve('../../.console/access-token'),
  'utf8',
).trim();
const api = '/api/console/v1';

// Read-only acceptance: replay saved device evidence; never load a model or
// submit inference while independent S05 experiments use this Jetson.
test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await page.getByLabel('访问令牌', { exact: true }).fill(token);
  await page.getByRole('button', { name: '进入工作台' }).click();
  await expect(page.getByRole('heading', { name: /从一个 token/ })).toBeVisible();
});

async function savedRuns(page: Page) {
  const response = await page.request.get(`${api}/runs?limit=100`);
  expect(response.ok()).toBe(true);
  return ((await response.json()) as { items: ObservedRun[] }).items.filter(
    (run) => run.kind === 'generate' && run.state === 'completed',
  );
}

test('saved operator evidence opens four real interactive visualizations by default', async ({
  page,
}) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  const run = (await savedRuns(page)).find(
    (entry) => entry.config.observation_mode === 'operators',
  );
  expect(run, 'acceptance requires an existing real operator capture').toBeDefined();
  await page.goto('/memory-flow');
  await expect(page.getByRole('heading', { name: '内存与数据流', exact: true })).toBeVisible();
  await expect(page.getByRole('link', { name: '查看完整运行 →', exact: true })).toHaveAttribute(
    'href',
    `/runs/${run!.id}`,
  );
  for (const id of [
    'memory-topology',
    'weight-treemap',
    'kv-layer-chart',
    'request-dataflow',
    'transfer-summary',
  ]) {
    await expect(page.getByTestId(id)).toBeVisible();
  }
  await expect(page.getByText('正在回放历史请求的真实证据')).toBeVisible();
  const tile = page.getByTestId('weight-treemap').getByRole('button').first();
  await tile.click();
  await expect(tile).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('.mf-selection-detail').getByRole('table')).toBeVisible();
  const layer = page.getByTestId('kv-layer-chart').getByRole('button').nth(1);
  await layer.click();
  await expect(layer).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('.mf-kv-detail')).toContainText('Layer 1');
  await page
    .getByTestId('request-dataflow')
    .getByRole('button', { name: /CUDA 输入/ })
    .click();
  await expect(page.locator('.mf-flow-detail')).toContainText('输入迁移之后启动 profiler');
  await page.screenshot({
    path: '../../reports/console/v021/memory-flow-desktop.png',
    fullPage: true,
  });
  expect(errors).toEqual([]);
});

test('copy paths match the real authenticated API and filter saved events', async ({ page }) => {
  const run = (await savedRuns(page)).find(
    (entry) => entry.config.observation_mode === 'operators',
  );
  expect(run).toBeDefined();
  const response = await page.request.get(`${api}/runs/${run!.id}/dataflow`);
  expect(response.ok()).toBe(true);
  const flow = (await response.json()) as Dataflow;
  expect(flow.status).toBe('available');
  expect(flow.totals.kernel_events).toBeGreaterThan(0);
  expect(flow.coverage.trace_sha256).toMatch(/^[a-f0-9]{64}$/);
  await page.goto(`/memory-flow?run=${run!.id}`);
  const summary = page.getByTestId('transfer-summary');
  await expect(summary).toContainText(flow.totals.copy_events.toLocaleString('zh-CN'));
  await expect(summary).toContainText(flow.totals.kernel_events.toLocaleString('zh-CN'));
  for (const edge of flow.edges) {
    const direction = page
      .locator('.mf-copy-path')
      .filter({ hasText: edge.direction === 'host_to_device' ? 'Host → Device' : 'Device → Host' });
    await expect(direction).toContainText(`${edge.count} 次`);
    if (edge.bytes != null && edge.bytes < 1024)
      await expect(direction).toContainText(`${edge.bytes} B`);
  }
  await page.getByRole('button', { name: /Host → Device/ }).click();
  await page.locator('.mf-event-details summary').filter({ hasText: '查看真实拷贝事件' }).click();
  const table = page.locator('.mf-event-details').getByRole('table');
  await expect(table).toContainText('Host → Device');
  await expect(table).not.toContainText('Device → Host');
  await page.reload();
  await expect(page.getByRole('link', { name: '查看完整运行 →', exact: true })).toHaveAttribute(
    'href',
    `/runs/${run!.id}`,
  );
  await expect(page.getByTestId('weight-treemap')).toBeVisible();
  await page.getByTestId('request-dataflow').scrollIntoViewIfNeeded();
  await page.screenshot({ path: '../../reports/console/v021/dataflow-detail.png' });
});

test('basic observation retains memory evidence without claiming zero GPU activity', async ({
  page,
}) => {
  const run = (await savedRuns(page)).find((entry) => entry.config.observation_mode === 'basic');
  expect(run, 'acceptance requires an existing basic observation').toBeDefined();
  await page.goto(`/memory-flow?run=${run!.id}`);
  await expect(page.getByTestId('memory-topology')).toBeVisible();
  await expect(page.getByTestId('request-dataflow')).toBeVisible();
  const summary = page.getByTestId('transfer-summary');
  await expect(summary).toContainText('未采集');
  await expect(summary).toContainText('总字节未完整采集');
  await expect(
    page.getByText('这个采集窗口没有可展示的 copy 方向证据。未捕获不代表没有发生传输。'),
  ).toBeVisible();
  await expect(page.getByText('读取数据流失败', { exact: true })).toHaveCount(0);
});

test('memory and flow diagrams fit a narrow viewport', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/memory-flow');
  await expect(page.getByTestId('weight-treemap')).toBeVisible();
  await expect(page.getByTestId('transfer-summary')).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(
    true,
  );
  await page.screenshot({
    path: '../../reports/console/v021/memory-flow-mobile.png',
    fullPage: true,
  });
});

test('an explicitly selected saved unobserved request explains its missing data', async ({
  page,
}) => {
  const run = (await savedRuns(page)).find((entry) => entry.config.observation_mode === 'off');
  expect(run).toBeDefined();
  await page.goto(`/memory-flow?run=${run!.id}`);
  await expect(
    page.getByText('本次请求未采集阶段与 allocator 观测', { exact: true }),
  ).toBeVisible();
  await expect(page.getByTestId('transfer-summary')).toContainText('未采集');
});

test('partial KV and genuine zero allocator are distinct from missing evidence', async ({
  page,
}) => {
  const run = (await savedRuns(page)).find((entry) => entry.config.observation_mode === 'basic');
  expect(run).toBeDefined();
  // Controlled presentation fixture, not a new hardware measurement. Keep the
  // real run/load identity and alter only the edge-case observation fields.
  await page.route(`**${api}/runs/${run!.id}`, async (route) => {
    const response = await route.fetch();
    const record = (await response.json()) as ObservedRun;
    const kv = record.metrics.kv_inventory!;
    kv.entries = kv.entries.filter((entry) => entry.kind === 'key').slice(0, 1);
    kv.entry_count = kv.entries.length;
    kv.truncated = true;
    kv.availability = 'partial';
    kv.unique_storage_bytes = kv.entries[0].storage_bytes;
    kv.logical_total_bytes = kv.entries[0].logical_bytes;
    for (const snapshot of record.metrics.observation!.memory_snapshots) {
      snapshot.allocated_bytes = 0;
      snapshot.reserved_bytes = 0;
    }
    await route.fulfill({ response, json: record });
  });
  await page.goto(`/memory-flow?run=${run!.id}`);
  await expect(page.getByText('仅绘制已观测的 KV 分量，缺失部分不是零容量')).toBeVisible();
  await expect(page.locator('.mf-kv-detail').getByText('V 未采集', { exact: true })).toBeVisible();
  await expect(
    page.getByRole('img', { name: 'Reserved 0 B 包含 Allocated 0 B', exact: true }),
  ).toBeVisible();
});
