import { test, expect } from '@playwright/test';
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import type { LocalInventory, LocalModel } from '../src/features/model-hub/types';

test.beforeEach(async ({ page }) => {
  if (process.env.HQSB_DELETE_PREVIEW === '1') {
    const dist = path.resolve('dist');
    await page.route('**/*', async (route) => {
      const pathname = new URL(route.request().url()).pathname;
      if (pathname.startsWith('/api/')) return route.continue();
      const file = path.join(
        dist,
        pathname.startsWith('/assets/') || pathname === '/favicon.svg' ? pathname : 'index.html',
      );
      if (!file.startsWith(dist + path.sep) || !existsSync(file)) return route.abort();
      await route.fulfill({
        body: readFileSync(file),
        contentType: file.endsWith('.js')
          ? 'application/javascript'
          : file.endsWith('.css')
            ? 'text/css'
            : file.endsWith('.svg')
              ? 'image/svg+xml'
              : 'text/html',
      });
    });
  }
});
const fixture = (): LocalModel => ({
  id: 'local-fixture',
  path: '/tmp/hqsb-delete-fixture/snapshot',
  model_id: 'Qwen/Delete-fixture',
  model_type: 'qwen3',
  revision_name: 'v1',
  kind: 'download',
  exists: true,
  complete: true,
  logical_bytes: 1024,
  allocated_bytes: 4096,
  file_count: 4,
  revision: 'b'.repeat(64),
  issues: [],
  references: [{ id: 'delete-fixture', name: '测试部署', state: 'unloaded' }],
  download_id: 'download-fixture',
  download_state: 'completed',
  deployment_id: 'delete-fixture',
  deletion: {
    allowed: true,
    reason: '',
    next: { to: '/devices#deployment-delete-fixture', label: '打开对应部署' },
  },
});
const report = (items: LocalModel[]): LocalInventory => ({
  items,
  roots: ['/tmp/hqsb-delete-fixture'],
  sampled_at: 1792392217,
  logical_bytes: items.reduce((s, r) => s + r.logical_bytes, 0),
  allocated_bytes: items.reduce((s, r) => s + r.allocated_bytes, 0),
  file_count: items.reduce((s, r) => s + r.file_count, 0),
  issues: [],
  volumes: [
    {
      device: 'fixture',
      path: '/tmp/hqsb-delete-fixture',
      total_bytes: 100 * 1024 ** 3,
      used_bytes: 50 * 1024 ** 3,
      free_bytes: 50 * 1024 ** 3,
    },
  ],
});

test('download page links completed transfers to the independent local inventory', async ({
  page,
}) => {
  const row = fixture();
  await page.route('**/model-hub/config', (r) =>
    r.fulfill({
      json: {
        publishers: [{ id: 'Qwen', name: 'Qwen', provenance: 'official' }],
        capacity: {
          gpu_total_bytes: 24 * 1024 ** 3,
          gpu_name: 'RTX 3090',
          host_available_bytes: 50 * 1024 ** 3,
          disk_free_bytes: 100 * 1024 ** 3,
          download_directory: '/tmp/fixture',
        },
        scope: '测试制品',
        runtime: 'FP16',
        disk_reserve_bytes: 1024 ** 3,
      },
    }),
  );
  await page.route('**/model-hub/catalog?*', (r) =>
    r.fulfill({
      json: {
        state: 'not_loaded',
        items: [],
        total: 0,
        indexed: 0,
        publisher: 'Qwen',
        error: null,
      },
    }),
  );
  await page.route('**/model-hub/downloads', (r) =>
    r.fulfill({
      json: {
        items: [
          {
            id: row.download_id,
            model_id: row.model_id,
            revision: 'v1',
            state: 'completed',
            downloaded_bytes: 1024,
            total_bytes: 1024,
            verified_files: 4,
            total_files: 4,
            attempt: 1,
          },
        ],
      },
    }),
  );
  await page.route('**/model-hub/local-models', (r) => r.fulfill({ json: report([row]) }));
  await page.goto('/models');
  await expect(page.getByRole('heading', { name: '模型下载', exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: '删除本地制品', exact: true })).toHaveCount(0);
  await expect(page.getByRole('button', { name: '登记为推理部署', exact: true })).toHaveCount(0);
  await page.getByRole('link', { name: '查看本地模型 →', exact: true }).click();
  await expect(page).toHaveURL(/\/models\/local\?download=download-fixture$/);
  await expect(page.getByRole('heading', { name: '本地模型管理', exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: '删除本地制品', exact: true })).toBeEnabled();
});

test('confirm exact directory, cancel, delete and preserve state after refresh', async ({
  page,
}) => {
  const row = fixture();
  let rows = [row],
    calls = 0;
  await page.route('**/model-hub/local-models', (r) => r.fulfill({ json: report(rows) }));
  await page.route('**/model-hub/local-models/*/delete', async (r) => {
    calls++;
    expect(r.request().postDataJSON()).toEqual({ revision: row.revision });
    rows = [];
    await r.fulfill({ json: { state: 'deleted' } });
  });
  await page.goto('/models/local');
  await page.getByRole('button', { name: '删除本地制品', exact: true }).click();
  const dialog = page.getByRole('dialog');
  await expect(dialog).toContainText(row.path);
  await expect(dialog).toContainText('历史推理、评测与观测记录继续保留');
  await dialog.getByRole('button', { name: '保留文件' }).click();
  expect(calls).toBe(0);
  await page.getByRole('button', { name: '删除本地制品', exact: true }).click();
  await dialog.getByRole('button', { name: '确认删除' }).click();
  await expect(page.getByRole('button', { name: '删除本地制品', exact: true })).toHaveCount(0);
  expect(calls).toBe(1);
  await page.reload();
  await expect(
    page.getByText('当前扫描范围内没有匹配的模型目录，可添加目录或到下载页取得模型。'),
  ).toBeVisible();
});

test('loaded models explain prerequisite on hover and keyboard focus', async ({ page }) => {
  const row = fixture();
  row.deletion.allowed = false;
  row.deletion.reason = '测试部署当前为 ready，请先在设备与部署页卸载';
  await page.route('**/model-hub/local-models', (r) => r.fulfill({ json: report([row]) }));
  await page.goto('/models/local');
  const button = page.getByRole('button', { name: '删除本地制品', exact: true });
  await expect(button).toBeDisabled();
  const wrapper = button.locator('..');
  await wrapper.hover();
  await expect(page.getByText(row.deletion.reason, { exact: true })).toBeVisible();
  await wrapper.focus();
  await expect(wrapper).toBeFocused();
  await expect(page.getByRole('link', { name: '打开对应部署 →' })).toHaveAttribute(
    'href',
    row.deletion.next!.to,
  );
});

test('cleanup errors remain visible and retry refreshes actual inventory', async ({ page }) => {
  const row = fixture();
  let rows = [row],
    calls = 0;
  await page.route('**/model-hub/local-models', (r) => r.fulfill({ json: report(rows) }));
  await page.route('**/model-hub/local-models/*/delete', async (r) => {
    if (++calls === 1)
      await r.fulfill({ status: 409, json: { error: { message: 'fixture disk busy' } } });
    else {
      rows = [];
      await r.fulfill({ json: { state: 'deleted' } });
    }
  });
  await page.goto('/models/local');
  await page.getByRole('button', { name: '删除本地制品', exact: true }).click();
  await page.getByRole('dialog').getByRole('button', { name: '确认删除' }).click();
  await page.getByRole('dialog').getByRole('button', { name: '保留文件' }).click();
  await expect(page.getByText('fixture disk busy')).toBeVisible();
  await page.getByRole('button', { name: '删除本地制品', exact: true }).click();
  await page.getByRole('dialog').getByRole('button', { name: '确认删除' }).click();
  await expect(page.getByRole('button', { name: '删除本地制品', exact: true })).toHaveCount(0);
  expect(calls).toBe(2);
});

test('local inventory supports search, measured sorting, refresh and another scan root', async ({
  page,
}) => {
  const small = fixture(),
    big = {
      ...fixture(),
      id: 'local-large',
      model_id: 'Vendor/Large',
      allocated_bytes: 1000000,
      path: '/tmp/large',
      kind: 'local',
    };
  let refreshed = 0;
  await page.route('**/model-hub/local-models', (r) => r.fulfill({ json: report([small, big]) }));
  await page.route('**/model-hub/local-models/refresh', (r) => {
    refreshed++;
    return r.fulfill({ json: report([small, big]) });
  });
  await page.route('**/model-hub/local-models/roots', (r) => {
    expect(r.request().postDataJSON()).toEqual({ path: '/tmp/another' });
    return r.fulfill({ json: { ...report([small, big]), roots: ['/tmp/another'] } });
  });
  await page.goto('/models/local');
  await expect(page.locator('tbody tr.ant-table-row').first()).toContainText('Vendor/Large');
  await page.getByRole('columnheader', { name: '磁盘占用' }).click();
  await expect(page.locator('tbody tr.ant-table-row').first()).toContainText('Qwen/Delete-fixture');
  await page.getByRole('textbox', { name: '搜索本地模型' }).fill('large');
  await expect(page.locator('tbody tr.ant-table-row')).toHaveCount(1);
  await page.getByRole('button', { name: '刷新空间统计', exact: true }).click();
  expect(refreshed).toBe(1);
  await page.locator('summary').filter({ hasText: '扫描范围与其他模型目录' }).click();
  await page.getByLabel('添加当前节点上的模型目录').fill('/tmp/another');
  await page.getByRole('button', { name: '添加并扫描', exact: true }).click();
  await expect(page.getByText('/tmp/another', { exact: true })).toBeVisible();
});

test('temporary node removes actual files and deployment while retaining the run', async ({
  page,
}) => {
  const info = process.env.HQSB_DELETE_FIXTURE_INFO;
  test.skip(!info, 'Requires an isolated fixture node');
  const saved = JSON.parse(readFileSync(info!, 'utf8'));
  expect(existsSync(saved.row.snapshot_path)).toBe(true);
  await page.goto('/models/local?download=' + saved.row.id);
  await page.getByRole('button', { name: '删除本地制品', exact: true }).click();
  await expect(page.getByRole('dialog')).toContainText(saved.row.snapshot_path);
  await page.getByRole('dialog').getByRole('button', { name: '确认删除' }).click();
  await expect(page.getByRole('button', { name: '删除本地制品', exact: true })).toHaveCount(0);
  expect(existsSync(saved.row.snapshot_path)).toBe(false);
  const deployments = await (await page.request.get('/api/console/v1/deployments')).json();
  expect(deployments.items.some((r: { id: string }) => r.id === saved.row.deployment_id)).toBe(
    false,
  );
  const run = await (await page.request.get('/api/console/v1/runs/' + saved.run_id)).json();
  expect(run.output).toBe('retained fixture output');
  await page.reload();
  await expect(page.getByRole('button', { name: '删除本地制品', exact: true })).toHaveCount(0);
});
