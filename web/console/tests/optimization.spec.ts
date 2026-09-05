import { test, expect } from '@playwright/test';
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { enterWorkspace } from './session';

const tokenFile = process.env.HQSB_CONSOLE_TOKEN_FILE ?? '../../.console/connected/access-token';
const prefix = '/api/console/v1';
test.afterEach(async ({ page }, testInfo) => {
  testInfo.setTimeout(testInfo.timeout + 60000);
  // Keep the isolated target free even if a browser assertion fails mid-path.
  const response = await page.request.get(prefix + '/deployments');
  if (!response.ok()) return;
  const deployment = (await response.json()).items[0];
  if (deployment.state === 'unloaded') return;
  await page.goto('/devices');
  const unloaded = await page.request.post(prefix + `/deployments/${deployment.id}/unload`, {
    data: {},
    headers: { 'X-HQSB-Client': 'console', 'Idempotency-Key': crypto.randomUUID() },
  });
  expect(unloaded.ok()).toBeTruthy();
  await expect
    .poll(
      async () => (await (await page.request.get(prefix + '/deployments')).json()).items[0].state,
      { timeout: 30000 },
    )
    .toBe('unloaded');
});
test('real optimization plan, serial jobs, sealed quality, downloads and reconnect', async ({
  page,
}) => {
  test.setTimeout(300000);
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await enterWorkspace(page, '/optimization', tokenFile);
  await expect(page.getByRole('heading', { name: '优化实验', exact: true })).toBeVisible();
  await page.getByRole('button', { name: '冻结并注册计划' }).click();
  await expect(page).toHaveURL(/experiment=exp_/);
  const id = new URL(page.url()).searchParams.get('experiment')!;
  await expect(page.getByText('采用判定：hold', { exact: true })).toBeVisible();
  if (await page.getByRole('button', { name: '加载执行部署' }).isEnabled()) {
    await expect(page.getByRole('button', { name: '依次执行未提交任务' })).toBeDisabled();
    await page.getByRole('button', { name: '加载执行部署' }).click();
  }
  await expect(page.getByRole('button', { name: '依次执行未提交任务' })).toBeEnabled({
    timeout: 180000,
  });
  await page.getByRole('button', { name: '依次执行未提交任务' }).click();
  await expect(page.getByText('已完成', { exact: true })).toHaveCount(6, { timeout: 120000 });
  await expect(page.getByRole('button', { name: '依次执行未提交任务' })).toBeDisabled();
  const get = async (url: string) => {
    const response = await page.request.get(prefix + url);
    expect(response.ok()).toBeTruthy();
    return response.json();
  };
  const jobs = await get(`/experiments/${id}/jobs`);
  expect(jobs.items).toHaveLength(6);
  expect(jobs.items.every((item: { state: string }) => item.state === 'completed')).toBeTruthy();
  await page.getByRole('button', { name: '刷新报告', exact: true }).click();
  await expect(page.getByText('数值合同 passed', { exact: true })).toBeVisible();
  const report = await get(`/experiments/${id}/report`);
  expect(report.completeness).toBe('complete');
  expect(report.quality_status).toBe('passed');
  expect(report.decision).toBe('hold');
  expect(report.performance.status).not.toBe('passed');
  await page.screenshot({ path: '../../reports/connection/desktop-quality.png', fullPage: true });
  await page.getByRole('tab', { name: '机制变化', exact: true }).click();
  await page.getByLabel('筛选机制对照指标').fill('reserved_bytes');
  await expect(
    page.getByRole('cell', { name: 'reserved_bytes', exact: true }).first(),
  ).toBeVisible();
  await page.screenshot({ path: '../../reports/connection/desktop-mechanism.png', fullPage: true });
  await page.reload();
  await expect(page.getByText('已完成', { exact: true })).toHaveCount(6);
  expect((await get(`/experiments/${id}/jobs`)).items).toHaveLength(6);
  const quality = jobs.items.find(
    (item: { role: string; arm: string }) => item.role === 'quality' && item.arm === 'candidate',
  );
  await page.goto(`/runs/${quality.run_id}?tab=experiment`);
  await expect(page.getByText('执行凭据：applied', { exact: true })).toBeVisible();
  await expect(page.getByText('限定质量评测：complete', { exact: true })).toBeVisible();
  const bundle = await get(`/runs/${quality.run_id}/experiment`);
  const index = bundle.manifest.artifacts.findIndex(
    (item: { path: string }) => item.path === 'execution-receipt.json',
  );
  const ready = page.waitForEvent('download');
  await page
    .getByRole('row')
    .filter({ has: page.getByRole('cell', { name: 'execution-receipt.json', exact: true }) })
    .getByRole('link', { name: '下载原件', exact: true })
    .click();
  const download = await ready;
  const sha = createHash('sha256')
    .update(readFileSync((await download.path())!))
    .digest('hex');
  expect(sha).toBe(bundle.manifest.artifacts[index].sha256);
  await page.screenshot({ path: '../../reports/connection/receipt.png', fullPage: true });
  // A normal inference also forwards the selected cleanup policy.
  await page.goto('/playground');
  await page.getByLabel('请求后清理策略').press('Enter');
  await page.getByText('保留 allocator 空闲缓存', { exact: true }).click();
  await page.getByLabel('推理输入').fill('用一句话解释 KV Cache。');
  await page.getByLabel('最大输出 token').fill('2');
  const submitted = page.waitForResponse(
    (response) =>
      response.url().endsWith(prefix + '/requests') && response.request().method() === 'POST',
  );
  await page.getByRole('button', { name: /开始推理/ }).click();
  const normal = await (await submitted).json();
  await expect(page.getByText('已完成', { exact: true })).toBeVisible({ timeout: 120000 });
  expect(
    (await get(`/runs/${normal.id}`)).metrics.optimization_receipt.effective.cleanup_policy,
  ).toBe('preserve_allocator');
  await page.goto(`/optimization?experiment=${id}`);
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByRole('heading', { name: '优化实验', exact: true })).toBeVisible();
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
  ).toBeTruthy();
  await page.screenshot({ path: '../../reports/connection/mobile.png', fullPage: true });
  const deployment = (await get('/deployments')).items[0];
  const unloadResponse = await page.request.post(prefix + `/deployments/${deployment.id}/unload`, {
    data: {},
    headers: { 'X-HQSB-Client': 'console', 'Idempotency-Key': crypto.randomUUID() },
  });
  expect(unloadResponse.ok()).toBeTruthy();
  await expect
    .poll(async () => (await get('/deployments')).items[0].state, { timeout: 30000 })
    .toBe('unloaded');
  expect(errors).toEqual([]);
  writeFileSync(
    '../../reports/connection/live-ui.json',
    JSON.stringify(
      {
        status: 'PASS',
        experiment_id: id,
        jobs: jobs.items,
        report,
        receipt_sha256: sha,
        ordinary_request: normal.id,
        model_unloaded: true,
        browser_errors: errors,
      },
      null,
      2,
    ),
  );
});

test('explicit stop cancels an accepted job and does not submit later slots', async ({ page }) => {
  test.setTimeout(240000);
  await enterWorkspace(page, '/optimization', tokenFile);
  const editor = page.getByLabel('实验计划 JSON');
  const spec = JSON.parse(await editor.inputValue());
  spec.output_tokens = 32;
  await editor.fill(JSON.stringify(spec, null, 2));
  await page.getByRole('button', { name: '冻结并注册计划' }).click();
  await expect(page).toHaveURL(/experiment=exp_/);
  const id = new URL(page.url()).searchParams.get('experiment')!;
  if (await page.getByRole('button', { name: '加载执行部署' }).isEnabled())
    await page.getByRole('button', { name: '加载执行部署' }).click();
  await expect(page.getByRole('button', { name: '依次执行未提交任务' })).toBeEnabled({
    timeout: 180000,
  });
  await page.getByRole('button', { name: '依次执行未提交任务' }).click();
  // Stop while the first submission may still be waiting for its accepted ID.
  await page.getByRole('button', { name: '停止后续提交并取消当前任务' }).click();
  await expect(page.getByText('已取消', { exact: true })).toBeVisible({ timeout: 30000 });
  const jobs = await (await page.request.get(prefix + `/experiments/${id}/jobs`)).json();
  expect(jobs.items).toHaveLength(1);
  expect(jobs.items[0].state).toBe('cancelled');
  const bundle = await (
    await page.request.get(prefix + `/runs/${jobs.items[0].run_id}/experiment`)
  ).json();
  expect(bundle.manifest.status).toBe('partial');
  await page.reload();
  await expect(page.getByText('已取消', { exact: true })).toBeVisible();
  expect(
    (await (await page.request.get(prefix + `/experiments/${id}/jobs`)).json()).items,
  ).toHaveLength(1);
  writeFileSync(
    '../../reports/connection/cancel-ui.json',
    JSON.stringify(
      {
        status: 'PASS',
        experiment_id: id,
        run_id: jobs.items[0].run_id,
        capture_status: bundle.manifest.status,
        submitted_jobs: 1,
      },
      null,
      2,
    ),
  );
});

test('sealed report is usable after unload on desktop and mobile; invalid drafts stay editable', async ({
  page,
}) => {
  const fixture = JSON.parse(readFileSync('../../reports/connection/live-ui.json', 'utf8'));
  await enterWorkspace(page, `/optimization?experiment=${fixture.experiment_id}`, tokenFile);
  await expect(page.getByText('数值合同 passed', { exact: true })).toBeVisible();
  await expect(page.getByText('未加载', { exact: true })).toBeVisible();
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.getByRole('tab', { name: '性能与观测成本', exact: true }).click();
  await expect(page.getByText('insufficient_evidence', { exact: true })).toBeVisible();
  await page.screenshot({
    path: '../../reports/connection/desktop-performance.png',
    fullPage: true,
  });
  await page.getByRole('tab', { name: '质量对照', exact: true }).click();
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByText('数值合同 passed', { exact: true })).toBeVisible();
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
  ).toBeTruthy();
  await page.screenshot({ path: '../../reports/connection/mobile-complete.png', fullPage: true });
  await page.getByRole('button', { name: '新建计划', exact: true }).click();
  const editor = page.getByLabel('实验计划 JSON');
  const raw = JSON.parse(await editor.inputValue());
  raw.candidate.changes = { paged_kv: true };
  raw.declared_changes = ['paged_kv'];
  const text = JSON.stringify(raw, null, 2);
  await editor.fill(text);
  await page.getByRole('button', { name: '冻结并注册计划' }).focus();
  await page.keyboard.press('Enter');
  await expect(page.getByText('操作未完成', { exact: true })).toBeVisible();
  await expect(editor).toHaveValue(text);
  expect(new URL(page.url()).searchParams.has('experiment')).toBeFalsy();
  await page.screenshot({ path: '../../reports/connection/invalid-draft.png', fullPage: true });
});
