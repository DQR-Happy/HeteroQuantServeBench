import { expect, test } from '@playwright/test';
const base = '/api/console/v1';

test('disabled L6 execution explains prerequisite by hover/focus and links to this model', async ({
  page,
}) => {
  const response = await page.request.get(base + '/deployments');
  const doc = await response.json();
  doc.items = doc.items.map((d: { id: string }) =>
    d.id === 'gemma3-1b' ? { ...d, state: 'unloaded', epoch: null } : d,
  );
  await page.route('**/api/console/v1/deployments', (route) => route.fulfill({ json: doc }));
  const writes: string[] = [];
  page.on('request', (r) => {
    if (r.method() === 'POST') writes.push(r.url());
  });
  await page.goto('/memory-cache/configure?deployment=gemma3-1b');
  const button = page.getByRole('button', { name: '运行 L6 方案', exact: true });
  await expect(button).toBeDisabled();
  const trigger = page.locator('[data-disabled-action]').filter({ has: button });
  await trigger.hover();
  await expect(page.getByText('所选模型尚未就绪，请先在设备与部署页加载模型。')).toBeVisible();
  await trigger.focus();
  await expect(page.getByRole('link', { name: '加载模型 →', exact: true })).toBeVisible();
  await page.getByRole('link', { name: '加载模型 →', exact: true }).click();
  await expect(page).toHaveURL(/\/devices#deployment-gemma3-1b$/);
  expect(writes).toEqual([]);
});

test('real sealed L6 records compare without executing a model; responsive subpages retain identity', async ({
  page,
}) => {
  const id = 'ms-6d9b6a4a3f944e041fd6';
  const response = await page.request.get(base + '/memory/runs?deployment_id=' + id);
  const { items } = await response.json();
  const candidate = items.find(
    (r: any) => r.state === 'completed' && r.config.memory_plan.mode === 'static',
  );
  test.skip(!candidate, 'Run the target L6 acceptance first');
  const baseline = items.find(
    (r: any) =>
      r.state === 'completed' &&
      r.config.memory_plan.mode === 'dynamic' &&
      r.config.epoch === candidate.config.epoch,
  );
  expect(baseline).toBeTruthy();
  await page.goto(`/memory-cache/records?deployment=${id}&run=${candidate.id}`);
  await expect(page.getByRole('heading', { name: '实际执行与存储', exact: true })).toBeVisible();
  await expect(page.getByRole('heading', { name: '缓存策略', exact: true })).toHaveCount(0);
  for (const [label, run] of [
    ['L6 基线', baseline],
    ['L6 候选', candidate],
  ] as const) {
    await page.getByRole('combobox', { name: label, exact: true }).click();
    await page
      .getByText(
        `${run.config.memory_plan.mode} · ${run.metrics.memory_receipt.prefix.lookup} · ${run.id}`,
        { exact: true },
      )
      .last()
      .click();
  }
  await page.getByRole('button', { name: '核对可比性与差异', exact: true }).click();
  await expect(page.getByText('同输入生成 Token 完全一致', { exact: true })).toBeVisible();
  await expect(page.getByRole('columnheader', { name: '候选 − 基线', exact: true })).toBeVisible();
  await page.screenshot({
    path: '../../reports/console/l6-memory/records-desktop.png',
    fullPage: true,
  });
  await page.getByRole('link', { name: '配置与试运行', exact: true }).click();
  await expect(page.getByRole('heading', { name: '缓存策略', exact: true })).toBeVisible();
  expect(new URL(page.url()).searchParams.get('run')).toBe(candidate.id);
  await expect(page.getByRole('heading', { name: '同输入运行对照', exact: true })).toHaveCount(0);
  await page.setViewportSize({ width: 390, height: 844 });
  await expect
    .poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 2))
    .toBe(true);
  await page.screenshot({
    path: '../../reports/console/l6-memory/configure-mobile.png',
    fullPage: true,
  });
});

test('production frontend submits an L6 request and shows applied real-model receipt', async ({
  page,
}) => {
  const overview = await (await page.request.get(base + '/overview')).json();
  test.skip(overview.pending !== 0, 'Target has an active task');
  const { items } = await (await page.request.get(base + '/deployments')).json();
  const current = items.find((d: any) => d.id === 'gemma3-1b' && d.state === 'ready');
  test.skip(!current, 'Restore the original Gemma3 deployment first');
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/memory-cache/configure?deployment=gemma3-1b');
  await page.getByRole('button', { name: '检查缓存能力', exact: true }).click();
  await expect(page.getByRole('button', { name: '运行 L6 方案', exact: true })).toBeEnabled();
  await page.getByRole('button', { name: '运行 L6 方案', exact: true }).click();
  await expect(page).toHaveURL(/run=run_/);
  await expect(page.getByRole('heading', { name: '实际执行与存储', exact: true })).toBeVisible({
    timeout: 60000,
  });
  await expect(page.getByText('applied', { exact: false }).first()).toBeVisible();
  const identity = new URL(page.url()).searchParams.get('run');
  const receipt = await (await page.request.get(`${base}/runs/${identity}/memory-cache`)).json();
  expect(receipt.status).toBe('applied');
  expect(receipt.forward_calls).toBe(32);
  expect(receipt.storage_before_release.bytes_by_device.cuda).toBeGreaterThan(0);
  await page.screenshot({
    path: '../../reports/console/l6-memory/configure-desktop.png',
    fullPage: true,
  });
  expect(errors).toEqual([]);
});

test('cancel a live L6 request, publish cleanup and allow the next submission', async ({
  page,
}) => {
  const overview = await (await page.request.get(base + '/overview')).json();
  test.skip(overview.pending !== 0, 'Target has an active task');
  const { items } = await (await page.request.get(base + '/deployments')).json();
  test.skip(
    !items.some((d: any) => d.id === 'gemma3-1b' && d.state === 'ready'),
    'Restore Gemma3 first',
  );
  await page.goto('/memory-cache/configure?deployment=gemma3-1b');
  await page.getByRole('button', { name: '检查缓存能力', exact: true }).click();
  await page.getByRole('combobox', { name: 'L6 输出 Token', exact: true }).focus();
  await page.getByRole('combobox', { name: 'L6 输出 Token', exact: true }).press('ArrowDown');
  await page.getByText('128', { exact: true }).last().click();
  await page.getByRole('button', { name: '运行 L6 方案', exact: true }).click();
  await expect(page).toHaveURL(/run=run_/);
  const id = new URL(page.url()).searchParams.get('run');
  await expect
    .poll(async () => {
      const run = await (await page.request.get(base + '/runs/' + id)).json();
      return run.output.length;
    })
    .toBeGreaterThan(0);
  await page.getByRole('button', { name: '取消当前运行', exact: true }).click();
  await expect
    .poll(async () => {
      const run = await (await page.request.get(base + '/runs/' + id)).json();
      return run.state;
    })
    .toBe('cancelled');
  const receipt = await (await page.request.get(`${base}/runs/${id}/memory-cache`)).json();
  expect(receipt.status).toBe('not_fully_applied');
  expect(receipt.prefix.snapshot_copy_bytes).toBe(0);
  await expect(page.getByRole('button', { name: '运行 L6 方案', exact: true })).toBeEnabled();
});
