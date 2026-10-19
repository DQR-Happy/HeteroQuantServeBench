import { expect, test } from '@playwright/test';
const api = '/api/console/v1';

test('real heterogeneous DP comparison labels different timing boundaries', async ({ page }) => {
  const rows = (await (await page.request.get(api + '/parallel/jobs')).json()).items;
  const peer = rows.find(
    (r: any) => r.state === 'completed' && r.metrics.parallel_receipt?.plan.strategy === 'dp_hosts',
  );
  expect(peer).toBeTruthy();
  const p = peer.metrics.parallel_receipt;
  const base = rows.find(
    (r: any) =>
      r.state === 'completed' &&
      r.metrics.parallel_receipt?.plan.strategy === 'single' &&
      r.metrics.parallel_receipt.source_digest === p.source_digest &&
      r.metrics.parallel_receipt.model_identity === p.model_identity &&
      r.metrics.output_tokens === peer.metrics.output_tokens,
  );
  expect(base).toBeTruthy();
  await page.goto('/parallel/compare?deployment=' + peer.config.deployment.id);
  await choice(page, 'L9 基线', base.id + ' · single');
  await choice(page, 'L9 候选', peer.id + ' · dp_hosts');
  await page.getByRole('button', { name: '比较 L9 结果', exact: true }).click();
  await expect(page.getByText('来源、负载与逐 Token 核对通过', { exact: true })).toBeVisible();
  await expect(page.getByText('不同：仅展示，不能求加速比', { exact: true })).toHaveCount(2);
  await page.screenshot({
    path: '../../reports/console/l9-parallel/peer-compare-desktop.png',
    fullPage: true,
    animations: 'disabled',
  });
});
async function wait(request: import('@playwright/test').APIRequestContext, id: string) {
  let row: any;
  await expect
    .poll(
      async () => {
        const r = await request.get(api + '/runs/' + id);
        expect(r.ok()).toBeTruthy();
        row = await r.json();
        return ['completed', 'failed', 'cancelled', 'timed_out'].includes(row.state);
      },
      { timeout: 150000 },
    )
    .toBe(true);
  expect(row.state, row.error ?? '').toBe('completed');
  expect(row.cleanup).toBe('succeeded');
  return row;
}
async function choice(page: import('@playwright/test').Page, label: string, text: string) {
  const input = page.getByRole('combobox', { name: label, exact: true });
  await input.focus();
  await input.press('ArrowDown');
  await page.getByText(text, { exact: true }).last().click();
  await page.keyboard.press('Escape');
}

test('L9 resident prerequisite is explained on hover and keyboard focus', async ({ page }) => {
  const d = await (await page.request.get(api + '/deployments')).json();
  const target = d.items.find((d: any) => d.state === 'ready');
  expect(target).toBeTruthy();
  await page.goto('/parallel/run?deployment=' + target.id);
  const button = page.getByRole('button', { name: '运行 L9 方案', exact: true });
  await expect(button).toBeDisabled();
  const trigger = page.locator('[data-disabled-action]').filter({ has: button });
  await trigger.hover();
  await trigger.focus();
  await expect(
    page
      .getByText('L9 独立进程需要拥有设备。请先在设备与部署页卸载当前驻留模型。', { exact: true })
      .last(),
  ).toBeVisible();
  await page.getByRole('link', { name: '卸载驻留模型 →', exact: true }).click();
  await expect(page).toHaveURL(new RegExp('/devices#deployment-' + target.id + '$'));
});

test('one GPU never enables a two GPU plan; subpages and mobile identity', async ({ page }) => {
  const d = await (await page.request.get(api + '/deployments')).json();
  const id = d.items[0].id;
  await page.route('**/api/console/v1/deployments', (r) =>
    r.fulfill({
      json: { items: d.items.map((d: any) => ({ ...d, state: 'unloaded', epoch: null })) },
    }),
  );
  await page.goto('/parallel/run?deployment=' + id);
  await choice(page, 'L9 并行策略', 'TP / Transformers 原生权重分片');
  const button = page.getByRole('button', { name: '运行 L9 方案', exact: true });
  await expect(button).toBeDisabled();
  await page.locator('[data-disabled-action]').filter({ has: button }).focus();
  await expect(page.getByText(/当前节点可见 1 张 GPU/).last()).toBeVisible();
  await page.getByRole('link', { name: '设备与通信', exact: true }).click();
  await expect(page.getByRole('heading', { name: '当前节点真实设备', exact: true })).toBeVisible();
  await page.screenshot({
    path: '../../reports/console/l9-parallel/topology-desktop.png',
    fullPage: true,
  });
  await page.getByRole('link', { name: '跨节点 DP', exact: true }).click();
  expect(new URL(page.url()).searchParams.get('deployment')).toBe(id);
  await page.setViewportSize({ width: 390, height: 844 });
  await expect
    .poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 2))
    .toBe(true);
  await page.screenshot({
    path: '../../reports/console/l9-parallel/peers-mobile.png',
    fullPage: true,
  });
});

test('real CUDA baseline, sealed evidence, comparison and old model restore', async ({ page }) => {
  test.setTimeout(240000);
  page.setDefaultTimeout(25000);
  const overview = await (await page.request.get(api + '/overview')).json();
  expect(overview.pending).toBe(0);
  const target = overview.deployments.find(
    (d: any) => d.state === 'ready' && d.provider === 'pytorch',
  );
  expect(target.quantization.method).toBe('none');
  let unloaded = false;
  async function post(path: string, body: unknown) {
    const response = await page.request.post(api + path, {
      data: body,
      headers: { 'X-HQSB-Client': 'console', 'Idempotency-Key': crypto.randomUUID() },
    });
    expect(response.ok(), await response.text()).toBeTruthy();
    return response.json();
  }
  try {
    await wait(page.request, (await post('/deployments/' + target.id + '/unload', {})).id);
    unloaded = true;
    await page.goto('/parallel/run?deployment=' + target.id);
    const button = page.getByRole('button', { name: '运行 L9 方案', exact: true });
    await expect(button).toBeEnabled();
    await button.click();
    await expect(page).toHaveURL(/run=run_/);
    const baseline = new URL(page.url()).searchParams.get('run')!;
    await wait(page.request, baseline);
    await expect(button).toBeEnabled();
    await button.click();
    await expect.poll(() => new URL(page.url()).searchParams.get('run')).not.toBe(baseline);
    const candidate = new URL(page.url()).searchParams.get('run')!;
    const record = await wait(page.request, candidate);
    expect(
      record.metrics.parallel_receipt.ranks[0].parameters.local_parameter_bytes,
    ).toBeGreaterThan(0);
    await expect(
      page.getByRole('heading', { name: 'L9 实际执行与资源凭据', exact: true }),
    ).toBeVisible();
    await page.screenshot({
      path: '../../reports/console/l9-parallel/run-desktop.png',
      fullPage: true,
      animations: 'disabled',
    });
    await page.getByRole('link', { name: '结果对照', exact: true }).click();
    await choice(page, 'L9 基线', baseline + ' · single');
    await choice(page, 'L9 候选', candidate + ' · single');
    await page.getByRole('button', { name: '比较 L9 结果', exact: true }).click();
    await expect(page.getByText('来源、负载与逐 Token 核对通过', { exact: true })).toBeVisible();
    await page.screenshot({
      path: '../../reports/console/l9-parallel/compare-desktop.png',
      fullPage: true,
      animations: 'disabled',
    });
    await page.goto('/runs/' + candidate);
    await expect(page).toHaveURL(
      new RegExp('/parallel/run\\?deployment=' + target.id + '&run=' + candidate),
    );
  } finally {
    if (unloaded) {
      const run = await post('/deployments/' + target.id + '/load', {
        precision: target.precision,
        kernel_profile: target.kernel_profile,
        quantization: { method: 'none' },
      });
      await wait(page.request, run.id);
    }
  }
});
