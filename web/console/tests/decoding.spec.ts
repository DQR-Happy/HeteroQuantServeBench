import { expect, test } from '@playwright/test';
const api = '/api/console/v1';
async function option(
  page: import('@playwright/test').Page,
  label: string,
  text: string,
  search = false,
) {
  const input = page.getByRole('combobox', { name: label, exact: true });
  await input.focus();
  await input.press('ArrowDown');
  if (search) await input.fill(text);
  await page.getByText(text, { exact: true }).last().click();
  await page.keyboard.press('Escape');
}
async function wait(request: import('@playwright/test').APIRequestContext, id: string) {
  let row: any;
  await expect
    .poll(
      async () => {
        const r = await request.get(api + '/runs/' + id);
        expect(r.ok()).toBeTruthy();
        row = await r.json();
        return ['completed', 'failed', 'cancelled', 'timed_out', 'interrupted'].includes(row.state);
      },
      { timeout: 120000 },
    )
    .toBe(true);
  expect(row.state, row.error ?? '').toBe('completed');
  return row;
}

test('unloaded L7 target explains prerequisite on hover and keyboard focus', async ({ page }) => {
  const d = await (await page.request.get(api + '/deployments')).json();
  const id = d.items[0].id;
  d.items = d.items.map((r: any) => ({ ...r, state: 'unloaded', epoch: null }));
  await page.route('**/api/console/v1/deployments', (r) => r.fulfill({ json: d }));
  const generation: string[] = [];
  page.on('request', (r) => {
    if (r.method() === 'POST' && r.url().endsWith('/requests')) generation.push(r.url());
  });
  await page.goto('/decoding/run?deployment=' + id);
  const button = page.getByRole('button', { name: '运行 L7 解码', exact: true });
  await expect(button).toBeDisabled();
  const trigger = page.locator('[data-disabled-action]').filter({ has: button });
  await trigger.hover();
  await trigger.focus();
  await expect(page.getByText('请先加载所选目标模型。', { exact: true }).last()).toBeVisible();
  await page.getByRole('link', { name: '加载模型 →', exact: true }).click();
  await expect(page).toHaveURL(new RegExp('/devices#deployment-' + id + '$'));
  expect(generation).toEqual([]);
});

test('L7 task subpages retain model identity and fit a small screen', async ({ page }) => {
  const ds = await (await page.request.get(api + '/deployments')).json();
  const id = ds.items.find((d: any) => d.state === 'ready').id;
  await page.goto('/decoding/run?deployment=' + id);
  await page.getByRole('link', { name: '草稿模型与资源', exact: true }).click();
  await expect(
    page.getByRole('heading', { name: '独立草稿模型的驻留与释放', exact: true }),
  ).toBeVisible();
  await expect(page.getByRole('heading', { name: '生成策略', exact: true })).toHaveCount(0);
  expect(new URL(page.url()).searchParams.get('deployment')).toBe(id);
  await page.getByRole('link', { name: '结果对照', exact: true }).click();
  await page.setViewportSize({ width: 390, height: 844 });
  await expect
    .poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 2))
    .toBe(true);
  await page.screenshot({
    path: '../../reports/console/l7-decoding/compare-mobile.png',
    fullPage: true,
  });
});

test('real L7 greedy and lookup stream, seal receipts, and compare in the UI', async ({ page }) => {
  test.setTimeout(150000);
  page.setDefaultTimeout(20000);
  const overview = await (await page.request.get(api + '/overview')).json();
  expect(overview.pending).toBe(0);
  const target = overview.deployments.find(
    (d: any) => d.state === 'ready' && d.provider === 'pytorch',
  );
  expect(target?.structure_artifact_id ?? null).toBeNull();
  expect(target.quantization.method).toBe('none');
  await page.goto('/decoding/run?deployment=' + target.id);
  const submit = page.getByRole('button', { name: '运行 L7 解码', exact: true });
  await expect(submit).toBeEnabled();
  await submit.click();
  await expect(page).toHaveURL(/run=run_/);
  const baseline = new URL(page.url()).searchParams.get('run')!;
  await wait(page.request, baseline);
  await expect(submit).toBeEnabled();
  await option(page, 'L7 策略', '提示词查找推测 / 贪心验证');
  await submit.click();
  await expect.poll(() => new URL(page.url()).searchParams.get('run')).not.toBe(baseline);
  const candidate = new URL(page.url()).searchParams.get('run')!;
  const result = await wait(page.request, candidate);
  expect(result.metrics.decoding_receipt.status).toBe('invoked');
  await expect(
    page.getByRole('heading', { name: 'L7 实际解码与验证凭据', exact: true }),
  ).toBeVisible();
  await page.screenshot({
    path: '../../reports/console/l7-decoding/run-desktop.png',
    fullPage: true,
    animations: 'disabled',
  });
  await page.getByRole('link', { name: '结果对照', exact: true }).click();
  await option(page, 'L7 基线', `autoregressive / greedy · ${baseline}`, true);
  await option(page, 'L7 候选', `prompt_lookup / greedy · ${candidate}`, true);
  await page.getByRole('button', { name: '比较 L7 结果', exact: true }).click();
  await expect(page.getByText('来源与负载核对通过', { exact: true })).toBeVisible();
  await expect(page.getByText(/greedy_exact_on_this_input/).first()).toBeVisible();
  await page.screenshot({
    path: '../../reports/console/l7-decoding/compare-desktop.png',
    fullPage: true,
    animations: 'disabled',
  });
  await page.goto('/runs/' + candidate + '?tab=decoding');
  await expect(
    page.getByRole('heading', { name: 'L7 实际解码与验证凭据', exact: true }),
  ).toBeVisible();
  expect(
    (await (await page.request.get(api + '/deployments')).json()).items.find(
      (d: any) => d.id === target.id,
    ).epoch,
  ).toBe(target.epoch);
});
