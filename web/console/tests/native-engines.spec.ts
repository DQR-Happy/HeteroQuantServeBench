import { test, expect } from '@playwright/test';

test('native tuning exposes controls, workload and evidence without login', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/engines/benchmark');
  await expect(page.getByRole('heading', { name: '原生推理引擎', exact: true })).toBeVisible();
  await expect(page.getByRole('textbox', { name: '固定输入', exact: true })).toHaveValue(
    /Compute 17 \+ 25/,
  );
  await expect(page.getByRole('button', { name: '运行公共负载', exact: true })).toBeDisabled();
  await expect(page.getByLabel('访问令牌', { exact: true })).toHaveCount(0);
  const catalog = await page.request.get('/api/console/v1/optimization-catalog?provider=vllm');
  expect(catalog.ok()).toBeTruthy();
  const data = await catalog.json();
  expect(data.engine).toBe('vllm');
  expect(data.items.find((row: { id: string }) => row.id === 'prefix_cache').layers).toContain(6);
  await page.getByRole('link', { name: '部署与参数', exact: true }).click();
  await expect(page.getByRole('textbox', { name: '固定输入', exact: true })).toHaveCount(0);
  const field = page.getByRole('spinbutton', { name: 'memory_fraction', exact: true });
  if (await field.count()) {
    await field.fill('0.55');
    await page.getByRole('link', { name: '公共压测', exact: true }).click();
    await expect(page.getByRole('textbox', { name: '固定输入', exact: true })).toBeVisible();
    await page.getByRole('link', { name: '部署与参数', exact: true }).click();
    await expect(field).toHaveValue('0.55');
  }
  await page.screenshot({
    path: '../../reports/console/native-engines/workspace.png',
    fullPage: true,
  });
  expect(errors).toEqual([]);
});

test('declared configuration change keeps a saved quality regression visible', async ({ page }) => {
  const baseline = process.env.HQSB_NATIVE_BASELINE_RUN;
  const candidate = process.env.HQSB_NATIVE_CANDIDATE_RUN;
  test.skip(
    !baseline || !candidate,
    'Requires saved native baseline and rejected candidate fixtures',
  );
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto(`/engines?run=${candidate}`);
  await expect(page.getByRole('link', { name: '运行与对照', exact: true })).toHaveAttribute(
    'aria-current',
    'page',
  );
  await page.getByRole('link', { name: '运行与对照', exact: true }).click();
  await expect(page).toHaveURL(/\/engines\/records\?/);
  async function choose(name: string, value: string) {
    await page.getByRole('combobox', { name, exact: true }).click();
    await page.getByRole('combobox', { name, exact: true }).fill(value);
    await page
      .locator('.ant-select-dropdown:visible .ant-select-item-option-content')
      .filter({ hasText: value })
      .click();
  }
  await choose('基线运行', baseline!);
  await choose('候选运行', candidate!);
  await choose('声明的配置变化', '前缀缓存');
  const pending = page.waitForResponse(
    (response) =>
      response.url().endsWith('/native-engines/comparisons') &&
      response.request().method() === 'POST',
  );
  await page.getByRole('button', { name: '检查可比性与差异', exact: true }).click();
  const report = await (await pending).json();
  expect(report.comparable).toBe(true);
  expect(report.quality_gate.status).toBe('failed');
  expect(report.eligible_for_adoption).toBe(false);
  const alert = page.getByText('该候选的固定样例质量发生回退，保留性能和机制差异供分析。', {
    exact: true,
  });
  await expect(alert).toBeVisible();
  await page.keyboard.press('Escape');
  await alert.scrollIntoViewIfNeeded();
  await page.screenshot({ path: '../../reports/console/native-engines/quality-gate-ui.png' });
  expect(errors).toEqual([]);
});
