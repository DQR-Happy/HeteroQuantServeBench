import { expect, test } from '@playwright/test';

const modelId = 'ms-6d9b6a4a3f944e041fd6';

async function modelState(
  page: import('@playwright/test').Page,
  overrides: Record<string, unknown>,
) {
  const response = await page.request.get('/api/console/v1/deployments');
  const body = await response.json();
  body.items = body.items.map((d: { id: string }) =>
    d.id === modelId ? { ...d, ...overrides } : d,
  );
  // A frozen read fixture avoids a poll's route.fetch outliving teardown.
  await page.route('**/api/console/v1/deployments', (route) => route.fulfill({ json: body }));
}

test('subpages keep model and run identity and mount only the selected workspace', async ({
  page,
}) => {
  const writes: string[] = [];
  page.on('request', (r) => {
    if (r.method() === 'POST') writes.push(r.url());
  });
  await page.goto(
    `/model-architecture?deployment=${modelId}&runtime_run=run_670b2a3e4d67413b819706bd6a25ec40&graph=onnx`,
  );
  await expect(page).toHaveURL(/\/model-architecture\/compiler\?/);
  await expect(page.getByRole('heading', { name: '计算图与 L3 执行', exact: true })).toBeVisible();
  await expect(page.getByRole('heading', { name: '通用模块与权重结构', exact: true })).toHaveCount(
    0,
  );
  await page.getByRole('link', { name: '模块与权重', exact: true }).click();
  await expect(page.getByRole('tree', { name: '通用模块树', exact: true })).toBeVisible();
  await expect(page.getByRole('heading', { name: '计算图与 L3 执行', exact: true })).toHaveCount(0);
  expect(new URL(page.url()).searchParams.get('runtime_run')).toBe(
    'run_670b2a3e4d67413b819706bd6a25ec40',
  );
  await page.screenshot({
    path: '../../reports/console/architecture-subpages/structure.png',
    fullPage: true,
  });
  await page.getByRole('link', { name: '来源与凭据', exact: true }).click();
  await expect(
    page.getByRole('heading', { name: '数据来源与采集范围', exact: true }),
  ).toBeVisible();
  await expect(page.getByRole('tree', { name: '通用模块树', exact: true })).toHaveCount(0);
  await page.getByRole('link', { name: '编译与导出', exact: true }).click();
  await expect(page.getByRole('combobox', { name: '计算图类型', exact: true })).toBeVisible();
  await page.setViewportSize({ width: 390, height: 844 });
  await expect
    .poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 2))
    .toBe(true);
  await page.screenshot({
    path: '../../reports/console/architecture-subpages/compiler-mobile.png',
    fullPage: true,
  });
  expect(writes).toEqual([]);
});

test('unloaded model explains disabled capture and links to its deployment without loading it', async ({
  page,
}) => {
  await modelState(page, {
    state: 'unloaded',
    epoch: null,
    kernel_profile: 'framework-native',
    quantization: { method: 'none' },
  });
  const writes: string[] = [];
  page.on('request', (r) => {
    if (r.method() === 'POST') writes.push(r.url());
  });
  await page.goto(`/model-architecture/compiler?deployment=${modelId}&graph=onnx`);
  const button = page.getByRole('button', { name: '采集 ONNX 导出诊断样例', exact: true });
  await expect(button).toBeDisabled();
  const trigger = page.locator('[data-disabled-action]').filter({ has: button });
  await trigger.hover();
  await expect(page.getByText(/当前模型尚未加载/)).toBeVisible();
  await trigger.focus();
  await expect(page.getByRole('link', { name: '加载或检查此模型', exact: false })).toBeVisible();
  await page.screenshot({
    path: '../../reports/console/architecture-subpages/disabled-unloaded.png',
  });
  await page.getByRole('link', { name: '加载或检查此模型', exact: false }).click();
  await expect(page).toHaveURL(new RegExp(`/devices#deployment-${modelId}$`));
  await expect(page.locator(`#deployment-${modelId}`)).toBeInViewport();
  expect(writes).toEqual([]);
});

test('quantized model offers keyboard accessible floating-point prerequisite with model identity', async ({
  page,
}) => {
  await modelState(page, {
    state: 'ready',
    epoch: 'read-only-fixture',
    kernel_profile: 'framework-native',
    quantization: { method: 'gptq', kv: { bits: 4 } },
  });
  const writes: string[] = [];
  page.on('request', (r) => {
    if (r.method() === 'POST') writes.push(r.url());
  });
  await page.goto(`/model-architecture/compiler?deployment=${modelId}&graph=inductor_post`);
  const button = page.getByRole('button', { name: '采集 Inductor IR 与代码诊断样例', exact: true });
  await expect(button).toBeDisabled();
  const trigger = page.locator('[data-disabled-action]').filter({ has: button });
  await trigger.focus();
  await expect(page.getByText(/编译或导出诊断需要原生浮点权重/)).toBeVisible();
  await trigger.press('Escape');
  await expect(
    page.getByRole('link', { name: '配置浮点权重与默认 KV', exact: false }),
  ).not.toBeVisible();
  await trigger.press('Tab');
  await trigger.focus();
  await page.getByRole('link', { name: '配置浮点权重与默认 KV', exact: false }).click();
  await expect(page).toHaveURL(new RegExp(`/precision\\?deployment=${modelId}$`));
  expect(writes).toEqual([]);
});

test('ATen diagnostic has its own page and unavailable action explanation', async ({ page }) => {
  await modelState(page, { state: 'unloaded', epoch: null });
  await page.goto(`/model-architecture?deployment=${modelId}&graph=aten`);
  await expect(page).toHaveURL(/\/model-architecture\/execution\?/);
  await expect(
    page.getByRole('heading', { name: '实际执行的算子依赖', exact: true }),
  ).toBeVisible();
  await expect(page.getByRole('heading', { name: '通用模块与权重结构', exact: true })).toHaveCount(
    0,
  );
  const button = page.getByRole('button', { name: '采集运行架构', exact: true });
  await expect(button).toBeDisabled();
  await page.locator('[data-disabled-action]').filter({ has: button }).hover();
  await expect(page.getByRole('link', { name: '加载或检查此模型', exact: false })).toBeVisible();
});
