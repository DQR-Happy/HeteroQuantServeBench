import { expect, test } from '@playwright/test';

test('L3 submission uses the selected strategy and the node deadline without reloading weights', async ({
  page,
  request,
}) => {
  const deployments = (await (await request.get('/api/console/v1/deployments')).json()).items;
  const model = deployments.find((d: { id: string }) => d.id === 'qwen-reference');
  test.skip(!model, 'requires the Qwen artifact');
  await page.route('**/api/console/v1/deployments', async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    body.items = body.items.map((d: { id: string }) =>
      d.id === model.id
        ? {
            ...d,
            state: 'ready',
            epoch: 'test-epoch',
            kernel_profile: 'framework-native',
            quantization: { method: 'none' },
          }
        : d,
    );
    await route.fulfill({ response, json: body });
  });
  await page.route('**/api/console/v1/runtime/catalog', (route) =>
    route.fulfill({ json: { limits: { max_deadline_ms: 42000 } } }),
  );
  const submissions: Record<string, unknown>[] = [];
  const lifecycles: string[] = [];
  page.on('request', (r) => {
    if (r.method() === 'POST' && /\/deployments\/.*\/(load|unload)$/.test(r.url()))
      lifecycles.push(r.url());
  });
  await page.route('**/api/console/v1/requests', async (route) => {
    submissions.push(route.request().postDataJSON());
    await route.fulfill({
      status: 422,
      json: { detail: 'test: execution intercepted before GPU work' },
    });
  });
  await page.goto(`/model-architecture?deployment=${model.id}&graph=fx`);
  await page.getByRole('combobox', { name: 'L3执行策略', exact: true }).press('ArrowDown');
  await page.getByText('Inductor 编译优化', { exact: true }).last().click();
  await page.getByRole('button', { name: '运行固定样例', exact: true }).click();
  await expect(
    page.getByText('test: execution intercepted before GPU work', { exact: true }),
  ).toBeVisible();
  expect(submissions).toHaveLength(1);
  expect(submissions[0].runtime_plan).toEqual({
    mode: 'inductor',
    shape_policy: 'auto',
    fullgraph: false,
  });
  expect(submissions[0].deadline_ms).toBe(42000);
  expect(submissions[0].observation_mode).toBe('off');
  expect(lifecycles).toEqual([]);
});

// Read-only target acceptance: never submits a model operation or inference.
test('L3 history joins compiler nodes to the existing architecture and run evidence', async ({
  page,
  request,
}) => {
  const deployments = (await (await request.get('/api/console/v1/deployments')).json()).items;
  let chosen: { deployment: string; run: string } | undefined;
  for (const d of deployments) {
    const response = await request.get(`/api/console/v1/runtime/runs?deployment_id=${d.id}`);
    if (!response.ok()) continue;
    const runs = (await response.json()).items;
    const candidate = runs.find(
      (r: { state: string; plan: { mode: string }; role: string }) =>
        r.state === 'completed' && r.plan.mode === 'inductor' && r.role === 'generation',
    );
    if (candidate) {
      chosen = { deployment: d.id, run: candidate.run_id };
      break;
    }
  }
  test.skip(!chosen, 'requires a sealed real-model Inductor run from validate_l3.py');
  const errors: string[] = [];
  const writes: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('request', (r) => {
    if (r.method() === 'POST' && !r.url().includes('/session')) writes.push(r.url());
  });
  await page.goto(
    `/model-architecture?deployment=${chosen!.deployment}&runtime_run=${chosen!.run}`,
  );
  await expect(page.getByRole('heading', { name: '计算图与 L3 执行', exact: true })).toBeVisible();
  await expect(page.getByText('inductor · invoked', { exact: true })).toBeVisible();
  await page.getByLabel('筛选编译节点', { exact: true }).fill('q_proj');
  const runtime = page.locator('.runtime-explorer');
  await expect(runtime.getByRole('cell', { name: /self_attn.q_proj/ }).first()).toBeVisible();
  await runtime
    .getByRole('cell', { name: /self_attn.q_proj/ })
    .first()
    .click();
  await page.getByText('所选编译节点、源码和关联权重', { exact: true }).click();
  await expect(runtime.getByText(/q_proj.weight/).first()).toBeVisible();
  await runtime.screenshot({ path: '../../reports/console/l3-runtime/runtime-panel.png' });
  await page.screenshot({
    path: '../../reports/console/l3-runtime/architecture-compiled.png',
    fullPage: true,
  });
  await page.goto(`/runs/${chosen!.run}`);
  await expect(page.getByRole('button', { name: '计算图与 L3 分析', exact: true })).toBeVisible();
  await page.getByRole('tab', { name: '优化与质量凭据', exact: true }).click();
  await expect(page.getByText('L3 实际执行与编译凭据', { exact: true })).toBeVisible();
  expect(errors).toEqual([]);
  expect(writes).toEqual([]);
});
