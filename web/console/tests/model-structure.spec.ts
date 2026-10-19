import { expect, test } from '@playwright/test';
const base = '/api/console/v1';
const tiny = 'ms-586060631c81c17023c0';
async function post(
  request: import('@playwright/test').APIRequestContext,
  path: string,
  body: unknown,
) {
  const r = await request.post(base + path, {
    data: body,
    headers: { 'X-HQSB-Client': 'console', 'Idempotency-Key': crypto.randomUUID() },
  });
  expect(r.ok(), await r.text()).toBeTruthy();
  return r.json();
}
async function wait(request: import('@playwright/test').APIRequestContext, id: string) {
  let run: any;
  await expect
    .poll(
      async () => {
        const response = await request.get(base + '/runs/' + id);
        expect(response.ok(), `Cannot read run ${id}`).toBeTruthy();
        run = await response.json();
        return ['completed', 'failed', 'cancelled', 'timed_out', 'interrupted'].includes(run.state);
      },
      { timeout: 180000 },
    )
    .toBe(true);
  expect(run.state, run.error ?? '').toBe('completed');
  return run;
}
async function option(page: import('@playwright/test').Page, name: string, value: string) {
  const input = page.getByRole('combobox', { name, exact: true });
  await input.focus();
  await input.press('ArrowDown');
  await input.fill(value);
  await page.getByText(value, { exact: true }).last().click();
}

test('L5 unloaded prerequisite is accessible and links to the selected deployment', async ({
  page,
}) => {
  const d = await (await page.request.get(base + '/deployments')).json();
  d.items = d.items.map((r: any) => (r.id === tiny ? { ...r, state: 'unloaded', epoch: null } : r));
  await page.route('**/api/console/v1/deployments', (route) => route.fulfill({ json: d }));
  const writes: string[] = [];
  page.on('request', (r) => {
    if (r.method() === 'POST') writes.push(r.url());
  });
  await page.goto('/model-structure/build?deployment=' + tiny);
  const button = page.getByRole('button', { name: '创建独立压缩制品', exact: true });
  await expect(button).toBeDisabled();
  const trigger = page.locator('[data-disabled-action]').filter({ has: button });
  await trigger.hover();
  await trigger.focus();
  await expect(page.getByText('请先加载所选模型。', { exact: true }).last()).toBeVisible();
  await page.getByRole('link', { name: '加载模型 →', exact: true }).click();
  await expect(page).toHaveURL(new RegExp('/devices#deployment-' + tiny + '$'));
  expect(writes).toEqual([]);
});

test('three independent L5 subpages retain model/artifact identity and work on mobile', async ({
  page,
}) => {
  const { items } = await (
    await page.request.get(base + '/structure/artifacts?deployment_id=' + tiny)
  ).json();
  const artifact = items.find((a: any) => a.recipe.method === 'mlp_channels');
  test.skip(!artifact, 'Complete target L5 acceptance first');
  const writes: string[] = [];
  page.on('request', (r) => {
    if (r.method() === 'POST') writes.push(r.url());
  });
  await page.goto(`/model-structure/artifacts?deployment=${tiny}&artifact=${artifact.artifact_id}`);
  await expect(
    page.getByRole('heading', { name: '独立保存的结构制品', exact: true }),
  ).toBeVisible();
  await expect(page.getByRole('heading', { name: '变换方法与范围', exact: true })).toHaveCount(0);
  await page.getByRole('link', { name: '评测与对照', exact: true }).click();
  expect(new URL(page.url()).searchParams.get('artifact')).toBe(artifact.artifact_id);
  await expect(
    page.getByRole('heading', { name: '原始基线与结构候选', exact: true }),
  ).toBeVisible();
  await expect(page.getByRole('heading', { name: '独立保存的结构制品', exact: true })).toHaveCount(
    0,
  );
  await page.setViewportSize({ width: 390, height: 844 });
  await expect
    .poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 2))
    .toBe(true);
  await page.screenshot({
    path: '../../reports/console/l5-structure/evaluation-mobile.png',
    fullPage: true,
  });
  await page.getByRole('link', { name: '创建压缩制品', exact: true }).click();
  await expect(page.getByRole('heading', { name: '变换方法与范围', exact: true })).toBeVisible();
  expect(writes).toEqual([]);
});

test('default model identity remains pinned when another deployment becomes ready', async ({
  page,
}) => {
  const d = await (await page.request.get(base + '/deployments')).json();
  const selected = d.items.find((r: any) => r.state === 'ready') ?? d.items[0];
  const alternate = d.items.find((r: any) => r.id !== selected.id);
  await page.goto('/model-structure');
  await expect.poll(() => new URL(page.url()).searchParams.get('deployment')).toBe(selected.id);
  await page.getByRole('link', { name: '制品与加载', exact: true }).click();
  expect(new URL(page.url()).searchParams.get('deployment')).toBe(selected.id);
  d.items = d.items.map((r: any) => ({
    ...r,
    state: r.id === alternate.id ? 'ready' : 'unloaded',
    epoch: null,
  }));
  await page.route('**/api/console/v1/deployments', (route) => route.fulfill({ json: d }));
  await page.reload();
  await page.getByRole('link', { name: '创建压缩制品', exact: true }).click();
  await expect(page.getByRole('button', { name: '检查可变换模块', exact: true })).toBeDisabled();
  expect(new URL(page.url()).searchParams.get('deployment')).toBe(selected.id);
});

test('production UI creates compact weights, loads them, scores PPL and compares original baseline', async ({
  page,
}) => {
  test.setTimeout(240000);
  page.setDefaultTimeout(20000);
  const overview = await (await page.request.get(base + '/overview')).json();
  test.skip(overview.pending !== 0, 'Another target task is active');
  const { items } = await (await page.request.get(base + '/deployments')).json();
  const original = items.find((d: any) => d.state === 'ready' && d.provider === 'pytorch');
  expect(original).toBeTruthy();
  const restore = async () => {
    const { items } = await (await page.request.get(base + '/deployments')).json();
    for (const d of items.filter((d: any) => d.state === 'ready' && d.provider === 'pytorch'))
      await wait(page.request, (await post(page.request, `/deployments/${d.id}/unload`, {})).id);
    const recipe = { ...original.detail.precision_receipt.resolved_recipe };
    delete recipe.calibration;
    if (original.quantization?.artifact_id) recipe.artifact_id = original.quantization.artifact_id;
    await wait(
      page.request,
      (
        await post(page.request, `/deployments/${original.id}/load`, {
          precision: original.precision,
          kernel_profile: original.kernel_profile ?? 'framework-native',
          quantization: recipe,
          ...(original.structure_artifact_id
            ? { structure_artifact_id: original.structure_artifact_id }
            : {}),
        })
      ).id,
    );
  };
  try {
    for (const d of items.filter((d: any) => d.state === 'ready' && d.provider === 'pytorch'))
      await wait(page.request, (await post(page.request, `/deployments/${d.id}/unload`, {})).id);
    await wait(
      page.request,
      (
        await post(page.request, `/deployments/${tiny}/load`, {
          precision: 'float16',
          kernel_profile: 'framework-native',
          quantization: { method: 'none' },
        })
      ).id,
    );
    await page.goto('/model-structure/evaluation?deployment=' + tiny);
    await page.getByRole('button', { name: '运行 WikiText-2 Mini / PPL', exact: true }).click();
    await expect(page).toHaveURL(/run=run_/);
    const baseline = new URL(page.url()).searchParams.get('run')!;
    await wait(page.request, baseline);
    await page.getByRole('link', { name: '创建压缩制品', exact: true }).click();
    await expect(page.getByRole('button', { name: '检查可变换模块', exact: true })).toBeEnabled();
    await page.getByRole('button', { name: '检查可变换模块', exact: true }).click();
    await option(page, 'L5 目标模块', 'model.layers.0.mlp');
    await page.keyboard.press('Escape');
    await page.getByRole('heading', { name: '变换方法与范围', exact: true }).click();
    await expect(page.getByRole('button', { name: '创建独立压缩制品', exact: true })).toBeEnabled();
    await page.screenshot({
      path: '../../reports/console/l5-structure/build-desktop.png',
      fullPage: true,
    });
    await page.getByRole('button', { name: '创建独立压缩制品', exact: true }).click();
    await expect(page).toHaveURL(/job=run_/);
    const job = new URL(page.url()).searchParams.get('job')!;
    const built = await wait(page.request, job);
    const identity = built.metrics.structure_artifact.artifact_id;
    expect(built.metrics.structure_artifact.statistics.parameters_after).toBeLessThan(
      built.metrics.structure_artifact.statistics.parameters_before,
    );
    await page.getByRole('link', { name: '查看压缩制品与加载 →', exact: true }).click();
    await page.getByRole('button', { name: '卸载当前模型并加载此制品', exact: true }).click();
    await expect
      .poll(
        async () => {
          const d = await (await page.request.get(base + '/deployments')).json();
          return d.items.find((d: any) => d.id === tiny).structure_artifact_id;
        },
        { timeout: 90000 },
      )
      .toBe(identity);
    await expect(
      page.getByRole('button', { name: '卸载当前模型并加载此制品', exact: true }),
    ).toBeEnabled();
    await page.screenshot({
      path: '../../reports/console/l5-structure/artifacts-desktop.png',
      fullPage: true,
    });
    await page.getByRole('link', { name: '评测与对照', exact: true }).click();
    await expect(
      page.getByRole('button', { name: '运行 WikiText-2 Mini / PPL', exact: true }),
    ).toBeEnabled();
    await page.getByRole('button', { name: '运行 WikiText-2 Mini / PPL', exact: true }).click();
    await expect(page).toHaveURL(/run=run_/);
    await expect.poll(() => new URL(page.url()).searchParams.get('run')).not.toBe(baseline);
    const candidate = new URL(page.url()).searchParams.get('run')!;
    const scored = await wait(page.request, candidate);
    expect(scored.metrics.optimization_receipt.structure_receipt.status).toBe('invoked');
    await option(page, 'L5 基线', `PPL · 原始 · ${baseline}`);
    await option(page, 'L5 候选', `PPL · 压缩 · ${candidate}`);
    await page.getByRole('button', { name: '比较质量或性能差异', exact: true }).click();
    await expect(page.getByText('来源、精度与评测协议核对通过', { exact: true })).toBeVisible();
    await expect(
      page.getByRole('columnheader', { name: '候选 − 基线', exact: true }),
    ).toBeVisible();
    await page.screenshot({
      path: '../../reports/console/l5-structure/evaluation-desktop.png',
      fullPage: true,
    });
    await page.getByRole('link', { name: '查看实际模块与权重 →', exact: true }).click();
    await expect(page.getByText(`当前加载 L5 制品：${identity}`, { exact: true })).toBeVisible();
    await page.screenshot({
      path: '../../reports/console/l5-structure/architecture-desktop.png',
      fullPage: true,
    });
  } finally {
    await restore();
  }
});
