import { expect, test, type Page } from '@playwright/test';
import type { Deployment } from '../src/api/types';

// Explicit fixtures exercise UI failures without changing any real deployment.
async function modelFixture(
  page: Page,
  options: { active?: boolean; failLoad?: boolean; capabilityError?: boolean } = {},
) {
  const common = {
    provider: 'pytorch',
    kernel_profile: 'framework-native',
    context_limit: 256,
    precision: 'float16',
    detail: {},
  };
  let items = [
    { ...common, id: 'fixture-gemma', name: 'Fixture Gemma', state: 'ready', epoch: 'gemma-epoch' },
    { ...common, id: 'fixture-qwen', name: 'Fixture Qwen', state: 'unloaded', epoch: null },
  ];
  const events: string[] = [];
  let failLoad = options.failLoad;
  let capabilityError = options.capabilityError;
  await page.route('**/api/console/v1/deployments', (route) => route.fulfill({ json: { items } }));
  await page.route('**/api/console/v1/scheduling-benchmarks', (route) =>
    route.fulfill({
      json: {
        items: [],
        active_id: options.active ? 'fixture-active' : null,
      },
    }),
  );
  await page.route('**/api/console/v1/deployments/*/scheduling-capabilities', (route) => {
    if (capabilityError) {
      capabilityError = false;
      return route.fulfill({ status: 409, json: { detail: 'Fixture device busy' } });
    }
    const gemma = route.request().url().includes('fixture-gemma');
    return route.fulfill({
      json: { supported: !gemma, reasons: gemma ? ['SLIDING_OR_HYBRID_CACHE_NOT_ADAPTED'] : [] },
    });
  });
  await page.route(/\/api\/console\/v1\/deployments\/fixture-[^/]+\/(load|unload)$/, (route) => {
    const [, id, action] = new URL(route.request().url()).pathname.match(
      /deployments\/([^/]+)\/(load|unload)$/,
    )!;
    events.push(`${action}:${id}:accepted`);
    items = items.map((d) =>
      d.id === id ? { ...d, state: action === 'load' ? 'loading' : 'draining' } : d,
    );
    return route.fulfill({
      json: { id: `fixture-${action}-${id}`, kind: action, state: 'running', cleanup: 'pending' },
    });
  });
  await page.route('**/api/console/v1/runs/fixture-*', (route) => {
    const id = new URL(route.request().url()).pathname.split('/').at(-1)!;
    const action = id.includes('fixture-load-') ? 'load' : 'unload';
    const deploymentId = id.replace(`fixture-${action}-`, '');
    const failed = action === 'load' && failLoad;
    if (action === 'load') failLoad = false;
    items = items.map((d) =>
      d.id === deploymentId
        ? {
            ...d,
            state: failed ? 'failed' : action === 'load' ? 'ready' : 'unloaded',
            epoch: action === 'load' && !failed ? 'new-epoch' : null,
            load_run_id: action === 'load' ? id : null,
          }
        : d,
    );
    events.push(`${action}:${deploymentId}:${failed ? 'failed' : 'completed'}`);
    return route.fulfill({
      json: {
        id,
        kind: action,
        state: failed ? 'failed' : 'completed',
        cleanup: 'succeeded',
        error: failed ? 'Fixture weight loading failed' : null,
      },
    });
  });
  return { events };
}

async function choose(page: Page, text: string) {
  await page.getByLabel('实验模型', { exact: true }).first().click({ timeout: 15000 });
  await page.getByTitle(text, { exact: true }).click({ timeout: 15000 });
}

test('unsupported model explains the block; switching waits for release and standalone unload works', async ({
  page,
}) => {
  test.setTimeout(45000);
  const fixture = await modelFixture(page);
  await page.goto('/scheduling');
  await expect(page.getByText(/该模型使用滑动窗口或混合 KV/).first()).toBeVisible();
  await expect(page.getByRole('button', { name: '运行调度对照', exact: true })).toBeDisabled();
  await choose(page, 'Fixture Qwen · unloaded');
  await expect(page.getByText(/将卸载当前模型并加载「Fixture Qwen」/)).toBeVisible();
  await page.getByRole('button', { name: '切换并加载模型', exact: true }).click();
  await expect(page.getByRole('button', { name: '运行调度对照', exact: true })).toBeEnabled();
  expect(fixture.events).toEqual([
    'unload:fixture-gemma:accepted',
    'unload:fixture-gemma:completed',
    'load:fixture-qwen:accepted',
    'load:fixture-qwen:completed',
  ]);
  await page.getByRole('button', { name: '卸载所选模型', exact: true }).click();
  await expect(page.getByText(/所选模型「Fixture Qwen」尚未加载/)).toBeVisible();
  await expect(page.getByRole('button', { name: '运行调度对照', exact: true })).toBeDisabled();
  await expect(page.getByRole('button', { name: '加载所选模型', exact: true })).toBeEnabled();
});

test('a failed model load keeps an actionable retry and its task record', async ({ page }) => {
  test.setTimeout(45000);
  await modelFixture(page, { failLoad: true });
  await page.goto('/scheduling');
  await choose(page, 'Fixture Qwen · unloaded');
  await page.getByRole('button', { name: '切换并加载模型', exact: true }).click();
  await expect(page.getByText('Fixture weight loading failed', { exact: true })).toBeVisible();
  await expect(page.getByRole('link', { name: '查看模型任务记录' })).toHaveAttribute(
    'href',
    '/runs/fixture-load-fixture-qwen',
  );
  await page.getByRole('button', { name: '加载所选模型', exact: true }).click();
  await expect(page.getByRole('button', { name: '运行调度对照', exact: true })).toBeEnabled();
});

test('active experiments prevent model mutation and capability errors can be retried', async ({
  page,
}) => {
  test.setTimeout(45000);
  await modelFixture(page, { active: true });
  await page.goto('/scheduling');
  await expect(page.getByText(/调度实验 fixture-active 正在运行/)).toBeVisible();
  await expect(page.getByRole('button', { name: '卸载所选模型', exact: true })).toBeDisabled();
  await choose(page, 'Fixture Qwen · unloaded');
  await expect(page.getByRole('button', { name: '切换并加载模型', exact: true })).toBeDisabled();
  await page.unrouteAll({ behavior: 'wait' });
  await modelFixture(page, { capabilityError: true });
  await page.reload();
  await expect(page.getByText(/模型能力检查失败：Fixture device busy/)).toBeVisible();
  await page.getByRole('button', { name: '重试检查', exact: true }).click();
  await expect(page.getByText(/该模型使用滑动窗口或混合 KV/).first()).toBeVisible();
});

test('live target switches model, runs multi-request inference, unloads and restores original model', async ({
  page,
}) => {
  test.skip(process.env.HQSB_L8_LIFECYCLE !== '1', 'Explicit real-model target validation only');
  const root = '/api/console/v1';
  const overview = await (await page.request.get(`${root}/overview`)).json();
  expect(overview.pending).toBe(0);
  const original = (overview.deployments as Deployment[]).filter(
    (d) => d.provider === 'pytorch' && d.state === 'ready',
  );
  const target = (overview.deployments as Deployment[]).find((d) => d.model === 'Qwen/Qwen3-0.6B')!;
  expect(target).toBeTruthy();
  const lifecycle = async (id: string, action: string, profile?: string) => {
    const accepted = await page.request.post(`${root}/deployments/${id}/${action}`, {
      headers: { 'X-HQSB-Client': 'console', 'Idempotency-Key': crypto.randomUUID() },
      data: action === 'load' ? { kernel_profile: profile ?? 'framework-native' } : {},
    });
    expect(accepted.ok()).toBeTruthy();
    const run = await accepted.json();
    await expect
      .poll(async () => (await (await page.request.get(`${root}/runs/${run.id}`)).json()).state, {
        timeout: 120000,
      })
      .toBe('completed');
  };
  try {
    await page.goto('/scheduling');
    await choose(page, `${target.name} · ${target.state}`);
    await page
      .getByRole('button', {
        name: original.length ? '切换并加载模型' : '加载所选模型',
        exact: true,
      })
      .click();
    await expect(page.getByRole('button', { name: '运行调度对照', exact: true })).toBeEnabled({
      timeout: 120000,
    });
    await page.screenshot({
      path: '../../reports/console/l8-model-controls/model-ready.png',
      fullPage: true,
    });
    await page.getByLabel('冻结请求负载 JSON').fill(
      JSON.stringify([
        { id: 'control-a', prompt: 'Once upon a time a little fox', output_tokens: 4 },
        { id: 'control-b', prompt: 'One day a cat', output_tokens: 4 },
      ]),
    );
    await page.getByRole('button', { name: '运行调度对照', exact: true }).click();
    await expect(page).toHaveURL(/experiment=run_/);
    const id = new URL(page.url()).searchParams.get('experiment')!;
    await expect
      .poll(
        async () =>
          (await (await page.request.get(`${root}/scheduling-benchmarks/${id}`)).json()).state,
        { timeout: 120000 },
      )
      .toBe('completed');
    const result = await (await page.request.get(`${root}/scheduling-benchmarks/${id}`)).json();
    expect(
      result.result.variants.find((v: { mode: string }) => v.mode === 'continuous')
        .max_actual_batch,
    ).toBeGreaterThan(1);
    expect(
      result.result.variants.every(
        (v: { output_tokens_per_s: number }) => v.output_tokens_per_s > 0,
      ),
    ).toBeTruthy();
    await expect(page.getByRole('columnheader', { name: 'Token/s', exact: true })).toBeVisible();
    await page.screenshot({
      path: '../../reports/console/l8-model-controls/results.png',
      fullPage: true,
    });
    await page.request
      .get(`${root}/runs/${id}`)
      .then(async (r) =>
        test
          .info()
          .attach('live-run.json', { body: await r.body(), contentType: 'application/json' }),
      );
    await page.getByRole('button', { name: '卸载所选模型', exact: true }).click();
    await expect(page.getByRole('button', { name: '加载所选模型', exact: true })).toBeEnabled({
      timeout: 60000,
    });
    const after = await (await page.request.get(`${root}/deployments`)).json();
    expect(after.items.find((d: Deployment) => d.id === target.id).state).toBe('unloaded');
  } finally {
    await expect
      .poll(async () => (await (await page.request.get(`${root}/overview`)).json()).pending, {
        timeout: 120000,
      })
      .toBe(0);
    const after = await (await page.request.get(`${root}/deployments`)).json();
    for (const row of after.items as Deployment[]) {
      if (
        row.provider === 'pytorch' &&
        row.state === 'ready' &&
        !original.some((d) => d.id === row.id)
      )
        await lifecycle(row.id, 'unload');
    }
    for (const row of original) {
      const now = await (await page.request.get(`${root}/deployments`)).json();
      if (now.items.find((d: Deployment) => d.id === row.id).state !== 'ready')
        await lifecycle(row.id, 'load', row.kernel_profile);
    }
  }
});
