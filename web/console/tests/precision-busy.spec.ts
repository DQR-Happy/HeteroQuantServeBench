import { expect, test, type Page } from '@playwright/test';

const root = '/api/console/v1';
const capability = {
  compute_dtypes: ['float32', 'float16', 'bfloat16'],
  methods: [
    { id: 'none', name: 'Floating point', availability: 'available_for_target_validation' },
  ],
  effective: {
    method: 'gptq',
    compute_dtype: 'torch.float16',
    execution: 'fused_dequant_weight_only',
    quantized_module_count: 42,
    quantized_storage_bytes: 4000000,
    remaining_float_parameter_bytes: 18000000,
  },
};

async function fixture(page: Page) {
  await page.route('**/api/console/v1/deployments', (route) =>
    route.fulfill({
      json: {
        items: [
          {
            id: 'fixture',
            name: 'Fixture',
            provider: 'pytorch',
            state: 'ready',
            epoch: 'quantized-epoch',
            precision: 'float16',
            kernel_profile: 'framework-native',
            context_limit: 256,
            quantization: { method: 'gptq' },
          },
        ],
      },
    }),
  );
  await page.route('**/api/console/v1/precision/catalog', (route) =>
    route.fulfill({ json: capability }),
  );
  await page.route('**/api/console/v1/precision/artifacts', (route) =>
    route.fulfill({ json: { items: [] } }),
  );
  await page.route('**/api/console/v1/precision/quality-benchmarks', (route) =>
    route.fulfill({
      json: {
        items: [
          {
            id: 'wikitext2-mini-v1',
            name: 'WikiText-2 Mini',
            sample_count: 8,
            max_tokens: 64,
            articles: ['Fixture article'],
            dataset_sha256: 'fixture',
            source: {
              url: 'https://huggingface.co/datasets/Salesforce/wikitext',
              variant: 'wikitext-2-raw-v1',
            },
          },
        ],
      },
    }),
  );
}

for (const outcome of ['completed', 'failed'] as const) {
  test(`L4 pauses probes during quality work and refreshes after ${outcome}`, async ({ page }) => {
    await fixture(page);
    let active = false,
      finish = false,
      polls = 0,
      probes = 0,
      collisions = 0,
      submissions = 0;
    const job = () => ({
      id: 'quality-fixture',
      kind: 'generate',
      state: finish ? outcome : 'running',
      cleanup: finish ? 'succeeded' : 'pending',
      config: {},
      output: '',
      error: finish && outcome === 'failed' ? 'QUALITY_FIXTURE_FAILURE' : null,
      metrics:
        finish && outcome === 'completed' ? { quality_evaluation: { perplexity: 1.75 } } : {},
    });
    await page.route('**/api/console/v1/deployments/*/precision-capabilities', (route) => {
      probes++;
      if (active) {
        collisions++;
        return route.fulfill({
          status: 409,
          json: {
            error: {
              code: 'DEVICE_BUSY',
              message: 'Wait for the current task before probing precision capabilities',
            },
          },
        });
      }
      return route.fulfill({ json: capability });
    });
    await page.route('**/api/console/v1/precision/quality-jobs', (route) => {
      submissions++;
      active = true;
      return route.fulfill({ json: job() });
    });
    await page.route('**/api/console/v1/runs/quality-fixture', (route) => {
      polls++;
      if (finish) active = false;
      return route.fulfill({ json: job() });
    });
    await page.goto('/precision');
    const refresh = page.getByRole('button', { name: '刷新实际凭据', exact: true });
    const evaluate = page.getByRole('button', { name: '运行独立质量评测', exact: true });
    await expect(page.getByText('torch.float16', { exact: true })).toBeVisible();
    await expect(refresh).toBeEnabled();
    const before = probes;
    await evaluate.click();
    await expect(evaluate).toBeDisabled();
    await expect(refresh).toBeDisabled();
    await expect(page.getByText(/精度凭据暂停刷新/)).toBeVisible();
    // Remain active longer than the old 3-second error retry interval.
    await expect.poll(() => polls).toBeGreaterThanOrEqual(5);
    expect(collisions).toBe(0);
    expect(probes).toBe(before);
    await expect(page.getByText('数据读取失败', { exact: true })).toHaveCount(0);
    finish = true;
    await expect(evaluate).toBeEnabled();
    await expect.poll(() => probes).toBeGreaterThan(before);
    await expect(refresh).toBeEnabled();
    expect(collisions).toBe(0);
    expect(submissions).toBe(1);
    if (outcome === 'failed')
      await expect(page.getByText('QUALITY_FIXTURE_FAILURE', { exact: true })).toBeVisible();
    else await expect(page.getByText(/PPL 1.75/)).toBeVisible();
    await evaluate.click();
    await expect.poll(() => submissions).toBe(2);
  });
}

for (const code of ['DEVICE_BUSY', 'L4_WORKER_UNSUPPORTED']) {
  test(`L4 classifies capability conflict ${code} without hiding other errors`, async ({
    page,
  }) => {
    await fixture(page);
    let available = false;
    await page.route('**/api/console/v1/deployments/*/precision-capabilities', (route) =>
      available
        ? route.fulfill({ json: capability })
        : route.fulfill({
            status: 409,
            json: { error: { code, message: 'CAPABILITY_FIXTURE_CONFLICT' } },
          }),
    );
    await page.goto('/precision');
    if (code === 'DEVICE_BUSY') {
      await expect(page.getByText(/精度凭据暂停刷新/)).toBeVisible();
      await expect(page.getByText('数据读取失败', { exact: true })).toHaveCount(0);
    } else {
      await expect(page.getByText('数据读取失败', { exact: true })).toBeVisible();
      await expect(page.getByText('CAPABILITY_FIXTURE_CONFLICT')).toBeVisible();
    }
    available = true;
    await expect(page.getByText('torch.float16', { exact: true })).toBeVisible();
    await expect(page.getByText(/精度凭据暂停刷新/)).toHaveCount(0);
    await expect(page.getByText('数据读取失败', { exact: true })).toHaveCount(0);
  });
}

test('real quantized deployment evaluates without competing capability probes', async ({
  page,
}) => {
  test.skip(process.env.HQSB_L4_CAPABILITY_LIVE !== '1', 'Explicit target-GPU acceptance only');
  const get = async (path: string) => {
    const response = await page.request.get(root + path);
    expect(response.ok(), await response.text()).toBeTruthy();
    return response.json();
  };
  const before = await get('/overview');
  expect(before.pending).toBe(0);
  const loaded = before.deployments.find(
    (d: any) => d.provider === 'pytorch' && d.state === 'ready',
  );
  expect(loaded?.quantization?.artifact_id).toBeTruthy();
  const conflicts: number[] = [];
  page.on('response', (r) => {
    if (r.url().includes('/precision-capabilities') && r.status() !== 200)
      conflicts.push(r.status());
  });
  await page.goto('/precision');
  await expect(page.getByRole('button', { name: '刷新实际凭据', exact: true })).toBeEnabled();
  await page.getByRole('combobox', { name: '质量评测数据集', exact: true }).click();
  await page.getByTitle('自定义短文本 · logits / 标准答案', { exact: true }).click();
  await page
    .getByLabel('独立质量文本', { exact: true })
    .fill('A red cat sat beside the window.\nThe young girl smiled at her sister.');
  const accepted = page.waitForResponse(
    (r) => r.url().endsWith('/precision/quality-jobs') && r.request().method() === 'POST',
  );
  const evaluate = page.getByRole('button', { name: '运行独立质量评测', exact: true });
  await evaluate.click();
  const response = await accepted;
  expect(response.ok(), await response.text()).toBeTruthy();
  const run = await response.json();
  await expect
    .poll(async () => (await get('/runs/' + run.id)).state, { timeout: 120000 })
    .toBe('completed');
  await expect(evaluate).toBeEnabled();
  await expect(page.getByRole('button', { name: '刷新实际凭据', exact: true })).toBeEnabled();
  const result = await get('/runs/' + run.id);
  expect(result.cleanup).toBe('succeeded');
  expect(result.metrics.quality_evaluation.status).toBe('complete');
  expect(Number.isFinite(result.metrics.quality_evaluation.perplexity)).toBeTruthy();
  expect(conflicts).toEqual([]);
  await expect(page.getByText('数据读取失败', { exact: true })).toHaveCount(0);
  const after = await get('/overview');
  expect(after.pending).toBe(0);
  const actual = after.deployments.find((d: any) => d.id === loaded.id);
  expect(actual.epoch).toBe(loaded.epoch);
  expect(actual.quantization).toEqual(loaded.quantization);
  await page.screenshot({
    path: '../../reports/console/l4-capability-busy/live-quality.png',
    fullPage: true,
  });
  const { mkdirSync, writeFileSync } = await import('node:fs');
  mkdirSync('../../reports/console/l4-capability-busy', { recursive: true });
  writeFileSync(
    '../../reports/console/l4-capability-busy/live-result.json',
    JSON.stringify(
      { result, before_epoch: loaded.epoch, after_epoch: actual.epoch, conflicts },
      null,
      2,
    ),
  );
});
