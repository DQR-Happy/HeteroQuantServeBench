import { expect, test } from '@playwright/test';

test('built-in corpus selection submits a fixed ID and presents PPL deltas', async ({ page }) => {
  const capability = {
    compute_dtypes: ['float16'],
    methods: [{ id: 'none', name: 'Floating point' }],
    effective: {
      method: 'none',
      compute_dtype: 'torch.float16',
      execution: 'framework',
      quantized_module_count: 0,
      quantized_storage_bytes: 0,
      remaining_float_parameter_bytes: 1024,
    },
  };
  await page.route('**/api/console/v1/deployments', (route) =>
    route.fulfill({
      json: {
        items: [
          {
            id: 'fixture',
            name: 'Fixture',
            provider: 'pytorch',
            state: 'ready',
            epoch: 'fixture-epoch',
            precision: 'float16',
            kernel_profile: 'framework-native',
          },
        ],
      },
    }),
  );
  await page.route('**/api/console/v1/precision/catalog', (route) =>
    route.fulfill({ json: capability }),
  );
  await page.route('**/api/console/v1/deployments/*/precision-capabilities', (route) =>
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
            name: 'WikiText-2 Mini · 8 段 / 64 Token',
            sample_count: 8,
            max_tokens: 64,
            dataset_sha256: 'fixture-data-hash',
            articles: ['Fixture article'],
            source: {
              url: 'https://huggingface.co/datasets/Salesforce/wikitext',
              variant: 'wikitext-2-raw-v1',
            },
          },
        ],
      },
    }),
  );
  let submitted: any;
  const finished = {
    id: 'corpus-fixture',
    kind: 'generate',
    state: 'completed',
    cleanup: 'succeeded',
    config: {},
    output: '',
    metrics: {
      quality_evaluation: {
        status: 'complete',
        score_mode: 'corpus_nll',
        benchmark_id: 'wikitext2-mini-v1',
        perplexity: 10,
        mean_nll: Math.log(10),
        scored_tokens: 504,
        document_count: 8,
      },
    },
  };
  await page.route('**/api/console/v1/precision/quality-jobs', (route) => {
    submitted = route.request().postDataJSON();
    return route.fulfill({ json: finished });
  });
  await page.route('**/api/console/v1/runs/corpus-fixture', (route) =>
    route.fulfill({ json: finished }),
  );
  await page.route('**/api/console/v1/precision/comparisons', (route) =>
    route.fulfill({
      json: {
        candidates: [
          {
            run_id: 'candidate',
            quality: {
              metric: 'token_perplexity',
              status: 'failed',
              baseline_perplexity: 10,
              candidate_perplexity: 20,
              mean_nll_increase: Math.log(2),
              ppl_ratio: 2,
            },
          },
        ],
      },
    }),
  );
  await page.goto('/precision');
  await expect(
    page.getByTitle('WikiText-2 Mini · 8 段 / 64 Token', { exact: true }).first(),
  ).toBeVisible();
  await expect(page.getByLabel('独立质量文本', { exact: true })).toBeHidden();
  await expect(page.getByText(/用于快速检查量化对英文语料预测的影响/)).toBeVisible();
  await page.getByLabel('预登记 PPL 阈值', { exact: true }).fill('{"max_mean_nll_increase":0.1}');
  await page.getByRole('button', { name: '运行独立质量评测', exact: true }).click();
  await expect.poll(() => submitted?.quality_plan?.benchmark_id).toBe('wikitext2-mini-v1');
  expect(submitted.quality_plan.samples).toBeUndefined();
  expect(submitted.ppl_limits).toEqual({ max_mean_nll_increase: 0.1 });
  await expect(page.getByText(/504 个计分 Token/)).toBeVisible();
  await page.getByLabel('质量基线运行 ID', { exact: true }).fill('baseline');
  await page.getByLabel('质量候选运行 ID', { exact: true }).fill('candidate');
  await expect(page.getByRole('button', { name: /生成质量对照/ })).toBeEnabled();
  await page.getByRole('button', { name: /生成质量对照/ }).click();
  await expect(page.getByRole('columnheader', { name: 'PPL 倍率 ↓' })).toBeVisible();
  await expect(page.getByText('超出预登记阈值', { exact: true })).toBeVisible();
});

test('real built-in PPL evidence is visible in the quantization workbench', async ({ page }) => {
  test.skip(
    !process.env.HQSB_PPL_BASELINE_RUN || !process.env.HQSB_PPL_CANDIDATE_RUN,
    'Requires target-GPU acceptance run IDs',
  );
  await page.goto('/precision');
  await expect(page.getByText(/用于快速检查量化对英文语料预测的影响/)).toBeVisible();
  await page
    .getByLabel('质量基线运行 ID', { exact: true })
    .fill(process.env.HQSB_PPL_BASELINE_RUN!);
  await page
    .getByLabel('质量候选运行 ID', { exact: true })
    .fill(process.env.HQSB_PPL_CANDIDATE_RUN!);
  await expect(page.getByRole('button', { name: '生成质量对照', exact: true })).toBeEnabled();
  const compared = page.waitForResponse(
    (r) => r.url().endsWith('/precision/comparisons') && r.request().method() === 'POST',
  );
  await page.getByRole('button', { name: '生成质量对照', exact: true }).click();
  const response = await compared;
  expect(response.ok(), await response.text()).toBeTruthy();
  await expect(page.getByRole('columnheader', { name: '平均 NLL 增量 ↓' })).toBeVisible();
  await expect(page.getByText('仅测量，未设阈值', { exact: true })).toBeVisible();
  await page.screenshot({
    path: '../../reports/console/quality-benchmark/live-workbench.png',
    fullPage: true,
  });
  await page
    .locator('section')
    .filter({ has: page.getByRole('heading', { name: '基线与候选质量对照', exact: true }) })
    .screenshot({
      path: '../../reports/console/quality-benchmark/live-comparison.png',
    });
});
