import { expect, test, type Page } from '@playwright/test';
test.use({ actionTimeout: 15000 });

const id = 'quality-record-fixture';
const run = {
  id,
  kind: 'generate',
  state: 'completed',
  cleanup: 'succeeded',
  seq: 1,
  created_at: 1,
  updated_at: 2,
  output: '',
  error: null,
  config: {
    experiment_role: 'quality',
    observation_mode: 'off',
    deployment: {
      id: 'fixture',
      name: 'Fixture Qwen',
      precision: 'float16',
      kernel: 'framework-native',
    },
  },
  metrics: {
    console_e2e_ms: 60000,
    quality_evaluation: {
      status: 'complete',
      score_mode: 'corpus_nll',
      benchmark_id: 'wikitext2-mini-v1',
      perplexity: 40.852101,
      mean_nll: 3.709958,
      scored_tokens: 504,
      document_count: 8,
    },
  },
};
const bundle = {
  manifest: { status: 'complete', publication: 'acknowledged', artifacts: [] },
  manifest_sha256: 'fixture-manifest',
  receipt: {
    status: 'applied',
    requested: {},
    resolved: {},
    effective: {},
    plan_sha256: 'fixture-plan',
    proof: {
      after_cleanup_allocator: {
        allocated_bytes: 1073741824,
        reserved_bytes: 2147483648,
        peak_allocated_bytes: 3221225472,
      },
    },
    precision_receipt: {
      method: 'gptq',
      compute_dtype: 'float16',
      kv: { bits: 2, backend: 'quanto', residual_length: 128 },
    },
  },
  quality: {
    ...run.metrics.quality_evaluation,
    dataset_sha256: 'a'.repeat(64),
    protocol: { forward_mode: 'incremental_decode', max_tokens: 64 },
    samples: [
      {
        id: 'science',
        slice: 'Science article',
        input_tokens: 64,
        scored_tokens: 63,
        nll_sum: 126,
        token_nll: Array.from({ length: 63 }, () => 2),
      },
      {
        id: 'history',
        slice: 'History article',
        input_tokens: 64,
        scored_tokens: 63,
        nll_sum: 189,
        token_nll: Array.from({ length: 63 }, () => 3),
      },
    ],
  },
};

async function fixture(page: Page, failedBundle = false) {
  await page.route(`**/api/console/v1/runs/${id}`, (route) => route.fulfill({ json: run }));
  await page.route(`**/api/console/v1/runs/${id}/events*`, (route) =>
    route.fulfill({ body: '', contentType: 'text/event-stream' }),
  );
  await page.route(`**/api/console/v1/runs/${id}/experiment`, (route) =>
    route.fulfill(
      failedBundle
        ? { status: 409, json: { detail: 'Saved evidence unavailable' } }
        : { json: bundle },
    ),
  );
}

test('quality record defaults to scores, samples, token NLL and scoped memory', async ({
  page,
}) => {
  await fixture(page);
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto(`/runs/${id}`);
  await expect(page.getByRole('heading', { name: '独立质量评测详情' })).toBeVisible();
  await expect(page.getByRole('tab', { name: '质量评分与样本' })).toHaveAttribute(
    'aria-selected',
    'true',
  );
  await expect(page.getByText('40.8521', { exact: true })).toBeVisible();
  await expect(page.getByRole('cell', { name: 'Science article', exact: true })).toBeVisible();
  await expect(page.getByRole('cell', { name: '7.389', exact: true })).toBeVisible();
  await expect(page.getByRole('img', { name: '逐 Token 负对数似然' })).toBeVisible();
  await expect(page.getByText('Worker 累计峰值', { exact: true })).toBeVisible();
  await expect(page.getByText('服务端首内容', { exact: true })).toHaveCount(0);
  await expect(page.getByText('此任务没有文本输出。', { exact: true })).toHaveCount(0);
  await page.getByLabel('评分样本', { exact: true }).first().click();
  await page.getByTitle('History article', { exact: true }).last().click();
  await expect(page.getByTitle('History article', { exact: true }).first()).toBeVisible();
  await expect(page.locator('.ant-select-dropdown:visible')).toHaveCount(0);
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByRole('cell', { name: 'Science article', exact: true })).toBeVisible();
  await expect
    .poll(() => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1))
    .toBeTruthy();
  expect(errors).toEqual([]);
});

test('old output URL and missing evidence preserve saved scores without invented metrics', async ({
  page,
}) => {
  await fixture(page, true);
  await page.goto(`/runs/${id}?tab=output`);
  await expect(page.getByText('40.8521', { exact: true })).toBeVisible();
  await expect(page.getByText('Saved evidence unavailable', { exact: true })).toBeVisible();
  await expect(page.getByText('封存原件读取失败', { exact: false })).toBeVisible();
  await expect(page.getByRole('img', { name: '逐 Token 负对数似然' })).toHaveCount(0);
  await page.getByRole('tab', { name: '全链路观测' }).click();
  await expect(page.getByText('本质量任务未采集性能与机制详录', { exact: true })).toBeVisible();
});

test('saved real quality evidence renders without rerunning or changing the model', async ({
  page,
}) => {
  test.skip(!process.env.HQSB_QUALITY_DETAIL_RUN, 'Read-only target run ID required');
  const identity = process.env.HQSB_QUALITY_DETAIL_RUN!;
  const mutations: string[] = [];
  page.on('request', (request) => {
    if (request.method() === 'POST') mutations.push(request.url());
  });
  await page.goto(`/runs/${identity}?tab=output`);
  await expect(page.getByText('172.4825', { exact: true })).toBeVisible();
  await expect(
    page.getByRole('cell', { name: 'Second Battle of Naktong Bulge', exact: true }),
  ).toBeVisible();
  await expect(page.getByRole('img', { name: '逐 Token 负对数似然' })).toBeVisible();
  await expect(page.getByText('Worker 清理后内存快照', { exact: true })).toBeVisible();
  await page.screenshot({
    path: '../../reports/console/quality-benchmark/quality-record.png',
    fullPage: true,
  });
  await page.goto('/runs?kind=quality');
  const row = page.locator('tbody tr').filter({ hasText: identity.slice(0, 20) });
  await expect(row.getByText('独立质量评测', { exact: true })).toBeVisible();
  await expect(row.getByText('172.483 PPL', { exact: true })).toBeVisible();
  expect(mutations).toEqual([]);
});
