import { expect, test, type Page } from '@playwright/test';

const id = 'baseline-ui-fixture';
const root = '/api/console/v1';
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
      id: 'qwen',
      name: 'Fixture Qwen',
      precision: 'float16',
      kernel: 'framework-native',
    },
  },
  metrics: {
    quality_evaluation: {
      status: 'complete',
      score_mode: 'corpus_nll',
      benchmark_id: 'wikitext2-mini-v1',
      perplexity: 172.482506,
      mean_nll: 5.150296,
      scored_tokens: 504,
      document_count: 8,
    },
  },
};
const comparison = {
  status: 'compared',
  reference_eligible: false,
  baseline: {
    run_id: 'saved-floating-reference',
    compute_dtype: 'float16',
    perplexity: 40.852101,
    mean_nll: 3.709958,
    manifest_sha256: 'sealed',
  },
  quality: {
    ppl_delta: 131.630405,
    ppl_relative_increase_percent: 322.21208,
    mean_nll_increase: 1.440338,
    task_accuracy_drop_percentage_points: null,
    documents: [{ id: 'one', article: 'Saved article', mean_nll_increase: 0.5 }],
  },
};

async function fixture(page: Page) {
  await page.route(`**${root}/runs/${id}`, (route) => route.fulfill({ json: run }));
  await page.route(`**${root}/runs/${id}/events*`, (route) =>
    route.fulfill({ body: '', contentType: 'text/event-stream' }),
  );
  await page.route(`**${root}/runs/${id}/experiment`, (route) =>
    route.fulfill({
      json: {
        receipt: null,
        manifest: { status: 'complete', artifacts: [] },
        quality: { ...run.metrics.quality_evaluation, samples: [] },
      },
    }),
  );
}

test('quantized quality result automatically presents its pinned floating baseline and signed degradation', async ({
  page,
}) => {
  await fixture(page);
  await page.route(`**${root}/runs/${id}/quality-baseline`, (route) =>
    route.fulfill({ json: comparison }),
  );
  await page.goto(`/runs/${id}`);
  const panel = page.getByRole('region', { name: '模型浮点基线对照' });
  await expect(panel.getByText('40.8521', { exact: true })).toBeVisible();
  await expect(panel.getByText('+322.21%', { exact: true })).toBeVisible();
  await expect(panel.getByText('+1.440338', { exact: true })).toBeVisible();
  await expect(panel.getByRole('link', { name: 'saved-floating-reference' })).toHaveAttribute(
    'href',
    '/runs/saved-floating-reference',
  );
  await expect(panel.getByRole('button', { name: '设为模型浮点基线' })).toHaveCount(0);
  await panel.getByText('逐段相对基线变化', { exact: true }).click();
  await expect(panel.getByRole('cell', { name: '+64.87%', exact: true })).toBeVisible();
});

test('missing or invalid baseline never invents a degradation percentage', async ({ page }) => {
  await fixture(page);
  await page.route(`**${root}/runs/${id}/quality-baseline`, (route) =>
    route.fulfill({ json: { status: 'missing', reference_eligible: false } }),
  );
  await page.goto(`/runs/${id}`);
  const panel = page.getByRole('region', { name: '模型浮点基线对照' });
  await expect(panel.getByText('尚未建立同协议的模型浮点基线', { exact: true })).toBeVisible();
  await expect(panel.getByText('PPL 相对变化', { exact: true })).toHaveCount(0);
  await expect(page.getByText('172.4825', { exact: true })).toBeVisible();
});

test('floating records replace the primary only through an explicit registration', async ({
  page,
}) => {
  await fixture(page);
  let registered = false;
  await page.route(`**${root}/runs/${id}/quality-baseline`, (route) =>
    route.fulfill({
      json: {
        ...comparison,
        status: registered ? 'baseline' : 'compared',
        reference_eligible: true,
        baseline: { ...comparison.baseline, run_id: registered ? id : comparison.baseline.run_id },
      },
    }),
  );
  await page.route(`**${root}/precision/quality-baselines`, (route) => {
    expect(route.request().postDataJSON()).toEqual({ run_id: id });
    registered = true;
    return route.fulfill({ status: 201, json: { status: 'baseline' } });
  });
  await page.goto(`/runs/${id}`);
  const panel = page.getByRole('region', { name: '模型浮点基线对照' });
  await panel.getByRole('button', { name: '设为模型浮点基线' }).click();
  await expect(panel.getByRole('link', { name: id, exact: true })).toBeVisible();
  await expect(panel.getByRole('button', { name: '设为模型浮点基线' })).toHaveCount(0);
});

test('real historical quantized record links to the model baseline without model operations', async ({
  page,
}) => {
  test.skip(!process.env.HQSB_BASELINE_LIVE, 'Read-only target evidence acceptance');
  const mutations: string[] = [];
  page.on('request', (r) => {
    if (r.method() === 'POST') mutations.push(r.url());
  });
  await page.goto('/runs/run_8137c350f7be4baeb2e46ffada0873f3');
  const panel = page.getByRole('region', { name: '模型浮点基线对照' });
  await expect(panel.getByText('40.8521', { exact: true })).toBeVisible();
  await expect(panel.getByText('+322.21%', { exact: true })).toBeVisible();
  await panel.screenshot({
    path: '../../reports/console/quality-baselines/real-baseline-comparison.png',
  });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect
    .poll(() => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1))
    .toBeTruthy();
  await page.goto('/runs?kind=quality');
  await expect(page.getByRole('columnheader', { name: 'PPL 相对基线', exact: true })).toBeVisible();
  await expect(
    page.getByRole('link', { name: '基线 · float16', exact: true }).first(),
  ).toBeVisible();
  const row = page.locator('tbody tr').filter({ hasText: 'run_8137c350f7be4baeb' });
  if (await row.count())
    await expect(row.getByRole('link', { name: '+322.21%', exact: true })).toBeVisible();
  expect(mutations).toEqual([]);
});
