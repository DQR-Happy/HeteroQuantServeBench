import { test, expect, type Page } from '@playwright/test';
import { mkdirSync, writeFileSync } from 'node:fs';
import { enterWorkspace } from './session';

const prefix = '/api/console/v1';
const reportRoot = '../../reports/console/benchmark-ui';
test.use({ actionTimeout: 20000 });
async function get(page: Page, path: string) {
  const response = await page.request.get(prefix + path);
  expect(response.ok(), path).toBeTruthy();
  return response.json();
}

test('fixed inputs and defaults are ready without automatic model execution', async ({ page }) => {
  const posts: string[] = [];
  page.on('request', (request) => {
    if (request.method() === 'POST') posts.push(request.url());
  });
  await enterWorkspace(page, '/benchmarks');
  await expect(page.getByRole('heading', { name: '性能与质量对照', exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: '开始固定输入对照', exact: true })).toBeEnabled();
  await expect(page.getByText('短答案 · 算术', { exact: true })).toBeVisible();
  await expect(page.getByText('结构化提取', { exact: true })).toBeVisible();
  await expect(page.getByText('上下文检索', { exact: true })).toBeVisible();
  await expect(page.getByText('解释生成', { exact: true })).toBeVisible();
  await expect(page.locator('tbody .ant-checkbox-checked')).toHaveCount(4);
  await page.locator('tbody .ant-table-row-expand-icon').first().click();
  await expect(page.locator('.benchmark-prompt')).toContainText('17 × 19');
  await page.reload();
  await expect(page.getByRole('button', { name: '开始固定输入对照', exact: true })).toBeEnabled();
  expect(posts.filter((url) => !url.endsWith('/session/login'))).toEqual([]);
  await page.getByRole('link', { name: '打开交互推理案例', exact: true }).click();
  await expect(page).toHaveURL(/\/playground/);
  await expect(page.getByLabel('推理输入')).toBeVisible();
});

test('unavailable default candidates cannot be submitted', async ({ page }) => {
  await page.route('**/benchmark-catalog?deployment_id=*', async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    for (const scheme of body.schemes)
      if (body.defaults.candidate_ids.includes(scheme.id))
        scheme.availability = { loadable: false, reason: 'KERNEL_NATIVE_MANIFEST_REQUIRED' };
    await route.fulfill({ response, json: body });
  });
  await enterWorkspace(page, '/benchmarks');
  await expect(page.getByRole('button', { name: '开始固定输入对照', exact: true })).toBeDisabled();
  await page.locator('.benchmark-candidates .ant-select-selector').click();
  await expect(
    page.locator('.ant-select-item-option-disabled').filter({ hasText: 'CUDA half2 向量化 · v2' }),
  ).toBeVisible();
});

test('real fixed suite compares outputs and performance, survives reload and restores model', async ({
  page,
}) => {
  test.skip(process.env.HQSB_BENCHMARK_UI_LIVE !== '1', 'Explicit Jetson acceptance only');
  test.setTimeout(900000);
  mkdirSync(reportRoot, { recursive: true });
  const errors: string[] = [],
    posts: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('request', (request) => {
    if (request.method() === 'POST' && request.url().endsWith(prefix + '/benchmarks'))
      posts.push(request.url());
  });
  await enterWorkspace(page, '/benchmarks');
  const initial = (await get(page, '/deployments')).items[0];
  expect(initial.state).toBe('ready');
  const catalog = await get(page, '/benchmark-catalog?deployment_id=qwen-reference');
  expect(catalog.native_artifact.status).toBe('files_verified');
  const acceptedResponse = page.waitForResponse(
    (response) =>
      response.url().endsWith(prefix + '/benchmarks') && response.request().method() === 'POST',
  );
  await page.getByRole('button', { name: '开始固定输入对照', exact: true }).click();
  const accepted = await (await acceptedResponse).json();
  expect(accepted.id).toMatch(/^bench_[0-9a-f]{32}$/);
  await expect(page).toHaveURL(new RegExp(`benchmark=${accepted.id}`));
  await page.reload();
  await expect(page.getByRole('button', { name: '取消对照并恢复部署' })).toBeVisible();
  let final: any;
  await expect
    .poll(
      async () => {
        final = await get(page, `/benchmarks/${accepted.id}`);
        writeFileSync(
          `${reportRoot}/live-progress.json`,
          JSON.stringify(
            {
              id: final.id,
              state: final.state,
              phase: final.phase,
              records: final.records.length,
              error: final.error,
            },
            null,
            2,
          ),
        );
        return ['completed', 'failed', 'cancelled', 'interrupted'].includes(final.state)
          ? final.state
          : 'running';
      },
      { timeout: 840000, intervals: [2000, 4000] },
    )
    .toBe('completed');
  expect(final.restore_error).toBeNull();
  expect(final.records).toHaveLength(24);
  expect(final.records.filter((row: any) => row.role === 'measurement')).toHaveLength(12);
  expect(final.report.rows).toHaveLength(12);
  const checks: any[] = [];
  for (const row of final.records.filter((item: any) => item.role === 'measurement')) {
    expect(row.state).toBe('completed');
    expect(row.cleanup).toBe('succeeded');
    expect(row.receipt.status).toBe('applied');
    expect(row.metrics.input_tokens).toBeGreaterThan(0);
    expect(row.metrics.runtime_e2e_ms).toBeGreaterThan(0);
    const run = await get(page, `/runs/${row.run_id}`);
    expect(run.config.benchmark_links.benchmark_id).toBe(final.id);
    expect(run.config.observation_mode).toBe('off');
    expect(run.config.messages).toEqual(
      final.plan.cases.find((item: any) => item.id === row.case_id).messages,
    );
    const bundle = await get(page, `/runs/${row.run_id}/experiment`);
    expect(bundle.manifest.publication).toBe('supervisor_acknowledged');
    expect(
      bundle.manifest.artifacts.some((artifact: any) => artifact.path === 'execution-receipt.json'),
    ).toBeTruthy();
    checks.push({
      run_id: row.run_id,
      case_id: row.case_id,
      scheme_id: row.scheme_id,
      receipt: row.receipt.kernel_receipt,
      bundle: bundle.manifest.publication,
    });
  }
  for (const task of final.plan.cases) {
    const records = final.records.filter(
      (row: any) => row.role === 'measurement' && row.case_id === task.id,
    );
    expect(new Set(records.map((row: any) => row.input_token_sha256)).size).toBe(1);
    const rows = final.report.rows.filter((row: any) => row.case_id === task.id);
    for (const row of rows) {
      expect(row.completed).toBe(1);
      if (task.scoring === 'unscored') expect(row.task_accuracy).toBeNull();
      if (row.reasons.includes('OUTPUT_LENGTH_CHANGED')) expect(row.relative_e2e).toBeNull();
    }
  }
  const restored = (await get(page, '/deployments')).items[0];
  expect(restored.state).toBe('ready');
  expect(restored.kernel_profile).toBe(initial.kernel_profile);
  await page.reload();
  await expect(page.getByText('完成测量 12 / 12', { exact: true })).toBeVisible();
  await expect(page.getByText('同一输入的输出对照', { exact: true })).toBeVisible();
  await page.screenshot({ path: `${reportRoot}/desktop.png`, fullPage: true });
  await page.getByRole('tab', { name: '结构化提取', exact: true }).click();
  await expect(page.locator('.benchmark-output')).toHaveCount(3);
  await page.screenshot({ path: `${reportRoot}/structured.png`, fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect
    .poll(() => page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth))
    .toBeLessThanOrEqual(2);
  await page.screenshot({ path: `${reportRoot}/mobile.png`, fullPage: true });
  expect(posts).toHaveLength(1);
  expect(errors).toEqual([]);
  writeFileSync(
    `${reportRoot}/live.json`,
    JSON.stringify(
      {
        status: 'PASS',
        benchmark: final,
        checks,
        initial: { state: initial.state, profile: initial.kernel_profile },
        restored: { state: restored.state, profile: restored.kernel_profile },
        posts: posts.length,
        browser_errors: errors,
      },
      null,
      2,
    ),
  );
});

test('saved comparison opens on results and configuration can be expanded explicitly', async ({
  page,
}) => {
  const catalog = await get(page, '/benchmarks');
  const saved = catalog.items.find(
    (item: any) => item.state === 'completed' && item.case_count === 4,
  );
  test.skip(!saved, 'Requires a completed real fixed-suite comparison');
  const posts: string[] = [];
  page.on('request', (request) => {
    if (request.method() === 'POST') posts.push(request.url());
  });
  await enterWorkspace(page, `/benchmarks?benchmark=${saved.id}`);
  await expect(page.getByText('同一输入的输出对照', { exact: true })).toBeVisible();
  await expect(page.getByLabel('每次输出预算 / tokens', { exact: true })).toHaveCount(0);
  await page.getByRole('button', { name: '配置下一轮对照', exact: true }).click();
  await expect(page.getByLabel('每次输出预算 / tokens', { exact: true })).toBeVisible();
  await page.reload();
  await expect(page.getByLabel('每次输出预算 / tokens', { exact: true })).toHaveCount(0);
  await page.getByRole('tab', { name: '解释生成', exact: true }).click();
  await expect(
    page.locator('.benchmark-output').getByText('未作客观任务评分', { exact: true }),
  ).toHaveCount(3);
  await page.screenshot({ path: `${reportRoot}/results-desktop.png`, fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect
    .poll(() => page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth))
    .toBeLessThanOrEqual(2);
  await page.screenshot({ path: `${reportRoot}/results-mobile.png`, fullPage: true });
  expect(posts.filter((url) => !url.endsWith('/session/login'))).toEqual([]);
});
