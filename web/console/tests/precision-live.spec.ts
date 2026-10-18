import { expect, test } from '@playwright/test';
import { mkdirSync, writeFileSync } from 'node:fs';

test('real L4 UI loads a compact model, evaluates, compares and restores', async ({ page }) => {
  test.skip(process.env.HQSB_L4_LIVE !== '1', 'Explicit target-GPU acceptance only');
  test.setTimeout(360000);
  const root = '/api/console/v1';
  const get = async (path: string) => {
    const response = await page.request.get(root + path);
    expect(response.ok()).toBeTruthy();
    return response.json();
  };
  const post = async (path: string, data: unknown) => {
    const response = await page.request.post(root + path, {
      headers: { 'X-HQSB-Client': 'console', 'Idempotency-Key': crypto.randomUUID() },
      data,
    });
    expect(response.ok(), await response.text()).toBeTruthy();
    return response.json();
  };
  const wait = async (id: string) => {
    await expect
      .poll(async () => (await get('/runs/' + id)).state, { timeout: 120000 })
      .toBe('completed');
    return get('/runs/' + id);
  };
  const before = await get('/overview');
  expect(before.pending).toBe(0);
  const original = before.deployments.filter(
    (d: any) => d.provider === 'pytorch' && d.state === 'ready',
  );
  const target = before.deployments.find(
    (d: any) => d.model === 'LLM-Research/tinyllama-15M-stories',
  );
  expect(target).toBeTruthy();
  const results: Record<string, unknown> = {};
  const apply = async () => {
    await page.getByRole('button', { name: '应用方案并重载模型', exact: true }).click();
    await expect(page.getByRole('button', { name: '测试当前已加载模型', exact: true })).toBeEnabled(
      { timeout: 120000 },
    );
  };
  const quality = async () => {
    const accepted = page.waitForResponse(
      (r) => r.url().endsWith('/precision/quality-jobs') && r.request().method() === 'POST',
    );
    await page.getByRole('button', { name: '运行独立质量评测', exact: true }).click();
    const response = await accepted;
    expect(response.ok()).toBeTruthy();
    const run = await response.json();
    const completed = await wait(run.id);
    await expect(page.getByRole('button', { name: '运行独立质量评测', exact: true })).toBeEnabled();
    return completed;
  };
  try {
    await page.goto('/precision');
    await page.getByLabel('L4 实验模型', { exact: true }).first().click();
    await page.getByTitle(`${target.name} · ${target.state}`, { exact: true }).click();
    await apply();
    const baseline = await quality();
    results.baseline = baseline;
    await page.getByLabel('量化方法', { exact: true }).first().click();
    await page.getByText('Round-to-nearest W4/W8', { exact: true }).last().click();
    await page.getByLabel('分组大小', { exact: true }).first().click();
    await page.getByTitle('32', { exact: true }).click();
    await apply();
    await expect(page.getByText('rtn', { exact: true }).first()).toBeVisible();
    const accepted = page.waitForResponse(
      (r) => r.url().endsWith('/requests') && r.request().method() === 'POST',
    );
    await page.getByRole('button', { name: '测试当前已加载模型', exact: true }).click();
    results.inference = await wait((await (await accepted).json()).id);
    await expect(page.getByRole('button', { name: '运行独立质量评测', exact: true })).toBeEnabled();
    const candidate = await quality();
    results.candidate = candidate;
    await page.getByLabel('质量基线运行 ID', { exact: true }).fill(baseline.id);
    await page.getByLabel('质量候选运行 ID', { exact: true }).fill(candidate.id);
    const compared = page.waitForResponse(
      (r) => r.url().endsWith('/precision/comparisons') && r.request().method() === 'POST',
    );
    await page.getByRole('button', { name: '生成质量对照', exact: true }).click();
    const response = await compared;
    expect(response.ok(), await response.text()).toBeTruthy();
    const comparison = await response.json();
    expect(comparison.candidates[0].quality.status).toBe('measured_no_acceptance_limits');
    results.comparison = comparison;
    await page.screenshot({ path: '../../reports/console/l4/live-desktop.png', fullPage: true });
    await page.setViewportSize({ width: 390, height: 844 });
    await page.screenshot({ path: '../../reports/console/l4/live-mobile.png', fullPage: true });
  } finally {
    for (const row of (await get('/deployments')).items)
      if (row.provider === 'pytorch' && row.state === 'ready')
        await wait((await post(`/deployments/${row.id}/unload`, {})).id);
    for (const row of original) {
      const requested = row.detail?.precision_artifact?.requested ?? {};
      if (row.quantization?.artifact_id) requested.artifact_id = row.quantization.artifact_id;
      await wait(
        (
          await post(`/deployments/${row.id}/load`, {
            precision: row.precision,
            kernel_profile: row.kernel_profile,
            quantization: requested,
          })
        ).id,
      );
    }
    mkdirSync('../../reports/console/l4', { recursive: true });
    writeFileSync('../../reports/console/l4/live-ui.json', JSON.stringify(results, null, 2));
  }
});
