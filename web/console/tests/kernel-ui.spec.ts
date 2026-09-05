import { test, expect, type Page } from '@playwright/test';
import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { enterWorkspace } from './session';

const prefix = '/api/console/v1';
const reportRoot = '../../reports/console/kernel-ui';
test.use({ actionTimeout: 20000 });
async function get(page: Page, path: string) {
  const response = await page.request.get(prefix + path);
  expect(response.ok(), path).toBeTruthy();
  return response.json();
}

test('kernel selector explains unavailable artifacts without submitting a load', async ({
  page,
}) => {
  const requests: string[] = [];
  page.on('request', (request) => {
    if (request.method() === 'POST') requests.push(request.url());
  });
  await page.route('**/kernels/profiles?deployment_id=*', async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    body.native_artifact = { status: 'missing', reason: 'KERNEL_NATIVE_MANIFEST_REQUIRED' };
    for (const profile of body.profiles)
      if (profile.requires_native_artifact)
        profile.availability = { loadable: false, reason: 'KERNEL_NATIVE_MANIFEST_REQUIRED' };
    await route.fulfill({ response, json: body });
  });
  await enterWorkspace(page, '/playground');
  await expect(page.getByText('原生制品：服务尚未配置原生算子制品', { exact: true })).toBeVisible();
  await page.locator('.kernel-control .ant-select-selector').click();
  await expect(
    page.locator('.ant-select-item-option-disabled').filter({ hasText: 'CUDA half2 向量化 · v2' }),
  ).toBeVisible();
  expect(requests.filter((url) => !url.endsWith('/session/login'))).toEqual([]);
});

test('a rejected unload keeps the desired version pending and shows the error', async ({
  page,
}) => {
  await page.route('**/api/console/v1/deployments', async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    body.items[0] = {
      ...body.items[0],
      state: 'ready',
      epoch: 'controlled-epoch',
      kernel_profile: 'framework-native',
    };
    await route.fulfill({ response, json: body });
  });
  await page.route('**/deployments/qwen-reference/unload', (route) =>
    route.fulfill({
      status: 409,
      json: { error: { message: 'Controlled unload failure' } },
    }),
  );
  await enterWorkspace(page, '/playground');
  await page.locator('.kernel-control .ant-select-selector').click();
  await page.getByRole('combobox', { name: '算子配置' }).fill('PyTorch SDPA math · v1');
  await page
    .locator('.ant-select-item-option')
    .filter({ hasText: 'PyTorch SDPA math · v1' })
    .click();
  await page.getByRole('button', { name: '应用并重载模型', exact: true }).click();
  await expect(page.getByText('Controlled unload failure', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: /开始推理/ })).toBeDisabled();
  await expect(page.locator('.kernel-current')).toContainText('framework-native');
  await expect(page.getByText('模型已就绪：qwen3-attention-math-v1', { exact: true })).toHaveCount(
    0,
  );
});

test('real model switches exact implementations, seals proof and retains observations', async ({
  page,
}) => {
  test.skip(process.env.HQSB_KERNEL_UI_LIVE !== '1', 'Explicit Jetson device acceptance only');
  test.setTimeout(600000);
  mkdirSync(reportRoot, { recursive: true });
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  const checks: Record<string, unknown>[] = [];
  await enterWorkspace(page, '/playground');
  await expect(page.getByLabel('访问令牌', { exact: true })).toHaveCount(0);
  const catalog = await get(page, '/kernels/profiles?deployment_id=qwen-reference');
  expect(catalog.native_artifact.status).toBe('files_verified');
  expect(
    catalog.profiles.every(
      (item: { availability: { loadable: boolean } }) => item.availability.loadable,
    ),
  ).toBeTruthy();
  const variants = [
    'framework-native',
    'qwen3-norm-vector-v2',
    'qwen3-norm-scalar-v1',
    'qwen3-local-fusions-v1',
  ];
  let epoch: string | null = null;
  for (const profileId of variants) {
    const option = catalog.profiles.find(
      (item: { profile_id: string }) => item.profile_id === profileId,
    );
    const select = page.getByRole('combobox', { name: '算子配置' });
    if (profileId !== 'framework-native') {
      await page.locator('.kernel-control .ant-select-selector').click();
      await select.fill(option.label);
      await page.locator('.ant-select-item-option').filter({ hasText: option.label }).click();
      await expect(page.getByRole('button', { name: /开始推理/ })).toBeDisabled();
      await expect(page.getByText('选择尚未应用，需要重载模型', { exact: true })).toBeVisible();
    }
    await page
      .getByRole('button', { name: /^(加载并应用算子|应用并重载模型|重新加载当前方案)$/ })
      .click();
    await expect(page.getByText(`模型已就绪：${profileId}`, { exact: true })).toBeVisible({
      timeout: 240000,
    });
    const deployment = (await get(page, '/deployments')).items[0];
    expect(deployment.kernel_profile).toBe(profileId);
    expect(deployment.epoch).not.toBe(epoch);
    epoch = deployment.epoch;
    await page.getByLabel('最大输出 token').fill('3');
    await page
      .getByLabel('推理输入')
      .fill('Explain why memory bandwidth matters for language models.');
    await page.locator('.observation-control .ant-select-selector').click();
    const mode =
      profileId === 'qwen3-norm-vector-v2'
        ? '算子诊断 · PyTorch Profiler'
        : '内存诊断 · 分配生命周期 / KV / OS';
    await page.getByText(mode, { exact: true }).last().click();
    const submitted = page.waitForResponse(
      (response) =>
        response.url().endsWith(prefix + '/requests') && response.request().method() === 'POST',
    );
    await page.getByRole('button', { name: /开始推理/ }).click();
    const accepted = await (await submitted).json();
    await expect
      .poll(async () => (await get(page, `/runs/${accepted.id}`)).state, { timeout: 120000 })
      .toBe('completed');
    const run = await get(page, `/runs/${accepted.id}`);
    const receipt = run.metrics.optimization_receipt;
    expect(receipt.status).toBe('applied');
    expect(receipt.effective.kernel_profile).toBe(profileId);
    expect(receipt.kernel_receipt.profile_id).toBe(profileId);
    expect(receipt.kernel_receipt.status).toBe(
      profileId === 'framework-native' ? 'reference' : 'invoked',
    );
    expect(receipt.kernel_receipt.fallbacks).toEqual([]);
    expect(receipt.kernel_receipt.not_exercised).toEqual([]);
    expect(run.config.epoch).toBe(epoch);
    expect(run.metrics.observation.phases.length).toBeGreaterThan(0);
    expect(run.metrics.observation.tokens.length).toBeGreaterThan(0);
    const bundle = await get(page, `/runs/${accepted.id}/experiment`);
    expect(bundle.manifest.publication).toBe('supervisor_acknowledged');
    expect(bundle.manifest.artifacts.map((item: { path: string }) => item.path)).toEqual(
      expect.arrayContaining([
        'kernel-plan.json',
        'kernel-bindings.json',
        'kernel-summary.json',
        'execution-receipt.json',
      ]),
    );
    const view = page.getByRole('region', { name: '实际算子调用凭据' });
    await expect(view.getByText(profileId, { exact: true })).toBeVisible();
    if (profileId !== 'framework-native')
      await expect(
        view.getByRole('cell', { name: Object.values(option.operators)[0] as string, exact: true }),
      ).toBeVisible();
    const summary = bundle.manifest.artifacts.find(
      (item: { path: string }) => item.path === 'kernel-summary.json',
    );
    const ready = page.waitForEvent('download');
    await page
      .getByRole('row')
      .filter({ has: page.getByRole('cell', { name: 'kernel-summary.json', exact: true }) })
      .getByRole('link', { name: '下载原件', exact: true })
      .click();
    const downloaded = await ready;
    expect(
      createHash('sha256')
        .update(readFileSync((await downloaded.path())!))
        .digest('hex'),
    ).toBe(summary.sha256);
    let observation: unknown = null;
    if (profileId === 'qwen3-norm-vector-v2') {
      const profile = run.metrics.observation.profile;
      expect(profile.activities).toContain('CUDA');
      expect(profile.trace_available).toBe(true);
      expect(
        profile.operators.some((item: { name: string }) => item.name.includes('l2_rms_norm')),
      ).toBeTruthy();
      observation = {
        status: profile.status,
        activities: profile.activities,
        trace_available: profile.trace_available,
      };
    } else {
      const mechanism = await get(page, `/runs/${accepted.id}/observations`);
      expect(mechanism.publication).toBe('supervisor_acknowledged');
      observation = {
        publication: mechanism.publication,
        capture_status: mechanism.capture_status,
        provenance: mechanism.provenance,
      };
    }
    checks.push({
      profile_id: profileId,
      run_id: run.id,
      epoch,
      receipt: receipt.kernel_receipt,
      artifact_sha256: summary.sha256,
      observation,
    });
    writeFileSync(
      `${reportRoot}/live-progress.json`,
      JSON.stringify({ status: 'RUNNING', checks }, null, 2),
    );
    await page.screenshot({ path: `${reportRoot}/${profileId}.png`, fullPage: true });
  }
  // Leave the existing reference workflow ready after the acceptance campaign.
  await page.locator('.kernel-control .ant-select-selector').click();
  await page.getByRole('combobox', { name: '算子配置' }).fill('Framework native · 模型默认实现');
  await page
    .locator('.ant-select-item-option')
    .filter({ hasText: 'Framework native · 模型默认实现' })
    .click();
  await page.getByRole('button', { name: '应用并重载模型', exact: true }).click();
  await expect(page.getByText('模型已就绪：framework-native', { exact: true })).toBeVisible({
    timeout: 240000,
  });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect
    .poll(() => page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth))
    .toBeLessThanOrEqual(1);
  await page.screenshot({ path: `${reportRoot}/mobile.png`, fullPage: true });
  expect(errors).toEqual([]);
  writeFileSync(
    `${reportRoot}/live.json`,
    JSON.stringify({ status: 'PASS', checks, errors, final_profile: 'framework-native' }, null, 2),
  );
});
