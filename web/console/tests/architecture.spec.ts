import { expect, test } from '@playwright/test';
import fs from 'node:fs/promises';

test.use({ actionTimeout: 15000 });

// Read-only live acceptance: never loads/unloads a model or starts inference.
test('artifact architecture supports layer selection, graph drilldown and evidence export', async ({
  page,
  request,
}) => {
  const deployments = (await (await request.get('/api/console/v1/deployments')).json()).items;
  const qwen = deployments.find((row: { model: string }) => row.model === 'Qwen/Qwen3-1.7B');
  test.skip(!qwen, 'requires the configured 3090 Qwen3 artifact');
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto(`/model-architecture?deployment=${qwen.id}`);
  await expect(page.getByText('制品关键结构与已核对实现匹配', { exact: true })).toBeVisible();
  await page.getByRole('tab', { name: 'Attention 内部', exact: true }).click();
  await expect(page.getByText(/Q、K、V 有共同输入/)).toBeVisible();
  await page.getByRole('button', { name: 'Q Linear，self_attn.q_proj', exact: true }).click();
  await expect(page.getByText('model.layers.0.self_attn.q_proj', { exact: true })).toBeVisible();
  await expect(
    page.getByRole('row').filter({ hasText: 'model.layers.0.self_attn.q_proj.weight' }),
  ).toHaveCount(1);
  await page.getByRole('button', { name: 'Q RoPE / reshape', exact: true }).click();
  await expect(page.getByText(/所选计算节点没有独立的模型权重/)).toBeVisible();
  await page.getByLabel('选择具体 Decoder 层', { exact: true }).press('ArrowDown');
  await page.getByText('Layer 1 · 全局 Attention', { exact: true }).click();
  await page.getByRole('button', { name: 'Q Linear，self_attn.q_proj', exact: true }).click();
  await expect(page.getByText('model.layers.1.self_attn.q_proj', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: '适应宽度', exact: true }).click();
  await page.locator('.architecture-canvas').scrollIntoViewIfNeeded();
  await page.screenshot({
    path: '../../reports/console/model-architecture/qwen3-attention-desktop.png',
  });
  await page.getByRole('tab', { name: 'MLP 内部', exact: true }).click();
  await expect(page.getByRole('button', { name: 'silu，mlp.act_fn', exact: true })).toBeVisible();
  const [download] = await Promise.all([
    page.waitForEvent('download'),
    page.getByRole('button', { name: '导出结构快照', exact: true }).click(),
  ]);
  await download.saveAs('../../reports/console/model-architecture/exported-architecture.json');
  const snapshot = JSON.parse(
    await fs.readFile(
      '../../reports/console/model-architecture/exported-architecture.json',
      'utf8',
    ),
  );
  expect(snapshot.deployment_id).toBe(qwen.id);
  expect(snapshot.snapshot_sha256).toMatch(/^[a-f0-9]{64}$/);
  expect(errors).toEqual([]);
});

test('Gemma distinguishes local and global attention and its additional norms', async ({
  page,
  request,
}) => {
  const deployments = (await (await request.get('/api/console/v1/deployments')).json()).items;
  const gemma = deployments.find((row: { model: string }) => row.model === 'google/gemma-3-1b-it');
  test.skip(!gemma, 'requires the configured Gemma text artifact');
  await page.goto(`/model-architecture?deployment=${gemma.id}`);
  await expect(
    page.getByRole('button', { name: 'FFN 后 RMSNorm，post_feedforward_layernorm', exact: true }),
  ).toBeAttached();
  await page.getByRole('tab', { name: 'Attention 内部', exact: true }).click();
  await expect(page.getByText(/本层 sliding_attention/)).toBeVisible();
  await page.getByLabel('选择具体 Decoder 层', { exact: true }).press('ArrowDown');
  await page.getByText('Layer 5 · 全局 Attention', { exact: true }).click();
  await expect(page.getByText(/本层 full_attention/)).toBeVisible();
  await page.getByRole('tab', { name: 'Decoder Block', exact: true }).click();
  await page.getByRole('button', { name: '适应宽度', exact: true }).click();
  await page.locator('.architecture-canvas').scrollIntoViewIfNeeded();
  await page.screenshot({
    path: '../../reports/console/model-architecture/gemma3-block-desktop.png',
  });
});

test('architecture remains contained at narrow width', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/model-architecture?deployment=qwen-reference');
  await expect(page.getByRole('tab', { name: 'Decoder Block', exact: true })).toBeVisible();
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 2),
  ).toBe(true);
  await page.screenshot({ path: '../../reports/console/model-architecture/mobile.png' });
});
