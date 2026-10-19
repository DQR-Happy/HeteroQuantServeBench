import { expect, test } from '@playwright/test';
const api = '/api/console/v1';

test('L1 prerequisite explains resident-model action and links to deployment', async ({ page }) => {
  await page.goto('/system/run?deployment=qwen-reference');
  const start = page.getByRole('button', { name: '运行 L1 配置对照', exact: true });
  await expect(start).toBeDisabled();
  const wrapper = page.locator('[data-disabled-action]').filter({ has: start });
  await wrapper.focus();
  await expect(
    page.getByText('L1 需要独立加载源模型。请先在设备与部署页卸载驻留模型。', { exact: true }),
  ).toBeVisible();
  await expect(page.getByRole('link', { name: '卸载模型 →', exact: true })).toHaveAttribute(
    'href',
    '/devices#deployment-qwen-reference',
  );
  await page.screenshot({
    path: '../../reports/console/l1-system/prerequisite-desktop.png',
    fullPage: true,
  });
});

test('real L1 record shows paired metrics, applied settings and mobile evidence', async ({
  page,
}) => {
  const rows = (await (await page.request.get(api + '/system/jobs')).json()).items;
  const run = rows.find(
    (r: any) => r.state === 'completed' && r.config.deployment.id === 'qwen-reference',
  );
  expect(run).toBeTruthy();
  await page.goto('/runs/' + run.id);
  await expect(page).toHaveURL(/\/system\/results\?/);
  await expect(page.getByText('固定输入的逐 Token 输出一致', { exact: true })).toBeVisible();
  await expect(page.getByText('进程清理 / 功耗恢复：succeeded', { exact: false })).toBeVisible();
  await expect(page.getByRole('columnheader', { name: '候选 − 基线', exact: true })).toBeVisible();
  await expect(page.getByText('生成 Token/s', { exact: true })).toBeVisible();
  await page.getByRole('link', { name: '硬件与能力', exact: true }).click();
  await expect(
    page.getByRole('heading', { name: '硬件事实与支持边界', exact: true }),
  ).toBeVisible();
  await expect(page.getByText('NVIDIA GeForce RTX 3090', { exact: true })).toBeVisible();
  await page.getByRole('link', { name: '结果与观测', exact: true }).first().click();
  await expect(page.getByText('固定输入的逐 Token 输出一致', { exact: true })).toBeVisible();
  await page.screenshot({
    path: '../../reports/console/l1-system/results-desktop.png',
    fullPage: true,
  });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByRole('heading', { name: '配置对照结果', exact: true })).toBeVisible();
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1),
  ).toBeTruthy();
  await page.screenshot({
    path: '../../reports/console/l1-system/results-mobile.png',
    fullPage: true,
  });
});
