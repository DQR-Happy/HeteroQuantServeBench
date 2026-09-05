import { test, expect } from '@playwright/test';

test('test workspace opens directly, reloads and changes routes without a token', async ({
  page,
}) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  const session = await page.request.get('/api/console/v1/session');
  expect(session.status()).toBe(200);
  expect((await session.json()).auth_required).toBe(false);
  for (const route of ['/optimization', '/runs', '/optimization']) {
    await page.goto(route);
    await expect(page.locator('.workbench-shell')).toBeVisible();
    await expect(page.getByLabel('访问令牌', { exact: true })).toHaveCount(0);
    await expect(page.getByRole('button', { name: '退出会话' })).toHaveCount(0);
  }
  await page.reload();
  await expect(page.getByRole('heading', { name: '优化实验', exact: true })).toBeVisible();
  await page.screenshot({ path: '../../reports/console/no-auth/workspace.png', fullPage: true });
  expect(errors).toEqual([]);
});

test('connection errors offer reconnect and never ask for a token', async ({ page }) => {
  let unavailable = true;
  await page.route('**/api/console/v1/session', (route) =>
    unavailable
      ? route.fulfill({ status: 503, json: { error: { message: 'Temporary test outage' } } })
      : route.continue(),
  );
  await page.goto('/optimization');
  await expect(page.getByText('数据读取失败', { exact: true })).toBeVisible();
  await expect(page.getByLabel('访问令牌', { exact: true })).toHaveCount(0);
  unavailable = false;
  await page.getByRole('button', { name: '重新连接', exact: true }).click();
  await expect(page.getByRole('heading', { name: '优化实验', exact: true })).toBeVisible();
});

test('an explicitly authenticated backend still shows the login screen', async ({ page }) => {
  await page.route('**/api/console/v1/session', (route) =>
    route.fulfill({ status: 401, json: { error: { message: 'Authentication required' } } }),
  );
  await page.goto('/optimization');
  await expect(page.getByLabel('访问令牌', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: '进入工作台' })).toBeDisabled();
});
