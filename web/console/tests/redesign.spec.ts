import { test, expect, type Page } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { enterWorkspace } from './session';
const report = JSON.parse(
  readFileSync(
    process.env.HQSB_OBSERVATION_REPORT ??
      '../../reports/observation/console-validation-final-code.json',
    'utf8',
  ),
);
async function login(page: Page) {
  await enterWorkspace(page);
  await expect(page.getByRole('heading', { name: '工作空间', exact: true })).toBeVisible();
}
test.beforeEach(async ({ page }) => {
  await login(page);
});

test('project navigation, jump dialog and URL-backed run filters retain context', async ({
  page,
}) => {
  await page.getByRole('button', { name: '快速跳转', exact: true }).click();
  await expect(page.getByLabel('搜索工作区')).toBeFocused();
  await page.getByLabel('搜索工作区').fill('运行记录');
  await page.keyboard.press('Enter');
  await expect(page).toHaveURL(/\/runs$/);
  await expect(page.getByRole('dialog')).toHaveCount(0);
  await page.getByRole('button', { name: /^已完成/ }).click();
  await page.getByLabel('搜索本页运行').fill(report.selected_runs.memory.slice(0, 18));
  await expect(page.getByRole('table')).toContainText(report.selected_runs.memory.slice(0, 18));
  await page.reload();
  await expect(page.getByLabel('搜索本页运行')).toHaveValue(
    report.selected_runs.memory.slice(0, 18),
  );
  await expect(page.getByRole('button', { name: /^已完成/ })).toHaveAttribute(
    'aria-pressed',
    'true',
  );
  await page.getByRole('button', { name: '清除筛选' }).click();
  await expect(page).toHaveURL(/\/runs$/);
  await page.screenshot({ path: '../../reports/console/v04/runs-desktop.png', fullPage: true });
  await page.getByRole('button', { name: '快速跳转', exact: true }).click();
  await page.getByLabel('搜索工作区').fill('no-such-workspace');
  await expect(page.getByText(/没有匹配的工作区/)).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(page.getByRole('button', { name: '快速跳转', exact: true })).toBeFocused();
});

test('observation directory opens the immutable object and returns to the catalogue', async ({
  page,
}) => {
  await page.goto('/observations');
  await page.getByLabel('筛选观测记录').fill(report.selected_runs.memory);
  await page.getByRole('link', { name: '检查证据 →', exact: true }).click();
  await expect(page).toHaveURL(new RegExp(`run=${report.selected_runs.memory}`));
  await expect(page.getByLabel('搜索观测指标')).toBeVisible();
  await page.getByRole('button', { name: '返回观测目录' }).click();
  await expect(page).toHaveURL(/\/observations$/);
});

test('optimization notebook reads the existing catalogue; malformed drafts remain editable', async ({
  page,
}) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/optimization');
  await expect(page.getByRole('heading', { name: '优化实验', exact: true })).toBeVisible();
  const editor = page.getByLabel('实验计划 JSON');
  await expect(editor).toBeVisible();
  await page.screenshot({
    path: '../../reports/console/v04/optimization-desktop.png',
    fullPage: true,
  });
  const saved = await editor.inputValue();
  await editor.fill('{invalid');
  await page.getByRole('button', { name: '冻结并注册计划' }).click();
  await expect(page.getByText('操作未完成', { exact: true })).toBeVisible();
  await expect(editor).toHaveValue('{invalid');
  await editor.fill(saved);
  await page.getByText('浏览十层优化能力目录', { exact: true }).click();
  await expect(page.getByRole('heading', { name: '十层优化能力目录', exact: true })).toBeVisible();
  expect(errors).toEqual([]);
});

test('all workspaces fit a 200 percent layout viewport and mobile navigation retains focus', async ({
  page,
}) => {
  // 720 CSS px is the layout viewport of a 1440px display at 200% zoom.
  // This checks reflow, not native browser zoom/OS scaling equivalence.
  await page.setViewportSize({ width: 720, height: 450 });
  for (const route of [
    '/',
    '/runs',
    '/optimization',
    '/evidence',
    '/devices',
    '/research',
    '/settings',
    `/observations?run=${report.selected_runs.memory}&view=kv`,
  ]) {
    await page.goto(route);
    await expect(page.locator('h1')).toBeVisible();
    await expect
      .poll(() => page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth))
      .toBeLessThanOrEqual(1);
  }
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole('button', { name: '打开导航' }).click();
  await page.getByRole('dialog').getByRole('link', { name: '优化实验', exact: true }).click();
  await expect(page).toHaveURL(/\/optimization$/);
  await expect(page.getByRole('dialog')).toHaveCount(0);
  await expect(page.getByLabel('实验计划 JSON')).toBeVisible();
  await expect
    .poll(() => page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth))
    .toBeLessThanOrEqual(1);
});
