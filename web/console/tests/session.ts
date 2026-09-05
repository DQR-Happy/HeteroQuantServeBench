import { expect, type Page } from '@playwright/test';
import { readFileSync } from 'node:fs';
import path from 'node:path';

export async function enterWorkspace(page: Page, destination = '/', tokenFile?: string) {
  await page.goto(destination);
  const response = await page.request.get(new URL('/api/console/v1/session', page.url()).href);
  if (response.status() === 401) {
    const token = readFileSync(
      tokenFile ??
        process.env.HQSB_CONSOLE_TOKEN_FILE ??
        path.resolve('../../.console/access-token'),
      'utf8',
    ).trim();
    await page.getByLabel('访问令牌', { exact: true }).fill(token);
    await page.getByRole('button', { name: '进入工作台' }).click();
  } else {
    expect(response.ok()).toBeTruthy();
  }
  return response.status() === 401;
}

export async function checkSessionExit(page: Page) {
  const response = await page.request.get('/api/console/v1/session');
  expect(response.ok()).toBeTruthy();
  if ((await response.json()).auth_required === false) {
    await expect(page.getByRole('button', { name: '退出会话' })).toHaveCount(0);
    await page.reload();
    await expect(page.locator('.workbench-shell')).toBeVisible();
    await expect(page.getByLabel('访问令牌', { exact: true })).toHaveCount(0);
  } else {
    await page.getByRole('button', { name: '退出会话' }).click();
    await expect(page.getByLabel('访问令牌', { exact: true })).toBeVisible();
  }
}
