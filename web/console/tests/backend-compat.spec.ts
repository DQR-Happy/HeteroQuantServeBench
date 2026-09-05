import { test, expect } from '@playwright/test';
import { readFileSync, existsSync } from 'node:fs';
import path from 'node:path';
import { enterWorkspace } from './session';
import { createHash } from 'node:crypto';

// Serve the new static UI in front of the existing API, without changing its
// code, data, port, authentication or running worker. No model jobs are started.
const target = process.env.HQSB_EXISTING_API_URL;
const archive = process.env.HQSB_EXISTING_REPORT;
const tokenFile = process.env.HQSB_EXISTING_TOKEN_FILE;
test.skip(!target || !archive, 'Requires the existing backend and its private archive');
test('redesigned static UI connects to the unchanged backend, sealed report and receipt download', async ({
  page,
}) => {
  const saved = JSON.parse(readFileSync(archive!, 'utf8'));
  const dist = path.resolve('dist');
  const errors: string[] = [];
  const calls: { method: string; path: string; status: number }[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('response', (response) => {
    const url = new URL(response.url());
    if (url.pathname.startsWith('/api/console/v1/'))
      calls.push({
        method: response.request().method(),
        path: url.pathname,
        status: response.status(),
      });
  });
  if (process.env.HQSB_USE_DEPLOYED_UI !== '1')
    await page.route(`${target}/**`, async (route) => {
      const pathname = new URL(route.request().url()).pathname;
      if (pathname.startsWith('/api/')) {
        await route.continue();
        return;
      }
      const asset = pathname.startsWith('/assets/') || pathname === '/favicon.svg';
      const filename = asset ? path.join(dist, pathname) : path.join(dist, 'index.html');
      if (!filename.startsWith(dist + path.sep) || !existsSync(filename)) {
        await route.abort();
        return;
      }
      const mime = filename.endsWith('.js')
        ? 'application/javascript'
        : filename.endsWith('.css')
          ? 'text/css'
          : filename.endsWith('.svg')
            ? 'image/svg+xml'
            : 'text/html';
      await route.fulfill({ status: 200, contentType: mime, body: readFileSync(filename) });
    });
  const requiresAuth = await enterWorkspace(
    page,
    `${target}/optimization?experiment=${saved.experiment_id}`,
    tokenFile,
  );
  await expect(page.getByText('数值合同 passed', { exact: true })).toBeVisible();
  await expect(page.locator('.workbench-footer')).toContainText('UI 0.4');
  await expect(page.getByText('采用判定：hold', { exact: true })).toBeVisible();
  await expect(page.getByText('已完成', { exact: true })).toHaveCount(6);
  await page.screenshot({
    path: '../../reports/console/v04/optimization-connected.png',
    fullPage: true,
  });
  await page.getByRole('tab', { name: '机制变化', exact: true }).click();
  await page.getByLabel('筛选机制对照指标').fill('reserved_bytes');
  await expect(
    page.getByRole('cell', { name: 'reserved_bytes', exact: true }).first(),
  ).toBeVisible();
  await page.getByRole('tab', { name: '性能与观测成本', exact: true }).click();
  await expect(page.getByText('insufficient_evidence', { exact: true })).toBeVisible();
  const quality = saved.jobs.find(
    (job: { role: string; arm: string }) => job.role === 'quality' && job.arm === 'candidate',
  );
  await page.goto(`${target}/runs/${quality.run_id}?tab=experiment`);
  await expect(page.getByText('执行凭据：applied', { exact: true })).toBeVisible();
  const response = await page.request.get(
    `${target}/api/console/v1/runs/${quality.run_id}/experiment`,
  );
  expect(response.ok()).toBeTruthy();
  const bundle = await response.json();
  const artifact = bundle.manifest.artifacts.find(
    (item: { path: string }) => item.path === 'execution-receipt.json',
  );
  const ready = page.waitForEvent('download');
  await page
    .getByRole('row')
    .filter({ has: page.getByRole('cell', { name: 'execution-receipt.json', exact: true }) })
    .getByRole('link', { name: '下载原件', exact: true })
    .click();
  const download = await ready;
  expect(
    createHash('sha256')
      .update(readFileSync((await download.path())!))
      .digest('hex'),
  ).toBe(artifact.sha256);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(`${target}/optimization?experiment=${saved.experiment_id}`);
  await expect(page.getByText('数值合同 passed', { exact: true })).toBeVisible();
  await expect
    .poll(() => page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth))
    .toBeLessThanOrEqual(1);
  await page.screenshot({
    path: '../../reports/console/v04/optimization-connected-mobile.png',
    fullPage: true,
  });
  expect(errors).toEqual([]);
  expect(calls.filter((call) => call.method === 'POST').map((call) => call.path)).toEqual(
    requiresAuth ? ['/api/console/v1/session/login'] : [],
  );
  expect(calls.filter((call) => call.status >= 400 && call.status !== 401)).toEqual([]);
});
