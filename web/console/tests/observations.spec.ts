import { test, expect } from '@playwright/test';
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { enterWorkspace } from './session';

const reportFile =
  process.env.HQSB_OBSERVATION_REPORT ??
  path.resolve('../../reports/observation/console-validation-final-code.json');
const report: { selected_runs: Record<string, string> } | undefined = existsSync(reportFile)
  ? JSON.parse(readFileSync(reportFile, 'utf8'))
  : undefined;

test('real sealed memory evidence is visible and downloadable after model unload', async ({
  page,
}) => {
  if (!report) {
    test.skip(true, 'Requires a real validate_console.py archive via HQSB_OBSERVATION_REPORT');
    return;
  }
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await enterWorkspace(page);
  await expect(page.getByRole('heading', { name: '工作空间', exact: true })).toBeVisible();
  await page.goto(`/runs/${report.selected_runs.memory}`);
  await page.getByRole('tab', { name: '全链路观测', exact: true }).click();
  await expect(page.getByText('内存诊断', { exact: true })).toBeVisible();
  await page.getByRole('tab', { name: '布局与系统机制', exact: true }).click();
  await expect(page.getByText('采集 complete · supervisor_acknowledged')).toBeVisible();
  await page.getByLabel('搜索观测指标').fill('cpu_minor_fault');
  await expect(
    page.getByRole('cell', { name: 'cpu_minor_fault_delta', exact: true }),
  ).toBeVisible();
  await page.getByRole('tab', { name: '申请与回收', exact: true }).click();
  await page.getByRole('tab', { name: 'Tensor 关联', exact: true }).click();
  await expect(
    page.getByRole('cell', { name: 'snapshot_correlated', exact: true }).first(),
  ).toBeVisible();
  await page.getByRole('tab', { name: '原始证据', exact: true }).click();
  const manifestReady = page.waitForEvent('download');
  await page.getByRole('link', { name: '下载 manifest（含源码哈希）', exact: true }).click();
  const manifestDownload = await manifestReady;
  const manifest = JSON.parse(readFileSync((await manifestDownload.path())!, 'utf8')) as {
    context: {
      run_id: string;
      host_time_origin_ns: string;
      source: { source_files: Record<string, string> };
    };
  };
  expect(manifest.context.run_id).toBe(report.selected_runs.memory);
  expect(manifest.context.host_time_origin_ns).toMatch(/^\d+$/);
  expect(Object.keys(manifest.context.source.source_files).length).toBeGreaterThan(0);
  const ready = page.waitForEvent('download');
  await page
    .getByRole('link', { name: /observations\/records/ })
    .first()
    .click();
  const download = await ready;
  const digest = createHash('sha256')
    .update(readFileSync((await download.path())!))
    .digest('hex');
  await expect(page.getByRole('cell', { name: digest, exact: true })).toBeVisible();
  await page.screenshot({ path: '../../reports/observation/console-memory.png', fullPage: true });
  expect(errors).toEqual([]);
});
