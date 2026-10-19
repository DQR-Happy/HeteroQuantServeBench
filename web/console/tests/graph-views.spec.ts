import { expect, test } from '@playwright/test';

test('sealed graph views can switch representation without starting work', async ({
  page,
  request,
}) => {
  const summary = await request.get(
    '/api/console/v1/runtime/runs?deployment_id=ms-586060631c81c17023c0',
  );
  const runs = (await summary.json()).items;
  let selected: { run_id: string; views: { kind: string }[] } | undefined;
  for (const run of runs.filter((r: { state: string }) => r.state === 'completed')) {
    const result = await request.get(`/api/console/v1/runs/${run.run_id}/graph-views`);
    if (!result.ok()) continue;
    const doc = await result.json();
    if (
      doc.views?.some(
        (v: { kind: string; status: string }) =>
          v.kind === 'inductor_post' && v.status === 'available',
      )
    ) {
      selected = { run_id: run.run_id, views: doc.views };
      break;
    }
  }
  test.skip(!selected, 'requires real sealed graph diagnostic acceptance');
  const writes: string[] = [],
    errors: string[] = [];
  page.on('request', (r) => {
    if (r.method() === 'POST') writes.push(r.url());
  });
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto(
    `/model-architecture?deployment=ms-586060631c81c17023c0&runtime_run=${selected!.run_id}&graph=inductor_post`,
  );
  await expect(page.locator('[data-architecture-kind="inductor_post"]')).toBeVisible();
  await expect(page.getByRole('link', { name: '下载原始编译文件', exact: true })).toBeVisible();
  await page.screenshot({
    path: '../../reports/console/graph-views/inductor-post.png',
    fullPage: true,
  });
  await page.getByRole('combobox', { name: '计算图类型', exact: true }).press('ArrowDown');
  await page.getByText('Inductor IR · 融合前', { exact: true }).last().click();
  await expect(page.locator('[data-architecture-kind="inductor_pre"]')).toBeVisible();
  await page.getByRole('combobox', { name: '计算图类型', exact: true }).press('ArrowDown');
  await page.getByText('Inductor · 生成代码', { exact: true }).last().click();
  await page.getByText('查看生成的 Triton / C++ / Python 代码', { exact: true }).click();
  await expect(page.locator('.graph-source')).toBeVisible();
  await page.screenshot({
    path: '../../reports/console/graph-views/generated-code.png',
    fullPage: true,
  });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect
    .poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth))
    .toBe(true);
  expect(writes).toEqual([]);
  expect(errors).toEqual([]);
});

test('ONNX shows exported nodes and historical missing views explicitly', async ({
  page,
  request,
}) => {
  const runs = (
    await (
      await request.get('/api/console/v1/runtime/runs?deployment_id=ms-586060631c81c17023c0')
    ).json()
  ).items;
  let identity = '';
  for (const r of runs.filter((r: { state: string }) => r.state === 'completed')) {
    const v = await request.get(`/api/console/v1/runs/${r.run_id}/graph-views`);
    if (
      v.ok() &&
      (await v.json()).views?.some(
        (v: { kind: string; status: string }) => v.kind === 'onnx' && v.status === 'available',
      )
    ) {
      identity = r.run_id;
      break;
    }
  }
  test.skip(!identity, 'requires real ONNX export');
  await page.goto(
    `/model-architecture?deployment=ms-586060631c81c17023c0&runtime_run=${identity}&graph=onnx`,
  );
  await expect(page.locator('[data-architecture-kind="onnx"]')).toBeVisible();
  await expect(page.getByText('ONNX checker: passed', { exact: true })).toBeVisible();
  await page.screenshot({ path: '../../reports/console/graph-views/onnx.png', fullPage: true });
  await page.getByRole('combobox', { name: '计算图类型', exact: true }).press('ArrowDown');
  await page.getByText('Inductor IR · 融合后', { exact: true }).last().click();
  await expect(page.getByText('此记录未采集所选图类型', { exact: true })).toBeVisible();
});
