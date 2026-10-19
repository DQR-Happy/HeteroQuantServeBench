import { expect, test } from '@playwright/test';

// UI fixtures only; real CUDA/API acceptance lives in validate_l4.py.
test('L4 exposes recipe, calibration and actual proof without claiming FP8 support', async ({
  page,
}) => {
  test.setTimeout(45000);
  const methods = [
    { id: 'none', name: 'Floating point' },
    { id: 'gptq', name: 'GPTQ', requires_calibration: true },
    { id: 'bnb_nf4', name: 'NormalFloat4 / double quant' },
    {
      id: 'fp8',
      name: 'Dynamic E4M3 W8A8',
      availability: 'unsupported',
      reasons: ['DEVICE_SM_BELOW_REQUIRED'],
    },
  ].map((m) => ({
    availability: 'available_for_target_validation',
    source: 'https://github.com/IST-DASLab/gptq',
    ...m,
  }));
  const capability = {
    compute_dtypes: ['float32', 'float16', 'bfloat16'],
    methods,
    effective: {
      method: 'none',
      compute_dtype: 'torch.float16',
      execution: 'framework',
      quantized_module_count: 0,
      quantized_storage_bytes: 0,
      remaining_float_parameter_bytes: 30000000,
    },
  };
  await page.route('**/api/console/v1/deployments', (route) =>
    route.fulfill({
      json: {
        items: [
          {
            id: 'fixture-tiny',
            name: 'Fixture Tiny',
            provider: 'pytorch',
            state: 'ready',
            epoch: 'fixture-epoch',
            kernel_profile: 'framework-native',
            precision: 'float16',
            context_limit: 256,
          },
        ],
      },
    }),
  );
  await page.route('**/api/console/v1/precision/catalog', (route) =>
    route.fulfill({ json: capability }),
  );
  await page.route('**/api/console/v1/deployments/*/precision-capabilities', (route) =>
    route.fulfill({ json: capability }),
  );
  await page.route('**/api/console/v1/precision/artifacts', (route) =>
    route.fulfill({ json: { items: [] } }),
  );
  await page.goto('/precision');
  await page.getByLabel('质量评测数据集', { exact: true }).first().click();
  await page.getByTitle('自定义短文本 · logits / 标准答案', { exact: true }).click();
  await expect(page.getByRole('heading', { name: '表示精度与量化', exact: true })).toBeVisible();
  await expect(page.getByText('torch.float16', { exact: true })).toBeVisible();
  await page.getByLabel('量化方法', { exact: true }).first().click();
  await expect(page.getByTitle('DEVICE_SM_BELOW_REQUIRED')).toHaveClass(/disabled/);
  await page.getByTitle('GPTQ', { exact: true }).click();
  await expect(page.getByLabel('校准文本', { exact: true })).toBeVisible();
  await expect(page.getByRole('combobox', { name: '权重位宽', exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: '应用方案并重载模型' })).toBeEnabled();
  await page.getByLabel('计算精度', { exact: true }).first().click();
  await page.getByTitle('float32', { exact: true }).click();
  await expect(page.getByRole('button', { name: '应用方案并重载模型' })).toBeDisabled();
  await expect(page.getByText(/量化执行请选择 FP16/)).toBeVisible();
  let submitted: { quality_plan?: { samples: { expected?: string }[] } } = {};
  const finished = {
    id: 'fixture-quality',
    kind: 'generate',
    state: 'completed',
    cleanup: 'succeeded',
    config: {},
    output: '',
    metrics: { quality_evaluation: { status: 'complete', perplexity: 1.5 } },
  };
  await page.route('**/api/console/v1/precision/quality-jobs', (route) => {
    submitted = route.request().postDataJSON();
    return route.fulfill({ json: finished });
  });
  await page.route('**/api/console/v1/runs/fixture-quality', (route) =>
    route.fulfill({ json: finished }),
  );
  await page.getByText('带标准答案的质量样本（可选）', { exact: true }).click();
  await page
    .getByLabel('带标签质量样本 JSON', { exact: true })
    .fill('[{"id":"math","text":"1+1=","expected":"2"}]');
  await page.getByRole('button', { name: '运行独立质量评测', exact: true }).click();
  await expect.poll(() => submitted.quality_plan?.samples[0].expected).toBe('2');
  await page.screenshot({ path: '../../reports/console/l4/desktop.png', fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({ path: '../../reports/console/l4/mobile.png', fullPage: true });
});
