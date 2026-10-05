import { test, expect } from '@playwright/test';

test.use({ actionTimeout: 15000 });

// Browser-only API fixtures: never submit a real model request or change evidence.
test('inference renders streaming Markdown, preserves source on copy and contains wide content', async ({
  page,
}) => {
  const errors: string[] = [];
  const fontResponses: { url: string; status: number }[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('response', (response) => {
    if (/KaTeX_.*\.(woff2?|ttf)(?:\?|$)/.test(response.url()))
      fontResponses.push({ url: response.url(), status: response.status() });
  });
  await page.addInitScript(() => {
    Object.defineProperty(navigator, 'clipboard', {
      value: {
        writeText: async (text: string) => {
          document.documentElement.dataset.copied = text;
        },
      },
    });
  });
  const partial =
    '## 带宽分析\n\n正在生成 **示例**：\n\n若 $x^2 = 16$，则 $x = \\pm 4$，和为 $\\boxed{0}$。\n\n```python\nprint("hello")';
  const unfinishedMath = '## 带宽分析\n\n公式生成中：$\\frac{1}{';
  const source = [
    partial,
    '\n```',
    '',
    '$$',
    '\\frac{1}{2} + \\sum_{i=1}^{n} i = \\frac{n(n+1)+1}{2}',
    '$$',
    '',
    '$$',
    Array.from({ length: 40 }, (_, index) => `x_{${index}}`).join(' + ') + '= 0',
    '$$',
    '',
    '无效公式仍保留内容：$\\frac{1}{$',
    '',
    '代码中的公式保留原文：`$x^2$`。',
    '',
    '> 权重读取受内存带宽限制。',
    '',
    '- **权重**：读取模型参数',
    '- 使用 `KV cache` 复用历史状态',
    '',
    '1. 固定输入',
    '2. 对照运行',
    '',
    '- [x] 已完成预热',
    '- [ ] 等待对照实验',
    '',
    '~~旧结论~~ [参考](https://example.com/reference)',
    '',
    '| 指标 | 数值 |',
    '| --- | ---: |',
    '| 内存 | 256 MiB |',
    '',
    '```text',
    'long-code-line-'.repeat(24),
    '```',
    '',
    '| 宽列 | 数值 |',
    '| --- | --- |',
    `| \`${'wide-table-cell-'.repeat(24)}\` | 保留完整数据 |`,
    '',
    '<script>window.markdownUnsafe = true</script>',
    '',
    '[危险链接](javascript:alert(1))',
  ].join('\n');
  let stage = 0;
  let submitted = 0;
  const run = () => ({
    id: 'markdown-fixture',
    state: stage === 2 ? 'completed' : 'running',
    kind: 'inference',
    deployment_id: 'markdown-model',
    output: stage === 2 ? source : stage === 1 ? unfinishedMath : partial,
    seq: stage + 1,
    config: {},
    metrics: { output_tokens: stage === 2 ? 128 : 16 },
    cleanup: stage === 2 ? 'completed' : 'pending',
  });
  await page.route('**/api/console/v1/**', async (route) => {
    const url = new URL(route.request().url());
    const path = url.pathname.replace('/api/console/v1', '');
    let body: unknown;
    if (path === '/session')
      body = { auth_required: false, version: 'markdown-fixture', limits: {} };
    else if (path === '/machines') body = { items: [{ id: 'local', name: 'Markdown 测试节点' }] };
    else if (path === '/deployments')
      body = {
        items: [
          {
            id: 'markdown-model',
            name: 'Markdown 测试模型（模拟数据）',
            provider: 'openai',
            state: 'ready',
            epoch: 'fixture-epoch',
            context_limit: 8192,
            max_output_tokens: 4096,
            detail: {},
          },
        ],
      };
    else if (path === '/requests' && route.request().method() === 'POST') {
      submitted++;
      expect(route.request().postDataJSON().messages.at(-1).content).toBe('Markdown 展示测试');
      body = { ...run(), output: '', seq: 0 };
    } else if (path === '/runs/markdown-fixture') body = run();
    else if (path === '/requests/markdown-fixture/events') {
      const snapshot = run();
      const event = {
        event_version: '1',
        request_id: snapshot.id,
        seq: snapshot.seq,
        kind: 'output.snapshot',
        data: { text: snapshot.output },
      };
      await route.fulfill({
        contentType: 'text/event-stream',
        body: `data: ${JSON.stringify(event)}\n\n`,
      });
      return;
    } else {
      throw new Error(`Unexpected fixture API call: ${path}`);
    }
    await route.fulfill({ json: body });
  });
  await page.goto('/playground');
  await page.getByLabel('推理输入').fill('Markdown 展示测试');
  await page.getByRole('button', { name: /开始推理$/ }).click();
  const output = page.getByLabel('模型输出内容');
  await expect(output.getByRole('heading', { name: '带宽分析' })).toBeVisible();
  await expect(output.locator('pre code')).toHaveText('print("hello")\n');
  await expect(output).toHaveAttribute('aria-busy', 'true');
  await expect(output.locator('.katex')).toHaveCount(3);
  await expect(output.locator('.katex-html').first()).toBeVisible();
  stage = 1;
  await expect(output).toContainText('公式生成中：$\\frac{1}{');
  await expect(output).toHaveAttribute('aria-busy', 'true');
  stage = 2;
  await expect(output).toHaveAttribute('aria-busy', 'false');
  await expect(output.locator('.katex-display')).toHaveCount(2);
  await expect(output.locator('.katex annotation').first()).toHaveText('x^2 = 16');
  await expect(output.locator('.katex-error')).toContainText('\\frac{1}{');
  await expect(output.locator('p > code').filter({ hasText: '$x^2$' })).toHaveText('$x^2$');
  await page.evaluate(() => document.fonts.ready.then(() => undefined));
  expect(fontResponses.length).toBeGreaterThan(0);
  expect(fontResponses.every((font) => font.status < 400)).toBe(true);
  expect(
    fontResponses.every((font) => new URL(font.url).origin === new URL(page.url()).origin),
  ).toBe(true);
  await expect(output.locator('strong').first()).toHaveText('示例');
  await expect(output.locator('blockquote')).toContainText('权重读取');
  await expect(output.locator('ul > li').first()).toHaveCSS('list-style-type', 'disc');
  await expect(output.locator('ol > li').first()).toHaveCSS('list-style-type', 'decimal');
  await expect(output.getByRole('cell', { name: '256 MiB' })).toBeVisible();
  await expect(output.locator('del')).toHaveText('旧结论');
  await expect(output.getByRole('checkbox').first()).toBeChecked();
  await expect(output.getByRole('checkbox').first()).toBeDisabled();
  await expect(output.getByRole('link', { name: '参考' })).toHaveAttribute(
    'href',
    'https://example.com/reference',
  );
  await expect(output.locator('script')).toHaveCount(0);
  await expect(output.getByText('危险链接')).not.toHaveAttribute('href', /^javascript:/i);
  await page.getByRole('button', { name: '复制输出', exact: true }).click();
  await expect(page.locator('html')).toHaveAttribute('data-copied', source);
  await page.screenshot({
    path: '../../reports/console/markdown-output/desktop.png',
    fullPage: true,
  });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect
    .poll(() => page.evaluate(() => document.documentElement.scrollWidth - innerWidth))
    .toBeLessThanOrEqual(1);
  await expect(output.locator('pre').last()).toHaveCSS('white-space', 'pre');
  expect(
    await output
      .locator('pre')
      .last()
      .evaluate((element) => element.scrollWidth > element.clientWidth),
  ).toBe(true);
  expect(
    await output
      .locator('.katex-display')
      .last()
      .evaluate((element) => element.scrollWidth > element.clientWidth),
  ).toBe(true);
  expect(
    await output
      .getByRole('region', { name: '输出表格' })
      .last()
      .evaluate((element) => element.scrollWidth > element.clientWidth),
  ).toBe(true);
  await output.scrollIntoViewIfNeeded();
  await page.screenshot({
    path: '../../reports/console/markdown-output/mobile.png',
    fullPage: true,
  });
  expect(submitted).toBe(1);
  expect(errors).toEqual([]);
});
