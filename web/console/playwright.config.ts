import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir: './tests',
  workers: 1,
  timeout: 180000,
  expect: { timeout: 15000 },
  reporter: [['list'], ['json', { outputFile: '../../reports/console/browser-results.json' }]],
  use: {
    baseURL: process.env.HQSB_CONSOLE_URL ?? 'http://127.0.0.1:8765',
    viewport: { width: 1512, height: 982 },
    screenshot: 'only-on-failure',
    trace: 'retain-on-failure',
    headless: true,
  },
  outputDir: 'test-results',
});
