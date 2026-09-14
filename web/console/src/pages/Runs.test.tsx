import { renderToStaticMarkup } from 'react-dom/server';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { describe, expect, it, vi } from 'vitest';
import type { Run } from '../api/types';
import { RunDetail } from './Runs';

const state = vi.hoisted(() => ({ run: undefined as Run | undefined }));
vi.mock('../hooks', () => ({ useRun: () => ({ data: state.run, isPending: false }) }));

function view(tab = '') {
  return renderToStaticMarkup(
    <MemoryRouter initialEntries={[`/runs/load-fixture${tab}`]}>
      <Routes>
        <Route path="/runs/:id" element={<RunDetail />} />
      </Routes>
    </MemoryRouter>,
  );
}

function loadedRun(): Run {
  return {
    id: 'load-fixture',
    kind: 'load',
    state: 'completed',
    created_at: 1,
    updated_at: 2,
    seq: 3,
    config: {
      deployment: {
        id: 'qwen3-8b',
        name: 'Qwen3 8B',
        model: 'Qwen/Qwen3-8B',
        precision: 'float16',
        kernel: 'framework-native',
      } as Run['config']['deployment'],
      epoch: null,
    },
    output: '',
    metrics: { console_e2e_ms: 100065.8 },
    error: null,
    cleanup: 'succeeded',
    quality: 'not_evaluated',
    result: {
      load_seconds: 80.9065,
      parameter_bytes: 16381470720,
      model_class: 'Qwen3ForCausalLM',
      device_name: 'NVIDIA GeForce RTX 4090 D',
      parameter_devices: ['cuda:0'],
      weight_verification: 'manifest_verified',
      memory: { peak_allocated_bytes: 16393476096 },
      parameter_inventory: { entry_count: 401, availability: 'complete' },
      observation: {
        schema_version: '1.0',
        mode: 'basic',
        clock_domain: 'worker_monotonic',
        time_origin: 'request_received_by_worker',
        tokens: [],
        execution: {},
        limitations: [],
        phases: [
          { name: 'model_load', start_ms: 0.1, duration_ms: 95960.78, source: 'host_monotonic' },
        ],
        memory_snapshots: [
          {
            label: 'after_warmup',
            time: 1,
            start_ms: 96348,
            host_total_bytes: null,
            host_available_bytes: null,
            allocated_bytes: 16392107008,
            reserved_bytes: 16557015040,
            peak_allocated_bytes: 16393476096,
            scope: 'worker_allocator',
            source: 'torch_cuda_allocator',
          },
        ],
      },
    },
  };
}

describe('lifecycle run details', () => {
  it('shows saved load results and observations even with generation metrics absent', () => {
    state.run = loadedRun();
    const html = view();
    expect(html).toContain('生命周期结果与观测');
    expect(html).toContain('Qwen3ForCausalLM');
    expect(html).toContain('80.9');
    expect(html).toContain('15.256');
    expect(html).toContain('manifest_verified');
    expect(html).toContain('401 项');
    expect(html).toContain('model_load');
    expect(html).toContain('after_warmup');
    expect(html).toContain('仍需在评测页点击');
    expect(html).not.toContain('此任务没有文本输出');
    expect(html).not.toContain('服务端首内容');
  });

  it('exposes the actual result in raw data and handles former generation-only tabs', () => {
    state.run = loadedRun();
    expect(view('?tab=metrics')).toContain('16381470720');
    expect(view('?tab=metrics')).toContain('parameter_inventory');
    expect(view('?tab=experiment')).toContain('Qwen3ForCausalLM');
    expect(view('?tab=observation')).toContain('after_warmup');
  });

  it('keeps absent lifecycle results unknown and does not invent generation evidence', () => {
    state.run = { ...loadedRun(), state: 'failed', error: 'load failed', result: undefined };
    const html = view();
    expect(html).toContain('load failed');
    expect(html).toContain('尚无结果');
    expect(html).toContain('没有阶段观测');
    expect(html).not.toContain('manifest_verified');
    expect(html).not.toContain('15.256');
  });

  it('preserves text and generation metrics for actual inference runs', () => {
    state.run = {
      ...loadedRun(),
      kind: 'generate',
      result: undefined,
      output: 'saved generated answer',
      metrics: {
        console_first_content_ms: 123.4,
        runtime_first_token_ms: 100,
        console_e2e_ms: 456,
      },
    };
    const html = view();
    expect(html).toContain('saved generated answer');
    expect(html).toContain('服务端首内容');
    expect(html).toContain('123.4');
    expect(html).not.toContain('生命周期结果与观测');
  });

  it('identifies unload records without inventing load measurements', () => {
    state.run = { ...loadedRun(), kind: 'unload', result: undefined };
    const html = view();
    expect(html).toContain('模型卸载记录');
    expect(html).not.toContain('参数存储');
    expect(html).not.toContain('Qwen3ForCausalLM');
  });
});
