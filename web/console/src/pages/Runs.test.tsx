import { renderToStaticMarkup } from 'react-dom/server';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { Bundle } from '../features/experiments/types';
import { describe, expect, it, vi } from 'vitest';
import type { Run } from '../api/types';
import { RunDetail } from './Runs';

const state = vi.hoisted(() => ({ run: undefined as Run | undefined }));
vi.mock('../hooks', () => ({ useRun: () => ({ data: state.run, isPending: false }) }));

function view(tab = '', bundle?: Bundle) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  if (bundle) client.setQueryData(['experiment-bundle', state.run!.id], bundle);
  return renderToStaticMarkup(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[`/runs/load-fixture${tab}`]}>
        <Routes>
          <Route path="/runs/:id" element={<RunDetail />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
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

function qualityRun(): Run {
  return {
    ...loadedRun(),
    kind: 'generate',
    result: undefined,
    config: { ...loadedRun().config, experiment_role: 'quality', observation_mode: 'off' },
    metrics: {
      console_e2e_ms: 12345,
      quality_evaluation: {
        status: 'complete',
        score_mode: 'corpus_nll',
        benchmark_id: 'wikitext2-mini-v1',
        perplexity: 172.482506,
        mean_nll: 5.150296,
        scored_tokens: 504,
        document_count: 8,
      },
    },
  };
}

describe('independent quality run details', () => {
  it('defaults to saved scores instead of generation placeholders, including old output URLs', () => {
    state.run = qualityRun();
    for (const tab of ['', '?tab=output']) {
      const html = view(tab);
      expect(html).toContain('独立质量评测详情');
      expect(html).toContain('172.4825');
      expect(html).toContain('5.150296');
      expect(html).toContain('504');
      expect(html).toContain('质量评分与样本');
      expect(html).not.toContain('此任务没有文本输出');
      expect(html).not.toContain('服务端首内容');
      expect(html).not.toContain('峰值分配内存');
    }
  });

  it('renders sealed document scores and correctly scoped allocator snapshots', () => {
    state.run = qualityRun();
    const bundle = {
      manifest_sha256: 'sealed',
      manifest: { status: 'complete', publication: 'acknowledged', artifacts: [] },
      receipt: {
        status: 'applied',
        requested: {},
        resolved: {},
        effective: {},
        plan_sha256: 'plan',
        measurement_role: 'quality',
        proof_scope: 'saved_fixture',
        experiment_links: {},
        proof: {
          after_cleanup_allocator: {
            allocated_bytes: 1073741824,
            reserved_bytes: 2147483648,
            peak_allocated_bytes: 3221225472,
          },
        },
        precision_receipt: {
          method: 'gptq',
          compute_dtype: 'float16',
          kv: { bits: 2, backend: 'quanto', residual_length: 128 },
        },
      },
      quality: {
        ...state.run.metrics.quality_evaluation,
        dataset_sha256: 'sealed-data-hash',
        protocol: { forward_mode: 'incremental_decode', max_tokens: 64 },
        samples: [
          {
            id: 'science',
            slice: 'Saved science article',
            input_tokens: 4,
            scored_tokens: 3,
            nll_sum: 6,
            token_nll: [1, 2, 3],
          },
        ],
      },
    } as Bundle;
    const html = view('', bundle);
    expect(html).toContain('Saved science article');
    expect(html).toContain('7.389');
    expect(html).toContain('sealed-data-hash');
    expect(html).toContain('quanto INT2');
    expect(html).toContain('逐 Token NLL');
    expect(html).toContain('Worker 累计峰值');
    expect(html).toContain('不能当作本次评测的独立峰值');
    expect(html).not.toContain('峰值分配内存');
  });

  it('keeps failed, pending and overflowed quality measurements explicit', () => {
    state.run = { ...qualityRun(), state: 'failed', error: 'SCORING_FAILED', metrics: {} };
    expect(view()).toContain('SCORING_FAILED');
    expect(view()).toContain('没有可用评分结果');
    state.run = { ...qualityRun(), state: 'running', metrics: {} };
    expect(view()).toContain('等待任务完成与封存');
    state.run = qualityRun();
    state.run.metrics.quality_evaluation = {
      status: 'complete',
      perplexity: null,
      perplexity_overflow: true,
      mean_nll: 1000,
    };
    expect(view()).toContain('数值溢出；请查看平均 NLL');
  });

  it('does not misrepresent uncollected quality diagnostics as missing inference data', () => {
    state.run = qualityRun();
    expect(view('?tab=observation')).toContain('本质量任务未采集性能与机制详录');
    expect(view('?tab=identity')).not.toContain('精确复现需再次提供原输入');
  });
});
