import type { Job, Plan, Slot } from './types';
import type { Run } from '../../api/types';

export function isQualityRun(run: Pick<Run, 'config' | 'metrics'>) {
  return run.config.experiment_role === 'quality' || !!run.metrics.quality_evaluation;
}
export const roleLabel = { performance: '性能', diagnostic: '机制诊断', quality: '质量' };
export const armLabel = { baseline: '基线', candidate: '候选' };
export const layerLabels = [
  '硬件与系统',
  '指令与 Kernel',
  '编译与运行时',
  '精度与量化',
  '模型结构',
  '内存与 KV',
  '解码算法',
  '请求调度',
  '分布式执行',
  '服务与应用',
];
export function slots(plan: Plan): Slot[] {
  return (['performance', 'diagnostic', 'quality'] as const).flatMap((role) =>
    Array.from({ length: role === 'performance' ? plan.spec.performance_pairs : 1 }, (_, pair) =>
      (pair % 2 ? (['candidate', 'baseline'] as const) : (['baseline', 'candidate'] as const)).map(
        (arm) => ({ arm, role, pair }),
      ),
    ).flat(),
  );
}
export const slotKey = (slot: Slot) => `${slot.role}:${slot.pair}:${slot.arm}`;
export const pendingSlots = (plan: Plan, jobs: Job[]) =>
  slots(plan).filter((slot) => !jobs.some((job) => slotKey(job) === slotKey(slot)));
export const initialSpec = {
  baseline: { changes: {} },
  candidate: { changes: { cleanup_policy: 'preserve_allocator' } },
  declared_changes: ['cleanup_policy'],
  performance_pairs: 1,
  input_tokens: 32,
  output_tokens: 8,
  max_relative_e2e: 0,
  quality_plan: {
    samples: [{ id: 'arithmetic-fixture', text: '1 + 1 =', expected: '2', slice: 'fixture' }],
    max_tokens: 16,
    generation_tokens: 4,
  },
  observation_plan: {
    mode: 'basic',
    sources: ['allocator', 'pages'],
    max_pages: 16,
    max_decode_steps: 2,
  },
  mechanism_expectations: [
    {
      metric_id: 'reserved_bytes',
      label: 'after_cleanup',
      direction: 'increase',
      minimum_delta: 0,
    },
  ],
};
