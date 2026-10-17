import { describe, expect, it } from 'vitest';
import { generationRate } from './throughput';
import type { Measurement, ResultRow } from './types';

const row = { scheme_id: 'base', case_id: 'case', completed: 3, metrics: {} } as ResultRow;
const samples = [[10, 1000], [100, 100], [60, 2000]].map(([count, time]) => ({
  receipt: { status: null, effective: null }, run_id: 'unit-sample', repeat: 0,
  output: '', error: null, finish_reason: 'length', score: null,
  scheme_id: 'base', case_id: 'case', role: 'measurement', state: 'completed', cleanup: 'succeeded',
  metrics: { output_tokens: count, runtime_e2e_ms: time },
})) as Measurement[];

describe('generation throughput for frozen reports', () => {
  it('takes the median of per-sample rates, excluding warmups and other cohorts', () => {
    const warmup = { ...samples[0], role: 'warmup', metrics: { output_tokens: 999, runtime_e2e_ms: 1 } };
    expect(generationRate(row, [...samples, warmup, { ...warmup, role: 'measurement', scheme_id: 'other' }]))
      .toEqual({ value: 30, derived: true });
    expect(samples[0].metrics).not.toHaveProperty('output_tokens_per_s');
  });
  it('uses recorded aggregates when available', () => {
    expect(generationRate({ ...row, metrics: { output_tokens_per_s: 23.5 } }, []))
      .toEqual({ value: 23.5, derived: false });
  });
  it('leaves missing evidence and zero duration unavailable', () => {
    expect(generationRate(row, samples.slice(1)).value).toBeNull();
    expect(generationRate(row, [samples[0], samples[1], { ...samples[2], metrics: { output_tokens: 10, runtime_e2e_ms: 0 } }]).value).toBeNull();
    expect(generationRate(row, [samples[0], samples[1], { ...samples[2], state: 'failed' }]).value).toBeNull();
  });
  it('supports a single output token without inventing a decode rate', () => {
    expect(generationRate({ ...row, completed: 1 }, [{ ...samples[0], metrics: { output_tokens: 1, runtime_e2e_ms: 100 } }]))
      .toEqual({ value: 10, derived: true });
  });
});
