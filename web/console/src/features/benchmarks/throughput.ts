import type { Measurement, ResultRow } from './types';

export function generationRate(row: ResultRow, records: Measurement[]) {
  const recorded = row.metrics.output_tokens_per_s;
  if (typeof recorded === 'number' && Number.isFinite(recorded) && recorded >= 0)
    return { value: recorded, derived: false };
  // Old frozen reports omit this aggregate. Derive each completed sample's
  // rate first, then take the median; never divide two aggregated medians.
  const samples = records.filter((sample) => sample.scheme_id === row.scheme_id &&
    sample.case_id === row.case_id && sample.role === 'measurement' &&
    sample.state === 'completed' && sample.cleanup === 'succeeded');
  if (!samples.length || samples.length !== row.completed) return { value: null, derived: false };
  const rates: number[] = [];
  for (const sample of samples) {
    const count = sample.metrics?.output_tokens;
    const elapsed = sample.metrics?.runtime_e2e_ms;
    if (typeof count !== 'number' || !Number.isInteger(count) || count < 0 ||
        typeof elapsed !== 'number' || !Number.isFinite(elapsed) || elapsed <= 0)
      return { value: null, derived: false };
    const value = count * 1000 / elapsed;
    if (!Number.isFinite(value)) return { value: null, derived: false };
    rates.push(value);
  }
  rates.sort((a, b) => a - b);
  const middle = Math.floor(rates.length / 2);
  return { value: rates.length % 2 ? rates[middle] : (rates[middle - 1] + rates[middle]) / 2,
    derived: true };
}
