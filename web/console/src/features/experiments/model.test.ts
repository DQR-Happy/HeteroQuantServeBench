import { describe, expect, it } from 'vitest';
import { pendingSlots, slots, slotKey } from './model';
import type { Job, Plan } from './types';
describe('frozen experiment task queue', () => {
  const plan = { spec: { performance_pairs: 2 } } as Plan;
  it('alternates performance arms and keeps quality/diagnostic as separate pairs', () => {
    const queue = slots(plan);
    expect(queue).toHaveLength(8);
    expect(queue.slice(0, 4).map(slotKey)).toEqual([
      'performance:0:baseline',
      'performance:0:candidate',
      'performance:1:candidate',
      'performance:1:baseline',
    ]);
    expect(queue.filter((slot) => slot.role === 'quality')).toHaveLength(2);
  });
  it('never resubmits accepted, failed or cancelled slots after reconnect', () => {
    const queue = slots(plan);
    const jobs = queue
      .slice(0, 3)
      .map((slot, i) => ({ ...slot, state: ['running', 'failed', 'cancelled'][i] })) as Job[];
    expect(pendingSlots(plan, jobs).map(slotKey)).toEqual(queue.slice(3).map(slotKey));
  });
});
