// Inclusive category indices keep a one-bin window usable on either axis.
export type AxisRange = [number, number];
export type DMATViewport = { time: AxisRange; address: AxisRange };

export function fullRange(count: number): AxisRange {
  return [0, Math.max(0, count - 1)];
}

export function panRange(range: AxisRange, offset: number, count: number): AxisRange {
  const size = Math.min(count, range[1] - range[0] + 1);
  const steps = Math.sign(offset) * Math.round(Math.abs(offset));
  const start = Math.max(0, Math.min(count - size, range[0] + steps));
  return [start, Math.max(start, start + size - 1)];
}

export function zoomRange(range: AxisRange, factor: number, count: number): AxisRange {
  const current = range[1] - range[0] + 1;
  const desired = Math.round(current * factor);
  const size = Math.max(
    1,
    Math.min(
      count,
      factor > 1
        ? Math.max(current + 1, desired)
        : factor < 1
          ? Math.min(current - 1, desired)
          : current,
    ),
  );
  const center = (range[0] + range[1]) / 2;
  const start = Math.max(0, Math.min(count - size, Math.round(center - (size - 1) / 2)));
  return [start, Math.max(start, start + size - 1)];
}

export function percentRange(start: number, end: number, count: number): AxisRange {
  const last = Math.max(0, count - 1);
  return [
    Math.round((Math.max(0, Math.min(100, start)) * last) / 100),
    Math.round((Math.max(start, Math.min(100, end)) * last) / 100),
  ];
}

export function timeWindow(
  range: AxisRange,
  start: number,
  quantum: number,
  end: number,
): AxisRange {
  return [
    Math.floor(start + range[0] * quantum),
    Math.min(end, Math.ceil(start + (range[1] + 1) * quantum)),
  ];
}
