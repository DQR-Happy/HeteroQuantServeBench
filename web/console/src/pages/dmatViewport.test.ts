import { describe, expect, it } from 'vitest';
import { fullRange, panRange, percentRange, timeWindow, zoomRange } from './dmatViewport';

describe('DMAT viewport bounds', () => {
  it('pans at fixed width and clamps at both ends', () => {
    expect(panRange([20, 39], 1000, 100)).toEqual([80, 99]);
    expect(panRange([20, 39], -1000, 100)).toEqual([0, 19]);
    expect(panRange([20, 39], 10, 100)).toEqual([30, 49]);
    expect(panRange([50, 50], -0.5, 100)).toEqual([49, 49]);
    expect(panRange([50, 50], 0.5, 100)).toEqual([51, 51]);
  });
  it('zooms around the center, preserves boundary detail and can fit all bins', () => {
    expect(zoomRange([0, 99], 0.5, 100)).toEqual([25, 74]);
    expect(zoomRange([90, 99], 2, 100)).toEqual([80, 99]);
    expect(zoomRange([50, 50], 0.5, 100)).toEqual([50, 50]);
    expect(zoomRange([50, 50], 1.12, 100)).toEqual([50, 51]);
    expect(zoomRange([20, 39], 100, 100)).toEqual(fullRange(100));
  });
  it('handles empty and one-bin captures', () => {
    for (const count of [0, 1]) {
      expect(fullRange(count)).toEqual([0, 0]);
      expect(panRange([0, 0], 100, count)).toEqual([0, 0]);
      expect(zoomRange([0, 0], 0.5, count)).toEqual([0, 0]);
      expect(percentRange(10, 90, count)).toEqual([0, 0]);
    }
  });
  it('maps slider percentages to bins and clips raw rebin requests to the actual end', () => {
    expect(percentRange(25, 75, 101)).toEqual([25, 75]);
    expect(timeWindow([10, 19], 1000, 64, 2250)).toEqual([1640, 2250]);
    expect(timeWindow([0, 0], 1000, 64, 2250)).toEqual([1000, 1064]);
  });
});
