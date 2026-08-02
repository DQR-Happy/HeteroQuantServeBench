import { describe, expect, it } from 'vitest';
import { groupWeights, layerKV, weightTiles } from './geometry';
import type { Inventory, TensorEntry } from './types';

const tensor = (
  name: string,
  bytes: number,
  storage: string | null,
  extra: Partial<TensorEntry> = {},
): TensorEntry => ({
  name,
  kind: 'parameter',
  shape: [bytes / 2],
  dtype: 'torch.float16',
  device: 'cuda:0',
  logical_bytes: bytes,
  storage_key: storage,
  storage_offset: 0,
  storage_bytes: storage == null ? null : bytes,
  ...extra,
});
const inventory = (entries: TensorEntry[]): Inventory => ({
  scope: 'test',
  availability: 'complete',
  entries,
  logical_total_bytes: null,
  unique_storage_bytes: null,
  entry_count: entries.length,
  truncated: false,
  limitations: [],
});

describe('memory evidence geometry', () => {
  it('counts tied storage only once across layers and preserves both aliases', () => {
    const { groups } = groupWeights(
      inventory([
        tensor('model.layers.0.weight', 80, 'shared'),
        tensor('model.layers.1.weight', 80, 'shared'),
        tensor('model.layers.0.bias', 20, 'local'),
      ]),
    );
    expect(groups.reduce((sum, group) => sum + group.bytes, 0)).toBe(100);
    expect(groups.find((group) => group.id === 'shared')).toMatchObject({ bytes: 80, storages: 1 });
    expect(groups.find((group) => group.id === 'shared')?.entries).toHaveLength(2);
  });
  it('keeps views of one layer together without adding logical bytes to storage bytes', () => {
    const result = groupWeights(
      inventory([
        tensor('model.layers.3.q_proj.weight', 40, 'packed', { storage_bytes: 128 }),
        tensor('model.layers.3.k_proj.weight', 16, 'packed', {
          storage_bytes: 128,
          storage_offset: 20,
          storage_offset_bytes: 40,
        }),
        tensor('model.layers.3.norm.weight', 8, 'norm'),
      ]),
    );
    expect(result.unknownEntries).toBe(0);
    expect(result.groups).toHaveLength(1);
    expect(result.groups[0]).toMatchObject({
      id: 'layer-3',
      kind: 'layer',
      bytes: 136,
      storages: 2,
    });
    expect(result.groups[0].entries).toHaveLength(3);
    expect(result.groups[0].entries.map((entry) => entry.logical_bytes)).toEqual([40, 16, 8]);
  });
  it('never fills unknown storage sizes with logical bytes', () => {
    const result = groupWeights(inventory([tensor('model.embed_tokens.weight', 999, null)]));
    expect(result.groups).toEqual([]);
    expect(result.unknownEntries).toBe(1);
    expect(weightTiles(result.groups)).toEqual([]);
  });
  it('distinguishes unknown storage metadata from a measured zero-byte storage', () => {
    const result = groupWeights(
      inventory([
        tensor('model.layers.0.known', 10, 'known'),
        tensor('model.layers.0.missing_size', 999, 'size-unknown', { storage_bytes: null }),
        tensor('model.layers.0.missing_key', 999, null, { storage_bytes: 999 }),
        tensor('model.layers.1.empty', 0, 'empty'),
      ]),
    );
    expect(result.unknownEntries).toBe(2);
    expect(result.groups.reduce((sum, group) => sum + group.bytes, 0)).toBe(10);
    expect(result.groups.flatMap((group) => group.entries).map((entry) => entry.name)).toEqual([
      'model.layers.0.known',
      'model.layers.1.empty',
    ]);
    expect(result.groups.find((group) => group.id === 'layer-1')).toMatchObject({
      bytes: 0,
      storages: 1,
    });
    expect(weightTiles(result.groups)).toHaveLength(1);
  });
  it('preserves byte proportions and canvas boundaries for every tile', () => {
    const { groups } = groupWeights(
      inventory(
        [100, 30, 20, 1].map((bytes, i) => tensor(`model.layers.${i}.weight`, bytes, `s${i}`)),
      ),
    );
    const tiles = weightTiles(groups, 840, 340);
    expect(tiles.reduce((sum, tile) => sum + tile.width * tile.height, 0)).toBeCloseTo(840 * 340);
    for (const tile of tiles) {
      expect((tile.width * tile.height) / (840 * 340)).toBeCloseTo(tile.bytes / 151);
      expect(tile.x).toBeGreaterThanOrEqual(0);
      expect(tile.y).toBeGreaterThanOrEqual(0);
      expect(tile.x + tile.width).toBeLessThanOrEqual(840.000001);
      expect(tile.y + tile.height).toBeLessThanOrEqual(340.000001);
    }
  });
  it.each([
    [840, 340],
    [300, 900],
    [400, 400],
  ])('does not overlap or inflate tied storage in a %s × %s canvas', (width, height) => {
    const { groups } = groupWeights(
      inventory([
        tensor('model.layers.0.weight', 180, 'tied'),
        tensor('model.layers.1.weight', 180, 'tied'),
        tensor('model.layers.2.weight', 50, 'layer2'),
        tensor('model.layers.3.weight', 8, 'layer3'),
        tensor('model.embed_tokens.weight', 1, 'embedding'),
      ]),
    );
    const tiles = weightTiles(groups, width, height);
    expect(tiles).toHaveLength(4);
    const area = width * height;
    for (const [index, tile] of tiles.entries()) {
      expect((tile.width * tile.height) / area).toBeCloseTo(tile.bytes / 239, 10);
      expect(tile.x).toBeGreaterThanOrEqual(0);
      expect(tile.y).toBeGreaterThanOrEqual(0);
      expect(tile.x + tile.width).toBeLessThanOrEqual(width + 1e-8);
      expect(tile.y + tile.height).toBeLessThanOrEqual(height + 1e-8);
      for (const other of tiles.slice(index + 1)) {
        const overlapX = Math.max(
          0,
          Math.min(tile.x + tile.width, other.x + other.width) - Math.max(tile.x, other.x),
        );
        const overlapY = Math.max(
          0,
          Math.min(tile.y + tile.height, other.y + other.height) - Math.max(tile.y, other.y),
        );
        expect(overlapX * overlapY).toBeLessThan(1e-8);
      }
    }
    expect(tiles.reduce((sum, tile) => sum + tile.width * tile.height, 0)).toBeCloseTo(area, 6);
  });
  it('uses logical K/V bytes independently from retained backing storage', () => {
    const layers = layerKV(
      inventory([
        tensor('layer.10.key', 20, 'k10', { layer: 10, kind: 'key', storage_bytes: 80 }),
        tensor('layer.2.value', 30, 'v2', { layer: 2, kind: 'value', storage_bytes: 90 }),
        tensor('layer.2.key', 25, 'k2', { layer: 2, kind: 'key' }),
      ]),
    );
    expect(layers.map((layer) => layer.layer)).toEqual(['2', '10']);
    expect(layers[0]).toMatchObject({ key: 25, value: 30, other: 0 });
    expect(layers[1].key).toBe(20);
  });
  it('preserves key and value logical views even when they share physical storage', () => {
    const result = layerKV(
      inventory([
        tensor('layer.0.key', 12, 'kv-packed', { layer: 0, kind: 'key', storage_bytes: 128 }),
        tensor('layer.0.value', 8, 'kv-packed', {
          layer: 0,
          kind: 'value',
          storage_bytes: 128,
          storage_offset: 6,
        }),
        tensor('layer.0.extra', 2, null, { layer: 0, kind: 'auxiliary' }),
        tensor('unresolved.key', 5, null, { kind: 'key' }),
      ]),
    );
    expect(result.find((layer) => layer.layer === '0')).toMatchObject({
      key: 12,
      value: 8,
      other: 2,
    });
    expect(result.find((layer) => layer.layer === '0')?.entries).toHaveLength(3);
    expect(result.find((layer) => layer.layer === '未知')).toMatchObject({
      key: 5,
      value: 0,
      other: 0,
    });
  });
});
