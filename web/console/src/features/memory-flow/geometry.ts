import type { Inventory, TensorEntry } from './types';

export type WeightGroup = {
  id: string;
  label: string;
  kind: 'layer' | 'embedding' | 'shared' | 'other';
  bytes: number;
  entries: TensorEntry[];
  storages: number;
};
function groupOf(entry: TensorEntry): Pick<WeightGroup, 'id' | 'label' | 'kind'> {
  const layer = entry.name.match(/(?:^|\.)(?:layers|h|blocks)\.(\d+)(?:\.|$)/);
  if (layer) return { id: `layer-${layer[1]}`, label: `Decoder ${layer[1]}`, kind: 'layer' };
  if (/(?:embed|embedding|lm_head|wte)/i.test(entry.name))
    return { id: 'embedding', label: 'Embedding / 输出头', kind: 'embedding' };
  return { id: 'other', label: 'Norm / 其他', kind: 'other' };
}

/** Deduplicate backing storage globally, including tied names across layers. */
export function groupWeights(inventory: Inventory | null | undefined) {
  const storages = new Map<string, TensorEntry[]>();
  let unknownEntries = 0;
  for (const entry of inventory?.entries ?? []) {
    if (entry.storage_key == null || entry.storage_bytes == null) {
      unknownEntries += 1;
      continue;
    }
    const aliases = storages.get(entry.storage_key) ?? [];
    aliases.push(entry);
    storages.set(entry.storage_key, aliases);
  }
  const groups = new Map<string, WeightGroup>();
  for (const aliases of storages.values()) {
    const owners = aliases.map(groupOf);
    const owner =
      new Set(owners.map((x) => x.id)).size > 1
        ? { id: 'shared', label: '跨组共享 Storage', kind: 'shared' as const }
        : owners[0];
    const group = groups.get(owner.id) ?? { ...owner, bytes: 0, entries: [], storages: 0 };
    group.bytes += Math.max(...aliases.map((entry) => entry.storage_bytes ?? 0));
    group.entries.push(...aliases);
    group.storages += 1;
    groups.set(owner.id, group);
  }
  return {
    groups: [...groups.values()].sort((a, b) => b.bytes - a.bytes || a.id.localeCompare(b.id)),
    unknownEntries,
  };
}

export type Tile = WeightGroup & { x: number; y: number; width: number; height: number };
/** Binary partition keeps each rectangle's area proportional to unique bytes. */
export function weightTiles(groups: WeightGroup[], width = 840, height = 340): Tile[] {
  const split = (items: WeightGroup[], x: number, y: number, w: number, h: number): Tile[] => {
    if (!items.length) return [];
    if (items.length === 1) return [{ ...items[0], x, y, width: w, height: h }];
    const sum = items.reduce((n, item) => n + item.bytes, 0);
    let pivot = 1;
    let left = items[0].bytes;
    while (
      pivot < items.length - 1 &&
      Math.abs(left + items[pivot].bytes - sum / 2) < Math.abs(left - sum / 2)
    ) {
      left += items[pivot].bytes;
      pivot += 1;
    }
    const ratio = left / sum;
    return w >= h
      ? [
          ...split(items.slice(0, pivot), x, y, w * ratio, h),
          ...split(items.slice(pivot), x + w * ratio, y, w * (1 - ratio), h),
        ]
      : [
          ...split(items.slice(0, pivot), x, y, w, h * ratio),
          ...split(items.slice(pivot), x, y + h * ratio, w, h * (1 - ratio)),
        ];
  };
  return split(
    groups.filter((g) => g.bytes > 0),
    0,
    0,
    width,
    height,
  );
}

export function layerKV(inventory: Inventory | null | undefined) {
  const layers = new Map<
    string,
    { layer: string; key: number; value: number; other: number; entries: TensorEntry[] }
  >();
  for (const entry of inventory?.entries ?? []) {
    const layer = String(entry.layer ?? '未知');
    const item = layers.get(layer) ?? { layer, key: 0, value: 0, other: 0, entries: [] };
    if (entry.kind === 'key') item.key += entry.logical_bytes;
    else if (entry.kind === 'value') item.value += entry.logical_bytes;
    else item.other += entry.logical_bytes;
    item.entries.push(entry);
    layers.set(layer, item);
  }
  return [...layers.values()].sort((a, b) =>
    a.layer.localeCompare(b.layer, undefined, { numeric: true }),
  );
}

export function memorySize(bytes: number | null | undefined): string {
  if (bytes == null || !Number.isFinite(bytes)) return '未采集';
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(2)} GiB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(2)} MiB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(2)} KiB`;
  return `${bytes} B`;
}
