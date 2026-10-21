"""Paper DMAT aggregation: thread-entry cycles or GPU time, tensor colours.

Address pages are software bins. Each enabled thread instruction contributes
one density event at its starting VA, matching Neutrino's DMAT definition.
"""
from math import prod

PALETTE = ['#0053d6', '#812be1', '#00bb00', '#b31805', '#b26531', '#636b74']


def grid(frame, rows, sites, *, page_bytes, time_bins, clock, tensors=(), window=None, bin_width=None, color_scale=None, address_origin="virtual"):
    import numpy as np
    if clock not in ('thread_entry_clock64', 'GPU_globaltimer'):
        raise ValueError('DMAT_CLOCK_DOMAIN')
    if 'clock' not in rows.dtype.names and clock == 'thread_entry_clock64':
        raise ValueError('DMAT_LEGACY_TRACE_REQUIRES_GLOBALTIMER')
    if 'clock' in rows.dtype.names and len(rows) and np.any(rows['clock'] < rows['entry']):
        raise ValueError('DMAT_ENTRY_CLOCK_ORDER')
    first = int(rows['time'].min()) if len(rows) else 0
    times = rows['clock'] - rows['entry'] if clock == 'thread_entry_clock64' else rows['time'] - first
    duration = int(times.max()) + 1 if len(rows) else 1
    left, right = window or (0, duration)
    if left < 0 or right <= left or right > 2**53:
        raise ValueError('DMAT_TIME_WINDOW')
    quantum = bin_width or max(1, (right - left + time_bins - 1) // time_bins)
    if quantum < 1 or (right - left + quantum - 1) // quantum > 512:
        raise ValueError('DMAT_TIME_GRID_BUDGET')
    time_bins = (right - left + quantum - 1) // quantum
    selected = (times >= left) & (times < right)
    visible, times = rows[selected], times[selected]
    # Narrowest containing view owns the event; aliases are retained in metadata.
    descriptors = sorted(tensors, key=lambda t: ({'Q': 0, 'K': 1, 'V': 2, 'O': 3}.get(t.get('role'), 4), t['name']))
    objects = [{**t, 'base': str(int(t['base'])), 'id': i, 'color': PALETTE[{'Q': 0, 'K': 1, 'V': 2, 'O': 3}.get(t.get('role'), i) % len(PALETTE)]}
               for i, t in enumerate(descriptors)]
    unknown = len(objects)
    objects.append({'id': unknown, 'name': 'Unknown', 'role': 'Unknown', 'base': '0', 'bytes': 0,
                    'color': PALETTE[-1]})
    owners = np.full(len(visible), unknown, dtype=np.int64)
    for obj in sorted(objects[:-1], key=lambda t: (int(t['bytes']), t['name'])):
        lo, size = int(obj['base']), int(obj['bytes'])
        fits = (visible['address'] >= lo) & (visible['address'] < lo + size)
        fits &= visible['width'].astype('uint64') <= (np.uint64(lo + size) - visible['address'])
        owners[(owners == unknown) & fits] = obj['id']
    pages = visible['address'] // page_bytes
    if address_origin == 'tensor_relative':
        pages = pages.copy()
        for obj in objects[:-1]:
            owned = owners == obj['id']
            pages[owned] = (visible['address'][owned] - np.uint64(int(obj['base']))) // page_bytes
    elif address_origin != 'virtual':
        raise ValueError('DMAT_ADDRESS_ORIGIN')
    pairs = np.stack((owners, pages.astype('int64')), axis=1) if len(visible) else np.empty((0, 2), dtype='int64')
    keys, y = np.unique(pairs, axis=0, return_inverse=True)
    if len(keys) > 1024:
        raise ValueError('DMAT_ADDRESS_VIEW_BUDGET: 缩小时间窗口或选择更大的地址粒度。')
    columns = ((times - left) // quantum).astype('int64')
    flat = y * time_bins + columns
    unique, inverse = np.unique(flat, return_inverse=True)
    total = np.bincount(inverse)
    read_sites = [s['site'] for s in sites if s['operation'] == 'read']
    reads = np.bincount(inverse, weights=np.isin(visible['site'], read_sites).astype('int64'), minlength=len(unique))
    byte_counts = np.bincount(inverse, weights=visible['width'].astype('int64'), minlength=len(unique))
    object_ids = [int(keys[int(v) // time_bins, 0]) for v in unique]
    scales = {}
    for obj in objects:
        values = total[np.asarray(object_ids) == obj['id']]
        role = obj.get('role', obj['name'])
        scales[str(obj['id'])] = list(color_scale[role]) if color_scale and role in color_scale else (np.percentile(values, [20, 40, 60, 80]).tolist() if len(values) else [0] * 4)
    cells = []
    for i, value in enumerate(unique):
        oid = object_ids[i]
        level = 1 + sum(int(total[i]) > t for t in scales[str(oid)])
        cells.append({'row': int(value) // time_bins, 'column': int(value) % time_bins,
                      'object': oid, 'events': int(total[i]), 'reads': int(reads[i]),
                      'writes': int(total[i] - reads[i]), 'instruction_bytes': int(byte_counts[i]), 'level': level})
    objects_by_id = {o['id']: o for o in objects}
    address_rows = []
    for oid, page in keys:
        obj = objects_by_id[int(oid)]
        origin = int(obj['base']) if address_origin == 'tensor_relative' and int(oid) != unknown else 0
        base = origin + int(page) * page_bytes
        address_rows.append({'object': int(oid), 'address': hex(base), 'offset_bytes': base - int(obj['base']),
                             'label': f"{obj.get('role', obj['name'])} +{hex(int(page) * page_bytes)}" if address_origin == 'tensor_relative' else f"{obj.get('role', obj['name'])} · {hex(base)}"})
    attempted = sum(int(x) for x in frame['counts'])
    per_site, numbers = np.unique(rows['site'], return_counts=True)
    site_counts = dict(zip((int(x) for x in per_site), (int(x) for x in numbers)))
    stats = []
    for obj in objects:
        owned = owners == obj['id']
        if not owned.any():
            continue
        object_cells = [c for c in cells if c['object'] == obj['id']]
        stats.append({**obj, 'events': int(owned.sum()), 'instruction_bytes': int(visible['width'][owned].sum()),
                      'pages': int(len(np.unique(pages[owned]))), 'occupied_cells': len(object_cells),
                      'color_thresholds': scales[str(obj['id'])]})
    return {'events': len(rows), 'visible_events': len(visible), 'attempted_events': attempted,
            'dropped_events': attempted - len(rows), 'complete': attempted == len(rows),
            'overflow_threads': int(np.count_nonzero(frame['counts'] > frame['capacity'])),
            'gpu_start_ns': str(first), 'duration_ns': int(rows['time'].max()) - first + 1 if len(rows) else 1,
            'duration': duration, 'clock_domain': clock, 'time_unit': 'cycles' if clock == 'thread_entry_clock64' else 'ns',
            'time_start': left, 'time_end': right, 'time_bin_ns': quantum, 'time_bin': quantum, 'time_bins': time_bins,
            'requested_time_bins': time_bins, 'address_bin_bytes': page_bytes, 'requested_address_bin_bytes': page_bytes,
            'addresses': [r['address'] for r in address_rows], 'address_rows': address_rows,
            'address_axis': 'tensor_relative_byte_bins' if address_origin == 'tensor_relative' else 'tensor_then_sorted_observed_VA_pages_gaps_collapsed',
            'density_semantics': 'enabled_thread_instruction_events_at_start_VA',
            'color_semantics': 'per_tensor_nonzero_cell_percentiles_20_40_60_80',
            'objects': stats, 'cells': cells, 'sites': [{**s, 'events': site_counts.get(s['site'], 0)} for s in sites],
            'mapped_events': int((owners != unknown).sum()), 'unknown_events': int((owners == unknown).sum()),
            'alias_policy': 'smallest_containing_view_then_name; tensor_aliases_retained_in_launch_context',
            'coverage': 'selected_kernel_supported_global_instructions', 'block_threads': prod(frame['block'])}
