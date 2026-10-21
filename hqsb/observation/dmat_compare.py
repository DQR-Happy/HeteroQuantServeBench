"""Align DMAT pairs by live tensor role/offset and common time/density scales."""
from pathlib import Path
import hashlib


def compare_memory(a, b, *, roots=None):
    import numpy as np
    from hqsb.observation.dmat_reports import read_frame, aggregate
    groups = [p.get('memory_launches', []) for p in (a, b)]
    reasons = []
    if len(groups[0]) != len(groups[1]) or not groups[0]:
        return {'comparable': False, 'mismatches': ['captured_launch_count']}
    pairs, objects = [], []
    for index, (left, right) in enumerate(zip(*groups)):
        for key in ('semantic_shape', 'operation', 'causal', 'scale', 'layer', 'phase'):
            if left.get('context', {}).get(key) != right.get('context', {}).get(key):
                reasons.append(f'launch:{index}:{key}')
        if not left.get('context', {}).get('semantic_shape') or not right.get('context', {}).get('semantic_shape'):
            reasons.append(f'launch:{index}:tensor_semantics_required')
        plots = [left['plot'], right['plot']]
        if any(not p['complete'] for p in plots):
            reasons.append(f'launch:{index}:partial_capture')
        if any(p.get('unknown_events', 1) for p in plots):
            reasons.append(f'launch:{index}:unmapped_events')
        if plots[0].get('clock_domain') != plots[1].get('clock_domain'):
            reasons.append(f'launch:{index}:clock_domain')
        if reasons:
            continue
        if roots:
            duration = max(p['duration'] for p in plots)
            bins = max(p['time_bins'] for p in plots)
            quantum = max(1, (duration + bins - 1) // bins)
            page_bytes = max(p['address_bin_bytes'] for p in plots)
            frames = [read_frame(Path(root) / launch['trace']) for root, launch in zip(roots, (left, right))]
            plots = [aggregate(f, launch['probe']['sites'], page_bytes=page_bytes, time_bins=bins,
                               clock=plots[0]['clock_domain'], tensors=launch['context']['tensors'],
                               window=(0, duration), bin_width=quantum, address_origin='tensor_relative') for f, launch in zip(frames, (left, right))]
            thresholds = {}
            for role in {o['role'] for p in plots for o in p['objects']}:
                values = [c['events'] for p in plots for c in p['cells']
                          if next(o['role'] for o in p['objects'] if o['id'] == c['object']) == role]
                thresholds[role] = np.percentile(values, [20, 40, 60, 80]).tolist()
            # Recolour existing common grids. Raw event decoding/sorting is
            # independent of the percentile palette and only needs one pass.
            for plot in plots:
                role_by_id = {o['id']: o['role'] for o in plot['objects']}
                for obj in plot['objects']:
                    obj['color_thresholds'] = thresholds[obj['role']]
                for cell in plot['cells']:
                    cell['level'] = 1 + sum(cell['events'] > t for t in thresholds[role_by_id[cell['object']]])
        else:
            return {'comparable': False, 'mismatches': ['raw_frames_required_for_common_scale']}
        for role in sorted({o['role'] for p in plots for o in p['objects']}):
            vals = [next((o for o in p['objects'] if o['role'] == role), {}) for p in plots]
            if any(not o for o in vals) or vals[0].get('shape') != vals[1].get('shape'):
                reasons.append(f'launch:{index}:tensor_shape:{role}')
                continue
            objects.append({'launch': index, 'role': role, 'shape': vals[0]['shape'],
                            'baseline_events': vals[0]['events'], 'candidate_events': vals[1]['events'],
                            'delta_events': vals[1]['events'] - vals[0]['events'],
                            'baseline_instruction_bytes': vals[0]['instruction_bytes'],
                            'candidate_instruction_bytes': vals[1]['instruction_bytes'],
                            'baseline_occupied_cells': vals[0]['occupied_cells'],
                            'candidate_occupied_cells': vals[1]['occupied_cells']})
        # VA allocations differ across processes. Align pages relative to each
        # tensor's first page; preserve padding offset as separate evidence.
        aligned = []
        for plot in plots:
            role_base = {o['id']: int(o['base']) for o in plot['objects']}
            role_name = {o['id']: o['role'] for o in plot['objects']}
            aligned.append({(role_name[c['object']], (int(plot['addresses'][c['row']], 16) - role_base[c['object']]) // plot['address_bin_bytes'], c['column']): c['events'] for c in plot['cells']})
        keys = aligned[0].keys() | aligned[1].keys()
        distance = sum(abs(aligned[1].get(k, 0) - aligned[0].get(k, 0)) for k in keys)
        pairs.append({'launch': index, 'baseline': plots[0], 'candidate': plots[1],
                      'aligned_density_l1': distance, 'alignment': 'tensor_role_relative_page_and_common_time_bin'})
    access_counts_changed = any(o['baseline_events'] != o['candidate_events'] or
                                o['baseline_instruction_bytes'] != o['candidate_instruction_bytes'] for o in objects)
    return {'comparable': not reasons, 'mismatches': reasons, 'pairs': pairs, 'objects': objects,
            'analysis_source_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                      for name in ('dmat_compare.py', 'dmat_grid.py', 'dmat_reports.py')},
            'structure_changed': access_counts_changed,
            'access_counts_changed': access_counts_changed,
            'density_matrix_changed': any(p['aligned_density_l1'] for p in pairs),
            'density_difference_interpretation': 'timing_variation_includes_scheduling_and_probe_perturbation; counts_and_repeated_experiments_support_attribution',
            'scale': 'shared_time_bins_and_pooled_per_tensor_nonzero_density_percentiles',
            'measurement_role': 'instrumented_mechanism_comparison',
            'timing_interpretation': 'probe_perturbed; use_separate_uninstrumented_benchmark_for_speed'}
