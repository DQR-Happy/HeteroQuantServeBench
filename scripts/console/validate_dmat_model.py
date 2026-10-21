"""Exercise the public load/generate/diagnostic/comparison flow on real Qwen3."""
import argparse
import json
from pathlib import Path
import time
import urllib.request
import uuid


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:8768')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=args.resume)

    def call(path, body=None):
        req = urllib.request.Request(args.url + '/api/console/v1' + path,
                data=json.dumps(body).encode() if body is not None else None,
                headers={'Content-Type': 'application/json', 'X-HQSB-Client': 'console', 'Idempotency-Key': uuid.uuid4().hex})
        with urllib.request.urlopen(req, timeout=120) as response:
            return json.load(response)

    def task(path, body):
        run = call(path, body)
        print(path, run['id'], flush=True)
        deadline = time.monotonic() + 1000
        last = None
        while time.monotonic() < deadline:
            run = call('/runs/' + run['id'])
            state = run['state'], run.get('progress', {}).get('stage')
            if state != last:
                print(state, flush=True)
                last = state
            if run['state'] in ('completed', 'failed', 'cancelled', 'timed_out', 'interrupted'):
                (root / (run['id'] + '.json')).write_text(json.dumps(run, indent=2))
                assert run['state'] == 'completed', run.get('error')
                return run
            time.sleep(1)
        raise TimeoutError(run['id'])

    def deployment():
        return next(d for d in call('/deployments')['items'] if d['id'] == 'qwen-reference')

    assert deployment()['state'] in ('unloaded', 'failed')
    prompt = 'Read the following numbers carefully. ' + 'The sequence is 1, 2, 3, 4, 5. ' * 12 + 'What is 17 plus 25? Answer briefly.'
    sources, captures = {}, {}
    saved = [json.loads(f.read_text()) for f in root.glob('run_*.json')] if args.resume else []
    for profile in ('framework-native', 'qwen3-attention-triton-n32-v1', 'qwen3-attention-triton-n64-v1'):
        source = next((r for r in saved if r['kind'] == 'generate' and r['state'] == 'completed'
            and r['config']['actual_runtime']['kernel_plan']['profile_id'] == profile
            and r['config'].get('messages') == [{'role': 'user', 'content': prompt}]), None)
        if source is None:
            try:
                task('/deployments/qwen-reference/load', {'kernel_profile': profile})
                state = deployment()
                source = task('/requests', {'deployment_id': state['id'], 'expected_epoch': state['epoch'],
                    'messages': [{'role': 'user', 'content': prompt}], 'max_output_tokens': 3,
                    'fixed_output_length': True, 'save_input': True, 'observation_mode': 'off'})
            finally:
                if deployment()['state'] == 'ready':
                    task('/deployments/qwen-reference/unload', {})
        sources[profile] = source
        if profile == 'framework-native':
            continue
        captures[profile] = {}
        for phase, skip in [('prefill', 0), ('decode', 28)]:
            run = next((r for r in saved if r['kind'] == 'diagnostic' and r['state'] == 'completed'
                and r['config']['source_run_id'] == source['id'] and r['config']['diagnostic_plan']['launch_skip'] == skip), None)
            run = run or task('/diagnostics/jobs', {'source_run_id': source['id'], 'tool': 'eretrace', 'ere_mode': 'memory',
                'ere_kernel': 'hqsb_dmat_attention', 'ere_codec': 'raw', 'launch_count': 1, 'launch_skip': skip,
                'memory_capacity': 256, 'memory_page_bytes': 4096, 'memory_time_bins': 160,
                'memory_clock': 'thread_entry_clock64', 'artifact_limit_mb': 512})
            receipt = run['metrics']['diagnostic']
            assert receipt['replay_outputs_match_source'] and receipt['replay_token_ids_match_source']
            assert receipt['replay_environment_matches_source'] and receipt['replay_workloads_match_source']
            launch = receipt['panel']['memory_launches'][0]
            assert launch['context']['phase'] == phase and launch['context']['layer'] == 'model.layers.0.self_attn'
            assert launch['plot']['unknown_events'] == 0 and launch['plot']['complete']
            captures[profile][phase] = run['id']
            # Rebin exact saved events in both physical clock domains.
            for clock in ('GPU_globaltimer', 'thread_entry_clock64'):
                view = call(f"/diagnostics/jobs/{run['id']}/memory/0?clock={clock}&page_bytes=4096&time_bins=160")
                assert view['plot']['visible_events'] == launch['plot']['events']
                (root / f'{profile}-{phase}-{clock}.json').write_text(json.dumps(view))
    comparisons = {}
    left, right = ('qwen3-attention-triton-n32-v1', 'qwen3-attention-triton-n64-v1')
    for phase in ('prefill', 'decode'):
        result = call('/diagnostics/comparisons', {'baseline_run_id': captures[left][phase], 'candidate_run_id': captures[right][phase]})
        assert result['comparable'] and result['dmat']['comparable'], result.get('mismatches')
        comparisons[phase] = result
    reference = sources['framework-native']['metrics']['generated_token_ids']
    quality = {profile: {'text': r['output'], 'token_ids': r['metrics']['generated_token_ids'],
                         'tokens_match_framework': r['metrics']['generated_token_ids'] == reference,
                         'metrics': {k: v for k, v in r['metrics'].items() if k in ('ttft_ms', 'tokens_per_second', 'output_tokens', 'duration_ms')}}
               for profile, r in sources.items()}
    assert all(q['tokens_match_framework'] for q in quality.values()), quality
    (root / 'acceptance.json').write_text(json.dumps({'passed': True, 'sources': {p: r['id'] for p, r in sources.items()},
         'captures': captures, 'quality': quality, 'comparisons': comparisons}, indent=2))
    print(json.dumps({'passed': True, 'captures': captures, 'quality': quality}, indent=2), flush=True)


if __name__ == '__main__':
    main()
