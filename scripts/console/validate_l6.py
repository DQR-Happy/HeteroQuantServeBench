"""Target-only L6 acceptance via production API; restore the resident recipe."""
import time
from pathlib import Path
from scripts.console import validate_l3 as api
from scripts.console import validate_graph_views as graph

OUT = Path('reports/console/l6-memory')


def terminal(run, label):
    while run['state'] not in {'completed','failed','cancelled','timed_out','interrupted'}:
        time.sleep(.3)
        run=api.call('/runs/'+run['id'])
    api.save(label,run)
    return run


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    api.OUT=OUT
    graph.OUT=OUT
    assert api.call('/overview')['pending']==0
    assert api.call('/scheduling-benchmarks')['active_id'] is None
    originals=[d for d in api.call('/deployments')['items'] if d['state']=='ready' and d['provider']=='pytorch']
    assert len(originals)==1
    original=originals[0]
    assert api.call('/benchmark-catalog?deployment_id='+original['id'])['active_id'] is None
    api.save('original',original)
    recipe={k:v for k,v in original['detail']['precision_receipt']['resolved_recipe'].items() if k!='calibration'}
    if original.get('quantization',{}).get('artifact_id'):recipe['artifact_id']=original['quantization']['artifact_id']
    summary={'models':{},'restored':False,'speedup_claim':False,'scope':'actual model generation, cache storage/transfer/prefix and scheduling block pool'}
    try:
        api.unload_all('initial')
        graph.restart()
        deadline=api.call('/memory/catalog')['max_deadline_ms']
        for identity in ('ms-586060631c81c17023c0','ms-6d9b6a4a3f944e041fd6','gemma3-1b'):
            print('MODEL',identity,flush=True)
            api.unload_all('next')
            api.wait(api.call(f'/deployments/{identity}/load',{'precision':'float16','kernel_profile':'framework-native','quantization':{'method':'none'}}),identity+'-load')
            d=next(d for d in api.call('/deployments')['items'] if d['id']==identity)
            cap=api.call('/memory/capabilities?deployment_id='+identity)
            api.save(identity+'-capability',cap)
            results=summary['models'][identity]={}
            body={'deployment_id':identity,'expected_epoch':d['epoch'],'prompt':'Once upon a time, there was a little fox. The fox',
                  'max_output_tokens':16,'fixed_output_length':True,'observation_mode':'off','save_input':True,'deadline_ms':deadline}
            for mode in ('dynamic','static','offloaded_dynamic','offloaded_static','recompute','model_default'):
                run=api.wait(api.call('/requests',{**body,'memory_plan':{'mode':mode}}),identity+'-'+mode)
                doc=api.call('/runs/'+run['id']+'/memory-cache')
                api.save(identity+'-'+mode+'-receipt',doc)
                assert doc['status']=='applied' and doc['forward_calls']==16
                results[mode]={'run_id':run['id'],'token_s':run['metrics']['output_tokens_per_s'],'memory':doc}
                if mode.startswith('offloaded'):
                    assert doc['transfers']['d2h_bytes']>0 and doc['transfers']['h2d_bytes']>0
                if mode!='dynamic':
                    comparison=api.call('/memory/comparisons',{'baseline_run_id':results['dynamic']['run_id'],'candidate_run_id':run['id']})
                    api.save(identity+'-'+mode+'-comparison',comparison)
                    assert comparison['comparable'],comparison['reasons']
                    # A different greedy decision is reported, never hidden by a timing result.
                    results[mode]['generation_check']=comparison['generation_check']
                    assert comparison['generation_check']['exact_tokens'],comparison['generation_check']
                    assert comparison['generation_check']['max_abs_selected_token_logprob_delta']<.04
                print(mode,run['id'],'Token/s',round(results[mode]['token_s'],2),flush=True)
                api.save('summary',summary)
            if cap['prefix_supported']:
                api.call('/memory/prefix/clear',{'deployment_id':identity,'expected_epoch':d['epoch']})
                for repeat in range(2):
                    run=api.wait(api.call('/requests',{**body,'memory_plan':{'mode':'dynamic','prefix_reuse':True}}),identity+f'-prefix-{repeat}')
                    doc=api.call('/runs/'+run['id']+'/memory-cache')
                    assert doc['prefix']['lookup']==('miss' if repeat==0 else 'hit')
                    if repeat: assert doc['prefix']['reused_tokens']>0
                    comparison=api.call('/memory/comparisons',{'baseline_run_id':results['dynamic']['run_id'],'candidate_run_id':run['id']})
                    assert comparison['comparable'] and comparison['generation_check']['exact_tokens'],comparison
                    results[f'prefix-{repeat}']={'run_id':run['id'],'prefix':doc['prefix']}
                cleared=api.call('/memory/prefix/clear',{'deployment_id':identity,'expected_epoch':d['epoch']})
                assert cleared['released_bytes']>0
            else:
                bad=terminal(api.call('/requests',{**body,'memory_plan':{'mode':'dynamic','prefix_reuse':True}}),identity+'-unsupported-prefix')
                assert bad['state']=='failed' and 'L6_PREFIX' in str(bad.get('error'))
                assert next(d for d in api.call('/deployments')['items'] if d['id']==identity)['state']=='ready'
            bad=terminal(api.call('/requests',{**body,'memory_plan':{'mode':'static','max_cache_tokens':1}}),identity+'-capacity-rejected')
            assert bad['state']=='failed'
            legacy=api.wait(api.call('/requests',{k:v for k,v in body.items() if k!='memory_plan'}),identity+'-legacy')
            results['legacy_preserved']=legacy['metrics']['generated_token_ids']==api.call('/runs/'+results['dynamic']['run_id'])['metrics']['generated_token_ids']
            assert results['legacy_preserved']
            if identity!='gemma3-1b':
                for layout in ('dense','block_pool'):
                    spec={'deployment_id':identity,'expected_epoch':d['epoch'], 'modes':['serial','continuous'],
                      'requests':[{'id':'a','prompt':'Once upon a time, a little fox','output_tokens':12},{'id':'b','prompt':'The bright red ball','output_tokens':8}],
                      'limits':{'max_active_sequences':2,'max_batch_sequences':2,'kv_capacity_tokens':256,'max_batched_tokens':64,'prefill_chunk_tokens':8,
                                'kv_layout':layout,'kv_block_tokens':8,'kv_pool_max_bytes':8*1024**2}}
                    experiment=api.call('/scheduling-benchmarks',spec)
                    while experiment['state'] not in {'completed','failed','cancelled','timed_out','interrupted'}:
                        time.sleep(.5)
                        experiment=api.call('/scheduling-benchmarks/'+experiment['id'])
                    api.save(identity+'-scheduling-'+layout,experiment)
                    assert experiment['state']=='completed',experiment.get('error')
                    variants=experiment['result']['variants']
                    assert all(v['cleanup']['device_kv_rows']==0 and v['cleanup']['resident_kv_bytes']==0 for v in variants)
                    if layout=='block_pool':assert all(v['memory_management']['peak_pool_storage_bytes']>0 for v in variants)
                    results['scheduling-'+layout]={'id':experiment['id'],'variants':[{k:v[k] for k in ('mode','kv_layout','kv_copy_bytes','memory_management','cleanup','output_tokens_per_s')} for v in variants]}
            api.save('summary',summary)
    finally:
        print('RESTORE',original['id'],flush=True)
        api.unload_all('restore')
        api.wait(api.call(f'/deployments/{original["id"]}/load',{'precision':original['precision'],'kernel_profile':original.get('kernel_profile','framework-native'),'quantization':recipe}),'restored-load')
        current=next(d for d in api.call('/deployments')['items'] if d['id']==original['id'])
        assert current['quantization']==original['quantization']
        assert current['precision']==original['precision'] and current.get('kernel_profile')==original.get('kernel_profile')
        summary['restored']=True
        api.save('restored',current)
        api.save('summary',summary)

if __name__=='__main__':main()
