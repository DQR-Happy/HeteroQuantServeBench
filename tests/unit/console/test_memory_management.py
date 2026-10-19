"""L6 contract, actual HF cache semantics and immutable prefix ownership."""
from types import SimpleNamespace
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from hqsb.core.memory_plan import MemoryPlan, validate_memory
from hqsb.backends.memory_session import MemorySession, MemoryPlanError
from hqsb.backends.block_kv import BlockKV
from hqsb.core.precision import PrecisionPlan


def provider():
    torch.manual_seed(7)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1)).eval()
    return SimpleNamespace(model=model, config={"context_limit": 128, "precision": "float32", "attention": "sdpa"},
        precision_plan=PrecisionPlan(), kernel_session=None, source_identity={"weights": "test"},
        tokenizer_identity={"tokenizer": "test"}, source_environment={"source_digest": "test"})


def run(p, mode, prefix=False, namespace="default"):
    session = MemorySession(p, {"memory_plan": {"mode": mode, "prefix_reuse": prefix, "prefix_namespace": namespace}})
    cache, reused = session.prepare([1, 2, 3, 4], 4)
    current = torch.tensor([[1, 2, 3, 4]])[:, reused:]
    tokens, logits = [], []
    with torch.inference_mode():
        for index in range(4):
            out = p.model(input_ids=current, attention_mask=torch.ones(1, 4 + index), past_key_values=cache, use_cache=session.use_cache)
            selected = int(out.logits[0, -1].argmax())
            tokens.append(selected)
            logits.append(out.logits[0, -1].clone())
            cache = out.past_key_values if session.use_cache else None
            session.after_forward(cache, current.shape[-1], 0.0)
            current = torch.tensor([[selected]]) if session.use_cache else torch.cat((current, torch.tensor([[selected]])), -1)
    return session, tokens, logits


def test_contract_and_composition():
    for data in ({"prefix_reuse": True}, {"mode": "dynamic", "max_cache_tokens": 8}, {"prefix_reuse": "false"}, {"mode": "paged_attention"}):
        with pytest.raises(ValueError):
            MemoryPlan.model_validate(data)
    for options in ({"native": False}, {"kv_bits": 4}, {"compiled": True}, {"quality": True}):
        with pytest.raises(ValueError):
            validate_memory({"mode": "static"}, **options)
    assert validate_memory({"mode": "model_default"}, kv_bits=4)


def test_real_cache_modes_and_prefix_do_not_alias(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    p = provider()
    baseline, tokens, logits = run(p, "dynamic", True)
    baseline.finish(True)
    hit, actual, actual_logits = run(p, "dynamic", True)
    assert hit.reused == 3 and actual == tokens
    for a, b in zip(logits, actual_logits):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
    assert p.prefix_store.retained_bytes > 0
    stored = next(iter(p.prefix_store.entries.values()))[1]
    assert stored[0][0].shape[-2] == 3  # decode did not mutate shared prefix
    assert stored[0][0].untyped_storage().data_ptr() != hit.cache.layers[0].keys.untyped_storage().data_ptr()
    for mode in ("static", "recompute", "model_default"):
        session, actual, values = run(p, mode)
        assert actual == tokens
        for a, b in zip(logits, values):
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
        doc = session.finish(True)
        assert doc["status"] == "applied"
        if mode == "recompute":
            assert not doc["use_cache"] and doc["processed_input_positions"] == 22
            assert not doc["storage_before_release"]["tensors"]
    miss, _, _ = run(p, "dynamic", True, namespace="other")
    assert miss.reused == 0
    miss.finish(False)
    p.prefix_store.trim(0, 0)
    assert p.prefix_store.retained_bytes == 0


def test_capacity_rejects_before_tensor_allocation():
    p = provider()
    session = MemorySession(p, {"memory_plan": {"mode": "static", "max_cache_tokens": 4}})
    with pytest.raises(MemoryPlanError, match="required=7"):
        session.prepare([1, 2, 3, 4], 4)
    assert session.cache is None


def test_real_block_pool_roundtrip_tail_waste_and_recycle():
    from transformers import DynamicCache
    pool = BlockKV(torch, block_size=4, capacity_blocks=4, max_bytes=65536)
    def dense(values):
        t = torch.tensor(values, dtype=torch.float32).reshape(len(values), 1, -1, 1)
        return DynamicCache(ddp_cache_data=[(t, t + 100)])
    cache = dense([[1, 2, 3], [0, 0, 9]])
    assert pool.commit(cache, ["a", "b"], [0, 0], [3, 1]) == 32
    packed, copied = pool.pack(["b", "a"], [1, 3])
    assert packed.layers[0].keys[:, 0, :, 0].tolist() == [[0, 0, 9], [1, 2, 3]]
    assert copied == 32
    # Existing prefix preserved; only the new suffix is copied back.
    appended = dense([[0, 0, 9, 10], [1, 2, 3, 4]])
    assert pool.commit(appended, ["b", "a"], [1, 3], [1, 1]) == 16
    assert pool.receipt()["tail_waste_tokens"] == 2
    assert pool.refusal(["a"], [3], [1]) == "BLOCK_KV_STALE_LENGTH"
    pool.release("a")
    assert pool.commit(dense([[8]]), ["c"], [0], [1]) == 8
    assert pool.receipt()["reused_blocks"] > 0
    pool.clear()
    assert not pool.rows and pool.bytes() == 0 and len(pool.free) == 4


def test_quantized_wrapper_inventory_counts_real_leaves_and_aliases():
    from hqsb.backends.memory_session import inventory
    class Wrapped:
        def __init__(self, data, scale): self.data,self.scale=data,scale
        def __tensor_flatten__(self): return ['data','scale'], None
    codes=torch.zeros(4,dtype=torch.uint8)
    scale=torch.ones(1)
    cache=SimpleNamespace(layers=[SimpleNamespace(keys=scale,values=scale,
        _quantized_keys=Wrapped(codes,scale),_quantized_values=Wrapped(codes,scale))])
    doc=inventory(cache)
    assert doc['bytes_by_device']['cpu']==codes.untyped_storage().nbytes()+scale.untyped_storage().nbytes()
    assert any('_quantized_keys.data'==t['role'] for t in doc['tensors'])


def test_comparison_rejects_confounded_or_incomplete_pairs():
    from copy import deepcopy
    from hqsb.core.memory_comparison import compare
    a={'id':'a','state':'completed','config':{'epoch':'same'},'metrics':{'generated_token_ids':[1,2],'output_tokens':2,
        'memory_receipt':{'selected_token_logprobs':[-1.,-2.]}}}
    b=deepcopy(a); b['id']='b'
    receipt={'status':'applied','model_identity':'model','tokenizer_identity':'tokenizer', 'source':{'source_digest':'source'},
        'proof':{'input_token_sha256':'input','requested_token_limit':2}, 'runtime':{'dtype':'float16'}}
    manifest={'status':'complete'}
    assert compare(a,b,receipt,receipt,manifest,manifest)['generation_check']['exact_tokens']
    b['metrics']['generated_token_ids']=[1,3]
    check=compare(a,b,receipt,receipt,manifest,manifest)['generation_check']
    assert not check['exact_tokens'] and check['max_abs_selected_token_logprob_delta'] is None
    b['config']['epoch']='changed'
    assert 'epoch_mismatch' in compare(a,b,receipt,receipt,manifest,manifest)['reasons']
    changed=deepcopy(receipt);changed['proof']['input_token_sha256']='other'
    assert not compare(a,b,receipt,changed,manifest,manifest)['comparable']
    assert not compare(a,a,receipt,receipt,{'status':'partial'},manifest)['comparable']


def test_serial_worker_controls_epoch_busy_and_sealed_integrity(tmp_path):
    import json
    from fastapi.testclient import TestClient
    from hqsb.console.app import create_app
    from hqsb.console.config import Settings, DeploymentConfig
    from hqsb.console.service import ConsoleService
    from hqsb.experiments.artifacts import seal
    calls=[]
    class Actor:
        def __init__(self,config): pass
        def close(self): pass
        def execute(self,kind,payload,*args):
            calls.append(kind)
            return {'completion':'cuda_synchronized'}
    settings=Settings(data_dir=tmp_path/'state',evidence_root=tmp_path,web_dist=tmp_path/'web',
        deployments=[DeploymentConfig(id='model',name='test')])
    svc=ConsoleService(settings,actor_factory=Actor)
    svc.states['model'].update(state='ready',epoch='epoch')
    headers={'X-HQSB-Client':'console','Idempotency-Key':'test'}
    with TestClient(create_app(settings,service=svc,monitor=False)) as client:
        base='/api/console/v1'
        assert client.get(base+'/memory/catalog').status_code==200
        assert client.post(base+'/memory/prefix/clear',headers=headers,json={'deployment_id':'model','expected_epoch':'stale'}).status_code==409
        assert not calls
        svc.pending=1
        assert client.get(base+'/memory/capabilities?deployment_id=model').status_code==409
        assert not calls
        svc.pending=0
        assert client.post(base+'/memory/prefix/clear',headers=headers,json={'deployment_id':'model','expected_epoch':'epoch'}).status_code==200
        assert calls==['memory_cache_clear']
        run,_=svc.store.create('generate',{'deployment':{'id':'model'},'epoch':'epoch','memory_plan':{'mode':'static'}},'record','record')
        directory=settings.data_dir/'captures'/run['id'];directory.mkdir(parents=True)
        doc={'run_id':run['id'],'epoch':'epoch','status':'applied'}
        receipt={'run_id':run['id'],'memory_receipt':doc}
        svc.store.update(run['id'],{'state':'completed','metrics':{'memory_receipt':doc}})
        (directory/'memory-cache.json').write_text(json.dumps(doc))
        (directory/'execution-receipt.json').write_text(json.dumps(receipt))
        seal(directory,run_id=run['id'])
        assert client.get(base+'/runs/'+run['id']+'/memory-cache').json()['publication']=='supervisor_acknowledged'
        (directory/'memory-cache.json').write_text(json.dumps({**doc,'epoch':'other'}))
        assert client.get(base+'/runs/'+run['id']+'/memory-cache').status_code==409
