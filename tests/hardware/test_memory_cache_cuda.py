"""Actual CUDA correctness/lifetime checks for L6, independent of downloaded weights."""
import torch
import pytest
from transformers import LlamaConfig, LlamaForCausalLM
from types import SimpleNamespace
from hqsb.core.precision import PrecisionPlan
from hqsb.backends.memory_session import MemorySession, clear_prefix
from hqsb.backends.step_executor import StepExecutor
from hqsb.core.contracts.scheduling import SchedulingLimits, Step

pytestmark = [pytest.mark.hardware, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA target required")]


def provider():
    torch.manual_seed(13)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
        attn_implementation="sdpa")).eval().half().cuda()
    return SimpleNamespace(model=model, config={"context_limit": 128, "precision": "float16", "attention": "sdpa"},
        precision_plan=PrecisionPlan(), kernel_session=None, source_identity={"weights": "cuda-test"},
        tokenizer_identity={"tokenizer": "test"}, source_environment={"source_digest": "cuda-test"},
        memory=lambda:{"peak_allocated_bytes":torch.cuda.max_memory_allocated(), "allocated_bytes":torch.cuda.memory_allocated()})


def test_cuda_cache_modes_prefix_offload_and_cancel_lifetime():
    p = provider()
    prompt = [1, 2, 3, 4, 5, 6]
    reference = []
    for mode in ("dynamic", "static", "offloaded_dynamic", "offloaded_static", "recompute"):
        session = MemorySession(p, {"memory_plan": {"mode": mode}})
        cache, _ = session.prepare(prompt, 5)
        with torch.inference_mode():
            for index in range(5):
                # Teacher force the SAME history to measure numerical drift,
                # independently of a near-tied greedy token decision.
                history = prompt + [7, 8, 9, 10][:index]
                inputs = history if not session.use_cache or index==0 else history[-1:]
                out = p.model(input_ids=torch.tensor([inputs],device='cuda'),attention_mask=torch.ones(1,len(history),device='cuda'),past_key_values=cache,use_cache=session.use_cache)
                if mode=='dynamic': reference.append(out.logits[:,-1].float().clone())
                else: torch.testing.assert_close(out.logits[:,-1].float(),reference[index],atol=0.003,rtol=0.02)
                cache = out.past_key_values if session.use_cache else None
                session.after_forward(cache,len(inputs),0.)
        doc = session.finish(True)
        assert doc['completion']=='cuda_all_streams_synchronized'
        if mode.startswith('offloaded'):
            assert doc['transfers']['d2h_bytes']>0 and doc['transfers']['h2d_bytes']>0
            assert doc['storage_before_release']['bytes_by_device']['cpu']>0
        del out,cache,session
    # A cancelled run must not publish an entry.
    cancelled = MemorySession(p, {'memory_plan':{'mode':'dynamic','prefix_reuse':True}})
    cache,_=cancelled.prepare(prompt,5)
    with torch.inference_mode():
        out=p.model(input_ids=torch.tensor([prompt],device='cuda'),past_key_values=cache,use_cache=True)
        cancelled.after_forward(out.past_key_values,len(prompt),0.)
    cancelled.finish(False)
    assert len(p.prefix_store.entries)==0
    assert clear_prefix(p)['released_entries']==0
    del cancelled,p
    torch.cuda.empty_cache()


def test_cuda_block_pool_matches_dense_and_releases_after_completion():
    p = provider()
    results=[]
    for layout in ('dense','block_pool'):
        limits=SchedulingLimits(kv_layout=layout,kv_block_tokens=4,kv_capacity_tokens=64,kv_pool_max_bytes=1024**2)
        executor=StepExecutor(p,1024,limits=limits)
        events=[]
        try:
            events.append(executor.execute([Step('a',0,(1,2,3),True,'prefill'),Step('b',0,(4,),True,'prefill')], 'first'))
            events.append(executor.execute([Step('b',1,(5,),True,'decode'),Step('a',3,(6,),True,'decode')], 'second'))
            executor.release('b')
            events.append(executor.execute([Step('a',4,(7,),True,'decode'),Step('c',0,(8,9),True,'prefill')],'join'))
            results.append(events)
            if layout=='block_pool':
                assert events[1].receipt['scatter_copy_bytes']<results[0][1].receipt['scatter_copy_bytes']
                assert events[-1].receipt['memory_management']['reused_blocks']>0
        finally:
            cleanup=executor.close()
            assert cleanup['device_kv_rows']==0 and cleanup['resident_kv_bytes']==0
    for a,b in zip(*results):
        assert a.tokens==b.tokens
        torch.testing.assert_close(torch.tensor(a.receipt['selected_logprobs']),torch.tensor(b.receipt['selected_logprobs']),atol=.002,rtol=.01)


def test_l4_default_quantized_kv_preserves_physical_leaf_accounting():
    pytest.importorskip('optimum.quanto')
    from hqsb.backends.precision_runtime import cache_receipt
    p=provider()
    p.model=LlamaForCausalLM(LlamaConfig(vocab_size=64,hidden_size=128,intermediate_size=256,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2)).eval().half().cuda()
    p.precision_plan=PrecisionPlan(kv_bits=4,kv_group_size=32,kv_residual_length=4)
    session=MemorySession(p,{'memory_plan':{'mode':'model_default'}})
    cache,_=session.prepare([1,2,3,4,5,6],2)
    with torch.inference_mode():
        out=p.model(input_ids=torch.tensor([[1,2,3,4,5,6]],device='cuda'),past_key_values=cache,use_cache=True)
        session.after_forward(out.past_key_values,6,0.)
    expected=cache_receipt(out.past_key_values)
    assert expected['quantized_tensor_leaves']>0
    doc=session.finish(True)
    assert sum(doc['storage_before_release']['bytes_by_device'].values())==expected['bytes']
    assert any('_quantized_keys' in t['role'] for t in doc['storage_before_release']['tensors'])
