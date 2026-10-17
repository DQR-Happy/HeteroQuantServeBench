"""Family-free structure and real executed Tensor dependencies; CPU evidence only."""
import pytest

from hqsb.models.architecture_structure import artifact_structure, semantic_family


def test_unknown_family_has_structure_and_known_families_only_annotate():
    rows=[{'name':'custom.blocks.0.proj.weight','stored_bytes':64}]
    result=artifact_structure(rows)
    assert any(row['path']=='custom.blocks.0.proj' for row in result['modules'])
    assert result['status']=='complete'
    assert not semantic_family({'model_type':'brand_new_legal_model'})['recognized']
    for kind in ['llama','qwen3_next','qwen3_5_text','gemma4_text','deepseek_v3','glm4_moe','mistral','phi3','gpt_oss','minimax_m2','internlm2']:
        assert semantic_family({'model_type':kind})['recognized']


def test_dispatch_records_forks_without_invented_serial_dependencies():
    import torch
    from hqsb.backends.architecture_capture import ArchitectureCapture, module_inventory
    class VendorUnknown(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.query=torch.nn.Linear(4,4,bias=False)
            self.key=torch.nn.Linear(4,4,bias=False)
        def forward(self,x):
            return self.query(x)+self.key(x)
    model=VendorUnknown().eval()
    x=torch.randn(2,4)
    expected=model(x)
    capture=ArchitectureCapture(model)
    with torch.inference_mode(),capture.forward('prefill'):
        actual=model(x)
    torch.testing.assert_close(actual,expected)
    doc=capture.document()
    assert doc['status']=='captured'
    query=next(row for row in doc['nodes'] if row['op']=='aten.linear.default' and row['module']=='query')
    key=next(row for row in doc['nodes'] if row['op']=='aten.linear.default' and row['module']=='key')
    assert not any({edge['source'],edge['target']}=={query['id'],key['id']} for edge in doc['edges'])
    add=next(row for row in doc['nodes'] if row['op']=='aten.add.Tensor')
    assert {edge['source'] for edge in doc['edges'] if edge['target']==add['id']} >= {query['id'],key['id']}
    assert len(module_inventory(model)['modules'])==3
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())


def test_alias_write_dependency_and_trace_budget_preserve_results():
    import torch
    from hqsb.backends.architecture_capture import ArchitectureCapture
    class Writes(torch.nn.Module):
        def forward(self,x):
            alias=x.view(-1)
            x.add_(1)
            return alias*2
    model=Writes()
    capture=ArchitectureCapture(model)
    with torch.inference_mode(),capture.forward('prefill'):
        result=model(torch.zeros(4))
    assert result.tolist()==[2]*4
    doc=capture.document()
    add=next(row['id'] for row in doc['nodes'] if row['op']=='aten.add_.Tensor')
    mul=next(row['id'] for row in doc['nodes'] if row['op']=='aten.mul.Tensor')
    assert any(edge['source']==add and edge['target']==mul and edge['kind']=='storage_write' for edge in doc['edges'])
    bounded=ArchitectureCapture(model,max_nodes=2,max_edges=1)
    with torch.inference_mode(),bounded.forward('prefill'):
        actual=model(torch.zeros(4))
    assert actual.tolist()==[2]*4
    assert bounded.document()['truncated'] and bounded.document()['status']=='partial'


def test_hooks_are_removed_on_forward_failure():
    import torch
    from hqsb.backends.architecture_capture import ArchitectureCapture
    class Broken(torch.nn.Module):
        def forward(self,x):
            x.sin()
            raise RuntimeError('intentional model failure')
    model=Broken()
    capture=ArchitectureCapture(model)
    with pytest.raises(RuntimeError,match='intentional'),capture.forward('prefill'):
        model(torch.ones(2))
    assert not model._forward_hooks and not model._forward_pre_hooks
    assert capture.document()['status']=='partial'


def test_buffer_sources_and_inventory_limits_are_explicit():
    import torch
    from hqsb.backends.architecture_capture import ArchitectureCapture,module_inventory
    class Buffered(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('offset',torch.ones(2))
            for i in range(70):
                self.register_parameter('p'+str(i),torch.nn.Parameter(torch.zeros(1)))
        def forward(self,x):
            return x+self.offset
    model=Buffered()
    capture=ArchitectureCapture(model)
    with torch.inference_mode(),capture.forward('prefill'):
        assert model(torch.zeros(2)).tolist()==[1,1]
    assert any(node['kind']=='buffer' and node['names']==['offset'] for node in capture.document()['nodes'])
    inventory=module_inventory(model)
    assert inventory['status']=='partial' and len(inventory['modules'][0]['parameters'])==64


@pytest.mark.parametrize('kind',['llama','qwen2','qwen3','gemma3_text','mistral','phi3','deepseek_v3'])
def test_native_model_families_and_moe_use_the_same_collector(kind):
    import torch
    from transformers import AutoConfig,AutoModelForCausalLM
    from hqsb.backends.architecture_capture import ArchitectureCapture
    before=torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        fields=dict(hidden_size=16,intermediate_size=32,num_hidden_layers=1,
            num_attention_heads=4,num_key_value_heads=2,head_dim=4,vocab_size=32,max_position_embeddings=64,
            pad_token_id=0,bos_token_id=1,eos_token_id=2,partial_rotary_factor=1.0,
            q_lora_rank=4,kv_lora_rank=4,qk_nope_head_dim=2,qk_rope_head_dim=2,v_head_dim=4,
            n_routed_experts=2,num_experts_per_tok=1,n_shared_experts=1,moe_intermediate_size=8,
            first_k_dense_replace=0,n_group=1,topk_group=1)
        if kind=='deepseek_v3':
            fields.pop('head_dim')  # MLA uses qk_rope_head_dim rather than dense-attention head_dim.
            fields['num_key_value_heads']=4  # Expanded MLA K/V already have one head per query head.
            fields['num_local_experts']=2
        config=AutoConfig.for_model(kind,**fields)
        model=AutoModelForCausalLM.from_config(config,attn_implementation='eager',
            **({'experts_implementation':'eager'} if kind=='deepseek_v3' else {})).eval()
        ids=torch.tensor([[1,4,5]])
        with torch.inference_mode():
            expected=model(ids,use_cache=False).logits
            capture=ArchitectureCapture(model)
            with capture.forward('prefill'):
                actual=model(ids,use_cache=False).logits
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        doc=capture.document()
        assert doc['nodes'] and doc['edges'] and doc['phases'][0]['completed']
        assert any(row['op'] in {'aten.linear.default','aten.mm.default','aten.addmm.default'} for row in doc['nodes'])
        assert not any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())
    finally:
        torch.set_num_threads(before)
