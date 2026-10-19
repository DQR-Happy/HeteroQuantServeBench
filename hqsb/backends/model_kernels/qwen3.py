"""Qwen3 dense FP16 adapter for the verified Transformers source revision.

Only model-instance methods change. Canonical Parameter objects, checkpoint
keys and cache ownership stay with the original model. No global monkeypatch.
"""
import hashlib
import inspect
from pathlib import Path

SOURCE_SHA256 = "fbdcfeeb1b54135ca67ba7df924da92f4b264e1252b517eea2de989289ebaeab"


def bind(model, session):
    import torch
    from transformers.models.qwen3 import modeling_qwen3 as source

    actual = hashlib.sha256(Path(inspect.getfile(source)).read_bytes()).hexdigest()
    cfg = model.config
    if actual != SOURCE_SHA256 or cfg.model_type != "qwen3":
        raise ValueError("KERNEL_MODEL_SOURCE_UNSUPPORTED:" + actual)
    if model.training or cfg.hidden_act != "silu" or getattr(cfg, "attention_dropout", 0) != 0:
        raise ValueError("KERNEL_MODEL_SEMANTICS_UNSUPPORTED")
    if (cfg.hidden_size, cfg.intermediate_size, cfg.num_attention_heads,
        cfg.num_key_value_heads, cfg.num_hidden_layers) != (2048, 6144, 16, 8, 28):
        raise ValueError("KERNEL_MODEL_SHAPE_UNSUPPORTED")
    for layer in model.model.layers:
        if layer.self_attn.sliding_window is not None:
            raise ValueError("KERNEL_SLIDING_WINDOW_UNSUPPORTED")
    before = {name: (id(p), p.data_ptr(), tuple(p.shape)) for name,p in model.named_parameters(remove_duplicate=False)}
    state_keys = tuple(model.state_dict())
    choices = session.profile["operators"]
    if choices.get("attention") == "torch-sdpa-flash-gqa-repeat-v1":
        original_forward = model.forward
        session.unpadded_causal = False

        def guarded_forward(self, *args, **kwargs):
            ids = kwargs.get("input_ids", args[0] if args else None)
            mask = kwargs.get("attention_mask")
            cache = kwargs.get("past_key_values")
            if ids is None or ids.ndim != 2 or ids.shape[0] != 1 or kwargs.get("inputs_embeds") is not None:
                raise ValueError("KERNEL_FLASH_BATCH1_TOKEN_INPUT_REQUIRED")
            if cache is not None and not isinstance(cache, source.DynamicCache):
                raise ValueError("KERNEL_FLASH_DYNAMIC_CACHE_ONLY")
            # One boundary check per model forward, not one sync per layer.
            # Its overhead is part of this research provider's total cost.
            if mask is not None and (mask.ndim != 2 or mask.shape[0] != 1 or not bool((mask == 1).all().item())):
                raise ValueError("KERNEL_FLASH_PADDING_OR_CUSTOM_MASK_UNSUPPORTED")
            session.unpadded_causal = True
            try:
                return original_forward(*args, **kwargs)
            finally:
                session.unpadded_causal = False
        session.install(model, guarded_forward)
        session.binding["attention_mask_contract"] = "batch1_unpadded_causal_checked_once_per_model_forward"
    ops = torch.ops.hqsb
    session.binding.update(adapter="qwen3-dense-source-pinned-v1", model_source_sha256=actual,
                           model_config_sha256=hashlib.sha256(cfg.to_json_string().encode()).hexdigest(),
                           parameter_count=len(before), parameter_map_sha256=hashlib.sha256(
                               repr([(name, shape) for name,(_,_,shape) in before.items()]).encode()).hexdigest())

    def norm_forward(path, module):
        variant = choices["rms_norm"]
        site = session.site(path, "rms_norm", variant)
        number = 2 if variant == "cuda-hf-half2-v2" else 1

        def forward(self, x):
            out = ops.l2_rms_norm(x, self.weight, self.variance_epsilon, number)
            session.hit(site)
            return out
        session.install(module, forward)

    if "rms_norm" in choices:
        census = 0
        for path,module in model.named_modules():
            if isinstance(module, source.Qwen3RMSNorm):
                census += 1
                # The residual fusion owns this logical norm; do not execute twice.
                if "residual_rms_norm" in choices and path.endswith("post_attention_layernorm"):
                    continue
                norm_forward(path, module)
        if census != 113:
            raise ValueError("KERNEL_NORM_CENSUS_MISMATCH")
        session.binding["norm_census"] = census

    for index,layer in enumerate(model.model.layers):
        prefix = f"model.layers.{index}"
        if "residual_rms_norm" in choices:
            site = session.site(prefix + ".post_attention_layernorm", "residual_rms_norm",
                                choices["residual_rms_norm"], covers=("residual_add", "post_attention_layernorm"))

            def decoder(self, hidden_states, attention_mask=None, position_ids=None, past_key_values=None,
                        use_cache=False, position_embeddings=None, _site=site, **kwargs):
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
                hidden_states, _ = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask,
                    position_ids=position_ids, past_key_values=past_key_values, use_cache=use_cache,
                    position_embeddings=position_embeddings, **kwargs)
                norm = self.post_attention_layernorm
                hidden_states, residual = ops.l2_residual_norm(hidden_states, residual, norm.weight, norm.variance_epsilon)
                session.hit(_site)
                return residual + self.mlp(hidden_states)
            session.install(layer, decoder)

        if "silu_mul" in choices:
            site = session.site(prefix + ".mlp", "silu_mul", choices["silu_mul"])

            def mlp(self, x, _site=site):
                product = ops.l2_silu_mul(self.gate_proj(x), self.up_proj(x))
                session.hit(_site)
                return self.down_proj(product)
            session.install(layer.mlp, mlp)

        if "rope" in choices or "attention" in choices:
            rope_site = session.site(prefix + ".self_attn", "rope", choices["rope"]) if "rope" in choices else None
            attn_site = session.site(prefix + ".self_attn", "attention", choices["attention"]) if "attention" in choices else None

            def attention(self, hidden_states, position_embeddings, attention_mask, past_key_values=None,
                          _rope=rope_site, _attn=attn_site, **kwargs):
                if kwargs.get("output_attentions") and _attn is not None:
                    raise ValueError("KERNEL_ATTENTION_WEIGHTS_UNSUPPORTED")
                shape = hidden_states.shape[:-1]
                head_shape = (*shape, -1, self.head_dim)
                q = self.q_norm(self.q_proj(hidden_states).view(head_shape))
                k = self.k_norm(self.k_proj(hidden_states).view(head_shape))
                v = self.v_proj(hidden_states).view(head_shape).transpose(1,2)
                cos,sin = position_embeddings
                if _rope is not None:
                    q,k = ops.l2_rope(q,cos,sin), ops.l2_rope(k,cos,sin)
                    session.hit(_rope)
                else:
                    q,k = source.apply_rotary_pos_emb(q.transpose(1,2), k.transpose(1,2), cos,sin)
                if past_key_values is not None:
                    k,v = past_key_values.update(k,v,self.layer_idx)
                if _attn is not None:
                    from hqsb.backends.model_kernels.sdpa import run

                    out = run(q,k,v,attention_mask,self.scaling,choices["attention"],
                              unpadded_causal=getattr(session, "unpadded_causal", False))
                    repeat_bytes = 2 * q.shape[0] * q.shape[1] * k.shape[2] * q.shape[3] * q.element_size() if q.shape[1] != k.shape[1] else 0
                    session.repeat_peak_bytes = max(session.repeat_peak_bytes, repeat_bytes)
                    session.repeat_total_bytes += repeat_bytes
                    weights = None
                    session.hit(_attn)
                else:
                    interface = source.ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation,
                                                                            source.eager_attention_forward)
                    out,weights = interface(self,q,k,v,attention_mask,dropout=0.0,scaling=self.scaling,
                                            sliding_window=self.sliding_window,**kwargs)
                return self.o_proj(out.reshape(*shape,-1).contiguous()),weights
            session.install(layer.self_attn, attention)

    if "linear" in choices:
        for path,module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if module.bias is not None:
                    raise ValueError("KERNEL_LINEAR_BIAS_UNSUPPORTED")
                # Select using the fixed, predeclared rule.
                lt_site = session.site(path, "linear", "cublaslt-fp32-acc-v1")
                gemv_site = session.site(path, "linear_small_m", "cuda-small-m-gemv-v1") if choices["linear"] == "cuda-small-m-gemv-v1" else None

                def linear(self, x, _lt=lt_site, _gemv=gemv_site):
                    small = _gemv is not None and x.numel() // x.shape[-1] <= 4
                    out = ops.l2_linear(x, self.weight, small)
                    session.hit(_gemv if small else _lt)
                    return out
                session.install(module, linear)
        # The two linear branches are alternatives, not both mandatory per run.
        for row in session.sites:
            if row["op"] in ("linear", "linear_small_m"):
                row["coverage_group"] = row["module_path"] + ":linear"
    after = {name: (id(p), p.data_ptr(), tuple(p.shape)) for name,p in model.named_parameters(remove_duplicate=False)}
    if before != after or state_keys != tuple(model.state_dict()):
        raise RuntimeError("KERNEL_PARAMETER_IDENTITY_CHANGED")
