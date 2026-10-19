"""Single-sequence greedy speculation with real block verification and KV rollback.

Model/cache ownership stays here; no HTTP, tokenizer formatting or streaming I/O.
At each published round the cache contains committed history except its last token.
"""

import time
import torch
from transformers.cache_utils import DynamicCache


class DecodingStopped(Exception):
    pass


def cache_for(model, total):
    config = (
        model.config.get_text_config()
        if hasattr(model.config, "get_text_config")
        else model.config
    )
    window = getattr(config, "sliding_window", None)
    if window and total >= window:
        raise ValueError("L7_ROLLBACK_REQUIRES_CONTEXT_BELOW_SLIDING_WINDOW")
    return DynamicCache(config=model.config)


def crop(cache, length):
    before = cache.get_seq_length()
    if not 0 <= length <= before:
        raise ValueError("L7_INVALID_CACHE_ROLLBACK_LENGTH")
    cache.crop(length)
    if cache.get_seq_length() != length:
        raise ValueError("L7_CACHE_ROLLBACK_NOT_APPLIED")
    return before - length


def lookup(history, plan, count):
    begin = max(0, len(history) - plan.lookup_window)
    for n in range(
        min(plan.lookup_max_ngram, len(history) - 1), plan.lookup_min_ngram - 1, -1
    ):
        suffix = history[-n:]
        for pos in range(len(history) - n - 1, begin - 1, -1):
            if history[pos : pos + n] == suffix:
                return history[pos + n : min(pos + n + count, len(history))]
    return []


class DraftSequence:
    def __init__(self, model, total, check, eos):
        self.model, self.check, self.eos = model, check, eos
        self.cache = cache_for(model, total)
        self.consumed = []
        self.calls = 0
        self.rollback_tokens = 0

    def forward(self, tokens):
        self.check()
        inputs = torch.tensor([tokens], device=self.model.device, dtype=torch.long)
        out = self.model(input_ids=inputs, past_key_values=self.cache, use_cache=True)
        self.cache = out.past_key_values
        self.calls += 1
        return out.logits[0, -1, :]

    def propose(self, history, count):
        common = 0
        for a, b in zip(self.consumed, history):
            if a != b:
                break
            common += 1
        keep = min(common, len(history) - 1)
        self.rollback_tokens += crop(self.cache, keep)
        logits = self.forward(history[keep:])
        self.consumed = history.copy()
        proposed = []
        for i in range(count):
            self.check()
            token = int(logits.argmax().item())
            proposed.append(token)
            if token in self.eos or i + 1 == count:
                break
            logits = self.forward([token])
            self.consumed.append(token)
        return proposed

    def close(self):
        self.cache = None
        self.consumed = []


class Decoder:
    def __init__(
        self, model, plan, sampler, input_ids, limit, eos, check, draft_model=None
    ):
        self.model, self.plan, self.sampler = model, plan, sampler
        self.history = list(input_ids)
        self.input_count = len(input_ids)
        self.limit, self.eos, self.check = limit, set(eos), check
        self.cache = cache_for(model, len(input_ids) + limit)
        self.draft = (
            DraftSequence(draft_model, len(input_ids) + limit, check, eos)
            if draft_model is not None
            else None
        )
        self.stats = {
            "target_forward_calls": 0,
            "target_input_tokens": 0,
            "verification_calls": 0,
            "proposed_tokens": 0,
            "accepted_tokens": 0,
            "discarded_proposal_tokens": 0,
            "target_rollback_tokens": 0,
            "rounds": [],
            "rounds_truncated": False,
            "proposal_host_ms": 0.0,
        }
        self.generated = []

    def forward(self, tokens):
        self.check()
        inputs = torch.tensor([tokens], device=self.model.device, dtype=torch.long)
        out = self.model(input_ids=inputs, past_key_values=self.cache, use_cache=True)
        self.cache = out.past_key_values
        if not isinstance(self.cache, DynamicCache):
            raise ValueError("L7_DYNAMIC_CROP_CAPABLE_CACHE_REQUIRED")
        self.stats["target_forward_calls"] += 1
        self.stats["target_input_tokens"] += len(tokens)
        return out.logits[0]

    def rounds(self):
        with torch.inference_mode():
            initial = self.forward(self.history)[-1:]
            while len(self.generated) < self.limit:
                self.check()
                remaining = self.limit - len(self.generated)
                begin = time.monotonic()
                if self.plan.method == "prompt_lookup":
                    proposals = lookup(
                        self.history,
                        self.plan,
                        min(self.plan.draft_tokens, remaining - 1),
                    )
                elif self.draft and remaining > 1:
                    proposals = self.draft.propose(
                        self.history, min(self.plan.draft_tokens, remaining - 1)
                    )
                else:
                    proposals = []
                self.stats["proposal_host_ms"] += (time.monotonic() - begin) * 1000
                n = len(proposals)
                if n or initial is None:
                    tokens = (
                        proposals
                        if initial is not None
                        else [self.history[-1]] + proposals
                    )
                    scores = self.forward(tokens)
                    scores = (
                        torch.cat((initial, scores), dim=0)
                        if initial is not None
                        else scores
                    )
                else:
                    scores = initial
                initial = None
                accepted = 0
                if self.plan.method == "autoregressive":
                    emitted = [
                        self.sampler.select(scores[-1], self.history, self.generated)
                    ]
                else:
                    if not bool(torch.isfinite(scores).all()):
                        raise ValueError("L7_NONFINITE_VERIFY_LOGITS")
                    predictions = scores.argmax(-1).tolist()
                    for proposed, predicted in zip(proposals, predictions):
                        if proposed != predicted:
                            break
                        accepted += 1
                    emitted = proposals[:accepted] + [predictions[accepted]]
                emitted = emitted[:remaining]
                if not getattr(self, "fixed", False):
                    for index, token in enumerate(emitted):
                        if token in self.eos:
                            emitted = emitted[: index + 1]
                            break
                # Trim rejected/superfluous future states before exposing any token.
                self.history.extend(emitted)
                self.generated.extend(emitted)
                rolled = crop(self.cache, len(self.history) - 1)
                retained = min(accepted, len(emitted))
                self.stats["target_rollback_tokens"] += rolled
                self.stats["proposed_tokens"] += n
                self.stats["accepted_tokens"] += retained
                self.stats["discarded_proposal_tokens"] += n - retained
                self.stats["verification_calls"] += int(n > 0)
                round_doc = {
                    "proposed": n,
                    "accepted": retained,
                    "emitted": len(emitted),
                    "target_rolled_back": rolled,
                    "committed_tokens": len(self.history),
                    "target_cache_tokens": self.cache.get_seq_length(),
                }
                if len(self.stats["rounds"]) < 256:
                    self.stats["rounds"].append(round_doc)
                else:
                    self.stats["rounds_truncated"] = True
                yield emitted
                if emitted[-1] in self.eos and not getattr(self, "fixed", False):
                    break

    def summary(self):
        s = dict(self.stats)
        s.update(
            draft_forward_calls=self.draft.calls if self.draft else 0,
            draft_rollback_tokens=self.draft.rollback_tokens if self.draft else 0,
            generated_tokens=len(self.generated),
            target_cache_tokens=self.cache.get_seq_length()
            if self.cache is not None
            else 0,
            acceptance_rate=s["accepted_tokens"] / s["proposed_tokens"]
            if s["proposed_tokens"]
            else None,
        )
        return s

    def close(self):
        self.cache = None
        if self.draft:
            self.draft.close()
