"""GPU logit selection and bounded token-trie constraints, independent of models."""

import torch


class ChoiceTrie:
    def __init__(self, tokenizer, choices, eos, limit, minimum):
        self.paths = [tokenizer.encode(c, add_special_tokens=False) for c in choices]
        if not eos or any(
            not p or len(p) > 64 or any(t in eos for t in p) for p in self.paths
        ):
            raise ValueError("L7_CHOICES_REQUIRE_EOS_AND_1_TO_64_NON_SPECIAL_TOKENS")
        if len({tuple(p) for p in self.paths}) != len(self.paths):
            raise ValueError("L7_CHOICES_HAVE_DUPLICATE_TOKEN_SEQUENCES")
        if limit < max(map(len, self.paths)) + 1 or minimum > min(map(len, self.paths)):
            raise ValueError(
                "L7_CHOICE_BUDGET_MUST_ALLOW_LONGEST_PATH_PLUS_EOS_AND_MINIMUM"
            )
        self.eos, self.root = set(eos), {}
        for path in self.paths:
            node = self.root
            for token in path:
                node = node.setdefault(token, {})
            node[None] = True

    def allowed(self, prefix):
        node = self.root
        for token in prefix:
            node = node[token]
        return sorted(
            [t for t in node if t is not None]
            + (list(self.eos) if None in node else [])
        )

    def valid(self, tokens):
        return bool(tokens and tokens[-1] in self.eos and tokens[:-1] in self.paths)


class Sampler:
    def __init__(self, plan, device, eos, trie=None):
        self.plan, self.eos, self.trie = plan, set(eos), trie
        self.generator = torch.Generator(device=device).manual_seed(plan.seed)
        self.calls = 0

    def scores(self, logits, history, generated):
        if logits.ndim != 1 or not bool(torch.isfinite(logits).all()):
            raise ValueError("L7_FINITE_SINGLE_VOCAB_LOGITS_REQUIRED")
        scores = logits.float().clone()
        if self.plan.repetition_penalty != 1:
            seen = torch.tensor(
                sorted(set(history)), device=scores.device, dtype=torch.long
            )
            old = scores[seen]
            scores[seen] = torch.where(
                old < 0,
                old * self.plan.repetition_penalty,
                old / self.plan.repetition_penalty,
            )
        if len(generated) < self.plan.min_new_tokens and self.eos:
            scores[list(self.eos)] = -torch.inf
        mapping = None
        if self.trie:
            allowed = self.trie.allowed(generated)
            if not allowed:
                raise ValueError("L7_CONSTRAINT_HAS_NO_VALID_SUCCESSOR")
            mapping = torch.tensor(allowed, device=scores.device, dtype=torch.long)
            if self.plan.constraint_backend == "gather":
                scores = scores[mapping]
            else:
                masked = torch.full_like(scores, -torch.inf)
                masked[mapping] = scores[mapping]
                scores, mapping = masked, None
        if self.plan.sampling == "multinomial":
            scores = scores / self.plan.temperature
            if self.plan.top_k:
                cutoff = torch.topk(
                    scores, min(self.plan.top_k, scores.numel())
                ).values[-1]
                scores = scores.masked_fill(scores < cutoff, -torch.inf)
            if self.plan.top_p < 1:
                ordered, indices = torch.sort(scores, descending=True, stable=True)
                remove = ordered.softmax(-1).cumsum(-1) > self.plan.top_p
                remove[1:] = remove[:-1].clone()
                remove[0] = False
                scores = scores.scatter(
                    0, indices, ordered.masked_fill(remove, -torch.inf)
                )
        if not bool(torch.isfinite(scores).any()):
            raise ValueError("L7_NO_FINITE_ALLOWED_LOGITS")
        return scores, mapping

    def select(self, logits, history, generated):
        scores, mapping = self.scores(logits, history, generated)
        if self.plan.sampling == "greedy":
            index = scores.argmax()
        else:
            index = torch.multinomial(scores.softmax(-1), 1, generator=self.generator)[
                0
            ]
        self.calls += 1
        return int((mapping[index] if mapping is not None else index).item())
