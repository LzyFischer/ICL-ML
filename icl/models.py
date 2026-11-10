"""Model wrappers.

HFModel  - local HuggingFace causal LM: exact label scoring (log-probabilities
           of each label word) for evaluation and for the LLM selector, plus
           chat generation for the rule optimizer.
APIModel - optional: an OpenAI-compatible chat endpoint, only used as the
           rule optimizer (``optimize_rules.py --optimizer-backend api``).
"""

from __future__ import annotations

import os

import numpy as np


class HFModel:
    def __init__(self, name: str, batch_size: int = 8, device: str | None = None, dtype: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.name = name
        self.batch_size = batch_size
        if device is None:
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        if dtype == "auto":
            torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
        else:
            torch_dtype = getattr(torch, dtype)

        self.tok = AutoTokenizer.from_pretrained(name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        try:
            self.model = AutoModelForCausalLM.from_pretrained(name, dtype=torch_dtype)
        except (ValueError, KeyError):
            # natively multimodal checkpoints (image-text-to-text) still work for text-only input
            from transformers import AutoModelForImageTextToText

            self.model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch_dtype)
        self.model.to(device).eval()
        cfg = getattr(self.model.config, "text_config", None) or self.model.config
        self.max_len = getattr(cfg, "max_position_embeddings", None) or getattr(cfg, "n_positions", None) or 4096
        self.max_len = min(self.max_len, 32768)

    # -- utilities -------------------------------------------------------

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return self.tok(text, add_special_tokens=add_special_tokens).input_ids  # BOS if the model uses one

    @property
    def has_chat_template(self) -> bool:
        return bool(getattr(self.tok, "chat_template", None))

    def chat_prompt(self, user: str, system: str | None = None, answer_choices=("Yes", "No")):
        """Render a single-turn chat prompt (thinking disabled) ending right
        where the assistant's answer starts. Returns (text, choices) where the
        choices are spelled so that they continue the text correctly. Falls
        back to plain text for models without a chat template."""
        if self.has_chat_template:
            msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=False)
            return text, list(answer_choices)
        text = (system + "\n\n" if system else "") + user + "\nAnswer:"
        return text, [" " + c for c in answer_choices]

    def chat(self, user: str, system: str | None = None, max_new_tokens: int = 1024, thinking: bool = False) -> str:
        """Free-form chat generation (used by the rule optimizer)."""
        if self.has_chat_template:
            msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=thinking)
        else:
            text = (system + "\n\n" if system else "") + user + "\n"
        enc = self.tok(text, return_tensors="pt", add_special_tokens=not self.has_chat_template).to(self.device)
        budget = min(max_new_tokens, self.max_len - enc.input_ids.shape[1])
        if budget < 16:
            print(f"[warn] chat prompt ({enc.input_ids.shape[1]} tokens) does not fit the context window")
            return ""
        with self.torch.no_grad():
            gen = self.model.generate(**enc, max_new_tokens=budget, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
        return self.tok.decode(gen[0, enc.input_ids.shape[1]:], skip_special_tokens=True)

    def n_tokens(self, text: str) -> int:
        return len(self.encode(text))

    def _forward_padded(self, seqs: list[list[int]]):
        """Right-pad a batch of id lists and return log-softmax logits."""
        torch = self.torch
        L = max(len(s) for s in seqs)
        pad = self.tok.pad_token_id
        ids = torch.full((len(seqs), L), pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s)
            mask[i, : len(s)] = 1
        out = self.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device))
        return torch.log_softmax(out.logits.float(), dim=-1)

    # -- scoring ---------------------------------------------------------

    def score_choices(self, prompts: list[str], choices: list[str], add_special_tokens: bool = True) -> np.ndarray:
        """Return an [N, C] array: total log P(choice | prompt).

        Uses one forward pass per prompt when every choice is a single,
        distinct token (the common case), else scores each (prompt, choice)
        pair explicitly.
        """
        torch = self.torch
        choice_ids = [self.tok(c, add_special_tokens=False).input_ids for c in choices]
        prompt_ids = [self.encode(p, add_special_tokens) for p in prompts]
        single = all(len(c) == 1 for c in choice_ids) and len({c[0] for c in choice_ids}) == len(choice_ids)
        scores = np.zeros((len(prompts), len(choices)), dtype=np.float64)

        order = np.argsort([-len(p) for p in prompt_ids])  # length-sorted batches = less padding
        with torch.no_grad():
            if single:
                first = torch.tensor([c[0] for c in choice_ids], device=self.device)
                for b in range(0, len(order), self.batch_size):
                    idx = order[b : b + self.batch_size]
                    seqs = [prompt_ids[i] for i in idx]
                    logp = self._forward_padded(seqs)
                    last = torch.tensor([len(s) - 1 for s in seqs], device=self.device)
                    rows = logp[torch.arange(len(seqs), device=self.device), last]  # [B, V]
                    scores[idx] = rows[:, first].cpu().numpy()
            else:
                pairs = [(i, j) for i in order for j in range(len(choices))]
                for b in range(0, len(pairs), self.batch_size):
                    chunk = pairs[b : b + self.batch_size]
                    seqs = [prompt_ids[i] + choice_ids[j] for i, j in chunk]
                    logp = self._forward_padded(seqs)
                    for r, (i, j) in enumerate(chunk):
                        p_len, c = len(prompt_ids[i]), choice_ids[j]
                        pos = torch.arange(p_len - 1, p_len - 1 + len(c), device=self.device)
                        tgt = torch.tensor(c, device=self.device)
                        scores[i, j] = logp[r, pos, tgt].sum().item()
        return scores


class APIModel:
    """OpenAI-compatible chat endpoint (OpenAI, DeepSeek, a local ``vllm serve``, ...)."""

    def __init__(self, name: str, base_url: str | None = None, api_key: str | None = None):
        from openai import OpenAI

        self.name = name
        self.client = OpenAI(base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
                             api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"))

    def chat(self, user: str, system: str | None = None, max_new_tokens: int = 1024, thinking: bool = False) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        r = self.client.chat.completions.create(model=self.name, messages=msgs, temperature=0,
                                                max_tokens=max_new_tokens)
        return r.choices[0].message.content or ""
