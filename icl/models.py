"""Model backends.

HFModel  - local HuggingFace causal LM. Supports exact label scoring
           (log-probabilities of each label word), which is the standard way to
           evaluate ICL on classification and needs no answer parsing.
APIModel - any OpenAI-compatible endpoint (OpenAI, a local ``vllm serve``,
           Together, DeepSeek, ...). Generation only; classification falls back
           to "generate a few tokens and match the label word".
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def truncate_at_stop(text: str, stop: list[str] | None) -> str:
    if not stop:
        return text
    cut = len(text)
    for s in stop:
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    return text[:cut]


class HFModel:
    supports_scoring = True

    def __init__(self, name: str, batch_size: int = 8, device: str | None = None, dtype: str = "auto",
                 trust_remote_code: bool = False):
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

        self.tok = AutoTokenizer.from_pretrained(name, trust_remote_code=trust_remote_code)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        try:
            self.model = AutoModelForCausalLM.from_pretrained(name, dtype=torch_dtype,
                                                              trust_remote_code=trust_remote_code)
        except (ValueError, KeyError):
            # natively multimodal checkpoints (image-text-to-text) still work for text-only input
            from transformers import AutoModelForImageTextToText

            self.model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch_dtype,
                                                                     trust_remote_code=trust_remote_code)
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

    # -- generation ------------------------------------------------------

    def generate(self, prompts: list[str], max_new_tokens: int = 256, stop: list[str] | None = None) -> list[str]:
        torch = self.torch
        self.tok.padding_side = "left"
        outs: list[str] = [""] * len(prompts)
        order = np.argsort([-len(p) for p in prompts])
        with torch.no_grad():
            for b in range(0, len(order), self.batch_size):
                idx = order[b : b + self.batch_size]
                enc = self.tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(self.device)
                kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tok.pad_token_id)
                if stop:
                    kwargs.update(stop_strings=stop, tokenizer=self.tok)
                gen = self.model.generate(**enc, **kwargs)
                new = gen[:, enc.input_ids.shape[1] :]
                for i, text in zip(idx, self.tok.batch_decode(new, skip_special_tokens=True)):
                    outs[i] = truncate_at_stop(text, stop)
        self.tok.padding_side = "right"
        return outs


class APIModel:
    """OpenAI-compatible endpoint. ``mode='completion'`` sends the raw few-shot
    prompt (best for base models served by vLLM); ``mode='chat'`` wraps it in a
    single user message (for chat-only APIs)."""

    supports_scoring = False

    CHAT_SYSTEM = (
        "You will see examples of a task followed by a new input. Continue the pattern: "
        "reply with only the answer for the last input, in the same format as the examples."
    )

    def __init__(self, name: str, base_url: str | None = None, api_key: str | None = None,
                 mode: str = "chat", max_workers: int = 8):
        from openai import OpenAI

        self.name = name
        self.mode = mode
        self.max_workers = max_workers
        self.client = OpenAI(base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
                             api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"))
        self.max_len = 10**9  # unknown; let the server reject over-long prompts

    def n_tokens(self, text: str) -> int:
        return len(text) // 4  # rough estimate, only used for logging

    def _one(self, prompt: str, max_new_tokens: int, stop: list[str] | None) -> str:
        stop = (stop or [])[:4]  # OpenAI allows at most 4 stop sequences
        if self.mode == "completion":
            r = self.client.completions.create(model=self.name, prompt=prompt, max_tokens=max_new_tokens,
                                               temperature=0, stop=stop or None)
            text = r.choices[0].text
        else:
            r = self.client.chat.completions.create(
                model=self.name, temperature=0, max_tokens=max_new_tokens, stop=stop or None,
                messages=[{"role": "system", "content": self.CHAT_SYSTEM}, {"role": "user", "content": prompt}],
            )
            text = r.choices[0].message.content or ""
        return truncate_at_stop(text, stop)

    def chat(self, user: str, system: str | None = None, max_new_tokens: int = 1024, thinking: bool = False) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        r = self.client.chat.completions.create(model=self.name, messages=msgs, temperature=0,
                                                max_tokens=max_new_tokens)
        return r.choices[0].message.content or ""

    def generate(self, prompts: list[str], max_new_tokens: int = 256, stop: list[str] | None = None) -> list[str]:
        with ThreadPoolExecutor(self.max_workers) as ex:
            return list(ex.map(lambda p: self._one(p, max_new_tokens, stop), prompts))


def load_model(backend: str, name: str, **kw):
    if backend == "hf":
        return HFModel(name, **{k: v for k, v in kw.items() if k in ("batch_size", "device", "dtype", "trust_remote_code")})
    if backend == "api":
        return APIModel(name, **{k: v for k, v in kw.items() if k in ("base_url", "api_key", "mode", "max_workers")})
    raise ValueError(backend)
