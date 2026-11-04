"""Model backends for the LLM Worker.

Two providers, same interface:

* ``OpenRouterBackend`` -- the paper's own setup (deepseek-v4-flash). Use it
  to reproduce their measured A3 rate.
* ``AnthropicBackend`` -- the official Anthropic SDK. This measures a
  *different* model's self-healing rate, which is a different experiment: it
  tells you whether MTPO's correctness claim holds on a frontier model, not
  whether the paper's number replicates.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# USD per token: (prompt, completion)
PRICING: Dict[str, Tuple[float, float]] = {
    # OpenRouter list price
    "deepseek/deepseek-v4-flash": (0.091e-6, 0.182e-6),
    "deepseek/deepseek-v4-pro": (1.600e-6, 3.200e-6),
    "deepseek/deepseek-v4.1-flash": (0.045e-6, 0.500e-6),
    # DeepSeek first-party, peak rate, cache miss
    "deepseek-flash": (0.30e-6, 1.20e-6),
    "deepseek-v4-pro": (1.32e-6, 3.96e-6),
    # Anthropic list price
    "claude-opus-5": (5.00e-6, 25.00e-6),
    "claude-opus-4-8": (5.00e-6, 25.00e-6),
    "claude-sonnet-5": (2.00e-6, 10.00e-6),
    "claude-haiku-4-5": (1.00e-6, 5.00e-6),
    "claude-fable-5-1": (10.00e-6, 50.00e-6),
}

DEFAULTS = {
    "openrouter": "deepseek/deepseek-v4-flash",
    "anthropic": "claude-opus-5",
}


def price(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pin, pout = PRICING.get(model, (0.0, 0.0))
    return prompt_tokens * pin + completion_tokens * pout


def load_key(var: str) -> Optional[str]:
    """Environment first, then a gitignored .env. Never logged or printed."""
    key = os.environ.get(var)
    if key:
        return key
    for base in (Path.cwd(), Path(__file__).resolve().parents[3]):
        envfile = base / ".env"
        if not envfile.is_file():
            continue
        for line in envfile.read_text().splitlines():
            line = line.strip()
            if line.startswith(f"{var}="):
                return line.split("=", 1)[1].strip().strip("\"'")
    return None


@dataclass
class Completion:
    parsed: Dict[str, Any]
    latency: float
    prompt_tokens: int
    completion_tokens: int
    raw: str = ""


def extract_json(text: str) -> Dict[str, Any]:
    """Tolerant parse: the loop must survive a model that adds prose."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    start = None
    return {"thought": text.strip()[:500], "done": True, "_unparsed": True}


class OpenRouterBackend:
    """The paper's provider, over the OpenAI-compatible chat endpoint.

    DeepSeek's first-party API speaks the same protocol, so pointing
    ``OPENROUTER_BASE_URL`` at ``https://api.deepseek.com/v1`` and supplying
    ``DEEPSEEK_API_KEY`` runs the same code against it. Their model names
    differ (``deepseek-flash``, not ``deepseek-v4-flash``), and for this
    workload they are dearer per token -- the loop re-reads a growing context
    rather than hitting a stable cached prefix, so their cache-hit rate does
    not apply.
    """

    provider = "openrouter"
    DEFAULT_BASE = "https://openrouter.ai/api/v1"

    def __init__(self, model: Optional[str] = None, temperature: float = 0.0):
        self.model = model or DEFAULTS["openrouter"]
        self.temperature = temperature
        base = os.environ.get("OPENROUTER_BASE_URL") or self.DEFAULT_BASE
        self.endpoint = base.rstrip("/") + "/chat/completions"
        self.api_key = load_key("OPENROUTER_API_KEY") or load_key("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "No key found. Set OPENROUTER_API_KEY (or DEEPSEEK_API_KEY "
                "with OPENROUTER_BASE_URL) in the environment or .env"
            )

    def complete(self, system: str, messages: List[Dict[str, str]]) -> Completion:
        import requests

        t0 = time.time()
        resp = requests.post(
            self.endpoint,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            json={
                "model": self.model,
                "messages": [{"role": "system", "content": system}] + messages,
                "temperature": self.temperature,
                "response_format": {"type": "json_object"},
            },
            timeout=180,
        )
        latency = time.time() - t0
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage", {}) or {}
        text = data["choices"][0]["message"]["content"]
        return Completion(
            parsed=extract_json(text), latency=latency,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            raw=text,
        )


class AnthropicBackend:
    """Official Anthropic SDK.

    Adaptive thinking is on by default: judging whether a notification
    invalidates a premise is exactly the reasoning step under measurement, so
    suppressing it would measure the wrong thing. ``effort`` trades depth
    against spend.
    """

    provider = "anthropic"

    def __init__(
        self,
        model: Optional[str] = None,
        effort: str = "medium",
        thinking: bool = True,
        max_tokens: int = 4096,
    ):
        import anthropic

        self.model = model or DEFAULTS["anthropic"]
        self.effort = effort
        self.thinking = thinking
        self.max_tokens = max_tokens
        key = load_key("ANTHROPIC_API_KEY")
        # A bare client also resolves an `ant auth login` profile, so only
        # fail when neither a key nor a profile is available.
        self.client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()

    # Haiku 4.5 and older models reject `output_config.effort`, and take
    # thinking as {"type": "enabled", budget_tokens: N} rather than adaptive.
    LEGACY = ("haiku-4-5", "sonnet-4-5", "haiku-3", "sonnet-3", "opus-3")

    @property
    def _legacy(self) -> bool:
        return any(tag in self.model for tag in self.LEGACY)

    def complete(self, system: str, messages: List[Dict[str, str]]) -> Completion:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": messages,
        }
        if self._legacy:
            if self.thinking:
                budget = max(1024, min(self.max_tokens - 512, 2048))
                kwargs["thinking"] = {"type": "enabled",
                                      "budget_tokens": budget}
        else:
            kwargs["output_config"] = {"effort": self.effort}
            if self.thinking:
                kwargs["thinking"] = {"type": "adaptive"}
        t0 = time.time()
        resp = self.client.messages.create(**kwargs)
        latency = time.time() - t0
        text = "".join(
            b.text for b in resp.content if getattr(b, "type", "") == "text"
        )
        if resp.stop_reason == "refusal":
            return Completion(
                parsed={"thought": "model declined", "done": True,
                        "_refused": True},
                latency=latency,
                prompt_tokens=resp.usage.input_tokens,
                completion_tokens=resp.usage.output_tokens,
                raw=text,
            )
        return Completion(
            parsed=extract_json(text), latency=latency,
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
            raw=text,
        )


def make_backend(provider: str, **kw):
    if provider == "anthropic":
        return AnthropicBackend(**kw)
    if provider == "openrouter":
        kw.pop("effort", None)
        kw.pop("thinking", None)
        kw.pop("max_tokens", None)
        return OpenRouterBackend(**kw)
    raise ValueError(f"unknown provider: {provider}")
