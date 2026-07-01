"""
The teacher: the source of (input -> action) supervision the system distills online.
The engine ONLY ever queries the teacher (and the execute channel); it never reads
the referee.

Three implementations, all interchangeable behind the same call signature
``teacher(text) -> Action``:

  OracleTeacher   -- backed by the workload ground truth. Perfect by construction.
                     CAVEAT: not a real LLM (see workload.py).
  NoisyTeacher    -- wraps any teacher and corrupts a fraction `delta` of its labels,
                     modelling a frozen LLM that is imperfect.
  OpenAITeacher   -- a REAL GPT model (default gpt-5.5, the current GA flagship as of
                     2026-06) via the OpenAI API + function calling. Drop-in for the
                     oracle. Requires `pip install openai` and OPENAI_API_KEY. This is
                     the path to a transfer claim; it is NOT exercised in the packaged
                     results because the build environment has no API key.

Also here: make_execute(), the production-confirmation channel (optionally noisy).
"""
from __future__ import annotations

import json
import os
import random
from typing import Callable, Dict, List, Optional

from .workload import Action, Family, FALLBACK, default_families

Teacher = Callable[[str], Action]


# --------------------------------------------------------------------------- #
# Oracle teacher (synthetic, perfect).
# --------------------------------------------------------------------------- #
class OracleTeacher:
    """Returns the workload's exact correct action. Perfect by construction -- a
    *mechanism* probe, not a real LLM. The engine cannot tell it apart from a real
    teacher at the interface; only the experimenter knows it is an oracle."""

    def __init__(self, ground_truth: Callable[[str], Action]):
        self._gt = ground_truth

    def __call__(self, text: str) -> Action:
        return self._gt(text)


# --------------------------------------------------------------------------- #
# Noisy teacher (models an imperfect frozen LLM).
# --------------------------------------------------------------------------- #
class NoisyTeacher:
    """Wraps a base teacher and, with probability `delta`, returns a WRONG action
    (a different tool drawn from the known tool set, or a corrupted parameter). This
    is the label-noise channel: a real frozen LLM makes mistakes, and "learning from a
    frozen LLM" must be tested under that noise. delta=0 reduces to the base teacher."""

    def __init__(self, base: Teacher, delta: float, rng: random.Random,
                 tools: Optional[List[str]] = None):
        self.base = base
        self.delta = delta
        self.rng = rng
        self.tools = tools or [f.tool for f in default_families()]

    def __call__(self, text: str) -> Action:
        a = self.base(text)
        if self.rng.random() >= self.delta or a.get("tool") == FALLBACK["tool"]:
            return a
        # corrupt: swap the tool to a different one (a plausible tool-selection error)
        others = [t for t in self.tools if t != a.get("tool")]
        if not others:
            return a
        wrong = dict(a)
        wrong["tool"] = self.rng.choice(others)
        return wrong


# --------------------------------------------------------------------------- #
# Real GPT teacher (gpt-5.5 by default). NOT run in packaged results.
# --------------------------------------------------------------------------- #
def openai_tools_from_families(families: List[Family]) -> List[dict]:
    """Build OpenAI function-calling tool specs from the workload families, so a real
    model is asked to select exactly the tools the referee scores against."""
    tools = []
    for f in families:
        tools.append({
            "type": "function",
            "function": {
                "name": f.tool,
                "description": f"Handle requests like: {f.phrasing.replace('{x}', '<value>')}",
                "parameters": {
                    "type": "object",
                    "properties": {f.param: {"type": "string"}},
                    "required": [f.param],
                },
            },
        })
    # an explicit escape hatch so the model can decline rather than hallucinate a call
    tools.append({
        "type": "function",
        "function": {
            "name": "__llm_other__",
            "description": "Use when no other tool fits the request.",
            "parameters": {"type": "object", "properties": {}},
        },
    })
    return tools


class OpenAITeacher:
    """A real GPT model as the teacher, via OpenAI function calling.

    Usage (in the author's own environment):
        pip install openai
        export OPENAI_API_KEY=sk-...
        teacher = OpenAITeacher(model="gpt-5.5", families=default_families())

    Notes:
      * model default is "gpt-5.5" (current GA flagship, 2026-06). "gpt-5.4" is a
        cheaper production option; "gpt-5.6" was preview-only at time of writing.
      * Returns the parsed tool call as {"tool":..., "params":{...}}; falls back to
        FALLBACK if the model declines or returns no tool call.
      * Consider wrapping with a cache in the caller to avoid paying per repeated input
        (the workload stream is Zipfian and repeats heavily).
    """

    def __init__(self, model: str = "gpt-5.5", families: Optional[List[Family]] = None,
                 system: Optional[str] = None, temperature: float = 0.0):
        try:
            from openai import OpenAI  # imported lazily so the package works without it
        except Exception as e:  # pragma: no cover - exercised only with the SDK present
            raise RuntimeError(
                "OpenAITeacher requires the openai package: pip install openai"
            ) from e
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("Set OPENAI_API_KEY to use OpenAITeacher.")
        self._client = OpenAI()
        self.model = model
        self.families = families or default_families()
        self.tools = openai_tools_from_families(self.families)
        self.temperature = temperature
        self.system = system or (
            "You are a tool-routing assistant. For each user request, call exactly one "
            "of the available tools with the correct argument, or call __llm_other__ if "
            "none fits. Do not answer in text."
        )

    def __call__(self, text: str) -> Action:  # pragma: no cover - needs a live API key
        kwargs = dict(
            model=self.model,
            tools=self.tools,
            tool_choice="required",
            messages=[{"role": "system", "content": self.system},
                      {"role": "user", "content": text}],
        )
        try:
            resp = self._client.chat.completions.create(
                temperature=self.temperature, **kwargs)
        except Exception as e:
            # Newer reasoning models (gpt-5.x) reject a non-default temperature.
            # Retry once at the model's default sampling rather than failing.
            if "temperature" in str(e):
                resp = self._client.chat.completions.create(**kwargs)
            else:
                raise
        msg = resp.choices[0].message
        calls = getattr(msg, "tool_calls", None) or []
        if not calls:
            return dict(FALLBACK)
        call = calls[0]
        name = call.function.name
        try:
            args = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        if name == "__llm_other__":
            return dict(FALLBACK)
        # normalize values to lowercase tokens to match the tokenizer's surface form
        params = {k: str(v).lower() for k, v in args.items()}
        return {"tool": name, "params": params}


# --------------------------------------------------------------------------- #
# Production-confirmation channel (execution), optionally noisy.
# --------------------------------------------------------------------------- #
def make_execute(ground_truth: Callable[[str], Action], eps: float,
                 rng: random.Random) -> Callable[[str, Action], bool]:
    """Returns execute(text, produced) -> bool. A correct action always confirms; a
    WRONG action confirms with probability eps (noisy execution / wrong-but-non-
    crashing). eps=0 is a clean executor."""
    def execute(text: str, produced: Action) -> bool:
        if produced == ground_truth(text):
            return True
        return rng.random() < eps
    return execute
