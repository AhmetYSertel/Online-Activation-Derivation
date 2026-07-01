"""
bfcl.py -- ROADMAP Phase B: a BFCL-grounded streaming workload, an AST-match
referee, and teacher adapters, replacing the author-written synthetic families.

WHY A *STREAMING* ADAPTATION (read this -- it is a modeling decision)
---------------------------------------------------------------------
BFCL v3 `simple` is a ONE-SHOT benchmark: 400 items, 370 unique function names, only
27 functions recur. OAD induces regularities from REPEATED decisions over a stable
tool set; on vanilla BFCL nothing recurs, so OAD would defer on everything -- the
regime its own paper says it is wrong for. So we do NOT run OAD on vanilla BFCL.

Instead we build the repetitive agent traffic OAD targets FROM real BFCL material:
  * TOOLS, SCHEMAS, and SURFACE TEMPLATES are taken verbatim from BFCL `simple`
    functions whose single required string argument appears as a substring of the
    question (38 such functions). Each becomes a single-slot family: the question
    with its argument value replaced by a slot.
  * REPETITION + SLOT VARIATION is the modeled part: a Zipfian stream over the stable
    tool set, each call filling the slot with a same-shape value. This is the only
    synthetic element -- it models "an agent that repeatedly invokes the same tools".
  * ABSTAIN PROBES are 100% real BFCL `irrelevance` items (a third-party
    should-not-call set), replacing the author-written adversarial probes (the F-1 /
    LIMITATIONS #3 critique).

So the tool distribution, schemas, phrasings, and the abstention test set are all
third-party; the agent-traffic repetition is the documented modeling choice. The
referee is BFCL-style AST matching. Everything is lowercased to match OAD's surface.
"""
from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from . import skeleton as sk

Action = Dict[str, object]
DECLINE: Action = {"tool": "__llm_other__", "params": {}}

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data", "bfcl")

_ALPHA_POOL = ("boston denver osaka cairo lisbon nairobi quito bergen dublin perth "
               "lima oslo madrid berlin tokyo seattle austin dallas miami atlanta "
               "geneva vienna prague warsaw athens cork galway turin naples porto "
               "kyoto sapporo busan incheon hanoi manila jakarta lagos accra tunis "
               "rabat amman doha muscat riga vilnius tallinn sofia zagreb skopje "
               "calgary ottawa regina laval ghent bruges leuven aalborg odense malmo").split()


def _read_jsonl(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _norm(s: str) -> str:
    return " ".join(sk.tokenize(s))   # lowercase, word-char tokens, space-joined


def _same_shape_fillers(value: str, k: int = 6) -> List[str]:
    """Generate k same-token-count, same-shape fillers including the original."""
    toks = _norm(value).split()
    if not toks:
        return []
    out = {" ".join(toks)}
    rng = random.Random(hash(value) & 0xFFFF)
    tries = 0
    while len(out) < k and tries < 200:
        tries += 1
        new = []
        for t in toks:
            if any(c.isdigit() for c in t):
                new.append(re.sub(r"\d+", lambda m: str(rng.randint(100, 999)), t))
            else:
                new.append(rng.choice(_ALPHA_POOL))
        out.add(" ".join(new))
    return list(out)


@dataclass
class BFCLFamily:
    tool: str
    arg: str
    prefix: str          # normalized surface before the slot
    suffix: str          # normalized surface after the slot
    fillers: List[str]

    def render(self, fill: str) -> str:
        return _norm(f"{self.prefix} {fill} {self.suffix}")

    def action_for(self, fill: str) -> Action:
        return {"tool": self.tool, "params": {self.arg: _norm(fill)}}


def build_families(max_families: int = 30, fillers_per_family: int = 6) -> List[BFCLFamily]:
    data = _read_jsonl(os.path.join(DATA, "BFCL_v3_simple.json"))
    ans = {e["id"]: e for e in _read_jsonl(
        os.path.join(DATA, "possible_answer", "BFCL_v3_simple.json"))}
    fams: List[BFCLFamily] = []
    seen_tools = set()
    for e in data:
        fn = e["function"][0]
        q = e["question"][0][0]["content"]
        gt = ans[e["id"]]["ground_truth"][0]
        name = list(gt)[0]
        if name in seen_tools:
            continue
        req = fn["parameters"].get("required", [])
        if len(req) != 1:
            continue
        a = req[0]
        vals = [v for v in gt[name].get(a, []) if isinstance(v, str) and v]
        val = next((v for v in vals if v.lower() in q.lower()), None)
        if not val:
            continue
        qn = _norm(q)
        vn = _norm(val)
        idx = qn.find(vn)
        if idx < 0:
            continue
        prefix = qn[:idx].strip()
        suffix = qn[idx + len(vn):].strip()
        if not prefix:                       # need a literal anchor before the slot
            continue
        fams.append(BFCLFamily(name, a, prefix, suffix,
                               _same_shape_fillers(val, k=fillers_per_family)))
        seen_tools.add(name)
        if len(fams) >= max_families:
            break
    return fams


def load_irrelevance(n: int = 120) -> List[str]:
    items = _read_jsonl(os.path.join(DATA, "BFCL_v3_irrelevance.json"))
    return [_norm(e["question"][0][0]["content"]) for e in items[:n]]


# --------------------------------------------------------------------------- #
# AST-match referee (BFCL-style): name must match; the produced arg value must be
# in the accepted set (here, the single filler). DECLINE matches DECLINE.
# --------------------------------------------------------------------------- #
def bfcl_score(produced: Optional[Action], gt: Action) -> bool:
    if gt.get("tool") == DECLINE["tool"]:
        return bool(produced) and str(produced.get("tool", "")).startswith("__")
    if not produced or produced.get("tool") != gt.get("tool"):
        return False
    gp, pp = gt.get("params", {}), produced.get("params", {})
    if set(gp) != set(pp):
        return False
    return all(_norm(str(pp.get(k, ""))) == _norm(str(v)) for k, v in gp.items())


# --------------------------------------------------------------------------- #
# Workload: Zipfian stream over families, mixed with real irrelevance probes.
# --------------------------------------------------------------------------- #
class BFCLWorkload:
    def __init__(self, seed: int = 0, families: Optional[List[BFCLFamily]] = None,
                 irrelevance: Optional[List[str]] = None):
        self.rng = random.Random(seed)
        self.families = families if families is not None else build_families()
        self.irrelevance = irrelevance if irrelevance is not None else load_irrelevance()
        self._by_text: Dict[str, Action] = {}     # memo of family ground truth
        for fam in self.families:
            for fill in fam.fillers:
                self._by_text[fam.render(fill)] = fam.action_for(fill)
        for t in self.irrelevance:
            self._by_text.setdefault(t, dict(DECLINE))

    def ground_truth(self, text: str) -> Action:
        return self._by_text.get(_norm(text), dict(DECLINE))

    def stream(self, n: int, junk_frac: float = 0.15, zipf_s: float = 0.8) -> List[str]:
        k = len(self.families)
        w = [1.0 / ((i + 1) ** zipf_s) for i in range(k)]
        out: List[str] = []
        for _ in range(n):
            if self.irrelevance and self.rng.random() < junk_frac:
                out.append(self.rng.choice(self.irrelevance))
            else:
                fam = self.rng.choices(self.families, weights=w, k=1)[0]
                out.append(fam.render(self.rng.choice(fam.fillers)))
        return out

    def tool_specs(self) -> List[dict]:
        """OpenAI function-calling specs for the whole stable tool set (offered every
        call), plus a decline option -- the realistic relevance/irrelevance setting."""
        specs = []
        for fam in self.families:
            specs.append({"type": "function", "function": {
                "name": fam.tool,
                "description": f"Handle requests of the form: {fam.prefix} <{fam.arg}> {fam.suffix}".strip(),
                "parameters": {"type": "object",
                               "properties": {fam.arg: {"type": "string"}},
                               "required": [fam.arg]}}})
        specs.append({"type": "function", "function": {
            "name": "__llm_other__",
            "description": "Use when no other tool fits the request.",
            "parameters": {"type": "object", "properties": {}}}})
        return specs


# --------------------------------------------------------------------------- #
# Teachers
# --------------------------------------------------------------------------- #
class OracleBFCLTeacher:
    """Offline stand-in: returns workload ground truth. Validates the pipeline without
    an API key. NOT a real LLM -- only the experimenter knows it is perfect."""
    def __init__(self, wl: BFCLWorkload):
        self._gt = wl.ground_truth

    def __call__(self, text: str) -> Action:
        return self._gt(text)


class OpenAIBFCLTeacher:
    """Real GPT teacher over the BFCL tool set via OpenAI function calling. Offers the
    full stable tool set every call (so declining on irrelevance is a real decision).

        export OPENAI_API_KEY=sk-...
        teacher = OpenAIBFCLTeacher(wl, model="gpt-5.5")
    """
    def __init__(self, wl: BFCLWorkload, model: str = "gpt-5.5",
                 temperature: float = 0.0):
        from openai import OpenAI
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("Set OPENAI_API_KEY to use OpenAIBFCLTeacher.")
        self._client = OpenAI()
        self.model = model
        # OpenAI function names must match ^[a-zA-Z0-9_-]+$, but BFCL tools are dotted
        # (e.g. "biology.get_cell_info"). Sanitize for the API and keep a reverse map so
        # the returned call name is translated back to the real dotted tool name.
        self.specs = []
        self._name_map: Dict[str, str] = {}
        for s in wl.tool_specs():
            orig = s["function"]["name"]
            san = re.sub(r"[^a-zA-Z0-9_-]", "_", orig)
            self._name_map[san] = orig
            spec = {"type": "function",
                    "function": {**s["function"], "name": san}}
            self.specs.append(spec)
        self.temperature = temperature
        self.system = ("You are a tool-routing assistant. For each request, call "
                       "exactly one available tool with the correct argument, or call "
                       "__llm_other__ if none fits. Do not answer in text.")

    def __call__(self, text: str) -> Action:
        kw = dict(model=self.model, tools=self.specs, tool_choice="required",
                  messages=[{"role": "system", "content": self.system},
                            {"role": "user", "content": text}])
        try:
            resp = self._client.chat.completions.create(temperature=self.temperature, **kw)
        except Exception as e:
            if "temperature" in str(e):
                resp = self._client.chat.completions.create(**kw)
            else:
                raise
        calls = getattr(resp.choices[0].message, "tool_calls", None) or []
        if not calls:
            return dict(DECLINE)
        name = self._name_map.get(calls[0].function.name, calls[0].function.name)
        try:
            args = json.loads(calls[0].function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        if name == "__llm_other__":
            return dict(DECLINE)
        return {"tool": name, "params": {k: _norm(str(v)) for k, v in args.items()}}
