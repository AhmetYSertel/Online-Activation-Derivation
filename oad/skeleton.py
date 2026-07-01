"""
Skeleton representation, tokenization, anchor-glob matching, and cross-example
slot induction (anti-unification).

A *skeleton* is the input's surface structure with variable positions abstracted.
It is represented as an ordered list of segments, each one of:

    ('lit',  (tok, tok, ...))   a fixed literal anchor
    ('slot', slot_id)           a variable span (captures >= 1 token)

This module is purely symbolic. No embeddings, no learned weights. The induction
of which positions are slots is a *cross-example* operation (a single example
cannot tell you which token varies); see ``induce_skeleton``.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Sequence, Tuple

Token = str
Segment = Tuple[str, object]  # ('lit', tuple[str,...]) | ('slot', int)

_TOK_RE = re.compile(r"[a-z0-9']+")

# Small, fixed stopword set used only to identify "content" words for the
# novel-content-word coherence check. Not learned.
STOPWORDS = frozenset(
    "a an the is are was were be been being to of in on at for from with my your "
    "me i you it this that these those what whats what's how hows how's please can "
    "could would do does did show find get tell give about and or".split()
)


def tokenize(text: str) -> List[Token]:
    """Lowercase + split on word characters. Digits are kept (so '1945' is a token)."""
    return _TOK_RE.findall(text.lower())


def content_words(tokens: Sequence[Token]) -> List[Token]:
    return [t for t in tokens if t not in STOPWORDS]


# --------------------------------------------------------------------------- #
# Matching: anchor-glob match of a skeleton against an input token sequence.
# --------------------------------------------------------------------------- #
def _find_subseq(tokens: Sequence[Token], anchor: Sequence[Token], start: int) -> Optional[int]:
    """Index >= start where tokens[idx:idx+len(anchor)] == anchor, else None."""
    if not anchor:
        return start
    L = len(anchor)
    for idx in range(start, len(tokens) - L + 1):
        if list(tokens[idx:idx + L]) == list(anchor):
            return idx
    return None


def match(skeleton: Sequence[Segment], tokens: Sequence[Token]) -> Optional[Dict[int, Tuple[Token, ...]]]:
    """
    Strict anchor-glob match. Literal segments must appear in order; slot
    segments capture the (non-empty) spans between anchors. Returns
    {slot_id: captured_tokens} on success, or None.
    """
    pos = 0
    i = 0
    n = len(skeleton)
    slots: Dict[int, Tuple[Token, ...]] = {}
    while i < n:
        kind, payload = skeleton[i]
        if kind == "lit":
            anchor = payload  # tuple of tokens
            L = len(anchor)
            if list(tokens[pos:pos + L]) != list(anchor):
                return None
            pos += L
            i += 1
        else:  # slot
            slot_id = payload
            # Look ahead for the next literal anchor (if any).
            if i + 1 < n and skeleton[i + 1][0] == "lit":
                anchor = skeleton[i + 1][1]
                found = _find_subseq(tokens, anchor, pos + 1)  # slot captures >= 1 token
                if found is None:
                    return None
                slots[slot_id] = tuple(tokens[pos:found])
                pos = found
                i += 1
            else:
                # Trailing slot: capture the rest (>= 1 token).
                if pos >= len(tokens):
                    return None
                slots[slot_id] = tuple(tokens[pos:])
                pos = len(tokens)
                i += 1
    if pos != len(tokens):
        return None
    return slots


def skeleton_literals(skeleton: Sequence[Segment]) -> List[Token]:
    out: List[Token] = []
    for kind, payload in skeleton:
        if kind == "lit":
            out.extend(payload)
    return out


def skeleton_str(skeleton: Sequence[Segment]) -> str:
    parts = []
    for kind, payload in skeleton:
        parts.append(" ".join(payload) if kind == "lit" else f"<{payload}>")
    return " ".join(parts)


def n_slots(skeleton: Sequence[Segment]) -> int:
    return sum(1 for k, _ in skeleton if k == "slot")


# --------------------------------------------------------------------------- #
# Induction: anti-unify a set of example token sequences into a skeleton.
#
# Design note (faithful to the spec): a regularity holds examples that share the
# same fixed phrasing and differ only in slot spans. Different phrasings form
# *different* skeletons (handled at assignment, not here). Induction therefore
# aligns every example to the first (reference) example, marks positions that
# ever vary as slot regions, and emits maximal literal/slot segments.
# --------------------------------------------------------------------------- #
def induce_skeleton(examples: Sequence[Sequence[Token]]) -> List[Segment]:
    seqs = [list(e) for e in examples]
    if not seqs:
        return []
    ref = seqs[0]
    if len(seqs) == 1:
        # Born fully literal -> over-specific. Loosens as more examples arrive.
        return [("lit", tuple(ref))] if ref else []

    var = [False] * len(ref)          # ref position is variable
    insert_after = [False] * (len(ref) + 1)  # an inserted slot sits in this gap

    for other in seqs[1:]:
        sm = SequenceMatcher(a=ref, b=other, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            if tag in ("replace", "delete"):
                for p in range(i1, i2):
                    var[p] = True
                if tag == "replace" and i1 == i2:  # pure insert flagged as replace edge
                    insert_after[i1] = True
            elif tag == "insert":
                insert_after[i1] = True

    # Build segments from ref + var mask + insertion gaps.
    segs: List[Segment] = []
    lit_buf: List[Token] = []
    slot_id = 0

    def flush_lit():
        nonlocal lit_buf
        if lit_buf:
            segs.append(("lit", tuple(lit_buf)))
            lit_buf = []

    def add_slot():
        nonlocal slot_id
        # Merge with a trailing slot if present.
        if segs and segs[-1][0] == "slot":
            return
        segs.append(("slot", slot_id))
        slot_id += 1

    for p in range(len(ref)):
        if insert_after[p]:
            flush_lit()
            add_slot()
        if var[p]:
            flush_lit()
            add_slot()
        else:
            lit_buf.append(ref[p])
    if insert_after[len(ref)]:
        flush_lit()
        add_slot()
    flush_lit()

    # Renumber slot ids left-to-right for stability.
    renum: List[Segment] = []
    k = 0
    for kind, payload in segs:
        if kind == "slot":
            renum.append(("slot", k))
            k += 1
        else:
            renum.append((kind, payload))
    return renum
