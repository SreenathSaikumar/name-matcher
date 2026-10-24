"""Shared name cleanup for the local scorers (not used for raw Jev input)."""

import re

from namematch.constants import (
    JOIN_SPLIT_CUT_END,
    JOIN_SPLIT_CUT_START,
    JOIN_SPLIT_MIN_LEN,
)

# Titles are not identity signal.
HONORIFICS = {
    "smt", "shri", "shree", "kumari", "km", "mr", "mrs", "ms", "miss",
    "dr", "prof", "sir", "md", "jr", "sr", "late",
}

# Keep relative names aside. Deleting them makes siblings look like one person.
RELATION_MARKERS = {"s/o", "d/o", "w/o", "c/o", "so", "do", "wo", "co"}


def _clean(text: str) -> str:
    text = text.lower().strip().replace(".", " ")
    text = re.sub(r"[^\w\s/]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def indic_romanize(token: str) -> str:
    # Common Indian English spellings → one form before Double Metaphone.
    t = token.lower()
    for a, b in [
        ("ksh", "x"), ("aa", "a"), ("ee", "i"), ("ii", "i"),
        ("oo", "u"), ("uu", "u"), ("ph", "f"), ("qh", "k"),
    ]:
        t = t.replace(a, b)
    return t.replace("w", "v").replace("q", "k").replace("x", "ks")


def parse_name(name: str) -> dict:
    parts = _clean(name).split()
    relative, tokens = [], []
    i = 0
    while i < len(parts):
        p = parts[i]
        if p in RELATION_MARKERS and i + 1 < len(parts):
            relative = parts[i + 1 :]
            break
        # "S O" written with a space instead of "S/O"
        if p in {"s", "d", "w", "c"} and i + 1 < len(parts) and parts[i + 1] == "o":
            relative = parts[i + 2 :]
            break
        if p not in HONORIFICS:
            tokens.append(p)
        i += 1
    return {"tokens": tokens, "relative": relative}


def join_split_variants(tokens: list[str]) -> list[list[str]]:
    # IDs disagree on spacing: Ram Kumar vs Ramkumar.
    variants = [tokens]
    if len(tokens) >= 2:
        variants.append(tokens[:-2] + [tokens[-2] + tokens[-1]])
        variants.append(["".join(tokens)])
    if len(tokens) == 1 and len(tokens[0]) >= JOIN_SPLIT_MIN_LEN:
        t = tokens[0]
        for cut in range(JOIN_SPLIT_CUT_START, min(JOIN_SPLIT_CUT_END, len(t))):
            variants.append([t[:cut], t[cut:]])
    return variants


def syllable_count(token: str) -> int:
    groups = re.findall(r"[aeiou]+", token.lower())
    return max(1, len(groups)) if token else 0


def is_initial_of(short: str, long: str) -> bool:
    return len(short) == 1 and len(long) > 1 and long[0] == short[0]


def align(a_tokens: list[str], b_tokens: list[str], score_fn) -> float:
    # Order-independent: surname-first vs given-first must still score high.
    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0

    pairs = sorted(
        ((score_fn(a, b), i, j) for i, a in enumerate(a_tokens) for j, b in enumerate(b_tokens)),
        reverse=True,
    )
    used_a, used_b, scored = set(), set(), []
    for s, i, j in pairs:
        if s <= 0 or i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        scored.append(s)
    if not scored:
        return 0.0
    # Coverage penalty: leftover tokens hurt (dropped middle is OK; inventing overlap is not).
    coverage = len(scored) / max(len(a_tokens), len(b_tokens))
    return (sum(scored) / len(scored)) * coverage
