"""Double Metaphone after Indic romanize: covers transliteration that JW misses."""

from metaphone import doublemetaphone

from namematch.constants import (
    PHONETIC_INITIAL_SCORE,
    PHONETIC_KEY_MATCH_SCORE,
    PHONETIC_MAX_LEN_DIFF,
    PHONETIC_MAX_SYLLABLE_DIFF,
)
from namematch.normalize import (
    align,
    indic_romanize,
    is_initial_of,
    join_split_variants,
    parse_name,
    syllable_count,
)


def _keys(token: str) -> set[str]:
    primary, alt = doublemetaphone(indic_romanize(token))
    return {k for k in (primary, alt) if k} or {token}


def _pair(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if is_initial_of(a, b) or is_initial_of(b, a):
        return PHONETIC_INITIAL_SCORE
    if (
        abs(len(a) - len(b)) > PHONETIC_MAX_LEN_DIFF
        or abs(syllable_count(a) - syllable_count(b)) > PHONETIC_MAX_SYLLABLE_DIFF
    ):
        return 0.0
    if not (_keys(a) & _keys(b)):
        return 0.0
    return PHONETIC_KEY_MATCH_SCORE * min(len(a), len(b)) / max(len(a), len(b))


def score(name_a: str, name_b: str) -> float:
    pa, pb = parse_name(name_a), parse_name(name_b)
    best = 0.0
    for va in join_split_variants(pa["tokens"]):
        for vb in join_split_variants(pb["tokens"]):
            best = max(best, align(va, vb, _pair))
    return best
