"""Token Jaro–Winkler: good for order, missing middles, initials — weak on transliteration."""

import jellyfish

from namematch.constants import JW_INITIAL_SCORE, JW_RELATIVE_BONUS
from namematch.normalize import align, is_initial_of, join_split_variants, parse_name


def _pair(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if is_initial_of(a, b) or is_initial_of(b, a):
        return JW_INITIAL_SCORE
    return jellyfish.jaro_winkler_similarity(a, b)


def score(name_a: str, name_b: str) -> float:
    pa, pb = parse_name(name_a), parse_name(name_b)
    best = 0.0
    for va in join_split_variants(pa["tokens"]):
        for vb in join_split_variants(pb["tokens"]):
            s = align(va, vb, _pair)
            if pa["relative"] and pb["relative"]:
                s = min(1.0, s + JW_RELATIVE_BONUS * align(pa["relative"], pb["relative"], _pair))
            best = max(best, s)
    return best
