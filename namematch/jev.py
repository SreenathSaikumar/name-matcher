"""TypeSafe Jev (Noul): criteria-driven same-person score on raw KYC strings."""

import hashlib
import json
from pathlib import Path

from dotenv import load_dotenv

# Frozen on purpose — rewriting after seeing errors is train-on-test.
# TypeSafe Noul keys are true/false, not yes/no.
JEV_CRITERIA = {
    "true": (
        "Same natural person on Indian KYC docs. Allow initials vs expanded names, "
        "surname-first order, transliteration (Mohammed/Mohammad, Lakshmi/Laxmi), "
        "missing middle names, honorifics, and Ram Kumar vs Ramkumar joins."
    ),
    "false": (
        "Different people. Reject siblings / near-duplicate givens with same surname, "
        "wrong initial expansions, gender near-phonetic pairs, shared surname alone."
    ),
}

CRITERIA_HASH = hashlib.sha256(json.dumps(JEV_CRITERIA, sort_keys=True).encode()).hexdigest()[:16]
CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "jev_scores.json"


def _key(pair_id: str, name_a: str, name_b: str) -> str:
    # Include criteria hash so an old prompt cannot reuse scores silently.
    return hashlib.sha256(
        f"{pair_id}|{name_a}|{name_b}|{CRITERIA_HASH}".encode()
    ).hexdigest()


def load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {"criteria_hash": CRITERIA_HASH, "model": None, "scores": {}}
    return json.loads(CACHE_PATH.read_text())


def save_cache(cache: dict) -> None:
    cache["criteria"] = JEV_CRITERIA
    cache["criteria_hash"] = CRITERIA_HASH
    CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")


def _live(name_a: str, name_b: str) -> tuple[float, str | None]:
    load_dotenv()
    from typesafe_sdk import Noul, TypeSafeClient

    with TypeSafeClient() as client:
        resp = client.system_one(
            state={"name_a": name_a, "name_b": name_b},
            questions={
                "same_person": Noul(
                    instructions="Do these two KYC name strings refer to the same person?",
                    criteria=JEV_CRITERIA,
                )
            },
        )
    prob = float(resp.answers["same_person"].noul)
    model = getattr(resp, "model", None)
    return prob, str(model) if model else None


def get_score(pair_id: str, name_a: str, name_b: str, cache: dict, refresh: bool = False) -> float:
    # Raw strings on purpose — we want Jev on document text, not our normalize pipeline.
    key = _key(pair_id, name_a, name_b)
    scores = cache.setdefault("scores", {})
    if not refresh and key in scores:
        return float(scores[key]["score"])
    if not refresh:
        raise KeyError(
            f"Missing Jev cache for id={pair_id}. Run: uv run python scripts/run_eval.py --refresh-jev"
        )

    prob, model = _live(name_a, name_b)
    scores[key] = {"id": pair_id, "name_a": name_a, "name_b": name_b, "score": prob}
    if model:
        cache["model"] = model
    return prob
