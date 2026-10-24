"""Train/holdout eval: tune thresholds on train, ship-call on synthetic holdout."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from namematch import indic_phonetic, jev, token_jw
from namematch.constants import (
    FALLBACK_T_HIGH,
    FALLBACK_T_LOW,
    JEV_CHECKPOINT_EVERY,
    JEV_PROGRESS_EVERY,
    MIN_AUTO_MATCHES_TRAIN,
    MIN_PRECISION,
    MIN_RECALL,
    SPLIT_SEED,
    T_HIGH_GRID_END,
    T_HIGH_GRID_START,
    T_HIGH_GRID_STEPS,
    T_LOW_GRID_END,
    T_LOW_GRID_START,
    T_LOW_GRID_STEPS,
    TEST_SIZE,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "name_pairs.csv"
THRESHOLDS = ROOT / "data" / "thresholds.json"
RESULTS = ROOT / "results"


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    # Stratify so rare categories are not all stuck in train or holdout by chance.
    strat = df["category"].astype(str) + "|" + df["label"].astype(str)
    _, holdout_idx = train_test_split(
        df.index, test_size=TEST_SIZE, random_state=SPLIT_SEED, stratify=strat
    )
    df = df.copy()
    df["split"] = "train"
    df.loc[holdout_idx, "split"] = "holdout"
    return df


def band(score: float, t_low: float, t_high: float) -> str:
    # Middle band: uncertain pairs cost ops time, not fraud or silent reject.
    if score >= t_high:
        return "match"
    if score >= t_low:
        return "review"
    return "non_match"


def score_all(df: pd.DataFrame, refresh_jev: bool = False) -> pd.DataFrame:
    out = df.copy()
    out["score_token_jw"] = [
        token_jw.score(a, b) for a, b in zip(out["name_a"], out["name_b"], strict=True)
    ]
    out["score_indic_phonetic"] = [
        indic_phonetic.score(a, b) for a, b in zip(out["name_a"], out["name_b"], strict=True)
    ]

    cache = jev.load_cache()
    scores = []
    n = len(out)
    for i, row in enumerate(out.itertuples(index=False), start=1):
        if refresh_jev and (i == 1 or i % JEV_PROGRESS_EVERY == 0 or i == n):
            print(f"Jev refresh {i}/{n} ...", flush=True)
        scores.append(jev.get_score(row.id, row.name_a, row.name_b, cache, refresh=refresh_jev))
        # Checkpoint so a mid-run failure does not lose all API work.
        if refresh_jev and i % JEV_CHECKPOINT_EVERY == 0:
            jev.save_cache(cache)
    if refresh_jev:
        jev.save_cache(cache)
    out["score_jev"] = scores
    return out


def metrics(y_true: pd.Series, pred: pd.Series) -> dict:
    y, p = y_true.astype(str), pred.astype(str)
    auto = p == "match"
    tp = int((auto & (y == "match")).sum())
    auto_n = int(auto.sum())
    n_pos = int((y == "match").sum())
    return {
        "precision": tp / auto_n if auto_n else float("nan"),
        "recall": tp / n_pos if n_pos else float("nan"),
        "dangerous_fp": int((auto & (y == "non_match")).sum()),
        "review_rate": float((p == "review").mean()) if len(y) else float("nan"),
        "auto_n": auto_n,
    }


def tune(train: pd.DataFrame, score_col: str) -> dict:
    # Train only — tuning on holdout would make the bake-off meaningless.
    y, scores = train["label"], train[score_col]
    best = None
    highs = np.round(np.linspace(T_HIGH_GRID_START, T_HIGH_GRID_END, T_HIGH_GRID_STEPS), 3)
    lows = np.round(np.linspace(T_LOW_GRID_START, T_LOW_GRID_END, T_LOW_GRID_STEPS), 3)
    for th in highs:
        for tl in lows:
            if tl >= th:
                continue
            m = metrics(y, scores.map(lambda s, a=tl, b=th: band(float(s), a, b)))
            if m["auto_n"] < MIN_AUTO_MATCHES_TRAIN or not (m["precision"] >= MIN_PRECISION):
                continue
            key = (m["recall"], -m["dangerous_fp"])
            if best is None or key > best[0]:
                best = (key, float(tl), float(th))
    if best is None:
        return {"t_low": FALLBACK_T_LOW, "t_high": FALLBACK_T_HIGH}
    return {"t_low": best[1], "t_high": best[2]}


def apply_bands(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    out = df.copy()
    for sys in ["token_jw", "indic_phonetic", "jev"]:
        t = thresholds[sys]
        out[f"band_{sys}"] = out[f"score_{sys}"].map(
            lambda s, tl=t["t_low"], th=t["t_high"]: band(float(s), tl, th)
        )

    # Locals decide when they agree; Jev only on the hard residue.
    def cascade(r):
        a, b = r["band_token_jw"], r["band_indic_phonetic"]
        if a == b == "match":
            return "match"
        if a == b == "non_match":
            return "non_match"
        return r["band_jev"]

    out["band_cascade"] = out.apply(cascade, axis=1)
    return out


def recommend(holdout: pd.DataFrame) -> dict:
    rows = []
    for sys in ["token_jw", "indic_phonetic", "jev", "cascade"]:
        m = metrics(holdout["label"], holdout[f"band_{sys}"])
        m["system"] = sys
        m["eligible"] = m["auto_n"] > 0 and m["recall"] == m["recall"] and m["recall"] >= MIN_RECALL
        rows.append(m)

    eligible = [r for r in rows if r["eligible"]]
    if not eligible:
        return {
            "winner": "cascade",
            "reason": f"No system hit recall>={MIN_RECALL}; prefer cascade + review, not auto-approve.",
        }

    # FP first: wrong auto-approve dominates ops cost of review.
    eligible.sort(
        key=lambda r: (
            r["dangerous_fp"],
            -(r["precision"] if r["precision"] == r["precision"] else -1),
            -(r["recall"] if r["recall"] == r["recall"] else -1),
        )
    )
    w = eligible[0]["system"]
    return {"winner": w, "reason": f"{w}: min dangerous_fp → max precision → max recall."}


def run(refresh_jev: bool = False) -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = score_all(load_dataset(), refresh_jev=refresh_jev)

    # Ambiguous nicknames are label noise — exclude from tune and ship-call.
    clear = df["ambiguous"].astype(str).str.lower().ne("yes")
    train = df[(df["split"] == "train") & clear]
    thresholds = {
        "token_jw": tune(train, "score_token_jw"),
        "indic_phonetic": tune(train, "score_indic_phonetic"),
        "jev": tune(train, "score_jev"),
        "cascade": {"rule": "locals agree → that band; else jev"},
    }
    df = apply_bands(df, thresholds)
    THRESHOLDS.write_text(json.dumps(thresholds, indent=2) + "\n")

    # Public figures inflate LLM prior; ship-call uses synthetic holdout only.
    primary = df[(df["split"] == "holdout") & clear & (df["source"] == "synthetic")]
    table = pd.DataFrame(
        [
            {"system": sys, **metrics(primary["label"], primary[f"band_{sys}"])}
            for sys in ["token_jw", "indic_phonetic", "jev", "cascade"]
        ]
    )
    rec = recommend(primary)

    cols = [
        "id", "name_a", "name_b", "label", "category", "source", "ambiguous", "split",
        "score_token_jw", "score_indic_phonetic", "score_jev",
        "band_token_jw", "band_indic_phonetic", "band_jev", "band_cascade",
    ]
    df[cols].to_csv(RESULTS / "predictions.csv", index=False)

    def fmt(x):
        return "—" if x != x else f"{x:.3f}" if isinstance(x, float) else str(x)

    lines = [
        "# Name matching evaluation summary",
        "",
        "Clear-label **synthetic** holdout (sklearn 70/30, `random_state=42`). "
        "Ambiguous / public-figure rows excluded from the ship-call.",
        "",
        f"**Winner:** `{rec['winner']}` — {rec['reason']}",
        "",
        "| system | precision | recall | dangerous_fp | review_rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, r in table.iterrows():
        lines.append(
            f"| {r['system']} | {fmt(r['precision'])} | {fmt(r['recall'])} | "
            f"{r['dangerous_fp']} | {fmt(r['review_rate'])} |"
        )
    lines += [
        "",
        "Thresholds (train-tuned, precision ≥ 0.95):",
        "",
        "```json",
        json.dumps(thresholds, indent=2),
        "```",
        "",
        "Jev scores: cached in `data/jev_scores.json` (refresh with `--refresh-jev`).",
        "",
    ]
    (RESULTS / "summary.md").write_text("\n".join(lines))

    print(table[["system", "precision", "recall", "dangerous_fp", "review_rate"]].to_string(index=False))
    print(f"\nRECOMMENDATION: {rec['winner']}\n{rec['reason']}")
    return rec
