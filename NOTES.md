# NOTES — KYC name matching

## Run

```bash
uv sync
cp .env.example .env   # TYPESAFE_API_KEY only needed for --refresh-jev

uv run python scripts/run_eval.py
uv run python scripts/run_eval.py --refresh-jev
```

Outputs: `results/predictions.csv`, `results/summary.md`, `data/thresholds.json`.

Rebuild dataset: `uv run python scripts/build_dataset.py`  
Split: sklearn `train_test_split`, 70/30, stratified by `category|label`, `random_state=42`.

## Why these metrics

A **false positive** here means auto-approving the wrong person. That is much worse than a **false negative** (friction or manual review).

So each system outputs one of three bands:

- **match** — auto-approve  
- **review** — send to a human  
- **non_match** — reject  

We rank systems by:

1. Lowest `dangerous_fp` (pred = match, label = non_match)  
2. Then highest **precision**  
3. Then highest **recall**  

A system needs recall ≥ `MIN_RECALL` so a “never auto-approve” model cannot win.

Thresholds are tuned on **train** only (target precision ≥ `MIN_PRECISION`). The ship call uses **clear-label synthetic holdout** only — public-figure names can inflate LLM scores.

## Constants (why these values)

All tunables live in [`namematch/constants.py`](namematch/constants.py). Short “why not X instead” notes:

| Constant | Value | Chosen because… | Rejected alternatives |
|---|---:|---|---|
| `JW_INITIAL_SCORE` | 0.82 | Helps true `S.`→`Suresh` without auto-approving every same letter | `1.0` = Suresh and Sunil both full matches; `≤0.70` kills initial recall under our `t_high` |
| `JW_RELATIVE_BONUS` | 0.05 | Small tie-break when both docs have aligning S/O lines | `0` ignores signal; `≥0.15` lets father name dominate the person |
| `PHONETIC_INITIAL_SCORE` | 0.80 | Slightly below JW — phonetic path already grants sound matches | Same as JW (0.82) would double-count initials as strong evidence |
| `PHONETIC_MAX_LEN_DIFF` | 3 | Allows Mohammed/Mohammad drift | `1–2` too tight; `≥5` lets Sunil/Sunila through |
| `PHONETIC_MAX_SYLLABLE_DIFF` | 1 | Smallest guard that still blocks Sunil/Sunila | `0` rejects real variants; `≥2` re-opens gender near-twins |
| `PHONETIC_KEY_MATCH_SCORE` | 0.92 | Shared DM key is strong, not identity | `1.0` = siblings auto-match; `≤0.85` under-rewards Lakshmi/Laxmi |
| `JOIN_SPLIT_MIN_LEN` | 6 | Only split glued tokens that might be compounds | Lower invents junk splits on short names (Ram, Ali) |
| `JOIN_SPLIT_CUT_*` | 3–8 | Typical Indian given/surname cut lengths | Cuts at 1–2 look like initials; past ~8 explodes variants |
| `TEST_SIZE` | 0.30 | ~60 holdout rows on n=200; usable per-category n | `0.2` too small; `0.4` starves train for threshold search |
| `MIN_PRECISION` | 0.95 | High bar for unsupervised auto-approve | `0.99` often infeasible; `0.90` looser than KYC comfort |
| `MIN_RECALL` | 0.40 | “Useful auto-approve volume” without demanding Jev recall | `0.2` lets cowards win; `0.6` excludes our safest locals here |
| `MIN_AUTO_MATCHES_TRAIN` | 3 | Avoids vacuous precision on one lucky pair | `1` is vacuous; `≥5` harsh on ~140-row train |
| `FALLBACK_T_*` | 0.34 / 0.85 | Cashfree-inspired seeds if no grid point hits 0.95 | Not calibrated — only used when search finds nothing |

## Algorithms

Shared prep in `normalize.py` (locals only):

- Strip honorifics (`Smt`, `Shri`, …) — not identity signal.  
- Keep `S/O` / `D/O` / `W/O` as a separate relative field. Deleting them makes siblings look like one person.  
- Try join/split variants (`Ram Kumar` ↔ `Ramkumar`).  
- Align tokens order-independently (greedy pairing × coverage) so surname-first formats still match.

Each score ∈ [0, 1] maps to bands with train-tuned `(t_low, t_high)`.

### 1. `token_jw`

Split into tokens; score pairs with **Jaro–Winkler** (`jellyfish`).

Most KYC noise is structural (order, missing middle, initials), not random typos.

- Exact token → 1.0  
- Initial vs expanded name → `JW_INITIAL_SCORE` only (never 1.0 — otherwise `S.` matches both Suresh and Sunil)  
- Else JW similarity  

**Good:** reorder, honorifics, dropped middles, careful initials.  
**Weak:** transliteration inside a token (`Lakshmi` / `Laxmi`); siblings with similar givens + same surname.

### 2. `indic_phonetic`

Light Indic romanize (`v`/`w`, long vowels, `ksh`, …), then **Double Metaphone** (primary + alternate keys). Tokens match if they share a key.

This covers transliteration that JW misses.

Guards: `PHONETIC_MAX_LEN_DIFF` / `PHONETIC_MAX_SYLLABLE_DIFF` block collapses like `Sunil` / `Sunila`. Shared key scores at most `PHONETIC_KEY_MATCH_SCORE` × length ratio so siblings do not get a free 1.0.

**Good:** Mohammed/Mohammad, Lakshmi/Laxmi.  
**Weak:** English-centric phonetics; near-name collisions; no rule for “wrong initial expansion.”

### 3. `jev`

TypeSafe **Noul** (yes/no) on **raw** document strings — we want the model’s judgment, not our normalize pipeline. Frozen criteria allow initials / transliteration / reorder; reject siblings, wrong expansions, surname-only overlap.

Scores are P(same person) from `jev-1.14.0`, cached in `data/jev_scores.json` (criteria hash in the key). Offline runs use the cache; `--refresh-jev` rewrites it.

**Good:** cultural / document rules hard to encode as string metrics.  
**Weak:** cost, latency, model drift; still produced dangerous FPs at the train-chosen threshold.

### 4. `cascade`

Not a fourth scorer — a fixed policy on bands:

1. Both locals `match` → `match`  
2. Both locals `non_match` → `non_match`  
3. Else → **`jev`**

Use cheap locals when they agree; pay for Jev only on hard cases. On this holdout, escalations still inherited Jev FPs — so the metric rule preferred `token_jw` alone for unsupervised auto-approve.

## Recommendation (after `--refresh-jev`)

| system | precision | recall | dangerous_fp | review_rate |
|---|---:|---:|---:|---:|
| **token_jw** | **1.000** | **0.441** | **0** | 0.712 |
| indic_phonetic | 0.938 | 0.441 | 1 | 0.635 |
| jev | 0.913 | 0.618 | 2 | 0.288 |
| cascade | 0.923 | 0.706 | 2 | 0.231 |

**Ship `token_jw` for auto-approve** under this metric: only zero-FP eligible system. High review rate is the cost of that safety.

**Production:** Cascade/Jev improve recall (0.62–0.71) but add 2 dangerous FPs. Prefer `token_jw` (or both-locals-agree) for auto-match; send disagreements to **human review**, not unsupervised Jev, unless Jev is retuned for zero FP.

See `results/summary.md`.

## Proudest edge case

Wrong initial expansion: `S. Kumar` vs `Sunil Kumar` when the person is Suresh. Full credit for initials auto-approves the wrong given name. Partial initial credit is what blocks that.

## If another day

**Hybrid evidence ladder.** Per token pair, try in order: exact → same Double Metaphone key → constrained initial → JW above a floor → else no evidence. Treat transliteration as stronger than fuzzy edit distance without blending unlike scores. Should cut sibling FPs.

**Rarity weighting.** Matching `Kumar` / `Singh` should count less than matching a rare given name. Today every token is equal, so common surnames inflate scores for different people. Use IDF (or a surname frequency table) so those cases fall into review.

**Larger Indic benchmark.** 200 hand-made pairs (~50/50) is fine for a take-home, weak for production cutoffs. Next: more pairs, natural class imbalance, bootstrap CIs on dangerous FP so “0 FP on n≈50” is not over-read.

**Safer Jev threshold.** Today Jev uses the same objective as locals (max recall at precision ≥ 0.95 on train). For a second pass, freeze a high `t_high` with **zero train dangerous FP**, accept more review, then measure holdout recall. That makes cascade safer as an auto-approve path.

**Extras:** structured S/O as a first-class feature; native-script Indic phonetics beyond romanized DM; DOB/gender as secondary signals when available.
