# Dataset construction

Hand-authored Indian KYC-style name pairs for **document-to-document same-person** checks (not watchlist screening).

## Columns

`id, name_a, name_b, label (match|non_match), category, source (synthetic|public), note, ambiguous (yes|no)`

## Why these categories

- **initials / wrong_expansion** — ID cards often store `S. Kumar`; wrong expansion is a classic false positive.
- **reorder** — surname-first formats on some state IDs.
- **transliteration** — Mohammed/Mohammad, Lakshmi/Laxmi, v/w, long vowels.
- **middle_name / so_line** — dropped middle names; S/O D/O W/O lines.
- **honorifics** — Smt/Shri/Kumari/Mr.
- **compound** — Ram Kumar vs Ramkumar.
- **south_indian_initials / public_variant** — multi-initial expansions; capped public figures.
- **hard_negative / easy_negative** — siblings, gender near-phonetic, shared surnames.
- **ambiguous_nickname** — tagged `ambiguous=yes`; excluded from threshold fitting and primary metrics.

## Sources

Mostly **synthetic** realistic patterns. **Public** rows are well-known published names/variants (≤~15–20%). Ship recommendation uses **clear-label synthetic-only holdout** so LLM prior on celebrities does not decide the bake-off.

## Labels

Binary `match` / `non_match` by construction intent. Ambiguous cultural nickname cases are tagged, not forced into primary metrics.

## Split

No static split file. At eval time, `sklearn.model_selection.train_test_split` does a 70/30 split stratified by `category|label` with `random_state=42`. Do not edit pairs after scoring.
