"""Tunable constants. Each value notes why it was chosen over nearby alternatives."""

# --- token_jw ---------------------------------------------------------------

# Initial vs expanded given name (S. ↔ Suresh).
# 1.0 would treat every same-letter expansion as a full match (Suresh and Sunil both win).
# ≤0.70 sits under common auto-match thresholds and kills recall on real initial pairs.
# 0.82 is strong enough to help true expansions, weak enough that wrong expansions need
# more evidence (surname alone is not enough for auto-match at our usual t_high).
JW_INITIAL_SCORE = 0.82

# Bonus when both sides have an S/O (or similar) relative line that also aligns.
# 0.0 ignores a useful signal. ≥0.15 lets a matching father name dominate the person tokens.
# 0.05 is a tie-break, not a second identity check.
JW_RELATIVE_BONUS = 0.05

# --- indic_phonetic ---------------------------------------------------------

# Same role as JW_INITIAL_SCORE, slightly lower: phonetic path already grants sound matches,
# so initials should not be the stronger signal here.
PHONETIC_INITIAL_SCORE = 0.80

# Max |len(a)-len(b)| before rejecting a phonetic pair.
# 1–2 is too tight (Mohammed/Mohammad). ≥5 lets Sunil/Sunila-style gender near-twins through.
# 3 allows common transliteration length drift without those collapses.
PHONETIC_MAX_LEN_DIFF = 3

# Max |syllable_count(a)-syllable_count(b)|.
# 0 rejects too many real variants. ≥2 re-opens Sunil/Sunila after Soundex-style coding.
# 1 is the smallest guard that still blocks that class.
PHONETIC_MAX_SYLLABLE_DIFF = 1

# Cap when Double Metaphone keys match.
# 1.0 treats shared sound code as identity (siblings / near-names).
# ≤0.85 under-rewards real transliteration (Lakshmi/Laxmi) before length_ratio.
# 0.92 × length_ratio = high but not automatic approve.
PHONETIC_KEY_MATCH_SCORE = 0.92

# --- normalize / join-split -------------------------------------------------

# Only try splitting a single glued token if it is at least this long.
# Shorter tokens (Ram, Ali) are usually atomic; splitting them creates noise.
JOIN_SPLIT_MIN_LEN = 6

# Character cut points when splitting one token into two (inclusive start, exclusive end).
# Cuts at 1–2 invent initials junk. Cuts past ~8 explode the variant set for long South names.
JOIN_SPLIT_CUT_START = 3
JOIN_SPLIT_CUT_END = 8

# --- eval / thresholds ------------------------------------------------------

# sklearn holdout fraction.
# 0.2 leaves too few holdout rows per category (~40). 0.4 eats train for threshold search.
# 0.3 ≈ Cashfree-style calibration split size and keeps ~60 holdout pairs on n=200.
TEST_SIZE = 0.30
SPLIT_SEED = 42

# Train objective: max recall subject to precision ≥ this.
# 0.99 is often infeasible and forces near-zero auto-match.
# 0.90 is looser than KYC auto-approve comfort.
# 0.95 matches a high-precision auto-approve bar (Cashfree "good partial" band is nearby).
MIN_PRECISION = 0.95

# Holdout eligibility floor on auto-match recall.
# 0.2 lets a coward system win. 0.6 excludes our safest high-precision locals on this set.
# 0.4 means "useful auto-approve volume" without demanding Jev-level recall.
MIN_RECALL = 0.40

# Ignore threshold grids that auto-match fewer than this many train rows.
# 1 allows vacuous precision on a single lucky pair. 5+ is harsh on a ~140-row train set.
MIN_AUTO_MATCHES_TRAIN = 3

# Fallback bands if no grid point hits MIN_PRECISION (Cashfree-inspired seeds, not calibrated).
FALLBACK_T_LOW = 0.34
FALLBACK_T_HIGH = 0.85

# Grid search ranges for (t_low, t_high). Wider than [0.5, 0.9] so JW's near-1.0 scores
# and softer phonetic scores can both find a feasible point.
T_HIGH_GRID_START = 0.55
T_HIGH_GRID_END = 0.99
T_HIGH_GRID_STEPS = 20
T_LOW_GRID_START = 0.20
T_LOW_GRID_END = 0.85
T_LOW_GRID_STEPS = 15

# Progress / checkpoint cadence during --refresh-jev (not scoring logic).
JEV_PROGRESS_EVERY = 10
JEV_CHECKPOINT_EVERY = 20
