# Name matching evaluation summary

Clear-label **synthetic** holdout (sklearn 70/30, `random_state=42`). Ambiguous / public-figure rows excluded from the ship-call.

**Winner:** `token_jw` — token_jw: min dangerous_fp → max precision → max recall.

| system | precision | recall | dangerous_fp | review_rate |
|---|---:|---:|---:|---:|
| token_jw | 1.000 | 0.441 | 0 | 0.712 |
| indic_phonetic | 0.938 | 0.441 | 1 | 0.635 |
| jev | 0.913 | 0.618 | 2 | 0.288 |
| cascade | 0.923 | 0.706 | 2 | 0.231 |

Thresholds (train-tuned, precision ≥ 0.95):

```json
{
  "token_jw": {
    "t_low": 0.2,
    "t_high": 0.99
  },
  "indic_phonetic": {
    "t_low": 0.2,
    "t_high": 0.921
  },
  "jev": {
    "t_low": 0.2,
    "t_high": 0.851
  },
  "cascade": {
    "rule": "locals agree \u2192 that band; else jev"
  }
}
```

Jev scores: cached in `data/jev_scores.json` (refresh with `--refresh-jev`).
