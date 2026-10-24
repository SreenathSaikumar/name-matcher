#!/usr/bin/env python3
"""CLI entry. Offline by default; --refresh-jev needs TYPESAFE_API_KEY."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from namematch.eval import run  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="KYC name-matching evaluation")
    p.add_argument(
        "--refresh-jev",
        action="store_true",
        help="Call TypeSafe Jev and rewrite data/jev_scores.json",
    )
    args = p.parse_args()
    run(refresh_jev=args.refresh_jev)


if __name__ == "__main__":
    main()
