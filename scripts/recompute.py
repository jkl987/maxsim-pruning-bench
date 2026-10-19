#!/usr/bin/env python3
"""Verify archives and recompute paired-query statistics, including under -O."""
import argparse
import json
from pathlib import Path
from maxsim_pruning.evidence import recompute

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    text = json.dumps(recompute(args.root), indent=2, allow_nan=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end='')
