#!/usr/bin/env python3
"""Load DeltaCorrector model and print its correction statistics."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifiers.delta_corrector import DeltaCorrector


def main():
    parser = argparse.ArgumentParser(description="Load DeltaCorrector and print stats")
    parser.add_argument(
        "model_path",
        nargs="?",
        default="correctors/delta_corrector_bins_9_12.joblib",
        help="Path to saved DeltaCorrector model",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    args = parser.parse_args()

    path = Path(args.model_path)
    if not path.exists():
        print(f"Model not found: {path}", file=sys.stderr)
        sys.exit(1)

    corrector = DeltaCorrector.load_model(str(path))
    stats = corrector.get_correction_stats()

    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print("DeltaCorrector stats")
        print("=" * 50)
        print(f"Method: {corrector.method}")
        print(f"Fitted: {corrector.is_fitted}")
        print()
        for var, s in stats.items():
            print(f"  {var}:")
            print(f"    bias: {s['bias']:.4f}")
            print(f"    method: {s['method']}")
            print(f"    correction_type: {s['correction_type']}")
            print()


if __name__ == "__main__":
    main()
