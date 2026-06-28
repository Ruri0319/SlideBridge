from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ibl2svs.acceptance import summarize_wsi_output


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect TIFF/SVS output structure.")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    summary = summarize_wsi_output(args.path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
