from __future__ import annotations

import sys

from .app_meta import APP_NAME, runtime_banner


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--version" in args:
        print(runtime_banner())
        return 0
    if "--worker" in args:
        from .backend_worker import main as worker_main

        return worker_main()
    print(f"{APP_NAME} no longer includes a Tkinter GUI.", file=sys.stderr)
    print("Use the React/Tauri desktop app, or run with --worker for JSONL sidecar mode.", file=sys.stderr)
    return 2
