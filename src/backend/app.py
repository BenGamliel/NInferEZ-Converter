from __future__ import annotations
import argparse
import contextlib
import json
from pathlib import Path
import signal
import sys
import threading
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "app" / "vendor"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True


def main():
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="NInferEZ Converter standalone backend. JSON request in stdin or --request FILE; NDJSON events in stdout.")
    parser.add_argument("command", choices=["capabilities", "probe", "analyze", "inspect", "convert", "validate", "raw"])
    parser.add_argument("--request", type=Path)
    args, extra = parser.parse_known_args()
    if args.command == "raw":
        from tools.convert.__main__ import main as raw_main
        raw_main(extra)
        return 0
    if extra:
        parser.error("Unknown arguments: " + " ".join(extra))
    output = sys.stdout
    lock = threading.Lock()
    def emit(event):
        with lock:
            output.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            output.flush()
    cancellation = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: cancellation.set())
    def watch():
        for line in sys.stdin:
            if line.strip().lower() in ("cancel", '{"command":"cancel"}'):
                cancellation.set()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            import core
            ctx = core.Context(emit, cancellation)
            if args.command == "capabilities":
                emit({"type": "capabilities", **core.capabilities()})
                return 0
            request = json.loads(args.request.read_text(encoding="utf-8-sig")) if args.request else json.loads(sys.stdin.readline())
            if not isinstance(request, dict):
                raise ValueError("Request must be a JSON object")
            threading.Thread(target=watch, daemon=True).start()
            result = getattr(core, args.command)(request, ctx)
            if args.command in ("probe", "analyze", "inspect", "validate"):
                emit({"type": "result", "success": True, **result})
        return 0
    except Exception as error:
        cancelled = error.__class__.__name__ == "Cancelled"
        emit({"type": "cancelled" if cancelled else "error", "success": False, "message": str(error), "errorClass": type(error).__name__})
        if not cancelled:
            traceback.print_exc(file=sys.stderr)
        return 2 if cancelled else 1


if __name__ == "__main__":
    raise SystemExit(main())
