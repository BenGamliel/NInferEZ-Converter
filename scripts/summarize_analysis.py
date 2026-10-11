"""Reduce a saved NDJSON analysis to a human-reviewable validation record."""
import json
from pathlib import Path
import sys

events = [json.loads(line) for line in Path(sys.argv[1]).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
result = next(e for e in reversed(events) if e.get("type") == "result")
summary = {"source": result["source"], "compatible": result["compatible"],
           "options": [{"profile": o["profile"], "supported": o["supported"],
                        **({"encodedBytes": o["plan"].get("encodedBytes"), "formats": o["plan"].get("formats"),
                            "operations": o["plan"].get("operations"), "warnings": o.get("warnings", [])} if o["supported"] else {"reason": o["reason"]})}
                       for o in result["options"]]}
Path(sys.argv[2]).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"compatible": summary["compatible"], "options": [{"profile": o["profile"], "supported": o["supported"], "GiB": round(o.get("encodedBytes", 0) / 1073741824, 2)} for o in summary["options"]]}))
