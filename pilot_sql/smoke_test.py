import json
from pathlib import Path
from .direct_harness import reference_solver, run_direct, smoke_tasks

def main():
    out = Path(__file__).parent / "data" / "smoke_results.jsonl"
    if out.exists(): raise SystemExit(f"Refusing to overwrite existing result: {out}")
    records = [run_direct(t, reference_solver, seed=20261008) for t in smoke_tasks()]
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    assert all(r["result"]["success"] for r in records)
    print(json.dumps({"harness": "direct", "n_tasks": len(records), "success": True, "output": str(out)}, ensure_ascii=False))

if __name__ == "__main__": main()
