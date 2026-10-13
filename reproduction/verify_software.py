"""Verify the extracted anonymous software supplement without network access."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE.parent / "software_verification")
    args = parser.parse_args()
    output = args.output.resolve()
    if output == HERE or HERE in output.parents:
        parser.error("Choose an output directory outside the extracted software folder.")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((HERE / "file_manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest.items():
        path = HERE / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError("Package integrity mismatch: " + relative)
    jobs = [
        ("metrics", "offline_reproduction", ["run_reproduction.py", "--out", str(output / "metrics")]),
        ("scorer_tests", "offline_reproduction", ["-m", "unittest", "discover", "-s", "frozen_bundle", "-p", "test_scorer.py", "-v"]),
        ("gpt_tests", "offline_reproduction", ["-m", "unittest", "-v", "test_reproduction"]),
        ("binding_metrics", "binding_joint", ["recompute.py", "--output", str(output / "binding")]),
        ("binding_tests", "binding_joint", ["-m", "unittest", "-v", "test_recompute"]),
    ]
    results = []
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
    for name, relative, arguments in jobs:
        completed = subprocess.run([sys.executable, "-B", "-X", "utf8", *arguments], cwd=HERE / relative,
                                   env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8")
        (output / (name + ".log")).write_text(completed.stdout, encoding="utf-8")
        results.append({"check": name, "exit_code": completed.returncode})
        print(name + ": " + ("PASS" if completed.returncode == 0 else "FAIL"), flush=True)
    passed = all(result["exit_code"] == 0 for result in results)
    report = {"status": "PASS" if passed else "FAIL", "checks": results, "api_calls": 0}
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
