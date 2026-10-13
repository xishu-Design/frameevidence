"""Verify the extracted FQ2 data supplement without inference or network access."""
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
    parser.add_argument("--output", type=Path, default=HERE.parent / "data_verification")
    args = parser.parse_args()
    output = args.output.resolve()
    if output == HERE or HERE in output.parents:
        parser.error("Choose an output directory outside the extracted data folder.")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((HERE / "file_manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest.items():
        path = HERE / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError("Package integrity mismatch: " + relative)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
    completed = subprocess.run([sys.executable, "-B", "-X", "utf8", "-m", "unittest", "-v", "test_trace", "test_bundled_images"],
                               cwd=HERE / "fq2_input_trace", env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding="utf-8")
    (output / "data_tests.log").write_text(completed.stdout, encoding="utf-8")
    passed = completed.returncode == 0
    report = {"status": "PASS" if passed else "FAIL", "exit_code": completed.returncode,
              "groups": 48, "published_conditions": 144, "additional_unlabelled_deletions": 94,
              "prepared_images": 125, "api_calls": 0}
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("data_tests: " + report["status"])
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
