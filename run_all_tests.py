#!/usr/bin/env python3
"""
run_all_tests.py

Runs every tests/test_*.py file as a subprocess (each is self-contained
and also runnable directly with `python3 tests/test_X.py`) and prints a
combined summary. No pytest dependency required.
"""
import subprocess
import sys
import glob
import os

TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests")


def main():
    test_files = sorted(glob.glob(os.path.join(TEST_DIR, "test_*.py")))
    total_pass = total_fail = 0
    failed_files = []

    for path in test_files:
        name = os.path.basename(path)
        result = subprocess.run([sys.executable, path], capture_output=True, text=True)
        last_line = next((l for l in reversed(result.stdout.strip().splitlines()) if "passed" in l), "")
        ok = result.returncode == 0
        status = "OK  " if ok else "FAIL"
        print(f"[{status}] {name:<40} {last_line}")
        if not ok:
            failed_files.append(name)
            print(result.stdout[-1500:])
            print(result.stderr[-1500:])
        try:
            import re
            m = re.search(r"(\d+)\s*passed,\s*(\d+)\s*failed", last_line)
            if m:
                total_pass += int(m.group(1))
                total_fail += int(m.group(2))
        except (IndexError, ValueError):
            pass

    print(f"\n{'='*60}\nTOTAL: {total_pass} passed, {total_fail} failed, "
         f"across {len(test_files)} files")
    if failed_files:
        print("Failed files:", ", ".join(failed_files))
        sys.exit(1)


if __name__ == "__main__":
    main()
