"""
tests/test_flight_software.py

Builds and runs the C flight-software layer (flight_software/) and
re-reports its result in the same "N passed, M failed out of T" format as
every other test file so run_all_tests.py counts it. Also builds the C code
with AddressSanitizer + UBSan (when the compiler supports it) and requires
a clean run.
"""
import os
import re
import subprocess
import sys

FSW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "flight_software")
SRCS = ["fsw_math.c", "fsw_frames.c", "fsw_peg.c", "fsw_guidance.c", "fsw_eskf.c", "fsw_crc.c", "fsw_telemetry.c", "fsw_fdir.c", "fsw_state.c", "fsw_watchdog.c", "fsw_core.c"]
PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")


def build(out, extra):
    for cc in ("gcc", "cc", "clang"):
        try:
            cmd = [cc, "-std=c99", "-O1", "-g", "-Wall", "-Wextra", "-I.", "-o", out] + extra + \
                  SRCS + ["test/fsw_selftest.c", "../embedded/pid_controller.c", "-lm"]
            r = subprocess.run(cmd, cwd=FSW, capture_output=True, text=True)
            if r.returncode == 0:
                return True, ""
            last = r.stderr
        except FileNotFoundError:
            last = "compiler not found"
            continue
    return False, last


def run(binary):
    r = subprocess.run([os.path.join(FSW, binary)], cwd=FSW, capture_output=True, text=True, timeout=120)
    m = re.search(r"(\d+) passed, (\d+) failed out of (\d+)", r.stdout)
    return r.returncode, (tuple(int(x) for x in m.groups()) if m else None), r.stdout


def main():
    os.makedirs(os.path.join(FSW, "build"), exist_ok=True)
    ok, err = build("build/selftest_plain", [])
    if not ok and "not found" in err:
        print("  [SKIP] no host C compiler; flight_software tests skipped")
        print("0 passed, 0 failed out of 0")
        return 0
    check("flight_software compiles with -Wall -Wextra", ok)
    if ok:
        rc, counts, out = run("build/selftest_plain")
        check("C self-test exits 0", rc == 0)
        check("C self-test reports a result line", counts is not None)
        if counts:
            check(f"all {counts[2]} C-side checks pass (0 failed)", counts[1] == 0 and counts[2] >= 205)
        if rc != 0:
            print("\n".join(l for l in out.splitlines() if "FAIL" in l))

    ok, err = build("build/selftest_san", ["-fsanitize=address,undefined", "-fno-sanitize-recover=all"])
    if ok:
        rc, counts, out = run("build/selftest_san")
        check("clean under AddressSanitizer + UBSan", rc == 0 and counts is not None and counts[1] == 0)
    else:
        print("  [SKIP] sanitizers unavailable on this toolchain")

    print(f"{PASS} passed, {FAIL} failed out of {PASS + FAIL}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
