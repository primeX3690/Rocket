#!/bin/sh
# One-command demo: builds the flight software with strict warnings, runs the whole host verification
# (227 checks, 28 closed-loop scenarios) and prints the headline numbers. Needs only gcc and make (~1-2 min).
cd "$(dirname "$0")" || exit 1
make test 2>&1 | awk '
  /^Scenario (1:|15|17|18|20|21)/ {print ""; print $0; next}
  /^     / && (in_hdr==0) {print $0}
  /passed, [0-9]+ failed/ {print ""; print "RESULT: " $0}
' | grep -vE "^\s+\[(PASS|FAIL)\]" | head -80
