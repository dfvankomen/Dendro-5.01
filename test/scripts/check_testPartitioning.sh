#!/bin/bash
# check_testPartitioning.sh <red baseline> <onehop 0|1> <command...>: runs testPartitioning (the command) and gates
# its report. The program reports by color and exits 0, so pass =
#   - the command exits 0 and prints FINISHED;
#   - every real gate (2b, 8be, 10, 11a-c, COVERAGE; 10o with onehop 1) prints its PASSED line, and none of them
#     prints a red or FAILED line, whatever the baseline holds;
#   - the red lines (known-benign diagnostics) equal the baseline exactly, counts included.
set -u
base=$1; onehop=$2; shift 2
esc=$'\x1b'
out=$("$@" 2>&1); rc=$?
printf '%s\n' "$out"
txt=$(printf '%s\n' "$out" | sed "s/${esc}\[[0-9;]*m//g")
red=$(printf '%s\n' "$out" | grep -aF "${esc}[1;31m" | sed "s/${esc}\[[0-9;]*m//g; s/[[:space:]]*\$//" | sort)
bad=0
fail() { echo "CHECK FAILED: $*"; bad=1; }

[ $rc -eq 0 ] || fail "exit code $rc"
printf '%s\n' "$txt" | grep -aq 'FINISHED' || fail "no FINISHED line"

gate='^[[:space:]]*(TEST (2b|8be|10|11[abc])|COVERAGE)'
need=('TEST 2b PASSED' 'COVERAGE TEST PASSED' 'TEST 8be .*PASSED'
      'TEST 10 \(AMR-cycle bit-identity, [0-9]+ cycles\) PASSED'
      'TEST 11a .*PASSED' 'TEST 11b .*PASSED' 'TEST 11c .*PASSED')
[ "$onehop" = 1 ] && need+=('TEST 10o cycle 0: onehop ok=1 .* mismatches=0')
for n in "${need[@]}"; do
    printf '%s\n' "$txt" | grep -aEq "^[[:space:]]*$n" || fail "missing: $n"
done
g=$(printf '%s\n' "$red" | grep -aE "$gate")
[ -z "$g" ] || fail "red line from a real gate: $g"
g=$(printf '%s\n' "$txt" | grep -aE "$gate" | grep -aE 'FAILED|^[[:space:]]*TEST 8be: |onehop ok=0')
[ -z "$g" ] || fail "failure line from a real gate: $g"

d=$(diff <(grep -av '^#' "$base" | sort) <(printf '%s\n' "$red" | grep -av '^$'))
[ -z "$d" ] || fail "red lines differ from the baseline $base (< baseline, > this run):"$'\n'"$d"

[ $bad -eq 0 ] && echo "CHECK PASSED: real gates passed; $(printf '%s\n' "$red" | grep -ac .) red lines match the baseline"
exit $bad
