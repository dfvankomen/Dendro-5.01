#!/usr/bin/env bash
# interleaved A/B runner.  run_ab.sh <outdir> <rounds> <np> <exe> <extra-args...>
# ARMS="name:parfile[:ENV1=a,ENV2=b] ..."   PROOF="regex" (must match >0 lines in the arm's run.log)
set -uo pipefail
OUT=$1; ROUNDS=$2; NP=$3; EXE=$4; shift 4; EXTRA="$*"
LAUNCH=${LAUNCH:-mpirun -np $NP}
ulimit -c 0
mkdir -p "$OUT"
for r in $(seq 1 "$ROUNDS"); do
  for arm in $ARMS; do
    IFS=: read -r name par envs <<< "$arm"
    d="$OUT/r${r}_${name}"; mkdir -p "$d/vtu"; cp "$par" "$d/"
    ( cd "$d"
      echo "start $(date -Is) loadavg $(cut -d' ' -f1-3 /proc/loadavg)" > meta.txt
      xenv=""; for kv in ${envs//,/ }; do [ -n "$kv" ] && { export "$kv"; xenv="$xenv -x ${kv%%=*}"; }; done
      t0=$(date +%s.%N)
      $LAUNCH $xenv "$EXE" "$(basename "$par")" $EXTRA </dev/null > run.log 2>&1; rc=$?
      t1=$(date +%s.%N)
      echo "rc=$rc wall=$(awk "BEGIN{printf \"%.1f\", $t1 - $t0}") end $(date -Is) loadavg $(cut -d' ' -f1-3 /proc/loadavg)" >> meta.txt
      [ -n "${PROOF:-}" ] && echo "proof_hits=$(grep -cE "$PROOF" run.log)" >> meta.txt
    )
    echo "round $r arm $name: $(tail -n 2 "$d/meta.txt" | tr '\n' ' ')"
  done
done
