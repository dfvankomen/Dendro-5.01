#!/usr/bin/env bash
# summarise an run_ab.sh output dir: wall per arm per round, proof hits, LTS stats where present
OUT=${1:?outdir}
for d in "$OUT"/r*_*; do
  n=$(basename "$d"); wall=$(grep -oE 'wall=[0-9.]+' "$d/meta.txt" | cut -d= -f2)
  proof=$(grep -oE 'proof_hits=[0-9]+' "$d/meta.txt" | cut -d= -f2)
  rc=$(grep -oE 'rc=[0-9]+' "$d/meta.txt" | head -1)
  eff=$(grep -oE 'lts_substep_eff=[0-9.]+' "$d/run.log" | tail -1)
  ldb=$(grep -E 'LD Bal' "$d/run.log" | tail -1 | tr -s '\t ' ' ' | sed 's/.*LD Bal: //')
  spd=$(grep -oE 'ets. speedup : [0-9.]+' "$d/run.log" | tail -1)
  rem=$(grep -oE 'graph-remesh\] .*total=[0-9.]+ms' "$d/run.log" | tail -1)
  echo "$n $rc wall=$wall proof=${proof:-n/a} ${eff:-} ${ldb:+LDBal=$ldb} ${spd:-} ${rem:-}"
done
