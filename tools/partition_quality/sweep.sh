#!/usr/bin/env bash
# Partition-quality sweep driver. Runs edgeCutCompare across a grid of
# (npes x max_depth x variant) and appends JSONL into one results dir, which
# analyze_partition_quality.py then reads.
#
# Works locally and under SLURM (see sweep.sbatch). No timing here -- edge cut
# is deterministic, so this needs no rep/interleave discipline. (That discipline
# is mandatory for WALL-CLOCK work; see Conventions and Constraints.)
#
#   ./sweep.sh                                   # defaults
#   NPES="2 4 8 16" DEPTHS="6 7 8" ./sweep.sh    # override the grid
#   OUT=/scratch/pq ./sweep.sh
#
# Env knobs:
#   BIN      path to edgeCutCompare           (default: build/edgeCutCompare)
#   OUT      results dir                      (default: ./pq_results)
#   NPES     rank counts to sweep             (default: "2 4 8")
#   DEPTHS   max octree depths                (default: "6 7")
#   WTOL     wavelet tolerance                (default: 1e-5)
#   GRAIN    dendro grain size                (default: 50)
#   PTOL     partition tolerance              (default: 0.1)
#   REFINE   sine|blob                        (default: blob)
#   VARIANTS which partitioners to score      (default: "sfc fastpart")
#   MPIRUN   launcher                         (default: mpirun)
#
# NOTE on REFINE: `sine` refines the WHOLE domain (11 periods across the box) and
# explodes past depth 6 -- it is the partitioningMeshTests function, kept for
# comparability, not for sweeps. `blob` is a centred gaussian: localised
# refinement, like EM4/BSSN, and tractable. But the blob is SYMMETRIC across the
# 8 top-level octants, so any equal-count SFC split gets equal work for free =>
# its work_imbalance column is an artefact (~1.000) and must NOT be read as
# "SFC balances work perfectly". Edge cut is unaffected.
# ---> For a WORK-BALANCE study use REFINE=offblob: an off-centre gaussian in the
#      +++ octant. The heavy refinement then clusters on one stretch of the
#      Hilbert curve and lands on a few ranks, so SFC's element-count balance
#      diverges from work balance and work_imbalance becomes meaningful. This is
#      the config that tests graph partitioning's one claimed advantage.
set -uo pipefail

BIN=${BIN:-build/edgeCutCompare}
OUT=${OUT:-./pq_results}
NPES=${NPES:-"2 4 8"}
DEPTHS=${DEPTHS:-"6 7"}
WTOL=${WTOL:-1e-5}
GRAIN=${GRAIN:-50}
PTOL=${PTOL:-0.1}
VARIANTS=${VARIANTS:-"sfc fastpart"}
REFINE=${REFINE:-blob}
MPIRUN=${MPIRUN:-mpirun}

if [ ! -x "$BIN" ]; then
    echo "ERROR: edgeCutCompare not found/executable at '$BIN'" >&2
    echo "  build it:  cmake --build build -j --target edgeCutCompare" >&2
    echo "  or set BIN=/path/to/edgeCutCompare" >&2
    exit 1
fi
BIN=$(readlink -f "$BIN")

mkdir -p "$OUT"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$OUT/sweep_$STAMP.log"
: > "$LOG"

echo "edgeCutCompare sweep"
echo "  bin      : $BIN"
echo "  out      : $OUT"
echo "  npes     : $NPES"
echo "  depths   : $DEPTHS"
echo "  variants : $VARIANTS"
echo "  log      : $LOG"
echo

# one run per (np, depth) scores every variant off the SAME mesh -> the arms are
# guaranteed to share identical adjacency, and we don't rebuild the octree N times.
VLIST=$(echo "$VARIANTS" | tr ' ' ',')

nrun=0; nfail=0
for np in $NPES; do
    for d in $DEPTHS; do
        out_jsonl="$OUT/pq_np${np}_d${d}_${REFINE}.jsonl"
        tag="np=$np depth=$d refine=$REFINE"
        printf '  %-34s ' "$tag"
        if $MPIRUN -np "$np" "$BIN" \
                --max-depth "$d" --wavelet-tol "$WTOL" \
                --grain "$GRAIN" --partition-tol "$PTOL" --refine "$REFINE" \
                --variants "$VLIST" --json "$out_jsonl" >> "$LOG" 2>&1; then
            echo "ok  -> $(basename "$out_jsonl")"
        else
            echo "FAIL (see $LOG)"
            nfail=$((nfail+1))
        fi
        nrun=$((nrun+1))
    done
done

echo
echo "$nrun runs, $nfail failed."
echo "analyze with:"
echo "  python3 tools/partition_quality/analyze_partition_quality.py $OUT --verdict"
[ "$nfail" -gt 0 ] && exit 1
exit 0
