#!/usr/bin/env bash
#SBATCH --job-name=partquality
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=00:30:00
#SBATCH --output=pq_%j.out
#SBATCH --error=pq_%j.err
#
# SLURM wrapper for the partition-quality sweep.
#
#   sbatch tools/partition_quality/sweep.sbatch
#   sbatch --ntasks=64 --nodes=4 tools/partition_quality/sweep.sbatch
#
# Override the grid without editing this file:
#   sbatch --export=ALL,NPES="8 16 32 64",DEPTHS="7 8 9" tools/partition_quality/sweep.sbatch
#
# NOTE this measures edge cut only -- static, deterministic, no timing. Safe to
# run alongside other jobs. Do NOT add wall-clock measurement here: that needs
# exclusive nodes + interleaved contemporaneous pairs (Conventions and Constraints).

set -uo pipefail

# --- site-specific: adjust module loads for your cluster -----------------
# module load gcc/12 openmpi/4.1 cmake
# -------------------------------------------------------------------------

REPO=${REPO:-$SLURM_SUBMIT_DIR}
cd "$REPO" || exit 1

export BIN=${BIN:-$REPO/build/edgeCutCompare}
export OUT=${OUT:-$REPO/pq_results_${SLURM_JOB_ID:-local}}
# default the sweep to the ranks SLURM actually gave us
export NPES=${NPES:-"${SLURM_NTASKS:-4}"}
export DEPTHS=${DEPTHS:-"6 7 8"}
export VARIANTS=${VARIANTS:-"sfc fastpart"}
# srun is the right launcher under SLURM; falls back to mpirun if absent
export MPIRUN=${MPIRUN:-srun --mpi=pmix}

echo "job ${SLURM_JOB_ID:-?} on ${SLURM_JOB_NODELIST:-localhost}"
echo "ntasks=${SLURM_NTASKS:-?}  npes sweep=$NPES  depths=$DEPTHS"

bash "$REPO/tools/partition_quality/sweep.sh"
rc=$?

echo
echo "=== summary ==="
python3 "$REPO/tools/partition_quality/analyze_partition_quality.py" "$OUT" --verdict \
    --csv "$OUT/summary.csv"

exit $rc
