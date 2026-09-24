#!/usr/bin/env bash
#SBATCH --job-name=repart_ab
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=repart_ab_%j.out
# usage: sbatch --export=ALL,EXP=A,NP=64 byu.sbatch    (EXP = A|B|C; see README)
set -uo pipefail
# module load gcc openmpi cmake        # CHPC (notchpeak/kingspeak): fill in
ulimit -c 0
ROOT=${ROOT:-$SLURM_SUBMIT_DIR/repart_bundle}; B=$SLURM_SUBMIT_DIR
export LAUNCH="srun --mpi=pmi2 -n ${NP:-$SLURM_NTASKS}"
echo "job $SLURM_JOB_ID nodes $SLURM_JOB_NODELIST np=${NP:-$SLURM_NTASKS} taskset=$(taskset -cp $$)"
case "${EXP:-A}" in
  A) ARMS="sfc:$B/par/em4_sfc.toml:DENDRO_LAZY_ZIPPLAN=1 graph:$B/par/em4_graph.toml:DENDRO_GRAPH_REFINE=1,DENDRO_LAZY_ZIPPLAN=1" PROOF='minimal-scatter\]|graph-part-ex' \
     $B/run_ab.sh $B/out_A_${NP} 3 "${NP:-64}" $ROOT/em4/build/solver/em4Solver ;;
  B) ARMS="sfc:$B/par/nlsm_sfc.par.json lw:$B/par/nlsm_lts.par.json:DENDRO_GRAPH_REFINE=2" PROOF='gain refinement|part-stats' \
     $B/run_ab.sh $B/out_B_${NP} 3 "${NP:-64}" $ROOT/nlsm/build/solver/nlsmNUTS 0 ;;
  C) ARMS="old:$B/par/bssn.par.toml:DENDRO_MINIMAL_SCATTER=0,DENDRO_ZIPPLAN_DIST=0,DENDRO_LAZY_ZIPPLAN=0 new:$B/par/bssn.par.toml" PROOF='minimal-scatter\]' \
     $B/run_ab.sh $B/out_C_${NP} 1 "${NP:-64}" $ROOT/bssn/build/BSSN_GR/bssnSolver ;;
esac
$B/check.sh $B/out_${EXP:-A}_${NP:-64}
