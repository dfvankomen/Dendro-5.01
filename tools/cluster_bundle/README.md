# Cluster bundle — repartitioning A/B (2026-09-24)

Three paired experiments, one allocation each, no timing on the dev box. Nothing here submits
anything by itself; `sbatch` is the user's call.

| exp | question | binary | arms |
|---|---|---|---|
| A | does the repartition cost gap close with rank count? | `em4Solver` | SFC (method 0) vs graph+refine (method 3, `DENDRO_GRAPH_REFINE=1`) at 32/64/128 ranks |
| B | does the LTS win exist in wall-clock? | `nlsmNUTS` (ts_mode 0) | SFC vs levelwise+refine (method 3, `DENDRO_GRAPH_REFINE=2`) at 32/64 |
| C | BSSN long-haul gate for the flipped defaults | `bssnSolver` | old defaults (`DENDRO_MINIMAL_SCATTER=0 DENDRO_ZIPPLAN_DIST=0 DENDRO_LAZY_ZIPPLAN=0`) vs new |

Pinned sources (see `build.sh`): fork = the jj tip after `docs(repart): speed push` (2026-09-24 night, push it
first), fastpart branch `dendro-refine` @ `0e848b0` **plus the uncommitted balance-phase diff** (commit it; the
`fastpart_oct_partopts` struct grew, so fork and library must be built together), EM4 `repartitioning` + uncommitted sandwich fixes, NLSM worktree branch
`lts-graph-fix`. The EM4/NLSM/BSSN edits must be committed and pushed before the cluster can build them.

Rules baked into `run_ab.sh` (from the vault's measurement hygiene):
- arms interleaved `A B A B …`, same nodes, one job; `/proc/loadavg` and `date` per run
- every run proves its knobs fired (grep counts in the log) — a run without proof is discarded
- `</dev/null` on every launcher, `ulimit -c 0`, `mkdir -p vtu`, `IO_OUTPUT_FREQ=0`
- BYU: modules export `OMP_NUM_THREADS=1` (fine for pure MPI); check `taskset -cp $$` inside the job
- exp B reads `dump_est_speedup` + `LD Bal` + `[part-stats] … lts_substep_eff`; exp A reads
  `em4*_prof` phase timers + `[graph-remesh] … total=`; exp C diffs the constraint CSVs

`check.sh <outdir>` summarises. Compare arms by *ratio within a round*, never across rounds.
