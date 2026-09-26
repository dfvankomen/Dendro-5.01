# Plan — price the repartitioning work in wall-clock on Kingspeak (2026-09-25)

User's OK given 2026-09-25 for Kingspeak (CHPC). Home is 100 % full; everything lives on
`/scratch/general/vast/u0720070/repart/`. The question the user asked: **does the drop in
communication help?** The vault's standing verdict is that no wall-clock win over SFC has ever been
measured; the dev box cannot time. Kingspeak's `kingspeak` partition (account `sundar`, 22 nodes, mixed
20/24/32-core, QOS wall 3 days) is the place: pairs on `c20` nodes only, `--exclusive`, arms interleaved
inside one job, ratio within a round.

## Sources
Working trees rsynced from the dev box (uncommitted sibling-repo fixes included, no git on the cluster):
fork = jj `rwnvuqwy` (probes, canonical fill, block-major opt-in, defaults flip); fastpart `dendro-refine`
@ `746b379` + working tree; EM4 `repartitioning` + sandwich fixes; NLSM `lts-graph-fix` + fixes; BSSN
repartitioning integration + tracker exchange. Toolchain: `gcc/13.3.0 openmpi/5.0.3-gcc13 cmake/3.31.11`.
CMake options copied from the local caches (EM4: `EM4_WITH_GRAPH_PARTITIONING=ON`, `EM4_ENABLE_PROFILING=ON`,
`EM4_COMPUTE_ANALYTICAL=ON`; NLSM: `NLSM_WITH_GRAPH_PARTITIONING=ON`, `NLSM_COMPARE_WITH_ANALYTICAL_SOL=ON`;
BSSN: `BSSN_WITH_GRAPH_PARTITIONING=ON`, `BSSN_WITH_AEH=OFF`, `BSSN_WITH_MSRK=OFF`; all Release,
`USE_64BIT_INDICES=ON`). The fork is built inside each solver (`DENDRO_dendrolib_DIR`), fastpart installed once.

## Experiments (one sbatch per rung, `run_ab.sh` interleaves the arms, 3 rounds)

**A — the communication drop, EM4 depth 9, wavelet tol 1e-8, AMR_FAC 0.004, t_end 5.0** (sized locally:
depth 7/1e-5 = 3k elements, depth 8 or 9 at 1e-6 = 4.1k, depth 9/1e-7 = 9.5k, depth 9/1e-8 = 22.5k at
t=0, growing as the pulse spreads; 560 elements per rank at np=40, 140 at np=160; the 0.1 AMR factor
flip-flops the init loop, 0.004 converges it). Arms, all with `DENDRO_COMM_STATS=1` (proof of fire + the byte counts):
`sfc_p0` (method 0, `DENDRO_ZIPSYNC_PRUNE=0`), `sfc` (method 0, defaults), `graph_p0` (method 3 plain,
prune 0), `graph` (method 3 plain, defaults), `refine` (method 3 + `DENDRO_GRAPH_REFINE=1`).
Rungs np = 40, 80, 160 (2/4/8 `c20` nodes). Read: `rkstep`/`unzip_async` phase timers, wall per run,
`total_bytes_sent` / `sync_bytes_sent`, `[graph-remesh] total=`. The pair `sfc_p0` vs `sfc` isolates the
side-channel removal on plain SFC; `graph_p0` vs `graph` the same on a ragged labelling; `sfc` vs
`refine` is the cost of repartitioning itself with the new defaults.

**B — the LTS win, `nlsmNUTS` depth 9** (bundle pars, t_end 4.0): `sfc` (method 0) vs `lw` (method 3,
`DENDRO_GRAPH_REFINE=2`). Rungs np = 40, 80, 160. Read: wall, `lts_substep_eff`, `dump_est_speedup`,
`LD Bal`.

**C — BSSN defaults gate at scale, q1 tinytest** (`BSSN_REFINEMENT_MODE=3`, t_end 3.0, IO off): `sfc` vs
`graph` (method 3), and `graph_p0` (prune 0) as the byte-identity control; np = 40, 1 round. Read:
constraint CSV md5 (`graph` = `graph_p0` bit-for-bit is the gate), wall as an indicator.

## Rules
Arms interleaved in one job on the same nodes; `/proc/loadavg`, `date`, `taskset -cp $$` recorded per
run; every run proves its knobs fired (`comm-stats-cumulative` line, `block-vote] reassigned` for
labelled arms) or is discarded; `</dev/null`, `ulimit -c 0`, `mkdir -p vtu`, `IO_OUTPUT_FREQ=0`.
Decisions on ratios within a round; report the deterministic counters next to every wall number.

## Log
- 15:55 sources rsynced to `repart/src/` (dendrolib 71M, bssn 113M, tpid 52M); bundle in `repart/bundle/`;
  build job 28408831 submitted on `kingspeak` (1 node, 16 cores, 2 h), pending on resources.
- 16:05 `kingspeak` fully allocated (17 of 22 nodes, rest mixed); `kingspeak-guest` (account `owner-guest`,
  preemptable) had 84 idle nodes. Build moved there (28408852, started at once on kp383). Rungs move to
  guest `c24&m128` nodes (22 idle: zpu 8, arup 8, hci 6): np = 48 / 96 / 192 on 2 / 4 / 8 nodes,
  `ab_guest.sbatch`; a preempted A/B round is discarded and rerun, never spliced.
- 16:20–16:40 build fixes: Eigen/GSL/OpenBLAS modules (`eigen/3.4.0 gsl/2.3 openblas/0.3.30`, `BLA_VENDOR=OpenBLAS`),
  the solvers' `CheckGit.cmake` needs a git repo → every shipped tree is a local snapshot commit; stale
  `git-state.txt` from a failed run breaks `LIST(GET)` → clean build dirs. Build 28408925 on kp295
  (E5-2680 v3, Haswell, c24 m128; `-march=native` pins runs to that type).
- 16:50 rungs queued with `afterok` on the build: A 28409204/205/206 (np 48/96/192), B 28409207/208/209,
  C 28409210 (np 48). Outputs `repart/runs/ab_<name>_<job>.out`, run dirs `repart/runs/<EXP>_np<N>_<job>/`.

## Results

**Experiment A, EM4 depth 9 (22.5k elements at t=0, 154 steps, 15 remeshes), Kingspeak guest c24 m128
nodes (E5-2680 v3), 24 ranks/node, five arms interleaved in one job, three rounds; `rkstep` /
`unzip_async` / `remesh` = max over ranks from `em4_prof_final.prof`; `ghost` = ghost-exchange bytes,
`sync` = side-channel bytes over the whole run. Repeatability inside a job is ≤ 1 %.**

| np | arm | wall s (3 rounds) | rkstep s | unzip_async s | remesh s | ghost GB | sync GB |
|---|---|---|---|---|---|---|---|
| 48 | sfc_p0 (full side-channel) | 222.5 / 222.3 / 222.4 | 184.6 / 184.7 / 184.8 | 127.4 / 127.3 / 127.3 | 15.0 / 15.0 / 15.1 | 290 | 1460 |
| 48 | sfc (defaults) | 138.5 / 138.6 / 138.4 | 107.1 / 107.2 / 106.9 | 60.9 / 61.1 / 60.9 | 15.9 / 16.1 / 16.0 | 290 | 0 |
| 48 | graph_p0 (plain fastpart, prune 0) | 422.1 / 418.6 / 423.2 | 331.9 / 328.7 / 332.3 | 253.3 / 251.2 / 254.4 | 52.6 / 52.5 / 52.7 | 328 | 2849 |
| 48 | graph (plain fastpart, defaults) | 208.9 / 208.8 / 208.6 | 131.8 / 131.7 / 131.8 | 81.2 / 80.9 / 81.1 | 56.9 / 56.9 / 56.7 | 328 | 0 |
| 48 | refine (`DENDRO_GRAPH_REFINE=1`, SFC-identical) | 181.3 / 181.2 / 181.8 | 111.7 / 111.7 / 112.0 | 60.4 / 60.5 / 60.5 | 51.8 / 51.8 / 52.0 | 285 | 0 |
| 96 | sfc_p0 | 175.2 / 175.0 / 174.9 | 142.8 / 143.0 / 142.8 | 107.5 / 107.6 / 107.3 | 11.5 / 11.4 / 11.5 | 400 | 2105 |
| 96 | sfc | 83.5 / 83.2 / 83.2 | 59.4 / 59.5 / 59.2 | 35.9 / 35.8 / 35.6 | 11.6 / 11.6 / 11.5 | 400 | 0 |
| 96 | graph_p0 | 408.4 / 404.9 / 407.8 | 325.6 / 322.2 / 324.9 | 268.9 / 265.7 / 268.2 | 42.9 / 42.9 / 42.9 | 443 | 4236 |
| 96 | graph | 141.6 / 141.8 / 141.9 | 80.7 / 80.9 / 80.7 | 54.0 / 54.3 / 54.0 | 44.3 / 44.1 / 44.3 | 443 | 0 |
| 96 | refine | 115.7 / 115.9 / 116.0 | 62.0 / 61.9 / 61.9 | 35.7 / 35.6 / 35.6 | 39.2 / 39.3 / 39.5 | 394 | 0 |
| 192 | sfc_p0 | 113.4 / 113.2 / 113.7 | 83.1 / 83.1 / 83.3 | 64.3 / 64.3 / 64.4 | 10.8 / 10.9 / 11.1 | 539 | 2978 |
| 192 | sfc | 59.4 / 60.0 / 59.6 | 34.1 / 34.6 / 34.5 | 21.2 / 21.7 / 21.4 | 11.0 / 11.1 / 10.9 | 539 | 0 |
| 192 | graph_p0, graph | `MPI_Abort(7)` ×3 each: a rank owns zero elements after labelling | | | | | |
| 192 | refine | 93.2 / 92.6 / 93.2 | 36.4 / 36.1 / 36.4 | 22.2 / 22.1 / 22.3 | 40.5 / 40.3 / 40.4 | 531 | 0 |

Ratios of round means (same nodes, same job):

| np | side-channel removal, rkstep (sfc_p0/sfc) | same on ragged labelling (graph_p0/graph) | refine/sfc rkstep | refine/sfc wall | refine/sfc remesh | graph/sfc rkstep |
|---|---|---|---|---|---|---|
| 48 | 1.72× | 2.51× | 1.045 | 1.31 | 3.2× | 1.23 |
| 96 | 2.41× | 4.01× | 1.043 | 1.39 | 3.4× | 1.36 |
| 192 | 2.42× | (crash) | 1.055 | 1.56 | 3.7× | — |

Readings (ratios inside a round, same nodes):
- **The side-channel removal is the answer to the question asked**: RK step 1.72× faster at np=48,
  2.40× at np=96, 2.43× at np=192 on plain SFC; 4.0× on the ragged labelling at np=96. The side-channel
  was moving 5–10× the ghost-exchange bytes and its share of the step grows with rank count.
- **Repartitioning vs SFC with the new defaults, per step**: the refined (SFC-identical) labelling is
  within 4 % (np=96) and 7 % (np=192) of SFC on `rkstep`, down from +27 % (July) and +10 % (this week,
  local). The plain ragged labelling is +36 % at np=96 (10 % more ghost bytes and a worse cut).
- **The sandwich tax is now the whole gap**: remesh 39–44 s per run vs 11–16 s for SFC (3.5×, 15
  remeshes), and it does not shrink with np. Wall: refine 1.39× SFC at np=96, 1.56× at np=192.
- On this symmetric EM4 case there is no win to have (the refined labelling *is* SFC); the repartitioning
  benefit lives on asymmetric/LTS cases, which this rung set could not measure (below).

**Experiment C, BSSN q1 tinytest (5.9k elements, `BSSN_REFINEMENT_MODE=3`, t_end 3.0), np=48, one round:**

| arm | wall s | BH locations | constraints | GW_L2 |
|---|---|---|---|---|
| sfc (defaults) | 72.7 | `00c4db2c…` | `249f7eec…` | `e5171c05…` |
| graph (defaults, vote 28 elements) | 124.7 | `00c4db2c…` | `3a282e62…` | `e5171c05…` |
| graph_p0 (prune 0 control) | 232.8 | `00c4db2c…` | `3a282e62…` | `e5171c05…` |

`graph = graph_p0` bit-for-bit at 48 ranks on a ragged labelling: the defaults gate holds at scale; the
side-channel removal is 1.87× in wall on BSSN's labelled path. BH locations agree with SFC bit-for-bit;
the constraint CSV differs between SFC and graph (as locally: different partition, same physics to the
printed digits, C_HAM 9.20684e-06 vs 9.20534e-06). Graph is still 1.7× SFC in wall on this 123-element-
per-rank case — the sandwich, not the step. NLSM (experiment B) was cancelled at np=48 too: 0 runs in
40 min on a 4.7k-element case; a scale-sized NLSM case is a separate, locally gated job.

**Defects surfaced at scale (not seen at np≤8):**
- ragged fastpart arms at np=192: `MPI_Abort(7)` from the fork's own guard `[repart] a rank owns zero
  elements after labelling` (mesh.cpp:19882) at the step-50 remesh: 24445 elements over 192 ranks, and
  the block-atomic vote (34 of 134 elements moved) left a rank empty. Needs a one-block floor per rank
  in the vote/partitioner (Roadmap), not a correctness issue. Also cosmetic: the labelled build prints
  `# of CG nodes (vertices) : 114682492337` (an unset counter) and the `DENDRO_ZIPSYNC_CHECK` report fired
  on ranks with an empty sync list (fixed locally, probe-only).
- `nlsmNUTS` levelwise arm hangs after `[repart] TOTAL` at np=96 and np=192: the gated depth-9 case has
  only 17 elements when the LTS driver first partitions, so most ranks are inactive — the labelled rebuild
  with inactive ranks hangs. Experiment B needs a case sized for ≥48 ranks (depth 10–11, tighter tol),
  gated locally first; not run today.

- 17:07 build 28408925 OK, but every run died in 2 s: OpenMPI `5.0.3-gcc13`'s UCX wants a Broadcom
  `bnxt_re0` device (another cluster's fabric); Kingspeak has Mellanox `mlx4_0`. The user's own running
  Kingspeak job (`thesis_kp_ladder.sbatch` → `chpc_env.sh`) loads `zlib/1.2.11 gcc/13.4.0 openmpi/5.0.3
  cmake/3.26.0 gsl/2.3` and launches with plain `mpirun`. Rebuilt on that toolchain (28409252), rungs
  re-queued: A 28409253/254/255, B 28409256/257/258, C 28409259. Probe 28409250 tests launch variants
  on the old binaries for the record.
- 17:45 second launch: MPI now fine (probe: `UCX_NET_DEVICES=mlx4_0:1` with `mpirun` works; `srun` has no
  PMIx plugin), but EM4 aborted at step ~50 (`lapack eigen solve failed`, `DGESV parameter 4 illegal`,
  C_DIVB ~20, 119k-element init mesh instead of 22k) and the NLSM SFC control blew up (U_PHI 1e13 at
  t=4): the `openblas/0.3.30` module is ILP64 (`USE64BITINT` in the .so) and every LAPACK call from the
  solvers' 32-bit prototypes is garbage — reference-element operators wrong, everything downstream
  garbage. The user's working build links the system reference LAPACK (`/lib64/liblapack.so.3`, LP64).
  Rebuilt with `BLA_VENDOR=Generic`, no openblas module. Experiment B also moved to the locally gated
  NLSM case (depth 9, tol 1e-5, t_end 0.5, grain 100) — the bundle's tol-1e-4/t_end-4 case was never
  gated and is not a valid control.

