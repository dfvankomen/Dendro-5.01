# Handoff — repartitioning: where it stands after the sandwich cuts, and what to optimize next (2026-09-26)

Read with the vault's Repartitioning `Next Session Handoff` / `Open Questions` / `Roadmap`, and, in this
repo, `docs/plan_2026-09-26_sandwich_cuts.md` (cut tables), `docs/plan_2026-09-25_kingspeak.md` (the
48/96/192-rank ladder, setup traps), `docs/audit_2026-09-25_s2g_cost.md`,
`docs/design_2026-09-25_remesh_without_twin.md`. Everything is on the jj main line, local only, never pushed.

## State in one paragraph
Correctness on genuinely ragged labellings is proven (global RK and ENUTS match SFC to 18 digits; the
BSSN four-arm gate is bit-for-bit). The fork's own per-step tax is gone (the zip-plan side-channel,
default pruned; the transport reads only source local slots). At 48–192 ranks on EM4 depth 9 the labelled
mesh is **within 4–6 % of SFC per RK step**, and the AMR sandwich is down from 3.17 to **1.21 s per remesh
at 192 ranks** (SFC's own remesh ≈ 0.7 s) after two cuts landed today: the ReMesh successor no longer
builds a zip plan it never uses (`DENDRO_LAZY_ZIPPLAN_SUCCESSOR`), and the zip-plan arbitration no longer
routes ghost claims that level-2 pruning would discard (`DENDRO_ZIPPLAN_CLAIM_FILTER`). **Wall vs SFC:
1.27× at 192 ranks, 1.19× at 48** (was 1.56× / 1.31× yesterday). Nowhere is the fork faster than SFC
yet — and it cannot be on this case: the EM4 gaussian is symmetric, the refined labelling is SFC-identical
and there is no imbalance to fix. The only thing the symmetric case can demonstrate is a tie, and the
remaining gap to that tie is entirely the sandwich.

## Defaults now (every knob `=0` reverts; all gated bit-for-bit on the defaults path)
`DENDRO_MINIMAL_SCATTER`, `DENDRO_ZIPPLAN_DIST`, `DENDRO_REDIST_ROUTE`, `DENDRO_REDIST_FLAGS` (EM4),
`DENDRO_ZIPSYNC_PRUNE=2` with labelled meshes pruned, `DENDRO_REDIST_CANON_FILL`,
`DENDRO_LAZY_ZIPPLAN_SUCCESSOR`, `DENDRO_ZIPPLAN_CLAIM_FILTER`. Opt-in only: `DENDRO_LAZY_ZIPPLAN` (global;
breaks NLSM via the g2s twin), `DENDRO_UNZIP_BLOCK_MAJOR` (byte-identical, slower), `DENDRO_UNZIP_BLOCK`
(upstream's unzip, A/B tool), the fastpart refine/balance knobs.

## The gate (run all of it for every change; a knob first, then the defaults path)
Scripts and references from this session live in the session scratchpad
`/tmp/claude-1000/-home-denv-research-dendrolib-dfvk/ee5e126c-*/scratchpad/{gate,ref,kp}` — copy them
somewhere durable first (`gate.sh`, `nuts2.sh`, `em4.sh`, `ref/tp_base`, `ref/*.param.toml`,
`ref/nlsm_lts_d9*.par.json`, `ref/tpid/`). If they are gone, the previous session's copies are described
in `docs/handoff_2026-09-25_speed_push.md`.
- testPartitioning four arms (m3, m3+refine, m4, ragged `PARTTEST_INIT=puncture`) at np=4 **and** np=8,
  TEST lines identical to base — exclude `[zipplan-prof]` lines if `DENDRO_ZIPPLAN_PROF=1` is on (they
  contain the word CONTESTED and match the grep).
- EM4 CSV md5: np=4 SFC `d7bbee89…` / labelled `a75a069e…`; np=8 `d7bbee89…` / `32eb202e…`.
- ragged depth-9 ENUTS (`nlsm_lts_d9.par.json`, method 3, vote ≈ 204) **and its SFC arm** (method 0):
  U_CHI min/max `-7.91133162581571663e-09 / 1.16722720164758065e+00`, U_PHI `-2.43354490884465102e+00 /
  5.02410073343503250e+00` (l2 differs by duplicate-slot counting; compare min/max).
- BSSN q1 tinytest (`BSSN_REFINEMENT_MODE=3`, t_end 6.0, `rit_q1_tpid_sol.bin` in the run dir): graph
  BH `7a18bafe…` / constraints `c1508613…` / GW_L2 `86470a02…` / GW_l2_m2 `ed1127e4…`; SFC `7a18bafe…` /
  `2ffba783…` / `86470a02…` / `b50bc05f…`. (Graph vs SFC differ in constraints and l2_m2 by partition —
  expected; each arm must reproduce its own checksums.)
- Every run must prove its knob fired (a print, a counter, a byte count); rc=0 alone is nothing.
Rules: never edit dendrolib sources while any build compiles them; one logical change per jj commit,
subject-only messages; never push; never put a process-name pattern on a command line; never edit a
running script; kill by PID from a script file.

## Kingspeak (the only place that can time; user's OK given for CHPC)
`/scratch/general/vast/u0720070/repart/` (home is full): `src/` = rsynced working trees turned into
snapshot git commits (the solvers' `CheckGit.cmake` needs a repo), `bundle/{build_guest,ab_guest}.sbatch`
+ `par/`, `bin/`, `runs/`. Toolchain = the user's `chpc_env.sh`: `zlib/1.2.11 gcc/13.4.0 openmpi/5.0.3
cmake/3.31.11 gsl/2.3 eigen/3.4.0`, **system reference LAPACK** (`BLA_VENDOR=Generic`; the
`openblas/0.3.30` module is ILP64 and silently corrupts every LAPACK call), `mpirun` with
`UCX_NET_DEVICES=mlx4_0:1` (the `-gcc13` openmpi module defaults to another cluster's `bnxt_re0`), build
on the run node type (`--constraint=c24&m128`, `-march=native`), guest partition (`owner-guest`,
preemptable, 24 ranks/node). `ab_guest.sbatch` arms: A (EM4 ladder), B (NLSM LTS — needs a bigger case),
C (BSSN gate), P/Q/S (phase and arbitration profiles, lazy and filter pairs). Compare arms only inside
one job, ratio within a round; deterministic counters (`[comm-stats-cumulative]`, `[graph-remesh]`,
`[repart] phase`, `[zipplan-arb]`, `[mesh-prof]`, `[unzip-prof]`) next to every wall number.

## Where the time is now (EM4 d9, 192 ranks, per remesh, refined labelling)
| term | ms | notes |
|---|---|---|
| g2s (no-block twin + transport) | ~195 | twin ctor ~40, transport ~150 |
| remesh + IGT on the twin | ~180 | ≈ SFC's own cost now that the successor is lazy |
| s2g: graph twin `createMesh(E2E_ONLY)` | ~70 | |
| s2g: labelled rebuild phases | ~500 | scatter map ~150, block decomposition ~170 (incl. ~95 of zip-plan work: arbitration 95, plan/inv), e2n-audit ~75, E2N expand ~60, partition decision 8, fetches 13 |
| s2g: `redistributeDVec` | ~200 | TN-hash directory rendezvous (3 Alltoallv) + orphan Allgatherv + canon pull |
| **total** | **~1210** | SFC ≈ 700 |

## What to optimize next, in order (estimated saving at 192 ranks; all behind knobs)
1. **The transport (`redistributeVec`, ~200 ms ×2 per remesh).** It runs a TN-hash directory rendezvous
   (Alltoall + 2 Alltoallv) to learn each element's destination rank, then an orphan/canonical-pull round
   with an **Allgatherv of all pull keys** (O(total keys) on every rank — only 68 keys on the BSSN swap,
   but check EM4 at scale with `DENDRO_REDIST_PROBE=1`). The g2s side already knows the twin's SFC
   splitters (destination = SFC rank of the element: a binary search, no rendezvous); the s2g side could
   carry the label vector instead of asking a directory. Gate: TEST 2b/8be (redistributeVec is under them)
   + the full set.
2. **The scatter-map rebuild (~150 ms) and its second build**: the audit found the minimal-scatter shrink
   rebuilds the whole nodal scatter map a second time (`DENDRO_MINIMAL_SCATTER_FILTER` idea: filter in
   place). Byte-identical by construction if the kept set is the same.
3. **Inherited labels across remeshes** (`DENDRO_REMESH_INHERIT_LABELS`, design memo S1): children take
   the parent's owner, fastpart only above an imbalance tolerance. Small direct saving (the partition
   decision is 8 ms) but it stabilizes the labelling (fewer element moves, fewer ghost changes) and is
   the entry to incremental rebuilds. Needs the one-block floor first.
4. **One-block floor per rank in the block vote** — plain fastpart at 192 ranks on 24k elements leaves a
   rank empty and hits the fork's own `MPI_Abort(7)` guard (`mesh.cpp` ~19882). Also the labelled
   rebuild must tolerate inactive ranks (the `nlsmNUTS` levelwise hang when the first LTS mesh has 17
   elements). Both are sizing floors, both block any ragged run at scale.
5. **Neighbour-routed claims / smaller claim messages** — second-order now (the two Alltoallv rounds are
   ~108 ms; `ClaimMsg` is 48 B and could be 28).
6. **Design A (remesh without the twin)** — scoped in `docs/design_2026-09-25_remesh_without_twin.md`;
   +0.3–0.5 s over 1–4 with all the bit-identity risk (flag sync-back, IGT's positional receive, the
   exact last-writer rule). Verdict: not yet.
Floor: the labelled rebuild will not go below SFC's own remesh; the honest target on the symmetric case
is a tie in wall.

## The experiment that would actually show a win (do this before more cuts)
The benefit cases are asymmetric meshes and LTS, and neither has a wall-clock number:
- **LTS (experiment B):** `nlsmNUTS`, SFC vs levelwise+refine (`DENDRO_GRAPH_REFINE=2`). The gated depth-9
  NLSM case has 17 elements when the LTS driver first partitions, which hangs the labelled rebuild at
  ≥ 48 ranks and crawls on SFC; the bundle's tol-1e-4 case blew up. Size a case locally first (depth
  10–11, tighter tolerance, ≥ 100 elements per rank at the *first* partition, gated to 18 digits vs SFC
  at np=4 and np=8), then the ladder. Deterministic readout: `lts_substep_eff`, `dump_est_speedup`,
  `LD Bal`; expected from the offline study: eff 0.7 → 1.0.
- **Asymmetric global RK:** the puncture mesh (`PARTEVOLVE_INIT=puncture` in testPartitioningEvolve, or
  BSSN q1 itself) with `DENDRO_GRAPH_REFINE=3` + `DENDRO_GRAPH_VTX_WEIGHTED=2` + `DENDRO_GRAPH_BALANCE_ROUNDS`
  (padded-volume imbalance 1.20 → 1.02 at np=16 in the offline study). BSSN at np=48 today: graph 124.7 s
  vs SFC 72.7 s on a 123-element-per-rank case — dominated by the sandwich; a case with ≥ 500 elements
  per rank and BH-driven refinement is the one to price.
- **Hybrid OpenMP+MPI** (user, 2026-09-25): after the `experimental` reconcile, 1–4 ranks per node with
  OpenMP inside; fastpart is weakest below ~64 ranks, so measure rather than assume. The fork's OpenMP
  partitioning path is opt-in and unvalidated (prove `OMP_NUM_THREADS=1` == serial first).

## Open items and traps carried forward
- GW `l2_m2` differs between the SFC and graph arms (BH, constraints, GW_L2 identical; each arm
  identical between prune levels) — unexplained, not a gate condition.
- The 1-ULP hanging-node residual on the fixed-mesh ragged ENUTS arm.
- BSSN/NLSM rebuild an O(N_global) Allgatherv bcast plan after every remesh (EM4 opts out) — separate
  item, may matter more than any of the above for BSSN at scale.
- The `experimental` reconcile (backup first; `mesh.tcc`/`ets.h` are the semantic conflict surfaces).
- Sibling repos still carry uncommitted edits (EM4 `repartitioning`, NLSM `lts-graph-fix`, the BSSN
  integration, fastpart `dendro-refine`); the cluster copies are snapshots of the working trees.
