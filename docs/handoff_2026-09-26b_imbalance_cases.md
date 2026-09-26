# Handoff — repartitioning after session 2 of 2026-09-26: the imbalance cases and four more cuts

Read with `docs/plan_2026-09-26_imbalance_and_transport.md` (every gate, table and job id of this
session) and the vault's Repartitioning Index / Open Questions / Roadmap. Supersedes
`docs/handoff_2026-09-26_sandwich_next.md` where they differ. Everything is on the jj line, local only.

## State in one paragraph
Four more changes are on by default, each gated knob-on then on the defaults path, bit-for-bit on every
reference (`=0` reverts each): the transport routes by the partition's labels (s2g) and origin ranks
(g2s) instead of a TN-hash directory (`DENDRO_REDIST_HINT`); a rank no label reaches gets one block of
its own pre-partition slice instead of `MPI_Abort(7)` (`DENDRO_BLOCK_FLOOR`); the minimal scatter map is
filtered in place instead of rebuilt (`DENDRO_MINIMAL_SCATTER_FILTER`); and the pos-bcast plan (NLSM,
BSSN) elects winners at a home rank instead of allgathering every claim (`DENDRO_BCAST_PLAN_DIST`) —
that last one is a **memory** fix: the legacy plan was O(N_global) per rank and killed every labelled
NLSM/BSSN arm at 48–192 ranks on meshes of 30–60k elements (OOM at the 47 GB/node cgroup cap). On the
symmetric EM4 case the first two cuts are worth −57 ms per remesh at 192 ranks (−5.6 % of the sandwich,
−0.6/−0.8 % of wall); refine vs SFC is 1.18× at 48 and 1.24× at 192 in wall. The first case where
repartitioning can *win* is sized and gated: NLSM d11 LTS, where SFC's weighted partition reaches only
`lts_substep_eff` 0.568 at np=48 (1.000 at np ≤ 8, where levelwise equals SFC).

## New defaults and knobs (this session)
Default on: `DENDRO_REDIST_HINT`, `DENDRO_BLOCK_FLOOR`, `DENDRO_MINIMAL_SCATTER_FILTER`,
`DENDRO_BCAST_PLAN_DIST`. Check modes: `DENDRO_REDIST_HINT_CHECK`, `DENDRO_MINIMAL_SCATTER_FILTER_CHECK`,
`DENDRO_BCAST_PLAN_CHECK` (each compares against the old path and prints a mismatch count). Probes:
`DENDRO_REDIST_PROF` (per-phase ms per transport call), `DENDRO_PART_STATS` (partition snapshot per
block-building mesh and per labelled rebuild: ele/work/padded imbalance, ghost volume,
`lts_substep_eff`), `DENDRO_BLOCK_FLOOR_TEST=r` (forces an empty rank).

## The gate harness moved
`~/research/repart_gate/` (never a session scratchpad — `/tmp` is RAM): `scripts/` (`build.sh <label>
[tp em4 nuts bssn]` → `bin/<label>/`; `gate.sh`, `em4.sh`, `nuts2.sh`, `bssn.sh`, `lts11.sh`,
`size_nuts.sh`, `size_bssn.sh`, `oom_probe.sh`), `ref/` (tp base, EM4 tomls, ENUTS pars incl.
`nlsm_lts_d11_{sfc,lw}.par.json`, TwoPunctures ID), `kp/` (ship/fetch + bundle with experiments T, L,
U). References unchanged: EM4 `d7bbee89`/`a75a069e`/`d7bbee89`/`32eb202e`; ENUTS min/max in `nuts2.sh`;
BSSN graph `7a18bafe`/`c1508613`/`86470a02`/`ed1127e4`, SFC `…/2ffba783/…/b50bc05f`. Run gate lines
with more than one knob through `bash`, not zsh (no word splitting).

## The cases
- **LTS (Kingspeak L)**: NLSM, `NLSM_MAXDEPTH=11`, tol 1e-7, `NLSM_MINDEPTH=5` (so the *first* partition
  has 32 796 elements, not ~540), `NLSM_DENDRO_AMR_FAC=0.01` (1.0 flip-flops 24 648 ↔ 44 080 forever),
  grain 50 (the first init remesh dips to ~10.3k; all 192 ranks stay active). Settles at 56 232 elements,
  levels 3–9, est. LTS speedup 2.52. Arms: sfc / lw (method 3, `DENDRO_GRAPH_REFINE=2`) / lvl (method 4).
- **Puncture (Kingspeak U)**: BSSN q1, `BSSN_MINDEPTH=5`, refinement mode 3 → 37 304 elements (777/rank
  at 48). Arms: sfc / pv (`DENDRO_GRAPH_REFINE=3` + `DENDRO_GRAPH_VTX_WEIGHTED=2` + `DENDRO_GRAPH_BALANCE_ROUNDS=20`).

## Results of round 2 (build 28412914, L 28412915–17, U 28412918–19)
*(pending when this was written — see the plan's log)*

## What to do next
1. Read L/U (`kp/fetch.sh`; lines include `part-stats`, `redist-prof`, `LD Bal`, wall). If lw/lvl
   lift `lts_substep_eff` toward 1.0 and wall drops vs sfc, that is the first measured win — write it
   up as such, with every ratio labelled.
2. The next sandwich cut is the **first** nodal scatter-map build (~140 ms per remesh at 192, now the
   largest s2g phase), then the canonical pull's key Allgatherv (~17 ms per call at 192), then inherited
   labels. The directory rendezvous turned out to be ~5 ms per call — do not re-estimate from totals.
3. Commit in the BSSN repo (user): `bssngr_main.cpp` deletes the ctx before its mesh (the teardown
   use-after-free; intermittent SIGSEGV after all output is written). `nlsm_sdc.cpp:224` has the same order.

## Traps met this session
- Two memory-pressure kills of my own background work from overlapping heavy runs; a 48-rank NLSM
  labelled run needs ~60 GB here. One heavy run at a time; killed builds leave truncated objects.
- A small-np LTS gate on a centred pulse is SFC-identical — check the labelling differs before treating
  it as evidence.
- `dumpPartitionStats` used to skip `computeMinMaxLevel`'s global Bcast on inactive ranks (hang/truncate);
  fixed.
