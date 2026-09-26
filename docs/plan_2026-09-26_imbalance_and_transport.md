# Plan — price an imbalance case, then cut the transport (2026-09-26, session 2)

From `docs/handoff_2026-09-26_sandwich_next.md`. Standing rules unchanged: local runs, Kingspeak only under
the handoff's setup; one logical change per jj commit, subject-only; new behaviour behind a `DENDRO_*` knob
with the default byte-identical, gated knob-on, then default flipped and the defaults path gated; every gate
has the ragged arm and the SFC ENUTS arm; deterministic metrics locally, wall only from Kingspeak (arms
interleaved in one job, ratio within a round).

Gate harness moved out of the session scratchpad to `~/research/repart_gate/` (`scripts/{build,gate,em4,
nuts2,bssn,size_nuts}.sh`, `ref/` = tp base, EM4 tomls, ENUTS pars, TwoPunctures ID; `bin/<label>/` =
the four binaries per build, `bin/base` = jj `nrynmmrr`).

## Task 1a — an LTS case sized for 48–192 ranks
The gated depth-9 NLSM ENUTS case partitions first on `function2Octree`'s output (540 octants), which the
LTS init remesh grows to 4096: 11 elements per rank at np=48, below the grain (100) — most ranks inactive,
and the labelled rebuild hangs with inactive ranks. Size by sweeping `NLSM_MAXDEPTH` 9–11 and
`NLSM_WAVELET_TOL` (SFC arm, np=8, one coarse step): read the first octant count, the post-init mesh, the
per-level histogram and `dump_est_speedup`. Target: first partition ≥ 4 800 octants (≥ 100 per rank at
np=48), ideally ≥ 19 200 (192 ranks), with ≥ 3 populated levels so sub-cycling matters, and a local cost
that fits np=4/8 gating.
Gate: SFC (method 0) vs levelwise+refine (method 3, `DENDRO_GRAPH_REFINE=2`) at np=4 and np=8, U_CHI/U_PHI
min/max identical to 18 digits, proof of fire `[block-vote]` + `[graph-part-ex]`. Then a Kingspeak job
(48/96/192, arms interleaved): wall, `lts_substep_eff`, `dump_est_speedup`, `LD Bal`.

## Task 1b — an asymmetric global-RK case
BSSN q1 with BH-driven refinement (or `PARTEVOLVE_INIT=puncture`), ≥ 500 elements per rank at the target
np; arms SFC vs `DENDRO_GRAPH_REFINE=3` + `DENDRO_GRAPH_VTX_WEIGHTED=2` + `DENDRO_GRAPH_BALANCE_ROUNDS`;
read `padded_vol_imbalance` and wall. Sizing first (element count at the first partition and after the
init remeshes).

Both need the one-block floor per rank in the block vote if a ragged arm hits the zero-element guard at
scale (`mesh.cpp` `[repart] a rank owns zero elements`).

## Task 2 — the transport (`redistributeVec`), both directions
Today every call learns each source element's destination rank by a TN-hash directory rendezvous (3
Alltoall + 3 Alltoallv + an Allreduce), although the answer is already known:
- **s2g** (src = the ReMesh successor, dst = the labelled twin): the twin was built from the successor's
  local elements positionally (`buildGraphTwin`), and the partition decision labelled exactly those
  elements; dst rank of src element i = label i.
- **g2s** (src = the labelled mesh, dst = its SFC twin): if the twin's SFC distribution equals the
  labelled mesh's pre-partition distribution, dst rank of a labelled element = the rank it came from
  (`upper_bound(ele_offsets, eid)`).
`DENDRO_REDIST_HINT=1`: `repartitionMeshGlobal` keeps the pre-partition local TreeNodes, their labels
and each new local element's origin rank; `redistributeVec` uses them when the src (s2g) or dst (g2s)
local element list equals the stored pre-partition list on every rank (one Allreduce), otherwise the
directory. Byte-identical by construction (same `srcTarget`); `DENDRO_REDIST_HINT_CHECK=1` runs both and
counts mismatches. Also fuse the fixed-shape collectives (the three count Alltoalls are one, the five
key Allgathervs are one, the two orphan count Alltoalls are one) — same bytes, fewer rounds.
A phase timer `DENDRO_REDIST_PROF=1` (barrier-timed, max over ranks) splits directory / pack+exchange /
write / canon pull so the Kingspeak pair can price each piece.
Then, in order: the orphan-key Allgatherv (only if the probe shows the key count matters at scale), the
scatter-map double build (`DENDRO_MINIMAL_SCATTER_FILTER`), inherited labels. Each gated as above and
re-priced on Kingspeak with an EXP=S-style on/off profile pair.

## Log
- Sizing, NLSM (SFC, np=8, one coarse step): the first partition is on the pre-init mesh (min-depth
  floor, ~512–540 octants), not `function2Octree`'s output. d9/1e-5 → 4 096 after init (est. LTS
  speedup 2.73); d10/1e-6 → 6 504 (3.66); d10/1e-7 → 44 248 (1.15); d11/1e-7 → 44 080 (2.30).
  `NLSM_MINDEPTH=5` makes the first mesh 32 796 (170/rank at 192). The init loop flip-flops
  24 648 ↔ 44 080 (AMR factor 1.0); the 24 648 state has est. speedup 4.04 (704 elements at level 9).
  AMR-factor sweep (0.01, 0.001) pending.
- Sizing, BSSN q1 (SFC, np=8, t_end 0.7): `BSSN_MINDEPTH=4` → 9 080 after the first remesh;
  `=5` → 37 304 (777/rank at 48, 389 at 96).
- `DENDRO_REDIST_HINT` (jj `nowultmy`) gate, knob on with `_CHECK` and `DENDRO_REDIST_PROF`:
  testPartitioning four arms incl. ragged at np=4 and np=8 IDENTICAL to base; every transport call took
  the hint (7 labels + 5 origins per run, 0 directory), 0 mismatches against the directory in 12 checks
  per run. EM4 / ENUTS / BSSN arms pending (the combined build was killed under memory pressure).
- Committed opt-in, not yet built: `DENDRO_BLOCK_FLOOR` (`kqzzstmp`), `DENDRO_MINIMAL_SCATTER_FILTER`
  (`lxwqxrrw`). Staged: `DENDRO_PART_STATS` print hook (padded_vol_imbalance / lts_substep_eff from
  both arms' meshes).
- `DENDRO_REDIST_HINT` knob-on gate COMPLETE (bin `fs` = jj `kysqtzrm`, `_CHECK` + `DENDRO_REDIST_PROF` on):
  EM4 np=4 `d7bbee89`/`a75a069e`, np=8 `d7bbee89`/`32eb202e`; ragged depth-9 ENUTS and its SFC arm at
  the 18-digit reference; BSSN q1 graph `7a18bafe`/`c1508613`/`86470a02`/`ed1127e4`, SFC
  `7a18bafe`/`2ffba783`/`86470a02`/`b50bc05f`. Proof: every labelled-arm transport call routed by hint
  (EM4 8 labels + 7 origins, ENUTS 12 + 11 — NLSM's weighted SFC twin still matches the pre-partition
  distribution — BSSN 10 + 9), 0 directory fallbacks, 0 mismatches vs the directory. Local
  `[redist-prof]` (np=8, indicative only; route includes the check's directory): EM4 16–32 ms per call
  (route 1–3, pack 5, xchg 7, write 4–10, canon 1–9), BSSN 209–243 ms (pack 67–74, xchg 66–86, write
  45–56, canon 27–44).
- Knob-on gates, bin `fs`: `DENDRO_BLOCK_FLOOR=1` tp ×8 IDENTICAL; forced empty rank
  (`DENDRO_BLOCK_FLOOR_TEST=1`, ragged np=4): floor on → 21 PASSED, `[block-floor] rounds=1
  ranks_filled=1 elements_moved=1`; floor off → `MPI_Abort(7)` "a rank owns zero elements" (the 192-rank
  Kingspeak failure). `DENDRO_MINIMAL_SCATTER_FILTER=1` + `_CHECK`: tp ×8 IDENTICAL, EM4 ×4 on
  references, ragged + SFC ENUTS 18 digits, BSSN graph + SFC on references; filter vs rebuild 0
  differing lists in every check (10 per tp arm, 8 per labelled EM4 run, 12 ENUTS, 10 BSSN).
- Defaults flipped (`yoyvsrkp` hint, `onktvnlq` floor, `tvpmvwuu` filter; `=0` reverts each) plus
  `fix(mesh)` for `dumpPartitionStats` on meshes with inactive ranks (a global Bcast inside
  `computeMinMaxLevel` the inactive ranks skipped — hung/truncated NLSM's pre-init mesh under
  `DENDRO_PART_STATS=1`). Defaults-path gate, bin `def` = `xzuzoonw`, nothing on the command line:
  tp ×8 IDENTICAL (hint 7+5 per run, filter active, floor silent), EM4 ×4 on references (hint 8+7),
  ragged + SFC ENUTS 18 digits (hint 12+11), BSSN SFC on references rc=0, **BSSN graph on all four
  references but rc=139: SIGSEGV on rank 0 after `ETS time (max)`, i.e. at teardown** — open, repro
  under gdb + `HINT=0` / `FILTER=0` bisection running. The flip is not closed until that is explained.
- d11 LTS gate (bin `def`, `DENDRO_PART_STATS=1`): SFC vs levelwise+refine identical to 18 digits at
  np=4 and np=8 (U_CHI −1.49655671350051605e-13 / 1.10899864922085034, U_PHI −2.36383893241567700 /
  2.50214029822932194; 3 coarse steps, 56 232 elements, est. LTS speedup 2.52). **But the labelling is
  SFC-identical at np ≤ 8** (block vote 0, identical ghost volume and `LD Bal`): the centred pulse is
  per-level balanced by SFC there (`lts_substep_eff` 1.000 both arms). Next: the same stats at np=48
  oversubscribed (deterministic) before any cluster time, and an off-centre variant.
- The BSSN rc=139 is a pre-existing use-after-free, not the flip: `bssngr_main.cpp:674` deleted the mesh
  before the ctx, and `~BSSNCtx` → `dealloc_mpi_ctx(m_uiMesh, …)` reads the freed mesh's scatter counts
  (core backtrace: `dealloc_mpi_ctx` ← `~BSSNCtx` ← `main`). Intermittent (1 of 4 graph-arm runs; the
  identical rerun under gdb and the `HINT=0` / `FILTER=0` arms exit 0, all on the references); the new
  defaults only moved the heap layout. Fixed in the BSSN repo (ctx deleted before its mesh; user
  commits); `nlsm_sdc.cpp:224` has the same order (not on any gated path).

## Kingspeak round 1 (build 28412535, jobs 28412536–42) — T priced, L/U killed by a memory blow-up
**T, EM4 d9, arms interleaved in one job, 2 rounds (wall s):**

| np | sfc | refine, new defaults | refine, old (`HINT=0 FILTER=0`) | plain fastpart |
|---|---|---|---|---|
| 48 | 130.7 / 131.0 | 155.0 / 154.5 | 155.8 / 155.7 | 179.8 / 180.0 |
| 192 | 51.2 | 63.5 / 63.2 | 64.0 | 80.3 |

Fork-before vs fork-after (the two cuts): −0.6 % at 48, −0.8 % at 192. Refine vs SFC: 1.18× at 48,
1.24× at 192. Per run at 192 (15 remeshes, 31 transport calls, max over ranks): transport 1690 → 1522 ms
(the directory itself was ~5 ms per call), `block-decomposition` phase 1455 → 890 ms (the filter), s2g
10163 → 9574 ms, sandwich 15139 → 14285 ms = −57 ms per remesh (−5.6 %). The first scatter-map build is
now the biggest s2g phase (~140 ms per remesh). Plain fastpart completes at 192 ranks (the floor fired:
1 rank filled, 64 elements) where it aborted every time on 2026-09-25.

**L and U: every labelled arm died of OOM** (~47 GB per node cgroup cap vs ~9 GB for SFC), ~10–26 s in,
right after the initial labelled transport. Root cause (local capped repro at np=24, gdb on the
ballooning rank): `Mesh::buildBcastPlan` ← `broadcastCgValuesByPhysPosPlanned` ← `nlsm::redistributeDVec`
← `remesh_and_gridtransfer` ← `ExplicitNUTS` ctor. The consensus pos-bcast plan (NLSM/BSSN; EM4 opts out)
allgathered every rank's CG claims (3×u64 + cg + flag per slot, ~10–20 M slots on these meshes) and
hashed all of them on every rank: O(N_global) memory per rank. SFC arms at np=48: `lts_substep_eff`
0.568 (np=96: ele_imbalance 1.82, padded 2.x; np=192: 2.49 / 3.x) — the headroom is real.

**Fix: `DENDRO_BCAST_PLAN_DIST`** (jj `xzuzoonw`, flipped `lkvzkqnx`): home-rank election by position
hash (local before ghost, then lowest rank, then lowest cg = the legacy order), winners returned to
claimants and notified to winners, plan vectors rebuilt in the legacy order; O(local) memory. Gate
(`_CHECK` on): tp ×8 and EM4 ×4 on references (neither builds the plan), ragged + SFC ENUTS 18 digits
(12 builds identical in all 9 vectors), BSSN graph + SFC on references (10 builds identical). Capped
np=24 probe: legacy dies at 26 s in the first plan; distributed runs the whole init (7 sandwich remeshes
to 56 232 elements). BSSN teardown fix confirmed 3/3 rc=0 on references. Round 2 submitted: build
28412914, L 28412915–17 (48/96/192), U 28412918–19 (48/96); local defaults-path gate for the flip running.
