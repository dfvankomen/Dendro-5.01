# Handoff — the speed push (2026-09-24 night → 2026-09-25 early morning)

Read with `docs/plan_2026-09-24_speed.md` (the plan and every result table) and the vault's
Repartitioning `Next Session Handoff` / `Open Questions`. Everything below is on the jj main line
(`jj log`), local only, never pushed.

## State in one paragraph
Correctness on genuinely ragged labellings is proven (global RK and ENUTS match SFC to 18 digits, ENUTS
through 10 remeshes). The labelled path is much faster than two days ago: rebuild 36 → 25 collectives,
sandwich hot loops halved (e2n-audit 290 → 50 ms, zip plan 514 → ~240 ms per remesh), unzip compute
−20 % SFC / −34 % labelled, unzip order and ENUTS dependency lists cached, redistribute route/flags
default on after np=8 byte-identity. The thesis handoff's padded-volume objective is met (SFC 1.20 →
1.02 at np=16 at SFC-level ghost volume; it conflicts with LTS efficiency, choose per solver). The one
big per-step lever — pruning the zip-plan side-channel sync that moves 4.7–5.9× the ghost-exchange
bytes on every mesh — is built, byte-identical on EM4 (both arms), NLSM ragged ENUTS and BSSN SFC (33 %
faster), but **opt-in** (`DENDRO_ZIPSYNC_PRUNE=2`) because graph-mode BSSN NaNs its constraint norm
with it on.

## The open defect, and exactly what is known
- Symptom: BSSN q1 tinytest, `BSSN_PARTITIONING_METHOD=3`, `BSSN_REFINEMENT_MODE=3`, any
  `DENDRO_ZIPSYNC_PRUNE ≥ 1`: `Constraint C_HAM L2 : (-nan)` from step 0; evolution, BH tracker and ψ4
  bit-identical to prune 0; solver deterministic (repeat run identical).
- Localised: 6 local constraint slots per rank at level 7 near the punctures. Rank 0's are zipped from a
  block whose EV padding is already NaN; rank 2's have **no zip-plan writer at all** and only ever
  received a value through the side-channel.
- Not the labelled mesh's prune: guarding the labelled rebuild off (full side-channel there) changes
  nothing. Not level 1, not the full scatter map, not the legacy zip, not adding sync send sources.
- Therefore the prune on the **ctor SFC meshes** (initial mesh, sandwich twins) removes values the
  initial transport reads — `redistributeDVec` (orphan fill by canonical phys, DG paths) reads
  source-mesh ghost slots that only the side-channel refreshed.
- `DENDRO_ZIPSYNC_CHECK=1` (full lists + report) shows dropped entries differ only on the first sync
  after a (re)build (transported vs primary value), never afterwards.
- Do first: instrument `redistributeDVec`/`redistributeVec` for reads of source ghost slots (cg outside
  the source's local range), then exchange those (or restrict the fill to source local slots), re-run
  `scratchpad/bssn/final2.sh`-style four arms (`sfc_p0 = sfc_p2`, `graph_p0 = graph_p2`), then flip the
  default back. Probes: `DENDRO_ZIP_NANCHK=1`, `DENDRO_ZIPSYNC_STATS=1`, `DENDRO_ZIPSYNC_CHECK=1`,
  `DENDRO_UNREF_CHECK=1`.

## Committed 2026-09-25 in the sibling repos (were uncommitted when this brief was first written)
- `dendrogr_dfvk_repartitioning`: branch `fix/bh-tracker-ghost-exchange` (`56250e4`, off master `639af21`) holds
  only the tracker exchange; master's working tree (the uncommitted repartitioning integration) is untouched.
- `em4_base_cpy` branch `repartitioning`: `02a7cfe` = flags ride the g2s transport with the default on.
- `fastpart` branch `dendro-refine`: `746b379` = the balance phase.
The items below describe those changes.
- `dendrogr_dfvk_repartitioning/BSSN_GR/src/bssnCtx.cpp`: ghost exchange of the three shift components
  before `computeBHLocations` (correctness fix; upstream has the same stale read).
- `em4_base_cpy/solver/src/solverCtx.cpp`: `DENDRO_REDIST_FLAGS` default on.
- `fastpart` branch `dendro-refine`: the balance phase (`refine_balance_rounds/_tol`); the option struct
  grew, so every fork build links `fastpart/build_balance/install` — never mix the old `.so` with new
  binaries.

## Gates and references (scratchpad `speed/`)
`gate.sh <label> <testPartitioning> [NP=8]` (four arms incl. ragged, diffs TEST lines against
`tp_base/np{4,8}_*.txt`), `nuts.sh <label> <nlsmNUTS>` (ragged d9 ENUTS, 18-digit reference in the
script), EM4 checksums: np=4 SFC `d7bbee89…`, labelled `a75a069e…`; np=8 SFC `d7bbee89…`, labelled
`32eb202e…` (the `4521dbc3…` noted on 09-24 was a sync-off artefact). BSSN references with the tracker
fix: SFC BH `7a18bafe…` / constraints `2ffba783…`; graph prune-0 BH `7a18bafe…` / constraints
`c1508613…`. `bin_c14/` holds the binaries of the final defaults gate.

## Next after the defect
Block-major unzip pass from a per-call DG buffer (the unzip agent's next lever); one-group floor in the
balance budget; the 1-ULP hanging-node residual on the fixed-mesh ragged ENUTS arm; cluster pricing
(`tools/cluster_bundle`, needs the user's pushes and OK); later, backup then reconcile with
`dfvankomen/experimental`.

## Traps that cost time
Never put a process-name pattern on the command line (self-kill ×3; use a script file). Never edit a
running bash script. Never edit dendrolib sources while any build compiles them (object-layout
mismatch → heap corruption that looks like an MPI bug); killing `make` leaves truncated objects with
fresh mtimes. BSSN TOML keys must precede `[AEH_PARAMS]`. `PARTEVOLVE_INIT=puncture` (evolve test) vs
`PARTTEST_INIT=puncture` (mesh test). Wall-clock on this box is an indicator only.

## Update, 2026-09-25 afternoon — the defect is closed, the prune is the default again

Full plan, probes and every table: `docs/plan_2026-09-25_transport_read.md`.

- **The reader** was `redistributeVec` step 2, but one level deeper than the brief guessed: it reads
  source ghost slots of an intergrid-transfer output that no ghost exchange ever filled (the BSSN init
  loop's last remesh + IGT is the swap's source; `interGridTransfer` ends with the duplicate sync only,
  which with full lists happened to deliver every ghost cg and pruned delivers nothing). The transport
  probe showed the slots as scatter-map-delivered and *all-zero* (malloc zero pages), not non-finite;
  rank 0's six NaN constraint cgs were exactly the six local slots the transport wrote from them. The
  s2g transport has the same dependency.
- **Fix** `DENDRO_REDIST_CANON_FILL` (jj `putpvosp`, default on since `xyozwlmw`; `=0` reverts):
  every destination local slot without an authoritative write, plus destination ghost read-set slots no
  destination exchange delivers, is pulled from a source local slot by cg2dg canonical position through
  the orphan-fill machinery (68 keys globally on the BSSN swap). Do not widen it to the whole ghost read
  set (first cut: 661k keys).
- **Gates green** with the fix on at prune 2 and again on the plain defaults path: testPartitioning
  four arms at np=4 and np=8 (prune 0 and 2) identical to base; EM4 np=4 `d7bbee89`/`a75a069e`, np=8
  `d7bbee89`/`32eb202e`; ragged depth-9 ENUTS 18 digits; four-arm BSSN q1 (t_end 6.0) `sfc_p0 = sfc_p2`
  and `graph_p0 = graph_p2` bit-for-bit on BH locations, constraints, GW_L2 and GW_l2_m2 (SFC
  constraints `2ffba783…`, graph `c1508613…`); defaults-path BSSN graph and SFC arms reproduce those
  files exactly with the labelled meshes pruned too. EM4 sync bytes on defaults: SFC 0, labelled
  3.3 MB (np=4) / 4.8 MB (np=8) against 43.8 GB with the full side-channel.
- **Defaults now** (`feat(repart)`, `xyozwlmw`): `DENDRO_REDIST_CANON_FILL` on, `DENDRO_ZIPSYNC_PRUNE` 2,
  `DENDRO_ZIPSYNC_PRUNE_LABELLED` on (the guard's premise was this bug); `=0` reverts each.
- **Block-major unzip** (`perf(unzip)`, `yyloslms`, `DENDRO_UNZIP_BLOCK_MAJOR=1`, opt-in): byte-identical
  on every gate, but unzip compute 13.1–13.4 s (scatter) vs 21.2–21.6 s (block-major v1), 3/3
  interleaved rounds at equal load. v2 (interpolants cached per element, squashed into the same commit) is byte-identical and
  still ~10 % slower (7.2 vs 7.9 s, 3/3 rounds on a quieter box): the lever is closed, knob opt-in.
- **Probes kept** (gated, in mesh.tcc): `DENDRO_REDIST_PROBE=1` (source ghost reads and destination
  slots by write path, non-finite and all-zero), `DENDRO_GHOSTREAD_CHECK=1` (unzip read set before and
  after each exchange, first 6 calls per mesh), `DENDRO_UNZIP_FILLCHK=1` (sentinel-filled padded
  buffer, unwritten slots per block). Any future probe for this class must count all-zero slots.
- **Open**: the GW `l2_m2` mode file differs between the SFC and graph arms (BH, constraints, GW_L2
  identical; each arm identical between prune levels) — not a gate condition, unexplained.
- Gate scripts and references for this session live in the session scratchpad `gate/` and `ref/`
  (copies of `tp_base/`, the EM4 param files, the NLSM d9 par, the q1 TOML and the TwoPunctures
  solution); `bin_def/` holds the defaults-gate binaries.

## Kingspeak, 2026-09-25 evening — the communication drop priced at 48–192 ranks
Plan, setup traps and every table: `docs/plan_2026-09-25_kingspeak.md`; memory
`project_kingspeak_setup_2026-09-25`. EM4 depth 9 (22.5k elements), guest c24 nodes, five arms
interleaved ×3, repeatability ≤ 1 %:
- **The side-channel removal (this week's default) is 1.72× / 2.41× / 2.42× on the RK step at np =
  48 / 96 / 192 on plain SFC** — the fork's SFC path before vs after, *not* a speedup over SFC; against
  SFC (no repartitioning) the fork is still 1.31 / 1.39 / 1.56× slower in wall, all of it sandwich (side-channel bytes 1.5–3.0 TB per run → 0), and 2.5× / 4.0× on the
  ragged fastpart labelling at 48 / 96. The old fork's SFC baseline was paying two thirds of its step
  for it at 192 ranks; every earlier "graph vs SFC" comparison was between two taxed arms.
- **Repartitioning vs SFC, per step, on the new defaults:** the refined (SFC-identical) labelling is
  within 4.5 % (48), 4.3 % (96) and 5.5 % (192) of SFC on `rkstep`; plain fastpart +23 % / +36 %.
- **What is left is the sandwich:** remesh 3.2–3.7× SFC's per run and flat with np, so wall is 1.31 /
  1.39 / 1.56× SFC. On this symmetric case there is no win to have; the benefit cases (asymmetric /
  LTS) still need a scale-sized, locally gated NLSM/BSSN case.
- BSSN q1 at np=48: `graph = graph_p0` bit-for-bit on a ragged labelling (the defaults gate holds at
  scale), 1.87× faster in wall; SFC 72.7 s vs graph 124.7 s (sandwich).
- **New defects at scale:** plain fastpart at np=192 on 24k elements leaves a rank with zero elements
  after the block vote → the fork's own `MPI_Abort(7)` guard (needs a one-block floor per rank);
  `nlsmNUTS` levelwise hangs when the first LTS mesh has fewer elements than ranks (17 at np ≥ 48).
- Kingspeak scratch layout and the four traps (ILP64 openblas, `bnxt_re0` UCX default of the `-gcc13`
  openmpi module, `CheckGit.cmake` needing a git repo, `-march=native`) are in the memory note.


## 2026-09-26 — sandwich cut 1 landed
`DENDRO_LAZY_ZIPPLAN_SUCCESSOR` (default on, `=0` reverts): the mesh `Mesh::ReMesh` builds defers its
zip plan to its first unzip; in the sandwich solvers it never unzips, so 0.8–1.0 s of its 1.0 s ctor per
remesh is gone (Kingspeak: sandwich 3.17 → 2.65 s at 192 ranks, 4.32 → 3.32 s at 48). Gated with the knob
on and on the defaults path (tp ×16, EM4 ×4 both np, ENUTS ragged + SFC, BSSN graph + SFC at t_end 6.0
bit-for-bit). Next: the zip-plan arbitration (`winners`, 1.0 s per rebuild at 192 ranks, the only phase
that grows with rank count, and ~all of plain SFC's own remesh) — `docs/plan_2026-09-26_sandwich_cuts.md`.

## 2026-09-26 — sandwich cut 2a landed: the zip-plan ghost-claim filter
`[zipplan-arb]` sub-timers (`DENDRO_ZIPPLAN_PROF=1`) showed the arbitration is per-claim CPU at 48
ranks and two dense Alltoallv rounds at 192, both ∝ claims, and 70–94 % of a rank's claims are ghost
claims that exist only to be pruned later. `DENDRO_ZIPPLAN_CLAIM_FILTER` (default on, `=0` reverts)
skips the ghost claims level-2 pruning would drop (winners are only consulted for local cgs, so the
lists are identical after the prune). Kingspeak np=192: arbitration 1152 → 95 ms, labelled sandwich
2.48 → 1.21 s per remesh, wall 85.0 → 65.9 s (SFC 59.6 → 51.9 s, its own arbitration 592 → 88 ms),
checksums identical; every local gate bit-for-bit (knob on and defaults path). Wall vs SFC at 192 ranks
is now 1.27× (was 1.56× yesterday). Commits: `vsymxzum` timers, `zrsmyuxw` filter, the flip, docs.
Plan + tables: `docs/plan_2026-09-26_sandwich_cuts.md`.

