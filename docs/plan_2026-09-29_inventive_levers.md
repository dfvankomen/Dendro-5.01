# Plan: make levelwise pay (2026-09-29)

Source: four read-only investigations (angles A–D) on 2026-09-29, summarised in
memory `project_inventive_levers_2026-09-29`. All claims below are unverified until the
step that tests them lands. Rules as usual: new behaviour behind a `DENDRO_*` / `NLSM_*`
knob; default path byte-identical; ragged arms in every gate; one heavy run at a time;
local only (Kingspeak needs the user's OK); jj commits in dendrolib_dfvk only (one-line
`feat(scope): subject`), the NLSM worktree stays uncommitted (the user commits there).

## Why levelwise hasn't paid yet (hypotheses)
1. NLSM init loop builds 7 labelled meshes per run that are never stepped on
   (`nlsm-gr_lts/solver/src/nlsmCtx.cpp:117-160, 199-256, 296-345`); ~40–50 % of the
   Kingspeak L wall gap (estimate: iter 1–6 `[repart] TOTAL` × main-loop sandwich ratio).
2. ENUTS pays a fixed, level-independent comm cost per partial step: full element-DG ghost
   exchange per stage + 2 element-vector exchanges + `MPI_Barrier`
   (`include/enuts.h` ~2259/2366/2372). lw: 1.75× send nodes, 38 vs 25 partners. Not in
   comm-stats.
3. 64-element atomic blocks cap lw eff (model: 0.99 → 0.61 at np192/1024).
4. Possible measurement contamination: NLSM `is_remesh` converts to the SFC twin before the
   wavelet; with no remesh the run keeps stepping on SFC (`nlsm_nuts.cpp` ~431-445/514-523).

## Steps
**S0 — instrumentation (no value change).**
- dendrolib: `readFromGhostBegin/End{EleDG,Element}Vec` tallies (bytes, calls) in
  `[comm-stats-cumulative]` under `DENDRO_COMM_STATS=1`; `DENDRO_ENUTS_PROF=1` per-partial-step
  timers in `partial_evolve` (rhs+zipDG, pack/begin, wait/end, sync_blk_timestep, element-vec
  exchanges, barrier wait), reduced min/mean/max at the end of `evolve()`/destructor.
- NLSM: per-step print (knob) of whether the stepping mesh is labelled
  (`isGraphPartitioned()` or equivalent); count steps stepped on SFC in a labelled arm.
- Gate: ENUTS d9 ragged (`scripts/nuts2.sh`) knob on == knob off == ref; tp np4 gate.

**S1 — NLSM init loop stays on SFC** (`NLSM_INIT_STAY_SFC=1`, opt-in first).
- Init iterations stay on ReMesh's SFC output; after the loop build the labelled mesh once
  (`buildGraphTwin`), then `init_grid()` on it (no s2g transport needed).
- If S0 shows silent SFC stepping, fix it the same way (convert back to labelled when
  `is_remesh` says no) behind its own knob.
- Gate: `lw3.sh <l> <bin> 16 sfc lvl lw fp` knob on vs off: min/max 18 digits, same remesh
  count, same element counts; nuts2 ragged d9; count of labelled rebuilds drops 9→3.
- Then flip default if identical.

**S2 — level-scoped ENUTS element-DG exchange** (`DENDRO_ENUTS_SCOPED_EXCH=1`).
- Once per mesh: stable-sort each peer's send segment by the element's timestep factor
  (`getBlkTimestepFac(level)`), receiver sorts its recv segment by the same key (ghost levels
  known). At partial step pt only entries with `pt % DT == 0` form a prefix; send/recv only
  that prefix, skip peers with 0. Assert the divisibility chain (`DENDRO_LTS_RATIO_CAP`).
- `_CHECK=1`: also do the full exchange into a shadow buffer; compare the ghost slots
  `sync_blk_timestep` of active blocks reads.
- Later (S2b): ghost times computed locally (DT·(⌊pt/DT⌋+1)) → drop the elem-vec exchanges
  and the barrier, with its own `_CHECK`.
- Gate: nuts2 ragged d9 + lw3 np16 four arms bit-identical; S0 counters show the byte cut.

**S3 — scaling-hazard fixes** (after S2): O(N) Allgathervs at `mesh.h:~4359` (connmap
missing-id fallback) and `mesh.tcc:~6600` (redistributeVec directory fallback), orphan-fill
O(P·o) + std::map (`mesh.tcc:~7064-7290`).

**Deciding experiment (needs Kingspeak OK):** P=48, L case at MINDEPTH+1 (~8× n/rank), arms
sfc / sfc_weighted / lvl / lw, with S0–S2 on. Measures how per-step tax and rebuild scale
with n.

**Deferred:** per-level vote granularity (S4, full gate), NLSM wavelet on the labelled mesh,
no-op remesh skip, coarse-level merge, minimax refine, offline async critical-path model.

## Log
(filled in per step)

### S0 (2026-09-29) instrumentation landed, no value change
- Code: `readFromGhostBegin{EleDGVec,ElementVec}` tallies -> `eledg_/elevec_{bytes_sent,calls}` in `[comm-stats-cumulative]` (DENDRO_COMM_STATS=1); `DENDRO_ENUTS_PROF=1` phase timers in `ExplicitNUTS::partial_evolve`, reduced (Allreduce on MPI_COMM_WORLD) and printed in the destructor. NLSM (uncommitted, patch at `~/research/repart_gate/work/s0/nlsm_s0.patch`): `NLSM_STEP_MESH_TRACE=1`, plus `dumpCommStatsCumulative` call at the end of nlsm_nuts.
- Gates: nuts2 d9 ragged knobs off == knobs on == ref (U_CHI/U_PHI 18 digits); tp np4 4 arms IDENTICAL.
- lw3 d11 np16 (3 coarse steps, 192 partial steps, mean 1.98 active levels/step), mean s over ranks, sfc | lw:
  rhs+zipDG 32.6|37.9, eledg_begin 6.0|7.9, eledg_end (wait) 36.5|34.2, sync_blk_ts 52.5|53.1, axpy+ele_ts 3.4|4.5, elevec_xch 6.7|5.1, waitActive 0.14|0.10, partial total 137.8|142.9.
  eledg bytes sent 106.0 GB|130.5 GB (768|771 calls) vs nodal CG ghost 1.73 GB|1.98 GB (496 exchanges): element-DG exchange is ~61x|66x the CG ghost volume. elevec 39|49 MB.
- Step-mesh trace: lw stepped 3/3 on labelled; but `is_remesh` always swaps labelled->SFC and only `remesh_and_gridtransfer` swaps back. With remesh returning false (d9, NLSM_REFINE_MODE=3, test freq 1): mesh stays SFC and the run aborts (rc 134) on the next step (ENUTS not resynced). No silent-SFC contamination in the normal remesh-every-test runs.

### S1 (2026-09-29) NLSM init loop stays on SFC + is_remesh-false restore (NLSM worktree only, uncommitted; patch `~/research/repart_gate/work/s1/nlsm_s1.patch`)
- `NLSM_INIT_STAY_SFC=1` (opt-in): init iterations keep ReMesh's SFC output (no per-iteration buildGraphTwin + s2g; the top-of-iteration g2s never fires). After the loop, if the last iteration refined (exactly the default path's "ends labelled" condition; an iteration that does not refine ends on SFC in the default path too), build the labelled mesh once via buildGraphTwin, recreate EV + working DVecs + mpi_ctx on it (no transport; `init_grid()` re-evaluates the whole vector incl. ghost slots), then `init_grid()` and the existing deriv-workspace resize.
- `NLSM_REMESH_FALSE_RESTORE=1` (opt-in): `is_remesh` transports a COPY of EV to the SFC twin and keeps the labelled mesh + original EV (ENUTS aliases its memory) alive; false -> swap back (recreate working DVecs, mpi_ctx, deriv workspace, restore synced flag); true -> free them. Costs one extra labelled mesh in RAM during the wavelet.
- Gates: nuts2 d9 ragged: off == ref, STAY on == ref (incl. l2), STAY+RESTORE on == ref. lw3 d11 np16 sfc/lvl/lw/fp: U_CHI/U_PHI min/max/l2 identical to all digits off vs on in every arm; remesh count, element counts and LD Bal identical. `[repart] TOTAL` lines 10 -> 4 per labelled arm (init s2g 7 -> 1); summed rank-0 repart time lvl 17.4->7.5 s, lw 21.1->8.9 s, fp 20.9->9.1 s (box throttles, relative only). Wall per labelled arm 337 -> ~265 s.
- Task B: d9 NLSM_REFINE_MODE=3, remesh test every step: default aborts (rc 139 here, 134 in S0); with RESTORE=1 rc 0, trace labelled on all 5 steps, and U_CHI/U_PHI min/max/l2 bit-identical to the freq=0 control.
- Recommendation: flip STAY default (identical in every arm). RESTORE is a pure bug fix for runs that previously aborted; default-on candidate too, but it costs RAM (extra mesh) so kept opt-in pending a big-mesh RSS check.

### S2 (2026-09-29) level-scoped ENUTS element-DG exchange (`DENDRO_ENUTS_SCOPED_EXCH=1`, opt-in, default off = byte-identical)
- Code: `Mesh::buildEleDGScopedPlan` (src/mesh.cpp, end) + `readFromGhostBegin/EndEleDGVecScoped` (mesh.tcc, before `writeFromGhostBegin`) + `ot::EleDGScopedPlan` (mesh.h); ENUTS `prepare_scoped_exchange`, `scoped_check_compare` and the branch in `partial_evolve`'s stage loop (enuts.h). Plan cached per mesh (cleared with `m_uiBlkDepEid`): each peer's send segment stably sorted by ascending DT of the element, the receiver sorts its recv segment by its ghost elements' DT (same k-order); per-peer prefix counts per distinct DT. At pt the row is the largest distinct DT dividing pt. Compact packing, peers with 0 skipped, Begin/End still called every stage on every rank (tag/ctx bookkeeping unchanged). Scoped bytes go to the existing `eledg_` counters.
- Correctness read: in `partial_evolve` the only reader of ghost slots is `sync_blk_timestep(rk_s>=1)`, which reads `m_uiStVec[0..rk-1]` of dependent elements (`m_uiBlkDepEid`); `m_uiEvarDG` is not read there (only in the rk_s==0 / IGT branch, called from the remesh resync after a full exchange of EvarDG and all StVec, enuts.h ~2987). `m_uiEvarDG` ghosts come from that resync exchange and the end-of-`evolve` labelled zip (`readFromGhost*EleDGVec(m_uiEvarDG)`), both untouched (full). A stage slot on the owner is written only by zipDG of an active block; activity = `pt % DT(level)==0` (plus a skip for blocks running ahead, so DT|pt is a superset). Blocks single-level asserted per mesh. Preconditions asserted per mesh (DT chain, element levels in [lmin,lmax], single-level blocks, all-rank agreement); failure prints and falls back to the full exchange.
- `DENDRO_ENUTS_SCOPED_EXCH_CHECK=1`: per stage runs the full exchange into a shadow copy, then compares bitwise every non-local dependent element of the active blocks (all dof, all nodes); aborts on mismatch. nuts2 d9 ragged: 322k-373k comparisons per rank, 0 mismatches.
- Gates: knob off: nuts2 == ref, tp np4 4 arms IDENTICAL. Knob on: nuts2 == ref (incl. l2), CHECK 0 mismatches. lw3 d11 np16 sfc/lvl/lw/fp with knob on: min/max/l2 md5 identical to work/lw3_s1on in every arm.
- Measured (np16 d11, 192 partial steps, COMM_STATS+PROF; off | on; box throttles, relative only): eledg bytes sent sfc 106.0 -> 42.6 GB (0.40x), lw 130.5 -> 51.2 GB (0.39x); eledg_begin (pack) sfc 6.05 -> 0.95 s, lw 7.92 -> 1.21 s; eledg_end (wait) sfc 36.1 -> 25.4 s, lw 35.0 -> 24.0 s; partial_total sfc 136.5 -> 119.4 s (-12.5 %), lw 145.0 -> 125.3 s (-13.6 %). Byte cut is smaller than the 2 of ~7 active levels suggests because the finest level holds most ghost elements and fires every step. Remaining eledg wait is finest-level traffic, so the next lever is S2b (drop it via local ghost times) or a smaller-payload exchange, not more scoping.
