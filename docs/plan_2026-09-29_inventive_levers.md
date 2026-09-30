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

### S3 (2026-09-29) O(N_global) gathers out of the remesh path (`DENDRO_CONNMAP_SPARSE`, `DENDRO_REDIST_ORPHAN_P2P`; landed opt-in, flipped default-on after the gates, `=0` reverts)
- Connmap missing-id fallback (`mesh.h` `buildOctantConnectivityMap`, the `numMissingGlobal > 0` block): hash rendezvous on (x,y,z,level). Each rank ships its local (key,gid) to the key's directory rank (Alltoall counts + Alltoallv), then queries every unresolved element slot (same set as the old Allgatherv resolved), answers come back by Alltoallv. O(local) memory, no `totalGlobal*5` int. Tally `[connmap-tally] path=sparse|allgatherv ...` (named to stay out of the gate's `conn-map` grep). Ragged tp np4: 662-2182 queries per call, all resolved; nuts2 d9 4 calls; lw np16 3 calls (29299 queries).
- redistributeVec directory: nothing to change. `DENDRO_REDIST_ROUTE` (TN-hash rendezvous) has been default-on since nmxuvsyp (`!= '0'`); the Allgatherv branch only runs with `=0`. The `[redist-hint] mode=directory` line means "hint missed, directory used", not "Allgatherv used", so the np192 log was the rendezvous path. Added `[redist-dir-tally] path=rendezvous|allgatherv elems=N` (under `DENDRO_REDIST_PROF=1`); lw np16 shows rendezvous.
- Orphan/canon fill (`mesh.tcc`, step 5, `orphanP2P` branch; old code kept as the else branch, also used with `DENDRO_ORPHAN_FILL_GEOM_KEY=1`): no Allgatherv of orphans and no std::map over every cg. Each rank Allgathers a 32^3 occupancy bitmap of its src-local canonical cg positions (O(P) words, exact superset filter), sends each orphan query only to ranks whose bitmap covers its cell, answers from an unordered_map of just the queried positions filled in one ascending pass over local cgs (first wins = lowest cg, as before). Write order kept (own rank first, then ascending source rank), hence bit-identical even for dup-LOCAL corners. Tally `[redist-orphan] path=p2p orphans=O queries_sent=Q (gather path: O*(P-1)) answered=A` under `DENDRO_REDIST_PROF=1`. Seen: ragged tp np4 O=14-24, Q=15-25 vs 42-72; lw np16 O=227, Q=254 vs 3405; nuts2 O=75, Q=52 vs 225. Bitmap filter is coarse: at very high P with fat SFC bboxes the query fan-out may rise; the tally shows it.
- Gates (knobs on, then off, then default-on rerun): tp np4 and np8 all four arms IDENTICAL; nuts2 d9 ragged == ref incl. l2; EM4 np4/np8 sfc d7bbee89, graph np4 2603beb3 / np8 32d48b0b; lw3 np16 sfc/lvl/lw/fp U_CHI/U_PHI md5 equal to lw3_s2off in every arm.
- Not done: no scale measurement (needs np>=96; box is np16 max); the bitmap Allgather is O(P) per rank but O(P^2) total, fine to a few thousand ranks.

### S4 (2026-09-29) per-level vote granularity (`DENDRO_LEVELWISE_VOTE_ADAPT=1`, opt-in, default off = byte-identical; `=2`/`=3` test modes: single-element / octet groups regardless of n_L)
- What enforces atomicity: not the block decomposition of the labelled mesh but the anchors shipped per element in `oct_data.blkAnchor*` (source SFC blocks, `deriveBlockInfoFromBlocks` mesh.cpp ~5978). `blockVote` (mesh.cpp ~20275) and `blockFloor` group by them; the labelled mesh rebuilds blocks by bucketing local elements by anchor (`buildBlocksFromCanonicalInfo` ~6008, called in the rebuild ~21797), so a group on one rank is a block. Blocks are single-level full cubes (`OCT2BLK_DECOMP_LEV_GAP 0`, volume equality in `octUtils.cpp` ~1120-1150). Nothing downstream needs the SFC block: neighbour discovery is geometric (`findBlockNeighborsWithoutSFC`), the unzip canonical writer rule is geometric (`buildUnzipCanonicalWriterTable`, level/xyz), `rotID` is unused after decomposition. So sub-blocks are legal.
- Design (`levelwiseLabels`, mesh.cpp ~20160-20260): per level s_L = largest power of two with s^3 <= n_L/(8P), floor 2 (n_L from the existing Allgather). Elements whose source block edge exceeds s_L get their block anchor rewritten to the ancestor octant at level L-log2(s_L) (only inside the source block; checked), so the labelled mesh builds blocks of that size. Each group is cut at its per-level prefix midpoint ((2*start+cnt)*P/(2*n_L)); the later vote then reassigns 0. `DENDRO_LEVELWISE_MATCH` and refine starts still work (they consume `lab`/anchors). Rank 0 prints `[levelwise-adapt]` (per-level n, s, #re-anchored, source block-edge histogram).
- Gates, knob on: testPartitioning np4/np8 x 4 arms all IDENTICAL to base (tp meshes only have blocks of edge <=2, so nothing is subdivided there). Stress: forced octets/singletons on uniform L5 (32768 el, 1024 blocks) and an eOrder-6 puncture mesh with edge-4 blocks: fails 0, TEST 10/11/COVERAGE maxErr 0, 8bo mism 0 (eOrder 6, pad 3). Caveat: TEST 8bo at eOrder 4 uses a width-3 stencil with pad 2, so it skips block-boundary nodes and is block-shape dependent by construction; it flags any block change (not a bug, use eOrder 6). Global RK: EM4 np4 method 4 (levelwise), 7 remeshes, knob 0/1/3 all CSV md5 d7bbee89 == SFC ref (512 elements re-blocked); nlsmNUTS GTS (ts_mode 1) d9 m4 forced octets on 14848 elements: min/max digits identical.
- ENUTS does NOT stay identical when blocks are subdivided: d9 m4 np4 forced octets on 3584 L7 elements (mode 3) moves U_PHI min -2.43354 -> -2.43126 (mode 1 rule subdivides nothing there, digits == ref). lw3 d11 np16 (mode 1, L8 4096 + L9 512 elements re-blocked): lvl U_PHI max 2.50214 -> 2.50147, l2 3054.228 -> 3053.703 (SFC == knob off). So the ENUTS scheme depends on block granularity (per-block times/dependents in `sync_blk_timestep`); not diagnosed further. Consequence: with the knob on, labelled != SFC numerics under ENUTS (same order as the LTS interface error); default stays off.
- lw3 d11 np16 part-stats off | on: lvl lts_substep_eff 0.973 | 0.990, work_imb 1.028 | 1.010, padded_vol_imb 1.050 | 1.039, total padded vol 41.22M | 42.00M (+1.9 %), ghost send nodes 9.33M | 9.41M, LD Bal max 88255 | 86743. lw eff 0.937 | 0.952, work_imb 1.063 | 1.049, padded_vol_imb 1.073 | 1.063, ghost send 7.16M | 7.35M, LD Bal max 91234 | 90036. np16 barely binds the cap (n_L/128 < 64 only on L<=6 and L9).
- Model (angleB, pm10 560k elems, `lw_blkad` = this rule with s_max 4, script `run2.py`; capped at 8 GB, wall time minutes): eff at np96/192/1024: lw 0.999/0.998/0.989, 64-atomic 0.940/0.898/0.614, adaptive 0.967/0.983/0.915; ghosts vs SFC 2.73/2.72/2.74x (lw), 2.64/2.57/2.19x (atomic), 2.66/2.70/2.65x (adaptive); scoped T(beta=.05) 0.875/0.849/0.710 (lw), 0.881/0.807/0.539 (atomic), 0.882/0.850/0.707 (adaptive). So the rule recovers nearly all of the atomic-block loss in eff at a 3-21 % ghost cost vs atomic.
- Verdict: correct and gated for global RK; opt-in because ENUTS numerics shift with subdivided blocks. Needs the Kingspeak np192 run to see whether eff 0.58 -> ~0.95 shows up in wall.

### Kingspeak K/K2 (2026-09-29) S0-S2 at scale, NLSM L case (d11), dendrolib snapshot 614493cb
- Jobs: K 28494836 (np48, 1 round), K2 28494838 (np96, 2 rounds), K2 28494842 (np192, 2 rounds). Min/max digits bit-identical within each labelling (lw = lw_sc = lw_initold; sfc = sfc_sc).
- Wall s (np48 / np96 / np192): sfc 140.2 / 88.8 / 63.9; sfc_sc 128.1 / 81.5 / 59.4; lvl_sc 125.2 / 79.7 / 54.0; lw_sc 129.2 / 81.3 / 55.5. np48 also: lw 153.8, lw_initold 164.7 (S1 = -11 s, S2 on lw = -25 s).
- lvl_sc / sfc = 0.89 / 0.90 / 0.85; lvl_sc / sfc_sc = 0.98 / 0.98 / 0.91 (was lvl/sfc 1.14 / 1.20 / 1.18 in run M). partial_total lvl_sc / sfc_sc = 0.93 / 0.92 / 0.82.
- np192 lts_substep_eff lvl 0.58 (block cap) -> S4 target.

### Kingspeak K3 (2026-09-29) S4 at np192, job 28495053, snapshot 8505ed25, 2 rounds
- Wall s: sfc_sc 59.25, lvl_sc 54.55, lvl_sc_ad 53.55, lw_sc_ad 53.85 (lvl_sc_ad / sfc_sc 0.90, / sfc(K2) 0.84).
- lvl_sc -> lvl_sc_ad: lts_substep_eff 0.58 -> 0.93, work_imb 1.61 -> 1.05, padded_vol_imb 1.55 -> 1.33, rhs+zipDG max 9.5 -> 7.0 s, eledg_end mean 10.8 -> 9.2 s, but sync_blk_ts+post mean 12.6 -> 15.1 s (smaller blocks = more dependent padding elements to time-sync); partial_total 37.5 -> 36.1 s. Ghost send nodes +4 %.
- Verdict: S4 recovers eff as modelled, but sync_blk_timestep eats most of it. Next lever: sync_blk_timestep cost (38 % of partial time on every arm, grows with block surface/volume).

### S2 default flip (2026-09-29) `DENDRO_ENUTS_SCOPED_EXCH` now default on (`=0` reverts)
- em4NUTS np4 (SOLVER_ETA_R0=1e-4 forces sub-cycling, t_end 8, ~20 remeshes): sfc and method 4 labelled, CSV md5 b33db3e6 off = on = CHECK (method 3 aborts at the first remesh in the off arm too, so method 4 is the labelled arm). CHECK 43008 compared elems/rank, 0 mismatches.
- bssnSolverNUTS np4 (q1 tinytest, refinement 3, t_end 1.6, 3 coarse steps, LTS 8 levels), sfc and method 4 labelled: alpha/constraint digest md5 5dabb4ff off = on = CHECK, final ||H||=1.58137806e+01 ||M||=9.54500527e+00, sfc = labelled. CHECK 61280 elems/rank, 0 mismatches. The labelled arm needed a post-IC maybe_apply_partitioning + sync_with_mesh in the driver (the COPY driver never applied the method, so its "labelled" arm was silently SFC); labelled arm ran with DENDRO_NUTS_NO_WPART=1 and remesh off (labelled ENUTS init remesh fails in the off arm too, pre-existing).
- BSSN driver: scratch build at scratchpad/bssnsrc = repartitioning tree + bssn_nuts.cpp from dendrogr_dfvk_COPY + the `#if 0` endif move + rhs_blk bssnrhs(t, constVars) signature fix; the other repos were not touched (COPY builds against dendrolib_upstream, fastpart-less).
- NLSM: nuts2 on = CHECK = default = ref digits (CHECK 317k-373k elems/rank, 0 mismatches); lw3 np16 lvl on = lw3_s3on/lvl digits (U_CHI -1.49655671350051605e-13 / 1.10899864922085034e+00, U_PHI -2.36385011495455943e+00 / 2.50214029822932194e+00).
- After the flip, no knobs: nuts2 = ref, gate.sh NP=4 all four arms IDENTICAL to base.
- Fix: the CHECK summary in `~ExplicitNUTS` touched the app ctx after drivers delete it (segfault at teardown in bssnSolverNUTS); it now uses MPI_COMM_WORLD rank.

### Correction (2026-09-29, checked from run logs)
- lw3 d11 np16 default: sfc, lvl, lw, fp all end at U_PHI (min, max) = (-2.36385011495455943, 2.50214029822932194), identical min/max to 18 digits across partitionings. With DENDRO_LEVELWISE_VOTE_ADAPT=1 (work/lw3_s4on/lvl): (-2.36383893241567700, 2.50146546299599537). So S4 moves the digits away from the SFC value (the S4 report quoted the direction reversed).

### LTS follow-ups (2026-09-30) em4NUTS labelled init remesh, sync_blk_timestep cost (`DENDRO_ENUTS_SYNC_FAST=1`, opt-in, default off = byte-identical)
- em4NUTS method 3 abort, root cause: `ExplicitNUTS` ctor (useWpart) calls `remesh_and_gridtransfer` before `init()`; with `SOLVER_PARTITIONING_METHOD != 0` that call runs the g2s/s2g sandwich on the still-SFC mesh and leaves the ctx on a LABELLED mesh. `SOLVERCtx::initialize()` (INIT_GRID_ITER=10) then runs its wavelet + raw `this->remesh()` (ReMesh assumes SFC-ordered local ranges) on that labelled mesh: "[Remesh Error] M1 & M2 front alignment failed", exit(0). One `[graph-remesh]` line precedes the error (the ctor one). Method 4 survived because its init wavelet happened to flag nothing; `SOLVER_INIT_GRID_ITER=0` also runs method 3 to the end (diagnostic). Not an enuts.h bug. NLSM is immune (its initialize converts to the SFC twin first, S1).
- Fix (solver side, patch `~/research/repart_gate/work/lts/em4_init_labelled.patch`, em4_base_cpy not touched): `SOLVERCtx::m_uiIsLabelled` (set by `swap_to_graph_partition` and the end of the graph `remesh_and_gridtransfer`); `initialize()` routes a refining iteration through `remesh_and_gridtransfer` (sandwich) when the mesh is labelled, else the old `remesh()+grid_transfer()+swap`. Same flags propagate from the labelled wavelet through `redistributedFlagVec`.
- Gates (em4NUTS np4, SOLVER_ETA_R0=1e-4, t_end 8): sfc csv b33db3e6 (== previous S2-flip ref); method 4 csv b33db3e6 (== sfc); method 3 now rc 0, csv 46cc7faa, nodes 428944 vs 428869 (labelled init wavelet differs slightly), U_E0/E1/E2/B*/ L2 ratio to sfc = 1.000 at every step (3 digits), final U_E0 L2 1.27598e-3 vs 1.27592e-3. `DENDRO_ENUTS_SCOPED_EXCH_CHECK=1` + SYNC_FAST: csv == non-check (46cc7faa), 0 mismatches on all ranks.
- bssnSolverNUTS labelled init remesh failure (diagnosis only): same signature (`[graph-remesh]` once from the ctor, then "Remesh Error ... front alignment failed"); `BSSNCtx::initialize()` calls `this->remesh()` on the labelled mesh exactly like EM4. Needs the same `m_uiIsLabelled` routing through `remesh_and_gridtransfer` (or convert to the SFC twin first as NLSM does).
- sync_blk_timestep (enuts.h) sub-timers under `DENDRO_ENUTS_PROF=1` (`DENDRO_ENUTS_PROF_CPU=1` = thread CPU time, for timing on an oversubscribed box) plus call/element/pair counters. Baseline nuts2 d9 np4 (CPU s, mean over ranks, 80 partial steps): sync 10.27 of partial_total 20.60; inside it same-time copy 1.15, coarse-future correction 0.47, fine-past 0 (never taken), unzip-into-block (padding fill) 8.26 (80 %), own-interior copy 0.34, post_stage_blk 0.01. lw3 np16 sfc baseline: sync 44.2 of 106.2; same-time 5.7, correction 2.2, padding fill 34.2, own 1.9. Inside the padding fill the parent-to-child interpolation of coarse neighbours is 4.6 of 8.26 (np4).
- `DENDRO_ENUTS_SYNC_FAST=1` (no FP reordering): (1) same-time neighbours are read from the stage vector instead of being memcpy'd into the DG temp (memcpy was a pure relay); (2) per-stage correction reuse: a per-element tag (pass, rk, branch, dt_c, dt_f, dt) says what m_uiDGTmp0 holds, so a neighbour corrected for one block is not recomputed for the next block of the same stage pass (62 % hit rate); (3) padding fill driven by a per-block plan of (src node, dst index) pairs in the original write order instead of redoing the double-precision coordinate arithmetic per node per stage; (4) direct-mapped cache of parent-to-child interpolations keyed by (elem, child, source tag), valid per stage pass (44 % hit); (5) row memcpy for the own-interior copy, persistent scratch instead of a 300 KB vector per call.
- Result (CPU s, mean over ranks): nuts2 np4 sync 10.27 -> 5.22, partial_total 20.60 -> 13.79; lw3 np16 sfc sync 44.2 -> 26.7, partial_total 106.2 -> 82.0 (box loaded, run-to-run CPU noise ~10 %). Remaining padding fill is gather-bound (neighbour element reads) plus the uncached p2c; next steps if worth it: p2c only on needed nodes (bit-identical, per-output FMA chains are replicable), run-length pairs.
- Gates: nuts2 knob off == ref; knob on == ref (all digits incl. l2); lw3 np16 sfc/lvl/lw/fp min/max/l2 md5 == work/lw3_s3on/<arm> : sfc 1e88c240cdd4, lvl a3da76f682bd, lw 36973a4a33c8, fp 8e8c0a89a2b0, all equal to lw3_s3on (v3 binary, knob on, PROF on). em4NUTS with CHECK+FAST: csv == non-knob, 0 mismatches.
