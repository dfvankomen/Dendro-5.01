# Repartitioning sweep plan — 2026-09-23

Written after a three-way static audit (fork plumbing, fastpart library, `origin/experimental`).
Executed locally only (dev box; no BYU/CHPC without explicit permission). Each sub-step is a jj commit.
Deterministic gates only (edge cut, per-sub-step eff, bit/byte-identity); the box cannot time.

## Findings that drive the plan

1. **fastpart's core is cut-oblivious.** `UpdateDiffusionLabels` moves a vertex to the neighbour
   part with the lowest *rate*, never to the part with the most neighbours; `vwgt` only flips a
   binary 0.1/0.8 rate; seeds are unweighted Hilbert midpoints; 20 fixed one-hop iterations.
   Mechanism for the 1.5–1.9× face cut: hop-count Voronoi on face6 is an L1 metric whose bisectors
   are staircases (≤√3× the area of an axis-aligned octant boundary). Fix = SFC-initialised labels +
   gain-aware balance-constrained boundary refinement + multi-constraint (per-level) weights.
   That is a fastpart-side research item, to hand to its author — not a dendrolib production path.
2. **The fork's real value is arbitrary labelling.** The pipeline consumes one `parts[]` (target
   rank per local source element in SFC order); everything downstream is label-agnostic. Only the
   fastpart branch produces `parts[]` and runs the block vote; Original/Random bypass the vote.
3. **Hidden O(N_global) collectives on default paths:** `broadcastCgValuesByPhysPos` every RK step
   (ON for NLSM/BSSN; 7-array Allgatherv of all claims, recomputed although mesh-invariant);
   `redistributeVec` step 1 (×2 per sandwich) + `redistributeFlags` (Allgatherv of all TreeNodes);
   `buildOctantConnectivityMap` gather; `buildZipPlan` gathers while `DENDRO_ZIPPLAN_DIST` is off.
4. **Latent hang:** `mesh.h:~4086` wraps `MPI_Allgather/Allgatherv` in `if (numMissing > 0)`.
5. **`__phase` does an unconditional `MPI_Barrier` + cout ~16×/repartition.**
6. **~3.5–4.5k of ~12k added lines are scaffolding** (Pass A/D/E, owner-mask pipeline, canonical-TN,
   legacy blocks/plan/zip, `createGlobalE2EMap`, `ReMeshRepartitioned`, 41 probe sites).
7. **experimental** requires contiguous SFC ranges per rank (`buildE2EMap` splitter search), so
   levelwise cannot enter via `set_wpart_function`. Conflict surface for a port: mesh.cpp trivial,
   mesh.tcc semantic (zip/unzip_scatter/ghost-pack), ets.h evolve 19 hunks, CMake moderate.
   A multi-constraint *contiguous* splitter (200–400 lines) may capture most of levelwise's benefit
   without the fork — must be scored offline first.
8. ENUTS bit-identity on a non-SFC labelling is unproven (enuts.h has no post-axpy sync/bcast and
   uses DG element-ghost exchange, not the plan zip).

## Phase 0 — do now (deterministic, local, low risk)

- **P0.1 Wire `sfc_levelwise` (`PartitioningOptions::SFCLevelwise`, method id 4).**
  Refactor `repartitionMeshGlobal` so every branch fills `parts[]`, then one shared block vote and
  one shared label→`my_partition` exchange (Original becomes `parts[i]=rank` → free neutrality gate).
  Levelwise labels: per-level local counts → `MPI_Allreduce` (n[L]) + `MPI_Exscan` (off[L]);
  `parts[i] = min(np-1, (off[L]+running[L]) * np / n[L])`. Equality-check vs
  `edgeCutCompare.cpp` labelling. Gate: testPartitioning TEST 2/2b/8/8b/10/11 + COVERAGE at
  np=2/4 with method 4; Original-via-vote path bit-identical to before.
- **P0.2 Offline scoring of the contiguous multi-constraint splitter** in `edgeCutCompare`
  (`sfc_mc_contig`: choose np−1 Hilbert cuts minimising max per-level imbalance). Report per-sub-step
  eff + cut vs `sfc_weighted` / `sfc_levelwise` / fastpart at np=2..32. This decides the
  experimental-branch strategy (Phase 2).
- **P0.3 Hardening:** make the connectivity-map gather unconditional (collective-safe); make
  `__phase` barriers opt-in (`DENDRO_REPART_PHASE_PROF=1`); keep numerics untouched.
- **P0.4 Save findings to the vault** (Repartitioning log + concept updates).

## Phase 1 — this session if time (byte-identical by construction, gated)

- **P1.1 Cache `broadcastCgValuesByPhysPos` plan per mesh** (claims/winners/send-recv lists are
  mesh-invariant; per step only the Alltoallv remains; batch across dofs). Gate: NLSM short-run
  bit-identity (17-digit samples) graph vs graph-before; EM4 unaffected (opts out).
- **P1.2 Label-routed sandwich transport:** s2g keeps post-vote `parts[]`; g2s routes by the SFC
  twin's splitters (binary search on 2P TreeNodes); flags ride the same exchange. Removes the three
  O(N) gathers. Gate: EM4 long-haul CSV byte-identity + testPartitioning.
- **P1.3 ENUTS-on-levelwise gate:** run `nlsmNUTS`/`em4NUTS` GTS mode on method 4 vs SFC on a fixed
  mesh; establish or refute bit-identity before any LTS claim.

## Phase 2 — later / needs a decision

- Scaffolding deletion campaign, one arm per jj commit, each behind a byte-identity A/B.
- Collapse the three solver sandwich copies into one dendrolib helper.
- experimental: if P0.2 says the contiguous splitter is within ~5% of levelwise eff, implement it
  there (200–400 lines, no fork port). Otherwise cherry-pick the minimal labelling core (option b,
  ~1.5–2 weeks): plan zip must replace the block-parallel zip on labelled meshes; unzip gating must
  be re-proved unnecessary; probes dropped (mutable statics race under OpenMP).
- fastpart proposal to the author (ranked): initial-labels input; gain-aware refinement; first-class
  weights + uint64 accumulate; convergence criterion; drop the O(P) Allreduce per iteration; hygiene
  (sorted CSR, `local_n==0`, `NO_LABEL`, edge-weighted early-return deadlock, verbose=0).

## Commit conventions
`feat(scope):` / `fix(scope):` / `test(scope):` / `refactor(scope):` subject only; no body unless
essential; no footers or attribution. `jj new -m` before each sub-step.

## 2026-09-24 addendum — audits of the rebuild rounds and the sandwich (vault: "Communication rounds of the labelled rebuild and the sandwich")

Landed today: ENUTS on labelled meshes (Block::firstElementID/numElements/iterator; init remesh routed
through ctx; solver overrides clear the ETS-synced flag), NLSM LTS driver fixes (worktree
`nlsm-gr_lts`, branch `lts-graph-fix`), transport trims (e2n_dg stub, fastpart verbose off, dead zip
gather skipped, stage-3 collective unconditional, zero-element-rank abort).

Next, in order (all byte-identical unless noted):
1. C-lite: flags ride the g2s `redistributeVec` (kill `redistributeFlags`' O(N_global) allgather) — <1 d.
2. Cache the `broadcastCgValuesByPhysPos` plan per mesh + one batched Alltoallv per step; count the (β)
   residual — ~150 lines.
3. Batch IGT's per-variable Alltoallv_sparse into one exchange.
4. Push partition (trank from labels) and forward-push R1/R2/R3 closure (14 → 8 rounds).
5. isDG bit packing; numMissing fold into the counts Allgather; drop the splitters Allgather.
6. Print `unifyE2NCgAcrossTNInstances`'s discarded counter; if 0 on labelled meshes, one-Alltoallv shrink.
7. Design A (labelled ReMesh + keyed IGT) behind an env gate with a shadow comparison — not byte-identical.
Open correctness item: NLSM global-RK graph ≠ SFC under the new defaults (bisect in progress); ENUTS on
labelled meshes evolves U_PHI ≡ 0 (ENUTS-specific; global RK keeps it).
