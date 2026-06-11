# Graph (fastpart) Repartitioning — Correctness Architecture

Single reference for how a graph-partitioned mesh is built and kept correct,
what each pass guarantees, which knobs are load-bearing vs diagnostic, and how
the result is validated. Replaces re-deriving context from the per-session
findings docs (`docs/findings_*.md`, `docs/handoff_*.md`); those remain as the
investigation record. Line numbers reference the tree as of 2026-06-10
(jj `partitioning-redist-batch` + this change).

---

## 1. The problem in one paragraph

The legacy pipeline assumes a Hilbert-SFC partition everywhere: local elements
are an SFC-contiguous, SFC-sorted range; ownership of a shared CG node is
derivable from SFC order; splitter search routes anything to its owner rank.
`fastpart` hands the mesh an *arbitrary* element→rank map, breaking all three.
Every structure downstream of the octree — E2E, E2N (CG/DG), nodal+element
scatter maps, block decomposition, unzip/zip plans — has a repartition-safe
rebuild path that must produce the *same physics* as the SFC build, ideally to
the bit.

## 2. Mesh-build pipeline (`Mesh::repartitionMeshGlobal`, src/mesh.cpp:18371)

Phases in execution order. "Gate" = env var or condition; defaults are what a
plain environment gets.

| # | Phase | Where | Gate / default |
|---|-------|-------|----------------|
| 0 | Octant connectivity map (e2e/edge/vertex neighbors, global ids) — partition-invariant input for everything below | 18402 | always |
| 1 | Partition decision (fastpart / NoPartition / OriginalPartition / Random) | 18425 | `m_partitionOption` |
| 1a | Block-atomic vote: keep canonical SFC blocks on one rank (tie → smallest rank) | 18504 | ON (`DENDRO_DISABLE_BLOCK_VOTE=1` reverts) |
| 2 | Keyset ghost fetch, rounds R1/R2/R3 (replaces old 7-round BFS) | 18623–18765 | always |
| 3 | global→local map, preferring LOCAL over ghost | 18786 | always |
| 4 | Rebuild E2E from oct connectivity | 18805 | always |
| 5 | Swap in new E2E / AllElements / OwnerMask / BlockInfo | 18834 | always |
| 6 | `buildE2NMap` at order 2 (hanging-node skeleton) | 18891 | always |
| 7 | `buildE2NWithSMRepartitioned` (src/mesh.cpp:3060): expand E2N to full order via cascade; orphan-cg cg2dg fix inside (ON, `DENDRO_DISABLE_ORPHAN_FIX=1` reverts) | 18914 | always |
| 8–10 | Mask validation / mask-ownership patch / TN-canonicalization — diagnostic A/B remnants | 18920–18967 | all OFF |
| 11 | **`auditAndRepairE2NCgPhysPos`** (src/mesh.cpp:6628) — see §3 | 18973 | ON (`DENDRO_E2N_AUDIT`) |
| 12 | Rebuild nodal scatter maps from repaired E2N (CG path R1/R2 interior, DG path R2/R3 boundary) | 18983 | always |
| 13–15 | Element scatter map, R1 ghost index, splitter nodes | 19236–19391 | always |
| 16 | Block creation: `buildBlocksFromCanonicalInfo` (5622, transported SFC blocks) with local-decomp fallback; `performBlocksSetupRepartitioned` (13712) → `findBlockNeighborsWithoutSFC` (13464, spatial hash, no SFC search); `buildUnzipCanonicalWriterTable` (6216); `buildZipPlan` (7274) | 19420–19473 | `do_block_creation` |

`m_uiAllElements` is **not SFC-sorted** after this (sorted by (target-rank,
id) then ghost rounds). Nothing downstream may binary-search it by SFC order;
point location must go through the spatial-hash/containment path
(`findContainingElementInAllNodes`, src/mesh.cpp:13648).

## 3. Correctness invariants and who establishes them

The bugs of 2026-04→05 were all violations of one of these four invariants.

**I1 — Every E2N_CG slot routes to the CG at its canonical phys position.**
Cascade (phase 7) gets this mostly right; partition-dependent ghost layers
break it at hanging faces / deep ghosts. `auditAndRepairE2NCgPhysPos` repairs
by phys-pos lookup (integer grid coords, no FP keys): canonical-CG election
via `better_canon` (6708 — LOCAL > ghost, has-writer > not, then owner-TN
(lev,x,y,z,sub) — a deterministic, partition-invariant total order), dangling
redirect, same-level bug-class patch. Up to 3 repair passes (7250).
The `unresolved` counter (same-level bug-class) MUST be zero — nonzero is
silent state corruption (now enforced, see §6). `dangling_unres` is normally
*large* (~2–4k/rank at maxDepth 7, np=4): ghost-fringe slots fed by ghost
exchange rather than zip; validated benign (EM4 long-haul is bit-identical
with these counts). The Morton-parent fallback for
DANGLING hanging slots was a 12× regression and is removed; the
`DENDRO_E2N_HANG_CANONICALIZE` branches are default-OFF diagnostics.

**I2 — At every multi-claim phys position, exactly one primary CG, and all
duplicates carry its value.** Established at zip time by `buildZipPlan`'s
interior-writer preference + duplicate-entry cleanup, maintained per RK
substage by the post-axpy `syncZipNonPrimary` (ets.h, default ON,
`DENDRO_DISABLE_POST_AXPY_SYNC=1` reverts) and optionally hardened by
`broadcastCgValuesByPhysPos` (mesh.tcc:831) — solver-controlled, see §4.

**I3 — redistributeVec delivers every dst CG, including orphans.** Orphan dst
CGs (not referenced by any received element) are filled by phys-pos match
against src local CGs, keyed by **cg2dg-derived canonical phys** — the
geometric-key variant was the root cause of the final 1.78× long-haul
residual (`DENDRO_ORPHAN_FILL_GEOM_KEY=1` reverts, A/B only). Fix B: no
`performGhostExchange` around redistribute — its `m_uiZipNonPrimaryToGhostCg`
mirror overwrites good LOCAL cgs with stale ghosts (was the 268× AMR spike).

**I4 — Blocks are partition-invariant.** Canonical SFC block info is
transported through the repartition (phase 1a vote + 16 reconstruction) so the
unzip writer set doesn't depend on the rank decomposition. Solvers must inject
src ownerMask + blockInfo into a fresh graph twin **before**
`repartitionMeshGlobal` (see §4) or the twin silently falls back to local
decomposition → partition-dependent hanging-face padding → ULP drift that
compounds into AMR-decision divergence (the step-46/52 class of bugs).
`flagBlockGhostDependancies` (13052) classifies UNZIP_INDEPENDENT vs DEPENDENT
without SFC assumptions (diagonal-edge `dir` offset + `nbrIsGhostDep` fixed
2026-06-09); `auditBlockTypeIndependence` (13221) NaN-poison-verifies it
(TEST 11).

## 4. The AMR sandwich and the per-solver machinery matrix

Remesh + intergrid transfer require SFC meshes (IGT builds its recv plan from
SFC splitter keys). **Never `interGridTransfer` into a graph mesh.** So every
remesh runs the sandwich: `buildSFCTwin` → `redistributeFlags` → remesh + IGT
on SFC → `buildGraphTwin` → `redistributeDVec` back. Implemented per solver in
`*_partitioning.h` (EM4 `em4_base_cpy`, NLSM `nlsm-gr_copy`, BSSN
`dendrogr_dfvk_repartitioning`).

**The SFC twin is built without blocks in EM4 and BSSN (2026-06-11) — NOT
NLSM.** NLSM's twin is not transient: nlsmCtx swaps `m_uiMesh` to the twin
and runs the wavelet refinement decision on it (unzip + `isReMeshUnzip`),
both in the init loop and `is_remesh()` — a no-block twin deadlocks NLSM
init in mismatched collectives. EM4/BSSN compute flags before g2s and only
pass the twin to ReMesh/IGT. Profiling
(`DENDRO_MESH_PROF=1`, rank-0 phase wall times in `createMesh` + the Mesh
ctor) showed the FDM twin ctor was ~90% block setup, and of that
`buildZipPlan` is 75–80% and `buildUnzipCanonicalWriterTable` ~18% — graph
machinery a transient twin never uses (it never unzips/zips; all its
consumers — redistributeVec dst, setMeshRefinementFlags, ReMesh-as-source,
IGT-as-source, ghost exchange, pos-bcast — read no block state, and the
zip-sync paths early-return on empty plans). EM4/BSSN `buildSFCTwin` now
passes `blockSetup=false` to `ot::createMesh` and then
`setBlockSetupFlag(true)` **only** so `ReMesh`'s successor mesh (which does
need blocks) inherits the flag. Measured: twin 281 → 38 ms/call (7.4×) on
EM4 np=4, EM4 graph
stress run (remesh every step) 27.2 → 22.8 s wall; bit-identity unchanged
(EM4 A/B 96/96 lines, TEST 10 maxErr 5.3e-15). Sort/dedup/balance in
`createMesh` measured negligible (~1.5 ms) at this scale — not worth a
skip-balance fast path yet.

**Zip plan + canon-writer table: eager by default; `DENDRO_LAZY_ZIPPLAN=1`
opt-in (2026-06-11).** Lazy construction (defer both to first
`unzip_scatter`/`zip` via `ensureZipPlanBuilt()`, mesh.h) lets the ReMesh
successor mesh in the graph sandwich skip them entirely — measured savings
~250 ms/remesh at EM4 scale and **~1.4 s/remesh at BSSN scale** (successor
ctor blk=1395 ms = zip-plan 1135 + canon-table 230). EM4 lazy was
bit-identical and NLSM A/B clean, BUT lazy mode exposed a **NaN in BSSN BBH
graph runs** (EV at the punctures goes NaN mid-step-4 with bit-identical BH
trajectories through step 3, before any sandwich; eager mode with identical
machinery is clean) — the signature of an allocation-timing-dependent
uninitialized read somewhere in the BSSN pipeline, not a plan-content
difference. Until that read is found (sanitizer hunt, see §8), lazy stays
opt-in. Two hardenings landed alongside, active in BOTH modes:
- `Mesh::unifyE2NCgAcrossTNInstances()` — the E2N_CG cross-instance repair
  formerly buried at the END of `buildZipPlan` (it MUTATES E2N_CG, so
  deferring it with the plan changed mesh semantics). Now hoisted: runs
  eagerly in both ctors and (idempotently) in `buildZipPlan` so the
  repartition path is unchanged. No-op on unique-TN (all ctor-built SFC)
  meshes.
- **Graph-safe `interpolateToCoords`** (include/daUtils.tcc): the SFC
  splitter+`SFC_treeSearch` point lookup silently fails on
  graph-partitioned meshes (unsorted `m_uiAllElements`, non-SFC splitters)
  — the BSSN BH-tracker "key not found" noise; in BBH runs the tracker was
  search-blind to one puncture for its first steps in EVERY config, with
  recovery by luck. A containment-rescue pass now scans the LOCAL element
  range for any point the SFC search failed to resolve (disjoint local
  ranges → at most one rank rescues each point; no-op when the search
  succeeds). BH punctures are now found deterministically from step 1.

BSSN BBH sandwich, np=4, steady state over 44 sandwiches (default
semantics, no env): **g2s 1653 → ~365 ms (4.5×)**, s2g ~256 ms,
remesh+IGT ~1760 ms (dominated by the successor's eager zip-plan ~1.4 s —
the opt-in lazy prize), **total 3460 → ~2385 ms (31%)**; mesh stable,
punctures inspiraling smoothly.

| Machinery | EM4 | NLSM | BSSN | Notes |
|---|---|---|---|---|
| buildGraphTwin: E2E_ONLY → FDM flip → repartition | yes | yes | yes | skips the FDM build repartition would discard |
| buildGraphTwin: ownerMask + blockInfo injection (I4) | yes | **yes (ported 2026-06-10)** | yes | absence = partition-dependent blocks |
| redistributeDVec: Fix B (no ghost exchange) | yes | yes | yes | I3 |
| redistributeDVec: DOF-batched `redistributeVec` call | yes | yes | yes | 3.4× on redistribute, bit-identical |
| redistributeDVec: post-redistribute pos-bcast | yes | **yes (ported 2026-06-10)** | yes | `DENDRO_DISABLE_REDIST_BCAST=1` A/B-reverts |
| consensus pos-bcast active on graph mesh | **no** (explicit opt-out) | yes (default) | yes (default) | see below |
| ets.h post-axpy `syncZipNonPrimary` (I2) | on | on | on* | global default; *BSSN uses a custom `remesh_and_gridtransfer`; its evolve path should be confirmed to pass through ets.h |

**Pos-bcast control (changed 2026-06-10):** `broadcastCgValuesByPhysPos` used
to no-op unless the ambient env var `DENDRO_FORCE_POS_BCAST=1` was set — i.e.
the validated NLSM/BSSN configs depended on environment setup that a plain
production run wouldn't have. It is now **on by default for every
graph-repartitioned mesh** (`repartitionMeshGlobal` sets
`m_uiPosBcastEnabled = true`); `DENDRO_FORCE_POS_BCAST=1/0` remains as an A/B
override in both directions. EM4 explicitly opts out via
`setPosBcastEnabled(false)` after its twin builds: the validated EM4-minimal
config ({orphan-fill fix + E2N audit} only) is bit-identical to SFC at
t_end=30 AND 1.7× faster than SFC, so the bcast buys nothing there.
**Minimal is EM4-specific** — NLSM (maxdepth 9) demonstrably needs the full
set; new solvers get the safe default with no wiring.

## 5. Env-knob registry

Load-bearing (ON by default, env *reverts* — keep):
`DENDRO_E2N_AUDIT` (I1 audit; never run graph without it),
`DENDRO_DISABLE_ORPHAN_FIX`, `DENDRO_DISABLE_BLOCK_VOTE`,
`DENDRO_USE_LEGACY_BLOCKS`, `DENDRO_DISABLE_POST_AXPY_SYNC`,
`DENDRO_DISABLE_REDIST_BCAST`, `DENDRO_ORPHAN_FILL_GEOM_KEY`,
`DENDRO_DISABLE_ZNP_MIRROR`.

Validation/diagnostic (default OFF — keep):
`DENDRO_E2N_AUDIT_STRICT` (abort on unresolved audit slots — set in CI /
validation runs), `DENDRO_FORCE_POS_BCAST` (A/B override of the solver flag),
`DENDRO_VALIDATE_MASK`, `DENDRO_E2N_AUDIT_DBG`, `DENDRO_EVOLVE_TRACE`,
`DENDRO_REDIST_PERDOF`, `DENDRO_S2G_SKIP_REPART` (**EM4-only A/B trap —
silently wrong on NLSM/BSSN; use `*_PARTITIONING_METHOD=0` as baseline
instead**).

Deletion candidates (refuted/redundant experiments, default OFF, kept only as
documented history — safe to remove in a cleanup change):
`DENDRO_ENABLE_PASS_A`, `DENDRO_ENABLE_PASS_DE` (+`DENDRO_DISABLE_PASS_D_RESCUE`),
`DENDRO_USE_MASK_OWNERSHIP`, `DENDRO_E2N_HANG_CANONICALIZE`,
`DENDRO_E2N_CANON_TN`, `DENDRO_USE_LEGACY_PLAN_BUILD`,
`DENDRO_USE_CASCADE_RULE`, `DENDRO_ZIPPLAN_USE_SMALLEST_TN`,
`DENDRO_BCAST_TIEBREAK_SUBN`, plus the `DENDRO_E2N_POSTZIP_*` /
`DENDRO_*_DBG` dump probes.

Build-time: `DENDRO_ENABLE_OPENMP_PARTITIONING` → `DENDRO_OMP_PART`
(audit/block-decomp hot loops; infra present, **not yet validated** at
`OMP_NUM_THREADS>1`).

## 6. Hard checks (added 2026-06-10)

- The E2N audit prints a per-rank `WARNING` whenever `unresolved` (same-level
  bug-class slots with no local cg at the expected phys) is nonzero;
  `DENDRO_E2N_AUDIT_STRICT=1` aborts instead. It also warns if the 3-pass
  repair cap was hit with work remaining. A graph run printing this warning
  is producing corrupted state — treat as a build-breaking bug, not noise.
  (`dangling_unres` deliberately excluded: routinely thousands per rank,
  validated benign — see §3.)
- `redistributeVec` counts duplicate-key collisions in its three identity maps
  (TN→rank, TN→local, canonical-phys→src CG) and warns per rank if any occur.
  These maps assume globally unique TNs / unique canonical phys per CG;
  a collision means a violated mesh invariant that was previously masked by
  silent first-wins.

## 7. Validation matrix

| Check | Where | Expected |
|---|---|---|
| testPartitioning (np=4, `mpirun -np 4 build/testPartitioning 3 7 1e-3 0.1 4 50 4`, `mkdir -p vtu` first) | dendrolib | TEST 2b `maxErr=0`, 8be `maxErr=0`, 11a/b/c PASS |
| TEST 10 AMR-cycle sandwich | dendrolib | 4 cycles, ~5e-15 per cycle |
| EM4 graph vs SFC, t_end=30 incl. AMR | em4_base_cpy | bit-identical (minimal machinery) |
| NLSM maxdepth 7 graph vs SFC | nlsm-gr_copy | byte-identical, **no env vars set** |
| NLSM maxdepth 9 long-haul | nlsm-gr_copy | U_CHI bit-identical samples; U_PHI transient re-converges |
| BSSN BBH WAMR / BH_LOC | dendrogr | bit-identical ~10 steps then round-off; BH track bit-identical over 30 remeshes |

Known-benign reds (NOT regressions): TEST 8bc (post-remesh zip roundtrip,
~hundreds of FP-order mismatches from partition-dependent ghost layout in the
roundtrip itself — real zip bugs show up as thousands, and TEST 10 covers the
solver path), "TEST 2" in some configs. Graph runs require
`SOLVER_/NLSM_/BSSN_PARTITIONING_METHOD=3` in the param file — without it the
run is silently SFC and proves nothing.

## 8. Open items / follow-ups

1. **Hunt the BSSN lazy-mode NaN** (unblocks ~1.4 s/remesh): EV at the
   punctures goes NaN at step 4 of a BBH graph run when `DENDRO_LAZY_ZIPPLAN=1`,
   after three bit-identical steps; eager is clean. Allocation timing is the
   only difference → almost certainly an uninitialized read whose content
   shifts with heap layout. Reproducer: bssn_smoke `bhloc.toml`, np=4,
   `DENDRO_LAZY_ZIPPLAN=1`, watch "Black Hole 1 new position" go NaN by
   sample 4. Hunt with ASan/MSan or valgrind on a small config.
1b. ~~BSSN BH-tracker ~55k "rank N key not found"~~ **FIXED 2026-06-11**
   (containment rescue in `interpolateToCoords`, see §4). Residual "not
   found" prints from the SFC-search stage are now informational only —
   every point gets rescued; consider silencing the print.
2. **Test gaps:** no dendrolib dof>1 redistribute test (solver-level only);
   suite exercised mainly at np=4, maxDepth=7; no empty-rank (inactive-rank)
   test; no standalone `redistributeFlags` test; OMP path unvalidated;
   confirm BSSN's evolve goes through ets.h post-axpy sync.
3. **NLSM maxdepth 9 re-validation** after the 2026-06-10 ports, before NLSM
   graph production use; one BSSN re-run with the explicit-flag path.
4. **Perf roadmap** lives in `docs/handoff_2026-06-10_partitioning_perf.md`
   §3–4 (#4 buildSFCTwin next; #2 Allgatherv removal blocked on a
   cluster-scale benchmark; long-term: partition-invariant IGT to delete the
   sandwich).
5. **Env-knob cleanup**: delete the §5 deletion-candidate gates and their
   dead branches once the team agrees the investigation record in docs/ is
   sufficient.
