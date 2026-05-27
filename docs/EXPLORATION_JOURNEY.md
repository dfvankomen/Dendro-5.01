# Graph partitioning → bit-identity with SFC: the journey

> **Status (2026-05-27):** Fixed. Graph mode is bit-identical to SFC
> reference at t_end=30 (230 steps, ~23 AMR remeshes) across all 8 EM4
> variables. See [The fix](#the-fix).

This is the long-form post-mortem of an investigation that ran from
2026-04-23 through 2026-05-27. It documents what was tried, what stuck,
what was a dead end, and where the answer was actually hiding.

The individual finding docs in `docs/archive/` capture each step in the
moment; this doc is the retrospective.

---

## The bug, in one paragraph

Graph-partitioning EM4 ran through AMR remeshes, but the simulation
state slowly diverged from the SFC reference. After landing many
intermediate fixes, the residual stabilized at ~1.78× U_E2 noise at
step 92 (and grew to ~6× by step 200) — the "1.78×" residual that this
investigation chased for weeks. The actual root cause was a single
keying bug in `Mesh::redistributeVec`'s orphan-fill step: it keyed
`srcPosToLocalCG` by element-node *geometric* phys, but at AMR
refinement boundaries hanging-face nodes have E2N pointing to a coarse
parent's cg whose canonical (cg2dg-derived) phys differs from the
element-node geometric phys. The geometric keying mapped the wrong
neighbor's value into dst orphan cgs at every AMR remesh, and the
post-redist bcast then broadcast that wrong value to all ranks.

## The fix

`include/mesh.tcc` ~line 6577: changed `srcPosToLocalCG` construction
to walk LOCAL src cgs and key by **cg2dg-derived (canonical) phys**
instead of element-node geometric phys.

```cpp
// before: walks src LOCAL elements, keys by element-node geom
for (e in src LOCAL elements)
    for (n in 0..npe)
        cg = E2N_CG[e*npe + n];
        if (cg LOCAL)
            srcPosToLocalCG.emplace(geom_phys(e, n), cg);

// after: walks src LOCAL cgs, keys by canonical phys
for (cg in src LOCAL cgs)
    dg = cg2dg[cg];
    if (dg valid)
        srcPosToLocalCG.emplace(cg2dg_phys(dg), cg);
```

Validation (t_end=30, 230 steps, ~23 AMR remeshes, 4 ranks):

| variable | fix vs skip ratio (step 230) | legacy vs skip (step 200) |
|---|---|---|
| U_E0 | 1.0000 | 1.51× |
| U_E1 | 1.0000 | (peer) |
| U_E2 | 1.0000 | **6.04×** |
| U_B0 | 1.0000 | 2.36× |
| U_B1 | 1.0000 | (peer) |
| U_B2 | 1.0000 | 1.00× |
| C_DIVE | 1.0000 | **21.98×** |
| C_DIVB | 1.0000 | **16.29×** |

No-AMR mode: fix is bit-identical to legacy and skip (orphan-fill
code path runs but never triggers without AMR remesh sandwiches → no
regression risk).

`testPartitioning`: 10/11 pass; Test 2 (ghost exchange) is a
pre-existing failure unrelated to this change.

See `findings_2026-05-26_orphan_fill_geom_key_bug.md` for the full
bisection trace at remesh 14 / phys (336, 264, 288).

---

## Timeline

The bug surface kept shifting as each fix removed one cause and
revealed the next. Brief milestones:

### Phase 0: setup (April)

- `partitioning_handoff_2026-04-23.md`, `cascade_rule_match_2026-04-27.md`,
  `mask_ownership_2026-04-28.md`, `canonical_blocks_2026-04-28.md` —
  initial framework for graph mode mesh construction, cascade rules,
  block-atomic vote, canonical blocks. By end of April, single-step
  testPartitioning was bit-perfect (8bo/8bp maxDiff=0), but EM4
  long-haul diverged.

### Phase 1: the first wave of fixes (May 5–11)

- `interior_writer_fix_2026-05-05.md` — buildZipPlan prefers writers
  whose buffer position is in the block interior (vs padding). Real
  graph-mesh corner zip bug fixed; step-1 unchanged.
- `em4_residual_findings_2026-05-05.md` — PassD's mirror loop was
  overwriting cascade-primary cgs with stale IC ghosts.
- `findings_2026-05-11.md` — extended cross-rank cg allgather to
  include GHOST cgs ("ghost-inclusive syncZipNonPrimary"); EM4 peak
  improved 2.18e-2 → 6.71e-3.

### Phase 2: cascade routing & E2N audit (May 11–14)

- `findings_2026-05-12_dangling_e2n.md` — DANGLING-CG audit redirects
  E2N_CG entries pointing to dangling cgs toward canonical cgs at the
  OWNER's phys.
- `findings_2026-05-12.md` ("E2N audit lands") — phys-pos audit
  catches/patches specific known bugs; some 10% step-130 improvement
  but long-haul peak unchanged.
- `findings_2026-05-14d.md` — `Mesh::auditAndRepairE2NCgPhysPos`
  hang-canonicalize branch (oLev<eLev and oLev>eLev): redirect E2N_CG
  to canonical Morton-tree-parent grid point. Eliminated +X corner
  drift.
- `findings_2026-05-12_post_axpy_sync.md` — post-axpy syncZipNonPrimary
  in `ETS::evolve` + intra-rank duplicate map. All 37 cross-rank
  duplicate phys positions match SFC bit-perfect at step 0.

### Phase 3: AMR-specific residual (May 12–15)

- `findings_2026-05-12_amr_ue2.md` — AMR long-haul completes 230 steps
  but U_E2 has 50× noise vs SFC. Physical vars match well.
- `findings_2026-05-15_spike_fix.md` — **Fix B:** skip
  `srcMesh->performGhostExchange` in `em4_partitioning.h::redistributeDVec`.
  Drops EM4 graph long-haul step 92 U_E2 L2 from 1.53e-2 to 1.02e-4
  (~150× improvement). Root cause was the ZNP mirror loop overwriting
  LOCAL non-primary cgs from their paired GHOST.

### Phase 4: chasing the 1.78× residual (May 15–21)

After Fix B, the residual stabilized at 1.78× U_E2 at step 92. This
phase chased it through many bisections, refuting one hypothesis at
a time:

- `findings_2026-05-15_ulp_drift.md` — `DENDRO_S2G_SKIP_REPART` makes
  graph mode bit-identical to SFC, so the residual is IN s2g.
- `findings_2026-05-15_audit_hangfix.md` — Morton-parent fallback for
  hanging slots in audit; eliminates step-31 routing mismatches, but
  doesn't move the 1.78× ratio.
- `findings_2026-05-15_orphan_bcast.md` — pre-sandwich bcast eliminates
  orphan-cg diffs at early-step AMR boundaries. Residual unchanged.
- `findings_2026-05-18_dup_local_priority.md` — diagnosed: graph creates
  2 LOCAL cgs at some corners; `broadcastCgValuesByPhysPos` picks the
  stale one. Hypothesis posted as root cause.
- `findings_2026-05-19_option_c_noop.md` — implemented structural
  tie-break for the bcast (Option C). **NO-OP.** Bcast is downstream
  of post-axpy sync which already converges multi-LOCAL phys correctly.
  Refuted the "wrong canonical pick" hypothesis.
- `findings_2026-05-19_block_decomp_probe.md` — block buffer is
  bit-identical at smoking-gun phys. Scatter map + block decomp
  exonerated.
- `findings_2026-05-19_hangfix_regression.md` — Morton-parent
  DANGLING fallback was a strict 12× regression. (Was later neutralized
  by the orphan-fill fix anyway.)
- `findings_2026-05-19_session_summary.md` — surfaces narrowed to
  cascade-interpolation source picks at hanging faces.
- `findings_2026-05-20_pz_neighbor_flip.md` — diagnosed: +Z neighbor of
  TN(72,72,88) flips owner between graph and skip. Mechanism for
  +X-face hotspot is FD stencil overlap into +Z padding.
- `findings_2026-05-21_option_a_no_op.md` — post-RHS stVec bcast in
  ETS::evolve. **NO-OP.** CG state already canonical at every checkpoint.
- `findings_2026-05-21_option_b_already_done.md` — R2 ghost layer
  expansion. Already done via R1+R2+R3 in repartitionMeshGlobal.
- `findings_2026-05-21_znp_no_op.md` — `DENDRO_DISABLE_ZNP_MIRROR`
  gate. **NO-OP** long-haul. Eliminates step-1 diff but residual
  unchanged.
- `findings_2026-05-21_cascade_refactor_scope.md` — designed a
  cross-rank cascade fetch refactor as the next attack.

### Phase 5: the actual fix (May 26)

- `findings_2026-05-26_phase0_canon_diff.md` — Phase 0 probe: 13%
  of multi-writer slots elect different canonical writers between
  graph and skip. Hypothesis: per-rank candidate set differs.
- `findings_2026-05-26_phase1_noop.md` — Phase 1 (global canon
  election): allgather all TNs, augment with global "extras". **NO-OP**:
  remote_winners=0 because R3 ghost already covers every winning TN.
- `findings_2026-05-26_orphan_fill_geom_key_bug.md` — bisected to
  remesh 14, traced to specific dst orphan cg=10785 receiving the
  wrong value from `srcPosToLocalCG.find(target_phys)`. The map entry
  came from a hanging-face element whose E2N points to a coarse
  parent's cg at a DIFFERENT phys. Fixed by re-keying with
  cg2dg-derived phys.

---

## What was tried — structured table

### Stuck (active in code)

| Approach | Where | Effect |
|---|---|---|
| Interior-writer preference in buildZipPlan | `src/mesh.cpp` (buildZipPlan) | Fixes corner zip; foundation for everything else |
| PassD mirror loop cleanup | `src/mesh.cpp` (buildZipPlan) | Eliminates PassD mirror overwriting cascade primaries |
| Ghost-inclusive syncZipNonPrimary | `include/mesh.tcc` (syncZipNonPrimary) | Cross-rank dup-LOCAL cgs converge bit-perfect |
| DANGLING-CG audit redirect | `src/mesh.cpp` (auditAndRepairE2NCgPhysPos) | E2N_CG misroutes to dangling cgs get patched |
| Hang-canonicalize audit (oLev<eLev & >eLev) | `src/mesh.cpp` (auditAndRepairE2NCgPhysPos) | Eliminates +X corner drift; routes hanging to Morton parent |
| Post-axpy syncZipNonPrimary | `ODE/include/ets.h` (evolve) | All cross-rank duplicate phys converge at end-of-step |
| Intra-rank duplicate cg map | `src/mesh.cpp` (buildZipPlan) + `include/mesh.tcc` (syncZipNonPrimary) | Local copies kept consistent without MPI |
| Skip src ghost exchange in s2g redist (Fix B) | `em4_partitioning.h::redistributeDVec` | ~150× drop in step-92 U_E2 |
| Canonical TN-sorted unzip iteration (mode 2) | `include/mesh.tcc` (unzip_scatter) | Partition-invariant element iteration order |
| Canonical-writer table per multi-writer slot | `src/mesh.cpp` (buildUnzipCanonicalWriterTable) + `include/mesh.tcc` (unzip_scatter) | Hard-pick winner per slot |
| Pre-sandwich bcast in remesh_and_gridtransfer | `em4_partitioning.h` (under DENDRO_FORCE_POS_BCAST) | Zeros orphan-cg diffs at AMR boundaries |
| Post-redist bcast in redistributeDVec | `em4_partitioning.h::redistributeDVec` | Consensus canonicalization post-s2g |
| **Orphan-fill cg2dg-key fix** | `include/mesh.tcc::redistributeVec` step 5 | **The fix that closed the 1.78× residual** |

### Refuted no-ops (removed 2026-05-27)

| Approach | Env knob | Why no-op |
|---|---|---|
| Per-substage bcast | `DENDRO_PER_STAGE_BCAST` | Doesn't close 1-ULP step-9 residual; equalizing intra-step state doesn't fix post-axpy drift |
| Post-RHS stVec bcast | `DENDRO_POST_RHS_BCAST` | CG state already canonical at every checkpoint |
| Disable ZNP mirror | `DENDRO_DISABLE_ZNP_MIRROR` | Eliminates step-1 diff but bit-identical long-haul to baseline; bcast canonicalizes from PRIMARY, not non-primary |
| Bcast structural tie-break (Option C) | `DENDRO_BCAST_TIEBREAK_SUBN` | Bit-identical to default rank-then-cg tie-break; bcast downstream of post-axpy sync which already converges |
| Global canon election (Phase 1) | `DENDRO_GLOBAL_CANON_ELECTION` | R3 ghost layer already covers every winning TN; remote_winners=0 |
| Unzip iter order alternatives | `DENDRO_UNZIP_CANONICAL_ITER`={1,3,4} | Mode 2 (coarser-last) only working choice; others tested and refuted |
| Canon-writer table disable | `DENDRO_UNZIP_CANON_WRITER_OFF` | Removing it caused regressions; always-on now |

### Refuted hypotheses (no code change needed)

| Hypothesis | Refuted by | Note |
|---|---|---|
| 1.78× was unzip writer-order | `findings_2026-05-14_writer_set_refuted.md` (project memory) | Writer SET bit-identical; dgWVec differs |
| 1.78× was IC noise at far-field | `findings_2026-05-14e.md` | Superseded by E2N hanging-face routing fix |
| 1.78× was duplicate-LOCAL bcast pick | `findings_2026-05-19_option_c_noop.md` | Bcast downstream of sync; tie-break doesn't matter |
| 1.78× was block-buffer drift | `findings_2026-05-19_block_decomp_probe.md` | Block buffer bit-identical at smoking gun |
| 1.78× was DANGLING Morton-parent fallback | `findings_2026-05-19_hangfix_regression.md` | That was a regression; removed |
| 1.78× was cascade source-pick | `findings_2026-05-21_*.md` | Designed refactor; bug was upstream of cascade entirely |

### Kept as A/B knobs (default on)

| Knob | Purpose |
|---|---|
| `DENDRO_FORCE_POS_BCAST` | Enables pre-sandwich + post-redist position bcast (required for graph mode bit-identity) |
| `DENDRO_S2G_SKIP_REPART` | Skips s2g rebuild; reverts to SFC after first AMR cycle. A/B knob, not a fix |
| `DENDRO_ORPHAN_FILL_GEOM_KEY` | Reverts the orphan-fill fix for A/B comparison |
| `DENDRO_E2N_AUDIT` | Enables the E2N audit/repair pass (default ON) |
| `DENDRO_DISABLE_*` family | Per-pass A/B knobs (Pass A, D, DE, D-rescue, etc.) for surgical bisection |
| `DENDRO_USE_LEGACY_*` family | Fallback to pre-rework code paths (blocks, plan_build, zip, mask_ownership) |

### Active debug-probe infrastructure

(Env-gated at runtime; **compiled out entirely** by default via CMake.
To enable: `cmake -DDENDRO_ENABLE_DEBUG_PROBES=ON .` then rebuild.
With probes ON, set the corresponding env var to activate per-call.)

| Probe family | Env prefix | Purpose |
|---|---|---|
| Redistribute send/recv probe | `EM4_REDIST_PROBE_*` | Logs src element-node writes + dst cg writes at a target phys |
| Remesh-sandwich tag dump | `EM4_REMESH_DUMP_*` | Dumps EV state at tag a/b/c/d boundaries within a remesh |
| Block-buffer dump | `DENDRO_BBUF_DUMP_*`, `EM4_BLOCK_DUMP*` | Dumps unzip block-buffer state |
| CG trace | `EM4_CG_TRACE_*` | Per-step CG state at target phys/bbox/tag |
| Canon-writer dump | `DENDRO_CANON_DUMP_*` | Dumps multi-writer slot canonical writer table |
| Unzip probe | `DENDRO_UNZIP_PROBE*` | Logs unzip writer/value at target mbox |
| Bcast claim probe | `EM4_BCAST_PROBE_*` | Dumps all advertised claims at target phys |
| Ghost-recv probe | `DENDRO_GHOST_RECV_PROBE*` | Logs ghost exchange recv at target TN/box |
| RHS dump | `EM4_DEBUG_RHS_*` | Dumps RHS in/out vectors |
| Element-set / hanging dumps | `EM4_ELEMSET_DUMP*`, `EM4_HANG_DUMP*` | Dumps mesh element/hanging structure |

---

## What made the bisection possible

Three things were load-bearing:

1. **The skip mode A/B knob (`DENDRO_S2G_SKIP_REPART`).** Bit-identical
   reference behaviour any time we wanted it. Without this, every
   investigation would have been graph-vs-graph noise.
2. **`EM4_REMESH_DUMP_*` tags a/b/c/d.** Localized divergence to a
   specific sandwich step (s2g) by checking each boundary in turn.
3. **`EM4_REDIST_PROBE_*` send/recv probe.** Once we knew s2g was the
   problem, this revealed that src was sending the wrong value
   (specifically, geometric-target writes resolving to cgs at
   different cg2dg-phys).

All three are env-gated and remain in the code.

---

## What I'd do differently

- **Get to the per-cg level sooner.** Many weeks were spent on L2-level
  hypotheses ("which cg group is contributing the noise?"). The
  bisection that actually worked was: pick ONE phys position, trace
  its value through one specific remesh sandwich, identify the wrong
  write at the writer level. That should have been the first move
  after Fix B.
- **Trust the A/B revert.** Many fixes were claimed "lands" or
  "neutral" without a bit-level revert test. The 2026-05-19
  hangfix-regression finding showed how that misled the investigation.
  An env-gated A/B revert at every fix would have caught the
  regression sooner.
- **`testPartitioning` is necessary, not sufficient.** Test 2b
  (`redistributeVec correct`) passed at maxErr=0 throughout — because
  the harness checks single-call redistribute, not the multi-remesh
  AMR sandwich path where orphan fill matters. The test suite needs an
  AMR-cycle test that exercises orphan fill at hanging-face boundaries.

---

## References

All individual finding docs are in `docs/archive/`. The chronological
list above gives the primary citation for each phase; cross-references
in those docs cover the rest.
