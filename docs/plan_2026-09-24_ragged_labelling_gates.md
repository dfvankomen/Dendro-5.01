# Plan — ragged labellings: close the blind spot, then price the win (2026-09-24, evening)

Follows `plan_2026-09-23_repartition_sweep.md`. Local only; every step gated deterministically; each
sub-step a jj commit.

## Why this plan exists

Every bit-identity gate the fork has ever passed ran on a symmetric mesh or an SFC-identical
labelling. The first genuinely ragged labelling through ENUTS (plain fastpart, block vote moving 234 of
2174 elements) exposed a fork defect that had been latent since the labelled path existed: the element
ghost exchange was a lossy push (fixed today, `fix(repart): pull the 26-closure…`). The same blind spot
can hide anything else that only a ragged partition exercises. One such candidate is already visible:
the ragged arm's last init remesh yields 9136 octants where SFC yields 8856.

## Step 0 — explain 9136 vs 8856 (in progress)

Two agents: a read-only map of every partition-dependent decision in the last-remesh path, and an
implementer adding `DENDRO_REMESH_FLAG_DUMP=<dir>` (per-rank `x y z lev flag` per local element, before
and after flag transport) plus a TreeNode-keyed diff. Outcomes and what each implies:

- flags identical, meshes differ → the decision moved inside `ReMesh` on the twin (sibling/coarsen
  rule vs what the labelled mesh delivered). Fix on the transport or twin side; must be byte-identical
  for SFC-identical labellings.
- flags differ at elements adjacent to the ragged partition boundary → wavelet reads partition-dependent
  padding (plan-unzip / hanging-face ULP class). Decide: accept as the known residual class, or make the
  wavelet read a partition-invariant quantity.
- flags differ elsewhere → a real defect in wavelet-on-index-list-blocks; fix and gate.

Result goes into `docs/` + vault (Open Questions closes, Concepts note if durable).

**Outcome (same evening):** none of the three. The differing octree is the step-0 wavelet, the first
one that sees data written on a labelled mesh; 24 elements flip COARSE→SPLIT with wMax orders of
magnitude above tol because 31 local CG slots that no element on their rank references were left at 0
by `createVector`. Fixed with `Mesh::fillUnreferencedLocalNodes` (cg2dg canonical position) in both
overloads; the ragged arm now follows the SFC mesh sequence (…4096 → 8856) and the IC check is clean.
The fill also fires on SFC-identical labellings (1–11 slots per rank on the puncture mesh), so the
gap was never ragged-only, just below the wavelet threshold. Remaining: final norms on the ragged arm
agree with SFC to 3–4 digits, not 18 — the fixed-mesh A/B decides ULP class vs defect.

## Step 1 — make ragged labellings a first-class gate

The gate must contain a labelling that is NOT SFC-identical, or it cannot see this class of bug.

1. `testPartitioning`: add an arm that runs the full TEST 2/2b/8/10/11 + COVERAGE set under
   `DENDRO_GRAPH_REFINE=0` on an asymmetric mesh (`PARTEVOLVE_INIT=puncture`-style initial data, or the
   existing ragged option), and assert `[block-vote] reassigned > 0` as proof of raggedness.
2. `partitioningEvolveTest`: global-RK ragged vs SFC on a FIXED mesh (remesh off) → bit-identity at
   sampled observables; then with remeshes → identical `createMesh:` sequence (the Step 0 outcome
   decides whether this is achievable or documented as a residual).
3. ENUTS on a fixed mesh: `nlsmNUTS` ragged vs SFC with `NLSM_REMESH_TEST_FREQ=0` → 18-digit identity
   is the bar. **Run (evening):** global RK on the ragged labelling IS identical to SFC to 18 digits over
   64 steps (min/max; L2 differs only because duplicate local slots at one position are counted
   twice). ENUTS is not: U_PHI min drifts 4e-7 → 3e-5 over 4 coarse steps. ENUTS-only ⇒ suspect the
   duplicate local CG slots at coarse–fine interfaces, written by different blocks at different
   sub-step times with no post-axpy/duplicate sync in `partial_evolve` (ets.h has one). Next: add the
   sync after each partial step (mesh-side helper, block-local), re-run this A/B.
4. Fold the `DENDRO_ENUTS_NANCHK` probe into the ENUTS gate run (cheap, catches the unfilled-padding
   class directly).

## Step 1b — padded-volume work objective (from `~/research/papers/thesis/PARTITIONER_HANDOFF.md`)

The thesis scaling session measured RHS imbalance 1.4–1.7 (max/mean) at 64–512 ranks with octant counts
balanced to 8 %: work is padded block volume, (6m+7)^3 per block of side m octants, so an isolated octant
costs ~6× an octant inside an 8^3 block, and a cut through a uniform region fragments cheap blocks into
expensive ones. Replay estimates: weighted prefix split −7…−31 %, block-aligned cuts −9…−34 %, growing
with np. What the fork adds, and how to test it locally:

1. `dumpPartitionStats`: print padded-volume per rank (Σ blocks (6m+7)^3) max/mean next to `work_imb`,
   for the labelled mesh and its SFC twin. Deterministic, one function.
2. `repartitionMeshGlobal` refine branch: `wpartMode=2` ⇒ `vwgt[i]` = padded volume of element i's
   source block ÷ elements in that block (blocks are already the refine groups via `blockIds()`), so
   the capacity constraint balances padded volume while moves stay block-atomic — the objective is then
   exactly additive per block (no fragmentation term), which is the report's option 3 without a custom
   prefix splitter, plus the 26-boundary and per-level constraints on top.
3. The twin's blocks are inherited by the labelled mesh, so fragmentation at the twin's cuts persists:
   give `buildSFCTwin` a padded-volume `getWeight` (option 1, `partitionW` already takes it) and/or
   snap its cuts to block boundaries (option 3) before the block-atomic refinement.
4. Score offline first on the puncture np=4/8/16 meshes with `edgeCutCompare` (add a `padded_vol`
   stat; the thesis dumps carry no coordinates, so they cannot be scored for adjacency, but their
   per-rank block lists can validate the (6m+7)^3 accounting), then in-Dendro with (1).
Correctness gate unchanged (bit-identity on SFC-identical labellings; ragged gates from Step 1).

**First numbers (puncture, np=4, eOrder 4, 11.6k elements; `testPartitioningEvolve`):**

| labelling | padded-vol imb | work imb (LTS) | ghost send nodes |
|---|---|---|---|
| SFC | 1.074 | 1.131 | 256k |
| REFINE=1 (SFC start), count or padded weights | 1.046 | 1.049 | 240k |
| REFINE=2 (levelwise), count weights | 1.073 | 1.037 | 278k |
| REFINE=2 (levelwise), padded weights | **1.023** | 1.066 | 281k |
| REFINE=3 (padded-weighted prefix start), padded weights | 1.050 | 1.083 | 244k |

Landed (all opt-in): `DENDRO_GRAPH_VTX_WEIGHTED=2` (padded-volume vwgt), `DENDRO_GRAPH_REFINE=3`
(weighted Hilbert prefix as the refiner's start), `padded_vol_imbalance` in `dumpPartitionStats`.
Reading: from an SFC start the weight is only a capacity cap and never binds, so the cut-driven refiner
leaves padded volume where SFC put it; only where caps bind (levelwise per-level caps) does padded
weighting move the number (1.073 → 1.023). At np=4 the whole effect is inside the noise the handoff
predicts (its gains start at np≥16). Next for this thread: score at np 8/16 (`--oversubscribe`), and give
the refiner a balance-driven move phase (or run fastpart's weighted diffusion before the gain
refinement) so padded volume is an objective, not just a constraint.

## Step 2 — scale the gates one notch

- np=8 (`--oversubscribe`) for the four testPartitioning arms + the EM4 CSV md5; then decide
  `DENDRO_REDIST_ROUTE` / `DENDRO_REDIST_FLAGS` defaults (both are O(local) replacements for
  O(N_global) paths; default-on only after np=8 identity).
- BSSN long-haul under the current defaults (RAM discipline: one run at a time).

## Step 3 — the remaining rounds (from the 2026-09-24 audits)

In order of payoff/risk: push partition (trank from labels, no lookup round) → forward-push R1/R2/R3
closure → fused request/response rounds → isDG packing + numMissing fold → IGT per-variable Alltoallv
batching. Each byte-identical, each behind a knob for one A/B, then default.

## Step 4 — price it (needs the user's cluster OK)

`tools/cluster_bundle/` with the four arms (SFC, refined, levelwise+refined, plain fastpart), ENUTS and
global RK, np 8→64. The only open question the box cannot answer is wall-clock.

## Not in this plan

- Design A (labelled ReMesh without the twin): not byte-identical, deferred until Steps 1–2 are green.
- Scaffolding deletion: still one arm per commit behind an A/B, after Step 2.
