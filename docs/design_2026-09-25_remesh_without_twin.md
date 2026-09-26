# Design A — remesh the labelled mesh without the SFC twin (2026-09-25)

Status: design memo from a read-only investigation (an exploration agent, transcribed by the session
that spawned it). Line numbers are for the tree at jj `mrwlpvxl`. Scope: EM4 first
(`em4_base_cpy/solver/src/solverCtx.cpp:3064-3383`); BSSN has the same pattern
(`bssnCtx.cpp:1718-1840`); NLSM is out of scope (its twin is not transient).

## 0. Prior art already in the tree
`Mesh::ReMeshRepartitioned` (src/mesh.cpp:16108-16162) is Design A's octree half: flags → unbalanced
octants from the *labelled* local range; `SFC_treeSort(TS_BALANCE_OCTREE)`; an `E2E_ONLY` SFC mesh;
`repartitionMeshGlobal`. EM4's legacy `rkSolver.cpp:306-312,1829-1856` used it and then called
`intergridTransferVars(old, new)` into the labelled mesh — the pattern now frozen as "never
interGridTransfer into a graph mesh". It has two I4-class holes: no flag sync-back to the old mesh
(ReMesh step 3 missing, so M2' != new octree when balancing changes flags) and no canonical block-info
injection. The octree half is known to run; the transfer and blocks halves are where it failed.

## 1. What assumes contiguous SFC ownership (C = correctness as coded, V = convenience)
**ReMesh** (src/mesh.cpp:15720-16106)
- C: COARSE assumes the 8 siblings sit at `ele..ele+7` in local order (15762-15766). Holds on a
  labelled mesh only if the siblings are co-local; local order is SFC order (rebuild sorts by
  (trank, eid), 20039-20044, mesh.h:4213-4216).
- V: splitters = Allgather of each rank's last unbalanced octant (15777).
- None: **the balance itself is partition-agnostic.** `SFC_treeSort` does the global Alltoallv
  (sfcSort.h:2194) *before* the local construct/balance (sfcSort.h:2230, 644-674). Input distribution
  does not matter; the output is SFC-partitioned.
- C as coded / V in principle: step 3 flag sync (15868-16080) sends balanced octants back by the old
  splitters and walks old-local vs new in lockstep ("M1 & M2 front alignment failed" otherwise). A
  TreeNode-keyed join does the same job with no contiguity requirement.
- V: output is SFC by design: `partitionW` (15792), `TS_REMOVE_DUPLICATES` (16083),
  `enforceSiblingsAreNotPartitioned` (16092, octUtils.cpp:1301), then the ctor (16097).

**setMeshRefinementFlags** (18978-19059): C: COARSE accepted only if `ele+7` is local with the same
parent (18997-19019); otherwise *silently* NO_CHANGE.

**IGT plan** (18745-18956): C: M2' must be sorted (18817); uses the destination's
`getSplitterElements()` per rank (18839-18849) plus `SFC_treeSearch` (18909); aborts if the receive
count != destination local count (18946).

**IGT receive** (include/mesh.tcc:5385-5422): C: received DG blocks are written *positionally*,
`wVec_m2[(ele - m2LocalElemBegin)*NpE]` — the rank-ordered concatenation of M2' must equal the
destination local range in SFC order. This is the real reason IGT cannot target a graph mesh.
C (bit identity): every local element writes each non-hanging, locally owned node, last writer wins in
local SFC order (5401-5419); no ghost exchange precedes IGT (include/dvec.h:409-433). Candidate values
at a shared node come from different operators, so which element wrote last shows in the bits
(the 5573-5589 comment records this as a real noise source).

**Blocks / I4** (octUtils.cpp:659-869 via performBlocksSetup, mesh.cpp:14242): C: `d_min` comes from
the local chunk (685-689) and a block needs `octVolume == blockVolume` inside the local range (791).
Blocks are defined by the SFC chunk, so Design A must reproduce the SFC chunking of the *new* octree.

**repartitionMeshGlobal input** (mesh.h:4120-4360, mesh.cpp:19379+): C: eid = local index +
`ele_offsets[rank]`; the push path finds an eid's source by `upper_bound(ele_offsets)` (19825-19827)
and `splitLocalRemote` uses the same ranges (19952). The input must be a mesh with contiguous eid ranges
per rank, eids in SFC order, full E2E and element ghosts. An SFC `E2E_ONLY` mesh of the new octree is
therefore irreducible unless the rebuild is rewritten.

**Solver glue** (em4_partitioning.h:91-116): V: the blockInfo copy is positional; ownerMask is read only
by default-off paths (mesh.cpp:20195-20211).

## 2. Minimal design ("A-min"), and what it cannot avoid
1. Flags on the labelled mesh (already computed there by `isReMeshUnzip`). Resolve COARSE by a sibling
   rendezvous keyed by the parent TreeNode, not by local contiguity (risk R1).
2. Unbalanced octants from the labelled local range, then the existing parallel
   `SFC_treeSort(TS_BALANCE_OCTREE)` (distribution-agnostic, §1). Do **not** write a local
   "fixed point over 26-neighbours" balance: the 2:1 ripple travels up to lmax−lmin rings. Then replay
   ReMesh's partition sequence (`partitionW`, `REMOVE_DUPLICATES`, `enforceSiblings`) so the SFC
   chunking matches.
3. Join: one TN-hash rendezvous (the directory idiom of mesh.tcc:6210-6281). Each balanced octant B
   queries B, parent(B) and child0(B) against the old (TreeNode → label, flag) directory: the final old
   flags (flag sync without splitters) and the inherited label (children take the parent's owner; a
   coarsened parent takes its co-owned children's owner).
4. E2E_ONLY SFC mesh built directly (`new ot::Mesh(balOct,…,E2E_ONLY)` as in ReMeshRepartitioned;
   skip createMesh's three extra treeSorts, meshUtils.cpp:134-165). Canonical block info from
   `octree2BlockDecomposition` on the local chunk plus the `deriveBlockInfoFromBlocks` logic
   (5851-5879), no FDM ctor.
5. Labelled rebuild: `repartitionMeshGlobal` with a preset-labels source at the partition decision
   (19638-19788), then the existing `blockVote` and push/rebuild; fastpart only when the inherited
   max/mean load exceeds a tolerance.
6. Direct transfer old labelled → new labelled: factor IGT's M2' loop (mesh.tcc:5172-5331) to run on
   the old labelled local range, key each M2' DG block by TreeNode (M2' == new octree after step 3),
   route through the redistributeVec directory to the new owner *and* its R1 ghost holders, write each
   new CG node from the SFC-canonical writer (owner element = coarsest sharer, ties by TreeNode; R = its
   SFC rank; writer = the max-eid candidate on R with the node non-hanging), then canon-fill any
   remainder (I3) and pos-bcast where the solver enables it. The R1/R2/R3 26-closure push (19791-19871)
   already brings every writer candidate for every local and read-set node.

Unavoidable in any variant: a sorted/balanced octree pass (SFC pays it too), the E2E_ONLY ctor, and the
full labelled rebuild (E2N, audit, scatter maps, blocks, zip plan).

## 3. Cost at 192 ranks (EM4 d9; today 3.0 s = g2s 0.17 + remesh+IGT 0.9 + s2g 1.9; SFC 0.73)
Removed by A-min: g2s (0.17); the ReMesh successor FDM ctor with blocks + zip plan (~0.5-0.6 of the
0.9); the SFC IGT (~0.1); the s2g redistributeDVec (~0.15); createMesh's re-sorts (~0.05); the fastpart
call at non-trigger remeshes (unmeasured; the np=48 cluster phase profile puts partition-decision at
8 ms, so small). Added: join (~0.05), chunk block decomposition (~0.01), direct transfer (~0.15-0.3).
**Estimate: A-min ≈ 1.5-2.2 s per remesh (saving 0.8-1.5 s, 27-50 %)**; over 15 remeshes 12-22 s of
the 33 s wall gap at np=192 (1.56× SFC → ~1.2-1.35×).
Floor: the labelled rebuild (≈ 0.8-1.5 s) already exceeds SFC's whole 0.73 s remesh. No
remesh-without-twin design gets below ~1.5-2× SFC per remesh unless the rebuild itself gets faster; at
117 elements/rank and flat with np it is latency- or O(P)-bound, not work-bound.
Skeptical split: ~0.4-1.1 s of the saving needs no new transfer code — T1: inherit labels inside
today's sandwich (stash (TN, rank) before `delete oldGraph`, solverCtx.cpp:3277-3279); T2: a
successor-lite ReMesh (the successor never unzips; it needs only octree2BlockDecomposition for
blockInfo, not zip plan or neighbour setup — essentially `DENDRO_LAZY_ZIPPLAN=1` promoted after its
long-haul A/B). A-min's unique increment over T1+T2 is ~0.3-0.5 s/remesh and carries all the
bit-identity risk.

## 4. Risks against the invariants and gates
- R1 coarsening needs co-owned siblings; setMeshRefinementFlags silently drops COARSE otherwise. Today
  guaranteed only indirectly (with LEV_GAP 0, octUtils.h:46, 8 leaf siblings fall inside one canonical
  block; the vote 19543-19571 keeps the block on one rank; breaks under `DENDRO_DISABLE_BLOCK_VOTE=1`,
  legacy blocks or tags — silent AMR divergence). Fix: the parent-keyed rendezvous.
- R2 IGT bits (biggest): A-min must reproduce SFC's last-writer choice (mesh.tcc:5401-5419) plus canon
  fill exactly; ULP drift compounds into AMR-decision divergence. The writer rule above is
  reconstructed from docs, not from `buildE2NMap` — verify first.
- R3 I4 chunking: blocks depend on the new octree's SFC chunk; ReMesh's chunk comes from routing to the
  *old* splitters, then REMOVE_DUP treeSort, enforceSiblings, ctor. A-min must prove the REMOVE_DUP
  sort's partition depends only on the global set.
- R4 Fix B / stale ghosts: M2' on the old labelled mesh reads ghost CG slots of boundary elements; the
  twin avoided this by pulling only source-local slots. Needs a mirror-free exchange or
  canonical-position sourcing before M2' ("exchange before you read").
- R5 I3 orphans: refinement creates new positions, so the phys-pos orphan fill cannot be reused;
  orphans must come from ghost-element M2' DG (the transfer must deliver R1 ghost DG).
- R6 "coarse duplicate instances" in labelled local ranges (correctness doc §8 1b; daUtils.tcc:641-653)
  contradicts the 1:1 eid→leaf construction (19857-19862). Assert disjoint leaves first.
- R7 load drift under inheritance (a front rank's load ×8 per level); the vote can still empty a rank
  (the np=192 `MPI_Abort(7)`, 19874-19883) → one-block floor.
- R8 solver deltas: BSSN pos-bcast on for graph meshes (repeat after the transfer) + tracker exchange;
  NLSM computes flags on the twin (restructure first).
Gates: TEST 2b/8be (redistributeVec untouched → maxErr=0), TEST 10/11, EM4 md5, ragged ENUTS 18 digits,
BSSN four-arm. With inherited labels the labelling differs, so EM4 md5 needs a new reference; the
partition-invariance bar is then 18-digit min/max vs SFC.

## 5. Staged plan (each behind a knob, unset = today)
- S0 (no code; go/no-go): np=192 refine-arm phase profile (`DENDRO_REPART_PHASE_PROF=1
  DENDRO_MESH_PROF=1 DENDRO_REPART_ROUNDS=1`) splitting s2g and remesh+IGT by phase; probe-gated
  assertions: labelled local range = disjoint leaves (R6), zero split sibling groups (R1).
- S1 `DENDRO_REMESH_INHERIT_LABELS=1` (+`_TOL`) inside the sandwich: stash old (TN, rank), join at s2g,
  preset labels → vote → rebuild, fastpart only above the tolerance. Gate: 18-digit min/max vs SFC,
  ragged ENUTS, BSSN four-arm under inheritance, TEST 10; log trigger rate and ghost bytes.
- S2 `DENDRO_REMESH_SUCCESSOR_LITE=1`: successor without zip plan / neighbour setup, blockInfo from
  chunk decomposition; buildGraphTwin constructs the E2E_ONLY mesh directly. Gate: byte-identical
  everywhere.
- S3 `DENDRO_REMESH_DIRECT=1` (A-min 1-4 and 6; labels from S1) with `DENDRO_REMESH_DIRECT_CHECK=1`
  running the sandwich alongside and comparing (a) the balanced TreeNode set, (b) per-element
  CanonicalBlockInfo, (c) the final labelled vector at every local and read-set cg, bitwise (a new
  TEST 10d). Then ENUTS 18 digits, then the BSSN four-arm.
- S4 default flips per solver after a Kingspeak re-price.
Where it most likely fails: S3(c) (writer rule / ghost sourcing, R2/R4); next S3(b) if the SFC
chunking depends on the input distribution (R3).

**Verdict: not yet.** Do S0-S2 first; build S3 only if the profile shows ≥ ~0.5 s/remesh left after
S1+S2.
