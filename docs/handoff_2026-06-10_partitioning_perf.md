# Handoff — Graph-Mode Repartitioning: Validation + Perf Cleanup (2026-06-10)

Snapshot of the graph (`fastpart`) repartitioning work in `dendrolib_dfvk` and the
three downstream solvers, written so the next session can pick up cold. Covers:
what landed, what's validated and how, the jj checkpoint stack, the performance
optimization roadmap (with the deep analysis for the deferred items), and the
build/run/validate recipes.

---

## 0. TL;DR

- Graph repartitioning is **correctness-complete** for EM4, NLSM, and now BSSN.
  Graph reproduces SFC physics to floating-point round-off.
- This session: fixed the block-type classifier, added TEST 10/11, wired BSSN,
  stripped dead probes, and **DOF-batched `redistributeVec` (measured 3.4× on the
  redistribute step, bit-identical)**.
- Next measurable lever is **#4 `buildSFCTwin`** (~1.6 s, the sandwich dominator).
  **#2** (drop the `Allgatherv`) is real but scaling-only and **deferred until a
  cluster / large-mesh benchmark exists** — it can't be verified at 4 ranks.

---

## 1. What landed this session (jj stack)

Repo `dendrolib_dfvk`, branch tip is `partitioning-redist-batch`. Each is an
independent, abandonable checkpoint, exported to git refs:

```
uzluzwuw  e0a3860e  partitioning-redist-batch    | perf: DOF-batch redistributeVec (build scatter plan once)
tszopons  23f52f32  partitioning-redist-cleanup  | perf: strip dead debug-probe scaffolding from redistributeVec
sqsossyu  e5fdc3fd  partitioning-blocktype-fix   | fix: graph UNZIP_INDEPENDENT block classification
wmvprwls  38e6360d  partitioning-perf3           | cleanup + micro-perf + opt-in OMP + AMR-cycle test
```

Rewind with `jj edit <name>` / `jj abandon <change-id>` / `jj undo`.

**GOTCHA (cost me a re-split this session):** jj auto-snapshots the working copy
into `@` on every operation. If `@` is a bookmarked commit you want to keep clean,
**editing files silently amends them into it.** Always `jj new -m "<step>"` BEFORE
starting a new sub-step's edits.

### 1a. Block-type classifier fix (`partitioning-blocktype-fix`)
`src/mesh.cpp::flagBlockGhostDependancies`. Graph mode mis-classified 11
`UNZIP_INDEPENDENT` blocks (they actually read ghost). Two bugs:
- diagonal-edge indexing dropped the `dir*(2*blk_ele_1d)` offset →
  `blkDiagMap[dir*(2*blk_ele_1d)+2*k+...]`.
- face/edge/vertex neighbor test used `== W_DEPENDENT`, missing `UNKWON`
  pure-ghost neighbors → switched to a `nbrIsGhostDep` lambda
  (`!= EType::INDEPENDENT`).
Validated by the new NaN-poison audit (`Mesh::auditBlockTypeIndependence`,
declared in mesh.h) wired as **TEST 11a/b/c** in
`test/src/partitioningMeshTests.cpp`. 0 misclassified on SFC/graph/remesh.

### 1b. TEST 10 sandwich fix (folded into `partitioning-perf3`)
TEST 10 aborted: it paired `ReMeshRepartitioned` (graph mesh) with
`interGridTransfer` (needs an SFC dest). Rewrote it as the SFC-twin sandwich.
**Rule: never `interGridTransfer` into a graph mesh** — IGT builds its recv plan
from SFC splitter keys.

### 1c. BSSN integration (separate repo, see §5)
`~/research/dendrogr_dfvk_repartitioning` (branch `master`). Full graph plumbing
ported from EM4. Builds + runs a real BBH in graph mode, both sandwich paths fire
clean. MSRK/AEH temporarily guarded off (dendrolib lags upstream).

### 1d. Probe strip (`partitioning-redist-cleanup`)
`Mesh::redistributeVec` was 957 lines, ~489 of them dead `EM4_REDIST_PROBE_*` /
`DENDRO_ENABLE_DEBUG_PROBES` scaffolding from resolved investigations. Removed →
469 lines. Bit-identical (TEST 2b/8be `maxErr=0`).

### 1e. DOF-batch redistributeVec (`partitioning-redist-batch`) — the perf win
`Mesh::redistributeVec` (now at `include/mesh.tcc:5937`) is **DOF-aware**:
`redistributeVec(dstMesh, vecIn, vecOut, dof=1)`. vecIn/vecOut laid out
`[dof][CG size]` (field v at `+ v*this->getDegOfFreedom()` /
`+ v*dstMesh->getDegOfFreedom()`). The scatter plan (global `TN→rank` map,
send/recv pattern, `tnToLocal`/`eAbs`, orphan-fill phys map) is built **once**;
only the packed values scale with dof. Single batched set of collectives instead
of one `Allgatherv`+hash+`Alltoallv`-set **per field**.

The three solver `redistributeDVec` wrappers now make one batched call instead of
a per-field loop:
- `em4_base_cpy/solver/include/em4_partitioning.h` (~line 173)
- `nlsm-gr_copy/solver/include/nlsm_partitioning.h` (~line 207)
- `dendrogr_dfvk_repartitioning/BSSN_GR/include/bssn_partitioning.h` (~line 146)
**These solver edits are uncommitted in their own git repos** (user owns commits).

---

## 2. Validation status (all green)

| Check | Result |
|---|---|
| testPartitioning TEST 2b (redistribute SFC→graph) | `maxErr=0` (bit-perfect) |
| testPartitioning TEST 8be (sfc→graph→sfc roundtrip) | `maxErr=0` |
| testPartitioning TEST 11a/b/c (block-type audit) | PASS, 0 misclassified |
| EM4 graph dof=8, batched | **bit-identical** to pre-batch (U_E0/E2/B2 l2 match all digits) |
| EM4 graph vs skip (long-haul, prior) | physical vars ~1e-6, analytical-zero ~1e-7 floor |
| BSSN BBH graph vs SFC, WAMR (no remesh) | bit-identical first ~10 steps, then 5e-8 by t=10 (round-off) |
| BSSN BBH graph vs SFC, BH_LOC (30 remeshes) | **BH track bit-identical** + grid identical at all 928 step-points |

Pre-existing reds (NOT regressions): TEST 8bc remesh zip-roundtrip (~138-385
mismatches; partition-artifact diagnostic), "TEST 2" in some configs.

### Performance measured (the only number we can show at this scale)
EM4 dof=8, env-toggle A/B (`DENDRO_REDIST_PERDOF=1` forced the old per-field
loop), `MPI_Wtime` around the redistribute, 9 redistribute calls over a short run:

| mode | per-call | total |
|---|---|---|
| per-DOF (old) | 62.9 ms | 566 ms |
| **batched (new)** | **18.3 ms** | **165 ms** |

→ **3.4×** on the redistribute step. Not 8× because batching removes the *fixed*
per-call overhead (global `Allgatherv` of the octree + hash build + collective
latency) ×(dof−1), but the `Alltoallv` payload bytes are unchanged:
`speedup = (dof·F + P)/(F + P)`. BSSN (dof=24) should be larger; not measured
(slow). **The instrumentation was reverted** — to re-measure, re-add an
`MPI_Wtime` + `DENDRO_REDIST_PERDOF` toggle around the `redistributeVec` call in a
solver's `redistributeDVec`.

---

## 3. Optimization roadmap (origin of #1–#7)

From the redistribute/sandwich profile. Status after this session:

| # | Item | Status |
|---|---|---|
| 1 | **Batch `redistributeVec` across DOFs** | ✅ DONE — 3.4×, bit-identical |
| 2 | **Drop the global `Allgatherv` in `redistributeVec`** | ⏸ DEFERRED — see §4 |
| 3 | OpenMP the audit + block-decomp hot loops | infra exists, gated `DENDRO_ENABLE_OPENMP_PARTITIONING`, OFF |
| 4 | **Make `buildSFCTwin` cheaper** | 👈 RECOMMENDED NEXT — measurable here |
| 5 | Cache TN/phys maps within a sandwich | not started (overlaps #1/#2) |
| 6 | Incremental block-decomposition | deferred (needs BBH profile, now have one) |
| 7 | Batch `broadcastCgValuesByPhysPos` across DOFs | not started (same pattern as #1) |

Profile shape (BSSN small mesh):
- `repartitionMeshGlobal` ≈ 456 ms: block-decomposition (288 ms, 63%) +
  e2n-audit (122 ms, 27%) ≈ 90%. → these are the #3 (OMP) targets.
- Sandwich ≈ 3.46 s: g2s ~1.65 s + remesh+IGT ~1.59 s dominate; s2g ~0.22 s
  (keep-partition fast path). g2s is dominated by `buildSFCTwin`. → #4.

---

## 4. Deep notes on the deferred / next items

### #4 — `buildSFCTwin` (RECOMMENDED NEXT, measurable at current scale)
`*_partitioning.h::buildSFCTwin` rebuilds a full FDM SFC mesh (`ot::createMesh`,
`SM_TYPE::FDM` = E2E + E2N + scatter map + blocks) from scratch every remesh,
~1.6 s. The twin is **transient** — used only for `redistributeFlags`,
`this->remesh`, and `interGridTransfer`, then discarded.

Plan: instrument `buildSFCTwin`'s internal phases (E2N / scatter-map /
block-decomp) with the same env-toggle + `MPI_Wtime` approach that gave the clean
3.4× number, find the dominant phase, and skip what the throwaway twin doesn't
need (e.g. block decomposition if IGT/remesh don't consume it). Measurable
before/after **here** — no cluster needed.

Bigger structural play (high risk, long-term): a **partition-invariant intergrid
transfer** that works directly on graph meshes, eliminating the SFC round-trip
entirely. This is the open problem behind the `DENDRO_S2G_SKIP_REPART` knob.

### #2 — drop the global `Allgatherv` (DEFERRED; here's exactly why + how)
`redistributeVec` Step 1 does `MPI_Allgatherv` of **every rank's local
TreeNodes** → builds a global `TN→rank` hash. O(N_global) memory + comm per rank,
per redistribute. This is the scaling bottleneck (invisible at 4 ranks / ≤60k
elements — the whole batched redistribute is 18 ms, mostly this + hash).

**Why it's not a quick edit:**
1. The partition is **Hilbert SFC** (`SFC::seqSort::SFC_treeSort`), NOT Morton
   `operator<`. So routing can't use a plain `std::` binary search — it needs
   Dendro's `SFC::seqSearch::SFC_treeSearch`.
2. **Reusable idiom exists**: `src/mesh.cpp:1529-1572` already does
   "splitters + `SFC_treeSearch` → per-rank index ranges". The dst partition's
   splitter octants are on the mesh already: `dstMesh->getSplitterElements()`
   (2·npes, no gather needed). Adapt: `SFC_treeSearch(dstSplitters,
   mySrcLocalElementsAsKeys)` → each dst splitter's `getSearchResult()` gives the
   index in my src list where that dst rank's range begins → assign src elements
   to dst ranks by those index ranges. O(npes·log N_local), no global gather.
3. **Detection problem**: this only works when dst is **SFC-contiguous**. Graph
   dst's splitters don't describe its (arbitrary) partition. But
   `buildSFCTwin` leaves the partition method at the **default `fastpart`**
   (mesh.h:1122) — so `getPartitioningMethod()` returns `fastpart` for both SFC
   twin AND graph mesh. Fix: have `buildSFCTwin` explicitly
   `setPartitioningMethod(OriginalPartition)`, then gate the fast path to
   non-`fastpart` dsts. **Safe by construction**: graph meshes are always
   `fastpart` (set in `buildGraphTwin`), so the fast path can *never* mis-route a
   graph dst — worst case is "no speedup," never "wrong."
4. **Only covers half the sandwich**: g2s has SFC dst (fast-pathable); s2g and the
   initial swap have **graph** dst → still need the gather, or a full distributed
   directory (hash-home two-hop shuffle) for the general case.

**Why deferred, not done:** it's a real Dendro-SFC-internals change with edge
cases (empty ranks, splitter validity, `OCT_FOUND` handling), and the payoff is
**pure scaling** — at 4 ranks I can only validate *correctness* (bit-identical),
not *speedup*. Optimizing an unmeasurable number invites silent regressions.
**Unblocker: a multi-node or large-mesh benchmark** where `Allgatherv` cost is
visible. Then #2 becomes verifiable AND prioritizable against #4.

User decision this session: do the **SFC-dst splitter fast path** increment — but
on finding the above depth, agreed to defer pending a real benchmark.

### #3 — OpenMP (infra present, OFF)
Track-C scaffolding exists (thread-local maps + deterministic serial merge for
the audit hash, audit phase-2 element loop, `buildZipPlan` myMin), gated behind
CMake `DENDRO_ENABLE_OPENMP_PARTITIONING` → `DENDRO_OMP_PART`. Targets the ~90%
of `repartitionMeshGlobal` time (block-decomp + e2n-audit). Needs enable +
bit-identity validation at `OMP_NUM_THREADS>1` (race-detection workload: TEST 11
+ AMR-cycle test on a maxDepth=8 mesh).

### #5 / #7 — map caching / bcast batching
Same per-DOF→single-pass idea as #1, applied to the within-sandwich repeated map
builds and `broadcastCgValuesByPhysPos`. Low effort, modest win. Do after #4.

---

## 5. BSSN repo (`~/research/dendrogr_dfvk_repartitioning`, branch `master`)

Graph plumbing mirrors EM4. See memory `project_bssn_repartitioning_integration`.
- NEW `BSSN_GR/include/bssn_partitioning.h` (namespace `bssn`, guard
  `BSSN_WITH_GRAPH_PARTITIONING`): applyPartitioning, buildGraphTwin, buildSFCTwin
  (`SM_TYPE::FDM`), redistributeDVec (Fix B + DOF-batched call), redistributeFlags.
- `bssnCtx.{h,cpp}`: `swap_to_graph_partition`, `maybe_apply_partitioning`,
  `remesh_and_gridtransfer` override (4-phase sandwich).
- `parameters.{h,cpp}`: `BSSN_PARTITIONING_METHOD` (toml key, set **=3** for graph,
  else silently SFC).
- `bssngr_main.cpp`: post-`ets->init()` swap hook.
- `CMakeLists.txt`: `BSSN_WITH_GRAPH_PARTITIONING` (+ TEMPORARY `BSSN_WITH_MSRK`,
  `BSSN_WITH_AEH`, default OFF).

**TEMPORARY guards** (clearly marked in-code; remove after rebasing dendrolib onto
upstream): `BSSN_WITH_MSRK` (no `ets_msrk.h` in dendrolib_dfvk),
`BSSN_WITH_AEH` (older 29-arg `AEH_BHaHAHA` ctor vs evolved 32-arg).

**Build:** `-DBSSN_WITH_GRAPH_PARTITIONING=ON` AND fastpart paths
(`-DFASTPART_INCLUDE_DIR=/home/denv/research/fastpart/build/install/include
 -DFASTPART_LIB_DIR=/home/denv/research/fastpart/build/install/lib`). Targets
`bssnSolver`, `tpid`.

**Run:** `tpid <par> <numthreads>` first (~18 min, writes `rit_q1_tpid_sol.bin`
~53 MB), then `bssnSolver` with `BSSN_ID_TYPE=1`. q1.tinytest + REFINEMENT_MODE=3
(BH_LOC) makes the sandwich fire during evolution; REFINEMENT_MODE=0 (WAMR) stays
pinned and never remeshes pre-merger.

**Open item — investigate:** ~55k `rank N key: ... not found` messages from the
BH-location tracker's point lookup in graph mode (BH tracking still correct).
Possibly graph-mesh point-locate behavior worth comparing vs SFC.

---

## 6. Build / run / validate recipes

```bash
# dendrolib testPartitioning (args: method depth wtol ld_tol order grain npe)
cd ~/research/dendrolib_dfvk
cmake --build build -j$(nproc) --target testPartitioning
mkdir -p vtu      # else rank-0 deadlocks in write_vtu
mpirun -np 4 build/testPartitioning 3 7 1e-3 0.1 4 50 4   # method 3 = graph
# look for: TEST 2b maxErr=0, TEST 8be maxErr=0, TEST 11a/b/c PASS

# EM4 graph vs skip A/B (DOF>1 correctness)
cd ~/research/em4_base_cpy
cmake --build build -j$(nproc) --target em4Solver
# toml: SOLVER_PARTITIONING_METHOD = 3 (graph) vs 0 (SFC); IO_OUTPUT_FREQ=0
# graph U_E0/E2/B2 _DIFF l2 must match the recorded bit-identical values

# BSSN graph BBH (see §5)
```

### Validation knobs / traps (from memory)
- `SOLVER_PARTITIONING_METHOD=3` / `NLSM_PARTITIONING_METHOD=3` /
  `BSSN_PARTITIONING_METHOD=3` — **without this, "graph" runs are silently SFC**
  and validation proves nothing.
- `DENDRO_S2G_SKIP_REPART` is **EM4-only**; it's an A/B trap on NLSM/BSSN. Use
  `*_PARTITIONING_METHOD=0` as the SFC baseline instead.
- `mkdir -p vtu` in any run dir (missing vtu/ silently deadlocks rank 0).
- `/tmp` is a 32 G tmpfs that fills fast → `IO_OUTPUT_FREQ=0`, clean run dirs;
  disk-full crashes masquerade as code bugs (exit 134/139/213).
- Fix B: `redistributeDVec` skips src/dst `performGhostExchange` (its
  `m_uiZipNonPrimaryToGhostCg` mirror overwrites good LOCAL cgs with stale ghost).
- `DENDRO_FORCE_POS_BCAST` gates the post-redistribute consensus bcast inside
  `broadcastCgValuesByPhysPosPublic`.
- `DENDRO_ORPHAN_FILL_GEOM_KEY=1` reverts the orphan-fill cg2dg-key fix (A/B only).

---

## 7. Key file map

| Path | What |
|---|---|
| `include/mesh.tcc:5937` | `Mesh::redistributeVec` (DOF-aware, post-batch) |
| `include/mesh.h:3418` | `redistributeVec` decl (`dof=1`) |
| `src/mesh.cpp` `flagBlockGhostDependancies` | block-type classifier (fixed) |
| `src/mesh.cpp` `auditBlockTypeIndependence` | NaN-poison block-type audit (TEST 11) |
| `src/mesh.cpp:1529-1572` | splitter + `SFC_treeSearch` idiom (template for #2) |
| `include/mesh.h:2155` `getSplitterElements()` | dst partition splitters (for #2) |
| `include/mesh.h:4276` `getPartitioningMethod()` | default `fastpart` — see #2 detection |
| `test/src/partitioningMeshTests.cpp` | TEST 10 (sandwich), TEST 11 (block-type) |
| `*/solver/include/*_partitioning.h` | per-solver sandwich + `redistributeDVec` |

---

## 8. Recommended next session

1. **Profile `buildSFCTwin` phases** (#4) — measurable now, the ~1.6 s sandwich
   dominator. Trim what the transient twin doesn't need.
2. **Set up a representative benchmark** (more ranks / larger mesh) — this is the
   single thing that unblocks #2 and lets us prioritize #2 vs #4 honestly.
3. With that benchmark: do #2 (SFC-dst splitter fast path → then maybe the
   distributed directory for graph dst) and enable/validate #3 (OMP).
4. Investigate the BSSN BH-tracker "key not found" noise (graph vs SFC).
5. Eventually: remove the TEMPORARY MSRK/AEH guards after rebasing dendrolib_dfvk
   onto upstream.
