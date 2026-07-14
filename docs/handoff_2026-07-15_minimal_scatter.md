# Handoff — DENDRO_MINIMAL_SCATTER: find what the remesh sandwich reads beyond R1

**Date:** 2026-07-15
**Branch/HEAD:** `partitioning-nan-hunt`; the work sits on a jj stack (see §8).
**Status:** ~~prototype landed, gated OFF; **not** bit-identical over AMR long-haul — root cause open.~~
**RESOLVED 2026-07-15 (same day, later session)** — see the addendum at the bottom (§11). Root cause
was NOT the sandwich: (a) the v1 `isGhostTwo` filter starved the unzip (`isGhostTwo` is a fetch-round
label, not a geometric ring — §3's coarsening attribution came from misread init-loop log lines; the
diverging remesh was a refinement), and (b) mirror last-writer parity. Redesigned as
full-map-then-consumer-slot-shrink; EM4 t_end=30 long-haul is **byte-identical** at a **3.13×** cut
(447,092 → 142,964 sends; 1.76× SFC). NLSM/BSSN long-hauls still pending before default-on.
Full record: Obsidian `Repartitioning/Logs/2026-07-15.md` + `Comms volume and LTS weighting` §I.

---

## 1. Goal

`DENDRO_MINIMAL_SCATTER=1` cuts graph-mode per-step ghost communication **3.78×** (447,092 → 118,284
CG-node sends at np=4/maxDepth7) by dropping the R2/R3 correctness halo from the per-step nodal scatter
map. It is **numerically bit-identical during evolution** but **diverges at AMR remeshes** (~3e-7 peak,
bounded). **Your job: find exactly what the remesh sandwich reads from the dropped ghost set, then either
(a) make the reduction surgical so it's bit-identical, or (b) confirm it's a limited, restorable set.**
Landing this = a real, validated per-step comms win on the biggest cost in graph mode.

---

## 2. What the change is (one filter)

`src/mesh.cpp` ~19273, inside `repartitionMeshGlobal`'s nodal scatter-map rebuild:
```cpp
const char* minScatterEnv = std::getenv("DENDRO_MINIMAL_SCATTER");
const bool minimalScatter = (minScatterEnv && minScatterEnv[0] == '1');
...
for (ele_id in 0..m_uiNumTotalElements) {
    const bool isR2 = new_oct_connectivity_map[ele_id].isGhostTwo;   // R2/R3 flag
    if (minimalScatter && isR2) continue;   // <-- THE FILTER: skip R2/R3 ghost elements
    ... push non-local CG nodes of this element into recvNodeSM_r[ownerTrank] ...
}
```
The send side follows automatically (resolved from recv requests via Alltoallv), and the DG-tagged send
path vanishes (it was R2/R3-boundary-only). Default OFF → zero change unless the env is set.

Background (why R2/R3 exist): the keyset ghost fetch (`src/mesh.cpp` ~18922-19053) pulls R1 (26-nbrs of
local elements), R2 (26-nbrs of R1), R3 (26-nbrs of R2). **R2/R3 are fetched only to stabilize the
setup-time canonical-owner cascade** (comment at ~19010-19016). The runtime **unzip reads only R1**
(proven — see §4).

---

## 3. The finding (EM4 t_end=30 long-haul, np=4, maxDepth7, remesh every 10)

Repro in §7. Compared graph-mode `DENDRO_MINIMAL_SCATTER` **off vs on**:

- **Steps 0-19: bit-identical** (byte-identical `_ANALYTICAL_DIFF.csv` rows). Includes iter-1 remesh at
  step 10 (a **refinement**, 736→1520) — still identical.
- **Divergence enters at step 20** = **iter-2 remesh = the first COARSENING (1520→1240).**
- Magnitude: onset ~3e-9, **peak ~3e-7 at step 129**, physical var U_E0 differs ~5e-8 relative at t=30.
  **Bounded — never blows up** across all 230 steps; both runs complete.
- **Mesh-size trajectory is IDENTICAL** in both runs (all 9 remeshes, same old→new sizes). ⇒ this is a
  **VALUE-only** perturbation, NOT structural AMR-decision divergence. The wavelet/remesh decision (which
  unzips → reads R1 → clean) picks the same mesh both ways; only the transferred VALUES differ.

### What this pins down
- The per-step exchange + unzip + RHS are **fine** (bit-identical between remeshes).
- The bug is in the **remesh sandwich**, specifically the **value transfer**, and it first bites on
  **coarsening (restriction)** — not refinement (prolongation).
- At the moment of remesh, the LOCAL state is bit-identical (steps 0-19 identical), so the ONLY thing
  that differs between off/on is the **contents of the graph mesh's ghost CG slots** (off = R2/R3 filled,
  on = R2/R3 stale/unrefreshed). **⇒ some sandwich step reads those ghost CG slots.**

---

## 4. What is already proven / ruled out

- **Unzip reads only R1.** Minimality test in `test/src/partitioningMeshTests.cpp` ~795 (added this
  session): compute excess = `recvNodeSM` set ∖ unzip-required set (unzip-required via
  `blkUnzipElementIDs` → `m_uiE2NMapping_CG` → ghost cgs, ~726-743); poison exactly the excess with NaN,
  set rest to 1.0, `unzip()` WITHOUT ghost exchange, count contaminated outputs. Result: excess = 304,186
  of 447,092 (68%), **contamination = 0**. With the filter ON: excess → **54**, contamination 0. So the
  unzip is genuinely R1-only.
- **Not structural** — mesh trajectory identical (§3).
- **Not the per-step exchange** — bit-identical between remeshes.
- `testPartitioning` gates PASS with the filter on: TEST 2b/8be `maxErr=0`, TEST 10 (4-cycle AMR
  sandwich) ~5.3e-15, 11a/b/c. **NOTE:** TEST 10 passing but the long-haul diverging means **TEST 10 does
  not exercise the divergent path** (likely: its analytic poly is smooth enough that the coarsening
  restriction error is below its ~5e-15 tolerance, OR it doesn't coarsen the way EM4 does). Do not trust
  TEST 10 as sufficient here.

---

## 5. Prime suspects (the sandwich, in order it runs)

Per-solver sandwich lives in `*_partitioning.h` (EM4: `~/research/em4_base_cpy/solver/include/em4_partitioning.h`). Steps:
1. **`buildSFCTwin`** (em4_partitioning.h ~308) — build throwaway SFC twin of current graph mesh.
2. **`redistributeFlags`** — move refinement flags to twin.
3. **remesh + `interGridTransfer`** on the SFC twin — the IGT (SFC).
4. **`buildGraphTwin`** → `repartitionMeshGlobal` — repartition the new mesh (rebuilds the scatter map).
5. **`redistributeDVec`** (em4_partitioning.h ~132) — move solved state back to the graph partition.

**Where a graph-mesh ghost CG could be read (ranked):**
1. **`Mesh::interGridTransfer` (include/mesh.tcc ~4868) — the COARSENING/restriction branch.** Strongest
   suspect: divergence is coarsening-only. Restriction averages child values into a parent; if children
   span a rank boundary, it reads ghost child cgs. If the transfer is happening on a mesh that still
   references the graph mesh's ghost layout (or g2s handed off stale ghosts), restriction reads stale.
   Look at `interGridTransfer` / `interGridTransferSendRecvCompute` (src/mesh.cpp:17960) — does the
   coarsening path read `vecIn[ghost cg]`?
2. **g2s state transfer (graph → SFC twin)** — `redistributeVec` (include/mesh.h ~3601). Does it read
   ghost cgs, or only local-owned? If only local, it's exonerated. Check the orphan-fill path (keyed by
   cg2dg canonical phys) — does it pull from ghost slots?
3. **`redistributeDVec` in em4_partitioning.h ~132** — note the big comment there re "Fix B: no
   src/dst `performGhostExchange`". So the sandwich deliberately does NOT refresh ghosts. That means any
   consumer that reads a graph ghost during the sandwich gets whatever the last per-step (reduced)
   exchange left — stale for R2/R3. This comment is a strong hint the ghosts are read somewhere.
4. **`broadcastCgValuesByPhysPos`** / **`syncZipNonPrimary`** — consensus/duplicate-CG side-channels
   (include/mesh.tcc). They reference cgs at phys positions; if any target R2/R3 ghost cgs, they'd read
   stale. (EM4 opts out of pos-bcast, but still runs post-axpy sync? verify.)

---

## 6. Diagnostic plan (do these in order)

> [!warning] TEST 10 will likely NOT reproduce this. It re-creates state each cycle from an analytic
> function (`createVector(func)`), so ghost slots are **re-evaluated**, not carried as **evolved** state
> — poisoned ghosts get overwritten before the sandwich reads them. That is almost certainly why TEST 10
> passes while the evolved solver diverges. The bug needs **evolved** state (ghost slots that hold a
> value only obtainable via exchange, not re-evaluable from a closed-form func). So prefer the
> **solver-side per-cg diff (6a-primary)** below; only pursue a testPartitioning poison if you first make
> its state non-re-evaluable (e.g. evolve a few RK steps, or fill ghosts by exchange then poison + sandwich
> WITHOUT re-createVector).

### 6a-primary. Solver-side per-cg diff at the first divergent remesh (the direct answer)
The mesh is **identical** off vs on, so cgs line up 1:1. In the EM4 solver, add an env-gated dump of the
FULL zipped node vector immediately **after** the step-20 sandwich (the first coarsening). Run off and
on, diff per-cg. For each differing cg print: cgIdx, its **phys position** (via cg2dg / E2N_DG decode),
and **level**. The phys positions will cluster in a specific ghost region — that tells you which layer
(R2 vs R3) and which operation pulled from it. Env-gate + phys-bbox scope to keep /tmp small
(`feedback_tmp_usage`). This is the shortest path to "which nodes, read by what."

Then walk the sandwich (`em4_partitioning.h`): dump the vector after **g2s** (graph→twin), after
**IGT**, after **s2g** (redistributeDVec) — whichever stage first shows a diff at those cgs is the
reader. Given "coarsening-only," `interGridTransfer`'s restriction branch (include/mesh.tcc ~4868,
src/mesh.cpp:17960) is the prime target.

### 6a-alt. Poison-through-sandwich in a NEW test
If you want a self-contained regression test: build a graph mesh, **evolve/exchange** so ghosts hold
non-re-evaluable values, poison the excess ghost cgs, run ONE sandwich cycle WITHOUT re-createVector, and
scan the result for NaN — bisecting per stage. Do NOT model it on TEST 10 as-is (see warning).

### 6b. Confirm it's the sandwich, not evolution (cheap)
Run EM4 long-haul with remesh effectively disabled (huge `SOLVER_REMESH_TEST_FREQ` or block-adaptive
fixed grid), off vs on. Expect **bit-identical** over 230 steps → confirms the bug is 100% in the
sandwich. (Strong prior already: steps 0-19 identical.)

### 6c. Refinement vs coarsening
Construct a case that only refines (never coarsens) and one that coarsens. Expect refine-only =
bit-identical, coarsen = diverges. Confirms the restriction path.

---

## 7. Repro commands

```bash
# build (rebuilds dendrolib; EM4 links it via add_subdirectory of dendrolib_dfvk)
cd ~/research/em4_base_cpy && cmake --build build -j$(nproc) --target em4Solver

# EM4 graph param: TOML (NO trailing commas!), set method 3, IO off, remesh/10
#   dsolve::SOLVER_PARTITIONING_METHOD = 3
#   dsolve::SOLVER_IO_OUTPUT_FREQ = 0 ; SOLVER_RK_TIME_END = 30.0 ; SOLVER_REMESH_TEST_FREQ = 10
mkdir -p off/vtu on/vtu
( cd off && mpirun -np 4 em4Solver graph.param.toml )                          # flag OFF
( cd on  && mpirun -np 4 -x DENDRO_MINIMAL_SCATTER=1 em4Solver graph.param.toml ) # flag ON
diff off/em4g_ANALYTICAL_DIFF.csv on/em4g_ANALYTICAL_DIFF.csv   # first diff = row 22 = step 20

# per-step max column divergence:
paste -d'|' off/em4g_ANALYTICAL_DIFF.csv on/em4g_ANALYTICAL_DIFF.csv | awk -F'|' \
 'NR==1{next}{n=split($1,a,",");split($2,b,",");md=0;for(i=3;i<=n;i++){d=a[i]-b[i];if(d<0)d=-d;if(d>md)md=d}if(md>0)print "step "a[1]": "md}'

# unzip minimality test (already in tree):
cd ~/research/dendrolib_dfvk && cmake --build build -j --target testPartitioning
DENDRO_MINIMAL_SCATTER=1 mpirun -np 4 build/testPartitioning 3 7 1e-3 0.1 4 50 4 | grep -A5 "MINIMALITY\|EFFICIENCY"
```
NLSM equivalent: `~/research/nlsm-gr_copy`, `NLSM_PARTITIONING_METHOD=3` (JSON param, not TOML), compares
via `(min, max, l2)` stdout lines (gated behind `NLSM_IO_OUTPUT_FREQ>0`). NLSM long-haul NOT yet run.

---

## 8. Fix paths (pick after diagnosis)

- **(a) Surgical reduction** — from 6a, identify the exact ghost set the sandwich reads, and keep those
  in the per-step map (drop only the truly-unread). Likely still a large reduction, but bit-safe.
- **(b) Restore full map around remeshes** — apply minimal-scatter only to the steady per-step exchange;
  rebuild/keep the full map for the sandwich. Keeps most of the win (evolution is the bulk of exchanges).
  Mechanism: the scatter map is rebuilt every `repartitionMeshGlobal`; you'd need the graph mesh used
  during the sandwich to carry the full map, and switch to reduced only for the RK loop's exchanges.
- **(c) Accept bounded ~3e-7** — it's below graph's own graph-vs-SFC AMR residual (~7.5e-5 on U_E2), so
  it's within the existing noise floor. But it FAILS the project's bit-identity bar; only viable if the
  team relaxes that for this optimization. Not recommended without (a)/(b) attempted.

---

## 9. jj state & revert

Stack (newest first), all this-session:
```
@  DENDRO_MINIMAL_SCATTER — drop unread R2/R3 halo from per-step nodal scatter map   <-- the change
○  minimality check — poison recv-SM excess, prove 68% of graph ghost sends unread
○  full-stencil adjacency opt-in (DENDRO_GRAPH_FULL_STENCIL)
○  C2: route weighted graph partition through fastpart _ex API
...
```
- Revert the change: `jj abandon` the `DENDRO_MINIMAL_SCATTER` commit (keeps the minimality test).
- Default is OFF, so nothing changes in production until the env is set.
- **fastpart** has uncommitted changes too (`_ex` API + full-stencil `oct_element` fields) at
  `~/research/fastpart` — a peer's repo, do NOT commit; needed for the graph build to link.
- **Never commit** in any of these repos — the user owns commits (jujutsu over git).

## 11. RESOLUTION ADDENDUM (2026-07-15, later session)

**Diagnosis** (details in `Repartitioning/Logs/2026-07-15.md`):
- §3's premise was wrong twice: the `iter : N (Remesh triggered)` lines are the *init-grid loop*
  (which never calls `remesh_and_gridtransfer`); the actual diverging remesh (call 1 = step 20) was
  **1240→1520, a refinement**, and the step-10 coarsening was bit-identical. §5's IGT-restriction
  suspect is dead.
- Stage dumps (`EM4_REMESH_DUMP_*`, calls mapped via header-only degenerate-bbox run): **g2s, IGT,
  s2g all bit-identical** — the sandwich never read the dropped ghosts. The reader was the per-step
  machinery on the NEW mesh.
- §4's minimality proof was one-sided: it proved recvSM∖need unread but never checked
  **need∖recvSM**. On EM4's meshes the v1 filter left 2.7k–21.8k unzip-needed slots per rank
  undelivered (NaN-poison → 455–4,713 contaminated unzip outputs/rank): **`isGhostTwo` is a
  fetch-round label, not a geometric ring — block padding references R2-labeled elements.**
- Residual after coverage fix: **mirror last-writer parity.** Plain `readFromGhostEnd`
  (mesh.tcc:1386) has no post-sync re-run, so dropping a Stage-3 mirror pair switches the
  post-exchange last writer of a non-primary local cg from the mirror (DG-tagged sender decode) to
  the begin-sync — ULP-visible on `compute_analytical`'s diff vector (EV itself stayed
  bit-identical at every RK tag).

**Fix (landed, same env gate, default OFF):** in `repartitionMeshGlobal`, always build the FULL map
(`buildNodalSM(nullptr)` lambda), let `buildZipPlan` derive mirror pairs from it, then rebuild with
keepSet = {local-element E2N ghost slots} ∪ {unzip block-padding slots} ∪
values(`m_uiZipNonPrimaryToGhostCg`) ∪ values(`m_uiPassDDemotedToGhostCg`).

**Validation:** testPartitioning default unchanged; flag-on gates green; new two-sided **COVERAGE
TEST** in `partitioningMeshTests.cpp`; runtime `DENDRO_SM_COVERAGE_CHECK=1` + `DENDRO_ZIPPLAN_STATS=1`
diagnostics; **EM4 t_end=30 long-haul (231 steps, 18 remeshes) CSV byte-identical**; sends
447,092 → **142,964** (3.13×; 1.76× SFC; +0.023% over exact unzip-need).

**Open before default-on:** NLSM long-haul (pos-bcast default-ON reads ALL cgs — latent trap),
BSSN long-haul, NLSM np=16 ghost-cut re-measure.

## 10. Reference
- Full write-up: Obsidian `Repartitioning/Experiments/Comms volume and LTS weighting.md` §§G-H (the
  over-communication finding, the fix, and the long-haul CORRECTION callout).
- Memory: `project_comms_harness_lts_weighting.md` (UPDATEs 5-7).
- Related invariants you must not break: `docs/graph_partitioning_correctness.md` (I1-I4); the orphan-fill
  cg2dg key, Fix B, never-IGT-into-graph.
