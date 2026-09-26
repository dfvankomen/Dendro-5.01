# Audit — what s2g (and g2s) cost per remesh, and what to cut (2026-09-25)

Read-only audit: no source edits, no builds. Question: s2g takes 1.9 s at 192 ranks (EM4 d9, ~24–37k
elements, Kingspeak), 1.84 s at 96, 2.35 s at 48. It barely shrinks with rank count, so what is in it,
and what can go? Inputs: `docs/plan_2026-09-25_kingspeak.md`, `plan_2026-09-24_speed.md`, the
handoff, `src/mesh.cpp` (`repartitionMeshGlobal` 19378–21160 and the functions it calls),
`include/mesh.tcc` (`redistributeVec`), `src/meshUtils.cpp` (`createMesh`),
`em4_base_cpy/solver/{include/em4_partitioning.h,src/solverCtx.cpp}` (the sandwich), and fastpart
`dendro-refine` (`src/octree.cpp`, `src/dgraph.cpp`).

## 1. What one sandwich runs (EM4, `solverCtx.cpp` 3175–3380)

| step | code | what it builds / moves |
|---|---|---|
| g2s | `buildSFCTwin` = `createMesh(FDM, blockSetup=false)` of the labelled local elements | 3 `SFC_treeSort`s (dedup, construct, balance) + Mesh ctor with E2E, E2N, nodal scatter map; no blocks |
| g2s | `redistributeDVec(m_uiMesh → sfcTwin, &flags)` | TN-hash rendezvous directory (2 dense rounds), element-wise payload Alltoallv, canon/orphan fill; flags ride along |
| remesh | `this->remesh` → `Mesh::ReMesh` → `new Mesh(…, m_uiIsBlockSetup=true, FDM)` | full SFC mesh **incl. `buildUnzipCanonicalWriterTable` + `buildZipPlan` + `pruneZipSync`** |
| IGT | `DVec::grid_transfer` → `interGridTransfer` | ends with `syncZipNonPrimary` on the new mesh (empty on SFC meshes at prune 2) |
| s2g | `buildGraphTwin` = `createMesh(E2E_ONLY)` of newMesh's local elements + copy owner masks and block info positionally | 3 treeSorts again on an already sorted, balanced, partitioned octree + E2E-only ctor |
| s2g | `graphMesh->repartitionMeshGlobal()` | labelled rebuild (phases below) |
| s2g | `redistributeDVec(newMesh → graphMesh)` + 6 `create_vector` + mpi_ctx realloc + 2 mesh deletes | |

**Is the graph twin redundant with newMesh?** Yes as a mesh: same element set, same SFC partition,
and newMesh already has everything `buildOctantConnectivityMap` reads (E2E, edge/vertex neighbours,
ghost octants, owner masks, block info). It exists only because `repartitionMeshGlobal` works in
place and the s2g transport needs the source layout alive. E2E_ONLY skips E2N, the nodal scatter map
and blocks (the ctor returns before the `[mesh-prof] ctor` print), so the twin is already light: the
cost left is `createMesh`'s three parallel treeSorts, a `MPI_Comm_dup`, and the E2E search/ghost-octant
exchange — 15 ms of 329 ms locally (4.6 %).

**Is the graph partition computed redundantly on every rank?** No. `fastpart_partgraph_octree_ex`
builds a distributed CSR from `vtx_dist` (one Alltoall + Alltoallv of boundary pairs) and every
array is `local_n + ghost_n` or `P`. `BFS_iter_count` is printed under `if(!this->my_rank)` — once
per call, not per rank. With `DENDRO_GRAPH_REFINE>0` (`init_labels` set) the BFS is skipped
entirely (one ghost label exchange), and `RefineByGain` does ≤30 rounds of a `C·P`-double Allreduce
plus neighbour label syncs. Locally `partition-decision` (fastpart + block vote + label hand-off) is
**0.5 % of s2g** (1.9 ms/call; 51 ms on the first call only).

## 2. Local phase profile (np=4, EM4 d7, `graph.param.toml`, 7 remeshes, `bin_def`)

Run: `DENDRO_REPART_PHASE_PROF=1 DENDRO_REPART_ROUNDS=1 DENDRO_ZIPPLAN_PROF=1 DENDRO_GRAPH_REFINE=1`,
second run adding `DENDRO_MESH_PROF=1`. CSV md5 `d7bbee89` (= SFC, as expected for REFINE=1) on both;
`[repart-rounds] total=25` per rebuild; block vote reassigned 0. Means per remesh (first-run numbers;
the phase timer barriers make each phase ≈ max over ranks). Sandwich: g2s 70 ms, remesh+IGT 258 ms,
**s2g 368 ms** (run 2: 50 / 199 / 329). Shares are the thing to read — the box throttles.

| phase (s2g) | ms | % s2g | per-rank scaling (from the code) |
|---|---|---|---|
| twin `createMesh(E2E_ONLY)` | 15 | 4.6 | 3 treeSorts (splitters O(P), sort O(local)), E2E ghost search O(local+ghost) |
| buildOctantConnectivityMap | 0.5 | 0.1 | O(local), Allgather(2P) |
| partition-decision (fastpart refine + vote) | 1.9 | 0.5 | O(local+ghost) per round, Allreduce(C·P) per round, `std::map` vote O(local log) |
| fetch-my-partition (3 ring claim rounds + push) | 0.7 | 0.2 | O(26·local) bytes, 8 Alltoall(v) with O(P) counts |
| keyset ghost fetch | 0.1 | 0.0 | O(local) |
| buildE2NMap(order=2) | 9.1 | 2.5 | O((local+ghost)·27) |
| buildE2NWithSMRepartitioned (expand) | 19.1 | 5.2 | O((local+ghost)·npe) hashes; Pass D (the O(N_global) Allgatherv) is off by default |
| e2n-audit | 31.0 | 8.4 | O(local·npe) on-demand maps |
| nodal-scatter-map | 31.5 | 8.6 | O(ghost nodes), Alltoall + Alltoallv round trip |
| blk: decomposition + setup + E2Block | 1.4 | 0.4 | O(local) |
| blk: buildUnzipCanonWriter | 7.2 | 1.9 | O(padded block volume) |
| **blk: buildZipPlan** | **165.0** | **44.9** | O(claims) = O((local+ghost)·npe); **"winners" = 119 ms (74 %) is `buildZipPlanArbitrateDistributed`: every claim (48 B, local *and* ghost) hashed to a home rank and sent in a dense Alltoallv, WinMsg (28 B) back, a request round** |
| "block-decomposition" tail | 35.4 | 9.6 | minimal-scatter keep set (`unordered_set`, O(local·npe + padding)) + a **second full `buildNodalSM`** + `pruneZipSync` (Alltoallv + Allreduce) |
| rest: `redistributeDVec` (directory + payload + canon fill) + 6 `create_vector` + mpi_ctx + 2 mesh deletes | ~50 | ~13 | directory = 2 dense hashed rounds over all local elements; payload O(local·npe·dof) |

Zip-plan claim counts: labelled rebuild 282k claims/rank, ghost/local = **1.64**; the SFC newMesh
219k, ghost/local = 1.05 (the labelled mesh carries 3 rings of 26-neighbour ghosts, `d ≤ 3` in the
push path, vs SFC's one).

**remesh+IGT (not s2g, but in the sandwich):** newMesh's ctor is e2e 12 + e2n/sm 18 + blocks with
canon ~7 + **zip plan ~105–125 ms** — the zip plan of a mesh that never unzips is ~60 % of remesh+IGT.

**g2s:** `buildSFCTwin` 32 ms of 50 ms (the FDM ctor; treeSorts < 1 ms), the rest the transport.

### Why s2g is flat in P on the cluster (code reading; not yet measured at 192)
- Every s2g phase is O(local + ghost); nothing on EM4's default path is O(N_global) (checked every
  Allgather/Allgatherv in `mesh.cpp`/`mesh.tcc`: Pass D, block-diag, legacy bcast, orphan-fill
  global gather and the non-DIST zip plan are off or empty by default; the orphan gather carried
  0 keys here). At 127–190 local elements per rank a 3-ring ghost shell is 5–9× the local volume,
  so per-rank work shrinks far slower than 1/P.
- The two **dense hash-routed exchanges** (zip-plan arbitration ×3 Alltoallv, redistribute
  directory ×2) send nearly every claim/element off-rank (P−1)/P and off-node 7/8 at 192 ranks,
  whatever the neighbourhood — a local problem made global. Per-rank bytes stay ~constant with P
  (ghost-dominated claims × 48 B ≈ 10–20 MB/rank/build), so this term is flat or rising.
- The same arbitration runs in every SFC mesh ctor, which fits plain SFC's own remesh being flat too
  (1.05 / 0.77 / 0.73 s at 48 / 96 / 192).
- Fit of s2g = a + b·(local+3-ring volume) to np=48/96 predicts 1.6 s at 192 vs 1.95 s measured: a
  P-growing term of ~0.3 s exists. The dense Alltoallvs are the only P-growing candidates found.

## 3. Inherited labelling — what of s2g remains

Idea: after ReMesh, an unchanged element keeps its owner, children inherit the parent's owner,
a coarsened parent takes its children's (e.g. first child's) owner; no partitioner runs except a
periodic rebalance when max/mean padded work > tol.

- **What it removes:** `partition-decision` only — 0.5 % of s2g locally (fastpart refine is already
  cheap; plain fastpart adds BFS ≥12 rounds + 20 diffusion iterations, still small). Everything else
  (E2N, audit, scatter maps, zip plan, blocks, transport) is rebuilding the labelled mesh's data
  structures and stays. **Inherited labels save <5 % of s2g by themselves.** Their value is label
  stability (fewer elements move → smaller push/transport, fewer changed partners) and being the
  prerequisite for an incremental rebuild (reuse unchanged regions), which is where s2g really goes.
- **Block vote still needed:** yes. Blocks are re-derived from newMesh's SFC decomposition
  (`deriveBlockInfoFromBlocks`), and a new block can straddle two inherited owners; the rank-atomic
  block invariant (correctness doc) needs the vote. It is O(local log local), cheap. It also needs
  the one-block-per-rank floor (the np=192 `MPI_Abort(7)`), since coarsening can starve a rank.
- **Input and how to produce it cheaply:** one label per local element of newMesh. (1) g2s: the
  labelled → sfcTwin transport already routes per-element payload by TreeNode and carries flags
  (`DENDRO_REDIST_FLAGS`); add the source rank as one more uint per element → `prevOwner` on
  sfcTwin. (2) remesh: `interGridTransferCellVec(…, CELLVEC_CPY)` exists (mesh.tcc 5829) and handles
  SPLIT/COARSE with one sparse Alltoallv → `prevOwner` per local element of newMesh. (3) feed it to
  `repartitionMeshGlobal` as the label vector (new option or `init_labels` with `refine_rounds=0`);
  rebalance = run the existing refine path with `init_labels = prevOwner` (fastpart's gain
  refinement starting from the inherited labelling, not from the SFC split).
- **Side benefit:** the s2g destination of every newMesh element is its label, known on the source
  rank, so `redistributeVec` in s2g does not need the TN-hash directory at all (cut 6 below) — true
  with or without inheritance (`pushLabels` already is that vector).

## 4. Ranked cuts (saving at 192 ranks per remesh; sandwich now 3.0 s, SFC remesh 0.73 s)

Savings at 192 scale the local shares onto the cluster's 1.95 s s2g / 0.89 s remesh+IGT; they are
estimates until a cluster phase profile exists (step 0).

| # | cut | est. saving @192 | risk | gate | knob |
|---|---|---|---|---|---|
| 1 | **Defer the zip plan on the sandwich's ReMesh product** (newMesh never unzips; it only feeds IGT, masks, block info and the s2g transport). **Measured locally with the existing knob:** remesh+IGT 199 → 65 ms (−67 %), CSV md5 unchanged on both EM4 arms (`d7bbee89` refine, `a75a069e` plain labelled). | **0.5–0.6 s** (of 0.89) | low for EM4; the global knob diverges NLSM (its wavelet decision on a twin) → scope it to sandwich successors | EM4 np=4/8 both arms, testPartitioning ×4, ragged ENUTS 18 digits, BSSN q1 4-arm | now: `DENDRO_LAZY_ZIPPLAN=1` (global, EM4 only); new: `DENDRO_SANDWICH_LAZY_ZIPPLAN` (sfcTwin passes "defer" to its successor like `setBlockSetupFlag`) |
| 2 | **Neighbour-local zip-plan arbitration**: home(pos) = min owner rank of the elements touching pos (known locally for every position whose touching set is inside local ∪ ghost), self-home claims never sent, the rest routed only to neighbour ranks (sparse p2p); hash home kept as fallback for outer-ring ghost positions. Same claims per position, same visit order → same winners. Hits s2g **and** every SFC mesh ctor. | 0.35–0.5 s in s2g (+ similar in the SFC remesh of plain runs) | medium: home consistency for outer-ring ghost claims, origin-rank visit order | a `_CHECK` mode that runs both homes and compares the winner arrays (abort on mismatch), then the usual four | `DENDRO_ZIPPLAN_HOME=1` (0 = hash, default) + `DENDRO_ZIPPLAN_HOME_CHECK=1` |
| 3 | **Filter ghost claims before arbitration**: ghost claims (TN_LEV_NONE) never win; keep only ghost claims at positions this rank also claims locally (mirror pairs) or reads (E2N of local elements, block padding = keep-set parts 1–2). Ring-2/3 ghost-only positions carry most of the 62 % ghost claims. | 0.1–0.2 s (claims −30–50 %) | low–medium (recv/sync lists shrink; prune level 2 already drops those entries afterwards) | compare post-prune sync lists and zip plan with/without; usual four | `DENDRO_ZIPPLAN_CLAIM_FILTER=1` |
| 4 | **Minimal-scatter tail without a second `buildNodalSM`**: filter the recv lists by the keep set and send one sparse drop list to owners, instead of rebuilding the map with its full round trip. | 0.1–0.15 s (tail is 9.6 %) | low–medium | byte-compare send/recv lists vs rebuild (`DENDRO_SM_COVERAGE_CHECK`), usual four | `DENDRO_MINIMAL_SCATTER_FILTER=1` |
| 5 | **Twin without `createMesh`**: in `buildGraphTwin` call `new ot::Mesh(localElems, …, E2E_ONLY)` directly (input already sorted, balanced, partitioned; drops 3 treeSorts + comm dup and makes the positional mask/block copy exact by construction). Next step: in-place `repartitionMeshGlobal` on newMesh with a transport snapshot (local TNs, element-wise DG payload, canonical-position map of local cgs), dropping the twin entirely. | 0.05–0.1 s (direct ctor); ~0.1 s (in place) | low (solver-side); medium (in place) | assert positional TN equality twin↔newMesh; EM4 md5 | `EM4_TWIN_DIRECT_CTOR=1`; later `DENDRO_S2G_IN_PLACE=1` |
| 6 | **Route the s2g transport by labels**, not the TN-hash directory: the destination of each newMesh element is `pushLabels[i]`, already on the source rank. Removes 2 dense hashed rounds. | 0.02–0.08 s | low (exact by construction) | redistribute route check vs directory; EM4 md5 | `DENDRO_REDIST_LABEL_ROUTE=1` |
| 7 | Hash-bound O(local+ghost) phases — e2n-audit 8.4 %, E2N expand 5.2 %, nodal SM 8.6 %: sort+fold instead of `unordered_map` inserts, as round 2 did for the zip plan. | 0.1–0.2 s combined | low | usual four | per-phase, byte-identical |
| 8 | Inherited labelling + rebalance-on-tolerance (section 3) | < 0.05 s directly | medium (drift of balance, empty ranks) | LTS eff / padded imbalance over a long run; usual four | `DENDRO_INHERIT_LABELS=1`, `DENDRO_INHERIT_REBAL_TOL=1.05` |
| — | **BSSN/NLSM only:** pos-bcast is left on for their labelled meshes, so the first bcast after every rebuild runs `buildBcastPlan` — an **O(N_global) Allgatherv of all local+ghost cgs (5 arrays) on every rank** (`mesh.cpp` 21278), called from their `redistributeDVec`. EM4 opts out. | EM4 0; BSSN: likely the largest single s2g term at scale | medium | BSSN q1 4-arm | distribute it like `DENDRO_ZIPPLAN_DIST` (hash or neighbour home) |

Cuts 1–4 together target ~1.1–1.4 s of the 2.9 s the labelled path spends per remesh beyond g2s;
with them, labelled remesh+IGT (0.3 s) would sit below plain SFC's remesh (0.73 s), and plain SFC
gains from 2 as well. Not a cut: fastpart (0.5 %), the fetch rounds (0.3 %), the block vote.

## 5. Plan (each knob default off, byte-identical, then flipped after the gate)

0. **Measure the split at scale first** (needs the user's OK for Kingspeak): one `refine` arm at np=48
   and np=192 with `DENDRO_REPART_PHASE_PROF=1 DENDRO_ZIPPLAN_PROF=1 DENDRO_MESH_PROF=1
   DENDRO_REPART_ROUNDS=1` (+ a second arm with `DENDRO_LAZY_ZIPPLAN=1`). This decides between the
   "ghost volume" and "dense all-to-all" explanations and re-ranks 2–4.
1. Locally, np=8 EM4 gate of `DENDRO_LAZY_ZIPPLAN=1` on both arms (`d7bbee89` / `32eb202e`); if green,
   EM4 runs can use it now. Then implement the scoped `DENDRO_SANDWICH_LAZY_ZIPPLAN` and gate NLSM
   ragged ENUTS + BSSN q1 with it.
2. `DENDRO_ZIPPLAN_CLAIM_FILTER` (3) — smallest code change of the zip-plan cuts, shrinks 2's input.
3. `DENDRO_ZIPPLAN_HOME` with its check mode (2).
4. `DENDRO_MINIMAL_SCATTER_FILTER` (4), `DENDRO_REDIST_LABEL_ROUTE` (6), `EM4_TWIN_DIRECT_CTOR` (5).
5. Inherited labels (8) only once 1–4 have moved s2g; it is the entry point to incremental rebuilds,
   not a saving by itself.

## Evidence and files
- Runs (scratchpad `agent_s2g/`): `run.log` (phase + rounds + zip-plan prof), `run2/run.log` (+ mesh
  prof), `lazy/run.log` (REFINE=1 + `DENDRO_LAZY_ZIPPLAN=1`, md5 `d7bbee89`), `lazy_plain/run.log`
  (plain method 3 + lazy, md5 `a75a069e`), `agg.py` (per-phase means).
- Sandwich means (ms, 7 remeshes): run2 g2s 50 / remesh+IGT 199 / s2g 329; lazy 50 / **65** / 329;
  lazy plain 50 / 66 / 394.
- Cluster (Kingspeak A, `refine` arm, last 3 remeshes): s2g 2404 / 1895 / 1948 ms at np 48 / 96 / 192;
  remesh+IGT 1277 / 954 / 890; plain SFC remesh per remesh 1.05 / 0.77 / 0.73 s.
