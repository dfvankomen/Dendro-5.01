# Plan: fast repartition communication, base case (2026-09-29b)

Goal (user): moving pieces during a repartition/remesh must be extremely fast, for global RK
(EM4 / BSSN / NLSM-GTS). LTS work is parked.

Source: read-only Opus ledger 2026-09-29 (EM4 np192, runs W/T). Numbers are from logs, unverified
until each step lands. Rules: opt-in `DENDRO_*` knobs, default byte-identical, ragged arms in
every gate, one heavy run at a time (>= 20 GB free), jj commits in dendrolib only (one-line),
solver repos stay uncommitted.

## Where the time is (EM4 np192, W 09-27)
- labelled remesh 0.87 s vs SFC 0.23 s, 11 remeshes = ~9.6 s of the 12.4 s gap.
- per remesh: g2s 0.13-0.18 s, remesh+IGT 0.15-0.28 s (SFC pays too), s2g 0.61-0.81 s
  (rebuild TOTAL 0.45-0.64 s). Each transport hop ~50 ms, of which only 6-11 ms is data exchange.
- ~9 MB/rank moves per remesh (~25 ms wire). ~145 labelled-only collectives (~70 dense A/Av).
- ghost shell g ~11x n (R1+R2+R3); the rebuild's local work scales with it.

## Steps
**C0 — measure.** PMPI shim (LD_PRELOAD, harness-side, no dendrolib edit): per-call count,
bytes, time, grouped by a phase tag, inside the sandwich. Fresh Kingspeak EXP R (EM4 np192,
REPART_PHASE_PROF/ROUNDS, REDIST_PROF, ZIPPLAN_PROF, MESH_PROF). Microbench dense A+Av vs NBX.

**K1 — `DENDRO_REMESH_KEEP_SFC=1`.** Keep ReMesh's SFC successor alive; the next g2s transports
into it (its local TNs == m_uiRepartPreTN) instead of rebuilding the SFC twin. `_CHECK=1`
builds the twin anyway and asserts equal local TNs / splitters. EM4 solverCtx.cpp ~3043, 3175,
3223, 3259, 3296-3304; NLSM analog. Gates: EM4 md5 np4/8 both arms, nuts2, lw3 np16 4 arms.
RAM: one extra mesh.

**K2 — `DENDRO_S2G_DIRECT=1`.** Repartition from newMesh without the E2E_ONLY graph twin
(`repartitionFrom(const Mesh&)`). mesh.cpp ~19994, mesh.h ~4130, em4_partitioning.h:55-116.
`_CHECK` compares E2N/SM/zip-plan arrays against the twin path.

**K3 — `DENDRO_REDIST_LEAN=1`.** One count exchange + packed per-peer buffer, splitter-routed
orphan queries for SFC sources (no bitmap Allgather), begin/end split to overlap the DG send
with ReMesh. mesh.tcc ~6921-6961, 7340-7463.

**K4 — `DENDRO_REPART_NBX=1`.** Sparse NBX (Issend/Iprobe/Ibarrier) helper replacing dense count
Alltoalls in the rebuild and transport; rank-ordered placement keeps bits.

**K5 — `DENDRO_REPART_R3_SLIM=1`.** R3 as topology-only records, out of node expansion / SM walk.
Gate tp 7b first (what broke RINGS=2).

**K6 — `DENDRO_REPART_CANON_KEYS=1`.** Push each element's canonical owner keys from the SFC
successor's E2N; labelled E2N_CG by key join, no cascade/audit; then retry RINGS=2.

**Deferred:** direct one-hop IGT into labels (Design A) — only if K1-K6 leave > 0.3 s/remesh.

## Log

### C0a Kingspeak baseline R (2026-09-29), EM4 np192, job 28495096, snapshot 3184d431, 2 rounds
- Wall: sfc 49.8 / 50.0, graph (method 3 + REFINE=1) 59.1 / 58.8 -> 1.18x (W 09-27: 1.24x).
- remesh(oct+LB) total: graph 10.37 s vs sfc 2.20 s over 12 remeshes = 8.2 s of the 9.0 s gap; rkstep 36.05 vs 34.47 s (+1.6 s, 4.6 %).
- Per labelled remesh: g2s 109-133 ms, remesh+IGT 140-400 ms (~SFC's), s2g 313-418 ms; [repart] TOTAL mean 248 ms (16 rebuilds).
- Rebuild phases summed over 16 rebuilds: e2n-audit 924 ms, E2N expand 475, E2N map 261, connmap 142, partition-decision 108, nodal SM 70, fetch 66, block-decomp 59 (zip plan sub-phases not in this sum).

### C0 (2026-09-29) PMPI shim landed: ~/research/repart_gate/pmpi/
libdpmpi.so (LD_PRELOAD, plain C, README + Makefile + run_em4.sh; sample tables sample_em4_np4_{sfc,graph}.txt). Counts/bytes/time
per MPI call type, whole run and "remesh windows" = bursts of Comm_dup/split/create/free (gap 0.5 s, DPMPI_GAP); pending-accumulator
makes the window end at the last comm event exactly. Also p2p peers/rank, Alltoallv nz-peers, dense-collective size histogram, per-window rows.
EM4 np4 md5 shim on/off identical (sfc d7bbee89, graph 2603beb3). Local np4 remesh windows w1..w7 (excl. initial w0), rank-0:
sfc ~7-17 MB, 112 calls, ~1-3 ms MPI; graph 25-54 MB, 575 calls, 12-35 ms MPI. Not measured: time before the first dup (g2s hop).

### C0b Kingspeak dpmpi shim, EM4 np192 (job 28495108, 1 round, REPART_PHASE_PROF on)
- Per remesh window (rank 0): SFC ~0.15 s / 0.08 s in MPI / 3-4 MB / ~128 calls / 20 comm events; labelled ~0.65 s / ~0.33 s in MPI / ~15 MB / ~1450 calls / 78 comm events.
- Labelled remesh windows, per rank over 16 windows: Alltoallv 1474 calls 241 MB 2.15 s; Allreduce 3169 calls 1.28 s; Barrier 304 calls 1.21 s (phase-prof barriers inflate this); Alltoall 1383 calls 0.91 s; Allgather 437 calls 0.61 s, 25 MB received (= S3 orphan bitmap 4 KB x P, 2 per remesh); Comm_dup 399 calls 0.32 s. SFC windows: MPI 3.0 s total.
- Verdict: bandwidth is trivial; cost = number of synchronising collectives (~200 Allreduce, ~86 Alltoall, ~92 Alltoallv, ~27 Allgather per remesh). ~1400 dense count exchanges of 512 B-1 KB. Next: per-call-site attribution in the shim.

### C0b-sites (2026-09-29) per-call-site attribution in the shim
dpmpi.c now unwinds (backtrace, 5 app frames, lib/par:: frames dropped) for collectives + Comm_* inside remesh windows only; rank 0 writes
dpmpi_sites.txt (top 60 by time and by count); `symbolize.sh <bin>` resolves with addr2line -f -C and re-aggregates by first non-par::/SFC:: frame.
EM4 np4 graph md5 unchanged (2603beb3), sfc d7bbee89. Release build has no -g (only function names; `??:?` lines) -- add -g for file:line.
Local np4 graph top sites: redistributeVec Alltoallv/Allgather/Alltoall/Allreduce (transport), buildZipPlanArbitrateDistributed Alltoall,
buildE2EMap Mpi_Alltoall/Allgather (Mesh ctor), createMesh Reduce, computeNodalScatterMap4 Alltoall/Comm_dup, repartitionMeshGlobal.
Sample: ~/research/repart_gate/pmpi/sample_sites_graph_np4.txt. Bundle copy: ~/research/repart_gate/kp/bundle/pmpi/.

### C0c call sites, EM4 np192 (job 28495110, rank 0, 16 windows, no phase-prof barriers; symbolize with addr2line at 0x400000+offset, binary is non-PIE)
- Twin mesh construction (createMesh -> Mesh ctor via buildGraphTwin/buildSFCTwin): buildE2EMap Alltoall 195 calls 0.89 s; SFC_treeSort Comm_dup 49 calls 0.77 s (~21 ms each); SFC_treeSort Alltoallv/splitComm2way Allgather 0.67 s; Mesh-ctor Alltoallv_sparse 0.34 s; createMesh Reduce 0.18 s; computeNodalScatterMap4 Allgather 0.18 s. Total ~3.0 s = ~190 ms/remesh -> K1 + K2.
- repartitionMeshGlobal ~1.0 s (one Alltoall 16 calls 0.70 s = 44 ms each: absorbs preceding compute imbalance; push Alltoallv 26 MB 0.21 s).
- zip-plan arbitration ~0.6 s; redistributeVec ~0.55 s (Allgather bitmap 0.22, Allreduce 0.21, data Alltoallv 97 MB 0.06 s); fastpart 750 Allreduce 0.03 s.

### K1 (2026-09-29): DENDRO_REMESH_KEEP_SFC landed in EM4 (uncommitted; hunks work/k1/em4_k1.patch)
- EM4 only, no dendrolib change. `SOLVERCtx::m_keptSFC` + `m_keptSFCFor` (labelled mesh it belongs to). s2g end keeps `newMesh` instead of deleting; init swap keeps the ctor SFC mesh; next g2s adopts it (Allreduce agreement, pointer-token guard) instead of `buildSFCTwin`, else falls back. `_CHECK=1` builds the old twin and MPI_Aborts (rc 77) on element/ghost/splitter/node-count mismatch.
- Gates: md5 np4/np8 sfc+graph knob off = on = _CHECK (d7bbee89 / 2603beb3 / d7bbee89 / 32d48b0b); np16 ragged (method 3 + GRAPH_REFINE=0: 0faf9dbb; method 4 + GRAPH_REFINE=2: d7bbee89) off = on = _CHECK. 7/7 remeshes reused=1 (incl. the first, from the init mesh), CHECK 0 mismatches.
- g2s (np8 graph, box throttled): ~40-56 ms -> ~14-29 ms (twin_build 22-27 ms saved per remesh; np192 expectation ~85-90 ms). RSS np4 max child 631 -> 618 MB (no visible cost at this size).
- NLSM analog: nlsmCtx runs the wavelet decision ON the twin (full-block twin required), so keep is only valid if the kept S_new has full blocks + built zip plan (lazy successor must be forced/ensureZipPlanBuilt before the wavelet) -- and its unzip state must be fresh; not done here.

### K1 Kingspeak EM4 np192 (job 28495686, dendrolib 313fabd4 + EM4 K1 patch, 2 rounds)
- Wall: sfc 50.0/49.6, graph 58.4/58.8, graph+K1 57.0/57.4 (1.18x -> 1.15x SFC), graph+K1+CHECK 58.4/58.9.
- g2s mean 125 ms -> 37 ms (-88 ms/remesh); [graph-remesh] total over 15 remeshes 9.7-9.8 s -> 8.5-8.6 s. CHECK: 15/15 remeshes check=ok at np192. EM4 CSV md5 graph = graph+K1 = CHECK = 68191978 (sfc af968a0b; graph != sfc as at np4/np8).
- Kingspeak trap found: zsh did not word-split the rsync exclude variable, so local build*/ dirs and .git pointers were shipped (all the stale-CMakeCache / CheckGit failures today); ship under bash.

### K2 (2026-09-29): DENDRO_S2G_DIRECT landed (dendrolib d66ca207 + EM4 hunk work/k2/em4_k2.patch, uncommitted in EM4)
- Design: the in-place rebuild makes a separate-source `repartitionMeshGlobal` a big refactor, so took the "next best": `Mesh::repartitionFrom(src, method, dmin, dmax)` = `Mesh(src, E2ETwinTag)` (mesh.cpp, new) + mask/blockInfo injection + in-place repartition. The clone copies the E2E-level state from newMesh's finished arrays (AllElements, E2E, local/ghost ranges + counts, LocalSplitter, R1 ghost index, element-DG scatter maps/proc lists/send buffer) with one Comm_dup; IsNodalMapValid is recomputed as buildE2EMap does; m_uiEmbeddedOctree stays empty (buildE2EMap swaps it out). No createMesh, no 3 SFC_treeSorts, no buildE2EMap Alltoalls, no nodal scatter map, no createMesh Reduce.
- Twin state repartitionMeshGlobal reads: AllElements + E2E + local range + R1 ghost/element-DG exchange (buildOctantConnectivityMap, mesh.h ~4253), ownerMask/blockInfo (injected), `m_uiLocalBlockList.empty()` for padW (clone keeps it empty like E2E_ONLY, so padW is identical); everything else (E2N, SM, blocks, zip plan) is rebuilt from scratch at mesh.cpp ~21097-22000. newMesh has all of it (only buildE2EMap writes the ghost element set; lazy zip plan irrelevant here). Falls back to the twin path (collective, printed) when src is not active on all ranks / npes<=1.
- `_CHECK=1` (EM4 buildGraphTwin): also builds the twin path, `Mesh::diffKeyArrays` compares element lists, E2E, R1/element scatter, E2N_CG/DG, CG2DG, nodal scatter maps + counts, zip plan (Cg, UnzipIdx, sync lists, dup lists, canon writer), E2BlkMap, block list (per-block fields), owner mask, block info, repart hints; MPI_Abort(77) on any diff. First run caught EmbeddedOctree (fixed). Fires on init + every remesh (8 rebuilds at np8/np16).
- Gates (knob on, with and without KEEP_SFC): EM4 md5 np4/np8 sfc+graph = refs (d7bbee89 / 2603beb3 / d7bbee89 / 32d48b0b); np16 ragged (m3, REFINE=0) 0faf9dbb and levelwise (m4, REFINE=2) d7bbee89, on / on+keep; CHECK (with and without KEEP_SFC) 0 mismatches in 8/8 rebuilds on all of np4/np8/np16 arms. tp gate NP=4 and NP=8 IDENTICAL x4 arms; nuts2 == ref.
- s2g (box throttled, relative; mean of 7 remeshes, np8 graph): off 221 ms -> on 126 ms (-95 ms; rebuild call itself 52-83 ms). np16 (m3 / m4): 2056 -> 254 ms / 1813 -> 206 ms (oversubscribed-ish, absolute unreliable). Expect ~190 ms/remesh at np192 per C0c (Comm_dup/treeSort wait time is the dominant term).
- NLSM: its graph twin is a different helper (nlsmCtx buildGraphTwin analog, wavelet decision on the twin); reuse `Mesh::repartitionFrom` there needs the same mask/blockInfo copy sites and a full-block source; not changed.
- Risk: clone omits E2E_ONLY's leaked raw arrays (m_uiSendKeyCount..., uninitialised in both, never read after build) and m_uiKeys; the clone ties comm to src's global comm dup.

### K1+K2 Kingspeak EM4 np192 (job 28495694, dendrolib f47c4fcc + EM4 K1/K2 patches, 2 rounds)
- Wall: sfc 49.1/49.2, graph 58.2/58.5, +K1 57.0/57.7, +K1+K2 55.9/56.9 (1.19x -> 1.15x SFC). md5 graph = +K1 = +K1+K2 = 68191978.
- Per-remesh mean (ms) g2s/IGT/s2g/total: graph 124/163/353/641; +K1 36/164/358/559; +K1+K2 37/151-206/300/489-543. remesh(oct+LB): 9.6 -> 8.4 -> 7.3-8.2 s (SFC 2.3 s).
- Remaining gap ~0.35 s/remesh: s2g 300 ms (rebuild ~250 + transport ~50) -> K3 (transport), K5/K6 (rebuild).

### K3 (2026-09-29): DENDRO_REDIST_LEAN=1 landed opt-in (dendrolib cc85d8b2, include/mesh.tcc redistributeVec only; no solver change)
- Hint agreement: no Allreduce. Each rank computes its hint code locally; the code rides in the step-3 header Alltoall. All equal and >0 -> proceed; else every rank (same view) runs the directory (`runDirectory` lambda, formerly inline), repacks and repeats the header exchange. `_HINT_CHECK=1` keeps the old Allreduce path (needs the directory alongside the hint).
- Step 3: 3 count Alltoalls + 3 Alltoallv -> 1 header Alltoall {tn count, hint code, src-is-sfc} + 1 Alltoallv. Flag/DG counts are npe*n and npe*dof*n, so no exchange. Per peer one chunk [TNs][DG][flags], regions padded to 8 B, counted in 8-byte words (same 2^31 limit as before, aborts 79 if hit). Receive side memcpys back into recvTN/recvDG/recvFlag, so the write passes are untouched.
- Orphans (canon/orphan fill), source SFC on every rank (`src-is-sfc`: !m_uiIsLabelledRebuild, splitters == local first/last element): no count Allgather, no bitmap Allgather. Each orphan is queried at the owners of the up-to-8 finest cells touching its canonical position (binary search over the Hilbert-ordered splitter firsts, TreeNode operator<), own rank skipped as before. Queries and replies go as sparse Isend/Irecv (only pairs with traffic); replies are positional (1 flag byte + dof values per query, in query order), so no reply count exchange and no dst-cg payload. Write order unchanged (own orphans first, then ascending peer, query order). Labelled source (g2s hop, unchanged bitmap filter): orphan count folded into the bitmap Allgather (1 Allgather instead of 2), zero-orphan early return kept; queries P2P, replies positional.
- Rounds per hop (collectives, rank 0): with orphans off 1 Ar + 6 A + 6 Av + 2 Ag = 15; lean s2g (SFC src) 2 A + 1 Av = 3 (+ sparse P2P); lean g2s (labelled src) 2 A + 1 Av + 1 Ag = 4. No orphans: off 8 (1 Ar + 3 A + 3 Av + 1 Ag), lean 3. PMPI shim, EM4 np8 graph whole run, 15 hops, rank 0: Alltoall 848 -> 802, Alltoallv 1019 -> 965, Allgather 276 -> 260, Allreduce 1131 -> 1116 (-131 collectives, ~8.7/hop; np8 has few orphan hops, np192 had 2 bitmap Allgathers/remesh so expect more). md5 32d48b0b both.
- Local np8 (shared memory, sync rounds ~free) REDIST_PROF sum over 15 hops: off 348 ms, lean 380 ms (+2 ms/hop: extra pack/unpack memcpy; uninitialised buffers cut it from +4 ms). No np192 timing yet; the win there is rounds, must be measured on Kingspeak (needs OK).
- Deviation: SFC-src lean does not early-return on "no orphans anywhere" (needs the global count); it then runs the residual zero-fill, which only differs from old when a dst-local cg is unwritten AND no orphans exist anywhere (old left it undefined).
- Debug knobs: `DENDRO_REDIST_LEAN_CHECK=1` also runs the bitmap orphan path and aborts (78) on any orphan slot that differs bitwise ([redist-lean-check] lines: 0 mismatches, np4/np8/np16 m3); `DENDRO_REDIST_LEAN_FORCEMISS=1` forces rank 0's hint to miss so the fallback path runs (15/15 hops fell back at np16, md5 unchanged).
- Gates (lean, and lean+KEEP_SFC+S2G_DIRECT): EM4 md5 np4/np8 sfc+graph = refs (d7bbee89/2603beb3/d7bbee89/32d48b0b) incl. forcemiss+check; np16 ragged m3 REFINE=0 0faf9dbb and levelwise m4 REFINE=2 d7bbee89 (on, on+keep+direct, check, forcemiss+check); HINT_CHECK+LEAN_CHECK np4/np8 mismatches=0; tp NP=4/NP=8 IDENTICAL x4 arms both combos; nuts2 == ref; lw3 np16 sfc/lvl/lw/fp min-line md5 = lw3_s3on.
- Not done: begin/end split (DG send overlapping ReMesh). The DG values are the source vector at the sandwich, ReMesh+IGT runs between g2s and s2g on the SFC twin, so an overlap needs the send posted before ReMesh and the dst mesh not yet built (recv side needs dst E2N); it is a driver-level restructure, not cheap. Not attempted.
- Risks: candidate-rank set assumes a rank holding a local cg at P owns a local element touching P (CHECK catches misses); a mixed hint/no-hint run pays one extra header exchange + directory; 8-byte word padding leaves uninitialised pad bytes on the wire.
- Recommendation: flip default after one Kingspeak np192 A/B (wall + shim collective counts); `=0` reverts.

### K3 Kingspeak EM4 np192 (job 28495719, dendrolib 8dd336df, 3 rounds) + default flip
- Wall: sfc 51.2/52.0/50.3; graph+K1+K2 68.4(outlier, IGT 623 ms mean)/57.4/57.9; +K3 57.9/57.1/57.6 (~1.13x SFC). md5 68191978 all graph arms.
- Per-remesh mean total (r2/r3): K1+K2 517/531 ms, +K3 512/504 ms; s2g 305/309 -> 308/297; g2s 37-39 -> 35-36. Wall-neutral at np192 (within noise); keeps the win for P scaling (no 4 KB x P bitmap Allgather, ~11 fewer collectives/hop).
- Flipped DENDRO_REDIST_LEAN default on (=0 reverts). Knob-less gates: EM4 np4/np8 sfc+graph d7bbee89/2603beb3/d7bbee89/32d48b0b; tp NP=4 4 arms IDENTICAL; nuts2 == ref.

### K6 (2026-09-30): DENDRO_REPART_CANON_KEYS=1, labelled E2N by SFC owner-key join (workspace dendrolib_dfvk_ws_k6, on 8dd336d)
- Design probe (read-only first): for every labelled slot the cascade+audit canonical position equals the SFC source's (0 misses in
  local/R1/R2 over all tp arms and EM4 np4), but the owner element differs at 0.1-1 % of slots on real labellings, all at positions
  where the labelled cascade made duplicate cgs (22-307 positions/rebuild) or picked another owner; the SFC keys are globally
  consistent (0 positions with >1 owner). R3 existed only to keep R2 rows' cascade (and their SM / mirror entries) right.
- Change: at push time every element carries per node the SFC owner (owner TN + sub, from CG2DG[E2N_CG] of the SFC source:
  `repartitionFrom`'s src, an in-place FDM mesh, or an E2E_ONLY twin whose SFC E2N is built on the spot). Labelled E2N =
  key join (owner dg -> sorted unique numbering, node ranges from element ranges); no buildE2NMap, no expansion/orphan passes,
  no audit. Records for all rings: the 26-lists miss finer face neighbours, so the unzip reads some round-2/3 elements
  (first try keyed only local+R1 -> COVERAGE starved). Owners outside the fetched set (outer rim) stay self-owned (counted).
  SM: CG path only (no R2/R3 DG path). Collective fallback to the cascade when no SFC source exists.
- `_CHECK=1`: also runs cascade+audit, prints per-ring slot diffs vs keys, aborts 77 on a canonical-phys miss in local/R1 or a
  keyed-E2N != key. Result everywhere (tp np4/np8 x4 arms, EM4 np4/np8/np16 ragged, NLSM nuts2): keyed-vs-keys 0/0/0,
  cascade-vs-keys phys 0, owner misses e.g. tp m3 np4 2874/904k local, 1611/202k R1; ragged 604/71k, 617/62k.
  Round-2/3 rows (R2 class): up to ~4k/rebuild slots keep a self-owned fallback where the owner lies outside the fetch (outer
  rim, reported as R2 phys misses, non-fatal); COVERAGE shows none of them is read.
- Correctness (new refs expected; user rule 2026-09-30: correct final solutions, not bit-identity to old labelled output):
  - EM4 AMR (t_end 10): graph md5 np4 2603beb3 -> d7bbee89, np8 32d48b0b -> d7bbee89, np16 ragged (m3 REFINE=0) 0faf9dbb ->
    d7bbee89 = the SFC md5 in every arm (levelwise m4 REFINE=2 was already d7bbee89, unchanged); also with S2G_DIRECT and
    with _CHECK. CSV max rel diff vs SFC: before 2.1e-4 / 2.7e-4 / 5.0e-4 (MAE columns), after 0.
  - EM4 fixed mesh (no AMR, np4): graph md5 4d24ebe8 (CSV max rel 7.2e-5 vs SFC) -> a5551721 = SFC.
  - Pointwise (new DENDRO_POS_DUMP_DIR, local cgs by canonical position, steps 10/40/70): local state was already bit-identical
    to SFC before K6 (AMR np4/8/16, no-AMR np4); the old CSV gap was duplicate local cgs counted twice in the norms.
  - tp np4+np8 x4 arms: 0 FAIL, COVERAGE 0, 7b/8/9/10/11 = base; 8bb/8bc/8bd zip roundtrip mismatches (m3 238/343/238, ragged
    np8 4/30/25) -> PASS; 8br dupPhysPos and 8bt disagreeing -> 0; 8bv 596 -> 413 (np4 m3; the test keys by the slot's own
    position, SFC ragged also shows 32).
  - BSSN q1 np4 graph constraints max rel diff vs SFC 7.8e-5 -> 1.0e-13 (md5 Constraints a83112e3 -> bf867af6; BH/GW md5 unchanged).
  - NLSM nuts2 (ENUTS d9) min/max = ref, unchanged; lw3 np16 sfc/lvl/lw/fp min/max = SFC, l2 moves in the 16th digit.
- Zip-sync (EM4 np8): dup_cgs 2.1 -> 0, mirror pairs 6.8 -> 0, pruned sync lists 13.6 -> 0 per rank (1 pair left on 2 of 8
  rebuilds, global_active stays 1 there), znp/passd mirror 0, local dup 0. STEP_PROF (1f1f107e on top, 2 rounds): sync:comm
  1.60 -> 0.07 s, gb:sync_pre 1.17 -> 0.045 s, post-axpy 0.33 -> 0.013 s, but ge:wait 1.0 -> 2.1 s; evolve 16.45/16.53 ->
  16.50/16.55 s (no local wall change: the sync was absorbing imbalance, the wait moved into the ghost exchange).
- Rebuild (EM4 np8 REPART_PHASE_PROF, 8 rebuilds): TOTAL mean 94 -> 55 ms; audit 264 + expand 70 + e2nmap 49 ms -> join 46 +
  index 2 ms; fetch 8 -> 35 ms (key payload, ~1.9 KB/element at p=6, transient). Ghost set unchanged (R1-R3 still fetched).
- K6c (GHOST_RINGS=2 + owner targets, experimental, not passing): owner targets splitter-routed to the SFC source and keyed;
  owners all resolved and COVERAGE clean, but tp m3 7b/8 get 22/20 large errors: the unzip padding needs elements that only the
  3rd ring delivered (26-lists incomplete across level jumps). Needs a geometrically complete neighbour fetch first.
- Knobs: DENDRO_REPART_CANON_KEYS (default 0), _CHECK; DENDRO_POS_DUMP_DIR/_STEPS (ets.h). Default path untouched.

### K1/K2 port to NLSM + BSSN (2026-09-30), default on, `=0` reverts; no dendrolib change (uncommitted in the solver repos; patches work/k12port/{nlsm,bssn}_k12.patch)
- Same env names / proof lines as EM4: `DENDRO_REMESH_KEEP_SFC[_CHECK]`, `DENDRO_S2G_DIRECT[_CHECK]`, `[keep-sfc] reused=..`, `[s2g-direct] rebuild n`, `[s2g-direct-check] .. 0 rank mismatches`. s2g uses `Mesh::repartitionFrom`; consensus pos-bcast stays ON (EM4 turns it off; NLSM/BSSN need the full machinery), BSSN re-registers phantom fill on the direct mesh.
- BSSN: verbatim EM4 shape (m_keptSFC/m_keptSFCFor, kept at init swap + s2g end, adopt in g2s).
- NLSM differs: the wavelet runs ON the twin, so there are 4 adopt sites (init top, init-refine, is_remesh, g2s) behind one `acquireSfcTwin(site)`; the kept mesh is a full-block ReMesh successor, adopt forces `ensureZipPlanBuiltPublic()` (lazy successor plan; wavelet unzips it; ~70 ms, vs 160-5000 ms twin build). Kept at: init STAY_SFC label-once, non-stay init s2g, remesh_and_gridtransfer s2g, and NLSM_REMESH_FALSE_RESTORE's no-remesh swap-back (flags reset to NO_CHANGE, re-stashed for the restored labelled mesh). `NLSM_INIT_STAY_SFC=0` composes too.
- NLSM CHECK is strict (elements/ghosts/splitters/block count) only without a weight function. With LTS weights (m_getWeight) the kept mesh is split by the weights of its ReMesh, the rebuilt twin by the current lmin/lmax weights, so rank slices legitimately differ (strict CHECK aborted rc=77 at lvl/fp np16); then CHECK compares the global octree (count + order-independent hash), prints `check=ok(set)`. Slice difference only costs the redistribute hint; results identical.
- Gates NLSM (nlsmNUTS): nuts2 np4 off = on = CHECK = ref. lw3 np16 sfc/lvl/lw/fp min-line md5 off = on = lw3_s3on (cfd49144/3a3e30dc/2f5728f0/bd0f2007); CHECK lvl/fp(set)/sfc(strict) rc=0, 0 mismatches. GTS global-RK (ts_mode 1, d9, np4, 8 remeshes, every-step dump md5 3ba98688): off = on = CHECK, also with FALSE_RESTORE=1 and STAY_SFC=0 (16 adopts, 16 check=ok); freq 3 (26 remeshes) FR off = on = CHECK = non-FR 71b376e1. Not exercised: FALSE_RESTORE no-remesh branch (every is_remesh test on d9 refines; tol>=1e-2 hangs a 176-element np4 mesh in the knob-off arm too).
- Gates BSSN (q1 tinytest, refinement 3, np4, graph): off = on = CHECK = ref md5 7a18bafe/a83112e3/86470a02/b50bc05f, C_HAM L2 9.20617e-06; CHECK 10/10 s2g-direct 0 mismatches, 10/10 keep-sfc check=ok (twin_build ~80 ms vs adopt ~1.3 ms).
- Timings (box shared, relative): BSSN [graph-remesh] mean of 9 g2s/IGT/s2g/total off 347/237/618/1202 ms, on 237/226/555/1018 ms. NLSM lw3 np16 (oversubscribed) [graph-remesh] total off 8.3/15.7/9.9 s -> on 5.2/8.2/7.0 s (lvl; fp same shape), g2s 6.0 -> 1.1 s, s2g 4.7-5.5 -> 1.4-2.5 s; nuts2 np4 g2s 172 -> 92 ms, s2g 276/581 -> 246/531 ms.
- Risks: adopted kept mesh holds one extra SFC mesh of RAM; NLSM restore-branch stash untested at runtime; BSSN/NLSM labelled refs unchanged (no K6 here).

### TAX + scaling, Kingspeak (2026-09-30; EM4, dendrolib 1f1f107e + EM4 K1/K2 default on)
- TAX np192 small case (job 28497505): evolve sfc 34.8 s, graph 36.7 s; zip-sync costs mean 0.37 s (sync_pre 0.21 + post 0.11 + postaxpy 0.05; max up to 2.9 s on a few ranks). DISABLE_POST_SYNC / +POST_AXPY_SYNC (timing only) did not help (37.6 / 37.1 s). So at np192 the syncs are not the per-step tax; K6's per-step ceiling there is ~0.4 s.
- Scaling, em4big (d12, tol 1e-10, 143.5k elements, t_end 2.5), K1-K3 on, 2 rounds:
  np192 (jobs 28497506): wall sfc 197.4/197.0, graph 209.8/209.4 (1.06x); rkstep sfc 161.3, graph 160.7-161.0 (no per-step tax); remesh total sfc 7.1 s, graph 22.1-22.5 s; per labelled remesh g2s 166 / IGT 470-500 / s2g 846-850 / total ~1.5 s vs SFC ~0.47 s.
  np384 (28497507): wall sfc 112.8/112.7, graph 128.2/127.5 (1.13x); rkstep 86.6 vs 91.9 (+6 %, 374 elem/rank); remesh 4.4 vs 15.1-15.3 s; per remesh g2s 92 / IGT 334-349 / s2g 583 / total ~1.0 s vs ~0.30 s.
  Labelled/SFC remesh ratio ~3.2-3.4x at both P; absolute overhead shrinks with P like SFC's, so no P-growth hazard yet (K4 NBX low priority). np768 cancelled (32 guest nodes unavailable), to re-run with K6.

### K6 default flip (2026-09-30): DENDRO_REPART_CANON_KEYS on by default (`=0` reverts), `_CHECK` and the no-SFC-source cascade fallback intact
- Code: one line in `Mesh::repartition` knob parse (`!(e && e[0]=='0')`). Build k6def (tp/em4/nuts/bssn), all gates knob-less.
- Gates (rule 2026-09-30: correctness vs SFC, not old labelled md5s):

| gate | result |
|---|---|
| EM4 np4/np8 sfc + graph | all d7bbee89 (graph was 2603beb3 / 32d48b0b) |
| EM4 np16 ragged (m3, REFINE=0) / levelwise (m4, REFINE=2) | d7bbee89 / d7bbee89 (ragged was 0faf9dbb) |
| tp NP=4, NP=8 | 0 FAIL; m4 identical; m3, m3ref1, ragged differ: COVERAGE and 7b unchanged, 8bb/8bc/8bd PASS maxErr 0, 8br dupPhysPos 0, TEST 10 maxErr unchanged (cg counts drop), 11 PASS (indep/dep counts +-few) |
| tp refs regenerated, rerun | IDENTICAL x8 (old kept as np{4,8}_{m3,m3ref1,ragged}.txt.pre_k6; m4 unchanged) |
| NLSM nuts2 np4 | == ref digits |
| NLSM lw3 np16 sfc/lvl/lw/fp | min/max of all arms == sfc; sfc/lvl md5 == lw3_s3on; lw/fp md5 move in l2 digits only (new baseline work/lw3_k6def: sfc 1e88c240, lvl a3da76f6, lw 84ea973a, fp b96c0e11; old lw 36973a4a, fp 8e8c0a89) |
| BSSN q1 np4 | graph Constraints a83112e3 -> bf867af6 (BH 7a18bafe, GW 86470a02/b50bc05f unchanged); graph vs sfc Constraints max rel diff 1.0e-13; C_HAM L2 9.20684e-06 both |
| `=0` revert, EM4 np4 graph | 2603beb3 (old) |

- Ref texts: scripts/em4.sh graph refs now d7bbee89 (old text in em4.sh.pre_k6); bssn.sh ref line for graph Constraints is now bf867af6 (was a83112e3).

### K7 probe (2026-09-30): do the IGT write candidates agree bitwise? (`DENDRO_IGT_WRITER_PROBE=1`, jj tvzrqryx)
In `interGridTransfer`'s write loop every (new local element, non-hanging node) is a candidate writer; the
probe routes all candidates by position, counts (node, var) groups whose candidates differ bitwise, and
compares the actual winner (last local non-hanging writer on the node's SFC owner rank) with
partition-free rules. Sums over all IGT calls of a run (EM4 t_end 10: 16 calls; tp: 4):

| run | groups | multi-cand | disagree | w != global-last | noW / multiW | max abs diff | big (>1e-12 rel) |
|---|---|---|---|---|---|---|---|
| EM4 sfc np4 | 44.4M | 16.1M | 265214 (80682 pos) | 75635 | 0 / 0 | 6.9e-18 | 0 |
| EM4 sfc np8 | 44.4M | 16.1M | 265578 | 111884 | 0 / 0 | 6.9e-18 | 0 |
| EM4 graph+K6 np4 / np8 | = sfc np4 / np8 | | | | | | |
| EM4 graph+K6 np16 ragged | 44.4M | 16.1M | 265269 | 140667 | 0 / 0 | 6.9e-18 | 0 |
| tp np4 m3 / m3ref1 / m4 | 4.70M | 2.50M | 213069 | 10263 | 0 / 0 | 3.1e-15 | 0 |
| tp np4 ragged | 0.78M | 0.41M | 76250 | 9004 | 0 / 0 | 4.4e-15 | 0 |

- Count is NONZERO: ~1.7 % of multi-candidate groups disagree, all at round-off (max 1 ULP-scale, no
  group above 1e-12 relative). The writer choice shows in the bits, so an IGT on a labelled
  distribution is not bit-identical by construction; it must reproduce the SFC writer.
- The SFC writer is itself partition-dependent (w != global-last grows 75k -> 112k -> 141k with np): the
  SFC IGT output differs at ULP level between np4 and np8 (CSV md5 hides it).
- Every owned node has exactly one actual writer (noW = multiW = 0), and the winner is never a
  partition-free rule (owner element / first / global-last all differ somewhere).
- Cost of reproducing it: the rule is local to the new SFC mesh S' (for each owned node: the last local
  non-hanging element on its S' owner rank, which owns the node's owner element too). It needs no data,
  so S' can compute a per-element writer mask and hand it out with the routing answer: cheap. See K7
  design below.

### K7 one-hop (design, 2026-09-30; workspace dendrolib_dfvk_ws_onehop)
Route (b): the IGT runs from the labelled source; the SFC octrees carry topology and flags only.
1. Flags g2s: `redistributeVec(dof=0, &flags)` (new flags-only mode, same hint/lean routing) labelled ->
   kept SFC twin, `setMeshRefinementFlags`, `ReMesh` -> SFC successor S'. No vector moves.
2. Final flags back: S (post-ReMesh, balance included) -> labelled source by the same flags-only call
   (hint mode labels), `setLocalElementFlags` (raw, no sibling filter).
3. Labelled successor L' = `buildGraphTwin(S')` (K2 `repartitionFrom`, K6 keys). Topology only.
4. `L.oneHopTransfer(S', L', in, out)`: M2' blocks computed on L's local elements with the IGT code
   (var-outer, one reused buffer, so coarse-injection leftovers are the SFC ones); each new element's
   SFC owner (S' splitters) is asked, per node, "is this element the IGT writer (last non-hanging local
   writer on the node's S' owner rank), and who owns the node on L'" (owner = label of the node's S'
   owner element, which is local there; K6 makes it L''s owner). One positional reply (npe ints per
   element, no count round). The source sends (position, dof values) only for writer nodes, straight
   to the node's L' owner: every L' local node is written exactly once, from the same element and the
   same bits as the SFC IGT. Then a ghost exchange on L' (DENDRO_REMESH_ONEHOP_GHOST, default on).
Invariants: canonical ownership (K6: L' owner of a node = label of its S' owner element); writer rule =
SFC IGT's, so the values equal the sandwich's by construction (no zip plan, orphan fill or pos-bcast
needed: S' has no orphans, every owned node has exactly one writer, noW = multiW = 0 in the probe);
COARSE siblings co-local on L (block vote); flags final before M2'.
Collective fallback to the two-hop sandwich when: no K6 keys on L', L' hints != S' local elements,
split COARSE sibling group on L, missing S' element / owner element, or any L' local node not written
exactly once (vecOut discarded, two-hop run from the still-alive L, S, S').
Moves per remesh: flags twice (4 B/element), one TN query (16 B/new element), one reply (4 B x npe per
new element), values (24 B key + 8 B x dof per owned node, each once) — vs two full CG hops plus the
SFC IGT exchange today.
Gates: `_CHECK=1` runs the sandwich alongside and aborts (76) on any differing local value (also counts
ghost slots); EM4 np4/np8 graph, np16 ragged (m3 REFINE=0) and levelwise (m4 REFINE=2) md5 = SFC
d7bbee89; tp np4/np8 4 arms 0 FAIL, COVERAGE 0, TEST 10 maxErr unchanged, new TEST 10o (onehop vs
sandwich bitwise per cycle) 0 mismatches.

### K7 prototype (2026-09-30): `DENDRO_REMESH_ONEHOP=1` (opt-in), dendrolib jj vlpzvtmv + uqsprzll, EM4 hunk work/onehop/em4_onehop.patch
- dendrolib: `Mesh::oneHopTransfer` (mesh.tcc), `setLocalElementFlags`, flags-only `redistributeVec(dof=0, &flags)`,
  `m_uiCanonKeysBuilt` (set by the K6 rebuild), `_PROF=1` phase line, `_GHOST=0` skips the final ghost exchange.
  testPartitioning TEST 10o: per cycle, final flags back to the labelled mesh + onehop, bitwise vs the sandwich.
- EM4 (`SOLVERCtx::remesh_onehop`): flags g2s -> ReMesh -> final flags back -> `buildGraphTwin` (K2) -> onehop;
  `_CHECK=1` runs the sandwich on a copy and aborts 76 on any differing local value. Needs K6 keys on (else fallback).
- Gates (K6 on): EM4 AMR md5 d7bbee89 (= SFC) at np4, np8, np16 ragged (m3 REFINE=0), np16 levelwise (m4 REFINE=2),
  7/7 remeshes onehop ok, 0 fallbacks; CHECK np4/np8/np16 ragged/np16 lw: 7/7 remeshes 0 local mismatches
  (2.1M-5.1M values per remesh at np4) and 0 ghost mismatches on every two-hop-written ghost slot; SFC arm untouched.
  tp np4 + np8 x4 arms: test lines identical to the knob-off base (0 FAIL, COVERAGE 0, TEST 10 maxErr 5.3e-15 /
  ragged 1.1e-14 unchanged), TEST 10o 4/4 cycles ok, 0 mismatches.
- Timing, EM4 np4/np8 local (throttled box, median of 7 remeshes, ms, g2s / remesh+IGT / s2g / total):
  np4 off 21.9/56.5/88.4/164 -> on 0.7/33.6/101.9/136 (-17 %); np8 off 23.8/48.4/93.2/167 and 23.4/48.8/88.6/160 ->
  on 1.0/30.9/92.6/124 and 0.6/33.4/91.7/124 (-22..-26 %). On: g2s = flags only; remesh+IGT = ReMesh + flags back;
  s2g = rebuild + onehop (xfer 24-81 ms at np4: m2p 5-17, query 2-10, xchg 4-11, write 8-31, ghost 3-8). np16 is
  oversubscribed (ReMesh ~1.5 s) and shows no signal.
- Risks: needs K6 + repart hints + co-local COARSE siblings (each checked, collective fallback); the final ghost
  exchange is one extra ghost-layer move (two-hop filled those slots through its transport); not ported to BSSN
  (pos-bcast, phantom fill) or NLSM (wavelet on the twin). Kingspeak np192 not measured.

### K6D Kingspeak np192 em4big (job 28498281, dendrolib 04d1c3a4 = K6 + CANON_TARGETS, 1 round, STEP_PROF + pos dumps)
- md5: sfc 34455595; graph K6+targets **34455595 (= SFC at np192)**; graph K6 targets=0 cb5ae14e. Targets: absent read slots 0 in every ring (10.5M target-row slots, none read); zip-sync pruned 0/0, global_active 0, local_dup/mirrors 0. targets=0: 7.7M absent R3 slots (153 read), zip-sync active.
- Wall sfc 195.8, K6+targets 217.1, targets=0 210.5. rkstep 161.6 / 171.1 / 166.3; remesh total 5.9 / 19.9 / 18.2 s; s2g 722 (targets add ~100 ms) vs 623.
- STEP_PROF gap K6+targets vs SFC (+9.5 s): evec_copy 5.0 -> 10.2 (+5.2, full-vector copies incl. the ~11x unread ghost layer), ge:wait +1.8, rhs_compute_other +1.2, mesh_waitAll +0.8, unzip_compute +0.4; zip-sync 0. Total bytes 1.50 TB vs SFC 1.52 TB.
- Next: DENDRO_EVEC_LOCAL_COPY (copy only local + delivered ghost slots), then stop allocating unread ghost cg slots; K7 one-hop at scale.

### K6 at scale (2026-10-01): absent owners in read rows, owner-target fix
- Kingspeak em4big np192/np384: K6 did not reproduce SFC (CSV 2.86e-4 unchanged), rkstep +2.5 %, every rank 34-40k
  "owner element absent" slots per rebuild.
- New counter `[canon-keys-absent]` (rank 0, every K6 rebuild): absent-owner slots by ring of their element
  (local/R1/R2/R3/target), and how many sit in elements the unzip reads (m_e2b_unzip_counts > 0); sum and max/rank.
  `[zipsync-summary]` now also prints max/rank of the pruned sync lists, local dups, ZNP mirrors, local non-primary.
- Local repro (main + counters, targets off): EM4 em4big np16 (50 GB RSS, stopped after 2 rebuilds): absent R2 723 /
  R3 1.81M slots, 125 R3 slots unzip-read. EM4 (maxdepth 7) np32: R2 316-726, R3 0.20-0.35M, read R2 0-27, R3 1-101,
  R1 6 in one rebuild; zip-sync stays globally active (pruned 1-26 pairs). The 26-lists miss finer neighbours, so the
  canonical owner of a read slot can sit outside the three fetched rings; the slot then falls back to a self-owned
  private cg (alternate owner / duplicate position) -> zip-sync back on and the state no longer the SFC one.
- Fix `DENDRO_REPART_CANON_TARGETS` (default 1, =0 reverts): the K6c owner-target fetch now also runs with three rings
  (splitter-routed to the SFC source, keyed records tagged 4). np32: 4.8-8.1k targets per rebuild (~150-250/rank), all
  absent slots left are on target rows (0 read in local/R1/R2/R3), zip-sync pruned 0 everywhere (global_active=0).
- Gates (targets on): tp np4+np8 x4 arms 0 FAIL, COVERAGE 0, tables = pre-fix K6 (only TEST 2 ghost-node counts and the
  informational 8bj slot count move: target rows add ghost cgs); EM4 md5 np4/np8/np16 ragged/np16 levelwise d7bbee89 =
  SFC, np32 795f6554 = SFC; NLSM nuts2 = ref.
- np48 oversubscribed EM4 hangs in the first graph repartition before buildOctantConnectivityMap with K6 on and off
  (600 s timeouts), so it is not K6; not chased.
- Kingspeak arm K6D in kp/bundle/ab_guest.sbatch (sfc / graph_k6 / graph_t0): DENDRO_STEP_PROF, COMM_STATS,
  ZIPSYNC_STATS, POS_DUMP_DIR=vtu at steps 5/25/65 ('/' separator, comma-split env lists), compare with poscmp.py.

### ALL + DX Kingspeak (2026-10-01; dendrolib 201163f6 = K6+targets default, + EVEC_LOCAL_COPY, + ONEHOP; em4big, 2 rounds)
- np192 (28498880): wall sfc 193.3/192.7, graph 212.7/215.0 (1.11x), +copy 202.1/201.0 (1.045x), +copy+onehop 197.2/197.7 (**1.023x**). rkstep sfc 159.6-160.2, +copy+onehop 154.0-154.6 (labelled stepping 3.5 % FASTER than SFC). Remesh total 5.7 vs 17.0 s; one-hop: g2s 170 -> 25 ms, IGT 420 -> 320 ms, s2g ~790 ms. All md5 = SFC 34455595.
- np384 (28498881): wall sfc 111.5/111.3, graph 1.16x, +copy 1.075x, +copy+onehop 117.3/118.2 (**1.057x**); rkstep 85.6-86.0 vs SFC 86.4-86.6. md5 graph arms 1335e24a != SFC (CSV max rel 3.9e-5); canon-keys-absent read=0, zip-sync off, so not K6.
- DX np384 (28499832) pointwise: steps 40/41/42/45 identical (0 diffs, 31.3M positions); step 50 (after the remesh) labelled has 1206 extra positions + 584 ULP diffs (4.3e-18) in one region (bbox [6976,8896,10048]-[14400,16864,11328]) => a refinement decision flipped at the step-50 remesh. Diagnosis in ws_flags (agent).

### FL (2026-10-01): np384 em4big flag flip at the step-50 remesh; DENDRO_REMESH_FLAGS_SFC (opt-in, EM4 hunk work/flags/em4_flags_sfc.patch)
- State is identical before the step-50 remesh (DX dumps), so the remesh flags differ. Two candidates: the labelled wavelet
  value differs at round-off (H1), or the sibling rule `setMeshRefinementFlags` runs on the labelled mesh in `isReMeshUnzip`
  (H2): a COARSE octet that is not contiguous on one labelled rank loses COARSE there, and the SFC twin never sees it. The
  extra labelled positions (fewer coarsenings) match H2.
- Fix `DENDRO_REMESH_FLAGS_SFC=1`: `Mesh::setRemeshRawFlags` makes `isReMeshUnzip` keep the raw wavelet flags on the
  labelled TNs; EM4 `is_remesh` then calls `Mesh::filterRefineFlagsOnSFC(keptSFC)` (flags-only redistribute, sibling rule +
  level cap on the twin, LOR), which is the SFC run's decision. The remesh path re-filters the same raw flags on the same twin.
  `_CHECK=1` prints `[remesh-flags-sfc]` raw/sfc/labelled coarse+split counts and `flag_diff` (twin flags old path vs new).
- Local: the mechanism is real (np16 with DENDRO_DISABLE_BLOCK_VOTE=1: flag_diff 248-424 per remesh), but with the block vote
  flag_diff = 0 at np4/8/16 ragged/16 lw/32, md5 d7bbee89 / 795f6554 = SFC with the knob on. Not reproducible below np384.
- Checkpoint restore (EM4_CHECKPT_WRITE=1, EM4_STOP_STEP=N, CHECKPT_FREQ=45): np4 sfc restore at 45 -> step 45/50 dumps and
  wave/flag dumps bit-identical to the full run; graph restore labels the restored SFC mesh and matches too.
- Kingspeak case FL (ab_guest.sbatch): sfc_cp writes the step-45 checkpoint, sfc_rs/graph_rs/graph_fx restore it; poscmp +
  kp/bundle/flagcmp.py (wave = raw decision + wMax per TN, flags = final flags at ReMesh) decide H1 vs H2.

### WAVE (2026-10-01): FL flip root cause = absent finer corner ghost; DENDRO_RING_FINER_NBRS=1 (opt-in, workspace ws_wave)
- Probe `DENDRO_WAVE_PROBE_TN=x/y/z/lev` (+`_DIR`): dumps the 13^3 x vars wavelet inputs of one element in isReMeshUnzip, every
  unzip commit into those slots (writer TN, local/ghost, same/finer/coarser, nodal inputs) and the elements touching it;
  compare with kp/bundle/wpcmp.py. Kingspeak WP (restore of FL step-45 cp, np384, job 28500241 / 28500387):
  SFC and labelled hold the same 2x2x2 L6 block (anchor 1536,1408,1664 L5); the target is its +x+y+z element. The 27 slots of the
  RIGHT_UP_FRONT corner padding are written in SFC by the finer ghost TN(1664,1536,1792,L7); on the labelled rank that element is
  ABSENT from the ghost set (28 of 29 touching elements present), so findBlockNeighborsWithoutSFC leaves the block vertex slot
  DEFAULT and the corner stays unwritten (stale zeros) -> wMax 7.7e-7 vs 1.4e-11. RK never reads 3D corner padding, so state stayed exact.
- Cause: buildOctantConnectivityMap fills edgeNeighbors/vertexNeighbors from getElementalEdge/VertexNeighbors, which drop finer
  neighbours (src/mesh.cpp level filter in both); the ring fetch only follows those lists, so a finer element touching only a corner
  (or edge) of a local element is fetched only if some other ring path reaches it.
- Fix `DENDRO_RING_FINER_NBRS=1`: a dropped edge/vertex slot takes the finer element touching the corner / edge low end (spatial
  lookup on the source mesh). More keys in the existing ring messages, no new collective: DENDRO_REPART_ROUNDS 36/35/36 on and off
  (np8), 36 per repartition at np384. `_CHECK=1` prints the filled slot counts (np384: ~23k edge, ~19k vertex, 0 without global id).
  `DENDRO_WAVE_COVER_CHECK=1` counts in-domain slots of every local element's wavelet stencil that no unzip writer filled (all unzips,
  incl. the remesh one): np384 restore 27 (one element) with the knob off, 0 with it on; np8/np16 ragged/np32 0 either way.
- Gates (knob on): WP restore graph = SFC at step 50 (31662889 positions, 0 extra, 0 diffs; wave/flag flips 0); full em4big np384
  graph (job 28500408) md5 34455595 = SFC; EM4 np4/np8 sfc+graph, np16 ragged, np16 levelwise d7bbee89; tp NP=4/8 0 FAIL (only TEST 2
  ghost-node counts and the informational 8bj pair counts move: more ghosts); nuts2 == ref.

## EVEC_LOCAL_COPY default flip (2026-10-01)

`DENDRO_EVEC_LOCAL_COPY` now default on (`=0` reverts; `_CHECK=1` still forces the checked path;
non-CG / non-host / non-double vectors still take the full copy). jj 51aeb470. Build `evdef`.

| gate | result |
|---|---|
| EM4 np4/np8 sfc + graph | all d7bbee89 |
| EM4 np16 ragged (m3, REFINE=0) / levelwise (m4, REFINE=2) | d7bbee89 / d7bbee89 |
| testPartitioning np4 + np8 (m3, m3ref1, m4, ragged) | 0 FAIL; ghost-count lines only differed (CANON_TARGETS), refs regenerated, rerun IDENTICAL |
| nuts2 | == ref digits |
| lw3 np16 sfc/lvl/lw/fp | min/max identical to sfc in every arm |
| bssn np4 q1 | graph bf867af6, sfc 2ffba783 (= refs) |
| EM4 np8 `_CHECK=1` | d7bbee89, 0 mismatches, no poison abort |
| EM4 np8 `=0` | d7bbee89 (revert works) |

Refs changed: ref/tp_base/np{4,8}_{m3,m3ref1,m4,ragged}.txt (TEST 2 node counts, 8bj pair counts); old copies kept as `.pre_targets`.

## EXACT ghost push (2026-10-01; workspace dendrolib_dfvk_ws_exact, on lpqproqq)
- `Mesh::computeExactGhostDest` (SFC source): D(e) = label(e) + labels of all geometrically touching elements
  (hash walk incl. finer, local+ghost of the SFC mesh) + K6 owner closure (D1(f) of every f with a node owned by e).
  Comm: 2 neighbour-sparse P2P rounds on the SFC R1 element pattern (ghost labels; owner claims) + 1 Allreduce
  (owners outside R1, splitter-routed Alltoall/Alltoallv only if any: seen 2 owners in NLSM's first rebuild).
- Premise holds: SFC R1 is a complete geometric ring (uncovered=0, unlabelled=0 in every tp/EM4/NLSM rebuild).
  Node owners not touching the element exist (owner_far ~8-14k per tp rebuild: hanging nodes routed to a
  coarser element's owner two hops away); they are all R1 ghosts of the source except the 2 NLSM cases.
- `DENDRO_REPART_EXACT_GHOST_CHECK=1|2`: (a) D vs delivered (rings + targets), (b) shell + padding volume
  cover and unzip-list completeness, labelled and SFC meshes. Current rebuild (np4-16, tp 4 arms, EM4
  m3/ragged/levelwise, RING_FINER_NBRS 0 and 1): missing 0 everywhere (R2/R3 backfill), covers clean; but
  touching elements arrive only as ring>=2 (nodal map invalid): EM4 np8 finer face 490 / edge 458+101 same /
  vertex 111+65 same, FINER_NBRS=1 -> face 316, edge 142(second edge child)+91 same, vertex 2+62 same.
  Over-fetch (EM4 np8, 8 rebuilds): rings deliver 73k ghost pushes vs exact 25k (R2 extra 11k, R3 24k, targets 13k).
- `DENDRO_REPART_EXACT_GHOST=1` (opt-in, needs K6): push from D, no claim rounds, no owner targets. Gates: EM4
  np4/np8 sfc+graph, np16 ragged (m3 REFINE=0) / levelwise (m4 REFINE=2) all d7bbee89 with _CHECK=2; tp NP=4/8
  0 FAIL, COVERAGE + SM_COVERAGE clean (only TEST 2 ghost-node counts and the informational 8bj move:
  m3 np4 0->108, ragged 4508->2555 / 9525->3087); nuts2 == ref.
- EM4 np8 (8 rebuilds, box-relative): rounds 35-36 -> 24-25, [repart] TOTAL 470/492 -> 309/317 ms, ghost
  elements/rank 1013 -> 326; np16 levelwise rounds 37 -> 26-28, TOTAL 873 -> 354 ms, ghosts 936 -> 293;
  per-step ghost send nodes identical.
- Kingspeak case `EG)` in kp/bundle/ab_guest.sbatch (np384 em4big, FL step-45 restore, sfc_rs/graph_rs/graph_ex).

## RING_FINER_NBRS default flip (2026-10-01)

`DENDRO_RING_FINER_NBRS` now default on (`=0` reverts; `_CHECK=1` unchanged). Correctness fix: without it a finer corner/edge
neighbour can be missing from the labelled ghost rings and np384 diverges from SFC. Build `rfdef`.

| gate | result |
|---|---|
| EM4 np4/np8 sfc + graph | all d7bbee89 |
| EM4 np16 ragged (m3, REFINE=0) / levelwise (m4, REFINE=2) | d7bbee89 / d7bbee89 |
| testPartitioning np4 + np8 (m3, m3ref1, m4, ragged) | 0 FAIL; only TEST 2 ghost-node counts and 8bj pair counts differed, refs regenerated, rerun IDENTICAL |
| nuts2 | == ref digits |
| lw3 np16 sfc/lvl/lw/fp | min/max identical to sfc in every arm |
| bssn np4 q1 | graph bf867af6, sfc 2ffba783 (= refs; C_HAM L2 9.20684e-06 both) |
| EM4 np8 graph `RING_FINER_NBRS=0` | d7bbee89 (revert works) |
| EM4 np8 graph `WAVE_COVER_CHECK=1` | 0 unwritten slots (332 report lines, none nonzero) |

Refs changed: ref/tp_base/np{4,8}_{m3,m3ref1,m4,ragged}.txt; old copies kept as `.pre_rf`.

### Solver port 2 (2026-10-01): inactive-rank guard + BSSN labelled init + BSSN one-hop (uncommitted in the solver repos; patches `~/research/repart_gate/work/port2/`)
- Inactive-rank guard, mirrors EM4, NLSM + BSSN: `buildGraphTwin` returns srcMesh when `!allRanksActive()`; s2g treats `graphMesh == newMesh` as "stayed SFC" (no redistribute, mesh stays SFC); BSSN `maybe_apply_partitioning` lost its `isActive` early return and `remesh_and_gridtransfer` does a plain remesh + `maybe_apply_partitioning` when `m_uiMesh` has inactive ranks (labels once all ranks are active). NLSM: init-loop and stay-SFC labelling guarded on `allRanksActive()`, s2g finishes as SFC (swap, no labelling). `redistributeDVec`'s both-inactive early return is left alone (never reached once buildGraphTwin returns src). Patches `nlsm_inactive.patch`, `bssn_inactive_init_labelled.patch`.
- BSSN labelled init (same patch): `BSSNCtx::m_uiIsLabelled` (set by `swap_to_graph_partition` and the end of the graph sandwich); `initialize()` routes a refining iteration through `remesh_and_gridtransfer` when labelled. bssnSolverNUTS is not wired in the BSSN repo (CMake commented), so gated with bssnSolver only. NLSM is immune (initialize converts to the SFC twin / stays SFC).
- BSSN one-hop `DENDRO_REMESH_ONEHOP` (+`_CHECK`), opt-in, `bssn_onehop.patch`: `BSSNCtx::remesh_onehop` = EM4's with BSSN's vectors (no analytic); pre-sandwich pos-bcast kept. NLSM one-hop NOT ported: its wavelet already needs the EV on the SFC twin (`is_remesh` g2s), so one-hop would only replace IGT+s2g and needs the keep-labelled state (`NLSM_REMESH_FALSE_RESTORE`) kept across the wavelet; separate design.
- Gates: nuts2 == ref; lw3 np16 sfc/lvl min/max == previous (U_CHI -1.4965567135e-13 / 1.1089986435; U_PHI -2.3638501150 / 2.5021402982); bssn np4 graph 7a18bafe/bf867af6/86470a02/b50bc05f, sfc 7a18bafe/2ffba783/86470a02/b50bc05f (= refs). One-hop `_CHECK` bssn np4: 9/9 remeshes 0 local mismatches, 0 ghost mismatches, md5s = graph refs; one-hop xfer ~330 ms + rebuild ~175 ms at np4.
- np48 oversubscribed (small meshes, some ranks inactive): NLSM d9 grain 300/600: graph == sfc (U_CHI/U_PHI min/max/l2 all digits); BSSN q1 grain 300: graph == sfc (all four md5s equal, mesh stays SFC); BSSN q1 grain 50 (labelled at times, 9 graph-remeshes): GW/BH md5 equal, Constraints differ at 1e-16 (as at np4). The pre-port binary did NOT hang on these small cases, so the guard is verified for non-regression only; the original np48 hang repro is EM4 maxdepth 7.
- Open: NLSM d9 grain 150 np48 (labelled after init, one graph-remesh) hangs at the step-1 output in `dump_analytical_error`/`compute_analytical` `readFromGhostEnd` vs `terminal_output` Reduce, and the SFC arm hangs identically, so it is not the port (min/max/l2 to step 1 equal across arms). Not investigated.

### Expanse np384 (2026-10-01; account byu105; dendrolib c04e15ef + EM4 patches; em4big)
- EX (job 54576524, 2 rounds, ~142 SU): md5 identical in every arm (38a4dfbc). rkstep sfc 121.8, graph+1hop 122.9-123.1, graph+exact+1hop 120.9-121.3; remesh total 6.6-7.2 / 36.8-37.7 / 26.5-26.9 s. Exact ghost cuts remesh 28 %. s2g ~2.1 s (1hop) vs Kingspeak ~0.58 s.
- Diagnosis: transport fine (UCX default, 1.26 us / 12.3 GB/s; TCP 12.7 us / 3 GB/s); dense vs sparse Alltoallv and tuned algorithms equal at np384 (microbench); UCX_TLS=dc_x no change. One-hop phase profile: ~85 % of the 1.3-1.6 s transfer is the final ghost exchange on the fresh labelled mesh (ghost=1.1-1.5 s, xchg 35-200 ms). Ghost fill off: transfer 0.21-0.27 s, remesh -2.7 s but rkstep +1.7 s (net -1 s / 3 remeshes) => the FIRST exchange on a new mesh is slow on Expanse (lazy per-mesh setup or first-touch with 128 ranks/node), not MPI.
- DENDRO_REMESH_ONEHOP_GHOST now default OFF (=1 restores): EM4 np4/np8 sfc+graph d7bbee89, BSSN q1 graph+onehop = all refs (9/9 onehop), Expanse np384 md5 identical ghost on/off.
- Expanse SU used today ~215 (EX 142, diag 25, a2av 14, short profiles 11+9+12.5, build/smoke/xport ~1).

### First ghost exchange on a fresh mesh (2026-10-01; local, OpenMPI 5.0.11, EM4 graph np8/np16)
- `DENDRO_FIRST_XCHG_PROF=1` (`_CALLS`=n, default 2): per mesh, the first n plain/ctx `readFromGhostBegin/End` get an entry barrier and a phase split (sync/alloc/post/pack/isend/wait/unpack/mirror/free), getrusage minflt/majflt, lazy zip-plan build time, max over ranks. `=2` adds probes before the exchange: zero-byte handshake with every peer on the world comm, then on the mesh comm, then two exchanges of the real sizes through persistent buffers.
- Local result: no first-use cost. Labelled meshes, first vs second exchange: np8 6.6-6.9 vs 7.0 ms, np16 12.4-15.0 vs 13.4-14.7 ms (one-hop on and off); wait/pack/unpack ~1/3 each; lazy zip-plan builds 0 (plan already built), sync 0, minflt <= ~300/rank, handshakes <= 0.02 ms. The entry barrier (imbalance arriving) is 6-27 ms, larger than the exchange itself. One-hop ghost phase 6.8 ms np8, 17-19 ms np16. Expanse's 1.1-1.5 s is therefore platform-specific (OpenMPI 4.1.8/UCX at 128 ranks/node): wire-up of new peer pairs, per-comm setup, or registration/first touch of fresh buffers; the diag case separates these.
- `DENDRO_XCHG_BUF_POOL=1` (`_CHECK=1` poisons each buffer with NaN bytes): `AsyncExchangeContex` send/recv buffers (plain exchange, `alloc_mpi_ctx` RK buffers, oda) come from a process-wide pool that persists across meshes, prefaulted when grown (+25 %). Locally time-neutral (RK ctx exchange minflt 200-500 -> 0, growth costs ~2 ms once per mesh); md5 unchanged everywhere.
- Expanse case `firstx.sbatch` (bundle): em4big short, graph+exact+one-hop with ghost fill on (the ohprof config), MODE=diag (default, pool off, probes) or MODE=pool. ~10 SU each.
- Gates (pool+CHECK unless noted): EM4 np4/np8 sfc+graph d7bbee89 (default and pool); tp NP=4 4 arms 0 FAIL, identical to base; nuts2 == ref; np16 r0 (m3, refine 0) and lw (m4, refine 2) d7bbee89 == ghoff baseline; BSSN q1 + `DENDRO_REMESH_ONEHOP=1` graph/sfc == all refs (9 onehop).
- 2026-10-01 late: Expanse job 54579252 (MODE=diag) hit the 6 min limit with no `[first-xchg]` output; the log tail is inside the FIRST mesh constructor (after `[unzip-canon r0]`), before any exchange the profiler instruments. Not reproduced locally: the old profiler (in-exchange `MPI_Barrier`, probe handshakes and `MPI_Reduce`) ran to completion on em4big np64 and on the gate case at np128, where active comms are 16/32-rank subsets. The profiler was redesigned anyway so it adds no MPI calls inside any exchange: slots are recorded locally and reduced in `~Mesh` next to the existing collective `MPI_Comm_free`; the probes run once at the end of `repartitionMeshGlobal` (`[first-xchg-probe]`), use the global comm only when it is `MPI_CONGRUENT` with the mesh comm, and `DENDRO_FIRST_XCHG_PROF_DOF` (default 8) sets the probe size. `wait=` now includes rank imbalance because the entry barrier is gone. Proofs: gate case at np64 and np128 (oversubscribed), firstx diag env and level 1, all rc=0 with `[first-xchg]` lines, md5 795f6554 for every arm; EM4 np4/np8 d7bbee89 with the knobs off and on (PROF=2 + pool + CHECK); tp NP=4 0 FAIL off and on. em4big does not fit in local RAM at np64 (MemAvailable fell to 3 GiB; the watchdog killed it).
- 2026-10-01 night, reduce-free profiler (jj prlwqwwp breadcrumbs + oryktonn): `DENDRO_BREADCRUMB_DIR=<dir>` makes every rank append unbuffered `t=<Wtime> tag` lines (one `write(2)`, `O_APPEND`) to `<dir>/r<rank>.txt`: mesh ctor/dtor with id + comm size, `remesh`/`igt`/`redist`/`onehop`/`repartGlobal`/`repartFrom` scopes, `evolve` steps, `xprobe` entry/exit. The profiler is now collective-free at level 1: each rank writes its own `[first-xchg]` record when the exchange finishes (no `~Mesh` reduce); level 2 keeps only the mesh-comm handshake + two warm exchanges (global-comm handshake and its `MPI_Reduce` dropped), waits spin on `MPI_Test`/`MPI_Ibarrier` and write `[first-xchg-stall]` with the pending peers after `DENDRO_FIRST_XCHG_PROF_STALL` s (default 20). Records go to the breadcrumb file, or `first_xchg/r<rank>.txt` (`DENDRO_BREADCRUMB_REC_DIR`) when breadcrumbs are off. `tools/breadcrumbs.py <dir>` gives the old max/sum lines; `--stall [--tail N]` groups ranks by last line, lists open scopes and meshes built/destroyed on a rank subset.
- Local repro attempt: em4big at d10, wavelet tol 1e-8, remesh freq 5 (t_end 0.5, 16 steps, 3 labelled remeshes, 3/3 `[onehop] ok=1`, exact ghost) at np64 and np128 oversubscribed with the firstx diag env: the ptstomsy profiler completed (md5 69deec0d both), no stall; every mesh built and destroyed on all ranks in the same order. The Expanse stall is not reproduced locally, so its exact point is unknown; the fix removes every profiler collective except the level-2 probe, which now self-reports a stall. Gates: np64 case rc=0 (69deec0d, same as no-fix); EM4 np4/np8 graph d7bbee89 off and on (PROF=2 + breadcrumbs + pool + one-hop + exact, stall path forced with STALL=1e-4); tp NP=4 4 arms 0 FAIL identical to base off and on. Bundle `firstx.sbatch` now sets `DENDRO_BREADCRUMB_DIR=$O/bc` and aggregates with `bundle/breadcrumbs.py`.

### Buffer pool default on (2026-10-02)
`DENDRO_XCHG_BUF_POOL` is now default ON (`=0` disables); `_CHECK=1` unchanged (poisons handed-out buffers).
- Evidence (Expanse np384, labelled short em4big, same nodes): remesh total default 3.66 / 5.50 s vs pool 2.87 s; s2g per remesh 713 / 1310 vs 520 ms; rkstep 23.16 / 22.29 vs 23.09 s (no stepping penalty); md5 identical.
- Root cause: RDMA memory registration of freshly allocated (~1 MB) exchange buffers. UCX rendezvous is zero-copy and pins the user buffer; fresh mmap'd buffers miss the registration cache, and at 128 ranks/node the pinning costs ~1 s per new mesh. The pool hands out the same buffers on every mesh, so the registration cache hits. This is not UCX-specific: any RDMA stack that registers user buffers lazily (UCX/verbs, libfabric/OFI incl. CXI on Slingshot, Cray MPICH) pays it, and it is worst at high ranks per node (shared NIC, serialized pinning in the kernel).
- Rejected: `UCX_RNDV_THRESH=16M` hides the cost (no zero-copy, no registration) but costs +8 % in stepping.
- Knobs: `DENDRO_XCHG_BUF_POOL=0` off; `DENDRO_XCHG_BUF_POOL_MAX_MB` per-rank cap on pooled bytes (default 512; requests that would exceed it fall back to plain malloc/free, never fail); `DENDRO_XCHG_BUF_POOL_REPORT=1` prints `[xbuf-pool] rank= peak_MB= cap_MB= fallbacks=` per rank at exit; with `DENDRO_COMM_STATS=1` `dumpCommStats` also prints `xchg_buf_pool_peak_MB_max` (max over active ranks).
- Memory: the pool holds the peak number of buffers in flight (+25 % headroom, prefaulted), per rank, for the process lifetime; it grows only to the largest exchange, so it is about the peak ghost-exchange footprint (measured per-rank peak: EM4 np8 22-30 MB, BSSN q1 np4 127-145 MB; with `MAX_MB=1` EM4 np8 stays <= 0.91 MB and ~95 fallbacks/rank, same md5), far below the cap.
- Threading: the pool is mutex-guarded. Production is MPI-per-core with no OpenMP; no OpenMP region calls the ghost exchanges, so the lock is uncontended.

## Exact ghost default flip (2026-10-02)

`DENDRO_REPART_EXACT_GHOST` is now default on (`=0` reverts to the R1-R3 rings + owner targets). The path still falls back to the rings when K6 keys are off or its preconditions fail; `_CHECK` and `DENDRO_RING_FINER_NBRS` are unchanged. Evidence: Kingspeak em4big np192/np384 md5 = SFC, remesh 23-27 % faster; Expanse np384 md5 equal, remesh -28 %.

Gates on the default build (bin exdef), no knobs:

| gate | result |
|---|---|
| em4.sh np4/np8 sfc+graph | all d7bbee89 |
| EM4 np16 ragged (m3, REFINE=0) / lvl (m4, REFINE=2) | d7bbee89 / d7bbee89 |
| testPartitioning np4, np8 (m3, m3ref1, m4, ragged) | 0 FAIL, COVERAGE clean, TEST 10/11 unchanged |
| nuts2.sh | == ref |
| lw3 np16 sfc/lvl/lw/fp | min/max of every arm equal to sfc (to ~1e-12 in l2 only) |
| bssn.sh graph | 7a18bafe / bf867af6 / 86470a02 / b50bc05f |
| EXACT_GHOST=0 EM4 np8 | d7bbee89 |
| EXACT_GHOST_CHECK=1 EM4 np8 graph | missing=0 |
| WAVE_COVER_CHECK=1 EM4 np8 graph | unwritten=0 |

tp refs: TEST 2 ghost-exchange node counts drop (np4 m3 319415 -> 265709, np8 m3 477074 -> 389177, m4/m3ref1 about -25 %). Refs regenerated (`ref/tp_base/np{4,8}_*.txt`, old kept as `.pre_exact`), rerun IDENTICAL.

Unexpected: np4 m3 TEST 8bj [canonical phys_pos] goes 0 -> 108 (e, sub) pairs routing differently graph vs SFC (maxDiff 4); `=0` gives 0 again. It is a red informational line (not counted as FAIL); pre/post-ghost-exchange 8bj bit-mismatches stay 0 and EM4 md5 is unchanged. Ragged 8bj improves (np4 4696 -> 2555, np8 9868 -> 3087). Worth a look but not a blocker.

## 2026-10-02 — one-hop + flat index + fast zip plan default on

- One-hop (`DENDRO_REMESH_ONEHOP`, EM4/BSSN solver repos, =0 reverts) and `DENDRO_ONEHOP_FLAT_INDEX` (a628f81c) default on;
  `DENDRO_ZIPPLAN_FAST` (bd062512 + 350f0d05) default on: zip plan straight from the local writer fold when every
  position has one claimant (K6 / SFC), one Allreduce, prune-time fallback to the full build; `_CHECK=1` aborts 83.
- Gates (all three on, plus `_CHECK`s): EM4 np4/8 sfc+graph, np16 ragged/lvl = d7bbee89; tp NP=4/8 identical to base;
  nuts2/lw3 identical; BSSN refs; zip-plan check 17/17 (EM4), 13/13 (BSSN), 0 differences.
- At scale (em4big, md5 34455595 everywhere): Kingspeak np192 labelled 1.00× SFC wall (rkstep −2 %), np384 1.027×
  (rkstep +0.6 %, remesh 8.7 vs 4.8 s); flat index write 110→50 ms/remesh at np192. Kingspeak ZC np192: fast plan
  taken 19/19 sfc + 19/19 graph, check-clean. Expanse np384 short: ~1.03×; zip plan 84→23 ms/rebuild; identical arms
  differ ±0.26 s remesh, so sub-0.3 s effects need the phase timers.
- Labelled rebuild at np384 now ~270–330 ms: sm exchange 61–65, fetch 37, partition decision 26, zip plan 23,
  connmap 20. One-hop ~120–200 ms (xchg 36–100). Next targets: sm exchange, one-hop records (owner TN, sub).
- Harness trap: scripts lacked `OMP_NUM_THREADS=1` (libgomp spin, timings 3–7× off); now pinned.
