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
