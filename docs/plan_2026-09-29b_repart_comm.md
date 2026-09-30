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
