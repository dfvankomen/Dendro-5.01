# Thesis results plan (2026-10-03)

Goal: a thesis chapter showing repartitioning is now overhead-free (done: em4big 1.00x np192, 1.013x np384,
Expanse ~1.006x, md5 = SFC) and what it enables where SFC does poorly. Code state: jj bookmark
`repart-thesis` (3fafd655); every number in the chapter cites the commit it ran on.

## A. Communication volume: SFC vs AMRaCut (fastpart) vs refined (local, deterministic)
Ghost bytes per step (total and busiest rank), ghost nodes, edge cut on the 26-closure and on faces, for
EM4 (em4d10 / em4big), NLSM d9/d11 and BSSN q1 meshes at np 16/32/64 (local, byte counts are exact under
oversubscription) plus np 96-384 from the Kingspeak runs below. Arms: sfc, amracut (method 3,
GRAPH_REFINE=0), refined (1), levelwise (4 / GRAPH_REFINE=2), pv (BSSN: 3 + VTX_WEIGHTED=2).

## B. BSSN where SFC balances badly (Kingspeak, np96 and np192)
q1 BBH, arms sfc / amracut / refined / pv, 2 interleaved rounds, comm + part stats. If an arm wins in
wall, fold it into the user's BSSN speedup ladder.

## C. LTS performance on the finished code (Kingspeak np192)
NLSM d11 (`nlsm_lts11_*`), arms sfc / lvl / lw, 2 rounds. Re-run of the 09-29 0.85x result.

## D. LTS accuracy (local, small)
1. Partition invariance under LTS: labelled vs SFC ENUTS agree to all printed digits (EM4 em4NUTS, NLSM).
2. LTS vs GTS error as the sub-cycle ratio grows (`DENDRO_LTS_RATIO_CAP` 1/2/4/uncapped): EM4 vs its
   analytical solution, NLSM vs a GTS reference on a fixed mesh (NLSM analytical is broken).

## E. BSSN analysis (from A + B)
Padded imbalance, ghost bytes, step breakdown, constraint norms SFC vs labelled.

Rules: Kingspeak one job at a time (chained), no Expanse without asking; every claim from interleaved
arms in one job; local MPI under the mpi.lock with >= 20 GB free; OMP_NUM_THREADS=1.
