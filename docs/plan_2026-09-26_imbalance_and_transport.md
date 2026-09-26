# Plan — price an imbalance case, then cut the transport (2026-09-26, session 2)

From `docs/handoff_2026-09-26_sandwich_next.md`. Standing rules unchanged: local runs, Kingspeak only under
the handoff's setup; one logical change per jj commit, subject-only; new behaviour behind a `DENDRO_*` knob
with the default byte-identical, gated knob-on, then default flipped and the defaults path gated; every gate
has the ragged arm and the SFC ENUTS arm; deterministic metrics locally, wall only from Kingspeak (arms
interleaved in one job, ratio within a round).

Gate harness moved out of the session scratchpad to `~/research/repart_gate/` (`scripts/{build,gate,em4,
nuts2,bssn,size_nuts}.sh`, `ref/` = tp base, EM4 tomls, ENUTS pars, TwoPunctures ID; `bin/<label>/` =
the four binaries per build, `bin/base` = jj `nrynmmrr`).

## Task 1a — an LTS case sized for 48–192 ranks
The gated depth-9 NLSM ENUTS case partitions first on `function2Octree`'s output (540 octants), which the
LTS init remesh grows to 4096: 11 elements per rank at np=48, below the grain (100) — most ranks inactive,
and the labelled rebuild hangs with inactive ranks. Size by sweeping `NLSM_MAXDEPTH` 9–11 and
`NLSM_WAVELET_TOL` (SFC arm, np=8, one coarse step): read the first octant count, the post-init mesh, the
per-level histogram and `dump_est_speedup`. Target: first partition ≥ 4 800 octants (≥ 100 per rank at
np=48), ideally ≥ 19 200 (192 ranks), with ≥ 3 populated levels so sub-cycling matters, and a local cost
that fits np=4/8 gating.
Gate: SFC (method 0) vs levelwise+refine (method 3, `DENDRO_GRAPH_REFINE=2`) at np=4 and np=8, U_CHI/U_PHI
min/max identical to 18 digits, proof of fire `[block-vote]` + `[graph-part-ex]`. Then a Kingspeak job
(48/96/192, arms interleaved): wall, `lts_substep_eff`, `dump_est_speedup`, `LD Bal`.

## Task 1b — an asymmetric global-RK case
BSSN q1 with BH-driven refinement (or `PARTEVOLVE_INIT=puncture`), ≥ 500 elements per rank at the target
np; arms SFC vs `DENDRO_GRAPH_REFINE=3` + `DENDRO_GRAPH_VTX_WEIGHTED=2` + `DENDRO_GRAPH_BALANCE_ROUNDS`;
read `padded_vol_imbalance` and wall. Sizing first (element count at the first partition and after the
init remeshes).

Both need the one-block floor per rank in the block vote if a ragged arm hits the zero-element guard at
scale (`mesh.cpp` `[repart] a rank owns zero elements`).

## Task 2 — the transport (`redistributeVec`), both directions
Today every call learns each source element's destination rank by a TN-hash directory rendezvous (3
Alltoall + 3 Alltoallv + an Allreduce), although the answer is already known:
- **s2g** (src = the ReMesh successor, dst = the labelled twin): the twin was built from the successor's
  local elements positionally (`buildGraphTwin`), and the partition decision labelled exactly those
  elements; dst rank of src element i = label i.
- **g2s** (src = the labelled mesh, dst = its SFC twin): if the twin's SFC distribution equals the
  labelled mesh's pre-partition distribution, dst rank of a labelled element = the rank it came from
  (`upper_bound(ele_offsets, eid)`).
`DENDRO_REDIST_HINT=1`: `repartitionMeshGlobal` keeps the pre-partition local TreeNodes, their labels
and each new local element's origin rank; `redistributeVec` uses them when the src (s2g) or dst (g2s)
local element list equals the stored pre-partition list on every rank (one Allreduce), otherwise the
directory. Byte-identical by construction (same `srcTarget`); `DENDRO_REDIST_HINT_CHECK=1` runs both and
counts mismatches. Also fuse the fixed-shape collectives (the three count Alltoalls are one, the five
key Allgathervs are one, the two orphan count Alltoalls are one) — same bytes, fewer rounds.
A phase timer `DENDRO_REDIST_PROF=1` (barrier-timed, max over ranks) splits directory / pack+exchange /
write / canon pull so the Kingspeak pair can price each piece.
Then, in order: the orphan-key Allgatherv (only if the probe shows the key count matters at scale), the
scatter-map double build (`DENDRO_MINIMAL_SCATTER_FILTER`), inherited labels. Each gated as above and
re-priced on Kingspeak with an EXP=S-style on/off profile pair.

## Log
