# Plan — the per-step labelled tax (2026-09-26, session 3)

From `docs/handoff_2026-09-26b_imbalance_cases.md`, item 1. Rules unchanged (local only, knob-gated,
default byte-identical until gated, one change per jj commit).

## What the per-step pos-bcast does (probe `DENDRO_BCAST_DIFF_STATS=1`)
NLSM/BSSN graph meshes run `broadcastCgValuesByPhysPos` after the post-axpy zip sync on every
ETS step (`ets.h`, EM4 opts out). BSSN q1 tinytest, np=4, graph arm, 24 dof, per call:

| destination | plan entries | entries whose write changes a value |
|---|---|---|
| local slot (recv) | 186 | 0 |
| ghost delivered by the ghost exchange | 683 117 | 683 111 (661 541 on the first call after a rebuild) |
| ghost the exchange never delivers | 1 212 866 | 1 212 140 |
| intra local / ghost-deliv / ghost-undeliv | 25 / 12 / 174 | 0 / 12 / 128 |
| phantom | 7 161 | 7 161 on the first call after a rebuild, then 0 |

~170 ms per call at np=4 locally (dense Alltoallv of 1.9 M × 24 doubles summed). It is a second
full ghost exchange, mostly into ghost slots the unzip never reads; the values it could change on
local slots never change. The pre-sandwich call (ghosts just exchanged by is_remesh) changes only
the undelivered ghosts.

## Cut: `DENDRO_BCAST_STEP_PRUNE=1` (opt-in, then flip)
`Mesh::broadcastCgValuesByPhysPosStep` (the ETS call site only; sandwich/transport calls keep the
full plan): apply only entries whose destination is a local slot, an undelivered ghost the unzip or
a mirror reads, or a ghost-exchange send source (the `pruneZipSync` level-2 keep set); p2p, plan
filtered once per mesh (`buildBcastStepPlan`, one keep-mask Alltoallv), reset with the full plan.
Check mode `DENDRO_BCAST_STEP_CHECK=1` runs the full plan on a copy: kept slots must be identical.
Stats `DENDRO_BCAST_STEP_STATS=1`; bytes in `[comm-stats-cumulative] bcast_bytes_sent`.

BSSN np=4: recv entries 1 896 169 → 192, intra 211 → 71, kept_mismatch 0.

Gates: BSSN graph + SFC md5 vs references; EM4 np=8 graph with `DENDRO_FORCE_POS_BCAST=1`, prune
0 vs 1 md5; tp/EM4/ENUTS unaffected (no ETS pos-bcast) but run on the defaults path after the flip.

## Gates (knob on, bin `bstep`, jj `tszvxsqm`)
- BSSN q1 tinytest np=4, `PRUNE=1 CHECK=1`: graph 7a18bafe / c1508613 / 86470a02 / ed1127e4, SFC
  7a18bafe / 2ffba783 / 86470a02 / b50bc05f — all on references; kept_mismatch 0 on every call;
  every rebuild recv 1 896 169 → 192, intra 211 → 71.
- EM4 np=8 with `DENDRO_FORCE_POS_BCAST=1`, prune 0 vs 1 (+CHECK): graph 32eb202e both, SFC d7bbee89.
  Bytes: the full plan ships ~101 MB per call (10.8 GB over 107 calls, more than the whole ghost
  exchange's 10.2 GB in that run); the pruned step plan ~5.7 kB per call (77 step calls).
