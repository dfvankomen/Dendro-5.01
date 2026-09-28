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

## Flip (jj `lvmprtyo`, bin `sdef`) — defaults path green
tp np=4 ×4 arms IDENTICAL to base (incl. ragged); EM4 np4 a75a069e/d7bbee89, np8 32eb202e/d7bbee89;
ragged ENUTS U_CHI/U_PHI on the 18-digit reference; BSSN graph + SFC on all 8 reference md5s.
Local wall is not evidence (shared box, another user's jobs). Next: price on Kingspeak (U pair at
np=48/96, where the handoff saw ~24 s per-step tax beyond sandwiches) — needs the user's OK.

## Kingspeak round 3 (build 28414343 = jj lvmprtyo, V 28414344/5, W 28414346)
**V, BSSN puncture (U case), np=96, 2 rounds (wall s):** sfc 171.6 / 177.2 · pv (step prune, default)
192.7 / 187.0 · pv_full (`DENDRO_BCAST_STEP_PRUNE=0`) 206.8 / 208.4. The prune is −14 to −21 s per run
(the handoff's ~24 s tax); labelled vs SFC 1.19× → 1.08×. Step plan per rebuild: recv 35 981 217 → 49.
**W, EM4 d9 np=192 (stamp dedup priced):** stamp 63.9 / 63.8 · set (`DENDRO_SM_SET_DEDUP=1`) 65.2 / 64.4 ·
sfc 51.7 / 51.3: −1 %, traffic byte-identical.
**L np=192 from round 2 (28412917):** sfc 65.4 / 64.9 · lvl 103.6 / 103.1 · lw 111.4 / 111.0 (1.59× /
1.71×); lts_substep_eff 0.34 → 0.58.

## The sandwich's pos-bcast election (NLSM d11 lw, np=8, `DENDRO_BCAST_PROF=1`)
After each labelled rebuild `nlsm::redistributeDVec` runs the full pos-bcast on the new mesh; its plan
election is 1.7–1.8 s per rebuild (apply ~70 ms), ~1/3 of s2g. Sub-phases: elect ~0.8 s, answer build
~0.65 s, answer exchange ~0.45 s over 1.5 M claims per rank — nearly all local slots at positions
nobody else claims.
- `DENDRO_BCAST_CLAIM_FILTER=1`: ghost slots outside the step keep set do not claim (a ghost claim
  never beats a local one; phantom positions are filled whoever wins). recv → ~0, build 1.75 → ~1.3 s.
  Gates green: d11 np8 sfc = lw 18 digits, ragged ENUTS, BSSN 8 md5s, EM4 np8 forced-bcast, tp ×4.
- `=2`: local slots no ghost element references and the exchange does not send resolve on-rank
  (winner = lowest local cg at the position, intra copy); a phantom position matching a skipped slot
  rebuilds at 1. Claims 1.5 M → 0.41 M per rank, build 1.75 → 0.72–0.87 s; remaining cost is the local
  sort (~0.4 s).
