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
- **V np=48** (28414344): sfc 281.8 / 281.8 · pv 335.3 / 334.7 · pv_full 343.3 / 343.3 — prune −8 s (−2.4 %),
  labelled vs SFC 1.22× → 1.19× (the dense Alltoallv it removes grows with np; −14–21 s at 96).
- Knob-on gates at =2: d11 np4 + np8 sfc = lw 18 digits, ragged ENUTS, BSSN 8 md5s, tp ×4, EM4 np8
  forced bcast on reference — **but EM4 forced hits the phantom check every rebuild (~100k) and falls
  back to 1**: EM4's minimal scatter drops exchange sends, so remote undelivered+read ghosts sit at
  "interior" positions. EM4 does not run the pos-bcast by default. NLSM/BSSN never fell back.
- Flip (jj `yzowpssm`, bin `cdef`): defaults path green (tp ×4, EM4 np4/8, d11 np8, ragged ENUTS, BSSN).
  Kingspeak X (build 28414412, np=48 28414413, np=96 28414414): L case sfc / lw / lw_f0 (`=0`).

## Kingspeak X (build 28414412; L case, 2 rounds, wall s)
| np | sfc | lw, filter 0 | lw, filter 2 (default) | lw/sfc |
|---|---|---|---|---|
| 48 | 144.9 / 144.8 | 191.9 / 192.7 | 179.8 / 179.2 | 1.33× → 1.24× |
| 96 | 97.1 / 95.3 | 136.9 / 137.2 | 125.8 / 126.4 | 1.43× → 1.31× |
Election per rebuild 1.7 s → 0.26 s (np=48), 1.57 → 0.15 s (np=96). Remaining per rebuild at np=96:
`[repart] TOTAL` ~1.75 s (10 rebuilds), g2s ~1.0 s on the init sandwich.

## The labelled rebuild (NLSM d11 lw, np=8, phase prof)
repart TOTAL ~3.8 s: buildZipPlan 2.2 s (claims 0.64, home arbitration 1.2 over 1.5 M local claims),
e2n-audit 0.5 s, buildE2NWithSM 0.29 s, unzip canon writer 0.13 s, buildE2NMap 0.11 s.
- `DENDRO_ZIPPLAN_LOCAL_ARB` (default on, jj mpmxmslz + uurlksqq): a position whose canonical
  position is not a node of any ghost element, holds no ghost claim of mine and no zip mirror slot is
  folded on-rank (same fold, same claim order); only ~4 % of claims are routed. A position the home
  saw with only ghost claims that a rank settled locally is re-routed (targeted, recursive) — never
  fired on any gate. First try marked every node of round-1 send elements + exchange send sources:
  71 % routed on the fragmented levelwise mesh.
- `DENDRO_ZIPPLAN_MYMIN_HASH` (default on): myMin grouped by an open-addressing table instead of a
  sort (the fold is a strict total order, so order-free): mymin 430 → 280 ms.
- `DENDRO_E2N_AUDIT_LAZY` (default on): the audit's sorted (pos, cg) index (~470 ms) is built on the
  first lookup; most rebuilds make 0–223 lookups. e2n-audit 500 → 146 ms.
Zip plan 2.2 → ~1.2 s; gates (knob on and defaults path) all bit-for-bit: tp ×8 (np4+np8, ragged),
EM4 ×4, d11 np4+np8 sfc = lw 18 digits, ragged ENUTS, BSSN 8 md5s.
Note: at np=96 per-rank data is ~12× smaller than np=8 locally, yet repart TOTAL is 1.75 s — most of
it is not local compute. Kingspeak Y (build 28415655, np=48 28415656, np=96 28415657): sfc / lw /
lw_old (the three cuts reverted), full phase profiles.

## Kingspeak Y and Z (L case, 2 rounds, wall s)
Y (build 28415655: local arb + myMin hash + lazy audit vs those reverted): np=48 172.7/172.8 vs
176.1/175.2 (−3 s), np=96 125.5/124.0 vs 125.2/125.6 (0) — local compute was not the np=96 cost.
Y's np=96 phase profile: ghost nodes per rank ~2× local (labelled 24.7 M vs SFC 13.8 M send nodes), so
every phase walking local+ghost elements stays flat with np; block-decomposition tail 334 ms
(hash sets), E2NWithSM 256 ms (sort-unique), zip `inv` 102 ms (hash of every recv entry).

Z (build 28415800 = defaults through jj lkzqtxqx; lw_old = every sandwich cut of 2026-09-27/28 reverted):
| np | sfc | lw_old | lw | lw/sfc |
|---|---|---|---|---|
| 48 | 137.8 / 137.4 | 183.6 / 187.9 | 166.8 / 164.7 | 1.35× → 1.20× |
| 96 | 87.7 / 87.0 | 127.3 / 125.8 | 112.5 / 108.8 | 1.45× → 1.27× |
| 192 | 67.5 / 66.7 | 124.1 / 110.3 | 89.9 / 89.0 | 1.75× → 1.33× |
np=192 rebuild 0.96 s: e2n-audit 250 (4197 lookups → index built), sm 222, zip plan 210, E2NWithSM 101,
E2NMap 63; bcast election 131; g2s 564 (init), SFC remesh+IGT 290–470.

## Cuts after Z (gated as a batch)
- SlotFlags (vector<char>) for the minimal-scatter keep set and pruneZipSync sets: tail 450 → 25 ms (np8).
- Sorted-unique DG indices by marking in both E2N builders (`DENDRO_E2N_SORT_UNIQUE=1` reverts):
  E2NWithSM 265 → 108 ms.
- `DENDRO_ZIPPLAN_INV_QUERY` (default on): resolve only the asked keys, inv 75 → ~2 ms.
- myMin slot groups (lookups by cg2dg element-node slot; interior nodes skip the table):
  claims 730 → 400, plan 270 → 40 ms.
- E2N audit: owner slot with a canonical writer returns early (walk 130 → 73 ms).
- Local-arb marking skips nodes strictly inside ghost elements (100–230 → 72–132 ms).
- Hoisted interior-writer block lookup in the myMin build.
- Pos-bcast level 2: marking by ghost-element boundary positions, hash grouping, phantom hits routed
  instead of a level-1 rebuild; build ~600 → ~450 ms (np8). EM4 forced bcast: one reroute per rebuild.

## Z2 (build 28415961 = batch through jj vwnvorlx) and where the gap is now
| np | sfc | lw_old | lw | lw/sfc |
|---|---|---|---|---|
| 48 | 136.5 / 135.8 | 183.3 / 183.3 | 162.9 / 162.9 | 1.35× → 1.20× |
| 96 | 86.6 / 85.6 | 125.2 / 125.2 | 107.3 / 107.9 | 1.45× → 1.25× |
| 192 | 67.1 / 65.4 | 109.7 / 109.3 | 88.8 / 88.4 | 1.65× → 1.34× |
Only −2–3 s over Z: the rebuild internals are in diminishing returns. The NLSM prof line says why: at
np=192 levelwise ranks carry **3.9× SFC's ghost elements** (mean 3 897 vs 999, max 6 203 vs 2 082) for the
same ~293 local elements — levelwise hands each rank a separate piece per level, each with its own ghost
shell. Every remaining rebuild phase (audit walk, zip marking, E2N expansion, sm walk) and the
per-step exchange scale with that shell. Next lever: labellings whose per-level pieces sit next to each
other (fewer ghost shells), or a cheaper per-ghost path; after this batch the global-to-local element map
is an unordered_map (sm decode 66 → 26 ms, gated after Z2).

## 2026-09-28 late — ghost shell, ring-1 marking, matched levelwise, and a correctness bug
- Ghost shell (subagent, `docs/investigation_2026-09-28_ghost_shell.md`): ~1.8× from the labelled rebuild's
  third ghost ring, ~1.3–1.5× levelwise geometry. `DENDRO_REPART_GHOST_RINGS=2` breaks results (tp 7b 100
  errors > 0.5, EM4 md5 moves): R2 canonical ownership needs R3. Kept as A/B knob only (jj pkwymqxy).
- Ring-1-only marking in both elections on labelled meshes (jj wyyynoky, gated): Kingspeak Z3 lw/sfc
  1.17× (np48), 1.21× (np96), 1.32× (np192 — but see below).
- Offline (`edgeCutCompare --oct` on the dumped d11 mesh, `DENDRO_DUMP_OCT`): levelwise ghosts/rank
  1.43–1.69× SFC; matched assignment (`levelwise_matched`) cuts ghosts ~8 %, comm volume ~15 %, lts_eff
  unchanged. Wired as `DENDRO_LEVELWISE_MATCH=1` (opt-in).
- np=48 split: lw spends ~10 s in its 10 rebuilds, ~13–17 s more than SFC in stepping (LD Bal 1.28 after
  block vote; 1.8× ghost volume per sub-step).
- **Correctness bug (pre-existing, bin/def reproduces it):** NLSM d11 LTS at np=16 on fastpart-produced
  labellings (method 3, refine 0 or 2) blows up after the first coarse step (U_PHI ±20 vs ±1.2);
  plain levelwise (method 4) is 18-digit identical to SFC. Kingspeak: lw fine at np48/96, blown up at
  np192 (both lw and lw_old) → the np192 wall ratios compare against a broken run. Investigation →
  `docs/investigation_2026-09-28_fastpart_lts_blowup.md` (subagent).
