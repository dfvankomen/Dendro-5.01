# Plan — sandwich cuts, in the order the Kingspeak profiles rank them (2026-09-26)

From `docs/plan_2026-09-25_kingspeak.md` (profiles at 48/96/192 ranks), `docs/audit_2026-09-25_s2g_cost.md`
and `docs/design_2026-09-25_remesh_without_twin.md`. Every cut behind a `DENDRO_*` knob, default off
until its gate passes; local gates = testPartitioning four arms incl. ragged at np=4 and np=8, EM4 CSV
md5 both arms at np=4 (`d7bbee89`/`a75a069e`) and np=8 (`d7bbee89`/`32eb202e`), ragged depth-9 ENUTS
18 digits (and its SFC arm, same reference), BSSN q1 tinytest graph and SFC at t_end 6.0 against the
defaults-gate checksums (graph BH `7a18bafe…` / constraints `c1508613…`, SFC `7a18bafe…` / `2ffba783…`).

## Cut 1 — lazy zip plan for the remesh successor (`DENDRO_LAZY_ZIPPLAN_SUCCESSOR=1`)
The mesh `Mesh::ReMesh` constructs is, in the sandwich solvers, only ever the source of the s2g
transport (EM4/BSSN/NLSM build the labelled mesh from it and its only plan consumer is the post-IGT
duplicate sync, which is empty on an SFC mesh at prune level 2). Its constructor still builds the canon
table + zip plan (0.8–1.0 s per remesh at 48–192 ranks, ≈ all of the successor's cost). Measured with the
global `DENDRO_LAZY_ZIPPLAN=1` on Kingspeak: remesh+IGT 1383 → 469 ms (np=48), 1004 → 518 ms (np=192),
CSV md5 identical. The global knob is opt-in because NLSM unzips on the g2s twin (its wavelet decision)
and diverged with it; scoping the laziness to the successor avoids the twin entirely.
Mechanism: a static one-shot flag set by `ReMesh` around its constructor call; the constructor treats it
like the global knob (plan deferred to `ensureZipPlanBuilt`, which builds table + plan + prune on first
use). On a plain-SFC solver the successor is the live mesh and the plan is built at its first unzip —
that is the global-lazy behaviour (EM4 byte-identical), so the SFC arms of every gate are part of the
proof.

## Cut 2 — a cheaper zip-plan election (next; bit-identity item)
`winners` (home-rank arbitration) is 0.96 s at np=48 and 1.06 s at np=192 per rebuild, the only phase
that grows with rank count, and 0.95 s of plain SFC's 0.93 s remesh. Options: neighbour-routed claims
(same volume, no dense pattern) with a check mode comparing winners; or a rank-independent geometric
rule computed locally from the closure with no exchange (changes primaries → bit-identity gate).

## Results
**Cut 1 gate (knob on):** testPartitioning four arms incl. ragged at np=4 and np=8 all IDENTICAL to base;
EM4 np=4 `d7bbee89`/`a75a069e`, np=8 `d7bbee89`/`32eb202e`; proof of fire: successor ctors with zero
zip-plan time 23/24 (labelled arms) and 16/17 (SFC arms; the one is the initial mesh); ragged depth-9
ENUTS and its SFC arm identical to the 18-digit reference; BSSN q1 t_end 6.0 graph `7a18bafe…` /
`c1508613…` / `86470a02…` / `ed1127e4…` and SFC `7a18bafe…` / `2ffba783…` / `86470a02…` / `b50bc05f…`,
all equal to the defaults-gate references. → default on (`=0` reverts), defaults-path gate below.
**Defaults-path gate (knob default on, nothing on the command line):** testPartitioning four arms at
np=4 and np=8 IDENTICAL; EM4 np=4 `d7bbee89`/`a75a069e`, np=8 `d7bbee89`/`32eb202e`; successor ctors
with zero zip time 23/24 and 16/17; ragged and SFC ENUTS at 18 digits. `DENDRO_LAZY_ZIPPLAN_SUCCESSOR`
is on by default (`=0` reverts). Expected at scale (measured with the equivalent global knob on
Kingspeak): sandwich 4.32 → 3.32 s per remesh at 48 ranks, 3.17 → 2.65 s at 192.

## Cut 2 evidence — inside the zip-plan arbitration (`[zipplan-arb]`, max over ranks, ms per rebuild)

| where | claims/rank | bucket | route Alltoallv | sort | elect | back Alltoallv | lists | requests+map | total |
|---|---|---|---|---|---|---|---|---|---|
| dev box np=8 (EM4 d7, labelled) | 292k (84k local, 208k ghost) | 31 | 9 | 36 | 19 | 5 | 49 | 35 | 185 |
| Kingspeak np=48 (EM4 d9, refine) | ~800–900k | 151–164 | 131–136 | 149–151 | 65–68 | 88 | 240–244 | 189–191 | ~1020 |
| Kingspeak np=192 | ~385–520k (≈25k local) | 79–82 | 375–383 | 76–85 | 31–32 | 408–429 | 160 | 106–114 | ~1130 |

At 48 ranks the cost is per-claim CPU (bucket, sort, lists, map ≈ 0.8 s); at 192 the two dense
Alltoallv rounds are 0.8 s of 1.13 s. Both are proportional to the claim count, and at 192 ranks
94 % of the claims are ghost claims (the 3-ring closure of 117 local elements).

**Cut 2a — `DENDRO_ZIPPLAN_CLAIM_FILTER=1`:** a ghost claim never wins and only feeds the side-channel
lists, which level-2 pruning keeps for a ghost slot only if it is read and undelivered or an exchange
send source; skip every other ghost claim before the arbitration (same lists after the prune; ghost-only
positions elect among the kept claims). Local A/B, EM4 both arms np=4 and np=8: md5 `d7bbee89` /
`a75a069e` / `d7bbee89` / `32eb202e` (identical); labelled np=8: claims 292k → 87k (288k ghost claims
skipped, 3.5k kept), `winners` 185 → 35 ms; SFC np=8: 210k → 84k, 45 ms. Full gate running
(chain 11); Kingspeak filter on/off pair at 48 and 192 ranks queued (EXP=S).

**Cut 2a on Kingspeak (EXP=S, one round each, lazy successor on in all arms; `winners` = arbitration
ms per rebuild, max over ranks; wall = whole 154-step run):**

| np | arm | claims/rank | winners | sandwich per remesh (g2s / remesh+IGT / s2g) | remesh(oct+LB) total | wall | md5 |
|---|---|---|---|---|---|---|---|
| 48 | refine | 824k | 967 | 279 / 396 / 2651 = 3326 ms | 41.6 s | 172.2 | `1e27c7ab` |
| 48 | refine + filter | 163k (539k skipped) | 199 | 306 / 359 / 1610 = 2276 ms | 28.4 s | 157.3 | `1e27c7ab` |
| 48 | sfc | 535k | 621 | — | 4.4 s | 140.4 | `af968a0b` |
| 48 | sfc + filter | 162k (310k skipped) | 198 | — | 4.3 s | 132.3 | `af968a0b` |
| 192 | refine | 396k | 1152 | 181 / 222 / 2080 = 2482 ms | 32.4 s | 85.0 | `68191978` |
| 192 | refine + filter | 41k (262k skipped) | 95 | 195 / 180 / 834 = 1209 ms | 15.7 s | 65.9 | `68191978` |
| 192 | sfc | 224k | 592 | — | 2.5 s | 59.6 | `af968a0b` |
| 192 | sfc + filter | 40k (155k skipped) | 88 | — | 2.5 s | 51.9 | `af968a0b` |

(SFC's remesh(oct+LB) is small here because the lazy successor moved SFC's own plan build to the
first unzip; its wall is the honest number.) Sandwich per remesh at 192 ranks: 3.17 s yesterday →
2.48 s (cut 1) → 1.21 s (cut 1 + 2a). Wall vs SFC (both arms with the same cuts): 1.27× at 192
(was 1.56×), 1.19× at 48 (was 1.31×). What remains in s2g at 192 ranks (0.83 s): the rebuild's other
phases (~0.5 s: scatter map, audit, E2N expand, block decomposition) and the transport.
**Cut 2a gates:** knob on — tp four arms np=4/8 IDENTICAL, EM4 both arms np=4/8 on the references, ENUTS
ragged + SFC 18 digits, BSSN graph/SFC bit-for-bit vs the defaults-gate checksums; defaults path (knob
default on, `DENDRO_ZIPPLAN_PROF=1` for proof) — the same, with the `[zipplan-prof]` lines excluded from
the tp comparison (they contain the word CONTESTED) and 89–512k ghost claims skipped per rebuild as
proof of fire. `DENDRO_ZIPPLAN_CLAIM_FILTER` is on by default (`=0` reverts).

## Where the sandwich stands (192 ranks, EM4 d9, per remesh)
3.17 s (2026-09-25) → 2.48 s (lazy successor) → 1.21 s (+ claim filter); SFC's own remesh ≈ 0.7 s.
Wall vs SFC 1.56× → 1.27×. Left in s2g (0.83 s): scatter map ~150, block decomposition ~170 (incl.
the remaining zip-plan work ~95), audit ~75, E2N expand ~60, transport ~200 ms; g2s 0.2 s; remesh+IGT
0.18 s. Next candidates: neighbour-routed claims (the two Alltoallv rounds are 108 of the 95+ ms
now — small), the transport's directory rendezvous, the scatter-map rebuild, and inherited labels.

