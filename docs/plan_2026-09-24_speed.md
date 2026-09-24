# Plan — make the labelled path fast, then show it pays (2026-09-24, night)

Follows `plan_2026-09-24_ragged_labelling_gates.md` (correctness is green on ragged labellings; the
labelled rebuild is 27 collectives). Local only; every change opt-in or byte-identical, gated by the
existing scripts (testPartitioning ×4 arms incl. ragged, EM4 CSV md5, ragged d9 ENUTS 18-digit); each
sub-step a jj commit. The box throttles, so wall-clock is only ever read as a ratio inside one
interleaved round, and the decisions are made on deterministic proxies (bytes, collectives, phase
shares, `lts_substep_eff`, `padded_vol_imbalance`).

## Step 0 — where the time goes now (done, np=4, EM4 d7 eOrder 6, 3032 elements, 8 remeshes, 2 rounds)

| arm | rkstep s | unzip_async s | deriv | rhs | zip | remesh(oct+LB) s | bytes sent | max-rank bytes |
|---|---|---|---|---|---|---|---|---|
| SFC | 25.8 / 28.0 | 16.8 / 18.2 | 3.8 | 1.1 | 1.0 | 2.6 / 2.8 | 4.81 GB | 1.32 GB |
| plain fastpart (method 3) | 35.7 / 36.3 | 25.5 / 25.9 | 4.4 | 1.3 | 0.9 | 8.7 / 8.9 | 7.37 GB | 2.13 GB |

Reading: `unzip_async` (ghost exchange + unzip) is 65 % of the step on SFC and 71 % on the labelled
mesh; the labelled excess (+8.7 s) tracks the 1.53× ghost bytes of the *unrefined* fastpart labelling,
not a slower unzip per byte. The sandwich costs ~0.77 s per remesh over SFC at this size. Refine and
levelwise arms measured next (Step 1.1) — the refined labelling sends 0.93× SFC bytes in the earlier
in-Dendro count, so its `unzip_async` should sit at SFC level; whatever is left is the per-step tax of
the labelled plumbing itself.

**Refine / levelwise arms (same run, `DENDRO_GRAPH_REFINE=1|2`):** on EM4's symmetric mesh both are
SFC-identical labellings (`[block-vote] reassigned 0`, CSV md5 = SFC, bytes = SFC). EM4 therefore
exercises only the plumbing on those arms; the ragged arm for EM4 is plain method 3.

**Sandwich split (`DENDRO_REPART_PHASE_PROF=1`, mean of 8 remeshes, np=4):** buildZipPlan 399 ms,
e2n-audit 225 ms, buildUnzipCanonWriter 117 ms, nodal-scatter-map 43 ms, rest < 25 ms; `[graph-remesh]`
s2g ≈ 1.0 s vs remesh+IGT ≈ 0.6 s. Read-only audit: all three are hash-map bound (about 1.5–2 M
allocating inserts per rebuild), with a full thread-local→global re-insertion when nThr == 1, a
map-of-vectors canon table, and a fixup block whose output nothing reads by default.

**The per-step finding (gated tally `sync_bytes_sent` in `[comm-stats-cumulative]`):** the zip-plan
side-channel `syncZipNonPrimary` runs twice per ghost-exchange batch on *every* mesh with P > 1 (before
the sends in `readFromGhostBegin`, again in `readFromGhostEnd`) and its recv list is every ghost cg on
the rank, so it moves far more than the exchange itself:

| arm | ghost-exchange bytes | zip-sync bytes | ratio |
|---|---|---|---|
| SFC | 4.81 GB | 22.4 GB | 4.7× |
| plain fastpart | 7.37 GB | 43.8 GB | 5.9× |

That is 82–86 % of all per-step traffic, plus 2 blocking Alltoallv per batch (8 per RK stage at
`ASYNC_COMM_K=4`), on the fork's SFC arm too — upstream Dendro has none of it. `pruneZipSync`
(opt-in `DENDRO_ZIPSYNC_PRUNE=1|2`) keeps only non-primary local slots, ghost slots the exchange does not
deliver, and send sources; level 2 also drops undelivered ghosts nothing on the rank reads. Level 1 is
byte-identical on both EM4 arms (md5 d7bbee…/a75a069…) and halves the sync volume (22.4 → 11.8 GB SFC,
43.8 → 27.7 GB labelled). Level 2 empties the side-channel on the SFC arm (0 bytes, 0 collectives)
and leaves 3.3 MB of 43.8 GB on the labelled arm; EM4 md5 identical on both arms, ragged d9 ENUTS
identical to 18 digits (min/max/l2), testPartitioning PASS gates identical on all four arms — the only
change is the informational TEST 8bk asymmetric-cg count, which scans the now-stale unread ghost slots.
Level 2 is the default (`DENDRO_ZIPSYNC_PRUNE=0` restores the full side-channel). Per-step traffic on
EM4 np=4: SFC 27.2 → 4.8 GB, labelled 51.2 → 7.4 GB. Also landed byte-identically: the unzip element
order is cached per mesh instead of sorted on every batch.

## Step 1 — the per-step tax

1. Same profile for `DENDRO_GRAPH_REFINE=1` and `=2` (levelwise), with `DENDRO_REPART_PHASE_PROF=1`
   for the sandwich split. Decide from the numbers which of 1.2 / 1.3 matters.
2. Split `unzip_async` into wait vs. compute on both mesh kinds (existing `DENDRO_MESH_PROF`, or two
   timers in the EM4 ctx around `readFromGhostEnd` and `unzip`). If the labelled unzip is slower *per
   ghost byte* than the SFC block unzip, the plan-zip slice path is the target (a per-block contiguous
   plan instead of per-entry gather; OpenMP is off in production, so it must be single-thread work
   reduction).
3. Per-step labelled-only work (agent audit of ets.h / enuts.h / mesh.tcc): anything that runs per RK
   stage on labelled meshes but is mesh-invariant gets cached per mesh (the pos-bcast plan already is);
   anything O(N_global) or O(P) per stage is a defect at scale and gets replaced. Each behind a knob for
   one A/B, then default; gate = EM4 md5 + ragged ENUTS 18-digit.

## Step 2 — the sandwich tax (per remesh)

- Phase split of `repartitionMeshGlobal` from `DENDRO_REPART_PHASE_PROF=1`; attack the top O(local)
  phase (earlier split: buildZipPlan 51 %, e2n-audit 25 %).
- Remaining rounds from the audit: `computeMinMaxLevel` as one Allreduce (−3), fused request/response
  rounds, keep-mask shrink in one Alltoallv, IGT per-variable Alltoallv batching (27 → ~20).
- Deterministic gate: `[repart-rounds]` count, EM4 md5, testPartitioning 4 arms.

## Step 3 — the win side (this is the "does it pay" evidence the box can give)

1. **LTS balance:** `nlsmNUTS` d9 puncture, SFC vs levelwise+refine, np=4 and np=8 (`--oversubscribe`):
   `lts_substep_eff`, `dump_est_speedup`, `LD Bal`, and the interleaved wall-clock ratio (≥3 rounds,
   ratio within a round only). Expected from the offline study: eff 0.53–0.87 → ~1.00 at every np.
2. **Padded-volume balance:** give fastpart's `RefineByGain` a balance phase (`refine_balance=1`,
   dendro-refine branch): after the gain rounds, boundary groups leave parts above target for the
   neighbour part with the most slack, ranked by least cut loss; then score `padded_vol_imbalance` at
   np 8/16/32 on the puncture mesh (`testPartitioningEvolve`, oversubscribed — deterministic). Target:
   below SFC's 1.07 at np=4 and a widening gap with np, as the thesis handoff predicts.
3. Write the result table into `docs/` and the vault: per-step tax (bytes, collectives), sandwich tax
   (s per remesh, collectives), LTS eff, padded imbalance — and the honest statement that only the
   cluster bundle (`tools/cluster_bundle`, ready) turns proxies into seconds.

## Results (night, all on the jj line after `nsurysmt`)

**Per-step.** `pruneZipSync` default level 2; per-step traffic EM4 np=4 SFC 27.2 → 4.8 GB, labelled
51.2 → 7.4 GB; collectives per RK stage on SFC 8 → 0. ENUTS per-block dependency lists and the unzip
element order cached per mesh.

**Sandwich (np=4, EM4 d7, ms per remesh, box loaded so rough):** e2n-audit 290 → 163, buildZipPlan
514 → 240, buildUnzipCanonWriter 128 → 9 (four byte-identical commits: no nThr==1 re-insertion, dense
canon fold, gated fixup map, cg-indexed winners). Remaining hash-bound work: `homeWin`/`myMin`
(sort+fold candidate), the audit's on-demand maps.

**Padded-volume objective (fastpart `dendro-refine` balance phase, opt-in `DENDRO_GRAPH_BALANCE_ROUNDS`
/ `_TOL`; puncture evolve mesh `PARTEVOLVE_INIT=puncture`, 11.6k elements, eOrder 4):**

| np | arm | padded imb | work imb | lts eff | ghost_max |
|---|---|---|---|---|---|
| 4 | SFC | 1.074 | 1.131 | 0.850 | 70k |
| 4 | levelwise + w2 | 1.023 | 1.066 | 0.915 | 75k |
| 4 | levelwise + w2 + balance | 1.007 | 1.040 | 0.942 | 84k |
| 4 | REFINE=3 + w2 + balance | 1.014 | 1.073 | 0.896 | 69k |
| 8 | SFC | 1.092 | 1.191 | 0.767 | 64k |
| 8 | levelwise + w2 + balance | 1.033 | 1.111 | 0.893 | 88k |
| 8 | REFINE=1 + w2 + balance | 1.015 | 1.160 | 0.790 | 63k |
| 8 | REFINE=3 + w2 + balance | 1.018 | 1.355 | 0.679 | 66k |
| 16 | SFC | 1.200 | 1.228 | 0.695 | 52k |
| 16 | levelwise + w2 + balance | 1.057 | 1.170 | 0.842 | 78k |
| 16 | REFINE=1 + w2 + balance | 1.039 | 1.433 | 0.620 | 48k |
| 16 | REFINE=3 + w2 + balance | 1.019 | 1.473 | 0.594 | 52k |

The thesis objective is met and widens with np (busiest-rank padded volume −15 % at np=16 at SFC-level
ghost volume) but fights LTS balance; levelwise's per-level caps bind (max_imb 1.33–1.52) and leave few
balance moves. Choose per solver: global RK → `REFINE=3`+w2+balance; LTS → levelwise+w2+balance. All
fork builds link `fastpart/build_balance/install` (the struct grew; never mix the old `.so` with new
binaries). Gates for the whole line: EM4 md5 ×2, four testPartitioning arms, ragged d9 ENUTS 18 digits.

**Still to do:** BSSN long-haul under the new default; a one-group floor in the balance budget; the
remaining sandwich hash maps and rebuild rounds; wall-clock only on a cluster.

## Step 4 — cluster (only with the user's OK)

Update `tools/cluster_bundle` to the new tip and the arms of Step 3; the user submits.

## Not in this plan

Design A (labelled ReMesh without the twin), scaffolding deletion, the `experimental` reconcile (backup
first, after the partitioning is settled).

## Round 2 (2026-09-24, late night) — after the prune, what is left per step and per remesh

Per-step, SFC arm: 0 collectives per stage, but `unzip_async` is still 12.4 of 21.8 s. Per-step,
labelled arm: 819 sync calls carrying 3.3 MB total remain (the mirrors' sources and non-primary locals).
Per remesh: buildZipPlan 240 ms and e2n-audit 163 ms are still hash-bound; the rebuild is 27 collectives.

1. **Split `unzip_async`** into exchange wait vs unzip compute (`DENDRO_MESH_PROF=1` accumulators in
   `readFromGhostEnd` and `unzip_scatter`, printed with the cumulative comm stats). If compute dominates,
   the plan-based `unzip_scatter` is the target: on SFC meshes the canonical-writer table is a no-op
   and `n_phases` is 1, so the per-slot gating can be skipped; the per-element interpolation path should
   be compared against upstream's block unzip.
2. **Labelled arm to 0 collectives per stage**: fold the surviving sync entries into the nodal scatter
   map (they are (primary rank, primary cg) → (rank, slot) pairs, exactly a scatter-map entry), so the
   side-channel disappears and only the p2p exchange remains. Byte-identical by construction if the
   delivered value is the same primary value; gate = the usual four.
3. **Sandwich hash maps** (agent, workspace `rounds`): `homeWin`/`myMin` as sort + fold with the
   insertion-order tie rule; audit maps built on demand from the query keys; `unify` via sort.
4. **Rounds** (same agent, after 3): `computeMinMaxLevel` as one Allreduce; fused request/response
   rounds where the request payload is small; IGT per-variable Alltoallv batching.
5. **np=8 gates** for the four arms + EM4 md5, then `DENDRO_REDIST_ROUTE` / `_FLAGS` defaults.
Every item byte-identical behind a knob for one A/B, then default; BSSN A/B (running) gates the prune.

**Round 2 results so far (late night; box saturated, so only byte-identity and the phase *split* are
read):**
- `unzip_async` split (`[unzip-prof]`, EM4 SFC np=4): ghost_begin 0.8 s, ghost_end 8.6 s, **unzip
  compute 49–63 s** — the plan-based `unzip_scatter` is ~85 % of the phase, the exchange wait ~15 %.
- `DENDRO_UNZIP_BLOCK=1` (SFC meshes take upstream's block-wise unzip): **byte-identical on the EM4 SFC
  arm** (md5 d7bbee…), labelled meshes fall back to the scatter (md5 a75a069…). So the scatter equals
  upstream's unzip on SFC meshes; whether the block path is faster needs a quiet box (both arms ran
  under load 20+, unusable).
- Pruned sync is now point-to-point among the surviving partners (no O(P) collective); byte-identical on
  all four gates.
- np=8: EM4 SFC md5 equals the np=4 one (d7bbee…), graph np=8 base 4521dbc3…; `DENDRO_REDIST_ROUTE=1`,
  `DENDRO_REDIST_FLAGS=1` and both together are byte-identical at np=8 on every arm → **both default on**
  (`=0` reverts; the flags knob lives in EM4's `solverCtx.cpp`, user commits). Rebuild at np=8: 28
  collectives.
- BSSN q1 tinytest A/B (SFC / graph full side-channel / graph pruned): SFC arm done, graph arms running.
- Defaults-path gate of the whole line (route + flags on, p2p sync, timers, block-unzip knob off): EM4
  md5 at np=4 and np=8 both arms, four testPartitioning arms at np=4 **and np=8**, ragged ENUTS 18
  digits — all identical. Rebuild at np=4 is 28 collectives with the routed redistribute (+1 for the
  O(local) directory in place of the O(N_global) gather).
- **BSSN, the reader the prune exposed:** the first pair (both arms SFC by a TOML-table accident,
  prune 0 vs 2) differed — constraints 5th digit, BH positions visibly — and a repeated run proved the
  solver deterministic. Cause: `BSSNCtx::post_timestep` → `computeBHLocations` interpolates the shift
  from the zipped vector after the final axpy with its ghost exchange commented out (fork and upstream),
  so it reads fresh local nodes next to one-stage-stale ghost nodes; the side-channel's post-axpy sync
  had masked it. Fix in the BSSN repo (uncommitted): exchange the three shift components before the
  tracker. Four-arm A/B (sfc/graph × prune 0/2) on the fixed solver running; the bar is
  sfc_p0 = sfc_p2 and graph_p0 = graph_p2.
