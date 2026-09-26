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

