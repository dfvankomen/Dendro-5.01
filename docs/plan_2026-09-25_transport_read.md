# Plan — the transport read that depends on the side-channel, then the block-major unzip (2026-09-25)

Follows `docs/handoff_2026-09-25_speed_push.md`. Local only; every change behind a `DENDRO_*` knob with
the default byte-identical until the gate passes; one jj commit per logical change; wall-clock is an
indicator only.

## Task 1 — the open defect

**Known.** Graph-mode BSSN q1 tinytest NaNs its constraint norm at every `DENDRO_ZIPSYNC_PRUNE ≥ 1`,
at step 0, in 6 local constraint slots per rank at level 7 near the punctures; the evolution is
bit-identical between prune levels; pruning the labelled mesh is not the cause. The initial swap
(`swap_to_graph_partition` → `redistributeDVec` → `Mesh::redistributeVec`) transports EV from the last
init-loop SFC mesh whose only ghost refresh was the unzip inside `init_grid`.

**The only source-ghost read in the transport** is step 2 of `redistributeVec`: it packs
`E2N_CG` values of every local element, ghost slots included, flagged non-authoritative. On the
destination those land unconditionally on ghost slots and on local slots that received no
authoritative write. The orphan fill reads source local slots only.

1. **Instrument** (`DENDRO_REDIST_PROBE=1`, header-only in `mesh.tcc`): per transport call, count
   source ghost reads (delivered by the scatter map or not, finite or not) and classify every
   destination non-finite slot by the write path that produced it (auth / non-auth / orphan /
   residual), local vs ghost. Run graph short (t_end 0.4) at prune 2 and prune 0.
2. **Fix** according to what the probe says, behind `DENDRO_REDIST_LOCAL_ONLY` (default off until
   the gate passes):
   - if the NaN enters through non-auth writes from source ghost slots: make every destination
     local slot authoritative — extend the orphan fill to every destination local slot without an
     authoritative write (canonical value from the source local slot by cg2dg phys, the machinery
     that already exists) — and stop pre-filling destination ghost slots from source ghost slots
     (the first exchange + sync on the destination delivers them; anything it does not deliver is
     not read, by the keep-set argument).
   - if instead the source ghost slot is a delivered slot that was never exchanged before the
     transport, exchange before reading (a copy of the source vector through
     `readFromGhostBegin/End`), which is the rule the vault already states.
3. **Gate** (the fix on, prune 2): four-arm BSSN q1 tinytest A/B, `sfc_p0 = sfc_p2` and
   `graph_p0 = graph_p2` bit-for-bit on BH locations, constraints and GW modes; testPartitioning
   four arms incl. ragged at np=4 and np=8; EM4 checksums both arms np=4 (SFC `d7bbee89…`,
   labelled `a75a069e…`) and np=8 (SFC `d7bbee89…`, labelled `32eb202e…`); ragged depth-9 ENUTS
   18 digits. Only then flip the fix default on and the prune default back to 2, and re-run the
   gate on the defaults path.

## Task 2 — block-major unzip from a per-call DG buffer

After Task 1 is green. Byte-identical under the canonical-writer argument (the handoff): build the
per-element DG buffer once per call, then walk blocks in order and copy rows; opt-in knob for one
A/B, interleaved timing as an indicator only, decision on byte-identity + the `[unzip-prof]` split.

## Results

**Probe 1 (`DENDRO_REDIST_PROBE=1`, BSSN q1 tinytest graph, np=4, t_end 0.4, prune 2 vs 0).** The
initial-swap transport is clean at both prune levels: every source ghost slot step 2 reads is a
scatter-map-delivered slot (r1 13862, r2 13862, r3 27422, r0 0 ghost reads; undelivered 0; non-finite
0), no source local slot is non-finite, and the destination vector has zero non-finite slots on every
write path, local and ghost. The handoff's hypothesis (the transport reads source ghost slots only the
side-channel refreshed) is refuted; the NaN enters on the labelled mesh after the swap, before the
step-0 constraint. Next probe: the unzip read set after the first exchange on the labelled mesh, and the
unzip's unwritten padded slots.

**Probes 2–3 (`DENDRO_GHOSTREAD_CHECK=1`: unzip read set scanned before and after every exchange;
`DENDRO_UNZIP_FILLCHK=1`: sentinel-filled padded buffer; the transport probe extended to all-zero
slots by write path).** The diagnosis, in order:
- The unzip never leaves an interior slot unwritten on any mesh; unwritten padding is domain-boundary
  padding only (SFC) or none (labelled). With the sentinel in place prune 0 stays finite, so the NaN is
  not an unwritten slot.
- Under prune 2 the labelled mesh's first exchange leaves all-zero slots in its unzip read set
  (r0: 6 ghost, 5 of them neither scatter-delivered nor in the sync list, plus 6 local; r1: 11 ghost;
  r2: 5 ghost + 6 local; r3: 12 ghost), none under prune 0. Rank 0's six all-zero local cgs
  (325083 … 325268) are exactly the six NaN constraint cgs, and the transport wrote them by the
  non-authoritative path (source ghost slot, no authoritative write on this rank).
- At the transport, every source ghost slot step 2 reads is all-zero under prune 2 (r1 13862/13862,
  r2 13862/13862, r3 27422/27422) and non-zero under prune 0, although all of them are
  scatter-map-delivered slots. Cause: the source is the mesh produced by the init loop's last remesh
  plus intergrid transfer, and no ghost exchange ever runs on it before the swap. `interGridTransfer`
  ends with `syncZipNonPrimary` only (the post-IGT duplicate sync). With full lists that call happened
  to deliver every ghost cg of the transferred vector; pruned, it delivers nothing, and the IGT output's
  ghost slots stay at their malloc contents (zero pages here, NaN heap garbage on the 2026-06 run).
- So the handoff's reader is `redistributeVec` step 2 after all, but the dependency is one level
  deeper: it reads a source vector that never had an exchange. The same holds for the sandwich's s2g
  transport (source = the remeshed SFC mesh after IGT), which is why the earlier check probe saw
  "dropped entries differ only on the first sync after a (re)build".

**Fix (`DENDRO_REDIST_CANON_FILL=1`, default off until the gate passes):** the transport pulls every
destination local slot without an authoritative write, plus every destination ghost slot in the unzip
read set without one, from a source LOCAL slot at the same canonical (cg2dg) position through the
existing orphan-fill machinery; the non-authoritative pass remains only as the fallback for positions
no source rank owns locally. Source ghost slots are then never load-bearing. Cost: the orphan key
Allgatherv grows from "unreferenced slots" to "slots without an authoritative write"; the counts are
printed (`[redist-canon]`) and decide whether a phys-hash rendezvous replaces the Allgatherv.

**Fix, first cut (short graph run, prune 2, `DENDRO_REDIST_CANON_FILL=1`):** constraint C_HAM L2
9.20625e-06, identical to prune 0; the labelled mesh's unzip read set has no all-zero slot at any
exchange. Pull counts on rank 0: 30 local slots without an authoritative write, but 189 248 ghost
read-set slots (661 398 keys gathered globally) — almost the whole ghost read set, since the
transport's element routing rarely gives a destination ghost slot an authoritative write and the first
exchange refreshes them anyway. Second cut: ghost pull restricted to read-set slots that neither the
scatter map nor the side-channel delivers (5–12 per rank in the probes). Gate chain re-run on it.

**Second cut (short graph run):** rank 0 pulls 30 local + 38 ghost slots, 68 keys globally; constraint
9.20625e-06 (= prune 0). Gate chain (tp four arms np=4/np=8 at prune 0 and 2, EM4 np=4/np=8 both
arms, ragged ENUTS, four-arm BSSN t_end 6.0) running with the fix on.

## Task 2 design (implemented behind `DENDRO_UNZIP_BLOCK_MAJOR=1`, default off)
`unzip_scatter_bm`: block → element lists by counting sort of the element → block map (per call,
O(pairs)); blocks walked in order, each block's feeding elements in ascending element index — the
element-major loop's natural order restricted to the block, so the last-writer result is identical with
or without the canonical-writer table (slots belong to one block; within an element the child order is
kept). Nodal values of elements feeding more than one block are gathered once per call into a static
cache (memory = multi-block elements × dof × npe); single-block elements are gathered into a scratch.
Same row-copy scatter with canonical gating as `scatter_grid`; parent→child interpolation recomputed per
(element, block). Falls back to the element-major loop when any unzip experiment or probe knob is set or
the element order is odd. Gate: EM4 md5 both arms, tp four arms, ragged ENUTS; interleaved timing
(`[unzip-prof]`, 3 rounds, SFC np=4) as an indicator only.

## Gate results (fix = `DENDRO_REDIST_CANON_FILL=1`, second cut; prune 2 unless noted)

| gate | result |
|---|---|
| testPartitioning, 4 arms incl. ragged, np=4, prune 2 and prune 0 | all 8 IDENTICAL to base (vote 555 / 0 / 0 / 203) |
| testPartitioning, 4 arms incl. ragged, np=8, prune 2 and prune 0 | all 8 IDENTICAL to base (vote 229 / 0 / 0 / 97) |
| EM4 np=4 SFC / labelled | `d7bbee89` / `a75a069e` (= refs) |
| EM4 np=8 SFC / labelled | `d7bbee89` / `32eb202e` (= refs) |
| ragged depth-9 ENUTS (10 remeshes, vote 204) | U_CHI / U_PHI min, max identical to the 18-digit reference |
| BSSN q1 graph, prune 2, fix on (t_end 6.0) | BH `7a18bafe…`, constraints `c1508613…` = the old graph prune-0 reference (fix off) |

Block-major unzip (`DENDRO_UNZIP_BLOCK_MAJOR=1`, on top of the fix, prune 2): EM4 np=4 SFC / labelled
`d7bbee89` / `a75a069e`, testPartitioning 4 arms np=4 IDENTICAL, ragged ENUTS identical to 18 digits.

**Block-major timing indicator (EM4 SFC np=4, d7, interleaved ×3, box under the BSSN four-arm at load
~8, `[unzip-prof]` unzip compute min/max over ranks; identical md5 on every run):**

| round | scatter (s) | block-major v1 (s) | wall scatter / bm (s) |
|---|---|---|---|
| 1 | 13.09 / 13.23 | 21.20 / 21.63 | 40 / 48 |
| 2 | 13.17 / 13.45 | 21.31 / 21.88 | 41 / 49 |
| 3 | 13.41 / 13.47 | 21.23 / 21.37 | 40 / 47 |

Direction robust (3/3 rounds, same load window): block-major v1 is ~1.6× slower in unzip compute. The
identifiable extra work is the parent→child interpolation recomputed per (coarse element, block) where
the element-major loop computes it once per element; v2 caches the interpolants per element and is
re-gated and re-timed behind the defaults gate. Whatever v2 gives, the knob stays opt-in unless it wins.

## Defaults-path gate (no knobs on the command line: `DENDRO_REDIST_CANON_FILL` on, `DENDRO_ZIPSYNC_PRUNE` 2, labelled meshes pruned)

| gate | result |
|---|---|
| testPartitioning 4 arms incl. ragged, np=4 and np=8 | all 8 IDENTICAL to base |
| EM4 np=4 SFC / labelled | `d7bbee89` / `a75a069e` |
| EM4 np=8 SFC / labelled | `d7bbee89` / `32eb202e` |
| EM4 sync bytes np=4 SFC / labelled | 0 / 3.29 MB (ghost exchange 4.81 / 7.37 GB; full side-channel was 22.4 / 43.8 GB) |
| EM4 sync bytes np=8 SFC / labelled | 0 / 4.82 MB (ghost exchange 7.80 / 10.2 GB) |
| ragged depth-9 ENUTS | identical to the 18-digit reference |
| BSSN q1 graph default / SFC default (t_end 6.0) | graph: BH `7a18bafe…`, constraints `c1508613…`, GW files = graph_p0/p2; SFC: `7a18bafe…` / `2ffba783…`, GW files = sfc_p0/p2 |

**Block-major v2 (parent→child interpolants cached per element), on the defaults binary; box at load
~4.5, interleaved ×3, byte-identical on EM4 both arms, tp four arms, ragged ENUTS:**

| round | scatter unzip (s) | block-major v2 unzip (s) | wall (s) |
|---|---|---|---|
| 1 | 7.16 / 7.24 | 7.86 / 8.02 | 23 / 23 |
| 2 | 7.17 / 7.24 | 7.89 / 7.96 | 23 / 24 |
| 3 | 7.19 / 7.35 | 7.96 / 8.00 | 23 / 24 |

**Verdict:** the interpolant recompute was most of v1's loss; what remains is ~10 % slower than the
element-major scatter, 3/3 rounds. Block-major order buys no locality here — the write pattern was not
the cost. `DENDRO_UNZIP_BLOCK_MAJOR` stays opt-in and the lever is closed (Dead Ends). Wall-clock is an
indicator only; the decision rests on the same direction in every interleaved round at two load levels.

