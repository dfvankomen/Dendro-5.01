# Hybrid performance levers

## Framing rule

A pure-MPI run parallelizes every phase of the time step, because every rank
runs every phase. A hybrid MPI+OpenMP build is therefore complete only when
every phase inside a rank is threaded: unzip, RHS, zip, ghost pack/unpack, and
the RK vector operations. A hybrid build that threads only the block loops is
an incomplete hybrid build, and its timings say nothing about hybrid as a
method. The levers below are that complete stack.

## How to read the table

- Gains are same-job ratios on the named machine. Rung gains (R06 to R11) are
  cumulative ladder steps measured on the combined stack, not isolated A/Bs of
  one commit. Do not sum or multiply them.
- "Gate" is the run that compared whole-run solver output with the lever on
  and off. The gates were passed by the thesis-base commits named in the
  ledger (`int/thesis-opt`). This branch carries the same patches on
  `experimental` and has not been through the cluster gates itself (see open
  items).
- Machines: BYU = AMD EPYC (znver3); Kingspeak (KP) = Intel Broadwell.
- Sources: **L** = thesis appendix `A-dendrolib-ledger.tex`, table
  `tab:ledger-campaign`; **M** = `thesis_data/MANIFEST.md`; **P** = vault note
  "Post-factorial optimizations"; **D** = vault note "Dead Ends".

## Levers

| Lever | What it does | Switch (default) | Measured gain (machine) | Gate (job) | Source |
|---|---|---|---|---|---|
| LPT block order | Visits unzip and RHS blocks largest first so threads finish together | runtime `BSSN_HYBRID_RHS_SCHEDULE="lpt"` (solver parameter) | rung R06, 1.03-1.09x per step (KP); base mesh 1.08x (KP, 8 nodes) | identical output R05-R11; KP 28407047 | L |
| Zip ownership plan | Replaces zip's per-point divide and ownership test with a per-mesh copy list | `DENDRO_ZIP_PLAN` (ON) | zip phase 1.49x (BYU), 1.30-1.58x (KP) | 24/24 files identical; 13887219, 28405623 | L |
| Threaded ghost pack/unpack | OpenMP over (neighbor, variable) pairs instead of neighbor only | `DENDRO_GHOST_PACK_FLAT` (ON) | pack 1.14x (BYU), 2.08x (KP); rung R08 | 24/24 files identical; 13887219, 28405623 | L, P |
| Unzip plan | Caches per-element hanging interpolations and replays the batched unzip as a per-cell copy plan; 75% of cells copy straight from CG | `DENDRO_UNZIP_PLAN` (ON); active only in an OpenMP unzip build | unzip 1.25x (BYU), 1.50x (KP); rung R09, 1.04-1.10x | 24/24 files identical; 13887219, 28405623 | L, P |
| Derivative pre-pass trim | Computes the mixed-derivative feeders only on points the chains read | solver `BSSN_DERIV_TRIM` (ON); the library routine is additive and has no switch | rung R10, 1.00-1.02x | identical output; `testMixedSetTrim` | L |
| RK fusion hooks | Builds each stage input and the final combine in one pass with constraint enforcement; about 29 to 17 state streams per step | `DENDRO_RK_FUSE` (ON), solver `BSSN_RK_FUSE` (follows it) | rung R11, 1.01-1.06x | identical output | L |
| MSRK through RK fusion | Routes MSRK stage input and combine through the same hooks and swaps history instead of copying | under `DENDRO_RK_FUSE` | MSRK2_1 1.12x, MSRK3 1.23x (KP, 8 nodes) | identical frozen and live; KP 28412414 | L, M |
| Block-size cap | Caps a merged block at 2^s elements per side, so one large block no longer sets the hybrid step | `OCT2BLK_MAX_SPAN` (2, at most 4^3 elements; 31 = no cap); env `DENDRO_OCT2BLK_MAX_SPAN` overrides at run time | cap 1: q12 hybrid 1.32x (16 nodes, 13937983), 1.16x (8 nodes, 13937984); cap 2: q12 RK4 ladder 1.48x to 1.91x (16 m12 nodes) | 80/80 files identical (13943757, 13947001); identical across live remeshes on q1 (KP 28500424) | L, M |
| Cap default of 2 | Cap 1 cost about 5% on q1; cap 2 does not | as above | q1 R11 0.4190 to 0.4187 s (KP) | identical output; KP 28502391, BYU 13953639 (q1 has no oversized blocks, so this gate cannot show the cap activating) | L, M |
| Threaded TwoPunctures (solver) | Evaluates the initial data over local nodes in a threaded loop without per-point MPI calls | none (threads under `DENDRO_HYBRID_OMP`) | hybrid startup 315 to 63 s (q12, 4 nodes) | 80/80 files identical; 13900868 | M |
| TwoPunctures release (solver) | Frees the restored spectral solution once the initial grid is built | none | pure-MPI node memory lower by 11.6 GB (q12) | identical output; 13909638 | M |

Whole-stack results over the August baseline, for orientation (M): Kingspeak
8 nodes q1 live AMR 2.67x (RK4) and 3.44x (MSRK2_1), job 28412908; BYU 1 node
q1 frozen 2.06x and 2.68x, job 13899726; q12 on 8 m12 nodes with the cap
1.95x and 2.52x, job 13943757; q24 on 16 m12 nodes 1.88x (RK4), job 13947212;
q12 at production output cadences 1.90x (RK4), job 13947211. Nearly all of
the gain from the plan and pack levers is hybrid-only: the same-job A/B moved
the hybrid step 1.10x (BYU, 13887219) and 1.23x (KP, 28405623), and the
all-on pure-MPI step 1.025x and 1.003x (P).

## Instrumentation (off unless requested)

| Name | What it prints | How to enable | Source |
|---|---|---|---|
| Block dependency statistics | Per rank: ghost-independent and ghost-dependent block counts, elements, and padded-volume share | env `DENDRO_BLKDEP_STATS` | library commit `b885d82` |
| Per-rank CSV (solver) | Per rank per step: phase times, elements, blocks, padded volume, send and receive counts | env `BSSN_PERRANK_CSV` (read on rank 0 and broadcast); output identical on and off (24/24) | P |

## Tried and not adopted

| Attempt | Deciding number | Source |
|---|---|---|
| KO dissipation interior trim (`opt/ko`) | output identical, but 0.59-1.08x | P |
| jemalloc with transparent huge pages (BYU) | 1.012x | P |
| Single-element blocks (`OCT2BLK_COARSEST_LEV=31`) | hybrid 1.14x at 16 nodes but 0.93x at 8 nodes on q12 (13934841, 13934842); the cap replaced it | M |
| Cap 1 as the default | costs about 5% on q1 (0.419 to 0.442 s, KP 28500423) | M |
| `DENDRO_UNZIP_OVERLAP` (unzip interior blocks during the exchange) | single node: 8x1 went 64 to 73 s, 4x2 went 61 to 78 s | D |
| Threading the unzip over elements | changes the answer: max abs 2.4e-2 against the serial reference | D |
| Threading the `e2n.1 dginit` loop | correct and race-free, 3.2x slower | D |
| Keeping state in block layout across RK stages | not attempted: the RK vectors live in zip layout | P |

## Open items

- **This branch is not cluster-gated.** The gates above ran on the thesis
  base. The integration on `experimental` needs its own runs: a switches-off
  build compared with `experimental`, the all-on ladder on q1 (frozen and
  live), and q12 with the cap.
- **Constraint-kernel half of the derivative trim.** The thesis-base stack
  dropped it (P), so the ladder gates did not cover it. It is included here
  under `BSSN_DERIV_TRIM` in the solver.
- **Unzip plan memory.** The copy plan stores 4 bytes per unzipped cell.
  Decide before it runs on memory-limited production meshes.
- **LPT order is re-sorted on every unzip call.** It can be cached per mesh.
- **Gate coverage.** The gates are whole-run output identity on EPYC (znver3)
  and Broadwell only. RK fusion depends on FMA contraction matching between
  the fused and unfused code; it does on those two targets.
- **No GPU check of the cap default.** CUDA builds force single-element
  blocks, so the default of 2 has not been exercised there.
- **MSRK stability.** MSRK aborts on the q12 and q24 production meshes at CFL
  0.25 (13966772, 13947212). This is a scheme stability question; diagnosis
  is pending.
