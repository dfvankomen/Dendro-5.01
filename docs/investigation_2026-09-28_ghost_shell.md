# Ghost-shell investigation (2026-09-28)

Question: why do labelled (levelwise / levelwise+refine) NLSM d11 meshes carry ~3.9x the ghost
elements per rank of SFC meshes at np=192 (Kingspeak: gele mean 3897 vs 999, lele ~293)?

## Verdict

About half **artifact, half geometry**, and the two factors multiply.

1. **Artifact: the labelled rebuild fetches 3 full 26-neighbour rings; the SFC ctor keeps ~2.**
   `repartitionMeshGlobal` pulls R1 + R2 + R3, each the full 26-closure of the ring before it, with
   no level filter (push path `src/mesh.cpp:20436` `for d = 1..3`, tags at `:20466`; legacy keyset path
   `:20583` R1, `:20607` R2, `:20629` R3). The SFC ctor keeps R1 (full 26 ring) plus R2 = face+edge
   neighbours of the elements sent in R1 with level <= the sender's (`src/mesh.cpp:2654-2838`), and
   has **no R3**. Control experiment: method 1 (OriginalPartition) runs the labelled rebuild on the
   **identical SFC partition**, so it has the same ghost send nodes to the byte. It still has
   **1.70x / 1.74x / 1.82x / 1.86x** SFC's ghost elements at np 8/12/16/24. R3 alone is 39–43 % of every
   labelled ghost layer. There are no duplicates or unreached elements (probe `dupOrMissing=0`,
   `ghostUnreached=0`). The extra rings do **not** add per-step traffic, because minimal scatter drops
   them. They do add rebuild work (E2N build, audit, zip plan, sm walk) and memory.
2. **Geometry: levelwise splits every level on its own, so each rank owns several disjoint
   pieces.** Under the SFC-ctor ghost rule (a model that matches the measured SFC gele to within
   0.1–1.4 %), levelwise needs 1.31x / 1.43x / 1.51x the SFC ghost count at np 12/16/24, and the factor
   grows with np. At np 24: 98 connected pieces vs 24 (4.1 per rank, up to 7), +36 % surface elements,
   1.39x R1, 1.65x per-step ghost send nodes. Refine (lwr) does not compact the pieces: 103
   components, and 335 per-level components vs 234.

Product at np 24: 1.81 (rule) x 1.51 (geometry) = **2.72x** measured. Kingspeak ratios are 3.14 / 3.46
/ 3.90 at np 48 / 96 / 192. That fits a rule factor of ~1.9–2.0 and a geometry factor of ~1.6–2.0,
but this is an extrapolation: no local run above np 24.

## Numbers (NLSM d11, t_end 0.02; per-rank means; np 8 is degenerate: all labellings equal SFC)

Column key:
- `ghost`: gele from the prof file.
- `R1/R2/R3`: the fetch rings, recomputed by BFS.
- `SFC-rule`: R1 plus face+edge+vertex level<= neighbours of R1, i.e. the SFC ctor's rule applied to
  this labelling.
- `send nodes`: total ghost send nodes, from part-stats.

| np | arm | lele | ghost | ghost / SFC | R1 | R2 | R3 | SFC-rule | pieces | send nodes |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 | sfc (m0) | 3515 | 3403 | 1.00 | | | | | 16 | 5.85M |
| 16 | orig (m1) | 3515 | 6199 | 1.82 | 1519 | 2003 | 2677 | 3477 | 16 | 5.85M |
| 16 | lw (m4) | 3515 | 8185 | 2.41 | 2074 | 2973 | 3137 | 4994 | 45 | 9.33M |
| 16 | lw+refine | 3575 | 7654 | 2.25 | 1881 | 2572 | 3201 | 4391 | 36 | 7.56M |
| 16 | fastpart | 3573 | 7293 | 2.14 | 1907 | 2468 | 2919 | 4318 | 20 | 7.62M |
| 24 | sfc | 2343 | 2804 | 1.00 | | | | | 24 | 7.13M |
| 24 | orig | 2343 | 5216 | 1.86 | 1220 | 1682 | 2314 | 2859 | 24 | 7.13M |
| 24 | lw (m4) | 2343 | 7636 | 2.72 | 1692 | 2738 | 3207 | 4354 | 98 | 11.80M |
| 24 | lw+refine | 2372 | 7661 | 2.73 | 1708 | 2616 | 3337 | 4231 | 103 | 10.93M |

np 12 (not tabulated): sfc 4177, orig 7286 (1.74x), lw 9314 (2.23x), lwr 9982, fp 8491.

Applying a level filter (same-or-coarser neighbours only) to R2 and R3 saves little. At np 24 lw it
shrinks R2 by 3 % and R3 by 14 %, because 91 % of elements sit on L7/L8. Only dropping R3 moves the
number: lw np 24 would go from 7636 to ~4430 per rank, 1.58x SFC.

## Why R3 exists (so it can't just be deleted)

The comment at `src/mesh.cpp:20629` and `docs/handoff_2026-07-15_minimal_scatter.md` §2 give three reasons:
- R3 lets R2's canonical-owner cascade and the E2N owner choice match the home rank.
- The DG-path routing in `buildNodalSM` handles the R2/R3 boundary.
- In the past, the remesh sandwich read R2/R3 ghost CG slots.

Unzip reads only R1. Per-step minimal scatter already ignores R2/R3.

## Recommendation

1. Ghost-layer construction is the bigger and np-independent lever, worth about 1.8x. Add an opt-in
   knob, e.g. `DENDRO_REPART_GHOST_RINGS=2`, that stops the claim/push loop after d=2 and skips the
   legacy R3. Run it through the full gate set:
   - testPartitioning TEST 10/11
   - EM4 + NLSM long-haul bit-identity
   - the ragged-labelling arm
   - `DENDRO_SM_COVERAGE_CHECK`
   - BSSN BBH

   A cheaper variant that may be safer: keep R3 for topology only (in E2E, no nodes/E2N expansion)
   and check whether the rebuild phases then scale with R1+R2.
   Confidence: high that the artifact is real and ~40 % of labelled ghosts; medium that removing it
   is value-neutral.
2. Labelling compactness is second and grows with np. The per-level pieces are disjoint. Refine does
   not merge them, and its 26-stencil gain only trims ghost nodes (−7 % vs lw at np 24). A labelling
   that co-locates a rank's per-level pieces (the pieces are the problem, not the shells) is needed
   to recover the remaining 1.5–2x. Confidence: medium.

## Reproduce

- Probe, opt-in `DENDRO_GHOST_SHELL_STATS=1`, reverted from `src/`: patch at
  `~/research/repart_gate/work/ghostinv/ghost_shell_probe.patch`.
- Binaries: `bin/ghostinv{,2}`.
- Runs: `~/research/repart_gate/work/ghostinv/{run,arms*,tab}.sh`.
