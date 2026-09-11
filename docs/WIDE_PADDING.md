# DENDRO_WIDE_PADDING — one extra ghost ring per block face (prototype)

Branch `pad4-prototype` (off `origin/experimental`), 2026-09-10. Paired with
`em4-gr` branch `pad4-prototype` (copy at `~/research/em4_pad4`).

## Why

Compact (BYU A6) derivatives lose most of their interior advantage over E6 to
the three closure rows that sit in the block padding: the closure error leaks
through P⁻¹ into the first active rows. Each extra padding ring roughly doubles
the deployed advantage (vault: *Padding width and the ghost layer*, *Face-type
accuracy map*). Same-level neighbour elements already carry eleOrder+1 nodes
and coarser neighbours are prolongated to 2·eleOrder+1, so a 4th padding point
is **already on-rank** for every face except one that abuts a *finer* octant
(a finer element has only eleOrder/2 nodes at block spacing). No new
communication; the ghost element set is unchanged (verified: identical ghost
element / node counts in the profiler with the flag on and off).

## What the flag does

CMake option `DENDRO_WIDE_PADDING` (default OFF) → compile definition
`DENDRO_WIDE_PADDING` via `dendro_config`. Every code site is wrapped in
`#ifdef DENDRO_WIDE_PADDING … #else … #endif`; `grep -rn DENDRO_WIDE_PADDING`
lists all of them. Central macros live in `include/dendro_padding.h`:

| macro | meaning |
|---|---|
| `DENDRO_WIDE_PADDING_EXTRA` (1) | extra rings per face |
| `DENDRO_PAD_WIDTH_FOR_ORDER(eo)` | `eo/2 + EXTRA` (ON) / `eo/2` (OFF) |
| `DENDRO_NATIVE_PAD_WIDTH(eo)` | `eo/2`, what a finer neighbour can fill; depth of the wavelet cube |
| `DENDRO_FINE_FACE_SHIFT`, `DENDRO_FINE_FACE_BIT(dir)`, `DENDRO_BFLAG_PHYS_MASK` | fine-face bits ride in `bflag` bits 6..11 for the derivative dispatch only |

### Sites (dendrolib)

| file | change |
|---|---|
| `src/block.cpp` | `m_uiPaddingWidth = DENDRO_PAD_WIDTH_FOR_ORDER(eleOrder)` (was `eleOrder>>1`); `m_uiFineFaceFlag` init |
| `include/block.h` | `m_uiFineFaceFlag`, `get/setBlkFineFaceFlag()` |
| `src/mesh.cpp` | `Mesh::computeBlkFineFaceFlags()`: after `buildE2BlockMap()` (both mesh ctors), walks `blkUnzipElementIDs` and sets a face bit when ANY face/edge/vertex padding element beyond that face plane is finer than the block (conservative: an edge-only finer neighbour flags both faces it touches, which is what mixed derivatives need). Env hook `DENDRO_WIDE_PADDING_FORCE_FINE=1` flags every face (equivalence test, see below) |
| `include/mesh.h` | declarations of `computeBlkFineFaceFlags`, `fillFineFaceRing` |
| `include/mesh.tcc` | `fillFineFaceRing()`: after `unzip_scatter` (serial and OMP paths) copies the nearest filled plane into the unfilled outer ring on flagged faces (x, then y, then z) so the buffer is finite (0·NaN = NaN in the GEMM); the trimmed operator never reads it. `getUnzipElementalNodalValues(isPadded=true)`: the wavelet cube stays `2·eOrder+1` and starts `paddWidth − eOrder/2` in from the block edge. `isReMeshUnzip` guard relaxed. Legacy per-direction `Mesh::unzip(blkIDs,…)` aborts under the flag (not made wide-padding aware; solvers use `unzip(in,out,dof)` → `unzip_scatter`) |
| `include/waveletAMR.tcc` | the two `pw == eOrder/2` guards |
| `include/derivatives.h`, `include/filters.h` | `p_pw = DENDRO_PAD_WIDTH_FOR_ORDER(eo)`, `p_n = eo+1+2·p_pw` |
| `src/derivatives/derivs_matrixonly.cpp`, `impl_hybrid_approaches.h`, `impl_boris.h` | eager block sizes `n = i·eo + 1 + 2·pw`; **fine-face operator variants** built by `build_fine_face_variants()` in all three matrix builders (plain, in-matrix filter, per-side diagonals) |
| `include/derivatives/derivs_utils.h` | `DerivMatrixStorage` gains `D_fine_left/right/leftright`, `D_fine_left_bdy_right`, `D_bdy_left_fine_right`, `fine_valid`; `get_deriv_mat_fine_face()` consulted first by `get_deriv_mat_by_bflag_{x,y,z}` |

### The fine-face variant

A block face abutting a finer octant has its outermost `EXTRA` padding rows
unfilled. The operator for that axis is built with `boundary_top/bottom =
EXTRA` at that end — the same `createMatrix(...)` trim used for physical
boundaries (`= pw` there): identity rows in the trimmed band, the Wu–Kim
closure starts one row in, i.e. it is exactly the stock (pad eo/2) operator on
that face. A physical boundary on the same face wins (there is no neighbour).
Schemes that build `DerivMatrixStorage` some other way (CCFD, Padé filter
matrices) leave `fine_valid=false` and the dispatch throws if a fine face is
ever requested — loud, not silently wrong.

Unchanged and untouched: explicit stencils and KO (they read at most eo/2
rings), ghost exchange, zip, `scatter_*` fast kernels (they clip to the
allocation), block/element maps.

### Solver side (em4-gr `pad4-prototype`)

`CMakeLists.txt` option + define; `parameters.cpp`
`SOLVER_PADDING_WIDTH = DENDRO_PAD_WIDTH_FOR_ORDER(...)` (3 sites);
`rhs.cpp` ORs `getBlkFineFaceFlag() << DENDRO_FINE_FACE_SHIFT` into the flag
passed to `solverrhs_compact_derivs`, which splits it into `dflag` (derivative
and in-matrix-filter calls) and `bflag = & DENDRO_BFLAG_PHYS_MASK` (boundary
conditions, KO); `dataUtils.cpp` assert. Unconditionally added: a
`[census]` line in `SOLVERCtx::interface_norms()` counting block faces by
type S/C/F/B per level (the "face census" from the roadmap).

## Tests (k0 / k1 = the frozen-AMR convergence meshes of
`scripts/run_em4_convergence.sh`, dipole ID, KO off, 4 ranks)

1. **Explicit control**: E6 on the stock and wide builds → identical to every
   printed digit (explicit never reads the extra ring).
2. **Equivalence**: `DENDRO_WIDE_PADDING_FORCE_FINE=1` on the wide build (all
   faces trimmed) reproduces the stock A6 numbers to every printed digit.
3. **A/B, A6 `BYU_A6_1ST_R060_OP5`, k0 (lmax 6, 1653 elements, t = 0.5)**,
   `[ifc] |dE|` rms by distance-to-coarser-face bin:

   | bin | stock | wide | ratio |
   |---|---|---|---|
   | d1 (next to a C face) | 3.38e-5 | 3.39e-5 | 1.00 |
   | d2 | 2.80e-5 | 2.10e-5 | 1.33 |
   | d3 | 8.10e-6 | 9.95e-6 | 0.81 |
   | d≥4 | 4.11e-5 | 2.79e-5 | 1.47 |
   | none (no C face on block) | 6.17e-5 | 4.29e-5 | 1.44 |

   Face census on that mesh: S 1460, C 434, F 284, B 96 (S+F = 77 %);
   197 of 379 blocks are single-element.
   Cost: `DOG(unzip)` 1.68M → 2.35M (×1.40); ETS wall 7.50 → 8.37 s (+12 %);
   ghost elements/nodes identical.

4. **A/B, A6, k1 (lmax 7, 13224 elements, all blocks 2-element, t = 0.5)**:

   | bin | \|dE\| stock | wide | ratio | divE stock | wide | ratio | divB ratio |
   |---|---|---|---|---|---|---|---|
   | d1 | 3.14e-7 | 3.17e-7 | 0.99 | 5.34e-6 | 3.59e-6 | 1.49 | 1.23 |
   | d2 | 4.24e-7 | 2.25e-7 | 1.89 | 1.80e-6 | 8.11e-7 | 2.22 | 1.67 |
   | d3 | 9.64e-8 | 9.65e-8 | 1.00 | 3.32e-7 | 1.99e-7 | 1.67 | 2.32 |
   | d≥4 | 6.74e-7 | 3.40e-7 | 1.98 | 1.24e-6 | 5.57e-7 | 2.22 | 3.94 |
   | none | 1.34e-6 | 6.50e-7 | 2.06 | 2.28e-6 | 1.68e-6 | 1.36 | 1.19 |

   Max \|dE\| in the "none" bin 5.4e-5 → 2.4e-5. Cost: `DOG(unzip)` 6.77M →
   8.36M (×1.23), ETS wall 114 → 137 s (+20 %, 4 ranks, no OMP).
   (First k1 pass showed the constraints getting *worse*: the constraint
   routine was still handed the plain bflag, so on fine faces it used the
   untrimmed operator over the copy-filled ring. Fixed in `solverCtx.cpp`
   compute_constraints — every compact-derivative consumer must receive the
   fine bits.)
5. **Dynamic AMR (REFINEMENT_MODE 0, remesh every step, k0 size)**: E6 in
   both builds → element counts identical at every step (wavelet cube
   extraction unchanged) and errors identical to 1e-17. With A6 the counts
   differ from step 2 on because the *solution* differs (expected).

## Not yet done / caveats

- `unzipDG_scatter` and the OMP unzip path are not exercised by EM4 builds
  here (OMP path has the ring fill but is compile-checked only).
- Dynamic AMR: verified equal remesh decisions with E6 (test 5); an
  `em4_simplified`-style long run with KO on and A6 has not been done yet.
- Mixed second derivatives (BSSN) are what the conservative edge/vertex
  flagging is for; EM4 only takes first derivatives.
- The extra-ring fill on fine faces is a plain copy of the neighbouring
  plane. A 6th-order extrapolation there would allow the *untrimmed* operator
  on F faces too (approximate); untested and not the default.
- Timing above is 4 ranks on a laptop; the interesting number is the
  production-mesh census (fraction of faces S/F) which this branch now prints.

## Performance caveat found 2026-09-10 late (not pad-4 specific)

All timings above use the convergence-sweep build config: `DENDRO_UNZIP_SPEEDUP`,
`DENDRO_UNZIP_SCATTER_FAST`, `DENDRO_TENSOR_SIMD`, `DENDRO_UNZIP_OMP` all OFF, `-march=native`,
and "E6" = the loop kernel (`E6Simd` is the vectorised one). perf on the k0 run: unzip_scatter +
parent-to-child interpolation + round = ~50 % of the A6 run; the libxsmm JIT kernels (the compact
derivative itself) are 3.3 %.

**The compact engine path costs 2× E6 here and it is NOT the GEMM.** `E6Matrix` (E6 through the
same dense-matrix engine) is exactly as slow as A6 (5.54 vs 5.67 s vs 2.95 s for the E6 loop; np=1:
15.9 vs 7.7 s). perf stat: same instructions (38.9G vs 41.3G), same branches, branch misses, LLC
misses, page faults, FP assists, machine clears, but 1.83× the cycles — IPC halves across *all*
code (unzip, interpolation, KO, axpy), not in the derivative calls. Ruled out: OpenMP
oversubscription (OMP_NUM_THREADS=1), denormals (FTZ/DAZ preload; assists.fp equal), libxsmm ISA
(LIBXSMM_TARGET=avx2 same), heap layout (MALLOC_MMAP_THRESHOLD_), core binding, hyperthread
sharing (one busy thread). Only elevated counter: `ld_blocks_partial.address_alias` 0.81G → 1.57G
(4K aliasing), which explains at most a third of the gap. The vault's kernel microbenchmark
(compact ≈ E6 loop per call at n = 13) cannot see this because it is a whole-process effect.
Open item: bisect inside `MatrixCompactDerivs::apply_*` (plan/scaled-operator memo, JIT cache,
workspace) with the ctx timers, on this laptop and on a cluster node.
