# Partition cost on a block-structured octree: why face-graph partitioners lose in Dendro, and what a repaired partitioner must optimise

David Van Komen's dendrolib fork, 2026-09-23. Companion to `plan_2026-09-23_repartition_sweep.md`;
numbers from `pq_oct_2026-09-23.jsonl` (paper exports, `edgeCutCompare --oct`) and the lattice
check reproduced in §2.

## 1. What Dendro pays for

Per RK stage each rank exchanges the nodal data of its **ghost elements**: every element that shares a
face, an edge, or a vertex with a local element (the unzip padding and the CG/hanging-node closure
need all three). Two consequences fix the objective:

- **Comm volume of rank p** ∝ |∂₂₆ Ω_p|, the number of elements of Ω_p that have a 26-neighbour outside
  Ω_p (equivalently, the R1 ghost count of its neighbours). Not the face-cut, not the face-boundary.
- **Compute of rank p per LTS coarse step** = Σ_ℓ 2^{ℓ−ℓ_min} · W_{p,ℓ}, where W_{p,ℓ} is the work in
  level ℓ on rank p, and every partial step is gated by the busiest rank at that level.

So the partition problem Dendro actually poses is

    minimise   Σ_p |∂₂₆ Ω_p|
    subject to max_p W_{p,ℓ} ≤ (1+ε) · W_ℓ / P      for every level ℓ        (multi-constraint balance)
               each Ω_p is a union of whole Dendro blocks                     (block atomicity)

The AMRaCut paper solved a different problem: minimise Σ_p |∂₆ Ω_p| (face-adjacent boundary
elements) with a single count constraint at ≤2× imbalance.

## 2. Boundary density of a lattice surface

Take a partition boundary as a surface S with unit normal field n(x), discretised on the cubic
lattice of leaf octants (locally uniform level). Per unit area of S, counted on a digital plane and
verified numerically on a 96³ lattice (orientations (100),(110),(111),(210),(321), agreement ≤ 3 %):

| quantity per unit area                | value  |
|---------------------------------------|--------|
| cut faces (6-graph edge cut)          | ‖n‖₁   |
| face-adjacent boundary cells (∂₆)     | ‖n‖∞   |
| 26-neighbour boundary cells (∂₂₆)     | ‖n‖₁   |

Hence for any partition

    |∂₆ Ω|  = ∫_S ‖n‖∞ dA,        |∂₂₆ Ω| = ∫_S ‖n‖₁ dA,

and the ratio ∂₂₆/∂₆ on a facet ranges from **1** (axis-aligned, n = e_i) to **3** (n ∝ (1,1,1)).
Isotropically distributed orientations give ⟨‖n‖₁⟩ = 1.50 and ⟨‖n‖∞⟩ = 0.83, ratio 1.8.

**Interpretation.** A diagonal facet is a staircase. Each stair cell exposes several faces, so few
cells carry many cut faces: cheap on the ∂₆ count, exactly as expensive as its cut on ∂₂₆. Axis-aligned
facets have one cell per cut face on both counts.

### Application to SFC vs fastpart

- **Hilbert segments** are unions of octants: n ∈ {±e_i} everywhere, so ∂₆ = ∂₂₆ = cut = A_sfc up to
  the edge/corner correction of finite blocks, ≈ 2·V^{−1/3} ≈ +8 % at V ≈ 8·10³ elements/rank.
  Measured: 12,305 → 13,249 (+7.7 %) on the paper's mesh2 at np = 8.
- **fastpart cells** are hop-count Voronoi cells of the face graph, i.e. L1-metric Voronoi cells,
  whose bisectors are diagonal. Measured ∂₆ ratio 0.66 with ⟨‖n‖∞⟩ ≈ 0.83 gives A_fp/A_sfc ≈ 0.79
  (Voronoi cells are geometrically a little more compact than curve segments); the model then predicts
  ∂₂₆ ratio ≈ 0.79 × 1.5 = 1.19 against measured 0.98 (np=8) – 1.13 (np ≥ 16). The residual is the
  non-isotropic facet mix of an L1 Voronoi diagram; the sign and size are right.

This is the whole explanation of the paper/fork contradiction: the paper's metric ∫‖n‖∞ rewards
diagonal boundaries by up to 3× relative to ∫‖n‖₁; Dendro is billed ∫‖n‖₁.

**Corollary (design rule).** Any partitioner for Dendro must score itself on the 26-neighbour graph
or, equivalently, on the L1 surface measure. Octant-aligned boundaries are optimal per unit area; the
only way to beat a Hilbert segment is by a smaller *area*, never by orientation.

## 3. Per-sub-step efficiency and why contiguity fails

For ENUTS with level time factors 2^{ℓ−ℓ_min},

    eff = Σ_ℓ 2^ℓ · (W_ℓ / P) / Σ_ℓ 2^ℓ · max_p W_{p,ℓ}.

Total-work balance (`sfc_weighted`) fixes Σ_ℓ 2^ℓ W_{p,ℓ} per rank but not each term. The finest level
on a puncture mesh occupies a short stretch of the curve, so any contiguous split gives max_p W_{p,ℓ_max}
≈ W_{ℓ_max} and the finest (most frequent) term dominates: measured eff 0.51–0.77 for every contiguous
labelling, including the optimal contiguous multi-constraint split (`sfc_mc_contig`, 0.30–0.52, because
forcing the fine level to spread also forces the coarse levels into starved ranks). The per-level split
(`sfc_levelwise`) makes each term equal by construction, eff → 1, at the price of k = n_lev disjoint
pieces per rank.

## 4. The price of k pieces and what refinement recovers

Isoperimetry on the lattice with the L1 surface measure: a connected region of V cells has
|∂₂₆| ≥ 6·V^{2/3} (cube), attained by axis-aligned cubes. A rank holding disjoint pieces with fractions
f_ℓ of its volume pays at least

    Σ_ℓ 6 (f_ℓ V)^{2/3} = 6 V^{2/3} · Σ_ℓ f_ℓ^{2/3} ≥ 6 V^{2/3},

with equality only for one piece; equal fractions give n_lev^{1/3} (1.44 for three levels). Measured
unrefined levelwise: 1.33–1.67× the SFC boundary.

The bound assumes the pieces are disjoint. If a rank's level-ℓ piece is *adjacent* to its own
level-(ℓ+1) piece, they merge into one region and the shared interface is not cut. Therefore the
optimal multi-constraint partition is a set of P compact regions each holding 1/P of every level,
which exists whenever the refinement bands are thick compared with the P^{−1/3} rank scale, and its
boundary is again ≈ A_sfc. The job of a refiner started from the levelwise labelling is to slide each
rank's per-level pieces until they touch.

Measured on mesh2, np = 8, after gain refinement on the 26-graph with per-level caps:

| cap (1+ε) | ∂₂₆ / SFC | eff   |
|-----------|-----------|-------|
| unrefined | 1.67      | 1.000 |
| 1.05      | 1.21      | 0.982 |
| 1.10      | 1.12      | 0.975 |

i.e. two thirds of the k-piece penalty is recovered at ε = 5 %, three quarters at 10 %, with eff
within 2.5 % of perfect. Started from the plain SFC split with a single constraint the same refiner
gives ∂₂₆ = 0.93× SFC at 3 % imbalance — the first fastpart configuration that beats the curve on the
quantity Dendro pays for.

## 5. The refinement as a constrained discrete gradient flow

Labels x: V → {0..P−1}. Objective F(x) = Σ_{(u,v)∈E} w_uv [x_u ≠ x_v] on the graph E that is given
(the 26-graph). For a group g (a Dendro block or a sibling octet) with current label a and candidate
b, the move gain is

    Δ_g(a→b) = w(g, b) − w(g, a),   w(g, q) = Σ_{u∈g, v∉g, x_v = q} w_uv,

and F decreases by exactly Δ_g when g moves alone. Each round every group proposes its best positive
gain; proposals are accepted in gain order against a per-part, per-constraint slack that all ranks
agree on through one allreduce (each part's free capacity is shared pro rata among the ranks asking
for it), so balance is never violated at the end of a round. Groups move as units, so block atomicity
is preserved without a post-hoc vote, and moving an octant-aligned group cannot create a diagonal facet
that the Hilbert start did not have. Simultaneous (Jacobi) moves of two adjacent groups can interfere;
a parity gate halves that and empirically the process converges in ≈ 30 rounds to a local minimum
(moves → 0). It is a Fiduccia–Mattheyses-style local search without the negative-gain escape, so it
recovers area, not topology: it cannot merge a piece with a far-away piece. That is the remaining
gap between 1.12 and 1.0 and the argument for a coarse (block-level) pass first.

Diffusion, by contrast, is inert from a balanced start: with all part sizes under the threshold every
rate is equal, the flux is symmetric, no value crosses ½, and the labels do not change (measured:
`fastpart_sfcinit` ≡ SFC). Its relabel rule never evaluates a cut; it can only trade balance for
boundary roughness, which is why 5 or 20 rounds give the same labels.

## 6. Block atomicity and the hybrid code

A Dendro block is a regular grid of 8^k same-level octants. Moving blocks instead of elements
(a) keeps boundaries octant-aligned, (b) bounds the imbalance added by atomicity to B/(N/P) per part,
(c) shrinks the graph by the block size, and (d) makes the per-level constraint trivial to account
(a block has one level). For the MPI+OpenMP branch the partition only has to be right at rank level;
OpenMP dynamic scheduling absorbs intra-rank imbalance but not the inter-rank per-level gating of §3,
so the objective and constraints above are unchanged there; only the number of parts falls by the
thread count, which lowers Σ_p |∂₂₆ Ω_p| by roughly (P/T)^{1/3}/P^{1/3}.

## 7. What is claimed, and what is not

Claimed and measured: (i) the face metric under-counts Dendro's ghost set by ∫‖n‖₁/∫‖n‖∞ ∈ [1,3];
(ii) hop-count Voronoi boundaries are the worst case for it; (iii) a gain-aware, capacity-constrained,
group-atomic refinement on the 26-graph started from the curve improves the Dendro objective in both
regimes (single-constraint: 0.93×; per-level: 1.67× → 1.12×). Not yet claimed: wall-clock (the box
cannot time), behaviour above 64 parts, and the coarse-pass extension. The fork's `repartitionMeshGlobal`
consumes any label vector, so the integration cost of the refined labelling is the existing
arbitrary-labelling pipeline plus passing block ids and levels into `fastpart_oct_partopts`.
