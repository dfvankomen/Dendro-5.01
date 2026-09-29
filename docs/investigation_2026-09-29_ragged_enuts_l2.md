# Ragged-labelling ENUTS L2 residual (NLSM d11, np=16): investigation (2026-09-29)

## Verdict
There were two separate causes, and they were stacked on each other.

1. **Most of the gap was in the diagnostic, not the solution.** It was ~99.99 % of the printed L2 difference. The NLSM `(min, max, l2)` line (and
   `[ANALYTIC-ERR]`) summed over the whole local node range. On labelled meshes that range also holds
   non-primary duplicate cg slots (cross-rank sync receivers and intra-rank dups), so they were counted twice. The
   evidence is already in the **t=0** line: U_CHI l2 was sfc 2.38333570647733450e3 and fp 2.38337139363700635e3, and the
   node counts were sfc/lvl 11821285, lw 11821890 and fp 11823134 (fp has 1849 duplicate slots).
   Fix (jj `oqkypopv`): `Mesh::getLocalNonPrimaryNodes()` records the local cgs that are recv targets or dup
   destinations in `buildZipPlan` (`src/mesh.cpp`), before `pruneZipSync` compacts the lists.
   `normL2Unique(mesh, v, &nUnique)` (`include/mathMeshUtils.h`) skips them. It uses the same arithmetic as
   `normL2(ptr, n, comm)`, so SFC output is byte-identical. The NLSM app now calls it in `nlsmCtx.cpp` terminal_output and
   dump_analytical_error. Those app edits are in `~/research/nlsm-gr_lts` and are uncommitted, because that repo is
   no-commit.
   With it, fp at t=0 gives 2.38333570647734314e3, which agrees with SFC to 14 digits, and the unique node counts are equal.

2. **A real ENUTS bug remained, and the L2 hid it.** Position-keyed dumps (`DENDRO_EVAR_DUMP=<pfx>
   DENDRO_EVAR_DUMP_POS=1`, a new app probe in `nlsm_nuts.cpp`) compare every grid point with SFC:
   - GTS (`nlsmNUTS par 1`) fp vs sfc: 0 points differ above 1e-13. **Global RK is exact.**
   - lvl vs sfc under LTS: 0 points differ.
   - fp vs sfc under LTS after **one coarse step**: 426 primary slots and 251 dup slots differ, at 344 positions, up to
     0.036 (U_PHI). **Every one of those positions is a duplicated position.** After 2 steps the error has spread to
     2.19 M points (max 0.058 on primaries). The L2 barely moves (1e-8 rel), because the differences have mixed signs and cancel in Σ(a−b)(a+b).

   **Root cause:** `ODE/include/enuts.h`, `ExplicitNUTS::build_cg_owner_map` (per-slot loop, ~line 945). At the
   end of each coarse step on a labelled mesh, each local cg is gathered from its "SFC-canonical owner's" DG copy.
   That owner is the coarsest non-hanging sharer, with ties broken by TreeNode order. The owner was chosen **among the elements whose E2N points to that
   cg slot**. On a labelled mesh the sharers of one position are split across the primary slot, its duplicate slots
   and ghost slots. So a slot's pick can be a finer sharer than the true SFC owner. After sub-cycling, the
   copies held by different sharers differ at truncation level, so the primary takes a wrong value, and the dup sync spreads it.
   Under GTS all copies agree, which is why only LTS shows it.

## Fix (jj `kqnzkynv`, default on, `DENDRO_ENUTS_OWNER_BYPOS=0` reverts)
`build_cg_owner_map` now keys the owner by **grid position**. It visits every local element and every ghost element
that the element DG exchange fills (`getRecvElementSM`), skips hanging nodes, and picks the coarsest element, with ties
broken by TreeNode order. Every local slot at that position then reads the same owner copy. The function only runs when the
mesh has non-SFC blocks, so SFC runs are untouched.
Result: fp vs sfc after one coarse step has **0 differing points (all slots, dups included, 1e-13)**.

## Gates (bins `~/research/repart_gate/bin/own2`, log `work/chain_own.log`)
All gates were run serially with the fix on:
- **gate.sh NP=4:** all four arms (m3, m3ref1, m4, ragged) are IDENTICAL to `ref/tp_base`, 0 FAIL.
- **em4.sh:** np4/np8, sfc d7bbee89 and graph 2603beb3 / 32d48b0b, all equal to the refs. EM4 is GTS, so it is unaffected.
- **bssn.sh:** graph and sfc md5s equal the refs.
- **nuts2.sh (ragged d9 ENUTS, np4):** min/max are 18-digit identical to the ref. For U_CHI l2, the fix gives 6.54318343478060683e2, SFC
  (method 0, same par) gives 6.54318343478058978e2, and `DENDRO_ENUTS_OWNER_BYPOS=0` gives 6.54318363669755854e2. So the
  old trajectory was 3e-8 off SFC, and the fix brings it to 14 digits.
- **lts11.sh NPS="4 8":** byte-identical to the kdef results (the lw arm there has no duplicate positions).
- **lw3.sh np16, last step (U_CHI l2 / U_PHI l2 / U_PHI max):**

  | arm | U_CHI l2 | U_PHI l2 | U_PHI max |
  |---|---|---|---|
  | sfc | 2.16461568286104557e3 | 3.05422796243521270e3 | 2.50214029822932194 |
  | lvl | …101874 | …518496 | identical |
  | lw | …102238 (was …63676339807807) | …517314 (was …3677819255226) | identical |
  | fp | …104739 (was …65610673942956) | …518769 (was …30450641200150) | identical (was …931483) |
  | lvlm | …102283 (was …62198606092443) | …518178 | identical |

  Every arm now agrees with SFC to 13–15 digits, which is summation order.

## Notes
- The EM4 app's aggregate L2/RMSE/MAE (`solverCtx.cpp` ~1210/2360/2475) has the same duplicate-slot overcount.
  That is the likely source of the "≤1e-5 relative, no systematic direction" graph-vs-SFC aggregate drift recorded
  after the ZNP DG fix. It was not changed here; switching it to `normL2Unique` would move the EM4 graph md5 refs.
- BSSN graph `bssn_profile_Constraints.dat` md5 (a83112e3) differs from SFC (2ffba783) while the other three files
  match; the constraint norms likely carry the same duplicate-slot overcount (not checked).
- L2 is a weak gate for labelled-vs-SFC agreement, because errors cancel. The position-keyed dump plus the `cmp.py` script
  (`~/research/repart_gate/work/gts_l2u/`) is the pointwise gate.
