# Fastpart LTS blow-up at np=16 / np=192: investigation (2026-09-28)

## Verdict
**Root cause:** the Stage-3 ZNP mirror map (`m_uiZipNonPrimaryToGhostCg`) was built from raw send-scatter codes
(`src/mesh.cpp`, `Mesh::buildZipPlan`, the inverse-scatter Alltoallv at ~9542, before the fix; the lookups are
at ~9642/9711). On labelled meshes some send entries are tagged DG (`m_uiScatterMapSendIsDG[k]=1`), and then
`m_uiScatterMapActualNodeSend[k]` holds `(localEle-localBegin)*NpE + sub`, not a cg index. The inverse map
treated that code as the sender's cg. When a DG code happened to equal the primary's cg index, the
non-primary local cg was paired with an unrelated ghost cg (samples: 50 to 190 index units away in space).
`readFromGhostEnd` (and the ctx and `performGhostExchange` variants, `include/mesh.tcc` ~1829/2238/786) then
copies that ghost into the local slot **after** `syncZipNonPrimary` has put the right value there. So every
ghost exchange corrupted a few local cgs, and the unzip padding of the next RHS read them.

- The bug is in **global RK too**, not only ENUTS. GTS (`nlsmNUTS par 1`) on the np=16 fp labelling gave
  U_PHI max 18.7 after one step (SFC 0.34).
- A collision needs a DG code that numerically equals some primary cg. Whether that happens depends on the
  index layout, which explains why it is sporadic: np=48/96 were clean, np=192 was not, and plain levelwise
  (method 4) happened to be collision-free. Fastpart labellings have many R2-boundary DG entries, so they
  collide more often.
- testPartitioning np=4 already showed it; the baseline recorded it as a non-FAIL red line. TEST 8bw (unzip
  SFC vs graph) had bufMism=6043 with maxDiff 1.25 (m3) and 1026 (ragged). TEST 8bi (async cycle) drifted
  by 0.46.

## Evidence chain
1. BLKHASH, sfc vs fp: the first coarse step's `init-unzip` has 122 blocks whose **padding** differs, by up to
   0.136 in max |pad|, while the interiors agree to ULP. Every later stage diverges from there.
2. GTS on the same labelling blows up after the first step. So it is not ENUTS.
3. `DENDRO_IC_CHECK=1` checks the state against the analytic IC at t=0. `after-init_grid` has 0 mismatches.
   The first check on the labelled mesh ("labelled", after the init `readFromGhost`) has 1–4 bad **local**
   cgs on 6 ranks (e.g. r13 cg at (976,1104,1088): chi 0.95 vs 1.244).
4. New probe `DENDRO_GHOST_POS_CHECK=1` puts a position code in every slot, then runs the sync and the ghost
   exchange. The `sync` phase is clean. The `exchange` phase has exactly those local cgs bad, every one of
   them a ZNP-mirror destination, and every one holding a far-away ghost's position.
5. The fix makes the probe and the IC check come back clean.

## Fix (jj `vvvslrvl`, default on, `DENDRO_ZIPPLAN_INV_DG=0` reverts for A/B)
The sender decodes DG-tagged entries to the cg it actually ships, `E2N_CG[(idx/NpE+eLB)*NpE+idx%NpE]`, the
same decode the send pack uses. SFC meshes have no DG tags, so they are byte-identical. Probe: jj `uvowkpky`.

## Gates (bins `~/research/repart_gate/bin/fix1`)
- **lw3.sh np=16, step-2 U_PHI (min, max).** sfc −2.36385011495455943, 2.50214029822932194. lw and lvl are
  18-digit identical. fp matches to 15 digits (max …931483); it was ±13.5 before the fix. Log:
  `work/fix1_gates.txt`.
- **GTS fp np=16, 7 steps:** min/max identical to SFC.
- **nuts2.sh (ragged d9 ENUTS):** 18 digits vs ref.
- **gate.sh NP=4:** 0 FAIL. m3ref1 and m4 are identical to base. m3 and ragged differ from base only by
  improvements: 8bw mismatch goes to 0 (maxDiff 6.7e-16), and 8bi goes from drift to PASSED maxDrift=0. The
  TEST 2/3 counts drop by a few slots, because minimal-scatter keeps mirror sources and the corrected mirror
  has different ones. The 8bk asymmetry count moves as well (maxDiff unchanged). `ref/tp_base` should be
  refreshed.
- **EM4 np4/np8:** the SFC md5 is unchanged (d7bbee89). The graph md5s move (np4 a75a069e→2603beb3, np8
  32eb202e→32d48b0b). The per-field MIN/MAX are unchanged; the L2/RMSE/MAE aggregates shift at ≤1e-5
  relative, with no systematic direction vs SFC. `DENDRO_ZIPPLAN_INV_DG=0` reproduces a75a069e exactly. The
  EM4 graph refs need a refresh.

## Remaining risk
- The np=192 Kingspeak lw runs, and every labelled run so far (BSSN, EM4 graph refs), may have carried this
  bug in some cgs. The Kingspeak np=192 ratios need a re-run.
- Other consumers of `m_uiScatterMapActualNodeSend` that ignore the DG tag were not audited, for example
  `generateBdyElementDiagonalSearchKeys` at `src/mesh.cpp` ~3784. It is likely SFC-only.
- `m_uiPassDDemotedToGhostCg` (the legacy PassD mirror) is a separate heuristic and was not implicated here.
  `DENDRO_DISABLE_PASSD_MIRROR=1` was a no-op.
