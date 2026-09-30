/**
 * @file bssn2adm_derivs.h
 * @brief Cartesian derivatives of the BSSN-to-ADM (physical) metric
 *        conversion (the g_ij = gt_ij/chi piece only -- see note below on
 *        why d_k K_ij is NOT included here).
 *
 * bssn2adm.h converts the BSSN conformal variables (chi, gtd, trK, Atd) into
 * the physical ADM variables (gd, Kd) POINTWISE. It is NOT valid to apply
 * bssn2adm.h a second time to the Cartesian derivatives of chi/gtd to get
 * the Cartesian derivatives of gd -- the conversion is nonlinear (a
 * quotient by chi), so the derivative must be built via the quotient rule.
 * This file does that, for g_ij only:
 *
 *   g_ij = gt_ij / chi
 *   d_k g_ij = (d_k gt_ij)/chi - gt_ij (d_k chi)/chi^2
 *
 * Note on K_ij: the analogous derivative, d_k K_ij (which would also need
 * d_k At_ij and d_k trK as inputs, via
 *   d_k K_ij = [d_k At_ij + (1/3)((d_k gt_ij) trK + gt_ij (d_k trK))]/chi
 *              - K_ij (d_k chi)/chi
 * ) is intentionally NOT computed here. SpECTRE's PreprocessCceWorldtube
 * AdmMetricNodal format (the consumer this was written for -- see
 * Dendro_CCE_v2.0.md Section 6.1) needs K_ij itself but does not need its
 * Cartesian derivatives, so computing d_k K_ij would require differentiating
 * At_ij/trK for no purpose. If a future consumer needs d_k K_ij, extend this
 * file following the pattern above (it needs dAtd[3][3][3] and dtrK[3] as
 * additional inputs).
 *
 * Usage: #include bssn2adm.h FIRST (this file uses its output gtd/chi,
 * which must already be in scope), then #include this file wherever the
 * Cartesian derivatives dchi[3], dgtd[3][3][3] are also in scope (index
 * 0 = d/dx, 1 = d/dy, 2 = d/dz).
 */

#ifndef DENDRO_GR_BSSN2ADM_DERIVS_H
#define DENDRO_GR_BSSN2ADM_DERIVS_H

// this is an include fragment, not a standalone header. Include bssn2adm.h
// before this file. In addition to bssn2adm.h's expected/produced variables,
// expects these to be in scope:
//   double dchi[3]        -- Cartesian derivatives of the conformal factor
//   double dgtd[3][3][3]  -- dgtd[k][i][j] = d_k (conformal metric)_ij
//
// produces:
//   double dgd[3][3][3]   -- dgd[k][i][j] = d_k (physical 3-metric)_ij

double dgd[3][3][3];
for (int k = 0; k < 3; k++) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            dgd[k][i][j] = dgtd[k][i][j] / chi -
                           gtd[i][j] * dchi[k] / (chi * chi);
        }
    }
}

#endif  // DENDRO_GR_BSSN2ADM_DERIVS_H
