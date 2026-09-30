/**
 * @file bssn2adm.h
 * @brief BSSN to ADM (physical) variable conversion.
 *
 * Converts BSSN conformal variables (chi, gtd, trK, Atd) into the
 * physical ADM variables (3-metric gd, extrinsic curvature Kd). This
 * is the inverse of the transformation in adm2bssn.h, and is needed
 * wherever code downstream of the BSSN evolution needs physical ADM
 * fields rather than the evolved conformal ones -- e.g. a worldtube
 * writer for Cauchy-Characteristic Extraction (CCE), which needs the
 * physical spatial metric and extrinsic curvature on a fixed-radius
 * sphere to hand off to an external characteristic-evolution code.
 *
 * This conversion is the same for BSSN, EMDA, CCZ4, and any other
 * formulation that uses the standard BSSN conformal decomposition.
 *
 * Usage: #include this file wherever chi, gtd[3][3], Atd[3][3], and
 * trK are already in scope (e.g. after unpacking U_CHI, U_SYMGT*,
 * U_SYMAT*, U_K from the evolved variable array). Mirrors the
 * include-fragment style of adm2bssn.h in this same directory.
 */

#ifndef DENDRO_GR_BSSN2ADM_H
#define DENDRO_GR_BSSN2ADM_H

// this is an include fragment, not a standalone header.
// expects these variables to be in scope:
//   double chi        -- conformal factor
//   double gtd[3][3]  -- conformal metric (tilde)
//   double Atd[3][3]  -- traceless conformal extrinsic curvature (tilde)
//   double trK        -- trace of extrinsic curvature
//
// produces:
//   double gd[3][3]   -- physical 3-metric
//   double Kd[3][3]   -- physical extrinsic curvature

// physical 3-metric: g_ij = gt_ij / chi
double gd[3][3];
for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
        gd[i][j] = gtd[i][j] / chi;

// physical extrinsic curvature: K_ij = (At_ij + (1/3) gt_ij trK) / chi
double Kd[3][3];
for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
        Kd[i][j] = (Atd[i][j] + (1.0 / 3.0) * gtd[i][j] * trK) / chi;

#endif  // DENDRO_GR_BSSN2ADM_H
