/**
 * @file adm2bssn.h
 * @brief ADM to BSSN variable conversion.
 *
 * Converts physical ADM variables (metric gd, extrinsic curvature Kd,
 * lapse alpha, shift beta) into BSSN conformal variables
 * (chi, gt, At, trK, Gt).
 *
 * This conversion is the same for BSSN, EMDA, CCZ4, and any other
 * formulation that uses the standard BSSN conformal decomposition.
 * Theory-specific fields (dilaton, EM, Z4 constraints) are left
 * untouched and should be set separately by the solver.
 *
 * Usage: #include this file inside the TwoPunctures data extraction
 * loop where gd[3][3] and Kd[3][3] are available.
 */

#ifndef DENDRO_GR_ADM2BSSN_H
#define DENDRO_GR_ADM2BSSN_H

// this is an include fragment, not a standalone header.
// expects these variables to be in scope:
//   double gd[3][3]  -- physical 3-metric
//   double Kd[3][3]  -- extrinsic curvature
//
// produces:
//   double chi        -- conformal factor
//   double gtd[3][3]  -- conformal metric (tilde)
//   double Atd[3][3]  -- traceless conformal extrinsic curvature (tilde)
//   double trK        -- trace of extrinsic curvature
//   double gu[3][3]   -- inverse physical metric

// compute determinant and inverse of physical metric
double t1 = gd[0][0];
double t2 = gd[1][1];
double t4 = gd[2][2];
double t6 = gd[1][2];
double t7 = t6 * t6;
double t9 = gd[0][1];
double t10 = t9 * t9;
double t12 = gd[0][2];
double t16 = t12 * t12;
double detgd = t1 * t2 * t4 - t1 * t7 - t10 * t4 + 2.0 * t9 * t12 * t6 - t16 * t2;
double idetgd = 1.0 / detgd;

double gu[3][3];
gu[0][0] = idetgd * (gd[1][1] * gd[2][2] - gd[1][2] * gd[1][2]);
gu[0][1] = idetgd * (-gd[0][1] * gd[2][2] + gd[0][2] * gd[1][2]);
gu[0][2] = idetgd * (gd[0][1] * gd[1][2] - gd[0][2] * gd[1][1]);
gu[1][0] = gu[0][1];
gu[1][1] = idetgd * (gd[0][0] * gd[2][2] - gd[0][2] * gd[0][2]);
gu[1][2] = idetgd * (-gd[0][0] * gd[1][2] + gd[0][1] * gd[0][2]);
gu[2][0] = gu[0][2];
gu[2][1] = gu[1][2];
gu[2][2] = idetgd * (gd[0][0] * gd[1][1] - gd[0][1] * gd[0][1]);

// conformal factor: chi = det(gd)^(-1/3)
double chi = pow(idetgd, 1.0 / 3.0);

// conformal metric: gt_ij = chi * g_ij
double gtd[3][3];
for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
        gtd[i][j] = chi * gd[i][j];

// trace of extrinsic curvature
double trK = 0.0;
for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
        trK += gu[i][j] * Kd[i][j];

// traceless conformal extrinsic curvature: At_ij = chi * (K_ij - (1/3) g_ij K)
double Atd[3][3];
for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
        Atd[i][j] = chi * (Kd[i][j] - (1.0 / 3.0) * gd[i][j] * trK);

#endif  // DENDRO_GR_ADM2BSSN_H
