/* TwoPunctures:  File  "Equations.c"*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <quadmath.h>
#include "TPUtilities.h"
#include "TwoPunctures.h"

/* U.d0[ivar]   = U[ivar];  (ivar = 0..nvar-1) */
/* U.d1[ivar]   = U[ivar]_x;  */
/* U.d2[ivar]   = U[ivar]_y;  */
/* U.d3[ivar]   = U[ivar]_z;  */
/* U.d11[ivar]  = U[ivar]_xx; */
/* U.d12[ivar]  = U[ivar]_xy; */
/* U.d13[ivar]  = U[ivar]_xz;*/
/* U.d22[ivar]  = U[ivar]_yy;*/
/* U.d23[ivar]  = U[ivar]_yz;*/
/* U.d33[ivar]  = U[ivar]_zz;*/

CCTK_REAL
TCP_BY_KKofxyz(CCTK_REAL x, CCTK_REAL y, CCTK_REAL z) {
    int i, j;
    CCTK_REAL r_plus, r2_plus, r3_plus, r_minus, r2_minus, r3_minus, np_Pp,
        nm_Pm, Aij, AijAij, n_plus[3], n_minus[3], np_Sp[3], nm_Sm[3];

    r2_plus = (x - TPID::par_b) * (x - TPID::par_b) + y * y + z * z;
    r2_minus = (x + TPID::par_b) * (x + TPID::par_b) + y * y + z * z;
    r_plus = sqrt(r2_plus);
    r_minus = sqrt(r2_minus);
    r3_plus = r_plus * r2_plus;
    r3_minus = r_minus * r2_minus;

    n_plus[0] = (x - TPID::par_b) / r_plus;
    n_minus[0] = (x + TPID::par_b) / r_minus;
    n_plus[1] = y / r_plus;
    n_minus[1] = y / r_minus;
    n_plus[2] = z / r_plus;
    n_minus[2] = z / r_minus;

    /* dot product: np_Pp = (n_+).(P_+); nm_Pm = (n_-).(P_-) */
    np_Pp = 0;
    nm_Pm = 0;
    for (i = 0; i < 3; i++) {
        np_Pp += n_plus[i] * TPID::par_P_plus[i];
        nm_Pm += n_minus[i] * TPID::par_P_minus[i];
    }
    /* cross product: np_Sp[i] = [(n_+) x (S_+)]_i; nm_Sm[i] = [(n_-) x
     * (S_-)]_i*/
    np_Sp[0] =
        n_plus[1] * TPID::par_S_plus[2] - n_plus[2] * TPID::par_S_plus[1];
    np_Sp[1] =
        n_plus[2] * TPID::par_S_plus[0] - n_plus[0] * TPID::par_S_plus[2];
    np_Sp[2] =
        n_plus[0] * TPID::par_S_plus[1] - n_plus[1] * TPID::par_S_plus[0];
    nm_Sm[0] =
        n_minus[1] * TPID::par_S_minus[2] - n_minus[2] * TPID::par_S_minus[1];
    nm_Sm[1] =
        n_minus[2] * TPID::par_S_minus[0] - n_minus[0] * TPID::par_S_minus[2];
    nm_Sm[2] =
        n_minus[0] * TPID::par_S_minus[1] - n_minus[1] * TPID::par_S_minus[0];
    AijAij = 0;
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) { /* Bowen-York-Curvature :*/
            Aij =
                +1.5 *
                    (TPID::par_P_plus[i] * n_plus[j] +
                     TPID::par_P_plus[j] * n_plus[i] +
                     np_Pp * n_plus[i] * n_plus[j]) /
                    r2_plus +
                1.5 *
                    (TPID::par_P_minus[i] * n_minus[j] +
                     TPID::par_P_minus[j] * n_minus[i] +
                     nm_Pm * n_minus[i] * n_minus[j]) /
                    r2_minus -
                3.0 * (np_Sp[i] * n_plus[j] + np_Sp[j] * n_plus[i]) / r3_plus -
                3.0 * (nm_Sm[i] * n_minus[j] + nm_Sm[j] * n_minus[i]) /
                    r3_minus;
            if (i == j) Aij -= +1.5 * (np_Pp / r2_plus + nm_Pm / r2_minus);
            AijAij += Aij * Aij;
        }
    }

    return AijAij;
}

void TCP_BY_Aijofxyz(CCTK_REAL x, CCTK_REAL y, CCTK_REAL z, CCTK_REAL Aij[3][3]) {
    int i, j;
    CCTK_REAL r_plus, r2_plus, r3_plus, r_minus, r2_minus, r3_minus, np_Pp,
        nm_Pm, n_plus[3], n_minus[3], np_Sp[3], nm_Sm[3];

    r2_plus = (x - TPID::par_b) * (x - TPID::par_b) + y * y + z * z;
    r2_minus = (x + TPID::par_b) * (x + TPID::par_b) + y * y + z * z;
    r2_plus = sqrt(pow(r2_plus, 2) + pow(TPID::TCP_epsilon, 4));
    r2_minus = sqrt(pow(r2_minus, 2) + pow(TPID::TCP_epsilon, 4));
    if (r2_plus < pow(TPID::TCP_Tiny, 2)) r2_plus = pow(TPID::TCP_Tiny, 2);
    if (r2_minus < pow(TPID::TCP_Tiny, 2)) r2_minus = pow(TPID::TCP_Tiny, 2);
    r_plus = sqrt(r2_plus);
    r_minus = sqrt(r2_minus);
    r3_plus = r_plus * r2_plus;
    r3_minus = r_minus * r2_minus;

    n_plus[0] = (x - TPID::par_b) / r_plus;
    n_minus[0] = (x + TPID::par_b) / r_minus;
    n_plus[1] = y / r_plus;
    n_minus[1] = y / r_minus;
    n_plus[2] = z / r_plus;
    n_minus[2] = z / r_minus;

    /* dot product: np_Pp = (n_+).(P_+); nm_Pm = (n_-).(P_-) */
    np_Pp = 0;
    nm_Pm = 0;
    for (i = 0; i < 3; i++) {
        np_Pp += n_plus[i] * TPID::par_P_plus[i];
        nm_Pm += n_minus[i] * TPID::par_P_minus[i];
    }
    /* cross product: np_Sp[i] = [(n_+) x (S_+)]_i; nm_Sm[i] = [(n_-) x
     * (S_-)]_i*/
    np_Sp[0] =
        n_plus[1] * TPID::par_S_plus[2] - n_plus[2] * TPID::par_S_plus[1];
    np_Sp[1] =
        n_plus[2] * TPID::par_S_plus[0] - n_plus[0] * TPID::par_S_plus[2];
    np_Sp[2] =
        n_plus[0] * TPID::par_S_plus[1] - n_plus[1] * TPID::par_S_plus[0];
    nm_Sm[0] =
        n_minus[1] * TPID::par_S_minus[2] - n_minus[2] * TPID::par_S_minus[1];
    nm_Sm[1] =
        n_minus[2] * TPID::par_S_minus[0] - n_minus[0] * TPID::par_S_minus[2];
    nm_Sm[2] =
        n_minus[0] * TPID::par_S_minus[1] - n_minus[1] * TPID::par_S_minus[0];
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) { /* Bowen-York-Curvature :*/
            Aij[i][j] =
                +1.5 *
                    (TPID::par_P_plus[i] * n_plus[j] +
                     TPID::par_P_plus[j] * n_plus[i] +
                     np_Pp * n_plus[i] * n_plus[j]) /
                    r2_plus +
                1.5 *
                    (TPID::par_P_minus[i] * n_minus[j] +
                     TPID::par_P_minus[j] * n_minus[i] +
                     nm_Pm * n_minus[i] * n_minus[j]) /
                    r2_minus -
                3.0 * (np_Sp[i] * n_plus[j] + np_Sp[j] * n_plus[i]) / r3_plus -
                3.0 * (nm_Sm[i] * n_minus[j] + nm_Sm[j] * n_minus[i]) /
                    r3_minus;
            if (i == j)
                Aij[i][j] -= +1.5 * (np_Pp / r2_plus + nm_Pm / r2_minus);
        }
    }
}

/*-----------------------------------------------------------*/
/********           Nonlinear Equations                ***********/
/*-----------------------------------------------------------*/
void
TCP_NonLinEquations (CCTK_REAL A, CCTK_REAL B, CCTK_REAL X, CCTK_REAL R,
                 CCTK_REAL x, CCTK_REAL r, CCTK_REAL phi,
                 CCTK_REAL y, CCTK_REAL z, derivs U, CCTK_REAL *values)
{
        volatile __float128 r_plus, r_minus, psi, psi2, psi4, psi6;
        volatile __float128 eta, varphi, kappa;
        volatile __float128 r3_plus, r3_minus;
        /* Derivatives of the electric potential */
        volatile __float128 varphi_dx, varphi_dy, varphi_dz;
        /* Derivatives of the gravitational potential */
        volatile __float128 eta_dx, eta_dy, eta_dz;
        /* Derivatives of kappa */
        volatile __float128 kappa_dx, kappa_dy, kappa_dz;
        volatile __float128 lapU, dkappa_dkappa, dvarphi_dvarphi, dvarphi_dkappa;
        volatile __float128 kappa2, varphi2;

        volatile __float128 rho_em = TCP_compute_rho_em(x, y, z);

    r_plus = sqrt((x - TPID::par_b) * (x - TPID::par_b) + y * y + z * z);
    r_minus = sqrt((x + TPID::par_b) * (x + TPID::par_b) + y * y + z * z);
    r3_plus  = r_plus * r_plus * r_plus;
    r3_minus = r_minus * r_minus * r_minus;

        eta    = (__float128)0.5 * TPID::par_m_plus / r_plus + (__float128)0.5 * TPID::par_m_minus / r_minus;
        varphi = (__float128)0.5 * TPID::par_q_plus / r_plus + (__float128)0.5 * TPID::par_q_minus / r_minus;
        kappa  = (__float128)1. + U.d0[0] + eta;

        kappa2  = kappa * kappa;
        varphi2 = varphi * varphi;

        psi = sqrtq(kappa2 - varphi2);
        psi2 = kappa2 - varphi2;
        psi4 = psi2 * psi2;
        psi6 = psi2 * psi4;

        eta_dx = - (__float128)0.5 * (TPID::par_m_plus * (x - TPID::par_b) / r3_plus + TPID::par_m_minus * (x + TPID::par_b) / r3_minus);
        eta_dy = - (__float128)0.5 * (TPID::par_m_plus * y / r3_plus           + TPID::par_m_minus * y / r3_minus);
        eta_dz = - (__float128)0.5 * (TPID::par_m_plus * z / r3_plus           + TPID::par_m_minus * z / r3_minus);
      
        varphi_dx = - (__float128)0.5 * (TPID::par_q_plus * (x - TPID::par_b) / r3_plus + TPID::par_q_minus * (x + TPID::par_b)/ r3_minus);
        varphi_dy = - (__float128)0.5 * (TPID::par_q_plus * y / r3_plus           + TPID::par_q_minus * y / r3_minus);
        varphi_dz = - (__float128)0.5 * (TPID::par_q_plus * z / r3_plus           + TPID::par_q_minus * z / r3_minus);
      
        kappa_dx = U.d1[0] + eta_dx;
        kappa_dy = U.d2[0] + eta_dy;
        kappa_dz = U.d3[0] + eta_dz;
      
        /* ∂ₐ κ ∂ᵃ κ */
        dkappa_dkappa = kappa_dx * kappa_dx + kappa_dy * kappa_dy + kappa_dz * kappa_dz;
      
        /* ∂ₐ φ ∂ᵃ φ */
        dvarphi_dvarphi = varphi_dx * varphi_dx + varphi_dy * varphi_dy + varphi_dz * varphi_dz;
      
        /* ∂ₐ κ ∂ᵃ φ */
        dvarphi_dkappa = kappa_dx * varphi_dx + kappa_dy * varphi_dy + kappa_dz * varphi_dz;
      
        /* Laplacian of u: ∇²u */
        lapU = U.d11[0] + U.d22[0] + U.d33[0];
      
        /* The Hamiltonian constraint is:
           κ ∇² u +  ∂ₐ κ ∂ᵃ κ (1 - κ²/ψ²) - ∂ₐ φ ∂ᵃ φ (1 + φ²/ψ²) + 2 κ φ / ψ² ∂ₐ κ ∂ᵃ ψ
           + 1/8 AᵢⱼAⁱʲ/ ψ⁶ + 2 π / ψ² ρ = 0
      
           This equation involves a lot of HUGE numbers when the collocation point is
           close to a puncture. In a perfect world, these numbers cancel and give
           zero. Due to floating-point arithmetic, this does not happen. We could not
           find a clever way to write the equation that avoid these problems, so to
           mitigate them, we use two tricks: 1. we switch to quadruple precision, 2.
           we use some clever summation methods.
        */
      
        __float128 *terms;
        int num_terms = 6;
        terms = (__float128*) malloc(num_terms * sizeof(__float128));
      
        terms[0] = + kappa * lapU;
        terms[1] = + dkappa_dkappa * ((__float128)1. - kappa2 / psi2);
        terms[2] = - dvarphi_dvarphi * ((__float128)1. + varphi2 / psi2);
        terms[3] = + (__float128)2. * kappa * varphi / psi2 * dvarphi_dkappa;
        terms[4] = + (__float128)0.125 * TCP_BY_KKofxyz (x, y, z) / psi6;
        terms[5] = + (__float128)2.0 * Pi / psi2 * rho_em;
      
        values[0] = Klein_sum(terms, num_terms);
      
        free(terms);
      
      }

/*-----------------------------------------------------------*/
/********               Linear Equations                ***********/
/*-----------------------------------------------------------*/
void TCP_LinEquations(CCTK_REAL A, CCTK_REAL B, CCTK_REAL X, CCTK_REAL R,
                  CCTK_REAL x, CCTK_REAL r, CCTK_REAL phi, CCTK_REAL y,
                  CCTK_REAL z, derivs dU, derivs U, CCTK_REAL *values) 
                  {
        volatile __float128 r_plus, r_minus, r3_plus, r3_minus, psi, psi2, psi4, psi8;
        volatile __float128 eta, varphi, eta_dx, eta_dy, eta_dz, varphi_dx, varphi_dy, varphi_dz,
            kappa, kappa_dx, kappa_dy, kappa_dz, kappa2, varphi2, eta2;
        volatile __float128 lapdU, lapU, dU_deta, dU_U, dkappa_ddU, dkappa_dkappa, dvarphi_dvarphi,
            dvarphi_dkappa, dvarphi_ddU;

        volatile __float128 rho_em = TCP_compute_rho_em(x, y, z);
    r_plus = sqrt((x - TPID::par_b) * (x - TPID::par_b) + y * y + z * z);
    r_minus = sqrt((x + TPID::par_b) * (x + TPID::par_b) + y * y + z * z);
    r3_plus  = r_plus * r_plus * r_plus;
    r3_minus = r_minus * r_minus * r_minus;


  eta    = (__float128)0.5 * TPID::par_m_plus / r_plus + (__float128)0.5 * TPID::par_m_minus / r_minus;
  varphi = (__float128)0.5 * TPID::par_q_plus / r_plus + (__float128)0.5 * TPID::par_q_minus / r_minus;
  kappa  = (__float128)1 + U.d0[0] + eta;

  eta2    = eta    * eta;
  varphi2 = varphi * varphi;
  kappa2  = kappa  * kappa;

  eta_dx = - (__float128)0.5 * (TPID::par_m_plus * (x - TPID::par_b) / r3_plus + TPID::par_m_minus * (x + TPID::par_b) / r3_minus);
  eta_dy = - (__float128)0.5 * (TPID::par_m_plus * y / r3_plus           + TPID::par_m_minus * y / r3_minus);
  eta_dz = - (__float128)0.5 * (TPID::par_m_plus * z / r3_plus           + TPID::par_m_minus * z / r3_minus);

  varphi_dx = - (__float128)0.5 * (TPID::par_q_plus * (x - TPID::par_b) / r3_plus + TPID::par_q_minus * (x + TPID::par_b)/ r3_minus);
  varphi_dy = - (__float128)0.5 * (TPID::par_q_plus * y / r3_plus           + TPID::par_q_minus * y / r3_minus);
  varphi_dz = - (__float128)0.5 * (TPID::par_q_plus * z / r3_plus           + TPID::par_q_minus * z / r3_minus);

  kappa_dx = U.d1[0] + eta_dx;
  kappa_dy = U.d2[0] + eta_dy;
  kappa_dz = U.d3[0] + eta_dz;

  psi  = sqrtq(kappa2 - varphi2);
  psi2 = psi * psi;
  psi4 = psi2 * psi2;
  psi8 = psi4 * psi4;

  /* Laplacian of δ u: ∇² δ u */
  lapdU = dU.d11[0] + dU.d22[0] + dU.d33[0];
  /* Laplacian of u: ∇²u */
  lapU  = U.d11[0] + U.d22[0] + U.d33[0];

  /* ∂ₐ κ ∂ᵃ δu */
  dkappa_ddU = kappa_dx * dU.d1[0] + kappa_dy * dU.d2[0] + kappa_dz * dU.d3[0];

  /* ∂ₐ φ ∂ᵃ δu */
  dvarphi_ddU = varphi_dx * dU.d1[0] + varphi_dy * dU.d2[0] + varphi_dz * dU.d3[0];

  /* ∂ₐ κ ∂ᵃ κ */
  dkappa_dkappa = kappa_dx * kappa_dx + kappa_dy * kappa_dy + kappa_dz * kappa_dz;

  /* ∂ₐ φ ∂ᵃ φ */
  dvarphi_dvarphi = varphi_dx * varphi_dx + varphi_dy * varphi_dy + varphi_dz * varphi_dz;

  /* ∂ₐ κ ∂ᵃ φ */
  dvarphi_dkappa = kappa_dx * varphi_dx + kappa_dy * varphi_dy + kappa_dz * varphi_dz;

  /* Given that δψ = δu κ / ψ */

  /* κ ∇²δu + δu ∇²u + 2 ∂ₐ κ ∂ᵃ δu + 2 ψ⁻³ δψ (κ² ∂ₐ κ ∂ᵃ κ + φ² ∂ₐ φ ∂ᵃ φ - 2 φ κ ∂ₐ κ ∂ᵃ φ) */
  /* - 2 (κ δu ∂ₐ κ  ∂ᵃ κ + κ² ∂ₐ κ  ∂ᵃ δu - φ δu ∂ₐ κ  ∂ᵃ φ - φ κ ∂ₐ δu ∂ᵃ φ ) / ψ²  */
  /* - 6/8 AᵢⱼAⁱʲ/ ψ⁷ δψ - 4 π / ψ³ ρ δ ψ = 0 */

  /* As in NonLinearEquations:
   *
   * This equation involves a lot of HUGE numbers when the collocation point is
     close to a puncture. In a perfect world, these numbers cancel and give
     zero. Due to floating-point arithmetic, this does not happen. We could not
     find a clever way to write the equation that avoid these problems, to
     mitigate them, we use two tricks: 1. we switch to quadruple precision, 2.
     we use some clever summation methods. */
  __float128 *terms;
  int num_terms = 12;
  terms = (__float128*) malloc(num_terms * sizeof(__float128));

  terms[0] = + lapdU * kappa;
  terms[1] = + dU.d0[0] * lapU;
  terms[2] = + (__float128)2 * dkappa_ddU;
  terms[3] = + (__float128)2 * kappa * dU.d0[0] / psi4 * kappa2 * dkappa_dkappa;
  terms[4] = + (__float128)2 * kappa * dU.d0[0] / psi4 * varphi2 * dvarphi_dvarphi;
  terms[5] = - (__float128)4 * kappa * dU.d0[0] / psi4 * kappa * varphi * dvarphi_dkappa;
  terms[6] = - (__float128)2 * kappa * dU.d0[0] * dkappa_dkappa / psi2;
  terms[7] = - (__float128)2 * kappa2 * dkappa_ddU / psi2;
  terms[8] = + (__float128)2 * varphi * dU.d0[0] * dvarphi_dkappa / psi2;
  terms[9] = + (__float128)2 * kappa * varphi * dvarphi_ddU / psi2;
  terms[10] = - (__float128)0.75 * TCP_BY_KKofxyz (x, y, z) / psi8 * kappa * dU.d0[0];
  terms[11] = - (__float128)4 * Pi / psi4 * rho_em * kappa * dU.d0[0];

  values[0] = Klein_sum(terms, num_terms);

  free(terms);
}



/*-----------------------------------------------------------*/
