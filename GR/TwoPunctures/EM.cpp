#include "TwoPunctures.h"
#include "TPUtilities.h"
#include <math.h>
#include <assert.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>
#include "parameters.h"
#include <quadmath.h>

void
TCP_compute_electromagnetic_fields_one(__float128 xb,
                                       __float128 xx, __float128 yy, __float128 zz,
                                       __float128 *ex, __float128 *ey, __float128 *ez,
                                       __float128 *bx, __float128 *by, __float128 *bz,
                                       int apply_ceiling){
                                        /* Coordinates */
  __float128 x, y, z, R, R2;
  x = xx - xb;
  y = yy;
  z = zz;

  /* Field parameters */
  __float128 Q;
  __float128 P;


  /* Electromagnetic fields */
  __float128 ER;
  __float128 BR;
  __float128 ex_tmp, ey_tmp, ez_tmp;
  __float128 bx_tmp, by_tmp, bz_tmp;

  Q = (xb >= 0) ? TPID::par_q_plus : TPID::par_q_minus;
  P = (xb >= 0) ? TPID::par_p_plus : TPID::par_p_minus;

  if (apply_ceiling != 0){
    R  = sqrtq(x * x + y * y + z * z + pow(TPID::TCP_epsilon, 4));
  }else{
    R  = sqrtq(x * x + y * y + z * z);
  }

  R2 = R * R;

  if (R < TPID::TCP_Tiny && apply_ceiling != 0)
    R = TPID::TCP_Tiny;

  ER = Q / R2;
  BR = P / R2;


  ex_tmp = ER * x / R;
  ey_tmp = ER * y / R;
  ez_tmp = ER * z / R;

  bx_tmp = BR * x / R;
  by_tmp = BR * y / R;
  bz_tmp = BR * z / R;


  *ex = ex_tmp;
  *ey = ey_tmp;
  *ez = ez_tmp;

  *bx = bx_tmp;
  *by = by_tmp;
  *bz = bz_tmp;
}
void
TCP_compute_electromagnetic_fields(__float128 xx, __float128 yy, __float128 zz,
                                   __float128 *ex, __float128 *ey, __float128 *ez,
                                   __float128 *bx, __float128 *by, __float128 *bz,
                                   int apply_ceiling){

  /* Compute the em fields in (xx,yy,zz) for two punctures */
  /* Electromagnetic fields for the two punctures */
  __float128 ex_p, ey_p, ez_p, bx_p, by_p, bz_p;
  __float128 ex_m, ey_m, ez_m, bx_m, by_m, bz_m;

  /* Plus puncture */
  TCP_compute_electromagnetic_fields_one(TPID::par_b, xx, yy, zz, &ex_p, &ey_p, &ez_p, &bx_p, &by_p, &bz_p, apply_ceiling);
  /* Minus puncture */
  TCP_compute_electromagnetic_fields_one(-TPID::par_b, xx, yy, zz, &ex_m, &ey_m, &ez_m, &bx_m, &by_m, &bz_m, apply_ceiling);

  *ex = (ex_p + ex_m);
  *ey = (ey_p + ey_m);
  *ez = (ez_p + ez_m);

  *bx = (bx_p + bx_m);
  *by = (by_p + by_m);
  *bz = (bz_p + bz_m);

}
void TCP_compute_dilaton_fields_one(
    __float128 xb,
    __float128 xx, __float128 yy, __float128 zz,
    __float128 *phi, __float128 *Dphi,
    int apply_ceiling)
{
    // ----------------------------
    // Coordinate shift (puncture-centered)
    // ----------------------------
    const __float128 x = xx - xb;
    const __float128 y = yy;
    const __float128 z = zz;

    // ----------------------------
    // Radius from puncture
    // ----------------------------
    __float128 R2 = x*x + y*y + z*z;

    if (apply_ceiling != 0) {
        // add epsilon^4 under sqrt 
        const __float128 eps  = TPID::TCP_epsilon;
        const __float128 eps2 = eps * eps;
        const __float128 eps4 = eps2 * eps2;
        R2 += eps4;
    }

    __float128 R = sqrtq(R2);

    const __float128 MIN_POS = TPID::TCP_Tiny;
    if (apply_ceiling && R < MIN_POS) R = MIN_POS;

    // ----------------------------
    // Select physical charge/mass by puncture sign
    // ----------------------------
    const __float128 Q_phys    = (xb >= 0) ? TPID::par_q_plus  : TPID::par_q_minus;
    const __float128 Mphys     = (xb >= 0) ? TPID::par_m_plus  : TPID::par_m_minus;

    // Guard against pathological input
    if (fabsq(Mphys) < MIN_POS) {
        *phi  = 0.0q;
        *Dphi = 0.0q;
        return;
    }

    // Integration parameter mass 
    const __float128 m = Mphys - 0.5q * Q_phys * Q_phys / Mphys;

    // alpha_ks definition MUST match reference: atanh(Q_phys/(Mphys*sqrt(2)))
    __float128 arg = Q_phys / (Mphys * sqrtq(2.0q));

    // Clamp to avoid NaNs at |arg|>=1 (atanh domain)
    // Choose a small margin;
    const __float128 margin = 10.0q * MIN_POS;
    const __float128 maxarg = 1.0q - margin;
    if (arg >  maxarg) arg =  maxarg;
    if (arg < -maxarg) arg = -maxarg;

    const __float128 alpha_ks = atanhq(arg);

    // sinh^2(alpha)
    __float128 s = sinhq(alpha_ks);
    const __float128 s2 = s * s;

    // ----------------------------
    // a_spin = 0 
    // f0 = (R + 0.5*m)^2
    // rho_bar^2 = f0^2 + 2*m*f0*R*sinh^2(alpha)
    // phi = -1/2 log( rho_bar^2 / f0^2 )
    // ----------------------------
    const __float128 Rp = (R + 0.5q * m);
    const __float128 f0 = Rp * Rp;

    // ratio = rho_bar^2 / f0^2 = 1 + 2*m*R*s2 / f0
    __float128 ratio = 1.0q + (2.0q * m * R * s2) / f0;

    // Clamp ratio positive for log safety (should be positive physically, but guard anyway)
    if (ratio < MIN_POS) ratio = MIN_POS;

    *phi = -0.5q * logq(ratio);

    // ----------------------------
    // Radial derivative dphi/dR 
    //
    // ratio(R) = 1 + 2*m*s2 * R / (R + 0.5*m)^2
    // dphi/dR = -1/2 * (1/ratio) * d(ratio)/dR
    //
    // d(ratio)/dR = 2*m*s2 * ( (0.5*m - R) / (R + 0.5*m)^3 )
    //
    // => dphi/dR = m*s2*(R - 0.5*m) / ( ratio * (R + 0.5*m)^3 )
    // ----------------------------
    __float128 denom = ratio * Rp * Rp * Rp;

    if (fabsq(denom) < MIN_POS) {
        denom = copysignq(MIN_POS, denom);
    }

    *Dphi = (m * s2 * (R - 0.5q * m)) / denom;
}

// Combine two punctures, scaled by alpha0
void TCP_compute_Dilaton_fields(
    __float128 xx, __float128 yy, __float128 zz,
    __float128 *phi, __float128 *Dphi,
    int apply_ceiling)
{
    __float128 phi_p, Dphi_p;
    __float128 phi_m, Dphi_m;

    // Plus puncture at +b
    TCP_compute_dilaton_fields_one(
        TPID::par_b, xx, yy, zz,
        &phi_p, &Dphi_p, apply_ceiling);

    // Minus puncture at -b
    TCP_compute_dilaton_fields_one(
        -TPID::par_b, xx, yy, zz,
        &phi_m, &Dphi_m, apply_ceiling);

    const __float128 a0 = TPID::par_alpha0;
    *phi  = a0 * (phi_p + phi_m);
    *Dphi = a0 * (Dphi_p + Dphi_m);   // <-- single factor, as desired
}

void TCP_compute_interpolated_dilaton_fields_one(__float128 xb,
                                                 __float128 xx,
                                                 __float128 yy,
                                                 __float128 zz,
                                                 __float128 *phi,
                                                 __float128 *Dphi,
                                                 int apply_ceiling)
{
 /* Coordinates relative to puncture */
  __float128 x = xx - xb;
  __float128 y = yy;
  __float128 z = zz;

  /* Radius R */
  __float128 R;
  if (apply_ceiling != 0) {
    R = sqrtq(x * x + y * y + z * z + powq(TPID::TCP_epsilon, 4));
  } else {
    R = sqrtq(x * x + y * y + z * z);
  }
  if (apply_ceiling != 0 && R < TPID::TCP_Tiny) R = TPID::TCP_Tiny;

  /* Parameters per puncture */
  __float128 Q    = (xb >= 0) ? TPID::par_q_plus : TPID::par_q_minus;
  __float128 M    = (xb >= 0) ? TPID::par_m_plus : TPID::par_m_minus;
  __float128 a0   = TPID::par_alpha0;

  /* Safety: a0 appears in denominator */
  const __float128 MIN_POS = TPID::TCP_Tiny;
  if (fabsl(a0) < MIN_POS) a0 = (a0 >= 0 ? MIN_POS : -MIN_POS);

  /* Common powers */
  __float128 a0_2 = a0 * a0;
  __float128 Q2   = Q  * Q;
  __float128 M2   = M  * M;

  /* disc = 1 - (1 - a0^2) Q^2/M^2 */
  __float128 disc = 1.0Q - (1.0Q - a0_2) * (Q2 / M2);

  /* Guard against tiny negatives from roundoff */
  if (disc < 0.0Q) disc = 0.0Q;

  __float128 sdisc = sqrtq(disc);

  /* rp, rm */
  __float128 rp = M * (1.0Q + sdisc);
  __float128 rm = (Q2 / M) * (1.0Q + a0_2) / (1.0Q + sdisc);

  /* rbar1, rbar2, rh */
  __float128 srp = sqrtq(rp);
  __float128 srm = sqrtq(rm);

  __float128 rbar1 = 0.25Q * (srp - srm) * (srp - srm);
  __float128 rbar2 = 0.25Q * (srp + srm) * (srp + srm);
  __float128 rh    = 0.25Q * (rp - rm);

  /* beta */
  __float128 beta = (2.0Q * a0_2) / (1.0Q + a0_2);

  /* Clamp log arguments if ceiling requested */
  __float128 A = R + rh;
  __float128 B = R + rbar1;
  __float128 C = R + rbar2;

  if (apply_ceiling != 0) {
    if (A < MIN_POS) A = MIN_POS;
    if (B < MIN_POS) B = MIN_POS;
    if (C < MIN_POS) C = MIN_POS;
  }

  /* phi */
  // phi = (1/(2a0)) * [ 2beta log(A) - beta log(B) - beta log(C) ]
  __float128 inv2a0 = 1.0Q / (2.0Q * a0);
  __float128 Phi    = inv2a0 * ( 2.0Q * beta * logq(A) - beta * logq(B) - beta * logq(C) );
  *phi = Phi;

  /* Dphi = dphi/dR */
  // dphi/dR = (beta/a0)/A - (beta/(2a0))*(1/B + 1/C)
  __float128 inva0 = 1.0Q / a0;

  __float128 invA = 1.0Q / A;
  __float128 invB = 1.0Q / B;
  __float128 invC = 1.0Q / C;

  __float128 dPhi_dR = (beta * inva0) * invA
                     - (beta * inv2a0) * (invB + invC);

  *Dphi = dPhi_dR;
 
}
void TCP_compute_interpolated_Dilaton_fields(__float128 xx,
                                             __float128 yy,
                                             __float128 zz,
                                             __float128 *phi,
                                             __float128 *Dphi,
                                             int apply_ceiling)
{
  __float128 phi_p,  Dphi_p;
  __float128 phi_m,  Dphi_m;

  // Plus puncture at +b
  TCP_compute_interpolated_dilaton_fields_one(TPID::par_b,
                                              xx, yy, zz,
                                              &phi_p, &Dphi_p,
                                              apply_ceiling);

  // Minus puncture at -b
  TCP_compute_interpolated_dilaton_fields_one(-TPID::par_b,
                                              xx, yy, zz,
                                              &phi_m, &Dphi_m,
                                              apply_ceiling);
//Alpha_0 should scale between our interpolative solutions
  *phi  = TPID::par_alpha0*(phi_p  + phi_m);
  *Dphi = TPID::par_alpha0*(Dphi_p + Dphi_m);
}


  __float128 TCP_compute_rho_em(__float128 xx, __float128 yy, __float128 zz) {
  // Compute the source of the Hamiltonian constraint

  __float128 ex, ey, ez, bx, by, bz, phi, Dphi, rho_em;

  TCP_compute_electromagnetic_fields(xx, yy, zz, &ex, &ey, &ez, &bx, &by, &bz, 0);

  rho_em = ex * ex + ey * ey + ez * ez + bx * bx + by * by + bz * bz;
  if (TPID::use_Dilaton) {
    TCP_compute_Dilaton_fields(xx, yy, zz, &phi, &Dphi, 0);
    
    rho_em *= expq(-2.0q * TPID::par_alpha0 * phi);
    rho_em += Dphi * Dphi;
}
else if (TPID::use_Interpolated_solution) {
  TCP_compute_interpolated_Dilaton_fields(xx, yy, zz, &phi, &Dphi, 0);
  // or TCP_compute_interpolated_Dilaton_fields(...)

  __float128 rho_kn  = rho_em;  // rho_em currently holds E^2 + B^2 up to here
  __float128 a = TPID::par_alpha0;

  // optional clamp
  if (a < 0.0q) a = 0.0q;
  if (a > 1.0q) a = 1.0q;

  __float128 rho_dil = rho_kn * expq(-2.0q * TPID::par_alpha0 * phi)
                     + Dphi * Dphi;

  rho_em = (1.0q - a) * rho_kn + a * rho_dil;
}

  return rho_em / (8.0q * Pi);
                                  }

void
TCP_compute_electromagnetic_potentials_one(CCTK_REAL xb, CCTK_REAL xx, CCTK_REAL yy, CCTK_REAL zz,
                                           CCTK_REAL *phi_em, CCTK_REAL *ax, CCTK_REAL *ay, CCTK_REAL *az,
                                           CCTK_INT apply_ceiling){
                                            /* Compute the em potentials in (xx,yy,zz) for a single source located in
     (x_b, 0, 0) */

  /* Coordinates */
  CCTK_REAL x, y, z, r, R, r2, rho2,            
  phi, th, cos_th, sin_th, cos_phi, sin_phi,  
  cos2_th, sin2_th, dxdphi, dydphi, dzdphi;

/* Field parameters */
CCTK_REAL Q;

/* Potentials */
CCTK_REAL Aphi, Ax, Ay, Az;
CCTK_REAL phi_em_tmp, ax_tmp, ay_tmp, az_tmp;

/* Move the coordinates to the origin */
x = xx - xb;
y = yy;
z = zz;

R = sqrt(x * x + y * y + z * z + pow(TPID::TCP_epsilon, 4));

if (R < TPID::TCP_Tiny && apply_ceiling != 0)
  R = TPID::TCP_Tiny;

Q = (xb >= 0) ? TPID::par_q_plus : TPID::par_q_minus;

ax_tmp = 0.;
ay_tmp = 0.;
az_tmp = 0.;

phi_em_tmp = - Q / R;

*phi_em = phi_em_tmp;
*ax = ax_tmp;
*ay = ay_tmp;
*az = az_tmp;
}
void
TCP_compute_electromagnetic_potentials(CCTK_REAL xx, CCTK_REAL yy, CCTK_REAL zz,
                                       CCTK_REAL *phi_em, CCTK_REAL *ax, CCTK_REAL *ay, CCTK_REAL *az,
                                       CCTK_INT apply_ceiling){
  /* Electromagnetic potentials for the two punctures */
  CCTK_REAL ax_p, ay_p, az_p, ax_m, ay_m, az_m;
  CCTK_REAL phi_em_p, phi_em_m;

  /* Plus puncture */
  TCP_compute_electromagnetic_potentials_one(TPID::par_b, xx, yy, zz, &phi_em_p, &ax_p, &ay_p, &az_p, apply_ceiling);
  /* Minus puncture */
  TCP_compute_electromagnetic_potentials_one(-TPID::par_b, xx, yy, zz, &phi_em_m, &ax_m, &ay_m, &az_m, apply_ceiling);

  *ax = (ax_p + ax_m);
  *ay = (ay_p + ay_m);
  *az = (az_p + az_m);

  *phi_em = (phi_em_p + phi_em_m);
}
