/* TwoPunctures:  File  "CoordTransf.c"*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <time.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>

#include "TPUtilities.h"
#include "TwoPunctures.h"
/*-----------------------------------------------------------*/
static inline CCTK_REAL min (CCTK_REAL const x, CCTK_REAL const y)
{
  return x < y ? x : y;
}

/*-----------------------------------------------------------*/
void TCP_AB_To_XR(int nvar, CCTK_REAL A, CCTK_REAL B, CCTK_REAL *X, CCTK_REAL *R,
              derivs U)
/* On Entrance: U.d0[]=U[]; U.d1[] =U[]_A;  U.d2[] =U[]_B;  U.d3[] =U[]_3;  */
/*                          U.d11[]=U[]_AA; U.d12[]=U[]_AB; U.d13[]=U[]_A3; */
/*                          U.d22[]=U[]_BB; U.d23[]=U[]_B3; U.d33[]=U[]_33; */
/* At Exit:     U.d0[]=U[]; U.d1[] =U[]_X;  U.d2[] =U[]_R;  U.d3[] =U[]_3;  */
/*                          U.d11[]=U[]_XX; U.d12[]=U[]_XR; U.d13[]=U[]_X3; */
/*                          U.d22[]=U[]_RR; U.d23[]=U[]_R3; U.d33[]=U[]_33; */
{
    CCTK_REAL At = 0.5 * (A + 1), A_X, A_XX, B_R, B_RR;
    int ivar;

    *X = 2 * atanh(At);
    *R = Pih + 2 * atan(B);

    A_X = 1 - At * At;
    A_XX = -At * A_X;
    B_R = 0.5 * (1 + B * B);
    B_RR = B * B_R;

    for (ivar = 0; ivar < nvar; ivar++) {
        U.d11[ivar] = A_X * A_X * U.d11[ivar] + A_XX * U.d1[ivar];
        U.d12[ivar] = A_X * B_R * U.d12[ivar];
        U.d13[ivar] = A_X * U.d13[ivar];
        U.d22[ivar] = B_R * B_R * U.d22[ivar] + B_RR * U.d2[ivar];
        U.d23[ivar] = B_R * U.d23[ivar];
        U.d1[ivar] = A_X * U.d1[ivar];
        U.d2[ivar] = B_R * U.d2[ivar];
    }
}

/*-----------------------------------------------------------*/
void TCP_C_To_c(int nvar, CCTK_REAL X, CCTK_REAL R, CCTK_REAL *x, CCTK_REAL *r,
            derivs U)
/* On Entrance: U.d0[]=U[]; U.d1[] =U[]_X;  U.d2[] =U[]_R;  U.d3[] =U[]_3;  */
/*                          U.d11[]=U[]_XX; U.d12[]=U[]_XR; U.d13[]=U[]_X3; */
/*                          U.d22[]=U[]_RR; U.d23[]=U[]_R3; U.d33[]=U[]_33; */
/* At Exit:     U.d0[]=U[]; U.d1[] =U[]_x;  U.d2[] =U[]_r;  U.d3[] =U[]_3;  */
/*                          U.d11[]=U[]_xx; U.d12[]=U[]_xr; U.d13[]=U[]_x3; */
/*                          U.d22[]=U[]_rr; U.d23[]=U[]_r3; U.d33[]=U[]_33; */
{
    CCTK_REAL C_c2, U_cb, U_CB;
    gsl_complex C, C_c, C_cc, c, c_C, c_CC, U_c, U_cc, U_C, U_CC;
    int ivar;

    C = gsl_complex_rect(X, R);

    c = gsl_complex_mul_real(gsl_complex_cosh(C), TPID::par_b); /* c=b*cosh(C)*/
    c_C = gsl_complex_mul_real(gsl_complex_sinh(C), TPID::par_b);
    c_CC = c;

    C_c = gsl_complex_inverse(c_C);
    C_cc = gsl_complex_negative(
        gsl_complex_mul(gsl_complex_mul(C_c, C_c), gsl_complex_mul(C_c, c_CC)));
    C_c2 = gsl_complex_abs2(C_c);

    for (ivar = 0; ivar < nvar; ivar++) {
        /* U_C = 0.5*(U_X3-i*U_R3)*/
        /* U_c = U_C*C_c = 0.5*(U_x3-i*U_r3)*/
        U_C = gsl_complex_rect(0.5 * U.d13[ivar], -0.5 * U.d23[ivar]);
        U_c = gsl_complex_mul(U_C, C_c);
        U.d13[ivar] = 2. * GSL_REAL(U_c);
        U.d23[ivar] = -2. * GSL_IMAG(U_c);

        /* U_C = 0.5*(U_X-i*U_R)*/
        /* U_c = U_C*C_c = 0.5*(U_x-i*U_r)*/
        U_C = gsl_complex_rect(0.5 * U.d1[ivar], -0.5 * U.d2[ivar]);
        U_c = gsl_complex_mul(U_C, C_c);
        U.d1[ivar] = 2. * GSL_REAL(U_c);
        U.d2[ivar] = -2. * GSL_IMAG(U_c);

        /* U_CC = 0.25*(U_XX-U_RR-2*i*U_XR)*/
        /* U_CB = d^2(U)/(dC*d\bar{C}) = 0.25*(U_XX+U_RR)*/
        U_CC = gsl_complex_rect(0.25 * (U.d11[ivar] - U.d22[ivar]),
                                -0.5 * U.d12[ivar]);
        U_CB = 0.25 * (U.d11[ivar] + U.d22[ivar]);

        /* U_cc = C_cc*U_C+(C_c)^2*U_CC*/
        U_cb = U_CB * C_c2;
        U_cc =
            gsl_complex_add(gsl_complex_mul(C_cc, U_C),
                            gsl_complex_mul(gsl_complex_mul(C_c, C_c), U_CC));

        /* U_xx = 2*(U_cb+Re[U_cc])*/
        /* U_rr = 2*(U_cb-Re[U_cc])*/
        /* U_rx = -2*Im[U_cc]*/
        U.d11[ivar] = 2 * (U_cb + GSL_REAL(U_cc));
        U.d22[ivar] = 2 * (U_cb - GSL_REAL(U_cc));
        U.d12[ivar] = -2 * GSL_IMAG(U_cc);
    }

    *x = GSL_REAL(c);
    *r = GSL_IMAG(c);
}

/*-----------------------------------------------------------*/
void TCP_rx3_To_xyz(int nvar, CCTK_REAL x, CCTK_REAL r, CCTK_REAL phi, CCTK_REAL *y,
                CCTK_REAL *z, derivs U)
/* On Entrance: U.d0[]=U[]; U.d1[] =U[]_x;  U.d2[] =U[]_r;  U.d3[] =U[]_3;  */
/*                          U.d11[]=U[]_xx; U.d12[]=U[]_xr; U.d13[]=U[]_x3; */
/*                          U.d22[]=U[]_rr; U.d23[]=U[]_r3; U.d33[]=U[]_33; */
/* At Exit:     U.d0[]=U[]; U.d1[] =U[]_x;  U.d2[] =U[]_y;  U.dz[] =U[]_z;  */
/*                          U.d11[]=U[]_xx; U.d12[]=U[]_xy; U.d1z[]=U[]_xz; */
/*                          U.d22[]=U[]_yy; U.d2z[]=U[]_yz; U.dzz[]=U[]_zz; */
{
    int jvar;
    CCTK_REAL
    sin_phi = sin(phi), cos_phi = cos(phi), sin2_phi = sin_phi * sin_phi,
    cos2_phi = cos_phi * cos_phi, sin_2phi = 2 * sin_phi * cos_phi,
    cos_2phi = cos2_phi - sin2_phi, r_inv = 1 / r, r_inv2 = r_inv * r_inv;

    *y = r * cos_phi;
    *z = r * sin_phi;

    for (jvar = 0; jvar < nvar; jvar++) {
        CCTK_REAL U_x = U.d1[jvar], U_r = U.d2[jvar], U_3 = U.d3[jvar],
                  U_xx = U.d11[jvar], U_xr = U.d12[jvar], U_x3 = U.d13[jvar],
                  U_rr = U.d22[jvar], U_r3 = U.d23[jvar], U_33 = U.d33[jvar];
        U.d1[jvar] = U_x;                                      /* U_x*/
        U.d2[jvar] = U_r * cos_phi - U_3 * r_inv * sin_phi;    /* U_y*/
        U.d3[jvar] = U_r * sin_phi + U_3 * r_inv * cos_phi;    /* U_z*/
        U.d11[jvar] = U_xx;                                    /* U_xx*/
        U.d12[jvar] = U_xr * cos_phi - U_x3 * r_inv * sin_phi; /* U_xy*/
        U.d13[jvar] = U_xr * sin_phi + U_x3 * r_inv * cos_phi; /* U_xz*/
        U.d22[jvar] = U_rr * cos2_phi +
                      r_inv2 * sin2_phi * (U_33 + r * U_r) /* U_yy*/
                      + sin_2phi * r_inv2 * (U_3 - r * U_r3);
        U.d23[jvar] =
            0.5 * sin_2phi * (U_rr - r_inv * U_r - r_inv2 * U_33) /* U_yz*/
            - cos_2phi * r_inv2 * (U_3 - r * U_r3);
        U.d33[jvar] = U_rr * sin2_phi +
                      r_inv2 * cos2_phi * (U_33 + r * U_r) /* U_zz*/
                      - sin_2phi * r_inv2 * (U_3 - r * U_r3);
    }
}

/*-----------------------------------------------------------*/
void
TCP_ijk_To_AB3 (int i, int j, int k, int n1, int n2, int n3,
            CCTK_REAL* const A, CCTK_REAL* const B, CCTK_REAL* const phi){
  /* Take i j k as input and return the A B phi coordinates */

  CCTK_REAL al, be;

  al = Pih * (2 * i + 1) / n1;
  *A = -cos (al);
  be = Pih * (2 * j + 1) / n2;
  *B = -cos (be);
  *phi = 2. * Pi * k / n3;

}
void
TCP_AB3_To_xyz (int nvar, CCTK_REAL A, CCTK_REAL B, CCTK_REAL phi, int n1, int n2, int n3, derivs U){
    CCTK_REAL At = 0.5 * (A + 1), A_X, B_R, X, R;
  int ivar;

  X = 2 * atanh (At);
  R = Pih + 2 * atan (B);

  A_X = 1 - At * At;
  B_R = 0.5 * (1 + B * B);

  for (ivar = 0; ivar < nvar; ivar++)
    {
      U.d1[ivar] = A_X * U.d1[ivar];
      U.d2[ivar] = B_R * U.d2[ivar];
    }

  CCTK_REAL C_c2, U_cb, U_CB, x, r;
  gsl_complex C, C_c, c, c_C, U_c, U_C;

  C = gsl_complex_rect (X, R);

  c = gsl_complex_mul_real (gsl_complex_cosh (C), TPID::par_b);	/* c=b*cosh(C)*/
  c_C = gsl_complex_mul_real (gsl_complex_sinh (C), TPID::par_b);
  C_c = gsl_complex_inverse (c_C);

  x = GSL_REAL(c);
  r = GSL_IMAG(c);

  for (ivar = 0; ivar < nvar; ivar++)
    {
      /* U_C = 0.5*(U_X-i*U_R)*/
      /* U_c = U_C*C_c = 0.5*(U_x-i*U_r)*/
      U_C = gsl_complex_rect (0.5 * U.d1[ivar], -0.5 * U.d2[ivar]);
      U_c = gsl_complex_mul (U_C, C_c);
      U.d1[ivar] = 2. * GSL_REAL(U_c);
      U.d2[ivar] = -2. * GSL_IMAG(U_c);
    }

  CCTK_REAL
    sin_phi = sin (phi),
    cos_phi = cos (phi);

  for (ivar = 0; ivar < nvar; ivar++)
    {
      CCTK_REAL U_x = U.d1[ivar], U_r = U.d2[ivar], U_3 = U.d3[ivar];

      U.d1[ivar] = U_x;		/* U_x*/
      U.d2[ivar] = U_r * cos_phi;
      /* if (r < 1e-3) printf("A %e B %e phi %e U_3 sin_phi %e\n", A, B, phi, U_3 * sin_phi); */
      if (r > pow(TPID::TCP_epsilon, 2)) U.d2[ivar] -= U_3 * sin_phi / r;	/* U_y*/
      /* if (r < 1e-3) printf("A %e B %e phi %e U_3 cos_phi %e\n", A, B, phi, U_3 * cos_phi); */
      U.d3[ivar] = U_r * sin_phi;
      if (r > pow(TPID::TCP_epsilon, 2)) U.d3[ivar] += U_3 * cos_phi / r;	/* U_z*/
    }

}
static CCTK_REAL
clamp_pm_one (CCTK_REAL val)
{
  return val < -1 ? -1 : val > 1 ? 1 : val;
}

void
TCP_xyz_To_AB3 (CCTK_REAL x, CCTK_REAL y, CCTK_REAL z,
            int n1, int n2, int n3,
            CCTK_REAL* const A, CCTK_REAL* const B, CCTK_REAL* const phi){
  /* Take x y z as input and return the A B phi coordinates */
  CCTK_REAL xs, ys, zs, rs2, X, R, aux1, aux2;

  xs = x / TPID::par_b;
  ys = y / TPID::par_b;
  zs = z / TPID::par_b;
  rs2 = ys * ys + zs * zs;
  *phi = atan2(zs, ys);
  if (*phi < 0)
      *phi += 2 * Pi;  

  aux1 = 0.5 * (xs * xs + rs2 - 1);
  aux2 = sqrt (aux1 * aux1 + rs2);
  X = asinh (sqrt (aux1 + aux2));
  R = asin (min(1.0, sqrt (-aux1 + aux2)));
  if (xs < 0)
    R = Pi - R;

  *A = clamp_pm_one(2 * tanh (0.5 * X) - 1);
  *B = clamp_pm_one(tan (0.5 * R - Piq));

}