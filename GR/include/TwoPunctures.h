#ifndef SFCSORTBENCH_TWOPUNCTURES_H
#define SFCSORTBENCH_TWOPUNCTURES_H

#define StencilSize 19
#define N_PlaneRelax 1
#define NRELAX 200
#define Step_Relax 1

#define TAYLOR_EXPANSION 0
#define EVALUATION 1

#define LAPSE_ANTISYMMETRIC 0
#define LAPSE_AVERAGED 1
#define LAPSE_PSIN 2
#define LAPSE_BROWNSVILLE 3

#define CS_FACTOR 0
#define CS_FACTOR_DERIVS 1
#define CS_FACTOR_SECOND_DERIVS 2

#define MT_STATIC_CONFORMAL 0
#define MT_STANDARD 1

#include "TPUtilities.h"

typedef struct DERIVS {
    CCTK_REAL *d0, *d1, *d2, *d3, *d11, *d12, *d13, *d22, *d23, *d33;
} derivs;

/*
Files of "TwoPunctures":
        TwoPunctures.c
        FuncAndJacobian.c
        CoordTransf.c
        Equations.c
        Newton.c
        utilities.c (see utilities.h)
**************************

*/

void TwoPunctures(const double xx1, const double yy1, const double zz1,
                  double *vars, double *mp, double *mm, double *mp_adm,
                  double *mm_adm, double *E, double *J1, double *J2,
                  double *J3);

/* Routines in  "TwoPunctures.c"*/
CCTK_REAL TCP_TestSolution(CCTK_REAL A, CCTK_REAL B, CCTK_REAL X, CCTK_REAL R,
                       CCTK_REAL phi);
void TCP_TestVector_w(CCTK_REAL *par, int nvar, int n1, int n2, int n3,
                  CCTK_REAL *w);

/* Routines in  "FuncAndJacobian.c"*/
int TCP_Index(int ivar, int i, int j, int k, int nvar, int n1, int n2, int n3);
void TCP_allocate_derivs(derivs *v, int n);
void TCP_free_derivs(derivs *v, int n);
void TCP_Derivatives_AB3(int nvar, int n1, int n2, int n3, derivs v);
void TCP_F_of_v(int nvar, int n1, int n2, int n3, derivs v, CCTK_REAL *F, derivs u);
void TCP_J_times_dv(int nvar, int n1, int n2, int n3, derivs dv, CCTK_REAL *Jdv,
                derivs u);
void TCP_JFD_times_dv(int i, int j, int k, int nvar, int n1, int n2, int n3,
                  derivs dv, derivs u, CCTK_REAL *values);
void TCP_SetMatrix_JFD(int nvar, int n1, int n2, int n3, derivs u, int *ncols,
                   int **cols, CCTK_REAL **Matrix);
CCTK_REAL TCP_PunctEvalAtArbitPosition(CCTK_REAL *v, int ivar, CCTK_REAL A,
                                   CCTK_REAL B, CCTK_REAL phi, int nvar, int n1,
                                   int n2, int n3);
void TCP_calculate_derivs(int i, int j, int k, int ivar, int nvar, int n1, int n2,
                      int n3, derivs v, derivs vv);
CCTK_REAL TCP_interpol(CCTK_REAL a, CCTK_REAL b, CCTK_REAL c, derivs v);
CCTK_REAL TCP_PunctTaylorExpandAtArbitPosition(int ivar, int nvar, int n1, int n2,
                                           int n3, derivs v, CCTK_REAL x,
                                           CCTK_REAL y, CCTK_REAL z);
CCTK_REAL TCP_PunctIntPolAtArbitPosition(int ivar, int nvar, int n1, int n2, int n3,
                                     derivs v, CCTK_REAL x, CCTK_REAL y,
                                     CCTK_REAL z);
void TCP_SpecCoef(int n1, int n2, int n3, int ivar, CCTK_REAL *v, CCTK_REAL *cf);
CCTK_REAL TCP_PunctEvalAtArbitPositionFast(CCTK_REAL *v, int ivar, CCTK_REAL A,
                                       CCTK_REAL B, CCTK_REAL phi, int nvar,
                                       int n1, int n2, int n3);
CCTK_REAL TCP_PunctIntPolAtArbitPositionFast(int ivar, int nvar, int n1, int n2,
                                         int n3, derivs v, CCTK_REAL x,
                                         CCTK_REAL y, CCTK_REAL z);

/* Routines in  "CoordTransf.c"*/
void TCP_AB_To_XR(int nvar, CCTK_REAL A, CCTK_REAL B, CCTK_REAL *X, CCTK_REAL *R,
              derivs U);
void TCP_C_To_c(int nvar, CCTK_REAL X, CCTK_REAL R, CCTK_REAL *x, CCTK_REAL *r,
            derivs U);
void TCP_rx3_To_xyz(int nvar, CCTK_REAL x, CCTK_REAL r, CCTK_REAL phi, CCTK_REAL *y,
                CCTK_REAL *z, derivs U);
void TCP_ijk_To_AB3 (int i, int j, int k, int n1, int n2, int n3,
                    CCTK_REAL* const A, CCTK_REAL* const B, CCTK_REAL* const phi);
void TCP_AB3_To_xyz (int nvar, CCTK_REAL A, CCTK_REAL B, CCTK_REAL phi, int n1,
                        int n2, int n3, derivs U);
void TCP_xyz_To_AB3 (CCTK_REAL x, CCTK_REAL y, CCTK_REAL z,
                        int n1, int n2, int n3,
                        CCTK_REAL* const A, CCTK_REAL* const B, CCTK_REAL* const phi);


/* Routines in  "Equations.c"*/
CCTK_REAL TCP_BY_KKofxyz(CCTK_REAL x, CCTK_REAL y, CCTK_REAL z);
void TCP_BY_Aijofxyz(CCTK_REAL x, CCTK_REAL y, CCTK_REAL z, CCTK_REAL Aij[3][3]);
void TCP_NonLinEquations (CCTK_REAL A, CCTK_REAL B, CCTK_REAL X, CCTK_REAL R,
    CCTK_REAL x, CCTK_REAL r, CCTK_REAL phi,
    CCTK_REAL y, CCTK_REAL z, derivs U, CCTK_REAL *values);
void TCP_LinEquations(CCTK_REAL A, CCTK_REAL B, CCTK_REAL X, CCTK_REAL R,
                  CCTK_REAL x, CCTK_REAL r, CCTK_REAL phi, CCTK_REAL y,
                  CCTK_REAL z, derivs dU, derivs U, CCTK_REAL *values);

/* Routines in  "Newton.c"*/
void TCP_TestRelax(int nvar, int n1, int n2, int n3, derivs v, CCTK_REAL *dv);
void TCP_Newton(int nvar, int n1, int n2, int n3, derivs v, CCTK_REAL tol,
            int itmax);
/* Routines in "EM.c" */
void TCP_compute_electromagnetic_fields_one(__float128 xb, __float128 xx, __float128 yy, __float128 zz,
    __float128 *ex, __float128 *ey, __float128 *ez,
    __float128 *bx, __float128 *by, __float128 *bz,
    int extend_radius);
void TCP_compute_electromagnetic_fields(__float128 xx, __float128 yy, __float128 zz,
__float128 *ex, __float128 *ey, __float128 *ez,
__float128 *bx, __float128 *by, __float128 *bz,
int extend_radius);
void TCP_compute_electromagnetic_potentials_one(CCTK_REAL xb, CCTK_REAL xx, CCTK_REAL yy, CCTK_REAL zz,
        CCTK_REAL *phi_em, CCTK_REAL *ax, CCTK_REAL *ay, CCTK_REAL *az,
        CCTK_INT extend_radius);
void TCP_compute_electromagnetic_potentials(CCTK_REAL xx, CCTK_REAL yy, CCTK_REAL zz,
    CCTK_REAL *phi_em, CCTK_REAL *ax, CCTK_REAL *ay, CCTK_REAL *az,
    CCTK_INT extend_radius);
__float128 TCP_compute_rho_em(__float128 xx, __float128 yy, __float128 zz);
//Adding fields for the dilaton:
void TCP_compute_dilaton_fields_one(__float128 xb,
    __float128 xx, __float128 yy, __float128 zz,
    __float128 *phi, __float128 *Dphi,
    int extend_radius);
void TCP_compute_Dilaton_fields(__float128 xx, __float128 yy, __float128 zz,
        __float128 *phi, __float128 *Dphi,
        int extend_radius);
/* Routines in  "ADM.c"*/
void TCP_compute_adm_integrals(int nvar, CCTK_REAL mp, CCTK_REAL mm,
                               int n1, int n2, int n3, derivs cf_v,
                               CCTK_REAL * const adm_mass,
                               CCTK_REAL * const adm_angular_momentum_x,
                               CCTK_REAL * const adm_angular_momentum_y,
                               CCTK_REAL * const adm_angular_momentum_z,
                               CCTK_REAL * const adm_linear_momentum_x,
                               CCTK_REAL * const adm_linear_momentum_y,
                               CCTK_REAL * const adm_linear_momentum_z,
                               CCTK_REAL * const charge,
                               CCTK_REAL radius);


/**@brief write the tp solve to a file*/
void TPStore(double *mp, double *mm, double *mp_adm, double *mm_adm, double *E,
             double *J1, double *J2, double *J3, const char *fprefix);
/**@brief restore the tp solve from file. */
void TPRestore(CCTK_REAL *&F, derivs &u, derivs &v, derivs &cf_v,
               const char *fprefix, bool mpi_bcast = true);

/*
 27: -1.325691774825335e-03
 37: -1.325691778944117e-03
 47: -1.325691778942711e-03

 17: -1.510625972641537e-03
 21: -1.511443006977708e-03
 27: -1.511440785153687e-03
 37: -1.511440809549005e-03
 39: -1.511440809597588e-03
 */
#endif
