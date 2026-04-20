/* TwoPunctures:  File  "utilities.h"*/
#pragma once
#include <math.h>

#include <cstdio>
#include <cstdlib>
#include "mpi.h"
#include "tpid_params.h"

static void code_exit(const char *s) {
    printf("EXITING SOLVER: %s\n", s);
    exit(-1);
}

#define CCTK_REAL double
#define CCTK_INT int
#define CCTK_ERROR code_exit

#define Pi 3.14159265358979323846264338328
#define Pih 1.57079632679489661923132169164 /* Pi/2*/
#define Piq 0.78539816339744830961566084582 /* Pi/4*/

#define TINY 1.0e-20
#define SWAP(a, b)  \
    {               \
        temp = (a); \
        (a) = (b);  \
        (b) = temp; \
    }

#define nrerror TCP_nrerror
#define ivector TCP_ivector
#define dvector TCP_dvector
#define imatrix TCP_imatrix
#define dmatrix TCP_dmatrix
#define d3tensor TCP_d3tensor
#define free_ivector TCP_free_ivector
#define free_dvector TCP_free_dvector
#define free_imatrix TCP_free_imatrix
#define free_dmatrix TCP_free_dmatrix
#define free_d3tensor TCP_free_d3tensor

#define minimum2 TCP_minimum2
#define minimum3 TCP_minimum3
#define maximum2 TCP_maximum2
#define maximum3 TCP_maximum3
#define pow_int TCP_pow_int

#define chebft_Zeros TCP_chebft_Zeros
#define chebft_Extremes TCP_chebft_Extremes
#define chder TCP_chder
#define chebev TCP_chebev
#define fourft TCP_fourft
#define fourder TCP_fourder
#define fourder2 TCP_fourder2
#define fourev TCP_fourev

#define norm1 TCP_norm1
#define norm2 TCP_norm2
#define scalarproduct TCP_scalarproduct

inline double EXTEND(double M, double r)
{
    const double R = TPID::TCP_Extend_Radius;

    if (R <= 0.0) return M / r;

    if (r >= R)
        return M / r;

    return M * ( 3.0/8.0 * pow(r, 4) / pow(R, 5)
               - 5.0/4.0 * pow(r, 2) / pow(R, 3)
               + 15.0/8.0 / R );
}


void nrerror(char error_text[]);
int *ivector(long nl, long nh);
CCTK_REAL *dvector(long nl, long nh);
int **imatrix(long nrl, long nrh, long ncl, long nch);
CCTK_REAL **dmatrix(long nrl, long nrh, long ncl, long nch);
CCTK_REAL ***d3tensor(long nrl, long nrh, long ncl, long nch, long ndl,
                      long ndh);
void free_ivector(int *v, long nl, long nh);
void free_dvector(CCTK_REAL *v, long nl, long nh);
void free_imatrix(int **m, long nrl, long nrh, long ncl, long nch);
void free_dmatrix(CCTK_REAL **m, long nrl, long nrh, long ncl, long nch);
void free_d3tensor(CCTK_REAL ***t, long nrl, long nrh, long ncl, long nch,
                   long ndl, long ndh);

int minimum2(int i, int j);
int minimum3(int i, int j, int k);
int maximum2(int i, int j);
int maximum3(int i, int j, int k);
int pow_int(int mantisse, int exponent);

void chebft_Zeros(CCTK_REAL u[], int n, int inv);
void chebft_Extremes(CCTK_REAL u[], int n, int inv);
void chder(CCTK_REAL *c, CCTK_REAL *cder, int n);
CCTK_REAL chebev(CCTK_REAL a, CCTK_REAL b, CCTK_REAL c[], int m, CCTK_REAL x);
void fourft(CCTK_REAL *u, int N, int inv);
void fourder(CCTK_REAL u[], CCTK_REAL du[], int N);
void fourder2(CCTK_REAL u[], CCTK_REAL d2u[], int N);
CCTK_REAL fourev(CCTK_REAL *u, int N, CCTK_REAL x);

CCTK_REAL norm1(CCTK_REAL *v, int n);
CCTK_REAL norm2(CCTK_REAL *v, int n);
CCTK_REAL scalarproduct(CCTK_REAL *v, CCTK_REAL *w, int n);

extern unsigned int TP_OMP_THREADS;
extern MPI_Comm TP_MPI_COMM;
extern unsigned int TP_SOLVE_VALID;

__float128 Klein_sum (__float128 *input, int num_terms);