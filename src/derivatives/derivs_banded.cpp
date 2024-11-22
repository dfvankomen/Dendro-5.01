
/**
 * This file implements methods for solving CFD schemes using
 *  routines for solving banded linear systems from LAPACK
 *
 * Here we also implement several schemes
 * @todo finish writing here
 * @todo implement schemes
 */

#include "derivatives/derivs_banded.h"

#include <algorithm>

#include "derivatives/derivs_utils.h"
#include "lapac.h"
#include "refel.h"

namespace dendroderivs {

/**
 * KL,KU should be for the matrix we will be banded solving for. This is likely
 *  the P matrix.
 */
BandedMatrixSolveVars::BandedMatrixSolveVars(char FACT, char TRANS, int N,
                                             int NRHS, int KL, int KU,
                                             double *AB) {
    // characters
    this->FACT  = new char{FACT};
    this->TRANS = new char{TRANS};
    this->EQUED =
        new char;  // only allocate; its value will be set upon factoriztion

    // numbers
    this->N     = new int{N};
    this->LDX   = this->N;  // shares a value with N, so use the same pointer.
                            // ASSUMING NEITHER OF THEM GET MODIFIED
    this->LDB   = this->N;  // shares a value with N, so use the same pointer.
                            // ASSUMING NEITHER OF THEM GET MODIFIED
    this->KL    = new int{KL};
    this->KU    = new int{KU};
    this->NRHS  = new int{NRHS};
    this->LDAB  = new int{KL + KU + 1};
    this->LDAFB = new int{2 * KL + KU + 1};
    this->INFO  = new int;     // only allocate; it's a return value
    this->RCOND = new double;  // only allocate; it's a return value

    // arrays
    this->AB    = AB;
    this->AFB   = new double[*(this->LDAFB) * N]{};
    this->IPIV  = new int[N]{};
    this->R     = new double[N]{};
    this->C     = new double[N]{};
    this->B =
        new double[*(this->LDB) * NRHS]{};  // RHS of the solver (input from
                                            // matrix vector multiply each step)
    this->X    = new double[*(this->LDX) * NRHS]{};
    this->FERR = new double[NRHS]{};
    std::cout << "FERR: " << this->FERR << std::endl;
    std::cout << "NRHS: " << NRHS << std::endl;
    this->BERR  = new double[NRHS]{};
    this->WORK  = new double[3 * N]{};
    this->IWORK = new int[N]{};
}

BandedMatrixSolveVars::~BandedMatrixSolveVars() {
#ifdef DEBUG
    std::cout << "in BandedMatrixSolveVars destructor" << std::endl;
#endif

    // IMPORTANT NOTE we do not delete AB as it is simply a pointer to
    //  an array that belongs to a class of type BandedCompactDerivs
    // It is the responsibility of this said class to delete AB
    delete[] AFB;
    delete[] IPIV;
    delete[] R;
    delete[] C;
    delete[] X;
    delete[] FERR;
    delete[] BERR;
    delete[] WORK;
    delete[] IWORK;

    delete N;
    // delete LDX; don't need to delete as N points to this
    // delete LDB; don't need to delete as N points to this
    delete KL;
    delete KU;
    delete NRHS;
    delete LDAB;
    delete LDAFB;
    delete RCOND;
    delete INFO;
}

/**
 * This method should be called by ANY class implementing BandedCompactDerivs.
 *
 * We would like to call the constructor for the derived class before
 *  that of the base class, but I don't know of a way to do this.
 * My solution is to do the necessary work in a method which we call
 *  from the derived class constructor after the necessary values are
 *  defined.
 *
 * At class instantiation, the call order is:
 *  1) Derivs constructor
 *  2) BandedCompactDerivs constructor
 *  3) (class extending BandedCompactDerivs) constructor
 *  4) BandedCompactDerivs::init called from 3)
 *
 * @todo re-examine how we are doing this; is there a better paradigm
 *  for what I'm trying to accomplish?
 * @todo using the kVals, add a check to put a minimum on n_
 * @todo implement one-time factorization at start of program (run with FACT=N
 * or E, then change to FACT=F)
 */
void BandedCompactDerivs::init(BandedMatrixDiagonalWidths *kVals,
                               MatrixDiagonalEntries *entries) {
#ifdef DEBUG
    std::cout << "initializing BandedCompactDerivs" << std::endl;
#endif

    // todo here
    // instantiate lapackvars
    // create matrices
    // banded store them
    // (@TODO) run the solver once to factorize the matrices
    // then set params to not factorize them anymore

    // allocate derivative arrays
    this->P_         = std::vector(p_n * p_n, 0.0);
    this->Q_         = std::vector(p_n * p_n, 0.0);
    this->Pb_        = new double[(kVals->pkl + kVals->pku + 1) * p_n]{};
    this->Qb_        = new double[(kVals->qkl + kVals->qku + 1) * p_n]{};

    // allocate workspace_
    // TODO: this needs to be modified to be larger!
    // This should *always* be overwritten, NOT zero initialized
    this->workspace_ = new double[p_n * p_n * p_n];

#ifdef DEBUG
    std::cout << "just allocated D_x and array, banded and full" << std::endl;
#endif

    // TODO: NRHS is whatever the maximum number of RHS variables we're going to
    // solve is this class needs to be modified if it's what we're using at some
    // point
    this->grad_xVars =
        new BandedMatrixSolveVars('E',  // FACT (set this way to factor it once,
                                        // then store this in AFB and IPIV)
                                  'N',  // TRANS
                                  p_n,  // N
                                  p_n,  // NRHS
                                  kVals->pkl,  // KL
                                  kVals->pku,  // KU
                                  this->Pb_    // AB
        );

#ifdef DEBUG
    std::cout << "just built grad_xVars" << std::endl;
#endif

    // build derivative matrices and banded store them
    buildMatrix(P_.data(), entries->PDiagInterior, entries->PDiagBoundary, 1.0,
                p_n);
    buildMatrix(Q_.data(), entries->QDiagInterior, entries->QDiagBoundary, -1.0,
                p_n);
    bandedMatrixStore(Pb_, P_.data(), kVals->pkl, kVals->pku, p_n);
    bandedMatrixStore(Qb_, Q_.data(), kVals->qkl, kVals->qku, p_n);
#ifdef DEBUG
    std::cout << "just built and banded stored D_x matrices" << std::endl;
#endif

    /**
     * Factor the matrices and store them
     * Then tell LAPACK that they're already factored for future runs
     */
    bandedMatrixSolve(grad_xVars);
    *(grad_xVars->FACT) = 'F';
#ifdef DEBUG
    std::cout << "just stored factorization of matrices" << std::endl;
#endif
}

/**
 * Destructor
 */
BandedCompactDerivs::~BandedCompactDerivs() {
#ifdef DEBUG
    std::cout << "in BandedCompactDerivs deconstructor" << std::endl;
#endif
    // delete[] P_;
    // delete[] Q_;

    delete[] Pb_;
    delete[] Qb_;

    delete[] workspace_;

    delete grad_xVars;
    delete kVals;
    delete diagEntries;
}

/**
 * @todo write this comment
 * @todo check if this actually works... am I done??
 * @todo fix copy result thing (unnecessary, but how will we do this? check use
 * case and how we use it there.)
 * @warning IF DX EVER CHANGES, THIS WILL LIKELY BREAK AS alpha IS STATIC
 * roughly:
 * du is to be found (derivative)
 * u is calculated rhs
 * dx is spacing parameter (h)
 *
 * this to be implemented with existing routines from utils.h
 *
 * The steps to be performed here are as follows:
 *  we begin with Pb_ du = (1 / dx) Qb_ u
 *  compute Qb_ u -> b_
 *  banded solve  Pb_ du = b_
 */
// TODO: be sure to actually properly implement do_grad_x, y and z
void BandedCompactDerivs::do_grad_x(double *const du, const double *const u,
                                    const double dx, const unsigned int *sz,
                                    const unsigned int bflag) {
    // First compute matrix product of Q1b_ and u and
    //  store in grad_xVars->B using banded matrix vector multiply
    static double alpha   = 1 / dx;

    // we need to iterate over all z slices

    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];

    for (unsigned int k = 0; k < nz; k++) {
        const double *const u_slice = u + k * nx * ny;
        double *const du_slice      = du + k * nx * ny;

        // then slice up along y axis
        for (unsigned int j = 0; j < ny; j++) {
            // do the multiplication across the way
            double *const u_teeny       = (double *)u_slice + j * nx;
            double *const workspace_bit = workspace_ + j * nx;

            // multiply the vector
            bandedMatrixVectorMult(workspace_bit, Qb_, u_teeny, kVals->qkl,
                                   kVals->qku, alpha, p_n);
        }

        lapack::dgbsvx_cpp_safe(
            grad_xVars->FACT, grad_xVars->TRANS, static_cast<int>(nx),
            grad_xVars->KL, grad_xVars->KU, static_cast<int>(ny),
            grad_xVars->AB, grad_xVars->LDAB, grad_xVars->AFB,
            grad_xVars->LDAFB, grad_xVars->IPIV, grad_xVars->EQUED,
            grad_xVars->R, grad_xVars->C, workspace_, static_cast<int>(nx),
            du_slice, static_cast<int>(nx), grad_xVars->RCOND, grad_xVars->FERR,
            grad_xVars->BERR, grad_xVars->WORK, grad_xVars->IWORK,
            grad_xVars->INFO);
    }
}

void BandedCompactDerivs::do_grad_y(double *const du, const double *const u,
                                    const double dx, const unsigned int *sz,
                                    const unsigned int bflag) {
    static double alpha          = 1 / dx;

    // we need to iterate over all z slices

    const unsigned int nx        = sz[0];
    const unsigned int ny        = sz[1];
    const unsigned int nz        = sz[2];

    double *const temp_transpose = workspace_;
    double *const intermediate   = workspace_ + nx * ny;

    for (unsigned int k = 0; k < nz; k++) {
        const double *const u_slice = u + k * nx * ny;
        double *const du_slice      = du + k * nx * ny;

        // then slice up along x axis
        for (unsigned int i = 0; i < nx; i++) {
            // do the multiplication across the way

            // worksapce is 3d, so i'll use the first nx x ny chunk as storage,
            // then the second chunk for work
            for (unsigned int j = 0; j < ny; j++) {
                temp_transpose[j] = u_slice[i + j * nx];
            }

            double *const intermediate_line = intermediate + nx * i;

            // multiply the vector
            bandedMatrixVectorMult(intermediate_line, Qb_, temp_transpose,
                                   kVals->qkl, kVals->qku, alpha, p_n);
        }

        // printArray_2D(temp_transpose, nx, ny);

        // printArray_2D(intermediate, nx, ny);

        // do the solve
        lapack::dgbsvx_cpp_safe(
            grad_xVars->FACT, grad_xVars->TRANS, static_cast<int>(ny),
            grad_xVars->KL, grad_xVars->KU, static_cast<int>(nx),
            grad_xVars->AB, grad_xVars->LDAB, grad_xVars->AFB,
            grad_xVars->LDAFB, grad_xVars->IPIV, grad_xVars->EQUED,
            grad_xVars->R, grad_xVars->C, intermediate, static_cast<int>(ny),
            temp_transpose, static_cast<int>(ny), grad_xVars->RCOND,
            grad_xVars->FERR, grad_xVars->BERR, grad_xVars->WORK,
            grad_xVars->IWORK, grad_xVars->INFO);

        // then transpose back out
        for (unsigned int i = 0; i < nx; i++) {
            for (unsigned int j = 0; j < ny; j++) {
                du_slice[INDEX_N2D(i, j, nx)] = temp_transpose[j + i * ny];
            }
        }
    }
}

void BandedCompactDerivs::do_grad_z(double *const du, const double *const u,
                                    const double dx, const unsigned int *sz,
                                    const unsigned int bflag) {
    // First compute matrix product of Q1b_ and u and
    //  store in grad_xVars->B using banded matrix vector multiply
    static double alpha      = 1 / dx;

    // we need to iterate over all z slices

    const unsigned int nx    = sz[0];
    const unsigned int ny    = sz[1];
    const unsigned int nz    = sz[2];

    double *const transposed = workspace_;
    double *const workspace  = workspace_ + nz * nx;

    for (unsigned int j = 0; j < ny; j++) {
        // start by extracing out the slice
        for (unsigned int i = 0; i < nx; i++) {
            for (unsigned int k = 0; k < nz; k++) {
                transposed[i * nz + k] = u[INDEX_3D(i, j, k)];
            }
        }

        // printArray_2D(transposed, nz, nz);

        // now we basically do what we did for "x", but over x
        for (unsigned int i = 0; i < nx; i++) {
            // do the multiplication across the way
            double *const transposed_chunk = transposed + i * nz;
            double *const workspace_chunk  = workspace + i * nz;

            // multiply the vector
            bandedMatrixVectorMult(workspace_chunk, Qb_, transposed_chunk,
                                   kVals->qkl, kVals->qku, alpha, p_n);
        }

        // printArray_2D(workspace, nz, nz);

        // now we do the dgbsvx
        lapack::dgbsvx_cpp_safe(
            grad_xVars->FACT, grad_xVars->TRANS, static_cast<int>(nz),
            grad_xVars->KL, grad_xVars->KU, static_cast<int>(nx),
            grad_xVars->AB, grad_xVars->LDAB, grad_xVars->AFB,
            grad_xVars->LDAFB, grad_xVars->IPIV, grad_xVars->EQUED,
            grad_xVars->R, grad_xVars->C, workspace, static_cast<int>(nz),
            transposed, static_cast<int>(nz), grad_xVars->RCOND,
            grad_xVars->FERR, grad_xVars->BERR, grad_xVars->WORK,
            grad_xVars->IWORK, grad_xVars->INFO);

        // printArray_2D(transposed, nz, nz);

        // exit(0);

        // then slot it back int
        for (unsigned int i = 0; i < nx; i++) {
            for (unsigned int k = 0; k < nz; k++) {
                du[INDEX_3D(i, j, k)] = transposed[k + i * nz];
            }
        }
    }
}

}  // namespace dendroderivs
