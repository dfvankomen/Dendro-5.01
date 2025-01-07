
#pragma once

#include <cmath>
#include <iostream>

#include "derivatives/derivs_compact.h"
#include "derivatives/derivs_utils.h"

namespace dendroderivs {

/*
 * Contains declarations for CFD schemes which will be solved
 * using banded solver routines from LAPACK.
 */
class BandedCompactDerivs : public CompactDerivs {
   public:
    // data used to call banded matrix vector solver
    BandedMatrixSolveVars *grad_xVars = nullptr;

   protected:
    // all KL and KU values for each P and Q matrix
    BandedMatrixDiagonalWidths *kVals = nullptr;

    // banded storage matrices
    double *Pb_                       = nullptr;  // P matrix (banded storage)
    double *Qb_                       = nullptr;  // Q matrix (banded storage)

    double *workspace_                = nullptr;

   protected:
    // YOU MUST CALL THIS IN YOUR IMPLEMENTATION
    void init(BandedMatrixDiagonalWidths *kVals,
              MatrixDiagonalEntries *entries);

   public:
    // this constructor will NOT allocate needed information!
    // when you create your own BandedCompactDerivs derived
    //  class implementation, you MUST call the method init()!!!!
    BandedCompactDerivs(unsigned int ele_order) : CompactDerivs{ele_order} {};
    /**
     * we implement a copy constructor just to print when it's called;
     * we would like to avoid accidental shallow copies
     */
    BandedCompactDerivs(const BandedCompactDerivs &obj) : CompactDerivs(obj) {
#ifdef DEBUG
        std::cout << "[copy constructor for BandedCompactDerivs was called!\n"
                  << "this is a mistake as there is no implementation]"
                  << std::endl;
#endif
    };

    // reminder, this will always be called as a child is destructing
    virtual ~BandedCompactDerivs();

    /**
     * @brief Pure virtual function to calculate the derivative.
     * @param du Pointer to where calculated derivative will be stored.
     * @param u Pointer to the input array.
     * @param dx The step size or grid spacing.
     * @param sz The number of points expected to be calculated in each dim
     * @param bflag The boundary flag
     */
    void do_grad_x(double *const du, const double *const u, const double dx,
                   const unsigned int *sz, const unsigned int bflag) override;
    void do_grad_y(double *const du, const double *const u, const double dx,
                   const unsigned int *sz, const unsigned int bflag) override;
    void do_grad_z(double *const du, const double *const u, const double dx,
                   const unsigned int *sz, const unsigned int bflag) override;

    inline double *getPBanded() const { return Pb_; }
    inline double *getQBanded() const { return Qb_; }
    inline double *getPScaleBand() const { return grad_xVars->AFB; }

    void set_maximum_block_size(size_t block_size) override {};
};
}  // namespace dendroderivs
