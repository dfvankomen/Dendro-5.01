#pragma once

#include "derivatives/derivs_banded.h"
#include "derivatives/derivs_matrixonly.h"

// TODO: remove this include
#include "refel.h"

namespace dendroderivs {

MatrixDiagonalEntries* createKimDiagonals();

/** 4th order tridiagonal Kim Derivative, First Order, Pure Matrix System */
class KimBoundO4_FirstOrder : public MatrixCompactDerivs<1> {
   public:
    template <typename... Args>
    KimBoundO4_FirstOrder(unsigned int n, unsigned int pw, Args&&...)
        : MatrixCompactDerivs<1>{n, pw} {
        MatrixDiagonalEntries* diagEntries = createKimDiagonals();

        P_ = create_P_from_diagonals(*diagEntries, n);
        Q_ = create_Q_from_diagonals(*diagEntries, n);

        this->init();

        // don't need the diagonal entries anymore
        delete diagEntries;
    }
    ~KimBoundO4_FirstOrder() {
#ifdef DEBUG
        std::cout << "in BandedCompactDerivs_KimBound4 deconstructor"
                  << std::endl;
#endif
    };

    DerivType getDerivType() const override { return DerivType::D_K4; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_FIRST_ORDER;
    }

    std::string toString() const override { return "KimBoundO4_FirstOrder"; };
};

/** 4th order tridiagonal Kim Derivative, First Order, Banded System */
class KimBoundO4_FirstOrder_Banded : public BandedCompactDerivs {
   public:
    template <typename... Args>
    KimBoundO4_FirstOrder_Banded(unsigned int n, unsigned int pw, Args&&...)
        : BandedCompactDerivs{n, pw} {
        kVals = new BandedMatrixDiagonalWidths{
            2,  // p1kl
            2,  // p1ku
            7,  // q1kl
            7,  // q1ku
        };

        diagEntries = createKimDiagonals();

        // build the matrices, allocate the data, etc
        this->init(kVals, diagEntries);

        std::vector<double> tempP =
            create_P_from_diagonals(*diagEntries, n, -1.0);

        printArray_2D_transpose(tempP.data(), n, n);

        std::vector<double> tempQ =
            create_Q_from_diagonals(*diagEntries, n, -1.0);

        printArray_2D_transpose(tempQ.data(), n, n);
    }
    ~KimBoundO4_FirstOrder_Banded() {
#ifdef DEBUG
        std::cout << "in BandedCompactDerivs_KimBound4 deconstructor"
                  << std::endl;
#endif
    };

    DerivType getDerivType() const override { return DerivType::D_K4; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_FIRST_ORDER;
    }

    std::string toString() const override {
        return "KimBoundO4_FirstOrder_Banded";
    };
};

/** 4th order tridiagonal Kim Derivative, First Order */
class KimBoundO4_SecondOrder : public BandedCompactDerivs {
   public:
    template <typename... Args>
    KimBoundO4_SecondOrder(unsigned int n, unsigned int pw)
        : BandedCompactDerivs{n, pw} {
        kVals = new BandedMatrixDiagonalWidths{
            1,  // p2kl -- no 2nd deriv
            1,  // p2ku -- FIXME no 2nd deriv
            1,  // q2kl -- FIXME no 2nd deriv
            1   // q2ku -- FIXME no 2nd deriv
        };
    }
    ~KimBoundO4_SecondOrder() {
#ifdef DEBUG
        std::cout << "in BandedCompactDerivs_KimBound4 deconstructor"
                  << std::endl;
#endif
    };

    DerivType getDerivType() const override { return DerivType::D_K4; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_SECOND_ORDER;
    }

    std::string toString() const override { return "KimBoundO4_FirstOrder"; };
};

}  // namespace dendroderivs
