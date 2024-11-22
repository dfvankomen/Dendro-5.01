#pragma once

#include "derivatives/derivs_compact.h"
#include "derivatives/derivs_matrixonly.h"

namespace dendroderivs {

MatrixDiagonalEntries* createE4DiagonalsFirstOrder();
MatrixDiagonalEntries* createE4DiagonalsSecondOrder();

class ExplicitDerivsO4_Matrix_FirstOrder : public MatrixCompactDerivs<1> {
   public:
    template <typename... Args>
    ExplicitDerivsO4_Matrix_FirstOrder(unsigned int n, unsigned int pw,
                                       Args&&...)
        : MatrixCompactDerivs{n, pw} {
        MatrixDiagonalEntries* diagEntries = createE4DiagonalsFirstOrder();

        P_ = create_P_from_diagonals(*diagEntries, n, 1.0);
        Q_ = create_Q_from_diagonals(*diagEntries, n, -1.0);

        this->init();

        // don't need the diagonal entries anymore
        delete diagEntries;
    }

    ~ExplicitDerivsO4_Matrix_FirstOrder() {}

    DerivType getDerivType() const override { return DerivType::D_E4; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_FIRST_ORDER;
    }

    std::string toString() const override {
        return "ExplicitDerivsO4_Matrix_FirstOrder";
    }
};

class ExplicitDerivsO4_Matrix_SecondOrder : public MatrixCompactDerivs<2> {
   public:
    template <typename... Args>
    ExplicitDerivsO4_Matrix_SecondOrder(unsigned int n, unsigned int pw,
                                        Args&&...)
        : MatrixCompactDerivs{n, pw} {
        MatrixDiagonalEntries* diagEntries = createE4DiagonalsSecondOrder();

        P_ = create_P_from_diagonals(*diagEntries, n, 1.0);
        Q_ = create_Q_from_diagonals(*diagEntries, n, 1.0);

        this->init();

        // don't need the diagonal entries anymore
        delete diagEntries;
    }
    ~ExplicitDerivsO4_Matrix_SecondOrder() {}

    DerivType getDerivType() const override { return DerivType::D_E4; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_SECOND_ORDER;
    }

    std::string toString() const override {
        return "ExplicitDerivsO4_Matrix_SecondOrder";
    }
};

}  // namespace dendroderivs
