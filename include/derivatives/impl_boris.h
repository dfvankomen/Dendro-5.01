#pragma once

#include "derivatives/derivs_compact.h"
#include "derivatives/derivs_matrixonly.h"
#include "derivatives/derivs_utils.h"

namespace dendroderivs {

MatrixDiagonalEntries* createBoris4thDiagonalsFirstOrder(
    unsigned int boundary_type, unsigned int pw);

MatrixDiagonalEntries* createBoris6thDiagonalsFirstOrder(
    unsigned int boundary_type, unsigned int pw);

MatrixDiagonalEntries* createBoris6thEtaDiagonalsFirstOrder(
    unsigned int boundary_type, unsigned int pw);

void inline boris_init_helper(
    std::unordered_map<unsigned int, std::unique_ptr<DerivMatrixStorage>>&
        D_storage_map,
    MatrixDiagonalEntries* diagEntries, MatrixDiagonalEntries* diagEntriesLeft,
    MatrixDiagonalEntries* diagEntriesRight,
    MatrixDiagonalEntries* diagEntriesLeftRight, unsigned int ele_order) {
    const unsigned pw = ele_order / 2;
    // create up to 5 fused blocks
    for (unsigned int i = 1; i <= 5; i++) {
        // calculate the size based on the element order
        const unsigned int n = (i + 1) * ele_order + 1;

        D_storage_map.emplace(
            n, createMatrixSystemForSingleSizeAllUniqueDiags<1>(
                   pw, n, diagEntries, diagEntriesLeft, diagEntriesRight,
                   diagEntriesLeftRight, i == 1));

#if 0
        if (i == 1) {
            // print out the matrix:
            std::cout << "NORMAL D MATRIX:" << std::endl;
            printArray_2D_transpose(D_storage_map[n]->D_original.data(), n, n);

            std::cout << std::endl << "LEFT D MATRIX:" << std::endl;
            printArray_2D_transpose(D_storage_map[n]->D_left.data(), n, n);

            std::cout << std::endl << "RIGHT D MATRIX:" << std::endl;
            printArray_2D_transpose(D_storage_map[n]->D_right.data(), n, n);
        }
#endif
    }
}

class Boris_BorisO4_FirstOrder : public MatrixCompactDerivs<1> {
    MatrixDiagonalEntries* diagEntriesLeft      = nullptr;
    MatrixDiagonalEntries* diagEntriesRight     = nullptr;
    MatrixDiagonalEntries* diagEntriesLeftRight = nullptr;

   public:
    template <typename... Args>
    Boris_BorisO4_FirstOrder(
        unsigned int ele_order,
        const std::vector<double>& coeffs_in = std::vector<double>(),
        const unsigned int matrixID          = 1, Args&&...)
        : MatrixCompactDerivs{ele_order} {
        // Matrix ID informs if we're using closure or not for this one
        // normal entries
        diagEntries      = createBoris4thDiagonalsFirstOrder(matrixID, p_pw);

        // ensure we build up the entries with DIRICHLET
        diagEntriesLeft  = createBoris4thDiagonalsFirstOrder(1, p_pw);
        diagEntriesRight = createBoris4thDiagonalsFirstOrder(1, p_pw);
        diagEntriesLeftRight = createBoris4thDiagonalsFirstOrder(1, p_pw);

        // now we can make sure that the Lower part of left is set to
        // non-closure
        diagEntriesLeft->PDiagBoundaryLower = diagEntries->PDiagBoundaryLower;
        diagEntriesLeft->QDiagBoundaryLower = diagEntries->QDiagBoundaryLower;

        // and the upper part for the Right
        diagEntriesRight->PDiagBoundary     = diagEntries->PDiagBoundary;
        diagEntriesRight->QDiagBoundary     = diagEntries->QDiagBoundary;

        boris_init_helper(D_storage_map_, diagEntries, diagEntriesLeft,
                          diagEntriesRight, diagEntriesLeftRight, p_ele_order);
    }

    ~Boris_BorisO4_FirstOrder() {}

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<Boris_BorisO4_FirstOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_BORISO4; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_FIRST_ORDER;
    }

    std::string toString() const override { return "Boris_BorisO4_FirstOrder"; }
};

class Boris_BorisO6_FirstOrder : public MatrixCompactDerivs<1> {
    MatrixDiagonalEntries* diagEntriesLeft      = nullptr;
    MatrixDiagonalEntries* diagEntriesRight     = nullptr;
    MatrixDiagonalEntries* diagEntriesLeftRight = nullptr;

   public:
    template <typename... Args>
    Boris_BorisO6_FirstOrder(
        unsigned int ele_order,
        const std::vector<double>& coeffs_in = std::vector<double>(),
        const unsigned int matrixID          = 1, Args&&...)
        : MatrixCompactDerivs{ele_order} {
        // Matrix ID informs if we're using closure or not for this one
        // normal entries
        diagEntries      = createBoris6thDiagonalsFirstOrder(matrixID, p_pw);

        // ensure we build up the entries with DIRICHLET
        diagEntriesLeft  = createBoris6thDiagonalsFirstOrder(1, p_pw);
        diagEntriesRight = createBoris6thDiagonalsFirstOrder(1, p_pw);
        diagEntriesLeftRight = createBoris6thDiagonalsFirstOrder(1, p_pw);

        // now we can make sure that the Lower part of left is set to
        // non-closure
        diagEntriesLeft->PDiagBoundaryLower = diagEntries->PDiagBoundaryLower;
        diagEntriesLeft->QDiagBoundaryLower = diagEntries->QDiagBoundaryLower;

        // and the upper part for the Right
        diagEntriesRight->PDiagBoundary     = diagEntries->PDiagBoundary;
        diagEntriesRight->QDiagBoundary     = diagEntries->QDiagBoundary;

        boris_init_helper(D_storage_map_, diagEntries, diagEntriesLeft,
                          diagEntriesRight, diagEntriesLeftRight, p_ele_order);
    }

    ~Boris_BorisO6_FirstOrder() {}

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<Boris_BorisO6_FirstOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_BORISO6; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_FIRST_ORDER;
    }

    std::string toString() const override { return "Boris_BorisO6_FirstOrder"; }
};

class Boris_BorisO6Eta_FirstOrder : public MatrixCompactDerivs<1> {
    MatrixDiagonalEntries* diagEntriesLeft      = nullptr;
    MatrixDiagonalEntries* diagEntriesRight     = nullptr;
    MatrixDiagonalEntries* diagEntriesLeftRight = nullptr;

   public:
    template <typename... Args>
    Boris_BorisO6Eta_FirstOrder(
        unsigned int ele_order,
        const std::vector<double>& coeffs_in = std::vector<double>(),
        const unsigned int matrixID          = 1, Args&&...)
        : MatrixCompactDerivs{ele_order} {
        // Matrix ID informs if we're using closure or not for this one
        // normal entries
        diagEntries      = createBoris6thEtaDiagonalsFirstOrder(matrixID, p_pw);

        // ensure we build up the entries with DIRICHLET
        diagEntriesLeft  = createBoris6thEtaDiagonalsFirstOrder(1, p_pw);
        diagEntriesRight = createBoris6thEtaDiagonalsFirstOrder(1, p_pw);
        diagEntriesLeftRight = createBoris6thEtaDiagonalsFirstOrder(1, p_pw);

        // now we can make sure that the Lower part of left is set to
        // non-closure
        diagEntriesLeft->PDiagBoundaryLower = diagEntries->PDiagBoundaryLower;
        diagEntriesLeft->QDiagBoundaryLower = diagEntries->QDiagBoundaryLower;

        // and the upper part for the Right
        diagEntriesRight->PDiagBoundary     = diagEntries->PDiagBoundary;
        diagEntriesRight->QDiagBoundary     = diagEntries->QDiagBoundary;

        boris_init_helper(D_storage_map_, diagEntries, diagEntriesLeft,
                          diagEntriesRight, diagEntriesLeftRight, p_ele_order);
    }

    ~Boris_BorisO6Eta_FirstOrder() {}

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<Boris_BorisO6Eta_FirstOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_BORISO6_ETA; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_FIRST_ORDER;
    }

    std::string toString() const override {
        return "Boris_BorisO6Eta_FirstOrder";
    }
};

}  // namespace dendroderivs
