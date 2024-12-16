#pragma once

#include <memory>

#include "derivatives/derivs_compact.h"
#include "derivatives/derivs_matrixonly.h"
#include "derivatives/impl_explicitmatrix.h"
#include "derivatives/impl_jonathantyler.h"

namespace dendroderivs {

class TestingHybridDerivatives_FirstOrder : public MatrixCompactDerivs<1> {
   private:
    MatrixDiagonalEntries* diagEntriesLeft      = nullptr;
    MatrixDiagonalEntries* diagEntriesRight     = nullptr;
    MatrixDiagonalEntries* diagEntriesLeftRight = nullptr;

   public:
    template <typename... Args>
    TestingHybridDerivatives_FirstOrder(unsigned int ele_order, Args&&...)
        : MatrixCompactDerivs{ele_order} {
        // idk, let's use JTT6 to start? then we'll override the boundaries
        diagEntries                         = createJTT6DiagonalsFirstOrder();

        // now we want to build up the left and right boundary diags, which
        // we'll overwrite
        diagEntriesLeft                     = createJTT6DiagonalsFirstOrder();
        diagEntriesRight                    = createJTT6DiagonalsFirstOrder();
        diagEntriesLeftRight                = createJTT6DiagonalsFirstOrder();

        // let's slot in E6's boundary terms'
        MatrixDiagonalEntries* temp_entries = createE6DiagonalsFirstOrder();

        diagEntriesLeft->PDiagBoundaryLower = temp_entries->PDiagBoundaryLower;
        diagEntriesLeft->QDiagBoundaryLower = temp_entries->QDiagBoundaryLower;

        diagEntriesRight->PDiagBoundary     = temp_entries->PDiagBoundary;
        diagEntriesRight->QDiagBoundary     = temp_entries->QDiagBoundary;

        diagEntriesLeftRight->PDiagBoundary = temp_entries->PDiagBoundary;
        diagEntriesLeftRight->QDiagBoundary = temp_entries->QDiagBoundary;
        diagEntriesLeftRight->PDiagBoundaryLower =
            temp_entries->PDiagBoundaryLower;
        diagEntriesLeftRight->QDiagBoundaryLower =
            temp_entries->QDiagBoundaryLower;

        // TEMP:
        diagEntriesLeft->PDiagBoundary       = temp_entries->PDiagBoundary;
        diagEntriesLeft->QDiagBoundary       = temp_entries->QDiagBoundary;
        diagEntriesLeft->PDiagBoundaryLower  = temp_entries->PDiagBoundaryLower;
        diagEntriesLeft->QDiagBoundaryLower  = temp_entries->QDiagBoundaryLower;
        diagEntriesRight->PDiagBoundary      = temp_entries->PDiagBoundary;
        diagEntriesRight->QDiagBoundary      = temp_entries->QDiagBoundary;
        diagEntriesRight->PDiagBoundaryLower = temp_entries->PDiagBoundaryLower;
        diagEntriesRight->QDiagBoundaryLower = temp_entries->QDiagBoundaryLower;
        diagEntries->PDiagBoundary           = temp_entries->PDiagBoundary;
        diagEntries->QDiagBoundary           = temp_entries->QDiagBoundary;
        diagEntries->PDiagBoundaryLower      = temp_entries->PDiagBoundaryLower;
        diagEntries->QDiagBoundaryLower      = temp_entries->QDiagBoundaryLower;

        delete temp_entries;

        init_true();
    }

    ~TestingHybridDerivatives_FirstOrder() {
        delete diagEntriesLeft;
        delete diagEntriesRight;
        delete diagEntriesLeftRight;
    }

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<TestingHybridDerivatives_FirstOrder>(*this);
    }

    // TODO: update this
    DerivType getDerivType() const override { return DerivType::D_E6; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_FIRST_ORDER;
    }

    std::string toString() const override {
        return "TestingHybridDerivatives_FirstOrder";
    }

    void init_true() {
        std::cout << "INSIDE MY INIT" << std::endl;
        for (unsigned int i = 1; i <= 5; i++) {
            // calculate the size based on the element order
            const unsigned int n = (i + 1) * p_ele_order + 1;

            D_storage_map_.emplace(
                n, createMatrixSystemForSingleSizeAllUniqueDiags<1>(
                       p_pw, n, diagEntries, diagEntriesLeft, diagEntriesRight,
                       diagEntriesLeftRight, i == 1));

            if (i == 1) {
                // print out the matrix:
                std::cout << "NORMAL D MATRIX:" << std::endl;
                printArray_2D_transpose(D_storage_map_[n]->D_original.data(), n,
                                        n);

                std::cout << std::endl << "LEFT D MATRIX:" << std::endl;
                printArray_2D_transpose(D_storage_map_[n]->D_left.data(), n, n);

                std::cout << std::endl << "RIGHT D MATRIX:" << std::endl;
                printArray_2D_transpose(D_storage_map_[n]->D_right.data(), n,
                                        n);
            }
        }
        std::cout << "FINISHED MY INIT" << std::endl;
    }
};

}  // namespace dendroderivs
