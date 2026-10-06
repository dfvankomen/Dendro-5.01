#pragma once

#include <memory>

#include "derivatives/derivs_compact.h"
#include "derivatives/derivs_matrixonly.h"
#include "derivatives/derivs_utils.h"
#include "derivatives/impl_explicitmatrix.h"
#include "derivatives/impl_jonathantyler.h"
#include "derivatives/nova_derivs.h"

namespace dendroderivs {

class TestingHybridDerivatives_FirstOrder : public MatrixCompactDerivs<1> {
   private:
    MatrixDiagonalEntries* diagEntriesLeft      = nullptr;
    MatrixDiagonalEntries* diagEntriesRight     = nullptr;
    MatrixDiagonalEntries* diagEntriesLeftRight = nullptr;

   public:
    TestingHybridDerivatives_FirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs{ele_order, in_matrix_filter, in_filter_coeffs} {
        // idk, let's use JTT6 to start? then we'll override the boundaries
        diagEntries                         = createJTT6DiagonalsFirstOrder();

        // now we want to build up the left and right boundary diags, which
        // we'll overwrite
        diagEntriesLeft                     = createJTT6DiagonalsFirstOrder();
        diagEntriesRight                    = createJTT6DiagonalsFirstOrder();
        diagEntriesLeftRight                = createJTT6DiagonalsFirstOrder();

        // let's slot in E6's boundary terms'
        MatrixDiagonalEntries* temp_entries = createE6DiagonalsFirstOrder();

        auto nova_bdys_normal =
            nova::create_nova_boundaries<double>(10, 3, 9, 1);
        auto nova_bdys_smaller =
            nova::create_nova_boundaries<double>(10, 3, 9, 1);

        auto p_bdrys_replace                     = generate_identity_bdys(3);

        diagEntriesLeft->PDiagBoundaryLower      = p_bdrys_replace;
        diagEntriesLeft->QDiagBoundaryLower      = nova_bdys_smaller;

        diagEntriesRight->PDiagBoundary          = p_bdrys_replace;
        diagEntriesRight->QDiagBoundary          = nova_bdys_smaller;

        diagEntriesLeftRight->PDiagBoundary      = p_bdrys_replace;
        diagEntriesLeftRight->QDiagBoundary      = nova_bdys_smaller;
        diagEntriesLeftRight->PDiagBoundaryLower = p_bdrys_replace;
        diagEntriesLeftRight->QDiagBoundaryLower = nova_bdys_smaller;

        // TEMP:

        // modify the "main" boundaries
        diagEntries->PDiagBoundary               = p_bdrys_replace;
        diagEntries->QDiagBoundary               = nova_bdys_normal;
        diagEntries->PDiagBoundaryLower          = p_bdrys_replace;
        diagEntries->QDiagBoundaryLower          = nova_bdys_normal;

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
        for (unsigned int i = 1; i <= 5; i++) {
            // calculate the size based on the element order
            const unsigned int n = (i + 1) * p_ele_order + 1;

            D_storage_map_.emplace(
                n, createMatrixSystemForSingleSizeAllUniqueDiags<1>(
                       p_pw, n, diagEntries, diagEntriesLeft, diagEntriesRight,
                       diagEntriesLeftRight, i == 1));

#if 0
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
#endif
        }
    }
};

}  // namespace dendroderivs
