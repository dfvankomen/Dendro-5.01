#pragma once

#include <memory>

#include "derivatives/derivs_compact.h"

namespace dendroderivs {

enum InMatFilterType {
    IMFT_NONE = 0,
    IMFT_BYUT6,
    IMFT_BYUT8,
    IMFT_KIM,
    IMFT_KIM_1_P6,
    IMFT_KIM_2_P6,
    IMFT_KIM_3_P6,
    IMFT_KIM_4_P6,
    IMFT_KIM_P6
};

// all we really need to do is "inject" into the matrix this stuff, depending on
// if our filter type is set to not-none
class InMatrixFilter {
   protected:
    MatrixDiagonalEntries* diagEntries = nullptr;

   public:
    InMatrixFilter(const std::vector<double>& input_coeffs) {}

    ~InMatrixFilter() {
        // make sure diagEntries is deleted!
        delete diagEntries;
    }

    MatrixDiagonalEntries* get_diag_entries() const { return diagEntries; }

    virtual InMatFilterType get_filter_type() const = 0;
};

class NoneFilter_InMatrix : public InMatrixFilter {
   public:
    NoneFilter_InMatrix(const std::vector<double>& input_coeffs)
        : InMatrixFilter(input_coeffs) {}

    ~NoneFilter_InMatrix() = default;

    InMatFilterType get_filter_type() const override {
        return InMatFilterType::IMFT_NONE;
    }
};

}  // namespace dendroderivs
