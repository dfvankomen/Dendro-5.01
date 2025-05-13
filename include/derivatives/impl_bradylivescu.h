#pragma once

#include "derivatives/derivs_banded.h"
#include "derivatives/derivs_compact.h"
#include "derivatives/derivs_matrixonly.h"
#include "derivatives/derivs_utils.h"

namespace dendroderivs {

MatrixDiagonalEntries *createBL6thDiagonalsFirstOrder(unsigned int matrixID);
void fill_alpha_beta_bl_6th(std::vector<std::vector<double>> &alpha,
                            std::vector<std::vector<double>> &beta,
                            unsigned int matrixID);

class BradyLivescu_BL6_FirstOrder : public MatrixCompactDerivs<1> {
   public:
    BradyLivescu_BL6_FirstOrder(
        unsigned int ele_order, const std::string &in_matrix_filter = "none",
        const std::vector<double> &in_filter_coeffs = std::vector<double>(),
        const std::vector<double> &coeffs_in        = std::vector<double>(),
        const unsigned int matrixID                 = 1)
        : MatrixCompactDerivs{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createBL6thDiagonalsFirstOrder(matrixID);

        this->init();
    }

    ~BradyLivescu_BL6_FirstOrder() {}

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<BradyLivescu_BL6_FirstOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_BL6; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_FIRST_ORDER;
    }

    std::string toString() const override {
        return "BradyLivescu_BLP6_FirstOrder";
    }
};

}  // namespace dendroderivs
