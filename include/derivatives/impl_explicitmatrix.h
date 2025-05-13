#pragma once

#include "derivatives/derivs_compact.h"
#include "derivatives/derivs_matrixonly.h"

namespace dendroderivs {

MatrixDiagonalEntries* createE4DiagonalsFirstOrder();
MatrixDiagonalEntries* createE4DiagonalsSecondOrder();
MatrixDiagonalEntries* createE6DiagonalsFirstOrder();
MatrixDiagonalEntries* createE6DiagonalsSecondOrder();
MatrixDiagonalEntries* createE8DiagonalsFirstOrder();
MatrixDiagonalEntries* createE8DiagonalsSecondOrder();

class ExplicitDerivsO4_Matrix_FirstOrder : public MatrixCompactDerivs<1> {
   public:
    ExplicitDerivsO4_Matrix_FirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createE4DiagonalsFirstOrder();

        this->init();
    }

    ~ExplicitDerivsO4_Matrix_FirstOrder() {}

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<ExplicitDerivsO4_Matrix_FirstOrder>(*this);
    }

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
    ExplicitDerivsO4_Matrix_SecondOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createE4DiagonalsSecondOrder();

        this->init();
    }
    ~ExplicitDerivsO4_Matrix_SecondOrder() {}

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<ExplicitDerivsO4_Matrix_SecondOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_E4; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_SECOND_ORDER;
    }

    std::string toString() const override {
        return "ExplicitDerivsO4_Matrix_SecondOrder";
    }
};

class ExplicitDerivsO6_Matrix_FirstOrder : public MatrixCompactDerivs<1> {
   public:
    ExplicitDerivsO6_Matrix_FirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createE6DiagonalsFirstOrder();

        this->init();
    }

    ~ExplicitDerivsO6_Matrix_FirstOrder() {}

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<ExplicitDerivsO6_Matrix_FirstOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_E6; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_FIRST_ORDER;
    }

    std::string toString() const override {
        return "ExplicitDerivsO6_Matrix_FirstOrder";
    }
};

class ExplicitDerivsO6_Matrix_SecondOrder : public MatrixCompactDerivs<2> {
   public:
    ExplicitDerivsO6_Matrix_SecondOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createE6DiagonalsSecondOrder();

        this->init();
    }
    ~ExplicitDerivsO6_Matrix_SecondOrder() {}

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<ExplicitDerivsO6_Matrix_SecondOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_E6; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_SECOND_ORDER;
    }

    std::string toString() const override {
        return "ExplicitDerivsO6_Matrix_SecondOrder";
    }
};

class ExplicitDerivsO8_Matrix_FirstOrder : public MatrixCompactDerivs<1> {
   public:
    ExplicitDerivsO8_Matrix_FirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createE8DiagonalsFirstOrder();

        this->init();
    }

    ~ExplicitDerivsO8_Matrix_FirstOrder() {}

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<ExplicitDerivsO8_Matrix_FirstOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_E8; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_FIRST_ORDER;
    }

    std::string toString() const override {
        return "ExplicitDerivsO8_Matrix_FirstOrder";
    }
};

class ExplicitDerivsO8_Matrix_SecondOrder : public MatrixCompactDerivs<2> {
   public:
    ExplicitDerivsO8_Matrix_SecondOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createE8DiagonalsSecondOrder();

        this->init();
    }
    ~ExplicitDerivsO8_Matrix_SecondOrder() {}

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<ExplicitDerivsO8_Matrix_SecondOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_E8; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_SECOND_ORDER;
    }

    std::string toString() const override {
        return "ExplicitDerivsO8_Matrix_SecondOrder";
    }
};

}  // namespace dendroderivs
