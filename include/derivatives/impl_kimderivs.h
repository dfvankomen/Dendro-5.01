#pragma once

#include "derivatives/derivs_banded.h"
#include "derivatives/derivs_matrixonly.h"

// TODO: remove this include
#include "refel.h"

namespace dendroderivs {

MatrixDiagonalEntries* createKimDiagonals();
MatrixDiagonalEntries* createKim07Diagonals();
MatrixDiagonalEntries* createKim16Diagonals();
MatrixDiagonalEntries* createBYU_KIM_1_Diagonals();
MatrixDiagonalEntries* createBYU_KIM_2_Diagonals();
MatrixDiagonalEntries* createBYU_KIM_3_Diagonals();
MatrixDiagonalEntries* createBYU_KIM_4_Diagonals();
MatrixDiagonalEntries* createBYU_KIM_5_Diagonals();
MatrixDiagonalEntries* createBYU_KIM_6_Diagonals();
MatrixDiagonalEntries* createBYU_KIM_7_Diagonals();
MatrixDiagonalEntries* createBYU_KIM_8_Diagonals();
MatrixDiagonalEntries* createBYU_KIM_9_Diagonals();
MatrixDiagonalEntries* createBYU_KIM_10_Diagonals();
MatrixDiagonalEntries* createBYU_KIM_11_Diagonals();
MatrixDiagonalEntries* createBYU_KIM_12_Diagonals();
MatrixDiagonalEntries* createBYU_KIM_13_Diagonals();
MatrixDiagonalEntries* createBYU_KIM_14_Diagonals();
MatrixDiagonalEntries* createBYU_KIM_15_Diagonals();
MatrixDiagonalEntries* createBYU_KIM_16_Diagonals();
MatrixDiagonalEntries* createBYU_KIM_17_Diagonals();
MatrixDiagonalEntries* createBYU_KIM_18_Diagonals();
MatrixDiagonalEntries* createBYU_KIM_19_Diagonals();



/** 4th order tridiagonal Kim Derivative, First Order, Pure Matrix System */
class KimBoundO4_FirstOrder : public MatrixCompactDerivs<1> {
   public:
    KimBoundO4_FirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs<1>{ele_order, in_matrix_filter,
                                 in_filter_coeffs} {
        diagEntries = createKimDiagonals();

        this->init();
    }
    ~KimBoundO4_FirstOrder() {
#ifdef DEBUG
        std::cout << "in BandedCompactDerivs_KimBound4 deconstructor"
                  << std::endl;
#endif
    };

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<KimBoundO4_FirstOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_K4; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_FIRST_ORDER;
    }

    std::string toString() const override { return "KimBoundO4_FirstOrder"; };
};



/** 4th order tridiagonal Kim Derivative, First Order, Pure Matrix System */
class Kim_07_BoundO4_FirstOrder : public MatrixCompactDerivs<1> {
   public:
    Kim_07_BoundO4_FirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs<1>{ele_order, in_matrix_filter,
                                 in_filter_coeffs} {
        diagEntries = createKim07Diagonals();

        this->init();
    }
    ~Kim_07_BoundO4_FirstOrder() {
#ifdef DEBUG
        std::cout << "in BandedCompactDerivs_KimBound4 deconstructor"
                  << std::endl;
#endif
    };

    std::unique_ptr<Derivs> clone() const override {
    return std::make_unique<Kim_07_BoundO4_FirstOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_K4_07; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_FIRST_ORDER;
    }

    std::string toString() const override { return "Kim_07BoundO4_FirstOrder"; };
};

/** 4th order tridiagonal Kim Derivative, First Order, Pure Matrix System */
class BYU_KIM_1_DiagonalsFirstOrder : public MatrixCompactDerivs<1> {
public:
    BYU_KIM_1_DiagonalsFirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs<1>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createBYU_KIM_1_Diagonals();
        this->init();
    }
    ~BYU_KIM_1_DiagonalsFirstOrder() {
#ifdef DEBUG
        std::cout << "in BYU_KIM_1_DiagonalsFirstOrder destructor" << std::endl;
#endif
    }

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<BYU_KIM_1_DiagonalsFirstOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_KIMBYU_1; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_FIRST_ORDER; }
    std::string toString() const override { return "BYU_KIM_1_DiagonalsFirstOrder"; }
};

/** 4th order tridiagonal Kim Derivative, First Order, Pure Matrix System */
class BYU_KIM_2_DiagonalsFirstOrder : public MatrixCompactDerivs<1> {
public:
    BYU_KIM_2_DiagonalsFirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs<1>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createBYU_KIM_2_Diagonals();
        this->init();
    }
    ~BYU_KIM_2_DiagonalsFirstOrder() {
#ifdef DEBUG
        std::cout << "in BYU_KIM_2_DiagonalsFirstOrder destructor" << std::endl;
#endif
    }

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<BYU_KIM_2_DiagonalsFirstOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_KIMBYU_2; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_FIRST_ORDER; }
    std::string toString() const override { return "BYU_KIM_2_DiagonalsFirstOrder"; }
};

/** 4th order tridiagonal Kim Derivative, First Order, Pure Matrix System */
class BYU_KIM_3_DiagonalsFirstOrder : public MatrixCompactDerivs<1> {
public:
    BYU_KIM_3_DiagonalsFirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs<1>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createBYU_KIM_3_Diagonals();
        this->init();
    }
    ~BYU_KIM_3_DiagonalsFirstOrder() {
#ifdef DEBUG
        std::cout << "in BYU_KIM_3_DiagonalsFirstOrder destructor" << std::endl;
#endif
    }

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<BYU_KIM_3_DiagonalsFirstOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_KIMBYU_3; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_FIRST_ORDER; }
    std::string toString() const override { return "BYU_KIM_3_DiagonalsFirstOrder"; }
};

/** 4th order tridiagonal Kim Derivative, First Order, Pure Matrix System */
class BYU_KIM_4_DiagonalsFirstOrder : public MatrixCompactDerivs<1> {
public:
    BYU_KIM_4_DiagonalsFirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs<1>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createBYU_KIM_4_Diagonals();
        this->init();
    }
    ~BYU_KIM_4_DiagonalsFirstOrder() {
#ifdef DEBUG
        std::cout << "in BYU_KIM_4_DiagonalsFirstOrder destructor" << std::endl;
#endif
    }

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<BYU_KIM_4_DiagonalsFirstOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_KIMBYU_4; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_FIRST_ORDER; }
    std::string toString() const override { return "BYU_KIM_4_DiagonalsFirstOrder"; }
};

/** 4th order tridiagonal Kim Derivative, First Order, Pure Matrix System */
class BYU_KIM_5_DiagonalsFirstOrder : public MatrixCompactDerivs<1> {
public:
    BYU_KIM_5_DiagonalsFirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs<1>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createBYU_KIM_5_Diagonals();
        this->init();
    }
    ~BYU_KIM_5_DiagonalsFirstOrder() {
#ifdef DEBUG
        std::cout << "in BYU_KIM_5_DiagonalsFirstOrder destructor" << std::endl;
#endif
    }

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<BYU_KIM_5_DiagonalsFirstOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_KIMBYU_5; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_FIRST_ORDER; }
    std::string toString() const override { return "BYU_KIM_5_DiagonalsFirstOrder"; }
};

/** 4th order tridiagonal Kim Derivative, First Order, Pure Matrix System */
class BYU_KIM_6_DiagonalsFirstOrder : public MatrixCompactDerivs<1> {
public:
    BYU_KIM_6_DiagonalsFirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs<1>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createBYU_KIM_6_Diagonals();
        this->init();
    }
    ~BYU_KIM_6_DiagonalsFirstOrder() {
#ifdef DEBUG
        std::cout << "in BYU_KIM_6_DiagonalsFirstOrder destructor" << std::endl;
#endif
    }

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<BYU_KIM_6_DiagonalsFirstOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_KIMBYU_6; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_FIRST_ORDER; }
    std::string toString() const override { return "BYU_KIM_6_DiagonalsFirstOrder"; }
};

/** 4th order tridiagonal Kim Derivative, First Order, Pure Matrix System */
class BYU_KIM_7_DiagonalsFirstOrder : public MatrixCompactDerivs<1> {
public:
    BYU_KIM_7_DiagonalsFirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs<1>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createBYU_KIM_7_Diagonals();
        this->init();
    }
    ~BYU_KIM_7_DiagonalsFirstOrder() {
#ifdef DEBUG
        std::cout << "in BYU_KIM_7_DiagonalsFirstOrder destructor" << std::endl;
#endif
    }

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<BYU_KIM_7_DiagonalsFirstOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_KIMBYU_7; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_FIRST_ORDER; }
    std::string toString() const override { return "BYU_KIM_7_DiagonalsFirstOrder"; }
};

/** 4th order tridiagonal Kim Derivative, First Order, Pure Matrix System */
class BYU_KIM_8_DiagonalsFirstOrder : public MatrixCompactDerivs<1> {
public:
    BYU_KIM_8_DiagonalsFirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs<1>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createBYU_KIM_8_Diagonals();
        this->init();
    }
    ~BYU_KIM_8_DiagonalsFirstOrder() {
#ifdef DEBUG
        std::cout << "in BYU_KIM_8_DiagonalsFirstOrder destructor" << std::endl;
#endif
    }

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<BYU_KIM_8_DiagonalsFirstOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_KIMBYU_8; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_FIRST_ORDER; }
    std::string toString() const override { return "BYU_KIM_8_DiagonalsFirstOrder"; }
};

/** 4th order tridiagonal Kim Derivative, First Order, Pure Matrix System */
class BYU_KIM_9_DiagonalsFirstOrder : public MatrixCompactDerivs<1> {
public:
    BYU_KIM_9_DiagonalsFirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs<1>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createBYU_KIM_9_Diagonals();
        this->init();
    }
    ~BYU_KIM_9_DiagonalsFirstOrder() {
#ifdef DEBUG
        std::cout << "in BYU_KIM_9_DiagonalsFirstOrder destructor" << std::endl;
#endif
    }

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<BYU_KIM_9_DiagonalsFirstOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_KIMBYU_9; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_FIRST_ORDER; }
    std::string toString() const override { return "BYU_KIM_9_DiagonalsFirstOrder"; }
};

/** 4th order tridiagonal Kim Derivative, First Order, Pure Matrix System */
class BYU_KIM_10_DiagonalsFirstOrder : public MatrixCompactDerivs<1> {
public:
    BYU_KIM_10_DiagonalsFirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs<1>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createBYU_KIM_10_Diagonals();
        this->init();
    }
    ~BYU_KIM_10_DiagonalsFirstOrder() {
#ifdef DEBUG
        std::cout << "in BYU_KIM_10_DiagonalsFirstOrder destructor" << std::endl;
#endif
    }

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<BYU_KIM_10_DiagonalsFirstOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_KIMBYU_10; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_FIRST_ORDER; }
    std::string toString() const override { return "BYU_KIM_10_DiagonalsFirstOrder"; }
};

/** 4th order tridiagonal Kim Derivative, First Order, Pure Matrix System */
class BYU_KIM_11_DiagonalsFirstOrder : public MatrixCompactDerivs<1> {
public:
    BYU_KIM_11_DiagonalsFirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs<1>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createBYU_KIM_11_Diagonals();
        this->init();
    }
    ~BYU_KIM_11_DiagonalsFirstOrder() {
#ifdef DEBUG
        std::cout << "in BYU_KIM_11_DiagonalsFirstOrder destructor" << std::endl;
#endif
    }

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<BYU_KIM_11_DiagonalsFirstOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_KIMBYU_11; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_FIRST_ORDER; }
    std::string toString() const override { return "BYU_KIM_11_DiagonalsFirstOrder"; }
};

/** 4th order tridiagonal Kim Derivative, First Order, Pure Matrix System */
class BYU_KIM_12_DiagonalsFirstOrder : public MatrixCompactDerivs<1> {
public:
    BYU_KIM_12_DiagonalsFirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs<1>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createBYU_KIM_12_Diagonals();
        this->init();
    }
    ~BYU_KIM_12_DiagonalsFirstOrder() {
#ifdef DEBUG
        std::cout << "in BYU_KIM_12_DiagonalsFirstOrder destructor" << std::endl;
#endif
    }

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<BYU_KIM_12_DiagonalsFirstOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_KIMBYU_12; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_FIRST_ORDER; }
    std::string toString() const override { return "BYU_KIM_12_DiagonalsFirstOrder"; }
};

/** 4th order tridiagonal Kim Derivative, First Order, Pure Matrix System */
class BYU_KIM_13_DiagonalsFirstOrder : public MatrixCompactDerivs<1> {
public:
    BYU_KIM_13_DiagonalsFirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs<1>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createBYU_KIM_13_Diagonals();
        this->init();
    }
    ~BYU_KIM_13_DiagonalsFirstOrder() {
#ifdef DEBUG
        std::cout << "in BYU_KIM_13_DiagonalsFirstOrder destructor" << std::endl;
#endif
    }

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<BYU_KIM_13_DiagonalsFirstOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_KIMBYU_13; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_FIRST_ORDER; }
    std::string toString() const override { return "BYU_KIM_13_DiagonalsFirstOrder"; }
};

/** 4th order tridiagonal Kim Derivative, First Order, Pure Matrix System */
class BYU_KIM_14_DiagonalsFirstOrder : public MatrixCompactDerivs<1> {
public:
    BYU_KIM_14_DiagonalsFirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs<1>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createBYU_KIM_14_Diagonals();
        this->init();
    }
    ~BYU_KIM_14_DiagonalsFirstOrder() {
#ifdef DEBUG
        std::cout << "in BYU_KIM_14_DiagonalsFirstOrder destructor" << std::endl;
#endif
    }

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<BYU_KIM_14_DiagonalsFirstOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_KIMBYU_14; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_FIRST_ORDER; }
    std::string toString() const override { return "BYU_KIM_14_DiagonalsFirstOrder"; }
};

/** 4th order tridiagonal Kim Derivative, First Order, Pure Matrix System */
class BYU_KIM_15_DiagonalsFirstOrder : public MatrixCompactDerivs<1> {
public:
    BYU_KIM_15_DiagonalsFirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs<1>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createBYU_KIM_15_Diagonals();
        this->init();
    }
    ~BYU_KIM_15_DiagonalsFirstOrder() {
#ifdef DEBUG
        std::cout << "in BYU_KIM_15_DiagonalsFirstOrder destructor" << std::endl;
#endif
    }

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<BYU_KIM_15_DiagonalsFirstOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_KIMBYU_15; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_FIRST_ORDER; }
    std::string toString() const override { return "BYU_KIM_15_DiagonalsFirstOrder"; }
};

/** 4th order tridiagonal Kim Derivative, First Order, Pure Matrix System */
class BYU_KIM_16_DiagonalsFirstOrder : public MatrixCompactDerivs<1> {
public:
    BYU_KIM_16_DiagonalsFirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs<1>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createBYU_KIM_16_Diagonals();
        this->init();
    }
    ~BYU_KIM_16_DiagonalsFirstOrder() {
#ifdef DEBUG
        std::cout << "in BYU_KIM_16_DiagonalsFirstOrder destructor" << std::endl;
#endif
    }

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<BYU_KIM_16_DiagonalsFirstOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_KIMBYU_16; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_FIRST_ORDER; }
    std::string toString() const override { return "BYU_KIM_16_DiagonalsFirstOrder"; }
};

/** 4th order tridiagonal Kim Derivative, First Order, Pure Matrix System */
class BYU_KIM_17_DiagonalsFirstOrder : public MatrixCompactDerivs<1> {
public:
    BYU_KIM_17_DiagonalsFirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs<1>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createBYU_KIM_17_Diagonals();
        this->init();
    }
    ~BYU_KIM_17_DiagonalsFirstOrder() {
#ifdef DEBUG
        std::cout << "in BYU_KIM_17_DiagonalsFirstOrder destructor" << std::endl;
#endif
    }

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<BYU_KIM_17_DiagonalsFirstOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_KIMBYU_17; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_FIRST_ORDER; }
    std::string toString() const override { return "BYU_KIM_17_DiagonalsFirstOrder"; }
};

/** 4th order tridiagonal Kim Derivative, First Order, Pure Matrix System */
class BYU_KIM_18_DiagonalsFirstOrder : public MatrixCompactDerivs<1> {
public:
    BYU_KIM_18_DiagonalsFirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs<1>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createBYU_KIM_18_Diagonals();
        this->init();
    }
    ~BYU_KIM_18_DiagonalsFirstOrder() {
#ifdef DEBUG
        std::cout << "in BYU_KIM_18_DiagonalsFirstOrder destructor" << std::endl;
#endif
    }

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<BYU_KIM_18_DiagonalsFirstOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_KIMBYU_18; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_FIRST_ORDER; }
    std::string toString() const override { return "BYU_KIM_18_DiagonalsFirstOrder"; }
};

/** 4th order tridiagonal Kim Derivative, First Order, Pure Matrix System */
class BYU_KIM_19_DiagonalsFirstOrder : public MatrixCompactDerivs<1> {
public:
    BYU_KIM_19_DiagonalsFirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs<1>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createBYU_KIM_19_Diagonals();
        this->init();
    }
    ~BYU_KIM_19_DiagonalsFirstOrder() {
#ifdef DEBUG
        std::cout << "in BYU_KIM_19_DiagonalsFirstOrder destructor" << std::endl;
#endif
    }

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<BYU_KIM_19_DiagonalsFirstOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_KIMBYU_19; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_FIRST_ORDER; }
    std::string toString() const override { return "BYU_KIM_19_DiagonalsFirstOrder"; }
};

/** 4th order tridiagonal Kim Derivative, First Order, Pure Matrix System */
class Kim_16_BoundO4_FirstOrder : public MatrixCompactDerivs<1> {
   public:
    Kim_16_BoundO4_FirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs<1>{ele_order, in_matrix_filter,
                                 in_filter_coeffs} {
        diagEntries = createKim16Diagonals();

        this->init();
    }
    ~Kim_16_BoundO4_FirstOrder() {
#ifdef DEBUG
        std::cout << "in BandedCompactDerivs_KimBound4 deconstructor"
                  << std::endl;
#endif
    };

    std::unique_ptr<Derivs> clone() const override {
    return std::make_unique<Kim_16_BoundO4_FirstOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_K4_16; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_FIRST_ORDER;
    }

    std::string toString() const override { return "Kim_016BoundO4_FirstOrder"; };
};


/** 4th order tridiagonal Kim Derivative, First Order, Banded System */
class KimBoundO4_FirstOrder_Banded : public BandedCompactDerivs {
   public:
    template <typename... Args>
    KimBoundO4_FirstOrder_Banded(unsigned int ele_order, Args&&...)
        : BandedCompactDerivs{ele_order} {
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
            create_P_from_diagonals(*diagEntries, p_n, -1.0);

        printArray_2D_transpose(tempP.data(), p_n, p_n);

        std::vector<double> tempQ =
            create_Q_from_diagonals(*diagEntries, p_n, 0.0);

        printArray_2D_transpose(tempQ.data(), p_n, p_n);
    }
    ~KimBoundO4_FirstOrder_Banded() {
#ifdef DEBUG
        std::cout << "in BandedCompactDerivs_KimBound4 deconstructor"
                  << std::endl;
#endif
    };

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<KimBoundO4_FirstOrder_Banded>(*this);
    }

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
    KimBoundO4_SecondOrder(unsigned int ele_order)
        : BandedCompactDerivs{ele_order} {
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

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<KimBoundO4_SecondOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_K4; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_SECOND_ORDER;
    }

    std::string toString() const override { return "KimBoundO4_FirstOrder"; };
};

}  // namespace dendroderivs
