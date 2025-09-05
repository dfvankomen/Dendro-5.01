#pragma once

#include "derivatives/derivs_banded.h"
#include "derivatives/derivs_matrixonly.h"

namespace dendroderivs {

std::vector<double> inline clean_coeffs(const std::vector<double>& coeffs_in,
                                        unsigned int max_coeffs) {
    std::vector<double> coeffs_out(max_coeffs, 0.0);

    // std::cout << "Applying coefficients: ";
    // as soon as one of these breaks, we exit, no need to check sizes
    for (unsigned int i = 0; i < max_coeffs && i < coeffs_in.size(); i++) {
        coeffs_out[i] = coeffs_in[i];
        // std::cout << coeffs_in[i] << " ";
    }
    // std::cout << std::endl;

    return coeffs_out;
}

void inline check_end_of_boundaries(std::vector<std::vector<double>>& coeff_in,
                                    const double threshold = 1e-10) {
    // we should only chekc the DIAGs and remove any values that are extremely
    // close to or equal to zero

    for (auto& vec : coeff_in) {
        // check the last value, if it's "bad" pop it back, otherwise it should
        // end
        while (!vec.empty() && std::abs(vec.back()) < threshold) {
            vec.pop_back();
        }
    }
}
MatrixDiagonalEntries* BYUDerivsT64R3DiagonalsFirstOrder(
    const std::vector<double>& D_coeffs);

MatrixDiagonalEntries* BYUDerivsT6R2DiagonalsFirstOrder(
    const std::vector<double>& D_coeffs);

MatrixDiagonalEntries* BYUDerivsT4R2DiagonalsFirstOrder(
    const std::vector<double>& D_coeffs);

MatrixDiagonalEntries* BYUDerivsT4R1DiagonalsFirstOrder(
    const std::vector<double>& D_coeffs);

MatrixDiagonalEntries* BYUDerivsT4R3DiagonalsFirstOrder(
    const std::vector<double>& D_coeffs);

MatrixDiagonalEntries* BYUDerivsT4R42DiagonalsFirstOrder(
    const std::vector<double>& D_coeffs);

MatrixDiagonalEntries* BYUDerivsT4R3DiagonalsSecondOrder(
    const std::vector<double>& D_coeffs);

MatrixDiagonalEntries* BYUDerivsT6R3DiagonalsFirstOrder(
    const std::vector<double>& D_coeffs);

MatrixDiagonalEntries* BYUDerivsT6R3DiagonalsSecondOrder(
    const std::vector<double>& D_coeffs);

MatrixDiagonalEntries* BYUDerivsT6R4DiagonalsFirstOrder(
    const std::vector<double>& D_coeffs);

MatrixDiagonalEntries* BYUDerivsT6R42DiagonalsFirstOrder(
    const std::vector<double>& D_coeffs);

MatrixDiagonalEntries* BYUDerivsT6R4DiagonalsSecondOrder(
    const std::vector<double>& D_coeffs);

MatrixDiagonalEntries* BYUDerivsT6R2DiagonalsSecondOrder(
    const std::vector<double>& D_coeffs);

MatrixDiagonalEntries* BYUDerivsP6R2DiagonalsSecondOrder(
    const std::vector<double>& D_coeffs);

MatrixDiagonalEntries* BYUDerivsP6R2DiagonalsFirstOrder(
    const std::vector<double>& D_coeffs);

MatrixDiagonalEntries* BYUDerivsP6R3DiagonalsFirstOrder(
    const std::vector<double>& D_coeffs);

MatrixDiagonalEntries* BYUDerivsT8R3DiagonalsFirstOrder(
    const std::vector<double>& D_coeffs);

MatrixDiagonalEntries* BYUDerivsP6R3DiagonalsSecondOrder(
    const std::vector<double>& D_coeffs);

MatrixDiagonalEntries* BYUDerivsP6R32DiagonalsSecondOrder(
    const std::vector<double>& D_coeffs);

MatrixDiagonalEntries* BYUDerivsP8R4DiagonalsSecondOrder(
    const std::vector<double>& D_coeffs);

MatrixDiagonalEntries* BYUDerivsP8R4DiagonalsFirstOrder(
    const std::vector<double>& D_coeffs);
 //First Derivatives
    // (Assumes 'struct MatrixDiagonalEntries;' is visible to this header)

// ---- A4 ----
MatrixDiagonalEntries* createA4_1_Diagonals();
MatrixDiagonalEntries* createA4_2_Diagonals();
MatrixDiagonalEntries* createA4_3_Diagonals();
MatrixDiagonalEntries* createA4_4_Diagonals();
MatrixDiagonalEntries* createA4_5_Diagonals();
MatrixDiagonalEntries* createA4_6_Diagonals();
MatrixDiagonalEntries* createA4_7_Diagonals();
MatrixDiagonalEntries* createA4_8_Diagonals();
MatrixDiagonalEntries* createA4_9_Diagonals();
MatrixDiagonalEntries* createA4_10_Diagonals();
MatrixDiagonalEntries* createA4_11_Diagonals();
MatrixDiagonalEntries* createA4_12_Diagonals();
MatrixDiagonalEntries* createA4_13_Diagonals();
MatrixDiagonalEntries* createA4_14_Diagonals();
MatrixDiagonalEntries* createA4_15_Diagonals();
MatrixDiagonalEntries* createA4_16_Diagonals();
MatrixDiagonalEntries* createA4_17_Diagonals();
MatrixDiagonalEntries* createA4_18_Diagonals();
MatrixDiagonalEntries* createA4_19_Diagonals();

// ---- B4 ----
MatrixDiagonalEntries* createB4_1_Diagonals();
MatrixDiagonalEntries* createB4_2_Diagonals();

MatrixDiagonalEntries* createC4_1_Diagonals();
MatrixDiagonalEntries* createC4_2_Diagonals();
MatrixDiagonalEntries* createC4_3_Diagonals();
MatrixDiagonalEntries* createC4_4_Diagonals();
MatrixDiagonalEntries* createC4_5_Diagonals();

MatrixDiagonalEntries* createA6_1_Diagonals();
MatrixDiagonalEntries* createA6_2_Diagonals();
MatrixDiagonalEntries* createA6_3_Diagonals();

//Second Derivatives
MatrixDiagonalEntries* create2B4_1_Diagonals();
// ---- 2A6 ----
MatrixDiagonalEntries* create2A6_1_Diagonals();

// ---- 2B6 ----
MatrixDiagonalEntries* create2B6_1_Diagonals();
MatrixDiagonalEntries* create2B6_2_Diagonals();
MatrixDiagonalEntries* create2B6_3_Diagonals();
MatrixDiagonalEntries* create2B6_4_Diagonals();
MatrixDiagonalEntries* create2B6_5_Diagonals();
MatrixDiagonalEntries* create2B6_6_Diagonals();
MatrixDiagonalEntries* create2B6_7_Diagonals();
MatrixDiagonalEntries* create2B6_8_Diagonals();
MatrixDiagonalEntries* create2B6_9_Diagonals();

// ---- 2C6 ----
MatrixDiagonalEntries* create2C6_1_Diagonals();
// TODO: banded matrix implementations
class BYUDerivsT64_R3_FirstOrder : public MatrixCompactDerivs<1> {
   private:
    std::vector<double> D_coeffs_;

    static constexpr unsigned int n_D_coeffs_ = 4;

   public:
    BYUDerivsT64_R3_FirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>(),
        const std::vector<double>& coeffs_in        = std::vector<double>())
        : MatrixCompactDerivs{ele_order, in_matrix_filter, in_filter_coeffs} {
        // check the coefficients

        D_coeffs_ =
            clean_coeffs(coeffs_in, BYUDerivsT64_R3_FirstOrder::n_D_coeffs_);

        diagEntries = BYUDerivsT64R3DiagonalsFirstOrder(D_coeffs_);

        this->init();
    }
    ~BYUDerivsT64_R3_FirstOrder() {}

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<BYUDerivsT64_R3_FirstOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_BYUT64R3; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_FIRST_ORDER;
    }

    std::string toString() const override {
        return "BYUDerivsT64_R3_FirstOrder";
    }
};
// TODO: banded matrix implementations
class BYUDerivsT6_R2_FirstOrder : public MatrixCompactDerivs<1> {
   private:
    std::vector<double> D_coeffs_;

    static constexpr unsigned int n_D_coeffs_ = 3;

   public:
    template <typename... Args>
    BYUDerivsT6_R2_FirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>(),
        const std::vector<double>& coeffs_in        = std::vector<double>())
        : MatrixCompactDerivs{ele_order, in_matrix_filter, in_filter_coeffs} {
        // check the coefficients

        D_coeffs_ =
            clean_coeffs(coeffs_in, BYUDerivsT6_R2_FirstOrder::n_D_coeffs_);

        diagEntries = BYUDerivsT6R2DiagonalsFirstOrder(D_coeffs_);

        this->init();
    }
    ~BYUDerivsT6_R2_FirstOrder() {}

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<BYUDerivsT6_R2_FirstOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_BYUT6R2; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_FIRST_ORDER;
    }

    std::string toString() const override {
        return "BYUDerivsT6_R2_FirstOrder";
    }
};

// TODO: banded matrix implementations
class BYUDerivsT4_R2_FirstOrder : public MatrixCompactDerivs<1> {
   private:
    std::vector<double> D_coeffs_;

    static constexpr unsigned int n_D_coeffs_ = 3;

   public:
    template <typename... Args>
    BYUDerivsT4_R2_FirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>(),
        const std::vector<double>& coeffs_in        = std::vector<double>())
        : MatrixCompactDerivs{ele_order, in_matrix_filter, in_filter_coeffs} {
        // check the coefficients

        D_coeffs_ =
            clean_coeffs(coeffs_in, BYUDerivsT4_R2_FirstOrder::n_D_coeffs_);

        diagEntries = BYUDerivsT4R2DiagonalsFirstOrder(D_coeffs_);

        this->init();
    }
    ~BYUDerivsT4_R2_FirstOrder() {}

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<BYUDerivsT4_R2_FirstOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_BYUT4R2; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_FIRST_ORDER;
    }

    std::string toString() const override {
        return "BYUDerivsT4_R2_FirstOrder";
    }
};
// TODO: banded matrix implementations
class BYUDerivsT4_R1_FirstOrder : public MatrixCompactDerivs<1> {
   private:
    std::vector<double> D_coeffs_;

    static constexpr unsigned int n_D_coeffs_ = 2;

   public:
    template <typename... Args>
    BYUDerivsT4_R1_FirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>(),
        const std::vector<double>& coeffs_in        = std::vector<double>())
        : MatrixCompactDerivs{ele_order, in_matrix_filter, in_filter_coeffs} {
        // check the coefficients

        D_coeffs_ =
            clean_coeffs(coeffs_in, BYUDerivsT4_R1_FirstOrder::n_D_coeffs_);

        diagEntries = BYUDerivsT4R1DiagonalsFirstOrder(D_coeffs_);

        this->init();
    }
    ~BYUDerivsT4_R1_FirstOrder() {}

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<BYUDerivsT4_R1_FirstOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_BYUT4R1; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_FIRST_ORDER;
    }

    std::string toString() const override {
        return "BYUDerivsT4_R1_FirstOrder";
    }
};

class BYUDerivsT4_R3_FirstOrder : public MatrixCompactDerivs<1> {
   private:
    std::vector<double> D_coeffs_;

    static constexpr unsigned int n_D_coeffs_ = 4;

   public:
    BYUDerivsT4_R3_FirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>(),
        const std::vector<double>& coeffs_in        = std::vector<double>())
        : MatrixCompactDerivs{ele_order, in_matrix_filter, in_filter_coeffs} {
        // check the coefficients

        D_coeffs_ =
            clean_coeffs(coeffs_in, BYUDerivsT4_R3_FirstOrder::n_D_coeffs_);

        diagEntries = BYUDerivsT4R3DiagonalsFirstOrder(D_coeffs_);

        this->init();
    }
    ~BYUDerivsT4_R3_FirstOrder() {}

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<BYUDerivsT4_R3_FirstOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_BYUT4; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_FIRST_ORDER;
    }

    std::string toString() const override {
        return "BYUDerivsT4_R3_FirstOrder";
    }
};

class BYUDerivsT4_R42_FirstOrder : public MatrixCompactDerivs<1> {
   private:
    std::vector<double> D_coeffs_;

    static constexpr unsigned int n_D_coeffs_ = 5;

   public:
    template <typename... Args>
    BYUDerivsT4_R42_FirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>(),
        const std::vector<double>& coeffs_in        = std::vector<double>())
        : MatrixCompactDerivs{ele_order, in_matrix_filter, in_filter_coeffs} {
        // check the coefficients

        D_coeffs_ =
            clean_coeffs(coeffs_in, BYUDerivsT4_R42_FirstOrder::n_D_coeffs_);

        diagEntries = BYUDerivsT4R42DiagonalsFirstOrder(D_coeffs_);

        this->init();
    }
    ~BYUDerivsT4_R42_FirstOrder() {}

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<BYUDerivsT4_R42_FirstOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_BYUT4R4; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_FIRST_ORDER;
    }

    std::string toString() const override {
        return "BYUDerivsT4_R42_FirstOrder";
    }
};

class BYUDerivsT8_R3_FirstOrder : public MatrixCompactDerivs<1> {
   private:
    std::vector<double> D_coeffs_;

    static constexpr unsigned int n_D_coeffs_ = 5;

   public:
    template <typename... Args>
    BYUDerivsT8_R3_FirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>(),
        const std::vector<double>& coeffs_in        = std::vector<double>())
        : MatrixCompactDerivs{ele_order, in_matrix_filter, in_filter_coeffs} {
        // check the coefficients

        D_coeffs_ =
            clean_coeffs(coeffs_in, BYUDerivsT8_R3_FirstOrder::n_D_coeffs_);

        diagEntries = BYUDerivsT8R3DiagonalsFirstOrder(D_coeffs_);

        this->init();
    }
    ~BYUDerivsT8_R3_FirstOrder() {}

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<BYUDerivsT8_R3_FirstOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_BYUT8; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_FIRST_ORDER;
    }

    std::string toString() const override {
        return "BYUDerivsT8_R3_FirstOrder";
    }
};

class BYUDerivsT4_R3_SecondOrder : public MatrixCompactDerivs<2> {
   private:
    std::vector<double> D_coeffs_;

    static constexpr unsigned int n_D_coeffs_ = 4;

   public:
    BYUDerivsT4_R3_SecondOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>(),
        const std::vector<double>& coeffs_in        = std::vector<double>())
        : MatrixCompactDerivs{ele_order, in_matrix_filter, in_filter_coeffs} {
        // check the coefficients
        D_coeffs_ =
            clean_coeffs(coeffs_in, BYUDerivsT4_R3_SecondOrder::n_D_coeffs_);

        diagEntries = BYUDerivsT4R3DiagonalsSecondOrder(D_coeffs_);

        this->init();
    }
    ~BYUDerivsT4_R3_SecondOrder() {}

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<BYUDerivsT4_R3_SecondOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_BYUT4; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_SECOND_ORDER;
    }

    std::string toString() const override {
        return "BYUDerivsT4_R3_SecondOrder";
    }
};
class BYUDerivsP6_R2_FirstOrder : public MatrixCompactDerivs<1> {
   private:
    std::vector<double> D_coeffs_;

    static constexpr unsigned int n_D_coeffs_ = 3;

   public:
    BYUDerivsP6_R2_FirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>(),
        const std::vector<double>& coeffs_in        = std::vector<double>())
        : MatrixCompactDerivs{ele_order, in_matrix_filter, in_filter_coeffs} {
        // check the coefficients

        D_coeffs_ =
            clean_coeffs(coeffs_in, BYUDerivsP6_R2_FirstOrder::n_D_coeffs_);

        diagEntries = BYUDerivsP6R2DiagonalsFirstOrder(D_coeffs_);

        this->init();
    }
    ~BYUDerivsP6_R2_FirstOrder() {}

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<BYUDerivsP6_R2_FirstOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_BYUP6R2; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_FIRST_ORDER;
    }

    std::string toString() const override {
        return "BYUDerivsP6_R2_FirstOrder";
    }
};
class BYUDerivsP6_R32_SecondOrder : public MatrixCompactDerivs<2> {
   private:
    std::vector<double> D_coeffs_;

    static constexpr unsigned int n_D_coeffs_ = 4;

   public:
    BYUDerivsP6_R32_SecondOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>(),
        const std::vector<double>& coeffs_in        = std::vector<double>())
        : MatrixCompactDerivs{ele_order, in_matrix_filter, in_filter_coeffs} {
        // check the coefficients
        D_coeffs_ =
            clean_coeffs(coeffs_in, BYUDerivsP6_R32_SecondOrder::n_D_coeffs_);

        diagEntries = BYUDerivsP6R32DiagonalsSecondOrder(D_coeffs_);

        this->init();
    }
    ~BYUDerivsP6_R32_SecondOrder() {}

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<BYUDerivsP6_R32_SecondOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_BYUP6R3; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_SECOND_ORDER;
    }

    std::string toString() const override {
        return "BYUDerivsP6_R32_SecondOrder";
    }
};

class BYUDerivsP6_R2_SecondOrder : public MatrixCompactDerivs<2> {
   private:
    std::vector<double> D_coeffs_;

    static constexpr unsigned int n_D_coeffs_ = 3;

   public:
    BYUDerivsP6_R2_SecondOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>(),
        const std::vector<double>& coeffs_in        = std::vector<double>())
        : MatrixCompactDerivs{ele_order, in_matrix_filter, in_filter_coeffs} {
        // check the coefficients
        D_coeffs_ =
            clean_coeffs(coeffs_in, BYUDerivsP6_R2_SecondOrder::n_D_coeffs_);

        diagEntries = BYUDerivsP6R2DiagonalsSecondOrder(D_coeffs_);

        this->init();
    }
    ~BYUDerivsP6_R2_SecondOrder() {}

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<BYUDerivsP6_R2_SecondOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_BYUP6R2; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_SECOND_ORDER;
    }

    std::string toString() const override {
        return "BYUDerivsP6_R2_SecondOrder";
    }
};

class BYUDerivsT6_R2_SecondOrder : public MatrixCompactDerivs<2> {
   private:
    std::vector<double> D_coeffs_;

    static constexpr unsigned int n_D_coeffs_ = 4;

   public:
    BYUDerivsT6_R2_SecondOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>(),
        const std::vector<double>& coeffs_in        = std::vector<double>())
        : MatrixCompactDerivs{ele_order, in_matrix_filter, in_filter_coeffs} {
        // check the coefficients
        D_coeffs_ =
            clean_coeffs(coeffs_in, BYUDerivsT6_R2_SecondOrder::n_D_coeffs_);

        diagEntries = BYUDerivsT6R2DiagonalsSecondOrder(D_coeffs_);

        this->init();
    }
    ~BYUDerivsT6_R2_SecondOrder() {}

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<BYUDerivsT6_R2_SecondOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_BYUT6R2; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_SECOND_ORDER;
    }

    std::string toString() const override {
        return "BYUDerivsT6_R2_SecondOrder";
    }
};

class BYUDerivsT6_R3_FirstOrder : public MatrixCompactDerivs<1> {
   private:
    std::vector<double> D_coeffs_;

    static constexpr unsigned int n_D_coeffs_ = 4;

   public:
    BYUDerivsT6_R3_FirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>(),
        const std::vector<double>& coeffs_in        = std::vector<double>())
        : MatrixCompactDerivs{ele_order, in_matrix_filter, in_filter_coeffs} {
        // check the coefficients
        D_coeffs_ =
            clean_coeffs(coeffs_in, BYUDerivsT6_R3_FirstOrder::n_D_coeffs_);

        diagEntries = BYUDerivsT6R3DiagonalsFirstOrder(D_coeffs_);

        this->init();
    }

    ~BYUDerivsT6_R3_FirstOrder() {}

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<BYUDerivsT6_R3_FirstOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_BYUT6R3; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_FIRST_ORDER;
    }

    std::string toString() const override {
        return "BYUDerivsT6_R3_SecondOrder";
    }
};

class BYUDerivsP6_R32_FirstOrder : public MatrixCompactDerivs<1> {
   private:
    std::vector<double> D_coeffs_;

    static constexpr unsigned int n_D_coeffs_ = 4;

   public:
    BYUDerivsP6_R32_FirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>(),
        const std::vector<double>& coeffs_in        = std::vector<double>())
        : MatrixCompactDerivs{ele_order, in_matrix_filter, in_filter_coeffs} {
        // check the coefficients
        D_coeffs_ =
            clean_coeffs(coeffs_in, BYUDerivsP6_R32_FirstOrder::n_D_coeffs_);

        diagEntries = BYUDerivsP6R3DiagonalsFirstOrder(D_coeffs_);

        this->init();
    }

    ~BYUDerivsP6_R32_FirstOrder() {}

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<BYUDerivsP6_R32_FirstOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_BYUP6R3; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_FIRST_ORDER;
    }

    std::string toString() const override {
        return "BYUDerivsP6_R32_FirstOrder";
    }
};

class BYUDerivsT6_R3_SecondOrder : public MatrixCompactDerivs<2> {
   private:
    std::vector<double> D_coeffs_;

    static constexpr unsigned int n_D_coeffs_ = 4;

   public:
    BYUDerivsT6_R3_SecondOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>(),
        const std::vector<double>& coeffs_in        = std::vector<double>())
        : MatrixCompactDerivs{ele_order, in_matrix_filter, in_filter_coeffs} {
        // check the coefficients
        D_coeffs_ =
            clean_coeffs(coeffs_in, BYUDerivsT6_R3_SecondOrder::n_D_coeffs_);

        diagEntries = BYUDerivsT6R3DiagonalsSecondOrder(D_coeffs_);

        this->init();
    }

    ~BYUDerivsT6_R3_SecondOrder() {}

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<BYUDerivsT6_R3_SecondOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_BYUT6R3; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_SECOND_ORDER;
    }

    std::string toString() const override {
        return "BYUDerivsT6_R3_SecondOrder";
    }
};

class BYUDerivsT6_R4_FirstOrder : public MatrixCompactDerivs<1> {
   private:
    std::vector<double> D_coeffs_;

    static constexpr unsigned int n_D_coeffs_ = 5;

   public:
    BYUDerivsT6_R4_FirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>(),
        const std::vector<double>& coeffs_in        = std::vector<double>())
        : MatrixCompactDerivs{ele_order, in_matrix_filter, in_filter_coeffs} {
        // check the coefficients
        D_coeffs_ =
            clean_coeffs(coeffs_in, BYUDerivsT6_R4_FirstOrder::n_D_coeffs_);

        diagEntries = BYUDerivsT6R4DiagonalsFirstOrder(D_coeffs_);

        this->init();
    }

    ~BYUDerivsT6_R4_FirstOrder() {}

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<BYUDerivsT6_R4_FirstOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_BYUT6; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_FIRST_ORDER;
    }

    std::string toString() const override {
        return "BYUDerivsT6_R4_SecondOrder";
    }
};

class BYUDerivsT6_R42_FirstOrder : public MatrixCompactDerivs<1> {
   private:
    std::vector<double> D_coeffs_;

    static constexpr unsigned int n_D_coeffs_ = 5;

   public:
    template <typename... Args>
    BYUDerivsT6_R42_FirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>(),
        const std::vector<double>& coeffs_in        = std::vector<double>())
        : MatrixCompactDerivs{ele_order, in_matrix_filter, in_filter_coeffs} {
        // check the coefficients
        D_coeffs_ =
            clean_coeffs(coeffs_in, BYUDerivsT6_R42_FirstOrder::n_D_coeffs_);

        diagEntries = BYUDerivsT6R42DiagonalsFirstOrder(D_coeffs_);

        this->init();
    }

    ~BYUDerivsT6_R42_FirstOrder() {}

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<BYUDerivsT6_R42_FirstOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_BYUT6R4; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_FIRST_ORDER;
    }

    std::string toString() const override {
        return "BYUDerivsT6_R42_SecondOrder";
    }
};

class BYUDerivsT6_R4_SecondOrder : public MatrixCompactDerivs<2> {
   private:
    std::vector<double> D_coeffs_;

    static constexpr unsigned int n_D_coeffs_ = 5;

   public:
    BYUDerivsT6_R4_SecondOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>(),
        const std::vector<double>& coeffs_in        = std::vector<double>())
        : MatrixCompactDerivs{ele_order, in_matrix_filter, in_filter_coeffs} {
        // check the coefficients
        D_coeffs_ =
            clean_coeffs(coeffs_in, BYUDerivsT6_R4_SecondOrder::n_D_coeffs_);

        diagEntries = BYUDerivsT6R4DiagonalsSecondOrder(D_coeffs_);

        this->init();
    }

    ~BYUDerivsT6_R4_SecondOrder() {}

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<BYUDerivsT6_R4_SecondOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_BYUT6; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_SECOND_ORDER;
    }

    std::string toString() const override {
        return "BYUDerivsT6_R4_SecondOrder";
    }
};

/**
 *
 * BYUDerivsP6_R3_FirstOrder
 *
 * Pentadiagonal system with O(6) accuracy.  There are 3 boundary rows.
 * There are 7 free parameters, one for the center stencil, and 2 for
 * each boundary row.
 *
 * The first and second derivatives are solved through a
 * linear system of the form
 *
 * P f'  = Q f
 * P f'' = Q f
 *
 * The LHS Matrix, P
 |  1    g01   g02     0      0      0     0     0     0   |
 |  g10   1    g12    g13     0      0     0     0     0   |
 |  g20  g21    1     g23    g24     0     0     0     0   |
 |  0   beta   alpha   1    alpha   beta   0     0     0   |
 |  0     0   beta   alpha   1    alpha   beta   0     0   |
 |  0     0     0    beta   alpha   1    alpha   beta  0   |
 |  0     0     0      0    g24   g23     1     g21   g20  |
 |  0     0     0      0     0    g13    g12     1    g10  |
 |  0     0     0      0     0     0     g02    g01    1   |


  The RHS Matrix, Q
| a00    a01    a02    a03    a04    a05    a06    a07     0   |
| a10    a11    a12    a13    a14    a15    a16    a17     0   |
| a20    a21    a22    a23    a24    a25    a26    a27     0   |
| 0    -a2/4h -a1/2h    0    a1/2h  a2/4h    0      0      0   |
| 0       0   -a2/4h -a1/2h    0    a1/2h  a2/4h    0      0   |
| 0       0      0   -a2/4h -a1/2h    0    a1/2h  a2/4h    0   |
| 0    -a27    -a26   -a25   -a24   -a23   -a22   -a21   -a20  |
| 0    -a17    -a16   -a15   -a14   -a13   -a12   -a11   -a10  |
| 0    -a07    -a06   -a05   -a04   -a03   -a02   -a01   -a00  |

This is a pentadiagonal system, as P has 5 total diagonals, and this is the
matrix to be inverted.

Each derivative system has three boundary rows.  Note that the minimum
number of boundary rows is two.  The third boundary row gives extra
freedom in defining the operator, which can be used to enhance accuracy
at the boundary or to improve stability.

The free parameters for the first derivative operator are:
   1. beta
   2. a06
   3. a17
   4. a27
   5. a07
   6. a11
   7. a22

The free parameters are set by the arguments coeffs1 and coeffs2.  These
are vectors.  The free parameters are filled in the order listed above, e.g.,

     coeffs1 = [ beta, a06, a17, a27, ... ]

Any parameters not specified are set to zero.  Any entries in coeffs1 or coeffs2
beyond 7 are ignored.

The parameters a11 and a22 are set to zero by default.  Setting them to non-zero
values seems to make the operators much less stable, although there is no proof
of this.

The free parameters for the 2nd derivative operator are:
   1. beta
   2. a07
   3. a17
   4. a27
   5. a08
   6. a18
   7. a28

 */
class BYUDerivsP6_R3_FirstOrder : public MatrixCompactDerivs<1> {
   private:
    std::vector<double> D_coeffs_;

    static constexpr unsigned int n_D_coeffs_ = 7;

   public:
    BYUDerivsP6_R3_FirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>(),
        const std::vector<double>& coeffs_in        = std::vector<double>())
        : MatrixCompactDerivs{ele_order, in_matrix_filter, in_filter_coeffs} {
        // check the coefficients
        D_coeffs_ =
            clean_coeffs(coeffs_in, BYUDerivsP6_R3_FirstOrder::n_D_coeffs_);

        diagEntries = BYUDerivsP6R3DiagonalsFirstOrder(D_coeffs_);

        this->init();
    }

    ~BYUDerivsP6_R3_FirstOrder() {}

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<BYUDerivsP6_R3_FirstOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_BYUT6; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_FIRST_ORDER;
    }

    std::string toString() const override {
        return "BYUDerivsP6_R3_SecondOrder";
    }
};

class BYUDerivsP6_R3_SecondOrder : public MatrixCompactDerivs<2> {
   private:
    std::vector<double> D_coeffs_;

    static constexpr unsigned int n_D_coeffs_ = 7;

   public:
    BYUDerivsP6_R3_SecondOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>(),
        const std::vector<double>& coeffs_in        = std::vector<double>())
        : MatrixCompactDerivs{ele_order, in_matrix_filter, in_filter_coeffs} {
        // check the coefficients
        D_coeffs_ =
            clean_coeffs(coeffs_in, BYUDerivsP6_R3_SecondOrder::n_D_coeffs_);

        diagEntries = BYUDerivsP6R3DiagonalsSecondOrder(D_coeffs_);

        this->init();
    }

    ~BYUDerivsP6_R3_SecondOrder() {}

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<BYUDerivsP6_R3_SecondOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_BYUP6; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_SECOND_ORDER;
    }

    std::string toString() const override {
        return "BYUDerivsP6_R3_SecondOrder";
    }
};

/**
 *
 * The first and second derivatives are solved through a linear system of the
form
 *
 * P f'  = Q f
 * P f'' = Q f
 *
 * The LHS Matrix, P
 |  1    g01   g02     0      0      0     0     0     0      0     0   |
 |  g10   1    g12    g13     0      0     0     0     0      0     0   |
 |  g20  g21    1     g23    g24     0     0     0     0      0     0   |
 |  0    g31   g32    1      g34    g35    0     0     0      0     0   |
 |  0     0   beta   alpha   1    alpha   beta   0     0      0     0   |
 |  0     0     0    beta  alpha   1    alpha  beta    0      0     0   |
 |  0     0     0      0   beta   alpha   1    alpha  beta    0     0   |
 |  0     0     0      0     0     g35   g34    1     g32    g31    0   |
 |  0     0     0      0     0      0    g24   g23     1     g21   g20  |
 |  0     0     0      0     0      0     0    g13    g12     1    g10  |
 |  0     0     0      0     0      0     0     0     g02    g01    1   |


* The RHS Matrix, Q
| a00    a01    a02    a03    a04    a05    a06    a07    a08    a09     0   |
| a10    a11    a12    a13    a14    a15    a16    a17    a18    a19     0   |
| a20    a21    a22    a23    a24    a25    a26    a27    a28    a29     0   |
| a30    a31    a32    a33    a34    a35    a36    a37    a38    a39     0   |
| 0    -a3/8h -a2/4h -a1/2h    0    a1/2h  a2/4h  a3/8h    0      0      0   |
| 0       0   -a3/8h -a2/4h -a1/2h    0    a1/2h  a2/4h  a3/8h    0      0   |
| 0       0      0   -a3/8h -a2/4h -a1/2h    0    a1/2h  a2/4h  a3/8h    0   |
| 0    -a39    -a38   -a37   -a36   -a35   -a34   -a33   -a32   -a31   -a30  |
| 0    -a29    -a28   -a27   -a26   -a25   -a24   -a23   -a22   -a21   -a20  |
| 0    -a19    -a18   -a17   -a16   -a15   -a14   -a13   -a12   -a11   -a10  |
| 0    -a09    -a08   -a07   -a06   -a05   -a04   -a03   -a02   -a01   -a00  |

This is a pentadiagonal system, as P has 5 total diagonals, and this is the
matrix to be inverted.

Each derivative system has four boundary rows.  Note that the minimum
number of boundary rows is three.  The fourth boundary row gives extra
freedom in defining the operator, which can be used to enhance accuracy
at the boundary or to improve stability.

The free parameters for these operators are:
   1. beta
   2. a08
   3. a19
   4. a29
   5. a39
   6. a09
   7. a11
   8. a22
   9. a33

The free parameters are set by the arguments coeffs1 and coeffs2.  These
are vectors.  The free parameters are filled in the order listed above, e.g.,

     coeffs1 = [ beta, a08, a19, a29, a39, ... ]

Any parameters not specified are set to zero.  Any entries in coeffs1 or coeffs2
beyond 9 are ignored.

The parameters a11, a22, and a33 are set to zero by default.  Setting them to
non-zero values seems to make the operators much less stable, although there is
no proof of this.
 *
 */
class BYUDerivsP8_R4_FirstOrder : public MatrixCompactDerivs<1> {
   private:
    std::vector<double> D_coeffs_;

    static constexpr unsigned int n_D_coeffs_ = 9;

   public:
    BYUDerivsP8_R4_FirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>(),
        const std::vector<double>& coeffs_in        = std::vector<double>())
        : MatrixCompactDerivs{ele_order, in_matrix_filter, in_filter_coeffs} {
        // check the coefficients
        D_coeffs_ =
            clean_coeffs(coeffs_in, BYUDerivsP8_R4_FirstOrder::n_D_coeffs_);

        diagEntries = BYUDerivsP8R4DiagonalsFirstOrder(D_coeffs_);

        this->init();
    }

    ~BYUDerivsP8_R4_FirstOrder() {}

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<BYUDerivsP8_R4_FirstOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_BYUP8; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_FIRST_ORDER;
    }

    std::string toString() const override {
        return "BYUDerivsP8_R4_SecondOrder";
    }
};

class BYUDerivsP8_R4_SecondOrder : public MatrixCompactDerivs<2> {
   private:
    std::vector<double> D_coeffs_;

    static constexpr unsigned int n_D_coeffs_ = 9;

   public:
    BYUDerivsP8_R4_SecondOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>(),
        const std::vector<double>& coeffs_in        = std::vector<double>())
        : MatrixCompactDerivs{ele_order, in_matrix_filter, in_filter_coeffs} {
        // check the coefficients
        D_coeffs_ =
            clean_coeffs(coeffs_in, BYUDerivsP8_R4_SecondOrder::n_D_coeffs_);

        diagEntries = BYUDerivsP8R4DiagonalsSecondOrder(D_coeffs_);

        this->init();
    }

    ~BYUDerivsP8_R4_SecondOrder() {}

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<BYUDerivsP8_R4_SecondOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_BYUP8; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_SECOND_ORDER;
    }

    std::string toString() const override {
        return "BYUDerivsP8_R4_SecondOrder";
    }
};
//First Derivatives:
// ----------------------------- A4 ---------------------------------
class A4_1_DiagonalsFirstOrder : public MatrixCompactDerivs<1> {
public:
    A4_1_DiagonalsFirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs<1>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createA4_1_Diagonals();
        this->init();
    }
    ~A4_1_DiagonalsFirstOrder() {
#ifdef DEBUG
        std::cout << "in A4_1_DiagonalsFirstOrder destructor" << std::endl;
#endif
    }
    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<A4_1_DiagonalsFirstOrder>(*this);
    }
    DerivType getDerivType() const override { return DerivType::D_A4_1; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_FIRST_ORDER; }
    std::string toString() const override { return "A4_1_DiagonalsFirstOrder"; }
};

class A4_2_DiagonalsFirstOrder : public MatrixCompactDerivs<1> {
public:
    A4_2_DiagonalsFirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs<1>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createA4_2_Diagonals();
        this->init();
    }
    ~A4_2_DiagonalsFirstOrder() {
#ifdef DEBUG
        std::cout << "in A4_2_DiagonalsFirstOrder destructor" << std::endl;
#endif
    }
    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<A4_2_DiagonalsFirstOrder>(*this);
    }
    DerivType getDerivType() const override { return DerivType::D_A4_2; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_FIRST_ORDER; }
    std::string toString() const override { return "A4_2_DiagonalsFirstOrder"; }
};

class A4_3_DiagonalsFirstOrder : public MatrixCompactDerivs<1> {
public:
    A4_3_DiagonalsFirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs<1>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createA4_3_Diagonals();
        this->init();
    }
    ~A4_3_DiagonalsFirstOrder() {
#ifdef DEBUG
        std::cout << "in A4_3_DiagonalsFirstOrder destructor" << std::endl;
#endif
    }
    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<A4_3_DiagonalsFirstOrder>(*this);
    }
    DerivType getDerivType() const override { return DerivType::D_A4_3; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_FIRST_ORDER; }
    std::string toString() const override { return "A4_3_DiagonalsFirstOrder"; }
};

class A4_4_DiagonalsFirstOrder : public MatrixCompactDerivs<1> {
public:
    A4_4_DiagonalsFirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs<1>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createA4_4_Diagonals();
        this->init();
    }
    ~A4_4_DiagonalsFirstOrder() {
#ifdef DEBUG
        std::cout << "in A4_4_DiagonalsFirstOrder destructor" << std::endl;
#endif
    }
    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<A4_4_DiagonalsFirstOrder>(*this);
    }
    DerivType getDerivType() const override { return DerivType::D_A4_4; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_FIRST_ORDER; }
    std::string toString() const override { return "A4_4_DiagonalsFirstOrder"; }
};

class A4_5_DiagonalsFirstOrder : public MatrixCompactDerivs<1> {
public:
    A4_5_DiagonalsFirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs<1>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createA4_5_Diagonals();
        this->init();
    }
    ~A4_5_DiagonalsFirstOrder() {
#ifdef DEBUG
        std::cout << "in A4_5_DiagonalsFirstOrder destructor" << std::endl;
#endif
    }
    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<A4_5_DiagonalsFirstOrder>(*this);
    }
    DerivType getDerivType() const override { return DerivType::D_A4_5; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_FIRST_ORDER; }
    std::string toString() const override { return "A4_5_DiagonalsFirstOrder"; }
};

class A4_6_DiagonalsFirstOrder : public MatrixCompactDerivs<1> {
public:
    A4_6_DiagonalsFirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs<1>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createA4_6_Diagonals();
        this->init();
    }
    ~A4_6_DiagonalsFirstOrder() {
#ifdef DEBUG
        std::cout << "in A4_6_DiagonalsFirstOrder destructor" << std::endl;
#endif
    }
    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<A4_6_DiagonalsFirstOrder>(*this);
    }
    DerivType getDerivType() const override { return DerivType::D_A4_6; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_FIRST_ORDER; }
    std::string toString() const override { return "A4_6_DiagonalsFirstOrder"; }
};

class A4_7_DiagonalsFirstOrder : public MatrixCompactDerivs<1> {
public:
    A4_7_DiagonalsFirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs<1>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createA4_7_Diagonals();
        this->init();
    }
    ~A4_7_DiagonalsFirstOrder() {
#ifdef DEBUG
        std::cout << "in A4_7_DiagonalsFirstOrder destructor" << std::endl;
#endif
    }
    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<A4_7_DiagonalsFirstOrder>(*this);
    }
    DerivType getDerivType() const override { return DerivType::D_A4_7; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_FIRST_ORDER; }
    std::string toString() const override { return "A4_7_DiagonalsFirstOrder"; }
};

class A4_8_DiagonalsFirstOrder : public MatrixCompactDerivs<1> {
public:
    A4_8_DiagonalsFirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs<1>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createA4_8_Diagonals();
        this->init();
    }
    ~A4_8_DiagonalsFirstOrder() {
#ifdef DEBUG
        std::cout << "in A4_8_DiagonalsFirstOrder destructor" << std::endl;
#endif
    }
    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<A4_8_DiagonalsFirstOrder>(*this);
    }
    DerivType getDerivType() const override { return DerivType::D_A4_8; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_FIRST_ORDER; }
    std::string toString() const override { return "A4_8_DiagonalsFirstOrder"; }
};

class A4_9_DiagonalsFirstOrder : public MatrixCompactDerivs<1> {
public:
    A4_9_DiagonalsFirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs<1>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createA4_9_Diagonals();
        this->init();
    }
    ~A4_9_DiagonalsFirstOrder() {
#ifdef DEBUG
        std::cout << "in A4_9_DiagonalsFirstOrder destructor" << std::endl;
#endif
    }
    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<A4_9_DiagonalsFirstOrder>(*this);
    }
    DerivType getDerivType() const override { return DerivType::D_A4_9; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_FIRST_ORDER; }
    std::string toString() const override { return "A4_9_DiagonalsFirstOrder"; }
};

class A4_10_DiagonalsFirstOrder : public MatrixCompactDerivs<1> {
public:
    A4_10_DiagonalsFirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs<1>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createA4_10_Diagonals();
        this->init();
    }
    ~A4_10_DiagonalsFirstOrder() {
#ifdef DEBUG
        std::cout << "in A4_10_DiagonalsFirstOrder destructor" << std::endl;
#endif
    }
    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<A4_10_DiagonalsFirstOrder>(*this);
    }
    DerivType getDerivType() const override { return DerivType::D_A4_10; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_FIRST_ORDER; }
    std::string toString() const override { return "A4_10_DiagonalsFirstOrder"; }
};

class A4_11_DiagonalsFirstOrder : public MatrixCompactDerivs<1> {
public:
    A4_11_DiagonalsFirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs<1>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createA4_11_Diagonals();
        this->init();
    }
    ~A4_11_DiagonalsFirstOrder() {
#ifdef DEBUG
        std::cout << "in A4_11_DiagonalsFirstOrder destructor" << std::endl;
#endif
    }
    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<A4_11_DiagonalsFirstOrder>(*this);
    }
    DerivType getDerivType() const override { return DerivType::D_A4_11; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_FIRST_ORDER; }
    std::string toString() const override { return "A4_11_DiagonalsFirstOrder"; }
};

class A4_12_DiagonalsFirstOrder : public MatrixCompactDerivs<1> {
public:
    A4_12_DiagonalsFirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs<1>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createA4_12_Diagonals();
        this->init();
    }
    ~A4_12_DiagonalsFirstOrder() {
#ifdef DEBUG
        std::cout << "in A4_12_DiagonalsFirstOrder destructor" << std::endl;
#endif
    }
    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<A4_12_DiagonalsFirstOrder>(*this);
    }
    DerivType getDerivType() const override { return DerivType::D_A4_12; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_FIRST_ORDER; }
    std::string toString() const override { return "A4_12_DiagonalsFirstOrder"; }
};

class A4_13_DiagonalsFirstOrder : public MatrixCompactDerivs<1> {
public:
    A4_13_DiagonalsFirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs<1>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createA4_13_Diagonals();
        this->init();
    }
    ~A4_13_DiagonalsFirstOrder() {
#ifdef DEBUG
        std::cout << "in A4_13_DiagonalsFirstOrder destructor" << std::endl;
#endif
    }
    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<A4_13_DiagonalsFirstOrder>(*this);
    }
    DerivType getDerivType() const override { return DerivType::D_A4_13; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_FIRST_ORDER; }
    std::string toString() const override { return "A4_13_DiagonalsFirstOrder"; }
};

class A4_14_DiagonalsFirstOrder : public MatrixCompactDerivs<1> {
public:
    A4_14_DiagonalsFirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs<1>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createA4_14_Diagonals();
        this->init();
    }
    ~A4_14_DiagonalsFirstOrder() {
#ifdef DEBUG
        std::cout << "in A4_14_DiagonalsFirstOrder destructor" << std::endl;
#endif
    }
    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<A4_14_DiagonalsFirstOrder>(*this);
    }
    DerivType getDerivType() const override { return DerivType::D_A4_14; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_FIRST_ORDER; }
    std::string toString() const override { return "A4_14_DiagonalsFirstOrder"; }
};

class A4_15_DiagonalsFirstOrder : public MatrixCompactDerivs<1> {
public:
    A4_15_DiagonalsFirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs<1>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createA4_15_Diagonals();
        this->init();
    }
    ~A4_15_DiagonalsFirstOrder() {
#ifdef DEBUG
        std::cout << "in A4_15_DiagonalsFirstOrder destructor" << std::endl;
#endif
    }
    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<A4_15_DiagonalsFirstOrder>(*this);
    }
    DerivType getDerivType() const override { return DerivType::D_A4_15; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_FIRST_ORDER; }
    std::string toString() const override { return "A4_15_DiagonalsFirstOrder"; }
};

class A4_16_DiagonalsFirstOrder : public MatrixCompactDerivs<1> {
public:
    A4_16_DiagonalsFirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs<1>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createA4_16_Diagonals();
        this->init();
    }
    ~A4_16_DiagonalsFirstOrder() {
#ifdef DEBUG
        std::cout << "in A4_16_DiagonalsFirstOrder destructor" << std::endl;
#endif
    }
    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<A4_16_DiagonalsFirstOrder>(*this);
    }
    DerivType getDerivType() const override { return DerivType::D_A4_16; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_FIRST_ORDER; }
    std::string toString() const override { return "A4_16_DiagonalsFirstOrder"; }
};

class A4_17_DiagonalsFirstOrder : public MatrixCompactDerivs<1> {
public:
    A4_17_DiagonalsFirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs<1>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createA4_17_Diagonals();
        this->init();
    }
    ~A4_17_DiagonalsFirstOrder() {
#ifdef DEBUG
        std::cout << "in A4_17_DiagonalsFirstOrder destructor" << std::endl;
#endif
    }
    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<A4_17_DiagonalsFirstOrder>(*this);
    }
    DerivType getDerivType() const override { return DerivType::D_A4_17; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_FIRST_ORDER; }
    std::string toString() const override { return "A4_17_DiagonalsFirstOrder"; }
};

class A4_18_DiagonalsFirstOrder : public MatrixCompactDerivs<1> {
public:
    A4_18_DiagonalsFirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs<1>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createA4_18_Diagonals();
        this->init();
    }
    ~A4_18_DiagonalsFirstOrder() {
#ifdef DEBUG
        std::cout << "in A4_18_DiagonalsFirstOrder destructor" << std::endl;
#endif
    }
    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<A4_18_DiagonalsFirstOrder>(*this);
    }
    DerivType getDerivType() const override { return DerivType::D_A4_18; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_FIRST_ORDER; }
    std::string toString() const override { return "A4_18_DiagonalsFirstOrder"; }
};

class A4_19_DiagonalsFirstOrder : public MatrixCompactDerivs<1> {
public:
    A4_19_DiagonalsFirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs<1>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createA4_19_Diagonals();
        this->init();
    }
    ~A4_19_DiagonalsFirstOrder() {
#ifdef DEBUG
        std::cout << "in A4_19_DiagonalsFirstOrder destructor" << std::endl;
#endif
    }
    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<A4_19_DiagonalsFirstOrder>(*this);
    }
    DerivType getDerivType() const override { return DerivType::D_A4_19; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_FIRST_ORDER; }
    std::string toString() const override { return "A4_19_DiagonalsFirstOrder"; }
};

// ----------------------------- B4 ---------------------------------
class B4_1_DiagonalsFirstOrder : public MatrixCompactDerivs<1> {
public:
    B4_1_DiagonalsFirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs<1>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createB4_1_Diagonals();
        this->init();
    }
    ~B4_1_DiagonalsFirstOrder() {
#ifdef DEBUG
        std::cout << "in B4_1_DiagonalsFirstOrder destructor" << std::endl;
#endif
    }
    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<B4_1_DiagonalsFirstOrder>(*this);
    }
    DerivType getDerivType() const override { return DerivType::D_B4_1; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_FIRST_ORDER; }
    std::string toString() const override { return "B4_1_DiagonalsFirstOrder"; }
};

class B4_2_DiagonalsFirstOrder : public MatrixCompactDerivs<1> {
public:
    B4_2_DiagonalsFirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs<1>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createB4_2_Diagonals();
        this->init();
    }
    ~B4_2_DiagonalsFirstOrder() {
#ifdef DEBUG
        std::cout << "in B4_2_DiagonalsFirstOrder destructor" << std::endl;
#endif
    }
    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<B4_2_DiagonalsFirstOrder>(*this);
    }
    DerivType getDerivType() const override { return DerivType::D_B4_2; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_FIRST_ORDER; }
    std::string toString() const override { return "B4_2_DiagonalsFirstOrder"; }
};
class C4_1_DiagonalsFirstOrder : public MatrixCompactDerivs<1> {
public:
    C4_1_DiagonalsFirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs<1>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createC4_1_Diagonals();
        this->init();
    }
    ~C4_1_DiagonalsFirstOrder() {
#ifdef DEBUG
        std::cout << "in C4_1_DiagonalsFirstOrder destructor" << std::endl;
#endif
    }

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<C4_1_DiagonalsFirstOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_C4_1; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_FIRST_ORDER; }
    std::string toString() const override { return "C4_1_DiagonalsFirstOrder"; }
};

class C4_2_DiagonalsFirstOrder : public MatrixCompactDerivs<1> {
public:
    C4_2_DiagonalsFirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs<1>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createC4_2_Diagonals();
        this->init();
    }
    ~C4_2_DiagonalsFirstOrder() {
#ifdef DEBUG
        std::cout << "in C4_2_DiagonalsFirstOrder destructor" << std::endl;
#endif
    }

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<C4_2_DiagonalsFirstOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_C4_2; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_FIRST_ORDER; }
    std::string toString() const override { return "C4_2_DiagonalsFirstOrder"; }
};

class C4_3_DiagonalsFirstOrder : public MatrixCompactDerivs<1> {
public:
    C4_3_DiagonalsFirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs<1>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createC4_3_Diagonals();
        this->init();
    }
    ~C4_3_DiagonalsFirstOrder() {
#ifdef DEBUG
        std::cout << "in C4_3_DiagonalsFirstOrder destructor" << std::endl;
#endif
    }

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<C4_3_DiagonalsFirstOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_C4_3; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_FIRST_ORDER; }
    std::string toString() const override { return "C4_3_DiagonalsFirstOrder"; }
};

class C4_4_DiagonalsFirstOrder : public MatrixCompactDerivs<1> {
public:
    C4_4_DiagonalsFirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs<1>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createC4_4_Diagonals();
        this->init();
    }
    ~C4_4_DiagonalsFirstOrder() {
#ifdef DEBUG
        std::cout << "in C4_4_DiagonalsFirstOrder destructor" << std::endl;
#endif
    }

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<C4_4_DiagonalsFirstOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_C4_4; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_FIRST_ORDER; }
    std::string toString() const override { return "C4_4_DiagonalsFirstOrder"; }
};

class C4_5_DiagonalsFirstOrder : public MatrixCompactDerivs<1> {
public:
    C4_5_DiagonalsFirstOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs<1>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createC4_5_Diagonals();
        this->init();
    }
    ~C4_5_DiagonalsFirstOrder() {
#ifdef DEBUG
        std::cout << "in C4_5_DiagonalsFirstOrder destructor" << std::endl;
#endif
    }

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<C4_5_DiagonalsFirstOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_C4_5; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_FIRST_ORDER; }
    std::string toString() const override { return "C4_5_DiagonalsFirstOrder"; }
};
class A6_1_DiagonalsFirstOrder : public MatrixCompactDerivs<1> {
public:
    A6_1_DiagonalsFirstOrder(
        unsigned int ele_order,
        const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs<1>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createA6_1_Diagonals();
        this->init();
    }

    ~A6_1_DiagonalsFirstOrder() {
    #ifdef DEBUG
        std::cout << "in A6_1_DiagonalsFirstOrder destructor" << std::endl;
    #endif
    }

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<A6_1_DiagonalsFirstOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_A6_1; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_FIRST_ORDER; }
    std::string toString() const override { return "A6_1_DiagonalsFirstOrder"; }
};

// =======================================
// First Derivatives: A6_2
// =======================================
class A6_2_DiagonalsFirstOrder : public MatrixCompactDerivs<1> {
public:
    A6_2_DiagonalsFirstOrder(
        unsigned int ele_order,
        const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs<1>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createA6_2_Diagonals();
        this->init();
    }

    ~A6_2_DiagonalsFirstOrder() {
    #ifdef DEBUG
        std::cout << "in A6_2_DiagonalsFirstOrder destructor" << std::endl;
    #endif
    }

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<A6_2_DiagonalsFirstOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_A6_2; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_FIRST_ORDER; }
    std::string toString() const override { return "A6_2_DiagonalsFirstOrder"; }
};

// =======================================
// First Derivatives: A6_3
// =======================================
class A6_3_DiagonalsFirstOrder : public MatrixCompactDerivs<1> {
public:
    A6_3_DiagonalsFirstOrder(
        unsigned int ele_order,
        const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>())
        : MatrixCompactDerivs<1>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = createA6_3_Diagonals();
        this->init();
    }

    ~A6_3_DiagonalsFirstOrder() {
    #ifdef DEBUG
        std::cout << "in A6_3_DiagonalsFirstOrder destructor" << std::endl;
    #endif
    }

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<A6_3_DiagonalsFirstOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_A6_3; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_FIRST_ORDER; }
    std::string toString() const override { return "A6_3_DiagonalsFirstOrder"; }
};
//Second Derivatives:
class TwoDerivs2B4_1_SecondOrder : public MatrixCompactDerivs<2> {
public:
    TwoDerivs2B4_1_SecondOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>(),
        const std::vector<double>& coeffs_in = std::vector<double>())
        : MatrixCompactDerivs{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = create2B4_1_Diagonals();
        this->init();
    }

    ~TwoDerivs2B4_1_SecondOrder() override = default;

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<TwoDerivs2B4_1_SecondOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_2B4_1; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_SECOND_ORDER; }
    std::string toString() const override { return "TwoDerivs2B4_1_SecondOrder"; }
};

// -------------- 2A6_1 ----------------
class TwoDerivs2A6_1_SecondOrder : public MatrixCompactDerivs<2> {
public:
    TwoDerivs2A6_1_SecondOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>(),
        const std::vector<double>& coeffs_in = std::vector<double>())
        : MatrixCompactDerivs<2>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = create2A6_1_Diagonals();
        this->init();
    }

    ~TwoDerivs2A6_1_SecondOrder() override = default;

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<TwoDerivs2A6_1_SecondOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_2A6_1; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_SECOND_ORDER; }
    std::string toString() const override { return "TwoDerivs2A6_1_SecondOrder"; }
};

// -------------- 2B6_1 ----------------
class TwoDerivs2B6_1_SecondOrder : public MatrixCompactDerivs<2> {
public:
    TwoDerivs2B6_1_SecondOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>(),
        const std::vector<double>& coeffs_in = std::vector<double>())
        : MatrixCompactDerivs<2>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = create2B6_1_Diagonals();
        this->init();
    }

    ~TwoDerivs2B6_1_SecondOrder() override = default;

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<TwoDerivs2B6_1_SecondOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_2B6_1; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_SECOND_ORDER; }
    std::string toString() const override { return "TwoDerivs2B6_1_SecondOrder"; }
};

// -------------- 2B6_2 ----------------
class TwoDerivs2B6_2_SecondOrder : public MatrixCompactDerivs<2> {
public:
    TwoDerivs2B6_2_SecondOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>(),
        const std::vector<double>& coeffs_in = std::vector<double>())
        : MatrixCompactDerivs<2>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = create2B6_2_Diagonals();
        this->init();
    }

    ~TwoDerivs2B6_2_SecondOrder() override = default;

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<TwoDerivs2B6_2_SecondOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_2B6_2; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_SECOND_ORDER; }
    std::string toString() const override { return "TwoDerivs2B6_2_SecondOrder"; }
};

// -------------- 2B6_3 ----------------
class TwoDerivs2B6_3_SecondOrder : public MatrixCompactDerivs<2> {
public:
    TwoDerivs2B6_3_SecondOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>(),
        const std::vector<double>& coeffs_in = std::vector<double>())
        : MatrixCompactDerivs<2>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = create2B6_3_Diagonals();
        this->init();
    }

    ~TwoDerivs2B6_3_SecondOrder() override = default;

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<TwoDerivs2B6_3_SecondOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_2B6_3; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_SECOND_ORDER; }
    std::string toString() const override { return "TwoDerivs2B6_3_SecondOrder"; }
};

// -------------- 2B6_4 ----------------
class TwoDerivs2B6_4_SecondOrder : public MatrixCompactDerivs<2> {
public:
    TwoDerivs2B6_4_SecondOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>(),
        const std::vector<double>& coeffs_in = std::vector<double>())
        : MatrixCompactDerivs<2>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = create2B6_4_Diagonals();
        this->init();
    }

    ~TwoDerivs2B6_4_SecondOrder() override = default;

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<TwoDerivs2B6_4_SecondOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_2B6_4; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_SECOND_ORDER; }
    std::string toString() const override { return "TwoDerivs2B6_4_SecondOrder"; }
};

// -------------- 2B6_5 ----------------
class TwoDerivs2B6_5_SecondOrder : public MatrixCompactDerivs<2> {
public:
    TwoDerivs2B6_5_SecondOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>(),
        const std::vector<double>& coeffs_in = std::vector<double>())
        : MatrixCompactDerivs<2>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = create2B6_5_Diagonals();
        this->init();
    }

    ~TwoDerivs2B6_5_SecondOrder() override = default;

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<TwoDerivs2B6_5_SecondOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_2B6_5; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_SECOND_ORDER; }
    std::string toString() const override { return "TwoDerivs2B6_5_SecondOrder"; }
};

// -------------- 2B6_6 ----------------
class TwoDerivs2B6_6_SecondOrder : public MatrixCompactDerivs<2> {
public:
    TwoDerivs2B6_6_SecondOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>(),
        const std::vector<double>& coeffs_in = std::vector<double>())
        : MatrixCompactDerivs<2>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = create2B6_6_Diagonals();
        this->init();
    }

    ~TwoDerivs2B6_6_SecondOrder() override = default;

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<TwoDerivs2B6_6_SecondOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_2B6_6; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_SECOND_ORDER; }
    std::string toString() const override { return "TwoDerivs2B6_6_SecondOrder"; }
};

// -------------- 2B6_7 ----------------
class TwoDerivs2B6_7_SecondOrder : public MatrixCompactDerivs<2> {
public:
    TwoDerivs2B6_7_SecondOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>(),
        const std::vector<double>& coeffs_in = std::vector<double>())
        : MatrixCompactDerivs<2>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = create2B6_7_Diagonals();
        this->init();
    }

    ~TwoDerivs2B6_7_SecondOrder() override = default;

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<TwoDerivs2B6_7_SecondOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_2B6_7; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_SECOND_ORDER; }
    std::string toString() const override { return "TwoDerivs2B6_7_SecondOrder"; }
};

// -------------- 2B6_8 ----------------
class TwoDerivs2B6_8_SecondOrder : public MatrixCompactDerivs<2> {
public:
    TwoDerivs2B6_8_SecondOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>(),
        const std::vector<double>& coeffs_in = std::vector<double>())
        : MatrixCompactDerivs<2>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = create2B6_8_Diagonals();
        this->init();
    }

    ~TwoDerivs2B6_8_SecondOrder() override = default;

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<TwoDerivs2B6_8_SecondOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_2B6_8; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_SECOND_ORDER; }
    std::string toString() const override { return "TwoDerivs2B6_8_SecondOrder"; }
};

// -------------- 2B6_9 ----------------
class TwoDerivs2B6_9_SecondOrder : public MatrixCompactDerivs<2> {
public:
    TwoDerivs2B6_9_SecondOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>(),
        const std::vector<double>& coeffs_in = std::vector<double>())
        : MatrixCompactDerivs<2>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = create2B6_9_Diagonals();
        this->init();
    }

    ~TwoDerivs2B6_9_SecondOrder() override = default;

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<TwoDerivs2B6_9_SecondOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_2B6_9; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_SECOND_ORDER; }
    std::string toString() const override { return "TwoDerivs2B6_9_SecondOrder"; }
};

// -------------- 2C6_1 ----------------
class TwoDerivs2C6_1_SecondOrder : public MatrixCompactDerivs<2> {
public:
    TwoDerivs2C6_1_SecondOrder(
        unsigned int ele_order, const std::string& in_matrix_filter = "none",
        const std::vector<double>& in_filter_coeffs = std::vector<double>(),
        const std::vector<double>& coeffs_in = std::vector<double>())
        : MatrixCompactDerivs<2>{ele_order, in_matrix_filter, in_filter_coeffs} {
        diagEntries = create2C6_1_Diagonals();
        this->init();
    }

    ~TwoDerivs2C6_1_SecondOrder() override = default;

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<TwoDerivs2C6_1_SecondOrder>(*this);
    }

    DerivType getDerivType() const override { return DerivType::D_2C6_1; }
    DerivOrder getDerivOrder() const override { return DerivOrder::D_SECOND_ORDER; }
    std::string toString() const override { return "TwoDerivs2C6_1_SecondOrder"; }
};


}  // namespace dendroderivs
