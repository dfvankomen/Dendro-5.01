#pragma once

#include "derivatives/derivs_banded.h"
#include "derivatives/derivs_matrixonly.h"

namespace dendroderivs {

std::vector<double> clean_coeffs(const std::vector<double>& coeffs_in,
                                 unsigned int max_coeffs = 3);

MatrixDiagonalEntries* BYUDerivsT4R3DiagonalsFirstOrder(
    const std::vector<double>& D_coeffs);

MatrixDiagonalEntries* BYUDerivsT4R3DiagonalsSecondOrder(
    const std::vector<double>& D_coeffs);

MatrixDiagonalEntries* BYUDerivsT6R3DiagonalsFirstOrder(
    const std::vector<double>& D_coeffs);

MatrixDiagonalEntries* BYUDerivsT6R3DiagonalsSecondOrder(
    const std::vector<double>& D_coeffs);

MatrixDiagonalEntries* BYUDerivsT6R4DiagonalsFirstOrder(
    const std::vector<double>& D_coeffs);

MatrixDiagonalEntries* BYUDerivsT6R4DiagonalsSecondOrder(
    const std::vector<double>& D_coeffs);

MatrixDiagonalEntries* BYUDerivsP6R3DiagonalsFirstOrder(
    const std::vector<double>& D_coeffs);

MatrixDiagonalEntries* BYUDerivsP6R3DiagonalsSecondOrder(
    const std::vector<double>& D_coeffs);

MatrixDiagonalEntries* BYUDerivsP8R4DiagonalsSecondOrder(
    const std::vector<double>& D_coeffs);

MatrixDiagonalEntries* BYUDerivsP8R4DiagonalsFirstOrder(
    const std::vector<double>& D_coeffs);

// TODO: banded matrix implementations

class BYUDerivsT4_R3_FirstOrder : public MatrixCompactDerivs<1> {
   private:
    std::vector<double> D_coeffs_;

    static constexpr unsigned int n_D_coeffs_ = 4;

   public:
    BYUDerivsT4_R3_FirstOrder(
        unsigned int ele_order,
        const std::vector<double>& coeffs_in = std::vector<double>())
        : MatrixCompactDerivs{ele_order} {
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

class BYUDerivsT4_R3_SecondOrder : public MatrixCompactDerivs<2> {
   private:
    std::vector<double> D_coeffs_;

    static constexpr unsigned int n_D_coeffs_ = 4;

   public:
    BYUDerivsT4_R3_SecondOrder(
        unsigned int ele_order,
        const std::vector<double>& coeffs_in = std::vector<double>())
        : MatrixCompactDerivs{ele_order} {
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

class BYUDerivsT6_R3_FirstOrder : public MatrixCompactDerivs<1> {
   private:
    std::vector<double> D_coeffs_;

    static constexpr unsigned int n_D_coeffs_ = 4;

   public:
    BYUDerivsT6_R3_FirstOrder(
        unsigned int ele_order,
        const std::vector<double>& coeffs_in = std::vector<double>())
        : MatrixCompactDerivs{ele_order} {
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

class BYUDerivsT6_R3_SecondOrder : public MatrixCompactDerivs<2> {
   private:
    std::vector<double> D_coeffs_;

    static constexpr unsigned int n_D_coeffs_ = 4;

   public:
    BYUDerivsT6_R3_SecondOrder(
        unsigned int ele_order,
        const std::vector<double>& coeffs_in = std::vector<double>())
        : MatrixCompactDerivs{ele_order} {
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
        unsigned int ele_order,
        const std::vector<double>& coeffs_in = std::vector<double>())
        : MatrixCompactDerivs{ele_order} {
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

class BYUDerivsT6_R4_SecondOrder : public MatrixCompactDerivs<2> {
   private:
    std::vector<double> D_coeffs_;

    static constexpr unsigned int n_D_coeffs_ = 5;

   public:
    BYUDerivsT6_R4_SecondOrder(
        unsigned int ele_order,
        const std::vector<double>& coeffs_in = std::vector<double>())
        : MatrixCompactDerivs{ele_order} {
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
        unsigned int ele_order,
        const std::vector<double>& coeffs_in = std::vector<double>())
        : MatrixCompactDerivs{ele_order} {
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
        unsigned int ele_order,
        const std::vector<double>& coeffs_in = std::vector<double>())
        : MatrixCompactDerivs{ele_order} {
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

    DerivType getDerivType() const override { return DerivType::D_BYUT6; }
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
        unsigned int ele_order,
        const std::vector<double>& coeffs_in = std::vector<double>())
        : MatrixCompactDerivs{ele_order} {
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
        unsigned int ele_order,
        const std::vector<double>& coeffs_in = std::vector<double>())
        : MatrixCompactDerivs{ele_order} {
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

}  // namespace dendroderivs
