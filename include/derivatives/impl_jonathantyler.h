#pragma once

#include "derivatives/derivs_banded.h"
#include "derivatives/derivs_matrixonly.h"

namespace dendroderivs {

MatrixDiagonalEntries* createJTT4DiagonalsFirstOrder();
MatrixDiagonalEntries* createJTT4DiagonalsSecondOrder();
MatrixDiagonalEntries* createJTT6DiagonalsFirstOrder();
MatrixDiagonalEntries* createJTT6DiagonalsSecondOrder();

/**
 *  Tridiagonal, 4th-order compact derivative from the thesis of Jonathan Tyler.
 */
class JonathanTyler_JTT4_FirstOrder : public MatrixCompactDerivs<1> {
   public:
    template <typename... Args>
    JonathanTyler_JTT4_FirstOrder(unsigned int ele_order, Args&&...)
        : MatrixCompactDerivs{ele_order} {
        diagEntries = createJTT4DiagonalsFirstOrder();

        this->init();
    }

    ~JonathanTyler_JTT4_FirstOrder() {}

    DerivType getDerivType() const override { return DerivType::D_JTT4; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_FIRST_ORDER;
    }

    std::string toString() const override {
        return "JonathanTyler_JTT4_FirstOrder";
    }
};

/**
 *  Tridiagonal, 4th-order compact derivative from the thesis of Jonathan Tyler.
 */
class JonathanTyler_JTT4_FirstOrder_Banded : public BandedCompactDerivs {
   public:
    template <typename... Args>
    JonathanTyler_JTT4_FirstOrder_Banded(unsigned int ele_order, Args&&...)
        : BandedCompactDerivs{ele_order} {
#ifdef DEBUG
        std::cout << "entered JTT4 constructor" << std::endl;
#endif
        kVals = new BandedMatrixDiagonalWidths{
            1,  // p1kl
            1,  // p1ku
            3,  // q1kl
            3,  // q1ku
        };

        diagEntries = createJTT4DiagonalsFirstOrder();

#ifdef DEBUG
        std::cout << "entering init method in JTT4 constructor" << std::endl;
#endif

        // build the matrices, allocate the data, etc
        init(kVals, diagEntries);

#ifdef DEBUG
        std::cout << "exiting JTT4 constructor" << std::endl;
#endif
    }
    ~JonathanTyler_JTT4_FirstOrder_Banded() {}

    DerivType getDerivType() const override { return DerivType::D_JTT4; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_FIRST_ORDER;
    }

    std::string toString() const override {
        return "JonathanTyler_JTT4_FirstOrder_Banded";
    }
};

/**
 *  Tridiagonal, 4th-order compact derivative from the thesis of Jonathan Tyler.
 */
class JonathanTyler_JTT4_SecondOrder : public MatrixCompactDerivs<2> {
   public:
    template <typename... Args>
    JonathanTyler_JTT4_SecondOrder(unsigned int ele_order, Args&&...)
        : MatrixCompactDerivs{ele_order} {
        diagEntries = createJTT4DiagonalsSecondOrder();

        this->init();
    }

    ~JonathanTyler_JTT4_SecondOrder() {}

    DerivType getDerivType() const override { return DerivType::D_JTT4; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_SECOND_ORDER;
    }

    std::string toString() const override {
        return "JonathanTyler_JTT4_SecondOrder";
    }
};

/**
 *  Tridiagonal, 4th-order compact derivative from the thesis of Jonathan Tyler.
 */
class JonathanTyler_JTT4_SecondOrder_Banded : public BandedCompactDerivs {
   public:
    template <typename... Args>
    JonathanTyler_JTT4_SecondOrder_Banded(unsigned int ele_order, Args&&...)
        : BandedCompactDerivs{ele_order} {
#ifdef DEBUG
        std::cout << "entered JTT4 constructor" << std::endl;
#endif
        kVals = new BandedMatrixDiagonalWidths{
            1,  // p2kl
            1,  // p2ku
            4,  // q2kl
            4   // q2ku
        };

        diagEntries = createJTT4DiagonalsSecondOrder();

#ifdef DEBUG
        std::cout << "entering init method in JTT4 constructor" << std::endl;
#endif

        // build the matrices, allocate the data, etc
        init(kVals, diagEntries);

#ifdef DEBUG
        std::cout << "exiting JTT4 constructor" << std::endl;
#endif
    }

    ~JonathanTyler_JTT4_SecondOrder_Banded() {}

    DerivType getDerivType() const override { return DerivType::D_JTT4; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_SECOND_ORDER;
    }

    std::string toString() const override {
        return "JonathanTyler_JTT4_FirstOrder_Banded";
    }
};

/**
 *  Tridiagonal, 6th-order compact derivative from the thesis of Jonathan Tyler.
 */
class JonathanTyler_JTT6_FirstOrder : public MatrixCompactDerivs<1> {
   public:
    template <typename... Args>
    JonathanTyler_JTT6_FirstOrder(unsigned int ele_order, Args&&...)
        : MatrixCompactDerivs{ele_order} {
        diagEntries = createJTT6DiagonalsFirstOrder();

        this->init();
    }

    ~JonathanTyler_JTT6_FirstOrder() {}

    DerivType getDerivType() const override { return DerivType::D_JTT6; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_FIRST_ORDER;
    }

    std::string toString() const override {
        return "JonathanTyler_JTT6_FirstOrder";
    }
};

/**
 *  Tridiagonal, 4th-order compact derivative from the thesis of Jonathan Tyler.
 */
class JonathanTyler_JTT6_FirstOrder_Banded : public BandedCompactDerivs {
   public:
    template <typename... Args>
    JonathanTyler_JTT6_FirstOrder_Banded(unsigned int ele_order, Args&&...)
        : BandedCompactDerivs{ele_order} {
#ifdef DEBUG
        std::cout << "entered JTT4 constructor" << std::endl;
#endif
        kVals = new BandedMatrixDiagonalWidths{
            1,  // p1kl
            1,  // p1ku
            5,  // q1kl
            5,  // q1ku
        };

        diagEntries = createJTT6DiagonalsFirstOrder();

#ifdef DEBUG
        std::cout << "entering init method in JTT4 constructor" << std::endl;
#endif

        // build the matrices, allocate the data, etc
        init(kVals, diagEntries);

#ifdef DEBUG
        std::cout << "exiting JTT4 constructor" << std::endl;
#endif
    }

    ~JonathanTyler_JTT6_FirstOrder_Banded() {}

    DerivType getDerivType() const override { return DerivType::D_JTT6; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_FIRST_ORDER;
    }

    std::string toString() const override {
        return "JonathanTyler_JTT6_FirstOrder_Banded";
    }
};

/**
 *  Tridiagonal, 6th-order compact derivative from the thesis of Jonathan Tyler.
 */
class JonathanTyler_JTT6_SecondOrder : public MatrixCompactDerivs<2> {
   public:
    template <typename... Args>
    JonathanTyler_JTT6_SecondOrder(unsigned int ele_order, Args&&...)
        : MatrixCompactDerivs{ele_order} {
        diagEntries = createJTT6DiagonalsSecondOrder();

        this->init();
    }
    ~JonathanTyler_JTT6_SecondOrder() {}

    DerivType getDerivType() const override { return DerivType::D_JTT6; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_SECOND_ORDER;
    }

    std::string toString() const override {
        return "JonathanTyler_JTT5_SecondOrder";
    }
};

/**
 *  Tridiagonal, 4th-order compact derivative from the thesis of Jonathan Tyler.
 */
class JonathanTyler_JTT6_SecondOrder_Banded : public BandedCompactDerivs {
   public:
    template <typename... Args>
    JonathanTyler_JTT6_SecondOrder_Banded(unsigned int ele_order, Args&&...)
        : BandedCompactDerivs{ele_order} {
#ifdef DEBUG
        std::cout << "entered JTT4 constructor" << std::endl;
#endif
        kVals = new BandedMatrixDiagonalWidths{
            1,  // p2kl
            1,  // p2ku
            6,  // q2kl
            6   // q2ku
        };

        diagEntries = createJTT6DiagonalsSecondOrder();

#ifdef DEBUG
        std::cout << "entering init method in JTT4 constructor" << std::endl;
#endif

        // build the matrices, allocate the data, etc
        init(kVals, diagEntries);

#ifdef DEBUG
        std::cout << "exiting JTT4 constructor" << std::endl;
#endif
    }

    ~JonathanTyler_JTT6_SecondOrder_Banded() {}

    DerivType getDerivType() const override { return DerivType::D_JTT6; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_SECOND_ORDER;
    }

    std::string toString() const override {
        return "JonathanTyler_JTT4_SecondOrder_Banded";
    }
};

}  // namespace dendroderivs
