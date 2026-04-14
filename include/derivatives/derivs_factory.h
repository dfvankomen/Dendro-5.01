#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "derivatives.h"
#include "derivatives/derivs_explicit.h"
#include "derivatives/filt_kodiss_explicit.h"
#include "derivatives/impl_boris.h"
#include "derivatives/impl_bradylivescu.h"
#include "derivatives/impl_byuderivs.h"
#include "derivatives/impl_explicitmatrix.h"
#include "derivatives/impl_hybrid_approaches.h"
#include "derivatives/impl_jonathantyler.h"
#include "derivatives/impl_kimderivs.h"
#include "filters.h"

namespace dendroderivs {

// all factory creators take the full parameter set — classes that don't
// need certain args just ignore them
using DerivCreatorFn = std::unique_ptr<Derivs> (*)(
    unsigned int ele_order, const std::string &filterType,
    const std::vector<double> &filterCoeffs,
    const std::vector<double> &coeffs_in, unsigned int matrixID);

// helper templates to cut down the boilerplate for each constructor pattern.
// pattern A: explicit stencils that only need ele_order
template <typename T>
std::unique_ptr<Derivs> make_explicit(unsigned int eo, const std::string &,
                                      const std::vector<double> &,
                                      const std::vector<double> &,
                                      unsigned int) {
    return std::make_unique<T>(eo);
}

// pattern B: matrix-only methods (ele_order + filter)
template <typename T>
std::unique_ptr<Derivs> make_matrix(unsigned int eo, const std::string &ft,
                                    const std::vector<double> &fc,
                                    const std::vector<double> &,
                                    unsigned int) {
    return std::make_unique<T>(eo, ft, fc);
}

// pattern C: coefficient-accepting methods (ele_order + filter + coeffs)
template <typename T>
std::unique_ptr<Derivs> make_with_coeffs(unsigned int eo, const std::string &ft,
                                         const std::vector<double> &fc,
                                         const std::vector<double> &coeffs,
                                         unsigned int) {
    return std::make_unique<T>(eo, ft, fc, coeffs);
}

// generic factory helpers that construct GenericMatrixDerivs directly,
// eliminating the need for per-scheme class definitions.
// the diagonal function pointer and metadata are template parameters.

template <unsigned int Order, DiagCreatorFn DiagFn, DerivType DType>
std::unique_ptr<Derivs> make_generic(unsigned int eo, const std::string &ft,
                                     const std::vector<double> &fc,
                                     const std::vector<double> &,
                                     unsigned int) {
    return std::make_unique<GenericMatrixDerivs<Order>>(
        DiagFn, DType, std::string("Generic_") + std::to_string(static_cast<int>(DType)),
        eo, ft, fc);
}

template <unsigned int Order, DiagCreatorWithCoeffsFn DiagFn, DerivType DType,
          unsigned int NCoeffs>
std::unique_ptr<Derivs> make_generic_coeffs(unsigned int eo,
                                            const std::string &ft,
                                            const std::vector<double> &fc,
                                            const std::vector<double> &coeffs,
                                            unsigned int) {
    return std::make_unique<GenericMatrixDerivsWithCoeffs<Order>>(
        DiagFn, DType,
        std::string("Generic_") + std::to_string(static_cast<int>(DType)),
        NCoeffs, eo, ft, fc, coeffs);
}

// pattern D: boris/BL methods that also need matrixID
template <typename T>
std::unique_ptr<Derivs> make_with_coeffs_and_id(unsigned int eo,
                                                const std::string &ft,
                                                const std::vector<double> &fc,
                                                const std::vector<double> &coeffs,
                                                unsigned int matID) {
    return std::make_unique<T>(eo, ft, fc, coeffs, matID);
}

// ============================================================
// first-order registry
// ============================================================
inline const std::unordered_map<std::string, DerivCreatorFn> &
get_first_order_registry() {
    static const std::unordered_map<std::string, DerivCreatorFn> reg = {
        // explicit stencils
        {"E4", make_explicit<ExplicitDerivsO4_DX>},
        {"E6", make_explicit<ExplicitDerivsO6_DX>},
        {"E8", make_explicit<ExplicitDerivsO8_DX>},

        // explicit matrix
        {"E4Matrix", make_generic<1, createE4DiagonalsFirstOrder, DerivType::D_E4>},
        {"E6Matrix", make_generic<1, createE6DiagonalsFirstOrder, DerivType::D_E6>},
        {"E8Matrix", make_generic<1, createE8DiagonalsFirstOrder, DerivType::D_E8>},

        // kim family — uses GenericMatrixDerivs, no per-scheme class needed
        {"KIM4", make_generic<1, createKimDiagonals, DerivType::D_K4>},
        {"KIM407", make_generic<1, createKim07Diagonals, DerivType::D_K4_07>},
        {"KIM416", make_generic<1, createKim16Diagonals, DerivType::D_K4_16>},
        {"KIMBYU_1", make_generic<1, createBYU_KIM_1_Diagonals, DerivType::D_KIMBYU_1>},
        {"KIMBYU_2", make_generic<1, createBYU_KIM_2_Diagonals, DerivType::D_KIMBYU_2>},
        {"KIMBYU_3", make_generic<1, createBYU_KIM_3_Diagonals, DerivType::D_KIMBYU_3>},
        {"KIMBYU_4", make_generic<1, createBYU_KIM_4_Diagonals, DerivType::D_KIMBYU_4>},
        {"KIMBYU_5", make_generic<1, createBYU_KIM_5_Diagonals, DerivType::D_KIMBYU_5>},
        {"KIMBYU_6", make_generic<1, createBYU_KIM_6_Diagonals, DerivType::D_KIMBYU_6>},
        {"KIMBYU_7", make_generic<1, createBYU_KIM_7_Diagonals, DerivType::D_KIMBYU_7>},
        {"KIMBYU_8", make_generic<1, createBYU_KIM_8_Diagonals, DerivType::D_KIMBYU_8>},
        {"KIMBYU_9", make_generic<1, createBYU_KIM_9_Diagonals, DerivType::D_KIMBYU_9>},
        {"KIMBYU_10", make_generic<1, createBYU_KIM_10_Diagonals, DerivType::D_KIMBYU_10>},
        {"KIMBYU_11", make_generic<1, createBYU_KIM_11_Diagonals, DerivType::D_KIMBYU_11>},
        {"KIMBYU_12", make_generic<1, createBYU_KIM_12_Diagonals, DerivType::D_KIMBYU_12>},
        {"KIMBYU_13", make_generic<1, createBYU_KIM_13_Diagonals, DerivType::D_KIMBYU_13>},
        {"KIMBYU_14", make_generic<1, createBYU_KIM_14_Diagonals, DerivType::D_KIMBYU_14>},
        {"KIMBYU_15", make_generic<1, createBYU_KIM_15_Diagonals, DerivType::D_KIMBYU_15>},
        {"KIMBYU_16", make_generic<1, createBYU_KIM_16_Diagonals, DerivType::D_KIMBYU_16>},
        {"KIMBYU_17", make_generic<1, createBYU_KIM_17_Diagonals, DerivType::D_KIMBYU_17>},
        {"KIMBYU_18", make_generic<1, createBYU_KIM_18_Diagonals, DerivType::D_KIMBYU_18>},
        {"KIMBYU_19", make_generic<1, createBYU_KIM_19_Diagonals, DerivType::D_KIMBYU_19>},

        // C4 family
        {"C4_1", make_generic<1, createC4_1_Diagonals, DerivType::D_C4_1>},
        {"C4_2", make_generic<1, createC4_2_Diagonals, DerivType::D_C4_2>},
        {"C4_3", make_generic<1, createC4_3_Diagonals, DerivType::D_C4_3>},
        {"C4_4", make_generic<1, createC4_4_Diagonals, DerivType::D_C4_4>},
        {"C4_5", make_generic<1, createC4_5_Diagonals, DerivType::D_C4_5>},

        // A4 family
        {"A4_1", make_generic<1, createA4_1_Diagonals, DerivType::D_A4_1>},
        {"A4_2", make_generic<1, createA4_2_Diagonals, DerivType::D_A4_2>},
        {"A4_3", make_generic<1, createA4_3_Diagonals, DerivType::D_A4_3>},
        {"A4_4", make_generic<1, createA4_4_Diagonals, DerivType::D_A4_4>},
        {"A4_5", make_generic<1, createA4_5_Diagonals, DerivType::D_A4_5>},
        {"A4_6", make_generic<1, createA4_6_Diagonals, DerivType::D_A4_6>},
        {"A4_7", make_generic<1, createA4_7_Diagonals, DerivType::D_A4_7>},
        {"A4_8", make_generic<1, createA4_8_Diagonals, DerivType::D_A4_8>},
        {"A4_9", make_generic<1, createA4_9_Diagonals, DerivType::D_A4_9>},
        {"A4_10", make_generic<1, createA4_10_Diagonals, DerivType::D_A4_10>},
        {"A4_11", make_generic<1, createA4_11_Diagonals, DerivType::D_A4_11>},
        {"A4_12", make_generic<1, createA4_12_Diagonals, DerivType::D_A4_12>},
        {"A4_13", make_generic<1, createA4_13_Diagonals, DerivType::D_A4_13>},
        {"A4_14", make_generic<1, createA4_14_Diagonals, DerivType::D_A4_14>},
        {"A4_15", make_generic<1, createA4_15_Diagonals, DerivType::D_A4_15>},
        {"A4_16", make_generic<1, createA4_16_Diagonals, DerivType::D_A4_16>},
        {"A4_17", make_generic<1, createA4_17_Diagonals, DerivType::D_A4_17>},
        {"A4_18", make_generic<1, createA4_18_Diagonals, DerivType::D_A4_18>},
        {"A4_19", make_generic<1, createA4_19_Diagonals, DerivType::D_A4_19>},
        {"A4_20", make_generic<1, createA4_20_Diagonals, DerivType::D_A4_20>},

        // B4 family
        {"B4_1", make_generic<1, createB4_1_Diagonals, DerivType::D_B4_1>},
        {"B4_2", make_generic<1, createB4_2_Diagonals, DerivType::D_B4_2>},

        // A6 family
        {"A6_1", make_generic<1, createA6_1_Diagonals, DerivType::D_A6_1>},
        {"A6_2", make_generic<1, createA6_2_Diagonals, DerivType::D_A6_2>},
        {"A6_3", make_generic<1, createA6_3_Diagonals, DerivType::D_A6_3>},
        {"A6_4", make_generic<1, createA6_4_Diagonals, DerivType::D_A6_4>},
        {"A6_5", make_generic<1, createA6_5_Diagonals, DerivType::D_A6_5>},
        {"A6_6", make_generic<1, createA6_6_Diagonals, DerivType::D_A6_6>},
        {"A6_7", make_generic<1, createA6_7_Diagonals, DerivType::D_A6_7>},
        {"A6_8", make_generic<1, createA6_8_Diagonals, DerivType::D_A6_8>},

        // C6 family
        {"C6_1", make_generic<1, createC6_1_Diagonals, DerivType::D_C6_1>},
        {"C6_2", make_generic<1, createC6_2_Diagonals, DerivType::D_C6_2>},
        {"C6_3", make_generic<1, createC6_3_Diagonals, DerivType::D_C6_3>},
        {"C6_4", make_generic<1, createC6_4_Diagonals, DerivType::D_C6_4>},
        {"C6_5", make_generic<1, createC6_5_Diagonals, DerivType::D_C6_5>},
        {"C6_6", make_generic<1, createC6_6_Diagonals, DerivType::D_C6_6>},
        {"C6_7", make_generic<1, createC6_7_Diagonals, DerivType::D_C6_7>},
        {"C6_8", make_generic<1, createC6_8_Diagonals, DerivType::D_C6_8>},
        {"C6_9", make_generic<1, createC6_9_Diagonals, DerivType::D_C6_9>},
        {"C6_10", make_generic<1, createC6_10_Diagonals, DerivType::D_C6_10>},
        {"C6_11", make_generic<1, createC6_11_Diagonals, DerivType::D_C6_11>},
        {"C6_12", make_generic<1, createC6_12_Diagonals, DerivType::D_C6_12>},
        {"C6_13", make_generic<1, createC6_13_Diagonals, DerivType::D_C6_13>},
        {"C6_14", make_generic<1, createC6_14_Diagonals, DerivType::D_C6_14>},
        {"C6_15", make_generic<1, createC6_15_Diagonals, DerivType::D_C6_15>},
        {"C6_16", make_generic<1, createC6_16_Diagonals, DerivType::D_C6_16>},

        // UC6 family
        {"UC6_1", make_generic<1, createUC6_1_Diagonals, DerivType::D_UC6_1>},
        {"UC6_2", make_generic<1, createUC6_2_Diagonals, DerivType::D_UC6_2>},
        {"UC6_3", make_generic<1, createUC6_3_Diagonals, DerivType::D_UC6_3>},
        {"UC6_4", make_generic<1, createUC6_4_Diagonals, DerivType::D_UC6_4>},

        // UA4 family
        {"UA4_4", make_generic<1, createUA4_4_Diagonals, DerivType::D_UA4_4>},
        {"UA4_5", make_generic<1, createUA4_5_Diagonals, DerivType::D_UA4_5>},
        {"UA4_6", make_generic<1, createUA4_6_Diagonals, DerivType::D_UA4_6>},

        // jonathan tyler compact schemes
        {"JTT4", make_generic<1, createJTT4DiagonalsFirstOrder, DerivType::D_JTT4>},
        {"JTT6", make_generic<1, createJTT6DiagonalsFirstOrder, DerivType::D_JTT6>},
        {"JTP6", make_generic<1, createJTP6DiagonalsFirstOrder, DerivType::D_JTP6>},
        {"JTT4Banded", make_explicit<JonathanTyler_JTT4_FirstOrder_Banded>},
        {"JTT6Banded", make_explicit<JonathanTyler_JTT6_FirstOrder_Banded>},
        {"JTP6Banded", make_explicit<JonathanTyler_JTP6_FirstOrder_Banded>},

        // brady-livescu (uses coeffs + matrixID)
        {"BL6", make_with_coeffs_and_id<BradyLivescu_BL6_FirstOrder>},

        // boris (uses coeffs + matrixID as closure ID)
        {"BorisO4", make_with_coeffs_and_id<Boris_BorisO4_FirstOrder>},
        {"BorisO6", make_with_coeffs_and_id<Boris_BorisO6_FirstOrder>},
        {"BorisO6Eta", make_with_coeffs_and_id<Boris_BorisO6Eta_FirstOrder>},

        // BYU tridiagonal (uses coeffs via GenericMatrixDerivsWithCoeffs)
        {"BYUT4", make_generic_coeffs<1, BYUDerivsT4R3DiagonalsFirstOrder, DerivType::D_BYUT4, 4>},
        {"BYUT4R1", make_generic_coeffs<1, BYUDerivsT4R1DiagonalsFirstOrder, DerivType::D_BYUT4R1, 2>},
        {"BYUT4R2", make_generic_coeffs<1, BYUDerivsT4R2DiagonalsFirstOrder, DerivType::D_BYUT4R2, 3>},
        {"BYUT4R4", make_generic_coeffs<1, BYUDerivsT4R42DiagonalsFirstOrder, DerivType::D_BYUT4R4, 5>},
        {"BYUT6", make_generic_coeffs<1, BYUDerivsT6R4DiagonalsFirstOrder, DerivType::D_BYUT6, 5>},
        {"BYUT62", make_generic_coeffs<1, BYUDerivsT6R42DiagonalsFirstOrder, DerivType::D_BYUT6R4, 5>},
        {"BYUT64R3", make_generic_coeffs<1, BYUDerivsT64R3DiagonalsFirstOrder, DerivType::D_BYUT64R3, 4>},
        {"BYUT6R2", make_generic_coeffs<1, BYUDerivsT6R2DiagonalsFirstOrder, DerivType::D_BYUT6R2, 3>},
        {"BYUT6R3", make_generic_coeffs<1, BYUDerivsT6R3DiagonalsFirstOrder, DerivType::D_BYUT6R3, 4>},

        // BYU pentadiagonal (uses coeffs)
        {"BYUP6", make_generic_coeffs<1, BYUDerivsP6R3DiagonalsFirstOrder, DerivType::D_BYUP6, 7>},
        {"BYUP6R2", make_generic_coeffs<1, BYUDerivsP6R2DiagonalsFirstOrder, DerivType::D_BYUP6R2, 3>},
        {"BYUP6R3", make_generic_coeffs<1, BYUDerivsP6R32DiagonalsFirstOrder, DerivType::D_BYUP6R3, 4>},
        {"BYUP8", make_generic_coeffs<1, BYUDerivsP8R4DiagonalsFirstOrder, DerivType::D_BYUP8, 9>},
    };
    return reg;
}

// ============================================================
// second-order registry
// ============================================================
inline const std::unordered_map<std::string, DerivCreatorFn> &
get_second_order_registry() {
    static const std::unordered_map<std::string, DerivCreatorFn> reg = {
        // explicit stencils
        {"E4", make_explicit<ExplicitDerivsO4_DXX>},
        {"E6", make_explicit<ExplicitDerivsO6_DXX>},
        {"E8", make_explicit<ExplicitDerivsO8_DXX>},

        // explicit matrix
        {"E4Matrix", make_generic<2, createE4DiagonalsSecondOrder, DerivType::D_E4>},
        {"E6Matrix", make_generic<2, createE6DiagonalsSecondOrder, DerivType::D_E6>},
        {"E8Matrix", make_generic<2, createE8DiagonalsSecondOrder, DerivType::D_E8>},

        // jonathan tyler
        {"JTT4", make_generic<2, createJTT4DiagonalsSecondOrder, DerivType::D_JTT4>},
        {"JTT6", make_generic<2, createJTT6DiagonalsSecondOrder, DerivType::D_JTT6>},
        {"JTP6", make_generic<2, createJTP6DiagonalsSecondOrder, DerivType::D_JTP6>},
        {"JTT4Banded", make_explicit<JonathanTyler_JTT4_SecondOrder_Banded>},
        {"JTT6Banded", make_explicit<JonathanTyler_JTT6_SecondOrder_Banded>},
        {"JTP6Banded", make_explicit<JonathanTyler_JTP6_SecondOrder_Banded>},

        // BYU tridiagonal 2nd order
        {"BYUT4", make_generic_coeffs<2, BYUDerivsT4R3DiagonalsSecondOrder, DerivType::D_BYUT4, 4>},
        {"BYUT6R3", make_generic_coeffs<2, BYUDerivsT6R3DiagonalsSecondOrder, DerivType::D_BYUT6R3, 4>},
        {"BYUT6", make_generic_coeffs<2, BYUDerivsT6R4DiagonalsSecondOrder, DerivType::D_BYUT6, 5>},
        {"BYUT6R2", make_generic_coeffs<2, BYUDerivsT6R2DiagonalsSecondOrder, DerivType::D_BYUT6R2, 3>},

        // BYU pentadiagonal 2nd order
        {"BYUP6", make_generic_coeffs<2, BYUDerivsP6R3DiagonalsSecondOrder, DerivType::D_BYUP6, 7>},
        {"BYUP6R2", make_generic_coeffs<2, BYUDerivsP6R2DiagonalsSecondOrder, DerivType::D_BYUP6R2, 3>},
        {"BYUP8", make_generic_coeffs<2, BYUDerivsP8R4DiagonalsSecondOrder, DerivType::D_BYUP8, 9>},

        // 2A family (no user coefficients despite old class using make_with_coeffs)
        {"2A_1", make_generic<2, create2A_1_Diagonals, DerivType::D_2A_1>},
        {"2A_2", make_generic<2, create2A_2_Diagonals, DerivType::D_2A_2>},
        {"2A_3", make_generic<2, create2A_3_Diagonals, DerivType::D_2A_3>},
        {"2A_4", make_generic<2, create2A_4_Diagonals, DerivType::D_2A_4>},
        {"2A_5", make_generic<2, create2A_5_Diagonals, DerivType::D_2A_5>},
        {"2A_6", make_generic<2, create2A_6_Diagonals, DerivType::D_2A_6>},
        {"2A_7", make_generic<2, create2A_7_Diagonals, DerivType::D_2A_7>},
        {"2A_8", make_generic<2, create2A_8_Diagonals, DerivType::D_2A_8>},
        {"2A_9", make_generic<2, create2A_9_Diagonals, DerivType::D_2A_9>},
        {"2A_10", make_generic<2, create2A_10_Diagonals, DerivType::D_2A_10>},
        {"2A_11", make_generic<2, create2A_11_Diagonals, DerivType::D_2A_11>},
        {"2A_12", make_generic<2, create2A_12_Diagonals, DerivType::D_2A_12>},

        // 2UA family
        {"2UA_1", make_generic<2, create2UA_1_Diagonals, DerivType::D_2UA_1>},
        {"2UA_2", make_generic<2, create2UA_2_Diagonals, DerivType::D_2UA_2>},

        // 2B4 family
        {"2B4_1", make_generic<2, create2B4_1_Diagonals, DerivType::D_2B4_1>},

        // 2B6 family
        {"2B6_1", make_generic<2, create2B6_1_Diagonals, DerivType::D_2B6_1>},
        {"2B6_2", make_generic<2, create2B6_2_Diagonals, DerivType::D_2B6_2>},
        {"2B6_3", make_generic<2, create2B6_3_Diagonals, DerivType::D_2B6_3>},
        {"2B6_4", make_generic<2, create2B6_4_Diagonals, DerivType::D_2B6_4>},
        {"2B6_5", make_generic<2, create2B6_5_Diagonals, DerivType::D_2B6_5>},
        {"2B6_6", make_generic<2, create2B6_6_Diagonals, DerivType::D_2B6_6>},
        {"2B6_7", make_generic<2, create2B6_7_Diagonals, DerivType::D_2B6_7>},
        {"2B6_8", make_generic<2, create2B6_8_Diagonals, DerivType::D_2B6_8>},
        {"2B6_9", make_generic<2, create2B6_9_Diagonals, DerivType::D_2B6_9>},

        // 2C4 family
        {"2C4_1", make_generic<2, create2C4_1_Diagonals, DerivType::D_2C4_1>},
        {"2C4_2", make_generic<2, create2C4_2_Diagonals, DerivType::D_2C4_2>},
        {"2C4_3", make_generic<2, create2C4_3_Diagonals, DerivType::D_2C4_3>},
        {"2C4_4", make_generic<2, create2C4_4_Diagonals, DerivType::D_2C4_4>},
        {"2C4_5", make_generic<2, create2C4_5_Diagonals, DerivType::D_2C4_5>},
        {"2C4_6", make_generic<2, create2C4_6_Diagonals, DerivType::D_2C4_6>},
        {"2C4_7", make_generic<2, create2C4_7_Diagonals, DerivType::D_2C4_7>},
        {"2C4_8", make_generic<2, create2C4_8_Diagonals, DerivType::D_2C4_8>},
        {"2C4_9", make_generic<2, create2C4_9_Diagonals, DerivType::D_2C4_9>},
        {"2C4_10", make_generic<2, create2C4_10_Diagonals, DerivType::D_2C4_10>},

        // 2C6 family
        {"2C6_1", make_generic<2, create2C6_1_Diagonals, DerivType::D_2C6_1>},
        {"2C6_2", make_generic<2, create2C6_2_Diagonals, DerivType::D_2C6_2>},
        {"2C6_3", make_generic<2, create2C6_3_Diagonals, DerivType::D_2C6_3>},
        {"2C6_4", make_generic<2, create2C6_4_Diagonals, DerivType::D_2C6_4>},
        {"2C6_5", make_generic<2, create2C6_5_Diagonals, DerivType::D_2C6_5>},
        {"2C6_6", make_generic<2, create2C6_6_Diagonals, DerivType::D_2C6_6>},
        {"2C6_7", make_generic<2, create2C6_7_Diagonals, DerivType::D_2C6_7>},

        // 2A6 family
        {"2A6_1", make_generic<2, create2A6_1_Diagonals, DerivType::D_2A6_1>},
        {"2A6_2", make_generic<2, create2A6_2_Diagonals, DerivType::D_2A6_2>},
        {"2A6_3", make_generic<2, create2A6_3_Diagonals, DerivType::D_2A6_3>},
        {"2A6_4", make_generic<2, create2A6_4_Diagonals, DerivType::D_2A6_4>},
        {"2A6_5", make_generic<2, create2A6_5_Diagonals, DerivType::D_2A6_5>},
        {"2A6_6", make_generic<2, create2A6_6_Diagonals, DerivType::D_2A6_6>},
        {"2A6_7", make_generic<2, create2A6_7_Diagonals, DerivType::D_2A6_7>},
    };
    return reg;
}

class DerivsFactory {
   public:
    static std::unique_ptr<Derivs> create_first_order(
        const std::string &name, const unsigned int ele_order,
        const std::vector<double> &coeffs_in, const unsigned int matrixID,
        const std::string &inMatrixFilterType,
        const std::vector<double> &in_mat_coeffs_in) {
        auto &reg = get_first_order_registry();
        auto it   = reg.find(name);
        if (it != reg.end()) {
            return it->second(ele_order, inMatrixFilterType, in_mat_coeffs_in,
                              coeffs_in, matrixID);
        }
        return nullptr;
    }

    static std::unique_ptr<Derivs> create_second_order(
        const std::string &name, const unsigned int ele_order,
        const std::vector<double> &coeffs_in, const unsigned int matrixID,
        const std::string &inMatrixFilterType,
        const std::vector<double> &in_mat_coeffs_in) {
        auto &reg = get_second_order_registry();
        auto it   = reg.find(name);
        if (it != reg.end()) {
            return it->second(ele_order, inMatrixFilterType, in_mat_coeffs_in,
                              coeffs_in, matrixID);
        }
        return nullptr;
    }
};

class FilterFactory {
   public:
    template <typename... Args>
    static std::unique_ptr<Filters> create_filter(const std::string &name,
                                                  Args &&...args) {
        if (name == "KO2") {
            return std::make_unique<ExplicitKODissO2>(
                std::forward<Args>(args)...);
        } else if (name == "KO4") {
            return std::make_unique<ExplicitKODissO4>(
                std::forward<Args>(args)...);
        } else if (name == "KO6") {
            return std::make_unique<ExplicitKODissO6>(
                std::forward<Args>(args)...);
        } else if (name == "KO8") {
            return std::make_unique<ExplicitKODissO8>(
                std::forward<Args>(args)...);
        }
        return nullptr;
    }
};

}  // namespace dendroderivs
