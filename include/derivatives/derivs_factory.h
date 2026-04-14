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
        {"E4Matrix", make_matrix<ExplicitDerivsO4_Matrix_FirstOrder>},
        {"E6Matrix", make_matrix<ExplicitDerivsO6_Matrix_FirstOrder>},
        {"E8Matrix", make_matrix<ExplicitDerivsO8_Matrix_FirstOrder>},

        // kim family
        {"KIM4", make_matrix<KimBoundO4_FirstOrder>},
        {"KIM407", make_matrix<Kim_07_BoundO4_FirstOrder>},
        {"KIM416", make_matrix<Kim_16_BoundO4_FirstOrder>},
        {"KIMBYU_1", make_matrix<BYU_KIM_1_DiagonalsFirstOrder>},
        {"KIMBYU_2", make_matrix<BYU_KIM_2_DiagonalsFirstOrder>},
        {"KIMBYU_3", make_matrix<BYU_KIM_3_DiagonalsFirstOrder>},
        {"KIMBYU_4", make_matrix<BYU_KIM_4_DiagonalsFirstOrder>},
        {"KIMBYU_5", make_matrix<BYU_KIM_5_DiagonalsFirstOrder>},
        {"KIMBYU_6", make_matrix<BYU_KIM_6_DiagonalsFirstOrder>},
        {"KIMBYU_7", make_matrix<BYU_KIM_7_DiagonalsFirstOrder>},
        {"KIMBYU_8", make_matrix<BYU_KIM_8_DiagonalsFirstOrder>},
        {"KIMBYU_9", make_matrix<BYU_KIM_9_DiagonalsFirstOrder>},
        {"KIMBYU_10", make_matrix<BYU_KIM_10_DiagonalsFirstOrder>},
        {"KIMBYU_11", make_matrix<BYU_KIM_11_DiagonalsFirstOrder>},
        {"KIMBYU_12", make_matrix<BYU_KIM_12_DiagonalsFirstOrder>},
        {"KIMBYU_13", make_matrix<BYU_KIM_13_DiagonalsFirstOrder>},
        {"KIMBYU_14", make_matrix<BYU_KIM_14_DiagonalsFirstOrder>},
        {"KIMBYU_15", make_matrix<BYU_KIM_15_DiagonalsFirstOrder>},
        {"KIMBYU_16", make_matrix<BYU_KIM_16_DiagonalsFirstOrder>},
        {"KIMBYU_17", make_matrix<BYU_KIM_17_DiagonalsFirstOrder>},
        {"KIMBYU_18", make_matrix<BYU_KIM_18_DiagonalsFirstOrder>},
        {"KIMBYU_19", make_matrix<BYU_KIM_19_DiagonalsFirstOrder>},

        // C4 family
        {"C4_1", make_matrix<C4_1_DiagonalsFirstOrder>},
        {"C4_2", make_matrix<C4_2_DiagonalsFirstOrder>},
        {"C4_3", make_matrix<C4_3_DiagonalsFirstOrder>},
        {"C4_4", make_matrix<C4_4_DiagonalsFirstOrder>},
        {"C4_5", make_matrix<C4_5_DiagonalsFirstOrder>},

        // A4 family
        {"A4_1", make_matrix<A4_1_DiagonalsFirstOrder>},
        {"A4_2", make_matrix<A4_2_DiagonalsFirstOrder>},
        {"A4_3", make_matrix<A4_3_DiagonalsFirstOrder>},
        {"A4_4", make_matrix<A4_4_DiagonalsFirstOrder>},
        {"A4_5", make_matrix<A4_5_DiagonalsFirstOrder>},
        {"A4_6", make_matrix<A4_6_DiagonalsFirstOrder>},
        {"A4_7", make_matrix<A4_7_DiagonalsFirstOrder>},
        {"A4_8", make_matrix<A4_8_DiagonalsFirstOrder>},
        {"A4_9", make_matrix<A4_9_DiagonalsFirstOrder>},
        {"A4_10", make_matrix<A4_10_DiagonalsFirstOrder>},
        {"A4_11", make_matrix<A4_11_DiagonalsFirstOrder>},
        {"A4_12", make_matrix<A4_12_DiagonalsFirstOrder>},
        {"A4_13", make_matrix<A4_13_DiagonalsFirstOrder>},
        {"A4_14", make_matrix<A4_14_DiagonalsFirstOrder>},
        {"A4_15", make_matrix<A4_15_DiagonalsFirstOrder>},
        {"A4_16", make_matrix<A4_16_DiagonalsFirstOrder>},
        {"A4_17", make_matrix<A4_17_DiagonalsFirstOrder>},
        {"A4_18", make_matrix<A4_18_DiagonalsFirstOrder>},
        {"A4_19", make_matrix<A4_19_DiagonalsFirstOrder>},
        {"A4_20", make_matrix<A4_20_DiagonalsFirstOrder>},

        // B4 family
        {"B4_1", make_matrix<B4_1_DiagonalsFirstOrder>},
        {"B4_2", make_matrix<B4_2_DiagonalsFirstOrder>},

        // A6 family
        {"A6_1", make_matrix<A6_1_DiagonalsFirstOrder>},
        {"A6_2", make_matrix<A6_2_DiagonalsFirstOrder>},
        {"A6_3", make_matrix<A6_3_DiagonalsFirstOrder>},
        {"A6_4", make_matrix<A6_4_DiagonalsFirstOrder>},
        {"A6_5", make_matrix<A6_5_DiagonalsFirstOrder>},
        {"A6_6", make_matrix<A6_6_DiagonalsFirstOrder>},
        {"A6_7", make_matrix<A6_7_DiagonalsFirstOrder>},
        {"A6_8", make_matrix<A6_8_DiagonalsFirstOrder>},

        // C6 family
        {"C6_1", make_matrix<C6_1_DiagonalsFirstOrder>},
        {"C6_2", make_matrix<C6_2_DiagonalsFirstOrder>},
        {"C6_3", make_matrix<C6_3_DiagonalsFirstOrder>},
        {"C6_4", make_matrix<C6_4_DiagonalsFirstOrder>},
        {"C6_5", make_matrix<C6_5_DiagonalsFirstOrder>},
        {"C6_6", make_matrix<C6_6_DiagonalsFirstOrder>},
        {"C6_7", make_matrix<C6_7_DiagonalsFirstOrder>},
        {"C6_8", make_matrix<C6_8_DiagonalsFirstOrder>},
        {"C6_9", make_matrix<C6_9_DiagonalsFirstOrder>},
        {"C6_10", make_matrix<C6_10_DiagonalsFirstOrder>},
        {"C6_11", make_matrix<C6_11_DiagonalsFirstOrder>},
        {"C6_12", make_matrix<C6_12_DiagonalsFirstOrder>},
        {"C6_13", make_matrix<C6_13_DiagonalsFirstOrder>},
        {"C6_14", make_matrix<C6_14_DiagonalsFirstOrder>},
        {"C6_15", make_matrix<C6_15_DiagonalsFirstOrder>},
        {"C6_16", make_matrix<C6_16_DiagonalsFirstOrder>},

        // UC6 family
        {"UC6_1", make_matrix<UC6_1_DiagonalsFirstOrder>},
        {"UC6_2", make_matrix<UC6_2_DiagonalsFirstOrder>},
        {"UC6_3", make_matrix<UC6_3_DiagonalsFirstOrder>},
        {"UC6_4", make_matrix<UC6_4_DiagonalsFirstOrder>},

        // UA4 family
        {"UA4_4", make_matrix<UA4_4_DiagonalsFirstOrder>},
        {"UA4_5", make_matrix<UA4_5_DiagonalsFirstOrder>},
        {"UA4_6", make_matrix<UA4_6_DiagonalsFirstOrder>},

        // jonathan tyler compact schemes
        {"JTT4", make_matrix<JonathanTyler_JTT4_FirstOrder>},
        {"JTT6", make_matrix<JonathanTyler_JTT6_FirstOrder>},
        {"JTP6", make_matrix<JonathanTyler_JTP6_FirstOrder>},
        {"JTT4Banded", make_explicit<JonathanTyler_JTT4_FirstOrder_Banded>},
        {"JTT6Banded", make_explicit<JonathanTyler_JTT6_FirstOrder_Banded>},
        {"JTP6Banded", make_explicit<JonathanTyler_JTP6_FirstOrder_Banded>},

        // brady-livescu (uses coeffs + matrixID)
        {"BL6", make_with_coeffs_and_id<BradyLivescu_BL6_FirstOrder>},

        // boris (uses coeffs + matrixID as closure ID)
        {"BorisO4", make_with_coeffs_and_id<Boris_BorisO4_FirstOrder>},
        {"BorisO6", make_with_coeffs_and_id<Boris_BorisO6_FirstOrder>},
        {"BorisO6Eta", make_with_coeffs_and_id<Boris_BorisO6Eta_FirstOrder>},

        // BYU tridiagonal (uses coeffs)
        {"BYUT4", make_with_coeffs<BYUDerivsT4_R3_FirstOrder>},
        {"BYUT4R1", make_with_coeffs<BYUDerivsT4_R1_FirstOrder>},
        {"BYUT4R2", make_with_coeffs<BYUDerivsT4_R2_FirstOrder>},
        {"BYUT4R4", make_with_coeffs<BYUDerivsT4_R42_FirstOrder>},
        {"BYUT6", make_with_coeffs<BYUDerivsT6_R4_FirstOrder>},
        {"BYUT62", make_with_coeffs<BYUDerivsT6_R42_FirstOrder>},
        {"BYUT64R3", make_with_coeffs<BYUDerivsT64_R3_FirstOrder>},
        {"BYUT6R2", make_with_coeffs<BYUDerivsT6_R2_FirstOrder>},
        {"BYUT6R3", make_with_coeffs<BYUDerivsT6_R3_FirstOrder>},

        // BYU pentadiagonal (uses coeffs)
        {"BYUP6", make_with_coeffs<BYUDerivsP6_R3_FirstOrder>},
        {"BYUP6R2", make_with_coeffs<BYUDerivsP6_R2_FirstOrder>},
        {"BYUP6R3", make_with_coeffs<BYUDerivsP6_R3_FirstOrder>},
        {"BYUP8", make_with_coeffs<BYUDerivsP8_R4_FirstOrder>},
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
        {"E4Matrix", make_matrix<ExplicitDerivsO4_Matrix_SecondOrder>},
        {"E6Matrix", make_matrix<ExplicitDerivsO6_Matrix_SecondOrder>},
        {"E8Matrix", make_matrix<ExplicitDerivsO8_Matrix_SecondOrder>},

        // jonathan tyler
        {"JTT4", make_matrix<JonathanTyler_JTT4_SecondOrder>},
        {"JTT6", make_matrix<JonathanTyler_JTT6_SecondOrder>},
        {"JTP6", make_matrix<JonathanTyler_JTP6_SecondOrder>},
        {"JTT4Banded", make_explicit<JonathanTyler_JTT4_SecondOrder_Banded>},
        {"JTT6Banded", make_explicit<JonathanTyler_JTT6_SecondOrder_Banded>},
        {"JTP6Banded", make_explicit<JonathanTyler_JTP6_SecondOrder_Banded>},

        // BYU tridiagonal
        {"BYUT4", make_with_coeffs<BYUDerivsT4_R3_SecondOrder>},
        {"BYUT6R3", make_with_coeffs<BYUDerivsT6_R3_SecondOrder>},
        {"BYUT6", make_with_coeffs<BYUDerivsT6_R4_SecondOrder>},
        {"BYUT6R2", make_with_coeffs<BYUDerivsT6_R2_SecondOrder>},
        // note: no BYUT8 second-order class exists currently

        // BYU pentadiagonal
        {"BYUP6", make_with_coeffs<BYUDerivsP6_R3_SecondOrder>},
        {"BYUP6R2", make_with_coeffs<BYUDerivsP6_R2_SecondOrder>},
        {"BYUP8", make_with_coeffs<BYUDerivsP8_R4_SecondOrder>},

        // 2A family
        {"2A_1", make_with_coeffs<TwoADerivs2A_1_SecondOrder>},
        {"2A_2", make_with_coeffs<TwoADerivs2A_2_SecondOrder>},
        {"2A_3", make_with_coeffs<TwoADerivs2A_3_SecondOrder>},
        {"2A_4", make_with_coeffs<TwoADerivs2A_4_SecondOrder>},
        {"2A_5", make_with_coeffs<TwoADerivs2A_5_SecondOrder>},
        {"2A_6", make_with_coeffs<TwoADerivs2A_6_SecondOrder>},
        {"2A_7", make_with_coeffs<TwoADerivs2A_7_SecondOrder>},
        {"2A_8", make_with_coeffs<TwoADerivs2A_8_SecondOrder>},
        {"2A_9", make_with_coeffs<TwoADerivs2A_9_SecondOrder>},
        {"2A_10", make_with_coeffs<TwoADerivs2A_10_SecondOrder>},
        {"2A_11", make_with_coeffs<TwoADerivs2A_11_SecondOrder>},
        {"2A_12", make_with_coeffs<TwoADerivs2A_12_SecondOrder>},

        // 2UA family
        {"2UA_1", make_with_coeffs<TwoUADerivs2UA_1_SecondOrder>},
        {"2UA_2", make_with_coeffs<TwoUADerivs2UA_2_SecondOrder>},

        // 2B4 family
        {"2B4_1", make_with_coeffs<TwoDerivs2B4_1_SecondOrder>},

        // 2B6 family
        {"2B6_1", make_with_coeffs<TwoDerivs2B6_1_SecondOrder>},
        {"2B6_2", make_with_coeffs<TwoDerivs2B6_2_SecondOrder>},
        {"2B6_3", make_with_coeffs<TwoDerivs2B6_3_SecondOrder>},
        {"2B6_4", make_with_coeffs<TwoDerivs2B6_4_SecondOrder>},
        {"2B6_5", make_with_coeffs<TwoDerivs2B6_5_SecondOrder>},
        {"2B6_6", make_with_coeffs<TwoDerivs2B6_6_SecondOrder>},
        {"2B6_7", make_with_coeffs<TwoDerivs2B6_7_SecondOrder>},
        {"2B6_8", make_with_coeffs<TwoDerivs2B6_8_SecondOrder>},
        {"2B6_9", make_with_coeffs<TwoDerivs2B6_9_SecondOrder>},

        // 2C4 family
        {"2C4_1", make_with_coeffs<TwoDerivs2C4_1_SecondOrder>},
        {"2C4_2", make_with_coeffs<TwoDerivs2C4_2_SecondOrder>},
        {"2C4_3", make_with_coeffs<TwoDerivs2C4_3_SecondOrder>},
        {"2C4_4", make_with_coeffs<TwoDerivs2C4_4_SecondOrder>},
        {"2C4_5", make_with_coeffs<TwoDerivs2C4_5_SecondOrder>},
        {"2C4_6", make_with_coeffs<TwoDerivs2C4_6_SecondOrder>},
        {"2C4_7", make_with_coeffs<TwoDerivs2C4_7_SecondOrder>},
        {"2C4_8", make_with_coeffs<TwoDerivs2C4_8_SecondOrder>},
        {"2C4_9", make_with_coeffs<TwoDerivs2C4_9_SecondOrder>},
        {"2C4_10", make_with_coeffs<TwoDerivs2C4_10_SecondOrder>},

        // 2C6 family
        {"2C6_1", make_with_coeffs<TwoDerivs2C6_1_SecondOrder>},
        {"2C6_2", make_with_coeffs<TwoDerivs2C6_2_SecondOrder>},
        {"2C6_3", make_with_coeffs<TwoDerivs2C6_3_SecondOrder>},
        {"2C6_4", make_with_coeffs<TwoDerivs2C6_4_SecondOrder>},
        {"2C6_5", make_with_coeffs<TwoDerivs2C6_5_SecondOrder>},
        {"2C6_6", make_with_coeffs<TwoDerivs2C6_6_SecondOrder>},
        {"2C6_7", make_with_coeffs<TwoDerivs2C6_7_SecondOrder>},

        // 2A6 family
        {"2A6_1", make_with_coeffs<TwoDerivs2A6_1_SecondOrder>},
        {"2A6_2", make_with_coeffs<TwoDerivs2A6_2_SecondOrder>},
        {"2A6_3", make_with_coeffs<TwoDerivs2A6_3_SecondOrder>},
        {"2A6_4", make_with_coeffs<TwoDerivs2A6_4_SecondOrder>},
        {"2A6_5", make_with_coeffs<TwoDerivs2A6_5_SecondOrder>},
        {"2A6_6", make_with_coeffs<TwoDerivs2A6_6_SecondOrder>},
        {"2A6_7", make_with_coeffs<TwoDerivs2A6_7_SecondOrder>},
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
