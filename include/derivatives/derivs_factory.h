#pragma once

#include <memory>
#include <stdexcept>

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

struct UnusedArg {};

class DerivsFactory {
   public:
    static std::unique_ptr<Derivs> create_first_order(
        const std::string &name, const unsigned int ele_order,
        const std::vector<double> &coeffs_in, const unsigned int matrixID,
        const std::string &inMatrixFilterType,
        const std::vector<double> &in_mat_coeffs_in) {
        if (name == "E4") {
            return std::unique_ptr<Derivs>(new ExplicitDerivsO4_DX(ele_order));
        } else if (name == "E6") {
            return std::unique_ptr<Derivs>(new ExplicitDerivsO6_DX(ele_order));
        } else if (name == "E8") {
            return std::unique_ptr<Derivs>(new ExplicitDerivsO8_DX(ele_order));
        } else if (name == "E4Matrix") {
            return std::unique_ptr<Derivs>(
                new ExplicitDerivsO4_Matrix_FirstOrder(
                    ele_order, inMatrixFilterType, in_mat_coeffs_in));
        } else if (name == "E6Matrix") {
            return std::unique_ptr<Derivs>(
                new ExplicitDerivsO6_Matrix_FirstOrder(
                    ele_order, inMatrixFilterType, in_mat_coeffs_in));
        } else if (name == "E8Matrix") {
            return std::unique_ptr<Derivs>(
                new ExplicitDerivsO8_Matrix_FirstOrder(
                    ele_order, inMatrixFilterType, in_mat_coeffs_in));
        } else if (name == "KIM4") {
            return std::unique_ptr<Derivs>(new KimBoundO4_FirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in));
        }else if (name == "KIM407") {
            return std::unique_ptr<Derivs>(new Kim_07_BoundO4_FirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in));
        } else if (name == "KIMBYU_1") {
            return std::unique_ptr<Derivs>(new BYU_KIM_1_DiagonalsFirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in));
        }else if (name == "KIMBYU_2") {
            return std::unique_ptr<Derivs>(new BYU_KIM_2_DiagonalsFirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in));
        }else if (name == "KIMBYU_3") {
            return std::unique_ptr<Derivs>(new BYU_KIM_3_DiagonalsFirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in));
        }else if (name == "KIMBYU_4") {
            return std::unique_ptr<Derivs>(new BYU_KIM_4_DiagonalsFirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in));
        }else if (name == "KIMBYU_5") {
            return std::unique_ptr<Derivs>(new BYU_KIM_5_DiagonalsFirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in));
        }else if (name == "KIMBYU_6") {
            return std::unique_ptr<Derivs>(new BYU_KIM_6_DiagonalsFirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in));
        }else if (name == "KIMBYU_7") {
            return std::unique_ptr<Derivs>(new BYU_KIM_7_DiagonalsFirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in));
        }else if (name == "KIMBYU_8") {
            return std::unique_ptr<Derivs>(new BYU_KIM_8_DiagonalsFirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in));
        }else if (name == "KIMBYU_9") {
            return std::unique_ptr<Derivs>(new BYU_KIM_9_DiagonalsFirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in));
        }else if (name == "KIMBYU_10") {
            return std::unique_ptr<Derivs>(new BYU_KIM_10_DiagonalsFirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in));
        }else if (name == "KIMBYU_11") {
            return std::unique_ptr<Derivs>(new BYU_KIM_11_DiagonalsFirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in));
        }else if (name == "KIMBYU_12") {
            return std::unique_ptr<Derivs>(new BYU_KIM_12_DiagonalsFirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in));
        }else if (name == "KIMBYU_13") {
            return std::unique_ptr<Derivs>(new BYU_KIM_13_DiagonalsFirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in));
        }else if (name == "KIMBYU_14") {
            return std::unique_ptr<Derivs>(new BYU_KIM_14_DiagonalsFirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in));
        }else if (name == "KIMBYU_15") {
            return std::unique_ptr<Derivs>(new BYU_KIM_15_DiagonalsFirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in));
        }else if (name == "KIMBYU_16") {
            return std::unique_ptr<Derivs>(new BYU_KIM_16_DiagonalsFirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in));
        }else if (name == "KIMBYU_17") {
            return std::unique_ptr<Derivs>(new BYU_KIM_17_DiagonalsFirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in));
        }else if (name == "KIMBYU_18") {
            return std::unique_ptr<Derivs>(new BYU_KIM_18_DiagonalsFirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in));
        }else if (name == "KIMBYU_19") {
            return std::unique_ptr<Derivs>(new BYU_KIM_19_DiagonalsFirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in));
        } if (name == "KIM416") {
            return std::unique_ptr<Derivs>(new Kim_16_BoundO4_FirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in));
        }else if (name == "C4_1") {
        return std::unique_ptr<Derivs>(new C4_1_DiagonalsFirstOrder(
            ele_order, inMatrixFilterType, in_mat_coeffs_in));
        } 
        else if (name == "C4_2") {
        return std::unique_ptr<Derivs>(new C4_2_DiagonalsFirstOrder(
            ele_order, inMatrixFilterType, in_mat_coeffs_in));
        } 
        else if (name == "C4_3") {
        return std::unique_ptr<Derivs>(new C4_3_DiagonalsFirstOrder(
            ele_order, inMatrixFilterType, in_mat_coeffs_in));
        } 
        else if (name == "C4_4") {
        return std::unique_ptr<Derivs>(new C4_4_DiagonalsFirstOrder(
            ele_order, inMatrixFilterType, in_mat_coeffs_in));
        } 
        else if (name == "C4_5") {
        return std::unique_ptr<Derivs>(new C4_5_DiagonalsFirstOrder(
            ele_order, inMatrixFilterType, in_mat_coeffs_in));
        }else if (name == "C6_1") {
            return std::unique_ptr<Derivs>(new C6_1_DiagonalsFirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in));
        } else if (name == "C6_2") {
            return std::unique_ptr<Derivs>(new C6_2_DiagonalsFirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in));
        } else if (name == "C6_3") {
            return std::unique_ptr<Derivs>(new C6_3_DiagonalsFirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in));
        } else if (name == "C6_4") {
            return std::unique_ptr<Derivs>(new C6_4_DiagonalsFirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in));
        } else if (name == "C6_5") {
            return std::unique_ptr<Derivs>(new C6_5_DiagonalsFirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in));
        } else if (name == "C6_6") {
            return std::unique_ptr<Derivs>(new C6_6_DiagonalsFirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in));
        } else if (name == "C6_7") {
            return std::unique_ptr<Derivs>(new C6_7_DiagonalsFirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in));
        } else if (name == "C6_8") {
            return std::unique_ptr<Derivs>(new C6_8_DiagonalsFirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in));
        } else if (name == "C6_9") {
            return std::unique_ptr<Derivs>(new C6_9_DiagonalsFirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in));
        } else if (name == "C6_10") {
            return std::unique_ptr<Derivs>(new C6_10_DiagonalsFirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in));
        } else if (name == "C6_11") {
            return std::unique_ptr<Derivs>(new C6_11_DiagonalsFirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in));
        } else if (name == "C6_12") {
            return std::unique_ptr<Derivs>(new C6_12_DiagonalsFirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in));
        } else if (name == "C6_13") {
            return std::unique_ptr<Derivs>(new C6_13_DiagonalsFirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in));
        } else if (name == "C6_14") {
            return std::unique_ptr<Derivs>(new C6_14_DiagonalsFirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in));
        } else if (name == "C6_15") {
            return std::unique_ptr<Derivs>(new C6_15_DiagonalsFirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in));
        } else if (name == "C6_16") {
            return std::unique_ptr<Derivs>(new C6_16_DiagonalsFirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in));
        }else if (name == "UC6_1") {
            return std::unique_ptr<Derivs>(new UC6_1_DiagonalsFirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in));
        } else if (name == "UC6_2") {
            return std::unique_ptr<Derivs>(new UC6_2_DiagonalsFirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in));
        } else if (name == "UC6_3") {
            return std::unique_ptr<Derivs>(new UC6_3_DiagonalsFirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in));
        } else if (name == "UC6_4") {
            return std::unique_ptr<Derivs>(new UC6_4_DiagonalsFirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in));
        }else if (name == "UA4_4") {
            return std::unique_ptr<Derivs>(new UA4_4_DiagonalsFirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in));
        } else if (name == "UA4_5") {
            return std::unique_ptr<Derivs>(new UA4_5_DiagonalsFirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in));
        } else if (name == "UA4_6") {
            return std::unique_ptr<Derivs>(new UA4_6_DiagonalsFirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in));
        }


 else if (name == "JTT4") {
            return std::unique_ptr<Derivs>(new JonathanTyler_JTT4_FirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in));
        } else if (name == "JTT6") {
            return std::unique_ptr<Derivs>(new JonathanTyler_JTT6_FirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in));
        } else if (name == "JTP6") {
            return std::unique_ptr<Derivs>(new JonathanTyler_JTP6_FirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in));
        } else if (name == "BL6") {
            return std::unique_ptr<Derivs>(new BradyLivescu_BL6_FirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in, coeffs_in,
                matrixID));
        } else if (name == "BorisO4") {
            // matrixID doubles as closure ID in the case of Boris!
            return std::unique_ptr<Derivs>(new Boris_BorisO4_FirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in, coeffs_in,
                matrixID));
        } else if (name == "BorisO6") {
            // matrixID doubles as closure ID in the case of Boris!
            return std::unique_ptr<Derivs>(new Boris_BorisO6_FirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in, coeffs_in,
                matrixID));
        } else if (name == "BorisO6Eta") {
            // matrixID doubles as closure ID in the case of Boris!
            return std::unique_ptr<Derivs>(new Boris_BorisO6Eta_FirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in, coeffs_in,
                matrixID));
        } else if (name == "BYUT4") {
            return std::unique_ptr<Derivs>(new BYUDerivsT4_R3_FirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in, coeffs_in));
        } else if (name == "BYUT4R1") {
            return std::unique_ptr<Derivs>(new BYUDerivsT4_R1_FirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in, coeffs_in));
        } else if (name == "BYUT4R4") {
            return std::unique_ptr<Derivs>(new BYUDerivsT4_R42_FirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in, coeffs_in));
        } else if (name == "BYUT4R2") {
            return std::unique_ptr<Derivs>(new BYUDerivsT4_R42_FirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in, coeffs_in));
        } else if (name == "BYUT64R3") {
            return std::unique_ptr<Derivs>(new BYUDerivsT64_R3_FirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in, coeffs_in));
        } else if (name == "BYUT6R2") {
            return std::unique_ptr<Derivs>(new BYUDerivsT6_R2_FirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in, coeffs_in));
        } else if (name == "BYUT6R3") {
            return std::unique_ptr<Derivs>(new BYUDerivsT6_R3_FirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in, coeffs_in));
        } else if (name == "BYUT6") {
            return std::unique_ptr<Derivs>(new BYUDerivsT6_R4_FirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in, coeffs_in));
        } else if (name == "BYUT62") {
            return std::unique_ptr<Derivs>(new BYUDerivsT6_R2_FirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in, coeffs_in));
        } else if (name == "BYUP6") {
            return std::unique_ptr<Derivs>(new BYUDerivsP6_R3_FirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in, coeffs_in));
        }  else if (name == "BYUP6R3") {
            return std::unique_ptr<Derivs>(new BYUDerivsP6_R32_FirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in, coeffs_in));
        } else if (name == "BYUP6R2") {
            return std::unique_ptr<Derivs>(new BYUDerivsP6_R2_FirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in, coeffs_in));
        }else if (name == "BYUP8") {
            return std::unique_ptr<Derivs>(new BYUDerivsP8_R4_FirstOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in, coeffs_in));
        } else if (name == "EXPERIMENTAL") {
            return std::unique_ptr<Derivs>(
                new TestingHybridDerivatives_FirstOrder(
                    ele_order, inMatrixFilterType, in_mat_coeffs_in));
        } else if (name == "JTT4Banded") {
            return std::unique_ptr<Derivs>(
                new JonathanTyler_JTT4_FirstOrder_Banded(ele_order));
        } else if (name == "JTT6Banded") {
            return std::unique_ptr<Derivs>(
                new JonathanTyler_JTT6_FirstOrder_Banded(ele_order));
        } else if (name == "JTP6Banded") {
            return std::unique_ptr<Derivs>(
                new JonathanTyler_JTP6_FirstOrder_Banded(ele_order));
        }

        return nullptr;
    }

    static std::unique_ptr<Derivs> create_second_order(
        const std::string &name, const unsigned int ele_order,
        const std::vector<double> &coeffs_in, const unsigned int matrixID,
        const std::string &inMatrixFilterType,
        const std::vector<double> &in_mat_coeffs_in) {
        if (name == "E4") {
            return std::unique_ptr<Derivs>(new ExplicitDerivsO4_DXX(ele_order));
        } else if (name == "E6") {
            return std::unique_ptr<Derivs>(new ExplicitDerivsO6_DXX(ele_order));
        } else if (name == "E8") {
            return std::unique_ptr<Derivs>(new ExplicitDerivsO8_DXX(ele_order));
        } else if (name == "E4Matrix") {
            return std::unique_ptr<Derivs>(
                new ExplicitDerivsO4_Matrix_SecondOrder(
                    ele_order, inMatrixFilterType, in_mat_coeffs_in));
        } else if (name == "E6Matrix") {
            return std::unique_ptr<Derivs>(
                new ExplicitDerivsO6_Matrix_SecondOrder(
                    ele_order, inMatrixFilterType, in_mat_coeffs_in));
        } else if (name == "E8Matrix") {
            return std::unique_ptr<Derivs>(
                new ExplicitDerivsO8_Matrix_SecondOrder(
                    ele_order, inMatrixFilterType, in_mat_coeffs_in));
        } else if (name == "KIM4") {
            throw std::runtime_error("There is no second order KIM4!");
        } else if (name == "JTT4") {
            return std::unique_ptr<Derivs>(new JonathanTyler_JTT4_SecondOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in));
        } else if (name == "JTT6") {
            return std::unique_ptr<Derivs>(new JonathanTyler_JTT6_SecondOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in));
        } else if (name == "JTP6") {
            return std::unique_ptr<Derivs>(new JonathanTyler_JTP6_SecondOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in));
        } else if (name == "BYUT4") {
            return std::unique_ptr<Derivs>(new BYUDerivsT4_R3_SecondOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in, coeffs_in));
        } else if (name == "BYUT6R3") {
            return std::unique_ptr<Derivs>(new BYUDerivsT6_R3_SecondOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in, coeffs_in));
        } else if (name == "BYUT6") {
            return std::unique_ptr<Derivs>(new BYUDerivsT6_R4_SecondOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in, coeffs_in));
        } else if (name == "BYUT6R2") {
            return std::unique_ptr<Derivs>(new BYUDerivsT6_R2_SecondOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in, coeffs_in));
        }else if (name == "BYUP6") {
            return std::unique_ptr<Derivs>(new BYUDerivsP6_R3_SecondOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in, coeffs_in));
        }else if (name == "BYUP6R2") {
            return std::unique_ptr<Derivs>(new BYUDerivsP6_R2_SecondOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in, coeffs_in));
        } else if (name == "BYUP6R3") {
            return std::unique_ptr<Derivs>(new BYUDerivsP6_R32_SecondOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in, coeffs_in));
        }  else if (name == "BYUT62") {
            return std::unique_ptr<Derivs>(new BYUDerivsT6_R2_SecondOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in, coeffs_in));
        }else if (name == "BYUP8") {
            return std::unique_ptr<Derivs>(new BYUDerivsP8_R4_SecondOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in, coeffs_in));
        } else if (name == "JTT4Banded") {
            return std::unique_ptr<Derivs>(
                new JonathanTyler_JTT4_SecondOrder_Banded(ele_order));
        } else if (name == "JTT6Banded") {
            return std::unique_ptr<Derivs>(
                new JonathanTyler_JTT6_SecondOrder_Banded(ele_order));
        } else if (name == "JTP6Banded") {
            return std::unique_ptr<Derivs>(
                new JonathanTyler_JTP6_SecondOrder_Banded(ele_order));
        } else if (name == "2A_1") {
        return std::unique_ptr<Derivs>(new TwoADerivs2A_1_SecondOrder(
        ele_order, inMatrixFilterType, in_mat_coeffs_in, coeffs_in));
        } else if (name == "2A_2") {
        return std::unique_ptr<Derivs>(new TwoADerivs2A_2_SecondOrder(
        ele_order, inMatrixFilterType, in_mat_coeffs_in, coeffs_in));
        } else if (name == "2A_3") {
        return std::unique_ptr<Derivs>(new TwoADerivs2A_3_SecondOrder(
        ele_order, inMatrixFilterType, in_mat_coeffs_in, coeffs_in));
        } else if (name == "2A_4") {
        return std::unique_ptr<Derivs>(new TwoADerivs2A_4_SecondOrder(
        ele_order, inMatrixFilterType, in_mat_coeffs_in, coeffs_in));
        } else if (name == "2A_5") {
        return std::unique_ptr<Derivs>(new TwoADerivs2A_5_SecondOrder(
        ele_order, inMatrixFilterType, in_mat_coeffs_in, coeffs_in));
        } else if (name == "2A_6") {
        return std::unique_ptr<Derivs>(new TwoADerivs2A_6_SecondOrder(
        ele_order, inMatrixFilterType, in_mat_coeffs_in, coeffs_in));
        } else if (name == "2A_7") {
        return std::unique_ptr<Derivs>(new TwoADerivs2A_7_SecondOrder(
        ele_order, inMatrixFilterType, in_mat_coeffs_in, coeffs_in));
        } else if (name == "2A_8") {
        return std::unique_ptr<Derivs>(new TwoADerivs2A_8_SecondOrder(
        ele_order, inMatrixFilterType, in_mat_coeffs_in, coeffs_in));
        } else if (name == "2A_9") {
        return std::unique_ptr<Derivs>(new TwoADerivs2A_9_SecondOrder(
        ele_order, inMatrixFilterType, in_mat_coeffs_in, coeffs_in));
        } else if (name == "2A_10") {
        return std::unique_ptr<Derivs>(new TwoADerivs2A_10_SecondOrder(
        ele_order, inMatrixFilterType, in_mat_coeffs_in, coeffs_in));
        } else if (name == "2A_11") {
        return std::unique_ptr<Derivs>(new TwoADerivs2A_11_SecondOrder(
        ele_order, inMatrixFilterType, in_mat_coeffs_in, coeffs_in));
        } else if (name == "2A_12") {
        return std::unique_ptr<Derivs>(new TwoADerivs2A_12_SecondOrder(
        ele_order, inMatrixFilterType, in_mat_coeffs_in, coeffs_in));
        }else if (name == "2UA_1") {
            return std::unique_ptr<Derivs>(new TwoUADerivs2UA_1_SecondOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in, coeffs_in));
        } else if (name == "2UA_2") {
            return std::unique_ptr<Derivs>(new TwoUADerivs2UA_2_SecondOrder(
                ele_order, inMatrixFilterType, in_mat_coeffs_in, coeffs_in));
        }else if (name == "2B4_1") {
        return std::unique_ptr<Derivs>(new TwoDerivs2B4_1_SecondOrder(
        ele_order, inMatrixFilterType, in_mat_coeffs_in, coeffs_in));
        }
        else if (name == "2B6_1") {
        return std::unique_ptr<Derivs>(new TwoDerivs2B6_1_SecondOrder(
        ele_order, inMatrixFilterType, in_mat_coeffs_in, coeffs_in));
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
            return std::unique_ptr<Filters>(
                new ExplicitKODissO2(std::forward<Args>(args)...));
        } else if (name == "KO4") {
            return std::unique_ptr<Filters>(
                new ExplicitKODissO4(std::forward<Args>(args)...));
        } else if (name == "KO6") {
            return std::unique_ptr<Filters>(
                new ExplicitKODissO6(std::forward<Args>(args)...));
        }

        return nullptr;
    }
};

}  // namespace dendroderivs
