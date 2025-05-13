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
    template <typename... Args>
    static std::unique_ptr<Derivs> create_first_order(const std::string &name,
                                                      Args &&...args) {
        if (name == "E4") {
            return std::unique_ptr<Derivs>(
                new ExplicitDerivsO4_DX(std::forward<Args>(args)...));
        } else if (name == "E6") {
            return std::unique_ptr<Derivs>(
                new ExplicitDerivsO6_DX(std::forward<Args>(args)...));
        } else if (name == "E8") {
            return std::unique_ptr<Derivs>(
                new ExplicitDerivsO8_DX(std::forward<Args>(args)...));
        } else if (name == "E4Matrix") {
            return std::unique_ptr<Derivs>(
                new ExplicitDerivsO4_Matrix_FirstOrder(
                    std::forward<Args>(args)...));
        } else if (name == "E6Matrix") {
            return std::unique_ptr<Derivs>(
                new ExplicitDerivsO6_Matrix_FirstOrder(
                    std::forward<Args>(args)...));
        } else if (name == "E8Matrix") {
            return std::unique_ptr<Derivs>(
                new ExplicitDerivsO8_Matrix_FirstOrder(
                    std::forward<Args>(args)...));
        } else if (name == "KIM4") {
            return std::unique_ptr<Derivs>(
                new KimBoundO4_FirstOrder(std::forward<Args>(args)...));
        } else if (name == "JTT4") {
            return std::unique_ptr<Derivs>(
                new JonathanTyler_JTT4_FirstOrder(std::forward<Args>(args)...));
        } else if (name == "JTT6") {
            return std::unique_ptr<Derivs>(
                new JonathanTyler_JTT6_FirstOrder(std::forward<Args>(args)...));
        } else if (name == "JTP6") {
            return std::unique_ptr<Derivs>(
                new JonathanTyler_JTP6_FirstOrder(std::forward<Args>(args)...));
        } else if (name == "BL6") {
            return std::unique_ptr<Derivs>(
                new BradyLivescu_BL6_FirstOrder(std::forward<Args>(args)...));
        } else if (name == "BorisO4") {
            return std::unique_ptr<Derivs>(
                new Boris_BorisO4_FirstOrder(std::forward<Args>(args)...));
        } else if (name == "BorisO6") {
            return std::unique_ptr<Derivs>(
                new Boris_BorisO6_FirstOrder(std::forward<Args>(args)...));
        } else if (name == "BorisO6Eta") {
            return std::unique_ptr<Derivs>(
                new Boris_BorisO6Eta_FirstOrder(std::forward<Args>(args)...));
        } else if (name == "BYUT4") {
            return std::unique_ptr<Derivs>(
                new BYUDerivsT4_R3_FirstOrder(std::forward<Args>(args)...));
        } else if (name == "BYUT4R1") {
            return std::unique_ptr<Derivs>(
                new BYUDerivsT4_R1_FirstOrder(std::forward<Args>(args)...));
        } else if (name == "BYUT4R4") {
            return std::unique_ptr<Derivs>(
                new BYUDerivsT4_R42_FirstOrder(std::forward<Args>(args)...));
        } else if (name == "BYUT4R2") {
            return std::unique_ptr<Derivs>(
                new BYUDerivsT4_R42_FirstOrder(std::forward<Args>(args)...));
        } else if (name == "BYUT64R3") {
            return std::unique_ptr<Derivs>(
                new BYUDerivsT64_R3_FirstOrder(std::forward<Args>(args)...));
        } else if (name == "BYUT6R2") {
            return std::unique_ptr<Derivs>(
                new BYUDerivsT6_R2_FirstOrder(std::forward<Args>(args)...));
        } else if (name == "BYUT6R3") {
            return std::unique_ptr<Derivs>(
                new BYUDerivsT6_R3_FirstOrder(std::forward<Args>(args)...));
        } else if (name == "BYUT6") {
            return std::unique_ptr<Derivs>(
                new BYUDerivsT6_R4_FirstOrder(std::forward<Args>(args)...));
        } else if (name == "BYUT62") {
            return std::unique_ptr<Derivs>(
                new BYUDerivsT6_R2_FirstOrder(std::forward<Args>(args)...));
        } else if (name == "BYUP6") {
            return std::unique_ptr<Derivs>(
                new BYUDerivsP6_R3_FirstOrder(std::forward<Args>(args)...));
        } else if (name == "BYUP8") {
            return std::unique_ptr<Derivs>(
                new BYUDerivsP8_R4_FirstOrder(std::forward<Args>(args)...));
        } else if (name == "EXPERIMENTAL") {
            return std::unique_ptr<Derivs>(
                new TestingHybridDerivatives_FirstOrder(
                    std::forward<Args>(args)...));
        } else if (name == "JTT4Banded") {
            return std::unique_ptr<Derivs>(
                new JonathanTyler_JTT4_FirstOrder_Banded(
                    std::forward<Args>(args)...));
        } else if (name == "JTT6Banded") {
            return std::unique_ptr<Derivs>(
                new JonathanTyler_JTT6_FirstOrder_Banded(
                    std::forward<Args>(args)...));
        } else if (name == "JTP6Banded") {
            return std::unique_ptr<Derivs>(
                new JonathanTyler_JTP6_FirstOrder_Banded(
                    std::forward<Args>(args)...));
        }

        return nullptr;
    }

    template <typename... Args>
    static std::unique_ptr<Derivs> create_second_order(const std::string &name,
                                                       Args &&...args) {
        if (name == "E4") {
            return std::unique_ptr<Derivs>(
                new ExplicitDerivsO4_DXX(std::forward<Args>(args)...));
        } else if (name == "E6") {
            return std::unique_ptr<Derivs>(
                new ExplicitDerivsO6_DXX(std::forward<Args>(args)...));
        } else if (name == "E8") {
            return std::unique_ptr<Derivs>(
                new ExplicitDerivsO8_DXX(std::forward<Args>(args)...));
        } else if (name == "E4Matrix") {
            return std::unique_ptr<Derivs>(
                new ExplicitDerivsO4_Matrix_SecondOrder(
                    std::forward<Args>(args)...));
        } else if (name == "E6Matrix") {
            return std::unique_ptr<Derivs>(
                new ExplicitDerivsO6_Matrix_SecondOrder(
                    std::forward<Args>(args)...));
        } else if (name == "E8Matrix") {
            return std::unique_ptr<Derivs>(
                new ExplicitDerivsO8_Matrix_SecondOrder(
                    std::forward<Args>(args)...));
        } else if (name == "KIM4") {
            throw std::runtime_error("There is no second order KIM4!");
        } else if (name == "JTT4") {
            return std::unique_ptr<Derivs>(new JonathanTyler_JTT4_SecondOrder(
                std::forward<Args>(args)...));
        } else if (name == "JTT6") {
            return std::unique_ptr<Derivs>(new JonathanTyler_JTT6_SecondOrder(
                std::forward<Args>(args)...));
        } else if (name == "JTP6") {
            return std::unique_ptr<Derivs>(new JonathanTyler_JTP6_SecondOrder(
                std::forward<Args>(args)...));
        } else if (name == "BYUT4") {
            return std::unique_ptr<Derivs>(
                new BYUDerivsT4_R3_SecondOrder(std::forward<Args>(args)...));
        } else if (name == "BYUT6R3") {
            return std::unique_ptr<Derivs>(
                new BYUDerivsT6_R3_SecondOrder(std::forward<Args>(args)...));
        } else if (name == "BYUT6") {
            return std::unique_ptr<Derivs>(
                new BYUDerivsT6_R4_SecondOrder(std::forward<Args>(args)...));
        } else if (name == "BYUP6") {
            return std::unique_ptr<Derivs>(
                new BYUDerivsP6_R3_SecondOrder(std::forward<Args>(args)...));
        } else if (name == "BYUP8") {
            return std::unique_ptr<Derivs>(
                new BYUDerivsP8_R4_SecondOrder(std::forward<Args>(args)...));
        } else if (name == "JTT4Banded") {
            return std::unique_ptr<Derivs>(
                new JonathanTyler_JTT4_SecondOrder_Banded(
                    std::forward<Args>(args)...));
        } else if (name == "JTT6Banded") {
            return std::unique_ptr<Derivs>(
                new JonathanTyler_JTT6_SecondOrder_Banded(
                    std::forward<Args>(args)...));
        } else if (name == "JTP6Banded") {
            return std::unique_ptr<Derivs>(
                new JonathanTyler_JTP6_SecondOrder_Banded(
                    std::forward<Args>(args)...));
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
                new ExplicitKODissO4(std::forward<Args>(args)...));
        }

        return nullptr;
    }
};

}  // namespace dendroderivs
