#pragma once

#include <stdexcept>

#include "derivatives.h"
#include "derivatives/derivs_explicit.h"
#include "derivatives/impl_byuderivs.h"
#include "derivatives/impl_explicitmatrix.h"
#include "derivatives/impl_jonathantyler.h"
#include "derivatives/impl_kimderivs.h"

namespace dendroderivs {

struct UnusedArg {};

class DerivsFactory {
   public:
    template <typename... Args>
    static std::unique_ptr<Derivs> create_first_order(const std::string &name,
                                                      Args &&...args) {
        std::cout << "NAME: " << name << std::endl;
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
        } else if (name == "KIM4") {
            return std::unique_ptr<Derivs>(
                new KimBoundO4_FirstOrder(std::forward<Args>(args)...));
        } else if (name == "JTT4") {
            return std::unique_ptr<Derivs>(
                new JonathanTyler_JTT4_FirstOrder(std::forward<Args>(args)...));
        } else if (name == "JTT6") {
            return std::unique_ptr<Derivs>(
                new JonathanTyler_JTT6_FirstOrder(std::forward<Args>(args)...));
        } else if (name == "BYUT4") {
            return std::unique_ptr<Derivs>(
                new BYUDerivsT4_R3_FirstOrder(std::forward<Args>(args)...));
        } else if (name == "BYUT6R3") {
            return std::unique_ptr<Derivs>(
                new BYUDerivsT6_R3_FirstOrder(std::forward<Args>(args)...));
        } else if (name == "BYUT6") {
            return std::unique_ptr<Derivs>(
                new BYUDerivsT6_R4_FirstOrder(std::forward<Args>(args)...));
        } else if (name == "BYUP6") {
            return std::unique_ptr<Derivs>(
                new BYUDerivsP6_R3_FirstOrder(std::forward<Args>(args)...));
        } else if (name == "BYUP8") {
            return std::unique_ptr<Derivs>(
                new BYUDerivsP8_R4_FirstOrder(std::forward<Args>(args)...));
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
        } else if (name == "KIM4") {
            throw std::runtime_error("There is no second order KIM4!");
        } else if (name == "JTT4") {
            return std::unique_ptr<Derivs>(new JonathanTyler_JTT4_SecondOrder(
                std::forward<Args>(args)...));
        } else if (name == "JTT6") {
            return std::unique_ptr<Derivs>(new JonathanTyler_JTT6_SecondOrder(
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
        }

        return nullptr;
    }
};

}  // namespace dendroderivs
