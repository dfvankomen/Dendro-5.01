#include "derivatives.h"

#include <iostream>
#include <ranges>

#include "derivatives/derivs_factory.h"

namespace dendroderivs {

DendroDerivatives::DendroDerivatives(
    const std::string derivType_1, const std::string derivType_2,
    const unsigned int ele_order, const std::vector<double> &coeffs_in_1,
    const std::vector<double> &coeffs_in_2, const unsigned int deriv1_matrixID,
    const unsigned int deriv2_matrixID, const std::string inMatrixFilterType_1,
    const std::string inMatrixFilterType_2,
    const std::vector<double> &in_matrix_coeffs_in_1,
    const std::vector<double> &in_matrix_coeffs_in_2,
    const std::string postRHSFilterType, const std::string fallback_1st,
    const std::string fallback_2nd)
    : _n_points_deriv_space(0),
      _n_vars_deriv_space(0),
      _derivative_space(nullptr) {
    // std::cout << "Attempting to create first order derivatives: " <<
    // derivType_1
    //           << std::endl;
    _first_deriv = DerivsFactory::create_first_order(
        derivType_1, ele_order, coeffs_in_1, deriv1_matrixID,
        inMatrixFilterType_1, in_matrix_coeffs_in_1);
    if (!_first_deriv) {
        throw std::runtime_error("Failed to create Derivs object of type: " +
                                 derivType_1);
    }

    // std::cout << "Attempting to create second order derivatives: "
    //           << derivType_2 << std::endl;
    _second_deriv = DerivsFactory::create_second_order(
        derivType_2, ele_order, coeffs_in_2, deriv1_matrixID,
        inMatrixFilterType_2, in_matrix_coeffs_in_2);
    if (!_second_deriv) {
        throw std::runtime_error("Failed to create Derivs object of type: " +
                                 derivType_2);
    }

    // then fetch the filter type
    std::string filterUse = postRHSFilterType;
    if (postRHSFilterType == "default") {
        // choose default KO based on padding size
        switch (ele_order) {
            case 4:
                filterUse = "KO2";
                break;
            case 6:
                filterUse = "KO4";
                break;
            case 8:
                filterUse = "KO6";
                break;
            case 10:
                filterUse = "KO8";
                break;
            default:
                filterUse = "KO4";
                break;
        }
    }
    _filter = FilterFactory::create_filter(filterUse, ele_order);
    if (!_filter) {
        throw std::runtime_error("Failed to create Filter object of type: " +
                                 filterUse);
    }

    // each Derivs instance carries its own pw (p_pw member); no global
    // shared state between DendroDerivatives objects anymore

    // Explicit puncture-block fallbacks: order auto-matched to the configured
    // scheme (or taken from fallback_*), clamped to what pw=ele_order/2 allows.
    const unsigned int pw       = ele_order >> 1u;
    const unsigned int max_eord = (pw >= 4) ? 8u : (pw >= 3) ? 6u : 4u;
    auto explicit_name = [&](const std::string &override_name,
                             const std::string &scheme) -> std::string {
        if (override_name != "auto" && !override_name.empty())
            return override_name;
        unsigned int ord = scheme_order_of_accuracy(scheme, ele_order);
        if (ord > max_eord) {
            dendro_log("[DendroDerivatives] explicit-fallback order " +
                       std::to_string(ord) + " for scheme '" + scheme +
                       "' exceeds what pw=" + std::to_string(pw) +
                       " supports; clamping to E" + std::to_string(max_eord));
            ord = max_eord;
        }
        return "E" + std::to_string(ord);
    };
    const std::string exp1 = explicit_name(fallback_1st, derivType_1);
    const std::string exp2 = explicit_name(fallback_2nd, derivType_2);
    _first_deriv_explicit  = DerivsFactory::create_first_order(
        exp1, ele_order, std::vector<double>(), 0, "none",
        std::vector<double>());
    _second_deriv_explicit = DerivsFactory::create_second_order(
        exp2, ele_order, std::vector<double>(), 0, "none",
        std::vector<double>());
    if (!_first_deriv_explicit || !_second_deriv_explicit) {
        throw std::runtime_error("Failed to create explicit-fallback Derivs (" +
                                 exp1 + ", " + exp2 + ")");
    }
    std::cout << "[DendroDerivatives] puncture explicit fallback: " << exp1
              << " (1st) / " << exp2 << " (2nd)" << std::endl;

    // cache raw stencil function pointers for fast dispatch
    _cache_raw_stencils();
}

}  // namespace dendroderivs
