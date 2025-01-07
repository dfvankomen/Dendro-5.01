#include "derivatives.h"

#include "derivatives/derivs_factory.h"

namespace dendroderivs {

unsigned int DENDRO_DERIVS_PW = 0;

DendroDerivatives::DendroDerivatives(const std::string derivType_1,
                                     const std::string derivType_2,
                                     const unsigned int ele_order,
                                     const std::vector<double> &coeffs_in_1,
                                     const std::vector<double> &coeffs_in_2)
    : _n_points_deriv_space(0),
      _n_vars_deriv_space(0),
      _derivative_space(nullptr) {
    // std::cout << "Attempting to create first order derivatives: " <<
    // derivType_1
    //           << std::endl;
    _first_deriv =
        DerivsFactory::create_first_order(derivType_1, ele_order, coeffs_in_1);
    if (!_first_deriv) {
        throw std::runtime_error("Failed to create Derivs object of type: " +
                                 derivType_1);
    }

    // std::cout << "Attempting to create second order derivatives: "
    //           << derivType_2 << std::endl;
    _second_deriv =
        DerivsFactory::create_second_order(derivType_2, ele_order, coeffs_in_2);
    if (!_second_deriv) {
        throw std::runtime_error("Failed to create Derivs object of type: " +
                                 derivType_2);
    }

    const unsigned int pw = ele_order / 2;

    // the value starts at 0, so it should be set. if we're not zero and it's
    // different, print a warning
    if (DENDRO_DERIVS_PW != 0 && DENDRO_DERIVS_PW != pw) {
        std::cout << "WARNING: Overwiting internal saved Padding Width from "
                  << DENDRO_DERIVS_PW << " to " << pw
                  << "! This could cause issues if there are multiple "
                     "DendroDerivative objects!";
        DENDRO_DERIVS_PW = pw;
    } else {
        // otherwise, we can just set it, because it's probably just 0
        DENDRO_DERIVS_PW = pw;
    }
}

}  // namespace dendroderivs
