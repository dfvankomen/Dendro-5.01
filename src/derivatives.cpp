#include "derivatives.h"

#include "derivatives/derivs_factory.h"

namespace dendroderivs {

void initialize_derivatives() { return; }

DendroDerivatives::DendroDerivatives(const std::string derivType_1,
                                     const std::string derivType_2,
                                     unsigned int n, unsigned int pw,
                                     const std::vector<double> &coeffs_in_1,
                                     const std::vector<double> &coeffs_in_2) {
    std::cout << "Attempting to create first order derivatives: " << derivType_1
              << std::endl;
    _first_deriv =
        DerivsFactory::create_first_order(derivType_1, n, pw, coeffs_in_1);
    if (!_first_deriv) {
        throw std::runtime_error("Failed to create Derivs object of type: " +
                                 derivType_1);
    }

    std::cout << "Attempting to create second order derivatives: "
              << derivType_2 << std::endl;
    _second_deriv =
        DerivsFactory::create_second_order(derivType_2, n, pw, coeffs_in_2);
    if (!_second_deriv) {
        throw std::runtime_error("Failed to create Derivs object of type: " +
                                 derivType_2);
    }
}

}  // namespace dendroderivs
