#include <cstdint>
#include <iostream>
#include <vector>

#include "nova_derivs.h"

int main(int argc, char *argv[]) {
    // std::vector<size_t> points = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    // std::vector<int32_t> points = {-1, 0, 1};
    // std::vector<int32_t> points = {-10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
    // 0};
    std::vector<int32_t> points = {-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6};
    size_t k                    = 7;

    std::vector<double> coeffs =
        dendroderivs::nova::generate_coeffs(points, k, 1, true, true, 1e-3);

    std::cout << "COEFFS: ";
    for (auto &coeff : coeffs) {
        std::cout << coeff << ", ";
    }
    std::cout << std::endl;

    auto coeffs_bdy =
        dendroderivs::nova::create_nova_boundaries<double>(13, 7, 7, 1);

    std::cout << "Boundary coefficients:" << std::endl;
    for (auto coefs : coeffs_bdy) {
        for (auto c : coefs) {
            std::cout << c << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;

    auto coeffs_again =
        dendroderivs::nova::create_continuous_nova<double>(13, 4, 7, 1);

    std::cout << "COEFFS: AGAIN: ";
    for (auto &coeff : coeffs_again) {
        std::cout << coeff << ", ";
    }
    std::cout << std::endl;

    return 0;
}
