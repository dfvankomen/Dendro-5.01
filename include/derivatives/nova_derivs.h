#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "lapac.h"

/**
 * NOVA Derivatives taken from William Kevin Black's repository:
 * 
 * https://bitbucket.org/wkblack/nova_alpha/src/master/
 *
 */

namespace dendroderivs {
namespace nova {

/**
 * @brief Adds negative of left shift to right shift
 *
 * This function takes in a vector, q, and adds the negative of its left shit
 * (right padded with zero) to its right shift (left padded with zero). Compared
 * to the original function in Python, this avoids creating two vectors and
 * instead creates one.
 *
 * @tparam T The type of elements in the vector (must support unary minus)
 * @param q The input vector to be negated
 * @return std::vector<T> A new vector that's been "shift added"
 *
 * @note This function creates a new vector each time!
 */
template <typename T>
inline std::vector<T> _shift_add(std::vector<T> q) {
    std::vector<T> left_shift_neg(q.size());

    // left_shift_neg should be negative q, so std::transform will handle that
    std::transform(q.begin(), q.end(), left_shift_neg.begin(),
                   [](const T& x) { return -x; });

    // add 0 to the end
    left_shift_neg.push_back(0);

    // then just add q to it, but offset by nothing at first
    for (size_t i = 0; i < left_shift_neg.size(); i++) {
        if (i == 0) {
            continue;
        }

        left_shift_neg[i] = left_shift_neg[i] + q[i - 1];
    }

    return left_shift_neg;
}

/**
 * @brief Calculate the quanta for a given derivative of "power" p
 *
 * This function gets the finite difference quanta for a derivative 'p'. It
 * recursively applies the _shift_add function p times, starting with [1].
 *
 * @tparam T the type of elements in the vector (must support unary minus)
 * @param p The number of times to apply _shift_add
 * @return std::vector<T> The resulting vector after computation
 *
 * @note This is a very "recursive" function that creates new vectors on each
 * input.
 */
template <typename T>
inline std::vector<T> quanta(uint64_t p) {
    if (p == 0) {
        return std::vector<T>{1};
    } else {
        return _shift_add<T>(quanta<T>(p - 1));
    }
}

template <typename T>
inline bool centered(const std::vector<T>& j) {
    int N        = j.size();

    auto abs_val = [](const T& val) ->
        typename std::conditional<std::is_floating_point<T>::value, T,
                                  double>::type {
            if constexpr (std::is_floating_point<T>::value) {
                return std::abs(val);
            } else if constexpr (std::is_integral<T>::value) {
                return std::abs(static_cast<double>(val));
            } else {
                return std::abs(std::real(val));
            }
        };

    // check if j is centered around 0
    bool is_centered = (N % 2 == 1) && (abs_val(j[N / 2]) < 1e-10);

    // check if j is symmetric
    for (int i = 0; i < N / 2; ++i) {
        if (abs_val(j[i] + j[N - 1 - i]) > 1e-10) {
            return false;
        }
    }

    return is_centered;
}

inline uint64_t factorial(uint32_t n) {
    if (n <= 1) return 1;

    uint64_t result = 1;
    for (uint32_t i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}

template <typename T = double>
std::vector<T> generate_coeffs(std::vector<int32_t> j, size_t k, size_t p = 1,
                               bool catch_errors = true,
                               bool use_analytic = true, double eps = 1e-12) {
    size_t N = j.size();

    std::cout << N << ", " << k << ", " << p << std::endl;

    // check the corner cases
    if (k < p + 1) {
        throw std::invalid_argument(
            "Insufficient constraints, k must be increased");
    }
    if (k > N) {
        throw std::invalid_argument(
            "Overconstraining equation, increase stencil size or decrease "
            "constraints");
    }
    if (p > (N - 1)) {
        throw std::invalid_argument("Insufficient stencil width, increase j");
    } else if (p == (N - 1)) {
        // early exit to use quanta if p == N-1
        return quanta<T>(p);
    }

    // check for analytic
    if (use_analytic) {
        bool is_centered = centered(j);

        // use explicit solutions if this is true
        if (is_centered) {
            size_t n = (N - 1) / 2;
            // pure first derivatives
            if (p == 1 && k == 2) {
                T factor = T(3.0 / (n * (n + 1) * (2 * n + 1)));

                std::vector<T> result(N);
                for (size_t i = 0; i < N; ++i) {
                    result[i] = factor * j[i];
                }
                return result;
            }
            // pure second derivative
            if (p == 2 && k == 3) {
                T factor = T(30.0 / ((2 * n + 1) * (n + 1) * n * (2 * n - 1) *
                                     (2 * n + 3)));

                std::vector<T> result(N);
                for (size_t i = 0; i < N; ++i) {
                    result[i] = factor * (3.0 * j[i] * j[i] - n * (n - 1));
                }
                return result;
            }
            // pure third derivative
            if (p == 3 && k == 4) {
                T denominator = T(n - 1) * n * (n + 1) * (n + 2) * (2 * n - 1) *
                                (2 * n + 1) * (2 * n + 3);

                std::vector<T> result(N);
                for (size_t i = 0; i < N; ++i) {
                    T j_cubed = j[i] * j[i] * j[i];
                    T numerator =
                        T(5.0 * j_cubed - j[i] * (3 * n * n + 3 * n - 1));
                    result[i] = T(210.0 * (numerator / denominator));
                }
                return result;
            }
        }
    }

    // otherwise, if none of that is true, then we want to generate non-exact

    // generate h values
    std::vector<T> h_values;
    h_values.reserve(2 * k - 1);

    for (size_t i = 0; i < 2 * k - 1; ++i) {
        std::vector<T> powered(j.size());
        for (size_t idx = 0; idx < j.size(); ++idx) {
            powered[idx] = std::pow(j[idx], i);
        }
        h_values.push_back(
            std::accumulate(powered.begin(), powered.end(), 0.0));
    }

    // then create the M matrix
    std::vector<T> M(k * k);

    // COLUMN order matrix
    for (int jj = 0; jj < k; ++jj) {
        for (int ii = 0; ii < k; ++ii) {
            M[ii + jj * k] = h_values[ii + jj];
        }
    }

    // B vector
    std::vector<T> b(k, 0.0);

    uint64_t p_bang = factorial(p);
    b[p]            = -2.0 * T(p_bang);

    // solve for the lambda values
    std::vector<T> lambs = lapack::solveLinearSystem(M, b, k, 1);

    // then calculate the coefficients
    std::vector<T> c(j.size(), 0);

    for (int ii = 0; ii < k; ++ii) {
        T lamb = lambs[ii];
        for (size_t idx = 0; idx < j.size(); ++idx) {
            c[idx] += lamb * std::pow(j[idx], ii);
        }
    }
    // be sure to scale by -0.5
    for (size_t idx = 0; idx < c.size(); ++idx) {
        c[idx] *= -0.5;
    }

    if (catch_errors) {
        for (size_t pwr = 0; pwr < p; ++pwr) {
            T sum = 0;
            for (size_t idx = 0; idx < j.size(); ++idx) {
                sum += std::pow(j[idx], pwr) * c[idx];
            }

            if (std::abs(sum) >= eps) {
                std::string mssg =
                    "\\sum_j j^" + std::to_string(pwr) + " a_j \\ne 0";
                throw std::runtime_error(mssg);
            }
        }

        T sum = 0;
        for (size_t idx = 0; idx < j.size(); ++idx) {
            sum += std::pow(j[idx], p) * c[idx];
        }

        if (std::abs(sum - p_bang) >= eps) {
            std::string mssg = "\\sum_j j^" + std::to_string(p) + " a_j \\ne " +
                               std::to_string(p) + "!";
            throw std::runtime_error(mssg);
        }
    }

    return c;
}

template <typename T>
inline std::vector<std::vector<T>> create_nova_boundaries(size_t n, size_t nb,
                                                          size_t k, size_t p,
                                                          double eps = 1e-3) {
    // vector of points to feed into the generate_coeffs function
    std::vector<int32_t> j(n);

    std::vector<std::vector<T>> coeffs;

    // we'll only generate "boundary" coefficients, another function should be
    // called for internal points
    for (int32_t ii = 0; ii < nb; ++ii) {
        // refill the j vector with the "offset" integers
        for (int32_t jj = 0; jj < n; ++jj) {
            j[jj] = jj - ii;
        }

        coeffs.push_back(generate_coeffs(j, k, p, true, true, eps));
    }

    // TODO: this is wrong
    return coeffs;
}

}  // namespace nova
}  // namespace dendroderivs
