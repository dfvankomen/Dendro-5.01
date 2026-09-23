#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "dendro.h"
#include "derivatives/compact_ko.h"

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
template <class F>
void rejects(F f) {
    bool rejected = false;
    try {
        f();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "invalid configuration accepted");
}
// Independent copy of the pre-refactor radius-one expression.
double original(const double* u, const double* g, unsigned int p, int s,
                double h, double sigma) {
    return (sigma / (4.0 * h)) * (u[p - s] - 2.0 * u[p] + u[p + s]) +
           (sigma / 8.0) * (g[p - s] - g[p + s]);
}
void test_hybrid() {
    using namespace dendroderivs;
    const unsigned int sz[3] = {11, 13, 15}, n = 11 * 13 * 15;
    const auto scheme        = CompactKOScheme::Radius1;
    const unsigned int lo[3] = {OCT_DIR_LEFT, OCT_DIR_DOWN, OCT_DIR_BACK};
    const unsigned int hi[3] = {OCT_DIR_RIGHT, OCT_DIR_UP, OCT_DIR_FRONT};
    std::vector<double> u(n), gx(n), gy(n), gz(n), out(n), cko(n);
    double* g[3]     = {gx.data(), gy.data(), gz.data()};
    const double nan = std::numeric_limits<double>::quiet_NaN();
    // Polynomial responses test the smooth-term match and h^3 scaling.
    // Fourier responses distinguish CKO from KO2/3 and detect double
    // application.
    for (unsigned int pw : {2u, 3u})
        for (unsigned int gpw : {pw, pw - 1})
            for (double scale : {0.5, 1.0, 2.0})
                for (double sigma : {0.0, 0.2, 0.4})
                    for (int degree : {0, 1, 2, 3, 4, -1}) {
                        const double h[3]     = {.125 * scale, .25 * scale,
                                                 .5 * scale};
                        const double theta[3] = {.7, 1.1, 1.7};
                        for (unsigned int flags = 0; flags < 64; ++flags) {
                            for (unsigned int k = 0; k < sz[2]; ++k)
                                for (unsigned int j = 0; j < sz[1]; ++j)
                                    for (unsigned int i = 0; i < sz[0]; ++i) {
                                        const unsigned int q[3] = {i, j, k},
                                                           p    = i + 11 * j +
                                                               143 * k;
                                        bool valid_field = true,
                                             valid_grad  = true;
                                        u[p]             = 0;
                                        for (int d = 0; d < 3; ++d) {
                                            const double x = q[d] * h[d];
                                            u[p] +=
                                                degree < 0
                                                    ? std::sin(theta[d] * q[d] +
                                                               .21)
                                                    : std::pow(x, degree);
                                            g[d][p] =
                                                degree < 0
                                                    ? theta[d] / h[d] *
                                                          std::cos(theta[d] *
                                                                       q[d] +
                                                                   .21)
                                                : degree == 0
                                                    ? 0
                                                    : degree *
                                                          std::pow(x,
                                                                   degree - 1);
                                            valid_grad &= q[d] >= gpw &&
                                                          q[d] < sz[d] - gpw;
                                            valid_field &=
                                                !((flags & (1u << lo[d])) &&
                                                  q[d] < pw);
                                            valid_field &=
                                                !((flags & (1u << hi[d])) &&
                                                  q[d] >= sz[d] - pw);
                                        }
                                        if (!valid_field) u[p] = nan;
                                        if (!valid_grad)
                                            gx[p] = gy[p] = gz[p] = nan;
                                    }
                            std::fill(out.begin(), out.end(), 7);
                            std::fill(cko.begin(), cko.end(), 7);
                            add_hybrid_ko(out.data(), u.data(), gx.data(),
                                          gy.data(), gz.data(), sz, h[0], h[1],
                                          h[2], pw, gpw, flags, sigma, scheme);
                            add_compact_ko(cko.data(), u.data(), gx.data(),
                                           gy.data(), gz.data(), sz, h[0], h[1],
                                           h[2], pw, gpw, flags, sigma, scheme);
                            for (unsigned int k = 0; k < sz[2]; ++k)
                                for (unsigned int j = 0; j < sz[1]; ++j)
                                    for (unsigned int i = 0; i < sz[0]; ++i) {
                                        const unsigned int q[3] = {i, j, k},
                                                           p    = i + 11 * j +
                                                               143 * k;
                                        bool active = true, all_cko = true;
                                        for (int d = 0; d < 3; ++d) {
                                            active &=
                                                q[d] >= pw && q[d] < sz[d] - pw;
                                            active &=
                                                !((flags & (1u << lo[d])) &&
                                                  q[d] == pw);
                                            active &=
                                                !((flags & (1u << hi[d])) &&
                                                  q[d] == sz[d] - pw - 1);
                                            all_cko &= q[d] > gpw &&
                                                       q[d] < sz[d] - gpw - 1;
                                        }
                                        double expected = 7;
                                        if (active)
                                            for (int d = 0; d < 3; ++d) {
                                                if (degree == 4)
                                                    expected -=
                                                        sigma *
                                                        std::pow(h[d], 3) / 2;
                                                else if (degree < 0) {
                                                    const double t = theta[d];
                                                    const bool compact =
                                                        q[d] > gpw &&
                                                        q[d] < sz[d] - gpw - 1;
                                                    const double symbol =
                                                        compact
                                                            ? sigma / h[d] *
                                                                  ((std::cos(
                                                                        t) -
                                                                    1) /
                                                                       2 +
                                                                   t *
                                                                       std::sin(
                                                                           t) /
                                                                       4)
                                                            : -sigma /
                                                                  (3 * h[d]) *
                                                                  std::pow(
                                                                      std::sin(
                                                                          t /
                                                                          2),
                                                                      4);
                                                    expected +=
                                                        symbol *
                                                        std::sin(t * q[d] +
                                                                 .21);
                                                }
                                            }
                                        require(std::isfinite(out[p]) &&
                                                    std::abs(out[p] -
                                                             expected) < 2e-10,
                                                "hybrid "
                                                "coverage/scaling/polynomial/"
                                                "Fourier mismatch");
                                        if (!active || all_cko)
                                            require(out[p] == cko[p],
                                                    "hybrid changed valid CKO "
                                                    "or boundary");
                                    }
                        }
                    }
    // Two-point interior: fallback on the opposite internal face must not
    // cross the nearby physical face. All physical field ghosts are poisoned.
    const unsigned int tiny[3] = {6, 6, 6};
    std::vector<double> tu(216, 1), tg(216, nan), tr(216, 0);
    for (unsigned int k = 2; k < 4; ++k)
        for (unsigned int j = 2; j < 4; ++j)
            for (unsigned int i = 2; i < 4; ++i) tg[i + 6 * j + 36 * k] = 0;
    for (unsigned int k = 0; k < 6; ++k)
        for (unsigned int j = 0; j < 6; ++j)
            for (unsigned int i = 0; i < 2; ++i) tu[i + 6 * j + 36 * k] = nan;
    add_hybrid_ko(tr.data(), tu.data(), tg.data(), tg.data(), tg.data(), tiny,
                  1, 1, 1, 2, 2, 1u << OCT_DIR_LEFT, .4, scheme);
    for (double v : tr) require(v == 0, "tiny hybrid read physical ghosts");
    rejects([&] {
        add_hybrid_ko(out.data(), u.data(), gx.data(), gy.data(), gz.data(), sz,
                      1, 1, 1, 1, 1, 0, .4, scheme);
    });
    rejects([&] {
        add_hybrid_ko(out.data(), u.data(), gx.data(), gy.data(), gz.data(), sz,
                      1, 1, 1, 2, 2, 0, 0, static_cast<CompactKOScheme>(99));
    });
}
}  // namespace

int main() {
    test_hybrid();
    using namespace dendroderivs;
    const auto scheme = CompactKOScheme::Radius1;
    require(compact_ko_radius(scheme) == 1, "wrong radius");
    const unsigned int sz[3] = {15, 17, 19};
    const unsigned int pw = 3, nx = sz[0], ny = sz[1];
    const auto index = [=](unsigned int i, unsigned int j, unsigned int k) {
        return i + nx * (j + ny * k);
    };
    const std::size_t n = std::size_t(nx) * ny * sz[2];
    std::vector<double> u(n), gx(n), gy(n), gz(n), rhs(n, 7);
    const double hx = .125, hy = .25, hz = .5, sigma = .4;
    for (unsigned int k = 0; k < sz[2]; ++k)
        for (unsigned int j = 0; j < sz[1]; ++j)
            for (unsigned int i = 0; i < sz[0]; ++i) {
                const auto p = index(i, j, k);
                u[p] =
                    std::sin(i * .31) + std::cos(j * .17) + std::sin(k * .23);
                gx[p] = .31 / hx * std::cos(i * .31);
                gy[p] = -.17 / hy * std::sin(j * .17);
                gz[p] = .23 / hz * std::cos(k * .23);
            }
    const auto u_saved = u, gx_saved = gx, gy_saved = gy, gz_saved = gz;
    const auto apply = [&](unsigned int gpw, unsigned int flags,
                           double strength, CompactKOScheme selected) {
        add_compact_ko(rhs.data(), u.data(), gx.data(), gy.data(), gz.data(),
                       sz, hx, hy, hz, pw, gpw, flags, strength, selected);
    };
    apply(pw, 0, sigma, scheme);
    double max_difference = 0;
    for (unsigned int k = pw + 2; k < sz[2] - pw - 2; ++k)
        for (unsigned int j = pw + 2; j < sz[1] - pw - 2; ++j)
            for (unsigned int i = pw + 2; i < sz[0] - pw - 2; ++i) {
                const auto p = index(i, j, k);
                const double old =
                    7 + (original(u.data(), gx.data(), p, 1, hx, sigma) +
                         original(u.data(), gy.data(), p, nx, hy, sigma) +
                         original(u.data(), gz.data(), p, nx * ny, hz, sigma));
                max_difference =
                    std::max(max_difference, std::abs(rhs[p] - old));
            }
    require(max_difference < 4e-15, "original formula changed");
    require(u == u_saved && gx == gx_saved && gy == gy_saved && gz == gz_saved,
            "inputs modified");

    // Poison all gradient ghosts: no undefined gradient may contribute.
    for (unsigned int k = 0; k < sz[2]; ++k)
        for (unsigned int j = 0; j < sz[1]; ++j)
            for (unsigned int i = 0; i < sz[0]; ++i)
                if (i < pw || i >= sz[0] - pw || j < pw || j >= sz[1] - pw ||
                    k < pw || k >= sz[2] - pw) {
                    gx[index(i, j, k)] = gy[index(i, j, k)] =
                        gz[index(i, j, k)] =
                            std::numeric_limits<double>::quiet_NaN();
                }
    for (unsigned int flags = 0; flags < 64; ++flags) {
        std::fill(rhs.begin(), rhs.end(), 7);
        apply(pw, flags, sigma, scheme);
        for (unsigned int k = 0; k < sz[2]; ++k)
            for (unsigned int j = 0; j < sz[1]; ++j)
                for (unsigned int i = 0; i < sz[0]; ++i) {
                    const auto p        = index(i, j, k);
                    const bool interior = i >= pw && i < sz[0] - pw &&
                                          j >= pw && j < sz[1] - pw &&
                                          k >= pw && k < sz[2] - pw;
                    const bool boundary =
                        (i == pw && (flags & (1u << OCT_DIR_LEFT))) ||
                        (i == sz[0] - pw - 1 &&
                         (flags & (1u << OCT_DIR_RIGHT))) ||
                        (j == pw && (flags & (1u << OCT_DIR_DOWN))) ||
                        (j == sz[1] - pw - 1 && (flags & (1u << OCT_DIR_UP))) ||
                        (k == pw && (flags & (1u << OCT_DIR_BACK))) ||
                        (k == sz[2] - pw - 1 &&
                         (flags & (1u << OCT_DIR_FRONT)));
                    double expected = 7;
                    if (interior && !boundary) {
                        if (i > pw && i < sz[0] - pw - 1)
                            expected +=
                                original(u.data(), gx.data(), p, 1, hx, sigma);
                        if (j > pw && j < sz[1] - pw - 1)
                            expected +=
                                original(u.data(), gy.data(), p, nx, hy, sigma);
                        if (k > pw && k < sz[2] - pw - 1)
                            expected += original(u.data(), gz.data(), p,
                                                 nx * ny, hz, sigma);
                    }
                    require(std::isfinite(rhs[p]) &&
                                std::abs(rhs[p] - expected) < 4e-15,
                            "invalid region/physical boundary/tangential term");
                }
    }
    gx = gx_saved;
    gy = gy_saved;
    gz = gz_saved;
    std::fill(rhs.begin(), rhs.end(), 7);
    apply(pw - 1, 0, sigma, scheme);
    for (unsigned int k = pw; k < sz[2] - pw; ++k)
        for (unsigned int j = pw; j < sz[1] - pw; ++j)
            for (unsigned int i = pw; i < sz[0] - pw; ++i) {
                const auto p = index(i, j, k);
                const double expected =
                    7 + original(u.data(), gx.data(), p, 1, hx, sigma) +
                    original(u.data(), gy.data(), p, nx, hy, sigma) +
                    original(u.data(), gz.data(), p, nx * ny, hz, sigma);
                require(std::abs(rhs[p] - expected) < 4e-15,
                        "gradient halo ignored");
            }
    const auto saved = rhs;
    apply(pw, 0, 0, scheme);
    require(rhs == saved, "zero strength changed RHS");
    rejects([&] { apply(pw, 0, 0, static_cast<CompactKOScheme>(99)); });
    rejects([&] { apply(pw + 1, 0, sigma, scheme); });
    rejects([&] { apply(pw, 0, -1, scheme); });
    const unsigned int empty[3] = {0, 0, 0}, tiny[3] = {6, 6, 6};
    add_compact_ko(nullptr, nullptr, nullptr, nullptr, nullptr, empty, 1, 1, 1,
                   3, 3, 0, 1, scheme);
    add_compact_ko(nullptr, nullptr, nullptr, nullptr, nullptr, tiny, 1, 1, 1,
                   3, 3, 0, 1, scheme);

    // Exact polynomial checks: constant is annihilated; x^4+y^4+z^4 gives
    // -sigma*(hx^3+hy^3+hz^3)/2 with exact directional first derivatives.
    for (int degree : {0, 4}) {
        for (unsigned int k = 0; k < sz[2]; ++k)
            for (unsigned int j = 0; j < sz[1]; ++j)
                for (unsigned int i = 0; i < sz[0]; ++i) {
                    const auto p   = index(i, j, k);
                    const double x = i * hx, y = j * hy, z = k * hz;
                    u[p]  = degree == 0 ? 1
                                        : std::pow(x, 4) + std::pow(y, 4) +
                                             std::pow(z, 4);
                    gx[p] = degree == 0 ? 0 : 4 * x * x * x;
                    gy[p] = degree == 0 ? 0 : 4 * y * y * y;
                    gz[p] = degree == 0 ? 0 : 4 * z * z * z;
                }
        std::fill(rhs.begin(), rhs.end(), 0);
        apply(pw - 1, 0, sigma, scheme);
        const double expected =
            degree == 0
                ? 0
                : -sigma / 2 * (hx * hx * hx + hy * hy * hy + hz * hz * hz);
        for (unsigned int k = pw; k < sz[2] - pw; ++k)
            for (unsigned int j = pw; j < sz[1] - pw; ++j)
                for (unsigned int i = pw; i < sz[0] - pw; ++i)
                    require(std::abs(rhs[index(i, j, k)] - expected) < 1e-11,
                            "polynomial check failed");
    }
    // Fourier damping for exact derivatives and the centered E4/E6/E8
    // modified wavenumbers. Test the numerical kernel, not just the symbol.
    for (int order : {0, 4, 6, 8}) {
        for (unsigned int mode = 1; mode <= 32; ++mode) {
            const double theta = std::acos(-1.0) * mode / 32;
            double modified    = theta;
            if (order == 4)
                modified = (8 * std::sin(theta) - std::sin(2 * theta)) / 6;
            if (order == 6)
                modified = (45 * std::sin(theta) - 9 * std::sin(2 * theta) +
                            std::sin(3 * theta)) /
                           30;
            if (order == 8)
                modified = 8.0 / 5 * std::sin(theta) -
                           2.0 / 5 * std::sin(2 * theta) +
                           8.0 / 105 * std::sin(3 * theta) -
                           1.0 / 140 * std::sin(4 * theta);
            const double symbol =
                sigma / hx *
                ((std::cos(theta) - 1) / 2 + modified * std::sin(theta) / 4);
            require(symbol <= 1e-14, "positive Fourier dissipation symbol");
            for (unsigned int k = 0; k < sz[2]; ++k)
                for (unsigned int j = 0; j < sz[1]; ++j)
                    for (unsigned int i = 0; i < sz[0]; ++i) {
                        const auto p = index(i, j, k);
                        u[p]         = std::sin(theta * i + .21);
                        gx[p] = modified / hx * std::cos(theta * i + .21);
                        gy[p] = gz[p] = 0;
                    }
            std::fill(rhs.begin(), rhs.end(), 0);
            apply(pw - 1, 0, sigma, scheme);
            for (unsigned int k = pw; k < sz[2] - pw; ++k)
                for (unsigned int j = pw; j < sz[1] - pw; ++j)
                    for (unsigned int i = pw; i < sz[0] - pw; ++i) {
                        const auto p = index(i, j, k);
                        require(std::abs(rhs[p] - symbol * u[p]) < 2e-14,
                                "Fourier response differs from damping symbol");
                    }
        }
    }
    std::cout << "Compact and hybrid KO tests passed; original-radius1 max difference = "
              << max_difference << '\n';
}
