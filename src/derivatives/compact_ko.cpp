#include "derivatives/compact_ko.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

#include "dendro.h"

namespace dendroderivs {
namespace {
// Radius-one formula based on field values and directional first derivatives.
// For exact first derivatives its leading smooth-field term is
// -sigma * spacing^3 * d^4(u)/dx^4 / 48. The derivative approximation and
// boundary treatment are part of the full operator; no order label is assumed.
// Keep the arithmetic ordering of the original implementation for comparison.
inline double compact_ko_term(const double* u, const double* du_axis,
                              std::size_t index, std::size_t axis_stride,
                              double spacing, double sigma) {
    return (sigma / (4.0 * spacing)) *
               (u[index - axis_stride] - 2.0 * u[index] +
                u[index + axis_stride]) +
           (sigma / 8.0) *
               (du_axis[index - axis_stride] - du_axis[index + axis_stride]);
}
}  // namespace

unsigned int compact_ko_radius(CompactKOScheme scheme) {
    switch (scheme) {
        case CompactKOScheme::Radius1:
            return 1;
    }
    throw std::invalid_argument("Unsupported Compact KO scheme");
}

static void add_ko(double* rhs, const double* u, const double* du_x,
                   const double* du_y, const double* du_z,
                   const unsigned int* sz, double hx, double hy, double hz,
                   unsigned int padding_width,
                   unsigned int gradient_padding_width, unsigned int bflag,
                   double sigma, CompactKOScheme scheme, bool hybrid) {
    // Validate even for zero strength, so invalid schemes never silently pass.
    const std::size_t radius = compact_ko_radius(scheme);
    if (!sz || gradient_padding_width > padding_width || !std::isfinite(hx) ||
        hx <= 0 || !std::isfinite(hy) || hy <= 0 || !std::isfinite(hz) ||
        hz <= 0 || !std::isfinite(sigma) || sigma < 0)
        throw std::invalid_argument("Invalid Compact KO block configuration");

    if (hybrid && padding_width < 2)
        throw std::invalid_argument(
            "Hybrid KO requires at least two field ghosts");
    const std::size_t pw = padding_width, gpw = gradient_padding_width;
    std::size_t volume = 1;
    for (unsigned int d = 0; d < 3; ++d) {
        if (sz[d] == 0 || pw >= sz[d] || pw >= sz[d] - pw) return;
        if (volume > std::numeric_limits<std::size_t>::max() / sz[d])
            throw std::invalid_argument("Compact KO block size overflows");
        volume *= sz[d];
    }
    if (sigma == 0) return;
    if (!rhs || !u || !du_x || !du_y || !du_z)
        throw std::invalid_argument("Null Compact KO field or gradient");

    const unsigned int lower_bit[3] = {OCT_DIR_LEFT, OCT_DIR_DOWN,
                                       OCT_DIR_BACK};
    const unsigned int upper_bit[3] = {OCT_DIR_RIGHT, OCT_DIR_UP,
                                       OCT_DIR_FRONT};
    std::size_t begin[3], end[3], valid_begin[3], valid_end[3];
    std::size_t field_begin[3], field_end[3];
    for (unsigned int d = 0; d < 3; ++d) {
        field_begin[d] = (bflag & (1u << lower_bit[d])) ? pw : 0;
        field_end[d]   = (bflag & (1u << upper_bit[d])) ? sz[d] - pw : sz[d];
        begin[d]       = pw + ((bflag & (1u << lower_bit[d])) ? 1 : 0);
        end[d]         = sz[d] - pw - ((bflag & (1u << upper_bit[d])) ? 1 : 0);
        valid_begin[d] = std::max(pw, gpw + radius);
        valid_end[d]   = sz[d] - gpw > radius ? sz[d] - gpw - radius : 0;
    }
    const std::size_t nx = sz[0], plane = nx * sz[1];
    for (std::size_t k = begin[2]; k < end[2]; ++k)
        for (std::size_t j = begin[1]; j < end[1]; ++j)
            for (std::size_t i = begin[0]; i < end[0]; ++i) {
                const std::size_t index     = i + nx * j + plane * k;
                double term                 = 0;
                const std::size_t coord[3]  = {i, j, k};
                const std::size_t stride[3] = {1, nx, plane};
                const double spacing[3]     = {hx, hy, hz};
                const double* gradient[3]   = {du_x, du_y, du_z};
                for (unsigned int d = 0; d < 3; ++d) {
                    if (coord[d] >= valid_begin[d] && coord[d] < valid_end[d]) {
                        term += compact_ko_term(u, gradient[d], index,
                                                stride[d], spacing[d], sigma);
                    } else if (hybrid && coord[d] >= field_begin[d] + 2 &&
                               coord[d] + 2 < field_end[d]) {
                        // Centered radius-two KO2 / 3. Never use physical
                        // ghosts or the legacy explicit KO boundary closures.
                        const auto s = stride[d];
                        term += -sigma / (48.0 * spacing[d]) *
                                (u[index - 2 * s] - 4 * u[index - s] +
                                 6 * u[index] - 4 * u[index + s] +
                                 u[index + 2 * s]);
                    }
                }
                rhs[index] += term;
            }
}
void add_compact_ko(double* rhs, const double* u, const double* du_x,
                    const double* du_y, const double* du_z,
                    const unsigned int* sz, double hx, double hy, double hz,
                    unsigned int padding_width,
                    unsigned int gradient_padding_width, unsigned int bflag,
                    double sigma, CompactKOScheme scheme) {
    add_ko(rhs, u, du_x, du_y, du_z, sz, hx, hy, hz, padding_width,
           gradient_padding_width, bflag, sigma, scheme, false);
}

void add_hybrid_ko(double* rhs, const double* u, const double* du_x,
                   const double* du_y, const double* du_z,
                   const unsigned int* sz, double hx, double hy, double hz,
                   unsigned int padding_width,
                   unsigned int gradient_padding_width, unsigned int bflag,
                   double sigma, CompactKOScheme scheme) {
    add_ko(rhs, u, du_x, du_y, du_z, sz, hx, hy, hz, padding_width,
           gradient_padding_width, bflag, sigma, scheme, true);
}

}  // namespace dendroderivs
