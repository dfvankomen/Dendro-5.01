#include "wideprolong.h"

#include <cmath>

namespace dendro {
namespace wideprolong {

void build_1d(unsigned int eleOrder, unsigned int child, unsigned int ext_lo,
              unsigned int ext_hi, unsigned int width,
              std::vector<double> &op, unsigned int &n_in) {
    const unsigned int nrp = eleOrder + 1;
    n_in                   = nrp + ext_lo + ext_hi;

    std::vector<double> xs(n_in);
    for (unsigned int j = 0; j < n_in; j++)
        xs[j] = ((double)j - (double)ext_lo) / (double)eleOrder;

    build_1d_at(eleOrder, child, xs, width, op);
}

void build_1d_at(unsigned int eleOrder, unsigned int child,
                 const std::vector<double> &xs, unsigned int width,
                 std::vector<double> &op) {
    const unsigned int nrp  = eleOrder + 1;
    const unsigned int n_in = (unsigned int)xs.size();

    // clamp to what the caller could actually reach; with no extension this
    // degenerates to the full-element interpolant, i.e. the narrow operator.
    if (width > n_in) width = n_in;
    if (width < 2) width = 2;

    op.assign((size_t)nrp * (size_t)n_in, 0.0);

    std::vector<double> w(width);

    for (unsigned int i = 0; i < nrp; i++) {
        const double xt =
            0.5 * (double)child + (double)i / (double)(2 * eleOrder);

        // Slide a contiguous window of `width` nodes as close to centred on
        // the target as the array allows; at the ends it goes one-sided.
        // Chosen by coordinate rather than index so a graded array (a
        // neighbour at another refinement level) is handled correctly.
        int centre = 0;
        while (centre + 1 < (int)n_in && xs[centre + 1] <= xt) centre++;
        int s = centre - (int)(width / 2) + 1;
        if (s < 0) s = 0;
        if (s + (int)width > (int)n_in) s = (int)n_in - (int)width;

        for (unsigned int a = 0; a < width; a++) {
            double num = 1.0;
            for (unsigned int b = 0; b < width; b++) {
                if (a == b) continue;
                num *= (xt - xs[s + b]) / (xs[s + a] - xs[s + b]);
            }
            w[a] = num;
        }

        for (unsigned int a = 0; a < width; a++)
            op[(size_t)i * (size_t)n_in + (size_t)s + a] = w[a];
    }
}

void apply_x(unsigned int n_in, unsigned int n_out, unsigned int ny,
             unsigned int nz, const double *A, const double *X, double *Y) {
    for (unsigned int k = 0; k < nz; k++)
        for (unsigned int j = 0; j < ny; j++) {
            const double *xrow = X + (size_t)(k * ny + j) * n_in;
            double *yrow       = Y + (size_t)(k * ny + j) * n_out;
            for (unsigned int o = 0; o < n_out; o++) {
                double acc = 0.0;
                for (unsigned int i = 0; i < n_in; i++)
                    acc += A[(size_t)o * n_in + i] * xrow[i];
                yrow[o] = acc;
            }
        }
}

void apply_y(unsigned int n_in, unsigned int n_out, unsigned int nx,
             unsigned int nz, const double *A, const double *X, double *Y) {
    for (unsigned int k = 0; k < nz; k++)
        for (unsigned int o = 0; o < n_out; o++)
            for (unsigned int i0 = 0; i0 < nx; i0++) {
                double acc = 0.0;
                for (unsigned int i = 0; i < n_in; i++)
                    acc += A[(size_t)o * n_in + i] *
                           X[(size_t)(k * n_in + i) * nx + i0];
                Y[(size_t)(k * n_out + o) * nx + i0] = acc;
            }
}

void apply_z(unsigned int n_in, unsigned int n_out, unsigned int nx,
             unsigned int ny, const double *A, const double *X, double *Y) {
    const size_t plane = (size_t)nx * ny;
    for (unsigned int o = 0; o < n_out; o++)
        for (size_t t = 0; t < plane; t++) {
            double acc = 0.0;
            for (unsigned int i = 0; i < n_in; i++)
                acc += A[(size_t)o * n_in + i] * X[(size_t)i * plane + t];
            Y[(size_t)o * plane + t] = acc;
        }
}

size_t scratch_size(unsigned int eleOrder, unsigned int nx_in,
                    unsigned int ny_in, unsigned int nz_in) {
    const size_t m  = eleOrder + 1;
    const size_t s1 = (size_t)nz_in * ny_in * m;  // after the x sweep
    const size_t s2 = (size_t)nz_in * m * m;      // after the y sweep
    return (s1 > s2) ? s1 : s2;
}

void apply_3d(unsigned int eleOrder, const double *opx, unsigned int nx_in,
              const double *opy, unsigned int ny_in, const double *opz,
              unsigned int nz_in, const double *in, double *out, double *w1,
              double *w2) {
    const unsigned int m = eleOrder + 1;

    // x: [nz_in][ny_in][nx_in] -> [nz_in][ny_in][m]
    apply_x(nx_in, m, ny_in, nz_in, opx, in, w1);
    // y: [nz_in][ny_in][m] -> [nz_in][m][m]
    apply_y(ny_in, m, m, nz_in, opy, w1, w2);
    // z: [nz_in][m][m] -> [m][m][m]
    apply_z(nz_in, m, m, m, opz, w2, out);
}

}  // namespace wideprolong
}  // namespace dendro
