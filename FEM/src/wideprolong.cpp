#include "wideprolong.h"

#include <cmath>

namespace dendro {
namespace wideprolong {

void build_1d(unsigned int eleOrder, unsigned int child, unsigned int ext_lo,
              unsigned int ext_hi, unsigned int width,
              std::vector<double> &op, unsigned int &n_in) {
    const unsigned int nrp = eleOrder + 1;
    n_in                   = nrp + ext_lo + ext_hi;

    // clamp to what the caller could actually reach; with no extension this
    // degenerates to the full-element interpolant, i.e. the narrow operator.
    if (width > n_in) width = n_in;
    if (width < 2) width = 2;

    op.assign((size_t)nrp * (size_t)n_in, 0.0);

    std::vector<double> xs(n_in);
    for (unsigned int j = 0; j < n_in; j++)
        xs[j] = ((double)j - (double)ext_lo) / (double)eleOrder;

    std::vector<double> w(width);

    for (unsigned int i = 0; i < nrp; i++) {
        const double xt =
            0.5 * (double)child + (double)i / (double)(2 * eleOrder);

        // slide a contiguous window of `width` nodes as close to centred on
        // the target as the array allows; at the ends it goes one-sided.
        int centre = (int)std::lround(xt * (double)eleOrder) + (int)ext_lo;
        int s      = centre - (int)(width / 2);
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

}  // namespace wideprolong
}  // namespace dendro
