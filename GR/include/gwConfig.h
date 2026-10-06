/**
 * @file gwConfig.h
 * @brief Configuration struct for gravitational wave extraction.
 *
 * Pass this to extractFarFieldPsi4() instead of relying on
 * solver-specific global parameters.
 */

#ifndef DENDRO_GR_GW_CONFIG_H
#define DENDRO_GR_GW_CONFIG_H

#include <string>
#include <vector>

namespace dendro_gr {

struct GWExtractionConfig {
    // which indices in the constraint variable array hold Psi4
    unsigned int psi4_real_idx;
    unsigned int psi4_imag_idx;

    // extraction radii (in code units). double, not unsigned: par files parse
    // these with as_floating() and fractional radii are legal.
    std::vector<double> radii;

    // l-modes to extract (m ranges from -l to +l for each). the SWSH table
    // ships l = 2 .. lmax, so every entry must be >= 2 and <= that lmax, but
    // the list itself need not be contiguous.
    std::vector<unsigned int> l_modes;

    // output file prefix
    std::string file_prefix;

    // domain information (needed for coordinate conversion)
    double compd_min[3];
    double compd_max[3];
    double octree_min[3];
    double octree_max[3];

    // output precision
    unsigned int output_precision = 10;
};

}  // namespace dendro_gr

#endif  // DENDRO_GR_GW_CONFIG_H
