/**
 * @file gwExtract.h
 * @brief Gravitational wave extraction via Psi4 decomposition.
 *
 * @author Milinda Fernando (original), refactored for generic use.
 *
 * Extracts gravitational wave content by interpolating Psi4 to
 * extraction spheres and decomposing into spin-weighted spherical
 * harmonic (l,m) modes. Theory-agnostic -- just needs mesh + Psi4.
 *
 * Based on: "Extraction of Gravitational Waves in Numerical Relativity"
 *           https://arxiv.org/abs/1606.02532
 */

#ifndef DENDRO_GR_GWEXTRACT_H
#define DENDRO_GR_GWEXTRACT_H

#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "daUtils.h"
#include "dendro.h"
#include "mesh.h"
#include "point.h"

#include "gwConfig.h"

namespace dendro_gr {

// lebedev quadrature points and SWSH values
#include "nrswsh.h"

/**
 * @brief Extract far-field Psi4 at extraction spheres and decompose into
 *        spin-weighted spherical harmonic modes.
 *
 * @param mesh Active mesh.
 * @param cVar Constraint variable array (indexed by constraint enum).
 * @param timestep Current timestep number.
 * @param time Current simulation time.
 * @param config Extraction configuration (radii, l-modes, Psi4 indices, etc.).
 */
template <typename T>
void extractFarFieldPsi4(const ot::Mesh* mesh, const T** cVar,
                         unsigned int timestep, double time,
                         const GWExtractionConfig& config) {

    const unsigned int rankActive = mesh->getMPIRank();
    const unsigned int npesActive = mesh->getMPICommSize();
    MPI_Comm commActive = mesh->getMPICommunicator();
    const unsigned int globalRank = mesh->getMPIRankGlobal();

    const unsigned int numRadii = config.radii.size();
    const unsigned int numLModes = config.l_modes.size();

    // count total (l,m) modes
    unsigned int totalModes = 0;
    for (unsigned int l = 0; l < numLModes; l++)
        totalModes += 2 * config.l_modes[l] + 1;

    const unsigned int TOTAL_MODES = totalModes;

    // allocate coefficient arrays
    DendroComplex* swsh_coeff = new DendroComplex[numRadii * TOTAL_MODES];
    DendroComplex* swsh_coeff_g = new DendroComplex[numRadii * TOTAL_MODES];

    std::vector<unsigned int> lmCounts(numLModes);
    std::vector<unsigned int> lmOffset(numLModes);

    for (unsigned int l = 0; l < numLModes; l++)
        lmCounts[l] = 2 * config.l_modes[l] + 1;

    lmOffset[0] = 0;
    omp_par::scan(&(*(lmCounts.begin())), &(*(lmOffset.begin())), numLModes);

    // psi4 L2 norms per radius
    std::vector<double> psi4L2R(numRadii, 0);
    std::vector<double> psi4L2I(numRadii, 0);
    std::vector<double> psi4L2R_g(numRadii, 0);
    std::vector<double> psi4L2I_g(numRadii, 0);

    if (mesh->isActive()) {
        const unsigned int numPts = LEBEDEV_NUM_PTS;

        std::vector<double> domain_coords(3 * numPts);
        std::vector<double> psi4_real(numPts);
        std::vector<double> psi4_imag(numPts);
        std::vector<unsigned int> validIndex;

        // domain bounds for interpolation
        Point grid_limits[2];
        Point domain_limits[2];

        grid_limits[0] = Point(config.octree_min[0], config.octree_min[1],
                               config.octree_min[2]);
        grid_limits[1] = Point(config.octree_max[0], config.octree_max[1],
                               config.octree_max[2]);

        domain_limits[0] = Point(config.compd_min[0], config.compd_min[1],
                                 config.compd_min[2]);
        domain_limits[1] = Point(config.compd_max[0], config.compd_max[1],
                                 config.compd_max[2]);

        for (unsigned int k = 0; k < numRadii; k++) {
            // build extraction sphere coordinates
            for (unsigned int pts = 0; pts < numPts; pts++) {
                domain_coords[3 * pts + 0] =
                    config.radii[k] * sin(LEBEDEV_THETA[pts]) * cos(LEBEDEV_PHI[pts]);
                domain_coords[3 * pts + 1] =
                    config.radii[k] * sin(LEBEDEV_THETA[pts]) * sin(LEBEDEV_PHI[pts]);
                domain_coords[3 * pts + 2] =
                    config.radii[k] * cos(LEBEDEV_THETA[pts]);
            }

            // interpolate Psi4 real part to sphere
            validIndex.clear();
            ot::da::interpolateToCoords(
                mesh, cVar[config.psi4_real_idx],
                domain_coords.data(), domain_coords.size(), grid_limits,
                domain_limits, &(*(psi4_real.begin())), validIndex);

            // interpolate Psi4 imaginary part to sphere
            validIndex.clear();
            ot::da::interpolateToCoords(
                mesh, cVar[config.psi4_imag_idx],
                domain_coords.data(), domain_coords.size(), grid_limits,
                domain_limits, &(*(psi4_imag.begin())), validIndex);

            // accumulate L2 norms
            for (unsigned int index = 0; index < validIndex.size(); index++) {
                psi4L2R[k] += psi4_real[validIndex[index]] * psi4_real[validIndex[index]];
                psi4L2I[k] += psi4_imag[validIndex[index]] * psi4_imag[validIndex[index]];
            }

            // project onto SWSH modes
            for (unsigned int l = 0; l < numLModes; l++) {
                for (unsigned int m = 0; m < 2 * config.l_modes[l] + 1; m++) {
                    swsh_coeff[k * TOTAL_MODES + lmOffset[l] + m] = DendroComplex(0.0, 0.0);
                    for (unsigned int index = 0; index < validIndex.size(); index++) {
                        DendroComplex psi4(psi4_real[validIndex[index]],
                                           psi4_imag[validIndex[index]]);
                        swsh_coeff[k * TOTAL_MODES + lmOffset[l] + m] +=
                            psi4 * std::conj(LEBEDEV_SWSH[lmOffset[l] + m][validIndex[index]]) *
                            LEBEDEV_W[validIndex[index]];
                    }
                    swsh_coeff[k * TOTAL_MODES + lmOffset[l] + m] *= (4 * M_PI);
                }
            }
        }
    }

    // reduce across all MPI ranks
    MPI_Reduce(swsh_coeff, swsh_coeff_g, numRadii * TOTAL_MODES,
               MPI_DOUBLE_COMPLEX, MPI_SUM, 0, commActive);
    MPI_Reduce(psi4L2R.data(), psi4L2R_g.data(), numRadii,
               MPI_DOUBLE, MPI_SUM, 0, commActive);
    MPI_Reduce(psi4L2I.data(), psi4L2I_g.data(), numRadii,
               MPI_DOUBLE, MPI_SUM, 0, commActive);

    // write output files (rank 0 only)
    if (!globalRank) {
        // L2 norm file
        char fName[256];
        sprintf(fName, "%s_GW_L2.dat", config.file_prefix.c_str());
        std::ofstream fileGW(fName, std::ofstream::app);

        if (timestep == 0) {
            fileGW << "TimeStep\t t\t";
            for (unsigned int r = 0; r < numRadii; r++)
                fileGW << "r=" << config.radii[r] << "\t";
            fileGW << std::endl;
        }

        fileGW << timestep << "\t" << time << "\t";
        for (unsigned int r = 0; r < numRadii; r++) {
            psi4L2R_g[r] = sqrt(psi4L2R_g[r]);
            psi4L2I_g[r] = sqrt(psi4L2I_g[r]);
            fileGW << "(" << psi4L2R_g[r] << "," << psi4L2I_g[r] << ")\t";
        }
        fileGW << std::endl;
        fileGW.close();

        // per-mode files
        for (unsigned int l = 0; l < numLModes; l++) {
            for (unsigned int m = 0; m < 2 * config.l_modes[l] + 1; m++) {
                sprintf(fName, "%s_GW_l%d_m%d.dat", config.file_prefix.c_str(),
                        config.l_modes[l], (int)(m - config.l_modes[l]));
                std::ofstream fileMode(fName, std::ofstream::app);

                if (timestep == 0) {
                    fileMode << "TimeStep\t t\t";
                    for (unsigned int r = 0; r < numRadii; r++)
                        fileMode << "r" << r << "\t";
                    fileMode << std::endl;
                }

                fileMode.precision(config.output_precision);
                fileMode << std::scientific;
                fileMode << timestep << "\t" << time << "\t";
                for (unsigned int r = 0; r < numRadii; r++)
                    fileMode << swsh_coeff_g[r * TOTAL_MODES + lmOffset[l] + m] << "\t";
                fileMode << std::endl;
                fileMode.close();
            }
        }
    }

    delete[] swsh_coeff;
    delete[] swsh_coeff_g;
}

}  // namespace dendro_gr

#endif  // DENDRO_GR_GWEXTRACT_H
