/**
 * @file grDataUtils.h
 * @brief Generic GR data utilities for Dendro-based solvers.
 *
 * Provides BH coordinate extraction, refinement helpers, and other
 * data processing utilities that are shared across GR formulations.
 */

#ifndef DENDRO_GR_DATAUTILS_H
#define DENDRO_GR_DATAUTILS_H

#include "TreeNode.h"
#include "mesh.h"
#include "point.h"

namespace dendro_gr {

/**
 * @brief Extracts black hole coordinates by finding local minima of a scalar field.
 *
 * Typically used with the conformal factor (chi) to locate BH punctures.
 *
 * @param pMesh Current mesh.
 * @param var Scalar field to search for minima (e.g., chi).
 * @param tolerance Threshold for considering a point as a minimum.
 * @param ptIn Previous known BH locations (for tracking).
 * @param numPt Number of BHs to track.
 * @param ptOut Output: new BH locations (valid on rank 0 of active comm).
 */
void extractBHCoords(const ot::Mesh *pMesh, const DendroScalar *var,
                     double tolerance, const Point *ptIn, unsigned int numPt,
                     Point *ptOut);

/**
 * @brief Write BH coordinates to a file.
 * @param fPrefix File prefix.
 * @param ptLocs BH locations.
 * @param numPt Number of BHs.
 * @param timestep Current timestep.
 * @param time Current simulation time.
 */
void writeBHCoords(const char *fPrefix, const Point *ptLocs, unsigned int numPt,
                   unsigned int timestep, double time);

}  // namespace dendro_gr

#endif  // DENDRO_GR_DATAUTILS_H
