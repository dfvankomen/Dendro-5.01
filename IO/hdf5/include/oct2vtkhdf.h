/**
 * @brief Writes a mesh and its variables to a single VTKHDF file (the
 * HDF5-based VTK format read natively by ParaView), with every rank writing
 * its own partition through parallel HDF5.
 */

#ifndef DENDRO_OCT2VTKHDF_H
#define DENDRO_OCT2VTKHDF_H

#include "mesh.h"

namespace io {
namespace vtkhdf {

/**
 * @brief Writes the given mesh to <fPrefix>.vtkhdf. Holds the same points,
 * cells and arrays as io::vtk::mesh2vtuFine, as one UnstructuredGrid
 * partition per active rank under /VTKHDF.
 *
 * The octree is also written under /Dendro so readers can locate points
 * without ParaView: Octants is (numElements, 4) of (x, y, z, level) in octree
 * coordinates, and the points of global element e are rows
 * [e * nPe, (e + 1) * nPe) of /VTKHDF/Points, x fastest. The group attributes
 * DomainMin, DomainMax, MaxDepth and ElementOrder map octree coordinates to
 * the domain.
 *
 * Collective over the mesh's active communicator.
 *
 * @param [in] pMesh: input mesh
 * @param [in] fPrefix: output file prefix
 * @param [in] numFieldData: number of field (scalar) data entries
 * @param [in] fieldDataNames: names of the field data
 * @param [in] fieldData: field data values
 * @param [in] numPointData: number of point variables
 * @param [in] pointDataNames: names of the point variables
 * @param [in] pointData: point variables, zipped (mesh node) layout
 * @param [in] nCellData: number of cell variables
 * @param [in] cellDNames: names of the cell variables
 * @param [in] cellData: cell variables, one value per local element
 * @param [in] isDGPData: true if the point variables are elemental DG vectors
 * @param [in] compressLevel: deflate level 1-9, 0 disables compression
 */
void mesh2vtkhdfFine(const ot::Mesh *pMesh, const char *fPrefix,
                     unsigned int numFieldData, const char **fieldDataNames,
                     const double *fieldData, unsigned int numPointData,
                     const char **pointDataNames, const double **pointData,
                     unsigned int nCellData = 0, const char **cellDNames = NULL,
                     const double **cellData = NULL, bool isDGPData = false,
                     unsigned int compressLevel = 0);

}  // namespace vtkhdf
}  // namespace io

#endif  // DENDRO_OCT2VTKHDF_H
