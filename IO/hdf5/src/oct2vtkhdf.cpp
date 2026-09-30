/**
 * @brief VTKHDF writer, see oct2vtkhdf.h.
 */

#include "oct2vtkhdf.h"

#include <hdf5.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "oct2vtk.h"

namespace io {
namespace vtkhdf {

namespace {

constexpr hsize_t VTKHDF_CHUNK_ROWS = 1u << 16;

/**@brief this rank's rows within a dataset shared by all ranks. */
struct Slice {
    hsize_t offset;
    hsize_t count;
    hsize_t total;
};

Slice make_slice(MPI_Comm comm, hsize_t count) {
    unsigned long long local = count, offset = 0, total = 0;
    MPI_Exscan(&local, &offset, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);
    MPI_Allreduce(&local, &total, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);

    int rank;
    MPI_Comm_rank(comm, &rank);
    if (!rank) offset = 0;
    return {offset, count, total};
}

/**@brief creates a (total x nComp) dataset and writes this rank's slice
 * collectively. */
void write_rows(hid_t loc, const char *name, hid_t fileType, hid_t memType,
                const Slice &s, hsize_t nComp, const void *buf, hid_t dxpl,
                unsigned int compressLevel) {
    const int nDims       = (nComp > 1) ? 2 : 1;
    const hsize_t dims[2] = {s.total, nComp};
    hid_t fSpace          = H5Screate_simple(nDims, dims, NULL);

    hid_t dcpl            = H5Pcreate(H5P_DATASET_CREATE);
    if (compressLevel > 0 && s.total > 0) {
        const hsize_t chunk[2] = {std::min(s.total, VTKHDF_CHUNK_ROWS), nComp};
        H5Pset_chunk(dcpl, nDims, chunk);
        H5Pset_deflate(dcpl, compressLevel);
    }

    hid_t dset =
        H5Dcreate2(loc, name, fileType, fSpace, H5P_DEFAULT, dcpl, H5P_DEFAULT);

    const hsize_t start[2] = {s.offset, 0};
    const hsize_t count[2] = {s.count, nComp};
    hid_t mSpace           = H5Screate_simple(nDims, count, NULL);
    if (s.count > 0) {
        H5Sselect_hyperslab(fSpace, H5S_SELECT_SET, start, NULL, count, NULL);
    } else {
        H5Sselect_none(fSpace);
        H5Sselect_none(mSpace);
    }

    H5Dwrite(dset, memType, mSpace, fSpace, dxpl, buf);

    H5Sclose(mSpace);
    H5Dclose(dset);
    H5Pclose(dcpl);
    H5Sclose(fSpace);
}

void write_string_attr(hid_t loc, const char *name, const char *value) {
    hid_t type = H5Tcopy(H5T_C_S1);
    H5Tset_size(type, std::strlen(value));
    H5Tset_strpad(type, H5T_STR_NULLPAD);
    H5Tset_cset(type, H5T_CSET_ASCII);

    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr  = H5Acreate2(loc, name, type, space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, type, value);

    H5Aclose(attr);
    H5Sclose(space);
    H5Tclose(type);
}

void write_array_attr(hid_t loc, const char *name, hid_t fileType,
                      hid_t memType, hsize_t n, const void *value) {
    hid_t space = H5Screate_simple(1, &n, NULL);
    hid_t attr =
        H5Acreate2(loc, name, fileType, space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, memType, value);

    H5Aclose(attr);
    H5Sclose(space);
}

}  // namespace

void mesh2vtkhdfFine(const ot::Mesh *pMesh, const char *fPrefix,
                     unsigned int numFieldData, const char **fieldDataNames,
                     const double *fieldData, unsigned int numPointData,
                     const char **pointDataNames, const double **pointData,
                     unsigned int nCellData, const char **cellDNames,
                     const double **cellData, bool isDGPData,
                     unsigned int compressLevel) {
    if (!(pMesh->isActive())) return;

    MPI_Comm comm           = pMesh->getMPICommunicator();
    unsigned int rank       = pMesh->getMPIRank();
    unsigned int npes       = pMesh->getMPICommSize();

    Point dmin              = pMesh->getDomainMinPt();
    Point dmax              = pMesh->getDomainMaxPt();
    const double invRg      = 1.0 / ((double)(1u << m_uiMaxDepth));

    const std::string fname = std::string(fPrefix) + ".vtkhdf";

    hid_t fapl              = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl, comm, MPI_INFO_NULL);
    hid_t file = H5Fcreate(fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);
    if (file < 0) {
        std::cout << "rank: " << rank
                  << "[IO Error]: Could not open the vtkhdf file. "
                  << std::endl;
        return;
    }

    hid_t dxpl = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE);

    const std::vector<ot::TreeNode> &pElements = pMesh->getAllElements();

    const unsigned int nPe                     = pMesh->getNumNodesPerElement();
    const unsigned int eleOrder                = pMesh->getElementOrder();
    const unsigned int ePe                     = eleOrder * eleOrder * eleOrder;
    const unsigned int eBegin                  = pMesh->getElementLocalBegin();
    const unsigned int eEnd                    = pMesh->getElementLocalEnd();
    const hsize_t num_elements = pMesh->getNumLocalMeshElements();
    const hsize_t num_cells    = num_elements * ePe;
    const hsize_t num_vertices = num_elements * nPe;
    const hsize_t num_conn     = num_cells * NUM_CHILDREN;

    const Slice sPart          = {rank, 1, npes};
    const Slice sPts           = make_slice(comm, num_vertices);
    const Slice sCells         = make_slice(comm, num_cells);
    const Slice sConn          = make_slice(comm, num_conn);
    const Slice sOffsets       = make_slice(comm, num_cells + 1);
    const Slice sElements      = make_slice(comm, num_elements);

    hid_t root =
        H5Gcreate2(file, "VTKHDF", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    const int version[2] = {2, 0};
    write_array_attr(root, "Version", H5T_STD_I32LE, H5T_NATIVE_INT, 2,
                     version);
    write_string_attr(root, "Type", "UnstructuredGrid");

    const int64_t nPts   = num_vertices;
    const int64_t nCells = num_cells;
    const int64_t nConn  = num_conn;
    write_rows(root, "NumberOfPoints", H5T_STD_I64LE, H5T_NATIVE_INT64, sPart,
               1, &nPts, dxpl, 0);
    write_rows(root, "NumberOfCells", H5T_STD_I64LE, H5T_NATIVE_INT64, sPart, 1,
               &nCells, dxpl, 0);
    write_rows(root, "NumberOfConnectivityIds", H5T_STD_I64LE, H5T_NATIVE_INT64,
               sPart, 1, &nConn, dxpl, 0);

    double sz;
    std::vector<DENDRO_NODE_COORD_DTYPE> coord_data(num_vertices * m_uiDim);
    for (unsigned int ele = eBegin; ele < eEnd; ele++) {
        sz = 1u << (m_uiMaxDepth - pElements[ele].getLevel());
        for (unsigned int k = 0; k < (eleOrder + 1); k++)
            for (unsigned int j = 0; j < (eleOrder + 1); j++)
                for (unsigned int i = 0; i < (eleOrder + 1); i++) {
                    const hsize_t p   = ((ele - eBegin) * nPe +
                                         k * (eleOrder + 1) * (eleOrder + 1) +
                                         j * (eleOrder + 1) + i) *
                                        m_uiDim;
                    coord_data[p + 0] = VTU_OCT_X_GRID_X(pElements[ele].getX() +
                                                         i * (sz / eleOrder));
                    coord_data[p + 1] = VTU_OCT_Y_GRID_Y(pElements[ele].getY() +
                                                         j * (sz / eleOrder));
                    coord_data[p + 2] = VTU_OCT_Z_GRID_Z(pElements[ele].getZ() +
                                                         k * (sz / eleOrder));
                }
    }
    write_rows(root, "Points", H5T_IEEE_F32LE, H5T_NATIVE_FLOAT, sPts, m_uiDim,
               coord_data.data(), dxpl, compressLevel);
    std::vector<DENDRO_NODE_COORD_DTYPE>().swap(coord_data);

    std::vector<int64_t> conn(num_conn);
    std::vector<int64_t> offsets(num_cells + 1);
    const unsigned int n1 = eleOrder + 1;
    for (hsize_t ele = 0; ele < num_elements; ele++) {
        for (unsigned int ek = 0; ek < eleOrder; ek++)
            for (unsigned int ej = 0; ej < eleOrder; ej++)
                for (unsigned int ei = 0; ei < eleOrder; ei++) {
                    const int64_t b = ele * nPe + ek * n1 * n1 + ej * n1 + ei;
                    int64_t *c = &conn[(ele * ePe + ek * eleOrder * eleOrder +
                                        ej * eleOrder + ei) *
                                       NUM_CHILDREN];
                    c[0]       = b;
                    c[1]       = b + 1;
                    c[2]       = b + n1 + 1;
                    c[3]       = b + n1;
                    c[4]       = b + n1 * n1;
                    c[5]       = b + n1 * n1 + 1;
                    c[6]       = b + n1 * n1 + n1 + 1;
                    c[7]       = b + n1 * n1 + n1;
                }
    }
    for (hsize_t il = 0; il <= num_cells; il++) offsets[il] = NUM_CHILDREN * il;

    write_rows(root, "Connectivity", H5T_STD_I64LE, H5T_NATIVE_INT64, sConn, 1,
               conn.data(), dxpl, compressLevel);
    write_rows(root, "Offsets", H5T_STD_I64LE, H5T_NATIVE_INT64, sOffsets, 1,
               offsets.data(), dxpl, compressLevel);
    std::vector<int64_t>().swap(conn);
    std::vector<int64_t>().swap(offsets);

    std::vector<uint8_t> types(num_cells, VTK_HEXAHEDRON);
    write_rows(root, "Types", H5T_STD_U8LE, H5T_NATIVE_UINT8, sCells, 1,
               types.data(), dxpl, compressLevel);
    std::vector<uint8_t>().swap(types);

    hid_t cellGroup =
        H5Gcreate2(root, "CellData", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    std::vector<unsigned int> cell_tmp(num_cells, rank);
    write_rows(cellGroup, "mpi_rank", H5T_STD_U32LE, H5T_NATIVE_UINT, sCells, 1,
               cell_tmp.data(), dxpl, compressLevel);

    for (unsigned int il = eBegin; il < eEnd; ++il)
        for (unsigned int w = 0; w < ePe; w++)
            cell_tmp[(il - eBegin) * ePe + w] = pElements[il].getLevel();
    write_rows(cellGroup, "cell_level", H5T_STD_U32LE, H5T_NATIVE_UINT, sCells,
               1, cell_tmp.data(), dxpl, compressLevel);
    std::vector<unsigned int>().swap(cell_tmp);

    if (nCellData > 0 && cellData != NULL) {
        std::vector<double> cell_dtmp(num_cells);
        for (unsigned int v = 0; v < nCellData; v++) {
            for (unsigned int il = eBegin; il < eEnd; ++il)
                for (unsigned int w = 0; w < ePe; w++)
                    cell_dtmp[(il - eBegin) * ePe + w] = cellData[v][il];
            write_rows(cellGroup, cellDNames[v], H5T_IEEE_F64LE,
                       H5T_NATIVE_DOUBLE, sCells, 1, cell_dtmp.data(), dxpl,
                       compressLevel);
        }
    }
    H5Gclose(cellGroup);

    if (numPointData > 0 && pointData != NULL) {
        hid_t pointGroup = H5Gcreate2(root, "PointData", H5P_DEFAULT,
                                      H5P_DEFAULT, H5P_DEFAULT);
        std::vector<double> nodalVal(nPe);
        std::vector<double> nodalVal_all(num_vertices);
        for (unsigned int pdata = 0; pdata < numPointData; pdata++) {
            for (unsigned int il = eBegin; il < eEnd; ++il) {
                pMesh->getElementNodalValues(pointData[pdata], nodalVal.data(),
                                             il, isDGPData);
                std::copy(nodalVal.begin(), nodalVal.end(),
                          nodalVal_all.begin() + (il - eBegin) * nPe);
            }
            write_rows(pointGroup, pointDataNames[pdata], H5T_IEEE_F64LE,
                       H5T_NATIVE_DOUBLE, sPts, 1, nodalVal_all.data(), dxpl,
                       compressLevel);
        }
        H5Gclose(pointGroup);
    }

    if (numFieldData > 0 && fieldData != NULL) {
        hid_t fieldGroup   = H5Gcreate2(root, "FieldData", H5P_DEFAULT,
                                        H5P_DEFAULT, H5P_DEFAULT);
        const Slice sField = {0, (hsize_t)(rank ? 0 : 1), 1};
        for (unsigned int fdata = 0; fdata < numFieldData; fdata++)
            write_rows(fieldGroup, fieldDataNames[fdata], H5T_IEEE_F64LE,
                       H5T_NATIVE_DOUBLE, sField, 1, &fieldData[fdata], dxpl,
                       0);
        H5Gclose(fieldGroup);
    }
    H5Gclose(root);

    hid_t dendro =
        H5Gcreate2(file, "Dendro", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    const double domMin[3]      = {dmin.x(), dmin.y(), dmin.z()};
    const double domMax[3]      = {dmax.x(), dmax.y(), dmax.z()};
    const unsigned int maxDepth = m_uiMaxDepth;
    write_array_attr(dendro, "DomainMin", H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, 3,
                     domMin);
    write_array_attr(dendro, "DomainMax", H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, 3,
                     domMax);
    write_array_attr(dendro, "MaxDepth", H5T_STD_U32LE, H5T_NATIVE_UINT, 1,
                     &maxDepth);
    write_array_attr(dendro, "ElementOrder", H5T_STD_U32LE, H5T_NATIVE_UINT, 1,
                     &eleOrder);

    std::vector<unsigned int> octants(num_elements * 4);
    for (unsigned int ele = eBegin; ele < eEnd; ele++) {
        unsigned int *o = &octants[(ele - eBegin) * 4];
        o[0]            = pElements[ele].getX();
        o[1]            = pElements[ele].getY();
        o[2]            = pElements[ele].getZ();
        o[3]            = pElements[ele].getLevel();
    }
    write_rows(dendro, "Octants", H5T_STD_U32LE, H5T_NATIVE_UINT, sElements, 4,
               octants.data(), dxpl, compressLevel);
    H5Gclose(dendro);

    H5Pclose(dxpl);
    H5Fclose(file);
}

}  // namespace vtkhdf
}  // namespace io
