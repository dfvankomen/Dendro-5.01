/**
 * @file testVTKHDF.cpp
 * @brief Gates io::vtkhdf::mesh2vtkhdfFine by reading the file back with
 * serial HDF5.
 *
 * Checks the VTKHDF partition bookkeeping (counts, offsets, connectivity
 * ranges, cell types), the cell and field data, and that every point value
 * equals a polynomial field evaluated at the node position rebuilt from
 * /Dendro/Octants, which is how a reader locates points. The field has degree
 * below the element order, so hanging-node interpolation reproduces it to
 * round-off. A deflate-compressed write must read back bit-identical, and an
 * x+z slice must pass the same checks while holding exactly the elements on
 * either plane.
 *
 * Usage: testVTKHDF [maxDepth] [waveletTol]
 */

#include <hdf5.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "TreeNode.h"
#include "dendro.h"
#include "mesh.h"
#include "meshUtils.h"
#include "oct2vtkhdf.h"
#include "octUtils.h"

namespace {

const double DOM_MIN = -8.0;
const double DOM_MAX = 12.0;

/**@brief cubic in domain coordinates, which createCGVector passes. */
double field(double x, double y, double z) {
    const double s = 1.0 / (DOM_MAX - DOM_MIN);
    const double a = (x - DOM_MIN) * s, b = (y - DOM_MIN) * s,
                 c = (z - DOM_MIN) * s;
    return 1.0 + a - 2.0 * b + 0.5 * c + a * a * b - b * c * c +
           3.0 * a * b * c;
}

int check(const char* name, bool ok) {
    std::printf("    %-34s %s\n", name, ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}

template <typename T>
std::vector<T> read_all(hid_t file, const char* path, hid_t memType) {
    hid_t dset  = H5Dopen2(file, path, H5P_DEFAULT);
    hid_t space = H5Dget_space(dset);
    hssize_t n  = H5Sget_simple_extent_npoints(space);
    std::vector<T> v(n > 0 ? n : 0);
    if (n > 0) H5Dread(dset, memType, H5S_ALL, H5S_ALL, H5P_DEFAULT, v.data());
    H5Sclose(space);
    H5Dclose(dset);
    return v;
}

int verify(const char* fname, unsigned int npes, unsigned int eOrder,
           double time) {
    int failures = 0;
    hid_t file   = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) return check("open file", false);

    const unsigned int n1  = eOrder + 1;
    const unsigned int nPe = n1 * n1 * n1;
    const unsigned int ePe = eOrder * eOrder * eOrder;

    auto nPts =
        read_all<int64_t>(file, "VTKHDF/NumberOfPoints", H5T_NATIVE_INT64);
    auto nCells =
        read_all<int64_t>(file, "VTKHDF/NumberOfCells", H5T_NATIVE_INT64);
    auto nConn  = read_all<int64_t>(file, "VTKHDF/NumberOfConnectivityIds",
                                    H5T_NATIVE_INT64);
    auto points = read_all<float>(file, "VTKHDF/Points", H5T_NATIVE_FLOAT);
    auto conn =
        read_all<int64_t>(file, "VTKHDF/Connectivity", H5T_NATIVE_INT64);
    auto offs   = read_all<int64_t>(file, "VTKHDF/Offsets", H5T_NATIVE_INT64);
    auto types  = read_all<uint8_t>(file, "VTKHDF/Types", H5T_NATIVE_UINT8);
    auto ranks  = read_all<unsigned int>(file, "VTKHDF/CellData/mpi_rank",
                                         H5T_NATIVE_UINT);
    auto levels = read_all<unsigned int>(file, "VTKHDF/CellData/cell_level",
                                         H5T_NATIVE_UINT);
    auto u = read_all<double>(file, "VTKHDF/PointData/u", H5T_NATIVE_DOUBLE);
    auto t = read_all<double>(file, "VTKHDF/FieldData/Time", H5T_NATIVE_DOUBLE);
    auto oct = read_all<unsigned int>(file, "Dendro/Octants", H5T_NATIVE_UINT);

    int64_t sumPts = 0, sumCells = 0, sumConn = 0;
    bool partsOk =
        nPts.size() == npes && nCells.size() == npes && nConn.size() == npes;
    for (unsigned int p = 0; partsOk && p < npes; p++) {
        sumPts += nPts[p];
        sumCells += nCells[p];
        sumConn += nConn[p];
        partsOk = partsOk && nPts[p] % nPe == 0 &&
                  nCells[p] == (nPts[p] / nPe) * ePe &&
                  nConn[p] == nCells[p] * NUM_CHILDREN;
    }
    failures += check("partition counts", partsOk);

    const int64_t nEle = (int64_t)oct.size() / 4;
    failures += check("dataset sizes",
                      partsOk && (int64_t)points.size() == 3 * sumPts &&
                          (int64_t)conn.size() == sumConn &&
                          (int64_t)offs.size() == sumCells + npes &&
                          (int64_t)types.size() == sumCells &&
                          (int64_t)ranks.size() == sumCells &&
                          (int64_t)levels.size() == sumCells &&
                          (int64_t)u.size() == sumPts && nEle * nPe == sumPts);
    if (failures) {
        H5Fclose(file);
        return failures;
    }

    // VTK hexahedron corner order as (x, y, z) bits
    const unsigned int corner[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0},
                                       {0, 1, 0}, {0, 0, 1}, {1, 0, 1},
                                       {1, 1, 1}, {0, 1, 1}};

    bool offsOk = true, connOk = true, cellOk = true, hexOk = true;
    int64_t cBase = 0, oBase = 0, kBase = 0, pBase = 0;
    for (unsigned int p = 0; p < npes; p++) {
        for (int64_t c = 0; c <= nCells[p]; c++)
            offsOk = offsOk && offs[oBase + c] == NUM_CHILDREN * c;
        for (int64_t k = 0; k < nConn[p]; k++)
            connOk =
                connOk && conn[kBase + k] >= 0 && conn[kBase + k] < nPts[p];
        for (int64_t c = 0; c < nCells[p]; c++)
            cellOk = cellOk && types[cBase + c] == 12 && ranks[cBase + c] == p;
        for (int64_t c = 0; connOk && c < nCells[p]; c++) {
            const int64_t* id = &conn[kBase + NUM_CHILDREN * c];
            const float* lo   = &points[3 * (pBase + id[0])];
            const float* hi   = &points[3 * (pBase + id[6])];
            for (unsigned int d = 0; d < 3; d++) hexOk = hexOk && lo[d] < hi[d];
            for (unsigned int m = 0; m < 8; m++)
                for (unsigned int d = 0; d < 3; d++)
                    hexOk = hexOk && points[3 * (pBase + id[m]) + d] ==
                                         (corner[m][d] ? hi[d] : lo[d]);
        }
        cBase += nCells[p];
        oBase += nCells[p] + 1;
        kBase += nConn[p];
        pBase += nPts[p];
    }
    failures += check("offsets per partition", offsOk);
    failures += check("connectivity in partition range", connOk);
    failures += check("hexahedra in VTK corner order", hexOk);
    failures += check("cell types and mpi_rank", cellOk);

    const double ext   = (double)(1u << m_uiMaxDepth);
    const double scale = (DOM_MAX - DOM_MIN) / ext;
    double maxErr = 0.0, maxPtErr = 0.0;
    bool levelOk = true;
    for (int64_t e = 0; e < nEle; e++) {
        const unsigned int* o = &oct[4 * e];
        const double sz       = (double)(1u << (m_uiMaxDepth - o[3]));
        for (unsigned int w = 0; w < ePe; w++)
            levelOk = levelOk && levels[e * ePe + w] == o[3];
        for (unsigned int k = 0; k < n1; k++)
            for (unsigned int j = 0; j < n1; j++)
                for (unsigned int i = 0; i < n1; i++) {
                    const int64_t r   = e * nPe + (k * n1 + j) * n1 + i;
                    const double x    = o[0] + i * sz / eOrder;
                    const double y    = o[1] + j * sz / eOrder;
                    const double z    = o[2] + k * sz / eOrder;
                    const double p[3] = {DOM_MIN + x * scale,
                                         DOM_MIN + y * scale,
                                         DOM_MIN + z * scale};
                    maxErr            = std::max(
                        maxErr, std::fabs(u[r] - field(p[0], p[1], p[2])));
                    for (unsigned int d = 0; d < 3; d++)
                        maxPtErr = std::max(
                            maxPtErr, std::fabs(points[3 * r + d] - p[d]));
                }
    }
    failures += check("cell_level matches octants", levelOk);
    failures += check("points match octants", maxPtErr < 1e-5);
    failures += check("point data at octant nodes", maxErr < 1e-10);
    if (maxErr >= 1e-10 || maxPtErr >= 1e-5)
        std::printf("      max |u - f| = %.3e, max |x - x_oct| = %.3e\n",
                    maxErr, maxPtErr);
    failures += check("field data", t.size() == 1 && t[0] == time);

    H5Fclose(file);
    return failures;
}

bool same_dataset(hid_t a, hid_t b, const char* path, hid_t memType,
                  size_t elemSz) {
    hid_t da = H5Dopen2(a, path, H5P_DEFAULT),
          db = H5Dopen2(b, path, H5P_DEFAULT);
    hid_t sa = H5Dget_space(da), sb = H5Dget_space(db);
    const hssize_t n = H5Sget_simple_extent_npoints(sa);
    bool ok          = n == H5Sget_simple_extent_npoints(sb);
    if (ok && n > 0) {
        std::vector<char> va(n * elemSz), vb(n * elemSz);
        H5Dread(da, memType, H5S_ALL, H5S_ALL, H5P_DEFAULT, va.data());
        H5Dread(db, memType, H5S_ALL, H5S_ALL, H5P_DEFAULT, vb.data());
        ok = va == vb;
    }
    H5Sclose(sa);
    H5Sclose(sb);
    H5Dclose(da);
    H5Dclose(db);
    return ok;
}

/**@brief checks an x+z slice holds exactly the elements whose lower corner
 * lies on x = c or z = c, each once. */
int check_slice(const char* fname, unsigned int c,
                unsigned long long expected) {
    hid_t file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) return check("open slice file", false);
    auto oct = read_all<unsigned int>(file, "Dendro/Octants", H5T_NATIVE_UINT);
    H5Fclose(file);

    std::vector<std::array<unsigned int, 4>> o(oct.size() / 4);
    unsigned long long onX = 0, onZ = 0, off = 0;
    for (size_t e = 0; e < o.size(); e++) {
        o[e] = {oct[4 * e], oct[4 * e + 1], oct[4 * e + 2], oct[4 * e + 3]};
        onX += o[e][0] == c;
        onZ += o[e][2] == c;
        off += o[e][0] != c && o[e][2] != c;
    }
    std::sort(o.begin(), o.end());
    const bool unique = std::adjacent_find(o.begin(), o.end()) == o.end();

    int failures      = 0;
    failures += check("slice holds only plane elements", off == 0);
    failures += check("slice holds both planes", onX > 0 && onZ > 0);
    failures += check("slice elements unique", unique);
    failures += check("slice element count", o.size() == expected);
    if (o.size() != expected)
        std::printf("      %zu elements, expected %llu\n", o.size(), expected);
    return failures;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, npes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);

    _InitializeHcurve(m_uiDim);
    m_uiMaxDepth             = (argc > 1) ? std::atoi(argv[1]) : 6;
    const double wavelet_tol = (argc > 2) ? std::atof(argv[2]) : 1e-3;

    int failures             = 0;
    for (unsigned int eOrder : {4u, 6u}) {
        const double scale = 1.0 / (double)(1u << m_uiMaxDepth);
        std::function<double(double, double, double)> refine_fn =
            [scale](double x, double y, double z) {
                const double dx = x * scale - 0.5, dy = y * scale - 0.5,
                             dz = z * scale - 0.5;
                return std::exp(-120.0 * (dx * dx + dy * dy + dz * dz));
            };

        std::vector<ot::TreeNode> tmpNodes;
        function2Octree(refine_fn, tmpNodes, m_uiMaxDepth, wavelet_tol, eOrder,
                        MPI_COMM_WORLD);
        ot::Mesh* mesh =
            ot::createMesh(tmpNodes.data(), tmpNodes.size(), eOrder,
                           MPI_COMM_WORLD, 1, ot::SM_TYPE::FDM,
                           DENDRO_DEFAULT_GRAIN_SZ, 0.1, DENDRO_DEFAULT_SF_K);
        mesh->setDomainBounds(Point(DOM_MIN, DOM_MIN, DOM_MIN),
                              Point(DOM_MAX, DOM_MAX, DOM_MAX));

        std::function<void(double, double, double, double*)> f =
            [](double x, double y, double z, double* v) {
                v[0] = field(x, y, z);
            };
        double* u = mesh->createCGVector<double>(f, 1);
        if (mesh->isActive()) {
            mesh->readFromGhostBegin(u, 1);
            mesh->readFromGhostEnd(u, 1);
        }

        const char* pNames[]   = {"u"};
        const double* pData[]  = {u};
        const char* fNames[]   = {"Time", "Cycle"};
        const double time      = 0.123456789012345;
        const double fData[]   = {time, 7.0};
        const std::string base = "testVTKHDF_eO" + std::to_string(eOrder);
        io::vtkhdf::mesh2vtkhdfFine(mesh, base.c_str(), 2, fNames, fData, 1,
                                    pNames, pData);
        io::vtkhdf::mesh2vtkhdfFine(mesh, (base + "_deflate").c_str(), 2,
                                    fNames, fData, 1, pNames, pData, 0, NULL,
                                    NULL, false, 4);

        const unsigned int c  = 1u << (m_uiMaxDepth - 1);
        unsigned int s_val[3] = {c, c, c};
        const bool s_axes[3]  = {true, false, true};
        io::vtkhdf::mesh2vtkhdf_slice(mesh, s_val, s_axes,
                                      (base + "_slice").c_str(), 2, fNames,
                                      fData, 1, pNames, pData);

        unsigned long long onPlanes = 0;
        if (mesh->isActive()) {
            const ot::TreeNode* pNodes = mesh->getAllElements().data();
            for (unsigned int e = mesh->getElementLocalBegin();
                 e < mesh->getElementLocalEnd(); e++)
                onPlanes += pNodes[e].minX() == c || pNodes[e].minZ() == c;
        }
        MPI_Allreduce(MPI_IN_PLACE, &onPlanes, 1, MPI_UNSIGNED_LONG_LONG,
                      MPI_SUM, MPI_COMM_WORLD);

        unsigned int activeNpes = mesh->isActive() ? mesh->getMPICommSize() : 0;
        MPI_Allreduce(MPI_IN_PLACE, &activeNpes, 1, MPI_UNSIGNED, MPI_MAX,
                      MPI_COMM_WORLD);

        if (!rank) {
            std::printf("  eOrder=%u ranks=%u\n", eOrder, activeNpes);
            failures +=
                verify((base + ".vtkhdf").c_str(), activeNpes, eOrder, time);

            hid_t a   = H5Fopen((base + ".vtkhdf").c_str(), H5F_ACC_RDONLY,
                                H5P_DEFAULT);
            hid_t b   = H5Fopen((base + "_deflate.vtkhdf").c_str(),
                                H5F_ACC_RDONLY, H5P_DEFAULT);
            bool same = a >= 0 && b >= 0;
            same = same && same_dataset(a, b, "VTKHDF/Points", H5T_NATIVE_FLOAT,
                                        sizeof(float));
            same = same && same_dataset(a, b, "VTKHDF/Connectivity",
                                        H5T_NATIVE_INT64, sizeof(int64_t));
            same = same && same_dataset(a, b, "VTKHDF/PointData/u",
                                        H5T_NATIVE_DOUBLE, sizeof(double));
            same = same && same_dataset(a, b, "Dendro/Octants", H5T_NATIVE_UINT,
                                        sizeof(unsigned int));
            if (a >= 0) H5Fclose(a);
            if (b >= 0) H5Fclose(b);
            failures += check("compressed write identical", same);

            std::printf("  x+z slice, %llu elements\n", onPlanes);
            failures += verify((base + "_slice.vtkhdf").c_str(), activeNpes,
                               eOrder, time);
            failures +=
                check_slice((base + "_slice.vtkhdf").c_str(), c, onPlanes);
        }

        delete[] u;
        delete mesh;
    }

    MPI_Bcast(&failures, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!rank) std::printf("%s\n", failures ? "FAILED" : "PASSED");
    MPI_Finalize();
    return failures ? 1 : 0;
}
