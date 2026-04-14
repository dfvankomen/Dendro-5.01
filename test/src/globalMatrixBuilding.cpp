#include <ctime>
#include <iostream>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "TreeNode.h"
#include "derivativeCtx.hpp"
#include "derivatives.h"
#include "hcurvedata.h"
#include "mesh.h"
#include "meshUtils.h"
#include "mpi.h"
#include "octUtils.h"

void blockAdaptiveOctree(std::vector<ot::TreeNode>& tmpNodes,
                         const Point& pt_min, const Point& pt_max,
                         const unsigned int regLev, const unsigned int maxDepth,
                         MPI_Comm comm) {
    int rank, npes;
    MPI_Comm_size(comm, &npes);
    MPI_Comm_rank(comm, &rank);

    double pt_g_min[3];
    double pt_g_max[3];

    pt_g_min[0] = X_TO_GRIDX(pt_min.x());
    pt_g_min[1] = Y_TO_GRIDY(pt_min.y());
    pt_g_min[2] = Z_TO_GRIDZ(pt_min.z());

    pt_g_max[0] = X_TO_GRIDX(pt_max.x());
    pt_g_max[1] = Y_TO_GRIDY(pt_max.y());
    pt_g_max[2] = Z_TO_GRIDZ(pt_max.z());

    assert(pt_g_min[0] >= 0 && pt_g_min[0] <= (1u << maxDepth));
    assert(pt_g_min[1] >= 0 && pt_g_min[1] <= (1u << maxDepth));
    assert(pt_g_min[2] >= 0 && pt_g_min[2] <= (1u << maxDepth));

    assert(pt_g_max[0] >= 0 && pt_g_max[0] <= (1u << maxDepth));
    assert(pt_g_max[1] >= 0 && pt_g_max[1] <= (1u << maxDepth));
    assert(pt_g_max[2] >= 0 && pt_g_max[2] <= (1u << maxDepth));

    unsigned int xRange_b, xRange_e;
    unsigned int yRange_b = pt_g_min[1], yRange_e = pt_g_max[1];
    unsigned int zRange_b = pt_g_min[2], zRange_e = pt_g_max[2];

    xRange_b =
        pt_g_min[0];  //(rank*(pt_g_max[0]-pt_g_min[0]))/npes + pt_g_min[0];
    xRange_e =
        pt_g_max[1];  //((rank+1)*(pt_g_max[0]-pt_g_min[0]))/npes + pt_g_min[0];

    unsigned int stepSz = 1u << (maxDepth - regLev);

    /* std::cout<<" x min: "<<xRange_b<<" x_max: "<<xRange_e<<std::endl;
     std::cout<<" y min: "<<yRange_b<<" y_max: "<<yRange_e<<std::endl;
     std::cout<<" z min: "<<zRange_b<<" z_max: "<<zRange_e<<std::endl;*/

    for (unsigned int x = xRange_b; x < xRange_e; x += stepSz)
        for (unsigned int y = yRange_b; y < yRange_e; y += stepSz)
            for (unsigned int z = zRange_b; z < zRange_e; z += stepSz) {
                if (x >= (1u << maxDepth)) x = x - 1;
                if (y >= (1u << maxDepth)) y = y - 1;
                if (z >= (1u << maxDepth)) z = z - 1;

                tmpNodes.push_back(
                    ot::TreeNode(x, y, z, regLev, m_uiDim, maxDepth));
            }

    return;
}

struct LookupPoint {
    double x, y, z;
    size_t lookupidx;
    size_t level;
    LookupPoint(double x, double y, double z, size_t lookupidx, size_t level)
        : x(x), y(y), z(z), lookupidx(lookupidx), level(level) {}

    bool operator==(const LookupPoint& other) const {
        return x == other.x && y == other.y && z == other.z &&
               level == other.level;
    }
};

struct PointHash {
    size_t operator()(const LookupPoint& p) const {
        return std::hash<double>()(p.x) ^ std::hash<double>()(p.y) ^
               std::hash<double>()(p.z) ^ std::hash<size_t>()(p.level);
    }
};

struct Line {
    std::vector<std::tuple<double, size_t, size_t>> x_values_idxs_level;
    double y, z;

    Line(std::vector<std::tuple<double, size_t, size_t>>&& xvit, double y,
         double z)
        : x_values_idxs_level(std::move(xvit)), y(y), z(z) {}
};

struct PairHash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ h2;
    }
};

std::vector<Line> find_lines(const std::vector<LookupPoint>& points) {
    std::unordered_set<LookupPoint, PointHash> unique_points;
    for (const auto& point : points) {
        unique_points.insert(point);
    }

    std::cout << "Number of original points: " << points.size()
              << " number of unique points: " << unique_points.size()
              << std::endl;

    // group points by y and z coordinates
    std::unordered_map<std::pair<double, double>,
                       std::vector<std::tuple<double, size_t, size_t>>,
                       PairHash>
        groups;
    for (const auto& point : unique_points) {
        groups[{point.y, point.z}].emplace_back(point.x, point.lookupidx,
                                                point.level);
    }

    // then extract the lines
    std::vector<Line> lines;
    lines.reserve(groups.size());

    for (auto& [yz, x_values_and_idxs] : groups) {
        std::sort(x_values_and_idxs.begin(), x_values_and_idxs.end());
        lines.push_back({std::move(x_values_and_idxs), yz.first, yz.second});
    }

    // Sort lines by y and z
    std::sort(lines.begin(), lines.end(), [](const Line& a, const Line& b) {
        return std::tie(a.y, a.z) < std::tie(b.y, b.z);
    });

    return lines;
}

void build_the_vector_thingy(ot::Mesh* mesh) {
    const ot::TreeNode* pNodes        = &(*(mesh->getAllElements().begin()));
    const unsigned int eleOrder       = mesh->getElementOrder();
    const unsigned int* e2n_cg        = &(*(mesh->getE2NMapping().begin()));
    const unsigned int* e2n_dg        = &(*(mesh->getE2NMapping_DG().begin()));
    const unsigned int nPe            = mesh->getNumNodesPerElement();
    const unsigned int nodeLocalBegin = mesh->getNodeLocalBegin();
    const unsigned int nodeLocalEnd   = mesh->getNodeLocalEnd();

    double dx, dy, dz;
    const Point pt_min(derivtest::DERIV_TEST_COMPD_MIN[0],
                       derivtest::DERIV_TEST_COMPD_MIN[1],
                       derivtest::DERIV_TEST_COMPD_MIN[2]);
    const Point pt_max(derivtest::DERIV_TEST_COMPD_MAX[0],
                       derivtest::DERIV_TEST_COMPD_MAX[1],
                       derivtest::DERIV_TEST_COMPD_MAX[2]);

    std::vector<LookupPoint> all_points;

    for (unsigned int elem = mesh->getElementLocalBegin();
         elem < mesh->getElementLocalEnd(); elem++) {
        for (unsigned int k = 0; k < (eleOrder + 1); k++)
            for (unsigned int j = 0; j < (eleOrder + 1); j++)
                for (unsigned int i = 0; i < (eleOrder + 1); i++) {
                    const unsigned int nodeLookUp_CG =
                        e2n_cg[elem * nPe +
                               k * (eleOrder + 1) * (eleOrder + 1) +
                               j * (eleOrder + 1) + i];
                    if (nodeLookUp_CG >= nodeLocalBegin &&
                        nodeLookUp_CG < nodeLocalEnd) {
                        const unsigned int nodeLookUp_DG =
                            e2n_dg[elem * nPe +
                                   k * (eleOrder + 1) * (eleOrder + 1) +
                                   j * (eleOrder + 1) + i];
                        unsigned int ownerID, ii_x, jj_y, kk_z;
                        mesh->dg2eijk(nodeLookUp_DG, ownerID, ii_x, jj_y, kk_z);

                        const DendroScalar len =
                            (double)(1u << (m_uiMaxDepth -
                                            pNodes[ownerID].getLevel()));
                        const DendroScalar x =
                            pNodes[ownerID].getX() + ii_x * (len / (eleOrder));
                        const DendroScalar y =
                            pNodes[ownerID].getY() + jj_y * (len / (eleOrder));
                        const DendroScalar z =
                            pNodes[ownerID].getZ() + kk_z * (len / (eleOrder));

                        // level determines

                        // these are all X points on the grid in the unzipped
                        // fashion

                        // std::cout << nodeLookUp_CG << " - " << len << ": " <<
                        // x
                        //           << ", " << y << ", " << z << std::endl;

                        all_points.push_back(
                            LookupPoint(x, y, z, nodeLookUp_CG,
                                        pNodes[ownerID].getLevel()));
                    }
                }
    }

    auto result = find_lines(all_points);

    size_t tmp  = 0;
    for (const auto& line : result) {
        if (tmp > 5) break;
        std::cout << "Line: y=" << line.y << ", z=" << line.z << std::endl;
        std::cout << "  x vals and indices: ";
        for (const auto& [x, idx, level] : line.x_values_idxs_level) {
            std::cout << "(" << x << ", " << idx << ", " << level << ") ";
        }
        std::cout << std::endl << std::endl;
        tmp++;
    }
}

int main(int argc, char* argv[]) {
    std::cout << "Now initializing program!" << std::endl;

    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, npes;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);

    int root                                      = std::min(1, npes - 1);

    derivtest::DERIV_TEST_MAXDEPTH                = 5;
    // make sure we're set to 3 for now
    derivtest::DERIV_TEST_ID_TYPE                 = 3;
    derivtest::DERIV_TEST_ENABLE_BLOCK_ADAPTIVITY = 1;

    _InitializeHcurve(3);
    m_uiMaxDepth = derivtest::DERIV_TEST_MAXDEPTH;

    std::vector<ot::TreeNode> tmpNodes;

    std::function<void(double, double, double, double*)> f_init =
        [](double x, double y, double z, double* var) {
            derivtest::initDataFuncToPhysCoords(x, y, z, var);
        };

    const unsigned int interpVars = derivtest::DERIV_TEST_NUM_VARS;

    unsigned int varIndex[interpVars];
    for (unsigned int i = 0; i < derivtest::DERIV_TEST_NUM_VARS; i++)
        varIndex[i] = i;

    if (false && derivtest::DERIV_TEST_ENABLE_BLOCK_ADAPTIVITY) {
        if (!rank)
            std::cout << YLW << "Using block adaptive mesh. AMR disabled "
                      << NRM << std::endl;
        const Point pt_min(derivtest::DERIV_TEST_COMPD_MIN[0],
                           derivtest::DERIV_TEST_COMPD_MIN[1],
                           derivtest::DERIV_TEST_COMPD_MIN[2]);
        const Point pt_max(derivtest::DERIV_TEST_COMPD_MAX[0],
                           derivtest::DERIV_TEST_COMPD_MAX[1],
                           derivtest::DERIV_TEST_COMPD_MAX[2]);
        blockAdaptiveOctree(tmpNodes, pt_min, pt_max, m_uiMaxDepth - 2,
                            m_uiMaxDepth, comm);
    } else {
        if (!rank)
            std::cout << YLW << "Using function2Octree. AMR enabled " << NRM
                      << std::endl;

        // f2olmin is like the max depth we want to refine to.
        // if we don't have two puncture initial data, then it should just be
        // the max depth minus three
        unsigned int maxDepthIn;

        // max depth in to the function2Octree must be 2 less than the max depth
        maxDepthIn = m_uiMaxDepth - 2;

        function2Octree(f_init, derivtest::DERIV_TEST_NUM_VARS, varIndex,
                        interpVars, tmpNodes, maxDepthIn,
                        derivtest::DERIV_TEST_WAVELET_TOL,
                        derivtest::DERIV_TEST_ELE_ORDER, comm);
    }

    // std::vector<ot::TreeNode> f2Octants(tmpNodes);
    ot::Mesh* mesh = ot::createMesh(
        tmpNodes.data(), tmpNodes.size(), derivtest::DERIV_TEST_ELE_ORDER, comm,
        1, ot::SM_TYPE::FDM, derivtest::DERIV_TEST_DENDRO_GRAIN_SZ,
        derivtest::DERIV_TEST_LOAD_IMB_TOL, derivtest::DERIV_TEST_SPLIT_FIX);
    mesh->setDomainBounds(Point(derivtest::DERIV_TEST_GRID_MIN_X,
                                derivtest::DERIV_TEST_GRID_MIN_Y,
                                derivtest::DERIV_TEST_GRID_MIN_Z),
                          Point(derivtest::DERIV_TEST_GRID_MAX_X,
                                derivtest::DERIV_TEST_GRID_MAX_Y,
                                derivtest::DERIV_TEST_GRID_MAX_Z));

    unsigned int lmin, lmax;
    mesh->computeMinMaxLevel(lmin, lmax);
    tmpNodes.clear();

    if (!rank) {
        std::cout << "================= Grid Info (Before init grid "
                     "converge):==============================================="
                     "========"
                  << std::endl;
        std::cout << "lmin: " << lmin << " lmax:" << lmax << std::endl;
        std::cout << "dx: "
                  << ((derivtest::DERIV_TEST_COMPD_MAX[0] -
                       derivtest::DERIV_TEST_COMPD_MIN[0]) *
                      ((1u << (m_uiMaxDepth - lmax)) /
                       ((double)derivtest::DERIV_TEST_ELE_ORDER)) /
                      ((double)(1u << (m_uiMaxDepth))))
                  << std::endl;
        std::cout << "dt: "
                  << derivtest::DERIV_TEST_CFL_FACTOR *
                         ((derivtest::DERIV_TEST_COMPD_MAX[0] -
                           derivtest::DERIV_TEST_COMPD_MIN[0]) *
                          ((1u << (m_uiMaxDepth - lmax)) /
                           ((double)derivtest::DERIV_TEST_ELE_ORDER)) /
                          ((double)(1u << (m_uiMaxDepth))))
                  << std::endl;
        std::cout << "========================================================="
                     "======================================================"
                  << std::endl;
    }

    // calculate the minimum dx

    derivtest::DERIV_TEST_RK45_TIME_STEP_SIZE =
        derivtest::DERIV_TEST_CFL_FACTOR *
        ((derivtest::DERIV_TEST_COMPD_MAX[0] -
          derivtest::DERIV_TEST_COMPD_MIN[0]) *
         ((1u << (m_uiMaxDepth - lmax)) /
          ((double)derivtest::DERIV_TEST_ELE_ORDER)) /
         ((double)(1u << (m_uiMaxDepth))));
    tmpNodes.clear();

    // enable block adaptivity disable remesh, once initialized, probably?
    derivtest::DERIV_TEST_ENABLE_BLOCK_ADAPTIVITY = 1;
    ot::Mesh* pMesh                               = mesh;

    build_the_vector_thingy(mesh);

    delete pMesh;

    return 0;
}
