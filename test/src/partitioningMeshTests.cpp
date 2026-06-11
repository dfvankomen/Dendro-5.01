

#include <random>

#include "TreeNode.h"
#include "dendro.h"
#include "dendroIO.h"
#include "fdCoefficient.h"
#include "functional"
#include "genPts_par.h"
#include "mesh.h"
#include "meshTestUtils.h"
#include "mpi.h"
#include "oct2vtk.h"
#include "octUtils.h"
#include "oda.h"
#include "rawIO.h"
#include "rkTransportUtils.h"
#include "sfcSort.h"
#include "stencil.h"
#include "waveletAMR.h"
#include "waveletRefEl.h"

namespace temp_data {

std::random_device rd;
std::mt19937 gen(rd());

std::normal_distribution<double> distnorm(0.0, 1.0);
std::uniform_real_distribution<double> distuniform(-1.0, 1.0);

}  // namespace temp_data

void enforceMinDepth(ot::Mesh* mesh, const unsigned int minDepth,
                     unsigned int grain_sz, double ld_tol, unsigned int sf_k) {
    std::vector<unsigned int> refine_flags;
    const unsigned int eleLocalBegin = mesh->getElementLocalBegin();
    const unsigned int eleLocalEnd   = mesh->getElementLocalEnd();
    bool isOctChange                 = false;

    if (mesh->isActive()) {
        refine_flags.resize(mesh->getNumLocalMeshElements(), OCT_NO_CHANGE);
        const ot::TreeNode* pNodes = mesh->getAllElements().data();
        for (unsigned int ele = eleLocalBegin; ele < eleLocalEnd; ele++) {
            if (pNodes[ele].getLevel() < minDepth) {
                refine_flags[ele - eleLocalBegin] = OCT_SPLIT;
            }
        }

        // then set the refinement flags
        isOctChange = mesh->setMeshRefinementFlags(refine_flags);
    }
    // communicate refinement between cores.
    bool isOctChange_g;
    MPI_Allreduce(&isOctChange, &isOctChange_g, 1, MPI_CXX_BOOL, MPI_LOR,
                  mesh->getMPIGlobalCommunicator());

    if (isOctChange_g) {
        std::cout << "Remeshing to enforce mindepth..." << std::endl;
        ot::Mesh* newMesh = mesh->ReMesh(grain_sz, ld_tol, sf_k);

        std::swap(mesh, newMesh);
        delete newMesh;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, npes;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);

    if (argc < 5) {
        if (!rank)
            std::cout << "Usage: " << argv[0]
                      << " maxDepth wavelet_tol partition_tol eleOrder"
                      << std::endl;
        MPI_Abort(comm, 0);
    }

    PartitioningOptions partitionOption =
        static_cast<PartitioningOptions>(atoi(argv[1]));
    m_uiMaxDepth                 = atoi(argv[2]);
    double wavelet_tol           = atof(argv[3]);
    double partition_tol         = atof(argv[4]);
    unsigned int eOrder          = atoi(argv[5]);
    unsigned int DENDRO_GRAIN_SZ = atoi(argv[6]);
    unsigned int minDepth        = atoi(argv[7]);
    unsigned int SPLIT_FIX       = 256;
    double LOAD_IMB_TOL          = 0.1;

    if (!rank) {
        std::cout << YLW << "Partitioning option: " << partitionOption
                  << std::endl;
        std::cout << YLW << "maxDepth: " << m_uiMaxDepth << NRM << std::endl;
        std::cout << YLW << "wavelet_tol: " << wavelet_tol << NRM << std::endl;
        std::cout << YLW << "partition_tol: " << partition_tol << NRM
                  << std::endl;
        std::cout << YLW << "eleOrder: " << eOrder << NRM << std::endl;
        std::cout << YLW << "GRAIN_SZ: " << DENDRO_GRAIN_SZ << NRM << std::endl;
        std::cout << YLW << "minDepth: " << minDepth << NRM << std::endl;
    }

    _InitializeHcurve(m_uiDim);

    // function that we need to interpolate.
    const double d_min = -5.5;
    const double d_max = 5.5;
    double dMin[]      = {d_min, d_min, d_min};
    double dMax[]      = {d_max, d_max, d_max};

    Point pt_min(d_min, d_min, d_min);
    Point pt_max(d_max, d_max, d_max);

    std::function<double(double, double, double)> func_flat =
        [d_min, d_max](const double x, const double y, const double z) {
            return 1e8 * temp_data::distuniform(temp_data::gen);
        };

    std::function<double(double, double, double)> func_sine =
        [d_min, d_max](const double x, const double y, const double z) {
            return sin(x) * sin(y) * sin(z);
        };

    //@note that based on how the functions are defined (f(x), dxf(x), etc)
    // the
    // compuatational domain is equivalent to the grid domain.
    std::function<double(double, double, double)> func =
        [d_min, d_max](const double x, const double y, const double z) {
            double xx = (x / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;
            double yy = (y / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;
            double zz = (z / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;
            return (sin(2 * M_PI * xx) * sin(2 * M_PI * yy) *
                    sin(2 * M_PI * zz));
        };
    std::function<double(double, double, double)> dx_func =
        [d_min, d_max](const double x, const double y, const double z) {
            double xx = (x / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;
            double yy = (y / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;
            double zz = (z / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;

            // if((xx < -0.5 || xx > 0.5) || ( yy < -0.5 || yy > 0.5) || (zz
            // < -0.5 || zz > 0.5) )
            //     return 0.0;

            return (2 * M_PI * (1.0 / (1u << m_uiMaxDepth) * (d_max - d_min))) *
                   (cos(2 * M_PI * xx) * sin(2 * M_PI * yy) *
                    sin(2 * M_PI * zz));
        };

    std::function<double(double, double, double)> dy_func =
        [d_min, d_max](const double x, const double y, const double z) {
            double xx = (x / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;
            double yy = (y / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;
            double zz = (z / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;

            // if((xx < -0.5 || xx > 0.5) || ( yy < -0.5 || yy > 0.5) || (zz
            // < -0.5 || zz > 0.5) )
            //     return 0.0;

            return (2 * M_PI * (1.0 / (1u << m_uiMaxDepth) * (d_max - d_min))) *
                   (sin(2 * M_PI * xx) * cos(2 * M_PI * yy) *
                    sin(2 * M_PI * zz));
        };

    std::function<double(double, double, double)> dz_func =
        [d_min, d_max](const double x, const double y, const double z) {
            double xx = (x / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;
            double yy = (y / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;
            double zz = (z / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;

            // if((xx < -0.5 || xx > 0.5) || ( yy < -0.5 || yy > 0.5) || (zz
            // < -0.5 || zz > 0.5) )
            //     return 0.0;

            return (2 * M_PI * (1.0 / (1u << m_uiMaxDepth) * (d_max - d_min))) *
                   (sin(2 * M_PI * xx) * sin(2 * M_PI * yy) *
                    cos(2 * M_PI * zz));
        };

    std::function<double(double, double, double)> func_alt =
        [d_min, d_max](const double x, const double y, const double z) {
            double xx = (x / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;
            double yy = (y / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;
            double zz = (z / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;
            // squared radial oscillation
            const double f  = (1.6 / 2.0) * 3.1415926;
            const double a  = 10.0;

            const double x0 = xx * xx + yy * yy + zz * zz;
            const double x1 = exp(-x0);

            // Main vars

            return a * (pow(x0, 2) * x1 * pow(sin(f * sqrt(x0)), 2) + x0 * x1);
        };

    std::function<double(double, double, double)> func_alt_dx =
        [d_min, d_max](const double x, const double y, const double z) {
            double xx = (x / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;
            double yy = (y / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;
            double zz = (z / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;
            // squared radial oscillation
            const double f         = (1.6 / 2.0) * 3.1415926;
            const double a         = 10.0;

            const double x0        = xx * xx + yy * yy + zz * zz;
            const double exp_x0    = exp(-x0);
            const double sqrt_x0   = sqrt(x0);
            const double sin_term  = sin(f * sqrt_x0);
            const double sin2_term = sin(2.0 * f * sqrt_x0);

            double derivative = (2.0 * x0 - x0 * x0) * sin_term * sin_term +
                                x0 * sqrt_x0 * f * sin2_term + (1.0 - x0);

            return 2.0 * a * xx * exp_x0 * derivative;
        };

    // call function to octree

    // f2olmin is like the max depth we want to refine to.
    // if we don't have two puncture initial data, then it should just be
    // the max depth minus three
    unsigned int maxDepthIn;

    std::vector<ot::TreeNode> tmpNodes;
    function2Octree(func, tmpNodes, m_uiMaxDepth, wavelet_tol, eOrder, comm);
    // function2Octree(func_sine, tmpNodes, m_uiMaxDepth, wavelet_tol, eOrder,
    //                 comm);

    // THIS MESH WILL NOT BE AFFECTED
    // MESH DEFAULT DATA
    ot::Mesh* mesh = ot::createMesh(tmpNodes.data(), tmpNodes.size(), eOrder,
                                    comm, 1, ot::SM_TYPE::FDM, DENDRO_GRAIN_SZ,
                                    LOAD_IMB_TOL, SPLIT_FIX);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << " ============================== FINISHED MESH 1 "
                     "========================\n";
    }
    MPI_Barrier(comm);

    // create a "copy" of the mesh (which will use the exact same
    // information as inputs)
    ot::Mesh* mesh_repartitioned = ot::createMesh(
        tmpNodes.data(), tmpNodes.size(), eOrder, comm, 1, ot::SM_TYPE::FDM,
        DENDRO_GRAIN_SZ, LOAD_IMB_TOL, SPLIT_FIX);

    unsigned int lmin = 1, lmax = 5;

    // set mesh data on "post" build
    mesh->setDomainBounds(pt_min, pt_max);
    mesh->computeMinMaxLevel(lmin, lmax);
    mesh_repartitioned->setDomainBounds(pt_min, pt_max);
    mesh_repartitioned->computeMinMaxLevel(lmin, lmax);

#if 0
    enforceMinDepth(mesh, minDepth, DENDRO_GRAIN_SZ, LOAD_IMB_TOL, SPLIT_FIX);
    enforceMinDepth(mesh_repartitioned, minDepth, DENDRO_GRAIN_SZ, LOAD_IMB_TOL,
                    SPLIT_FIX);
#endif

    MPI_Barrier(MPI_COMM_WORLD);

#ifdef EXPORT_MESH
    std::string save_prefix = "test_mesh_npes" + std::to_string(npes);
    io::vtk::mesh2vtu(mesh, save_prefix.c_str(), 0, nullptr, nullptr, 0,
                      nullptr, nullptr);
#endif

    // Capture SFC partition metrics before repartitioning
    unsigned int sfcSendTotal = 0, sfcRecvTotal = 0;
    unsigned int sfcNumProcs  = 0;
    if (mesh_repartitioned->isActive()) {
        const auto& sc = mesh_repartitioned->getNodalSendCounts();
        const auto& rc = mesh_repartitioned->getNodalRecvCounts();
        for (unsigned int p = 0; p < (unsigned int)npes; p++) {
            sfcSendTotal += sc[p];
            sfcRecvTotal += rc[p];
            if (sc[p] > 0 || rc[p] > 0) sfcNumProcs++;
        }
    }

    mesh_repartitioned->setPartitioningMethod(partitionOption);
    mesh_repartitioned->repartitionMeshGlobal();

    // Compare SFC vs graph partition scatter maps
    {
        unsigned int gphSendTotal = 0, gphRecvTotal = 0;
        unsigned int gphNumProcs  = 0;
        if (mesh_repartitioned->isActive()) {
            const auto& sc = mesh_repartitioned->getNodalSendCounts();
            const auto& rc = mesh_repartitioned->getNodalRecvCounts();
            for (unsigned int p = 0; p < (unsigned int)npes; p++) {
                gphSendTotal += sc[p];
                gphRecvTotal += rc[p];
                if (sc[p] > 0 || rc[p] > 0) gphNumProcs++;
            }
        }

        // Gather global totals
        unsigned int sfcSendGlobal, gphSendGlobal;
        unsigned int sfcProcsGlobal, gphProcsGlobal;
        MPI_Allreduce(&sfcSendTotal, &sfcSendGlobal, 1, MPI_UNSIGNED,
                      MPI_SUM, comm);
        MPI_Allreduce(&gphSendTotal, &gphSendGlobal, 1, MPI_UNSIGNED,
                      MPI_SUM, comm);
        MPI_Allreduce(&sfcNumProcs, &sfcProcsGlobal, 1, MPI_UNSIGNED,
                      MPI_SUM, comm);
        MPI_Allreduce(&gphNumProcs, &gphProcsGlobal, 1, MPI_UNSIGNED,
                      MPI_SUM, comm);

        unsigned int sfcLocalEle =
            mesh->getNumLocalMeshElements();
        unsigned int gphLocalEle =
            mesh_repartitioned->getNumLocalMeshElements();
        unsigned int sfcEleMax, sfcEleMin, gphEleMax, gphEleMin;
        MPI_Allreduce(&sfcLocalEle, &sfcEleMax, 1, MPI_UNSIGNED, MPI_MAX,
                      comm);
        MPI_Allreduce(&sfcLocalEle, &sfcEleMin, 1, MPI_UNSIGNED, MPI_MIN,
                      comm);
        MPI_Allreduce(&gphLocalEle, &gphEleMax, 1, MPI_UNSIGNED, MPI_MAX,
                      comm);
        MPI_Allreduce(&gphLocalEle, &gphEleMin, 1, MPI_UNSIGNED, MPI_MIN,
                      comm);

        if (!rank) {
            std::cout << "\n--- PARTITION METRICS ---\n";
            std::cout << "Element balance (min/max):\n";
            std::cout << "  SFC:   " << sfcEleMin << " / " << sfcEleMax
                      << "  (imbalance "
                      << (sfcEleMax > 0
                              ? 100.0 * (sfcEleMax - sfcEleMin) / sfcEleMax
                              : 0)
                      << "%)\n";
            std::cout << "  Graph: " << gphEleMin << " / " << gphEleMax
                      << "  (imbalance "
                      << (gphEleMax > 0
                              ? 100.0 * (gphEleMax - gphEleMin) / gphEleMax
                              : 0)
                      << "%)\n";
            std::cout << "Ghost exchange volume (total nodes sent):\n";
            std::cout << "  SFC:   " << sfcSendGlobal << "\n";
            std::cout << "  Graph: " << gphSendGlobal;
            if (sfcSendGlobal > 0) {
                double pct = 100.0 * ((double)gphSendGlobal - (double)sfcSendGlobal) /
                             (double)sfcSendGlobal;
                std::cout << "  (" << (pct >= 0 ? "+" : "") << pct << "%)";
            }
            std::cout << "\n";
            std::cout << "Communication neighbors (total rank pairs):\n";
            std::cout << "  SFC:   " << sfcProcsGlobal << "\n";
            std::cout << "  Graph: " << gphProcsGlobal << "\n";
            std::cout << "---\n\n";
        }
    }

    /**
     *
     * TESTS
     *
     */

    // ---- TEST 1: createVector on repartitioned mesh ----
    std::vector<double> funcVal_repartitioned;
    mesh_repartitioned->createVector(funcVal_repartitioned, func);
    // Verify LOCAL node values match the analytic function
    {
        int localErrors       = 0;
        const double localTol = 1e-6;
        const auto* pNodes    = mesh_repartitioned->getAllElements().data();
        const unsigned int* e2n_cg_v =
            mesh_repartitioned->getE2NMapping().data();
        const unsigned int* e2n_dg_v =
            mesh_repartitioned->getE2NMapping_DG().data();
        const unsigned int npe_v  = mesh_repartitioned->getNumNodesPerElement();
        const unsigned int eOrd_v = mesh_repartitioned->getElementOrder();
        unsigned int nlb_v = mesh_repartitioned->getNodeLocalBegin();
        unsigned int nle_v = mesh_repartitioned->getNodeLocalEnd();
        for (unsigned int ele = mesh_repartitioned->getElementLocalBegin();
             ele < mesh_repartitioned->getElementLocalEnd(); ele++) {
            for (unsigned int n = 0; n < npe_v; n++) {
                unsigned int cg = e2n_cg_v[ele * npe_v + n];
                if (cg < nlb_v || cg >= nle_v) continue;
                unsigned int dg = e2n_dg_v[ele * npe_v + n];
                unsigned int oe = dg / npe_v;
                unsigned int rem = dg % npe_v;
                unsigned int ok = rem / ((eOrd_v+1)*(eOrd_v+1));
                unsigned int oj = (rem / (eOrd_v+1)) % (eOrd_v+1);
                unsigned int oi = rem % (eOrd_v+1);
                double len = 1u << (m_uiMaxDepth - pNodes[oe].getLevel());
                double x = pNodes[oe].getX() + oi * (len / eOrd_v);
                double y = pNodes[oe].getY() + oj * (len / eOrd_v);
                double z = pNodes[oe].getZ() + ok * (len / eOrd_v);
                double expected = func(x, y, z);
                double got      = funcVal_repartitioned[cg];
                if (std::abs(expected - got) > localTol)
                    localErrors++;
            }
        }
        int totalLocalErrors;
        MPI_Allreduce(&localErrors, &totalLocalErrors, 1, MPI_INT, MPI_SUM,
                      comm);
        if (!rank) {
            if (totalLocalErrors == 0)
                std::cout << GRN << "TEST 1 PASSED: createVector correct"
                          << NRM << std::endl;
            else
                std::cout << RED << "TEST 1 FAILED: " << totalLocalErrors
                          << " local node value errors" << NRM << std::endl;
        }
    }

    // ---- TEST 2: nodal ghost exchange ----
    mesh_repartitioned->performGhostExchange(funcVal_repartitioned);

    {
        int ghostErrors        = 0;
        int ghostNodesChecked  = 0;
        double ghostMaxErr     = 0.0;
        const double ghostTol  = 1e-6;
        const auto* pNodes     = mesh_repartitioned->getAllElements().data();
        const unsigned int* e2n_cg =
            mesh_repartitioned->getE2NMapping().data();
        const unsigned int npe  = mesh_repartitioned->getNumNodesPerElement();
        const unsigned int eOrd = mesh_repartitioned->getElementOrder();
        const unsigned int lb   = mesh_repartitioned->getElementLocalBegin();
        const unsigned int le   = mesh_repartitioned->getElementLocalEnd();

        // collect CG indices covered by the recv scatter map
        const auto& recvCount  = mesh_repartitioned->getNodalRecvCounts();
        const auto& recvOffset = mesh_repartitioned->getNodalRecvOffsets();
        const auto& recvSM     = mesh_repartitioned->getRecvNodeSM();
        std::set<unsigned int> recvCovered;
        for (unsigned int p = 0; p < (unsigned int)npes; p++)
            for (unsigned int n = recvOffset[p];
                 n < recvOffset[p] + recvCount[p]; n++)
                recvCovered.insert(recvSM[n]);

        // helper: check one ghost element's nodes using the E2N DG
        // encoding to find the correct physical coordinates (important
        // for hanging nodes where the owner element differs from the
        // ghost element)
        const unsigned int* e2n_dg_t =
            mesh_repartitioned->getE2NMapping_DG().data();
        auto checkGhostEle = [&](unsigned int ele) {
            for (unsigned int n = 0; n < npe; n++) {
                unsigned int cgIdx = e2n_cg[ele * npe + n];
                if (cgIdx >= mesh_repartitioned->getNodeLocalBegin() &&
                    cgIdx < mesh_repartitioned->getNodeLocalEnd())
                    continue;
                if (recvCovered.find(cgIdx) == recvCovered.end())
                    continue;
                ghostNodesChecked++;
                // use the DG encoding to find the owner element
                // and sub-coordinates for the correct physical position
                unsigned int dgIdx = e2n_dg_t[ele * npe + n];
                unsigned int oe    = dgIdx / npe;
                unsigned int rem   = dgIdx % npe;
                unsigned int ok = rem / ((eOrd + 1) * (eOrd + 1));
                unsigned int oj = (rem / (eOrd + 1)) % (eOrd + 1);
                unsigned int oi = rem % (eOrd + 1);
                double len =
                    1u << (m_uiMaxDepth - pNodes[oe].getLevel());
                double x = pNodes[oe].getX() + oi * (len / eOrd);
                double y = pNodes[oe].getY() + oj * (len / eOrd);
                double z = pNodes[oe].getZ() + ok * (len / eOrd);
                double expected = func(x, y, z);
                double got      = funcVal_repartitioned[cgIdx];
                double err      = std::abs(expected - got);
                if (err > ghostMaxErr) ghostMaxErr = err;
                if (err > ghostTol) {
                    ghostErrors++;
                }
            }
        };

        // check pre-ghost and post-ghost (level-1 only)
        for (unsigned int ele = mesh_repartitioned->getElementPreGhostBegin();
             ele < mesh_repartitioned->getElementPreGhostEnd(); ele++)
            checkGhostEle(ele);
        for (unsigned int ele = mesh_repartitioned->getElementPostGhostBegin();
             ele < mesh_repartitioned->getElementPostGhostEnd(); ele++)
            checkGhostEle(ele);

        int totalErrors, totalChecked;
        double gMaxErr;
        MPI_Allreduce(&ghostErrors, &totalErrors, 1, MPI_INT, MPI_SUM, comm);
        MPI_Allreduce(&ghostNodesChecked, &totalChecked, 1, MPI_INT, MPI_SUM,
                      comm);
        MPI_Allreduce(&ghostMaxErr, &gMaxErr, 1, MPI_DOUBLE, MPI_MAX, comm);
        if (!rank) {
            if (totalErrors == 0) {
                std::cout << GRN << "TEST 2 PASSED: ghost exchange correct ("
                          << totalChecked << " nodes, maxErr="
                          << gMaxErr << ")" << NRM << std::endl;
            } else {
                std::cout << RED << "TEST 2 FAILED: ghost exchange had "
                          << totalErrors << " / " << totalChecked
                          << " errors, maxErr=" << gMaxErr
                          << NRM << std::endl;
            }
        }
    }

    // ---- TEST 2b: redistributeVec preserves values SFC → graph ----
    // Populates mesh (SFC) via createVector(func), then uses the
    // Mesh::redistributeVec helper to transfer to mesh_repartitioned
    // (graph partition). Verifies that every LOCAL CG slot on the
    // graph mesh holds f(physical_pos) after the redistribute +
    // ghost exchange. Isolates redistribute bugs from ghost-exchange
    // bugs (Test 2 validates ghost-exchange alone).
    {
        std::vector<double> funcVal_sfc;
        mesh->createVector(funcVal_sfc, func);
        mesh->performGhostExchange(funcVal_sfc);

        std::vector<double> funcVal_redist;
        mesh_repartitioned->createVector(funcVal_redist, (double)0);
        mesh->redistributeVec(mesh_repartitioned, funcVal_sfc.data(),
                              funcVal_redist.data());
        mesh_repartitioned->performGhostExchange(funcVal_redist);

        int errs             = 0, checked = 0;
        double maxErr        = 0.0;
        double maxErrPos[3]  = {0, 0, 0};
        const double tol     = 1e-6;
        const auto* pNodes   = mesh_repartitioned->getAllElements().data();
        const unsigned int* e2n_cg_v =
            mesh_repartitioned->getE2NMapping().data();
        const unsigned int* e2n_dg_v =
            mesh_repartitioned->getE2NMapping_DG().data();
        const unsigned int npe_v  = mesh_repartitioned->getNumNodesPerElement();
        const unsigned int eOrd_v = mesh_repartitioned->getElementOrder();
        const unsigned int nlb_v  = mesh_repartitioned->getNodeLocalBegin();
        const unsigned int nle_v  = mesh_repartitioned->getNodeLocalEnd();
        for (unsigned int ele = mesh_repartitioned->getElementLocalBegin();
             ele < mesh_repartitioned->getElementLocalEnd(); ele++) {
            for (unsigned int n = 0; n < npe_v; n++) {
                unsigned int cg = e2n_cg_v[ele * npe_v + n];
                if (cg < nlb_v || cg >= nle_v) continue;
                unsigned int dg  = e2n_dg_v[ele * npe_v + n];
                unsigned int oe  = dg / npe_v;
                unsigned int rem = dg % npe_v;
                unsigned int ok  = rem / ((eOrd_v + 1) * (eOrd_v + 1));
                unsigned int oj  = (rem / (eOrd_v + 1)) % (eOrd_v + 1);
                unsigned int oi  = rem % (eOrd_v + 1);
                double len       = 1u << (m_uiMaxDepth - pNodes[oe].getLevel());
                double x         = pNodes[oe].getX() + oi * (len / eOrd_v);
                double y         = pNodes[oe].getY() + oj * (len / eOrd_v);
                double z         = pNodes[oe].getZ() + ok * (len / eOrd_v);
                double expected  = func(x, y, z);
                double got       = funcVal_redist[cg];
                double err       = std::abs(expected - got);
                checked++;
                if (err > maxErr) {
                    maxErr       = err;
                    maxErrPos[0] = x;
                    maxErrPos[1] = y;
                    maxErrPos[2] = z;
                }
                if (err > tol) {
                    errs++;
                }
            }
        }
        int totalErrs     = 0, totalChecked = 0;
        double globalMax  = 0.0;
        MPI_Allreduce(&errs, &totalErrs, 1, MPI_INT, MPI_SUM, comm);
        MPI_Allreduce(&checked, &totalChecked, 1, MPI_INT, MPI_SUM, comm);
        MPI_Allreduce(&maxErr, &globalMax, 1, MPI_DOUBLE, MPI_MAX, comm);
        if (!rank) {
            if (totalErrs == 0) {
                std::cout << GRN << "TEST 2b PASSED: redistributeVec "
                                    "correct ("
                          << totalChecked << " nodes, maxErr="
                          << globalMax << ")" << NRM
                          << std::endl;
            } else {
                std::cout << RED << "TEST 2b FAILED: redistributeVec had "
                          << totalErrs << " / " << totalChecked
                          << " errors, maxErr=" << globalMax
                          << NRM << std::endl;
                // Print location of max-error node on rank 0 only if
                // rank 0 saw it
                if (maxErr == globalMax && maxErr > tol)
                    std::cout << "    max-err @ (" << maxErrPos[0] << ", "
                              << maxErrPos[1] << ", " << maxErrPos[2]
                              << ")  L=" << "?" << "\n";
            }
        }
    }

    // ---- TEST 3: scatter map send/recv symmetry ----
    {
        const auto& sendCount = mesh_repartitioned->getNodalSendCounts();
        const auto& recvCount = mesh_repartitioned->getNodalRecvCounts();
        unsigned int totalSend = 0, totalRecv = 0;
        for (unsigned int p = 0; p < (unsigned int)npes; p++) {
            totalSend += sendCount[p];
            totalRecv += recvCount[p];
        }
        unsigned int globalSend, globalRecv;
        MPI_Allreduce(&totalSend, &globalSend, 1, MPI_UNSIGNED, MPI_SUM,
                      comm);
        MPI_Allreduce(&totalRecv, &globalRecv, 1, MPI_UNSIGNED, MPI_SUM,
                      comm);
        if (!rank) {
            if (globalSend == globalRecv) {
                std::cout << GRN << "TEST 3 PASSED: scatter map symmetric "
                             "(total send = recv = "
                          << globalSend << ")" << NRM << std::endl;
            } else {
                std::cout << RED << "TEST 3 FAILED: send=" << globalSend
                          << " recv=" << globalRecv << NRM << std::endl;
            }
        }
    }

    // ---- TEST 4: block decomposition ----
    {
        const auto& blkList = mesh_repartitioned->getLocalBlockList();
        std::set<unsigned int> elementsInBlocks;
        for (const auto& blk : blkList) {
            for (auto elemId : blk) {
                elementsInBlocks.insert(elemId);
            }
        }

        int blkErrors = 0;
        for (unsigned int e = mesh_repartitioned->getElementLocalBegin();
             e < mesh_repartitioned->getElementLocalEnd(); e++) {
            if (elementsInBlocks.find(e) == elementsInBlocks.end())
                blkErrors++;
        }

        int totalBlkErrors;
        MPI_Allreduce(&blkErrors, &totalBlkErrors, 1, MPI_INT, MPI_SUM,
                      comm);
        if (!rank) {
            if (totalBlkErrors == 0) {
                std::cout << GRN << "TEST 4 PASSED: all local elements in "
                             "blocks (" << blkList.size() << " blocks)"
                          << NRM << std::endl;
            } else {
                std::cout << RED << "TEST 4 FAILED: " << totalBlkErrors
                          << " local elements missing from blocks" << NRM
                          << std::endl;
            }
        }
    }

    // ---- TEST 5: element ghost exchange ----
    {
        unsigned int* eleVec =
            mesh_repartitioned->createElementVector<unsigned int>(
                LOOK_UP_TABLE_DEFAULT, 1);

        // fill local elements with a known value (global element index)
        for (unsigned int e = mesh_repartitioned->getElementLocalBegin();
             e < mesh_repartitioned->getElementLocalEnd(); e++) {
            eleVec[e] = e;  // local index as marker
        }

        mesh_repartitioned->readFromGhostBeginElementVec(eleVec, 1);
        mesh_repartitioned->readFromGhostEndElementVec(eleVec, 1);

        // verify: every ghost element in the Round1 index should now
        // have a value != LOOK_UP_TABLE_DEFAULT
        const auto& ghostR1Idx =
            mesh_repartitioned->getLevel1GhostElementIndices();
        int eleGhostErrors = 0;
        for (unsigned int k = 0; k < ghostR1Idx.size(); k++) {
            if (eleVec[ghostR1Idx[k]] == LOOK_UP_TABLE_DEFAULT) {
                eleGhostErrors++;
            }
        }

        int totalEleGhostErrors;
        MPI_Allreduce(&eleGhostErrors, &totalEleGhostErrors, 1, MPI_INT,
                      MPI_SUM, comm);
        if (!rank) {
            if (totalEleGhostErrors == 0) {
                std::cout << GRN << "TEST 5 PASSED: element ghost exchange ("
                          << ghostR1Idx.size() << " round-1 ghosts)" << NRM
                          << std::endl;
            } else {
                std::cout << RED << "TEST 5 FAILED: " << totalEleGhostErrors
                          << " ghost elements not received" << NRM
                          << std::endl;
            }
        }
        delete[] eleVec;
    }

    // ---- SCATTER MAP EFFICIENCY ANALYSIS ----
    // Compare: (a) total nodes in recv scatter map vs
    //          (b) nodes that are actually on ghost element boundaries
    //              (the minimal set needed for block padding)
    {
        const unsigned int* e2e =
            mesh_repartitioned->getE2EMapping().data();
        const unsigned int* e2n_cg_a =
            mesh_repartitioned->getE2NMapping().data();
        const unsigned int npe_a =
            mesh_repartitioned->getNumNodesPerElement();
        const unsigned int eOrd_a =
            mesh_repartitioned->getElementOrder();
        const unsigned int ndir =
            mesh_repartitioned->getNumDirections();
        const unsigned int lb_a =
            mesh_repartitioned->getElementLocalBegin();
        const unsigned int le_a =
            mesh_repartitioned->getElementLocalEnd();
        const unsigned int nlb_a =
            mesh_repartitioned->getNodeLocalBegin();
        const unsigned int nle_a =
            mesh_repartitioned->getNodeLocalEnd();

        // Count unique ghost CG nodes ACTUALLY needed for unzip:
        // use blkUnzipElementIDs to get the exact ghost element set
        // that each block reads from, then collect their CG nodes.
        std::set<unsigned int> unzipGhostNodes;
        std::set<unsigned int> unzipGhostElements;
        {
            std::vector<unsigned int> blkEleIDs;
            const auto& blkList = mesh_repartitioned->getLocalBlockList();
            for (unsigned int b = 0; b < blkList.size(); b++) {
                mesh_repartitioned->blkUnzipElementIDs(b, blkEleIDs);
                for (unsigned int eid : blkEleIDs) {
                    if (eid >= lb_a && eid < le_a) continue;
                    unzipGhostElements.insert(eid);
                    for (unsigned int n = 0; n < npe_a; n++) {
                        unsigned int cg = e2n_cg_a[eid * npe_a + n];
                        if (cg >= nlb_a && cg < nle_a) continue;
                        unzipGhostNodes.insert(cg);
                    }
                }
            }
        }

        // Also count all nodes in the recv scatter map
        const auto& recvSM_a = mesh_repartitioned->getRecvNodeSM();
        std::set<unsigned int> scatterMapNodes(recvSM_a.begin(),
                                               recvSM_a.end());

        // Compare with the SFC mesh
        std::set<unsigned int> sfcScatterNodes;
        {
            const auto& sfcRecvSM = mesh->getRecvNodeSM();
            sfcScatterNodes.insert(sfcRecvSM.begin(), sfcRecvSM.end());
        }

        unsigned int unzipNodes = unzipGhostNodes.size();
        unsigned int unzipEles = unzipGhostElements.size();
        unsigned int smNodes   = scatterMapNodes.size();
        unsigned int sfcNodes  = sfcScatterNodes.size();
        unsigned int gUnzipNodes, gUnzipEles, gSmNodes, gSfcNodes;
        MPI_Allreduce(&unzipNodes, &gUnzipNodes, 1, MPI_UNSIGNED, MPI_SUM,
                      comm);
        MPI_Allreduce(&unzipEles, &gUnzipEles, 1, MPI_UNSIGNED, MPI_SUM,
                      comm);
        MPI_Allreduce(&smNodes, &gSmNodes, 1, MPI_UNSIGNED, MPI_SUM, comm);
        MPI_Allreduce(&sfcNodes, &gSfcNodes, 1, MPI_UNSIGNED, MPI_SUM, comm);

        if (!rank) {
            std::cout << "\n--- SCATTER MAP EFFICIENCY ---\n";
            std::cout << "Ghost elements needed by unzip:  " << gUnzipEles
                      << "\n";
            std::cout << "Ghost CG nodes needed by unzip:  " << gUnzipNodes
                      << "\n";
            std::cout << "SFC scatter map sends:           " << gSfcNodes
                      << "\n";
            std::cout << "Graph scatter map sends:         " << gSmNodes;
            if (gUnzipNodes > 0) {
                double pct =
                    100.0 * ((double)gSmNodes - (double)gUnzipNodes) /
                    (double)gUnzipNodes;
                std::cout << "  (+" << pct << "% vs unzip need)";
            }
            std::cout << "\n";
            if (gSfcNodes > 0) {
                double sfcPct =
                    100.0 * ((double)gSfcNodes - (double)gUnzipNodes) /
                    (double)gUnzipNodes;
                std::cout << "SFC overhead vs unzip need:      +" << sfcPct
                          << "%\n";
            }
            std::cout << "---\n\n";
        }
    }

    // ---- TEST 6: setMeshRefinementFlags + ReMesh ----
    ot::Mesh* remeshed = nullptr;
    {
        std::vector<unsigned int> refine_flags;
        if (mesh_repartitioned->isActive()) {
            const unsigned int numLocal =
                mesh_repartitioned->getNumLocalMeshElements();
            refine_flags.resize(numLocal, OCT_NO_CHANGE);

            // Refine the first few local elements to force actual remeshing
            const ot::TreeNode* pNodesR =
                mesh_repartitioned->getAllElements().data();
            unsigned int lbR = mesh_repartitioned->getElementLocalBegin();
            for (unsigned int e = 0; e < numLocal && e < 4; e++) {
                if (pNodesR[lbR + e].getLevel() < m_uiMaxDepth)
                    refine_flags[e] = OCT_SPLIT;
            }
        }

        bool isOctChange =
            mesh_repartitioned->setMeshRefinementFlags(refine_flags);
        bool isOctChange_g;
        MPI_Allreduce(&isOctChange, &isOctChange_g, 1, MPI_CXX_BOOL, MPI_LOR,
                      comm);

        if (!rank) {
            std::cout << (isOctChange_g ? YLW : GRN)
                      << "TEST 6a: setMeshRefinementFlags "
                      << (isOctChange_g ? "(mesh changed - unexpected for "
                                          "no-change flags)"
                                        : "PASSED (no change)")
                      << NRM << std::endl;
        }

        // Remesh: use ReMeshRepartitioned for graph-partitioned meshes,
        // standard ReMesh for SFC-partitioned (OriginalPartition).
        if (partitionOption == PartitioningOptions::OriginalPartition ||
            partitionOption == PartitioningOptions::NoPartition) {
            remeshed = mesh_repartitioned->ReMesh(
                DENDRO_GRAIN_SZ, LOAD_IMB_TOL, SPLIT_FIX);
        } else {
            remeshed = mesh_repartitioned->ReMeshRepartitioned(
                DENDRO_GRAIN_SZ, LOAD_IMB_TOL, SPLIT_FIX);
        }

        if (remeshed != nullptr) {
            unsigned int remeshLocalCount =
                remeshed->getNumLocalMeshElements();
            unsigned int totalRemeshEle;
            MPI_Allreduce(&remeshLocalCount, &totalRemeshEle, 1, MPI_UNSIGNED,
                          MPI_SUM, comm);

            if (!rank) {
                std::cout << GRN << "TEST 6b PASSED: ReMesh succeeded ("
                          << totalRemeshEle << " total elements)" << NRM
                          << std::endl;
            }
        } else {
            if (!rank) {
                std::cout << YLW
                          << "TEST 6b: ReMesh returned nullptr (mesh "
                             "unchanged, this is OK for no-change)"
                          << NRM << std::endl;
            }
        }
    }

    // ---- TEST 7: unzip correctness on repartitioned mesh ----
    // ---- TEST 8: unzip correctness on remeshed mesh ----
    // Verify that unzip produces correct values in the ghost padding
    // region by comparing against the analytic function at each
    // point's physical coordinates.
    auto testUnzip = [&](ot::Mesh* testMesh, const char* label) {
        if (!testMesh || !testMesh->isActive()) return;

        const unsigned int eOrd_u = testMesh->getElementOrder();
        const unsigned int npe_u  = testMesh->getNumNodesPerElement();

        // Create vector, ghost exchange, unzip
        std::vector<double> zVec;
        testMesh->createVector(zVec, func);
        testMesh->performGhostExchange(zVec);

        double* uzVec = testMesh->createUnZippedVector<double>(0.0);
        testMesh->unzip(zVec.data(), uzVec);

        const auto& blkList = testMesh->getLocalBlockList();
        const auto* pNodes  = testMesh->getAllElements().data();

        int totalErrors = 0, totalChecked = 0;
        int largeErrors = 0;
        double maxErr = 0;
        const double uzTol = 1e-3;

        for (unsigned int b = 0; b < blkList.size(); b++) {
            const ot::Block& blk   = blkList[b];
            const unsigned int pw   = blk.get1DPadWidth();
            const unsigned int lx   = blk.getAllocationSzX();
            const unsigned int ly   = blk.getAllocationSzY();
            const unsigned int lz   = blk.getAllocationSzZ();
            const DendroIntL offset = blk.getOffset();

            const ot::TreeNode bn = blk.getBlockNode();

            // Physical spacing at the block's regular grid level
            const double hx =
                (1u << (m_uiMaxDepth - blk.getRegularGridLev())) /
                (double)eOrd_u;
            const double xmin = bn.getX() - pw * hx;
            const double ymin = bn.getY() - pw * hx;
            const double zmin = bn.getZ() - pw * hx;

            // Check every point in the unzipped block
            for (unsigned int k = 0; k < lz; k++) {
                for (unsigned int j = 0; j < ly; j++) {
                    for (unsigned int i = 0; i < lx; i++) {
                        double x = xmin + i * hx;
                        double y = ymin + j * hx;
                        double z = zmin + k * hx;

                        // Skip points outside the domain
                        if (x < 0 || y < 0 || z < 0) continue;
                        unsigned int domMax = 1u << m_uiMaxDepth;
                        if (x > domMax || y > domMax || z > domMax) continue;

                        double expected = func(x, y, z);
                        double got = uzVec[offset + k * lx * ly + j * lx + i];

                        double err = std::abs(expected - got);
                        if (err > maxErr) maxErr = err;
                        if (err > uzTol) {
                            totalErrors++;
                            if (err > 0.5) {
                                largeErrors++;
                                if (largeErrors <= 12) {
                                    std::cout << "U7L r=" << rank
                                              << " blk=" << b
                                              << " (i,j,k)=(" << i << ","
                                              << j << "," << k << ")"
                                              << " xyz=(" << x << "," << y
                                              << "," << z
                                              << ") exp=" << expected
                                              << " got=" << got << "\n";
                                }
                            }
                        }
                        totalChecked++;
                    }
                }
            }
        }

        int globalErrors, globalChecked;
        MPI_Allreduce(&totalErrors, &globalErrors, 1, MPI_INT, MPI_SUM, comm);
        MPI_Allreduce(&totalChecked, &globalChecked, 1, MPI_INT, MPI_SUM,
                      comm);
        int globalLargeErrors;
        double globalMaxErr;
        MPI_Allreduce(&largeErrors, &globalLargeErrors, 1, MPI_INT, MPI_SUM,
                      comm);
        MPI_Allreduce(&maxErr, &globalMaxErr, 1, MPI_DOUBLE, MPI_MAX, comm);
        if (!rank) {
            if (globalErrors == 0) {
                std::cout << GRN << label << " PASSED: unzip correct ("
                          << globalChecked << " points, maxErr="
                          << globalMaxErr << ")" << NRM << std::endl;
            } else {
                std::cout << (globalLargeErrors > 0 ? RED : YLW)
                          << label << ": " << globalErrors << " / "
                          << globalChecked << " errors (tol=" << uzTol
                          << "), " << globalLargeErrors
                          << " large (>0.5), maxErr=" << globalMaxErr
                          << NRM << std::endl;
            }
        }

        delete[] uzVec;
    };

    testUnzip(mesh, "TEST 7a (SFC baseline)");
    testUnzip(mesh_repartitioned, "TEST 7b (repartitioned)");
    if (remeshed) {
        testUnzip(remeshed, "TEST 8  (after remesh)");
    }

    // ---- TEST 11: UNZIP_INDEPENDENT/DEPENDENT classification correctness ----
    // Ground-truth check that every block flagged UNZIP_INDEPENDENT genuinely
    // reads no ghost data (safe for compute/comm overlap). auditBlockType-
    // Independence poisons all ghost cgs with NaN, unzips, and counts any
    // independent block whose output catches a NaN. Must be 0 on SFC and graph.
    auto testBlockType = [&](ot::Mesh* testMesh, const char* label) {
        if (!testMesh || !testMesh->isActive()) return;
        DendroIntL bad = testMesh->auditBlockTypeIndependence(label);
        if (!rank) {
            if (bad == 0)
                std::cout << GRN << label
                          << " PASSED: no UNZIP_INDEPENDENT block reads ghost"
                          << NRM << std::endl;
            else
                std::cout << RED << label << " FAILED: " << bad
                          << " UNZIP_INDEPENDENT blocks read ghost data" << NRM
                          << std::endl;
        }
    };
    testBlockType(mesh, "TEST 11a (SFC blocktype)     ");
    testBlockType(mesh_repartitioned, "TEST 11b (graph blocktype)   ");
    if (remeshed) testBlockType(remeshed, "TEST 11c (remesh blocktype)  ");

    // ---- TEST 8b: zip roundtrip (createVector → unzip → zip) ----
    // vecA = createVector(func). ghost exchange + unzip → uz. zip(uz) →
    // vecB. Compare vecA[cg] vs vecB[cg] at every local cg.
    // Any difference is a zip bug. Unzip is verified exact in 7b so any
    // error here is attributable to zip.
    auto testZipRoundtrip = [&](ot::Mesh* testMesh, const char* label) {
        if (!testMesh || !testMesh->isActive()) return;
        std::vector<double> vecA, vecB;
        testMesh->createVector(vecA, func);
        testMesh->createVector(vecB, (double)0);
        testMesh->performGhostExchange(vecA);
        double* uzVec = testMesh->createUnZippedVector<double>(0.0);
        testMesh->unzip(vecA.data(), uzVec);
        testMesh->zip(uzVec, vecB.data());
        delete[] uzVec;

        const unsigned int nLB = testMesh->getNodeLocalBegin();
        const unsigned int nLE = testMesh->getNodeLocalEnd();
        int errors    = 0;
        int checked   = 0;
        double maxE   = 0;
        const double tol = 1e-10;
        for (unsigned int cg = nLB; cg < nLE; cg++) {
            double d = std::abs(vecA[cg] - vecB[cg]);
            if (d > maxE) maxE = d;
            if (d > tol) errors++;
            checked++;
        }
        int gErr, gChk;
        double gMax;
        MPI_Allreduce(&errors, &gErr, 1, MPI_INT, MPI_SUM, comm);
        MPI_Allreduce(&checked, &gChk, 1, MPI_INT, MPI_SUM, comm);
        MPI_Allreduce(&maxE, &gMax, 1, MPI_DOUBLE, MPI_MAX, comm);
        if (!rank) {
            if (gErr == 0)
                std::cout << GRN << label
                          << " PASSED: zip roundtrip (" << gChk
                          << " cgs, maxErr=" << gMax << ")" << NRM
                          << std::endl;
            else
                std::cout << RED << label << ": " << gErr << " / " << gChk
                          << " zip mismatches (tol=" << tol
                          << "), maxErr=" << gMax << NRM << std::endl;
        }
    };
    testZipRoundtrip(mesh, "TEST 8ba (SFC zip roundtrip)");
    testZipRoundtrip(mesh_repartitioned, "TEST 8bb (repart zip roundtrip)");
    if (remeshed) {
        testZipRoundtrip(remeshed, "TEST 8bc (remesh zip roundtrip)");
    }

    // TEST 8b-cycle: N rounds of ghost_exchange + unzip + zip on the
    // same vec. If identity, vec stays bit-exact. Any drift means one
    // of the ops is leaking state somewhere — a real bug.
    auto testCycleStability = [&](ot::Mesh* testMesh, const char* label) {
        if (!testMesh || !testMesh->isActive()) return;
        std::vector<double> vec;
        testMesh->createVector(vec, func);
        std::vector<double> ref = vec;  // snapshot
        double* uzVec = testMesh->createUnZippedVector<double>(0.0);
        const int N = 20;
        double maxDrift = 0.0;
        for (int iter = 0; iter < N; iter++) {
            testMesh->performGhostExchange(vec);
            std::fill_n(uzVec, testMesh->getDegOfFreedomUnZip(), 0.0);
            testMesh->unzip(vec.data(), uzVec);
            testMesh->zip(uzVec, vec.data());
        }
        delete[] uzVec;
        const unsigned int nLB = testMesh->getNodeLocalBegin();
        const unsigned int nLE = testMesh->getNodeLocalEnd();
        for (unsigned int cg = nLB; cg < nLE; cg++) {
            double d = std::abs(vec[cg] - ref[cg]);
            if (d > maxDrift) maxDrift = d;
        }
        double globalDrift;
        MPI_Allreduce(&maxDrift, &globalDrift, 1, MPI_DOUBLE, MPI_MAX, comm);
        if (!rank) {
            if (globalDrift == 0)
                std::cout << GRN << label
                          << " PASSED: cycle stable (" << N
                          << " iters, maxDrift=0)" << NRM << std::endl;
            else
                std::cout << (globalDrift > 1e-10 ? RED : YLW) << label
                          << ": cycle drift maxDrift=" << globalDrift
                          << " after " << N << " iters" << NRM
                          << std::endl;
        }
    };
    testCycleStability(mesh, "TEST 8bf (SFC cycle stability)");
    testCycleStability(mesh_repartitioned, "TEST 8bg (repart cycle stability)");

    // TEST 8b-async: same as 8bf/bg but uses ASYNC readFromGhostBegin/End
    // (via Mesh API directly, mimicking Ctx::unzip's code path in NLSM).
    auto testAsyncCycleStability = [&](ot::Mesh* testMesh,
                                       const char* label) {
        if (!testMesh || !testMesh->isActive()) return;
        std::vector<double> vec;
        testMesh->createVector(vec, func);
        std::vector<double> ref = vec;
        double* uzVec = testMesh->createUnZippedVector<double>(0.0);
        const int N = 20;
        for (int iter = 0; iter < N; iter++) {
            testMesh->readFromGhostBegin(vec.data(), 1);
            testMesh->readFromGhostEnd(vec.data(), 1);
            std::fill_n(uzVec, testMesh->getDegOfFreedomUnZip(), 0.0);
            testMesh->unzip(vec.data(), uzVec);
            testMesh->zip(uzVec, vec.data());
        }
        delete[] uzVec;
        const unsigned int nLB = testMesh->getNodeLocalBegin();
        const unsigned int nLE = testMesh->getNodeLocalEnd();
        double maxDrift = 0;
        for (unsigned int cg = nLB; cg < nLE; cg++) {
            double d = std::abs(vec[cg] - ref[cg]);
            if (d > maxDrift) maxDrift = d;
        }
        double globalDrift;
        MPI_Allreduce(&maxDrift, &globalDrift, 1, MPI_DOUBLE, MPI_MAX, comm);
        if (!rank) {
            if (globalDrift == 0)
                std::cout << GRN << label
                          << " PASSED: async cycle stable (" << N
                          << " iters, maxDrift=0)" << NRM << std::endl;
            else
                std::cout << (globalDrift > 1e-10 ? RED : YLW) << label
                          << ": async cycle drift maxDrift="
                          << globalDrift << " after " << N << " iters"
                          << NRM << std::endl;
        }
    };
    testAsyncCycleStability(mesh, "TEST 8bh (SFC async cycle)");
    testAsyncCycleStability(mesh_repartitioned,
                            "TEST 8bi (repart async cycle)");

    // TEST 8bj: direct ghost-region cross-check.
    // Populate both SFC (mesh) and graph (mesh_repartitioned) with the
    // SAME func, ghost exchange both.
    //
    // For each element common to both meshes, at each sub-index, ask:
    // what value does SFC have at vSfc[E2N_CG_sfc[e, n]] vs what does
    // graph have at vGraph[E2N_CG_graph[e, n]]? Element + sub uniquely
    // determines phys_pos, so this is a direct value comparison at
    // matched physical points. No cg2dg involved — robust to orphan-fix
    // redirects.
    // Helper: compute phys_pos that (elem, sub)'s E2N_DG canonical
    // routes to. SAME input (elem TN + sub) should give SAME output
    // on both meshes if canonical routing is consistent.
    auto physPosOfCanonical = [&](const ot::Mesh* m, unsigned int e,
                                  unsigned int n,
                                  double& X, double& Y, double& Z) {
        const unsigned int npe = m->getNumNodesPerElement();
        const unsigned int eOrd = m->getElementOrder();
        const auto* pN = m->getAllElements().data();
        const auto& e2ndg = m->getE2NMapping_DG();
        unsigned int dg = e2ndg[e * npe + n];
        unsigned int oe = dg / npe;
        unsigned int rem = dg % npe;
        unsigned int ok = rem / ((eOrd + 1) * (eOrd + 1));
        unsigned int oj = (rem / (eOrd + 1)) % (eOrd + 1);
        unsigned int oi = rem % (eOrd + 1);
        double len =
            (double)(1u << (m_uiMaxDepth - pN[oe].getLevel()));
        X = pN[oe].getX() + oi * (len / (double)eOrd);
        Y = pN[oe].getY() + oj * (len / (double)eOrd);
        Z = pN[oe].getZ() + ok * (len / (double)eOrd);
    };

    auto cmp_elem_sub = [&](bool doGhostExchange, const char* tag) {
        std::vector<double> vSfc, vGraph;
        mesh->createVector(vSfc, func);
        mesh_repartitioned->createVector(vGraph, func);
        if (doGhostExchange) {
            mesh->performGhostExchange(vSfc);
            mesh_repartitioned->performGhostExchange(vGraph);
        }

        // For shared (TN, sub), compare canonical-owner phys_pos
        // between SFC and graph. Any mismatch is a real routing bug.
        int canonMismatch = 0;
        double maxCanonDiff = 0;
        if (mesh->isActive() && mesh_repartitioned->isActive()) {
            // Build TN → element-idx map for quick lookup on SFC side.
            struct TNHash {
                size_t operator()(const ot::TreeNode& t) const noexcept {
                    uint64_t h = (uint64_t)t.getX();
                    h = (h << 21) ^ (uint64_t)t.getY();
                    h = (h << 21) ^ (uint64_t)t.getZ();
                    h = (h << 6) ^ (uint64_t)t.getLevel();
                    h ^= h >> 33;
                    h *= 0xff51afd7ed558ccdULL;
                    h ^= h >> 33;
                    return (size_t)h;
                }
            };
            std::unordered_map<ot::TreeNode, unsigned int, TNHash>
                sfcTnToElem;
            const auto* sfcPN = mesh->getAllElements().data();
            for (unsigned int e = mesh->getElementPreGhostBegin();
                 e < mesh->getElementPostGhostEnd(); e++)
                sfcTnToElem.emplace(sfcPN[e], e);

            const unsigned int npe =
                mesh_repartitioned->getNumNodesPerElement();
            const auto* gPN =
                mesh_repartitioned->getAllElements().data();
            for (unsigned int e =
                     mesh_repartitioned->getElementPreGhostBegin();
                 e < mesh_repartitioned->getElementPostGhostEnd(); e++) {
                auto it = sfcTnToElem.find(gPN[e]);
                if (it == sfcTnToElem.end()) continue;
                unsigned int eSfc = it->second;
                for (unsigned int n = 0; n < npe; n++) {
                    double sx, sy, sz, grx, gry, grz;
                    physPosOfCanonical(mesh, eSfc, n, sx, sy, sz);
                    physPosOfCanonical(mesh_repartitioned, e, n,
                                       grx, gry, grz);
                    double d = std::max({std::abs(sx - grx),
                                         std::abs(sy - gry),
                                         std::abs(sz - grz)});
                    if (d > maxCanonDiff) maxCanonDiff = d;
                    if (d > 0) canonMismatch++;
                }
            }
        }
        int gCM;
        double gMCD;
        MPI_Allreduce(&canonMismatch, &gCM, 1, MPI_INT, MPI_SUM, comm);
        MPI_Allreduce(&maxCanonDiff, &gMCD, 1, MPI_DOUBLE, MPI_MAX, comm);
        if (!rank && strcmp(tag, "pre-ghost-exchange") == 0) {
            std::cout << (gCM > 0 ? RED : GRN)
                      << "TEST 8bj [canonical phys_pos] "
                      << gCM << " (e, sub) pairs route to different "
                      << "phys_pos on graph vs SFC, maxDiff="
                      << gMCD << NRM << std::endl;
        }

        // For every element on SFC, index (TreeNode, sub) → value.
        // Use global TreeNode ID so we can match to graph's view.
        struct TNSubKey {
            uint64_t x, y, z;
            uint32_t lvl;
            uint32_t sub;
            bool operator<(const TNSubKey& o) const {
                return std::tie(x, y, z, lvl, sub) <
                       std::tie(o.x, o.y, o.z, o.lvl, o.sub);
            }
        };
        // Build sfcElementValues from LOCAL elements only — these are
        // the cgs that SFC actually populates and uses for evolution.
        std::map<TNSubKey, double> sfcElementValues;
        if (mesh->isActive()) {
            const unsigned int npe =
                mesh->getNumNodesPerElement();
            const auto* pN = mesh->getAllElements().data();
            const auto& e2n = mesh->getE2NMapping();
            const unsigned int eBegin = mesh->getElementLocalBegin();
            const unsigned int eEnd = mesh->getElementLocalEnd();
            for (unsigned int e = eBegin; e < eEnd; e++) {
                TNSubKey k;
                k.x = pN[e].getX();
                k.y = pN[e].getY();
                k.z = pN[e].getZ();
                k.lvl = pN[e].getLevel();
                for (unsigned int n = 0; n < npe; n++) {
                    k.sub = n;
                    double val = vSfc[e2n[e * npe + n]];
                    sfcElementValues.emplace(k, val);
                }
            }
        }

        // For each graph element's sub-index, look up same (TN, sub) in
        // SFC map and compare.
        int checkedLocal = 0, checkedGhost = 0;
        int bitMismatchLocal = 0, bitMismatchGhost = 0;
        double maxDiffLocal = 0, maxDiffGhost = 0;
        if (mesh_repartitioned->isActive()) {
            const unsigned int npe =
                mesh_repartitioned->getNumNodesPerElement();
            const auto* pN =
                mesh_repartitioned->getAllElements().data();
            const auto& e2n = mesh_repartitioned->getE2NMapping();
            const unsigned int nLB =
                mesh_repartitioned->getNodeLocalBegin();
            const unsigned int nLE =
                mesh_repartitioned->getNodeLocalEnd();
            const unsigned int eBegin =
                mesh_repartitioned->getElementLocalBegin();
            const unsigned int eEnd =
                mesh_repartitioned->getElementLocalEnd();
            int printed = 0;
            for (unsigned int e = eBegin; e < eEnd; e++) {
                TNSubKey k;
                k.x = pN[e].getX();
                k.y = pN[e].getY();
                k.z = pN[e].getZ();
                k.lvl = pN[e].getLevel();
                for (unsigned int n = 0; n < npe; n++) {
                    k.sub = n;
                    auto it = sfcElementValues.find(k);
                    if (it == sfcElementValues.end()) continue;
                    unsigned int cg = e2n[e * npe + n];
                    double d = std::abs(vGraph[cg] - it->second);
                    bool isLocal = (cg >= nLB && cg < nLE);
                    if (isLocal) {
                        checkedLocal++;
                        if (d > maxDiffLocal) maxDiffLocal = d;
                        if (vGraph[cg] != it->second)
                            bitMismatchLocal++;
                    } else {
                        checkedGhost++;
                        if (d > maxDiffGhost) maxDiffGhost = d;
                        if (vGraph[cg] != it->second)
                            bitMismatchGhost++;
                    }
                    // print first few mismatches with debug info
                    if (doGhostExchange && d > 1e-10 && printed < 10
                        && rank == 0 && !isLocal) {
                        std::cout
                            << "  [diag " << tag << "] e=" << e
                            << " TN=(" << k.x << "," << k.y << ","
                            << k.z << ",L" << k.lvl << ") sub=" << n
                            << " cg=" << cg
                            << " (local=" << isLocal << ")"
                            << " vGraph=" << vGraph[cg]
                            << " vSfc=" << it->second
                            << " diff=" << (vGraph[cg] - it->second)
                            << std::endl;
                        printed++;
                    }
                }
            }
        }
        int gCL, gCG, gBL, gBG;
        double gML, gMG;
        MPI_Allreduce(&checkedLocal, &gCL, 1, MPI_INT, MPI_SUM, comm);
        MPI_Allreduce(&checkedGhost, &gCG, 1, MPI_INT, MPI_SUM, comm);
        MPI_Allreduce(&bitMismatchLocal, &gBL, 1, MPI_INT, MPI_SUM,
                      comm);
        MPI_Allreduce(&bitMismatchGhost, &gBG, 1, MPI_INT, MPI_SUM,
                      comm);
        MPI_Allreduce(&maxDiffLocal, &gML, 1, MPI_DOUBLE, MPI_MAX, comm);
        MPI_Allreduce(&maxDiffGhost, &gMG, 1, MPI_DOUBLE, MPI_MAX, comm);
        if (!rank) {
            std::cout << (gBL + gBG > 0 ? YLW : GRN)
                      << "TEST 8bj [" << tag << "] "
                      << "local: " << gCL << " checked, "
                      << gBL << " bit-mismatch, maxDiff=" << gML
                      << "; ghost: " << gCG << " checked, "
                      << gBG << " bit-mismatch, maxDiff=" << gMG
                      << NRM << std::endl;
        }
    };
    cmp_elem_sub(false, "pre-ghost-exchange");
    cmp_elem_sub(true, "post-ghost-exchange");

    // TEST 8bk: mirror-symmetry preservation under ghost exchange.
    // Populate graph with a function that is MIRROR-SYMMETRIC about
    // the domain center (i.e., f(domCenter + d) = f(domCenter - d)).
    // Since the function is symmetric and the mesh geometry is
    // symmetric, EACH CG at (domCenter + d) must equal the CG at
    // (domCenter - d) bit-exactly — on any partition. If graph breaks
    // this symmetry, we have a real bug.
    {
        // Center of domain in octree coords.
        const double dC = (double)(1u << m_uiMaxDepth) * 0.5;

        // NLSM-like Gaussian shell at radius R (mirror-symmetric about
        // the domain center). Uses similar parameters to NLSM's IC
        // (amp=1.3, R=1, delta=0.5) scaled to the octree coordinate
        // system where domain is [0, 1<<maxDepth].
        const double DOMAIN = (double)(1u << m_uiMaxDepth);
        std::function<double(double, double, double)> symFunc =
            [dC, DOMAIN](double x, double y, double z) {
                // Map octree coords to [-10, 10] domain (NLSM's domain).
                double fx = (x - dC) / DOMAIN * 20.0;
                double fy = (y - dC) / DOMAIN * 20.0;
                double fz = (z - dC) / DOMAIN * 20.0;
                double r  = std::sqrt(fx * fx + fy * fy + fz * fz);
                const double amp = 1.3, R = 1.0, delta = 0.5;
                return amp * std::exp(-(r - R) * (r - R) /
                                      (delta * delta));
            };

        auto testMirror = [&](ot::Mesh* m, const char* label,
                              int cycles, bool useAsync, bool rhsOp) {
            if (!m || !m->isActive()) return;
            std::vector<double> v;
            m->createVector(v, symFunc);
            // Do cycles of ghost exchange + unzip + rhsOp + zip, mimicking RK4.
            double* uz = m->createUnZippedVector<double>(0.0);
            const auto& blkList = m->getLocalBlockList();
            for (int it = 0; it < cycles; it++) {
                if (useAsync) {
                    m->readFromGhostBegin(v.data(), 1);
                    m->readFromGhostEnd(v.data(), 1);
                } else {
                    m->performGhostExchange(v);
                }
                std::fill_n(uz, m->getDegOfFreedomUnZip(), 0.0);
                m->unzip(v.data(), uz);
                if (rhsOp) {
                    // Apply a simple 2nd-derivative-like FD stencil
                    // (a symmetric 3-point central) scaled by small dt
                    // to simulate RK4 rhs. The stencil is symmetric, so
                    // symmetric input must produce symmetric output.
                    const double dt = 1e-3;
                    for (size_t b = 0; b < blkList.size(); b++) {
                        const auto& blk = blkList[b];
                        const unsigned int lx = blk.getAllocationSzX();
                        const unsigned int ly = blk.getAllocationSzY();
                        const unsigned int lz = blk.getAllocationSzZ();
                        const unsigned int pw = blk.get1DPadWidth();
                        const size_t offset = blk.getOffset();
                        // new_val = old + dt * (u[i+1] + u[i-1] - 2*u[i])
                        // in x, y, z. Symmetric.
                        std::vector<double> tmp(lx * ly * lz);
                        for (unsigned int k = pw; k < lz - pw; k++)
                            for (unsigned int j = pw; j < ly - pw; j++)
                                for (unsigned int i = pw;
                                     i < lx - pw; i++) {
                                    size_t c = offset +
                                               k * (ly * lx) + j * lx + i;
                                    double u0 = uz[c];
                                    double dx = uz[c + 1] + uz[c - 1]
                                                - 2 * u0;
                                    double dy = uz[c + lx]
                                                + uz[c - lx] - 2 * u0;
                                    double dz = uz[c + lx * ly]
                                                + uz[c - lx * ly]
                                                - 2 * u0;
                                    tmp[k * (ly * lx) + j * lx + i] =
                                        u0 + dt * (dx + dy + dz);
                                }
                        for (unsigned int k = pw; k < lz - pw; k++)
                            for (unsigned int j = pw; j < ly - pw; j++)
                                for (unsigned int i = pw;
                                     i < lx - pw; i++) {
                                    size_t c = offset +
                                               k * (ly * lx) + j * lx + i;
                                    uz[c] = tmp[k * (ly * lx) + j * lx
                                                + i];
                                }
                    }
                }
                m->zip(uz, v.data());
            }
            delete[] uz;
            if (useAsync) {
                m->readFromGhostBegin(v.data(), 1);
                m->readFromGhostEnd(v.data(), 1);
            } else {
                m->performGhostExchange(v);
            }

            // For each local CG, find its mirror about domain center
            // and compare. Build phys_pos → value map for cross-rank
            // lookup via MPI allgather.
            const unsigned int npe  = m->getNumNodesPerElement();
            const unsigned int eOrd = m->getElementOrder();
            const auto* pN          = m->getAllElements().data();
            const auto& e2n         = m->getE2NMapping();

            struct Entry {
                uint64_t x, y, z;
                double val;
            };
            std::vector<Entry> mine;
            std::set<unsigned int> seen;
            for (unsigned int e = m->getElementLocalBegin();
                 e < m->getElementLocalEnd(); e++) {
                for (unsigned int n = 0; n < npe; n++) {
                    unsigned int cg = e2n[e * npe + n];
                    if (!seen.insert(cg).second) continue;
                    uint64_t len = (uint64_t)1
                                   << (m_uiMaxDepth - pN[e].getLevel());
                    uint64_t x =
                        (uint64_t)pN[e].getX() * eOrd +
                        (uint64_t)(n % (eOrd + 1)) * len;
                    uint64_t y =
                        (uint64_t)pN[e].getY() * eOrd +
                        (uint64_t)((n / (eOrd + 1)) % (eOrd + 1)) * len;
                    uint64_t z =
                        (uint64_t)pN[e].getZ() * eOrd +
                        (uint64_t)(n / ((eOrd + 1) * (eOrd + 1))) *
                            len;
                    mine.push_back({x, y, z, v[cg]});
                }
            }

            // Allgatherv all (x, y, z, val) to every rank.
            int myCount = (int)mine.size();
            std::vector<int> counts(npes), offs(npes, 0);
            MPI_Allgather(&myCount, 1, MPI_INT, counts.data(), 1,
                          MPI_INT, comm);
            int total = 0;
            for (int p = 0; p < npes; p++) {
                offs[p] = total;
                total += counts[p];
            }
            std::vector<uint64_t> aX(total), aY(total), aZ(total);
            std::vector<double> aV(total);
            {
                std::vector<uint64_t> mX(myCount), mY(myCount),
                    mZ(myCount);
                std::vector<double> mV(myCount);
                for (int i = 0; i < myCount; i++) {
                    mX[i] = mine[i].x;
                    mY[i] = mine[i].y;
                    mZ[i] = mine[i].z;
                    mV[i] = mine[i].val;
                }
                MPI_Allgatherv(mX.data(), myCount, MPI_UINT64_T,
                               aX.data(), counts.data(), offs.data(),
                               MPI_UINT64_T, comm);
                MPI_Allgatherv(mY.data(), myCount, MPI_UINT64_T,
                               aY.data(), counts.data(), offs.data(),
                               MPI_UINT64_T, comm);
                MPI_Allgatherv(mZ.data(), myCount, MPI_UINT64_T,
                               aZ.data(), counts.data(), offs.data(),
                               MPI_UINT64_T, comm);
                MPI_Allgatherv(mV.data(), myCount, MPI_DOUBLE,
                               aV.data(), counts.data(), offs.data(),
                               MPI_DOUBLE, comm);
            }
            std::map<std::tuple<uint64_t, uint64_t, uint64_t>, double>
                posToVal;
            for (int i = 0; i < total; i++)
                posToVal.emplace(std::make_tuple(aX[i], aY[i], aZ[i]),
                                 aV[i]);

            // 2 * dC * eOrd in scaled units — this is the scaled-int
            // of domain center times 2 for mirror flip:
            // mirror(x_scaled) = 2 * (dC * eOrd) - x_scaled
            const uint64_t center2 =
                (uint64_t)((uint64_t)(1u << m_uiMaxDepth) * eOrd);
            int mismatches = 0;
            double maxDiff = 0;
            for (auto& kv : posToVal) {
                auto [x, y, z] = kv.first;
                if (x > center2 || y > center2 || z > center2) continue;
                uint64_t mx = center2 - x, my = center2 - y,
                         mz = center2 - z;
                auto it = posToVal.find(std::make_tuple(mx, my, mz));
                if (it == posToVal.end()) continue;
                double d = std::abs(kv.second - it->second);
                if (d > maxDiff) maxDiff = d;
                if (d > 0) mismatches++;
            }
            // Only rank 0 reports (values are allgather'd identically)
            if (!rank) {
                std::cout << (maxDiff > 1e-12 ? RED : GRN) << label
                          << " mirror symmetry: " << mismatches
                          << " cgs asymmetric (maxDiff=" << maxDiff
                          << ")" << NRM << std::endl;
            }
        };
        // TEST 8bm: check block decomposition symmetry on graph.
        // For each block on graph, look up the mirror block. They
        // should have the same size and level. If not, asymmetric
        // block structure could lead to asymmetric unzip/rhs.
        if (mesh_repartitioned->isActive()) {
            // Gather all local block info across ranks.
            struct BlkInfo {
                uint32_t x, y, z, lvl;   // block node
                uint32_t szX, szY, szZ;  // alloc sizes
                uint32_t bflag;
                uint32_t regLev;
            };
            std::vector<BlkInfo> mine;
            for (const auto& b :
                 mesh_repartitioned->getLocalBlockList()) {
                auto bn = b.getBlockNode();
                BlkInfo bi;
                bi.x = bn.getX();
                bi.y = bn.getY();
                bi.z = bn.getZ();
                bi.lvl = bn.getLevel();
                bi.szX = b.getAllocationSzX();
                bi.szY = b.getAllocationSzY();
                bi.szZ = b.getAllocationSzZ();
                bi.bflag = b.getBlkNodeFlag();
                bi.regLev = b.getRegularGridLev();
                mine.push_back(bi);
            }
            int myCount = (int)mine.size();
            std::vector<int> counts(npes), offs(npes, 0);
            MPI_Allgather(&myCount, 1, MPI_INT, counts.data(), 1,
                          MPI_INT, comm);
            int total = 0;
            for (int p = 0; p < npes; p++) {
                offs[p] = total;
                total += counts[p];
            }
            std::vector<int> iCounts(npes), iOffs(npes, 0);
            const int ints_per_blk = sizeof(BlkInfo) / sizeof(int);
            for (int p = 0; p < npes; p++) {
                iCounts[p] = counts[p] * ints_per_blk;
                if (p > 0) iOffs[p] = iOffs[p-1] + iCounts[p-1];
            }
            std::vector<int> all(total * ints_per_blk);
            MPI_Allgatherv(mine.data(),
                           myCount * ints_per_blk, MPI_INT,
                           all.data(), iCounts.data(), iOffs.data(),
                           MPI_INT, comm);
            // Look up blocks by block-node position
            std::map<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>,
                     BlkInfo> byPos;
            const BlkInfo* allBlk = (const BlkInfo*)all.data();
            for (int i = 0; i < total; i++) {
                byPos[std::make_tuple(allBlk[i].x, allBlk[i].y,
                                      allBlk[i].z, allBlk[i].lvl)] =
                    allBlk[i];
            }
            // Mirror flip
            const uint32_t DOMAIN = 1u << m_uiMaxDepth;
            int asymBlks = 0;
            for (auto& kv : byPos) {
                auto& [kx, ky, kz, klvl] = kv.first;
                auto& bi = kv.second;
                // Block size in octree coords at level lvl:
                uint32_t blkSz = 1u << (m_uiMaxDepth - klvl);
                // Mirror: block from (x,y,z) size blkSz to
                // (DOMAIN-x-blkSz, DOMAIN-y-blkSz, DOMAIN-z-blkSz)
                uint32_t mx = DOMAIN - kx - blkSz;
                uint32_t my = DOMAIN - ky - blkSz;
                uint32_t mz = DOMAIN - kz - blkSz;
                auto it =
                    byPos.find(std::make_tuple(mx, my, mz, klvl));
                if (it == byPos.end()) {
                    asymBlks++;
                    continue;
                }
                const BlkInfo& m = it->second;
                if (bi.szX != m.szX || bi.szY != m.szY ||
                    bi.szZ != m.szZ || bi.regLev != m.regLev)
                    asymBlks++;
            }
            if (!rank)
                std::cout << (asymBlks > 0 ? YLW : GRN)
                          << "TEST 8bm (graph block mirror symmetry): "
                          << asymBlks << " / " << total
                          << " blocks have no matching mirror"
                          << NRM << std::endl;
        }

        // TEST: do blocks at matching positions have matching bflag?
        // For graph vs SFC, same domain region should be covered by
        // blocks (possibly different decompositions) but the domain
        // boundary flag should only depend on where the block node is.
        if (mesh->isActive() && mesh_repartitioned->isActive()) {
            std::map<std::tuple<uint32_t, uint32_t, uint32_t,
                                uint32_t>, unsigned int> sfcBlkFlag;
            for (const auto& b : mesh->getLocalBlockList()) {
                auto bn = b.getBlockNode();
                sfcBlkFlag[std::make_tuple(bn.getX(), bn.getY(),
                                           bn.getZ(), bn.getLevel())] =
                    b.getBlkNodeFlag();
            }
            int flagMismatch = 0;
            for (const auto& b :
                 mesh_repartitioned->getLocalBlockList()) {
                auto bn = b.getBlockNode();
                auto k = std::make_tuple(bn.getX(), bn.getY(),
                                         bn.getZ(), bn.getLevel());
                auto it = sfcBlkFlag.find(k);
                if (it == sfcBlkFlag.end()) continue;
                if (it->second != b.getBlkNodeFlag()) flagMismatch++;
            }
            int gFM;
            MPI_Allreduce(&flagMismatch, &gFM, 1, MPI_INT, MPI_SUM,
                          comm);
            if (!rank)
                std::cout << (gFM > 0 ? RED : GRN)
                          << "TEST 8bk-block-bflag: "
                          << gFM << " matching blocks have different "
                          << "bflag between SFC and graph" << NRM
                          << std::endl;
        }
        // TEST 8bl: symmetric SFC vec → redistribute to graph → check
        // graph still symmetric. This is the path NLSM uses during
        // init_grid: build SFC mesh + IC, then buildGraphTwin +
        // redistributeDVec to graph.
        if (mesh->isActive() && mesh_repartitioned->isActive()) {
            std::vector<double> vSfc, vGraph;
            mesh->createVector(vSfc, symFunc);
            mesh_repartitioned->createVector(vGraph, (double)0);
            mesh->performGhostExchange(vSfc);
            mesh->redistributeVec(mesh_repartitioned, vSfc.data(),
                                  vGraph.data());
            mesh_repartitioned->performGhostExchange(vGraph);

            // Check graph's mirror symmetry after redistribute.
            const unsigned int npe =
                mesh_repartitioned->getNumNodesPerElement();
            const unsigned int eOrd =
                mesh_repartitioned->getElementOrder();
            const auto* pN =
                mesh_repartitioned->getAllElements().data();
            const auto& e2n = mesh_repartitioned->getE2NMapping();
            struct Entry {
                uint64_t x, y, z;
                double val;
            };
            std::vector<Entry> mine;
            std::set<unsigned int> seen;
            for (unsigned int e =
                     mesh_repartitioned->getElementLocalBegin();
                 e < mesh_repartitioned->getElementLocalEnd(); e++) {
                for (unsigned int n = 0; n < npe; n++) {
                    unsigned int cg = e2n[e * npe + n];
                    if (!seen.insert(cg).second) continue;
                    uint64_t len =
                        (uint64_t)1 << (m_uiMaxDepth -
                                        pN[e].getLevel());
                    uint64_t x =
                        (uint64_t)pN[e].getX() * eOrd +
                        (uint64_t)(n % (eOrd + 1)) * len;
                    uint64_t y =
                        (uint64_t)pN[e].getY() * eOrd +
                        (uint64_t)((n / (eOrd + 1)) %
                                   (eOrd + 1)) * len;
                    uint64_t z =
                        (uint64_t)pN[e].getZ() * eOrd +
                        (uint64_t)(n / ((eOrd + 1) *
                                        (eOrd + 1))) * len;
                    mine.push_back({x, y, z, vGraph[cg]});
                }
            }
            int myCount = (int)mine.size();
            std::vector<int> counts(npes), offs(npes, 0);
            MPI_Allgather(&myCount, 1, MPI_INT, counts.data(), 1,
                          MPI_INT, comm);
            int total = 0;
            for (int p = 0; p < npes; p++) {
                offs[p] = total;
                total += counts[p];
            }
            std::vector<uint64_t> aX(total), aY(total), aZ(total);
            std::vector<double> aV(total);
            {
                std::vector<uint64_t> mX(myCount), mY(myCount),
                    mZ(myCount);
                std::vector<double> mV(myCount);
                for (int i = 0; i < myCount; i++) {
                    mX[i] = mine[i].x;
                    mY[i] = mine[i].y;
                    mZ[i] = mine[i].z;
                    mV[i] = mine[i].val;
                }
                MPI_Allgatherv(mX.data(), myCount, MPI_UINT64_T,
                               aX.data(), counts.data(), offs.data(),
                               MPI_UINT64_T, comm);
                MPI_Allgatherv(mY.data(), myCount, MPI_UINT64_T,
                               aY.data(), counts.data(), offs.data(),
                               MPI_UINT64_T, comm);
                MPI_Allgatherv(mZ.data(), myCount, MPI_UINT64_T,
                               aZ.data(), counts.data(), offs.data(),
                               MPI_UINT64_T, comm);
                MPI_Allgatherv(mV.data(), myCount, MPI_DOUBLE,
                               aV.data(), counts.data(), offs.data(),
                               MPI_DOUBLE, comm);
            }
            std::map<std::tuple<uint64_t, uint64_t, uint64_t>, double>
                posToVal;
            for (int i = 0; i < total; i++)
                posToVal.emplace(std::make_tuple(aX[i], aY[i], aZ[i]),
                                 aV[i]);
            const uint64_t center2 =
                (uint64_t)((uint64_t)(1u << m_uiMaxDepth) * eOrd);
            double maxDiff = 0;
            int mismatches = 0;
            for (auto& kv : posToVal) {
                auto [x, y, z] = kv.first;
                if (x > center2 || y > center2 || z > center2)
                    continue;
                uint64_t mx = center2 - x, my = center2 - y,
                         mz = center2 - z;
                auto it = posToVal.find(std::make_tuple(mx, my, mz));
                if (it == posToVal.end()) continue;
                double d = std::abs(kv.second - it->second);
                if (d > maxDiff) maxDiff = d;
                if (d > 0) mismatches++;
            }
            if (!rank)
                std::cout << (maxDiff > 1e-12 ? RED : GRN)
                          << "TEST 8bl (SFC→graph redistribute "
                          << "symmetry): " << mismatches
                          << " cgs asymmetric, maxDiff=" << maxDiff
                          << NRM << std::endl;
        }

        // TEST 8bn: check if UNZIPPED BLOCK VALUES are mirror-symmetric.
        // This tests: does the physical position (x,y,z) in block B's
        // unzipped buffer hold the same value as position (-x,-y,-z)
        // in some other block's unzipped buffer?
        // Unzip is what NLSM's RHS reads. If unzip gives asymmetric
        // values at mirror cells, RHS produces asymmetric output.
        auto testUnzipMirror = [&](ot::Mesh* m, const char* label) {
            if (!m || !m->isActive()) return;
            std::vector<double> v;
            m->createVector(v, symFunc);
            m->performGhostExchange(v);
            double* uz = m->createUnZippedVector<double>(0.0);
            m->unzip(v.data(), uz);

            // Gather all unzipped interior values + their phys_pos.
            const auto& blkList = m->getLocalBlockList();
            const double DOMAIN = (double)(1u << m_uiMaxDepth);
            struct Entry {
                double x, y, z, val;
            };
            std::vector<Entry> mine;
            for (size_t b = 0; b < blkList.size(); b++) {
                const auto& blk = blkList[b];
                const unsigned int lx = blk.getAllocationSzX();
                const unsigned int ly = blk.getAllocationSzY();
                const unsigned int lz = blk.getAllocationSzZ();
                const unsigned int pw = blk.get1DPadWidth();
                const size_t offset = blk.getOffset();
                const auto bn = blk.getBlockNode();
                const unsigned int regLev = blk.getRegularGridLev();
                const double h =
                    (double)(1u << (m_uiMaxDepth - regLev)) /
                    (double)(m->getElementOrder());
                // ALL cells including padding — FD stencil reads padding
                for (unsigned int k = 0; k < lz; k++)
                    for (unsigned int j = 0; j < ly; j++)
                        for (unsigned int i = 0; i < lx; i++) {
                            double x =
                                (double)bn.getX() +
                                (double)((int)i - (int)pw) * h;
                            double y =
                                (double)bn.getY() +
                                (double)((int)j - (int)pw) * h;
                            double z =
                                (double)bn.getZ() +
                                (double)((int)k - (int)pw) * h;
                            size_t c = offset +
                                       k * (ly * lx) + j * lx + i;
                            mine.push_back({x, y, z, uz[c]});
                        }
            }
            delete[] uz;
            int myCount = (int)mine.size();
            std::vector<int> counts(npes), offs(npes, 0);
            MPI_Allgather(&myCount, 1, MPI_INT, counts.data(), 1,
                          MPI_INT, comm);
            int total = 0;
            for (int p = 0; p < npes; p++) {
                offs[p] = total;
                total += counts[p];
            }
            std::vector<double> aX(total), aY(total), aZ(total),
                aV(total);
            {
                std::vector<double> mX(myCount), mY(myCount),
                    mZ(myCount), mV(myCount);
                for (int i = 0; i < myCount; i++) {
                    mX[i] = mine[i].x;
                    mY[i] = mine[i].y;
                    mZ[i] = mine[i].z;
                    mV[i] = mine[i].val;
                }
                MPI_Allgatherv(mX.data(), myCount, MPI_DOUBLE,
                               aX.data(), counts.data(), offs.data(),
                               MPI_DOUBLE, comm);
                MPI_Allgatherv(mY.data(), myCount, MPI_DOUBLE,
                               aY.data(), counts.data(), offs.data(),
                               MPI_DOUBLE, comm);
                MPI_Allgatherv(mZ.data(), myCount, MPI_DOUBLE,
                               aZ.data(), counts.data(), offs.data(),
                               MPI_DOUBLE, comm);
                MPI_Allgatherv(mV.data(), myCount, MPI_DOUBLE,
                               aV.data(), counts.data(), offs.data(),
                               MPI_DOUBLE, comm);
            }
            // Scaled-int key to avoid FP comparison issues.
            const double SCALE = 1e6;
            std::map<std::tuple<int64_t, int64_t, int64_t>,
                     std::vector<double>> posToVals;
            for (int i = 0; i < total; i++) {
                auto k = std::make_tuple(
                    (int64_t)std::round(aX[i] * SCALE),
                    (int64_t)std::round(aY[i] * SCALE),
                    (int64_t)std::round(aZ[i] * SCALE));
                posToVals[k].push_back(aV[i]);
            }
            int intraBlockMismatch = 0;
            double intraMaxDiff = 0;
            // Check same-position multiple unzip writes (should match)
            for (auto& kv : posToVals) {
                for (size_t i = 1; i < kv.second.size(); i++) {
                    double d = std::abs(kv.second[i] - kv.second[0]);
                    if (d > intraMaxDiff) intraMaxDiff = d;
                    if (d > 0) intraBlockMismatch++;
                }
            }
            // Mirror symmetry
            const int64_t center2 = (int64_t)(DOMAIN * SCALE);
            int mirrorMismatch = 0;
            double mirrorMaxDiff = 0;
            for (auto& kv : posToVals) {
                auto [x, y, z] = kv.first;
                if (x > center2 / 2 && y > center2 / 2 &&
                    z > center2 / 2)
                    continue;  // only half the space
                int64_t mx = center2 - x, my = center2 - y,
                        mz = center2 - z;
                auto it = posToVals.find(std::make_tuple(mx, my, mz));
                if (it == posToVals.end()) continue;
                double d = std::abs(kv.second[0] - it->second[0]);
                if (d > mirrorMaxDiff) mirrorMaxDiff = d;
                if (d > 0) mirrorMismatch++;
            }
            if (!rank)
                std::cout
                    << (mirrorMaxDiff > 1e-12 || intraMaxDiff > 1e-12
                            ? RED
                            : GRN)
                    << label << " unzip: intra-mismatch="
                    << intraBlockMismatch
                    << " (max=" << intraMaxDiff << "), "
                    << "mirror-mismatch=" << mirrorMismatch
                    << " (max=" << mirrorMaxDiff << ")" << NRM
                    << std::endl;
        };
        testUnzipMirror(mesh, "TEST 8bn-SFC");
        testUnzipMirror(mesh_repartitioned, "TEST 8bn-graph");

        // TEST 8bp: directly compare unzipped values (interior + padding)
        // between SFC and graph at MATCHING phys_pos.
        if (mesh->isActive() && mesh_repartitioned->isActive()) {
            auto gather_unzip = [&](ot::Mesh* m,
                                    std::map<std::tuple<int64_t, int64_t,
                                                        int64_t>,
                                             std::vector<double>>& out) {
                std::vector<double> v;
                m->createVector(v, symFunc);
                m->performGhostExchange(v);
                double* uz = m->createUnZippedVector<double>(0.0);
                m->unzip(v.data(), uz);
                const auto& blkList = m->getLocalBlockList();
                const double SCALE = 1e6;
                struct E { double x, y, z, val; };
                std::vector<E> mine;
                for (size_t b = 0; b < blkList.size(); b++) {
                    const auto& blk = blkList[b];
                    const unsigned int lx = blk.getAllocationSzX();
                    const unsigned int ly = blk.getAllocationSzY();
                    const unsigned int lz = blk.getAllocationSzZ();
                    const unsigned int pw = blk.get1DPadWidth();
                    const size_t offset = blk.getOffset();
                    const auto bn = blk.getBlockNode();
                    const unsigned int regLev = blk.getRegularGridLev();
                    const double h =
                        (double)(1u << (m_uiMaxDepth - regLev)) /
                        (double)(m->getElementOrder());
                    for (unsigned int k = 0; k < lz; k++)
                        for (unsigned int j = 0; j < ly; j++)
                            for (unsigned int i = 0; i < lx; i++) {
                                double x = (double)bn.getX() +
                                    (double)((int)i - (int)pw) * h;
                                double y = (double)bn.getY() +
                                    (double)((int)j - (int)pw) * h;
                                double z = (double)bn.getZ() +
                                    (double)((int)k - (int)pw) * h;
                                size_t c =
                                    offset + k*(ly*lx) + j*lx + i;
                                mine.push_back({x, y, z, uz[c]});
                            }
                }
                delete[] uz;
                int myCount = (int)mine.size();
                std::vector<int> counts(npes), offs(npes, 0);
                MPI_Allgather(&myCount, 1, MPI_INT, counts.data(), 1,
                              MPI_INT, comm);
                int total = 0;
                for (int p = 0; p < npes; p++) {
                    offs[p] = total;
                    total += counts[p];
                }
                std::vector<double> aX(total), aY(total), aZ(total),
                    aV(total);
                std::vector<double> mX(myCount), mY(myCount),
                    mZ(myCount), mV(myCount);
                for (int i = 0; i < myCount; i++) {
                    mX[i] = mine[i].x;
                    mY[i] = mine[i].y;
                    mZ[i] = mine[i].z;
                    mV[i] = mine[i].val;
                }
                MPI_Allgatherv(mX.data(), myCount, MPI_DOUBLE,
                               aX.data(), counts.data(), offs.data(),
                               MPI_DOUBLE, comm);
                MPI_Allgatherv(mY.data(), myCount, MPI_DOUBLE,
                               aY.data(), counts.data(), offs.data(),
                               MPI_DOUBLE, comm);
                MPI_Allgatherv(mZ.data(), myCount, MPI_DOUBLE,
                               aZ.data(), counts.data(), offs.data(),
                               MPI_DOUBLE, comm);
                MPI_Allgatherv(mV.data(), myCount, MPI_DOUBLE,
                               aV.data(), counts.data(), offs.data(),
                               MPI_DOUBLE, comm);
                for (int i = 0; i < total; i++) {
                    auto k = std::make_tuple(
                        (int64_t)std::round(aX[i] * SCALE),
                        (int64_t)std::round(aY[i] * SCALE),
                        (int64_t)std::round(aZ[i] * SCALE));
                    out[k].push_back(aV[i]);
                }
            };
            std::map<std::tuple<int64_t, int64_t, int64_t>,
                     std::vector<double>>
                sfcU, graphU;
            gather_unzip(mesh, sfcU);
            gather_unzip(mesh_repartitioned, graphU);
            int matched = 0, mism = 0;
            double maxD = 0;
            int graphIntra = 0;
            double graphIntraMax = 0;
            int sfcIntra = 0;
            double sfcIntraMax = 0;
            for (auto& kv : graphU) {
                // intra check on graph
                for (size_t i = 1; i < kv.second.size(); i++) {
                    double d = std::abs(kv.second[i] - kv.second[0]);
                    if (d > graphIntraMax) graphIntraMax = d;
                    if (d > 0) graphIntra++;
                }
                auto it = sfcU.find(kv.first);
                if (it == sfcU.end()) continue;
                matched++;
                double d = std::abs(kv.second[0] - it->second[0]);
                if (d > maxD) maxD = d;
                if (d > 0) mism++;
            }
            for (auto& kv : sfcU) {
                for (size_t i = 1; i < kv.second.size(); i++) {
                    double d = std::abs(kv.second[i] - kv.second[0]);
                    if (d > sfcIntraMax) sfcIntraMax = d;
                    if (d > 0) sfcIntra++;
                }
            }
            if (!rank)
                std::cout
                    << (maxD > 1e-12 || graphIntraMax > 1e-12 ? RED : GRN)
                    << "TEST 8bp (unzip SFC vs graph at matching pos):"
                    << " matched=" << matched << " mism=" << mism
                    << " maxDiff=" << maxD
                    << " graphIntra=" << graphIntra
                    << "(max=" << graphIntraMax << ")"
                    << " sfcIntra=" << sfcIntra
                    << "(max=" << sfcIntraMax << ")"
                    << NRM << std::endl;
        }

        // TEST 8bo: compare SFC vs graph RHS-ish output (laplacian-like
        // 6th-order stencil). For each phys_pos covered by both
        // meshes' zip result, compare SFC's vs graph's values.
        // This is a direct check of "does evolution code give same
        // values at same phys_pos across partitions?"
        auto applyLap = [&](ot::Mesh* m, std::vector<double>& vec_out) {
            if (!m->isActive()) return;
            std::vector<double> vec_in;
            m->createVector(vec_in, symFunc);
            m->performGhostExchange(vec_in);
            double* uz_in = m->createUnZippedVector<double>(0.0);
            double* uz_out = m->createUnZippedVector<double>(0.0);
            m->unzip(vec_in.data(), uz_in);
            // Apply 6th-order Laplacian-like central stencil
            const auto& blkList = m->getLocalBlockList();
            for (size_t b = 0; b < blkList.size(); b++) {
                const auto& blk = blkList[b];
                const unsigned int lx = blk.getAllocationSzX();
                const unsigned int ly = blk.getAllocationSzY();
                const unsigned int lz = blk.getAllocationSzZ();
                const unsigned int pw = blk.get1DPadWidth();
                const size_t offset = blk.getOffset();
                for (unsigned int k = 3; k < lz - 3; k++)
                    for (unsigned int j = 3; j < ly - 3; j++)
                        for (unsigned int i = 3; i < lx - 3; i++) {
                            size_t c = offset + k*(ly*lx) + j*lx + i;
                            // 6th-order d2/dx2 (same coeffs as NLSM)
                            double dxx =
                                (2.0 * uz_in[c - 3] -
                                 27.0 * uz_in[c - 2] +
                                 270.0 * uz_in[c - 1] -
                                 490.0 * uz_in[c] +
                                 270.0 * uz_in[c + 1] -
                                 27.0 * uz_in[c + 2] +
                                 2.0 * uz_in[c + 3]) / 180.0;
                            double dyy =
                                (2.0 * uz_in[c - 3*lx] -
                                 27.0 * uz_in[c - 2*lx] +
                                 270.0 * uz_in[c - 1*lx] -
                                 490.0 * uz_in[c] +
                                 270.0 * uz_in[c + 1*lx] -
                                 27.0 * uz_in[c + 2*lx] +
                                 2.0 * uz_in[c + 3*lx]) / 180.0;
                            double dzz =
                                (2.0 * uz_in[c - 3*lx*ly] -
                                 27.0 * uz_in[c - 2*lx*ly] +
                                 270.0 * uz_in[c - 1*lx*ly] -
                                 490.0 * uz_in[c] +
                                 270.0 * uz_in[c + 1*lx*ly] -
                                 27.0 * uz_in[c + 2*lx*ly] +
                                 2.0 * uz_in[c + 3*lx*ly]) / 180.0;
                            uz_out[c] = dxx + dyy + dzz;
                        }
            }
            m->createVector(vec_out, (double)0);
            m->zip(uz_out, vec_out.data());
            // Mirror NLSM's rhs path: post-zip ghost exchange so
            // local CGs whose canonical DG owner lives on a
            // different rank get their values delivered. Without
            // this, both SFC and graph leave such slots at 0
            // (different sets of slots on each), producing spurious
            // 8bo mismatches that don't reflect a real partition bug.
            m->performGhostExchange(vec_out);
            delete[] uz_in;
            delete[] uz_out;
        };

        std::vector<double> vecSfcLap, vecGraphLap;
        applyLap(mesh, vecSfcLap);
        applyLap(mesh_repartitioned, vecGraphLap);

        // Compare at matching element+sub.
        if (mesh->isActive() && mesh_repartitioned->isActive()) {
            struct TNSubKey {
                uint64_t x, y, z;
                uint32_t lvl, sub;
                bool operator<(const TNSubKey& o) const {
                    return std::tie(x, y, z, lvl, sub) <
                           std::tie(o.x, o.y, o.z, o.lvl, o.sub);
                }
            };
            std::map<TNSubKey, double> sfcLapMap;
            {
                const unsigned int npe =
                    mesh->getNumNodesPerElement();
                const auto* pN = mesh->getAllElements().data();
                const auto& e2n = mesh->getE2NMapping();
                for (unsigned int e =
                         mesh->getElementLocalBegin();
                     e < mesh->getElementLocalEnd(); e++) {
                    TNSubKey k;
                    k.x = pN[e].getX();
                    k.y = pN[e].getY();
                    k.z = pN[e].getZ();
                    k.lvl = pN[e].getLevel();
                    for (unsigned int n = 0; n < npe; n++) {
                        k.sub = n;
                        sfcLapMap.emplace(
                            k, vecSfcLap[e2n[e * npe + n]]);
                    }
                }
            }
            int chk = 0, mism = 0;
            double maxD = 0;
            int printed = 0;
            {
                const unsigned int npe =
                    mesh_repartitioned->getNumNodesPerElement();
                const auto* pN =
                    mesh_repartitioned->getAllElements().data();
                const auto& e2n =
                    mesh_repartitioned->getE2NMapping();
                for (unsigned int e =
                         mesh_repartitioned->getElementLocalBegin();
                     e < mesh_repartitioned->getElementLocalEnd();
                     e++) {
                    TNSubKey k;
                    k.x = pN[e].getX();
                    k.y = pN[e].getY();
                    k.z = pN[e].getZ();
                    k.lvl = pN[e].getLevel();
                    for (unsigned int n = 0; n < npe; n++) {
                        k.sub = n;
                        auto it = sfcLapMap.find(k);
                        if (it == sfcLapMap.end()) continue;
                        double g = vecGraphLap[e2n[e * npe + n]];
                        double d = std::abs(g - it->second);
                        if (d > maxD) maxD = d;
                        if (d > 0) mism++;
                        chk++;
                        if (g == 0.0 && std::abs(it->second) > 1e-10) {
                            if (printed < 10) {
                                std::cout
                                    << "  [r" << rank << "] graph-zero e="
                                    << e << " TN=(" << k.x << "," << k.y
                                    << "," << k.z << ",L" << k.lvl
                                    << ") sub=" << n
                                    << " graph=0 sfc=" << it->second << "\n";
                                printed++;
                            }
                        }
                        if (d > 1e-10 && printed < 10) {
                            std::cout
                                << "  [r" << rank << "] mismatch e="
                                << e << " TN=(" << k.x << "," << k.y
                                << "," << k.z << ",L" << k.lvl
                                << ") sub=" << n
                                << " graph=" << g
                                << " sfc=" << it->second
                                << " d=" << d << "\n";
                            printed++;
                        }
                    }
                }
            }
            int gChk, gMism;
            double gMaxD;
            MPI_Allreduce(&chk, &gChk, 1, MPI_INT, MPI_SUM, comm);
            MPI_Allreduce(&mism, &gMism, 1, MPI_INT, MPI_SUM, comm);
            MPI_Allreduce(&maxD, &gMaxD, 1, MPI_DOUBLE, MPI_MAX, comm);
            if (!rank)
                std::cout
                    << (gMaxD > 1e-12 ? RED : GRN)
                    << "TEST 8bo (FD laplacian SFC vs graph after zip):"
                    << " checked=" << gChk << " mism=" << gMism
                    << " maxDiff=" << gMaxD << NRM << std::endl;
        }

        // TEST 8bq: orphan-slot check. Iterate the graph mesh's LOCAL
        // CG index range directly (NOT via local elements) so we
        // visit orphans — local CGs not referenced by any local
        // element. For each, look up the same phys_pos on SFC and
        // compare the post-zip+ghost-exchange Laplacian value.
        // Hypothesis: orphans whose canonical DG owner lives on a
        // DIFFERENT rank are never written (zip skips, no recv-SM
        // entry on this rank, no ghost exchange delivers a value),
        // so they retain stale data while non-orphans match SFC.
        if (mesh->isActive() && mesh_repartitioned->isActive()) {
            const double SCALE = 1e6;

            // Build a global phys_pos -> SFC-value map by gathering
            // every rank's local CG slot.
            auto gatherPosMap = [&](ot::Mesh* m, std::vector<double>& vec)
                -> std::map<std::tuple<int64_t, int64_t, int64_t>, double> {
                std::map<std::tuple<int64_t, int64_t, int64_t>, double> out;
                if (!m->isActive()) return out;
                const auto& cg2dg = m->getCG2DGMap();
                const auto* pN = m->getAllElements().data();
                const unsigned int eOrd = m->getElementOrder();
                const unsigned int nPe = m->getNumNodesPerElement();
                const unsigned int nLB = m->getNodeLocalBegin();
                const unsigned int nLE = m->getNodeLocalEnd();
                std::vector<double> mX, mY, mZ, mV;
                mX.reserve(nLE - nLB);
                for (unsigned int cg = nLB; cg < nLE; cg++) {
                    unsigned int dg = cg2dg[cg];
                    unsigned int oe = dg / nPe;
                    unsigned int os = dg % nPe;
                    double len = (double)(1u << (m_uiMaxDepth - pN[oe].getLevel()));
                    double x = (double)pN[oe].getX()
                             + (double)(os % (eOrd + 1)) * (len / eOrd);
                    double y = (double)pN[oe].getY()
                             + (double)((os / (eOrd + 1)) % (eOrd + 1))
                                 * (len / eOrd);
                    double z = (double)pN[oe].getZ()
                             + (double)(os / ((eOrd + 1) * (eOrd + 1)))
                                 * (len / eOrd);
                    mX.push_back(x);
                    mY.push_back(y);
                    mZ.push_back(z);
                    mV.push_back(vec[cg]);
                }
                int myCount = (int)mX.size();
                std::vector<int> counts(npes), offs(npes, 0);
                MPI_Allgather(&myCount, 1, MPI_INT, counts.data(), 1,
                              MPI_INT, comm);
                int total = 0;
                for (int p = 0; p < npes; p++) {
                    offs[p] = total;
                    total += counts[p];
                }
                std::vector<double> aX(total), aY(total), aZ(total),
                    aV(total);
                MPI_Allgatherv(mX.data(), myCount, MPI_DOUBLE, aX.data(),
                               counts.data(), offs.data(), MPI_DOUBLE, comm);
                MPI_Allgatherv(mY.data(), myCount, MPI_DOUBLE, aY.data(),
                               counts.data(), offs.data(), MPI_DOUBLE, comm);
                MPI_Allgatherv(mZ.data(), myCount, MPI_DOUBLE, aZ.data(),
                               counts.data(), offs.data(), MPI_DOUBLE, comm);
                MPI_Allgatherv(mV.data(), myCount, MPI_DOUBLE, aV.data(),
                               counts.data(), offs.data(), MPI_DOUBLE, comm);
                for (int i = 0; i < total; i++) {
                    auto k = std::make_tuple(
                        (int64_t)std::round(aX[i] * SCALE),
                        (int64_t)std::round(aY[i] * SCALE),
                        (int64_t)std::round(aZ[i] * SCALE));
                    out[k] = aV[i];
                }
                return out;
            };
            auto sfcMap = gatherPosMap(mesh, vecSfcLap);

            // Identify orphans on graph: local CGs not referenced by
            // any local element via E2N_CG.
            const unsigned int nLB =
                mesh_repartitioned->getNodeLocalBegin();
            const unsigned int nLE =
                mesh_repartitioned->getNodeLocalEnd();
            const unsigned int nPe =
                mesh_repartitioned->getNumNodesPerElement();
            const unsigned int eOrd =
                mesh_repartitioned->getElementOrder();
            const auto& cg2dg = mesh_repartitioned->getCG2DGMap();
            const auto& e2n = mesh_repartitioned->getE2NMapping();
            const auto* pN =
                mesh_repartitioned->getAllElements().data();

            std::vector<unsigned char> refByLocal(nLE - nLB, 0);
            for (unsigned int e =
                     mesh_repartitioned->getElementLocalBegin();
                 e < mesh_repartitioned->getElementLocalEnd(); e++) {
                for (unsigned int n = 0; n < nPe; n++) {
                    unsigned int cg = e2n[e * nPe + n];
                    if (cg >= nLB && cg < nLE)
                        refByLocal[cg - nLB] = 1;
                }
            }

            // "xRankCan": local CG whose canonical DG owner element
            // is NOT in this rank's local element range. This rank's
            // zip will skip it (no local canonical writer), so
            // without a "reverse" ghost exchange that pulls the
            // value from the owner rank, the slot retains stale data.
            // Distinct from "orphan" (no local element refs the CG
            // at all) — both lead to "zip skips" but only the
            // unreferenced kind is fixed by the existing
            // orphanPreGather redirect.
            const unsigned int eLB =
                mesh_repartitioned->getElementLocalBegin();
            const unsigned int eLE =
                mesh_repartitioned->getElementLocalEnd();

            int orphans = 0, orphMism = 0, nonOrphMism = 0;
            double orphMaxD = 0, nonOrphMaxD = 0;
            int xRankCan = 0, xRankMism = 0;
            double xRankMaxD = 0;
            int unmatched = 0;
            int printed = 0;
            for (unsigned int cg = nLB; cg < nLE; cg++) {
                bool isOrphan = !refByLocal[cg - nLB];
                if (isOrphan) orphans++;
                unsigned int dgRaw = cg2dg[cg];
                unsigned int canEle = dgRaw / nPe;
                bool isXRankCan = (canEle < eLB || canEle >= eLE);
                if (isXRankCan) xRankCan++;
                unsigned int dg = cg2dg[cg];
                unsigned int oe = dg / nPe;
                unsigned int os = dg % nPe;
                double len = (double)(1u << (m_uiMaxDepth -
                                              pN[oe].getLevel()));
                double x = (double)pN[oe].getX()
                         + (double)(os % (eOrd + 1)) * (len / eOrd);
                double y = (double)pN[oe].getY()
                         + (double)((os / (eOrd + 1)) % (eOrd + 1))
                             * (len / eOrd);
                double z = (double)pN[oe].getZ()
                         + (double)(os / ((eOrd + 1) * (eOrd + 1)))
                             * (len / eOrd);
                auto k = std::make_tuple(
                    (int64_t)std::round(x * SCALE),
                    (int64_t)std::round(y * SCALE),
                    (int64_t)std::round(z * SCALE));
                auto it = sfcMap.find(k);
                if (it == sfcMap.end()) {
                    unmatched++;
                    continue;
                }
                double g = vecGraphLap[cg];
                double d = std::abs(g - it->second);
                if (isOrphan) {
                    if (d > 0) orphMism++;
                    if (d > orphMaxD) orphMaxD = d;
                    if (d > 1e-10 && printed < 10) {
                        std::cout << "  [r" << rank
                                  << "] ORPHAN cg=" << cg
                                  << " dg=(" << oe << "," << os << ")"
                                  << " pos=(" << x << "," << y << "," << z
                                  << ") graph=" << g
                                  << " sfc=" << it->second
                                  << " d=" << d << "\n";
                        printed++;
                    }
                } else {
                    if (d > 0) nonOrphMism++;
                    if (d > nonOrphMaxD) nonOrphMaxD = d;
                }
                if (isXRankCan) {
                    if (d > 0) xRankMism++;
                    if (d > xRankMaxD) xRankMaxD = d;
                    if (d > 1e-10 && printed < 10) {
                        std::cout << "  [r" << rank
                                  << "] XRANKCAN cg=" << cg
                                  << " canEle=" << canEle
                                  << " (eLB=" << eLB << ",eLE=" << eLE
                                  << ") pos=(" << x << "," << y << ","
                                  << z << ") graph=" << g
                                  << " sfc=" << it->second
                                  << " d=" << d << "\n";
                        printed++;
                    }
                }
            }

            int gOrph = 0, gOrphMism = 0, gNonOrphMism = 0,
                gUnmatched = 0;
            int gXRank = 0, gXRankMism = 0;
            double gOrphMaxD = 0, gNonOrphMaxD = 0, gXRankMaxD = 0;
            MPI_Allreduce(&orphans, &gOrph, 1, MPI_INT, MPI_SUM, comm);
            MPI_Allreduce(&orphMism, &gOrphMism, 1, MPI_INT, MPI_SUM,
                          comm);
            MPI_Allreduce(&nonOrphMism, &gNonOrphMism, 1, MPI_INT,
                          MPI_SUM, comm);
            MPI_Allreduce(&unmatched, &gUnmatched, 1, MPI_INT,
                          MPI_SUM, comm);
            MPI_Allreduce(&orphMaxD, &gOrphMaxD, 1, MPI_DOUBLE,
                          MPI_MAX, comm);
            MPI_Allreduce(&nonOrphMaxD, &gNonOrphMaxD, 1, MPI_DOUBLE,
                          MPI_MAX, comm);
            MPI_Allreduce(&xRankCan, &gXRank, 1, MPI_INT, MPI_SUM,
                          comm);
            MPI_Allreduce(&xRankMism, &gXRankMism, 1, MPI_INT,
                          MPI_SUM, comm);
            MPI_Allreduce(&xRankMaxD, &gXRankMaxD, 1, MPI_DOUBLE,
                          MPI_MAX, comm);
            if (!rank)
                std::cout
                    << ((gOrphMaxD > 1e-12 || gXRankMaxD > 1e-12)
                            ? RED
                            : GRN)
                    << "TEST 8bq (orphan/xRankCanonical rhs check):"
                    << " orphans=" << gOrph
                    << " orphMism=" << gOrphMism
                    << " orphMaxDiff=" << gOrphMaxD
                    << " xRankCan=" << gXRank
                    << " xRankMism=" << gXRankMism
                    << " xRankMaxDiff=" << gXRankMaxD
                    << " nonOrphMism=" << gNonOrphMism
                    << " nonOrphMaxDiff=" << gNonOrphMaxD
                    << " unmatched=" << gUnmatched
                    << NRM << std::endl;

            // TEST 8br: duplicate-local-ownership check. The graph
            // partition's per-rank canonical determination can land
            // on different elements on different ranks for the same
            // shared physical CG, leaving multiple ranks each
            // claiming `isLocal=1` on the same phys_pos. Neither
            // rank's zip writes the slot (their E2N_DG was rewritten
            // inconsistently), and no ghost exchange delivers the
            // value — this is the actual root cause of NLSM step-1
            // divergence on graph (see plan starry-weaving-duckling).
            // SFC must report 0 duplicates by construction.
            {
                std::vector<int64_t> myKeys;
                myKeys.reserve(3 * (nLE - nLB));
                for (unsigned int cg = nLB; cg < nLE; cg++) {
                    unsigned int dg = cg2dg[cg];
                    unsigned int oe = dg / nPe;
                    unsigned int os = dg % nPe;
                    double len = (double)(1u << (m_uiMaxDepth -
                                                  pN[oe].getLevel()));
                    double x = (double)pN[oe].getX()
                             + (double)(os % (eOrd + 1)) * (len / eOrd);
                    double y = (double)pN[oe].getY()
                             + (double)((os / (eOrd + 1)) % (eOrd + 1))
                                 * (len / eOrd);
                    double z = (double)pN[oe].getZ()
                             + (double)(os / ((eOrd + 1) * (eOrd + 1)))
                                 * (len / eOrd);
                    myKeys.push_back((int64_t)std::round(x * SCALE));
                    myKeys.push_back((int64_t)std::round(y * SCALE));
                    myKeys.push_back((int64_t)std::round(z * SCALE));
                }
                int myCount = (int)(myKeys.size() / 3);
                std::vector<int> counts(npes), offs(npes, 0);
                MPI_Allgather(&myCount, 1, MPI_INT, counts.data(),
                              1, MPI_INT, comm);
                int total = 0;
                for (int p = 0; p < npes; p++) {
                    offs[p] = total;
                    total += counts[p];
                }
                std::vector<int> bcnts(npes), boffs(npes, 0);
                int bTotal = 0;
                for (int p = 0; p < npes; p++) {
                    bcnts[p] = counts[p] * 3;
                    boffs[p] = bTotal;
                    bTotal += bcnts[p];
                }
                std::vector<int64_t> allKeys(bTotal);
                MPI_Allgatherv(myKeys.data(), myKeys.size(),
                               MPI_INT64_T, allKeys.data(),
                               bcnts.data(), boffs.data(),
                               MPI_INT64_T, comm);
                std::map<std::tuple<int64_t, int64_t, int64_t>,
                         int> ownerCount;
                for (int p = 0; p < npes; p++) {
                    for (int i = 0; i < counts[p]; i++) {
                        int idx = boffs[p] + i * 3;
                        auto k = std::make_tuple(
                            allKeys[idx],
                            allKeys[idx + 1],
                            allKeys[idx + 2]);
                        ownerCount[k]++;
                    }
                }
                int dups = 0;
                int maxOwners = 0;
                for (auto& kv : ownerCount) {
                    if (kv.second > 1) {
                        dups++;
                        if (kv.second > maxOwners)
                            maxOwners = kv.second;
                    }
                }
                if (!rank)
                    std::cout
                        << (dups > 0 ? RED : GRN)
                        << "TEST 8br (duplicate local-CG owners):"
                        << " dupPhysPos=" << dups
                        << " maxOwners=" << maxOwners
                        << " uniquePhysPos=" << ownerCount.size()
                        << NRM << std::endl;
            }
        }

        testMirror(mesh, "TEST 8bk-SFC (no cycles, sync)",
                   0, false, false);
        testMirror(mesh_repartitioned,
                   "TEST 8bk-graph (no cycles, sync)",
                   0, false, false);
        testMirror(mesh, "TEST 8bk-SFC (20 cycles+rhs, sync)",
                   20, false, true);
        testMirror(mesh_repartitioned,
                   "TEST 8bk-graph (20 cycles+rhs, sync)",
                   20, false, true);
        testMirror(mesh, "TEST 8bk-SFC (20 cycles+rhs, async)",
                   20, true, true);
        testMirror(mesh_repartitioned,
                   "TEST 8bk-graph (20 cycles+rhs, async)",
                   20, true, true);
    }

    // TEST 8be: redistributeVec roundtrip sfc→graph→sfc.
    // Does the graph-twin redistribute introduce drift?
    // Uses the exact NLSM buildGraphTwin path (E2E_ONLY → FDM flip).
    if (partitionOption != PartitioningOptions::NoPartition &&
        partitionOption != PartitioningOptions::OriginalPartition) {
        // Fresh SFC source with analytic values.
        std::vector<double> vecSfc;
        mesh->createVector(vecSfc, func);
        mesh->performGhostExchange(vecSfc);

        // Build graph twin (NLSM buildGraphTwin path).
        std::vector<ot::TreeNode> oct;
        if (mesh->isActive()) {
            const auto* pN = mesh->getAllElements().data();
            for (unsigned int e = mesh->getElementLocalBegin();
                 e < mesh->getElementLocalEnd(); e++)
                oct.push_back(pN[e]);
        }
        ot::Mesh* gTwin = ot::createMesh(
            oct.data(), oct.size(), eOrder, comm, 1,
            ot::SM_TYPE::E2E_ONLY, DENDRO_GRAIN_SZ, LOAD_IMB_TOL, SPLIT_FIX);
        gTwin->setDomainBounds(pt_min, pt_max);
        gTwin->setPartitioningMethod(partitionOption);
        gTwin->setScatterMapType(ot::SM_TYPE::FDM);
        gTwin->repartitionMeshGlobal();

        // sfc → graph
        std::vector<double> vecGraph;
        gTwin->createVector(vecGraph, (double)0);
        mesh->redistributeVec(gTwin, vecSfc.data(), vecGraph.data());
        gTwin->performGhostExchange(vecGraph);

        // graph → sfc (back to original SFC mesh)
        std::vector<double> vecSfc2;
        mesh->createVector(vecSfc2, (double)0);
        gTwin->redistributeVec(mesh, vecGraph.data(), vecSfc2.data());
        mesh->performGhostExchange(vecSfc2);

        const unsigned int nLB = mesh->getNodeLocalBegin();
        const unsigned int nLE = mesh->getNodeLocalEnd();
        int errs = 0, chk = 0;
        double mxE = 0;
        for (unsigned int cg = nLB; cg < nLE; cg++) {
            double d = std::abs(vecSfc[cg] - vecSfc2[cg]);
            if (d > mxE) mxE = d;
            if (d > 1e-10) errs++;
            chk++;
        }
        int gE, gC;
        double gM;
        MPI_Allreduce(&errs, &gE, 1, MPI_INT, MPI_SUM, comm);
        MPI_Allreduce(&chk, &gC, 1, MPI_INT, MPI_SUM, comm);
        MPI_Allreduce(&mxE, &gM, 1, MPI_DOUBLE, MPI_MAX, comm);
        if (!rank) {
            if (gE == 0)
                std::cout << GRN
                          << "TEST 8be (sfc→graph→sfc redistribute) "
                          << "PASSED: " << gC << " cgs, maxErr=" << gM
                          << NRM << std::endl;
            else
                std::cout << RED
                          << "TEST 8be: " << gE << " / " << gC
                          << " redistribute roundtrip mismatches, maxErr="
                          << gM << NRM << std::endl;
        }
        delete gTwin;
    }

    // TEST 8bd: mimics NLSM's buildGraphTwin path exactly:
    //   createMesh(SM_TYPE::E2E_ONLY) -> setScatterMapType(FDM)
    //   -> repartitionMeshGlobal.
    // Test mesh_repartitioned used SM_TYPE::FDM directly — that path is
    // verified exact in 8bb. If 8bd fails, the E2E_ONLY -> FDM flip is
    // the NLSM pre-remesh evolution divergence root cause.
    if (partitionOption != PartitioningOptions::NoPartition &&
        partitionOption != PartitioningOptions::OriginalPartition) {
        std::vector<ot::TreeNode> oct;
        if (mesh->isActive()) {
            const auto* pN = mesh->getAllElements().data();
            for (unsigned int e = mesh->getElementLocalBegin();
                 e < mesh->getElementLocalEnd(); e++)
                oct.push_back(pN[e]);
        }
        ot::Mesh* twin = ot::createMesh(
            oct.data(), oct.size(), eOrder, comm, 1,
            ot::SM_TYPE::E2E_ONLY, DENDRO_GRAIN_SZ, LOAD_IMB_TOL, SPLIT_FIX);
        twin->setDomainBounds(pt_min, pt_max);
        twin->setPartitioningMethod(partitionOption);
        twin->setScatterMapType(ot::SM_TYPE::FDM);
        twin->repartitionMeshGlobal();
        testZipRoundtrip(twin, "TEST 8bd (E2E_ONLY->FDM twin roundtrip)");
        delete twin;
    }

    // TEST 8bs: EM4-style buildGraphTwin path (E2E_ONLY -> FDM flip
    // WITH ownerMask + blockInfo injection) — does unzip produce
    // bit-perfect output vs the source SFC mesh?
    //
    // 8bp proves the IN-PLACE repartition path is bit-perfect. EM4 uses
    // a structurally different path; this test isolates whether that
    // path also produces bit-perfect unzip. If yes, EM4 long-haul
    // divergence is at the solver level (RHS/RK), not the mesh level.
    if (partitionOption != PartitioningOptions::NoPartition &&
        partitionOption != PartitioningOptions::OriginalPartition) {
        std::vector<ot::TreeNode> oct;
        if (mesh->isActive()) {
            const auto* pN = mesh->getAllElements().data();
            for (unsigned int e = mesh->getElementLocalBegin();
                 e < mesh->getElementLocalEnd(); e++)
                oct.push_back(pN[e]);
        }
        ot::Mesh* gTwin = ot::createMesh(
            oct.data(), oct.size(), eOrder, comm, 1,
            ot::SM_TYPE::E2E_ONLY, DENDRO_GRAIN_SZ, LOAD_IMB_TOL, SPLIT_FIX);
        gTwin->setDomainBounds(pt_min, pt_max);
        gTwin->setPartitioningMethod(partitionOption);
        gTwin->setScatterMapType(ot::SM_TYPE::FDM);

        // mirror em4_partitioning.h's buildGraphTwin injection
        if (mesh->isActive() && gTwin->isActive()) {
            const auto& srcMask = mesh->getOwnerMask();
            const unsigned int srcLB = mesh->getElementLocalBegin();
            const unsigned int srcLE = mesh->getElementLocalEnd();
            const unsigned int dstLB = gTwin->getElementLocalBegin();
            const size_t totalElems = gTwin->getAllElements().size();
            std::vector<uint32_t> dstMask(totalElems, 0u);
            if (srcMask.size() >= srcLE) {
                for (unsigned int i = 0;
                     i < (srcLE - srcLB) && (dstLB + i) < totalElems; i++)
                    dstMask[dstLB + i] = srcMask[srcLB + i];
            }
            gTwin->setOwnerMask(std::move(dstMask));

            const auto& srcBlk = mesh->getBlockInfo();
            std::vector<ot::Mesh::CanonicalBlockInfo> dstBlk(totalElems);
            if (srcBlk.size() >= srcLE) {
                for (unsigned int i = 0;
                     i < (srcLE - srcLB) && (dstLB + i) < totalElems; i++)
                    dstBlk[dstLB + i] = srcBlk[srcLB + i];
            }
            gTwin->setBlockInfo(std::move(dstBlk));
        }
        gTwin->repartitionMeshGlobal();

        // run the same gather_unzip + compare logic as 8bp, but
        // against gTwin instead of mesh_repartitioned
        if (mesh->isActive() && gTwin->isActive()) {
            const double DOMAIN_LOC = (double)(1u << m_uiMaxDepth);
            const double dC_LOC =
                0.5 * (double)(1u << m_uiMaxDepth);
            std::function<double(double, double, double)> symFuncLoc =
                [dC_LOC, DOMAIN_LOC](double x, double y, double z) {
                    double fx = (x - dC_LOC) / DOMAIN_LOC * 20.0;
                    double fy = (y - dC_LOC) / DOMAIN_LOC * 20.0;
                    double fz = (z - dC_LOC) / DOMAIN_LOC * 20.0;
                    double r  = std::sqrt(fx * fx + fy * fy + fz * fz);
                    const double amp = 1.3, R = 1.0, delta = 0.5;
                    return amp * std::exp(-(r - R) * (r - R) /
                                          (delta * delta));
                };
            auto gather_unzip = [&](ot::Mesh* m,
                                    std::map<std::tuple<int64_t, int64_t,
                                                        int64_t>,
                                             std::vector<double>>& out) {
                std::vector<double> v;
                m->createVector(v, symFuncLoc);
                m->performGhostExchange(v);
                double* uz = m->createUnZippedVector<double>(0.0);
                m->unzip(v.data(), uz);
                const auto& blkList = m->getLocalBlockList();
                const double SCALE = 1e6;
                struct E { double x, y, z, val; };
                std::vector<E> mine;
                for (size_t b = 0; b < blkList.size(); b++) {
                    const auto& blk = blkList[b];
                    const unsigned int lx = blk.getAllocationSzX();
                    const unsigned int ly = blk.getAllocationSzY();
                    const unsigned int lz = blk.getAllocationSzZ();
                    const unsigned int pw = blk.get1DPadWidth();
                    const size_t offset = blk.getOffset();
                    const auto bn = blk.getBlockNode();
                    const unsigned int regLev = blk.getRegularGridLev();
                    const double h =
                        (double)(1u << (m_uiMaxDepth - regLev)) /
                        (double)(m->getElementOrder());
                    for (unsigned int k = 0; k < lz; k++)
                        for (unsigned int j = 0; j < ly; j++)
                            for (unsigned int i = 0; i < lx; i++) {
                                double x = (double)bn.getX() +
                                    (double)((int)i - (int)pw) * h;
                                double y = (double)bn.getY() +
                                    (double)((int)j - (int)pw) * h;
                                double z = (double)bn.getZ() +
                                    (double)((int)k - (int)pw) * h;
                                size_t c =
                                    offset + k*(ly*lx) + j*lx + i;
                                mine.push_back({x, y, z, uz[c]});
                            }
                }
                delete[] uz;
                int myCount = (int)mine.size();
                std::vector<int> counts(npes), offs(npes, 0);
                MPI_Allgather(&myCount, 1, MPI_INT, counts.data(), 1,
                              MPI_INT, comm);
                int total = 0;
                for (int p = 0; p < npes; p++) {
                    offs[p] = total;
                    total += counts[p];
                }
                std::vector<double> aX(total), aY(total), aZ(total),
                    aV(total);
                std::vector<double> mX(myCount), mY(myCount),
                    mZ(myCount), mV(myCount);
                for (int i = 0; i < myCount; i++) {
                    mX[i] = mine[i].x;
                    mY[i] = mine[i].y;
                    mZ[i] = mine[i].z;
                    mV[i] = mine[i].val;
                }
                MPI_Allgatherv(mX.data(), myCount, MPI_DOUBLE,
                               aX.data(), counts.data(), offs.data(),
                               MPI_DOUBLE, comm);
                MPI_Allgatherv(mY.data(), myCount, MPI_DOUBLE,
                               aY.data(), counts.data(), offs.data(),
                               MPI_DOUBLE, comm);
                MPI_Allgatherv(mZ.data(), myCount, MPI_DOUBLE,
                               aZ.data(), counts.data(), offs.data(),
                               MPI_DOUBLE, comm);
                MPI_Allgatherv(mV.data(), myCount, MPI_DOUBLE,
                               aV.data(), counts.data(), offs.data(),
                               MPI_DOUBLE, comm);
                for (int i = 0; i < total; i++) {
                    auto k = std::make_tuple(
                        (int64_t)std::round(aX[i] * SCALE),
                        (int64_t)std::round(aY[i] * SCALE),
                        (int64_t)std::round(aZ[i] * SCALE));
                    out[k].push_back(aV[i]);
                }
            };
            std::map<std::tuple<int64_t, int64_t, int64_t>,
                     std::vector<double>>
                sfcU, twinU;
            gather_unzip(mesh, sfcU);
            gather_unzip(gTwin, twinU);
            int matched = 0, mism = 0;
            double maxD = 0;
            int twinIntra = 0;
            double twinIntraMax = 0;
            for (auto& kv : twinU) {
                for (size_t i = 1; i < kv.second.size(); i++) {
                    double d = std::abs(kv.second[i] - kv.second[0]);
                    if (d > twinIntraMax) twinIntraMax = d;
                    if (d > 0) twinIntra++;
                }
                auto it = sfcU.find(kv.first);
                if (it == sfcU.end()) continue;
                matched++;
                double d = std::abs(kv.second[0] - it->second[0]);
                if (d > maxD) maxD = d;
                if (d > 0) mism++;
            }
            if (!rank)
                std::cout
                    << (maxD > 1e-12 || twinIntraMax > 1e-12 ? RED : GRN)
                    << "TEST 8bs (EM4 buildGraphTwin path unzip "
                       "vs SFC at matching pos):"
                    << " matched=" << matched << " mism=" << mism
                    << " maxDiff=" << maxD
                    << " twinIntra=" << twinIntra
                    << "(max=" << twinIntraMax << ")"
                    << NRM << std::endl;
        }

        // TEST 8bt: simulate a step-1-like evolution and check CG sync.
        // Mirrors what EM4's solver does: createVector → unzip → modify
        // unzipped buffer → zip back → ghost-exchange. If multiple CG
        // slots at the same physical position end up with different
        // values, the sync is broken — which is exactly the EM4
        // long-haul divergence symptom.
        if (mesh->isActive() && gTwin->isActive()) {
            std::function<double(double,double,double)> probeFunc =
                [](double x, double y, double z) {
                    return std::sin(0.01 * x) + std::cos(0.01 * y)
                         + std::sin(0.01 * z);
                };
            auto evolveAndCheck = [&](ot::Mesh* m, const char* label) {
                std::vector<double> v;
                m->createVector(v, probeFunc);
                m->performGhostExchange(v);
                double* uz = m->createUnZippedVector<double>(0.0);
                m->unzip(v.data(), uz);
                // simulated rhs: each block, multiply by 1.001
                const auto& blkList = m->getLocalBlockList();
                for (size_t b = 0; b < blkList.size(); b++) {
                    const auto& blk = blkList[b];
                    const size_t off  = blk.getOffset();
                    const size_t lx   = blk.getAllocationSzX();
                    const size_t ly   = blk.getAllocationSzY();
                    const size_t lz   = blk.getAllocationSzZ();
                    for (size_t k = 0; k < lz; k++)
                        for (size_t j = 0; j < ly; j++)
                            for (size_t i = 0; i < lx; i++) {
                                size_t c = off + k*(ly*lx) + j*lx + i;
                                uz[c] *= 1.001;
                            }
                }
                std::vector<double> vNew;
                m->createVector(vNew, (double)0);
                m->zip(uz, vNew.data());
                m->performGhostExchange(vNew);
                delete[] uz;

                // gather all (phys_pos, value) pairs from all elements
                // (every element's getElementNodalValues for its 343
                // sub-positions). check: at each phys_pos seen by
                // multiple elements/ranks, do all values match?
                const unsigned int eOrd = m->getElementOrder();
                const unsigned int nPe  = m->getNumNodesPerElement();
                std::vector<double> nodalVal(nPe);
                struct E { double x, y, z, val; };
                std::vector<E> mine;
                const auto& aE = m->getAllElements();
                for (unsigned int e = m->getElementLocalBegin();
                     e < m->getElementLocalEnd(); e++) {
                    m->getElementNodalValues(vNew.data(), nodalVal.data(),
                                              e, false);
                    const unsigned int sz =
                        1u << (m_uiMaxDepth - aE[e].getLevel());
                    for (unsigned int kk = 0; kk <= eOrd; kk++)
                        for (unsigned int jj = 0; jj <= eOrd; jj++)
                            for (unsigned int ii = 0; ii <= eOrd; ii++) {
                                double x = aE[e].getX() + ii * (double)sz / eOrd;
                                double y = aE[e].getY() + jj * (double)sz / eOrd;
                                double z = aE[e].getZ() + kk * (double)sz / eOrd;
                                size_t s = kk*(eOrd+1)*(eOrd+1) + jj*(eOrd+1) + ii;
                                mine.push_back({x, y, z, nodalVal[s]});
                            }
                }
                int myCount = (int)mine.size();
                std::vector<int> counts(npes), offs(npes, 0);
                MPI_Allgather(&myCount, 1, MPI_INT, counts.data(),
                              1, MPI_INT, comm);
                int total = 0;
                for (int p = 0; p < npes; p++) {
                    offs[p] = total;
                    total += counts[p];
                }
                std::vector<double> aX(total), aY(total), aZ(total),
                    aV(total);
                std::vector<double> mX(myCount), mY(myCount), mZ(myCount),
                    mV(myCount);
                for (int i = 0; i < myCount; i++) {
                    mX[i] = mine[i].x; mY[i] = mine[i].y;
                    mZ[i] = mine[i].z; mV[i] = mine[i].val;
                }
                MPI_Allgatherv(mX.data(), myCount, MPI_DOUBLE, aX.data(),
                               counts.data(), offs.data(), MPI_DOUBLE, comm);
                MPI_Allgatherv(mY.data(), myCount, MPI_DOUBLE, aY.data(),
                               counts.data(), offs.data(), MPI_DOUBLE, comm);
                MPI_Allgatherv(mZ.data(), myCount, MPI_DOUBLE, aZ.data(),
                               counts.data(), offs.data(), MPI_DOUBLE, comm);
                MPI_Allgatherv(mV.data(), myCount, MPI_DOUBLE, aV.data(),
                               counts.data(), offs.data(), MPI_DOUBLE, comm);
                std::map<std::tuple<int64_t,int64_t,int64_t>,
                         std::vector<double>> byPos;
                const double SCALE = 1e6;
                for (int i = 0; i < total; i++) {
                    auto k = std::make_tuple(
                        (int64_t)std::round(aX[i] * SCALE),
                        (int64_t)std::round(aY[i] * SCALE),
                        (int64_t)std::round(aZ[i] * SCALE));
                    byPos[k].push_back(aV[i]);
                }
                int multi = 0, badPos = 0;
                double maxIntra = 0;
                for (auto& kv : byPos) {
                    if (kv.second.size() <= 1) continue;
                    multi++;
                    double mn = kv.second[0], mx = kv.second[0];
                    for (double v : kv.second) {
                        if (v < mn) mn = v;
                        if (v > mx) mx = v;
                    }
                    double diff = mx - mn;
                    if (diff > maxIntra) maxIntra = diff;
                    if (diff > 1e-12) badPos++;
                }
                if (!rank)
                    std::cout << (badPos > 0 ? RED : GRN)
                              << "TEST 8bt " << label
                              << ": multi-element positions=" << multi
                              << " disagreeing=" << badPos
                              << " maxIntra=" << maxIntra
                              << NRM << std::endl;
            };
            evolveAndCheck(mesh, "(SFC)        ");
            evolveAndCheck(gTwin, "(EM4 graph twin)");

            // TEST 8bu: for every multi-element shared physical
            // position, check whether ALL element's E2N_CG entries
            // point at the SAME cg index. if any position has
            // diverging cg indices, zip writes only one canonical
            // slot and the others stay stale → EM4-style divergence
            // when the per-node arithmetic loop reads non-canonical
            // slots with stale data.
            auto checkCgUnification = [&](ot::Mesh* m, const char* label) {
                if (!m->isActive()) return;
                const auto& aE = m->getAllElements();
                const unsigned int eOrd = m->getElementOrder();
                const unsigned int nPe  = m->getNumNodesPerElement();
                const auto& E2NCG = m->getE2NMapping();
                const unsigned int npe  = nPe;

                struct R { double x, y, z; uint32_t cgIdx; uint32_t rank; };
                std::vector<R> mine;
                for (unsigned int e = m->getElementLocalBegin();
                     e < m->getElementLocalEnd(); e++) {
                    const unsigned int sz =
                        1u << (m_uiMaxDepth - aE[e].getLevel());
                    for (unsigned int kk = 0; kk <= eOrd; kk++)
                        for (unsigned int jj = 0; jj <= eOrd; jj++)
                            for (unsigned int ii = 0; ii <= eOrd; ii++) {
                                size_t s = kk*(eOrd+1)*(eOrd+1)
                                         + jj*(eOrd+1) + ii;
                                double x = aE[e].getX() + ii * (double)sz / eOrd;
                                double y = aE[e].getY() + jj * (double)sz / eOrd;
                                double z = aE[e].getZ() + kk * (double)sz / eOrd;
                                mine.push_back({x, y, z,
                                                E2NCG[e * npe + s],
                                                (uint32_t)rank});
                            }
                }
                int myCount = (int)mine.size();
                std::vector<int> counts(npes), offs(npes, 0);
                MPI_Allgather(&myCount, 1, MPI_INT, counts.data(),
                              1, MPI_INT, comm);
                int total = 0;
                for (int p = 0; p < npes; p++) {
                    offs[p] = total;
                    total += counts[p];
                }
                std::vector<double> aX(total), aY(total), aZ(total);
                std::vector<uint32_t> aCG(total), aR(total);
                std::vector<double> mX(myCount), mY(myCount), mZ(myCount);
                std::vector<uint32_t> mCG(myCount), mR(myCount);
                for (int i = 0; i < myCount; i++) {
                    mX[i] = mine[i].x; mY[i] = mine[i].y;
                    mZ[i] = mine[i].z;
                    mCG[i] = mine[i].cgIdx; mR[i] = mine[i].rank;
                }
                MPI_Allgatherv(mX.data(), myCount, MPI_DOUBLE, aX.data(),
                               counts.data(), offs.data(), MPI_DOUBLE, comm);
                MPI_Allgatherv(mY.data(), myCount, MPI_DOUBLE, aY.data(),
                               counts.data(), offs.data(), MPI_DOUBLE, comm);
                MPI_Allgatherv(mZ.data(), myCount, MPI_DOUBLE, aZ.data(),
                               counts.data(), offs.data(), MPI_DOUBLE, comm);
                MPI_Allgatherv(mCG.data(), myCount, MPI_UINT32_T, aCG.data(),
                               counts.data(), offs.data(), MPI_UINT32_T, comm);
                MPI_Allgatherv(mR.data(), myCount, MPI_UINT32_T, aR.data(),
                               counts.data(), offs.data(), MPI_UINT32_T, comm);

                // group by phys position (within rank scope: cg indices
                // are local to each rank, so we group by (rank, pos))
                std::map<std::tuple<uint32_t, int64_t, int64_t, int64_t>,
                         std::set<uint32_t>> byPosRank;
                const double SCALE = 1e6;
                for (int i = 0; i < total; i++) {
                    auto k = std::make_tuple(
                        aR[i],
                        (int64_t)std::round(aX[i] * SCALE),
                        (int64_t)std::round(aY[i] * SCALE),
                        (int64_t)std::round(aZ[i] * SCALE));
                    byPosRank[k].insert(aCG[i]);
                }
                int multi = 0, badPos = 0;
                int maxIdxCount = 0;
                for (auto& kv : byPosRank) {
                    if (kv.second.size() <= 1) continue;
                    multi++;
                    badPos++;
                    if ((int)kv.second.size() > maxIdxCount)
                        maxIdxCount = (int)kv.second.size();
                }
                if (!rank)
                    std::cout << (badPos > 0 ? RED : GRN)
                              << "TEST 8bu " << label
                              << ": (rank,pos) with multiple cg-indices="
                              << badPos
                              << " (max distinct cg per (rank,pos)="
                              << maxIdxCount << ")"
                              << NRM << std::endl;
            };
            checkCgUnification(mesh, "(SFC)        ");
            checkCgUnification(gTwin, "(EM4 graph twin)");

            // TEST 8bv: cross-rank canonical-cg uniqueness.
            // For each shared physical position, exactly ONE rank
            // should have it as a LOCAL cg slot (canonical owner);
            // other ranks holding the same position should have it
            // as a GHOST cg slot (cg index outside [nodeLocalBegin,
            // nodeLocalEnd)). If multiple ranks have it as LOCAL,
            // CG ownership unification failed across ranks → multiple
            // independent canonical writers → no sync between them →
            // EM4-style desync at face nodes.
            auto checkCrossRankCanonical = [&](ot::Mesh* m,
                                               const char* label) {
                if (!m->isActive()) return;
                const auto& aE = m->getAllElements();
                const unsigned int eOrd = m->getElementOrder();
                const unsigned int nPe  = m->getNumNodesPerElement();
                const auto& E2NCG = m->getE2NMapping();
                const unsigned int nLB = m->getNodeLocalBegin();
                const unsigned int nLE = m->getNodeLocalEnd();
                const unsigned int npe  = nPe;
                struct R { double x, y, z; uint8_t isLocal; };
                std::vector<R> mine;
                for (unsigned int e = m->getElementLocalBegin();
                     e < m->getElementLocalEnd(); e++) {
                    const unsigned int sz =
                        1u << (m_uiMaxDepth - aE[e].getLevel());
                    for (unsigned int kk = 0; kk <= eOrd; kk++)
                        for (unsigned int jj = 0; jj <= eOrd; jj++)
                            for (unsigned int ii = 0; ii <= eOrd; ii++) {
                                size_t s = kk*(eOrd+1)*(eOrd+1)
                                         + jj*(eOrd+1) + ii;
                                double x = aE[e].getX() + ii * (double)sz / eOrd;
                                double y = aE[e].getY() + jj * (double)sz / eOrd;
                                double z = aE[e].getZ() + kk * (double)sz / eOrd;
                                unsigned int cg = E2NCG[e * npe + s];
                                uint8_t loc = (cg >= nLB && cg < nLE) ? 1 : 0;
                                mine.push_back({x, y, z, loc});
                            }
                }
                int myCount = (int)mine.size();
                std::vector<int> counts(npes), offs(npes, 0);
                MPI_Allgather(&myCount, 1, MPI_INT, counts.data(),
                              1, MPI_INT, comm);
                int total = 0;
                for (int p = 0; p < npes; p++) {
                    offs[p] = total;
                    total += counts[p];
                }
                std::vector<double> aX(total), aY(total), aZ(total);
                std::vector<uint8_t> aLoc(total);
                std::vector<double> mX(myCount), mY(myCount), mZ(myCount);
                std::vector<uint8_t> mLoc(myCount);
                for (int i = 0; i < myCount; i++) {
                    mX[i] = mine[i].x; mY[i] = mine[i].y;
                    mZ[i] = mine[i].z; mLoc[i] = mine[i].isLocal;
                }
                MPI_Allgatherv(mX.data(), myCount, MPI_DOUBLE, aX.data(),
                               counts.data(), offs.data(), MPI_DOUBLE, comm);
                MPI_Allgatherv(mY.data(), myCount, MPI_DOUBLE, aY.data(),
                               counts.data(), offs.data(), MPI_DOUBLE, comm);
                MPI_Allgatherv(mZ.data(), myCount, MPI_DOUBLE, aZ.data(),
                               counts.data(), offs.data(), MPI_DOUBLE, comm);
                MPI_Allgatherv(mLoc.data(), myCount, MPI_BYTE, aLoc.data(),
                               counts.data(), offs.data(), MPI_BYTE, comm);
                // tag each entry with its source rank
                std::vector<uint32_t> aRank(total);
                for (int p = 0; p < npes; p++)
                    for (int i = 0; i < counts[p]; i++)
                        aRank[offs[p] + i] = (uint32_t)p;
                std::map<std::tuple<int64_t, int64_t, int64_t>,
                         std::set<uint32_t>> ranksOwningLocal;
                const double SCALE = 1e6;
                for (int i = 0; i < total; i++) {
                    if (!aLoc[i]) continue;
                    auto k = std::make_tuple(
                        (int64_t)std::round(aX[i] * SCALE),
                        (int64_t)std::round(aY[i] * SCALE),
                        (int64_t)std::round(aZ[i] * SCALE));
                    ranksOwningLocal[k].insert(aRank[i]);
                }
                int multiOwner = 0;
                int maxRanks = 0;
                for (auto& kv : ranksOwningLocal) {
                    if ((int)kv.second.size() > 1) {
                        multiOwner++;
                        if ((int)kv.second.size() > maxRanks)
                            maxRanks = (int)kv.second.size();
                    }
                }
                if (!rank)
                    std::cout << (multiOwner > 0 ? RED : GRN)
                              << "TEST 8bv " << label
                              << ": positions with LOCAL cg slot on >1 rank="
                              << multiOwner
                              << " (max ranks per pos=" << maxRanks << ")"
                              << NRM << std::endl;
            };
            checkCrossRankCanonical(mesh, "(SFC)        ");
            checkCrossRankCanonical(gTwin, "(EM4 graph twin)");
        }

        // TEST 8bw: multi-component RK45-stage-1-like cycle on the
        // EM4 buildGraphTwin path. Mimics what EM4 actually does in
        // stage 1 → stage 2 of a single RK45 step, with dof=6 to
        // expose interactions between vector components and the
        // multi-claim-local-CG sync chain.
        //
        // Pattern (per mesh):
        //   1. createVector u (dof=6, smooth IC).
        //   2. ghost-exchange u.
        //   3. unzip u → uz_in.
        //   4. simulated stage-1 RHS: each interior cell of each
        //      block writes a curl-like value into uz_out. PADDING
        //      cells stay 0 (mirrors EM4 RHS, which only touches
        //      block interior).
        //   5. zip uz_out → k1.
        //   6. axpy: u_stage2 = u + (1/5) * k1.
        //   7. ghost-exchange u_stage2 (this is where syncZipNon-
        //      Primary + standard scatter + PassD/Stage3 mirror
        //      runs).
        //   8. unzip u_stage2 → uz_stage2_in.
        //
        // Then compare SFC vs graph_twin uz_stage2_in at every
        // matching grid position. Also report off-axis (face-
        // interior, edge) positions where the bug concentrates.
        //
        // The legacy 8bp/8bs tests use 1 dof and skip steps 4-6
        // (no zip in the path), so they don't exercise the multi-
        // claim sync chain that EM4 hits at level-transition
        // corners. 8bt does run zip + sync but uses 1 dof and a
        // padding-touching simulated RHS, which masks the bug
        // because every cell gets the same ratio.
        if (mesh->isActive() && gTwin->isActive()) {
            const unsigned int DOF = 6;
            const double DOM_W =
                (double)(1u << m_uiMaxDepth);
            std::function<double(int, double, double, double)> ic =
                [DOM_W](int v, double x, double y, double z) {
                    double fx = x / DOM_W * 6.28319;
                    double fy = y / DOM_W * 6.28319;
                    double fz = z / DOM_W * 6.28319;
                    switch (v) {
                        case 0: return std::sin(fx) * std::cos(fy);
                        case 1: return std::cos(fx) * std::sin(fz);
                        case 2: return std::sin(fy) * std::cos(fz);
                        case 3: return std::cos(fx) * std::cos(fz);
                        case 4: return std::sin(fy) * std::sin(fz);
                        default: return std::cos(fy) * std::sin(fx);
                    }
                };
            auto runStageCycle = [&](ot::Mesh* m,
                std::map<std::tuple<int64_t,int64_t,int64_t>,
                         std::vector<double>>& outBuf,
                std::map<std::tuple<int64_t,int64_t,int64_t>,
                         std::vector<double>>& outCgs) {
                if (!m->isActive()) return;
                const unsigned int nCG = m->getDegOfFreedom();
                const unsigned int nUZ = m->getDegOfFreedomUnZip();
                std::vector<double> u(DOF * nCG, 0.0);
                // step 1: createVector for each dof slice
                for (unsigned int v = 0; v < DOF; v++) {
                    std::vector<double> slice;
                    int v_id = (int)v;
                    std::function<double(double,double,double)> fv =
                        [&ic, v_id](double x, double y, double z) {
                            return ic(v_id, x, y, z);
                        };
                    m->createVector(slice, fv);
                    std::copy(slice.begin(), slice.end(),
                              u.begin() + v * nCG);
                }
                m->readFromGhostBegin(u.data(), DOF);
                m->readFromGhostEnd(u.data(), DOF);
                std::vector<double> uz(DOF * nUZ, 0.0);
                m->unzip(u.data(), uz.data(), DOF);
                // step 4: simulated stage-1 RHS. Curl-like: for each
                // interior cell, k1[v] = sum of axis differences of
                // (v+1)%6 and (v+2)%6. Reads neighbors at (i±1, j±1,
                // k±1) so it touches block-edge cells.
                std::vector<double> uz_out(DOF * nUZ, 0.0);
                const auto& blkList = m->getLocalBlockList();
                for (size_t b = 0; b < blkList.size(); b++) {
                    const auto& blk = blkList[b];
                    const unsigned int lx = blk.getAllocationSzX();
                    const unsigned int ly = blk.getAllocationSzY();
                    const unsigned int lz = blk.getAllocationSzZ();
                    const unsigned int pw = blk.get1DPadWidth();
                    const size_t off = blk.getOffset();
                    for (unsigned int k = pw; k < lz - pw; k++)
                    for (unsigned int j = pw; j < ly - pw; j++)
                    for (unsigned int i = pw; i < lx - pw; i++) {
                        size_t c = off + k*(ly*lx) + j*lx + i;
                        for (unsigned int v = 0; v < DOF; v++) {
                            unsigned int va = (v + 1) % DOF;
                            unsigned int vb = (v + 2) % DOF;
                            double dxa = uz[va*nUZ + c + 1]
                                       - uz[va*nUZ + c - 1];
                            double dyb = uz[vb*nUZ + c + lx]
                                       - uz[vb*nUZ + c - lx];
                            double dzc = uz[v*nUZ + c + lx*ly]
                                       - uz[v*nUZ + c - lx*ly];
                            uz_out[v*nUZ + c] = 0.5 * (dxa - dyb)
                                              + 0.25 * dzc;
                        }
                    }
                }
                // step 5: zip → k1 (per-dof; zip API has no dof param)
                std::vector<double> k1(DOF * nCG, 0.0);
                for (unsigned int v = 0; v < DOF; v++)
                    m->zip(uz_out.data() + v * nUZ, k1.data() + v * nCG);
                // step 6: axpy u_stage2 = u + (1/5) * k1
                std::vector<double> u2(DOF * nCG, 0.0);
                const double h = 0.2;
                const unsigned int nLB = m->getNodeLocalBegin();
                const unsigned int nLE = m->getNodeLocalEnd();
                for (unsigned int v = 0; v < DOF; v++)
                    for (unsigned int cg = nLB; cg < nLE; cg++)
                        u2[v*nCG + cg] = u[v*nCG + cg]
                                       + h * k1[v*nCG + cg];
                // step 7: ghost-exchange u_stage2 (THE critical sync,
                // multi-dof to mirror EM4's Ctx::unzip path)
                m->readFromGhostBegin(u2.data(), DOF);
                m->readFromGhostEnd(u2.data(), DOF);
                // step 8: unzip
                std::vector<double> uz2(DOF * nUZ, 0.0);
                m->unzip(u2.data(), uz2.data(), DOF);

                // gather block-buffer values at each (phys_pos, dof)
                struct E { double x, y, z; uint8_t v; double val; };
                std::vector<E> mine;
                for (size_t b = 0; b < blkList.size(); b++) {
                    const auto& blk = blkList[b];
                    const unsigned int lx = blk.getAllocationSzX();
                    const unsigned int ly = blk.getAllocationSzY();
                    const unsigned int lz = blk.getAllocationSzZ();
                    const unsigned int pw = blk.get1DPadWidth();
                    const size_t off = blk.getOffset();
                    const auto bn = blk.getBlockNode();
                    const unsigned int regLev = blk.getRegularGridLev();
                    const double hh =
                        (double)(1u << (m_uiMaxDepth - regLev)) /
                        (double)(m->getElementOrder());
                    for (unsigned int k = 0; k < lz; k++)
                    for (unsigned int j = 0; j < ly; j++)
                    for (unsigned int i = 0; i < lx; i++) {
                        double x = bn.getX() + (int(i) - int(pw)) * hh;
                        double y = bn.getY() + (int(j) - int(pw)) * hh;
                        double z = bn.getZ() + (int(k) - int(pw)) * hh;
                        size_t c = off + k*(ly*lx) + j*lx + i;
                        for (unsigned int v = 0; v < DOF; v++)
                            mine.push_back({x, y, z, (uint8_t)v,
                                            uz2[v*nUZ + c]});
                    }
                }
                int myCount = (int)mine.size();
                std::vector<int> counts(npes), offs(npes, 0);
                MPI_Allgather(&myCount, 1, MPI_INT, counts.data(),
                              1, MPI_INT, comm);
                int total = 0;
                for (int p = 0; p < npes; p++) {
                    offs[p] = total;
                    total += counts[p];
                }
                std::vector<double> aX(total), aY(total), aZ(total),
                    aV(total);
                std::vector<uint8_t> aD(total);
                std::vector<double> mX(myCount), mY(myCount), mZ(myCount),
                    mV(myCount);
                std::vector<uint8_t> mD(myCount);
                for (int i = 0; i < myCount; i++) {
                    mX[i] = mine[i].x; mY[i] = mine[i].y;
                    mZ[i] = mine[i].z; mV[i] = mine[i].val;
                    mD[i] = mine[i].v;
                }
                MPI_Allgatherv(mX.data(), myCount, MPI_DOUBLE, aX.data(),
                               counts.data(), offs.data(), MPI_DOUBLE,
                               comm);
                MPI_Allgatherv(mY.data(), myCount, MPI_DOUBLE, aY.data(),
                               counts.data(), offs.data(), MPI_DOUBLE,
                               comm);
                MPI_Allgatherv(mZ.data(), myCount, MPI_DOUBLE, aZ.data(),
                               counts.data(), offs.data(), MPI_DOUBLE,
                               comm);
                MPI_Allgatherv(mV.data(), myCount, MPI_DOUBLE, aV.data(),
                               counts.data(), offs.data(), MPI_DOUBLE,
                               comm);
                MPI_Allgatherv(mD.data(), myCount, MPI_BYTE, aD.data(),
                               counts.data(), offs.data(), MPI_BYTE,
                               comm);
                const double SCALE = 1e6;
                for (int i = 0; i < total; i++) {
                    auto k = std::make_tuple(
                        (int64_t)std::round(aX[i] * SCALE * 16
                                          + (int)aD[i]),
                        (int64_t)std::round(aY[i] * SCALE),
                        (int64_t)std::round(aZ[i] * SCALE));
                    outBuf[k].push_back(aV[i]);
                }

                // Also gather CG values per phys_pos (for diagnostics).
                const auto& aE = m->getAllElements();
                const unsigned int eOrd = m->getElementOrder();
                const unsigned int nPe = m->getNumNodesPerElement();
                std::vector<double> nodalVal(DOF * nPe);
                struct G { double x, y, z; uint8_t v; double val; };
                std::vector<G> mineG;
                for (unsigned int e = m->getElementLocalBegin();
                     e < m->getElementLocalEnd(); e++) {
                    for (unsigned int dv = 0; dv < DOF; dv++) {
                        m->getElementNodalValues(u2.data() + dv * nCG,
                                                  nodalVal.data() + dv * nPe,
                                                  e, false);
                    }
                    const unsigned int sz =
                        1u << (m_uiMaxDepth - aE[e].getLevel());
                    for (unsigned int kk = 0; kk <= eOrd; kk++)
                    for (unsigned int jj = 0; jj <= eOrd; jj++)
                    for (unsigned int ii = 0; ii <= eOrd; ii++) {
                        double x = aE[e].getX() + ii * (double)sz / eOrd;
                        double y = aE[e].getY() + jj * (double)sz / eOrd;
                        double z = aE[e].getZ() + kk * (double)sz / eOrd;
                        size_t s = kk*(eOrd+1)*(eOrd+1) + jj*(eOrd+1) + ii;
                        for (unsigned int dv = 0; dv < DOF; dv++)
                            mineG.push_back({x, y, z, (uint8_t)dv,
                                             nodalVal[dv * nPe + s]});
                    }
                }
                int myCG = (int)mineG.size();
                std::vector<int> cCG(npes), oCG(npes, 0);
                MPI_Allgather(&myCG, 1, MPI_INT, cCG.data(), 1, MPI_INT,
                              comm);
                int totCG = 0;
                for (int p = 0; p < npes; p++) {
                    oCG[p] = totCG; totCG += cCG[p];
                }
                std::vector<double> aXg(totCG), aYg(totCG), aZg(totCG),
                    aVg(totCG);
                std::vector<uint8_t> aDg(totCG);
                std::vector<double> mXg(myCG), mYg(myCG), mZg(myCG),
                    mVg(myCG);
                std::vector<uint8_t> mDg(myCG);
                for (int i = 0; i < myCG; i++) {
                    mXg[i] = mineG[i].x; mYg[i] = mineG[i].y;
                    mZg[i] = mineG[i].z; mVg[i] = mineG[i].val;
                    mDg[i] = mineG[i].v;
                }
                MPI_Allgatherv(mXg.data(), myCG, MPI_DOUBLE, aXg.data(),
                               cCG.data(), oCG.data(), MPI_DOUBLE, comm);
                MPI_Allgatherv(mYg.data(), myCG, MPI_DOUBLE, aYg.data(),
                               cCG.data(), oCG.data(), MPI_DOUBLE, comm);
                MPI_Allgatherv(mZg.data(), myCG, MPI_DOUBLE, aZg.data(),
                               cCG.data(), oCG.data(), MPI_DOUBLE, comm);
                MPI_Allgatherv(mVg.data(), myCG, MPI_DOUBLE, aVg.data(),
                               cCG.data(), oCG.data(), MPI_DOUBLE, comm);
                MPI_Allgatherv(mDg.data(), myCG, MPI_BYTE, aDg.data(),
                               cCG.data(), oCG.data(), MPI_BYTE, comm);
                for (int i = 0; i < totCG; i++) {
                    auto k = std::make_tuple(
                        (int64_t)std::round(aXg[i] * SCALE * 16
                                          + (int)aDg[i]),
                        (int64_t)std::round(aYg[i] * SCALE),
                        (int64_t)std::round(aZg[i] * SCALE));
                    outCgs[k].push_back(aVg[i]);
                }
            };

            std::map<std::tuple<int64_t,int64_t,int64_t>,
                     std::vector<double>>
                sfcBuf, twinBuf, sfcCgs, twinCgs;
            runStageCycle(mesh, sfcBuf, sfcCgs);
            runStageCycle(gTwin, twinBuf, twinCgs);
            // Compare SFC vs twin block-buffer values
            int matched = 0, mism = 0;
            double maxD = 0;
            std::tuple<int64_t,int64_t,int64_t> worstKey;
            for (auto& kv : twinBuf) {
                auto it = sfcBuf.find(kv.first);
                if (it == sfcBuf.end()) continue;
                matched++;
                double sV = it->second[0];
                double gV = kv.second[0];
                double d = std::abs(sV - gV);
                if (d > maxD) { maxD = d; worstKey = kv.first; }
                if (d > 1e-12) mism++;
            }
            // Compare SFC vs twin element-nodal CG values (multi-element)
            int cgMatched = 0, cgMism = 0;
            double cgMaxD = 0;
            int twinCgIntra = 0;
            for (auto& kv : twinCgs) {
                for (size_t i = 1; i < kv.second.size(); i++) {
                    if (std::abs(kv.second[i] - kv.second[0]) > 1e-12)
                        twinCgIntra++;
                }
                auto it = sfcCgs.find(kv.first);
                if (it == sfcCgs.end()) continue;
                cgMatched++;
                double d = std::abs(kv.second[0] - it->second[0]);
                if (d > cgMaxD) cgMaxD = d;
                if (d > 1e-12) cgMism++;
            }
            if (!rank) {
                std::cout << (maxD > 1e-12 || cgMaxD > 1e-12
                              || twinCgIntra > 0 ? RED : GRN)
                    << "TEST 8bw (RK45-stage dof=6 unzip SFC vs graph):"
                    << " bufMatched=" << matched
                    << " bufMism=" << mism
                    << " bufMaxDiff=" << maxD
                    << " | cgMatched=" << cgMatched
                    << " cgMism=" << cgMism
                    << " cgMaxDiff=" << cgMaxD
                    << " twinCgIntra=" << twinCgIntra
                    << NRM << std::endl;
                if (maxD > 1e-12) {
                    auto k = worstKey;
                    int64_t encX = std::get<0>(k);
                    int v_id = (int)(encX % 16);
                    double x = (double)((encX - v_id) / 16) / 1e6;
                    double y = (double)std::get<1>(k) / 1e6;
                    double z = (double)std::get<2>(k) / 1e6;
                    std::cout << "  worst @ (" << x << "," << y
                              << "," << z << ") dof=" << v_id
                              << std::endl;
                }
            }
        }
        delete gTwin;
    }

    // ---- TEST 9: derivative correctness on unzipped blocks ----
    // Applies axis (d/dx, d/dy, d/dz), pure-2nd (d2/dx2, ...), and
    // mixed-2nd (d2/dxdy, d2/dydz, d2/dxdz) finite-difference stencils
    // at every interior point of every block and compares to the
    // analytic derivative. Mixed derivatives hit block EDGE padding
    // (2 coords in padding), which is what bssn-style physics reads.
    //
    // stencils (4th-order central, reach +/-2, requires pw >= 2):
    //   d/dx   : (1, -8, 0, 8, -1) / (12h)
    //   d2/dx2 : (-1, 16, -30, 16, -1) / (12 h^2)
    //   d2/dxdy : composition c[a]*c[b] / (12h)^2 over 5x5 tensor
    auto testDerivatives = [&](ot::Mesh* testMesh, const char* label) {
        if (!testMesh || !testMesh->isActive()) return;
        const unsigned int eOrd = testMesh->getElementOrder();
        if (eOrd < 4) {
            if (!rank)
                std::cout << YLW << label << ": SKIPPED (eleOrder=" << eOrd
                          << " < 4)" << NRM << std::endl;
            return;
        }

        // use a quadratic polynomial so 4th-order stencils are exact
        // to machine precision — any residual error is a padding /
        // ghost-exchange bug rather than truncation noise.
        //   f = x^2 + y^2 + z^2 + x*y + y*z + x*z  (physical coords)
        // analytic:
        //   df/dx = 2x + y + z    df/dy = 2y + x + z    df/dz = 2z + x + y
        //   d2f/dx2 = d2f/dy2 = d2f/dz2 = 2
        //   d2f/dxdy = d2f/dydz = d2f/dxdz = 1
        std::function<double(double, double, double)> test_func =
            [d_min, d_max](const double xg, const double yg, const double zg) {
                double xp = (xg / (double)(1u << m_uiMaxDepth)) *
                                (d_max - d_min) +
                            d_min;
                double yp = (yg / (double)(1u << m_uiMaxDepth)) *
                                (d_max - d_min) +
                            d_min;
                double zp = (zg / (double)(1u << m_uiMaxDepth)) *
                                (d_max - d_min) +
                            d_min;
                return xp * xp + yp * yp + zp * zp + xp * yp + yp * zp +
                       xp * zp;
            };

        std::vector<double> zVec;
        testMesh->createVector(zVec, test_func);
        testMesh->performGhostExchange(zVec);
        double* uzVec = testMesh->createUnZippedVector<double>(0.0);
        testMesh->unzip(zVec.data(), uzVec);

        const auto& blkList = testMesh->getLocalBlockList();

        const double domMax        = (double)(1u << m_uiMaxDepth);
        const double phys_per_grid = (d_max - d_min) / domMax;

        const int NUM_D = 9;
        const char* names[NUM_D] = {"dx",  "dy",  "dz",  "dxx",  "dyy",
                                    "dzz", "dxdy","dydz","dxdz"};
        int checked[NUM_D] = {0};
        int errs[NUM_D]    = {0};
        int large[NUM_D]   = {0};
        double maxE[NUM_D] = {0};

        const double tol       = 1e-2;   // 4th-order trunc err << this for smooth f
        const double large_tol = 0.5;

        const double c1[5] = {1.0, -8.0, 0.0, 8.0, -1.0};
        const double c2[5] = {-1.0, 16.0, -30.0, 16.0, -1.0};

        for (unsigned int b = 0; b < blkList.size(); b++) {
            const ot::Block& blk   = blkList[b];
            const unsigned int pw  = blk.get1DPadWidth();
            if (pw < 2) continue;  // need reach 2
            const unsigned int lx   = blk.getAllocationSzX();
            const unsigned int ly   = blk.getAllocationSzY();
            const DendroIntL offset = blk.getOffset();
            const ot::TreeNode bn   = blk.getBlockNode();
            const unsigned int rL   = blk.getRegularGridLev();
            const unsigned int n1D  = blk.getElemSz1D();

            const double hx_grid = (1u << (m_uiMaxDepth - rL)) / (double)eOrd;
            const double h       = hx_grid * phys_per_grid;   // physical stencil h
            const double xmin    = bn.getX() - pw * hx_grid;
            const double ymin    = bn.getY() - pw * hx_grid;
            const double zmin    = bn.getZ() - pw * hx_grid;

            const unsigned int iLo = pw;
            const unsigned int iHi = pw + n1D * eOrd;  // inclusive upper

            for (unsigned int k = iLo; k <= iHi; k++) {
                for (unsigned int j = iLo; j <= iHi; j++) {
                    for (unsigned int i = iLo; i <= iHi; i++) {
                        // skip domain boundary cells (stencil would read
                        // outside domain padding that doesn't exist)
                        double xg = xmin + i * hx_grid;
                        double yg = ymin + j * hx_grid;
                        double zg = zmin + k * hx_grid;
                        if (xg < 2 * hx_grid || yg < 2 * hx_grid ||
                            zg < 2 * hx_grid)
                            continue;
                        if (xg > domMax - 2 * hx_grid ||
                            yg > domMax - 2 * hx_grid ||
                            zg > domMax - 2 * hx_grid)
                            continue;

                        double xp = (xg / domMax) * (d_max - d_min) + d_min;
                        double yp = (yg / domMax) * (d_max - d_min) + d_min;
                        double zp = (zg / domMax) * (d_max - d_min) + d_min;

                        auto IDX = [&](int ii, int jj, int kk) {
                            return offset + ii + lx * (jj + ly * kk);
                        };

                        // axis stencils
                        double dx = (c1[0] * uzVec[IDX(i - 2, j, k)] +
                                     c1[1] * uzVec[IDX(i - 1, j, k)] +
                                     c1[3] * uzVec[IDX(i + 1, j, k)] +
                                     c1[4] * uzVec[IDX(i + 2, j, k)]) /
                                    (12.0 * h);
                        double dy = (c1[0] * uzVec[IDX(i, j - 2, k)] +
                                     c1[1] * uzVec[IDX(i, j - 1, k)] +
                                     c1[3] * uzVec[IDX(i, j + 1, k)] +
                                     c1[4] * uzVec[IDX(i, j + 2, k)]) /
                                    (12.0 * h);
                        double dz = (c1[0] * uzVec[IDX(i, j, k - 2)] +
                                     c1[1] * uzVec[IDX(i, j, k - 1)] +
                                     c1[3] * uzVec[IDX(i, j, k + 1)] +
                                     c1[4] * uzVec[IDX(i, j, k + 2)]) /
                                    (12.0 * h);

                        // pure 2nd derivs
                        double dxx = (c2[0] * uzVec[IDX(i - 2, j, k)] +
                                      c2[1] * uzVec[IDX(i - 1, j, k)] +
                                      c2[2] * uzVec[IDX(i, j, k)] +
                                      c2[3] * uzVec[IDX(i + 1, j, k)] +
                                      c2[4] * uzVec[IDX(i + 2, j, k)]) /
                                     (12.0 * h * h);
                        double dyy = (c2[0] * uzVec[IDX(i, j - 2, k)] +
                                      c2[1] * uzVec[IDX(i, j - 1, k)] +
                                      c2[2] * uzVec[IDX(i, j, k)] +
                                      c2[3] * uzVec[IDX(i, j + 1, k)] +
                                      c2[4] * uzVec[IDX(i, j + 2, k)]) /
                                     (12.0 * h * h);
                        double dzz = (c2[0] * uzVec[IDX(i, j, k - 2)] +
                                      c2[1] * uzVec[IDX(i, j, k - 1)] +
                                      c2[2] * uzVec[IDX(i, j, k)] +
                                      c2[3] * uzVec[IDX(i, j, k + 1)] +
                                      c2[4] * uzVec[IDX(i, j, k + 2)]) /
                                     (12.0 * h * h);

                        // mixed 2nd derivs: 5x5 tensor product of 1st-deriv
                        // coeffs reads edge cells (2 coords in padding)
                        double dxdy = 0, dydz = 0, dxdz = 0;
                        for (int a = 0; a < 5; a++) {
                            for (int bb = 0; bb < 5; bb++) {
                                int da = a - 2, db = bb - 2;
                                dxdy += c1[a] * c1[bb] *
                                        uzVec[IDX(i + da, j + db, k)];
                                dydz += c1[a] * c1[bb] *
                                        uzVec[IDX(i, j + da, k + db)];
                                dxdz += c1[a] * c1[bb] *
                                        uzVec[IDX(i + da, j, k + db)];
                            }
                        }
                        dxdy /= (144.0 * h * h);
                        dydz /= (144.0 * h * h);
                        dxdz /= (144.0 * h * h);

                        // analytic (physical-coord) derivatives of
                        //   f = x^2 + y^2 + z^2 + x*y + y*z + x*z
                        double aDx   = 2 * xp + yp + zp;
                        double aDy   = 2 * yp + xp + zp;
                        double aDz   = 2 * zp + xp + yp;
                        double aDxx  = 2.0;
                        double aDyy  = 2.0;
                        double aDzz  = 2.0;
                        double aDxy  = 1.0;
                        double aDyz  = 1.0;
                        double aDxz  = 1.0;

                        double got[NUM_D] = {dx,   dy,   dz,   dxx, dyy,
                                             dzz,  dxdy, dydz, dxdz};
                        double exp_[NUM_D] = {aDx,  aDy,  aDz,  aDxx, aDyy,
                                              aDzz, aDxy, aDyz, aDxz};
                        for (int t = 0; t < NUM_D; t++) {
                            double e = std::abs(got[t] - exp_[t]);
                            checked[t]++;
                            if (e > tol) {
                                errs[t]++;
                                if (e > large_tol) large[t]++;
                            }
                            if (e > maxE[t]) maxE[t] = e;
                        }
                    }
                }
            }
        }

        int gChecked[NUM_D], gErrs[NUM_D], gLarge[NUM_D];
        double gMaxE[NUM_D];
        MPI_Allreduce(checked, gChecked, NUM_D, MPI_INT, MPI_SUM, comm);
        MPI_Allreduce(errs, gErrs, NUM_D, MPI_INT, MPI_SUM, comm);
        MPI_Allreduce(large, gLarge, NUM_D, MPI_INT, MPI_SUM, comm);
        MPI_Allreduce(maxE, gMaxE, NUM_D, MPI_DOUBLE, MPI_MAX, comm);

        if (!rank) {
            bool anyLarge = false;
            for (int t = 0; t < NUM_D; t++)
                if (gLarge[t] > 0) anyLarge = true;
            std::cout << (anyLarge ? RED : GRN) << label << ":" << NRM;
            for (int t = 0; t < NUM_D; t++) {
                std::cout << " " << names[t] << "=" << gErrs[t] << "/"
                          << gChecked[t] << "[L=" << gLarge[t]
                          << ",m=" << std::scientific << gMaxE[t]
                          << std::defaultfloat << "]";
            }
            std::cout << std::endl;
        }

        delete[] uzVec;
    };

    testDerivatives(mesh, "TEST 9a (SFC baseline)       ");
    testDerivatives(mesh_repartitioned, "TEST 9b (repartitioned)      ");
    if (remeshed) {
        testDerivatives(remeshed, "TEST 9c (after remesh)       ");
        delete remeshed;
    }

    // ---- TEST 10: AMR-cycle value preservation over N remesh cycles ----
    // Carry a smooth low-degree polynomial through N graph-mode AMR cycles
    // and verify the transferred values still match the analytical f(phys)
    // at every local cg. orphan-fill / canonical-cg regressions show up as
    // analytical-divergence that accumulates across cycles — single-cycle
    // tests (8be / 8 / 9c) miss this because the bug only manifests when
    // stale values get carried forward.
    //
    // IMPORTANT: the standard interGridTransfer builds its recv plan from
    // the destination mesh's SFC splitter keys, so it CANNOT target a
    // graph-partitioned mesh directly (it aborts on a recv-count mismatch).
    // ReMeshRepartitioned returns a graph mesh, so pairing it with
    // interGridTransfer is invalid. Instead we mirror the solver's grid-
    // transfer "sandwich": move the graph values to an SFC twin, remesh +
    // interGridTransfer entirely in SFC space (both valid there), then
    // redistribute the result back onto a fresh graph mesh of the new
    // octree. redistributeVec moves values between two meshes that share an
    // octree but differ in partition (validated by TEST 2b).
    //
    // refinement flags are derived from each element's TN (not its local
    // index), so the decision is partition-invariant on every rank.
    // a degree-1 polynomial is preserved exactly by intergrid p2c+c2p, so
    // any error > ~1e-10 is a real bug.
    if (partitionOption != PartitioningOptions::NoPartition &&
        partitionOption != PartitioningOptions::OriginalPartition) {
        std::function<double(double, double, double)> fAmr =
            [d_min, d_max](double x, double y, double z) {
                double xx = (x / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;
                double yy = (y / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;
                double zz = (z / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;
                return 1.0 + 0.5 * xx + 0.25 * yy + 0.125 * zz;
            };

        // helper: gather a mesh's LOCAL octree into a flat vector.
        auto localOct = [](ot::Mesh* m) {
            std::vector<ot::TreeNode> oct;
            if (m->isActive()) {
                const auto* pN = m->getAllElements().data();
                for (unsigned int e = m->getElementLocalBegin();
                     e < m->getElementLocalEnd(); e++)
                    oct.push_back(pN[e]);
            }
            return oct;
        };
        // helper: build a graph-partitioned mesh from an octree.
        auto makeGraphMesh = [&](std::vector<ot::TreeNode>& oct) {
            ot::Mesh* g = ot::createMesh(oct.data(), oct.size(), eOrder, comm, 1,
                                         ot::SM_TYPE::FDM, DENDRO_GRAIN_SZ,
                                         LOAD_IMB_TOL, SPLIT_FIX);
            g->setDomainBounds(pt_min, pt_max);
            g->setPartitioningMethod(partitionOption);
            g->repartitionMeshGlobal();
            return g;
        };
        // helper: TN-based refinement flags for a mesh (partition-invariant).
        auto refineFlags = [&](ot::Mesh* m, unsigned int k) {
            std::vector<unsigned int> flags;
            if (m->isActive()) {
                const unsigned int numLocal = m->getNumLocalMeshElements();
                flags.assign(numLocal, OCT_NO_CHANGE);
                const auto* pNR        = m->getAllElements().data();
                const unsigned int lbR = m->getElementLocalBegin();
                const unsigned int domMax = 1u << m_uiMaxDepth;
                const unsigned int xCut =
                    (k & 1u) ? (domMax * 3u) / 4u : (domMax) / 4u;
                for (unsigned int e = 0; e < numLocal; e++) {
                    const auto& tn = pNR[lbR + e];
                    if (tn.getLevel() >= m_uiMaxDepth) continue;
                    if ((k & 1u) ? tn.getX() >= xCut : tn.getX() < xCut)
                        flags[e] = OCT_SPLIT;
                }
            }
            return flags;
        };

        std::vector<ot::TreeNode> octCyc = localOct(mesh);
        ot::Mesh* mCyc = makeGraphMesh(octCyc);  // graph mesh

        std::vector<double> vCyc;
        mCyc->createVector(vCyc, fAmr);
        mCyc->performGhostExchange(vCyc);

        const unsigned int N_CYCLES   = 4;
        const double cycTol           = 1e-10;
        int anyFailGlobal             = 0;
        double maxErrGlobalOverCycles = 0;

        for (unsigned int k = 0; k < N_CYCLES; k++) {
            // --- sandwich step 1: graph mCyc -> SFC twin (same octree) ---
            // blockSetup=false + flag re-arm mirrors the solvers'
            // buildSFCTwin (transient twin never unzips; ReMesh successor
            // still inherits block setup).
            std::vector<ot::TreeNode> octCur = localOct(mCyc);
            ot::Mesh* sfcCur = ot::createMesh(
                octCur.data(), octCur.size(), eOrder, comm, 1,
                ot::SM_TYPE::FDM, DENDRO_GRAIN_SZ, LOAD_IMB_TOL, SPLIT_FIX,
                /*getWeight*/ NULL, /*blockSetup*/ false);
            sfcCur->setDomainBounds(pt_min, pt_max);
            sfcCur->setBlockSetupFlag(true);
            std::vector<double> vSfc;
            sfcCur->createVector(vSfc, (double)0);
            mCyc->redistributeVec(sfcCur, vCyc.data(), vSfc.data());
            sfcCur->performGhostExchange(vSfc);

            // --- step 2: remesh + intergrid transfer entirely in SFC ---
            std::vector<unsigned int> flags = refineFlags(sfcCur, k);
            sfcCur->setMeshRefinementFlags(flags);
            ot::Mesh* sfcNext =
                sfcCur->ReMesh(DENDRO_GRAIN_SZ, LOAD_IMB_TOL, SPLIT_FIX);
            if (sfcNext == nullptr) {
                if (!rank)
                    std::cout << YLW << "TEST 10 cycle " << k
                              << ": ReMesh returned nullptr (no change), "
                                 "skipping"
                              << NRM << std::endl;
                delete sfcCur;
                continue;
            }
            sfcCur->interGridTransfer(vSfc, sfcNext);  // valid: SFC -> SFC
            sfcNext->performGhostExchange(vSfc);

            // --- step 3: SFC result -> fresh graph mesh of the new octree ---
            std::vector<ot::TreeNode> octNext = localOct(sfcNext);
            ot::Mesh* mNext = makeGraphMesh(octNext);
            std::vector<double> vNext;
            mNext->createVector(vNext, (double)0);
            sfcNext->redistributeVec(mNext, vSfc.data(), vNext.data());
            mNext->performGhostExchange(vNext);

            // --- verify against the analytical field on the graph mesh ---
            int localFail = 0;
            int localChk  = 0;
            double localMaxErr = 0;
            if (mNext->isActive()) {
                const unsigned int npe   = mNext->getNumNodesPerElement();
                const unsigned int eOrdN = mNext->getElementOrder();
                const auto* pNN = mNext->getAllElements().data();
                const auto& cg2dgN = mNext->getCG2DGMap();
                for (unsigned int cg = mNext->getNodeLocalBegin();
                     cg < mNext->getNodeLocalEnd(); cg++) {
                    if (cg >= cg2dgN.size()) continue;
                    unsigned int dg = cg2dgN[cg];
                    if (dg == LOOK_UP_TABLE_DEFAULT) continue;
                    unsigned int e = dg / npe;
                    unsigned int n = dg % npe;
                    if (e >= mNext->getAllElements().size()) continue;
                    const auto& tn = pNN[e];
                    const unsigned int ni = n % (eOrdN + 1);
                    const unsigned int nj = (n / (eOrdN + 1)) % (eOrdN + 1);
                    const unsigned int nk = n / ((eOrdN + 1) * (eOrdN + 1));
                    const unsigned long long len =
                        (unsigned long long)1u
                        << (m_uiMaxDepth - tn.getLevel());
                    const double gx = (double)tn.getX()
                                      + (double)ni * len / (double)eOrdN;
                    const double gy = (double)tn.getY()
                                      + (double)nj * len / (double)eOrdN;
                    const double gz = (double)tn.getZ()
                                      + (double)nk * len / (double)eOrdN;
                    const double exp = fAmr(gx, gy, gz);
                    const double err = std::abs(vNext[cg] - exp);
                    if (err > cycTol) localFail++;
                    if (err > localMaxErr) localMaxErr = err;
                    localChk++;
                }
            }
            int gFail = 0, gChk = 0;
            double gMaxErr = 0;
            MPI_Allreduce(&localFail, &gFail, 1, MPI_INT, MPI_SUM, comm);
            MPI_Allreduce(&localChk, &gChk, 1, MPI_INT, MPI_SUM, comm);
            MPI_Allreduce(&localMaxErr, &gMaxErr, 1, MPI_DOUBLE, MPI_MAX, comm);
            if (gMaxErr > maxErrGlobalOverCycles)
                maxErrGlobalOverCycles = gMaxErr;
            if (gFail > 0) anyFailGlobal++;
            if (!rank) {
                std::cout << (gFail > 0 ? RED : GRN) << "TEST 10 cycle " << k
                          << ": cg=" << gChk << " fail=" << gFail
                          << " maxErr=" << std::scientific << gMaxErr
                          << std::defaultfloat << NRM << std::endl;
            }

            delete sfcCur;
            delete sfcNext;
            delete mCyc;
            mCyc = mNext;
            vCyc = std::move(vNext);
        }

        if (!rank) {
            std::cout << (anyFailGlobal > 0 ? RED : GRN)
                      << "TEST 10 (AMR-cycle bit-identity, "
                      << N_CYCLES << " cycles)";
            if (anyFailGlobal > 0)
                std::cout << " FAILED in " << anyFailGlobal << " cycle(s)";
            else
                std::cout << " PASSED";
            std::cout << " worst maxErr=" << std::scientific
                      << maxErrGlobalOverCycles << std::defaultfloat
                      << NRM << std::endl;
        }

        delete mCyc;
    } else if (!rank) {
        std::cout << YLW
                  << "TEST 10 (AMR-cycle): SKIPPED for non-graph partition"
                  << NRM << std::endl;
    }

    // END CLEANUP
    delete mesh;
    delete mesh_repartitioned;

    MPI_Barrier(comm);

    if (!rank) {
        std::cout << "---------------------------------------" << std::endl;
        std::cout << "               FINISHED                " << std::endl;
        std::cout << "---------------------------------------" << std::endl;
    }

    MPI_Finalize();
}
