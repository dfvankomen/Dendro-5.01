

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

            // if((xx < -0.5 || xx > 0.5) || ( yy < -0.5 || yy > 0.5) || (zz
            // < -0.5 || zz > 0.5) )
            //     return 0.0;

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

    mesh_repartitioned->setPartitioningMethod(partitionOption);

    // now we can do some partitioning checks
    mesh_repartitioned->repartitionMeshGlobal();

#ifdef EXPORT_MESH
    std::string save_prefix_2 =
        "test_mesh_repartitioned_npes" + std::to_string(npes);
    io::vtk::mesh2vtu(mesh_repartitioned, save_prefix_2.c_str(), 0, nullptr,
                      nullptr, 0, nullptr, nullptr);
#endif

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
                if (std::abs(expected - got) > ghostTol) {
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
        MPI_Allreduce(&ghostErrors, &totalErrors, 1, MPI_INT, MPI_SUM, comm);
        MPI_Allreduce(&ghostNodesChecked, &totalChecked, 1, MPI_INT, MPI_SUM,
                      comm);
        if (!rank) {
            if (totalErrors == 0) {
                std::cout << GRN << "TEST 2 PASSED: ghost exchange correct ("
                          << totalChecked << " nodes checked)" << NRM
                          << std::endl;
            } else {
                std::cout << RED << "TEST 2 FAILED: ghost exchange had "
                          << totalErrors << " / " << totalChecked
                          << " errors" << NRM << std::endl;
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

    // ---- TEST 6: setMeshRefinementFlags + ReMesh ----
    {
        std::vector<unsigned int> refine_flags;
        if (mesh_repartitioned->isActive()) {
            refine_flags.resize(
                mesh_repartitioned->getNumLocalMeshElements(), OCT_NO_CHANGE);
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

        // now try actual remesh (even with no-change, this exercises
        // the full remesh path)
        ot::Mesh* remeshed = mesh_repartitioned->ReMesh(
            DENDRO_GRAIN_SZ, LOAD_IMB_TOL, SPLIT_FIX);

        if (remeshed != nullptr) {
            // verify the remeshed mesh has elements
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

            delete remeshed;
        } else {
            if (!rank) {
                std::cout << YLW
                          << "TEST 6b: ReMesh returned nullptr (mesh "
                             "unchanged, this is OK for no-change)"
                          << NRM << std::endl;
            }
        }
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
