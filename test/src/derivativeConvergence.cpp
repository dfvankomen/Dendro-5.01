/**
 * @file derivativeScaling.cpp
 *
 * @copyright Copyright (c) 2024
 *
 */

#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>

#include "TreeNode.h"
#include "derivativeCtx.hpp"
#include "mathUtils.h"
#include "mesh.h"
#include "meshUtils.h"
#include "mpi.h"
#include "octUtils.h"

// #define DO_WEAK_SCALING_TESTS

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

int derivtest_driver(MPI_Comm comm, std::ostream& outfile) {
    int rank, npes;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);

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

    // if (bssn::BSSN_RESTORE_SOLVER == 0)
    //     pMesh = bssn::weakScalingReMesh(mesh, npes);

    // build up the mesh
    derivtest::DerivTestCtx* theCtx = new derivtest::DerivTestCtx(pMesh);
    theCtx->initialize();

    // then we'll just calculate the derivs stuff

    theCtx->prepare_derivatives();
    theCtx->doDerivCalculation();
    theCtx->terminal_output();
    theCtx->write_vtu();

    // ets->m_uiCtxpt
    // std::cout<<"reached end:"<<rank<<std::endl;

    ot::Mesh* tmp_mesh = theCtx->get_mesh();
    delete theCtx;
    delete tmp_mesh;

    return 0;
}

int main(int argc, char** argv) {
    // if (argc < 2) {
    //     std::cout << "Usage: " << argv[0] << " paramFile " << std::endl;
    //     return 0;
    // }
    //
    std::cout << "Now initializing program!" << std::endl;

    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, npes;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);

    // Print out CMAKE options
    if (!rank) {
    }

    // 1 . read the parameter file.
    // if (!rank) std::cout << " reading parameter file :" << argv[1] <<
    // std::endl; readParamFile(argv[1], comm);

    int root = std::min(1, npes - 1);
    // dumpParamFile(std::cout, root, comm);

    _InitializeHcurve(3);
    m_uiMaxDepth = derivtest::DERIV_TEST_MAXDEPTH;

    std::cout << "DERIV TYPES: " << derivtest::DERIV_TEST_DERIVTYPE_FIRST
              << " - " << derivtest::DERIV_TEST_DERIVTYPE_SECOND << std::endl;

    derivtest::DERIV_TEST_DERIVS =
        std::make_unique<dendroderivs::DendroDerivatives>(
            derivtest::DERIV_TEST_DERIVTYPE_FIRST,
            derivtest::DERIV_TEST_DERIVTYPE_SECOND,
            derivtest::DERIV_TEST_ELE_ORDER,
            derivtest::DERIV_TEST_DERIV_FIRST_COEFFS,
            derivtest::DERIV_TEST_DERIV_SECOND_COEFFS,
            derivtest::DERIV_TEST_DERIVFIRST_MATID,
            derivtest::DERIV_TEST_DERIVSECOND_MATID);

    std::cout << "DENDRO DERIVS: " << derivtest::DERIV_TEST_DERIVS->toString()
              << std::endl;

    std::ofstream outfile;
    char fname[256];
    sprintf(fname, "derivtest_testing_%d.txt", npes);

    if (!rank) {
        outfile.open(fname, std::ios_base::app);
        time_t now = time(0);
        // convert now to string form
        char* dt   = ctime(&now);
        outfile
            << "============================================================"
            << std::endl;
        outfile << "Current time : " << dt << " --- " << std::endl;
        outfile
            << "============================================================"
            << std::endl;
    }

    derivtest_driver(comm, outfile);

    if (!rank) outfile.close();

#ifdef DO_WEAK_SCALING_TESTS

    if (!rank)
        std::cout << "========================================================="
                     "============="
                  << std::endl;
    if (!rank) std::cout << "     Weak Scaling Run Begin.   " << std::endl;
    if (!rank)
        std::cout << "========================================================="
                     "============="
                  << std::endl;

    int proc_group = 0;
    int min_np     = 2;
    for (int i = npes; rank < i && i >= min_np; i = i >> 1) proc_group++;
    MPI_Comm comm_ws;

    MPI_Comm_split(comm, proc_group, rank, &comm_ws);

    MPI_Comm_rank(comm_ws, &rank);
    MPI_Comm_size(comm_ws, &npes);

    if (!rank) outfile.open(fname, std::ios_base::app);
    MPI_Barrier(comm_ws);

    derivtest_driver(comm_ws, outfile);

    MPI_Barrier(comm_ws);
    if (!rank) outfile.close();

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);
    MPI_Barrier(comm);

    if (!rank)
        std::cout << "========================================================="
                     "============="
                  << std::endl;
    if (!rank) std::cout << "     Weak Scaling Run Complete.   " << std::endl;
    if (!rank)
        std::cout << "========================================================="
                     "============="
                  << std::endl;

#endif

    MPI_Finalize();
    return 0;
}
