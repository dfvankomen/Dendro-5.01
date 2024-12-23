#include "derivativeCtx.hpp"

#include <ostream>

namespace derivtest {

double DERIV_TEST_GRID_MIN_X          = -4.0;
double DERIV_TEST_GRID_MAX_X          = 4.0;
double DERIV_TEST_GRID_MIN_Y          = -4.0;
double DERIV_TEST_GRID_MAX_Y          = 4.0;
double DERIV_TEST_GRID_MIN_Z          = -5.5;
double DERIV_TEST_GRID_MAX_Z          = 5.5;
unsigned int DERIV_TEST_ELE_ORDER     = 6;
unsigned int DERIV_TEST_PADDING_WIDTH = DERIV_TEST_ELE_ORDER / 2;
double DERIV_TEST_RK45_TIME_STEP_SIZE = 0.0;
unsigned int DERIV_TEST_MAXDEPTH      = 7;
double DERIV_TEST_COMPD_MIN[3]  = {DERIV_TEST_GRID_MIN_X, DERIV_TEST_GRID_MIN_Y,
                                   DERIV_TEST_GRID_MIN_Z};
double DERIV_TEST_COMPD_MAX[3]  = {DERIV_TEST_GRID_MAX_X, DERIV_TEST_GRID_MAX_Y,
                                   DERIV_TEST_GRID_MAX_Z};
double DERIV_TEST_OCTREE_MIN[3] = {0.0, 0.0, 0.0};
double DERIV_TEST_OCTREE_MAX[3] = {(double)(1u << DERIV_TEST_MAXDEPTH),
                                   (double)(1u << DERIV_TEST_MAXDEPTH),
                                   (double)(1u << DERIV_TEST_MAXDEPTH)};
unsigned int DERIV_TEST_ID_TYPE = 0;
bool DERIV_TEST_ENABLE_BLOCK_ADAPTIVITY            = false;

std::string DERIV_TEST_DERIVTYPE_FIRST             = "EXPERIMENTAL";
std::string DERIV_TEST_DERIVTYPE_SECOND            = "JTT6";
std::vector<double> DERIV_TEST_DERIV_FIRST_COEFFS  = {};
std::vector<double> DERIV_TEST_DERIV_SECOND_COEFFS = {};

std::string DERIV_TEST_VTU_FILE_PREFIX             = "derivTest";
bool DERIV_TEST_VTU_Z_SLICE_ONLY                   = true;

std::unique_ptr<dendroderivs::DendroDerivatives> DERIV_TEST_DERIVS =
    std::make_unique<dendroderivs::DendroDerivatives>(
        DERIV_TEST_DERIVTYPE_FIRST, DERIV_TEST_DERIVTYPE_SECOND,
        DERIV_TEST_ELE_ORDER, DERIV_TEST_DERIV_FIRST_COEFFS,
        DERIV_TEST_DERIV_SECOND_COEFFS);

DerivTestCtx::DerivTestCtx(ot::Mesh* pMesh) : Ctx() {
    m_uiMesh = pMesh;
    // variable allocation for evolution variables

    // "zipped" representation
    m_var[VL::CPU_EV].create_vector(m_uiMesh, ot::DVEC_TYPE::OCT_SHARED_NODES,
                                    ot::DVEC_LOC::HOST,
                                    DERIV_TEST_NUM_VARS_TRUE, true);
    m_var[VL::CPU_DERIVS].create_vector(
        m_uiMesh, ot::DVEC_TYPE::OCT_SHARED_NODES, ot::DVEC_LOC::HOST,
        DERIV_TEST_NUM_VARS, true);

    // unzipped representation
    m_var[VL::CPU_EV_UZ].create_vector(
        m_uiMesh, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::HOST,
        DERIV_TEST_NUM_VARS_TRUE, true);
    m_var[VL::CPU_DERIVS_UZ].create_vector(
        m_uiMesh, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::HOST,
        DERIV_TEST_NUM_VARS, true);

    // store the Analytic Derivative Solutions
    m_var[VL::CPU_DERIVS_CALC].create_vector(
        m_uiMesh, ot::DVEC_TYPE::OCT_SHARED_NODES, ot::DVEC_LOC::HOST,
        DERIV_TEST_NUM_VARS, true);
    m_var[VL::CPU_DERIVS_DIFF].create_vector(
        m_uiMesh, ot::DVEC_TYPE::OCT_SHARED_NODES, ot::DVEC_LOC::HOST,
        DERIV_TEST_NUM_VARS, true);

    m_uiTinfo._m_uiStep = 0;
    m_uiTinfo._m_uiT    = 0;
    // NOTE: this CTX object shouldn't be used for evolution
    m_uiTinfo._m_uiTb   = 0.0;
    m_uiTinfo._m_uiTe   = 0.0;
    m_uiTinfo._m_uiTh   = 0.0;

    m_uiElementOrder    = DERIV_TEST_ELE_ORDER;

    m_uiMinPt           = Point(DERIV_TEST_GRID_MIN_X, DERIV_TEST_GRID_MIN_Y,
                                DERIV_TEST_GRID_MIN_Z);
    m_uiMaxPt           = Point(DERIV_TEST_GRID_MAX_X, DERIV_TEST_GRID_MAX_Y,
                                DERIV_TEST_GRID_MAX_Z);

    // deallocate_deriv_workspace();
    // allocate_deriv_workspace(m_uiMesh, 1);

    ot::dealloc_mpi_ctx<DendroScalar>(m_uiMesh, m_mpi_ctx, DERIV_TEST_NUM_VARS,
                                      DERIV_TEST_ASYNC_COMM_K);
    ot::alloc_mpi_ctx<DendroScalar>(m_uiMesh, m_mpi_ctx, DERIV_TEST_NUM_VARS,
                                    DERIV_TEST_ASYNC_COMM_K);

    // set up the appropriate derivs
    // dendro_derivs::set_appropriate_derivs(NLSM_PADDING_WIDTH);

    if (!m_uiMesh->getMPIRankGlobal())
        std::cout << "Deriv Object Contains: " << DERIV_TEST_DERIVS->toString()
                  << std::endl;

    return;
}

DerivTestCtx::~DerivTestCtx() {
    for (unsigned int i = 0; i < VL::END; i++) m_var[i].destroy_vector();

    // deallocate_deriv_workspace();
    ot::dealloc_mpi_ctx<DendroScalar>(m_uiMesh, m_mpi_ctx, DERIV_TEST_NUM_VARS,
                                      DERIV_TEST_ASYNC_COMM_K);
}

int DerivTestCtx::initialize() {
    if (!DERIV_TEST_INIT_GRID_REINIT_EACH_TIME) this->init_grid();

    // go through the refinement
    bool isRefine = false;
    DendroIntL oldElements, oldElements_g;
    DendroIntL newElements, newElements_g;

    DendroIntL oldGridPoints, oldGridPoints_g;
    DendroIntL newGridPoints, newGridPoints_g;

    unsigned int iterCount         = 1;
    const unsigned int max_iter    = DERIV_TEST_INIT_GRID_ITER;
    const unsigned int rank_global = m_uiMesh->getMPIRankGlobal();
    MPI_Comm gcomm                 = m_uiMesh->getMPIGlobalCommunicator();

    DendroScalar* unzipVar[DERIV_TEST_NUM_VARS];
    unsigned int refineVarIds[DERIV_TEST_NUM_REFINE_VARS];

    for (unsigned int vIndex = 0; vIndex < DERIV_TEST_NUM_REFINE_VARS; vIndex++)
        refineVarIds[vIndex] = DERIV_TEST_REFINE_VARS[vIndex];

    double wTol = DERIV_TEST_WAVELET_TOL;
    std::function<double(double, double, double, double* hx)> waveletTolFunc =
        [](double x, double y, double z, double* hx) {
            return computeWTolDCoords(x, y, z, hx);
        };

    DVec& m_evar     = m_var[VL::CPU_EV];
    DVec& m_evar_unz = m_var[VL::CPU_EV_UZ];

    do {
        // initialize the grid at each step if desired
        if (DERIV_TEST_INIT_GRID_REINIT_EACH_TIME) this->init_grid();

        // we can allow compression at these steps
        this->unzip(m_evar, m_evar_unz, DERIV_TEST_ASYNC_COMM_K);

        m_evar_unz.to_2d(unzipVar);
        // isRefine=this->is_remesh();
        // enforce WARM refinement based on refinement initially

        // TODO: don't refine initially if AMR is off
        if (max_iter == 0)
            isRefine = false;
        else {
            if (DERIV_TEST_DO_REFINE) {
                // isRefine =
                //     dsolve::isReMeshWAMR(m_uiMesh, (const double
                //     **)unzipVar,
                //                        refineVarIds,
                //                        nlsm::NLSM_NUM_REFINE_VARS,
                //                        waveletTolFunc,
                //                        nlsm::NLSM_DENDRO_AMR_FAC);
                isRefine = m_uiMesh->isReMeshUnzip(
                    (const double**)unzipVar, refineVarIds,
                    DERIV_TEST_NUM_REFINE_VARS, waveletTolFunc,
                    DERIV_TEST_DENDRO_AMR_FAC);

            } else {
                isRefine = false;
            }
        }

        if (isRefine) {
            ot::Mesh* newMesh =
                this->remesh(DERIV_TEST_DENDRO_GRAIN_SZ,
                             DERIV_TEST_LOAD_IMB_TOL, DERIV_TEST_SPLIT_FIX);

            oldElements   = m_uiMesh->getNumLocalMeshElements();
            newElements   = newMesh->getNumLocalMeshElements();

            oldGridPoints = m_uiMesh->getNumLocalMeshNodes();
            newGridPoints = newMesh->getNumLocalMeshNodes();

            par::Mpi_Allreduce(&oldElements, &oldElements_g, 1, MPI_SUM, gcomm);
            par::Mpi_Allreduce(&newElements, &newElements_g, 1, MPI_SUM, gcomm);

            par::Mpi_Allreduce(&oldGridPoints, &oldGridPoints_g, 1, MPI_SUM,
                               m_uiMesh->getMPIGlobalCommunicator());
            par::Mpi_Allreduce(&newGridPoints, &newGridPoints_g, 1, MPI_SUM,
                               m_uiMesh->getMPIGlobalCommunicator());

            if (!rank_global) {
                std::cout << "[solverCtx] iter : " << iterCount
                          << " (Remesh triggered) ->  old mesh : "
                          << oldElements_g << " new mesh : " << newElements_g
                          << std::endl;

                std::cout << "[solverCtx] iter : " << iterCount
                          << " (Remesh triggered) ->  old mesh (zip nodes) : "
                          << oldGridPoints_g
                          << " new mesh (zip nodes) : " << newGridPoints_g
                          << std::endl;
            }

            this->grid_transfer(newMesh);

            std::swap(m_uiMesh, newMesh);
            delete newMesh;

            // then update the size of the grid, no need to recompute
            m_uiGlobalMeshElements = newElements_g;
            m_uiGlobalGridPoints   = newGridPoints_g;

#ifdef __CUDACC__
            // TODO: CUDA STUFF
#endif
        }

        iterCount += 1;

    } while (isRefine &&
             (newElements_g != oldElements_g ||
              newGridPoints_g != oldGridPoints_g) &&
             (iterCount < max_iter));

    // initialize the grid!
    this->init_grid();

    // with the grid now defined, we can allocate the workspace for
    // derivatives
    // deallocate_deriv_workspace();
    // allocate_deriv_workspace(m_uiMesh, 1);

#ifdef NLSM_USE_COMPRESSION
    this->prepareBytesVectors();
#endif

    // Now we need to make sure we sync the grid because we might have
    // increased our value
    // NOTE: this isn't necessary, it's left in for testing purposes
    m_uiMesh->readFromGhostBegin(m_var[VL::CPU_EV].get_vec_ptr(),
                                 m_var[VL::CPU_EV].get_dof());
    m_uiMesh->readFromGhostEnd(m_var[VL::CPU_EV].get_vec_ptr(),
                               m_var[VL::CPU_EV].get_dof());

    unsigned int lmin, lmax;
    m_uiMesh->computeMinMaxLevel(lmin, lmax);
    DERIV_TEST_RK45_TIME_STEP_SIZE =
        DERIV_TEST_CFL_FACTOR *
        ((DERIV_TEST_COMPD_MAX[0] - DERIV_TEST_COMPD_MIN[0]) *
         ((1u << (m_uiMaxDepth - lmax)) / ((double)DERIV_TEST_ELE_ORDER)) /
         ((double)(1u << (m_uiMaxDepth))));
    m_uiTinfo._m_uiTh = DERIV_TEST_RK45_TIME_STEP_SIZE;

    if (!m_uiMesh->getMPIRankGlobal()) {
        const DendroScalar dx_finest =
            ((DERIV_TEST_COMPD_MAX[0] - DERIV_TEST_COMPD_MIN[0]) *
             ((1u << (m_uiMaxDepth - lmax)) / ((double)DERIV_TEST_ELE_ORDER)) /
             ((double)(1u << (m_uiMaxDepth))));
        const DendroScalar dt_finest = DERIV_TEST_CFL_FACTOR * dx_finest;

        std::cout << "================= Grid Info (After init grid "
                     "convergence):========================================"
                     "======="
                     "========"
                  << std::endl;
        std::cout << "lmin: " << lmin << " lmax:" << lmax << std::endl;
        std::cout << "dx: " << dx_finest << std::endl;
        std::cout << "dt: " << dt_finest << std::endl;
        std::cout << "========================================================="
                     "======================================================"
                  << std::endl;
    }

    MPI_Barrier(m_uiMesh->getMPIGlobalCommunicator());

    return 0;
}

int DerivTestCtx::init_grid() {
    // GRID INITIALIZATION
    DVec& m_evar                = m_var[VL::CPU_EV];

    const ot::TreeNode* pNodes  = &(*(m_uiMesh->getAllElements().begin()));
    const unsigned int eleOrder = m_uiMesh->getElementOrder();
    const unsigned int* e2n_cg  = &(*(m_uiMesh->getE2NMapping().begin()));
    const unsigned int* e2n_dg  = &(*(m_uiMesh->getE2NMapping_DG().begin()));
    const unsigned int nPe      = m_uiMesh->getNumNodesPerElement();
    const unsigned int nodeLocalBegin = m_uiMesh->getNodeLocalBegin();
    const unsigned int nodeLocalEnd   = m_uiMesh->getNodeLocalEnd();

    DendroScalar* zipIn[DERIV_TEST_NUM_VARS_TRUE];
    m_evar.to_2d(zipIn);

    DendroScalar var1[DERIV_TEST_NUM_VARS_TRUE];

    for (unsigned int elem = m_uiMesh->getElementLocalBegin();
         elem < m_uiMesh->getElementLocalEnd(); elem++) {
        DendroScalar var[DERIV_TEST_NUM_VARS_TRUE];
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
                        m_uiMesh->dg2eijk(nodeLookUp_DG, ownerID, ii_x, jj_y,
                                          kk_z);

                        const DendroScalar len =
                            (double)(1u << (m_uiMaxDepth -
                                            pNodes[ownerID].getLevel()));
                        const DendroScalar x =
                            pNodes[ownerID].getX() + ii_x * (len / (eleOrder));
                        const DendroScalar y =
                            pNodes[ownerID].getY() + jj_y * (len / (eleOrder));
                        const DendroScalar z =
                            pNodes[ownerID].getZ() + kk_z * (len / (eleOrder));

                        // the data to physical coords takes in the octree
                        // coords and autoconverts them and returns the data
                        // based on things
                        initDataFuncToPhysCoords((double)x, (double)y,
                                                 (double)z, var);

                        for (unsigned int v = 0; v < DERIV_TEST_NUM_VARS_TRUE;
                             v++)
                            zipIn[v][nodeLookUp_CG] = var[v];
                    }
                }
    }

    // for (unsigned int node = m_uiMesh->getNodeLocalBegin();
    //      node < m_uiMesh->getNodeLocalEnd(); node++) {
    //     enforce_constraints(zipIn, node);
    // }

    return 0;
}

int DerivTestCtx::finalize() { return 0; }

void DerivTestCtx::doDerivCalculation() {
    if (!m_uiMesh->isActive()) return;

    // get the zipped variables
    DVec& m_evar        = m_var[VL::CPU_EV];
    DVec& m_evar_unz    = m_var[VL::CPU_EV_UZ];
    DVec& m_derivs      = m_var[VL::CPU_DERIVS];
    DVec& m_derivs_unz  = m_var[VL::CPU_DERIVS_UZ];
    DVec& m_derivs_diff = m_var[VL::CPU_DERIVS_DIFF];

    // prep the pointers
    DendroScalar* derivsUnzipVar[DERIV_TEST_NUM_VARS];
    DendroScalar* derivsVar[DERIV_TEST_NUM_VARS];

    DendroScalar* evolUnzipVar[DERIV_TEST_NUM_VARS_TRUE];
    DendroScalar* evolVar[DERIV_TEST_NUM_VARS_TRUE];

    m_evar_unz.to_2d(evolUnzipVar);
    m_derivs_unz.to_2d(derivsUnzipVar);

    m_evar.to_2d(evolVar);
    m_derivs.to_2d(derivsVar);

    if (!(m_uiMesh->getMPIRankGlobal())) {
        std::cout << BLU << "[DERIV_TEST] - Now computing DERIVS" << NRM
                  << std::endl;
    }

    // do the zip/unzip
    this->unzip(m_evar, m_evar_unz);

    const std::vector<ot::Block> blkList = m_uiMesh->getLocalBlockList();

    unsigned int offset;
    double ptmin[3], ptmax[3];
    unsigned int sz[3];
    unsigned int bflag;
    double dx, dy, dz;
    const Point pt_min(DERIV_TEST_COMPD_MIN[0], DERIV_TEST_COMPD_MIN[1],
                       DERIV_TEST_COMPD_MIN[2]);
    const Point pt_max(DERIV_TEST_COMPD_MAX[0], DERIV_TEST_COMPD_MAX[1],
                       DERIV_TEST_COMPD_MAX[2]);
    const unsigned int PW = DERIV_TEST_PADDING_WIDTH;

    for (unsigned int blk = 0; blk < blkList.size(); blk++) {
        offset   = blkList[blk].getOffset();
        sz[0]    = blkList[blk].getAllocationSzX();
        sz[1]    = blkList[blk].getAllocationSzY();
        sz[2]    = blkList[blk].getAllocationSzZ();

        bflag    = blkList[blk].getBlkNodeFlag();

        dx       = blkList[blk].computeDx(pt_min, pt_max);
        dy       = blkList[blk].computeDy(pt_min, pt_max);
        dz       = blkList[blk].computeDz(pt_min, pt_max);

        ptmin[0] = GRIDX_TO_X(blkList[blk].getBlockNode().minX()) - PW * dx;
        ptmin[1] = GRIDY_TO_Y(blkList[blk].getBlockNode().minY()) - PW * dy;
        ptmin[2] = GRIDZ_TO_Z(blkList[blk].getBlockNode().minZ()) - PW * dz;

        ptmax[0] = GRIDX_TO_X(blkList[blk].getBlockNode().maxX()) + PW * dx;
        ptmax[1] = GRIDY_TO_Y(blkList[blk].getBlockNode().maxY()) + PW * dy;
        ptmax[2] = GRIDZ_TO_Z(blkList[blk].getBlockNode().maxZ()) + PW * dz;

        const double* const U      = &evolUnzipVar[0][offset];
        double* const Udx          = &derivsUnzipVar[0][offset];
        double* const Udy          = &derivsUnzipVar[1][offset];
        double* const Udz          = &derivsUnzipVar[2][offset];
        double* const Udxx         = &derivsUnzipVar[3][offset];
        double* const Udyy         = &derivsUnzipVar[4][offset];
        double* const Udzz         = &derivsUnzipVar[5][offset];

        const unsigned int i       = 5;
        const unsigned int j       = 5;
        const unsigned int k       = 5;
        const unsigned int testidx = ((i) + sz[0] * ((j) + sz[1] * (k)));

        // calculate the derivatives
        DERIV_TEST_DERIVS->grad_x(Udx, U, dx, sz, bflag);
        DERIV_TEST_DERIVS->grad_y(Udy, U, dy, sz, bflag);
        DERIV_TEST_DERIVS->grad_z(Udz, U, dz, sz, bflag);
        DERIV_TEST_DERIVS->grad_xx(Udxx, U, dx, sz, bflag);
        DERIV_TEST_DERIVS->grad_yy(Udyy, U, dy, sz, bflag);
        DERIV_TEST_DERIVS->grad_zz(Udzz, U, dz, sz, bflag);
    }

    this->zip(m_derivs_unz, m_derivs);
    m_uiMesh->readFromGhostBegin(m_derivs.get_vec_ptr(), m_derivs.get_dof());
    m_uiMesh->readFromGhostEnd(m_derivs.get_vec_ptr(), m_derivs.get_dof());

    if (!(m_uiMesh->getMPIRankGlobal())) {
        std::cout << BLU << "[DERIV_TEST] - Finished computing DERIVATIVES!"
                  << NRM << std::endl;

        std::cout << YLW
                  << "[DERIV_TEST] - Now computing analytical derivatives..."
                  << NRM << std::endl;
    }

    // then compute the analytical deriv solution
    const ot::TreeNode* pNodes  = &(*(m_uiMesh->getAllElements().begin()));
    const unsigned int eleOrder = m_uiMesh->getElementOrder();
    const unsigned int* e2n_cg  = &(*(m_uiMesh->getE2NMapping().begin()));
    const unsigned int* e2n_dg  = &(*(m_uiMesh->getE2NMapping_DG().begin()));
    const unsigned int nPe      = m_uiMesh->getNumNodesPerElement();
    const unsigned int nodeLocalBegin = m_uiMesh->getNodeLocalBegin();
    const unsigned int nodeLocalEnd   = m_uiMesh->getNodeLocalEnd();

    DVec& m_analytic_deriv            = m_var[VL::CPU_DERIVS_CALC];
    DVec& m_analytic_deriv_diff       = m_var[VL::CPU_DERIVS_DIFF];
    DendroScalar* analytical_deriv_var[DERIV_TEST_NUM_VARS];
    DendroScalar* analytical_deriv_diff[DERIV_TEST_NUM_VARS];

    m_analytic_deriv.to_2d(analytical_deriv_var);
    m_analytic_deriv_diff.to_2d(analytical_deriv_diff);

    for (unsigned int elem = m_uiMesh->getElementLocalBegin();
         elem < m_uiMesh->getElementLocalEnd(); elem++) {
        DendroScalar var[DERIV_TEST_NUM_VARS];
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
                        m_uiMesh->dg2eijk(nodeLookUp_DG, ownerID, ii_x, jj_y,
                                          kk_z);

                        const DendroScalar len =
                            (double)(1u << (m_uiMaxDepth -
                                            pNodes[ownerID].getLevel()));
                        const DendroScalar x =
                            pNodes[ownerID].getX() + ii_x * (len / (eleOrder));
                        const DendroScalar y =
                            pNodes[ownerID].getY() + jj_y * (len / (eleOrder));
                        const DendroScalar z =
                            pNodes[ownerID].getZ() + kk_z * (len / (eleOrder));

                        analyticalDerivs((double)x, (double)y, (double)z, var);

                        for (unsigned int v = 0; v < DERIV_TEST_NUM_VARS; v++) {
                            analytical_deriv_var[v][nodeLookUp_CG] = var[v];
                            analytical_deriv_diff[v][nodeLookUp_CG] =
                                derivsVar[v][nodeLookUp_CG] - var[v];
                        }
                    }
                }
    }

    if (!(m_uiMesh->getMPIRank())) {
        std::cout << YLW
                  << "[DERIV_TEST] - Finished analytical, now syncing owners..."
                  << NRM << std::endl;
    }

    // NOTE: not sure if I need to actually do the communication here or
    // not I think BSSN did it simply because of the extraction step for
    // grav waves
    m_uiMesh->readFromGhostBegin(m_analytic_deriv.get_vec_ptr(),
                                 m_analytic_deriv.get_dof());
    m_uiMesh->readFromGhostEnd(m_analytic_deriv.get_vec_ptr(),
                               m_analytic_deriv.get_dof());
    m_uiMesh->readFromGhostBegin(m_analytic_deriv_diff.get_vec_ptr(),
                                 m_analytic_deriv_diff.get_dof());
    m_uiMesh->readFromGhostEnd(m_analytic_deriv_diff.get_vec_ptr(),
                               m_analytic_deriv_diff.get_dof());

    if (!(m_uiMesh->getMPIRank())) {
        std::cout
            << BLU
            << "[DERIV_TEST] - Success! Finished full test of derivatives!"
            << NRM << std::endl;
    }
}

int DerivTestCtx::rhs(DVec* in, DVec* out, unsigned int sz, DendroScalar time) {
    return 0;
}

void DerivTestCtx::compute_constraints() { return; }

int DerivTestCtx::rhs_blk(const DendroScalar* in, DendroScalar* out,
                          unsigned int dof, unsigned int local_blk_id,
                          DendroScalar blk_time) {
    return 0;
}

int DerivTestCtx::write_vtu() {
    if (!m_uiMesh->isActive()) return 0;

    DVec& m_evar = m_var[VL::CPU_EV];
    DendroScalar* evolVar[DERIV_TEST_NUM_VARS_TRUE];

    m_evar.to_2d(evolVar);

    // no constraints to compute

    DVec& m_derivs          = m_var[VL::CPU_DERIVS];
    DVec& m_derivs_analytic = m_var[VL::CPU_DERIVS_CALC];
    DVec& m_derivs_diff     = m_var[VL::CPU_DERIVS_DIFF];

    DendroScalar* derivsVar[DERIV_TEST_NUM_VARS];
    DendroScalar* derivsAnalyticVar[DERIV_TEST_NUM_VARS];
    DendroScalar* derivsDiffVar[DERIV_TEST_NUM_VARS];

    m_derivs.to_2d(derivsVar);
    m_derivs_analytic.to_2d(derivsAnalyticVar);
    m_derivs_diff.to_2d(derivsDiffVar);

    std::vector<std::string> pDataNames;
    const unsigned int numEvolVars  = 1;

    const unsigned int totalVTUVars = numEvolVars + 3 * DERIV_TEST_NUM_VARS;
    double* pData[totalVTUVars];

    unsigned int outputOffset = 0;

    pDataNames.push_back("U_ORIGINAL");
    pData[0] = evolVar[0];

    outputOffset += 1;

    for (unsigned int i = 0; i < DERIV_TEST_NUM_VARS; i++) {
        pDataNames.push_back(std::string(DERIV_TEST_DERIV_NAMES[i]));
        pData[outputOffset + i] = derivsVar[i];
    }
    outputOffset += DERIV_TEST_NUM_VARS;

    for (unsigned int i = 0; i < DERIV_TEST_NUM_VARS; i++) {
        pDataNames.push_back(std::string(DERIV_TEST_DERIV_NAMES[i]) + "_ANLYT");
        pData[outputOffset + i] = derivsAnalyticVar[i];
    }
    outputOffset += DERIV_TEST_NUM_VARS;

    for (unsigned int i = 0; i < DERIV_TEST_NUM_VARS; i++) {
        pDataNames.push_back(std::string(DERIV_TEST_DERIV_NAMES[i]) + "_DIFF");
        pData[outputOffset + i] = derivsDiffVar[i];
    }
    outputOffset += DERIV_TEST_NUM_VARS;

    std::vector<char*> pDataNames_char;
    pDataNames_char.reserve(pDataNames.size());

    for (unsigned int i = 0; i < pDataNames.size(); i++)
        pDataNames_char.push_back(const_cast<char*>(pDataNames[i].c_str()));

    const char* fDataNames[] = {"Time", "Cycle"};
    const double fData[]     = {m_uiTinfo._m_uiT, (double)m_uiTinfo._m_uiStep};

    char fPrefix[256];
    sprintf(fPrefix, "%s_%06d", DERIV_TEST_VTU_FILE_PREFIX.c_str(),
            m_uiTinfo._m_uiStep);

    if (DERIV_TEST_VTU_Z_SLICE_ONLY) {
        unsigned int s_val[3]  = {1u << (m_uiMaxDepth - 1),
                                  1u << (m_uiMaxDepth - 1),
                                  1u << (m_uiMaxDepth - 1)};
        unsigned int s_norm[3] = {0, 0, 1};
        io::vtk::mesh2vtu_slice(m_uiMesh, s_val, s_norm, fPrefix, 2, fDataNames,
                                fData, totalVTUVars,
                                (const char**)&pDataNames_char[0],
                                (const double**)pData);
    } else
        io::vtk::mesh2vtuFine(m_uiMesh, fPrefix, 2, fDataNames, fData,
                              totalVTUVars, (const char**)&pDataNames_char[0],
                              (const double**)pData);

    return 0;
}

int DerivTestCtx::pre_timestep(DVec sIn) { return 0; }

int DerivTestCtx::pre_stage(DVec sIn) { return 0; }

int DerivTestCtx::post_stage(DVec sIn) { return 0; }

int DerivTestCtx::post_timestep(DVec sIn) { return 0; }

bool DerivTestCtx::is_remesh() {
#ifdef __PROFILE_CTX__
    m_uiCtxpt[ts::CTXPROFILE::IS_REMESH].start();
#endif

    bool isRefine = false;
    if (DERIV_TEST_ENABLE_BLOCK_ADAPTIVITY) return isRefine;

    MPI_Comm comm    = m_uiMesh->getMPIGlobalCommunicator();

    DVec& m_evar     = m_var[VL::CPU_EV];
    DVec& m_evar_unz = m_var[VL::CPU_EV_UZ];

    this->unzip(m_evar, m_evar_unz, DERIV_TEST_ASYNC_COMM_K);

    DendroScalar* unzipVar[DERIV_TEST_NUM_VARS];
    m_evar_unz.to_2d(unzipVar);

    unsigned int refineVarIds[DERIV_TEST_NUM_REFINE_VARS];
    for (unsigned int vIndex = 0; vIndex < DERIV_TEST_NUM_REFINE_VARS; vIndex++)
        refineVarIds[vIndex] = DERIV_TEST_REFINE_VARS[vIndex];

    double wTol = DERIV_TEST_WAVELET_TOL;
    std::function<double(double, double, double, double*)> waveletTolFunc =
        [wTol](double x, double y, double z, double* hx) {
            return computeWTolDCoords(x, y, z, hx);
        };

    if (DERIV_TEST_DO_REFINE) {
        isRefine = m_uiMesh->isReMeshUnzip(
            (const double**)unzipVar, refineVarIds, DERIV_TEST_NUM_REFINE_VARS,
            waveletTolFunc, DERIV_TEST_DENDRO_AMR_FAC);
    } else {
        isRefine = false;
    }

#ifdef __PROFILE_CTX__
    m_uiCtxpt[ts::CTXPROFILE::IS_REMESH].stop();
#endif

    return isRefine;
}

DVec& DerivTestCtx::get_evolution_vars() { return m_var[CPU_EV]; }

DVec& DerivTestCtx::get_constraint_vars() { return m_var[CPU_EV]; }

int DerivTestCtx::terminal_output() {
    if (!m_uiMesh->isActive()) return 0;

    std::streamsize ss = std::cout.precision();
    std::streamsize sw = std::cout.width();
    DVec& m_evar       = m_var[VL::CPU_EV];
    DendroScalar* zippedUp[DERIV_TEST_NUM_VARS];
    m_var[VL::CPU_DERIVS_DIFF].to_2d(zippedUp);

    // update cout precision and scientific view

    std::cout << std::scientific;

    std::cout.precision(7);

    for (unsigned int i = 0; i < DERIV_TEST_NUM_VARS; i++) {
        double l_min   = vecMin(&zippedUp[i][m_uiMesh->getNodeLocalBegin()],
                                (m_uiMesh->getNumLocalMeshNodes()),
                                m_uiMesh->getMPICommunicator());
        double l_max   = vecMax(&zippedUp[i][m_uiMesh->getNodeLocalBegin()],
                                (m_uiMesh->getNumLocalMeshNodes()),
                                m_uiMesh->getMPICommunicator());
        double l2_norm = normL2(&zippedUp[i][m_uiMesh->getNodeLocalBegin()],
                                (m_uiMesh->getNumLocalMeshNodes()),
                                m_uiMesh->getMPICommunicator());

        if (!(m_uiMesh->getMPIRank())) {
            std::cout << "\t[var - diff]:  " << std::setw(12)
                      << DERIV_TEST_DERIV_NAMES[i];
            std::cout << " (min, max, l2) : \t ( " << l_min << ", " << l_max
                      << ", " << l2_norm << ") " << std::endl;
        }
    }

    // and then the difference to analytical!

    std::cout.precision(ss);
    std::cout << std::setw(sw);
    std::cout.unsetf(std::ios_base::floatfield);

    return 0;
}

unsigned int DerivTestCtx::getBlkTimestepFac(unsigned int blev,
                                             unsigned int lmin,
                                             unsigned int lmax) {
    const unsigned int ldiff = 0;
    if ((lmax - blev) <= ldiff)
        return 1;
    else {
        return 1u << (lmax - blev - ldiff);
    }
}

int DerivTestCtx::grid_transfer(const ot::Mesh* m_new) {
#ifdef __PROFILE_CTX__
    m_uiCtxpt[ts::CTXPROFILE::GRID_TRASFER].start();
#endif
    DVec& m_evar = m_var[VL::CPU_EV];
    DVec::grid_transfer(m_uiMesh, m_new, m_evar);
    // printf("igt ended\n");

    m_var[VL::CPU_EV_UZ].destroy_vector();
    m_var[VL::CPU_DERIVS_UZ].destroy_vector();

    m_var[VL::CPU_DERIVS].destroy_vector();
    m_var[VL::CPU_DERIVS_CALC].destroy_vector();
    m_var[VL::CPU_DERIVS_DIFF].destroy_vector();

    m_var[VL::CPU_DERIVS].create_vector(m_new, ot::DVEC_TYPE::OCT_SHARED_NODES,
                                        ot::DVEC_LOC::HOST, DERIV_TEST_NUM_VARS,
                                        true);

    m_var[VL::CPU_EV_UZ].create_vector(
        m_new, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::HOST,
        DERIV_TEST_NUM_VARS_TRUE, true);
    m_var[VL::CPU_DERIVS_UZ].create_vector(
        m_new, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::HOST,
        DERIV_TEST_NUM_VARS, true);

    m_var[VL::CPU_DERIVS_CALC].create_vector(
        m_new, ot::DVEC_TYPE::OCT_SHARED_NODES, ot::DVEC_LOC::HOST,
        DERIV_TEST_NUM_VARS, true);
    m_var[VL::CPU_DERIVS_DIFF].create_vector(
        m_new, ot::DVEC_TYPE::OCT_SHARED_NODES, ot::DVEC_LOC::HOST,
        DERIV_TEST_NUM_VARS, true);

    ot::dealloc_mpi_ctx<DendroScalar>(m_uiMesh, m_mpi_ctx, DERIV_TEST_NUM_VARS,
                                      DERIV_TEST_ASYNC_COMM_K);
    ot::alloc_mpi_ctx<DendroScalar>(m_new, m_mpi_ctx, DERIV_TEST_NUM_VARS,
                                    DERIV_TEST_ASYNC_COMM_K);

    m_uiIsETSSynced = false;

#ifdef __PROFILE_CTX__
    m_uiCtxpt[ts::CTXPROFILE::GRID_TRASFER].stop();
#endif

    // deallocate_deriv_workspace();
    // allocate_deriv_workspace(m_uiMesh, 1);
    return 0;
}

void DerivTestCtx::calculate_full_grid_size() {
    if (!m_uiMesh->isActive()) {
        return;
    }
    // number of mesh elements
    DendroIntL mesh_elements = m_uiMesh->getNumLocalMeshElements();

    DendroIntL grid_points   = m_uiMesh->getNumLocalMeshNodes();

    // perform an all reduce on the mesh
    par::Mpi_Reduce(&mesh_elements, &m_uiGlobalMeshElements, 1, MPI_SUM, 0,
                    m_uiMesh->getMPICommunicator());

    par::Mpi_Reduce(&grid_points, &m_uiGlobalGridPoints, 1, MPI_SUM, 0,
                    m_uiMesh->getMPICommunicator());
}

void initDataType0(const double xx, const double yy, const double zz,
                   double* var) {
    // boris initialization

    // simple initialization!
    var[0] = 0.5 * exp(-sin(2 * xx) - sin(2 * yy) - sin(2 * zz));
}

void initDataType0_AnalyticalDerivs(const double xx, const double yy,
                                    const double zz, double* var) {
    // first order analytical derivative
    var[0] = -1.0 * exp(-sin(2 * xx) - sin(2 * yy) - sin(2 * zz)) * cos(2 * xx);
    var[1] = -1.0 * exp(-sin(2 * xx) - sin(2 * yy) - sin(2 * zz)) * cos(2 * yy);
    var[2] = -1.0 * exp(-sin(2 * xx) - sin(2 * yy) - sin(2 * zz)) * cos(2 * zz);

    // second order analytical derivative
    var[3] = 2.0 * (sin(2 * xx) + pow(cos(2 * xx), 2)) *
             exp(-sin(2 * xx) - sin(2 * yy) - sin(2 * zz));
    var[4] = 2.0 * (sin(2 * yy) + pow(cos(2 * yy), 2)) *
             exp(-sin(2 * xx) - sin(2 * yy) - sin(2 * zz));
    var[5] = 2.0 * (sin(2 * zz) + pow(cos(2 * zz), 2)) *
             exp(-sin(2 * xx) - sin(2 * yy) - sin(2 * zz));
}

void initDataType1(const double xx, const double yy, const double zz,
                   double* var) {
    // sine initialization

    const double amplitude_x = 1.0;
    const double amplitude_y = 1.0;
    const double amplitude_z = 1.0;
    var[0] =
        amplitude_x * sin(xx) + amplitude_y * sin(yy) + amplitude_z * sin(zz);
}

void initDataType1_AnalyticalDerivs(const double xx, const double yy,
                                    const double zz, double* var) {
    const double amplitude_x = 1.0;
    const double amplitude_y = 1.0;
    const double amplitude_z = 1.0;
    // first order analytical derivative
    var[0]                   = amplitude_x * cos(xx);
    var[1]                   = amplitude_y * cos(yy);
    var[2]                   = amplitude_z * cos(zz);

    // second order analytical derivative
    var[3]                   = -amplitude_x * sin(xx);
    var[4]                   = -amplitude_y * sin(yy);
    var[5]                   = -amplitude_z * sin(zz);
}

void initDataType2(const double xx, const double yy, const double zz,
                   double* var) {
    // Rosenbrock function initialization

    const double freq = (1.0 / 2.0) * PI;
    var[0] =
        cos(freq * xx) * cos(freq * yy) * exp(zz) + xx * xx * yy * yy * zz * zz;
}

void initDataType2_AnalyticalDerivs(const double xx, const double yy,
                                    const double zz, double* var) {
    const double freq = (1.0 / 2.0) * PI;
    // first order analytical derivative
    var[0]            = -freq * sin(freq * xx) * cos(freq * yy) * exp(zz) +
             2 * xx * yy * yy * zz * zz;
    var[1] = -freq * cos(freq * xx) * sin(freq * yy) * exp(zz) +
             2 * xx * xx * yy * zz * zz;
    var[2] =
        cos(freq * xx) * cos(freq * yy) * exp(zz) + 2 * xx * xx * yy * yy * zz;

    // second order analytical derivative
    var[3] = -freq * freq * cos(freq * xx) * cos(freq * yy) * exp(zz) +
             2 * yy * yy * zz * zz;
    var[4] = -freq * freq * cos(freq * xx) * cos(freq * yy) * exp(zz) +
             2 * xx * xx * zz * zz;
    var[5] = cos(freq * xx) * cos(freq * yy) * exp(zz) + 2 * xx * xx * yy * yy;
}

}  // namespace derivtest
