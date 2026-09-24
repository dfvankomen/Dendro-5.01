/**
 * @file enuts.h
 * @author Milinda Fernando (milinda@cs.utah.edu)
 * @brief Spatially adaptive non uniform time stepping framework.
 * @version 0.1
 * @date 2019-07-12
 * @copyright Copyright (c) 2019, School of Computing, University of Utah.
 *
 */
#pragma once
#include <assert.h>

#include <bitset>
#include <fstream>
#include <iomanip>
#include <functional>
#include <iostream>

#include "blkAsync.h"
#include "block.h"
#include "ets.h"
#include "mathMeshUtils.h"
#include "mesh.h"
#include "meshUtils.h"
#include "oct2vtk.h"
#include "point.h"
#include "subSM.h"
#include "ts.h"
#include "waveletAMR.h"

#define __LTS_BLK_SYNC_TIME_COMP_TOL__ 1e-6

namespace ts {
/**@brief: Meta-data structure to compute the correct weight for the LTS wpart.
 */
struct WeightedPartitionData {
    unsigned int lmin;
    unsigned int lmax;
    unsigned int (*tfac)(unsigned int blev, unsigned int lmin,
                         unsigned int lmax);
};

/**@brief:  wpart meta data, (only should be used in the LTS class. )*/
static WeightedPartitionData wpart_meta;

/**
 * @brief : simple structure to support storing of a single time step (explicit
 * methods) note that the stages are numbered from 1 to m_uiNumStages.
 * @tparam T : vector type
 */
template <typename T>
struct BlockTimeStep {
   public:
    /**@brief: ets stages
     * stage 0    : input vector
     * stage 1    : ets stage 1
     *            .
     *            .
     * stage p    : ets stage p
     * stage p+1  : output vector
     */
    std::vector<BlockAsyncVector<T>> _vec;

    /**@brief: rk stage*/
    unsigned int _rks  = ETS_STAGE_DEFAULT;

    /**@brief: block time*/
    unsigned int _time = LOOK_UP_TABLE_DEFAULT;

    /**
     * @brief allocate the Block time step vector.
     * @param numVec : number of vectors per block.
     * @param blkid  : block id
     * @param sz     : sizes of the each dimension
     * @param dof    : degrees of freedoms.
     */
    void alloc_vec(unsigned int numVec, unsigned int blkid,
                   const unsigned int* sz, unsigned int dof = 1) {
        _vec.resize(numVec);
        for (unsigned int k = 0; k < _vec.size(); k++)
            _vec[k].createVec(blkid, sz, false, BLK_ASYNC_VEC_MODE::BLK_UNZIP,
                              dof);
    }

    /**
     * @brief deallocate the vectors.
     */
    void dealloc_vec() {
        for (unsigned int k = 0; k < _vec.size(); k++) _vec[k].destroyVec();

        _vec.clear();
    }
};

/**
 * @brief Basic: class for performing non-uniform time stepping. (explicit time
 * stepping) In order to perform non uniform time stepping the octree to block
 * decomposition needed to be completed. We assume that the blocks are setup in
 * the mesh class.
 *
 * Note: The stages are numbered from 1 to m_uiNumStages.
 */
#ifdef __PROFILE_ENUTS__
enum ENUTSPROFILE {
    ENUTS_EVOLVE = 0,
    BLK_SYNC,
    NUTS_CORRECTION,
    ENUTS_BLK_UNZIP,
    ENUTS_BLK_ZIP,
    ENUTS_LAST
};
#endif

template <typename T, typename Ctx>
class ExplicitNUTS : public ETS<T, Ctx> {
    using ETS<T, Ctx>::m_uiAppCtx;
    using ETS<T, Ctx>::m_uiNumStages;
    using ETS<T, Ctx>::m_uiEVar;
    using ETS<T, Ctx>::m_uiStVec;
    using ETS<T, Ctx>::m_uiEVecTmp;
    using ETS<T, Ctx>::m_uiTimeInfo;
    using ETS<T, Ctx>::m_uiAij;
    using ETS<T, Ctx>::m_uiBi;
    using ETS<T, Ctx>::m_uiCi;
    using ETS<T, Ctx>::m_uiType;
    using ETS<T, Ctx>::m_uiIsInternalAlloc;

#ifdef __PROFILE_ENUTS__

   public:
    std::vector<profiler_t> m_uiPt =
        std::vector<profiler_t>(static_cast<int>(ENUTSPROFILE::ENUTS_LAST));
    const char* ENUTSPROFILE_NAMES[static_cast<int>(ENUTSPROFILE::ENUTS_LAST)] =
        {"evolve", "blk_sync", "nuts_corr", "blk_unzip", "blk_zip"};

    void init_pt() {
        for (unsigned int i = 0; i < m_uiPt.size(); i++) m_uiPt[i].start();

        m_uiAppCtx->init_pt();
    }

    void reset_pt() {
        for (unsigned int i = 0; i < m_uiPt.size(); i++) m_uiPt[i].snapreset();

        m_uiAppCtx->reset_pt();
    }

    void dump_pt(std::ostream& outfile) {
        const ot::Mesh* m_uiMesh = m_uiAppCtx->get_mesh();

        if (!(m_uiMesh->isActive())) return;

        int rank                       = m_uiMesh->getMPIRank();
        int npes                       = m_uiMesh->getMPICommSize();

        MPI_Comm comm                  = m_uiMesh->getMPICommunicator();
        const unsigned int currentStep = m_uiAppCtx->get_ts_info()._m_uiStep;

        double t_stat;
        double t_stat_g[3];

        if (!rank) {
            // writes the header
            if (currentStep <= 1)
                outfile << "step_nuts\t act_npes\t glb_npes\t maxdepth\t "
                           "numOcts\t dof_cg\t dof_uz\t"
                        << "gele_min\t gele_mean\t gele_max\t"
                           "lele_min\t lele_mean\t lele_max\t"
                           "lnodes_min\t lnodes_mean\t lnodes_max\t"
                           "remsh_igt_min\t remesh_igt_mean\t remesh_igt_max\t"
                           "evolve_min\t evolve_mean\t evolve_max\t"
                           "blk_sync_min\t blk_sync_mean\t blk_sync_max\t"
                           "nuts_corr_min\t nuts_corr_mean\t nuts_corr_max\t"
                           "blk_unzip_min\t blk_unzip_mean\t blk_unzip_max\t"
                           "rhs_blk_min\t rhs_blk_mean\t rhs_blk_max\t"
                           "blk_zip_min\t blk_zip_mean\t blk_zip_max\t"
                        << std::endl;
        }

        if (!rank) outfile << currentStep << "\t ";
        if (!rank) outfile << m_uiMesh->getMPICommSize() << "\t ";
        if (!rank) outfile << m_uiMesh->getMPICommSizeGlobal() << "\t ";
        if (!rank) outfile << m_uiMaxDepth << "\t ";

        DendroIntL localSz = m_uiMesh->getNumLocalMeshElements();
        DendroIntL globalSz;

        par::Mpi_Reduce(&localSz, &globalSz, 1, MPI_SUM, 0, comm);
        if (!rank) outfile << globalSz << "\t ";

        localSz = m_uiMesh->getNumLocalMeshNodes();
        par::Mpi_Reduce(&localSz, &globalSz, 1, MPI_SUM, 0, comm);
        if (!rank) outfile << globalSz << "\t ";

        localSz = m_uiMesh->getDegOfFreedomUnZip();
        par::Mpi_Reduce(&localSz, &globalSz, 1, MPI_SUM, 0, comm);
        if (!rank) outfile << globalSz << "\t ";

        DendroIntL ghostElements = m_uiMesh->getNumPreGhostElements() +
                                   m_uiMesh->getNumPostGhostElements();
        DendroIntL localElements = m_uiMesh->getNumLocalMeshElements();

        t_stat                   = ghostElements;
        min_mean_max(&t_stat, t_stat_g, comm);
        if (!rank)
            outfile << t_stat_g[0] << "\t " << t_stat_g[1] << "\t "
                    << t_stat_g[2] << "\t ";

        t_stat = localElements;
        min_mean_max(&t_stat, t_stat_g, comm);
        if (!rank)
            outfile << t_stat_g[0] << "\t " << t_stat_g[1] << "\t "
                    << t_stat_g[2] << "\t ";

        DendroIntL ghostNodes =
            m_uiMesh->getNumPreMeshNodes() + m_uiMesh->getNumPostMeshNodes();
        DendroIntL localNodes = m_uiMesh->getNumLocalMeshNodes();

        t_stat                = localNodes;
        min_mean_max(&t_stat, t_stat_g, comm);
        if (!rank)
            outfile << t_stat_g[0] << "\t " << t_stat_g[1] << "\t "
                    << t_stat_g[2] << "\t ";

        t_stat = m_uiAppCtx->m_uiCtxpt[CTXPROFILE::REMESH].snap;
        min_mean_max(&t_stat, t_stat_g, comm);
        if (!rank)
            outfile << t_stat_g[0] << "\t " << t_stat_g[1] << "\t "
                    << t_stat_g[2] << "\t ";

        t_stat = m_uiPt[ENUTSPROFILE::ENUTS_EVOLVE].snap;
        min_mean_max(&t_stat, t_stat_g, comm);
        if (!rank)
            outfile << t_stat_g[0] << "\t " << t_stat_g[1] << "\t "
                    << t_stat_g[2] << "\t ";

        t_stat = m_uiPt[ENUTSPROFILE::BLK_SYNC].snap;
        min_mean_max(&t_stat, t_stat_g, comm);
        if (!rank)
            outfile << t_stat_g[0] << "\t " << t_stat_g[1] << "\t "
                    << t_stat_g[2] << "\t ";

        t_stat = m_uiPt[ENUTSPROFILE::NUTS_CORRECTION].snap;
        min_mean_max(&t_stat, t_stat_g, comm);
        if (!rank)
            outfile << t_stat_g[0] << "\t " << t_stat_g[1] << "\t "
                    << t_stat_g[2] << "\t ";

        t_stat = m_uiPt[ENUTSPROFILE::ENUTS_BLK_UNZIP].snap;
        min_mean_max(&t_stat, t_stat_g, comm);
        if (!rank)
            outfile << t_stat_g[0] << "\t " << t_stat_g[1] << "\t "
                    << t_stat_g[2] << "\t ";

        t_stat = m_uiAppCtx->m_uiCtxpt[CTXPROFILE::RHS_BLK].snap;
        min_mean_max(&t_stat, t_stat_g, comm);
        if (!rank)
            outfile << t_stat_g[0] << "\t " << t_stat_g[1] << "\t "
                    << t_stat_g[2] << "\t ";

        t_stat = m_uiPt[ENUTSPROFILE::ENUTS_BLK_ZIP].snap;
        min_mean_max(&t_stat, t_stat_g, comm);
        if (!rank)
            outfile << t_stat_g[0] << "\t " << t_stat_g[1] << "\t "
                    << t_stat_g[2] << "\t ";

        if (!rank) outfile << std::endl;
    }

#endif

   protected:
    /**@brief: minimum level of the grid. */
    unsigned int m_uiLevMin;

    /**@brief: max level of the grid. */
    unsigned int m_uiLevMax;

    /**@brief: element to block map. */
    std::vector<unsigned int> m_uiE2B;

    /**@brief: unzip vector for m_uiEVar*/
    DVec m_uiEvarUzip;

    /**@brief: DG vector for m_uiEVar*/
    DVec m_uiEvarDG;

    /**@brief per local cg, the DG slot of its SFC-canonical owner (coarsest
     * sharer, ties by TreeNode order): the writer buildE2NMap picks on an
     * SFC mesh. Rebuilt per mesh; labelled meshes zip through it. */
    std::vector<unsigned int> m_uiCgOwnerDG;
    const ot::Mesh* m_uiCgOwnerMesh = nullptr;

    /**@brief: temp vector of the DG scheme*/
    DVec m_uiDGTmp0;

    /**@brief: List of block async vector. */
    std::vector<ts::BlockTimeStep<T>> m_uiBVec;

    /**@brief : keep track of the element time. */
    std::vector<unsigned int> m_uiEleTime;

    /**@brief : keep track of the element DT (int) to get the double dt,
     * m_uiDT*DT[e] */
    std::vector<unsigned int> m_uiEleDT;

    /**@brief min and max time of blocks for a given level. */
    std::vector<unsigned int> m_uiBlkTimeLevMinMax;

    /**@brief: explicit timer interpolation operators for ENUTS */
    ENUTSOp* m_uiECOp = NULL;

    /**@brief: store the partial step id for the latest evolution. */
    unsigned int m_uiPt;

    /**@brief: store the active block IDs evolved by the partial blocks (unzip
     * independent block ids)*/
    std::vector<unsigned int> m_uiIndependentActiveBlkIDs;

    /**@brief: store the active block IDs evolved by the partial blocks (unzip
     * dependent block ids)*/
    std::vector<unsigned int> m_uiDependentActiveBlkIDs;

    /**@brief: The union of block independent and dependent block ids. */
    std::vector<unsigned int> m_uiActiveBlkIDs;

    /**@brief : if true weighted partition is used*/
    bool m_uiUseWeightedPart;

   private:
    /**@brief: allocates the data stuctures and initialize with the current mesh
     * stores in the class. */
    void init_data_structures();

    /**@brief: freee the allocated data structures. */
    void free_data_structures();

   private:
    /**
     * @brief Allocates internal variables for the time stepper.
     * @return int
     */
    int allocate_internal_vars();

    /**@brief: Deallocate internal variables*/
    int deallocate_internal_vars();

    /**@bried: synchronize the block vectors*/
    void debug_nan_check(const char* tag, unsigned int blk, unsigned int s,
                         unsigned int pt, unsigned int rk);
    void debug_blk_hash(const char* tag, unsigned int blk, unsigned int s,
                        unsigned int pt, unsigned int rk);
    void build_cg_owner_map();
    void zip_labelled_via_dg();

    void sync_blk_timestep(unsigned int blk, unsigned int rk_s,
                           bool isIGTSync = false);

    /**@brief: update the element timestep*/
    void update_ele_timestep();

   public:
    /**@brief: constructor
     * Assumptions: Note that blocks result in from octree to block
     * decomposition can be not 2:1 balanced. But to perform NUTS, blocks should
     * be 2:1 balanced.
     */
    ExplicitNUTS(Ctx* ctx, bool useWpart = true);

    /**@brief: default destructor */
    ~ExplicitNUTS();

    /**@brief: build all the required data structures for the non uniform TS*/
    void init();

    /**@brief : evolve the variables to the next coarsest time step (i.e. loop
     * over the block sequence. ) Evolution in the sense of the  NUTS*/
    void evolve();

    /**
     * @brief : Perform parital step for the evolution.
     * @param[in] pt: partial step id.
     */
    void partial_evolve(unsigned int pt);

    void gts_evolve(unsigned int pt);

    /**@brief: compute the evolution with remesh triggered at partial steps. */
    void evolve_with_remesh(unsigned int remesh_freq = 5);

    /**@brief: returns  a constant pointer to the sub scatter maps. */
    // inline ot::SubScatterMap* const get_sub_scatter_maps() const {return
    // m_uiSubSM;}

    /**@brief: perform remesh for the nuts class. */
    void remesh(unsigned int grain_sz = DENDRO_DEFAULT_GRAIN_SZ,
                double ld_tol         = DENDRO_DEFAULT_LB_TOL,
                unsigned int sf_k     = DENDRO_DEFAULT_SF_K);

    /**
     * @brief compute the wavelets for the LTS vectors.
     *
     * @param varIds : variable ids considered for remeshing.
     * @param numvars : number of variables.
     * @param amr_coarsen_fac : AMR coarsening factor.
     * @return true : if needed remeshing.
     * @return false
     */
    bool isRemeshEvars(
        const unsigned int* varIds, unsigned int numvars,
        std::function<double(double, double, double, double*)> wavelet_tol,
        double amr_coarsen_fac = 0.1);

    /**@brief: update all the internal data strucutures, with a new mesh
     * pointer. */
    int sync_with_mesh();

    /**@brief returns the dt min value */
    T get_dt_min() const {
        return m_uiTimeInfo._m_uiTh * m_uiAppCtx->getBlkTimestepFac(
                                          m_uiLevMax, m_uiLevMin, m_uiLevMax);
    }

    /**@brief returns the dt max value */
    T get_dt_max() const {
        return m_uiTimeInfo._m_uiTh * m_uiAppCtx->getBlkTimestepFac(
                                          m_uiLevMin, m_uiLevMin, m_uiLevMax);
    }

    /**@brief: prints the load balance statistis*/
    void dump_load_statistics(std::ostream& sout);

    /**@brief: computes the estimated speedup for a given mesh. */
    void dump_est_speedup(std::ostream& sout, bool verbose = false);

    /**@brief : perform blk time step vector to vec CG zip operation.
     * @param blkVec : pointer to the block vector.
     * @param vecCG  : allocated CG vector.
     * @param s: blkVec.__vec[s] access parameter.
     */
    void blk_vec_to_zipCG(const ts::BlockTimeStep<T>* blkVec, DVec& vecCG,
                          unsigned int s = 0);

    /**
     * @brief perform zip operation (DG) to get the vecDG from block vector.
     * @param blkVec : blk vector in
     * @param vecDG  : allocated DG vector.
     * @param s: blkVec.__vec[s] access parameter.
     */
    void blk_vec_to_zipDG(const ts::BlockTimeStep<T>* blkVec, DVec& vecDG,
                          unsigned int s = 0);

    /**
     * @brief copy the block vector, to unzip vector,
     * @param blkVec : array of block async vector
     * @param vecUzip : unzip vector
     * @param s : s index of the block async vector
     */
    void blk_vec_to_unzipDG(const ts::BlockTimeStep<T>* blkVec, DVec& vecUzip,
                            unsigned int s = 0);

    /**
     * @brief copy DG vector to the block vector.
     *
     * @param vecDG DG vector
     * @param blkVec array of block async vector,
     * @param s : s index, of the block async vector.
     */
    void vecDG_to_blk_vec(DVec& vecDG, ts::BlockTimeStep<T>* blkVec,
                          unsigned int s = 0);

    /**@brief: compute weight for a wpart. */
    static unsigned int getOctWeight(const ot::TreeNode* pNode) {
        const unsigned int finest_t = wpart_meta.tfac(
            wpart_meta.lmax, wpart_meta.lmin,
            wpart_meta
                .lmax);  // m_uiAppCtx->getBlkTimestepFac(m_uiLevMax,m_uiLevMin,m_uiLevMax);
                         // // finest  time level.
        const unsigned int coarset_t = wpart_meta.tfac(
            wpart_meta.lmin, wpart_meta.lmin,
            wpart_meta
                .lmax);  // m_uiAppCtx->getBlkTimestepFac(m_uiLevMin,m_uiLevMin,m_uiLevMax);
                         // // coarset time level.
        const unsigned int weight =
            ((coarset_t - finest_t) /
             (wpart_meta.tfac(pNode->getLevel(), wpart_meta.lmin,
                              wpart_meta.lmax)));  //*nPe;
        return std::max(1u, weight);
    }

    /**@brief: update wpart meta data from current variables. */
    void update_wpart_metadata() {
        wpart_meta.lmin = m_uiLevMin;
        wpart_meta.lmax = m_uiLevMax;
        wpart_meta.tfac = &(Ctx::getBlkTimestepFac);

        return;
    }
};

template <typename T, typename Ctx>
ExplicitNUTS<T, Ctx>::ExplicitNUTS(Ctx* ctx, bool useWpart) : ETS<T, Ctx>(ctx) {
    m_uiUseWeightedPart = useWpart;
    if (m_uiUseWeightedPart) {
        ot::Mesh* pMesh = m_uiAppCtx->get_mesh();
        pMesh->computeMinMaxLevel(m_uiLevMin, m_uiLevMax);

        update_wpart_metadata();

        if (!m_uiAppCtx->is_wpart_func_set())
            m_uiAppCtx->set_wpart_function(getOctWeight);

        // the ctx path is virtual: solvers with a labelled mesh route it
        // through their SFC-twin sandwich
        m_uiAppCtx->remesh_and_gridtransfer();
    }
    this->init_data_structures();
}

template <typename T, typename Ctx>
ExplicitNUTS<T, Ctx>::~ExplicitNUTS() {
    this->deallocate_internal_vars();
    this->free_data_structures();
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::dump_est_speedup(std::ostream& sout, bool verbose) {
    const ot::Mesh* pMesh = m_uiAppCtx->get_mesh();

    if (pMesh->isActive()) {
        const unsigned int rank     = pMesh->getMPIRank();
        const unsigned int npes     = pMesh->getMPICommSize();
        MPI_Comm comm               = pMesh->getMPICommunicator();

        const unsigned int finest_t = m_uiAppCtx->getBlkTimestepFac(
            m_uiLevMax, m_uiLevMin, m_uiLevMax);  // finest  time level.
        const unsigned int coarset_t = m_uiAppCtx->getBlkTimestepFac(
            m_uiLevMin, m_uiLevMin, m_uiLevMax);  // coarset time level.

        double red_stat[3];  // for mpi reduction stats.
        const std::vector<ot::Block>& blkList = pMesh->getLocalBlockList();
        const ot::TreeNode* pNodes            = pMesh->getAllElements().data();

        // initialize the block async vectors
        for (unsigned int blk = 0; blk < blkList.size(); blk++) {
            m_uiBVec[blk]._time = 0;
            m_uiBVec[blk]._rks  = 0;
        }

        double localSz = 0;

        for (unsigned int blk = 0; blk < blkList.size(); blk++) {
            double nnx = (blkList[blk].getAllocationSzX() -
                          2 * blkList[blk].get1DPadWidth());
            // localSz += (nnx*nnx*nnx);  //(blkList[blk].getLocalElementEnd() -
            // blkList[blk].getLocalElementBegin());
            localSz += blkList[blk].numElements();
        }

        double globalESz;
        par::Mpi_Reduce(&localSz, &globalESz, 1, MPI_SUM, 0, comm);

        double globalBlkSz;
        localSz = pMesh->getLocalBlockList().size();
        par::Mpi_Reduce(&localSz, &globalBlkSz, 1, MPI_SUM, 0, comm);

        std::vector<DendroIntL> octByLev;
        std::vector<DendroIntL> octByLev_g;
        octByLev.resize(m_uiMaxDepth + 1, 0);
        octByLev_g.resize(m_uiMaxDepth + 1, 0);

        for (unsigned int ele = pMesh->getElementLocalBegin();
             ele < pMesh->getElementLocalEnd(); ele++)
            octByLev[pNodes[ele].getLevel()]++;

        par::Mpi_Reduce(octByLev.data(), octByLev_g.data(), m_uiMaxDepth,
                        MPI_SUM, 0, comm);

        if (!rank) {
            sout << "level\t num_elements\t\n" << std::endl;
            for (int l = octByLev_g.size() - 1; l >= 0; l--)
                sout << l << "\t" << octByLev_g[l] << std::endl;
        }

        if (verbose && !rank)
            sout << "partial_step\t active_elements\t all_elements\n"
                 << std::endl;

        double totalW = 0;
        for (unsigned int pt = 0; pt < coarset_t; pt++) {
            double numBlkEvolved = 0;

            for (unsigned int blk = 0; blk < blkList.size(); blk++) {
                const unsigned int BLK_T = m_uiBVec[blk]._time;
                const unsigned int NN    = m_uiBVec[blk]._vec[0].getSz();
                const unsigned int bLev =
                    pNodes[blkList[blk].firstElementID()].getLevel();
                const unsigned int BLK_DT =
                    m_uiAppCtx->getBlkTimestepFac(bLev, m_uiLevMin, m_uiLevMax);
                if (pt % BLK_DT != 0) continue;

                double nnx = (blkList[blk].getAllocationSzX() -
                              2 * blkList[blk].get1DPadWidth());
                // std::cout<<"pt: "<<pt<<" blk id : "<<blk<<" blklevel :
                // "<<bLev<<" is now at btime : "<<BLK_T<<" dt:
                // "<<BLK_DT<<std::endl;

                // numBlkEvolved += (nnx*nnx*nnx);
                // //(blkList[blk].getLocalElementEnd() -
                // blkList[blk].getLocalElementBegin());
                numBlkEvolved += blkList[blk].numElements();

                m_uiBVec[blk]._time += BLK_DT;
                m_uiBVec[blk]._rks = 0;
            }

            par::Mpi_Reduce(&numBlkEvolved, red_stat, 1, MPI_MIN, 0, comm);
            par::Mpi_Reduce(&numBlkEvolved, red_stat + 1, 1, MPI_SUM, 0, comm);
            par::Mpi_Reduce(&numBlkEvolved, red_stat + 2, 1, MPI_MAX, 0, comm);

            totalW += red_stat[1];

            if (verbose && !rank)
                sout << pt << "\t" << red_stat[1] << "\t" << globalESz << "\t"
                     << (red_stat[1] / globalESz) << std::endl;
            // std::cout<<" partial step : "<<pt<< " number of dof evolved (min,
            // sum, max): ("<<red_stat[0]<<",\t "<<red_stat[1]<<",\t
            // "<<red_stat[2]<<") out of  "<<globalESz<<std::endl;

            // compute the time step vector and increment time.
            // for(unsigned int blk =0; blk < blkList.size(); blk++)
            // {

            //     const unsigned int BLK_T = m_uiBVec[blk]._time;
            //     const unsigned int NN   =  m_uiBVec[blk]._vec[0].getSz();
            //     const unsigned int bLev =
            //     pNodes[blkList[blk].firstElementID()].getLevel(); const
            //     unsigned int BLK_DT =
            //     m_uiAppCtx->getBlkTimestepFac(bLev,m_uiLevMin,m_uiLevMax);
            //     if( pt% BLK_DT !=0 )
            //         continue;

            //     m_uiBVec[blk]._time += BLK_DT;
            //     m_uiBVec[blk]._rks=0;

            // }
        }

        if (!rank) {
            sout << "number of LTS partial timesteps " << coarset_t
                 << std::endl;
            sout << "LTS Work : " << totalW
                 << " GTS Work: " << globalESz * coarset_t << std::endl;
            sout << " ets. speedup : "
                 << (globalESz * coarset_t / (double)(totalW)) << std::endl;
        }
    }
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::dump_load_statistics(std::ostream& sout) {
    const ot::Mesh* pMesh = m_uiAppCtx->get_mesh();

    if (pMesh->isActive()) {
        double local_weight        = 0;
        const ot::TreeNode* pNodes = pMesh->getAllElements().data();
        for (unsigned int ele = pMesh->getElementLocalBegin();
             ele < pMesh->getElementLocalEnd(); ele++)
            local_weight += getOctWeight(&pNodes[ele]);

        double ld_stat[3];
        MPI_Comm aComm = pMesh->getMPICommunicator();

        par::Mpi_Reduce(&local_weight, ld_stat + 0, 1, MPI_MIN, 0, aComm);
        par::Mpi_Reduce(&local_weight, ld_stat + 1, 1, MPI_SUM, 0, aComm);
        ld_stat[1] = ld_stat[1] / (double)pMesh->getMPICommSize();
        par::Mpi_Reduce(&local_weight, ld_stat + 2, 1, MPI_MAX, 0, aComm);

        if (!pMesh->getMPIRank())
            std::cout << YLW << "\t LD Bal: (min,mean,max): " << ld_stat[0]
                      << "|\t" << ld_stat[1] << "|\t" << ld_stat[2] << NRM
                      << std::endl;
    }

    return;
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::init_data_structures() {
    // identify dependent and independent blocks.
    const ot::Mesh* pMesh = m_uiAppCtx->get_mesh();
    const bool isActive   = pMesh->isActive();

    pMesh->computeMinMaxLevel(m_uiLevMin, m_uiLevMax);
    assert((m_uiLevMin > 0) && (m_uiLevMax <= m_uiMaxDepth));

    update_wpart_metadata();
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::free_data_structures() {
    return;
}

template <typename T, typename Ctx>
int ExplicitNUTS<T, Ctx>::allocate_internal_vars() {
    assert(m_uiNumStages > 0);
    const unsigned int DOF = m_uiEVar.get_dof();

    if (m_uiIsInternalAlloc)
        return 0;  // no need to allocated again if the internal vars are
                   // allocated.

    m_uiStVec.resize(m_uiNumStages);

    for (unsigned int i = 0; i < m_uiNumStages; i++)
        m_uiStVec[i].create_vector(
            m_uiAppCtx->get_mesh(), ot::DVEC_TYPE::OCT_LOCAL_NODES,
            ot::DVEC_LOC::HOST, m_uiEVar.get_dof(), true);

    m_uiEVecTmp[0].create_vector(m_uiAppCtx->get_mesh(),
                                 ot::DVEC_TYPE::OCT_SHARED_NODES,
                                 ot::DVEC_LOC::HOST, m_uiEVar.get_dof(), true);
    m_uiEVecTmp[1].create_vector(m_uiAppCtx->get_mesh(),
                                 ot::DVEC_TYPE::OCT_SHARED_NODES,
                                 ot::DVEC_LOC::HOST, m_uiEVar.get_dof(), true);

    // allocate tmp DG vectors
    m_uiDGTmp0.create_vector(m_uiAppCtx->get_mesh(),
                             ot::DVEC_TYPE::OCT_LOCAL_NODES, ot::DVEC_LOC::HOST,
                             m_uiEVar.get_dof(), true);

    m_uiEvarUzip.create_vector(m_uiAppCtx->get_mesh(),
                               ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING,
                               ot::DVEC_LOC::HOST, m_uiEVar.get_dof(), true);
    m_uiCgOwnerMesh = nullptr;
    m_uiEvarDG.create_vector(m_uiAppCtx->get_mesh(),
                             ot::DVEC_TYPE::OCT_LOCAL_NODES, ot::DVEC_LOC::HOST,
                             m_uiEVar.get_dof(), true);

    const ot::Mesh* pMesh = m_uiAppCtx->get_mesh();
    if (pMesh->isActive()) {
        m_uiEleTime.resize(pMesh->getAllElements().size(), 0);
        m_uiEleDT.resize(pMesh->getAllElements().size(), 0);

        const std::vector<ot::Block>& blkList = pMesh->getLocalBlockList();
        m_uiBVec.resize(blkList.size());

        for (unsigned int blk = 0; blk < blkList.size(); blk++) {
            const unsigned int sz[3] = {blkList[blk].getAllocationSzX(),
                                        blkList[blk].getAllocationSzY(),
                                        blkList[blk].getAllocationSzZ()};
            m_uiBVec[blk].alloc_vec(m_uiNumStages + 2, blk, sz, DOF);
        }
    }

    m_uiECOp            = new ENUTSOp(m_uiType);
    m_uiIsInternalAlloc = true;
    m_uiBlkTimeLevMinMax.resize(2 * m_uiMaxDepth, 0);

    m_uiActiveBlkIDs.reserve(pMesh->getLocalBlockList().size());
    m_uiDependentActiveBlkIDs.reserve(pMesh->getLocalBlockList().size());
    m_uiIndependentActiveBlkIDs.reserve(pMesh->getLocalBlockList().size());
    return 0;
}

template <typename T, typename Ctx>
int ExplicitNUTS<T, Ctx>::deallocate_internal_vars() {
    if (!m_uiIsInternalAlloc) return 0;

    for (unsigned int i = 0; i < m_uiNumStages; i++) {
        this->m_uiStVec[i].destroy_vector();
    }

    this->m_uiStVec.clear();
    this->m_uiEVecTmp[0].destroy_vector();
    this->m_uiEVecTmp[1].destroy_vector();

    // deallocate DG tmp.
    m_uiDGTmp0.destroy_vector();

    m_uiEvarUzip.destroy_vector();
    m_uiEvarDG.destroy_vector();

    m_uiEleTime.clear();
    m_uiEleDT.clear();

    for (unsigned int k = 0; k < m_uiBVec.size(); k++) {
        for (unsigned int j = 0; j < m_uiBVec[k]._vec.size(); j++)
            m_uiBVec[k]._vec[j].destroyVec();

        m_uiBVec[k]._vec.clear();
    }

    m_uiBVec.clear();

    // deallocate m_uiStVec, which is based on the number of stages
    for (unsigned int i = 0; i < m_uiNumStages; i++) {
        m_uiStVec[i].destroy_vector();
    }
    m_uiStVec.clear();

    m_uiBlkTimeLevMinMax.clear();

    m_uiActiveBlkIDs.clear();
    m_uiIndependentActiveBlkIDs.clear();
    m_uiDependentActiveBlkIDs.clear();

    delete m_uiECOp;
    m_uiIsInternalAlloc = false;
    return 0;
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::blk_vec_to_zipCG(const ts::BlockTimeStep<T>* blkVec,
                                            DVec& vecCG, unsigned int s) {
    ot::Mesh* pMesh                       = m_uiAppCtx->get_mesh();
    const std::vector<ot::Block>& blkList = pMesh->getLocalBlockList();
    const unsigned int DOF                = vecCG.get_dof();

    for (unsigned int blk = 0; blk < blkList.size(); blk++)
        blkVec[blk]._vec[s].zip(pMesh, vecCG.get_vec_ptr(), DOF);

    return;
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::blk_vec_to_zipDG(const ts::BlockTimeStep<T>* blkVec,
                                            DVec& vecDG, unsigned int s) {
    ot::Mesh* pMesh                       = m_uiAppCtx->get_mesh();
    const std::vector<ot::Block>& blkList = pMesh->getLocalBlockList();
    const unsigned int DOF                = vecDG.get_dof();

    for (unsigned int blk = 0; blk < blkList.size(); blk++)
        blkVec[blk]._vec[s].zipDG(pMesh, vecDG.get_vec_ptr(), DOF);

    return;
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::blk_vec_to_unzipDG(
    const ts::BlockTimeStep<T>* blkVec, DVec& vecUzip, unsigned int s) {
    ot::Mesh* pMesh                       = m_uiAppCtx->get_mesh();
    const std::vector<ot::Block>& blkList = pMesh->getLocalBlockList();
    const unsigned int DOF                = vecUzip.get_dof();

    for (unsigned int v = 0; v < DOF; v++) {
        T* d_ptr = vecUzip.get_vec_ptr() + v * pMesh->getDegOfFreedomUnZip();

        for (unsigned int blk = 0; blk < blkList.size(); blk++) {
            const unsigned int offset = blkList[blk].getOffset();

            const unsigned int lx     = blkList[blk].getAllocationSzX();
            const unsigned int ly     = blkList[blk].getAllocationSzY();
            const unsigned int lz     = blkList[blk].getAllocationSzZ();

            const T* bvec = blkVec[blk]._vec[s].data() + (v * lx * ly * lz);

            std::memcpy(d_ptr + offset, bvec, sizeof(T) * (lx * ly * lz));
        }
    }

    return;
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::vecDG_to_blk_vec(DVec& vecDG,
                                            ts::BlockTimeStep<T>* blkVec,
                                            unsigned int s) {
    ot::Mesh* pMesh                       = m_uiAppCtx->get_mesh();
    const std::vector<ot::Block>& blkList = pMesh->getLocalBlockList();
    const unsigned int DOF                = vecDG.get_dof();

    for (unsigned int blk = 0; blk < blkList.size(); blk++) {
        blkVec[blk]._vec[s].copyFromVecDG(pMesh, vecDG.get_vec_ptr(), DOF);
        blkVec[blk]._vec[s].mark_unsynced();
    }
}

template <typename T, typename Ctx>
int ExplicitNUTS<T, Ctx>::sync_with_mesh() {
    if (m_uiAppCtx->is_ets_synced()) return 0;

    this->deallocate_internal_vars();
    this->free_data_structures();

    m_uiEVar = m_uiAppCtx->get_evolution_vars();
    this->init_data_structures();
    this->allocate_internal_vars();

    m_uiAppCtx->set_ets_synced(true);

    return 0;
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::init() {
    m_uiAppCtx->initialize();
    m_uiTimeInfo = m_uiAppCtx->get_ts_info();
    allocate_internal_vars();
    m_uiAppCtx->compute_lts_ts_offset();

    if (m_uiUseWeightedPart) {
        ot::Mesh* pMesh = m_uiAppCtx->get_mesh();
        pMesh->computeMinMaxLevel(m_uiLevMin, m_uiLevMax);

        update_wpart_metadata();

        if (!m_uiAppCtx->is_wpart_func_set())
            m_uiAppCtx->set_wpart_function(getOctWeight);

        DendroIntL oldElements = pMesh->getNumLocalMeshElements(), oldElements_g = 0;
        par::Mpi_Reduce(&oldElements, &oldElements_g, 1, MPI_SUM, 0,
                        pMesh->getMPIGlobalCommunicator());
        m_uiAppCtx->remesh_and_gridtransfer();
        pMesh = m_uiAppCtx->get_mesh();
        DendroIntL newElements = pMesh->getNumLocalMeshElements(), newElements_g = 0;
        par::Mpi_Reduce(&newElements, &newElements_g, 1, MPI_SUM, 0,
                        pMesh->getMPIGlobalCommunicator());
        if (!(pMesh->getMPIRankGlobal()))
            std::cout << GRN << "[LTS Init(Wpart)]: \told mesh: " << oldElements_g
                      << "\tnew mesh:" << newElements_g << NRM << std::endl;
    }

    // Ctx initialize might have changed the mesh i.e. converge untill mesh
    // adapted to the initial data.
    m_uiAppCtx->set_ets_synced(false);
    this->sync_with_mesh();
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::build_cg_owner_map() {
    ot::Mesh* pMesh = m_uiAppCtx->get_mesh();
    if (m_uiCgOwnerMesh == pMesh && m_uiCgOwnerDG.size() == pMesh->getDegOfFreedom()) return;
    m_uiCgOwnerMesh = pMesh;
    const unsigned int eo = pMesh->getElementOrder(), nPe = pMesh->getNumNodesPerElement();
    const std::vector<ot::TreeNode>& all = pMesh->getAllElements();
    const unsigned int* e2n = pMesh->getE2NMapping().data();
    m_uiCgOwnerDG.assign(pMesh->getDegOfFreedom(), LOOK_UP_TABLE_DEFAULT);
    // sharers of a cg = every (element, non-hanging node) mapped to it; the
    // SFC build makes the coarsest sharer the owner, ties by TreeNode order
    for (unsigned int e = pMesh->getElementPreGhostBegin(); e < pMesh->getElementPostGhostEnd(); e++) {
        if (e < pMesh->getElementLocalBegin() || e >= pMesh->getElementLocalEnd()) {
            bool touchesLocal = false;
            for (unsigned int nd = 0; nd < nPe && !touchesLocal; nd++) {
                const unsigned int cg = e2n[e * nPe + nd];
                touchesLocal = cg >= pMesh->getNodeLocalBegin() && cg < pMesh->getNodeLocalEnd();
            }
            if (!touchesLocal) continue;
        }
        for (unsigned int k = 0; k <= eo; k++)
            for (unsigned int j = 0; j <= eo; j++)
                for (unsigned int i = 0; i <= eo; i++) {
                    const unsigned int nd = k * (eo + 1) * (eo + 1) + j * (eo + 1) + i;
                    const unsigned int cg = e2n[e * nPe + nd];
                    if (cg < pMesh->getNodeLocalBegin() || cg >= pMesh->getNodeLocalEnd()) continue;
                    if (pMesh->isNodeHanging(e, i, j, k)) continue;
                    unsigned int& cur = m_uiCgOwnerDG[cg];
                    if (cur == LOOK_UP_TABLE_DEFAULT) { cur = e * nPe + nd; continue; }
                    const unsigned int ce = cur / nPe;
                    if (all[e].getLevel() < all[ce].getLevel() ||
                        (all[e].getLevel() == all[ce].getLevel() && all[e] < all[ce]))
                        cur = e * nPe + nd;
                }
    }
    // slots no element references (or only hanging ones): keep the mesh's own canonical
    const std::vector<unsigned int>& cg2dg = pMesh->getCG2DGMap();
    for (unsigned int cg = pMesh->getNodeLocalBegin(); cg < pMesh->getNodeLocalEnd(); cg++)
        if (m_uiCgOwnerDG[cg] == LOOK_UP_TABLE_DEFAULT && cg < cg2dg.size()) m_uiCgOwnerDG[cg] = cg2dg[cg];
    if (std::getenv("DENDRO_ENUTS_NANCHK")) {
        const unsigned int* e2nDG = pMesh->getE2NMapping_DG().data();
        unsigned int set = 0, agree = 0, disagree = 0;
        for (unsigned int cg = pMesh->getNodeLocalBegin(); cg < pMesh->getNodeLocalEnd(); cg++) {
            if (m_uiCgOwnerDG[cg] == LOOK_UP_TABLE_DEFAULT) continue;
            set++;
            unsigned int oe, oi, oj, ok;
            pMesh->dg2eijk(m_uiCgOwnerDG[cg], oe, oi, oj, ok);
            const unsigned int nd = ok * (eo + 1) * (eo + 1) + oj * (eo + 1) + oi;
            if (e2nDG[oe * nPe + nd] == m_uiCgOwnerDG[cg] && e2n[oe * nPe + nd] == cg) { agree++; continue; }
            disagree++;
            if (disagree <= 8) {
                const unsigned int dgE = pMesh->getCG2DGMap()[cg];
                unsigned int de, di, dj, dk; pMesh->dg2eijk(dgE, de, di, dj, dk);
                const bool oeLocal = oe >= pMesh->getElementLocalBegin() && oe < pMesh->getElementLocalEnd();
                const bool deLocal = de >= pMesh->getElementLocalBegin() && de < pMesh->getElementLocalEnd();
                std::cout << "[cg-owner r" << pMesh->getMPIRankGlobal() << "]   cg " << cg << " mine " << all[oe] << " (" << oi << ',' << oj << ',' << ok << ") local=" << oeLocal
                          << "  e2nDG " << all[de] << " (" << di << ',' << dj << ',' << dk << ") local=" << deLocal
                          << " e2nDG-self=" << (e2nDG[de * nPe + dk * (eo + 1) * (eo + 1) + dj * (eo + 1) + di] == dgE)
                          << " tnless(mine<e2n)=" << (all[oe] < all[de]) << std::endl;
            }
        }
        std::cout << "[cg-owner r" << pMesh->getMPIRankGlobal() << "] local="
                  << (pMesh->getNodeLocalEnd() - pMesh->getNodeLocalBegin()) << " set=" << set
                  << " agreeE2NDG=" << agree << " disagree=" << disagree << std::endl;
    }
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::zip_labelled_via_dg() {
    ot::Mesh* pMesh = m_uiAppCtx->get_mesh();
    const unsigned int DOF  = m_uiEVar.get_dof();
    const unsigned int cgSz = pMesh->getDegOfFreedom();
    const unsigned int dgSz = pMesh->getDegOfFreedomDG();
    T* cg       = m_uiEVar.get_vec_ptr();
    const T* dg = m_uiEvarDG.get_vec_ptr();
    for (unsigned int v = 0; v < DOF; v++)
        for (unsigned int c = pMesh->getNodeLocalBegin(); c < pMesh->getNodeLocalEnd(); c++)
            if (m_uiCgOwnerDG[c] != LOOK_UP_TABLE_DEFAULT)
                cg[v * cgSz + c] = dg[v * dgSz + m_uiCgOwnerDG[c]];
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::debug_blk_hash(const char* tag, unsigned int blk,
                                          unsigned int s, unsigned int pt,
                                          unsigned int rk) {
    static const char* dir = std::getenv("DENDRO_ENUTS_BLKHASH");
    if (!dir) return;
    ot::Mesh* pMesh      = m_uiAppCtx->get_mesh();
    static std::ofstream out;
    if (!out.is_open())
        out.open(std::string(dir) + "/blkhash_r" +
                 std::to_string(pMesh->getMPIRankGlobal()) + ".txt");
    const T* v            = m_uiBVec[blk]._vec[s].data();
    const unsigned int n  = m_uiEVar.get_dof() * m_uiBVec[blk]._vec[s].getSz();
    const ot::Block& b    = pMesh->getLocalBlockList()[blk];
    const unsigned int NN = b.getAlignedBlockSz();
    const unsigned int lx = b.getAllocationSzX(), ly = b.getAllocationSzY(),
                       lz = b.getAllocationSzZ(), PW = b.get1DPadWidth();
    const double hx0 = (1u << (m_uiMaxDepth - pMesh->getAllElements()[b.firstElementID()].getLevel())) /
                       (double)pMesh->getElementOrder();
    const double dmax = (double)(1u << m_uiMaxDepth);
    unsigned long long hi = 1469598103934665603ull, hp = 1469598103934665603ull;
    double mi = 0, mp = 0;
    for (unsigned int i = 0; i < n; i++) {
        const unsigned int l = i % NN, x = l % lx, y = (l / lx) % ly, z = l / (lx * ly);
        const bool interior = !(x < PW || x >= lx - PW || y < PW || y >= ly - PW || z < PW || z >= lz - PW);
        if (!interior) {
            const double px = b.getBlockNode().minX() - PW * hx0 + x * hx0;
            const double py = b.getBlockNode().minY() - PW * hx0 + y * hx0;
            const double pz = b.getBlockNode().minZ() - PW * hx0 + z * hx0;
            if (px < 0 || py < 0 || pz < 0 || px > dmax || py > dmax || pz > dmax) continue;
        }
        unsigned long long bits; std::memcpy(&bits, &v[i], sizeof(bits));
        unsigned long long& h = interior ? hi : hp;
        double& m = interior ? mi : mp;
        h ^= bits; h *= 1099511628211ull;
        if (std::fabs(v[i]) > m) m = std::fabs(v[i]);
    }
    const ot::TreeNode bn = b.getBlockNode();
    std::vector<unsigned int> eid;
    pMesh->blkUnzipElementIDs(blk, eid);
    std::vector<ot::TreeNode> eidTN;
    for (const unsigned int e : eid) eidTN.push_back(pMesh->getAllElements()[e]);
    std::sort(eidTN.begin(), eidTN.end());
    unsigned long long he = 1469598103934665603ull, ho = he, hv = he;
    for (const ot::TreeNode& t : eidTN) {
        for (unsigned int c : {t.getX(), t.getY(), t.getZ(), t.getLevel()}) { he ^= c; he *= 1099511628211ull; }
    }
    for (const unsigned int e : eid) {
        const ot::TreeNode& t = pMesh->getAllElements()[e];
        for (unsigned int c : {t.getX(), t.getY(), t.getZ(), t.getLevel()}) { ho ^= c; ho *= 1099511628211ull; }
    }
    {
        std::vector<unsigned int> byTN(eid);
        std::sort(byTN.begin(), byTN.end(), [&](unsigned int p, unsigned int q) {
            return pMesh->getAllElements()[p] < pMesh->getAllElements()[q]; });
        const unsigned int nPe = pMesh->getNumNodesPerElement();
        const unsigned int dgSz = pMesh->getAllElements().size() * nPe;
        const T* src = (s == 0) ? m_uiEvarDG.get_vec_ptr()
                       : (s <= m_uiNumStages) ? m_uiStVec[s - 1].get_vec_ptr() : nullptr;
        const bool srcOk = src != nullptr && std::strcmp(tag, "init-unzip") != 0;
        if (srcOk) for (const unsigned int e : byTN)
            for (unsigned int vv = 0; vv < m_uiEVar.get_dof(); vv++)
                for (unsigned int k = 0; k < nPe; k++) {
                    unsigned long long bits; std::memcpy(&bits, &src[vv * dgSz + e * nPe + k], sizeof(bits));
                    hv ^= bits; hv *= 1099511628211ull;
                }
    }
    if (const char* sel = std::getenv("DENDRO_ENUTS_BLKDUMP")) {
        unsigned int sx, sy, sz, sl; unsigned int spt, srk; char stag[32];
        static unsigned int matches = 0;
        unsigned int socc = 1;
        if (std::sscanf(sel, "%u,%u,%u,%u,%u,%u,%31[^,],%u", &sx, &sy, &sz, &sl, &spt, &srk, stag, &socc) >= 7 &&
            sx == bn.getX() && sy == bn.getY() && sz == bn.getZ() && sl == bn.getLevel() &&
            spt == pt && srk == rk && std::strcmp(stag, tag) == 0 && ++matches == socc) {
            std::ofstream d(std::string(dir) + "/blkdump_r" + std::to_string(pMesh->getMPIRankGlobal()) + ".txt");
            {
                // per non-padding slot: cg, my owner, e2nDG owner
                const unsigned int eo = pMesh->getElementOrder(), nPe = pMesh->getNumNodesPerElement();
                const std::vector<ot::TreeNode>& all = pMesh->getAllElements();
                const unsigned int* e2nCG = pMesh->getE2NMapping().data();
                const unsigned int* e2nDG = pMesh->getE2NMapping_DG().data();
                const unsigned int regLev = b.getRegularGridLev();
                std::set<unsigned int> done;
                for (const unsigned int e : b) {
                    const unsigned int ei = (all[e].getX() - bn.getX()) >> (m_uiMaxDepth - regLev);
                    const unsigned int ej = (all[e].getY() - bn.getY()) >> (m_uiMaxDepth - regLev);
                    const unsigned int ek = (all[e].getZ() - bn.getZ()) >> (m_uiMaxDepth - regLev);
                    for (unsigned int k = 0; k <= eo; k++) for (unsigned int j = 0; j <= eo; j++) for (unsigned int i = 0; i <= eo; i++) {
                        const unsigned int sx2 = ei * eo + i + PW, sy2 = ej * eo + j + PW, sz2 = ek * eo + k + PW;
                        const unsigned int slot = sz2 * lx * ly + sy2 * lx + sx2;
                        if (!done.insert(slot).second) continue;
                        const unsigned int nd = k * (eo + 1) * (eo + 1) + j * (eo + 1) + i;
                        const unsigned int cg = e2nCG[e * nPe + nd], dg = e2nDG[e * nPe + nd];
                        unsigned int oe, oi, oj, ok; pMesh->dg2eijk(dg, oe, oi, oj, ok);
                        const bool cgLocal = cg >= pMesh->getNodeLocalBegin() && cg < pMesh->getNodeLocalEnd();
                        d << "slot " << sx2 << ' ' << sy2 << ' ' << sz2 << " cg " << cg << " cgLocal " << cgLocal << " hang " << pMesh->isNodeHanging(e, i, j, k)
                          << " e2nDG " << all[oe].getX() << ' ' << all[oe].getY() << ' ' << all[oe].getZ() << ' ' << all[oe].getLevel() << ' ' << oi << ' ' << oj << ' ' << ok
                          << " local " << (oe >= pMesh->getElementLocalBegin() && oe < pMesh->getElementLocalEnd());
                        if (cgLocal && cg < m_uiCgOwnerDG.size() && m_uiCgOwnerDG[cg] != LOOK_UP_TABLE_DEFAULT) {
                            unsigned int me, mi, mj, mk; pMesh->dg2eijk(m_uiCgOwnerDG[cg], me, mi, mj, mk);
                            d << " mine " << all[me].getX() << ' ' << all[me].getY() << ' ' << all[me].getZ() << ' ' << all[me].getLevel() << ' ' << mi << ' ' << mj << ' ' << mk
                              << " local " << (me >= pMesh->getElementLocalBegin() && me < pMesh->getElementLocalEnd());
                        } else d << " mine none";
                        d << '\n';
                    }
                }
            }
            for (const unsigned int e : eid) {
                const ot::TreeNode& t = pMesh->getAllElements()[e];
                d << "eid " << e << ' ' << t.getX() << ' ' << t.getY() << ' ' << t.getZ() << ' ' << t.getLevel()
                  << ' ' << (e >= pMesh->getElementLocalBegin() && e < pMesh->getElementLocalEnd()) << '\n';
            }
            for (unsigned int i = 0; i < n; i++) {
                const unsigned int l = i % NN, x = l % lx, y = (l / lx) % ly, z = l / (lx * ly);
                d << i / NN << ' ' << x << ' ' << y << ' ' << z << ' ' << std::setprecision(17) << v[i] << '\n';
            }
        }
    }
    out << bn.getX() << ' ' << bn.getY() << ' ' << bn.getZ() << ' ' << bn.getLevel()
        << ' ' << pt << ' ' << rk << ' ' << tag << ' ' << s << ' ' << std::hex << hi << ' ' << hp
        << std::dec << ' ' << std::setprecision(17) << mi << ' ' << mp << ' ' << b.isNonSFC()
        << ' ' << m_uiBVec[blk]._time << ' ' << eid.size() << ' ' << std::hex << he << ' ' << ho << ' ' << hv << std::dec << '\n';
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::debug_nan_check(const char* tag, unsigned int blk,
                                           unsigned int s, unsigned int pt,
                                           unsigned int rk) {
    static const bool on = [] {
        const char* e = std::getenv("DENDRO_ENUTS_NANCHK");
        return e && e[0] == '1';
    }();
    if (!on) return;
    ot::Mesh* pMesh      = m_uiAppCtx->get_mesh();
    const T* v           = m_uiBVec[blk]._vec[s].data();
    const unsigned int n = m_uiEVar.get_dof() * m_uiBVec[blk]._vec[s].getSz();
    const ot::Block& b   = pMesh->getLocalBlockList()[blk];
    const unsigned int NN = b.getAlignedBlockSz();
    const unsigned int lx = b.getAllocationSzX(), ly = b.getAllocationSzY(),
                       lz = b.getAllocationSzZ(), PW = b.get1DPadWidth();
    const bool interiorOnly = (std::strcmp(tag, "post-rhs") == 0);
    const double hx0 = (1u << (m_uiMaxDepth - pMesh->getAllElements()[b.firstElementID()].getLevel())) /
                       (double)pMesh->getElementOrder();
    const double dmax = (double)(1u << m_uiMaxDepth);
    for (unsigned int i = 0; i < n; i++) {
        const unsigned int l = i % NN, x = l % lx, y = (l / lx) % ly, z = l / (lx * ly);
        if (interiorOnly) {
            if (x < PW || x >= lx - PW || y < PW || y >= ly - PW || z < PW || z >= lz - PW) continue;
        }
        {
            const double px = b.getBlockNode().minX() - PW * hx0 + x * hx0;
            const double py = b.getBlockNode().minY() - PW * hx0 + y * hx0;
            const double pz = b.getBlockNode().minZ() - PW * hx0 + z * hx0;
            if (px < 0 || py < 0 || pz < 0 || px > dmax || py > dmax || pz > dmax) continue;
        }
        if (std::isnan(v[i]) || std::isinf(v[i])) {
            std::cout << "[enuts-nan r" << pMesh->getMPIRankGlobal() << "] "
                      << tag << " pt=" << pt << " rk=" << rk << " blk=" << blk
                      << " s=" << s << " idx=" << i << " of " << n << " lev="
                      << pMesh->getAllElements()[b.firstElementID()].getLevel()
                      << " nonSFC=" << b.isNonSFC()
                      << " nele=" << b.numElements()
                      << " node=" << b.getBlockNode()
                      << " time=" << m_uiBVec[blk]._time << std::endl;
            if (std::strcmp(tag, "post-sync") == 0) {
                std::vector<unsigned int> eid;
                pMesh->blkUnzipElementIDs(blk, eid);
                const unsigned int nPe = pMesh->getNumNodesPerElement();
                const unsigned int dgSz = pMesh->getAllElements().size() * nPe;
                const T* st = m_uiStVec[s - 1].get_vec_ptr();
                const std::vector<unsigned int>& r1 = pMesh->getLevel1GhostElementIndices();
                {
                    const unsigned int l = i % NN, x = l % lx, y = (l / lx) % ly, z = l / (lx * ly);
                    const unsigned int eo = pMesh->getElementOrder();
                    const double hx = (1u << (m_uiMaxDepth - pMesh->getAllElements()[b.firstElementID()].getLevel())) / (double)eo;
                    const double px = b.getBlockNode().minX() - PW * hx + x * hx;
                    const double py = b.getBlockNode().minY() - PW * hx + y * hx;
                    const double pz = b.getBlockNode().minZ() - PW * hx + z * hx;
                    std::cout << "[enuts-nan r" << pMesh->getMPIRankGlobal() << "]   slot (" << x << "," << y << "," << z
                              << ") pt (" << px << "," << py << "," << pz << ") PW=" << PW << " lx=" << lx << std::endl;
                    const std::vector<ot::TreeNode>& all = pMesh->getAllElements();
                    for (unsigned int e = 0; e < all.size(); e++) {
                        if (px < all[e].minX() || px > all[e].maxX() || py < all[e].minY() || py > all[e].maxY() ||
                            pz < all[e].minZ() || pz > all[e].maxZ()) continue;
                        const bool local = e >= pMesh->getElementLocalBegin() && e < pMesh->getElementLocalEnd();
                        const bool inR1 = std::find(r1.begin(), r1.end(), e) != r1.end();
                        const bool inEid = std::find(eid.begin(), eid.end(), e) != eid.end();
                        std::cout << "[enuts-nan r" << pMesh->getMPIRankGlobal() << "]   covering elem " << e << " " << all[e]
                                  << " local=" << local << " r1=" << inR1 << " inEid=" << inEid << std::endl;
                    }
                    std::cout << "[enuts-nan r" << pMesh->getMPIRankGlobal() << "]   eid size " << eid.size() << " :";
                    for (const unsigned int e : eid) std::cout << " " << all[e];
                    std::cout << std::endl;
                }
                for (const unsigned int e : eid) {
                    unsigned int bad = 0;
                    for (unsigned int v = 0; v < m_uiEVar.get_dof(); v++)
                        for (unsigned int k = 0; k < nPe; k++)
                            if (std::isnan(st[v * dgSz + e * nPe + k])) bad++;
                    if (!bad) continue;
                    const bool local = e >= pMesh->getElementLocalBegin() && e < pMesh->getElementLocalEnd();
                    const bool inR1 = std::find(r1.begin(), r1.end(), e) != r1.end();
                    std::cout << "[enuts-nan r" << pMesh->getMPIRankGlobal() << "]   dep elem " << e
                              << " " << pMesh->getAllElements()[e] << " bad=" << bad
                              << " local=" << local << " r1=" << inR1
                              << " etime=" << m_uiEleTime[e] << " edt=" << m_uiEleDT[e] << std::endl;
                }
            }
            MPI_Abort(pMesh->getMPICommunicator(), 7);
        }
    }
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::sync_blk_timestep(unsigned int blk,
                                             unsigned int rk_s,
                                             bool isIGTSync) {
#ifdef __PROFILE_ENUTS__
    m_uiPt[ENUTSPROFILE::BLK_SYNC].start();
#endif

    ot::Mesh* pMesh = (ot::Mesh*)m_uiAppCtx->get_mesh();
    if ((!(pMesh->isActive())) || m_uiBVec[blk]._vec[rk_s].isSynced()) return;

    const ot::TreeNode* pNodes  = pMesh->getAllElements().data();
    const ot::Block* blkList    = pMesh->getLocalBlockList().data();

    const unsigned int regLevel = blkList[blk].getRegularGridLev();
    const ot::TreeNode blkNode  = blkList[blk].getBlockNode();
    const unsigned int PW       = blkList[blk].get1DPadWidth();

    const unsigned int eOrder   = pMesh->getElementOrder();
    const unsigned int nPe      = pMesh->getNumNodesPerElement();

    const unsigned int* etVec   = m_uiEleTime.data();
    const unsigned int* dtELev  = m_uiEleDT.data();

    MPI_Comm comm               = pMesh->getMPICommunicator();

    const unsigned int lx       = blkList[blk].getAllocationSzX();
    const unsigned int ly       = blkList[blk].getAllocationSzY();
    const unsigned int lz       = blkList[blk].getAllocationSzZ();
    const unsigned int offset   = blkList[blk].getOffset();

    const unsigned int dgSz =
        pMesh->getAllElements().size() * pMesh->getNumNodesPerElement();
    const unsigned int cgSz = pMesh->getDegOfFreedom();
    const unsigned int unSz = pMesh->getDegOfFreedomUnZip();

    const unsigned int* e2n = pMesh->getE2NMapping().data();
    const unsigned int* e2e = pMesh->getE2EMapping().data();

    const unsigned int bLev =
        pNodes[blkList[blk].firstElementID()].getLevel();
    const unsigned int bTime = etVec[blkList[blk].firstElementID()];
    const unsigned int bDT   = dtELev[blkList[blk].firstElementID()];

    T dt;
    unsigned int tl  = 0;
    unsigned int dtl = 0;  // dt time level.
    unsigned int lookUp;
    const unsigned int cSz[3] = {eOrder + 1, eOrder + 1, eOrder + 1};

    unsigned int fchild[4];
    unsigned int echild[2];

    const unsigned int dof = m_uiEVar.get_dof();
    std::vector<T> cVec;
    cVec.resize(dof * nPe * (m_uiNumStages + 1));

    T* cVin[(m_uiNumStages + 1) * dof];   // correction vec. pointer in,
    T* cVout[(m_uiNumStages + 1) * dof];  // correction vec. pointer out,

    T* dgWVec  = m_uiDGTmp0.get_vec_ptr();
    // T * cgWVec = m_uiEVecTmp[0].get_vec_ptr();
    T* uzWVec  = m_uiEvarUzip.get_vec_ptr();

    T* m_uiVec = (T*)m_uiBVec[blk]._vec[rk_s].data();

    T* dgStages[m_uiNumStages];
    for (unsigned int i = 0; i < m_uiNumStages; i++)
        dgStages[i] = m_uiStVec[i].get_vec_ptr();

    T* dgEVar = m_uiEvarDG.get_vec_ptr();

    std::vector<unsigned int> eid;
    computeBlockUnzipDepElements(pMesh, blk, eid);

    if (rk_s == 0) {  // Sync operation for U(t,.)

        for (unsigned int m = 0; m < eid.size(); m++) {
            const unsigned int elem = eid[m];
            tl  = etVec[elem];   // this is the current time the block is at, in
                                 // the block loop time.
            dtl = dtELev[elem];  // this is time step size level used to get to
                                 // this time point.
            // You can use the element/block level to compute the future time
            // step size level.

            T t_blk = m_uiTimeInfo._m_uiT + bTime * m_uiTimeInfo._m_uiTh;
            T t_ele = m_uiTimeInfo._m_uiT + tl * m_uiTimeInfo._m_uiTh;

            if (tl == bTime) {
                // both elements are in the same time.
                dt = 0;
#pragma unroll
                for (unsigned int v = 0; v < dof; v++)
                    std::memcpy(dgWVec + v * dgSz + elem * nPe,
                                dgEVar + v * dgSz + elem * nPe,
                                sizeof(T) * nPe);

            } else if (tl > bTime) {
                assert(dtl >= bDT);
                // elem is in the future.
                const T dt_c = dtl * (m_uiTimeInfo._m_uiTh);
                const T dt_f = bDT * (m_uiTimeInfo._m_uiTh);
                dt           = (tl - bTime) * (m_uiTimeInfo._m_uiTh);

                for (unsigned int v = 0; v < dof; v++) {
                    for (unsigned int s = 0; s < m_uiNumStages; s++) {
                        cVin[v * (m_uiNumStages + 1) + s] =
                            dgStages[s] + v * dgSz + elem * nPe;
                        cVout[v * (m_uiNumStages + 1) + s] =
                            cVec.data() + v * (m_uiNumStages + 1) * nPe +
                            s * nPe;
                    }

                    cVin[v * (m_uiNumStages + 1) + m_uiNumStages] =
                        dgEVar + v * dgSz + elem * nPe;
                    cVout[v * (m_uiNumStages + 1) + m_uiNumStages] =
                        cVec.data() + v * (m_uiNumStages + 1) * nPe +
                        m_uiNumStages * nPe;
                }

#ifdef __PROFILE_ENUTS__
                m_uiPt[ENUTSPROFILE::NUTS_CORRECTION].start();
#endif

                m_uiECOp->coarser_finer_ut_correction(cVout, (const T**)cVin,
                                                      cSz, dt_c, dt_f, dt, dof);

#ifdef __PROFILE_ENUTS__
                m_uiPt[ENUTSPROFILE::NUTS_CORRECTION].stop();
#endif

#pragma unroll
                for (unsigned int v = 0; v < dof; v++)
                    std::memcpy(dgWVec + v * dgSz + elem * nPe,
                                cVout[v * (m_uiNumStages + 1) + m_uiNumStages],
                                sizeof(T) * nPe);

            } else {
                // elem is in the past.
                assert(tl < bTime);
                assert(dtl <= bDT);
                // elem is in the future.
                const T dt_c = bDT * (m_uiTimeInfo._m_uiTh);
                const T dt_f = dtl * (m_uiTimeInfo._m_uiTh);

                dt           = (bTime - tl) * (m_uiTimeInfo._m_uiTh);

                for (unsigned int v = 0; v < dof; v++) {
                    for (unsigned int s = 0; s < m_uiNumStages; s++) {
                        cVin[v * (m_uiNumStages + 1) + s] =
                            dgStages[s] + v * dgSz + elem * nPe;
                        cVout[v * (m_uiNumStages + 1) + s] =
                            cVec.data() + v * (m_uiNumStages + 1) * nPe +
                            s * nPe;
                    }

                    cVin[v * (m_uiNumStages + 1) + m_uiNumStages] =
                        dgEVar + v * dgSz + elem * nPe;
                    cVout[v * (m_uiNumStages + 1) + m_uiNumStages] =
                        cVec.data() + v * (m_uiNumStages + 1) * nPe +
                        m_uiNumStages * nPe;
                }

#ifdef __PROFILE_ENUTS__
                m_uiPt[ENUTSPROFILE::NUTS_CORRECTION].start();
#endif

                m_uiECOp->finer_coarser_ut_correction(cVout, (const T**)cVin,
                                                      cSz, dt_c, dt_f, dt, dof);

#ifdef __PROFILE_ENUTS__
                m_uiPt[ENUTSPROFILE::NUTS_CORRECTION].stop();
#endif

#pragma unroll
                for (unsigned int v = 0; v < dof; v++)
                    std::memcpy(dgWVec + v * dgSz + elem * nPe,
                                cVout[v * (m_uiNumStages + 1) + m_uiNumStages],
                                sizeof(T) * nPe);
            }
        }

    } else {
        assert(rk_s >= 1 && rk_s <= m_uiNumStages);
        assert(PW > 0);

        // apply time interpolation correction operators.
        for (unsigned int m = 0; m < eid.size(); m++) {
            const unsigned int elem = eid[m];
            tl  = etVec[elem];   // this is the current time the block is at, in
                                 // the block loop time.
            dtl = dtELev[elem];  // this is time step size level used to get to
                                 // this time point.
            // You can use the element/block level to compute the future time
            // step size level.

            T t_blk = m_uiTimeInfo._m_uiT + bTime * m_uiTimeInfo._m_uiTh;
            T t_ele = m_uiTimeInfo._m_uiT + tl * m_uiTimeInfo._m_uiTh;

            const T t_blk_prev = m_uiTimeInfo._m_uiT +
                                 bTime * m_uiTimeInfo._m_uiTh -
                                 bDT * m_uiTimeInfo._m_uiTh;
            const T t_ele_prev = m_uiTimeInfo._m_uiT +
                                 tl * m_uiTimeInfo._m_uiTh -
                                 dtl * m_uiTimeInfo._m_uiTh;

            if (isIGTSync) {
                // update where these stages are computed.
                t_blk = t_blk_prev;
                t_ele = t_ele_prev;

                if (fabs(t_ele - t_blk) <= __LTS_BLK_SYNC_TIME_COMP_TOL__) {
// no interpolation is needed.
#pragma unroll
                    for (unsigned int v = 0; v < dof; v++)
                        std::memcpy(dgWVec + v * dgSz + elem * nPe,
                                    dgStages[rk_s - 1] + v * dgSz + elem * nPe,
                                    sizeof(T) * nPe);

                } else if ((t_ele - t_blk) > __LTS_BLK_SYNC_TIME_COMP_TOL__) {
                    // elem is in future.
                    T dt_c = dtl * (m_uiTimeInfo._m_uiTh);
                    T dt_f = bDT * (m_uiTimeInfo._m_uiTh);

                    if (dt_c < dt_f) std::swap(dt_c, dt_f);

                    dt = (t_ele - t_blk);

                    for (unsigned int v = 0; v < dof; v++)
                        for (unsigned int s = 1; s <= rk_s; s++) {
                            cVin[v * rk_s + (s - 1)] =
                                dgStages[s - 1] + v * dgSz + elem * nPe;
                            cVout[v * rk_s + (s - 1)] =
                                cVec.data() + v * rk_s * nPe + (s - 1) * nPe;
                        }

#ifdef __PROFILE_ENUTS__
                    m_uiPt[ENUTSPROFILE::NUTS_CORRECTION].start();
#endif

                    m_uiECOp->Ccf(cVout, (const T**)cVin, cSz, rk_s, dt_c, dt_f,
                                  dt, dof);

#ifdef __PROFILE_ENUTS__
                    m_uiPt[ENUTSPROFILE::NUTS_CORRECTION].stop();
#endif

#pragma unroll
                    for (unsigned int v = 0; v < dof; v++)
                        std::memcpy(dgWVec + v * dgSz + elem * nPe,
                                    cVout[v * rk_s + (rk_s - 1)],
                                    sizeof(T) * nPe);
                } else if ((t_blk - t_ele) > __LTS_BLK_SYNC_TIME_COMP_TOL__) {
                    // blk is in future
                    T dt_c = (bDT) * (m_uiTimeInfo._m_uiTh);
                    T dt_f = (dtl) * (m_uiTimeInfo._m_uiTh);

                    if (dt_c < dt_f) std::swap(dt_c, dt_f);

                    dt = (t_blk - t_ele);

                    for (unsigned int v = 0; v < dof; v++)
                        for (unsigned int s = 1; s <= rk_s; s++) {
                            cVin[v * rk_s + (s - 1)] =
                                dgStages[s - 1] + v * dgSz + elem * nPe;
                            cVout[v * rk_s + (s - 1)] =
                                cVec.data() + v * rk_s * nPe + (s - 1) * nPe;
                        }

#ifdef __PROFILE_ENUTS__
                    m_uiPt[ENUTSPROFILE::NUTS_CORRECTION].start();
#endif

                    m_uiECOp->Cfc(cVout, (const T**)cVin, cSz, rk_s, dt_c, dt_f,
                                  dt, dof);

#ifdef __PROFILE_ENUTS__
                    m_uiPt[ENUTSPROFILE::NUTS_CORRECTION].stop();
#endif

#pragma unroll
                    for (unsigned int v = 0; v < dof; v++)
                        std::memcpy(dgWVec + v * dgSz + elem * nPe,
                                    cVout[v * rk_s + (rk_s - 1)],
                                    sizeof(T) * nPe);
                }

            } else {
                if (bTime == tl) {
// no interpolation is needed.
#pragma unroll
                    for (unsigned int v = 0; v < dof; v++)
                        std::memcpy(dgWVec + v * dgSz + elem * nPe,
                                    dgStages[rk_s - 1] + v * dgSz + elem * nPe,
                                    sizeof(T) * nPe);

                } else if (tl > bTime) {
                    // elem is in future.
                    T dt_c = dtl * (m_uiTimeInfo._m_uiTh);
                    T dt_f = bDT * (m_uiTimeInfo._m_uiTh);

                    if (dt_c < dt_f) std::swap(dt_c, dt_f);

                    dt = (t_ele - t_blk);

                    for (unsigned int v = 0; v < dof; v++)
                        for (unsigned int s = 1; s <= rk_s; s++) {
                            cVin[v * rk_s + (s - 1)] =
                                dgStages[s - 1] + v * dgSz + elem * nPe;
                            cVout[v * rk_s + (s - 1)] =
                                cVec.data() + v * rk_s * nPe + (s - 1) * nPe;
                        }

#ifdef __PROFILE_ENUTS__
                    m_uiPt[ENUTSPROFILE::NUTS_CORRECTION].start();
#endif

                    m_uiECOp->Ccf(cVout, (const T**)cVin, cSz, rk_s, dt_c, dt_f,
                                  dt, dof);

#ifdef __PROFILE_ENUTS__
                    m_uiPt[ENUTSPROFILE::NUTS_CORRECTION].stop();
#endif

#pragma unroll
                    for (unsigned int v = 0; v < dof; v++)
                        std::memcpy(dgWVec + v * dgSz + elem * nPe,
                                    cVout[v * rk_s + (rk_s - 1)],
                                    sizeof(T) * nPe);
                } else {
                    assert(tl < bTime);
                    // blk is in future
                    T dt_c = (bDT) * (m_uiTimeInfo._m_uiTh);
                    T dt_f = (dtl) * (m_uiTimeInfo._m_uiTh);

                    if (dt_c < dt_f) std::swap(dt_c, dt_f);

                    dt = (t_blk - t_ele);

                    for (unsigned int v = 0; v < dof; v++)
                        for (unsigned int s = 1; s <= rk_s; s++) {
                            cVin[v * rk_s + (s - 1)] =
                                dgStages[s - 1] + v * dgSz + elem * nPe;
                            cVout[v * rk_s + (s - 1)] =
                                cVec.data() + v * rk_s * nPe + (s - 1) * nPe;
                        }

#ifdef __PROFILE_ENUTS__
                    m_uiPt[ENUTSPROFILE::NUTS_CORRECTION].start();
#endif

                    m_uiECOp->Cfc(cVout, (const T**)cVin, cSz, rk_s, dt_c, dt_f,
                                  dt, dof);

#ifdef __PROFILE_ENUTS__
                    m_uiPt[ENUTSPROFILE::NUTS_CORRECTION].stop();
#endif

#pragma unroll
                    for (unsigned int v = 0; v < dof; v++)
                        std::memcpy(dgWVec + v * dgSz + elem * nPe,
                                    cVout[v * rk_s + (rk_s - 1)],
                                    sizeof(T) * nPe);
                }
            }
        }

        // for(unsigned int elem = blkList[blk].getLocalElementBegin(); elem <
        // blkList[blk].getLocalElementEnd(); elem++)
        // {
        //     const unsigned int
        //     ei=(pNodes[elem].getX()-blkNode.getX())>>(m_uiMaxDepth-regLevel);
        //     const unsigned int
        //     ej=(pNodes[elem].getY()-blkNode.getY())>>(m_uiMaxDepth-regLevel);
        //     const unsigned int
        //     ek=(pNodes[elem].getZ()-blkNode.getZ())>>(m_uiMaxDepth-regLevel);

        //     const unsigned int emin = 0;
        //     const unsigned int emax = (1u<<(regLevel-blkNode.getLevel()))-1;

        //     assert(etVec[elem] == bTime );

        //     #pragma unroll
        //     for(unsigned int v=0; v < dof; v++)
        //         std::memcpy(dgWVec + v*dgSz + elem * nPe , dgStages[rk_s-1] +
        //         v*dgSz + elem * nPe, sizeof(T)*nPe );

        // }

        // for(unsigned int elem = blkList[blk].getLocalElementBegin(); elem <
        // blkList[blk].getLocalElementEnd(); elem++)
        //     eid.push_back(elem);
        // pMesh->DG2CGVec(dgWVec,cgWVec,true,eid.data(),eid.size(),dof);
        // pMesh->unzip(cgWVec,uzWVec, &blk, 1, dof);

        // #pragma unroll
        // for(unsigned int v=0; v < dof; v++)
        //     std::memcpy(m_uiVec + v*lx*ly*lz , uzWVec + v* unSz + offset,
        //     sizeof(T)*(lx*ly*lz));

        // return;
    }

    // now need to copy to the block unzip/ block asyncVector
    const double hx   = (1u << (m_uiMaxDepth - bLev)) / (double)eOrder;

    const double xmin = blkNode.minX() - PW * hx;
    const double xmax = blkNode.maxX() + PW * hx;
    const double ymin = blkNode.minY() - PW * hx;
    const double ymax = blkNode.maxY() + PW * hx;
    const double zmin = blkNode.minZ() - PW * hx;
    const double zmax = blkNode.maxZ() + PW * hx;

    std::vector<ot::TreeNode> childOct;
    childOct.reserve(NUM_CHILDREN);

    std::vector<T> p2cI;
    p2cI.resize(nPe);

    const double d_compar_tol = 1e-6;

    for (unsigned int m = 0; m < eid.size(); m++) {
        const unsigned int ele = eid[m];

        // no interpolation needed just copy.
        if (pNodes[ele].getLevel() == bLev) {
            const double hh = (1u << (m_uiMaxDepth - pNodes[ele].getLevel())) /
                              (double)eOrder;
            const double invhh = 1.0 / hh;

            for (unsigned int k = 0; k < eOrder + 1; k++) {
                double zz = pNodes[ele].minZ() + k * hh;

                if (fabs(zz - zmin) < d_compar_tol) zz = zmin;
                if (fabs(zz - zmax) < d_compar_tol) zz = zmax;

                if (zz < zmin || zz > zmax) continue;
                const unsigned int kkz = std::round((zz - zmin) * invhh);
                assert(std::fabs(zz - zmin - kkz * hh) < 1e-6);
                assert(kkz >= 0 && kkz < lz);

                for (unsigned int j = 0; j < eOrder + 1; j++) {
                    double yy = pNodes[ele].minY() + j * hh;

                    if (fabs(yy - ymin) < d_compar_tol) yy = ymin;
                    if (fabs(yy - ymax) < d_compar_tol) yy = ymax;

                    if (yy < ymin || yy > ymax) continue;
                    const unsigned int jjy = std::round((yy - ymin) * invhh);
                    // std::cout<<"yy: "<<yy<<" (ymin + hh*jjy): "<<(ymin +
                    // hh*jjy)<<std::endl;
                    assert(std::fabs(yy - ymin - jjy * hh) < 1e-6);
                    assert(jjy >= 0 && jjy < ly);

                    for (unsigned int i = 0; i < eOrder + 1; i++) {
                        double xx = pNodes[ele].minX() + i * hh;

                        if (fabs(xx - xmin) < d_compar_tol) xx = xmin;
                        if (fabs(xx - xmax) < d_compar_tol) xx = xmax;

                        if (xx < xmin || xx > xmax) continue;
                        const unsigned int iix =
                            std::round((xx - xmin) * invhh);
                        assert(std::fabs(xx - xmin - iix * hh) < 1e-6);
                        assert(iix >= 0 && iix < lx);

                        // std::cout<<"blk: "<<blk<<" copy : (i,j,k): ("<<kkz<<"
                        // , "<<jjy<<", "<<iix<<")"<<" of : "<<lx<<std::endl;

                        // if(blkNode.isAncestor(pNodes[ele]))
                        // {
                        //     const unsigned int
                        //     ei=(pNodes[ele].getX()-blkNode.getX())>>(m_uiMaxDepth-regLevel);
                        //     const unsigned int
                        //     ej=(pNodes[ele].getY()-blkNode.getY())>>(m_uiMaxDepth-regLevel);
                        //     const unsigned int
                        //     ek=(pNodes[ele].getZ()-blkNode.getZ())>>(m_uiMaxDepth-regLevel);

                        //     std::cout<<"blk: "<<blk<<" copy : (i,j,k):
                        //     ("<<iix<<" , "<<jjy<<", "<<kkz<<")"<<" of :
                        //     "<<lx<<" ei: "<<ei<<" ej: "<<ej<<" ek: "<<ek << "
                        //     xx: "<<xx<<" yy: "<<yy<<" zz:"<<zz<<" xmin:
                        //     "<<xmin<<" ymin: "<<ymin<<" zmin: "<<zmin<<" hh :
                        //     "<<hh<<" hhx : "<<hx<<std::endl;

                        // }

                        for (unsigned int v = 0; v < dof; v++) {
                            // double v1 = m_uiVec[(v*lx*ly*lz) + kkz*lx*ly +
                            // jjy*lx + iix]; double v2 = dgWVec[v*dgSz +
                            // ele*nPe + k*(eOrder+1)*(eOrder+1)+ j*(eOrder+1) +
                            // i]; if(rk_s==0 && fabs(v2-v1)>1e-3)
                            // {
                            //     std::cout<<"var : "<<v<<" equal level copy:
                            //     "<<blk<<" old : "<<v1<<" new :
                            //     "<<v2<<std::endl;
                            // }
                            m_uiVec[(v * lx * ly * lz) + kkz * lx * ly +
                                    jjy * lx + iix] =
                                dgWVec[v * dgSz + ele * nPe +
                                       k * (eOrder + 1) * (eOrder + 1) +
                                       j * (eOrder + 1) + i];
                        }
                    }
                }
            }

        } else if (pNodes[ele].getLevel() > bLev) {
            assert((bLev + 1) == pNodes[ele].getLevel());
            const unsigned int cnum = pNodes[ele].getMortonIndex();
            ot::TreeNode tmpParent  = pNodes[ele].getParent();

            const double hh = (1u << (m_uiMaxDepth - pNodes[ele].getLevel())) /
                              (double)eOrder;
            const double invhh = 1.0 / (2 * hh);

            assert(eOrder > 1);
            const unsigned int cb = (eOrder % 2 == 0) ? 0 : 1;

            for (unsigned int k = cb; k < eOrder + 1; k += 2) {
                double zz = (pNodes[ele].minZ() + k * hh);
                if (fabs(zz - zmin) < d_compar_tol) zz = zmin;
                if (fabs(zz - zmax) < d_compar_tol) zz = zmax;

                if (zz < zmin || zz > zmax) continue;
                const unsigned int kkz = std::round((zz - zmin) * invhh);
                assert(kkz >= 0 && kkz < lz);

                for (unsigned int j = cb; j < eOrder + 1; j += 2) {
                    double yy = pNodes[ele].minY() + j * hh;

                    if (fabs(yy - ymin) < d_compar_tol) yy = ymin;
                    if (fabs(yy - ymax) < d_compar_tol) yy = ymax;

                    if (yy < ymin || yy > ymax) continue;

                    const unsigned int jjy = std::round((yy - ymin) * invhh);
                    assert(jjy >= 0 && jjy < ly);

                    for (unsigned int i = cb; i < eOrder + 1; i += 2) {
                        double xx = pNodes[ele].minX() + i * hh;

                        if (fabs(xx - xmin) < d_compar_tol) xx = xmin;
                        if (fabs(xx - xmax) < d_compar_tol) xx = xmax;

                        if (xx < xmin || xx > xmax) continue;
                        const unsigned int iix =
                            std::round((xx - xmin) * invhh);
                        assert(iix >= 0 && iix < lx);

                        // if(blkNode.isAncestor(pNodes[ele]))
                        // {
                        //     const unsigned int
                        //     ei=(pNodes[ele].getX()-blkNode.getX())>>(m_uiMaxDepth-regLevel);
                        //     const unsigned int
                        //     ej=(pNodes[ele].getY()-blkNode.getY())>>(m_uiMaxDepth-regLevel);
                        //     const unsigned int
                        //     ek=(pNodes[ele].getZ()-blkNode.getZ())>>(m_uiMaxDepth-regLevel);

                        //     std::cout<<"blk: "<<blk<<" copy : (i,j,k):
                        //     ("<<iix<<" , "<<jjy<<", "<<kkz<<")"<<" of :
                        //     "<<lx<<" ei: "<<ei<<" ej: "<<ej<<" ek: "<<ek << "
                        //     xx: "<<xx<<" yy: "<<yy<<" zz:"<<zz<<" xmin:
                        //     "<<xmin<<" ymin: "<<ymin<<" zmin: "<<zmin<<" hh :
                        //     "<<hh<<" hhx : "<<hx<<std::endl;

                        // }

                        // std::cout<<"blk: "<<blk<<" blk copy : (i,j,k):
                        // ("<<iix<<" , "<<jjy<<", "<<kkz<<")"<<" of : "<<lx<<"
                        // xx: "<<xx<<" yy: "<<yy<<" zz:"<<zz<<" xmin:
                        // "<<xmin<<" ymin: "<<ymin<<" zmin: "<<zmin<<" hh :
                        // "<<hh<<" hhx : "<<hx<<std::endl;
                        for (unsigned int v = 0; v < dof; v++) {
                            // double v1 = m_uiVec[(v*lx*ly*lz) + kkz*lx*ly +
                            // jjy*lx + iix]; double v2 = dgWVec[v*dgSz +
                            // ele*nPe + k*(eOrder+1)*(eOrder+1)+ j*(eOrder+1) +
                            // i]; if(rk_s==0 && fabs(v2-v1)>1e-3)
                            // {
                            //     std::cout<<"var : "<<v<<" copy from finer
                            //     octant: "<<blk<<" old : "<<v1<<" new :
                            //     "<<v2<<std::endl;
                            // }
                            m_uiVec[(v * lx * ly * lz) + kkz * lx * ly +
                                    jjy * lx + iix] =
                                dgWVec[v * dgSz + ele * nPe +
                                       k * (eOrder + 1) * (eOrder + 1) +
                                       j * (eOrder + 1) + i];
                        }
                    }
                }
            }

        } else {
            assert((bLev) == (pNodes[ele].getLevel() + 1));
            childOct.clear();
            pNodes[ele].addChildren(
                childOct);  // note this is the ordering of SFC (depends on
                            // Hilbert or Morton. )

            for (unsigned int child = 0; child < NUM_CHILDREN; child++) {
                if ((childOct[child].maxX() < xmin ||
                     childOct[child].minX() >= xmax) ||
                    (childOct[child].maxY() < ymin ||
                     childOct[child].minY() >= ymax) ||
                    (childOct[child].maxZ() < zmin ||
                     childOct[child].minZ() >= zmax))
                    continue;

                // std::cout<<"blk: "<<blk<<" blkNode: "<<blkNode<<" child:
                // "<<child<<" child node "<<childOct[child]<<" parent :
                // "<<pNodes[ele]<<std::endl;
                const double hh =
                    (1u << (m_uiMaxDepth - childOct[child].getLevel())) /
                    (double)eOrder;
                const double invhh = 1.0 / hh;

                for (unsigned int v = 0; v < dof; v++) {
                    const unsigned int cnum = childOct[child].getMortonIndex();
                    pMesh->parent2ChildInterpolation(
                        &dgWVec[v * dgSz + ele * nPe], p2cI.data(), cnum,
                        m_uiDim);

                    for (unsigned int k = 0; k < eOrder + 1; k++) {
                        double zz = childOct[child].minZ() + k * hh;

                        if (fabs(zz - zmin) < d_compar_tol) zz = zmin;
                        if (fabs(zz - zmax) < d_compar_tol) zz = zmax;

                        if (zz < zmin || zz > zmax) continue;
                        const unsigned int kkz =
                            std::round((zz - zmin) * invhh);
                        assert(kkz >= 0 && kkz < lz);

                        for (unsigned int j = 0; j < eOrder + 1; j++) {
                            double yy = childOct[child].minY() + j * hh;

                            if (fabs(yy - ymin) < d_compar_tol) yy = ymin;
                            if (fabs(yy - ymax) < d_compar_tol) yy = ymax;

                            if (yy < ymin || yy > ymax) continue;

                            const unsigned int jjy =
                                std::round((yy - ymin) * invhh);
                            assert(jjy >= 0 && jjy < ly);

                            for (unsigned int i = 0; i < eOrder + 1; i++) {
                                double xx = childOct[child].minX() + i * hh;

                                if (fabs(xx - xmin) < d_compar_tol) xx = xmin;
                                if (fabs(xx - xmax) < d_compar_tol) xx = xmax;

                                if (xx < xmin || xx > xmax) continue;
                                const unsigned int iix =
                                    std::round((xx - xmin) * invhh);
                                assert(iix >= 0 && iix < lx);
                                // std::cout<<"blk: "<<blk<<" blk copy :
                                // (i,j,k): ("<<iix<<" , "<<jjy<<",
                                // "<<kkz<<")"<<" of : "<<lx<<" xx: "<<xx<<" yy:
                                // "<<yy<<" zz:"<<zz<<" xmin: "<<xmin<<" ymin:
                                // "<<ymin<<" zmin: "<<zmin<<" hh : "<<hh<<" hhx
                                // : "<<hx<<" child: "<<child<<std::endl;
                                //  double v1 = m_uiVec[(v*lx*ly*lz) + kkz*lx*ly
                                //  + jjy*lx + iix]; double v2 =
                                //  p2cI[k*(eOrder+1)*(eOrder+1)+ j*(eOrder+1) +
                                //  i]; if(rk_s==0 && fabs(v2-v1)>1e-3)
                                //  {
                                //      std::cout<<"var : "<<v<<" copy from
                                //      coaser octant(p2c): "<<blk<<" old :
                                //      "<<v1<<" new : "<<v2<<std::endl;
                                //  }
                                m_uiVec[(v * lx * ly * lz) + kkz * lx * ly +
                                        jjy * lx + iix] =
                                    p2cI[k * (eOrder + 1) * (eOrder + 1) +
                                         j * (eOrder + 1) + i];
                            }
                        }
                    }
                }
            }
        }
    }

    // block internal copy.
    if (rk_s == 0) {
        for (const unsigned int elem : blkList[blk]) {
            const unsigned int ei = (pNodes[elem].getX() - blkNode.getX()) >>
                                    (m_uiMaxDepth - regLevel);
            const unsigned int ej = (pNodes[elem].getY() - blkNode.getY()) >>
                                    (m_uiMaxDepth - regLevel);
            const unsigned int ek = (pNodes[elem].getZ() - blkNode.getZ()) >>
                                    (m_uiMaxDepth - regLevel);

            const unsigned int emin = 0;
            const unsigned int emax =
                (1u << (regLevel - blkNode.getLevel())) - 1;

            assert(etVec[elem] == bTime);

            // #pragma unroll
            // for(unsigned int v=0; v < dof; v++)
            //     std::memcpy(dgWVec + v*dgSz + elem * nPe , dgStages[rk_s-1] +
            //     v*dgSz + elem * nPe, sizeof(T)*nPe );

            for (unsigned int v = 0; v < dof; v++)
                for (unsigned int k = 0; k < (eOrder + 1); k++)
                    for (unsigned int j = 0; j < (eOrder + 1); j++)
                        for (unsigned int i = 0; i < (eOrder + 1); i++) {
                            // double v1 = m_uiVec[v*lx*ly*lz +
                            // (ek*eOrder+k+PW)*(ly*lx)+(ej*eOrder+j+PW)*(lx)+(ei*eOrder+i+PW)];
                            // double v2 = dgEVar[ v*dgSz + elem * nPe +
                            // k*(eOrder+1)*(eOrder+1) + j*(eOrder+1) + i];

                            // if(fabs(v1-v2)>1e-3)
                            //     std::cout<<"i: "<<i<< "j: "<<j<<" k: "<<k<<"
                            //     [internal]: v1(old): "<<v1<<" v2:(new):
                            //     "<<v2<<std::endl;

                            m_uiVec[v * lx * ly * lz +
                                    (ek * eOrder + k + PW) * (ly * lx) +
                                    (ej * eOrder + j + PW) * (lx) +
                                    (ei * eOrder + i + PW)] =
                                dgEVar[v * dgSz + elem * nPe +
                                       k * (eOrder + 1) * (eOrder + 1) +
                                       j * (eOrder + 1) + i];
                        }
        }

    } else {
        for (const unsigned int elem : blkList[blk]) {
            const unsigned int ei = (pNodes[elem].getX() - blkNode.getX()) >>
                                    (m_uiMaxDepth - regLevel);
            const unsigned int ej = (pNodes[elem].getY() - blkNode.getY()) >>
                                    (m_uiMaxDepth - regLevel);
            const unsigned int ek = (pNodes[elem].getZ() - blkNode.getZ()) >>
                                    (m_uiMaxDepth - regLevel);

            const unsigned int emin = 0;
            const unsigned int emax =
                (1u << (regLevel - blkNode.getLevel())) - 1;

            assert(etVec[elem] == bTime);

            // #pragma unroll
            // for(unsigned int v=0; v < dof; v++)
            //     std::memcpy(dgWVec + v*dgSz + elem * nPe , dgStages[rk_s-1] +
            //     v*dgSz + elem * nPe, sizeof(T)*nPe );

            for (unsigned int v = 0; v < dof; v++)
                for (unsigned int k = 0; k < (eOrder + 1); k++)
                    for (unsigned int j = 0; j < (eOrder + 1); j++)
                        for (unsigned int i = 0; i < (eOrder + 1); i++)
                            m_uiVec[v * lx * ly * lz +
                                    (ek * eOrder + k + PW) * (ly * lx) +
                                    (ej * eOrder + j + PW) * (lx) +
                                    (ei * eOrder + i + PW)] =
                                dgStages[rk_s - 1]
                                        [v * dgSz + elem * nPe +
                                         k * (eOrder + 1) * (eOrder + 1) +
                                         j * (eOrder + 1) + i];
        }
    }

#ifdef __PROFILE_ENUTS__
    m_uiPt[ENUTSPROFILE::BLK_SYNC].stop();
#endif
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::update_ele_timestep() {
    ot::Mesh* pMesh = m_uiAppCtx->get_mesh();

    if (!(pMesh->isActive())) return;

    const ot::TreeNode* pNodes            = pMesh->getAllElements().data();
    const std::vector<ot::Block>& blkList = pMesh->getLocalBlockList();
    for (unsigned int blk = 0; blk < blkList.size(); blk++) {
        const unsigned int bLev =
            pNodes[blkList[blk].firstElementID()].getLevel();
        const unsigned int BLK_DT =
            m_uiAppCtx->getBlkTimestepFac(bLev, m_uiLevMin, m_uiLevMax);

        for (const unsigned int ele : blkList[blk]) {
            m_uiEleTime[ele] = m_uiBVec[blk]._time;
            m_uiEleDT[ele]   = BLK_DT;
        }
    }
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::partial_evolve(unsigned int pt) {
    ot::Mesh* pMesh = m_uiAppCtx->get_mesh();

    if (!(pMesh->isActive())) return;

    const std::vector<ot::Block>& blkList = pMesh->getLocalBlockList();
    const ot::TreeNode* pNodes            = pMesh->getAllElements().data();
    const double current_t                = m_uiTimeInfo._m_uiT;
    const unsigned int DOF                = m_uiEVar.get_dof();

    double current_t_adv                  = current_t;

    const unsigned int finest_t           = m_uiAppCtx->getBlkTimestepFac(
        m_uiLevMax, m_uiLevMin, m_uiLevMax);  // finest  time level.
    const unsigned int coarset_t =
        m_uiAppCtx->getBlkTimestepFac(m_uiLevMin, m_uiLevMin, m_uiLevMax);

    const double dt_finest  = m_uiTimeInfo._m_uiTh;
    const double dt_coarset = coarset_t * m_uiTimeInfo._m_uiTh;

    assert(blkList.size() > 0);

    // note that this is needed if we are planing to perform parital remesh.
    for (unsigned int blk = 0; blk < blkList.size(); blk++) {
        const unsigned int bLev =
            pNodes[blkList[blk].firstElementID()].getLevel();
        m_uiBlkTimeLevMinMax[2 * bLev + 0] = LOOK_UP_TABLE_DEFAULT;
        m_uiBlkTimeLevMinMax[2 * bLev + 1] = 0;
    }

    for (unsigned int blk = 0; blk < blkList.size(); blk++) {
        const unsigned int BLK_T = m_uiBVec[blk]._time;
        const unsigned int bLev =
            pNodes[blkList[blk].firstElementID()].getLevel();

        if (BLK_T < m_uiBlkTimeLevMinMax[2 * bLev + 0])
            m_uiBlkTimeLevMinMax[2 * bLev + 0] = BLK_T;

        if (BLK_T > m_uiBlkTimeLevMinMax[2 * bLev + 1])
            m_uiBlkTimeLevMinMax[2 * bLev + 1] = BLK_T;
    }

    m_uiActiveBlkIDs.clear();
    m_uiIndependentActiveBlkIDs.clear();
    m_uiDependentActiveBlkIDs.clear();

    // compute the active block list for this partial timestep.
    for (unsigned int blk = 0; blk < blkList.size(); blk++) {
        const unsigned int BLK_T = m_uiBVec[blk]._time;
        const unsigned int NN    = m_uiBVec[blk]._vec[0].getSz();
        const unsigned int bLev =
            pNodes[blkList[blk].firstElementID()].getLevel();
        const unsigned int BLK_DT =
            m_uiAppCtx->getBlkTimestepFac(bLev, m_uiLevMin, m_uiLevMax);

        const T blk_dt   = m_uiTimeInfo._m_uiTh * BLK_DT;

        bool isBlkSynced = true;
        if (m_uiBlkTimeLevMinMax[2 * bLev + 0] !=
            m_uiBlkTimeLevMinMax[2 * bLev + 1])
            isBlkSynced = false;

        // if(!isBlkSynced)
        //     std::cout<<"isBlkSynced : "<<isBlkSynced<<" blk_lev:"<<bLev<<"
        //     blk min : "<<m_uiBlkTimeLevMinMax[2*bLev+0]<<" blk max "<<
        //     m_uiBlkTimeLevMinMax[2*bLev+1]<<std::endl;
        // skip blocks is they are not synced, and the time is equal to the max
        // time.
        if ((!isBlkSynced && BLK_T == m_uiBlkTimeLevMinMax[2 * bLev + 1]) ||
            (pt % BLK_DT != 0))
            continue;

        m_uiActiveBlkIDs.push_back(blk);

        // 02/12/2021 : The overlapped ghost exchange has a bug.
        // if(blkList[blk].getBlockType()==ot::BlockType::UNZIP_INDEPENDENT)
        // {
        //     m_uiIndependentActiveBlkIDs.push_back(blk);
        // }else
        // {
        //     m_uiDependentActiveBlkIDs.push_back(blk);
        // }
    }

    // std::cout<<"rank: "<<pMesh->getMPIRank()<<"pt: "<<pt<<" active block
    // size: "<<m_uiActiveBlkIDs.size()<<std::endl;

    // 10/07/2020 (Milinda): Synchronize data u(x,t)
    // for(unsigned int bb=0; bb < m_uiActiveBlkIDs.size(); bb++)
    // {
    //     const unsigned int blk = m_uiActiveBlkIDs[bb];
    //     m_uiBVec[blk]._vec[0].zipDG(pMesh,m_uiEvarDG.get_vec_ptr(),DOF);
    // }

    // // Now need to synchronize the out vector just to smooth out solution
    // between refinement boundaries.
    // pMesh->readFromGhostBeginEleDGVec(m_uiEvarDG.get_vec_ptr(),DOF);
    // pMesh->readFromGhostEndEleDGVec(m_uiEvarDG.get_vec_ptr(),DOF);

    // for(unsigned int bb=0; bb < m_uiActiveBlkIDs.size(); bb++)
    // {
    //     const unsigned int blk = m_uiActiveBlkIDs[bb];
    //     const unsigned int BLK_T = m_uiBVec[blk]._time;
    //     const unsigned int NN   =  m_uiBVec[blk]._vec[0].getSz();
    //     const unsigned int bLev =
    //     pNodes[blkList[blk].firstElementID()].getLevel(); const
    //     unsigned int BLK_DT =
    //     m_uiAppCtx->getBlkTimestepFac(bLev,m_uiLevMin,m_uiLevMax);

    //     m_uiBVec[blk]._vec[0].mark_unsynced();
    //     sync_blk_timestep(blk,0);
    //     m_uiBVec[blk]._vec[0].mark_synced();

    // }

    for (unsigned int rk = 1; rk <= m_uiNumStages; rk++) {
        const unsigned int BLK_S = rk;
        for (unsigned int bb = 0; bb < m_uiActiveBlkIDs.size(); bb++) {
            const unsigned int blk   = m_uiActiveBlkIDs[bb];
            const unsigned int BLK_T = m_uiBVec[blk]._time;
            const unsigned int NN    = m_uiBVec[blk]._vec[0].getSz();
            const unsigned int bLev =
                pNodes[blkList[blk].firstElementID()].getLevel();
            const unsigned int BLK_DT =
                m_uiAppCtx->getBlkTimestepFac(bLev, m_uiLevMin, m_uiLevMax);

            const T blk_dt = m_uiTimeInfo._m_uiTh * BLK_DT;

            assert(m_uiBVec[blk]._vec[rk - 1].isSynced());
            m_uiAppCtx->pre_timestep_blk((T*)m_uiBVec[blk]._vec[0].data(), DOF,
                                         blk, current_t + dt_finest * BLK_T);

            m_uiBVec[blk]._vec[m_uiNumStages + 1].vecCopy(
                m_uiBVec[blk]._vec[0]);  // copy the in vector.
            T* out_ptr = (T*)m_uiBVec[blk]._vec[m_uiNumStages + 1].data();

            for (unsigned int s = 1; s < BLK_S; s++) {
                const T* stage_ptr = m_uiBVec[blk]._vec[s].data();
                for (unsigned int n = 0; n < DOF * NN; n++) {
                    out_ptr[n] +=
                        (blk_dt *
                         m_uiAij[(BLK_S - 1) * m_uiNumStages + (s - 1)] *
                         stage_ptr[n]);
                    // printf("s[%d]: %f\n", n, out_ptr[n]);
                    // printf("s: %d : stage[%d]: %f\n", s,  n, stage_ptr[n]);
                }
            }

            m_uiAppCtx->pre_stage_blk(
                (T*)m_uiBVec[blk]._vec[m_uiNumStages + 1].data(), DOF, blk,
                current_t + dt_finest * BLK_T);
            debug_nan_check("pre-rhs-in", blk, m_uiNumStages + 1, pt, rk);
            debug_blk_hash("pre-rhs-in", blk, m_uiNumStages + 1, pt, rk);
            m_uiBVec[blk]._vec[BLK_S].computeVec(
                m_uiAppCtx, m_uiBVec[blk]._vec[m_uiNumStages + 1],
                current_t + dt_finest * BLK_T);
            debug_nan_check("post-rhs", blk, BLK_S, pt, rk);
            debug_blk_hash("post-rhs", blk, BLK_S, pt, rk);
            m_uiBVec[blk]._vec[BLK_S].mark_unsynced();
            // std::cout<<"rk:"<<rk<<std::endl;
            // m_uiBVec[blk]._vec[BLK_S].dump_vec(std::cout);

#ifdef __PROFILE_ENUTS__
            m_uiPt[ENUTSPROFILE::ENUTS_BLK_ZIP].start();
#endif

            m_uiBVec[blk]._vec[BLK_S].zipDG(
                pMesh, m_uiStVec[BLK_S - 1].get_vec_ptr(), DOF);
            m_uiBVec[blk]._rks = rk;

#ifdef __PROFILE_ENUTS__
            m_uiPt[ENUTSPROFILE::ENUTS_BLK_ZIP].stop();
#endif
        }

        pMesh->readFromGhostBeginEleDGVec(m_uiStVec[BLK_S - 1].get_vec_ptr(),
                                          DOF);
        pMesh->readFromGhostEndEleDGVec(m_uiStVec[BLK_S - 1].get_vec_ptr(),
                                        DOF);

        for (unsigned int bb = 0; bb < m_uiActiveBlkIDs.size(); bb++) {
            const unsigned int blk   = m_uiActiveBlkIDs[bb];
            const unsigned int BLK_T = m_uiBVec[blk]._time;
            const unsigned int NN    = m_uiBVec[blk]._vec[0].getSz();
            const unsigned int bLev =
                pNodes[blkList[blk].firstElementID()].getLevel();
            const unsigned int BLK_DT =
                m_uiAppCtx->getBlkTimestepFac(bLev, m_uiLevMin, m_uiLevMax);

            sync_blk_timestep(blk, BLK_S);
            debug_nan_check("post-sync", blk, BLK_S, pt, rk);
            debug_blk_hash("post-sync", blk, BLK_S, pt, rk);
            m_uiAppCtx->post_stage_blk((T*)m_uiBVec[blk]._vec[BLK_S].data(),
                                       DOF, blk, current_t + dt_finest * BLK_T);
            m_uiBVec[blk]._vec[BLK_S].mark_synced();
            // std::cout<<"[NUTS]: pt :  "<<pt<<" blk : "<<blk<<" rk : "<<rk<<"
            // step size: "<<BLK_DT<<" is synced:
            // "<<m_uiBVec[blk]._vec[BLK_S].isSynced()<<std::endl;
        }

        // 02/12/2021 : The overlapped ghost exchange has a bug. - look at line
        // 1768 when uncommenting
        /*
        // do the DG vec ghost sync.
        pMesh->readFromGhostBeginEleDGVec(m_uiStVec[BLK_S-1].get_vec_ptr(),DOF);



        for(unsigned int bb=0; bb < m_uiIndependentActiveBlkIDs.size(); bb++)
        {
            const unsigned int blk = m_uiIndependentActiveBlkIDs[bb];
            const unsigned int BLK_T = m_uiBVec[blk]._time;
            const unsigned int NN   =  m_uiBVec[blk]._vec[0].getSz();
            const unsigned int bLev =
        pNodes[blkList[blk].firstElementID()].getLevel(); const unsigned
        int BLK_DT = m_uiAppCtx->getBlkTimestepFac(bLev,m_uiLevMin,m_uiLevMax);

            sync_blk_timestep(blk,BLK_S);
            m_uiAppCtx->post_stage_blk((T*)m_uiBVec[blk]._vec[BLK_S].data(),DOF,blk,current_t
        + dt_finest*BLK_T); m_uiBVec[blk]._vec[BLK_S].mark_synced();
            //std::cout<<"[NUTS]: pt :  "<<pt<<" blk : "<<blk<<" rk : "<<rk<<"
        step size: "<<BLK_DT<<" is synced:
        "<<m_uiBVec[blk]._vec[BLK_S].isSynced()<<std::endl;

        }

        pMesh->readFromGhostEndEleDGVec(m_uiStVec[BLK_S-1].get_vec_ptr(),DOF);

        for(unsigned int bb=0; bb < m_uiDependentActiveBlkIDs.size(); bb++)
        {
            const unsigned int blk = m_uiDependentActiveBlkIDs[bb];
            const unsigned int BLK_T = m_uiBVec[blk]._time;
            const unsigned int NN   =  m_uiBVec[blk]._vec[0].getSz();
            const unsigned int bLev =
        pNodes[blkList[blk].firstElementID()].getLevel(); const unsigned
        int BLK_DT = m_uiAppCtx->getBlkTimestepFac(bLev,m_uiLevMin,m_uiLevMax);

            sync_blk_timestep(blk,BLK_S);
            m_uiAppCtx->post_stage_blk((T*)m_uiBVec[blk]._vec[BLK_S].data(),DOF,blk,current_t
        + dt_finest*BLK_T); m_uiBVec[blk]._vec[BLK_S].mark_synced();
            //std::cout<<"[NUTS]: pt :  "<<pt<<" blk : "<<blk<<" rk : "<<rk<<"
        step size: "<<BLK_DT<<" is synced:
        "<<m_uiBVec[blk]._vec[BLK_S].isSynced()<<std::endl;

        }
        */
    }

    // compute the time step vector and increment time.
    for (unsigned int bb = 0; bb < m_uiActiveBlkIDs.size(); bb++) {
        const unsigned int blk   = m_uiActiveBlkIDs[bb];
        const unsigned int BLK_T = m_uiBVec[blk]._time;
        const unsigned int NN    = m_uiBVec[blk]._vec[0].getSz();
        const unsigned int bLev =
            pNodes[blkList[blk].firstElementID()].getLevel();
        const unsigned int BLK_DT =
            m_uiAppCtx->getBlkTimestepFac(bLev, m_uiLevMin, m_uiLevMax);
        const T blk_dt = m_uiTimeInfo._m_uiTh * BLK_DT;

        T* out_ptr     = (T*)m_uiBVec[blk]._vec[0].data();
        for (unsigned int s = 1; s <= m_uiNumStages; s++) {
            assert(m_uiBVec[blk]._vec[s].isSynced());
            const T* stage_ptr = m_uiBVec[blk]._vec[s].data();
            for (unsigned int n = 0; n < DOF * NN; n++)
                out_ptr[n] += (blk_dt * m_uiBi[(s - 1)] * stage_ptr[n]);
        }

        m_uiBVec[blk]._time += BLK_DT;
        m_uiBVec[blk]._rks = 0;
        m_uiAppCtx->post_timestep_blk((T*)m_uiBVec[blk]._vec[0].data(), DOF,
                                      blk, current_t + dt_finest * BLK_T);
        m_uiBVec[blk]._vec[0].mark_synced();
        debug_blk_hash("post-axpy", blk, 0, pt, 0);

        // zipDG to the stVec[0]
        // mark the stage vars as junck.
        for (unsigned int s = 1; s <= m_uiNumStages; s++)
            m_uiBVec[blk]._vec[s].mark_unsynced();
    }

    update_ele_timestep();

    pMesh->readFromGhostBeginElementVec(m_uiEleTime.data(), 1);
    pMesh->readFromGhostEndElementVec(m_uiEleTime.data(), 1);

    pMesh->readFromGhostBeginElementVec(m_uiEleDT.data(), 1);
    pMesh->readFromGhostEndElementVec(m_uiEleDT.data(), 1);

    pMesh->waitActive();
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::gts_evolve(unsigned int pt) {
    ot::Mesh* pMesh = m_uiAppCtx->get_mesh();

    if (!(pMesh->isActive())) return;

    const std::vector<ot::Block>& blkList = pMesh->getLocalBlockList();
    const ot::TreeNode* pNodes            = pMesh->getAllElements().data();
    const double current_t                = m_uiTimeInfo._m_uiT;
    const unsigned int DOF                = m_uiEVar.get_dof();

    double current_t_adv                  = current_t;

    const unsigned int finest_t           = m_uiAppCtx->getBlkTimestepFac(
        m_uiLevMax, m_uiLevMin, m_uiLevMax);  // finest  time level.
    const unsigned int coarset_t =
        m_uiAppCtx->getBlkTimestepFac(m_uiLevMin, m_uiLevMin, m_uiLevMax);

    const double dt_finest  = m_uiTimeInfo._m_uiTh;
    const double dt_coarset = coarset_t * m_uiTimeInfo._m_uiTh;

    assert(blkList.size() > 0);
    m_uiActiveBlkIDs.clear();
    for (unsigned int blk = 0; blk < blkList.size(); blk++) {
        m_uiActiveBlkIDs.push_back(blk);

        if (blkList[blk].getBlockType() == ot::BlockType::UNZIP_INDEPENDENT) {
            m_uiIndependentActiveBlkIDs.push_back(blk);
        } else {
            m_uiDependentActiveBlkIDs.push_back(blk);
        }
    }

    // std::cout<<"rank: "<<pMesh->getMPIRank()<<"pt: "<<pt<<" active block
    // size: "<<m_uiActiveBlkIDs.size()<<std::endl;

    // 10/07/2020 (Milinda): Synchronize data u(x,t)
    // for(unsigned int bb=0; bb < m_uiActiveBlkIDs.size(); bb++)
    // {
    //     const unsigned int blk = m_uiActiveBlkIDs[bb];
    //     m_uiBVec[blk]._vec[0].zipDG(pMesh,m_uiEvarDG.get_vec_ptr(),DOF);
    // }

    // // Now need to synchronize the out vector just to smooth out solution
    // between refinement boundaries.
    // pMesh->readFromGhostBeginEleDGVec(m_uiEvarDG.get_vec_ptr(),DOF);
    // pMesh->readFromGhostEndEleDGVec(m_uiEvarDG.get_vec_ptr(),DOF);

    // for(unsigned int bb=0; bb < m_uiActiveBlkIDs.size(); bb++)
    // {
    //     const unsigned int blk = m_uiActiveBlkIDs[bb];
    //     const unsigned int BLK_T = m_uiBVec[blk]._time;
    //     const unsigned int NN   =  m_uiBVec[blk]._vec[0].getSz();
    //     const unsigned int bLev =
    //     pNodes[blkList[blk].firstElementID()].getLevel(); const
    //     unsigned int BLK_DT =
    //     m_uiAppCtx->getBlkTimestepFac(bLev,m_uiLevMin,m_uiLevMax);

    //     m_uiBVec[blk]._vec[0].mark_unsynced();
    //     sync_blk_timestep(blk,0);
    //     m_uiBVec[blk]._vec[0].mark_synced();

    // }

    for (unsigned int rk = 1; rk <= m_uiNumStages; rk++) {
        const unsigned int BLK_S = rk;
        for (unsigned int bb = 0; bb < m_uiActiveBlkIDs.size(); bb++) {
            const unsigned int blk   = m_uiActiveBlkIDs[bb];
            const unsigned int BLK_T = m_uiBVec[blk]._time;
            const unsigned int NN    = m_uiBVec[blk]._vec[0].getSz();
            const unsigned int bLev =
                pNodes[blkList[blk].firstElementID()].getLevel();
            const unsigned int BLK_DT =
                m_uiAppCtx->getBlkTimestepFac(bLev, m_uiLevMin, m_uiLevMax);

            const T blk_dt = m_uiTimeInfo._m_uiTh * BLK_DT;

            assert(m_uiBVec[blk]._vec[rk - 1].isSynced());
            m_uiAppCtx->pre_timestep_blk((T*)m_uiBVec[blk]._vec[0].data(), DOF,
                                         blk, current_t + dt_finest * BLK_T);

            m_uiBVec[blk]._vec[m_uiNumStages + 1].vecCopy(
                m_uiBVec[blk]._vec[0]);  // copy the in vector.
            T* out_ptr = (T*)m_uiBVec[blk]._vec[m_uiNumStages + 1].data();

            for (unsigned int s = 1; s < BLK_S; s++) {
                const T* stage_ptr = m_uiBVec[blk]._vec[s].data();
                for (unsigned int n = 0; n < DOF * NN; n++) {
                    out_ptr[n] +=
                        (blk_dt *
                         m_uiAij[(BLK_S - 1) * m_uiNumStages + (s - 1)] *
                         stage_ptr[n]);
                    // printf("s[%d]: %f\n", n, out_ptr[n]);
                    // printf("s: %d : stage[%d]: %f\n", s,  n, stage_ptr[n]);
                }
            }

            m_uiAppCtx->pre_stage_blk(
                (T*)m_uiBVec[blk]._vec[m_uiNumStages + 1].data(), DOF, blk,
                current_t + dt_finest * BLK_T);
            debug_nan_check("pre-rhs-in", blk, m_uiNumStages + 1, pt, rk);
            debug_blk_hash("pre-rhs-in", blk, m_uiNumStages + 1, pt, rk);
            m_uiBVec[blk]._vec[BLK_S].computeVec(
                m_uiAppCtx, m_uiBVec[blk]._vec[m_uiNumStages + 1],
                current_t + dt_finest * BLK_T);
            debug_nan_check("post-rhs", blk, BLK_S, pt, rk);
            debug_blk_hash("post-rhs", blk, BLK_S, pt, rk);
            m_uiBVec[blk]._vec[BLK_S].mark_unsynced();
            // std::cout<<"rk:"<<rk<<std::endl;
            // m_uiBVec[blk]._vec[BLK_S].dump_vec(std::cout);

#ifdef __PROFILE_ENUTS__
            m_uiPt[ENUTSPROFILE::ENUTS_BLK_ZIP].start();
#endif

            m_uiBVec[blk]._vec[BLK_S].zip(
                pMesh, m_uiStVec[BLK_S - 1].get_vec_ptr(), DOF);
            m_uiBVec[blk]._rks = rk;

#ifdef __PROFILE_ENUTS__
            m_uiPt[ENUTSPROFILE::ENUTS_BLK_ZIP].stop();
#endif
        }

        // do the DG vec ghost sync.
        pMesh->readFromGhostBegin(m_uiStVec[BLK_S - 1].get_vec_ptr(), DOF);
        pMesh->readFromGhostEnd(m_uiStVec[BLK_S - 1].get_vec_ptr(), DOF);

        pMesh->unzip(m_uiStVec[BLK_S - 1].get_vec_ptr(),
                     m_uiEvarUzip.get_vec_ptr(), DOF);

        for (unsigned int bb = 0; bb < m_uiActiveBlkIDs.size(); bb++) {
            const unsigned int blk   = m_uiActiveBlkIDs[bb];
            const unsigned int BLK_T = m_uiBVec[blk]._time;
            const unsigned int NN    = m_uiBVec[blk]._vec[0].getSz();
            const unsigned int bLev =
                pNodes[blkList[blk].firstElementID()].getLevel();
            const unsigned int BLK_DT =
                m_uiAppCtx->getBlkTimestepFac(bLev, m_uiLevMin, m_uiLevMax);

            m_uiBVec[blk]._vec[BLK_S].copyFromUnzip(
                pMesh, m_uiEvarUzip.get_vec_ptr(), true, DOF);
            // sync_blk_timestep(blk,BLK_S);
            m_uiAppCtx->post_stage_blk((T*)m_uiBVec[blk]._vec[BLK_S].data(),
                                       DOF, blk, current_t + dt_finest * BLK_T);
            m_uiBVec[blk]._vec[BLK_S].mark_synced();
            // std::cout<<"[NUTS]: pt :  "<<pt<<" blk : "<<blk<<" rk : "<<rk<<"
            // step size: "<<BLK_DT<<" is synced:
            // "<<m_uiBVec[blk]._vec[BLK_S].isSynced()<<std::endl;
        }
    }

    // compute the time step vector and increment time.
    for (unsigned int bb = 0; bb < m_uiActiveBlkIDs.size(); bb++) {
        const unsigned int blk   = m_uiActiveBlkIDs[bb];
        const unsigned int BLK_T = m_uiBVec[blk]._time;
        const unsigned int NN    = m_uiBVec[blk]._vec[0].getSz();
        const unsigned int bLev =
            pNodes[blkList[blk].firstElementID()].getLevel();
        const unsigned int BLK_DT =
            m_uiAppCtx->getBlkTimestepFac(bLev, m_uiLevMin, m_uiLevMax);
        const T blk_dt = m_uiTimeInfo._m_uiTh * BLK_DT;

        T* out_ptr     = (T*)m_uiBVec[blk]._vec[0].data();
        for (unsigned int s = 1; s <= m_uiNumStages; s++) {
            assert(m_uiBVec[blk]._vec[s].isSynced());
            const T* stage_ptr = m_uiBVec[blk]._vec[s].data();
            for (unsigned int n = 0; n < DOF * NN; n++)
                out_ptr[n] += (blk_dt * m_uiBi[(s - 1)] * stage_ptr[n]);
        }

        m_uiBVec[blk]._time += BLK_DT;
        m_uiBVec[blk]._rks = 0;
        m_uiAppCtx->post_timestep_blk((T*)m_uiBVec[blk]._vec[0].data(), DOF,
                                      blk, current_t + dt_finest * BLK_T);
        m_uiBVec[blk]._vec[0].mark_synced();
        debug_blk_hash("post-axpy", blk, 0, pt, 0);

        // zipDG to the stVec[0]
        // mark the stage vars as junck.
        for (unsigned int s = 1; s <= m_uiNumStages; s++)
            m_uiBVec[blk]._vec[s].mark_unsynced();
    }

    update_ele_timestep();

    pMesh->readFromGhostBeginElementVec(m_uiEleTime.data(), 1);
    pMesh->readFromGhostEndElementVec(m_uiEleTime.data(), 1);

    pMesh->readFromGhostBeginElementVec(m_uiEleDT.data(), 1);
    pMesh->readFromGhostEndElementVec(m_uiEleDT.data(), 1);

    pMesh->waitActive();
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::evolve() {
#ifdef __PROFILE_ENUTS__
    m_uiPt[ENUTSPROFILE::ENUTS_EVOLVE].start();
#endif

    ot::Mesh* pMesh             = m_uiAppCtx->get_mesh();
    m_uiTimeInfo                = m_uiAppCtx->get_ts_info();
    const double current_t      = m_uiTimeInfo._m_uiT;
    double current_t_adv        = current_t;
    // Assumption: m_uiEvar is time synced accross all the blocks.

    const unsigned int finest_t = m_uiAppCtx->getBlkTimestepFac(
        m_uiLevMax, m_uiLevMin, m_uiLevMax);  // finest  time level.
    const unsigned int coarset_t = m_uiAppCtx->getBlkTimestepFac(
        m_uiLevMin, m_uiLevMin, m_uiLevMax);  // coarset time level.

    const double dt_finest  = m_uiTimeInfo._m_uiTh;
    const double dt_coarset = coarset_t * m_uiTimeInfo._m_uiTh;

    m_uiAppCtx->pre_timestep(m_uiEVar);

    if (pMesh->isActive()) {
        const unsigned int rank = pMesh->getMPIRank();
        const unsigned int npes = pMesh->getMPICommSize();

        assert(m_uiEVar.get_dof() == m_uiEVecTmp[0].get_dof());
        assert(m_uiEVar.get_dof() == m_uiEVecTmp[1].get_dof());

        const unsigned int DOF           = m_uiEVar.get_dof();
        MPI_Comm comm                    = pMesh->getMPICommunicator();

        const ot::TreeNode* const pNodes = pMesh->getAllElements().data();

#ifdef __PROFILE_ENUTS__
        m_uiPt[ENUTSPROFILE::ENUTS_BLK_UNZIP].start();
#endif

        m_uiAppCtx->unzip(
            m_uiEVar, m_uiEvarUzip,
            m_uiAppCtx
                ->get_async_batch_sz());  // unzip m_uiEVec to m_uiEVarUnzip
        const std::vector<ot::Block>& blkList = pMesh->getLocalBlockList();

        // initialize the block async vectors
        for (unsigned int blk = 0; blk < blkList.size(); blk++) {
            m_uiBVec[blk]._time = 0;
            m_uiBVec[blk]._rks  = 0;
            m_uiBVec[blk]._vec[0].copyFromUnzip(
                pMesh, m_uiEvarUzip.get_vec_ptr(), true, DOF);
            m_uiBVec[blk]._vec[0].mark_synced();
            debug_blk_hash("init-unzip", blk, 0, 0, 0);

            for (unsigned int s = 1; s <= m_uiNumStages; s++)
                m_uiBVec[blk]._vec[s].mark_unsynced();
        }

#ifdef __PROFILE_ENUTS__
        m_uiPt[ENUTSPROFILE::ENUTS_BLK_UNZIP].stop();
#endif

        for (unsigned int ele = pMesh->getElementPreGhostBegin();
             ele < pMesh->getElementPostGhostEnd(); ele++)
            m_uiEleTime[ele] = 0;

        this->update_ele_timestep();

        pMesh->readFromGhostBeginElementVec(m_uiEleTime.data(), 1);
        pMesh->readFromGhostBeginElementVec(m_uiEleDT.data(), 1);

        pMesh->readFromGhostEndElementVec(m_uiEleTime.data(), 1);
        pMesh->readFromGhostEndElementVec(m_uiEleDT.data(), 1);

        for (unsigned int pt = 0; pt < coarset_t; pt++) {
            // unsigned int numBlkEvolved = 0;
            // for(unsigned int blk =0; blk < blkList.size(); blk++)
            // {
            //     const unsigned int BLK_T = m_uiBVec[blk]._time;
            //     const unsigned int NN   =  m_uiBVec[blk]._vec[0].getSz();
            //     const unsigned int bLev =
            //     pNodes[blkList[blk].firstElementID()].getLevel(); const
            //     unsigned int BLK_DT =
            //     m_uiAppCtx->getBlkTimestepFac(bLev,m_uiLevMin,m_uiLevMax);;
            //     if( pt% BLK_DT !=0 )
            //         continue;
            //     numBlkEvolved++;
            // }

            // std::cout<<"[ENUTS] : pt: "<<pt<<" number of blocks evolved :
            // "<<numBlkEvolved<<" out of "<<blkList.size()<<std::endl;
            this->partial_evolve(pt);
            // this->gts_evolve(pt);
            m_uiPt                     = pt;
            const unsigned int pt_freq = std::min(10u, coarset_t);
            if (m_uiPt % pt_freq == 0) {
                if (!pMesh->getMPIRankGlobal())
                    std::cout << "[LTS] : local ts: " << m_uiPt << " of total "
                              << coarset_t << std::endl;
            }
        }

#ifdef __PROFILE_ENUTS__
        m_uiPt[ENUTSPROFILE::ENUTS_BLK_ZIP].start();
#endif

        // on labelled meshes the zip plan may source a cg from a block's
        // padding, which is stale after the last sub-step: bring every block's
        // padding to the common end-of-step time first
        // labelled meshes: the block zip plan follows the labelled E2N_DG
        // owner, which can differ from the SFC build's owner; the two block
        // copies of a shared node differ after sub-cycling, so gather each cg
        // from its SFC-canonical owner's block copy through the DG vector
        bool anyNonSFC = false;
        for (const ot::Block& b : blkList) anyNonSFC |= b.isNonSFC();
        if (anyNonSFC) {
            build_cg_owner_map();
            blk_vec_to_zipDG(m_uiBVec.data(), m_uiEvarDG, 0);
            pMesh->readFromGhostBeginEleDGVec(m_uiEvarDG.get_vec_ptr(), DOF);
            pMesh->readFromGhostEndEleDGVec(m_uiEvarDG.get_vec_ptr(), DOF);
            zip_labelled_via_dg();
        }

        for (unsigned int blk = 0; anyNonSFC == false && blk < blkList.size(); blk++) {
            const unsigned int BLK_T = m_uiBVec[blk]._time;
            const unsigned int NN    = m_uiBVec[blk]._vec[0].getSz();
            const unsigned int bLev =
                pNodes[blkList[blk].firstElementID()].getLevel();
            const unsigned int BLK_DT =
                m_uiAppCtx->getBlkTimestepFac(bLev, m_uiLevMin, m_uiLevMax);

            m_uiBVec[blk]._vec[0].zip(pMesh, m_uiEVar.get_vec_ptr(), DOF);
        }

#ifdef __PROFILE_ENUTS__
        m_uiPt[ENUTSPROFILE::ENUTS_BLK_ZIP].stop();
#endif
    }

    m_uiAppCtx->post_timestep(m_uiEVar);
    m_uiAppCtx->increment_ts_info((T)coarset_t, 1);
    m_uiTimeInfo = m_uiAppCtx->get_ts_info();
    pMesh->waitAll();

#ifdef __PROFILE_ENUTS__
    m_uiPt[ENUTSPROFILE::ENUTS_EVOLVE].stop();
#endif
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::remesh(unsigned int grain_sz, double ld_tol,
                                  unsigned int sf_k) {
    ot::Mesh* pMesh        = m_uiAppCtx->get_mesh();
    ot::Mesh* newMesh      = m_uiAppCtx->remesh(grain_sz, ld_tol, sf_k);

    DendroIntL oldElements = pMesh->getNumLocalMeshElements();
    DendroIntL newElements = newMesh->getNumLocalMeshElements();

    DendroIntL oldElements_g, newElements_g;

    par::Mpi_Reduce(&oldElements, &oldElements_g, 1, MPI_SUM, 0,
                    pMesh->getMPIGlobalCommunicator());
    par::Mpi_Reduce(&newElements, &newElements_g, 1, MPI_SUM, 0,
                    newMesh->getMPIGlobalCommunicator());

    unsigned int lminmax[2];
    newMesh->computeMinMaxLevel(lminmax[0], lminmax[1]);

    if (!(pMesh->getMPIRankGlobal())) {
        std::cout << "[LTS]: remesh " << m_uiPt
                  << ": \ttime : " << m_uiTimeInfo._m_uiT
                  << "\told mesh: " << oldElements_g
                  << "\tnew mesh:" << newElements_g << std::endl;
        printf(
            "[LTS]: old level (min, max): (%d,%d)  --> new level (min, max): "
            "(%d, %d) \n",
            m_uiLevMin, m_uiLevMax, lminmax[0], lminmax[1]);
        // printf("[LTS]: old number of blocks %d new mesh number of blocks : %d
        // \n",pMesh->getLocalBlockList().size(),
        // newMesh->getLocalBlockList().size());
    }

    const unsigned int DOF = m_uiEvarDG.get_dof();

    // 0 : synced vector
    // 1 : rk stage
    //. : ....
    // m_uiNumStages : last rk stage.
    std::vector<DVec> nMeshVec;
    nMeshVec.resize(m_uiNumStages + 1);

    for (unsigned int i = 0; i < (m_uiNumStages + 1); i++)
        nMeshVec[i].create_vector(newMesh, ot::DVEC_TYPE::OCT_LOCAL_NODES,
                                  ot::DVEC_LOC::HOST, DOF, true);

    // element time vector.
    std::vector<unsigned int> eleTimeVec;
    eleTimeVec.resize(newMesh->getAllElements().size(), 0);

    // time level in dt.
    std::vector<unsigned int> eleTimeLev;
    eleTimeLev.resize(newMesh->getAllElements().size(), 0);

    // input block vector.
    blk_vec_to_zipDG(m_uiBVec.data(), m_uiEvarDG, 0);

    // computed stage vectors.
    for (unsigned int s = 1; s <= m_uiNumStages; s++)
        blk_vec_to_zipDG(m_uiBVec.data(), m_uiStVec[s - 1], s);

    const unsigned int dg_sz_old = pMesh->getDegOfFreedomDG();
    const unsigned int dg_sz_new = newMesh->getDegOfFreedomDG();
    const unsigned int nPe       = pMesh->getNumNodesPerElement();

#if 0
        for(unsigned int v=0; v< DOF; v++)
        {

            double vmin = vecMin(pMesh,m_uiEvarDG.get_vec_ptr() + v * pMesh->getDegOfFreedomDG(), ot::VEC_TYPE::DG_NODAL,true);
            double vmax = vecMax(pMesh,m_uiEvarDG.get_vec_ptr() + v * pMesh->getDegOfFreedomDG(), ot::VEC_TYPE::DG_NODAL,true);

            if(!pMesh->getMPIRankGlobal())
                std::cout<<" Evar before IGT var: "<<v<< " vmin: "<<vmin<<" vmax: "<<vmax<<std::endl;

            for(unsigned int s=0; s < m_uiNumStages; s++)
            {
                double vmin = vecMin(pMesh,m_uiStVec[s].get_vec_ptr() + v * pMesh->getDegOfFreedomDG(), ot::VEC_TYPE::DG_NODAL,true);
                double vmax = vecMax(pMesh,m_uiStVec[s].get_vec_ptr() + v * pMesh->getDegOfFreedomDG(), ot::VEC_TYPE::DG_NODAL,true);

                if(!pMesh->getMPIRankGlobal())
                    std::cout<<" stage: "<<s<<" before IGT var: "<<v<< " vmin: "<<vmin<<" vmax: "<<vmax<<std::endl;

            }
        
        }
#endif

    pMesh->interGridTransfer_DG(m_uiEvarDG.get_vec_ptr(),
                                nMeshVec[0].get_vec_ptr(), newMesh, DOF);
    for (unsigned int i = 0; i < (m_uiNumStages); i++)
        pMesh->interGridTransfer_DG(m_uiStVec[i].get_vec_ptr(),
                                    nMeshVec[i + 1].get_vec_ptr(), newMesh,
                                    DOF);

    // intergrid transfer for the elemental vector.
    pMesh->interGridTransferCellVec(m_uiEleTime.data(), eleTimeVec.data(),
                                    newMesh, 1,
                                    ot::INTERGRID_TRANSFER_MODE::CELLVEC_CPY);

    // intergrid transfer for the elemental time level dt.
    pMesh->interGridTransferCellVec(m_uiEleDT.data(), eleTimeLev.data(),
                                    newMesh, 1,
                                    ot::INTERGRID_TRANSFER_MODE::CELLVEC_CPY);

    // regroup blocks with element time tag values.
    newMesh->performBlocksSetup(
        0, eleTimeVec.data() + newMesh->getElementLocalBegin(),
        newMesh->getNumLocalMeshElements());

    // {
    //         const unsigned int coarset_t =
    //         m_uiAppCtx->getBlkTimestepFac(m_uiLevMin,m_uiLevMin,m_uiLevMax);
    //         // coarset time level. std::vector<double> eleT;
    //         eleT.resize(newMesh->getAllElements().size(),0);

    //         for(unsigned int ele= newMesh->getElementLocalBegin(); ele <
    //         newMesh->getElementLocalEnd(); ele++)
    //             eleT[ele] = eleTimeVec[ele];//std::cout<<"ele : "<<ele<<"
    //             time : "<<m_uiEleTime[ele]<<std::endl;

    //         const char*  pVarNames []  = {"CHI","PHI"};
    //         const double* pVarData []  = {nMeshVec[0].get_vec_ptr() ,
    //         nMeshVec[0].get_vec_ptr() + newMesh->getDegOfFreedomDG()}; const
    //         char*  cVarNames []  = {"time_level"}; const double* cVarData []
    //         = {eleT.data()};

    //         char fname[256];
    //         int lts_step = this->curr_step() * coarset_t + m_uiPt;

    //         sprintf(fname,"remesh_lts_%d",lts_step);
    //         io::vtk::mesh2vtuFine(newMesh,fname,0,NULL,NULL,2,pVarNames,pVarData,1,cVarNames,cVarData,true);
    // }

    m_uiAppCtx->grid_transfer(newMesh);

    std::swap(pMesh, newMesh);
    delete newMesh;

    m_uiAppCtx->set_mesh(pMesh);

    // to sync with ETS variables. (GTS) timestepper class.
    m_uiEVar = m_uiAppCtx->get_evolution_vars();
    m_uiAppCtx->set_ets_synced(true);

    // sync with ENUTS (LTS) timestepper class.
    this->deallocate_internal_vars();
    this->free_data_structures();

    this->init_data_structures();
    this->allocate_internal_vars();

    // vecDG_to_blk_vec(nMeshVec[0], m_uiBVec.data(), 0);

    // for(unsigned int s=1; s <=m_uiNumStages; s++)
    //     vecDG_to_blk_vec(nMeshVec[s], m_uiBVec.data(), s);

    std::swap(nMeshVec[0], m_uiEvarDG);
    for (unsigned int s = 1; s <= m_uiNumStages; s++)
        std::swap(nMeshVec[s], m_uiStVec[s - 1]);

    std::swap(m_uiEleTime, eleTimeVec);
    std::swap(m_uiEleDT, eleTimeLev);
    eleTimeVec.clear();
    eleTimeLev.clear();

    for (unsigned int i = 0; i < (m_uiNumStages + 1); i++)
        nMeshVec[i].destroy_vector();

    // ghost sync begin.
    pMesh->readFromGhostBeginEleDGVec(m_uiEvarDG.get_vec_ptr(),
                                      m_uiEvarDG.get_dof());

    for (unsigned int i = 0; i < m_uiNumStages; i++)
        pMesh->readFromGhostBeginEleDGVec(m_uiStVec[i].get_vec_ptr(),
                                          m_uiStVec[i].get_dof());

    pMesh->readFromGhostBeginElementVec(m_uiEleTime.data(), 1);
    pMesh->readFromGhostBeginElementVec(m_uiEleDT.data(), 1);

    pMesh->readFromGhostEndEleDGVec(m_uiEvarDG.get_vec_ptr(),
                                    m_uiEvarDG.get_dof());

    for (unsigned int i = 0; i < m_uiNumStages; i++)
        pMesh->readFromGhostEndEleDGVec(m_uiStVec[i].get_vec_ptr(),
                                        m_uiStVec[i].get_dof());

    pMesh->readFromGhostEndElementVec(m_uiEleTime.data(), 1);
    pMesh->readFromGhostEndElementVec(m_uiEleDT.data(), 1);

#if 0
        for(unsigned int v=0; v< DOF; v++)
        {

            double vmin = vecMin(pMesh,m_uiEvarDG.get_vec_ptr() + v * pMesh->getDegOfFreedomDG(), ot::VEC_TYPE::DG_NODAL,true);
            double vmax = vecMax(pMesh,m_uiEvarDG.get_vec_ptr() + v * pMesh->getDegOfFreedomDG(), ot::VEC_TYPE::DG_NODAL,true);

            if(!pMesh->getMPIRankGlobal())
                std::cout<<" Evar after IGT var: "<<v<< " vmin: "<<vmin<<" vmax: "<<vmax<<std::endl;

            for(unsigned int s=0; s < m_uiNumStages; s++)
            {
                double vmin = vecMin(pMesh,m_uiStVec[s].get_vec_ptr() + v * pMesh->getDegOfFreedomDG(), ot::VEC_TYPE::DG_NODAL,true);
                double vmax = vecMax(pMesh,m_uiStVec[s].get_vec_ptr() + v * pMesh->getDegOfFreedomDG(), ot::VEC_TYPE::DG_NODAL,true);

                if(!pMesh->getMPIRankGlobal())
                    std::cout<<" stage: "<<s<<" after IGT var: "<<v<< " vmin: "<<vmin<<" vmax: "<<vmax<<std::endl;
            }

        }
#endif

    // update block time.
    const std::vector<ot::Block>& blkList = pMesh->getLocalBlockList();
    const ot::TreeNode* pNodes            = pMesh->getAllElements().data();

    // restore the block time.
    bool isBlkEleTimeSynced               = true;
    for (unsigned int blk = 0; blk < blkList.size(); blk++) {
        const unsigned int eletime =
            m_uiEleTime[blkList[blk].firstElementID()];
        for (const unsigned int ele : blkList[blk]) {
            if (m_uiEleTime[ele] != eletime) {
                std::cout << "rank: " << pMesh->getMPIRank()
                          << " element : " << pNodes[ele]
                          << " in block : " << blk
                          << " is time wise not synchronized  eletime : "
                          << eletime
                          << " current eletime : " << m_uiEleTime[ele]
                          << " ele dt1 : "
                          << m_uiEleDT[blkList[blk].firstElementID()]
                          << " ele dt2 : " << m_uiEleDT[ele] << std::endl;
                isBlkEleTimeSynced = false;
            }
        }
        m_uiBVec[blk]._time = eletime;
        m_uiBVec[blk]._vec[0].mark_unsynced();
    }

    if (!isBlkEleTimeSynced) {
        std::cout << "rank: " << pMesh->getMPIRankGlobal()
                  << " elements in blocks synced constraint failed. "
                  << std::endl;
        MPI_Abort(pMesh->getMPICommunicator(), 0);
    }

    for (unsigned int blk = 0; blk < blkList.size(); blk++) {
        m_uiBVec[blk]._vec[0].mark_unsynced();
        sync_blk_timestep(blk, 0);
        m_uiBVec[blk]._vec[0].mark_synced();
        debug_nan_check("igt-sync-u", blk, 0, 0, 0);
        debug_blk_hash("igt-sync-u", blk, 0, 0, 0);

        for (unsigned int s = 1; s <= m_uiNumStages; s++) {
            m_uiBVec[blk]._vec[s].mark_unsynced();
            sync_blk_timestep(blk, s, true);
            debug_nan_check("igt-sync-stage", blk, s, 0, s);
            debug_blk_hash("igt-sync-stage", blk, s, 0, s);
        }
    }

    return;
}

template <typename T, typename Ctx>
bool ExplicitNUTS<T, Ctx>::isRemeshEvars(
    const unsigned int* varIds, unsigned int numvars,
    std::function<double(double, double, double, double*)> wavelet_tol,
    double amr_coarsen_fac) {
    ot::Mesh* pMesh = (ot::Mesh*)m_uiAppCtx->get_mesh();
    std::vector<unsigned int> refine_flags;
    bool isMeshGlobalChanged = false;
    bool isMeshLocalChanged  = false;

    if (pMesh->isActive()) {
        const std::vector<ot::Block>& blkList = pMesh->getLocalBlockList();
        const unsigned int eOrder             = pMesh->getElementOrder();

        const unsigned int unz_ele =
            (2 * eOrder + 1) * (2 * eOrder + 1) * (2 * eOrder + 1);
        const unsigned int DOF = m_uiEVar.get_dof();

        std::vector<T> uEleVec;
        uEleVec.resize(unz_ele * DOF);

        std::vector<T> uEleRefIn;
        uEleRefIn.resize(unz_ele * numvars);

        std::vector<double> wMax;
        wMax.resize(pMesh->getNumLocalMeshElements(), 0.0);

        const unsigned int eleOfst = pMesh->getElementLocalBegin();
        RefElement* refEl          = (RefElement*)pMesh->getReferenceElement();
        wavelet::WaveletEl wRefEl(refEl);

        const ot::TreeNode* pNodes = pMesh->getAllElements().data();

        for (unsigned int blk = 0; blk < blkList.size(); blk++) {
            for (const unsigned int ele : blkList[blk]) {
                bool isBdyEle = pMesh->isBoundaryOctant(ele);
                m_uiBVec[blk]._vec[0].getUnzipElementalValues(pMesh, ele,
                                                              uEleVec.data());

                for (unsigned int v = 0; v < numvars; v++) {
                    const unsigned int vid = varIds[v];
                    std::memcpy(uEleRefIn.data() + v * unz_ele,
                                uEleVec.data() + vid * unz_ele,
                                sizeof(T) * unz_ele);
                }

                const double oct_dx =
                    (1u << (m_uiMaxDepth - pNodes[ele].getLevel())) /
                    (double(pMesh->getElementOrder()));
                Point oct_pt1 = Point(pNodes[ele].minX(), pNodes[ele].minY(),
                                      pNodes[ele].minZ());
                Point oct_pt2 = Point(pNodes[ele].minX() + oct_dx,
                                      pNodes[ele].minY() + oct_dx,
                                      pNodes[ele].minZ() + oct_dx);
                Point domain_pt1, domain_pt2, dx_domain;
                pMesh->octCoordToDomainCoord(oct_pt1, domain_pt1);
                pMesh->octCoordToDomainCoord(oct_pt2, domain_pt2);
                dx_domain    = domain_pt2 - domain_pt1;
                double hx[3] = {dx_domain.x(), dx_domain.y(), dx_domain.z()};
                const double tol_ele = wavelet_tol(
                    domain_pt1.x(), domain_pt1.y(), domain_pt1.z(), hx);

                wMax[ele - eleOfst] = wavelet::compute_element_wavelet(
                    pMesh, (const wavelet::WaveletEl*)&wRefEl, uEleRefIn.data(),
                    tol_ele, numvars, isBdyEle);

                // if(isBdyEle)
                // std::cout<<"ele : "<<ele<<" comp. coefficient :
                // "<<wMax[ele-eleOfst]<<std::endl;
            }
        }

        refine_flags.clear();
        refine_flags.resize(pMesh->getNumLocalMeshElements(), OCT_NO_CHANGE);

        for (unsigned int ele = pMesh->getElementLocalBegin();
             ele < pMesh->getElementLocalEnd(); ele++) {
            const double oct_dx =
                (1u << (m_uiMaxDepth - pNodes[ele].getLevel())) /
                (double(pMesh->getElementOrder()));
            Point oct_pt1 = Point(pNodes[ele].minX(), pNodes[ele].minY(),
                                  pNodes[ele].minZ());
            Point oct_pt2 =
                Point(pNodes[ele].minX() + oct_dx, pNodes[ele].minY() + oct_dx,
                      pNodes[ele].minZ() + oct_dx);
            Point domain_pt1, domain_pt2, dx_domain;
            pMesh->octCoordToDomainCoord(oct_pt1, domain_pt1);
            pMesh->octCoordToDomainCoord(oct_pt2, domain_pt2);
            dx_domain    = domain_pt2 - domain_pt1;
            double hx[3] = {dx_domain.x(), dx_domain.y(), dx_domain.z()};
            const double tol_ele =
                wavelet_tol(domain_pt1.x(), domain_pt1.y(), domain_pt1.z(), hx);

            const double l_max = wMax[ele - eleOfst];

            if (pNodes[ele].getLevel() < m_uiLevMax && l_max > tol_ele) {
                refine_flags[(ele - eleOfst)] = OCT_SPLIT;
                isMeshLocalChanged            = true;

            } else if (pNodes[ele].getLevel() > m_uiLevMin &&
                       l_max < amr_coarsen_fac * tol_ele) {
                // 08/17/2020 : time sub-cycling mess up when we coarsen during
                // the LTS remesh. (currently coarsen enabled at synced remesh.
                // )
                refine_flags[(ele - eleOfst)] = OCT_NO_CHANGE;  // OCT_COARSE;
                // isMeshLocalChanged =true;
            } else {
                refine_flags[(ele - eleOfst)] = OCT_NO_CHANGE;
            }

            // std::cout<<"ele: "<<ele<<" refinement flag:
            // "<<refine_flags[(ele-eleOfst)]<<" wc :
            // "<<wMax[ele-eleOfst]<<std::endl;
        }
    }
    // std::cout<<"grank : "<<pMesh->getMPIRankGlobal()<<std::endl;
    MPI_Allreduce(&isMeshLocalChanged, &isMeshGlobalChanged, 1, MPI_CXX_BOOL,
                  MPI_LOR, pMesh->getMPIGlobalCommunicator());

    if (isMeshGlobalChanged) pMesh->setMeshRefinementFlags(refine_flags);

    return isMeshGlobalChanged;
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::evolve_with_remesh(unsigned int remesh_freq) {
#ifdef __PROFILE_ENUTS__
    m_uiPt[ENUTSPROFILE::ENUTS_EVOLVE].start();
#endif

    ot::Mesh* pMesh             = m_uiAppCtx->get_mesh();
    m_uiTimeInfo                = m_uiAppCtx->get_ts_info();
    const double current_t      = m_uiTimeInfo._m_uiT;
    double current_t_adv        = current_t;

    // Assumption: m_uiEvar is time synced accross all the blocks.
    const unsigned int finest_t = m_uiAppCtx->getBlkTimestepFac(
        m_uiLevMax, m_uiLevMin, m_uiLevMax);  // finest  time level.
    const unsigned int coarset_t = m_uiAppCtx->getBlkTimestepFac(
        m_uiLevMin, m_uiLevMin, m_uiLevMax);  // coarset time level.
    const unsigned int sync_level = m_uiLevMin;

    const double dt_finest        = m_uiTimeInfo._m_uiTh;
    const double dt_coarset       = coarset_t * m_uiTimeInfo._m_uiTh;

    m_uiAppCtx->pre_timestep(m_uiEVar);

    if (pMesh->isActive()) {
        const unsigned int rank = pMesh->getMPIRank();
        const unsigned int npes = pMesh->getMPICommSize();

        assert(m_uiEVar.get_dof() == m_uiEVecTmp[0].get_dof());
        assert(m_uiEVar.get_dof() == m_uiEVecTmp[1].get_dof());

        const unsigned int DOF                = m_uiEVar.get_dof();
        MPI_Comm comm                         = pMesh->getMPICommunicator();

        const ot::TreeNode* const pNodes      = pMesh->getAllElements().data();
        const std::vector<ot::Block>& blkList = pMesh->getLocalBlockList();

#ifdef __PROFILE_ENUTS__
        m_uiPt[ENUTSPROFILE::ENUTS_BLK_UNZIP].start();
#endif

        m_uiAppCtx->unzip(
            m_uiEVar, m_uiEvarUzip,
            m_uiAppCtx
                ->get_async_batch_sz());  // unzip m_uiEVec to m_uiEVarUnzip

        // initialize the block async vectors
        for (unsigned int blk = 0; blk < blkList.size(); blk++) {
            m_uiBVec[blk]._time = 0;
            m_uiBVec[blk]._rks  = 0;
            m_uiBVec[blk]._vec[0].copyFromUnzip(
                pMesh, m_uiEvarUzip.get_vec_ptr(), true, DOF);
            m_uiBVec[blk]._vec[0].mark_synced();
            debug_blk_hash("init-unzip", blk, 0, 0, 0);

            for (unsigned int s = 1; s <= m_uiNumStages; s++)
                m_uiBVec[blk]._vec[s].mark_unsynced();
        }

#ifdef __PROFILE_ENUTS__
        m_uiPt[ENUTSPROFILE::ENUTS_BLK_UNZIP].stop();
#endif

        for (unsigned int ele = pMesh->getElementPreGhostBegin();
             ele < pMesh->getElementPostGhostEnd(); ele++)
            m_uiEleTime[ele] = 0;

        this->update_ele_timestep();

        pMesh->readFromGhostBeginElementVec(m_uiEleTime.data(), 1);
        pMesh->readFromGhostBeginElementVec(m_uiEleDT.data(), 1);

        pMesh->readFromGhostEndElementVec(m_uiEleTime.data(), 1);
        pMesh->readFromGhostEndElementVec(m_uiEleDT.data(), 1);
    }

    unsigned int pt = 0;
    m_uiPt          = pt;

    while (pt < coarset_t) {
        pMesh = m_uiAppCtx->get_mesh();

        this->partial_evolve(pt);

        pt++;
        m_uiPt                     = pt;

        const unsigned int pt_freq = std::min(10u, coarset_t);
        if (m_uiPt % pt_freq == 0) {
            if (!pMesh->getMPIRankGlobal())
                std::cout << "[LTS] : local ts: " << m_uiPt << " of total "
                          << coarset_t << std::endl;
        }
        // if(!pMesh->getMPIRankGlobal())
        // std::cout<<"local ts: "<<m_uiPt<<" of total "<<coarset_t<<std::endl;
        // {
        //     blk_vec_to_zipDG(m_uiBVec.data(),m_uiEvarDG,0);
        //     std::vector<double> eleT;
        //     eleT.resize(pMesh->getAllElements().size(),0);

        //     for(unsigned int ele= pMesh->getElementLocalBegin(); ele <
        //     pMesh->getElementLocalEnd(); ele++)
        //         eleT[ele] = m_uiEleTime[ele];//std::cout<<"ele : "<<ele<<"
        //         time : "<<m_uiEleTime[ele]<<std::endl;

        //     const char*  pVarNames []  = {"CHI","PHI"};
        //     const double* pVarData []  = {m_uiEvarDG.get_vec_ptr() ,
        //     m_uiEvarDG.get_vec_ptr() + pMesh->getDegOfFreedomDG()}; const
        //     char*  cVarNames []  = {"time_level"}; const double* cVarData []
        //     = {eleT.data()};

        //     char fname[256];
        //     int lts_step = this->curr_step() * coarset_t + m_uiPt;

        //     sprintf(fname,"lts_step_%d",lts_step);
        //     io::vtk::mesh2vtuFine(pMesh,fname,0,NULL,NULL,2,pVarNames,pVarData,1,cVarNames,cVarData,true);
        // }

        if ((pt < coarset_t) && (pt % remesh_freq) == 0) {
            const unsigned int* refVIDs = m_uiAppCtx->get_refine_var_ids();
            unsigned int numRefVars     = m_uiAppCtx->get_num_refine_vars();
            std::function<double(double, double, double, double*)>
                waveletTolFunc = m_uiAppCtx->get_wtol_function();
            bool isRemesh      = (numRefVars > 0)
                                     ? this->isRemeshEvars(refVIDs, numRefVars,
                                                           waveletTolFunc, 0.1)
                                     : false;
            // bool isRemesh =
            // this->isRemeshEvars(refVIDs,numRefVars,waveletTolFunc,0.1);
            if (isRemesh) {
                // create new mesh.
                // do intergrid trasfer
                // 1. zipDG vector.
                // 2. timeCell vector

                // compute new min and max levels.
                // std::cout<<"Remesh triggered but not doing remesh:
                // )"<<std::endl;
                this->remesh();
            }

            // blk_vec_to_zipDG(m_uiBVec.data(),m_uiEvarDG,0);
            // pMesh->readFromGhostBeginEleDGVec(m_uiEvarDG.get_vec_ptr(),m_uiEvarDG.get_dof());
            // pMesh->readFromGhostEndEleDGVec(m_uiEvarDG.get_vec_ptr(),m_uiEvarDG.get_dof());

            // const std::vector<ot::Block>& blkList =
            // pMesh->getLocalBlockList(); for(unsigned int blk=0; blk<
            // blkList.size(); blk++)
            // {
            //     // T* tmp = (T*) m_uiBVec[blk]._vec[0].data();
            //     // for(unsigned int w=0; w < m_uiBVec[blk]._vec[0].get_dof()*
            //     m_uiBVec[blk]._vec[0].getSz(); w++)
            //     //     tmp[w]=0;

            //     m_uiBVec[blk]._vec[0].mark_unsynced();
            //     sync_blk_timestep(blk,0);
            //     m_uiBVec[blk]._vec[0].mark_synced();

            // }
        }
    }

    pMesh = m_uiAppCtx->get_mesh();

    if (pMesh->isActive()) {
        const std::vector<ot::Block>& blkList = pMesh->getLocalBlockList();

#ifdef __PROFILE_ENUTS__
        m_uiPt[ENUTSPROFILE::ENUTS_BLK_ZIP].start();
#endif

        const unsigned int DOF = m_uiEVar.get_dof();
        // on labelled meshes the zip plan may source a cg from a block's
        // padding, which is stale after the last sub-step: bring every block's
        // padding to the common end-of-step time first
        // labelled meshes: the block zip plan follows the labelled E2N_DG
        // owner, which can differ from the SFC build's owner; the two block
        // copies of a shared node differ after sub-cycling, so gather each cg
        // from its SFC-canonical owner's block copy through the DG vector
        bool anyNonSFC = false;
        for (const ot::Block& b : blkList) anyNonSFC |= b.isNonSFC();
        if (anyNonSFC) {
            build_cg_owner_map();
            blk_vec_to_zipDG(m_uiBVec.data(), m_uiEvarDG, 0);
            pMesh->readFromGhostBeginEleDGVec(m_uiEvarDG.get_vec_ptr(), DOF);
            pMesh->readFromGhostEndEleDGVec(m_uiEvarDG.get_vec_ptr(), DOF);
            zip_labelled_via_dg();
        }

        for (unsigned int blk = 0; anyNonSFC == false && blk < blkList.size(); blk++) {
            // const unsigned int BLK_T = m_uiBVec[blk]._time;
            // const unsigned int NN   =  m_uiBVec[blk]._vec[0].getSz();
            // const unsigned int bLev =
            // pNodes[blkList[blk].firstElementID()].getLevel(); const
            // unsigned int BLK_DT =
            // m_uiAppCtx->getBlkTimestepFac(bLev,m_uiLevMin,m_uiLevMax);
            m_uiBVec[blk]._vec[0].zip(pMesh, m_uiEVar.get_vec_ptr(), DOF);
        }

#ifdef __PROFILE_ENUTS__
        m_uiPt[ENUTSPROFILE::ENUTS_BLK_ZIP].stop();
#endif
    }

    m_uiAppCtx->post_timestep(m_uiEVar);
    m_uiAppCtx->increment_ts_info((T)coarset_t, 1);
    m_uiTimeInfo = m_uiAppCtx->get_ts_info();
    pMesh->waitAll();

#ifdef __PROFILE_ENUTS__
    m_uiPt[ENUTSPROFILE::ENUTS_EVOLVE].stop();
#endif
}

}  // namespace ts

namespace ts {
// utility functions
/**
 * @brief Perform block padding synchronization for a block vector.
 *
 * @tparam T vector type.
 * @param pMesh :  underlying mesh bVec is created and allocated for.
 * @param bVec  : list of block vectors,
 * @param blk   : block id to synchronize the padding regions.
 * @param bVecIndex : bVec[blk]._vec[bVecIndex] will be synchronized.
 */
template <typename T>
void sync_blk_padding(const ot::Mesh* pMesh, const T* const dgWVec,
                      std::vector<ts::BlockTimeStep<T>>& bVec, unsigned int blk,
                      unsigned int bVecIndex, unsigned int dof = 1);
}  // namespace ts
