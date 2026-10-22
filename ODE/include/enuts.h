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

#include <time.h>
#include <bitset>
#include <fstream>
#include <iomanip>
#include <functional>
#include <iostream>
#include <atomic>
#include <memory>
#include <unordered_map>
#include <functional>
#include <map>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>

#include "blkAsync.h"
#include "block.h"
#include "ets.h"
#include "lts.h"
#include "mathMeshUtils.h"
#include "mesh.h"
#include "meshUtils.h"
#include "oct2vtk.h"
#include "point.h"
#include "subSM.h"
#include "ts.h"
#include "waveletAMR.h"

// the block loops of partial_evolve run threaded in hybrid builds, for a context
// that declares lts_thread_safe
#if defined(DENDRO_HYBRID_OMP) && defined(_OPENMP)
#include <omp.h>
#define DENDRO_ENUTS_HAS_OMP 1
#endif

#define __LTS_BLK_SYNC_TIME_COMP_TOL__ 1e-6

#include "enuts_prof.h"

namespace ts {
/**@brief: Meta-data structure to compute the correct weight for the LTS wpart.
 */
struct WeightedPartitionData {
    unsigned int lmin;
    unsigned int lmax;
    unsigned int offset;
    unsigned int cap;
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

/**@brief out[i] = (..((out[i] + c[0] * st[0][i]) + c[1] * st[1][i]) ..), one
 * pass over the block instead of one per stage, same order per entry. */
template <typename T>
static inline void enuts_axpy_stages(T* __restrict__ out, const T* const* st,
                                     const T* c, unsigned int ns, size_t n) {
    switch (ns) {
        case 0:
            break;
        case 1: {
            const T c0 = c[0];
            const T* a = st[0];
            for (size_t i = 0; i < n; i++) out[i] += (c0 * a[i]);
        } break;
        case 2: {
            const T c0 = c[0], c1 = c[1];
            const T *a = st[0], *b = st[1];
            for (size_t i = 0; i < n; i++) {
                T t = out[i];
                t += (c0 * a[i]);
                t += (c1 * b[i]);
                out[i] = t;
            }
        } break;
        case 3: {
            const T c0 = c[0], c1 = c[1], c2 = c[2];
            const T *a = st[0], *b = st[1], *d = st[2];
            for (size_t i = 0; i < n; i++) {
                T t = out[i];
                t += (c0 * a[i]);
                t += (c1 * b[i]);
                t += (c2 * d[i]);
                out[i] = t;
            }
        } break;
        case 4: {
            const T c0 = c[0], c1 = c[1], c2 = c[2], c3 = c[3];
            const T *a = st[0], *b = st[1], *d = st[2], *e = st[3];
            for (size_t i = 0; i < n; i++) {
                T t = out[i];
                t += (c0 * a[i]);
                t += (c1 * b[i]);
                t += (c2 * d[i]);
                t += (c3 * e[i]);
                out[i] = t;
            }
        } break;
        default:
            for (unsigned int s = 0; s < ns; s++) {
                const T cs = c[s];
                const T* a = st[s];
                for (size_t i = 0; i < n; i++) out[i] += (cs * a[i]);
            }
    }
}

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

    /**@brief level offset of the step ladder (ctx compute_lts_ts_offset)*/
    unsigned int m_uiLevOffset = 0;

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
    /**@brief per local block, its unzip dependent elements (SFC order); rebuilt per mesh */
    std::vector<std::vector<unsigned int>> m_uiBlkDepEid;
    /** per block: m_uiBlkDepEid plus the owners of the nodes of the block's
     * elements and of its dependents (owner sync) */
    std::vector<std::vector<unsigned int>> m_uiBlkOwnEid;

    /**@brief level-scoped element-DG exchange plan;
     * rebuilt per mesh, cleared with m_uiBlkDepEid. m_uiScopedOk false =
     * preconditions failed (divisibility chain, level range), full exchange. */
    ot::EleDGScopedPlan m_uiScopedPlan;
    const ot::Mesh* m_uiScopedMesh = nullptr;
    bool m_uiScopedOk              = false;
    std::vector<unsigned int> m_uiScopedDTs;   // ascending distinct factors
    std::vector<unsigned int> m_uiScopedZero;  // all-zero count row
    std::vector<T> m_uiScopedShadow;
    bool prepare_scoped_exchange(ot::Mesh* pMesh);
    void scoped_check_compare(ot::Mesh* pMesh, const T* real, const T* shadow,
                              unsigned int DOF, unsigned int pt, unsigned int rk);

    /**@brief: temp vector of the DG scheme*/
    DVec m_uiDGTmp0;

    /** DENDRO_ENUTS_IFACE_FIX bit 4: g = k4 - 2 k3 + k1 of every element's
     *  latest completed step (DG, like the stage vectors); zero after every
     *  (re)allocation */
    DVec m_uiStG;
    bool m_uiStGOn = false;
    /** IFACE_FIX bit 4: per element, whether update_stage_g wrote its g since the last (re)allocation */
    std::vector<unsigned char> m_uiStGValid;
    /** per class: whether the class completed a step since the (re)allocation
     *  (every element of a class completes together, so this is the same on
     *  every rank, unlike a per-ghost-element flag before its dt is known) */
    std::vector<unsigned char> m_uiStGValidCls;
    bool gValidCls(unsigned int c) const { return c < m_uiStGValidCls.size() && m_uiStGValidCls[c]; }
    void update_stage_g(unsigned int pt);

    /** per-element tag of the correction currently held in m_uiDGTmp0 */
    struct SyncTag {
        unsigned long long pass = 0;
        unsigned int rk         = 0;
        unsigned char fut       = 0;
        T dt_c = 0, dt_f = 0, dt = 0;
        bool operator==(const SyncTag& o) const {
            return pass == o.pass && rk == o.rk && fut == o.fut &&
                   dt_c == o.dt_c && dt_f == o.dt_f && dt == o.dt;
        }
    };
    /** per-stage-pass store of parent-to-child interpolations: every (coarse
     *  dependent, child) of a plan has a dense index (m_uiP2cIdx, assigned as
     *  plans are built), its slot in the pool is handed out on first use in a
     *  pass. No hashing, no collisions; the pool is as large as one pass needs
     *  (capped, DENDRO_ENUTS_P2C_STORE_MB). */
    std::vector<int> m_uiP2cIdx;
    std::vector<unsigned long long> m_uiP2cPassOf;
    std::vector<unsigned int> m_uiP2cSlotOf;
    std::vector<SyncTag> m_uiP2cTagOf;
    /** prof: coarse step of the last use of a p2c store slot (distinct census) */
    std::vector<unsigned long long> m_uiP2cStepOf;
    unsigned long long m_uiCoarseStep = 0;
    std::vector<T> m_uiP2cPool;
    size_t m_uiP2cUsed                = 0;
    unsigned long long m_uiP2cPoolPass = 0;
    unsigned int p2c_index(unsigned int ele, unsigned int cnum);
    T* p2c_slot(unsigned int idx, const SyncTag& key, size_t stride, bool& hit);
    /** own_p2c_prepass bookkeeping of the current pass */
    bool m_uiP2cSharedOn = false;            // this pass ran the pre-pass
    std::vector<unsigned int> m_uiP2cWorkRow;   // unique rows of the pass
    std::vector<unsigned int> m_uiP2cWorkBlk;   // a reader block per row
    std::vector<unsigned int> m_uiP2cWorkOff;   // children offsets per row
    std::vector<unsigned int> m_uiP2cWorkStore; // store idx per child
    std::vector<unsigned char> m_uiP2cWorkCnum; // child number per child
    std::vector<unsigned int> m_uiP2cWorkRdr, m_uiP2cWorkOutB, m_uiP2cWorkOutE;  // reader block + its runs
    std::vector<T> m_uiOwnRowBuf2;              // one child's values
    std::vector<unsigned long long> m_uiP2cRowPass;  // row -> pass it was listed in
    std::vector<unsigned int> m_uiP2cRowAt;          // row -> index in the work list
    std::vector<SyncTag> m_uiSyncTag;
    unsigned long long m_uiSyncPass = 0;
    bool m_uiSyncPassOpen           = false;
    std::vector<T> m_uiSyncCVec;

    /** per local block, the padding-fill copy plan (per-element sync, fast form):
     *  per dependent element, (src node, dst block index) pairs in the exact
     *  write order of the coordinate loops; cnum >= 0 = coarse parent, p2c
     *  interpolate child cnum first */
    struct SyncPlan {
        struct Ent {
            unsigned int ele;
            int cnum;
            unsigned int b, e;
            unsigned int store;  // p2c store index (cnum >= 0)
        };
        bool built = false;
        std::vector<Ent> ents;
        std::vector<unsigned int> src, dst;
    };
    std::vector<SyncPlan> m_uiSyncPlan;
    void build_sync_plan(unsigned int blk, const std::vector<unsigned int>& eid,
                         SyncPlan& pl);

    /** per local block, the owner-sync fill: every
     *  block slot the padding fill and the interior copy write, with the one
     *  source that survives (their last writer), read through the owner of
     *  the source node. Three kinds of source:
     *   - a node: the owner's DG slot, grouped by the owner's time-step class
     *     (one stage-conversion row per class and stage pass). Owner elements
     *     that feed many slots are converted whole into m_uiDGTmp0, once per
     *     pass (m_uiSyncTag), and copied in runs; the rest is converted per
     *     node;
     *   - a hanging face / edge of an element: its 2D / 1D parent-to-child
     *     interpolation of owner values;
     *   - a child of a coarser dependent: the 3D interpolation of the
     *     dependent's reconciled row (m_uiOwnRow*). */
    struct OwnPlan {
        bool built = false;
        // per class: runs (rdst, rsrc, rlen) reading the elements cele
        std::vector<unsigned int> runOff, celeOff;
        std::vector<unsigned int> rdst, rsrc, cele;
        std::vector<unsigned short> rlen;
        // per class: single nodes
        std::vector<unsigned int> clsOff;
        std::vector<unsigned int> pdst, psrc;
        struct Op {
            unsigned int inB, segB, segE, outB, outE;
            unsigned char dim, cnum;
        };
        std::vector<Op> ops;
        // op inputs sorted by class: (input index, owner DG slot); segments
        // (class, end offset relative to inB)
        std::vector<unsigned int> opSrc, opT;
        std::vector<unsigned short> opSegEnd;
        std::vector<unsigned char> opSegCls;
        std::vector<unsigned int> oDst;
        std::vector<unsigned short> oT;
        struct Child {
            unsigned int row, store, outB, outE;
            unsigned char cnum;
        };
        std::vector<Child> ch;  // outputs: runs (child node, block slot, length)
        std::vector<unsigned int> cDst;
        std::vector<unsigned short> cT, cLen;
    };
    std::vector<OwnPlan> m_uiOwnPlan;
    /** stage-conversion of one owner class into a block's time frame */
    struct OwnConv {
        // g: the last of the ns inputs is m_uiStG
        unsigned char same = 1, ns = 0, g = 0;
        T m[8] = {};
    };
    /** stage-conversion rows of one block class for this pass (thread-local
     *  cache); bTime / bDT of the reader block */
    const OwnConv* own_conv_rows(unsigned int bc, unsigned int rk_s, unsigned int bTime,
                                 unsigned int bDT);
    /** the reconciled DG row of a dependent (owner values, then its hanging
     *  faces and edges) in the frame of a reader block of class rows cv */
    void own_row_build(unsigned int row, unsigned int rk_s, const OwnConv* cv,
                       T* rowBuf, T* g, T* pin, T* pout, double* im1, double* im2);
    /** before each stage's sync loop (owner plan): list the unique children
     *  (parent element, child) of the active blocks once; a threaded pass over
     *  the unique parent rows builds each row once and writes each child's
     *  outputs straight into every reader block (the child slots have no other
     *  writer), so the block loop skips them. Same row build and interpolation
     *  code as the per-block path, so bitwise. */
    void own_p2c_prepass(unsigned int rk_s, const unsigned int* actIds, unsigned int nAct,
                         bool thr);
    const ot::Mesh* m_uiOwnMesh = nullptr;
    bool m_uiOwnOk              = false;
    unsigned int m_uiOwnPt      = 0;
    /** ascending distinct time-step factors (the owner classes) */
    std::vector<unsigned int> m_uiOwnDTs;
    bool m_uiOwnChain = false;
    /** per cg node, the class of its owner element (ghost nodes: as sent by
     *  the owning rank) */
    std::vector<unsigned char> m_uiNodeCls;
    /** per element, its hanging face / edge interpolations and corner
     *  re-gathers in the order of Mesh::getElementNodalValues: id | cnum << 5,
     *  id < 6 face, < 18 edge, else corner 18 + c */
    std::vector<unsigned int> m_uiOwnOpOff;
    std::vector<unsigned char> m_uiOwnOps;
    std::vector<unsigned int> m_uiOwnSlots;
    /** rows of the coarser dependents: the nodes another element owns, per
     *  class (m_uiOwnRowOff, nCls + 1 offsets per row), and their owner DG
     *  slots; the rest is the element's own DG row (class m_uiOwnRowPc) */
    std::vector<unsigned char> m_uiOwnRowPc;
    std::vector<int> m_uiOwnRowOf;
    std::vector<unsigned int> m_uiOwnRowEle;
    std::vector<unsigned int> m_uiOwnRowOff;
    std::vector<unsigned int> m_uiOwnRowSrc, m_uiOwnRowN;
    /** the reconciled row computed last (all dof) */
    std::vector<T> m_uiOwnRowBuf;
    unsigned int m_uiOwnRowBufRow           = ~0u;
    unsigned long long m_uiOwnRowBufPass    = 0;
    /** x and x-y sweeps of one row's 3D interpolation, per variable and
     *  child parity (6 * nPe each): children of one parent share them */
    std::vector<T> m_uiOwnSweep;
    std::vector<unsigned char> m_uiOwnSweepOk;
    /** RefElement::I3D_Parent2Child of one variable sweep by sweep (x, y, z,
     *  each with the operator of the child's parity): the x sweep serves the
     *  four children of one x parity, the x-y sweep two. Same kernels, same
     *  operands, same result. */
    static inline void p2c3_shared(const ot::Mesh* pMesh, const T* row, T* out,
                                   unsigned int cnum, T* sw, unsigned char* ok) {
        const unsigned int n1  = pMesh->getElementOrder() + 1;
        const size_t nPe       = pMesh->getNumNodesPerElement();
        const double* ipc[2]   = {pMesh->getReferenceElement()->getIMChild0(),
                                  pMesh->getReferenceElement()->getIMChild1()};
        const unsigned int cx = cnum & 1u, cy = (cnum >> 1) & 1u, cz = (cnum >> 2) & 1u;
        T* xs = sw + (size_t)cx * nPe;
        T* ys = sw + (size_t)(2 + 2 * cx + cy) * nPe;
        if (!ok[cx]) {
            DENDRO_TENSOR_IIAX_APPLY_ELEM(n1, ipc[cx], row, xs);
            ok[cx] = 1;
        }
        if (!ok[2 + 2 * cx + cy]) {
            DENDRO_TENSOR_IAIX_APPLY_ELEM(n1, ipc[cy], xs, ys);
            ok[2 + 2 * cx + cy] = 1;
        }
        DENDRO_TENSOR_AIIX_APPLY_ELEM(n1, ipc[cz], ys, out);
    }
    std::vector<OwnConv> m_uiOwnConv;
    std::vector<unsigned long long> m_uiOwnConvPass;
    std::vector<T> m_uiOwnScr;
    /** owner-node exchange (more than one rank): per neighbor the owner DG
     *  slots of the shared nodes sorted by class, cumulative counts per class */
    std::vector<unsigned char> m_uiOwnNodeRecv;
    std::vector<unsigned int> m_uiOwnSendSlot, m_uiOwnRecvSlot;
    std::vector<unsigned int> m_uiOwnSendOff, m_uiOwnRecvOff;
    std::vector<unsigned int> m_uiOwnSendCum, m_uiOwnRecvCum;
    /** G_XCH with the scoped exchange: the unscoped lists, kept at the re-sort (the
     *  scoped order sends a slot only when a class that reads it is
     *  stepping, which a g written at a step end would miss) */
    std::vector<unsigned int> m_uiOwnSendSlotU, m_uiOwnRecvSlotU, m_uiOwnSendCumU, m_uiOwnRecvCumU;
    std::vector<T> m_uiOwnSendBuf, m_uiOwnRecvBuf;
    std::vector<MPI_Request> m_uiOwnReq;
    /** the fill converts a listed element whole (every node of its row) only
     *  when its class differs from the reader block's and it is coarser, or
     *  the equal-time interface fix converts it anyway; otherwise the block
     *  copies the run slots alone (the padding footprint) */
    bool xch_whole_row(unsigned int ele, unsigned int bc) const {
        const ot::Mesh* pMesh = m_uiAppCtx->get_mesh();
        const unsigned int l  = pMesh->getAllElements()[ele].getLevel();
        const unsigned int dt = step_ratio(l);
        const unsigned int ce = (unsigned int)(
            std::lower_bound(m_uiOwnDTs.begin(), m_uiOwnDTs.end(), dt) - m_uiOwnDTs.begin());
        return ce != bc && (ce > bc || (ts::EnutsProf::ifaceFix() & 1));
    }
    /** DENDRO_ENUTS_XCH_CENSUS: per (stage, DG slot) 0 never received, 1
     *  received and not read since, 2 read; per DG slot, whether a built plan
     *  reads it at all */
    std::vector<unsigned char> m_uiXchFlag, m_uiXchInPlan;
    size_t m_uiXchDg = 0;
    /** every stage input a conversion row c reads at slot: k_{rk_s} alone when
     *  the class is at the block's time, else the ns stage inputs (g aside) */
    inline void xch_mark(unsigned int slot, const OwnConv& c, unsigned int rk_s) {
        unsigned char* f = m_uiXchFlag.data() + slot;
        if (c.same) {
            f[(size_t)(rk_s - 1) * m_uiXchDg] = 2;
            return;
        }
        const unsigned int ns = c.ns - c.g;
        for (unsigned int k = 0; k < ns; k++) f[(size_t)k * m_uiXchDg] = 2;
    }
    inline void xch_mark_ele(unsigned int ele, unsigned int nPe, const OwnConv& c,
                             unsigned int rk_s) {
        for (unsigned int n = 0; n < nPe; n++) xch_mark(ele * nPe + n, c, rk_s);
    }
    /** DENDRO_ENUTS_XCH_TRACE: per recv partner, its clock minus mine (NaN:
     *  no round trip possible), the records of every exchange call */
    std::vector<double> m_uiXchOff, m_uiXchLat, m_uiXchSpread;
    std::vector<unsigned int> m_uiXchOffRank;
    double m_uiXchT0 = 0;
    bool m_uiXchCal  = false;
    struct XchRec {
        unsigned int pt, s, j, arg, lateArg;
        unsigned char argSend;
        double tw, wait, rwait, first, late, lateMax, nrecv, argN;
    };
    std::vector<XchRec> m_uiXchRec;
    void xch_calibrate(ot::Mesh* pMesh);
    void xch_dump();
    /** owner exchange scoped by reader (owner plan, chained classes): once
     *  every block's plan exists (after the first sync pass) each rank tells
     *  each sender, per shared node, the class of the finest local block that
     *  reads it (255: none), and both sides re-sort the node list by
     *  max(owner class, reader class): a stage is sent only at sub-steps where
     *  a reader of it steps, never-read nodes never. Bitwise: a slot not sent
     *  is not read before it is sent again. True once the lists are re-sorted
     *  (once per mesh). */
    bool m_uiXchScoped = false;
   private:
    bool xch_scope_build(ot::Mesh* pMesh);
    /** owner sync: before a block takes its stage data, give every shared or
     *  hanging node of the block's elements and of its unzip dependents the
     *  value of the node's owner (what zip + unzip does in ETS): by the
     *  per-block plan (OwnPlan, owner values exchanged per node on more than
     *  one rank), or per element where the plan is unavailable. True when the
     *  plan is built for the current mesh. */
    bool prepare_owner_sync(ot::Mesh* pMesh);
    void build_own_plan(unsigned int blk, const std::vector<unsigned int>& eid,
                        OwnPlan& op);
    void own_sync_fill(unsigned int blk, unsigned int rk_s,
                       const std::vector<unsigned int>& eid);
    void own_row_ops(unsigned int ele, const T* g, T* out, T* pin, T* pout,
                     double* im1, double* im2) const;
    void own_node_exchange_begin(T* stv, unsigned int dof, unsigned int pt);
    void own_node_exchange_end(T* stv, unsigned int dof, unsigned int pt);
    /** with m_uiSyncPlan / m_uiBlkDepEid: the plans hold p2c store indices */
    void own_reset() {
        m_uiOwnMesh = nullptr;
        m_uiOwnOk   = false;
        m_uiOwnPlan.clear();
        m_uiP2cIdx.clear();
        m_uiP2cPassOf.clear();
        m_uiP2cSlotOf.clear();
        m_uiP2cTagOf.clear();
        m_uiP2cStepOf.clear();
        m_uiP2cPoolPass  = 0;
        m_uiOwnRowBufRow = ~0u;
        m_uiXchFlag.clear();
        m_uiXchInPlan.clear();
        m_uiXchCal    = false;
        m_uiXchScoped = false;
    }

    /** DENDRO_ENUTS_OMP: the scratch and per-pass caches of worker thread
     *  t >= 1 in sync_blk_timestep / own_sync_fill / p2c_slot (thread 0 keeps
     *  the shared members). dgW / cgW stand in for m_uiDGTmp0 / m_uiEVecTmp[0]
     *  and are left uninitialised: every row read was written by the same
     *  thread (in the block call, or under this thread's tags). The caches
     *  only skip recomputing identical values, so per-thread copies give the
     *  same result. */
    struct SyncThr {
        ENUTSOp* op = nullptr;
        std::unique_ptr<T[]> dgW, cgW;
        std::vector<T> cVec;
        std::vector<SyncTag> tag;
        // p2c pass store (p2c_slot)
        std::vector<unsigned long long> p2cPassOf;
        std::vector<unsigned int> p2cSlotOf;
        std::vector<SyncTag> p2cTagOf;
        std::vector<T> p2cPool;
        std::vector<T> ownRowBuf2;
        size_t p2cUsed                = 0;
        unsigned long long p2cPoolPass = 0;
        // owner-sync plan (own_sync_fill) and the shared p2c sweeps
        std::vector<OwnConv> ownConv;
        std::vector<unsigned long long> ownConvPass;
        std::vector<T> ownScr, ownRowBuf, ownSweep;
        std::vector<unsigned char> ownSweepOk;
        unsigned int ownRowBufRow        = ~0u;
        unsigned long long ownRowBufPass = 0;
    };
    std::vector<SyncThr> m_uiSyncThr;
    /** active blocks of the partial step, largest first (DENDRO_ENUTS_OMP) */
    std::vector<unsigned int> m_uiActiveOrder;
    /** prof: seconds spent on each active block in the current rhs loop call */
    std::vector<double> m_uiRlBlkT;
    /** DENDRO_ENUTS_CLASS_WORK=1 (prof): this rank's rhs block loop by
     *  time-step class (blocks, elements, padded and interior points of a
     *  pass, calls, thread-s) and by sub-step type (the coarsest class
     *  active at pt: loop calls, wall, threads x wall, thread-s); the
     *  static columns describe the last mesh, printed per rank at the end */
    std::vector<unsigned int> m_uiCwDTs;
    std::vector<unsigned char> m_uiCwBlkCls;
    std::vector<double> m_uiCwCls;
    std::vector<double> m_uiCwType;
    static bool classWork() {
        static const int v = []() {
            const char* e = std::getenv("DENDRO_ENUTS_CLASS_WORK");
            return (e && e[0] == '1') ? 1 : 0;
        }();
        return v == 1;
    }
    void class_work_setup() {
        const ot::Mesh* pMesh    = m_uiAppCtx->get_mesh();
        const ot::Block* blkList = pMesh->getLocalBlockList().data();
        const unsigned int nBlk  = pMesh->getLocalBlockList().size();
        const ot::TreeNode* pNodes = pMesh->getAllElements().data();
        m_uiCwDTs.clear();
        for (unsigned int l = m_uiLevMin; l <= m_uiLevMax; l++)
            m_uiCwDTs.push_back(step_ratio(l));
        std::sort(m_uiCwDTs.begin(), m_uiCwDTs.end());
        m_uiCwDTs.erase(std::unique(m_uiCwDTs.begin(), m_uiCwDTs.end()), m_uiCwDTs.end());
        const size_t nCls = m_uiCwDTs.size();
        std::vector<double> cls(nCls * 6, 0.0), typ(nCls * 4, 0.0);
        // keep the dynamic columns across a remesh
        for (size_t k = 0; k < nCls && k * 6 + 5 < m_uiCwCls.size(); k++) {
            cls[k * 6 + 4] = m_uiCwCls[k * 6 + 4];
            cls[k * 6 + 5] = m_uiCwCls[k * 6 + 5];
        }
        for (size_t i = 0; i < std::min(typ.size(), m_uiCwType.size()); i++) typ[i] = m_uiCwType[i];
        m_uiCwBlkCls.assign(nBlk, 0);
        for (unsigned int blk = 0; blk < nBlk; blk++) {
            const unsigned int lev = pNodes[blkList[blk].firstElementID()].getLevel();
            const unsigned int dt  = step_ratio(lev);
            const unsigned int k   = (unsigned int)(
                std::lower_bound(m_uiCwDTs.begin(), m_uiCwDTs.end(), dt) - m_uiCwDTs.begin());
            m_uiCwBlkCls[blk] = (unsigned char)k;
            const unsigned int lx = blkList[blk].getAllocationSzX(), ly = blkList[blk].getAllocationSzY(),
                               lz = blkList[blk].getAllocationSzZ(), pw = blkList[blk].get1DPadWidth();
            cls[k * 6 + 0] += 1;
            for (const unsigned int e : blkList[blk]) { (void)e; cls[k * 6 + 1] += 1; }
            cls[k * 6 + 2] += (double)lx * ly * lz;
            cls[k * 6 + 3] += (double)(lx - 2 * pw) * (ly - 2 * pw) * (lz - 2 * pw);
        }
        m_uiCwCls.swap(cls);
        m_uiCwType.swap(typ);
    }
    /** the per-mesh tables sync_blk_timestep builds on first use */
    void sync_lazy_build(bool ownEid);
    /** threads: everything a block call would build on first use (tables and
     *  the plans of all blocks), before the parallel region */
    void sync_prebuild(bool ownPlan, bool ownEid, bool syncFast);
    void sync_thread_setup(bool cgScratch);
    void free_sync_threads();

    /**@brief: List of block async vector. */
    std::vector<ts::BlockTimeStep<T>> m_uiBVec;

    /** The stage input (the block solution plus the stage terms) is one
     *  scratch vector per thread, sized for the largest block: it is live only
     *  between its build and the RHS call of the same block and stage.
     *
     *  With the ring (tableaus with a_{s,j} = 0 for j < s-1, i.e. RK4) a block
     *  keeps one stage vector instead of NS, plus an accumulator: the stage
     *  input of stage s needs only k_{s-1}, and the final update
     *  U + sum_s b_s dt k_s is accumulated in the same per-entry order as the
     *  one-pass update, its term b_{s-1} dt k_{s-1} added in the stage-input
     *  pass of stage s (the last one in the sync loop). The dropped
     *  zero-coefficient terms 0 * k_j only turn a -0.0 entry of U into +0.0
     *  where some k_j has its sign bit clear; a per-block sign mask reproduces
     *  that, so the stage inputs are bit-identical to full storage. */
    bool m_uiStageRing = false;
    /** per-thread stage-input scratch, sized for the largest block */
    std::vector<BlockAsyncVector<T>> m_uiStageIn;
    /** ring: per block, bit i set when a dropped stage k_j (j < s-1) had its
     *  sign bit clear at entry i (0 * k_j = +0.0 there) */
    std::vector<std::vector<uint64_t>> m_uiStageSgn;
    /** ring slots: 0 = solution, 1 = the one stage vector, 2 = accumulator */
    static constexpr unsigned int RING_K = 1, RING_ACC = 2;
    /** slot of RK stage s (0 = solution) in m_uiBVec[blk]._vec */
    inline unsigned int stage_slot(unsigned int s) const {
        return (m_uiStageRing && s > 0) ? RING_K : s;
    }
    /** the stage-input vector of a block: the thread's scratch */
    inline BlockAsyncVector<T>& stage_in(unsigned int blk) {
        return m_uiStageIn[ts::EnutsProf::tid()];
    }
    /** vector behind a debug index s in 0..NS+1 (NS+1 = the stage input) */
    inline const BlockAsyncVector<T>& stage_vec_dbg(unsigned int blk,
                                                    unsigned int s) {
        return (s == m_uiNumStages + 1) ? stage_in(blk)
                                        : m_uiBVec[blk]._vec[stage_slot(s)];
    }
    inline void mark_stages_unsynced(unsigned int blk) {
        for (unsigned int s = 1; s < m_uiBVec[blk]._vec.size(); s++)
            m_uiBVec[blk]._vec[s].mark_unsynced();
    }
    /** temporaries only some paths read are allocated on first use */
    void ensure_evar_dg() {
        if (m_uiEvarDG.get_vec_ptr() == nullptr)
            m_uiEvarDG.create_vector(m_uiAppCtx->get_mesh(),
                                     ot::DVEC_TYPE::OCT_LOCAL_NODES,
                                     ot::DVEC_LOC::HOST, m_uiEVar.get_dof(),
                                     true);
    }
    void ensure_cg_tmp() {
        if (m_uiEVecTmp[0].get_vec_ptr() == nullptr)
            m_uiEVecTmp[0].create_vector(m_uiAppCtx->get_mesh(),
                                         ot::DVEC_TYPE::OCT_SHARED_NODES,
                                         ot::DVEC_LOC::HOST, m_uiEVar.get_dof(),
                                         true);
    }
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
    unsigned int m_uiPartialStepId;

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

    /**@brief DENDRO_TS_MEM_INFO=1: print the bytes of every buffer family */
    void mem_report();

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

    /**@brief: update the element timestep (all blocks, or the given ones)*/
    void update_ele_timestep(const std::vector<unsigned int>* blks = nullptr);

    /**@brief: per-block stepping rules out embedded error control. */
    bool supports_error_control() const override { return false; }

   public:
    /// LTS stepper over the context's mesh. Blocks of the octree-to-block
    /// decomposition must be 2:1 balanced. Collective: no (init() is).
    /// Aborts unless `opts.iface` is 3 or 15.
    /// @param useWpart  repartition by LTS work after the initial grid
    ExplicitNUTS(Ctx* ctx, bool useWpart = true,
                 const ts::LtsOptions& opts = ts::LtsOptions());

    using ETS<T, Ctx>::set_ets_coefficients;
    /// The Runge-Kutta tableau; aborts unless ts::lts_tableau_supported
    /// (RK3, RK4). Collective: no.
    int set_ets_coefficients(ETSType type) {
        if (!ts::lts_tableau_supported(type)) {
            std::cerr << "[enuts] the LTS stepper runs RK3 or RK4, not ETSType "
                      << (int)type << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        return ETS<T, Ctx>::set_ets_coefficients(type);
    }

    /// Step of a block at level `blev` in units of the finest step.
    /// Collective: no. Thread-safe: yes.
    unsigned int step_ratio(unsigned int blev) const {
        return ts::step_ratio(blev, m_uiLevMax, m_uiLevOffset,
                              ts::EnutsProf::options().ratio_cap);
    }

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


    /**@brief: returns  a constant pointer to the sub scatter maps. */
    // inline ot::SubScatterMap* const get_sub_scatter_maps() const {return
    // m_uiSubSM;}


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
        return m_uiTimeInfo._m_uiTh * step_ratio(m_uiLevMax);
    }

    /**@brief returns the dt max value */
    T get_dt_max() const {
        return m_uiTimeInfo._m_uiTh * step_ratio(m_uiLevMin);
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
        const WeightedPartitionData& w = wpart_meta;
        const unsigned int finest_t  = ts::step_ratio(w.lmax, w.lmax, w.offset, w.cap);
        const unsigned int coarset_t = ts::step_ratio(w.lmin, w.lmax, w.offset, w.cap);
        const unsigned int weight =
            (coarset_t - finest_t) /
            ts::step_ratio(pNode->getLevel(), w.lmax, w.offset, w.cap);
        return std::max(1u, weight);
    }

    /**@brief: update wpart meta data from current variables. */
    void update_wpart_metadata() {
        wpart_meta.lmin   = m_uiLevMin;
        wpart_meta.lmax   = m_uiLevMax;
        wpart_meta.offset = m_uiLevOffset;
        wpart_meta.cap    = ts::EnutsProf::options().ratio_cap;

        return;
    }
};

template <typename T, typename Ctx>
ExplicitNUTS<T, Ctx>::ExplicitNUTS(Ctx* ctx, bool useWpart,
                                   const ts::LtsOptions& opts)
    : ETS<T, Ctx>(ctx) {
    if (!ts::lts_options_valid(opts)) {
        std::cerr << "[enuts] LtsOptions::iface is 3 or 15, not " << opts.iface
                  << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    ts::EnutsProf::options()           = opts;
#ifdef DENDRO_ENUTS_HAS_OMP
    ts::EnutsProf::threaded() = ts::lts_thread_safe_v<Ctx>;
#endif
    ts::EnutsProf::options().ratio_cap = ts::snap_ratio_cap(opts.ratio_cap);
    m_uiUseWeightedPart                = useWpart;
    this->init_data_structures();
}

template <typename T, typename Ctx>
ExplicitNUTS<T, Ctx>::~ExplicitNUTS() {
    if (ts::EnutsProf::scopedCheck() && ts::EnutsProf::get().chk_cmp) {
        // drivers may delete the app ctx before the ENUTS object: no ctx access here
        int chk_rk = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &chk_rk);
        std::cout << "[enuts-scoped-check r" << chk_rk
                  << "] compared_elems=" << ts::EnutsProf::get().chk_cmp
                  << " mismatches=" << ts::EnutsProf::get().chk_bad << std::endl;
    }
    if (ts::EnutsProf::on()) {
        // collective over MPI_COMM_WORLD: every rank destroys its ENUTS object
        xch_dump();
        ts::EnutsProf& P = ts::EnutsProf::get();
        double loc[ts::EnutsProf::NT + 4];
        for (int i = 0; i < ts::EnutsProf::NT; i++) loc[i] = P.t[i];
        loc[ts::EnutsProf::NT + 0] = P.partial;
        loc[ts::EnutsProf::NT + 1] = P.lev_sum;
        loc[ts::EnutsProf::NT + 2] = P.blk_sum;
        loc[ts::EnutsProf::NT + 3] = P.blk_tot;
        double mn[ts::EnutsProf::NT + 4], mx[ts::EnutsProf::NT + 4],
            sm[ts::EnutsProf::NT + 4];
        MPI_Allreduce(loc, mn, ts::EnutsProf::NT + 4, MPI_DOUBLE, MPI_MIN,
                      MPI_COMM_WORLD);
        MPI_Allreduce(loc, mx, ts::EnutsProf::NT + 4, MPI_DOUBLE, MPI_MAX,
                      MPI_COMM_WORLD);
        MPI_Allreduce(loc, sm, ts::EnutsProf::NT + 4, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
        int rk = 0, np = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rk);
        MPI_Comm_size(MPI_COMM_WORLD, &np);
        if (rk == 0 && mx[ts::EnutsProf::NT] > 0) {
            static const char* nm[ts::EnutsProf::NT] = {
                "rhs+zipDG",  "eledg_begin", "eledg_end", "sync_blk_ts+post",
                "axpy+ele_ts", "elevec_xch",  "waitActive", "partial_total",
                // owner-sync plan reuses the first four: plan + conversions,
                // node copies, hanging faces / edges, children of coarser dependents
                "  sync:same_copy|own:prep", "  sync:corr_coarse_fut|own:copy", "  sync:corr_fine_past|own:hang", "  sync:unzip_pad|own:child", "  sync:own_interior",
                "  sync:post_stage_blk", "  sync:  of unzip: p2c interp",
                "  sync:  of unzip: row build", "  sync:  of unzip: child copy",
                "  sync:  p2c pre-pass (shared store)",
                "  sync:blk_ts call total (thread-s)", "  sync:ring axpy (thread-s)",
                "  sync:block loop wall", "  sync:prof census (serial, prof-only)",
                "  xch: wait all (owner path)", "  xch:   of wait: to first recv done", "  xch: unpack",
                "  count: sync calls", "  count: dep elems", "  count: copy entries", "  count: copy pairs",
                "  count: p2c children", "  count: corrections", "  count: corr cache hits", "  count: p2c cache hits",
                "  count: p2c unique/pass", "  count: p2c cross-thread dups", "  count: p2c pool overflows",
                "  count: xch doubles recv", "  count: xch doubles sent", "  count: xch calls",
                "  count: coarse steps", "  count: p2c distinct children (sum over coarse steps)",
                "  count: hoisted-p2c store doubles (classes x (NS+1) x child, sum over coarse steps)",
                "  sync:    of row build: parent hanging ops", "  count: rows built", "  count: rows with hanging ops",
                "  xch:   of wait: arg-max partner late by (sum of positive; trace)",
                "  xch:   of wait: max partner lateness (sum of positive; trace)",
                "  count: xch calls whose last request was a send (trace)",
                "  count: xch calls with unknown arg-max lateness (trace)",
                "  count: xch doubles recv, no local plan reads the slot (census)",
                "  count: xch doubles recv, in a plan, overwritten unread (census)",
                "  count: xch doubles recv, read before overwritten (census)",
                "  count: xch calls with lateness > wait (trace; clock offset error)",
                // the rhs block loop (the rhs+zipDG row is its wall): per-block work in thread-s
                "  rhs:  stage input sweep (thread-s)", "  rhs:  pre_stage_blk (thread-s)",
                "  rhs:  rhs_blk (thread-s)", "  rhs:  zipDG (thread-s)",
                "  rhs:  other per block: pre_timestep_blk, marks (thread-s)",
                "  rhs:  threads x loop wall (thread-s capacity)",
                // the same loop by sub-step class: finest-class-only sub-steps / all others
                "  rhs:  finest-only sub-steps: loop wall", "  rhs:    threads x wall",
                "  rhs:    per-block work (thread-s)", "  rhs:    largest block's seconds (sum of per-call max)",
                "  count: rhs loop calls on finest-only sub-steps",
                "  rhs:  other sub-steps: loop wall", "  rhs:    threads x wall",
                "  rhs:    per-block work (thread-s)", "  rhs:    largest block's seconds (sum of per-call max)",
                "  count: rhs loop calls on other sub-steps",
                // partial_evolve before its stage loop, and evolve() around the partial steps
                "partial_prolog (active list, owner plan, prebuild)",
                "evolve_total", "  evolve: ctx pre_timestep", "  evolve: ctx unzip (load)",
                "  evolve: block load from unzip", "  evolve: element time init + exchange",
                "  evolve: end-of-step zip", "  evolve: ctx post_timestep + waitAll",
                // rhs_blk by block span (elements per side): thread-s, then padded points
                "  rhs:  rhs_blk on 1-element blocks (thread-s)", "  rhs:  rhs_blk on 2-element blocks (thread-s)",
                "  rhs:  rhs_blk on 4-element blocks (thread-s)", "  rhs:  rhs_blk on 8+-element blocks (thread-s)",
                "  count: rhs_blk points on 1-element blocks", "  count: rhs_blk points on 2-element blocks",
                "  count: rhs_blk points on 4-element blocks", "  count: rhs_blk points on 8+-element blocks",
                // IFACE_FIX bit 4: nodes whose stage conversion read a stored g that no step had written yet
                // (first step of the element's class after a (re)allocation), and the (re)allocations
                "  count: stage conversions reading g = 0 (nodes, once per node)", "  count: g vector (re)allocations",
                "  count: stage conversions reading g (nodes, once per node, valid or not)"};
            std::cout << "[enuts-prof] npes=" << np
                      << " partial_steps(max rank)=" << mx[ts::EnutsProf::NT]
                      << " mean_active_levels/step="
                      << mx[ts::EnutsProf::NT + 1] /
                             std::max(1.0, mx[ts::EnutsProf::NT])
                      << " mean_active_blk_frac(local,rank-avg)="
                      << sm[ts::EnutsProf::NT + 2] /
                             std::max(1.0, sm[ts::EnutsProf::NT + 3])
                      << std::endl;
#ifdef DENDRO_ENUTS_HAS_OMP
            std::cout << "[enuts-prof] enuts_omp=" << ts::EnutsProf::ompOn()
                      << " omp_max_threads=" << omp_get_max_threads()
                      << (ts::EnutsProf::ompOn()
                              ? " (sync:* rows and counts are summed over threads)"
                              : "")
                      << std::endl;
#endif
            std::cout << "[enuts-prof] phase min mean max (s over ranks)"
                      << std::endl;
            for (int i = 0; i < ts::EnutsProf::NT; i++)
                std::cout << "[enuts-prof] " << nm[i] << " " << mn[i] << " "
                          << sm[i] / np << " " << mx[i] << std::endl;
            // the two sub-step classes partition the rhs loop calls
            const double cw = sm[ts::EnutsProf::RC_WALL0] + sm[ts::EnutsProf::RC_WALL1];
            if (std::fabs(cw - sm[ts::EnutsProf::RHS]) > 1e-9 * std::max(1.0, sm[ts::EnutsProf::RHS]))
                std::cout << "[enuts-prof] WARNING: rhs class walls " << cw
                          << " != rhs+zipDG " << sm[ts::EnutsProf::RHS] << std::endl;
        }
        if (classWork()) {
            // per rank and class: the static columns and the loop by sub-step type
            int nc = (int)m_uiCwDTs.size(), ncm = 0;
            MPI_Allreduce(&nc, &ncm, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
            const int W = 1 + 11 * ncm;
            std::vector<double> row((size_t)W, 0.0), all(rk == 0 ? (size_t)W * np : 0);
            row[0] = nc;
            for (int k = 0; k < nc; k++) {
                row[1 + k * 11] = m_uiCwDTs[k];
                for (int i = 0; i < 6; i++) row[2 + k * 11 + i] = m_uiCwCls[(size_t)k * 6 + i];
                for (int i = 0; i < 4; i++) row[8 + k * 11 + i] = m_uiCwType[(size_t)k * 4 + i];
            }
            MPI_Gather(row.data(), W, MPI_DOUBLE, all.data(), W, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            if (rk == 0) {
                std::cout << "[lts-class-work] rank cls ratio blocks elements pad_pts int_pts rhs_calls "
                             "rhs_thread_s type_loops type_wall type_cap type_thread_s" << std::endl;
                for (int r = 0; r < np; r++) {
                    const double* a = all.data() + (size_t)r * W;
                    for (int k = 0; k < (int)a[0]; k++) {
                        std::cout << "[lts-class-work] " << r << " " << k;
                        for (int i = 0; i < 11; i++) std::cout << " " << a[1 + k * 11 + i];
                        std::cout << std::endl;
                    }
                }
            }
        }
        for (int i = 0; i < ts::EnutsProf::NT; i++) P.t[i] = 0;
        P.partial = P.lev_sum = P.blk_sum = P.blk_tot = 0;
    }
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

        const unsigned int finest_t = step_ratio(m_uiLevMax);  // finest  time level.
        const unsigned int coarset_t = step_ratio(m_uiLevMin);  // coarset time level.

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
                    step_ratio(bLev);
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
            //     step_ratio(bLev);
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
            const DendroIntL offset = blkList[blk].getOffset();

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
void ExplicitNUTS<T, Ctx>::init() {
    m_uiAppCtx->initialize();
    m_uiTimeInfo = m_uiAppCtx->get_ts_info();
    allocate_internal_vars();
    m_uiLevOffset = m_uiAppCtx->compute_lts_ts_offset();

    if (m_uiUseWeightedPart) {
        ot::Mesh* pMesh = m_uiAppCtx->get_mesh();
        pMesh->computeMinMaxLevel(m_uiLevMin, m_uiLevMax);

        update_wpart_metadata();

        if (!m_uiAppCtx->is_wpart_func_set())
            m_uiAppCtx->set_wpart_function(getOctWeight);

        DendroIntL oldElements = pMesh->getNumLocalMeshElements(), oldElements_g = 0;
        par::Mpi_Reduce(&oldElements, &oldElements_g, 1, MPI_SUM, 0,
                        pMesh->getMPIGlobalCommunicator());
        // the IGT reads ghost cgs of local elements, and the context's
        // initialize may end on a local-only init_grid after an IGT, which
        // leaves those ghosts zero: elements that change rank would carry the
        // zeros to their new owner
        if (pMesh->isActive()) {
            auto& ev = m_uiAppCtx->get_evolution_vars();
            pMesh->readFromGhostBegin(ev.get_vec_ptr(), ev.get_dof());
            pMesh->readFromGhostEnd(ev.get_vec_ptr(), ev.get_dof());
        }
        m_uiAppCtx->remesh_and_gridtransfer();
        m_uiAppCtx->on_mesh_changed(false);
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
    const T* v            = stage_vec_dbg(blk, s).data();
    const unsigned int n  = m_uiEVar.get_dof() * stage_vec_dbg(blk, s).getSz();
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
        if (s == 0) ensure_evar_dg();
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
    const T* v           = stage_vec_dbg(blk, s).data();
    const unsigned int n = m_uiEVar.get_dof() * stage_vec_dbg(blk, s).getSz();
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
void ExplicitNUTS<T, Ctx>::update_ele_timestep(
    const std::vector<unsigned int>* blks) {
    ot::Mesh* pMesh = m_uiAppCtx->get_mesh();

    if (!(pMesh->isActive())) return;

    const ot::TreeNode* pNodes            = pMesh->getAllElements().data();
    const std::vector<ot::Block>& blkList = pMesh->getLocalBlockList();
    // only the given blocks moved: the others keep their time, and dt is a
    // function of the level
    const unsigned int nb = blks ? blks->size() : blkList.size();
    for (unsigned int bi = 0; bi < nb; bi++) {
        const unsigned int blk = blks ? (*blks)[bi] : bi;
        const unsigned int bLev =
            pNodes[blkList[blk].firstElementID()].getLevel();
        const unsigned int BLK_DT =
            step_ratio(bLev);

        for (const unsigned int ele : blkList[blk]) {
            m_uiEleTime[ele] = m_uiBVec[blk]._time;
            m_uiEleDT[ele]   = BLK_DT;
        }
    }
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::update_stage_g(unsigned int pt) {
    // the elements that finished a step in this partial step (dt | pt), local
    // and ghost: their stage rows are current (with the owner-node exchange,
    // the owner slots a block reads)
    ot::Mesh* pMesh = m_uiAppCtx->get_mesh();
    if (!pMesh->isActive()) return;
    const unsigned int nPe = pMesh->getNumNodesPerElement();
    const unsigned int dof = m_uiEVar.get_dof();
    const size_t nEle      = pMesh->getAllElements().size();
    const size_t dgSz      = nEle * nPe;
    const T* k1            = m_uiStVec[0].get_vec_ptr();
    const T* k3            = m_uiStVec[2].get_vec_ptr();
    const T* k4            = m_uiStVec[3].get_vec_ptr();
    T* g                   = m_uiStG.get_vec_ptr();
    for (size_t e = 0; e < nEle; e++) {
        if (!m_uiEleDT[e] || (pt % m_uiEleDT[e])) continue;
        for (unsigned int v = 0; v < dof; v++) {
            const size_t o = v * dgSz + e * nPe;
            for (unsigned int q = 0; q < nPe; q++)
                g[o + q] = (k4[o + q] - 2 * k3[o + q]) + k1[o + q];
        }
        if (e < m_uiStGValid.size()) m_uiStGValid[e] = 1;
    }
    if (m_uiStGValidCls.size() != m_uiOwnDTs.size()) m_uiStGValidCls.assign(m_uiOwnDTs.size(), 0);
    for (size_t c = 0; c < m_uiOwnDTs.size(); c++)
        if (pt % m_uiOwnDTs[c] == 0) m_uiStGValidCls[c] = 1;
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::partial_evolve(unsigned int pt) {
    ot::Mesh* pMesh = m_uiAppCtx->get_mesh();

    if (!(pMesh->isActive())) return;
    const bool prof = ts::EnutsProf::on();
    double tp0 = 0, tp1 = 0, tpT = 0;
    if (prof) tpT = ts::EnutsProf::now();
    if (prof && pt == 0) {
        m_uiCoarseStep++;
        ts::EnutsProf::get().t[ts::EnutsProf::CN_CSTEPS] += 1;
    }
#define ENUTS_TIC() do { if (prof) tp0 = ts::EnutsProf::now(); } while (0)
#define ENUTS_TOC(k) do { if (prof) { tp1 = ts::EnutsProf::now(); ts::EnutsProf::get().t[ts::EnutsProf::k] += tp1 - tp0; } } while (0)

    const std::vector<ot::Block>& blkList = pMesh->getLocalBlockList();
    const ot::TreeNode* pNodes            = pMesh->getAllElements().data();
    const double current_t                = m_uiTimeInfo._m_uiT;
    const unsigned int DOF                = m_uiEVar.get_dof();

    double current_t_adv                  = current_t;

    const unsigned int finest_t           = step_ratio(m_uiLevMax);  // finest  time level.
    const unsigned int coarset_t =
        step_ratio(m_uiLevMin);

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
            step_ratio(bLev);

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
    //     step_ratio(bLev);

    //     m_uiBVec[blk]._vec[0].mark_unsynced();
    //     sync_blk_timestep(blk,0);
    //     m_uiBVec[blk]._vec[0].mark_synced();

    // }

    // prof: a finest-only sub-step has no active block with a dt above the finest
    bool finestOnly = true;
    if (prof) {
        ts::EnutsProf& P = ts::EnutsProf::get();
        P.partial += 1;
        P.blk_sum += (double)m_uiActiveBlkIDs.size();
        P.blk_tot += (double)blkList.size();
        unsigned int nl = 0;
        for (unsigned int l = m_uiLevMin; l <= m_uiLevMax; l++)
            if (pt % step_ratio(l) == 0) nl++;
        P.lev_sum += nl;
        for (unsigned int bb = 0; bb < m_uiActiveBlkIDs.size(); bb++) {
            const unsigned int bLev =
                pNodes[blkList[m_uiActiveBlkIDs[bb]].firstElementID()].getLevel();
            if (step_ratio(bLev) != finest_t)
                finestOnly = false;
        }
    }

    // level-scoped element-DG exchange: the elements whose level fires at pt
    // (DT | pt) are the plan's prefix; row d = largest distinct DT dividing pt
    // owner-sync plan: a block reads owner values only, so the exchange is
    // per node (the owner's rank sends), not per ghost element
    const bool ownPlan = prepare_owner_sync(pMesh);
    m_uiOwnPt = pt;
    if (ownPlan) {
        // the plan takes an owner's time from its dt: every block must sit at
        // the next multiple of its dt (true inside evolve())
        for (unsigned int blk = 0; blk < blkList.size(); blk++) {
            const unsigned int bdt =
                step_ratio(pNodes[blkList[blk].firstElementID()].getLevel());
            if (m_uiBVec[blk]._time != ((pt + bdt - 1) / bdt) * bdt) {
                std::cout << "[enuts-owner-sync r" << pMesh->getMPIRankGlobal()
                          << "] blk " << blk << " time " << m_uiBVec[blk]._time
                          << " dt " << bdt << " at partial step " << pt
                          << ": block times are not level-uniform; aborting"
                          << std::endl;
                MPI_Abort(pMesh->getMPICommunicator(), 14);
            }
        }
    }
    const bool scopedOn = !ownPlan && prepare_scoped_exchange(pMesh);
    const bool scopedChk = scopedOn && ts::EnutsProf::scopedCheck();
    const unsigned int* scSend = nullptr;
    const unsigned int* scRecv = nullptr;
    if (scopedOn) {
        int d = -1;
        for (size_t i = 0; i < m_uiScopedDTs.size(); i++)
            if (pt % m_uiScopedDTs[i] == 0) d = (int)i;
        if (d < 0) {
            scSend = scRecv = m_uiScopedZero.data();
        } else {
            scSend = m_uiScopedPlan.sendCnt.data() + (size_t)d * m_uiScopedPlan.npes;
            scRecv = m_uiScopedPlan.recvCnt.data() + (size_t)d * m_uiScopedPlan.npes;
        }
    }

    // DENDRO_ENUTS_OMP: the three block loops below run threaded over the
    // active list, largest block first (schedule(dynamic,1)); every block
    // writes only its own block vectors and its own elements' DG rows. off =
    // the original order on one thread.
    bool thr                   = ts::EnutsProf::ompOn();
    const unsigned int* actIds = m_uiActiveBlkIDs.data();
    const unsigned int nAct    = m_uiActiveBlkIDs.size();
#ifdef DENDRO_ENUTS_HAS_OMP
    if (thr) {
        // the per-block debug probes share static state
        static const bool dbg = std::getenv("DENDRO_ENUTS_BLKHASH") ||
                                std::getenv("DENDRO_ENUTS_NANCHK");
        if (dbg || omp_in_parallel()) thr = false;
    }
    if (thr) {
        m_uiActiveOrder.assign(m_uiActiveBlkIDs.begin(), m_uiActiveBlkIDs.end());
        std::stable_sort(m_uiActiveOrder.begin(), m_uiActiveOrder.end(),
                         [this](unsigned int a, unsigned int b) {
                             return m_uiBVec[a]._vec[0].getSz() >
                                    m_uiBVec[b]._vec[0].getSz();
                         });
        actIds = m_uiActiveOrder.data();
        // everything sync_blk_timestep builds on first use, and the scratch
        // of the worker threads, before any parallel region
        sync_prebuild(ownPlan, !ownPlan, false);
        sync_thread_setup(!ownPlan);
        if (!ownPlan) ensure_evar_dg();  // before the parallel loops
    }
#endif

    if (prof) {
        ts::EnutsProf::get().t[ts::EnutsProf::PE_PROLOG] += ts::EnutsProf::now() - tpT;
        m_uiRlBlkT.assign(nAct, 0.0);
        if (classWork() && m_uiCwBlkCls.size() != pMesh->getLocalBlockList().size())
            class_work_setup();
    }
    for (unsigned int rk = 1; rk <= m_uiNumStages; rk++) {
        const unsigned int BLK_S = rk;
        ENUTS_TIC();
#ifdef DENDRO_ENUTS_HAS_OMP
#pragma omp parallel for schedule(dynamic, 1) if (thr)
#endif
        for (unsigned int bb = 0; bb < nAct; bb++) {
            const unsigned int blk   = actIds[bb];
            const unsigned int BLK_T = m_uiBVec[blk]._time;
            const unsigned int NN    = m_uiBVec[blk]._vec[0].getSz();
            const unsigned int bLev =
                pNodes[blkList[blk].firstElementID()].getLevel();
            const unsigned int BLK_DT =
                step_ratio(bLev);

            const T blk_dt = m_uiTimeInfo._m_uiTh * BLK_DT;

            // prof: the per-block split (thread-s, merged after the sync loop)
            double tb0 = 0, tb1 = 0;
            if (prof) tb0 = tb1 = ts::EnutsProf::now();
#define ENUTS_BLK_TOC(k) do { if (prof) { const double tb2 = ts::EnutsProf::now(); ts::EnutsProf::get().t[ts::EnutsProf::k] += tb2 - tb1; tb1 = tb2; } } while (0)

            assert(m_uiBVec[blk]._vec[stage_slot(rk - 1)].isSynced());
            m_uiAppCtx->pre_timestep_blk((T*)m_uiBVec[blk]._vec[0].data(), DOF,
                                         blk, current_t + dt_finest * BLK_T);
            ENUTS_BLK_TOC(RL_OTH);

            BlockAsyncVector<T>& sIn = stage_in(blk);
            T* out_ptr = (T*)sIn.data();

            if (m_uiStageRing && BLK_S > 1) {
                // ring: U, the -0.0 -> +0.0 of the dropped 0 * k_j terms
                // (j < s-1) where the mask is set, the live term
                // a_{s,s-1} dt k_{s-1}, the accumulator term b_{s-1} dt k_{s-1}
                // and the mask update, in one pass per entry
                sIn.copyMeta(m_uiBVec[blk]._vec[0]);
                std::vector<uint64_t>& sg = m_uiStageSgn[blk];
                const size_t n            = (size_t)DOF * NN;
                const T* u                = m_uiBVec[blk]._vec[0].data();
                const T* k                = m_uiBVec[blk]._vec[RING_K].data();
                const T c0 = blk_dt * m_uiAij[(BLK_S - 1) * m_uiNumStages + (BLK_S - 2)];
                const bool launder = BLK_S >= 3;
                const bool upd     = BLK_S + 1 <= m_uiNumStages;
                const size_t nw    = (n + 63) / 64;
                // acc = U + b_1 dt k_1 (stage 2) or acc += b_{s-1} dt k_{s-1}
                T* acc     = (T*)m_uiBVec[blk]._vec[RING_ACC].data();
                const T cb = blk_dt * m_uiBi[BLK_S - 2];
                for (size_t w = 0; w < nw; w++) {
                    const size_t i0 = w * 64, i1 = std::min(n, i0 + 64);
                    const uint64_t old = sg[w];
                    uint64_t add = 0;
                    if (!launder || !old) {
                        for (size_t i = i0; i < i1; i++) {
                            T t = u[i];
                            t += (c0 * k[i]);
                            out_ptr[i] = t;
                        }
                    } else {
                        for (size_t i = i0; i < i1; i++) {
                            T t = u[i];
                            if (((old >> (i - i0)) & 1) && t == (T)0 && std::signbit(t)) t = (T)0;
                            t += (c0 * k[i]);
                            out_ptr[i] = t;
                        }
                    }
                    if (BLK_S == 2)
                        for (size_t i = i0; i < i1; i++) {
                            T q = u[i];
                            q += (cb * k[i]);
                            acc[i] = q;
                        }
                    else
                        for (size_t i = i0; i < i1; i++) acc[i] += (cb * k[i]);
                    if (upd)
                        for (size_t i = i0; i < i1; i++)
                            if (!std::signbit(k[i])) add |= (uint64_t)1 << (i - i0);
                    sg[w] = old | add;
                }
            } else {
                sIn.vecCopy(m_uiBVec[blk]._vec[0]);
                if (m_uiStageRing) {
                    // stage 1: no stage term yet, a fresh sign mask
                    std::fill(m_uiStageSgn[blk].begin(), m_uiStageSgn[blk].end(), 0);
                } else {
                    // all stage terms in one pass (same order per entry)
                    const T* stp[16];
                    T stc[16];
                    for (unsigned int s = 1; s < BLK_S; s++) {
                        stp[s - 1] = m_uiBVec[blk]._vec[s].data();
                        stc[s - 1] = blk_dt *
                                     m_uiAij[(BLK_S - 1) * m_uiNumStages + (s - 1)];
                    }
                    enuts_axpy_stages(out_ptr, stp, stc, BLK_S - 1, (size_t)DOF * NN);
                }
            }

            ENUTS_BLK_TOC(RL_IN);

            // the RK stage time, as ETS passes it: a time-dependent RHS term
            // sees its forcing at the stage, not at the step start
#if defined(DENDRO_ENUTS_TEST_SABOTAGE) && DENDRO_ENUTS_TEST_SABOTAGE == 1
            // test builds only: the stage time frozen at the step start
            const T t_stage = current_t + dt_finest * BLK_T;
#else
            const T t_stage = current_t + dt_finest * BLK_T + m_uiCi[BLK_S - 1] * blk_dt;
#endif
            m_uiAppCtx->pre_stage_blk((T*)sIn.data(), DOF, blk, t_stage);
            ENUTS_BLK_TOC(RL_PRE);
            debug_nan_check("pre-rhs-in", blk, m_uiNumStages + 1, pt, rk);
            debug_blk_hash("pre-rhs-in", blk, m_uiNumStages + 1, pt, rk);
            const double tr0 = tb1;
            m_uiBVec[blk]._vec[stage_slot(BLK_S)].computeVec(m_uiAppCtx, sIn,
                                                             t_stage);
            ENUTS_BLK_TOC(RL_RHS);
            if (prof) {
                // rhs_blk seconds and points by block span (elements per side)
                const unsigned int bw = blkList[blk].getAllocationSzX() -
                                        2 * blkList[blk].get1DPadWidth() - 1;
                const unsigned int ne = bw / pMesh->getElementOrder();
                const unsigned int bi = ne >= 8 ? 3 : ne >= 4 ? 2 : ne >= 2 ? 1 : 0;
                ts::EnutsProf::get().t[ts::EnutsProf::RB_T0 + bi] += tb1 - tr0;
                ts::EnutsProf::get().t[ts::EnutsProf::RB_P0 + bi] += (double)NN;
            }
            debug_nan_check("post-rhs", blk, BLK_S, pt, rk);
            debug_blk_hash("post-rhs", blk, BLK_S, pt, rk);
            m_uiBVec[blk]._vec[stage_slot(BLK_S)].mark_unsynced();
            // std::cout<<"rk:"<<rk<<std::endl;
            // m_uiBVec[blk]._vec[BLK_S].dump_vec(std::cout);

#ifdef __PROFILE_ENUTS__
            m_uiPt[ENUTSPROFILE::ENUTS_BLK_ZIP].start();
#endif

            m_uiBVec[blk]._vec[stage_slot(BLK_S)].zipDG(
                pMesh, m_uiStVec[BLK_S - 1].get_vec_ptr(), DOF);
            ENUTS_BLK_TOC(RL_ZIP);
            m_uiBVec[blk]._rks = rk;
            if (prof) m_uiRlBlkT[bb] = tb1 - tb0;
#undef ENUTS_BLK_TOC

#ifdef __PROFILE_ENUTS__
            m_uiPt[ENUTSPROFILE::ENUTS_BLK_ZIP].stop();
#endif
        }

        ENUTS_TOC(RHS);
        if (prof) {
            // the loop wall by sub-step class, its thread capacity, the per-block
            // work and the longest block of this call (granularity: max / wall)
            ts::EnutsProf& P = ts::EnutsProf::get();
            const double wall = tp1 - tp0;
            double sum = 0, mxb = 0;
            for (unsigned int bb = 0; bb < nAct; bb++) {
                sum += m_uiRlBlkT[bb];
                mxb = std::max(mxb, m_uiRlBlkT[bb]);
            }
            int nthr = 1;
#ifdef DENDRO_ENUTS_HAS_OMP
            if (thr) nthr = omp_get_max_threads();
#endif
            const int c = finestOnly ? 0 : ts::EnutsProf::RC_WALL1 - ts::EnutsProf::RC_WALL0;
            P.t[ts::EnutsProf::RL_CAP] += nthr * wall;
            P.t[ts::EnutsProf::RC_WALL0 + c] += wall;
            P.t[ts::EnutsProf::RC_CAP0 + c] += nthr * wall;
            P.t[ts::EnutsProf::RC_SUM0 + c] += sum;
            P.t[ts::EnutsProf::RC_MAX0 + c] += mxb;
            P.t[ts::EnutsProf::RC_CALLS0 + c] += 1;
            if (classWork() && !m_uiCwDTs.empty()) {
                // sub-step type: the coarsest class active at pt
                unsigned int tt = 0;
                for (unsigned int k = 0; k < m_uiCwDTs.size(); k++)
                    if (pt % m_uiCwDTs[k] == 0) tt = k;
                double* ty = m_uiCwType.data() + (size_t)tt * 4;
                ty[0] += 1;
                ty[1] += wall;
                ty[2] += nthr * wall;
                ty[3] += sum;
                for (unsigned int bb = 0; bb < nAct; bb++) {
                    double* cl = m_uiCwCls.data() + (size_t)m_uiCwBlkCls[actIds[bb]] * 6;
                    cl[4] += 1;
                    cl[5] += m_uiRlBlkT[bb];
                }
            }
        }
        ENUTS_TIC();
        ot::g_eledg_pack_omp = thr;
        if (ownPlan) {
            T* stv = m_uiStVec[BLK_S - 1].get_vec_ptr();
            own_node_exchange_begin(stv, DOF, pt);
            ENUTS_TOC(EDG_BEGIN);
            ENUTS_TIC();
            own_node_exchange_end(stv, DOF, pt);
            ENUTS_TOC(EDG_END);
        } else if (scopedOn) {
            T* stv = m_uiStVec[BLK_S - 1].get_vec_ptr();
            if (scopedChk) {
                // shadow: what the full exchange would leave in every ghost slot
                const size_t n = (size_t)DOF * pMesh->getAllElements().size() *
                                 pMesh->getNumNodesPerElement();
                m_uiScopedShadow.assign(stv, stv + n);
                pMesh->readFromGhostBeginEleDGVec(m_uiScopedShadow.data(), DOF);
                pMesh->readFromGhostEndEleDGVec(m_uiScopedShadow.data(), DOF);
            }
            pMesh->readFromGhostBeginEleDGVecScoped(stv, DOF, m_uiScopedPlan,
                                                    scSend, scRecv);
            ENUTS_TOC(EDG_BEGIN);
            ENUTS_TIC();
            pMesh->readFromGhostEndEleDGVecScoped(stv, DOF, m_uiScopedPlan,
                                                  scRecv);
            ENUTS_TOC(EDG_END);
            if (scopedChk)
                scoped_check_compare(pMesh, stv, m_uiScopedShadow.data(), DOF,
                                     pt, rk);
        } else {
        pMesh->readFromGhostBeginEleDGVec(m_uiStVec[BLK_S - 1].get_vec_ptr(),
                                          DOF);
        ENUTS_TOC(EDG_BEGIN);
        ENUTS_TIC();
        pMesh->readFromGhostEndEleDGVec(m_uiStVec[BLK_S - 1].get_vec_ptr(),
                                        DOF);
        ENUTS_TOC(EDG_END);
        }
        ot::g_eledg_pack_omp = false;
        ENUTS_TIC();
        m_uiSyncPass++;
        m_uiSyncPassOpen = true;
        if (ownPlan) {
            const double tpre = prof ? ts::EnutsProf::now() : 0.0;
            own_p2c_prepass(BLK_S, actIds, nAct, thr);
            if (prof) ts::EnutsProf::get().t[ts::EnutsProf::SY_PRE] += ts::EnutsProf::now() - tpre;
        } else {
            m_uiP2cSharedOn = false;
        }

        const double tloop0 = prof ? ts::EnutsProf::now() : 0.0;
#ifdef DENDRO_ENUTS_HAS_OMP
#pragma omp parallel for schedule(dynamic, 1) if (thr)
#endif
        for (unsigned int bb = 0; bb < nAct; bb++) {
            const unsigned int blk   = actIds[bb];
            const unsigned int BLK_T = m_uiBVec[blk]._time;
            const unsigned int NN    = m_uiBVec[blk]._vec[0].getSz();
            const unsigned int bLev =
                pNodes[blkList[blk].firstElementID()].getLevel();
            const unsigned int BLK_DT =
                step_ratio(bLev);

            sync_blk_timestep(blk, BLK_S);
            debug_nan_check("post-sync", blk, BLK_S, pt, rk);
            debug_blk_hash("post-sync", blk, BLK_S, pt, rk);
            const double tps0 = prof ? ts::EnutsProf::now() : 0.0;
            m_uiAppCtx->post_stage_blk(
                (T*)m_uiBVec[blk]._vec[stage_slot(BLK_S)].data(), DOF, blk,
                current_t + dt_finest * BLK_T);
            if (prof) ts::EnutsProf::get().t[ts::EnutsProf::SY_POST] += ts::EnutsProf::now() - tps0;
            m_uiBVec[blk]._vec[stage_slot(BLK_S)].mark_synced();
            if (m_uiStageRing && BLK_S == m_uiNumStages) {
                // ring: the last stage's accumulator term (the earlier ones
                // went in with the next stage's input pass)
                BlockAsyncVector<T>& acc = m_uiBVec[blk]._vec[RING_ACC];
                const T* k               = m_uiBVec[blk]._vec[RING_K].data();
                const size_t n           = (size_t)DOF * NN;
                const T blk_dt           = m_uiTimeInfo._m_uiTh * BLK_DT;
                const T* stp[1] = {k};
                const T stc[1]  = {blk_dt * m_uiBi[BLK_S - 1]};
                const double tr0 = prof ? ts::EnutsProf::now() : 0.0;
                if (BLK_S == 1) acc.vecCopy(m_uiBVec[blk]._vec[0]);
                enuts_axpy_stages((T*)acc.data(), stp, stc, 1, n);
                if (prof) ts::EnutsProf::get().t[ts::EnutsProf::SY_RING] += ts::EnutsProf::now() - tr0;
            }
            // std::cout<<"[NUTS]: pt :  "<<pt<<" blk : "<<blk<<" rk : "<<rk<<"
            // step size: "<<BLK_DT<<" is synced:
            // "<<m_uiBVec[blk]._vec[BLK_S].isSynced()<<std::endl;
        }
        m_uiSyncPassOpen = false;
        m_uiP2cSharedOn  = false;
        if (prof) {
            const double tcen0 = ts::EnutsProf::now();
            ts::EnutsProf::get().t[ts::EnutsProf::SY_LOOP] += tcen0 - tloop0;
            // p2c store census of this pass: children computed by any thread
            // (unique) and computed again by another thread (duplicates)
            size_t tot = 0, uniq = 0;
            const size_t nst = m_uiP2cPassOf.size();
            for (size_t i = 0; i < nst; i++) {
                bool any = false;
                if (m_uiP2cPassOf[i] == m_uiSyncPass) { tot++; any = true; }
                for (size_t t = 1; t < m_uiSyncThr.size(); t++)
                    if (m_uiSyncThr[t].p2cPassOf.size() > i &&
                        m_uiSyncThr[t].p2cPassOf[i] == m_uiSyncPass) { tot++; any = true; }
                uniq += any;
            }
            ts::EnutsProf::get().t[ts::EnutsProf::CN_P2C_UNIQ] += (double)uniq;
            ts::EnutsProf::get().t[ts::EnutsProf::CN_P2C_DUP] += (double)(tot - uniq);
            ts::EnutsProf::get().t[ts::EnutsProf::SY_CENSUS] += ts::EnutsProf::now() - tcen0;
        }
        if (thr && prof) ts::EnutsProf::mergeThreads();
        ENUTS_TOC(SYNC_BLK);

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
        int BLK_DT = step_ratio(bLev);

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
        int BLK_DT = step_ratio(bLev);

            sync_blk_timestep(blk,BLK_S);
            m_uiAppCtx->post_stage_blk((T*)m_uiBVec[blk]._vec[BLK_S].data(),DOF,blk,current_t
        + dt_finest*BLK_T); m_uiBVec[blk]._vec[BLK_S].mark_synced();
            //std::cout<<"[NUTS]: pt :  "<<pt<<" blk : "<<blk<<" rk : "<<rk<<"
        step size: "<<BLK_DT<<" is synced:
        "<<m_uiBVec[blk]._vec[BLK_S].isSynced()<<std::endl;

        }
        */
    }

    ENUTS_TIC();
    // compute the time step vector and increment time.
#ifdef DENDRO_ENUTS_HAS_OMP
#pragma omp parallel for schedule(dynamic, 1) if (thr)
#endif
    for (unsigned int bb = 0; bb < nAct; bb++) {
        const unsigned int blk   = actIds[bb];
        const unsigned int BLK_T = m_uiBVec[blk]._time;
        const unsigned int NN    = m_uiBVec[blk]._vec[0].getSz();
        const unsigned int bLev =
            pNodes[blkList[blk].firstElementID()].getLevel();
        const unsigned int BLK_DT =
            step_ratio(bLev);
        const T blk_dt = m_uiTimeInfo._m_uiTh * BLK_DT;

        T* out_ptr     = (T*)m_uiBVec[blk]._vec[0].data();
        if (m_uiStageRing) {
            // ring: the accumulator already holds U + sum_s b_s dt k_s
            m_uiBVec[blk]._vec[0].swapData(m_uiBVec[blk]._vec[RING_ACC]);
        } else {
            const T* stp[16];
            T stc[16];
            for (unsigned int s = 1; s <= m_uiNumStages; s++) {
                assert(m_uiBVec[blk]._vec[s].isSynced());
                stp[s - 1] = m_uiBVec[blk]._vec[s].data();
                stc[s - 1] = blk_dt * m_uiBi[(s - 1)];
            }
            enuts_axpy_stages(out_ptr, stp, stc, m_uiNumStages, (size_t)DOF * NN);
        }

        m_uiBVec[blk]._time += BLK_DT;
        m_uiBVec[blk]._rks = 0;
        m_uiAppCtx->post_timestep_blk((T*)m_uiBVec[blk]._vec[0].data(), DOF,
                                      blk, current_t + dt_finest * BLK_T);
        m_uiBVec[blk]._vec[0].mark_synced();
        debug_blk_hash("post-axpy", blk, 0, pt, 0);

        // zipDG to the stVec[0]
        // mark the stage vars as junck.
        mark_stages_unsynced(blk);
    }

    if (m_uiStGOn) update_stage_g(pt);
    if (m_uiStGOn && ownPlan && ts::EnutsProf::gXch()) {
        // ghost rows of g from their owners (the classes that completed at
        // pt), on the unscoped lists when the stage exchange is scoped
        const bool sw = m_uiXchScoped && !m_uiOwnRecvCumU.empty();
        if (sw) {
            m_uiOwnRecvSlot.swap(m_uiOwnRecvSlotU); m_uiOwnRecvCum.swap(m_uiOwnRecvCumU);
            m_uiOwnSendSlot.swap(m_uiOwnSendSlotU); m_uiOwnSendCum.swap(m_uiOwnSendCumU);
        }
        own_node_exchange_begin(m_uiStG.get_vec_ptr(), DOF, pt);
        own_node_exchange_end(m_uiStG.get_vec_ptr(), DOF, pt);
        if (sw) {
            m_uiOwnRecvSlot.swap(m_uiOwnRecvSlotU); m_uiOwnRecvCum.swap(m_uiOwnRecvCumU);
            m_uiOwnSendSlot.swap(m_uiOwnSendSlotU); m_uiOwnSendCum.swap(m_uiOwnSendCumU);
        }
    }
    update_ele_timestep(&m_uiActiveBlkIDs);
    ENUTS_TOC(AXPY_ELETS);
    ENUTS_TIC();

    // the owner path never reads a ghost element time
    const bool skipEt = ownPlan;
    if (!skipEt) {
        pMesh->readFromGhostBeginElementVec(m_uiEleTime.data(), 1);
        pMesh->readFromGhostEndElementVec(m_uiEleTime.data(), 1);
    }
    // m_uiEleDT is not exchanged here: it depends on the level only and
    // evolve() exchanged it at the start of the step
    ENUTS_TOC(EVEC_XCH);
    ENUTS_TIC();

    if (!skipEt) pMesh->waitActive();
    ENUTS_TOC(WAIT);
    if (prof) ts::EnutsProf::get().t[ts::EnutsProf::TOTAL] += ts::EnutsProf::now() - tpT;
#undef ENUTS_TIC
#undef ENUTS_TOC
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

    const unsigned int finest_t = step_ratio(m_uiLevMax);  // finest  time level.
    const unsigned int coarset_t = step_ratio(m_uiLevMin);  // coarset time level.

    const double dt_finest  = m_uiTimeInfo._m_uiTh;
    const double dt_coarset = coarset_t * m_uiTimeInfo._m_uiTh;

    // prof: what evolve() does around the partial steps (serial, per coarse step)
    const bool eprof = ts::EnutsProf::on();
    double te0 = 0, teT = 0;
    if (eprof) te0 = teT = ts::EnutsProf::now();
#define ENUTS_EV_TOC(k) do { if (eprof) { const double te1 = ts::EnutsProf::now(); ts::EnutsProf::get().t[ts::EnutsProf::k] += te1 - te0; te0 = te1; } } while (0)

    m_uiAppCtx->pre_timestep(m_uiEVar);
    ENUTS_EV_TOC(EV_PRETS);

    if (pMesh->isActive()) {
        const unsigned int rank = pMesh->getMPIRank();
        const unsigned int npes = pMesh->getMPICommSize();


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
        ENUTS_EV_TOC(EV_UNZIP);
        const std::vector<ot::Block>& blkList = pMesh->getLocalBlockList();

        // initialize the block async vectors
        for (unsigned int blk = 0; blk < blkList.size(); blk++) {
            m_uiBVec[blk]._time = 0;
            m_uiBVec[blk]._rks  = 0;
            m_uiBVec[blk]._vec[0].copyFromUnzip(
                pMesh, m_uiEvarUzip.get_vec_ptr(), true, DOF);
            m_uiBVec[blk]._vec[0].mark_synced();
            debug_blk_hash("init-unzip", blk, 0, 0, 0);

            mark_stages_unsynced(blk);
        }

#ifdef __PROFILE_ENUTS__
        m_uiPt[ENUTSPROFILE::ENUTS_BLK_UNZIP].stop();
#endif
        ENUTS_EV_TOC(EV_LOAD);

        for (unsigned int ele = pMesh->getElementPreGhostBegin();
             ele < pMesh->getElementPostGhostEnd(); ele++)
            m_uiEleTime[ele] = 0;

        this->update_ele_timestep();

        pMesh->readFromGhostBeginElementVec(m_uiEleTime.data(), 1);
        pMesh->readFromGhostBeginElementVec(m_uiEleDT.data(), 1);

        pMesh->readFromGhostEndElementVec(m_uiEleTime.data(), 1);
        pMesh->readFromGhostEndElementVec(m_uiEleDT.data(), 1);
        ENUTS_EV_TOC(EV_ETIME);

        for (unsigned int pt = 0; pt < coarset_t; pt++) {
            // unsigned int numBlkEvolved = 0;
            // for(unsigned int blk =0; blk < blkList.size(); blk++)
            // {
            //     const unsigned int BLK_T = m_uiBVec[blk]._time;
            //     const unsigned int NN   =  m_uiBVec[blk]._vec[0].getSz();
            //     const unsigned int bLev =
            //     pNodes[blkList[blk].firstElementID()].getLevel(); const
            //     unsigned int BLK_DT =
            //     step_ratio(bLev);;
            //     if( pt% BLK_DT !=0 )
            //         continue;
            //     numBlkEvolved++;
            // }

            // std::cout<<"[ENUTS] : pt: "<<pt<<" number of blocks evolved :
            // "<<numBlkEvolved<<" out of "<<blkList.size()<<std::endl;
            this->partial_evolve(pt);
            m_uiPartialStepId                     = pt;
            const unsigned int pt_freq = std::min(10u, coarset_t);
            if (m_uiPartialStepId % pt_freq == 0) {
                if (!pMesh->getMPIRankGlobal())
                    std::cout << "[LTS] : local ts: " << m_uiPartialStepId << " of total "
                              << coarset_t << std::endl;
            }
        }

#ifdef __PROFILE_ENUTS__
        m_uiPt[ENUTSPROFILE::ENUTS_BLK_ZIP].start();
#endif
        if (eprof) te0 = ts::EnutsProf::now();  // partial steps have their own rows

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
            ensure_evar_dg();
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
                step_ratio(bLev);

            m_uiBVec[blk]._vec[0].zip(pMesh, m_uiEVar.get_vec_ptr(), DOF);
        }

#ifdef __PROFILE_ENUTS__
        m_uiPt[ENUTSPROFILE::ENUTS_BLK_ZIP].stop();
#endif
        ENUTS_EV_TOC(EV_ZIP);
    }

    m_uiAppCtx->post_timestep(m_uiEVar);
    m_uiAppCtx->increment_ts_info((T)coarset_t, 1);
    m_uiTimeInfo = m_uiAppCtx->get_ts_info();
    pMesh->waitAll();
    ENUTS_EV_TOC(EV_POST);
    if (eprof) ts::EnutsProf::get().t[ts::EnutsProf::EV_TOTAL] += ts::EnutsProf::now() - teT;
#undef ENUTS_EV_TOC

#ifdef __PROFILE_ENUTS__
    m_uiPt[ENUTSPROFILE::ENUTS_EVOLVE].stop();
#endif
}

}  // namespace ts

#include "enuts_remesh.tcc"
#include "enuts_sync.tcc"
#include "enuts_owner.tcc"
#include "enuts_exchange.tcc"
