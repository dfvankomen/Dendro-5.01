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

// DENDRO_ENUTS_OMP=1 threads the block loops of partial_evolve; hybrid builds only
#if defined(DENDRO_HYBRID_OMP) && defined(_OPENMP)
#include <omp.h>
#define DENDRO_ENUTS_HAS_OMP 1
#endif

#define __LTS_BLK_SYNC_TIME_COMP_TOL__ 1e-6

namespace ts {
/**@brief DENDRO_ENUTS_PROF=1 wall-clock accumulators for partial_evolve (per
 * rank, process-wide). env read once; off = zero cost, zero collectives. */
struct EnutsProf {
    enum { RHS = 0, EDG_BEGIN, EDG_END, SYNC_BLK, AXPY_ELETS, EVEC_XCH, WAIT, TOTAL,
           SY_SAME, SY_CORR, SY_CORR_FC, SY_UNZIP, SY_OWN, SY_POST, SY_P2C,
           SY_ROW, SY_CPY, SY_PRE, SY_BLK, SY_RING, SY_LOOP, SY_CENSUS,
           EX_WAIT, EX_FIRST, EX_UNPACK,
           CN_CALLS, CN_EID, CN_ENT, CN_PAIRS, CN_P2C, CN_CORR, CN_HIT, CN_P2C_HIT,
           CN_P2C_UNIQ, CN_P2C_DUP, CN_P2C_OVF, CN_XCH_RECV, CN_XCH_SEND, CN_XCH_CALLS,
           CN_CSTEPS, CN_P2C_DISTINCT, CN_HOIST_DBL, SY_ROWOPS, CN_ROWS, CN_ROWS_OPS,
           EX_LATE, EX_LATEMAX, CN_XCH_ARGSEND, CN_XCH_NOOFF, CN_XCH_NOPLAN, CN_XCH_UNREAD,
           CN_XCH_READ, CN_XCH_LATEBAD,
           RL_IN, RL_PRE, RL_RHS, RL_ZIP, RL_OTH, RL_CAP,
           RC_WALL0, RC_CAP0, RC_SUM0, RC_MAX0, RC_CALLS0,
           RC_WALL1, RC_CAP1, RC_SUM1, RC_MAX1, RC_CALLS1,
           PE_PROLOG, EV_TOTAL, EV_PRETS, EV_UNZIP, EV_LOAD, EV_ETIME, EV_ZIP, EV_POST,
           ST_PASS, CN_STS_ROUNDS,
           RB_T0, RB_T1, RB_T2, RB_T3, RB_P0, RB_P1, RB_P2, RB_P3, CN_G0, CN_GRESET, CN_GALL, NT };
    double t[NT]     = {};
    double partial   = 0;      // partial steps taken (this rank was active)
    double lev_sum   = 0;      // sum over partial steps of globally active levels
    double blk_sum   = 0;      // sum over partial steps of locally active blocks
    double blk_tot   = 0;      // sum over partial steps of local blocks
    static bool on() {
        static const int v = []() {
            const char* e = std::getenv("DENDRO_ENUTS_PROF");
            return (e && e[0] == '1') ? 1 : 0;
        }();
        return v == 1;
    }
    /** the process-wide accumulators */
    static EnutsProf& global() {
        static EnutsProf p;
        return p;
    }
    /** inside an active parallel region every thread accumulates into its own
     *  copy (folded into global() by mergeThreads()); global() otherwise */
    static EnutsProf& get() {
#ifdef DENDRO_ENUTS_HAS_OMP
        if (omp_in_parallel()) {
            static thread_local EnutsProf tp;
            return tp;
        }
#endif
        return global();
    }
    /** add the per-thread accumulators to global() and zero them; call outside
     *  a parallel region. the folded sync:* rows are thread-seconds. */
    static void mergeThreads() {
#ifdef DENDRO_ENUTS_HAS_OMP
        EnutsProf& g = global();
#pragma omp parallel
        {
            EnutsProf& tp = get();
#pragma omp critical(dendro_enuts_prof)
            if (&tp != &g)
                for (int i = 0; i < NT; i++) {
                    g.t[i] += tp.t[i];
                    tp.t[i] = 0;
                }
        }
#endif
    }
    /** DENDRO_ENUTS_OMP=1 (default off; needs a DENDRO_HYBRID_OMP build):
     *  thread the per-block loops of partial_evolve (stage input + rhs_blk +
     *  zipDG, sync_blk_timestep + post_stage_blk, final update) over the
     *  active blocks, largest first, and the element-DG exchange pack/unpack */
    static bool ompOn() {
#ifdef DENDRO_ENUTS_HAS_OMP
        static const int v = []() {
            const char* e = std::getenv("DENDRO_ENUTS_OMP");
            return (e && e[0] == '1') ? 1 : 0;
        }();
        return v == 1;
#else
        return false;
#endif
    }
    /** thread id inside an active parallel region, 0 otherwise */
    static int tid() {
#ifdef DENDRO_ENUTS_HAS_OMP
        return omp_in_parallel() ? omp_get_thread_num() : 0;
#else
        return 0;
#endif
    }
    /** wall clock, or thread CPU time with DENDRO_ENUTS_PROF_CPU=1 (for timing on an oversubscribed box) */
    static double now() {
        static const int cpu = []() {
            const char* e = std::getenv("DENDRO_ENUTS_PROF_CPU");
            return (e && e[0] == '1') ? 1 : 0;
        }();
        if (cpu) {
            struct timespec ts_;
            clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts_);
            return ts_.tv_sec + 1e-9 * ts_.tv_nsec;
        }
        return MPI_Wtime();
    }
    /** level-scoped element-DG ghost exchange, default on (bit-identical
     *  off/on); DENDRO_ENUTS_SCOPED_EXCH=0 reverts to the full exchange;
     *  verification via DENDRO_ENUTS_SCOPED_EXCH_CHECK=1 */
    static bool scoped() {
        static const int v = []() {
            const char* e = std::getenv("DENDRO_ENUTS_SCOPED_EXCH");
            return (e && e[0] == '0') ? 0 : 1;
        }();
        return v == 1;
    }
    static bool scopedCheck() {
        static const int v = []() {
            const char* e = std::getenv("DENDRO_ENUTS_SCOPED_EXCH_CHECK");
            return (e && e[0] == '1') ? 1 : 0;
        }();
        return v == 1;
    }
    unsigned long long chk_cmp = 0, chk_bad = 0;
    /** DENDRO_ENUTS_SYNC_FAST=1 (default off = the original loops, until the
     *  gate covers labelled meshes, NLSM and BSSN):
     *  bit-identical sync_blk_timestep shortcuts (no same-time DG copy,
     *  per-stage correction reuse across blocks, padding-fill plan, per-pass
     *  store and shared sweeps of the parent-to-child interpolations) */
    /** DENDRO_ENUTS_IFACE_FIX (bit mask, default 0 = unchanged): stage data a
     *  block takes from a neighbour that steps with a different dt.
     *  1: same time -> convert the neighbour's stages to this block's dt
     *     (the plain copy is first order in dt).
     *  2: neighbour ahead -> use all of its completed stages for the time
     *     shift instead of the first rk_s.
     *  4: (with 1 and 2, RK4) conversion that stays fourth order for a
     *     nonlinear right-hand side (ENUTSOp::stageRowNL); the same-time
     *     conversion of stages 2 and 3 takes g = k4 - 2 k3 + k1 from the
     *     neighbour's previous step (m_uiStG).
     *  8: neighbour ahead -> time shift measured from the start of the
     *     neighbour's step (block time - (element time - element dt)). The
     *     original shift, element time - block time, is the same number only
     *     when the block sits at the midpoint of the element's step: true
     *     between 2:1 neighbours, not for an owner two time levels away
     *     (reached through a hanging node of a dependent). */
    /** the solution-changing settings of the process's stepper, set by the
     *  ExplicitNUTS constructor */
    static ts::LtsOptions& options() {
        static ts::LtsOptions o;
        return o;
    }
    static int ifaceFix() { return (int)options().iface; }
    /** DENDRO_ENUTS_STAGE_TIME=1 (default off = unchanged): pass the RK stage
     *  time t + c_s * dt_blk to the RHS and to pre_stage_blk, as ETS does
     *  (ets.h: current_t + m_uiCi[stage] * dt). The original passes the
     *  block's step-start time to every stage; a time-dependent RHS term
     *  (BSSN's slow-start lapse, exp(-t^2 / 2 sigma^2)) then sees a forcing
     *  lagged by ~dt_blk / 2, growing with the block's step: a lapse offset
     *  of 1e-3 on q1 at the full ladder. */
    /** with interface setting bit 4 and the owner sync, exchange the stored
     *  vector g of the classes that completed a step at this partial step
     *  (the owner-node exchange, on the unscoped lists when the stage exchange
     *  is scoped), so a ghost row's g is its owner's value. Without it g on
     *  ghost rows is formed locally from stage slots the exchange only partly
     *  delivered, and the result depends on the partition. */
    static bool gXch() { return options().g_exchange; }
    static bool stageTime() {
        static const int v = []() {
            const char* e = std::getenv("DENDRO_ENUTS_STAGE_TIME");
            return (e && e[0] == '1') ? 1 : 0;
        }();
        return v == 1;
    }
    /** DENDRO_ENUTS_STAGE_SCRATCH=1 (default off): the stage input (the copy
     *  of the block solution plus the stage terms, _vec[NS+1]) is one scratch
     *  vector per thread sized for the largest block instead of one per block.
     *  It is live only between its build and the RHS call of the same block
     *  and stage, so the result is the same. Saves one padded copy. */
    static bool stageScratch() {
        static const int v = []() {
            const char* e = std::getenv("DENDRO_ENUTS_STAGE_SCRATCH");
            return (e && e[0] == '1') ? 1 : 0;
        }();
        return v == 1 || stageRing();
    }
    /** DENDRO_ENUTS_STAGE_RING=1 (default off; implies STAGE_SCRATCH): one
     *  stage vector per block instead of NS, plus an accumulator. The stage
     *  input of stage s needs only k_{s-1} when the tableau has a_{s,j} = 0
     *  for j < s-1 (RK4), and the final update U + sum_s b_s k_s is formed by
     *  adding each b_s k_s right after stage s is synced, in the same order
     *  per entry as the one-pass update. The dropped zero-coefficient terms
     *  0 * k_j only turn a -0.0 entry of U into +0.0 when some k_j has its sign
     *  bit clear there; a per-block sign mask reproduces that, so the stage
     *  inputs are bit-identical for finite stages. Saves NS - 1 - 1 = 2 padded
     *  copies per block for RK4 (6 -> 3 with the scratch). */
    static bool stageRing() { return stageRingMode() != 0; }
    /** 1 = ring with the sign mask; 2 = ring without it (test only: shows the
     *  mask is what keeps the -0.0 entries bit-identical) */
    static int stageRingMode() {
        static const int v = []() {
            const char* e = std::getenv("DENDRO_ENUTS_STAGE_RING");
            return e ? std::atoi(e) : 0;
        }();
        return (v == 1 || v == 2) ? v : 0;
    }
    /** DENDRO_ENUTS_STAGE_IN_FUSED=1 (default off; ring only): the stage
     *  input U + a_{s,s-1} dt k_{s-1} is built in one pass over the block
     *  (copy, sign launder and axpy per entry, mask update in the same
     *  loop) instead of a copy pass and an axpy pass; same arithmetic.
     *  Note the whole padded block is needed: the padding of U is carried
     *  from sub-step to sub-step by the update itself (the per-sub-step
     *  stage-0 sync is not run), so no pass here may stop at the interior. */
    static bool stageInFused() {
        static const int v = []() {
            const char* e = std::getenv("DENDRO_ENUTS_STAGE_IN_FUSED");
            return (e && e[0] == '1') ? 1 : 0;
        }();
        return (v == 1 || ringFused()) && stageRing();
    }
    /** DENDRO_ENUTS_RING_FUSED=1 (default off; ring only, implies
     *  STAGE_IN_FUSED): the accumulator term b_{s-1} dt k_{s-1} is added in
     *  the stage-input pass of stage s, which reads k_{s-1} anyway (stage 2
     *  forms acc = U + b_1 dt k_1 there, in place of the copy); only the
     *  last stage's term stays in the sync loop. Same operation sequence per
     *  entry; 19 instead of 31 block sweeps per RK4 sub-step with the input
     *  passes. */
    static bool ringFused() {
        static const int v = []() {
            const char* e = std::getenv("DENDRO_ENUTS_RING_FUSED");
            return (e && e[0] == '1') ? 1 : 0;
        }();
        return v == 1 && stageRing();
    }
    /** DENDRO_ENUTS_LAZY_TMP=1 (default off): temporaries only some paths
     *  read are allocated on first use (m_uiEVecTmp[0], the DG copy of the
     *  solution m_uiEvarDG), m_uiEVecTmp[1] (never read) is not allocated,
     *  and the DG work array m_uiDGTmp0 is allocated untouched (its rows are
     *  written before they are read). Same values, fewer resident pages. */
    static bool lazyTmp() {
        static const int v = []() {
            const char* e = std::getenv("DENDRO_ENUTS_LAZY_TMP");
            return (e && e[0] == '1') ? 1 : 0;
        }();
        return v == 1;
    }
    /** DENDRO_ENUTS_WPART_LATE=1 (default off): register the LTS work weight
     *  and repartition only in init(), after the context's initialize(), not
     *  in the constructor. The constructor's early registration makes the
     *  context's initial-grid iterations run on work-weighted partitions, so
     *  a weighted run converges to a different initial octree than the
     *  unweighted twin (7050 vs 7036 elements on q1); with this knob both
     *  start from the same octree and only the partition differs. */
    /** DENDRO_ENUTS_OWNER_SYNC_LABELLED=1 (default off): build the owner-sync
     *  plan on labelled meshes too. Allowed only when the labelled rebuild
     *  joined on the SFC source's canonical ownership keys (K6): then every
     *  node has one owner element (cg2dg), locally owned nodes have a local
     *  owner element and the send map holds cg indices, which is the model the
     *  plan assumes; the checks below verify it and the decision is collective.
     *  The end-of-step zip then takes the same owner (cg2dg) as the plan. */
    static bool ownerSyncLabelled() {
        static const int v = []() {
            const char* e = std::getenv("DENDRO_ENUTS_OWNER_SYNC_LABELLED");
            return (e && e[0] == '1') ? 1 : 0;
        }();
        return v == 1;
    }
    /** DENDRO_ENUTS_OWN_NO_SELFCOPY=1 (default off): the owner plan does not
     *  copy, onto the block's own element slots, nodes that element itself
     *  owns: the source is that element's DG row, which the zip just filled
     *  from this very block, so the copy is a no-op. Bitwise. */
    static bool ownNoSelfCopy() {
        static const int v = []() {
            const char* e = std::getenv("DENDRO_ENUTS_OWN_NO_SELFCOPY");
            return (e && e[0] == '1') ? 1 : 0;
        }();
        return v == 1;
    }
    /** DENDRO_ENUTS_OWN_NO_ETIME_XCH=1 (default off): with the owner plan
     *  the per-sub-step exchange of the element time vector and the barrier
     *  after it are skipped. The owner path reads element times of local
     *  elements only (the block's first element; owner times come from the
     *  sub-step counter); the ghost entries feed the per-element sync path.
     *  Bitwise. */
    static bool ownNoEtimeXch() {
        static const int v = []() {
            const char* e = std::getenv("DENDRO_ENUTS_OWN_NO_ETIME_XCH");
            return (e && e[0] == '1') ? 1 : 0;
        }();
        return v == 1;
    }
    /** DENDRO_ENUTS_OWN_XCH_OMP=1 (default off; hybrid builds, with
     *  DENDRO_ENUTS_OMP): the pack and unpack of the owner exchange run
     *  threaded over the variables (plain indexed copies). Bitwise. */
    static bool ownXchOmp() {
#ifdef DENDRO_ENUTS_HAS_OMP
        static const int v = []() {
            const char* e = std::getenv("DENDRO_ENUTS_OWN_XCH_OMP");
            return (e && e[0] == '1') ? 1 : 0;
        }();
        return v == 1 && ompOn();
#else
        return false;
#endif
    }
    /** DENDRO_ENUTS_XCH_CENSUS=1 (prof-only, with DENDRO_ENUTS_PROF=1): of the
     *  owner-exchange doubles received, count at the next overwrite of each
     *  (stage, DG slot) whether a local plan read it since (rows: read / in a
     *  plan but unread / in no plan). Marks every plan read with a byte
     *  store; no effect on results. */
    static bool xchCensus() { return xchCensusLvl() >= 1 && on(); }
    /** =2: rank 0 also prints per call the unread slots by owner class */
    static int xchCensusLvl() {
        static const int v = []() {
            const char* e = std::getenv("DENDRO_ENUTS_XCH_CENSUS");
            return (e && e[0] >= '1' && e[0] <= '9') ? e[0] - '0' : 0;
        }();
        return v;
    }
    /** DENDRO_ENUTS_XCH_TRACE=1|<dir> (prof-only): per owner-exchange call the
     *  wait, the partner whose request completed last and how late it sent
     *  (its send stamp, one extra double per partner message; clocks
     *  calibrated per partner at plan build), written per rank to
     *  <dir>/enuts_xch_r<rank>.txt (1 = cwd); sums in the rows. */
    static const char* xchTraceDir() {
        static const std::string d = []() {
            const char* e = std::getenv("DENDRO_ENUTS_XCH_TRACE");
            if (!e || !e[0] || e[0] == '0') return std::string();
            return (e[0] == '1' && !e[1]) ? std::string(".") : std::string(e);
        }();
        return (d.empty() || !on()) ? nullptr : d.c_str();
    }
    static bool xchTrace() { return xchTraceDir() != nullptr; }
    /** DENDRO_ENUTS_XCH_SCOPED=1 (default off; owner plan, chained classes):
     *  once every block's plan exists (after the first sync pass) each rank
     *  tells each sender, per shared node, the class of the finest local
     *  block that reads it (255: none), and both sides re-sort the node list
     *  by max(owner class, reader class): a stage is sent only at sub-steps
     *  where a reader of it steps, never-read nodes never. Bitwise: a slot
     *  not sent is not read before it is sent again. */
    static bool xchScoped() {
        static const int v = []() {
            const char* e = std::getenv("DENDRO_ENUTS_XCH_SCOPED");
            return (e && e[0] == '1') ? 1 : 0;
        }();
        return v == 1;
    }
    /** DENDRO_ENUTS_OWN_P2C_SHARED=1 (default off): before each stage's sync
     *  loop, the unique children (parent element, child) of every active
     *  block are listed once, each gets a slot in one shared pool, and a
     *  threaded pass over the unique parent rows builds each row once and
     *  interpolates its children into the pool; the block loop then only
     *  copies. Same row build and interpolation code as the per-block path,
     *  so bitwise. Removes the cross-thread duplicates and the per-thread
     *  pool-cap recomputes. DENDRO_ENUTS_P2C_SHARED_MB (default 2048) caps
     *  the shared pool; children past the cap take the per-block path. */
    static bool ownP2cShared() { return ownP2cSharedMode() != 0; }
    /** 1 = shared pool, block loop copies from it; 2 = the pre-pass writes
     *  each child's outputs straight into every reader block (no pool, no
     *  second copy; the child slots have no other writer). Both bitwise. */
    static int ownP2cSharedMode() {
        static const int v = []() {
            const char* e = std::getenv("DENDRO_ENUTS_OWN_P2C_SHARED");
            const int i   = e ? std::atoi(e) : 0;
            return (i == 1 || i == 2) ? i : 0;
        }();
        return v;
    }
    static size_t p2cSharedBytes() {
        static const size_t v = []() {
            const char* e = std::getenv("DENDRO_ENUTS_P2C_SHARED_MB");
            return (size_t)(e ? std::atol(e) : 2048) << 20;
        }();
        return v;
    }
    static bool wpartLate() {
        static const int v = []() {
            const char* e = std::getenv("DENDRO_ENUTS_WPART_LATE");
            return (e && e[0] == '1') ? 1 : 0;
        }();
        return v == 1;
    }
    static bool ifaceNL() { return (ifaceFix() & 7) == 7; }
    static bool ifaceShift() { return (ifaceFix() & 8) != 0; }
    /** DENDRO_ENUTS_OWNER_SYNC=1: before a block takes its stage data, give
     *  every shared or hanging node of the block's elements and of its unzip
     *  dependents the value of the node's owner (what zip + unzip does in ETS).
     *  1: per-block plan (OwnPlan), owner values exchanged per node on more
     *     than one rank.
     *  3: the original per-element zip + gather (one rank only).
     *  2: as 3, also poison the scratch cg vector and count unwritten reads. */
    static int ownerSync() {
        static const int v = []() {
            const char* e = std::getenv("DENDRO_ENUTS_OWNER_SYNC");
            return e ? std::atoi(e) : 0;
        }();
        return v;
    }
    static bool syncFast() {
        static const int v = []() {
            const char* e = std::getenv("DENDRO_ENUTS_SYNC_FAST");
            return (e && e[0] == '1') ? 1 : 0;
        }();
        return v == 1;
    }
    /** DENDRO_ENUTS_OWNER_SYNC_CHECK=1: verify the hanging-node recipes of the
     *  owner-sync plan against Mesh::getElementNodalValues once per mesh */
    static bool ownerSyncCheck() {
        static const int v = []() {
            const char* e = std::getenv("DENDRO_ENUTS_OWNER_SYNC_CHECK");
            return (e && e[0] == '1') ? 1 : 0;
        }();
        return v == 1;
    }
    /** DENDRO_ENUTS_P2C_STORE_MB (default 128): cap of the per-stage-pass
     *  store of parent-to-child interpolations */
    static size_t p2cStoreBytes() {
        static const size_t v = []() {
            const char* e = std::getenv("DENDRO_ENUTS_P2C_STORE_MB");
            return (size_t)(e ? std::max(0, std::atoi(e)) : 128) << 20;
        }();
        return v;
    }
};
}  // namespace ts

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
     * elements and of its dependents (DENDRO_ENUTS_OWNER_SYNC) */
    std::vector<std::vector<unsigned int>> m_uiBlkOwnEid;

    /**@brief level-scoped element-DG exchange plan (DENDRO_ENUTS_SCOPED_EXCH);
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
    /** OWN_P2C_SHARED: shared-pool bookkeeping of the current pass */
    bool m_uiP2cSharedOn = false;            // this pass used the shared pool
    std::vector<unsigned int> m_uiP2cWorkRow;   // unique rows of the pass
    std::vector<unsigned int> m_uiP2cWorkBlk;   // a reader block per row
    std::vector<unsigned int> m_uiP2cWorkOff;   // children offsets per row
    std::vector<unsigned int> m_uiP2cWorkStore; // store idx per child
    std::vector<unsigned char> m_uiP2cWorkCnum; // child number per child
    std::vector<unsigned int> m_uiP2cWorkRdr, m_uiP2cWorkOutB, m_uiP2cWorkOutE;  // mode 2: reader block + its runs
    std::vector<T> m_uiOwnRowBuf2;              // mode 2: one child's values
    std::vector<unsigned long long> m_uiP2cRowPass;  // row -> pass it was listed in
    std::vector<unsigned int> m_uiP2cRowAt;          // row -> index in the work list
    std::vector<SyncTag> m_uiSyncTag;
    unsigned long long m_uiSyncPass = 0;
    bool m_uiSyncPassOpen           = false;
    std::vector<T> m_uiSyncCVec;

    /** per local block, the padding-fill copy plan (DENDRO_ENUTS_SYNC_FAST):
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

    /** per local block, the owner-sync fill (DENDRO_ENUTS_OWNER_SYNC=1): every
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
    /** OWN_P2C_SHARED: list the pass's unique children and interpolate them */
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
    /** G_XCH with XCH_SCOPED: the unscoped lists, kept at the re-sort (the
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
    /** DENDRO_ENUTS_XCH_SCOPED: lists re-sorted by reader class (done once per mesh) */
    bool m_uiXchScoped = false;
    /** super-time-stepped damping, stepper side (default off): the solver
     *  registers one variable, s(ratio) and a stage operator with set_sts().
     *  At the end of partial_evolve every class whose dt divides pt has just
     *  completed a block step; coarsest first, its blocks run s stages of the
     *  operator on that variable, with one same-class padding round per stage
     *  (one-variable zip, class-range owner exchange, run / single-node copy
     *  from the plan). Padding of other classes, and hanging-node interfaces
     *  inside a class, stay frozen at U's value for the pass. */
    bool m_uiStsOn        = false;
    unsigned int m_uiStsVar = 0;
    std::function<unsigned int(unsigned int)> m_uiStsStages;
    /** (blk, j, s, dt, y0, y_{j-1}, y_{j-2}, ly0, y_j): write Y_j at the
     *  block's points (and L(y0) into ly0 when j == 1); y_j arrives holding
     *  y_{j-1}, so the padding carries over */
    std::function<void(unsigned int, unsigned int, unsigned int, T, const T*, const T*,
                       const T*, T*, T*)> m_uiStsStage;
    std::vector<std::vector<T>> m_uiStsScr;  // per block: Y_a, Y_b, Y_c, L(Y0)
    std::vector<T> m_uiStsDG;                // one-variable DG vector
    std::vector<T> m_uiStsChk;
    /** the identity test operator is installed (node-coincident rounds only,
     *  no cross-class refresh: bitwise by construction) */
    bool m_uiStsIdent = false;
    /** scratch of the one-variable fill (g, row, pin, pout, 6 sweeps | im1,
     *  im2; T is double here as in own_sync_fill) */
    mutable std::vector<T> m_uiStsOpScr;
    /** per block, the padded slots the plan fills from other classes (test
     *  modes: the break and the dump's shell flag; empty until asked) */
    std::vector<std::vector<unsigned char>> m_uiStsXMask;
    /** STS_TEST_C: per block the damping applied so far (interior slots),
     *  per class the largest |Y_s - Y_0| of any pass, coarse steps dumped */
    std::vector<std::vector<T>> m_uiStsAcc;
    std::vector<std::vector<T>> m_uiStsBrk;  // STS_TEST_BREAK=1: first-pass padding
    std::vector<T> m_uiStsInc;
    unsigned int m_uiStsSteps = 0;
    T m_uiStsDt0 = 0;
    /** DENDRO_ENUTS_STS_TEST=<s> (test only): the identity operator with s
     *  stages on every class of ratio >= 4; the pass must leave U bitwise
     *  unchanged, and DENDRO_ENUTS_STS_CHECK=1 verifies that per block */
    static int stsTest() {
        static const int v = []() {
            const char* e = std::getenv("DENDRO_ENUTS_STS_TEST");
            return e ? std::atoi(e) : 0;
        }();
        return v;
    }
    static bool stsCheck() {
        static const int v = []() {
            const char* e = std::getenv("DENDRO_ENUTS_STS_CHECK");
            return (e && e[0] == '1') ? 1 : 0;
        }();
        return v == 1;
    }
    /** DENDRO_ENUTS_STS_TEST_C=<C> (test only): explicit diffusion of
     *  variable DENDRO_ENUTS_STS_TEST_VAR (default 0) with RKL2 stages,
     *  Y += D sum_dirs (y_- - 2 y + y_+) at the block's interior points, D =
     *  C (dt_block / dt0) (h_finest / h_block)^2, dt0 =
     *  DENDRO_ENUTS_STS_TEST_DT0 (default: the fine step at the first pass,
     *  so D = C on the finest level), stage count from (s^2 + s - 2) / 4 >=
     *  6 D plus one. The first interior layer inside a domain face gets
     *  L = 0. DENDRO_ENUTS_STS_TEST_DUMP=<dir>: at the end of every coarse
     *  step, per rank, the blocks' interior points with the shell flag
     *  (within the padding width of a slot another class fills), the
     *  accumulated damping and U (sts_dump). DENDRO_ENUTS_STS_TEST_BREAK=1
     *  breaks the interface rule on purpose: other-class padding is pinned
     *  to its value at the block's first pass (=2: to zero). */
    static double stsTestC() {
        static const double v = []() {
            const char* e = std::getenv("DENDRO_ENUTS_STS_TEST_C");
            return e ? std::atof(e) : 0.0;
        }();
        return v;
    }
    static unsigned int stsTestVar() {
        static const unsigned int v = []() {
            const char* e = std::getenv("DENDRO_ENUTS_STS_TEST_VAR");
            return e ? (unsigned int)std::atoi(e) : 0u;
        }();
        return v;
    }
    static double stsTestDt0() {
        static const double v = []() {
            const char* e = std::getenv("DENDRO_ENUTS_STS_TEST_DT0");
            return e ? std::atof(e) : 0.0;
        }();
        return v;
    }
    static const char* stsTestDump() {
        static const char* v = std::getenv("DENDRO_ENUTS_STS_TEST_DUMP");
        return v;
    }
    static int stsTestBreak() {
        static const int v = []() {
            const char* e = std::getenv("DENDRO_ENUTS_STS_TEST_BREAK");
            return e ? std::atoi(e) : 0;
        }();
        return v;
    }
    /** DENDRO_ENUTS_STS_FROZEN_ONLY=1: no cross-class refresh before a pass
     *  (the A/B of the interface rule: lag of one coarse step) */
    static bool stsFrozenOnly() {
        static const int v = []() {
            const char* e = std::getenv("DENDRO_ENUTS_STS_FROZEN_ONLY");
            return (e && e[0] == '1') ? 1 : 0;
        }();
        return v == 1;
    }
    /** damping number of the test operator for a block step dt at level lev */
    double sts_test_D(T dt, unsigned int lev) const {
        return stsTestC() * (double)(dt / m_uiStsDt0) *
               std::ldexp(1.0, -2 * (int)(m_uiLevMax - lev));
    }
    void sts_hook(unsigned int pt, const unsigned int* actIds, unsigned int nAct, bool ownPlan,
                  bool thr);
    void sts_pass(unsigned int cls, unsigned int s, T dt, const unsigned int* actIds,
                  unsigned int nAct, bool thr, unsigned int pt);
    void sts_zip(unsigned int blk, const T* y, T* dg) const;
    /** the plan's class-cls sources into y: node-coincident copies always,
     *  hanging faces / edges and the children of coarser dependents of that
     *  class when hang (one variable, every source at its value in dg) */
    void sts_fill(unsigned int blk, unsigned int cls, const T* dg, T* y, bool hang) const;
    const std::vector<unsigned char>& sts_xmask(unsigned int blk);
    void sts_test_stage(unsigned int blk, unsigned int j, unsigned int s, const T* y0,
                        const T* y1, const T* y2, T* ly0, T* yj, double D) const;
    void sts_test_acc(unsigned int blk, unsigned int cls, const T* y0, const T* ys);
    void sts_dump();

   public:
    /** the block's U, every variable padded (stride: the block's padded
     *  size), at the time of the pass: what a stage operator reads besides
     *  its own variable */
    const T* sts_blk_u(unsigned int blk) const { return m_uiBVec[blk]._vec[0].data(); }
    /** register the super-time-stepped variable: stages(ratio) -> s (0 keeps
     *  the variable's term in the RHS for that class), stage(...) as above */
    void set_sts(unsigned int var, std::function<unsigned int(unsigned int)> stages,
                 std::function<void(unsigned int, unsigned int, unsigned int, T, const T*,
                                    const T*, const T*, T*, T*)> stage) {
        m_uiStsVar    = var;
        m_uiStsStages = std::move(stages);
        m_uiStsStage  = std::move(stage);
        m_uiStsOn     = (bool)m_uiStsStage;
    }

   private:
    bool xch_scope_build(ot::Mesh* pMesh);
    bool prepare_owner_sync(ot::Mesh* pMesh);
    void build_own_plan(unsigned int blk, const std::vector<unsigned int>& eid,
                        OwnPlan& op);
    void own_sync_fill(unsigned int blk, unsigned int rk_s,
                       const std::vector<unsigned int>& eid);
    void own_row_ops(unsigned int ele, const T* g, T* out, T* pin, T* pout,
                     double* im1, double* im2) const;
    /** cls >= 0: one class's slot range instead of the prefix of pt (STS
     *  rounds; no census, trace or prof rows) */
    void own_node_exchange_begin(T* stv, unsigned int dof, unsigned int pt, int cls = -1);
    void own_node_exchange_end(T* stv, unsigned int dof, unsigned int pt, int cls = -1);
    /** with m_uiSyncPlan / m_uiBlkDepEid: the plans hold p2c store indices */
    void own_reset() {
        m_uiOwnMesh = nullptr;
        m_uiOwnOk   = false;
        m_uiOwnPlan.clear();
        m_uiStsXMask.clear();
        m_uiStsBrk.clear();
        m_uiStsAcc.clear();
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

    /** DENDRO_ENUTS_STAGE_SCRATCH / _RING state (see EnutsProf) */
    bool m_uiStageScratch = false;
    bool m_uiStageRing    = false;
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
    /** the stage-input vector of a block: _vec[NS+1] or the thread's scratch */
    inline BlockAsyncVector<T>& stage_in(unsigned int blk) {
        return m_uiStageScratch ? m_uiStageIn[ts::EnutsProf::tid()]
                                : m_uiBVec[blk]._vec[m_uiNumStages + 1];
    }
    /** vector behind a debug index s in 0..NS+1 (NS+1 = the stage input) */
    inline const BlockAsyncVector<T>& stage_vec_dbg(unsigned int blk,
                                                    unsigned int s) {
        return (s == m_uiNumStages + 1) ? stage_in(blk)
                                        : m_uiBVec[blk]._vec[stage_slot(s)];
    }
    inline void mark_stages_unsynced(unsigned int blk) {
        for (unsigned int s = 1; s < m_uiBVec[blk]._vec.size(); s++)
            if (m_uiStageScratch || s <= m_uiNumStages)
                m_uiBVec[blk]._vec[s].mark_unsynced();
    }
    /** DENDRO_ENUTS_LAZY_TMP: allocate a temporary on first use */
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

   public:
    /**@brief: constructor
     * Assumptions: Note that blocks result in from octree to block
     * decomposition can be not 2:1 balanced. But to perform NUTS, blocks should
     * be 2:1 balanced.
     * @param ctx : the solver context
     * @param useWpart : repartition by LTS work after the initial grid
     * @param opts : the settings that change the solution
     */
    ExplicitNUTS(Ctx* ctx, bool useWpart = true,
                 const ts::LtsOptions& opts = ts::LtsOptions());

    /**@brief step of a block at level blev in units of the finest step*/
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
    ts::EnutsProf::options()           = opts;
    ts::EnutsProf::options().ratio_cap = ts::snap_ratio_cap(opts.ratio_cap);
    m_uiUseWeightedPart                = useWpart;
    // WPART_LATE: leave the weight unregistered until init(), after the
    // context's initialize(), so the initial grid is the unweighted one
    if (m_uiUseWeightedPart && !ts::EnutsProf::wpartLate()) {
        ot::Mesh* pMesh = m_uiAppCtx->get_mesh();
        pMesh->computeMinMaxLevel(m_uiLevMin, m_uiLevMax);

        update_wpart_metadata();

        if (!m_uiAppCtx->is_wpart_func_set())
            m_uiAppCtx->set_wpart_function(getOctWeight);

        // the ctx path is virtual: solvers with a labelled mesh route it
        // through their SFC-twin sandwich
        m_uiAppCtx->remesh_and_gridtransfer();
        m_uiAppCtx->on_mesh_changed(false);
    }
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
                "sts: passes (stages + one-variable rounds; wall)", "  count: sts rounds",
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

    const bool lazyTmp = ts::EnutsProf::lazyTmp();
    if (!lazyTmp) {
        m_uiEVecTmp[0].create_vector(m_uiAppCtx->get_mesh(),
                                     ot::DVEC_TYPE::OCT_SHARED_NODES,
                                     ot::DVEC_LOC::HOST, m_uiEVar.get_dof(), true);
        m_uiEVecTmp[1].create_vector(m_uiAppCtx->get_mesh(),
                                     ot::DVEC_TYPE::OCT_SHARED_NODES,
                                     ot::DVEC_LOC::HOST, m_uiEVar.get_dof(), true);
    }

    // allocate tmp DG vectors (untouched with LAZY_TMP: a work array)
    m_uiDGTmp0.create_vector(m_uiAppCtx->get_mesh(),
                             ot::DVEC_TYPE::OCT_LOCAL_NODES, ot::DVEC_LOC::HOST,
                             m_uiEVar.get_dof(), true, !lazyTmp);

    m_uiStGOn = ts::EnutsProf::ifaceNL() && m_uiType == ETSType::RK4;
    if (m_uiStGOn) {
        m_uiStG.create_vector(m_uiAppCtx->get_mesh(),
                              ot::DVEC_TYPE::OCT_LOCAL_NODES, ot::DVEC_LOC::HOST,
                              m_uiEVar.get_dof(), true);
        std::fill(m_uiStG.get_vec_ptr(),
                  m_uiStG.get_vec_ptr() + m_uiStG.get_size(), (T)0);
        m_uiStGValid.assign(m_uiAppCtx->get_mesh()->getAllElements().size(), 0);
        m_uiStGValidCls.assign(m_uiOwnDTs.size(), 0);
        if (ts::EnutsProf::on()) ts::EnutsProf::get().t[ts::EnutsProf::CN_GRESET] += 1;
    }

    m_uiEvarUzip.create_vector(m_uiAppCtx->get_mesh(),
                               ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING,
                               ot::DVEC_LOC::HOST, m_uiEVar.get_dof(), true);
    m_uiCgOwnerMesh = nullptr;
    m_uiBlkDepEid.clear();
    m_uiBlkOwnEid.clear();
    m_uiSyncPlan.clear();
    own_reset();
    free_sync_threads();
    m_uiScopedMesh = nullptr;
    m_uiScopedOk   = false;
    // LAZY_TMP: the DG copy of the solution is read by the element-wise sync
    // path (owner sync off), the labelled zip and the legacy remesh; with the
    // owner plan it is allocated where those paths start (ensure_evar_dg)
    if (!lazyTmp || ts::EnutsProf::ownerSync() != 1)
        m_uiEvarDG.create_vector(m_uiAppCtx->get_mesh(),
                                 ot::DVEC_TYPE::OCT_LOCAL_NODES,
                                 ot::DVEC_LOC::HOST, m_uiEVar.get_dof(), true);

    const ot::Mesh* pMesh = m_uiAppCtx->get_mesh();
    if (pMesh->isActive()) {
        m_uiEleTime.resize(pMesh->getAllElements().size(), 0);
        m_uiEleDT.resize(pMesh->getAllElements().size(), 0);

        const std::vector<ot::Block>& blkList = pMesh->getLocalBlockList();
        m_uiBVec.resize(blkList.size());

        m_uiStageScratch = ts::EnutsProf::stageScratch();
        m_uiStageRing    = ts::EnutsProf::stageRing();
        if (m_uiStageRing) {
            // the ring needs a_{s,j} = 0 for j < s-1 (stage s reads k_{s-1} only)
            for (unsigned int st = 3; st <= m_uiNumStages; st++)
                for (unsigned int j = 1; j < st - 1; j++)
                    if (m_uiAij[(st - 1) * m_uiNumStages + (j - 1)] != (T)0) {
                        std::cerr << "[enuts] DENDRO_ENUTS_STAGE_RING needs a "
                                     "tableau with a_{s,j} = 0 for j < s-1"
                                  << std::endl;
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }
        }
        // slots: solution, NS stages (ring: one), stage input (scratch: none)
        const unsigned int nvec = m_uiStageRing      ? 3
                                  : m_uiStageScratch ? m_uiNumStages + 1
                                                     : m_uiNumStages + 2;
        unsigned int maxNN = 0;
        for (unsigned int blk = 0; blk < blkList.size(); blk++) {
            const unsigned int sz[3] = {blkList[blk].getAllocationSzX(),
                                        blkList[blk].getAllocationSzY(),
                                        blkList[blk].getAllocationSzZ()};
            m_uiBVec[blk].alloc_vec(nvec, blk, sz, DOF);
            maxNN = std::max(maxNN, sz[0] * sz[1] * sz[2]);
        }
        if (m_uiStageScratch) {
            unsigned int nt = 1;
#ifdef DENDRO_ENUTS_HAS_OMP
            nt = (unsigned int)omp_get_max_threads();
#endif
            m_uiStageIn.resize(nt);
            const unsigned int ssz[3] = {maxNN, 1, 1};
            for (unsigned int t = 0; t < nt; t++)
                m_uiStageIn[t].createVec(0, ssz, false,
                                         BLK_ASYNC_VEC_MODE::BLK_UNZIP, DOF);
        }
        if (m_uiStageRing) {
            m_uiStageSgn.resize(blkList.size());
            for (unsigned int blk = 0; blk < blkList.size(); blk++)
                m_uiStageSgn[blk].assign(
                    ((size_t)DOF * m_uiBVec[blk]._vec[0].getSz() + 63) / 64, 0);
        }
    }

    m_uiECOp            = new ENUTSOp(m_uiType);
    m_uiIsInternalAlloc = true;
    m_uiBlkTimeLevMinMax.resize(2 * m_uiMaxDepth, 0);

    m_uiActiveBlkIDs.reserve(pMesh->getLocalBlockList().size());
    m_uiDependentActiveBlkIDs.reserve(pMesh->getLocalBlockList().size());
    m_uiIndependentActiveBlkIDs.reserve(pMesh->getLocalBlockList().size());
    if (ts::ts_mem_info()) mem_report();
    return 0;
}

/** DENDRO_TS_MEM_INFO=1: bytes per buffer family (collective, rank 0 prints).
 *  blk_*: the per-block padded vectors (_vec[0] solution, [1..NS] stages,
 *  [NS+1] stage input / output scratch); thr_dgW_omp: the per-thread sync
 *  scratch sync_thread_setup would allocate at this thread count. */
template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::mem_report() {
    static int ncall = 0;
    char tag[32];
    std::snprintf(tag, sizeof(tag), ncall ? "remesh%d" : "init", ncall);
    ncall++;
    const ot::Mesh* pMesh = m_uiAppCtx->get_mesh();
    const bool a          = pMesh->isActive();
    const double B        = (double)sizeof(T);
    const unsigned int DOF = m_uiEVar.get_dof();
    double blkN = 0;  // padded nodes over all local blocks (one copy, one dof)
    double stageSlots = 0, scratchSlots = 0;
    for (unsigned int blk = 0; blk < m_uiBVec.size(); blk++) {
        const double nn = (double)m_uiBVec[blk]._vec[0].getSz();
        blkN += nn;
        const unsigned int nv = (unsigned int)m_uiBVec[blk]._vec.size();
        scratchSlots += m_uiStageScratch ? 0 : nn;
        stageSlots += nn * (nv - 1 - (m_uiStageScratch ? 0 : 1));
    }
    for (unsigned int t = 0; t < m_uiStageIn.size(); t++)
        scratchSlots += (double)m_uiStageIn[t].getSz();
    double thr = 0;
#ifdef DENDRO_ENUTS_HAS_OMP
    if (a) {
        const double nt = (double)omp_get_max_threads();
        thr = (nt > 1 ? nt - 1 : 0) * B * DOF * pMesh->getAllElements().size() *
              pMesh->getNumNodesPerElement();
    }
#endif
    const char* nm[] = {"ctx_evar",   "blk_sol",   "blk_stages", "blk_scratch",
                        "evar_uzip",  "st_dg",     "dg_tmp0",    "evar_dg",
                        "cg_tmp_x2",  "stG",       "ele_time_dt", "thr_dgW_omp",
                        "ctx_unzip_x2_est"};
    double v[] = {
        a ? B * m_uiEVar.get_size() : 0,
        B * DOF * blkN,
        B * DOF * stageSlots,
        B * DOF * scratchSlots,
        a ? B * m_uiEvarUzip.get_size() : 0,
        a ? B * (double)m_uiNumStages * m_uiStVec[0].get_size() : 0,
        a ? B * m_uiDGTmp0.get_size() : 0,
        (a && m_uiEvarDG.get_vec_ptr()) ? B * m_uiEvarDG.get_size() : 0,
        a ? B * ((m_uiEVecTmp[0].get_vec_ptr() ? m_uiEVecTmp[0].get_size() : 0) +
                 (m_uiEVecTmp[1].get_vec_ptr() ? m_uiEVecTmp[1].get_size() : 0))
          : 0,
        (a && m_uiStGOn) ? B * m_uiStG.get_size() : 0,
        (double)sizeof(unsigned int) * (m_uiEleTime.size() + m_uiEleDT.size()),
        thr,
        a ? B * 2.0 * pMesh->getDegOfFreedomUnZip() * DOF : 0};
    ts::ts_mem_counts<T>(pMesh, DOF, "enuts", tag);
    ts::ts_mem_print("enuts", tag, 13, nm, v, true);

    // stage locality census: which blocks sit next to a time-level change and
    // which element rows of m_uiStVec are read at all / read by a block with a
    // smaller step (those must survive past the sub-step they were written in)
    double c[7] = {0, 0, 0, 0, 0, 0, 0};
    const char* cn[] = {"blocks",        "blocks_jump",  "blk_nodes",
                        "blk_nodes_jump", "elems_all",    "elems_live",
                        "elems_persist"};
    if (a && m_uiLevMin >= 1 && m_uiLevMax <= m_uiMaxDepth &&
        m_uiLevMin <= m_uiLevMax) {
        const std::vector<ot::TreeNode>& all  = pMesh->getAllElements();
        const std::vector<ot::Block>& blkList = pMesh->getLocalBlockList();
        const bool own = ts::EnutsProf::ownerSync() != 0;
        sync_lazy_build(own);  // same lists partial_evolve builds lazily
        std::vector<unsigned int> dtOf(all.size(), 0);
        for (size_t e = 0; e < all.size(); e++)
            dtOf[e] = step_ratio(all[e].getLevel());
        std::vector<unsigned char> live(all.size(), 0), persist(all.size(), 0);
        for (unsigned int b = 0; b < blkList.size(); b++) {
            const unsigned int bdt = dtOf[blkList[b].firstElementID()];
            const std::vector<unsigned int>& R =
                (own && b < m_uiBlkOwnEid.size()) ? m_uiBlkOwnEid[b]
                                                  : m_uiBlkDepEid[b];
            bool jump = false;
            for (const unsigned int e : R) {
                live[e] = 1;
                if (dtOf[e] != bdt) jump = true;
                if (dtOf[e] > bdt) persist[e] = 1;
            }
            const double nn = (double)m_uiBVec[b]._vec[0].getSz();
            c[0] += 1;
            c[2] += nn;
            if (jump) {
                c[1] += 1;
                c[3] += nn;
            }
        }
        const std::vector<unsigned int>& sm = pMesh->getSendElementSM();
        const unsigned int lb                = pMesh->getElementLocalBegin();
        for (const unsigned int e : sm) live[e + lb] = 1;
        c[4] = (double)all.size();
        for (size_t e = 0; e < all.size(); e++) {
            c[5] += live[e];
            c[6] += persist[e];
        }
    }
    ts::ts_mem_print("enuts", tag, 7, cn, c, false);
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
    if (m_uiStGOn) {
        m_uiStG.destroy_vector();
        m_uiStGOn = false;
    }

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
    for (unsigned int t = 0; t < m_uiStageIn.size(); t++)
        m_uiStageIn[t].destroyVec();
    m_uiStageIn.clear();
    m_uiStageSgn.clear();
    m_uiBlkDepEid.clear();
    m_uiBlkOwnEid.clear();
    m_uiSyncPlan.clear();
    own_reset();
    free_sync_threads();
    m_uiScopedMesh = nullptr;
    m_uiScopedOk   = false;

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
int ExplicitNUTS<T, Ctx>::sync_with_mesh() {
    if (m_uiAppCtx->is_ets_synced()) return 0;

    // the level offset belongs to the new mesh, and the ladder built below reads it
    m_uiLevOffset = m_uiAppCtx->compute_lts_ts_offset();

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
        // DENDRO_ENUTS_WPART_GHOST_SYNC=1: ghost-exchange the evolution vector
        // before the weighted remesh. the IGT reads ghost cgs of local elements,
        // and ctx initialize may end with a local-only init_grid after an IGT
        // (which writes local cgs only), leaving those ghosts zero; elements that
        // change rank then carry the zeros to their new owner.
        static const bool wpartGhostSync = [] {
            const char* e = std::getenv("DENDRO_ENUTS_WPART_GHOST_SYNC");
            return e && e[0] == '1';
        }();
        if (wpartGhostSync && pMesh->isActive()) {
            auto& ev            = m_uiAppCtx->get_evolution_vars();
            T* evp              = ev.get_vec_ptr();
            const unsigned int dof = ev.get_dof();
            const unsigned int nn  = pMesh->getDegOfFreedom();
            const unsigned int gb = pMesh->getNodeLocalBegin();
            const unsigned int ge = pMesh->getNodeLocalEnd();
            auto countGhostZeros = [&]() {
                DendroIntL z = 0;
                for (unsigned int v = 0; v < dof; v++)
                    for (unsigned int n = 0; n < nn; n++)
                        if ((n < gb || n >= ge) && evp[v * nn + n] == T(0)) z++;
                return z;
            };
            DendroIntL z0 = countGhostZeros();
            pMesh->readFromGhostBegin(evp, dof);
            pMesh->readFromGhostEnd(evp, dof);
            DendroIntL z1 = countGhostZeros(), z0g = 0, z1g = 0;
            par::Mpi_Reduce(&z0, &z0g, 1, MPI_SUM, 0, pMesh->getMPICommunicator());
            par::Mpi_Reduce(&z1, &z1g, 1, MPI_SUM, 0, pMesh->getMPICommunicator());
            if (!pMesh->getMPIRank())
                std::cout << "[LTS Init(Wpart)]: ghost sync, zero ghost entries "
                          << z0g << " -> " << z1g << std::endl;
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
    // DENDRO_ENUTS_OWNER_BYPOS (default on, =0 reverts): pick the owner per grid
    // POSITION over every local and received-ghost element, not per cg slot. on a
    // labelled mesh the sharers of one position are split across a primary slot,
    // its duplicate slots and ghost slots, so the per-slot pick above can land on
    // a finer sharer's copy than the SFC owner (and differ between a primary and
    // its dups); after sub-cycling those copies differ well above rounding
    static const bool byPos = [] {
        const char* e = std::getenv("DENDRO_ENUTS_OWNER_BYPOS");
        return !(e && e[0] == '0');
    }();
    const std::vector<unsigned int>& cg2dg = pMesh->getCG2DGMap();
    if (byPos) {
        const unsigned int md = m_uiMaxDepth;
        struct KeyHash {
            size_t operator()(const unsigned __int128& k) const {
                const uint64_t a = (uint64_t)k, b = (uint64_t)(k >> 64);
                return std::hash<uint64_t>()(a ^ (b * 0x9E3779B97F4A7C15ull));
            }
        };
        auto key = [&](unsigned int e, unsigned int i, unsigned int j, unsigned int k) {
            const unsigned long long len = 1ull << (md - all[e].getLevel());
            const unsigned long long x = (unsigned long long)all[e].getX() * eo + i * len;
            const unsigned long long y = (unsigned long long)all[e].getY() * eo + j * len;
            const unsigned long long z = (unsigned long long)all[e].getZ() * eo + k * len;
            return ((unsigned __int128)x << 84) | ((unsigned __int128)y << 42) | (unsigned __int128)z;
        };
        std::unordered_map<unsigned __int128, unsigned int, KeyHash> best;
        best.reserve(pMesh->getNodeLocalEnd() - pMesh->getNodeLocalBegin());
        for (unsigned int cg = pMesh->getNodeLocalBegin(); cg < pMesh->getNodeLocalEnd(); cg++) {
            if (cg >= cg2dg.size() || cg2dg[cg] == LOOK_UP_TABLE_DEFAULT) continue;
            unsigned int e, i, j, k;
            pMesh->dg2eijk(cg2dg[cg], e, i, j, k);
            best.emplace(key(e, i, j, k), LOOK_UP_TABLE_DEFAULT);
        }
        auto visit = [&](unsigned int e) {
            for (unsigned int k = 0; k <= eo; k++)
                for (unsigned int j = 0; j <= eo; j++)
                    for (unsigned int i = 0; i <= eo; i++) {
                        if (pMesh->isNodeHanging(e, i, j, k)) continue;
                        auto it = best.find(key(e, i, j, k));
                        if (it == best.end()) continue;
                        const unsigned int dg = e * nPe + k * (eo + 1) * (eo + 1) + j * (eo + 1) + i;
                        unsigned int& cur = it->second;
                        if (cur == LOOK_UP_TABLE_DEFAULT) { cur = dg; continue; }
                        const unsigned int ce = cur / nPe;
                        if (all[e].getLevel() < all[ce].getLevel() ||
                            (all[e].getLevel() == all[ce].getLevel() && all[e] < all[ce]))
                            cur = dg;
                    }
        };
        for (unsigned int e = pMesh->getElementLocalBegin(); e < pMesh->getElementLocalEnd(); e++) visit(e);
        for (unsigned int e : pMesh->getRecvElementSM()) visit(e);  // ghosts the DG exchange fills
        for (unsigned int cg = pMesh->getNodeLocalBegin(); cg < pMesh->getNodeLocalEnd(); cg++) {
            if (cg >= cg2dg.size() || cg2dg[cg] == LOOK_UP_TABLE_DEFAULT) continue;
            unsigned int e, i, j, k;
            pMesh->dg2eijk(cg2dg[cg], e, i, j, k);
            const unsigned int w = best[key(e, i, j, k)];
            if (w != LOOK_UP_TABLE_DEFAULT) m_uiCgOwnerDG[cg] = w;
        }
    }
    // slots no element references (or only hanging ones): keep the mesh's own canonical
    for (unsigned int cg = pMesh->getNodeLocalBegin(); cg < pMesh->getNodeLocalEnd(); cg++)
        if (m_uiCgOwnerDG[cg] == LOOK_UP_TABLE_DEFAULT && cg < cg2dg.size()) m_uiCgOwnerDG[cg] = cg2dg[cg];
    if (ts::EnutsProf::ownerSyncLabelled() && pMesh->canonKeysBuilt()) {
        // OWNER_SYNC_LABELLED: the plan reads owners through cg2dg; the zip
        // must take the same copy. count where the by-position pick differs
        size_t diff = 0;
        for (unsigned int cg = pMesh->getNodeLocalBegin(); cg < pMesh->getNodeLocalEnd() && cg < cg2dg.size(); cg++) {
            if (m_uiCgOwnerDG[cg] != cg2dg[cg]) diff++;
            m_uiCgOwnerDG[cg] = cg2dg[cg];
        }
        std::cout << "[enuts-owner-sync r" << pMesh->getMPIRankGlobal()
                  << "] end-of-step owner pinned to cg2dg; by-position pick differed at "
                  << diff << " local cgs" << std::endl;
    }
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
    ensure_evar_dg();
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
void ExplicitNUTS<T, Ctx>::build_sync_plan(unsigned int blk,
                                           const std::vector<unsigned int>& eid,
                                           SyncPlan& pl) {
    ot::Mesh* pMesh = (ot::Mesh*)m_uiAppCtx->get_mesh();
    const ot::TreeNode* pNodes = pMesh->getAllElements().data();
    const ot::Block* blkList   = pMesh->getLocalBlockList().data();
    const ot::TreeNode blkNode = blkList[blk].getBlockNode();
    const unsigned int PW      = blkList[blk].get1DPadWidth();
    const unsigned int eOrder  = pMesh->getElementOrder();
    const unsigned int lx      = blkList[blk].getAllocationSzX();
    const unsigned int ly      = blkList[blk].getAllocationSzY();
    const unsigned int lz      = blkList[blk].getAllocationSzZ();
    const unsigned int bLev = pNodes[blkList[blk].firstElementID()].getLevel();

    const double hx   = (1u << (m_uiMaxDepth - bLev)) / (double)eOrder;
    const double xmin = blkNode.minX() - PW * hx;
    const double xmax = blkNode.maxX() + PW * hx;
    const double ymin = blkNode.minY() - PW * hx;
    const double ymax = blkNode.maxY() + PW * hx;
    const double zmin = blkNode.minZ() - PW * hx;
    const double zmax = blkNode.maxZ() + PW * hx;
    const double d_compar_tol = 1e-6;
    const unsigned int n1     = eOrder + 1;

    pl.ents.clear();
    pl.src.clear();
    pl.dst.clear();

    // one node's target index in the block, or false when it lies outside
    auto clampIdx = [&](double c, double cmin, double cmax, double invhh,
                        unsigned int& idx) -> bool {
        if (fabs(c - cmin) < d_compar_tol) c = cmin;
        if (fabs(c - cmax) < d_compar_tol) c = cmax;
        if (c < cmin || c > cmax) return false;
        idx = std::round((c - cmin) * invhh);
        return true;
    };

    std::vector<ot::TreeNode> childOct;
    childOct.reserve(NUM_CHILDREN);

    for (unsigned int m = 0; m < eid.size(); m++) {
        const unsigned int ele = eid[m];
        const unsigned int lev = pNodes[ele].getLevel();

        if (lev == bLev || lev > bLev) {
            const bool fine = lev > bLev;
            const double hh = (1u << (m_uiMaxDepth - lev)) / (double)eOrder;
            const double invhh = fine ? 1.0 / (2 * hh) : 1.0 / hh;
            const unsigned int cb   = fine ? ((eOrder % 2 == 0) ? 0 : 1) : 0;
            const unsigned int step = fine ? 2 : 1;
            const unsigned int b0   = pl.src.size();
            for (unsigned int k = cb; k < n1; k += step) {
                unsigned int kkz;
                if (!clampIdx(pNodes[ele].minZ() + k * hh, zmin, zmax, invhh, kkz)) continue;
                for (unsigned int j = cb; j < n1; j += step) {
                    unsigned int jjy;
                    if (!clampIdx(pNodes[ele].minY() + j * hh, ymin, ymax, invhh, jjy)) continue;
                    for (unsigned int i = cb; i < n1; i += step) {
                        unsigned int iix;
                        if (!clampIdx(pNodes[ele].minX() + i * hh, xmin, xmax, invhh, iix)) continue;
                        pl.src.push_back(k * n1 * n1 + j * n1 + i);
                        pl.dst.push_back(kkz * lx * ly + jjy * lx + iix);
                    }
                }
            }
            if (pl.src.size() > b0)
                pl.ents.push_back({ele, -1, b0, (unsigned int)pl.src.size(), 0u});
        } else {
            childOct.clear();
            pNodes[ele].addChildren(childOct);
            for (unsigned int child = 0; child < NUM_CHILDREN; child++) {
                if ((childOct[child].maxX() < xmin || childOct[child].minX() >= xmax) ||
                    (childOct[child].maxY() < ymin || childOct[child].minY() >= ymax) ||
                    (childOct[child].maxZ() < zmin || childOct[child].minZ() >= zmax))
                    continue;
                const double hh = (1u << (m_uiMaxDepth - childOct[child].getLevel())) / (double)eOrder;
                const double invhh = 1.0 / hh;
                const unsigned int b0 = pl.src.size();
                for (unsigned int k = 0; k < n1; k++) {
                    unsigned int kkz;
                    if (!clampIdx(childOct[child].minZ() + k * hh, zmin, zmax, invhh, kkz)) continue;
                    for (unsigned int j = 0; j < n1; j++) {
                        unsigned int jjy;
                        if (!clampIdx(childOct[child].minY() + j * hh, ymin, ymax, invhh, jjy)) continue;
                        for (unsigned int i = 0; i < n1; i++) {
                            unsigned int iix;
                            if (!clampIdx(childOct[child].minX() + i * hh, xmin, xmax, invhh, iix)) continue;
                            pl.src.push_back(k * n1 * n1 + j * n1 + i);
                            pl.dst.push_back(kkz * lx * ly + jjy * lx + iix);
                        }
                    }
                }
                if (pl.src.size() > b0)
                    pl.ents.push_back(
                        {ele, (int)childOct[child].getMortonIndex(), b0,
                         (unsigned int)pl.src.size(),
                         p2c_index(ele, childOct[child].getMortonIndex())});
            }
        }
    }
    pl.built = true;
}

template <typename T, typename Ctx>
unsigned int ExplicitNUTS<T, Ctx>::p2c_index(unsigned int ele,
                                             unsigned int cnum) {
    const ot::Mesh* pMesh = m_uiAppCtx->get_mesh();
    const size_t n        = (size_t)pMesh->getAllElements().size() * NUM_CHILDREN;
    if (m_uiP2cIdx.size() != n) {
        m_uiP2cIdx.assign(n, -1);
        m_uiP2cPassOf.clear();
        m_uiP2cSlotOf.clear();
        m_uiP2cTagOf.clear();
        m_uiP2cStepOf.clear();
    }
    int& i = m_uiP2cIdx[(size_t)ele * NUM_CHILDREN + cnum];
    if (i < 0) {
        i = (int)m_uiP2cPassOf.size();
        m_uiP2cPassOf.push_back(0);
        m_uiP2cSlotOf.push_back(0);
        m_uiP2cTagOf.push_back(SyncTag());
    }
    return (unsigned int)i;
}

template <typename T, typename Ctx>
T* ExplicitNUTS<T, Ctx>::p2c_slot(unsigned int idx, const SyncTag& key,
                                  size_t stride, bool& hit) {
    // [ENUTS_OMP] worker threads keep their own pass store (see SyncThr)
    SyncThr* const thr_ =
        ts::EnutsProf::tid() ? &m_uiSyncThr[ts::EnutsProf::tid()] : nullptr;
    unsigned long long& m_uiP2cPoolPass = thr_ ? thr_->p2cPoolPass : this->m_uiP2cPoolPass;
    size_t& m_uiP2cUsed                 = thr_ ? thr_->p2cUsed : this->m_uiP2cUsed;
    std::vector<unsigned long long>& m_uiP2cPassOf = thr_ ? thr_->p2cPassOf : this->m_uiP2cPassOf;
    std::vector<unsigned int>& m_uiP2cSlotOf       = thr_ ? thr_->p2cSlotOf : this->m_uiP2cSlotOf;
    std::vector<SyncTag>& m_uiP2cTagOf             = thr_ ? thr_->p2cTagOf : this->m_uiP2cTagOf;
    std::vector<T>& m_uiP2cPool                    = thr_ ? thr_->p2cPool : this->m_uiP2cPool;
    hit = false;
    if (m_uiP2cSharedOn) {
        // OWN_P2C_SHARED: the pre-pass filled one shared pool; read-only here
        if (this->m_uiP2cPassOf[idx] == m_uiSyncPass && this->m_uiP2cTagOf[idx] == key) {
            hit = true;
            if (this->m_uiP2cSlotOf[idx] == ~0u) return nullptr;  // written into the blocks already
            return this->m_uiP2cPool.data() + (size_t)this->m_uiP2cSlotOf[idx] * stride;
        }
        if (ts::EnutsProf::on()) ts::EnutsProf::get().t[ts::EnutsProf::CN_P2C_OVF] += 1;
        return nullptr;  // past the shared cap: computed, not stored
    }
    if (m_uiP2cPoolPass != m_uiSyncPass) {
        m_uiP2cPoolPass = m_uiSyncPass;
        m_uiP2cUsed     = 0;
    }
    if (m_uiP2cPassOf[idx] == m_uiSyncPass) {
        // same (element, child) seen in this pass: same data unless the
        // source differs (its tag)
        if (m_uiP2cTagOf[idx] == key)
            hit = true;
        else
            m_uiP2cTagOf[idx] = key;
        return m_uiP2cPool.data() + (size_t)m_uiP2cSlotOf[idx] * stride;
    }
    const size_t cap =
        std::max<size_t>(64, ts::EnutsProf::p2cStoreBytes() / (stride * sizeof(T)));
    if (m_uiP2cUsed >= cap) {
        if (ts::EnutsProf::on()) ts::EnutsProf::get().t[ts::EnutsProf::CN_P2C_OVF] += 1;
        return nullptr;
    }
    if ((m_uiP2cUsed + 1) * stride > m_uiP2cPool.size())
        m_uiP2cPool.resize(std::min(
            cap * stride,
            std::max<size_t>(2 * m_uiP2cPool.size(), 64 * stride)));
    m_uiP2cSlotOf[idx] = (unsigned int)m_uiP2cUsed++;
    m_uiP2cPassOf[idx] = m_uiSyncPass;
    m_uiP2cTagOf[idx]  = key;
    return m_uiP2cPool.data() + (size_t)m_uiP2cSlotOf[idx] * stride;
}

template <typename T, typename Ctx>
bool ExplicitNUTS<T, Ctx>::prepare_owner_sync(ot::Mesh* pMesh) {
    if (m_uiOwnMesh == pMesh) return m_uiOwnOk;
    m_uiOwnPlan.clear();
    m_uiOwnRowBufRow = ~0u;
    m_uiOwnOk        = false;
    m_uiOwnMesh      = pMesh;
    if (!pMesh->isActive()) return false;

    const unsigned int nPe  = pMesh->getNumNodesPerElement();
    const unsigned int eO   = pMesh->getElementOrder();
    const unsigned int eo1  = eO + 1;
    const unsigned int nEle = pMesh->getAllElements().size();
    const unsigned int cgSz = pMesh->getDegOfFreedom();
    const int npes          = pMesh->getMPICommSize();
    const ot::TreeNode* pNodes          = pMesh->getAllElements().data();
    const unsigned int* e2n             = pMesh->getE2NMapping().data();
    const unsigned int* e2nDG           = pMesh->getE2NMapping_DG().data();
    const std::vector<unsigned int>& c2d = pMesh->getCG2DGMap();

    // DENDRO_ENUTS_OWNER_SYNC_PROBE=1 (default off): on a labelled mesh run
    // every consistency check of the plan, print what breaks and how often,
    // and fall back to the per-element sync instead of aborting (diagnostic
    // for building the plan on labelled meshes; the values are unchanged)
    static const bool probeOn = [] {
        const char* e = std::getenv("DENDRO_ENUTS_OWNER_SYNC_PROBE");
        return e && e[0] == '1';
    }();
    const bool labelled = pMesh->useLabelledUnzip();
    // OWNER_SYNC_LABELLED: like the probe, every check runs and records; the
    // outcome is then agreed collectively (the per-element fallback must be
    // taken by every rank or none: the two paths use different exchanges)
    const bool lab   = labelled && !probeOn && ts::EnutsProf::ownerSyncLabelled();
    const bool probe = (probeOn && labelled) || lab;
    bool probeBad    = false;
    // one rank: fall back to the per-element form; more: there is no fallback
    auto fail = [&](const char* why, size_t n) -> bool {
        std::cout << "[enuts-owner-sync r" << pMesh->getMPIRankGlobal()
                  << "] plan unavailable: " << why << " (" << n << ")"
                  << (probe ? "; probe" : npes > 1 ? "; aborting" : "; per-element owner sync")
                  << std::endl;
        if (probe) {
            probeBad = true;
            return false;
        }
        if (npes > 1) MPI_Abort(pMesh->getMPICommunicator(), 11);
        return false;
    };
    if (labelled && !probe) return fail("labelled mesh", 0);
    if (lab && !pMesh->canonKeysBuilt()) fail("labelled mesh without canonical ownership keys", 0);
    if (lab) {
        // the plan's model needs one owner slot per node and cg-indexed sends
        pMesh->ensureZipPlanBuiltPublic();
        const size_t nonPrim = pMesh->getLocalNonPrimaryNodes().size();
        if (nonPrim) fail("non-primary duplicate cg slots", nonPrim);
        size_t dgTag = 0;
        for (const unsigned char f : pMesh->getScatterMapSendIsDG()) dgTag += f;
        if (dgTag) fail("DG-tagged send-map entries", dgTag);
        size_t notInc = 0;
        for (size_t n = 1; n < c2d.size(); n++) notInc += (c2d[n] <= c2d[n - 1]);
        if (notInc) fail("cg2dg not strictly increasing", notInc);
    }
    if (c2d.size() < cgSz) return fail("cg2dg map smaller than the cg vector", c2d.size());
    if (eo1 * eo1 > 65535u) return fail("element order", eO);

    // the plan reads a node through cg2dg; it must be the owner slot the
    // element-wise zip uses (e2n_dg), for every element
    {
        size_t bad = 0;
        for (size_t i = 0; i < (size_t)nEle * nPe; i++)
            if (e2n[i] >= cgSz || c2d[e2n[i]] != e2nDG[i]) bad++;
        if (bad && !probe) return fail("e2n_dg is not cg2dg o e2n", bad);
        if (bad) fail("e2n_dg is not cg2dg o e2n", bad);
    }

    m_uiOwnDTs.clear();
    for (unsigned int l = m_uiLevMin; l <= m_uiLevMax; l++)
        m_uiOwnDTs.push_back(
            step_ratio(l));
    std::sort(m_uiOwnDTs.begin(), m_uiOwnDTs.end());
    m_uiOwnDTs.erase(std::unique(m_uiOwnDTs.begin(), m_uiOwnDTs.end()),
                     m_uiOwnDTs.end());
    const unsigned int nCls = m_uiOwnDTs.size();
    if (nCls == 0 || nCls > 250 || m_uiOwnDTs[0] < 1)
        return fail("time-step classes", nCls);
    m_uiOwnChain = true;
    for (unsigned int i = 1; i < nCls; i++)
        if (m_uiOwnDTs[i] % m_uiOwnDTs[i - 1] != 0) m_uiOwnChain = false;

    auto clsOfEle = [&](unsigned int e) -> unsigned int {
        const unsigned int l = pNodes[e].getLevel();
        if (e >= nEle || l < m_uiLevMin || l > m_uiLevMax) return 255u;
        const unsigned int dt =
            step_ratio(l);
        return (unsigned int)(std::lower_bound(m_uiOwnDTs.begin(),
                                               m_uiOwnDTs.end(), dt) -
                              m_uiOwnDTs.begin());
    };
    m_uiNodeCls.assign(cgSz, 255);
    for (unsigned int n = 0; n < cgSz; n++)
        if (c2d[n] / nPe < nEle) m_uiNodeCls[n] = (unsigned char)clsOfEle(c2d[n] / nPe);

    m_uiOwnNodeRecv.clear();
    m_uiOwnSendSlot.clear();
    m_uiOwnRecvSlot.clear();
    m_uiOwnSendOff.clear();
    m_uiOwnRecvOff.clear();
    m_uiOwnSendCum.clear();
    m_uiOwnRecvCum.clear();
    if (npes > 1) {
        const unsigned int nb = pMesh->getNodeLocalBegin();
        const unsigned int ne = pMesh->getNodeLocalEnd();
        const unsigned int eb = pMesh->getElementLocalBegin();
        const unsigned int ee = pMesh->getElementLocalEnd();
        // a node owned by this rank has its owner element here: its stage
        // value is local data, sent to the ranks that hold it as a ghost
        size_t badOwn = 0;
        std::vector<T> tmp(cgSz, (T)-1);
        for (unsigned int n = nb; n < ne; n++) {
            const unsigned int oe = c2d[n] / nPe;
            if (oe < eb || oe >= ee || m_uiNodeCls[n] == 255) badOwn++;
            tmp[n] = (T)m_uiNodeCls[n];
        }
        pMesh->readFromGhostBegin(tmp.data(), 1);
        pMesh->readFromGhostEnd(tmp.data(), 1);

        const std::vector<unsigned int>& sOff = pMesh->getNodalSendOffsets();
        const std::vector<unsigned int>& sCnt = pMesh->getNodalSendCounts();
        const std::vector<unsigned int>& rOff = pMesh->getNodalRecvOffsets();
        const std::vector<unsigned int>& rCnt = pMesh->getNodalRecvCounts();
        const std::vector<unsigned int>& sPr  = pMesh->getSendProcList();
        const std::vector<unsigned int>& rPr  = pMesh->getRecvProcList();
        const std::vector<unsigned int>& sSM  = pMesh->getSendNodeSM();
        const std::vector<unsigned int>& rSM  = pMesh->getRecvNodeSM();

        m_uiOwnNodeRecv.assign(cgSz, 0);
        size_t badSend = 0, badRecv = 0, badCls = 0, clsDiff = 0;
        std::vector<unsigned int> ord;
        // per neighbor: nodes in scatter-map order, stably sorted by class
        // (both sides sort the same sequence of classes)
        auto build = [&](const std::vector<unsigned int>& pr,
                         const std::vector<unsigned int>& off,
                         const std::vector<unsigned int>& cnt,
                         const std::vector<unsigned int>& sm, bool recv,
                         std::vector<unsigned int>& slot,
                         std::vector<unsigned int>& poff,
                         std::vector<unsigned int>& cum) {
            poff.assign(pr.size() + 1, 0);
            cum.assign(pr.size() * nCls, 0);
            for (size_t pi = 0; pi < pr.size(); pi++) {
                const unsigned int p = pr[pi];
                ord.resize(cnt[p]);
                for (unsigned int k = 0; k < cnt[p]; k++) ord[k] = sm[off[p] + k];
                for (unsigned int k = 0; k < cnt[p]; k++) {
                    const unsigned int n = ord[k];
                    if (recv) {
                        if (n >= cgSz || (n >= nb && n < ne)) {
                            badRecv++;
                            continue;
                        }
                        const T c = tmp[n];
                        if (!(c >= 0 && c < (T)nCls)) {
                            badCls++;
                            continue;
                        }
                        if (m_uiNodeCls[n] != (unsigned char)c) clsDiff++;
                        m_uiNodeCls[n]     = (unsigned char)c;
                        m_uiOwnNodeRecv[n] = 1;
                        const unsigned int oe = c2d[n] / nPe;
                        // the slot that keeps the owner's value must not be
                        // one a local block writes
                        if (oe >= eb && oe < ee) badRecv++;
                    } else if (n < nb || n >= ne) {
                        badSend++;
                    }
                }
                if (badRecv || badSend || badCls) continue;
                std::stable_sort(ord.begin(), ord.end(),
                                 [&](unsigned int a, unsigned int b) {
                                     return m_uiNodeCls[a] < m_uiNodeCls[b];
                                 });
                for (unsigned int k = 0; k < cnt[p]; k++) {
                    slot.push_back(c2d[ord[k]]);
                    cum[pi * nCls + m_uiNodeCls[ord[k]]]++;
                }
                for (unsigned int c = 1; c < nCls; c++)
                    cum[pi * nCls + c] += cum[pi * nCls + c - 1];
                poff[pi + 1] = (unsigned int)slot.size();
            }
        };
        build(sPr, sOff, sCnt, sSM, false, m_uiOwnSendSlot, m_uiOwnSendOff,
              m_uiOwnSendCum);
        build(rPr, rOff, rCnt, rSM, true, m_uiOwnRecvSlot, m_uiOwnRecvOff,
              m_uiOwnRecvCum);
        if (badOwn && !probe) return fail("local nodes whose owner element is not local", badOwn);
        if (badSend && !probe) return fail("send-map entries that are not local cg nodes", badSend);
        if (badRecv && !probe) return fail("ghost nodes kept in a local element's slot", badRecv);
        if (badCls && !probe) return fail("ghost nodes without an owner class", badCls);
        if (probe) {
            if (badOwn) fail("local nodes whose owner element is not local", badOwn);
            if (badSend) fail("send-map entries that are not local cg nodes", badSend);
            if (badRecv) fail("ghost nodes kept in a local element's slot", badRecv);
            if (badCls) fail("ghost nodes without an owner class", badCls);
            if (clsDiff) fail("ghost node class differs from the owner's", clsDiff);
        }
        if (lab) {
            int okl = probeBad ? 0 : 1, okg = 0;
            MPI_Allreduce(&okl, &okg, 1, MPI_INT, MPI_MIN, pMesh->getMPICommunicator());
            if (!okg) {
                if (!pMesh->getMPIRank())
                    std::cout << "[enuts-owner-sync] labelled mesh: plan not buildable on some rank; "
                                 "per-element sync on every rank" << std::endl;
                m_uiOwnPlan.clear();
                return false;
            }
        } else if (probe && !lab) {
            std::cout << "[enuts-owner-sync r" << pMesh->getMPIRankGlobal()
                      << "] probe: labelled mesh, elems " << nEle << " local "
                      << (ee - eb) << " cg " << cgSz << " local " << (ne - nb)
                      << " nonprimary " << pMesh->getLocalNonPrimaryNodes().size()
                      << " clsDiff " << clsDiff << " classes " << nCls
                      << (probeBad ? " -> plan NOT buildable" : " -> all checks pass")
                      << std::endl;
            m_uiOwnPlan.clear();
            return false;  // per-element sync for this run
        }
        if (lab && npes == 1 && probeBad) {
            m_uiOwnPlan.clear();
            return false;
        }
        if (std::getenv("DENDRO_ENUTS_OWNER_SYNC_INFO"))
            std::cout << "[enuts-owner-sync r" << pMesh->getMPIRankGlobal()
                      << "] classes " << nCls << " chain " << m_uiOwnChain
                      << " send nodes " << m_uiOwnSendSlot.size()
                      << " recv nodes " << m_uiOwnRecvSlot.size()
                      << " ghost nodes whose local owner had another class "
                      << clsDiff << std::endl;
    }

    // hanging face / edge interpolations per element, in the order of
    // Mesh::getElementNodalValues (same construction as Mesh::buildUnzipPlan)
    static const unsigned int faceAx[6] = {0, 0, 1, 1, 2, 2};
    static const unsigned int faceHi[6] = {0, 1, 0, 1, 0, 1};
    static const unsigned int faceDir[6] = {OCT_DIR_LEFT, OCT_DIR_RIGHT,
                                            OCT_DIR_DOWN, OCT_DIR_UP,
                                            OCT_DIR_BACK, OCT_DIR_FRONT};
    static const unsigned int edgeFree[12] = {2, 2, 1, 1, 2, 2, 1, 1, 0, 0, 0, 0};
    static const unsigned int edgeF[12][2] = {
        {0, 2}, {0, 3}, {0, 4}, {0, 5}, {1, 2}, {1, 3},
        {1, 4}, {1, 5}, {2, 4}, {2, 5}, {3, 4}, {3, 5}};
    static const unsigned int edgeDir[12] = {
        OCT_DIR_LEFT_DOWN,  OCT_DIR_LEFT_UP,     OCT_DIR_LEFT_BACK,
        OCT_DIR_LEFT_FRONT, OCT_DIR_RIGHT_DOWN,  OCT_DIR_RIGHT_UP,
        OCT_DIR_RIGHT_BACK, OCT_DIR_RIGHT_FRONT, OCT_DIR_DOWN_BACK,
        OCT_DIR_DOWN_FRONT, OCT_DIR_UP_BACK,     OCT_DIR_UP_FRONT};
    m_uiOwnSlots.clear();
    auto slot = [eo1](const unsigned int* x) {
        return x[2] * eo1 * eo1 + x[1] * eo1 + x[0];
    };
    for (unsigned int f = 0; f < 6; f++) {
        const unsigned int a  = faceAx[f];
        const unsigned int lo = (a == 0) ? 1 : 0, hi = (a == 2) ? 1 : 2;
        unsigned int x[3];
        x[a] = faceHi[f] * eO;
        for (x[hi] = 0; x[hi] < eo1; x[hi]++)
            for (x[lo] = 0; x[lo] < eo1; x[lo]++) m_uiOwnSlots.push_back(slot(x));
    }
    for (unsigned int e = 0; e < 12; e++) {
        unsigned int x[3];
        for (unsigned int s = 0; s < 2; s++)
            x[faceAx[edgeF[e][s]]] = faceHi[edgeF[e][s]] * eO;
        const unsigned int a = edgeFree[e];
        for (x[a] = 0; x[a] < eo1; x[a]++) m_uiOwnSlots.push_back(slot(x));
    }
    m_uiOwnOpOff.assign(nEle + 1, 0);
    m_uiOwnOps.clear();
    unsigned int cnum;
    for (unsigned int ele = 0; ele < nEle; ele++) {
        m_uiOwnOpOff[ele] = (unsigned int)m_uiOwnOps.size();
        bool faceHang[6];
        unsigned char cornerSet = 0;
        for (unsigned int f = 0; f < 6; f++) {
            faceHang[f] = pMesh->isFaceHanging(ele, faceDir[f], cnum);
            if (!faceHang[f]) continue;
            m_uiOwnOps.push_back((unsigned char)(f | (cnum << 5)));
            for (unsigned int c = 0; c < 8; c++)
                if (((c >> faceAx[f]) & 1u) == faceHi[f]) cornerSet |= 1u << c;
        }
        for (unsigned int e = 0; e < 12; e++) {
            if (faceHang[edgeF[e][0]] || faceHang[edgeF[e][1]]) continue;
            if (!pMesh->isEdgeHanging(ele, edgeDir[e], cnum)) continue;
            m_uiOwnOps.push_back((unsigned char)((6 + e) | (cnum << 5)));
            for (unsigned int c = 0; c < 8; c++) {
                bool on = true;
                for (unsigned int s = 0; s < 2; s++)
                    on = on && (((c >> faceAx[edgeF[e][s]]) & 1u) ==
                                faceHi[edgeF[e][s]]);
                if (on) cornerSet |= 1u << c;
            }
        }
        // an interpolated corner that is not hanging is gathered again
        for (unsigned int c = 0; c < 8; c++)
            if (((cornerSet >> c) & 1u) &&
                !pMesh->isNodeHanging(ele, (c & 1u) * eO, ((c >> 1) & 1u) * eO,
                                      ((c >> 2) & 1u) * eO))
                m_uiOwnOps.push_back((unsigned char)(18 + c));
    }
    m_uiOwnOpOff[nEle] = (unsigned int)m_uiOwnOps.size();

    m_uiOwnRowOf.assign(nEle, -1);
    m_uiOwnRowEle.clear();
    m_uiOwnRowSrc.clear();
    m_uiOwnRowN.clear();
    m_uiOwnRowOff.clear();
    m_uiOwnRowPc.clear();
    m_uiOwnConv.assign((size_t)nCls * nCls, OwnConv());
    m_uiOwnConvPass.assign(nCls, 0);
    m_uiOwnScr.assign((size_t)5 * nPe, 0);

    if (ts::EnutsProf::ownerSyncCheck()) {
        // recipe == Mesh::getElementNodalValues, bit for bit, on arbitrary data
        std::vector<T> cg(cgSz), g(nPe), a(nPe), b(nPe), pin(eo1 * eo1), pout(eo1 * eo1);
        std::vector<double> im1(nPe), im2(nPe);
        unsigned long long x = 88172645463325252ull;
        for (unsigned int n = 0; n < cgSz; n++) {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            cg[n] = (T)((double)(x >> 11) / 9007199254740992.0 - 0.5);
        }
        size_t bad = 0, hang = 0;
        for (unsigned int ele = 0; ele < nEle; ele++) {
            for (unsigned int n = 0; n < nPe; n++) g[n] = cg[e2n[(size_t)ele * nPe + n]];
            own_row_ops(ele, g.data(), a.data(), pin.data(), pout.data(), im1.data(), im2.data());
            pMesh->getElementNodalValues(cg.data(), b.data(), ele);
            hang += (m_uiOwnOpOff[ele + 1] > m_uiOwnOpOff[ele]);
            if (std::memcmp(a.data(), b.data(), sizeof(T) * nPe) != 0) bad++;
        }
        std::cout << "[enuts-owner-sync-check r" << pMesh->getMPIRankGlobal()
                  << "] elements " << nEle << " with hanging ops " << hang
                  << " recipe mismatches " << bad << std::endl;
        if (bad) MPI_Abort(pMesh->getMPICommunicator(), 12);
    }
    if (ts::EnutsProf::xchCensus()) {
        const size_t dgSz = (size_t)nEle * nPe;
        m_uiXchDg         = dgSz;
        m_uiXchFlag.assign((size_t)m_uiNumStages * dgSz, 0);
        m_uiXchInPlan.assign(dgSz, 0);
    }
    if (ts::EnutsProf::xchTrace() && npes > 1) xch_calibrate(pMesh);
    m_uiOwnOk = true;
    return true;
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::own_row_ops(unsigned int ele, const T* g, T* out,
                                       T* pin, T* pout, double* im1,
                                       double* im2) const {
    const ot::Mesh* pMesh  = m_uiAppCtx->get_mesh();
    const unsigned int nPe = pMesh->getNumNodesPerElement();
    const unsigned int eO  = pMesh->getElementOrder();
    const unsigned int eo1 = eO + 1;
    if (out != g) std::memcpy(out, g, sizeof(T) * nPe);
    for (unsigned int o = m_uiOwnOpOff[ele]; o < m_uiOwnOpOff[ele + 1]; o++) {
        const unsigned int id = m_uiOwnOps[o] & 31u;
        if (id >= 18) {
            const unsigned int c = id - 18;
            const unsigned int n = (c >> 2) * eO * eo1 * eo1 +
                                   ((c >> 1) & 1u) * eO * eo1 + (c & 1u) * eO;
            out[n] = g[n];
            continue;
        }
        const bool face        = id < 6;
        const unsigned int len = face ? eo1 * eo1 : eo1;
        const unsigned int* s  = m_uiOwnSlots.data() +
                                (face ? id * eo1 * eo1 : 6 * eo1 * eo1 + (id - 6) * eo1);
        for (unsigned int t = 0; t < len; t++) pin[t] = g[s[t]];
        pMesh->parent2ChildInterpolation(pin, pout, m_uiOwnOps[o] >> 5,
                                         face ? 2 : 1, im1, im2);
        for (unsigned int t = 0; t < len; t++) out[s[t]] = pout[t];
    }
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::build_own_plan(unsigned int blk,
                                          const std::vector<unsigned int>& eid,
                                          OwnPlan& op) {
    ot::Mesh* pMesh            = (ot::Mesh*)m_uiAppCtx->get_mesh();
    const ot::TreeNode* pNodes = pMesh->getAllElements().data();
    const ot::Block* blkList   = pMesh->getLocalBlockList().data();
    const ot::TreeNode blkNode = blkList[blk].getBlockNode();
    const unsigned int regLevel = blkList[blk].getRegularGridLev();
    const unsigned int PW      = blkList[blk].get1DPadWidth();
    const unsigned int eOrder  = pMesh->getElementOrder();
    const unsigned int n1      = eOrder + 1;
    const unsigned int nPe     = pMesh->getNumNodesPerElement();
    const unsigned int lx      = blkList[blk].getAllocationSzX();
    const unsigned int ly      = blkList[blk].getAllocationSzY();
    const unsigned int lz      = blkList[blk].getAllocationSzZ();
    const size_t lxyz          = (size_t)lx * ly * lz;
    const unsigned int cgSz    = pMesh->getDegOfFreedom();
    const unsigned int nCls    = m_uiOwnDTs.size();
    const unsigned int* e2n    = pMesh->getE2NMapping().data();
    const std::vector<unsigned int>& c2d = pMesh->getCG2DGMap();
    const bool par             = pMesh->getMPICommSize() > 1;
    const unsigned int nb      = pMesh->getNodeLocalBegin();
    const unsigned int ne      = pMesh->getNodeLocalEnd();

    if (m_uiSyncPlan.size() != pMesh->getLocalBlockList().size())
        m_uiSyncPlan.assign(pMesh->getLocalBlockList().size(), SyncPlan());
    SyncPlan& pl = m_uiSyncPlan[blk];
    if (!pl.built) build_sync_plan(blk, eid, pl);

    // last writer of every block slot: the padding fill in plan order, then
    // the block's own elements
    const unsigned int NONE = ~0u;
    std::vector<unsigned int> wEle(lxyz, NONE), wNode(lxyz, 0);
    std::vector<signed char> wC(lxyz, -1);
    std::vector<unsigned char> wOwn(lxyz, 0);  // slot of one of the block's own elements
    for (const typename SyncPlan::Ent& en : pl.ents)
        for (unsigned int q = en.b; q < en.e; q++) {
            const unsigned int d = pl.dst[q];
            wEle[d]  = en.ele;
            wC[d]    = (signed char)en.cnum;
            wNode[d] = pl.src[q];
        }
    for (const unsigned int elem : blkList[blk]) {
        const unsigned int ei = (pNodes[elem].getX() - blkNode.getX()) >> (m_uiMaxDepth - regLevel);
        const unsigned int ej = (pNodes[elem].getY() - blkNode.getY()) >> (m_uiMaxDepth - regLevel);
        const unsigned int ek = (pNodes[elem].getZ() - blkNode.getZ()) >> (m_uiMaxDepth - regLevel);
        for (unsigned int k = 0; k < n1; k++)
            for (unsigned int j = 0; j < n1; j++)
                for (unsigned int i = 0; i < n1; i++) {
                    const size_t d = (size_t)(ek * eOrder + k + PW) * (ly * lx) +
                                     (size_t)(ej * eOrder + j + PW) * lx +
                                     (ei * eOrder + i + PW);
                    wEle[d]  = elem;
                    wC[d]    = -1;
                    wNode[d] = k * n1 * n1 + j * n1 + i;
                    wOwn[d]  = 1;
                }
    }
    const bool noSelf = ts::EnutsProf::ownNoSelfCopy();
    size_t nSelf      = 0;  // self-copy slots dropped (reported with OWNER_SYNC_INFO)

    size_t uncovered = 0, badNode = 0;
    // class of a source node; it must be a value this rank has
    auto need = [&](unsigned int cg) -> unsigned int {
        if (cg >= cgSz || m_uiNodeCls[cg] >= nCls) {
            badNode++;
            return 0;
        }
        if (par && (cg < nb || cg >= ne) && !m_uiOwnNodeRecv[cg]) uncovered++;
        return m_uiNodeCls[cg];
    };

    std::vector<std::vector<unsigned int>> bd(nCls), bs(nCls);
    struct Inst {
        unsigned int ele, o;
        std::vector<std::pair<unsigned short, unsigned int>> out;
    };
    std::vector<Inst> opI, chI;
    std::unordered_map<unsigned long long, unsigned int> opMap, chMap;
    // per element with hanging ops: for each node the op that writes it last
    // (-1: the gathered value) and its index in that op's slot list
    std::unordered_map<unsigned int, std::vector<int>> lastOp;
    auto lastOpOf = [&](unsigned int e) -> const std::vector<int>& {
        auto it = lastOp.find(e);
        if (it != lastOp.end()) return it->second;
        std::vector<int>& lo = lastOp[e];
        lo.assign(2 * nPe, -1);
        for (unsigned int o = m_uiOwnOpOff[e]; o < m_uiOwnOpOff[e + 1]; o++) {
            const unsigned int id = m_uiOwnOps[o] & 31u;
            if (id >= 18) {
                const unsigned int c = id - 18;
                lo[2 * ((c >> 2) * eOrder * n1 * n1 + ((c >> 1) & 1u) * eOrder * n1 +
                        (c & 1u) * eOrder)] = -1;
                continue;
            }
            const unsigned int len = (id < 6) ? n1 * n1 : n1;
            const unsigned int* s  = m_uiOwnSlots.data() +
                                    ((id < 6) ? id * n1 * n1 : 6 * n1 * n1 + (id - 6) * n1);
            for (unsigned int t = 0; t < len; t++) {
                lo[2 * s[t]]     = (int)(o - m_uiOwnOpOff[e]);
                lo[2 * s[t] + 1] = (int)t;
            }
        }
        return lo;
    };

    for (size_t d = 0; d < lxyz; d++) {
        const unsigned int e = wEle[d];
        if (e == NONE) continue;
        const unsigned int n = wNode[d];
        if (wC[d] >= 0) {
            const unsigned long long key = (unsigned long long)e * NUM_CHILDREN + wC[d];
            auto it = chMap.find(key);
            if (it == chMap.end()) {
                it = chMap.emplace(key, (unsigned int)chI.size()).first;
                chI.push_back({e, (unsigned int)wC[d], {}});
            }
            chI[it->second].out.emplace_back((unsigned short)n, (unsigned int)d);
            continue;
        }
        int o = -1, t = 0;
        if (m_uiOwnOpOff[e + 1] > m_uiOwnOpOff[e]) {
            const std::vector<int>& lo = lastOpOf(e);
            o = lo[2 * n];
            t = lo[2 * n + 1];
        }
        if (o < 0) {
            const unsigned int cg = e2n[(size_t)e * nPe + n];
            // OWN_NO_SELFCOPY: the element owns this node, so the source is
            // its own DG row = this block's value (zipped from it): skip
            if (noSelf && wOwn[d] && cg < cgSz && c2d[cg] == (size_t)e * nPe + n) {
                nSelf++;
                continue;
            }
            const unsigned int c  = need(cg);
            bd[c].push_back((unsigned int)d);
            bs[c].push_back(cg < cgSz ? c2d[cg] : 0);
        } else {
            const unsigned long long key = (unsigned long long)e * 32 + o;
            auto it = opMap.find(key);
            if (it == opMap.end()) {
                it = opMap.emplace(key, (unsigned int)opI.size()).first;
                opI.push_back({e, (unsigned int)o, {}});
            }
            opI[it->second].out.emplace_back((unsigned short)t, (unsigned int)d);
        }
    }

    op = OwnPlan();
    op.clsOff.assign(nCls + 1, 0);
    op.runOff.assign(nCls + 1, 0);
    op.celeOff.assign(nCls + 1, 0);
    // an owner element that feeds at least this many slots of the block is
    // converted whole (shared between blocks of a pass), fewer: per node
    const unsigned int wholeMin = std::max(8u, nPe / 6);
    std::unordered_map<unsigned int, unsigned int> cnt;
    for (unsigned int c = 0; c < nCls; c++) {
        cnt.clear();
        for (const unsigned int sl : bs[c]) cnt[sl / nPe]++;
        for (const auto& kv : cnt)
            if (kv.second >= wholeMin) op.cele.push_back(kv.first);
        std::sort(op.cele.begin() + op.celeOff[c], op.cele.end());
        op.celeOff[c + 1] = (unsigned int)op.cele.size();
        bool open = false;
        for (size_t q = 0; q < bd[c].size(); q++) {
            if (cnt[bs[c][q] / nPe] < wholeMin) {
                op.pdst.push_back(bd[c][q]);
                op.psrc.push_back(bs[c][q]);
                continue;
            }
            if (open && op.rlen.back() < 65535 &&
                op.rdst.back() + op.rlen.back() == bd[c][q] &&
                op.rsrc.back() + op.rlen.back() == bs[c][q] &&
                op.rsrc.back() / nPe == bs[c][q] / nPe) {
                op.rlen.back()++;
            } else {
                op.rdst.push_back(bd[c][q]);
                op.rsrc.push_back(bs[c][q]);
                op.rlen.push_back(1);
                open = true;
            }
        }
        op.clsOff[c + 1] = (unsigned int)op.pdst.size();
        op.runOff[c + 1] = (unsigned int)op.rdst.size();
        if (c + 1 == nCls && nSelf && std::getenv("DENDRO_ENUTS_OWNER_SYNC_INFO"))
            std::cout << "[enuts-owner-sync] blk " << blk << ": " << nSelf
                      << " self-copy slots dropped" << std::endl;
    }
    for (const Inst& in : opI) {
        const unsigned char code = m_uiOwnOps[m_uiOwnOpOff[in.ele] + in.o];
        const unsigned int id    = code & 31u;
        const unsigned int len   = (id < 6) ? n1 * n1 : n1;
        const unsigned int* s    = m_uiOwnSlots.data() +
                                  ((id < 6) ? id * n1 * n1 : 6 * n1 * n1 + (id - 6) * n1);
        typename OwnPlan::Op w;
        w.inB  = (unsigned int)op.opSrc.size();
        w.outB = (unsigned int)op.oDst.size();
        w.dim  = (id < 6) ? 2 : 1;
        w.cnum = code >> 5;
        w.segB = (unsigned int)op.opSegCls.size();
        {
            std::vector<unsigned char> ic(len);
            for (unsigned int t = 0; t < len; t++)
                ic[t] = (unsigned char)need(e2n[(size_t)in.ele * nPe + s[t]]);
            for (unsigned int c = 0; c < nCls; c++) {
                unsigned int k = 0;
                for (unsigned int t = 0; t < len; t++) {
                    if (ic[t] != c) continue;
                    const unsigned int cg = e2n[(size_t)in.ele * nPe + s[t]];
                    op.opT.push_back(t);
                    op.opSrc.push_back(cg < cgSz ? c2d[cg] : 0);
                    k++;
                }
                if (!k) continue;
                op.opSegCls.push_back((unsigned char)c);
                op.opSegEnd.push_back((unsigned short)(op.opSrc.size() - w.inB));
            }
        }
        w.segE = (unsigned int)op.opSegCls.size();
        for (const auto& pr : in.out) {
            op.oT.push_back(pr.first);
            op.oDst.push_back(pr.second);
        }
        w.outE = (unsigned int)op.oDst.size();
        op.ops.push_back(w);
    }
    // children of one parent together: its reconciled row is built once
    std::sort(chI.begin(), chI.end(), [](const Inst& a, const Inst& b) {
        return a.ele != b.ele ? a.ele < b.ele : a.o < b.o;
    });
    for (const Inst& in : chI) {
        if (m_uiOwnRowOf[in.ele] < 0) {
            m_uiOwnRowOf[in.ele] = (int)m_uiOwnRowEle.size();
            m_uiOwnRowEle.push_back(in.ele);
            std::vector<unsigned char> rc(nPe);
            for (unsigned int n = 0; n < nPe; n++)
                rc[n] = (unsigned char)need(e2n[(size_t)in.ele * nPe + n]);
            {
                const unsigned int dt = step_ratio(pNodes[in.ele].getLevel());
                m_uiOwnRowPc.push_back((unsigned char)(
                    std::lower_bound(m_uiOwnDTs.begin(), m_uiOwnDTs.end(), dt) -
                    m_uiOwnDTs.begin()));
            }
            for (unsigned int c = 0; c < nCls; c++) {
                m_uiOwnRowOff.push_back((unsigned int)m_uiOwnRowN.size());
                for (unsigned int n = 0; n < nPe; n++) {
                    if (rc[n] != c) continue;
                    const unsigned int cg = e2n[(size_t)in.ele * nPe + n];
                    // its own node: part of the element's DG row
                    if (cg < cgSz && c2d[cg] == in.ele * nPe + n &&
                        c == m_uiOwnRowPc.back())
                        continue;
                    m_uiOwnRowN.push_back(n);
                    m_uiOwnRowSrc.push_back(cg < cgSz ? c2d[cg] : 0);
                }
            }
            m_uiOwnRowOff.push_back((unsigned int)m_uiOwnRowN.size());
        }
        typename OwnPlan::Child w;
        w.row   = (unsigned int)m_uiOwnRowOf[in.ele];
        w.store = p2c_index(in.ele, in.o);
        w.cnum  = (unsigned char)in.o;
        w.outB  = (unsigned int)op.cDst.size();
        for (const auto& pr : in.out) {
            if (op.cDst.size() > w.outB && op.cLen.back() < 65535 &&
                op.cT.back() + op.cLen.back() == pr.first &&
                op.cDst.back() + op.cLen.back() == pr.second) {
                op.cLen.back()++;
            } else {
                op.cT.push_back(pr.first);
                op.cDst.push_back(pr.second);
                op.cLen.push_back(1);
            }
        }
        w.outE = (unsigned int)op.cDst.size();
        op.ch.push_back(w);
    }
    if (badNode || uncovered) {
        std::cout << "[enuts-owner-sync r" << pMesh->getMPIRankGlobal()
                  << "] blk " << blk << ": source nodes without a cg index or "
                  << "class " << badNode << ", ghost nodes no rank sends "
                  << uncovered << "; aborting" << std::endl;
        MPI_Abort(pMesh->getMPICommunicator(), 13);
    }
    if (ts::EnutsProf::xchCensus() && !m_uiXchInPlan.empty()) {
        unsigned char* ip = m_uiXchInPlan.data();
        const unsigned int bDT = m_uiEleDT[pMesh->getLocalBlockList()[blk].firstElementID()];
        const unsigned int bc  = (unsigned int)(
            std::lower_bound(m_uiOwnDTs.begin(), m_uiOwnDTs.end(), bDT) - m_uiOwnDTs.begin());
        for (const unsigned int e : op.cele)
            if (xch_whole_row(e, bc)) std::memset(ip + (size_t)e * nPe, 1, nPe);
        for (size_t r = 0; r < op.rsrc.size(); r++)
            std::memset(ip + op.rsrc[r], 1, op.rlen[r]);
        for (const unsigned int q : op.psrc) ip[q] = 1;
        for (const unsigned int q : op.opSrc) ip[q] = 1;
        for (const typename OwnPlan::Child& w : op.ch) {
            std::memset(ip + (size_t)m_uiOwnRowEle[w.row] * nPe, 1, nPe);
            const unsigned int* ro = m_uiOwnRowOff.data() + (size_t)w.row * (nCls + 1);
            for (unsigned int q = ro[0]; q < ro[nCls]; q++) ip[m_uiOwnRowSrc[q]] = 1;
        }
    }
    op.built = true;
}


template <typename T, typename Ctx>
const typename ExplicitNUTS<T, Ctx>::OwnConv* ExplicitNUTS<T, Ctx>::own_conv_rows(
    unsigned int bc, unsigned int rk_s, unsigned int bTime, unsigned int bDT) {
    SyncThr* const thr_ =
        ts::EnutsProf::tid() ? &m_uiSyncThr[ts::EnutsProf::tid()] : nullptr;
    ENUTSOp* const m_uiECOp           = thr_ ? thr_->op : this->m_uiECOp;
    std::vector<OwnConv>& m_uiOwnConv = thr_ ? thr_->ownConv : this->m_uiOwnConv;
    std::vector<unsigned long long>& m_uiOwnConvPass = thr_ ? thr_->ownConvPass : this->m_uiOwnConvPass;
    const unsigned int nCls = m_uiOwnDTs.size();
    OwnConv* cv = m_uiOwnConv.data() + (size_t)bc * nCls;
    if (m_uiOwnConvPass[bc] != m_uiSyncPass) {
        const bool ifxEq  = (ts::EnutsProf::ifaceFix() & 1);
        const bool ifxAll = (ts::EnutsProf::ifaceFix() & 2);
        for (unsigned int c = 0; c < nCls; c++) {
            const unsigned int dtl = m_uiOwnDTs[c];
            // every level steps at the multiples of its dt
            const unsigned int tl = ((m_uiOwnPt + dtl - 1) / dtl) * dtl;
            OwnConv& w            = cv[c];
            w                     = OwnConv();
            if (bTime == tl && !(ifxEq && dtl != bDT)) continue;
            const T t_blk     = m_uiTimeInfo._m_uiT + bTime * m_uiTimeInfo._m_uiTh;
            const T t_ele     = m_uiTimeInfo._m_uiT + tl * m_uiTimeInfo._m_uiTh;
            const bool eqT    = (tl == bTime);
            const bool eleFut = eqT ? (dtl > bDT) : (tl > bTime);
            const unsigned int nsIn =
                (ifxAll && eleFut && !eqT) ? m_uiNumStages : rk_s;
            T dt_c = (eleFut ? dtl : bDT) * (m_uiTimeInfo._m_uiTh);
            T dt_f = (eleFut ? bDT : dtl) * (m_uiTimeInfo._m_uiTh);
            if (dt_c < dt_f) std::swap(dt_c, dt_f);
            T dt = eqT ? 0 : (eleFut ? (t_ele - t_blk) : (t_blk - t_ele));
            if (ts::EnutsProf::ifaceShift() && eleFut && !eqT)
                dt = (T)(bTime + dtl - tl) * m_uiTimeInfo._m_uiTh;
            w.same     = 0;
            w.ns       = (unsigned char)nsIn;
            if (m_uiStGOn && (eqT || nsIn == m_uiNumStages)) {
                T row[5];
                if (m_uiECOp->stageRowNL(eleFut, row, rk_s, dt_c, dt_f, dt, nsIn)) {
                    w.g       = 1;
                    row[nsIn] = row[4];
                    w.ns      = (unsigned char)(nsIn + 1);
                }
                for (unsigned int k = 0; k < w.ns; k++) w.m[k] = row[k];
            } else
            m_uiECOp->stageRow(eleFut, w.m, rk_s, dt_c, dt_f, dt);
        }
        m_uiOwnConvPass[bc] = m_uiSyncPass;
    }
    return cv;
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::own_row_build(unsigned int row_, unsigned int rk_s,
                                         const OwnConv* cv, T* rowBuf, T* g, T* pin,
                                         T* pout, double* im1, double* im2) {
    SyncThr* const thr_ =
        ts::EnutsProf::tid() ? &m_uiSyncThr[ts::EnutsProf::tid()] : nullptr;
    ENUTSOp* const m_uiECOp = thr_ ? thr_->op : this->m_uiECOp;
    const ot::Mesh* pMesh   = m_uiAppCtx->get_mesh();
    const unsigned int nPe  = pMesh->getNumNodesPerElement();
    const size_t dgSz       = (size_t)pMesh->getAllElements().size() * nPe;
    const unsigned int dof  = m_uiEVar.get_dof();
    const unsigned int nCls = m_uiOwnDTs.size();
    const T* S[8];
    const T* SG[8];
    const T* stG = m_uiStGOn ? m_uiStG.get_vec_ptr() : nullptr;
    struct { unsigned int row; } w = {row_};
    const unsigned int ele  = m_uiOwnRowEle[w.row];
    const unsigned int* ro  = m_uiOwnRowOff.data() + (size_t)w.row * (nCls + 1);
    const bool hasOps       = m_uiOwnOpOff[ele + 1] > m_uiOwnOpOff[ele];
    if (ts::EnutsProf::xchCensus() && !m_uiXchFlag.empty()) {
        xch_mark_ele(ele, nPe, cv[m_uiOwnRowPc[w.row]], rk_s);
        for (unsigned int c = 0; c < nCls; c++)
            for (unsigned int q = ro[c]; q < ro[c + 1]; q++)
                xch_mark(m_uiOwnRowSrc[q], cv[c], rk_s);
    }
    for (unsigned int v = 0; v < dof; v++) {
        for (unsigned int k = 0; k < m_uiNumStages; k++)
            SG[k] = S[k] = m_uiStVec[k].get_vec_ptr() + (size_t)v * dgSz;
        if (stG) SG[rk_s] = stG + (size_t)v * dgSz;
        T* row = rowBuf + (size_t)v * nPe;
        T* gg  = hasOps ? g : row;
        {
            // the element's own DG row in the block's frame, then the
            // nodes other elements own
            const OwnConv& pc = cv[m_uiOwnRowPc[w.row]];
            if (pc.same) {
                std::memcpy(gg, S[rk_s - 1] + (size_t)ele * nPe, sizeof(T) * nPe);
            } else {
                const T* Se[8];
                for (unsigned int k = 0; k < m_uiNumStages; k++)
                    Se[k] = S[k] + (size_t)ele * nPe;
                if (pc.g) Se[pc.ns - 1] = stG + (size_t)v * dgSz + (size_t)ele * nPe;
                if (ts::EnutsProf::on() && pc.g && !v) {
                    ts::EnutsProf::get().t[ts::EnutsProf::CN_GALL] += nPe;
                    if (!gValidCls(m_uiOwnRowPc[w.row])) ts::EnutsProf::get().t[ts::EnutsProf::CN_G0] += nPe;
                }
                m_uiECOp->applyRow(gg, Se, pc.m, pc.ns, nPe);
            }
        }
        for (unsigned int c = 0; c < nCls; c++) {
            const unsigned int n = ro[c + 1] - ro[c];
            if (!n) continue;
            const unsigned int* rn = m_uiOwnRowN.data() + ro[c];
            const unsigned int* rs = m_uiOwnRowSrc.data() + ro[c];
            if (cv[c].same) {
                const T* sv = S[rk_s - 1];
                for (unsigned int q = 0; q < n; q++) gg[rn[q]] = sv[rs[q]];
            } else {
                m_uiECOp->gatherRow(gg, rn, rs, n, cv[c].g ? SG : S, cv[c].m, cv[c].ns);
                if (ts::EnutsProf::on() && cv[c].g && !v) {
                    ts::EnutsProf::get().t[ts::EnutsProf::CN_GALL] += n;
                    if (!gValidCls(c)) ts::EnutsProf::get().t[ts::EnutsProf::CN_G0] += n;
                }
            }
        }
        if (hasOps) {
            const bool rprof = ts::EnutsProf::on();
            const double to0 = rprof ? ts::EnutsProf::now() : 0.0;
            own_row_ops(ele, g, row, pin, pout, im1, im2);
            if (rprof) ts::EnutsProf::get().t[ts::EnutsProf::SY_ROWOPS] += ts::EnutsProf::now() - to0;
        }
    }
    if (ts::EnutsProf::on()) {
        ts::EnutsProf::get().t[ts::EnutsProf::CN_ROWS] += 1;
        ts::EnutsProf::get().t[ts::EnutsProf::CN_ROWS_OPS] += hasOps;
    }
}
template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::own_p2c_prepass(unsigned int rk_s, const unsigned int* actIds,
                                           unsigned int nAct, bool thr) {
    ot::Mesh* pMesh          = (ot::Mesh*)m_uiAppCtx->get_mesh();
    const ot::Block* blkList = pMesh->getLocalBlockList().data();
    const unsigned int nPe   = pMesh->getNumNodesPerElement();
    const unsigned int dof   = m_uiEVar.get_dof();
    const size_t stride      = (size_t)dof * nPe;
    const unsigned int nCls  = m_uiOwnDTs.size();
    const size_t cap = std::max<size_t>(64, ts::EnutsProf::p2cSharedBytes() / (stride * sizeof(T)));
    m_uiP2cSharedOn = false;
    if (m_uiOwnPlan.size() != pMesh->getLocalBlockList().size())
        m_uiOwnPlan.assign(pMesh->getLocalBlockList().size(), OwnPlan());
    SyncTag key;
    key.pass = m_uiSyncPass;
    key.rk   = rk_s;
    key.fut  = 3;
    // 1. the unique children of the active blocks, one shared slot each, grouped by parent row
    const bool direct = ts::EnutsProf::ownP2cSharedMode() == 2;
    struct Item { unsigned int row, store, blk, outB, outE; unsigned char cnum; };
    std::vector<Item> items;
    size_t slots = 0, over = 0;
    for (unsigned int bb = 0; bb < nAct; bb++) {
        const unsigned int blk = actIds[bb];
        OwnPlan& op            = m_uiOwnPlan[blk];
        if (!op.built) {
            sync_lazy_build(true);
            build_own_plan(blk, m_uiBlkDepEid[blk], op);
        }
        for (const typename OwnPlan::Child& w : op.ch) {
            if (m_uiP2cPassOf[w.store] != m_uiSyncPass) {
                if (!direct && slots >= cap) { over++; continue; }
                m_uiP2cPassOf[w.store] = m_uiSyncPass;
                m_uiP2cSlotOf[w.store] = direct ? ~0u : (unsigned int)slots;
                m_uiP2cTagOf[w.store]  = key;
                slots++;
            } else if (!direct) {
                continue;  // pool mode: one entry per unique child
            }
            items.push_back({w.row, w.store, blk, w.outB, w.outE, w.cnum});
        }
    }
    if (ts::EnutsProf::on() && over) ts::EnutsProf::get().t[ts::EnutsProf::CN_P2C_OVF] += (double)over;
    if (ts::EnutsProf::on()) {
        // distinct children of this coarse step and the store a hoisted p2c
        // (one interpolated child per contributing class and stage) would need
        if (m_uiP2cStepOf.size() != m_uiP2cPassOf.size()) m_uiP2cStepOf.assign(m_uiP2cPassOf.size(), 0);
        double nd = 0, dbl = 0;
        for (const Item& it : items) {
            if (m_uiP2cStepOf[it.store] == m_uiCoarseStep) continue;
            m_uiP2cStepOf[it.store] = m_uiCoarseStep;
            const unsigned int* ro = m_uiOwnRowOff.data() + (size_t)it.row * (nCls + 1);
            unsigned int cls = 0;
            for (unsigned int c = 0; c < nCls; c++)
                cls += (ro[c + 1] > ro[c] || c == m_uiOwnRowPc[it.row]);
            nd += 1;
            dbl += (double)cls * (m_uiNumStages + 1) * stride;
        }
        ts::EnutsProf::get().t[ts::EnutsProf::CN_P2C_DISTINCT] += nd;
        ts::EnutsProf::get().t[ts::EnutsProf::CN_HOIST_DBL] += dbl;
    }
    if (!slots) return;
    // by row, then by child: the readers of one child are contiguous
    std::stable_sort(items.begin(), items.end(), [](const Item& a, const Item& b) {
        return a.row != b.row ? a.row < b.row : a.store < b.store;
    });
    m_uiP2cWorkRow.clear();
    m_uiP2cWorkBlk.clear();
    m_uiP2cWorkOff.assign(1, 0);
    m_uiP2cWorkStore.resize(items.size());
    m_uiP2cWorkCnum.resize(items.size());
    m_uiP2cWorkRdr.resize(items.size());
    m_uiP2cWorkOutB.resize(items.size());
    m_uiP2cWorkOutE.resize(items.size());
    for (size_t i = 0; i < items.size(); i++) {
        if (i == 0 || items[i].row != items[i - 1].row) {
            m_uiP2cWorkRow.push_back(items[i].row);
            m_uiP2cWorkBlk.push_back(items[i].blk);
            if (i) m_uiP2cWorkOff.push_back((unsigned int)i);
        }
        m_uiP2cWorkStore[i] = items[i].store;
        m_uiP2cWorkCnum[i]  = items[i].cnum;
        m_uiP2cWorkRdr[i]   = items[i].blk;
        m_uiP2cWorkOutB[i]  = items[i].outB;
        m_uiP2cWorkOutE[i]  = items[i].outE;
    }
    m_uiP2cWorkOff.push_back((unsigned int)items.size());
    if (!direct && m_uiP2cPool.size() < slots * stride) m_uiP2cPool.resize(slots * stride);
    m_uiP2cPoolPass  = m_uiSyncPass;
    m_uiP2cUsed      = direct ? 0 : slots;
    m_uiP2cSharedOn  = true;
    T* pool          = direct ? nullptr : m_uiP2cPool.data();
    const unsigned int nRows = (unsigned int)m_uiP2cWorkRow.size();
    // 2. each unique row once: its reconciled row, then its children into the pool
#ifdef DENDRO_ENUTS_HAS_OMP
#pragma omp parallel for schedule(dynamic, 1) if (thr)
#endif
    for (unsigned int r = 0; r < nRows; r++) {
        SyncThr* const thr_ =
            ts::EnutsProf::tid() ? &m_uiSyncThr[ts::EnutsProf::tid()] : nullptr;
        std::vector<T>& rowBufV   = thr_ ? thr_->ownRowBuf : this->m_uiOwnRowBuf;
        std::vector<T>& scrV      = thr_ ? thr_->ownScr : this->m_uiOwnScr;
        std::vector<T>& sweepV    = thr_ ? thr_->ownSweep : this->m_uiOwnSweep;
        std::vector<unsigned char>& sweepOkV = thr_ ? thr_->ownSweepOk : this->m_uiOwnSweepOk;
        if (rowBufV.size() < stride) rowBufV.resize(stride);
        if (sweepV.size() < 6 * stride) sweepV.resize(6 * stride);
        if (sweepOkV.size() != (size_t)6 * dof) sweepOkV.assign((size_t)6 * dof, 0);
        T* scr      = scrV.data();
        T* g        = scr;
        T* pin      = scr + nPe;
        T* pout     = scr + 2 * nPe;
        double* im1 = (double*)(scr + 3 * nPe);
        double* im2 = (double*)(scr + 4 * nPe);
        const unsigned int blk   = m_uiP2cWorkBlk[r];
        const unsigned int bTime = m_uiEleTime[blkList[blk].firstElementID()];
        const unsigned int bDT   = m_uiEleDT[blkList[blk].firstElementID()];
        const unsigned int bc    = (unsigned int)(
            std::lower_bound(m_uiOwnDTs.begin(), m_uiOwnDTs.end(), bDT) - m_uiOwnDTs.begin());
        const OwnConv* cv = own_conv_rows(bc, rk_s, bTime, bDT);
        const bool pprof  = ts::EnutsProf::on();
        const double tr0  = pprof ? ts::EnutsProf::now() : 0.0;
        own_row_build(m_uiP2cWorkRow[r], rk_s, cv, rowBufV.data(), g, pin, pout, im1, im2);
        if (pprof) ts::EnutsProf::get().t[ts::EnutsProf::SY_ROW] += ts::EnutsProf::now() - tr0;
        std::fill(sweepOkV.begin(), sweepOkV.end(), 0);
        if (!direct) {
            for (unsigned int i = m_uiP2cWorkOff[r]; i < m_uiP2cWorkOff[r + 1]; i++) {
                T* sd = pool + (size_t)m_uiP2cSlotOf[m_uiP2cWorkStore[i]] * stride;
                for (unsigned int v = 0; v < dof; v++)
                    p2c3_shared(pMesh, rowBufV.data() + (size_t)v * nPe, sd + (size_t)v * nPe,
                                m_uiP2cWorkCnum[i], sweepV.data() + (size_t)v * 6 * nPe,
                                sweepOkV.data() + (size_t)v * 6);
            }
        } else {
            // one child at a time into pout, then its output runs into every reader block
            std::vector<T>& childV = thr_ ? thr_->ownRowBuf2 : this->m_uiOwnRowBuf2;
            if (childV.size() < stride) childV.resize(stride);
            unsigned int i = m_uiP2cWorkOff[r];
            while (i < m_uiP2cWorkOff[r + 1]) {
                const unsigned int st = m_uiP2cWorkStore[i];
                const double tp0 = pprof ? ts::EnutsProf::now() : 0.0;
                for (unsigned int v = 0; v < dof; v++)
                    p2c3_shared(pMesh, rowBufV.data() + (size_t)v * nPe, childV.data() + (size_t)v * nPe,
                                m_uiP2cWorkCnum[i], sweepV.data() + (size_t)v * 6 * nPe,
                                sweepOkV.data() + (size_t)v * 6);
                const double tc0 = pprof ? ts::EnutsProf::now() : 0.0;
                if (pprof) ts::EnutsProf::get().t[ts::EnutsProf::SY_P2C] += tc0 - tp0;
                for (; i < m_uiP2cWorkOff[r + 1] && m_uiP2cWorkStore[i] == st; i++) {
                    const unsigned int rb = m_uiP2cWorkRdr[i];
                    const OwnPlan& rop    = m_uiOwnPlan[rb];
                    T* blkVec = (T*)m_uiBVec[rb]._vec[stage_slot(rk_s)].data();
                    const size_t lxyz = (size_t)blkList[rb].getAllocationSzX() *
                                        blkList[rb].getAllocationSzY() *
                                        blkList[rb].getAllocationSzZ();
                    for (unsigned int v = 0; v < dof; v++) {
                        T* dv       = blkVec + (size_t)v * lxyz;
                        const T* pv = childV.data() + (size_t)v * nPe;
                        for (unsigned int q = m_uiP2cWorkOutB[i]; q < m_uiP2cWorkOutE[i]; q++) {
                            const T* sp = pv + rop.cT[q];
                            T* dp       = dv + rop.cDst[q];
                            for (unsigned int k = 0; k < rop.cLen[q]; k++) dp[k] = sp[k];
                        }
                    }
                }
                if (pprof) ts::EnutsProf::get().t[ts::EnutsProf::SY_CPY] += ts::EnutsProf::now() - tc0;
            }
        }
        (void)nCls;
    }
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::own_sync_fill(unsigned int blk, unsigned int rk_s,
                                         const std::vector<unsigned int>& eid) {
    // [ENUTS_OMP begin] worker threads (tid >= 1) of the threaded sync loop:
    // the locals below shadow the shared scratch / per-pass caches for the
    // rest of this function (see SyncThr). the plans were built by
    // sync_prebuild(). thr_ == nullptr = serial / thread 0.
    SyncThr* const thr_ =
        ts::EnutsProf::tid() ? &m_uiSyncThr[ts::EnutsProf::tid()] : nullptr;
    ENUTSOp* const m_uiECOp           = thr_ ? thr_->op : this->m_uiECOp;
    std::vector<OwnConv>& m_uiOwnConv = thr_ ? thr_->ownConv : this->m_uiOwnConv;
    std::vector<unsigned long long>& m_uiOwnConvPass = thr_ ? thr_->ownConvPass : this->m_uiOwnConvPass;
    std::vector<T>& m_uiOwnScr        = thr_ ? thr_->ownScr : this->m_uiOwnScr;
    std::vector<SyncTag>& m_uiSyncTag = thr_ ? thr_->tag : this->m_uiSyncTag;
    std::vector<T>& m_uiOwnSweep      = thr_ ? thr_->ownSweep : this->m_uiOwnSweep;
    std::vector<unsigned char>& m_uiOwnSweepOk = thr_ ? thr_->ownSweepOk : this->m_uiOwnSweepOk;
    std::vector<T>& m_uiOwnRowBuf     = thr_ ? thr_->ownRowBuf : this->m_uiOwnRowBuf;
    unsigned int& m_uiOwnRowBufRow    = thr_ ? thr_->ownRowBufRow : this->m_uiOwnRowBufRow;
    unsigned long long& m_uiOwnRowBufPass = thr_ ? thr_->ownRowBufPass : this->m_uiOwnRowBufPass;
    // [ENUTS_OMP end]
    ot::Mesh* pMesh          = (ot::Mesh*)m_uiAppCtx->get_mesh();
    const ot::Block* blkList = pMesh->getLocalBlockList().data();
    const unsigned int nPe   = pMesh->getNumNodesPerElement();
    const unsigned int n1    = pMesh->getElementOrder() + 1;
    const size_t dgSz        = (size_t)pMesh->getAllElements().size() * nPe;
    const size_t lxyz        = (size_t)blkList[blk].getAllocationSzX() *
                               blkList[blk].getAllocationSzY() *
                               blkList[blk].getAllocationSzZ();
    const unsigned int dof   = m_uiEVar.get_dof();
    const unsigned int nCls  = m_uiOwnDTs.size();
    const bool sprof         = ts::EnutsProf::on();
    const double t0          = sprof ? ts::EnutsProf::now() : 0.0;
    double tp2c              = 0;

    if (m_uiOwnPlan.size() != pMesh->getLocalBlockList().size())
        m_uiOwnPlan.assign(pMesh->getLocalBlockList().size(), OwnPlan());
    OwnPlan& op = m_uiOwnPlan[blk];
    if (!op.built) build_own_plan(blk, eid, op);

    const unsigned int bTime = m_uiEleTime[blkList[blk].firstElementID()];
    const unsigned int bDT   = m_uiEleDT[blkList[blk].firstElementID()];
    const unsigned int bc    = (unsigned int)(
        std::lower_bound(m_uiOwnDTs.begin(), m_uiOwnDTs.end(), bDT) - m_uiOwnDTs.begin());
    assert(bc < nCls && m_uiOwnDTs[bc] == bDT);

    // one conversion row per owner class for this block class and stage pass:
    // what the per-element loop decides from (element time, element dt)
    const OwnConv* cv = own_conv_rows(bc, rk_s, bTime, bDT);
    if (ts::EnutsProf::xchCensus() && !m_uiXchFlag.empty()) {
        // the stage slots this call reads (rows: in own_row_build)
        for (unsigned int c = 0; c < nCls; c++) {
            const OwnConv& w = cv[c];
            if (!w.same)
                for (unsigned int i = op.celeOff[c]; i < op.celeOff[c + 1]; i++)
                    xch_mark_ele(op.cele[i], nPe, w, rk_s);
            else
                for (unsigned int r = op.runOff[c]; r < op.runOff[c + 1]; r++)
                    for (unsigned int i = 0; i < op.rlen[r]; i++)
                        xch_mark(op.rsrc[r] + i, w, rk_s);
            for (unsigned int q = op.clsOff[c]; q < op.clsOff[c + 1]; q++)
                xch_mark(op.psrc[q], w, rk_s);
        }
        for (const typename OwnPlan::Op& w : op.ops) {
            unsigned int b = 0;
            for (unsigned int sg = w.segB; sg < w.segE; sg++) {
                const OwnConv& c     = cv[op.opSegCls[sg]];
                const unsigned int e = op.opSegEnd[sg];
                for (unsigned int t = b; t < e; t++) xch_mark(op.opSrc[w.inB + t], c, rk_s);
                b = e;
            }
        }
    }

    T* blkVec = (T*)m_uiBVec[blk]._vec[stage_slot(rk_s)].data();
    // SG: the inputs of a class that reads g (k1 .. k_rk_s, g)
    const T* S[8];
    const T* SG[8];
    const T* stG = m_uiStGOn ? m_uiStG.get_vec_ptr() : nullptr;
    T* scr      = m_uiOwnScr.data();  // g, pin, pout | im1, im2
    T* g        = scr;
    T* pin      = scr + nPe;
    T* pout     = scr + 2 * nPe;
    double* im1 = (double*)(scr + 3 * nPe);
    double* im2 = (double*)(scr + 4 * nPe);

    // owner elements of a converted class that feed many slots: the whole
    // element into m_uiDGTmp0, unless a block of this pass and class did it
    // [ENUTS_OMP] per-thread stand-in for the shared work vector
    T* dgW = thr_ ? thr_->dgW.get() : m_uiDGTmp0.get_vec_ptr();
    if (m_uiSyncTag.size() != pMesh->getAllElements().size())
        m_uiSyncTag.assign(pMesh->getAllElements().size(), SyncTag());
    SyncTag etag;
    etag.pass = m_uiSyncPass;
    etag.rk   = rk_s;
    etag.fut  = 4;
    etag.dt_c = (T)bc;
    unsigned int nconv = 0, nreuse = 0;
    for (unsigned int c = 0; c < nCls; c++) {
        if (cv[c].same) continue;
        for (unsigned int i = op.celeOff[c]; i < op.celeOff[c + 1]; i++) {
            const unsigned int ele = op.cele[i];
            if (m_uiSyncPassOpen && m_uiSyncTag[ele] == etag) {
                nreuse++;
                continue;
            }
            nconv++;
            for (unsigned int v = 0; v < dof; v++) {
                for (unsigned int k = 0; k < m_uiNumStages; k++)
                    S[k] = m_uiStVec[k].get_vec_ptr() + (size_t)v * dgSz + (size_t)ele * nPe;
                if (cv[c].g) S[cv[c].ns - 1] = stG + (size_t)v * dgSz + (size_t)ele * nPe;
                if (sprof && cv[c].g && !v) {
                    ts::EnutsProf::get().t[ts::EnutsProf::CN_GALL] += nPe;
                    if (!gValidCls(c)) ts::EnutsProf::get().t[ts::EnutsProf::CN_G0] += nPe;
                }
                m_uiECOp->applyRow(dgW + (size_t)v * dgSz + (size_t)ele * nPe, S,
                                   cv[c].m, cv[c].ns, nPe);
            }
            m_uiSyncTag[ele] = m_uiSyncPassOpen ? etag : SyncTag();
        }
    }
    const double t1 = sprof ? ts::EnutsProf::now() : 0.0;
    double tops     = 0;

    for (unsigned int v = 0; v < dof; v++) {
        for (unsigned int k = 0; k < m_uiNumStages; k++)
            SG[k] = S[k] = m_uiStVec[k].get_vec_ptr() + (size_t)v * dgSz;
        if (stG) SG[rk_s] = stG + (size_t)v * dgSz;
        T* dv = blkVec + (size_t)v * lxyz;
        for (unsigned int c = 0; c < nCls; c++) {
            if (op.runOff[c + 1] > op.runOff[c]) {
                const T* base = cv[c].same ? S[rk_s - 1]
                                           : (const T*)(dgW + (size_t)v * dgSz);
                const unsigned int* rd   = op.rdst.data();
                const unsigned int* rs   = op.rsrc.data();
                const unsigned short* rl = op.rlen.data();
                for (unsigned int r = op.runOff[c]; r < op.runOff[c + 1]; r++) {
                    const T* sp = base + rs[r];
                    T* dp       = dv + rd[r];
                    for (unsigned int i = 0; i < rl[r]; i++) dp[i] = sp[i];
                }
            }
            const unsigned int b = op.clsOff[c], n = op.clsOff[c + 1] - b;
            if (!n) continue;
            const unsigned int* dp = op.pdst.data() + b;
            const unsigned int* sp = op.psrc.data() + b;
            if (cv[c].same) {
                const T* sv = S[rk_s - 1];
                for (unsigned int q = 0; q < n; q++) dv[dp[q]] = sv[sp[q]];
            } else {
                m_uiECOp->gatherRow(dv, dp, sp, n, cv[c].g ? SG : S, cv[c].m, cv[c].ns);
                if (sprof && cv[c].g && !v) {
                    ts::EnutsProf::get().t[ts::EnutsProf::CN_GALL] += n;
                    if (!gValidCls(c)) ts::EnutsProf::get().t[ts::EnutsProf::CN_G0] += n;
                }
            }
        }
        const double to0 = sprof ? ts::EnutsProf::now() : 0.0;
        for (const typename OwnPlan::Op& w : op.ops) {
            const unsigned int len    = (w.dim == 2) ? n1 * n1 : n1;
            const unsigned int* src = op.opSrc.data() + w.inB;
            const unsigned int* ti  = op.opT.data() + w.inB;
            unsigned int b          = 0;
            for (unsigned int sg = w.segB; sg < w.segE; sg++) {
                const OwnConv& c     = cv[op.opSegCls[sg]];
                const unsigned int e = op.opSegEnd[sg];
                if (c.same) {
                    const T* sv = S[rk_s - 1];
                    for (unsigned int t = b; t < e; t++) pin[ti[t]] = sv[src[t]];
                } else {
                    m_uiECOp->gatherRow(pin, ti + b, src + b, e - b, c.g ? SG : S, c.m, c.ns);
                    if (sprof && c.g && !v) {
                        ts::EnutsProf::get().t[ts::EnutsProf::CN_GALL] += e - b;
                        if (!gValidCls(op.opSegCls[sg])) ts::EnutsProf::get().t[ts::EnutsProf::CN_G0] += e - b;
                    }
                }
                b = e;
            }
            pMesh->parent2ChildInterpolation(pin, pout, w.cnum, w.dim, im1, im2);
            for (unsigned int q = w.outB; q < w.outE; q++)
                dv[op.oDst[q]] = pout[op.oT[q]];
        }
        if (sprof) tops += ts::EnutsProf::now() - to0;
    }
    const double t2 = sprof ? ts::EnutsProf::now() : 0.0;

    const size_t stride = (size_t)dof * nPe;
    if (m_uiOwnSweep.size() < 6 * stride) m_uiOwnSweep.resize(6 * stride);
    if (m_uiOwnSweepOk.size() != (size_t)6 * dof) m_uiOwnSweepOk.assign((size_t)6 * dof, 0);
    SyncTag key;
    key.pass = m_uiSyncPass;
    key.rk   = rk_s;
    key.fut  = 3;
    unsigned int nhit = 0;
    double trow = 0, tcpy = 0;
    for (const typename OwnPlan::Child& w : op.ch) {
        bool hit = false;
        T* sd    = m_uiSyncPassOpen ? p2c_slot(w.store, key, stride, hit) : nullptr;
        nhit += hit;
        if (hit && !sd) continue;  // OWN_P2C_SHARED=2: the pre-pass wrote this child
        if (!hit && !(m_uiOwnRowBufRow == w.row && m_uiOwnRowBufPass == m_uiSyncPass &&
                      m_uiSyncPassOpen)) {
            const double tr0 = sprof ? ts::EnutsProf::now() : 0.0;
            // the dependent's row as its zip + gather gives it: owner values,
            // then its own hanging faces and edges
            if (m_uiOwnRowBuf.size() < stride) m_uiOwnRowBuf.resize(stride);
            own_row_build(w.row, rk_s, cv, m_uiOwnRowBuf.data(), g, pin, pout, im1, im2);
            m_uiOwnRowBufRow  = w.row;
            m_uiOwnRowBufPass = m_uiSyncPass;
            std::fill(m_uiOwnSweepOk.begin(), m_uiOwnSweepOk.end(), 0);
            if (sprof) trow += ts::EnutsProf::now() - tr0;
        }
        if (!hit) {
            const double tp = sprof ? ts::EnutsProf::now() : 0.0;
            for (unsigned int v = 0; v < dof; v++) {
                T* dv = blkVec + (size_t)v * lxyz;
                T* pv = sd ? sd + (size_t)v * nPe : pout;
                p2c3_shared(pMesh, m_uiOwnRowBuf.data() + (size_t)v * nPe, pv, w.cnum,
                            m_uiOwnSweep.data() + (size_t)v * 6 * nPe,
                            m_uiOwnSweepOk.data() + (size_t)v * 6);
                if (!sd)
                    for (unsigned int q = w.outB; q < w.outE; q++) {
                        const T* sp = pv + op.cT[q];
                        T* dp       = dv + op.cDst[q];
                        for (unsigned int i = 0; i < op.cLen[q]; i++) dp[i] = sp[i];
                    }
            }
            if (sprof) tp2c += ts::EnutsProf::now() - tp;
            if (!sd) continue;
        }
        const double tc0 = sprof ? ts::EnutsProf::now() : 0.0;
        for (unsigned int v = 0; v < dof; v++) {
            T* dv       = blkVec + (size_t)v * lxyz;
            const T* pv = sd + (size_t)v * nPe;
            for (unsigned int q = w.outB; q < w.outE; q++) {
                const T* sp = pv + op.cT[q];
                T* dp       = dv + op.cDst[q];
                for (unsigned int i = 0; i < op.cLen[q]; i++) dp[i] = sp[i];
            }
        }
        if (sprof) tcpy += ts::EnutsProf::now() - tc0;
    }

    if (sprof) {
        ts::EnutsProf& P = ts::EnutsProf::get();
        const double t3 = ts::EnutsProf::now();
        // plan path: same_copy = plan + rows + whole-element conversions,
        // corr_coarse_fut = node copies, corr_fine_past = hanging faces/edges,
        // unzip_pad = children of coarser dependents (p2c inside)
        P.t[ts::EnutsProf::SY_SAME] += t1 - t0;
        P.t[ts::EnutsProf::SY_CORR] += (t2 - t1) - tops;
        P.t[ts::EnutsProf::SY_CORR_FC] += tops;
        P.t[ts::EnutsProf::SY_UNZIP] += t3 - t2;
        P.t[ts::EnutsProf::SY_P2C] += tp2c;
        P.t[ts::EnutsProf::SY_ROW] += trow;
        P.t[ts::EnutsProf::SY_CPY] += tcpy;
        P.t[ts::EnutsProf::CN_ENT] += op.ops.size();
        P.t[ts::EnutsProf::CN_CORR] += nconv;
        P.t[ts::EnutsProf::CN_HIT] += nreuse;
        size_t nrun = 0;
        for (const unsigned short l : op.rlen) nrun += l;
        size_t ncr = 0;
        for (const unsigned short l : op.cLen) ncr += l;
        P.t[ts::EnutsProf::CN_PAIRS] += nrun + op.pdst.size() + op.oDst.size() + ncr;
        P.t[ts::EnutsProf::CN_P2C] += op.ch.size();
        P.t[ts::EnutsProf::CN_P2C_HIT] += nhit;
    }
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::own_node_exchange_begin(T* stv, unsigned int dof,
                                                   unsigned int pt, int cls) {
    ot::Mesh* pMesh = (ot::Mesh*)m_uiAppCtx->get_mesh();
    if (!pMesh->isActive() || pMesh->getMPICommSize() < 2) return;
    const unsigned int nCls = m_uiOwnDTs.size();
    const size_t dgSz = (size_t)pMesh->getAllElements().size() * pMesh->getNumNodesPerElement();
    // the owners that computed this stage are the classes whose dt divides pt:
    // a prefix of the ascending classes when the dts form a divisibility chain
    int j = -1;
    if (!m_uiOwnChain)
        j = (int)nCls - 1;
    else
        for (unsigned int i = 0; i < nCls; i++)
            if (pt % m_uiOwnDTs[i] == 0) j = (int)i;
    const std::vector<unsigned int>& sPr = pMesh->getSendProcList();
    const std::vector<unsigned int>& rPr = pMesh->getRecvProcList();
    MPI_Comm comm = pMesh->getMPICommunicator();
    const int tag = 7153;
    m_uiOwnReq.assign(sPr.size() + rPr.size(), MPI_REQUEST_NULL);
    if (j < 0) return;
    if (ts::EnutsProf::xchScoped() && !m_uiXchScoped && m_uiOwnChain) xch_scope_build(pMesh);
    // one class: its slot range [cum[cls-1], cum[cls]) per partner (the lists
    // are sorted by class, scoped or not)
    const bool one = cls >= 0;
    if (one) j = cls;
    auto lo = [&](const unsigned int* cum) { return one && cls ? cum[cls - 1] : 0u; };
    auto nn = [&](const unsigned int* cum) { return cum[j] - lo(cum); };
    if (!one && ts::EnutsProf::xchCensus() && !m_uiXchFlag.empty()) {
        // what became of the previous message into these slots (this stage):
        // read by a plan since, or overwritten unread; never-read slots no
        // plan lists are the dead weight of the map itself
        unsigned int s = 0;
        for (unsigned int k = 0; k < m_uiNumStages; k++)
            if (m_uiStVec[k].get_vec_ptr() == stv) s = k;
        const unsigned char* fl = m_uiXchFlag.data() + (size_t)s * dgSz;
        double nRd = 0, nUn = 0, nNo = 0;
        std::vector<double> unC(nCls, 0), rdC(nCls, 0);
        for (size_t i = 0; i < rPr.size(); i++) {
            const unsigned int cnt = m_uiOwnRecvCum[i * nCls + j];
            const unsigned int* sl = m_uiOwnRecvSlot.data() + m_uiOwnRecvOff[i];
            const unsigned int* cum = m_uiOwnRecvCum.data() + i * nCls;
            unsigned int c = 0;
            for (unsigned int q = 0; q < cnt; q++) {
                while (q >= cum[c]) c++;
                const unsigned char f = fl[sl[q]];
                if (!f) continue;
                if (!m_uiXchInPlan[sl[q]])
                    nNo++;
                else if (f == 1) {
                    nUn++;
                    unC[c]++;
                } else {
                    nRd++;
                    rdC[c]++;
                }
            }
        }
        if (ts::EnutsProf::xchCensusLvl() >= 2 && !pMesh->getMPIRankGlobal()) {
            std::cout << "[xch-census r0] pt " << pt << " s " << s + 1 << " j " << j << " read " << nRd
                      << " unread " << nUn << " noplan " << nNo << " | per class read/unread:";
            for (unsigned int c = 0; c < nCls; c++) std::cout << " " << rdC[c] << "/" << unC[c];
            std::cout << std::endl;
        }
        ts::EnutsProf& P = ts::EnutsProf::get();
        P.t[ts::EnutsProf::CN_XCH_NOPLAN] += nNo * dof;
        P.t[ts::EnutsProf::CN_XCH_UNREAD] += nUn * dof;
        P.t[ts::EnutsProf::CN_XCH_READ] += nRd * dof;
    }
    // XCH_TRACE: one extra double per partner message, the sender's stamp
    const size_t xt = (!one && ts::EnutsProf::xchTrace()) ? 1 : 0;
    size_t rTot = 0, sTot = 0;
    for (size_t i = 0; i < rPr.size(); i++) rTot += nn(m_uiOwnRecvCum.data() + i * nCls);
    for (size_t i = 0; i < sPr.size(); i++) sTot += nn(m_uiOwnSendCum.data() + i * nCls);
    if (m_uiOwnRecvBuf.size() < rTot * dof + xt * rPr.size())
        m_uiOwnRecvBuf.resize(rTot * dof + xt * rPr.size());
    if (m_uiOwnSendBuf.size() < sTot * dof + xt * sPr.size())
        m_uiOwnSendBuf.resize(sTot * dof + xt * sPr.size());
    size_t off = 0;
    for (size_t i = 0; i < rPr.size(); i++) {
        const unsigned int cnt = nn(m_uiOwnRecvCum.data() + i * nCls);
        if (!cnt) continue;
        par::Mpi_Irecv(m_uiOwnRecvBuf.data() + off, (int)(cnt * dof + xt), rPr[i], tag,
                       comm, &m_uiOwnReq[i]);
        off += (size_t)cnt * dof + xt;
    }
    off = 0;
    for (size_t i = 0; i < sPr.size(); i++) {
        const unsigned int* cum = m_uiOwnSendCum.data() + i * nCls;
        const unsigned int cnt  = nn(cum);
        if (!cnt) continue;
        const unsigned int* sl = m_uiOwnSendSlot.data() + m_uiOwnSendOff[i] + lo(cum);
        T* b                   = m_uiOwnSendBuf.data() + off + xt;
        const bool thr         = ts::EnutsProf::ownXchOmp();
#ifdef DENDRO_ENUTS_HAS_OMP
#pragma omp parallel for schedule(static) if (thr && !omp_in_parallel())
#endif
        for (unsigned int v = 0; v < dof; v++) {
            const T* sv = stv + (size_t)v * dgSz;
            for (unsigned int q = 0; q < cnt; q++) b[(size_t)v * cnt + q] = sv[sl[q]];
        }
        if (xt) b[-1] = (T)ts::EnutsProf::now();
        par::Mpi_Isend(b - xt, (int)(cnt * dof + xt), sPr[i], tag, comm,
                       &m_uiOwnReq[rPr.size() + i]);
        off += (size_t)cnt * dof + xt;
    }
    if (ts::EnutsProf::on() && !one) {
        ts::EnutsProf::get().t[ts::EnutsProf::CN_XCH_SEND] += (double)sTot * dof;
        ts::EnutsProf::get().t[ts::EnutsProf::CN_XCH_RECV] += (double)rTot * dof;
        ts::EnutsProf::get().t[ts::EnutsProf::CN_XCH_CALLS] += 1;
    }
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::own_node_exchange_end(T* stv, unsigned int dof,
                                                 unsigned int pt, int cls) {
    ot::Mesh* pMesh = (ot::Mesh*)m_uiAppCtx->get_mesh();
    if (!pMesh->isActive() || pMesh->getMPICommSize() < 2) return;
    const unsigned int nCls = m_uiOwnDTs.size();
    const size_t dgSz = (size_t)pMesh->getAllElements().size() * pMesh->getNumNodesPerElement();
    int j = -1;
    if (!m_uiOwnChain)
        j = (int)nCls - 1;
    else
        for (unsigned int i = 0; i < nCls; i++)
            if (pt % m_uiOwnDTs[i] == 0) j = (int)i;
    if (j < 0) return;
    const bool one = cls >= 0;
    if (one) j = cls;
    auto lo = [&](const unsigned int* cum) { return one && cls ? cum[cls - 1] : 0u; };
    auto nn = [&](const unsigned int* cum) { return cum[j] - lo(cum); };
    const bool xprof = ts::EnutsProf::on() && !one;
    const double tw0 = xprof ? ts::EnutsProf::now() : 0.0;
    const std::vector<unsigned int>& rPr = pMesh->getRecvProcList();
    const std::vector<unsigned int>& sPr = pMesh->getSendProcList();
    const size_t xt = (!one && ts::EnutsProf::xchTrace()) ? 1 : 0;
    if (xt) {
        // XCH_TRACE: completion order of every request; the last one is the
        // partner this call waited on. its lateness = its send stamp in my
        // clock minus the time I started waiting (a send request completes
        // when the receiver posted its receive, in its own exchange begin,
        // so its stamp to me stands in for that too)
        XchRec r{};
        r.pt = pt;
        r.j  = (unsigned int)j;
        for (unsigned int k = 0; k < m_uiNumStages; k++)
            if (m_uiStVec[k].get_vec_ptr() == stv) r.s = k;
        r.tw   = tw0 - m_uiXchT0;
        r.late = r.lateMax = std::nan("");
        r.arg = r.lateArg = ~0u;
        const int nReq = (int)m_uiOwnReq.size();
        int argIdx = -1, argR = -1, argS = -1;
        double tlast = tw0, tfirst = -1, tlr = tw0, tls = tw0;
        for (int k = 0; k < nReq; k++) {
            int idx = MPI_UNDEFINED;
            MPI_Waitany(nReq, m_uiOwnReq.data(), &idx, MPI_STATUS_IGNORE);
            if (idx == MPI_UNDEFINED) break;
            const double tn = ts::EnutsProf::now();
            if (idx < (int)rPr.size()) {
                if (tfirst < 0) tfirst = tn;
                tlr  = tn;
                argR = idx;
            } else {
                tls  = tn;
                argS = idx;
            }
            tlast = tn;
        }
        // the last receive is the partner waited on, unless a send (the
        // partner's receive posted late) outlasted it by more than a poll
        argIdx    = (argS >= 0 && tls > tlr + 5e-5) ? argS : argR;
        r.wait    = tlast - tw0;
        r.rwait   = tlr - tw0;
        r.first   = tfirst < 0 ? 0 : tfirst - tw0;
        r.argSend = (argIdx >= (int)rPr.size());
        if (argIdx >= 0) r.arg = r.argSend ? sPr[argIdx - rPr.size()] : rPr[argIdx];
        size_t off = 0;
        for (size_t i = 0; i < rPr.size(); i++) {
            const unsigned int cnt = m_uiOwnRecvCum[i * nCls + j];
            if (!cnt) continue;
            r.nrecv += (double)cnt * dof;
            const double L = std::isnan(m_uiXchOff[i])
                                 ? std::nan("")
                                 : (double)m_uiOwnRecvBuf[off] + m_uiXchOff[i] - tw0;
            if (rPr[i] == r.arg) {
                r.late = L;
                r.argN = (double)cnt * dof;
            }
            if (!std::isnan(L) && (std::isnan(r.lateMax) || L > r.lateMax)) {
                r.lateMax = L;
                r.lateArg = rPr[i];
            }
            off += (size_t)cnt * dof + xt;
        }
        if (r.argSend && argIdx >= 0)
            r.argN = (double)m_uiOwnSendCum[(argIdx - rPr.size()) * nCls + j] * dof;
        ts::EnutsProf& P = ts::EnutsProf::get();
        P.t[ts::EnutsProf::EX_FIRST] += r.first;
        if (!std::isnan(r.late)) P.t[ts::EnutsProf::EX_LATE] += std::max(0.0, r.late);
        if (!std::isnan(r.lateMax)) P.t[ts::EnutsProf::EX_LATEMAX] += std::max(0.0, r.lateMax);
        P.t[ts::EnutsProf::CN_XCH_ARGSEND] += r.argSend;
        P.t[ts::EnutsProf::CN_XCH_NOOFF] += (argIdx >= 0 && std::isnan(r.late));
        P.t[ts::EnutsProf::CN_XCH_LATEBAD] += (!std::isnan(r.lateMax) && r.lateMax > r.wait + 1e-4);
        m_uiXchRec.push_back(r);
    } else {
    if (xprof && !rPr.empty()) {
        // split of the wait: until the first receive completes (a late
        // partner) vs the rest (transfer / the slowest partner)
        int idx = MPI_UNDEFINED;
        MPI_Waitany((int)rPr.size(), m_uiOwnReq.data(), &idx, MPI_STATUS_IGNORE);
        ts::EnutsProf::get().t[ts::EnutsProf::EX_FIRST] += ts::EnutsProf::now() - tw0;
    }
    if (!m_uiOwnReq.empty())
        MPI_Waitall((int)m_uiOwnReq.size(), m_uiOwnReq.data(), MPI_STATUSES_IGNORE);
    }
    const double tu0 = xprof ? ts::EnutsProf::now() : 0.0;
    if (xprof) ts::EnutsProf::get().t[ts::EnutsProf::EX_WAIT] += tu0 - tw0;
    const bool cen = !one && ts::EnutsProf::xchCensus() && !m_uiXchFlag.empty();
    unsigned char* fl = nullptr;
    if (cen) {
        unsigned int s = 0;
        for (unsigned int k = 0; k < m_uiNumStages; k++)
            if (m_uiStVec[k].get_vec_ptr() == stv) s = k;
        fl = m_uiXchFlag.data() + (size_t)s * dgSz;
    }
    size_t off = 0;
    for (size_t i = 0; i < rPr.size(); i++) {
        const unsigned int* cum = m_uiOwnRecvCum.data() + i * nCls;
        const unsigned int cnt  = nn(cum);
        if (!cnt) continue;
        const unsigned int* sl = m_uiOwnRecvSlot.data() + m_uiOwnRecvOff[i] + lo(cum);
        const T* b             = m_uiOwnRecvBuf.data() + off + xt;
        const bool thr         = ts::EnutsProf::ownXchOmp();
#ifdef DENDRO_ENUTS_HAS_OMP
#pragma omp parallel for schedule(static) if (thr && !omp_in_parallel())
#endif
        for (unsigned int v = 0; v < dof; v++) {
            T* sv = stv + (size_t)v * dgSz;
            for (unsigned int q = 0; q < cnt; q++) sv[sl[q]] = b[(size_t)v * cnt + q];
        }
        if (cen)
            for (unsigned int q = 0; q < cnt; q++) fl[sl[q]] = 1;
        off += (size_t)cnt * dof + xt;
    }
    if (xprof) ts::EnutsProf::get().t[ts::EnutsProf::EX_UNPACK] += ts::EnutsProf::now() - tu0;
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::sts_hook(unsigned int pt, const unsigned int* actIds,
                                     unsigned int nAct, bool ownPlan, bool thr) {
    ot::Mesh* pMesh = (ot::Mesh*)m_uiAppCtx->get_mesh();
    if (!m_uiStsOn && stsTestC() > 0) {
        // test operator: explicit diffusion with RKL2 stages (stsTestC); the
        // stage count of a class from the largest D of its levels
        if (m_uiStsDt0 <= 0)
            m_uiStsDt0 = stsTestDt0() > 0 ? (T)stsTestDt0() : m_uiTimeInfo._m_uiTh;
        set_sts(stsTestVar(),
                [this](unsigned int r) -> unsigned int {
                    double D = 0;
                    for (unsigned int l = m_uiLevMin; l <= m_uiLevMax; l++)
                        if (step_ratio(l) == r)
                            D = std::max(D, sts_test_D((T)(m_uiTimeInfo._m_uiTh * r), l));
                    unsigned int s = 2;
                    while ((double)(s * s + s - 2) / 4.0 < 6.0 * D) s++;
                    return s + 1;
                },
                [this](unsigned int blk, unsigned int j, unsigned int s, T dt, const T* y0,
                       const T* y1, const T* y2, T* ly0, T* yj) {
                    const unsigned int lev =
                        m_uiAppCtx->get_mesh()->getLocalBlockList()[blk].getRegularGridLev();
                    sts_test_stage(blk, j, s, y0, y1, y2, ly0, yj, sts_test_D(dt, lev));
                });
        m_uiStsIdent = false;
        if (!pMesh->getMPIRankGlobal())
            std::cout << "[enuts-sts] test operator: diffusion C " << stsTestC() << " on var "
                      << stsTestVar() << " dt0 " << m_uiStsDt0 << std::endl;
    } else if (!m_uiStsOn && stsTest() > 0) {
        // test operator: identity, s stages on every class of ratio >= 4
        const unsigned int ts_ = (unsigned int)stsTest();
        set_sts(0, [ts_](unsigned int r) { return r >= 4 ? ts_ : 0u; },
                [](unsigned int, unsigned int j, unsigned int, T, const T*, const T* y1,
                   const T*, T* ly0, T* yj) {
                    (void)j; (void)y1; (void)ly0; (void)yj;  // y_j arrives as y_{j-1}
                });
        m_uiStsIdent = true;
    }
    if (!ownPlan || !m_uiOwnChain) {
        if (!pMesh->getMPIRankGlobal())
            std::cout << "[enuts-sts] the super-time-stepped pass needs the owner-sync plan "
                         "with chained classes; aborting" << std::endl;
        MPI_Abort(pMesh->getMPICommunicator(), 15);
    }
    const bool prof = ts::EnutsProf::on();
    const double t0 = prof ? ts::EnutsProf::now() : 0.0;
    const unsigned int nCls = m_uiOwnDTs.size();
    // coarsest first: a finer class then sees its coarser neighbors diffused
    for (unsigned int k = nCls; k-- > 0;) {
        const unsigned int r = m_uiOwnDTs[k];
        if (pt % r) continue;
        const unsigned int s = m_uiStsStages(r);
        if (!s) continue;
        sts_pass(k, s, (T)(m_uiTimeInfo._m_uiTh * r), actIds, nAct, thr, pt);
    }
    // end of the coarse step: every class at one time
    if (!m_uiStsIdent && stsTestC() > 0 && pt + 1 == m_uiOwnDTs.back()) {
        m_uiStsSteps++;
        if (stsTestDump()) sts_dump();
    }
    if (prof) ts::EnutsProf::get().t[ts::EnutsProf::ST_PASS] += ts::EnutsProf::now() - t0;
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::sts_zip(unsigned int blk, const T* y, T* dg) const {
    const ot::Mesh* pMesh      = m_uiAppCtx->get_mesh();
    const ot::Block& b         = pMesh->getLocalBlockList()[blk];
    const unsigned int regLev  = b.getRegularGridLev();
    const ot::TreeNode* pNodes = pMesh->getAllElements().data();
    const ot::TreeNode bn      = b.getBlockNode();
    const unsigned int eo      = pMesh->getElementOrder();
    const unsigned int pw      = b.get1DPadWidth();
    const unsigned int nPe     = pMesh->getNumNodesPerElement();
    const unsigned int lx = b.getAllocationSzX(), ly = b.getAllocationSzY();
    for (const unsigned int elem : b) {
        const unsigned int ei = (pNodes[elem].getX() - bn.getX()) >> (m_uiMaxDepth - regLev);
        const unsigned int ej = (pNodes[elem].getY() - bn.getY()) >> (m_uiMaxDepth - regLev);
        const unsigned int ek = (pNodes[elem].getZ() - bn.getZ()) >> (m_uiMaxDepth - regLev);
        T* d = dg + (size_t)elem * nPe;
        for (unsigned int k = 0; k <= eo; k++)
            for (unsigned int j = 0; j <= eo; j++)
                for (unsigned int i = 0; i <= eo; i++)
                    d[k * (eo + 1) * (eo + 1) + j * (eo + 1) + i] =
                        y[(size_t)(ek * eo + k + pw) * ly * lx + (ej * eo + j + pw) * lx +
                          (ei * eo + i + pw)];
    }
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::sts_fill(unsigned int blk, unsigned int cls, const T* dg, T* y,
                                     bool hang) const {
    const OwnPlan& op = m_uiOwnPlan[blk];
    // node-coincident copies: runs and single nodes
    for (unsigned int r = op.runOff[cls]; r < op.runOff[cls + 1]; r++)
        std::memcpy(y + op.rdst[r], dg + op.rsrc[r], (size_t)op.rlen[r] * sizeof(T));
    for (unsigned int q = op.clsOff[cls]; q < op.clsOff[cls + 1]; q++)
        y[op.pdst[q]] = dg[op.psrc[q]];
    if (!hang || (op.ops.empty() && op.ch.empty())) return;
    // level jumps: hanging faces / edges with an input of this class, and the
    // children of the coarser dependents of this class (the row as the full
    // fill builds it: the element's own DG row, the nodes other elements
    // own, its own hanging faces and edges), no stage conversion
    const ot::Mesh* pMesh   = m_uiAppCtx->get_mesh();
    const unsigned int nPe  = pMesh->getNumNodesPerElement();
    const unsigned int nCls = m_uiOwnDTs.size();
    if (m_uiStsOpScr.size() < 12 * (size_t)nPe) m_uiStsOpScr.assign(12 * (size_t)nPe, (T)0);
    T* scr      = m_uiStsOpScr.data();
    T* g        = scr;
    T* row      = scr + nPe;
    T* pin      = scr + 2 * nPe;
    T* pout     = scr + 3 * nPe;
    T* sw       = scr + 4 * nPe;
    double* im1 = (double*)(scr + 10 * nPe);
    double* im2 = (double*)(scr + 11 * nPe);
    for (const typename OwnPlan::Op& w : op.ops) {
        bool mine = false;
        for (unsigned int sg = w.segB; sg < w.segE; sg++) mine |= op.opSegCls[sg] == cls;
        if (!mine) continue;
        const unsigned int n = op.opSegEnd[w.segE - 1];
        for (unsigned int t = 0; t < n; t++) pin[op.opT[w.inB + t]] = dg[op.opSrc[w.inB + t]];
        pMesh->parent2ChildInterpolation(pin, pout, w.cnum, w.dim, im1, im2);
        for (unsigned int q = w.outB; q < w.outE; q++) y[op.oDst[q]] = pout[op.oT[q]];
    }
    unsigned char ok[6];
    for (const typename OwnPlan::Child& w : op.ch) {
        if (m_uiOwnRowPc[w.row] != cls) continue;
        const unsigned int ele = m_uiOwnRowEle[w.row];
        const unsigned int* ro = m_uiOwnRowOff.data() + (size_t)w.row * (nCls + 1);
        const bool hasOps      = m_uiOwnOpOff[ele + 1] > m_uiOwnOpOff[ele];
        T* gg                  = hasOps ? g : row;
        std::memcpy(gg, dg + (size_t)ele * nPe, sizeof(T) * nPe);
        for (unsigned int c = 0; c < nCls; c++)
            for (unsigned int q = ro[c]; q < ro[c + 1]; q++)
                gg[m_uiOwnRowN[q]] = dg[m_uiOwnRowSrc[q]];
        if (hasOps) own_row_ops(ele, g, row, pin, pout, im1, im2);
        std::memset(ok, 0, sizeof(ok));
        p2c3_shared(pMesh, row, pout, w.cnum, sw, ok);
        for (unsigned int q = w.outB; q < w.outE; q++)
            std::memcpy(y + op.cDst[q], pout + op.cT[q], (size_t)op.cLen[q] * sizeof(T));
    }
}

template <typename T, typename Ctx>
const std::vector<unsigned char>& ExplicitNUTS<T, Ctx>::sts_xmask(unsigned int blk) {
    const ot::Mesh* pMesh    = m_uiAppCtx->get_mesh();
    const ot::Block* blkList = pMesh->getLocalBlockList().data();
    const unsigned int nBlk  = pMesh->getLocalBlockList().size();
    if (m_uiStsXMask.size() != nBlk) m_uiStsXMask.assign(nBlk, std::vector<unsigned char>());
    std::vector<unsigned char>& m = m_uiStsXMask[blk];
    if (!m.empty() || m_uiOwnPlan.size() != nBlk || !m_uiOwnPlan[blk].built) return m;
    const size_t lxyz = (size_t)blkList[blk].getAllocationSzX() *
                        blkList[blk].getAllocationSzY() * blkList[blk].getAllocationSzZ();
    m.assign(lxyz, 0);
    const OwnPlan& op       = m_uiOwnPlan[blk];
    const unsigned int nCls = m_uiOwnDTs.size();
    const unsigned int bc   = (unsigned int)(
        std::lower_bound(m_uiOwnDTs.begin(), m_uiOwnDTs.end(),
                         m_uiEleDT[blkList[blk].firstElementID()]) - m_uiOwnDTs.begin());
    for (unsigned int c = 0; c < nCls; c++) {
        if (c == bc) continue;
        for (unsigned int r = op.runOff[c]; r < op.runOff[c + 1]; r++)
            for (unsigned int i = 0; i < op.rlen[r]; i++) m[op.rdst[r] + i] = 1;
        for (unsigned int q = op.clsOff[c]; q < op.clsOff[c + 1]; q++) m[op.pdst[q]] = 1;
    }
    for (const typename OwnPlan::Op& w : op.ops) {
        bool other = false;
        for (unsigned int sg = w.segB; sg < w.segE; sg++) other |= op.opSegCls[sg] != bc;
        if (!other) continue;
        for (unsigned int q = w.outB; q < w.outE; q++) m[op.oDst[q]] = 1;
    }
    for (const typename OwnPlan::Child& w : op.ch) {
        if (m_uiOwnRowPc[w.row] == bc) continue;
        for (unsigned int q = w.outB; q < w.outE; q++)
            for (unsigned int i = 0; i < op.cLen[q]; i++) m[op.cDst[q] + i] = 1;
    }
    return m;
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::sts_test_stage(unsigned int blk, unsigned int j, unsigned int s,
                                           const T* y0, const T* y1, const T* y2, T* ly0,
                                           T* yj, double D) const {
    const ot::Block& b       = m_uiAppCtx->get_mesh()->getLocalBlockList()[blk];
    const unsigned int lx    = b.getAllocationSzX(), ly = b.getAllocationSzY(),
                       lz    = b.getAllocationSzZ(), pw = b.get1DPadWidth();
    const unsigned int bflag = b.getBlkNodeFlag();
    // RKL2 (Meyer, Balsara, Aslam 2014); dt is inside D
    const double w1 = 4.0 / (double)(s * s + s - 2);
    auto bq = [](unsigned int q) {
        return q < 2 ? 1.0 / 3.0 : (double)(q * q + q - 2) / (2.0 * q * (q + 1));
    };
    double mu = 1, nu = 0, mut = bq(1) * w1, gt = 0;
    if (j >= 2) {
        mu  = (2.0 * j - 1) / j * bq(j) / bq(j - 1);
        nu  = -(j - 1.0) / j * bq(j) / bq(j - 2);
        mut = mu * w1;
        gt  = -(1.0 - bq(j - 1)) * mut;
    }
    const size_t sx = 1, sy = lx, sz = (size_t)lx * ly;
    const bool bl = bflag & (1u << OCT_DIR_LEFT), br = bflag & (1u << OCT_DIR_RIGHT),
               bd = bflag & (1u << OCT_DIR_DOWN), bu = bflag & (1u << OCT_DIR_UP),
               bb = bflag & (1u << OCT_DIR_BACK), bf = bflag & (1u << OCT_DIR_FRONT);
    for (unsigned int k = pw; k < lz - pw; k++)
        for (unsigned int jj = pw; jj < ly - pw; jj++)
            for (unsigned int i = pw; i < lx - pw; i++) {
                const size_t p = k * sz + jj * sy + i;
                const bool edge = (bl && i == pw) || (br && i + pw + 1 == lx) ||
                                  (bd && jj == pw) || (bu && jj + pw + 1 == ly) ||
                                  (bb && k == pw) || (bf && k + pw + 1 == lz);
                auto L = [&](const T* y) -> double {
                    return edge ? 0.0
                                : D * (y[p - sx] + y[p + sx] + y[p - sy] + y[p + sy] +
                                       y[p - sz] + y[p + sz] - 6.0 * y[p]);
                };
                if (j == 1) ly0[p] = (T)L(y0);
                const double l1 = (j == 1) ? (double)ly0[p] : L(y1);
                yj[p] = (T)(mu * y1[p] + nu * y2[p] + (1.0 - mu - nu) * y0[p] + mut * l1 +
                            gt * ly0[p]);
            }
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::sts_test_acc(unsigned int blk, unsigned int cls, const T* y0,
                                         const T* ys) {
    const ot::Mesh* pMesh   = m_uiAppCtx->get_mesh();
    const ot::Block& b      = pMesh->getLocalBlockList()[blk];
    const unsigned int nBlk = pMesh->getLocalBlockList().size();
    const unsigned int lx = b.getAllocationSzX(), ly = b.getAllocationSzY(),
                       lz = b.getAllocationSzZ(), pw = b.get1DPadWidth();
    const size_t lxyz = (size_t)lx * ly * lz;
    if (m_uiStsAcc.size() != nBlk) m_uiStsAcc.assign(nBlk, std::vector<T>());
    if (m_uiStsInc.size() != m_uiOwnDTs.size()) m_uiStsInc.assign(m_uiOwnDTs.size(), (T)0);
    std::vector<T>& a = m_uiStsAcc[blk];
    if (a.size() != lxyz) a.assign(lxyz, (T)0);
    T mx = 0;
    for (unsigned int k = pw; k < lz - pw; k++)
        for (unsigned int jj = pw; jj < ly - pw; jj++)
            for (unsigned int i = pw; i < lx - pw; i++) {
                const size_t p = (size_t)k * lx * ly + (size_t)jj * lx + i;
                const T d      = ys[p] - y0[p];
                a[p] += d;
                mx = std::max(mx, std::abs(d));
            }
    m_uiStsInc[cls] = std::max(m_uiStsInc[cls], mx);
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::sts_dump() {
    // per rank <dir>/sts_r<rank>.bin: header (magic, blocks, classes, coarse
    // steps, dt0, fine dt, per class ratio and largest pass increment), then
    // per block (id, level, lx, ly, lz, pad, class, ratio, origin x y z) and
    // its interior points in k, j, i order: shell flag, damping so far, U
    const ot::Mesh* pMesh    = m_uiAppCtx->get_mesh();
    const ot::Block* blkList = pMesh->getLocalBlockList().data();
    const unsigned int nBlk  = pMesh->getLocalBlockList().size();
    const unsigned int nCls  = m_uiOwnDTs.size();
    const unsigned int var   = m_uiStsVar;
    const std::string fn     = std::string(stsTestDump()) + "/sts_r" +
                           std::to_string(pMesh->getMPIRankGlobal()) + ".bin";
    FILE* f = std::fopen(fn.c_str(), "wb");
    if (!f) {
        std::cout << "[enuts-sts r" << pMesh->getMPIRankGlobal() << "] cannot write " << fn
                  << std::endl;
        return;
    }
    auto w32 = [&](unsigned int v) { std::fwrite(&v, 4, 1, f); };
    auto wd  = [&](double v) { std::fwrite(&v, 8, 1, f); };
    w32(0x53545301u);
    w32(nBlk);
    w32(nCls);
    w32(m_uiStsSteps);
    wd((double)m_uiStsDt0);
    wd((double)m_uiTimeInfo._m_uiTh);
    for (unsigned int c = 0; c < nCls; c++) {
        w32(m_uiOwnDTs[c]);
        wd(c < m_uiStsInc.size() ? (double)m_uiStsInc[c] : 0.0);
    }
    std::vector<unsigned char> dil, tmp;
    for (unsigned int blk = 0; blk < nBlk; blk++) {
        const ot::Block& b    = blkList[blk];
        const unsigned int lx = b.getAllocationSzX(), ly = b.getAllocationSzY(),
                           lz = b.getAllocationSzZ(), pw = b.get1DPadWidth();
        const size_t lxyz     = (size_t)lx * ly * lz;
        const ot::TreeNode bn = b.getBlockNode();
        const unsigned int bc = (unsigned int)(
            std::lower_bound(m_uiOwnDTs.begin(), m_uiOwnDTs.end(),
                             m_uiEleDT[b.firstElementID()]) - m_uiOwnDTs.begin());
        // shell: within pw (Chebyshev) of a slot another class fills; three
        // 1D max filters
        const std::vector<unsigned char>& xm = sts_xmask(blk);
        dil.assign(lxyz, 0);
        if (xm.size() == lxyz) {
            dil = xm;
            tmp.assign(lxyz, 0);
            const unsigned int n[3]   = {lx, ly, lz};
            const size_t st[3]        = {1, lx, (size_t)lx * ly};
            for (unsigned int d = 0; d < 3; d++) {
                for (size_t p = 0; p < lxyz; p++) {
                    if (!dil[p]) continue;
                    const unsigned int c = (unsigned int)((p / st[d]) % n[d]);
                    const unsigned int lo = c > pw ? c - pw : 0, hi = std::min(c + pw, n[d] - 1);
                    for (unsigned int q = lo; q <= hi; q++) tmp[p + ((size_t)q - c) * st[d]] = 1;
                }
                dil.swap(tmp);
                std::fill(tmp.begin(), tmp.end(), 0);
            }
        }
        const T* u   = m_uiBVec[blk]._vec[0].data() + (size_t)var * lxyz;
        const T* acc = (blk < m_uiStsAcc.size() && m_uiStsAcc[blk].size() == lxyz)
                           ? m_uiStsAcc[blk].data()
                           : nullptr;
        w32(blk);
        w32(b.getRegularGridLev());
        w32(lx);
        w32(ly);
        w32(lz);
        w32(pw);
        w32(bc);
        w32(m_uiOwnDTs[bc]);
        w32(bn.getX());
        w32(bn.getY());
        w32(bn.getZ());
        for (unsigned int k = pw; k < lz - pw; k++)
            for (unsigned int jj = pw; jj < ly - pw; jj++)
                for (unsigned int i = pw; i < lx - pw; i++) {
                    const size_t p = (size_t)k * lx * ly + (size_t)jj * lx + i;
                    std::fputc(dil[p], f);
                    wd(acc ? (double)acc[p] : 0.0);
                    wd((double)u[p]);
                }
    }
    std::fclose(f);
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::sts_pass(unsigned int cls, unsigned int s, T dt,
                                     const unsigned int* actIds, unsigned int nAct, bool thr,
                                     unsigned int pt) {
    ot::Mesh* pMesh          = (ot::Mesh*)m_uiAppCtx->get_mesh();
    const ot::Block* blkList = pMesh->getLocalBlockList().data();
    const size_t dgSz = (size_t)pMesh->getAllElements().size() * pMesh->getNumNodesPerElement();
    const unsigned int dof = m_uiEVar.get_dof();
    const unsigned int var = m_uiStsVar;
    // the blocks of this class among the active ones
    std::vector<unsigned int> bl;
    for (unsigned int bb = 0; bb < nAct; bb++) {
        const unsigned int blk = actIds[bb];
        if (m_uiEleDT[blkList[blk].firstElementID()] == m_uiOwnDTs[cls]) bl.push_back(blk);
    }
    if (m_uiStsScr.size() != pMesh->getLocalBlockList().size())
        m_uiStsScr.assign(pMesh->getLocalBlockList().size(), std::vector<T>());
    if (m_uiStsDG.size() != dgSz) m_uiStsDG.assign(dgSz, (T)0);
    const bool chk = stsCheck();
    std::vector<size_t> chkOff;
    if (chk) {
        m_uiStsChk.clear();
        for (const unsigned int blk : bl) {
            const size_t NN = m_uiBVec[blk]._vec[0].getSz();
            const T* y0     = m_uiBVec[blk]._vec[0].data() + (size_t)var * NN;
            chkOff.push_back(m_uiStsChk.size());
            m_uiStsChk.insert(m_uiStsChk.end(), y0, y0 + NN);
        }
    }
    for (const unsigned int blk : bl) {
        const size_t NN = m_uiBVec[blk]._vec[0].getSz();
        if (m_uiStsScr[blk].size() != 4 * NN) m_uiStsScr[blk].assign(4 * NN, (T)0);
    }
    // U of the class into the one-variable DG vector, exchanged in the class
    // range (a dependent's row reads the class's nodes from it), then the
    // padding from every coarser class at this time: its pass left its
    // damped U in the vector, and the block's U padding holds the owner
    // sync's value without that damping. The identity test keeps the
    // node-coincident rounds only (bitwise); STS_FROZEN_ONLY keeps the lag
    // of one coarse step; STS_TEST_BREAK pins that padding instead.
    const bool test = stsTestC() > 0 && !m_uiStsIdent;
    const bool hang = !m_uiStsIdent;
    const int brk   = test ? stsTestBreak() : 0;
    for (const unsigned int blk : bl)
        sts_zip(blk, m_uiBVec[blk]._vec[0].data() + (size_t)var * m_uiBVec[blk]._vec[0].getSz(),
                m_uiStsDG.data());
    own_node_exchange_begin(m_uiStsDG.data(), 1, pt, (int)cls);
    own_node_exchange_end(m_uiStsDG.data(), 1, pt, (int)cls);
    if (ts::EnutsProf::on()) ts::EnutsProf::get().t[ts::EnutsProf::CN_STS_ROUNDS] += 1;
    if (brk) {
        if (m_uiStsBrk.size() != pMesh->getLocalBlockList().size())
            m_uiStsBrk.assign(pMesh->getLocalBlockList().size(), std::vector<T>());
        for (const unsigned int blk : bl) {
            const size_t NN = m_uiBVec[blk]._vec[0].getSz();
            T* y0           = (T*)m_uiBVec[blk]._vec[0].data() + (size_t)var * NN;
            const std::vector<unsigned char>& xm = sts_xmask(blk);
            if (xm.size() != NN) continue;
            if (m_uiStsBrk[blk].empty()) {
                m_uiStsBrk[blk].assign(y0, y0 + NN);
                if (brk == 2) std::fill(m_uiStsBrk[blk].begin(), m_uiStsBrk[blk].end(), (T)0);
            }
            for (size_t i = 0; i < NN; i++)
                if (xm[i]) y0[i] = m_uiStsBrk[blk][i];
        }
    } else if (hang && !stsFrozenOnly()) {
        const unsigned int nCls = m_uiOwnDTs.size();
        for (unsigned int c = cls + 1; c < nCls; c++) {
            if ((pt + m_uiOwnDTs[cls]) % m_uiOwnDTs[c] || !m_uiStsStages(m_uiOwnDTs[c])) continue;
            for (const unsigned int blk : bl)
                sts_fill(blk, c, m_uiStsDG.data(),
                         (T*)m_uiBVec[blk]._vec[0].data() + (size_t)var * m_uiBVec[blk]._vec[0].getSz(),
                         true);
        }
    }
    // ring of three stage slots: a = Y_{j-1}, b = Y_{j-2}, c = Y_j
    unsigned int ia = 0, ib = 1, ic = 2;
    for (unsigned int j = 1; j <= s; j++) {
        const unsigned int nb = bl.size();
#ifdef DENDRO_ENUTS_HAS_OMP
#pragma omp parallel for schedule(dynamic, 1) if (thr)
#endif
        for (unsigned int q = 0; q < nb; q++) {
            const unsigned int blk = bl[q];
            const size_t NN = m_uiBVec[blk]._vec[0].getSz();
            const T* y0     = m_uiBVec[blk]._vec[0].data() + (size_t)var * NN;
            T* scr          = m_uiStsScr[blk].data();
            const T* y1     = j >= 2 ? scr + (size_t)ia * NN : y0;
            const T* y2     = j >= 3 ? scr + (size_t)ib * NN : y0;
            T* ly0          = scr + 3 * NN;
            T* yj           = scr + (size_t)ic * NN;
            std::memcpy(yj, y1, NN * sizeof(T));  // padding carries over
            m_uiStsStage(blk, j, s, dt, y0, y1, y2, ly0, yj);
        }
        // the round: same-class padding of Y_j from the neighbors' Y_j
        for (const unsigned int blk : bl)
            sts_zip(blk, m_uiStsScr[blk].data() + (size_t)ic * m_uiBVec[blk]._vec[0].getSz(),
                    m_uiStsDG.data());
        own_node_exchange_begin(m_uiStsDG.data(), 1, pt, (int)cls);
        own_node_exchange_end(m_uiStsDG.data(), 1, pt, (int)cls);
        for (const unsigned int blk : bl)
            sts_fill(blk, cls, m_uiStsDG.data(),
                     m_uiStsScr[blk].data() + (size_t)ic * m_uiBVec[blk]._vec[0].getSz(), hang);
        if (ts::EnutsProf::on()) ts::EnutsProf::get().t[ts::EnutsProf::CN_STS_ROUNDS] += 1;
        const unsigned int t = ib;
        ib = ia; ia = ic; ic = t;
    }
    // write Y_s back into U; other-class padding was frozen at U's value
    size_t bad = 0;
    for (size_t q = 0; q < bl.size(); q++) {
        const unsigned int blk = bl[q];
        const size_t NN = m_uiBVec[blk]._vec[0].getSz();
        T* y0           = (T*)m_uiBVec[blk]._vec[0].data() + (size_t)var * NN;
        const T* ys     = m_uiStsScr[blk].data() + (size_t)ia * NN;
        if (test) sts_test_acc(blk, cls, y0, ys);
        if (chk) {
            const T* ref = m_uiStsChk.data() + chkOff[q];
            for (size_t i = 0; i < NN; i++)
                if (std::memcmp(&ys[i], &ref[i], sizeof(T))) bad++;
        }
        std::memcpy(y0, ys, NN * sizeof(T));
    }
    (void)dof;
    if (chk && bad)
        std::cout << "[enuts-sts-check r" << pMesh->getMPIRankGlobal() << "] pt " << pt
                  << " class " << cls << " s " << s << " blocks " << bl.size()
                  << " entries changed by the identity pass: " << bad << std::endl;
}

template <typename T, typename Ctx>
bool ExplicitNUTS<T, Ctx>::xch_scope_build(ot::Mesh* pMesh) {
    // XCH_SCOPED: every rank must have every block's plan (sync_prebuild
    // builds them all in the first sync pass); until then, unscoped
    const unsigned int nBlk = pMesh->getLocalBlockList().size();
    int ok = (m_uiOwnPlan.size() == nBlk);
    for (unsigned int b = 0; ok && b < nBlk; b++) ok = m_uiOwnPlan[b].built;
    MPI_Comm comm = pMesh->getMPICommunicator();
    int g         = 0;
    MPI_Allreduce(&ok, &g, 1, MPI_INT, MPI_MIN, comm);
    if (!g) return false;
    const ot::Block* blkList = pMesh->getLocalBlockList().data();
    const unsigned int nPe   = pMesh->getNumNodesPerElement();
    const size_t dgSz        = (size_t)pMesh->getAllElements().size() * nPe;
    const unsigned int nCls  = m_uiOwnDTs.size();
    // per DG slot, the class of the finest local block whose plan reads it
    std::vector<unsigned char> rcls(dgSz, 255);
    auto low = [&](size_t slot, unsigned char c) {
        if (c < rcls[slot]) rcls[slot] = c;
    };
    for (unsigned int b = 0; b < nBlk; b++) {
        const OwnPlan& op = m_uiOwnPlan[b];
        const unsigned int bDT = m_uiEleDT[blkList[b].firstElementID()];
        const unsigned char bc = (unsigned char)(
            std::lower_bound(m_uiOwnDTs.begin(), m_uiOwnDTs.end(), bDT) - m_uiOwnDTs.begin());
        for (const unsigned int e : op.cele)
            if (xch_whole_row(e, bc))
                for (unsigned int n = 0; n < nPe; n++) low((size_t)e * nPe + n, bc);
        for (size_t r = 0; r < op.rsrc.size(); r++)
            for (unsigned int i = 0; i < op.rlen[r]; i++) low((size_t)op.rsrc[r] + i, bc);
        for (const unsigned int q : op.psrc) low(q, bc);
        for (const unsigned int q : op.opSrc) low(q, bc);
        for (const typename OwnPlan::Child& w : op.ch) {
            const size_t e = m_uiOwnRowEle[w.row];
            for (unsigned int n = 0; n < nPe; n++) low(e * nPe + n, bc);
            const unsigned int* ro = m_uiOwnRowOff.data() + (size_t)w.row * (nCls + 1);
            for (unsigned int q = ro[0]; q < ro[nCls]; q++) low(m_uiOwnRowSrc[q], bc);
        }
    }
    // per receive partner, the effective class of each slot in list order
    // (max of the owner class the list is sorted by and the reader class);
    // tell the sender, hear ours from each receiver
    const std::vector<unsigned int>& sPr = pMesh->getSendProcList();
    const std::vector<unsigned int>& rPr = pMesh->getRecvProcList();
    std::vector<unsigned char> rEff(m_uiOwnRecvSlot.size(), 255), sEff(m_uiOwnSendSlot.size(), 255);
    for (size_t i = 0; i < rPr.size(); i++) {
        const unsigned int* cum = m_uiOwnRecvCum.data() + i * nCls;
        for (unsigned int c = 0; c < nCls; c++)
            for (unsigned int q = (c ? cum[c - 1] : 0); q < cum[c]; q++) {
                const unsigned char r = rcls[m_uiOwnRecvSlot[m_uiOwnRecvOff[i] + q]];
                rEff[m_uiOwnRecvOff[i] + q] = (r == 255) ? 255 : std::max((unsigned char)c, r);
            }
    }
    std::vector<MPI_Request> rq(sPr.size() + rPr.size(), MPI_REQUEST_NULL);
    for (size_t i = 0; i < sPr.size(); i++) {
        const int n = (int)(m_uiOwnSendOff[i + 1] - m_uiOwnSendOff[i]);
        if (n) MPI_Irecv(sEff.data() + m_uiOwnSendOff[i], n, MPI_UNSIGNED_CHAR, sPr[i], 7156, comm, &rq[i]);
    }
    for (size_t i = 0; i < rPr.size(); i++) {
        const int n = (int)(m_uiOwnRecvOff[i + 1] - m_uiOwnRecvOff[i]);
        if (n) MPI_Isend(rEff.data() + m_uiOwnRecvOff[i], n, MPI_UNSIGNED_CHAR, rPr[i], 7156, comm, &rq[sPr.size() + i]);
    }
    MPI_Waitall((int)rq.size(), rq.data(), MPI_STATUSES_IGNORE);
    // both sides: the same stable sort of the same sequence by effective
    // class; the prefix counts now count effective classes, 255 never sent
    auto resort = [&](size_t np, std::vector<unsigned int>& slot, const std::vector<unsigned int>& off,
                      std::vector<unsigned int>& cum, const std::vector<unsigned char>& eff) {
        std::vector<unsigned int> idx;
        std::vector<unsigned int> tmp;
        for (size_t i = 0; i < np; i++) {
            const unsigned int b = off[i], n = off[i + 1] - off[i];
            idx.resize(n);
            for (unsigned int q = 0; q < n; q++) idx[q] = q;
            std::stable_sort(idx.begin(), idx.end(), [&](unsigned int a, unsigned int c) {
                return eff[b + a] < eff[b + c];
            });
            tmp.assign(slot.begin() + b, slot.begin() + b + n);
            for (unsigned int q = 0; q < n; q++) slot[b + q] = tmp[idx[q]];
            for (unsigned int c = 0; c < nCls; c++) cum[i * nCls + c] = 0;
            for (unsigned int q = 0; q < n; q++)
                if (eff[b + q] < nCls) cum[i * nCls + eff[b + q]]++;
            for (unsigned int c = 1; c < nCls; c++) cum[i * nCls + c] += cum[i * nCls + c - 1];
        }
    };
    if (ts::EnutsProf::gXch()) {
        m_uiOwnRecvSlotU = m_uiOwnRecvSlot;
        m_uiOwnRecvCumU  = m_uiOwnRecvCum;
        m_uiOwnSendSlotU = m_uiOwnSendSlot;
        m_uiOwnSendCumU  = m_uiOwnSendCum;
    }
    resort(rPr.size(), m_uiOwnRecvSlot, m_uiOwnRecvOff, m_uiOwnRecvCum, rEff);
    resort(sPr.size(), m_uiOwnSendSlot, m_uiOwnSendOff, m_uiOwnSendCum, sEff);
    if (std::getenv("DENDRO_ENUTS_OWNER_SYNC_INFO")) {
        size_t never = 0;
        for (const unsigned char e : rEff) never += (e == 255);
        std::cout << "[enuts-owner-sync r" << pMesh->getMPIRankGlobal() << "] scoped send lists: recv nodes "
                  << rEff.size() << " never read " << never << std::endl;
    }
    m_uiXchScoped = true;
    return true;
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::xch_calibrate(ot::Mesh* pMesh) {
    // XCH_TRACE: per receive partner the offset of its clock from mine, from
    // round trips (stamp each way, then the measured one-way values each
    // way; the sample with the least latency wins). partners I only receive
    // from stay NaN.
    const std::vector<unsigned int>& rPr = pMesh->getRecvProcList();
    const std::vector<unsigned int>& sPr = pMesh->getSendProcList();
    MPI_Comm comm                        = pMesh->getMPICommunicator();
    const size_t nr                      = rPr.size();
    m_uiXchOff.assign(nr, std::nan(""));
    m_uiXchLat.assign(nr, std::nan(""));
    m_uiXchSpread.assign(nr, std::nan(""));
    m_uiXchOffRank.assign(rPr.begin(), rPr.end());
    std::vector<int> both(nr, 0);
    for (size_t i = 0; i < nr; i++)
        both[i] = std::find(sPr.begin(), sPr.end(), rPr[i]) != sPr.end();
    std::vector<double> st(nr, 0), mine(nr, 0), tr(nr, 0), dm(nr, 0), dp(nr, 0), best(nr, 1e300);
    const int nRound = 64;
    std::vector<double> latS(nr * nRound, 1e300), offS(nr * nRound, 0);
    std::vector<MPI_Request> rq(2 * nr, MPI_REQUEST_NULL);
    for (int round = 0; round < nRound; round++) {
        for (size_t i = 0; i < nr; i++)
            if (both[i]) MPI_Irecv(&st[i], 1, MPI_DOUBLE, rPr[i], 7154, comm, &rq[i]);
        for (size_t i = 0; i < nr; i++)
            if (both[i]) {
                mine[i] = ts::EnutsProf::now();
                MPI_Isend(&mine[i], 1, MPI_DOUBLE, rPr[i], 7154, comm, &rq[nr + i]);
            }
        for (size_t k = 0; k < 2 * nr; k++) {
            int idx = MPI_UNDEFINED;
            MPI_Waitany((int)(2 * nr), rq.data(), &idx, MPI_STATUS_IGNORE);
            if (idx == MPI_UNDEFINED) break;
            if (idx < (int)nr) tr[idx] = ts::EnutsProf::now();
        }
        for (size_t i = 0; i < nr; i++) dm[i] = tr[i] - st[i];  // latency - offset
        for (size_t i = 0; i < nr; i++)
            if (both[i]) {
                MPI_Irecv(&dp[i], 1, MPI_DOUBLE, rPr[i], 7155, comm, &rq[i]);
                MPI_Isend(&dm[i], 1, MPI_DOUBLE, rPr[i], 7155, comm, &rq[nr + i]);
            }
        MPI_Waitall((int)(2 * nr), rq.data(), MPI_STATUSES_IGNORE);
        for (size_t i = 0; i < nr; i++) {
            if (!both[i]) continue;
            const double lat          = 0.5 * (dm[i] + dp[i]);
            latS[i * nRound + round] = lat;
            offS[i * nRound + round] = 0.5 * (dp[i] - dm[i]);
            if (lat < best[i]) best[i] = lat;
        }
    }
    // the offset: median over the samples within 2 us of the least latency
    // (a descheduled poll inflates one direction); spread = their range
    for (size_t i = 0; i < nr; i++) {
        if (!both[i]) continue;
        std::vector<double> good;
        for (int k = 0; k < nRound; k++)
            if (latS[i * nRound + k] <= best[i] + 2e-6) good.push_back(offS[i * nRound + k]);
        std::sort(good.begin(), good.end());
        m_uiXchOff[i]    = good[good.size() / 2];
        m_uiXchLat[i]    = best[i];
        m_uiXchSpread[i] = good.back() - good.front();
    }
    m_uiXchT0  = ts::EnutsProf::now();
    m_uiXchCal = true;
    m_uiXchRec.clear();
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::xch_dump() {
    const char* dir = ts::EnutsProf::xchTraceDir();
    if (!dir || m_uiXchRec.empty()) return;
    int rk = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rk);
    std::ofstream f(std::string(dir) + "/enuts_xch_r" + std::to_string(rk) + ".txt");
    // XCH_TRACE: the calibration instant in this rank's clock, so entry times (t_wait_enter, since
    // it) chain into one clock across ranks through the partner offsets
    f << "# t0 " << std::setprecision(12) << m_uiXchT0 << "\n";
    f << "# clock offsets (partner rank: partner clock - mine, s; nan = no round trip; then the best\n"
         "# one-way latency of the 64 samples and the range of the offsets among the low-latency\n"
         "# samples, s; then the largest lateness - wait seen with that partner as arg-max (> 0:\n"
         "# the offset is at least that much too large; the corrected sums subtract it)\n";
    std::map<unsigned int, double> bad;
    for (const XchRec& r : m_uiXchRec)
        if (!std::isnan(r.late) && r.late - r.wait > bad[r.arg]) bad[r.arg] = r.late - r.wait;
    for (size_t i = 0; i < m_uiXchOff.size(); i++)
        f << "# off " << m_uiXchOffRank[i] << " " << std::setprecision(9) << m_uiXchOff[i] << " "
          << m_uiXchLat[i] << " " << m_uiXchSpread[i] << " " << bad[m_uiXchOffRank[i]] << "\n";
    f << "# call pt stage j t_wait_enter nrecv_doubles wait_s last_recv_s first_recv_s argmax_partner "
         "argmax_kind(r/s) argmax_late_s argmax_doubles max_late_s max_late_partner\n";
    std::map<unsigned int, std::pair<unsigned int, double>> hist;
    double sw = 0, sl = 0, slm = 0, slc = 0;
    unsigned int nbad = 0;
    for (size_t k = 0; k < m_uiXchRec.size(); k++) {
        const XchRec& r = m_uiXchRec[k];
        f << k << " " << r.pt << " " << r.s + 1 << " " << r.j << " " << std::setprecision(9)
          << r.tw << " " << std::setprecision(6) << r.nrecv << " " << r.wait << " " << r.rwait << " " << r.first
          << " " << (int)r.arg << " " << (r.argSend ? 's' : 'r') << " " << r.late << " "
          << r.argN << " " << r.lateMax << " " << (int)r.lateArg << "\n";
        hist[r.arg].first++;
        hist[r.arg].second += r.wait;
        sw += r.wait;
        if (!std::isnan(r.late)) {
            sl += std::max(0.0, r.late);
            slc += std::max(0.0, std::min(r.late - bad[r.arg], r.wait));
            nbad += (r.late > r.wait + 1e-4);
        }
        if (!std::isnan(r.lateMax)) slm += std::max(0.0, r.lateMax);
    }
    std::vector<std::pair<unsigned int, std::pair<unsigned int, double>>> hv(hist.begin(), hist.end());
    std::sort(hv.begin(), hv.end(), [](const auto& a, const auto& b) {
        return a.second.second > b.second.second;
    });
    f << "# summary calls " << m_uiXchRec.size() << " wait " << sw << " argmax_late_pos " << sl
      << " argmax_late_corrected " << slc << " max_late_pos " << slm << " late_gt_wait_calls " << nbad
      << "\n";
    f << "# argmax partner: calls wait_s (by wait)\n";
    for (size_t i = 0; i < hv.size() && i < 12; i++)
        f << "# arg " << (int)hv[i].first << " " << hv[i].second.first << " " << hv[i].second.second
          << "\n";
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::sync_lazy_build(bool ownEid) {
    ot::Mesh* pMesh          = (ot::Mesh*)m_uiAppCtx->get_mesh();
    const ot::Block* blkList = pMesh->getLocalBlockList().data();
    const unsigned int nPe   = pMesh->getNumNodesPerElement();
    if (m_uiBlkDepEid.size() != pMesh->getLocalBlockList().size()) {
        m_uiBlkDepEid.assign(pMesh->getLocalBlockList().size(), {});
        for (unsigned int b = 0; b < m_uiBlkDepEid.size(); b++)
            computeBlockUnzipDepElements(pMesh, b, m_uiBlkDepEid[b]);
    }
    if (ownEid && m_uiBlkOwnEid.size() != m_uiBlkDepEid.size()) {
        const unsigned int* e2n_dg = pMesh->getE2NMapping_DG().data();
        m_uiBlkOwnEid.assign(m_uiBlkDepEid.size(), {});
        std::vector<unsigned char> seen(pMesh->getAllElements().size(), 0);
        for (unsigned int b = 0; b < m_uiBlkOwnEid.size(); b++) {
            std::vector<unsigned int>& oe = m_uiBlkOwnEid[b];
            oe = m_uiBlkDepEid[b];
            for (const unsigned int e : oe) seen[e] = 1;
            for (const unsigned int e : blkList[b]) seen[e] = 1;
            const size_t nd = oe.size();
            auto addOwners = [&](unsigned int e) {
                for (unsigned int q = 0; q < nPe; q++) {
                    const unsigned int o = e2n_dg[e * nPe + q] / nPe;
                    if (!seen[o]) {
                        seen[o] = 1;
                        oe.push_back(o);
                    }
                }
            };
            for (size_t m = 0; m < nd; m++) addOwners(oe[m]);
            for (const unsigned int e : blkList[b]) addOwners(e);
            for (const unsigned int e : oe) seen[e] = 0;
            for (const unsigned int e : blkList[b]) seen[e] = 0;
        }
    }
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::sync_prebuild(bool ownPlan, bool ownEid,
                                         bool syncFast) {
    ot::Mesh* pMesh         = (ot::Mesh*)m_uiAppCtx->get_mesh();
    const unsigned int nBlk = pMesh->getLocalBlockList().size();
    sync_lazy_build(ownEid);
    // building a plan appends to tables all blocks share (p2c store indices,
    // rows of the coarser dependents): all blocks here, once per mesh, so a
    // block call inside the parallel region only reads them
    if (ownPlan) {
        if (m_uiOwnPlan.size() != nBlk) m_uiOwnPlan.assign(nBlk, OwnPlan());
        for (unsigned int b = 0; b < nBlk; b++)
            if (!m_uiOwnPlan[b].built)
                build_own_plan(b, m_uiBlkDepEid[b], m_uiOwnPlan[b]);
    } else if (syncFast) {
        if (m_uiSyncPlan.size() != nBlk) m_uiSyncPlan.assign(nBlk, SyncPlan());
        for (unsigned int b = 0; b < nBlk; b++)
            if (!m_uiSyncPlan[b].built)
                build_sync_plan(b, m_uiBlkDepEid[b], m_uiSyncPlan[b]);
    }
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::free_sync_threads() {
    for (SyncThr& st : m_uiSyncThr) delete st.op;
    m_uiSyncThr.clear();
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::sync_thread_setup(bool cgScratch) {
#ifdef DENDRO_ENUTS_HAS_OMP
    ot::Mesh* pMesh = (ot::Mesh*)m_uiAppCtx->get_mesh();
    const size_t nt = (size_t)omp_get_max_threads();
    const size_t dof = m_uiEVar.get_dof();
    const size_t nPe = pMesh->getNumNodesPerElement();
    const size_t dgN = dof * pMesh->getAllElements().size() * nPe;
    const size_t cgN = dof * pMesh->getDegOfFreedom();
    if (m_uiSyncThr.size() != nt) {
        free_sync_threads();
        m_uiSyncThr.resize(nt);
    }
    // slot 0 stays empty: thread 0 uses the shared members
    for (size_t t = 1; t < nt; t++) {
        SyncThr& st = m_uiSyncThr[t];
        if (!st.op) st.op = new ENUTSOp(m_uiType);
        if (!st.dgW) st.dgW.reset(new T[dgN]);
        if (cgScratch && !st.cgW) st.cgW.reset(new T[cgN]);
        // mirrors of the shared per-pass caches, sized like them
        if (st.p2cPassOf.size() != m_uiP2cPassOf.size()) {
            st.p2cPassOf.assign(m_uiP2cPassOf.size(), 0);
            st.p2cSlotOf.assign(m_uiP2cPassOf.size(), 0);
            st.p2cTagOf.assign(m_uiP2cPassOf.size(), SyncTag());
            st.p2cUsed     = 0;
            st.p2cPoolPass = 0;
        }
        if (st.ownConv.size() != m_uiOwnConv.size()) {
            st.ownConv.assign(m_uiOwnConv.size(), OwnConv());
            st.ownConvPass.assign(m_uiOwnConvPass.size(), 0);
            st.ownRowBufRow = ~0u;
        }
        if (st.ownScr.size() != m_uiOwnScr.size())
            st.ownScr.assign(m_uiOwnScr.size(), 0);
    }
#endif
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::sync_blk_timestep(unsigned int blk,
                                             unsigned int rk_s,
                                             bool isIGTSync) {
#ifdef __PROFILE_ENUTS__
    m_uiPt[ENUTSPROFILE::BLK_SYNC].start();
#endif

    ot::Mesh* pMesh = (ot::Mesh*)m_uiAppCtx->get_mesh();
    if ((!(pMesh->isActive())) || m_uiBVec[blk]._vec[stage_slot(rk_s)].isSynced()) return;
    const bool sprof = ts::EnutsProf::on();
    double tcorrCf = 0, tcorrFc = 0, tsy0 = 0;
    if (sprof) tsy0 = ts::EnutsProf::now();
    if (sprof) ts::EnutsProf::get().t[ts::EnutsProf::CN_CALLS] += 1;

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
    const DendroIntL offset   = blkList[blk].getOffset();

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
    const int ownSync   = (rk_s >= 1 && !isIGTSync) ? ts::EnutsProf::ownerSync() : 0;
    // ownSync 1 runs the plan (below, once the dependents are known); here it
    // is the per-element form: modes 2 / 3, or 1 where the plan is unavailable
    const bool syncFast = ts::EnutsProf::syncFast() && !ownSync;
    const bool ifxEq  = (ts::EnutsProf::ifaceFix() & 1) && rk_s >= 1 && !isIGTSync;
    const bool ifxAll = (ts::EnutsProf::ifaceFix() & 2) && rk_s >= 1 && !isIGTSync;
    // [ENUTS_OMP edit 1/5 begin] worker threads (tid >= 1) of the threaded sync
    // loop get their own scratch: the locals below shadow the shared members
    // for the rest of this function (p2c_slot and own_sync_fill do the same).
    // thr_ == nullptr = serial / thread 0.
    SyncThr* const thr_ =
        ts::EnutsProf::tid() ? &m_uiSyncThr[ts::EnutsProf::tid()] : nullptr;
    ENUTSOp* const m_uiECOp           = thr_ ? thr_->op : this->m_uiECOp;
    std::vector<T>& m_uiSyncCVec      = thr_ ? thr_->cVec : this->m_uiSyncCVec;
    std::vector<SyncTag>& m_uiSyncTag = thr_ ? thr_->tag : this->m_uiSyncTag;
    std::vector<T>& m_uiOwnSweep      = thr_ ? thr_->ownSweep : this->m_uiOwnSweep;
    std::vector<unsigned char>& m_uiOwnSweepOk = thr_ ? thr_->ownSweepOk : this->m_uiOwnSweepOk;
    // [ENUTS_OMP edit 1/5 end]
    std::vector<T> cVecLocal;
    std::vector<T>& cVec = syncFast ? m_uiSyncCVec : cVecLocal;
    if (cVec.size() < (size_t)dof * nPe * (m_uiNumStages + 1))
        cVec.resize(dof * nPe * (m_uiNumStages + 1));
    if (syncFast && m_uiSyncTag.size() != pMesh->getAllElements().size())
        m_uiSyncTag.assign(pMesh->getAllElements().size(), SyncTag());

    T* cVin[(m_uiNumStages + 1) * dof];   // correction vec. pointer in,
    T* cVout[(m_uiNumStages + 1) * dof];  // correction vec. pointer out,

    // [ENUTS_OMP edit 2/5] per-thread stand-in for the shared work vector
    T* dgWVec  = thr_ ? thr_->dgW.get() : m_uiDGTmp0.get_vec_ptr();
    // T * cgWVec = m_uiEVecTmp[0].get_vec_ptr();
    T* uzWVec  = m_uiEvarUzip.get_vec_ptr();

    T* m_uiVec = (T*)m_uiBVec[blk]._vec[stage_slot(rk_s)].data();

    T* dgStages[m_uiNumStages];
    for (unsigned int i = 0; i < m_uiNumStages; i++)
        dgStages[i] = m_uiStVec[i].get_vec_ptr();

    T* dgEVar = m_uiEvarDG.get_vec_ptr();

    // [ENUTS_OMP edit 3/5] the m_uiBlkDepEid build (here) and the
    // m_uiBlkOwnEid build (below) moved verbatim to sync_lazy_build(); the
    // threaded sync loop runs sync_prebuild() before the parallel region, so
    // in a worker these are read-only size checks. any new table or plan
    // built on first use belongs in sync_prebuild().
    sync_lazy_build(false);
    const std::vector<unsigned int>& eid = m_uiBlkDepEid[blk];
    if (sprof) ts::EnutsProf::get().t[ts::EnutsProf::CN_EID] += eid.size();
    if (ownSync == 1 && m_uiSyncPassOpen && prepare_owner_sync(pMesh)) {
        own_sync_fill(blk, rk_s, eid);
        if (sprof) ts::EnutsProf::get().t[ts::EnutsProf::SY_BLK] += ts::EnutsProf::now() - tsy0;
#ifdef __PROFILE_ENUTS__
        m_uiPt[ENUTSPROFILE::BLK_SYNC].stop();
#endif
        return;
    }
    if (ownSync) sync_lazy_build(true);  // [ENUTS_OMP edit 3/5, second part]
    if (!thr_ && m_uiEvarDG.get_vec_ptr() == nullptr) ensure_evar_dg();  // LAZY_TMP, serial path
    // elements whose stage is brought to this block's time frame
    const std::vector<unsigned int>& cel = ownSync ? m_uiBlkOwnEid[blk] : eid;

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
                if (!syncFast) {
#pragma unroll
                    for (unsigned int v = 0; v < dof; v++)
                        std::memcpy(dgWVec + v * dgSz + elem * nPe,
                                    dgEVar + v * dgSz + elem * nPe,
                                    sizeof(T) * nPe);
                }

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

                { const double tc0_ = sprof ? ts::EnutsProf::now() : 0.0;
 m_uiECOp->coarser_finer_ut_correction(cVout, (const T**)cVin,
                                                      cSz, dt_c, dt_f, dt, dof);
 if (sprof) tcorrCf += ts::EnutsProf::now() - tc0_; }

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

                { const double tc0_ = sprof ? ts::EnutsProf::now() : 0.0;
 m_uiECOp->finer_coarser_ut_correction(cVout, (const T**)cVin,
                                                      cSz, dt_c, dt_f, dt, dof);
 if (sprof) tcorrFc += ts::EnutsProf::now() - tc0_; }

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
        for (unsigned int m = 0; m < cel.size(); m++) {
            const unsigned int elem = cel[m];
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

                    { const double tc0_ = sprof ? ts::EnutsProf::now() : 0.0;
 m_uiECOp->Ccf(cVout, (const T**)cVin, cSz, rk_s, dt_c, dt_f,
                                  dt, dof);
 if (sprof) tcorrCf += ts::EnutsProf::now() - tc0_; }

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

                    { const double tc0_ = sprof ? ts::EnutsProf::now() : 0.0;
 m_uiECOp->Cfc(cVout, (const T**)cVin, cSz, rk_s, dt_c, dt_f,
                                  dt, dof);
 if (sprof) tcorrFc += ts::EnutsProf::now() - tc0_; }

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
                if (bTime == tl && !(ifxEq && dtl != bDT)) {
                    // fast: the padding fill below reads the stage vector directly
                    if (!syncFast) {
#pragma unroll
                        for (unsigned int v = 0; v < dof; v++)
                            std::memcpy(dgWVec + v * dgSz + elem * nPe,
                                        dgStages[rk_s - 1] + v * dgSz + elem * nPe,
                                        sizeof(T) * nPe);
                    }

                } else {
                    // tl > bTime: elem is in the future (Ccf), else blk is (Cfc)
                    // eqT (ifxEq only): same time but a different step, so the
                    // element's stage k_s was built with its own dt; convert it
                    // with a zero time shift (Ccf from a coarser element)
                    const bool eqT    = (tl == bTime);
                    const bool eleFut = eqT ? (dtl > bDT) : (tl > bTime);
                    assert(eqT || eleFut || tl < bTime);
                    // ifxAll: an element ahead of the block has finished its
                    // step, so all of its stages enter the time shift
                    const unsigned int nsIn =
                        (ifxAll && eleFut && !eqT) ? m_uiNumStages : rk_s;
                    T dt_c = (eleFut ? dtl : bDT) * (m_uiTimeInfo._m_uiTh);
                    T dt_f = (eleFut ? bDT : dtl) * (m_uiTimeInfo._m_uiTh);

                    if (dt_c < dt_f) std::swap(dt_c, dt_f);

                    dt = eqT ? 0 : (eleFut ? (t_ele - t_blk) : (t_blk - t_ele));
                    if (ts::EnutsProf::ifaceShift() && eleFut && !eqT)
                        dt = (T)(bTime + dtl - tl) * m_uiTimeInfo._m_uiTh;

                    if (syncFast && m_uiSyncPassOpen &&
                        m_uiSyncTag[elem].pass == m_uiSyncPass &&
                        m_uiSyncTag[elem].rk == rk_s &&
                        m_uiSyncTag[elem].fut == (unsigned char)eleFut &&
                        m_uiSyncTag[elem].dt_c == dt_c &&
                        m_uiSyncTag[elem].dt_f == dt_f &&
                        m_uiSyncTag[elem].dt == dt) {
                        // another block of this stage pass already left the
                        // same correction in dgWVec
                        if (sprof) ts::EnutsProf::get().t[ts::EnutsProf::CN_HIT] += 1;
                    } else {
                        for (unsigned int v = 0; v < dof; v++)
                            for (unsigned int s = 1; s <= nsIn; s++) {
                                cVin[v * nsIn + (s - 1)] =
                                    dgStages[s - 1] + v * dgSz + elem * nPe;
                                cVout[v * nsIn + (s - 1)] =
                                    cVec.data() + v * nsIn * nPe + (s - 1) * nPe;
                            }

                        if (sprof) ts::EnutsProf::get().t[ts::EnutsProf::CN_CORR] += 1;
                        {
                            const double tc0_ = sprof ? ts::EnutsProf::now() : 0.0;
                            if (m_uiStGOn && (eqT || nsIn == m_uiNumStages)) {
                                // nonlinear form; g of an element's previous
                                // step where its k3, k4 do not exist yet
                                T row[5], m5[5];
                                const T* in5[5];
                                const bool useG = m_uiECOp->stageRowNL(
                                    eleFut, row, rk_s, dt_c, dt_f, dt, nsIn);
                                for (unsigned int k = 0; k < nsIn; k++) m5[k] = row[k];
                                if (useG) m5[nsIn] = row[4];
                                for (unsigned int v = 0; v < dof; v++) {
                                    for (unsigned int k = 0; k < nsIn; k++)
                                        in5[k] = cVin[v * nsIn + k];
                                    if (useG)
                                        in5[nsIn] = m_uiStG.get_vec_ptr() +
                                                    (size_t)v * dgSz + (size_t)elem * nPe;
                                    m_uiECOp->applyRow(cVout[v * nsIn + (rk_s - 1)], in5,
                                                       m5, nsIn + (useG ? 1 : 0), nPe);
                                }
                            } else if (eleFut)
                                m_uiECOp->Ccf(cVout, (const T**)cVin, cSz, rk_s,
                                              dt_c, dt_f, dt, dof, nsIn);
                            else
                                m_uiECOp->Cfc(cVout, (const T**)cVin, cSz, rk_s,
                                              dt_c, dt_f, dt, dof, nsIn);
                            if (sprof) (eleFut ? tcorrCf : tcorrFc) += ts::EnutsProf::now() - tc0_;
                        }

#pragma unroll
                        for (unsigned int v = 0; v < dof; v++)
                            std::memcpy(dgWVec + v * dgSz + elem * nPe,
                                        cVout[v * nsIn + (rk_s - 1)],
                                        sizeof(T) * nPe);
                        if (syncFast && m_uiSyncPassOpen)
                            m_uiSyncTag[elem] = {m_uiSyncPass, rk_s,
                                                 (unsigned char)eleFut, dt_c,
                                                 dt_f, dt};
                    }
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

    if (ownSync) {
        // dgWVec holds the stage of the dependents and of the owners of their
        // nodes in this block's time frame; add the block's own elements, give
        // every cg node its owner's value and read each element back through
        // the cg vector (what zip + unzip does for a stage in ETS)
        const unsigned int* e2n_dg = pMesh->getE2NMapping_DG().data();
        // [ENUTS_OMP edit 4/5] per-thread stand-in for the shared cg scratch
        if (!thr_) ensure_cg_tmp();  // LAZY_TMP: thread 0 / serial only
        T* cgW = thr_ ? thr_->cgW.get() : m_uiEVecTmp[0].get_vec_ptr();
        std::vector<unsigned int> es(eid.begin(), eid.end());
        for (const unsigned int elem : blkList[blk]) {
            es.push_back(elem);
            for (unsigned int v = 0; v < dof; v++)
                std::memcpy(dgWVec + v * dgSz + elem * nPe,
                            dgStages[rk_s - 1] + v * dgSz + elem * nPe,
                            sizeof(T) * nPe);
        }
        if (ownSync == 2) {
            for (size_t q = 0; q < (size_t)dof * cgSz; q++)
                cgW[q] = std::numeric_limits<T>::quiet_NaN();
            // nan left by the rhs itself (padding outside the domain) is not ours
            for (size_t q = 0; q < (size_t)dof * lx * ly * lz; q++)
                if (std::isnan(m_uiVec[q])) m_uiVec[q] = 0;
        }
        for (const unsigned int ele : es)
            for (unsigned int q = 0; q < nPe; q++) {
                const unsigned int cg = e2n[ele * nPe + q];
                const unsigned int dg = e2n_dg[ele * nPe + q];
                for (unsigned int v = 0; v < dof; v++)
                    cgW[v * cgSz + cg] = dgWVec[v * dgSz + dg];
            }
        for (const unsigned int ele : es)
            for (unsigned int v = 0; v < dof; v++)
                pMesh->getElementNodalValues(cgW + v * cgSz,
                                             dgWVec + v * dgSz + ele * nPe, ele);
    }

    double tun0 = 0;
    if (sprof) {
        tun0 = ts::EnutsProf::now();
        ts::EnutsProf::get().t[ts::EnutsProf::SY_CORR] += tcorrCf;
        ts::EnutsProf::get().t[ts::EnutsProf::SY_CORR_FC] += tcorrFc;
        ts::EnutsProf::get().t[ts::EnutsProf::SY_SAME] += (tun0 - tsy0) - tcorrCf - tcorrFc;
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

    if (syncFast) {
        if (m_uiSyncPlan.size() != pMesh->getLocalBlockList().size())
            m_uiSyncPlan.assign(pMesh->getLocalBlockList().size(), SyncPlan());
        SyncPlan& pl = m_uiSyncPlan[blk];
        if (!pl.built) build_sync_plan(blk, eid, pl);
        const size_t lxyz = (size_t)lx * ly * lz;
        if (sprof) {
            ts::EnutsProf::get().t[ts::EnutsProf::CN_ENT] += pl.ents.size();
            ts::EnutsProf::get().t[ts::EnutsProf::CN_PAIRS] += pl.src.size();
            unsigned int nc = 0;
            for (const typename SyncPlan::Ent& en : pl.ents) nc += (en.cnum >= 0);
            ts::EnutsProf::get().t[ts::EnutsProf::CN_P2C] += nc;
        }
        const size_t p2cStride = (size_t)dof * nPe;
        const bool useP2cCache = m_uiSyncPassOpen && !isIGTSync && p2cStride > 0;
        // sweeps of the 3D interpolation shared by the children of one source row
        if (m_uiOwnSweep.size() < 6 * p2cStride) m_uiOwnSweep.resize(6 * p2cStride);
        if (m_uiOwnSweepOk.size() != (size_t)6 * dof) m_uiOwnSweepOk.assign((size_t)6 * dof, 0);
        unsigned int swEle = ~0u;
        const T* swBase    = nullptr;
        for (const typename SyncPlan::Ent& en : pl.ents) {
            const T* srcBase = dgWVec;
            const bool sameTime = !isIGTSync && etVec[en.ele] == bTime &&
                                  !(ifxEq && dtELev[en.ele] != bDT);
            if (sameTime)
                srcBase = (rk_s == 0) ? (const T*)dgEVar : (const T*)dgStages[rk_s - 1];
            const unsigned int* sp = pl.src.data();
            const unsigned int* dp = pl.dst.data();
            if (en.cnum < 0) {
                for (unsigned int v = 0; v < dof; v++) {
                    T* dv       = m_uiVec + v * lxyz;
                    const T* sv = srcBase + (size_t)v * dgSz + (size_t)en.ele * nPe;
                    for (unsigned int q = en.b; q < en.e; q++) dv[dp[q]] = sv[sp[q]];
                }
                continue;
            }
            T* slotData = nullptr;
            bool hit    = false;
            if (useP2cCache) {
                SyncTag key;
                if (sameTime) {
                    key.pass = m_uiSyncPass;
                    key.rk   = rk_s;
                    key.fut  = 2;
                } else {
                    key = m_uiSyncTag[en.ele];
                }
                // nullptr: the pass store is full, interpolate into scratch
                slotData = p2c_slot(en.store, key, p2cStride, hit);
            }
            if (sprof) ts::EnutsProf::get().t[ts::EnutsProf::CN_P2C_HIT] += hit;
            for (unsigned int v = 0; v < dof; v++) {
                T* dv       = m_uiVec + v * lxyz;
                const T* sv = srcBase + (size_t)v * dgSz + (size_t)en.ele * nPe;
                T* pv       = slotData ? slotData + (size_t)v * nPe : p2cI.data();
                if (!hit) {
                    const double tp2 = sprof ? ts::EnutsProf::now() : 0.0;
                    if (m_uiDim == 3) {
                        if (v == 0 && (swEle != en.ele || swBase != srcBase)) {
                            std::fill(m_uiOwnSweepOk.begin(), m_uiOwnSweepOk.end(), 0);
                            swEle  = en.ele;
                            swBase = srcBase;
                        }
                        p2c3_shared(pMesh, sv, pv, en.cnum,
                                    m_uiOwnSweep.data() + (size_t)v * 6 * nPe,
                                    m_uiOwnSweepOk.data() + (size_t)v * 6);
                    } else {
                        pMesh->parent2ChildInterpolation(sv, pv, en.cnum, m_uiDim);
                    }
                    if (sprof) ts::EnutsProf::get().t[ts::EnutsProf::SY_P2C] += ts::EnutsProf::now() - tp2;
                }
                for (unsigned int q = en.b; q < en.e; q++) dv[dp[q]] = pv[sp[q]];
            }
        }
    } else
    for (unsigned int m = 0; m < eid.size(); m++) {
        const unsigned int ele = eid[m];
        const T* srcv          = dgWVec;
        if (syncFast && !isIGTSync && etVec[ele] == bTime &&
            !(ifxEq && dtELev[ele] != bDT))
            srcv = (rk_s == 0) ? (const T*)dgEVar : (const T*)dgStages[rk_s - 1];

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
                            // jjy*lx + iix]; double v2 = srcv[v*dgSz +
                            // ele*nPe + k*(eOrder+1)*(eOrder+1)+ j*(eOrder+1) +
                            // i]; if(rk_s==0 && fabs(v2-v1)>1e-3)
                            // {
                            //     std::cout<<"var : "<<v<<" equal level copy:
                            //     "<<blk<<" old : "<<v1<<" new :
                            //     "<<v2<<std::endl;
                            // }
                            m_uiVec[(v * lx * ly * lz) + kkz * lx * ly +
                                    jjy * lx + iix] =
                                srcv[v * dgSz + ele * nPe +
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
                            // jjy*lx + iix]; double v2 = srcv[v*dgSz +
                            // ele*nPe + k*(eOrder+1)*(eOrder+1)+ j*(eOrder+1) +
                            // i]; if(rk_s==0 && fabs(v2-v1)>1e-3)
                            // {
                            //     std::cout<<"var : "<<v<<" copy from finer
                            //     octant: "<<blk<<" old : "<<v1<<" new :
                            //     "<<v2<<std::endl;
                            // }
                            m_uiVec[(v * lx * ly * lz) + kkz * lx * ly +
                                    jjy * lx + iix] =
                                srcv[v * dgSz + ele * nPe +
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
                        &srcv[v * dgSz + ele * nPe], p2cI.data(), cnum,
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
    double town0 = 0;
    if (sprof) {
        town0 = ts::EnutsProf::now();
        ts::EnutsProf::get().t[ts::EnutsProf::SY_UNZIP] += town0 - tun0;
    }
    if (syncFast) {
        const T* ownSrc = (rk_s == 0) ? (const T*)dgEVar : (const T*)dgStages[rk_s - 1];
        const unsigned int n1 = eOrder + 1;
        for (const unsigned int elem : blkList[blk]) {
            const unsigned int ei = (pNodes[elem].getX() - blkNode.getX()) >>
                                    (m_uiMaxDepth - regLevel);
            const unsigned int ej = (pNodes[elem].getY() - blkNode.getY()) >>
                                    (m_uiMaxDepth - regLevel);
            const unsigned int ek = (pNodes[elem].getZ() - blkNode.getZ()) >>
                                    (m_uiMaxDepth - regLevel);
            assert(etVec[elem] == bTime);
            for (unsigned int v = 0; v < dof; v++) {
                const T* sv = ownSrc + (size_t)v * dgSz + (size_t)elem * nPe;
                T* dv       = m_uiVec + (size_t)v * lx * ly * lz;
                for (unsigned int k = 0; k < n1; k++)
                    for (unsigned int j = 0; j < n1; j++)
                        std::memcpy(dv + (size_t)(ek * eOrder + k + PW) * (ly * lx) +
                                        (size_t)(ej * eOrder + j + PW) * lx +
                                        (ei * eOrder + PW),
                                    sv + k * n1 * n1 + j * n1, sizeof(T) * n1);
            }
        }
    } else if (rk_s == 0) {
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
                                (ownSync ? dgWVec : dgStages[rk_s - 1])
                                        [v * dgSz + elem * nPe +
                                         k * (eOrder + 1) * (eOrder + 1) +
                                         j * (eOrder + 1) + i];
        }
    }

    if (ownSync == 2) {
        // [ENUTS_OMP edit 5/5] atomic: blocks may run concurrently
        static std::atomic<unsigned long long> nanTot{0}, calls{0};
        unsigned long long nn = 0;
        for (size_t q = 0; q < (size_t)dof * lx * ly * lz; q++)
            nn += std::isnan(m_uiVec[q]);
        nanTot += nn;
        if (nn && calls++ < 20)
            std::cout << "[owner-sync] blk " << blk << " rk " << rk_s << " lev " << bLev
                      << " nan " << nn << " of " << (size_t)dof * lx * ly * lz
                      << " total " << nanTot << std::endl;
    }
    if (sprof) {
        const double tend = ts::EnutsProf::now();
        ts::EnutsProf::get().t[ts::EnutsProf::SY_OWN] += tend - town0;
        ts::EnutsProf::get().t[ts::EnutsProf::SY_BLK] += tend - tsy0;
    }
#ifdef __PROFILE_ENUTS__
    m_uiPt[ENUTSPROFILE::BLK_SYNC].stop();
#endif
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
bool ExplicitNUTS<T, Ctx>::prepare_scoped_exchange(ot::Mesh* pMesh) {
    if (m_uiScopedMesh == pMesh) return m_uiScopedOk;
    m_uiScopedMesh = pMesh;
    m_uiScopedOk   = false;
    m_uiScopedPlan = ot::EleDGScopedPlan();
    m_uiScopedDTs.clear();
    if (!pMesh->isActive()) return false;
    // one active rank: no element ghost scatter maps exist, nothing to scope
    if (pMesh->getMPICommSize() < 2) return false;

    // ascending distinct timestep factors; they must form a divisibility chain
    // so the elements firing at pt (DT | pt) are a prefix of the sorted segment
    for (unsigned int l = m_uiLevMin; l <= m_uiLevMax; l++)
        m_uiScopedDTs.push_back(
            step_ratio(l));
    std::sort(m_uiScopedDTs.begin(), m_uiScopedDTs.end());
    m_uiScopedDTs.erase(
        std::unique(m_uiScopedDTs.begin(), m_uiScopedDTs.end()),
        m_uiScopedDTs.end());
    bool ok = !m_uiScopedDTs.empty() && m_uiScopedDTs[0] >= 1;
    for (size_t i = 1; ok && i < m_uiScopedDTs.size(); i++)
        ok = (m_uiScopedDTs[i] % m_uiScopedDTs[i - 1] == 0);

    // element -> DT by its own level; must be in range, and every local block
    // must be single-level (activity is decided by the block's first element)
    const std::vector<ot::TreeNode>& all = pMesh->getAllElements();
    std::vector<unsigned int> dtOf(all.size(), 0);
    for (size_t e = 0; ok && e < all.size(); e++) {
        const unsigned int l = all[e].getLevel();
        if (l < m_uiLevMin || l > m_uiLevMax) {
            ok = false;
            break;
        }
        dtOf[e] = step_ratio(l);
    }
    const std::vector<ot::Block>& blkList = pMesh->getLocalBlockList();
    for (unsigned int b = 0; ok && b < blkList.size(); b++) {
        const unsigned int bl = all[blkList[b].firstElementID()].getLevel();
        for (const unsigned int e : blkList[b])
            if (all[e].getLevel() != bl) {
                ok = false;
                break;
            }
    }
    // all ranks must agree (Begin/End are collective in effect)
    int loc = ok ? 1 : 0, glob = 1;
    MPI_Allreduce(&loc, &glob, 1, MPI_INT, MPI_MIN, pMesh->getMPICommunicator());
    ok = glob == 1;
    if (!ok) {
        if (!pMesh->getMPIRankGlobal())
            std::cout << "[enuts-scoped] preconditions failed (DT chain / level "
                         "range / multi-level block): full exchange"
                      << std::endl;
        return false;
    }
    pMesh->buildEleDGScopedPlan(dtOf, m_uiScopedDTs, m_uiScopedPlan);
    m_uiScopedZero.assign(std::max(1u, m_uiScopedPlan.npes), 0u);
    m_uiScopedOk = true;
    return true;
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::scoped_check_compare(ot::Mesh* pMesh, const T* real,
                                                const T* shadow,
                                                unsigned int DOF,
                                                unsigned int pt,
                                                unsigned int rk) {
    if (m_uiBlkDepEid.size() != pMesh->getLocalBlockList().size()) {
        m_uiBlkDepEid.assign(pMesh->getLocalBlockList().size(), {});
        for (unsigned int b = 0; b < m_uiBlkDepEid.size(); b++)
            computeBlockUnzipDepElements(pMesh, b, m_uiBlkDepEid[b]);
    }
    const unsigned int nPe  = pMesh->getNumNodesPerElement();
    const size_t dgSz       = (size_t)pMesh->getAllElements().size() * nPe;
    const unsigned int lb   = pMesh->getElementLocalBegin();
    const unsigned int le   = pMesh->getElementLocalEnd();
    std::set<unsigned int> seen;
    unsigned long long bad = 0;
    for (const unsigned int blk : m_uiActiveBlkIDs)
        for (const unsigned int e : m_uiBlkDepEid[blk]) {
            if (e >= lb && e < le) continue;
            if (!seen.insert(e).second) continue;
            ts::EnutsProf::get().chk_cmp++;
            bool diff = false;
            for (unsigned int v = 0; v < DOF; v++)
                if (std::memcmp(real + v * dgSz + (size_t)e * nPe,
                                shadow + v * dgSz + (size_t)e * nPe,
                                sizeof(T) * nPe) != 0)
                    diff = true;
            if (diff) {
                bad++;
                if (bad <= 5)
                    std::cout << "[enuts-scoped-check r"
                              << pMesh->getMPIRankGlobal() << "] MISMATCH pt=" << pt
                              << " rk=" << rk << " ghost elem " << e << " "
                              << pMesh->getAllElements()[e] << " etime="
                              << m_uiEleTime[e] << " edt=" << m_uiEleDT[e]
                              << std::endl;
            }
        }
    ts::EnutsProf::get().chk_bad += bad;
    if (bad) {
        std::cout << "[enuts-scoped-check r" << pMesh->getMPIRankGlobal()
                  << "] " << bad << " mismatching dependent ghost elems at pt="
                  << pt << " rk=" << rk << "; aborting" << std::endl;
        MPI_Abort(pMesh->getMPICommunicator(), 9);
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
    const bool ownPlan =
        ts::EnutsProf::ownerSync() == 1 && prepare_owner_sync(pMesh);
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
    const bool scopedOn = !ownPlan && ts::EnutsProf::scoped() &&
                          prepare_scoped_exchange(pMesh);
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
        const int own = ts::EnutsProf::ownerSync();
        sync_prebuild(ownPlan, own && !ownPlan, ts::EnutsProf::syncFast() && !own);
        sync_thread_setup(own && !ownPlan);
        if (!ownPlan) ensure_evar_dg();  // LAZY_TMP: before the parallel loops
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

            if (m_uiStageRing && ts::EnutsProf::stageInFused() && BLK_S > 1) {
                // STAGE_IN_FUSED: the block below in one pass per entry
                sIn.copyMeta(m_uiBVec[blk]._vec[0]);
                std::vector<uint64_t>& sg = m_uiStageSgn[blk];
                const size_t n            = (size_t)DOF * NN;
                const T* u                = m_uiBVec[blk]._vec[0].data();
                const T* k                = m_uiBVec[blk]._vec[RING_K].data();
                const T c0 = blk_dt * m_uiAij[(BLK_S - 1) * m_uiNumStages + (BLK_S - 2)];
                const bool launder = BLK_S >= 3 && ts::EnutsProf::stageRingMode() == 1;
                const bool upd     = BLK_S + 1 <= m_uiNumStages;
                const size_t nw    = (n + 63) / 64;
                // RING_FUSED: acc = U + b_1 dt k_1 (stage 2) or acc += b_{s-1} dt k_{s-1}
                const bool facc = ts::EnutsProf::ringFused();
                T* acc          = facc ? (T*)m_uiBVec[blk]._vec[RING_ACC].data() : nullptr;
                const T cb      = facc ? blk_dt * m_uiBi[BLK_S - 2] : (T)0;
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
                    if (facc) {
                        if (BLK_S == 2)
                            for (size_t i = i0; i < i1; i++) {
                                T q = u[i];
                                q += (cb * k[i]);
                                acc[i] = q;
                            }
                        else
                            for (size_t i = i0; i < i1; i++) acc[i] += (cb * k[i]);
                    }
                    if (upd)
                        for (size_t i = i0; i < i1; i++)
                            if (!std::signbit(k[i])) add |= (uint64_t)1 << (i - i0);
                    sg[w] = old | add;
                }
            } else {
            sIn.vecCopy(m_uiBVec[blk]._vec[0]);  // copy the in vector.
            if (m_uiStageRing) {
                // STAGE_RING: U, then the dropped 0 * k_j terms (j < s-1) as
                // their only effect, -0.0 -> +0.0 where the mask is set, then
                // the one live term a_{s,s-1} k_{s-1} from the ring slot
                std::vector<uint64_t>& sg = m_uiStageSgn[blk];
                const size_t n            = (size_t)DOF * NN;
                if (BLK_S == 1) std::fill(sg.begin(), sg.end(), 0);
                if (BLK_S >= 3 && ts::EnutsProf::stageRingMode() == 1) {
                    for (size_t w = 0; w < sg.size(); w++) {
                        uint64_t bits = sg[w];
                        while (bits) {
                            const int b    = __builtin_ctzll(bits);
                            const size_t i = w * 64 + b;
                            bits &= bits - 1;
                            if (i < n && out_ptr[i] == (T)0 && std::signbit(out_ptr[i]))
                                out_ptr[i] = (T)0;
                        }
                    }
                }
                if (BLK_S > 1) {
                    const T* k = m_uiBVec[blk]._vec[RING_K].data();  // k_{s-1}
                    // k_{s-1} is a dropped 0 * k term of stages s+1 .. NS
                    if (BLK_S + 1 <= m_uiNumStages)
                        for (size_t i = 0; i < n; i++)
                            if (!std::signbit(k[i])) sg[i >> 6] |= (uint64_t)1 << (i & 63);
                    const T* stp[1] = {k};
                    const T stc[1]  = {blk_dt * m_uiAij[(BLK_S - 1) * m_uiNumStages +
                                                        (BLK_S - 2)]};
                    enuts_axpy_stages(out_ptr, stp, stc, 1, n);
                }
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

            // stage time: step start, plus c_s * dt_blk with DENDRO_ENUTS_STAGE_TIME
            const T t_stage = current_t + dt_finest * BLK_T +
                              (ts::EnutsProf::stageTime() ? m_uiCi[BLK_S - 1] * blk_dt : (T)0);
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
        if (ownPlan && ts::EnutsProf::ownP2cShared()) {
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
            if (m_uiStageRing &&
                (!ts::EnutsProf::ringFused() || BLK_S == m_uiNumStages)) {
                // STAGE_RING: acc = U, then += b_s dt k_s after each synced
                // stage: the same per-entry sequence as the one-pass update.
                // RING_FUSED: stages 1 .. NS-1 are added by the next stage
                // input pass instead (k_s is still in the ring slot there)
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
            // STAGE_RING: the accumulator already holds U + sum_s b_s dt k_s
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
    if (m_uiStsOn || stsTest() || stsTestC() > 0) sts_hook(pt, actIds, nAct, ownPlan, thr);
    ENUTS_TIC();

    // OWN_NO_ETIME_XCH: the owner path never reads a ghost element time
    const bool skipEt = ownPlan && ts::EnutsProf::ownNoEtimeXch();
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

        assert(ts::EnutsProf::lazyTmp() || m_uiEVar.get_dof() == m_uiEVecTmp[0].get_dof());
        assert(ts::EnutsProf::lazyTmp() || m_uiEVar.get_dof() == m_uiEVecTmp[1].get_dof());

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
