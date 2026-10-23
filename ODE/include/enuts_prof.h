/**
 * @file enuts_prof.h
 * @brief EnutsProf: the ENUTS options, env switches and profiling accumulators.
 * (code moved from enuts.h, which includes this file)
 */
#pragma once

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
    /// True when the per-block loops of partial_evolve run threaded over the
    /// active blocks, largest first: a hybrid build and a context that
    /// declares lts_thread_safe. Set by the ExplicitNUTS constructor.
    static bool& threaded() {
        static bool on = false;
        return on;
    }
    static bool ompOn() { return threaded(); }
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
    /** DENDRO_ENUTS_SCOPED_EXCH_CHECK=1: the level-scoped element-DG ghost
     *  exchange (per-element sync path) also runs the full exchange into a
     *  shadow and compares every ghost slot (output only) */
    static bool scopedCheck() {
        static const int v = []() {
            const char* e = std::getenv("DENDRO_ENUTS_SCOPED_EXCH_CHECK");
            return (e && e[0] == '1') ? 1 : 0;
        }();
        return v == 1;
    }
    unsigned long long chk_cmp = 0, chk_bad = 0;
    /** interface setting (LtsOptions::iface, 3 or 15), a bit mask: how a
     *  block converts stage data it takes from a neighbor that steps with a
     *  different dt.
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
    /// The solution-changing settings of the process's stepper. Set by the
    /// ExplicitNUTS constructor; one stepper per process.
    static ts::LtsOptions& options() {
        static ts::LtsOptions o;
        return o;
    }
    // DENDRO_ENUTS_TEST_SABOTAGE=2 (test builds only): the original interface
    // conversion, first order at level interfaces; testEnutsOrder must fail
#if defined(DENDRO_ENUTS_TEST_SABOTAGE) && DENDRO_ENUTS_TEST_SABOTAGE == 2
    static int ifaceFix() { return 0; }
#else
    static int ifaceFix() { return (int)options().iface; }
#endif
    /** with interface setting bit 4 and the owner sync, exchange the stored
     *  vector g of the classes that completed a step at this partial step
     *  (the owner-node exchange, on the unscoped lists when the stage exchange
     *  is scoped), so a ghost row's g is its owner's value. Without it g on
     *  ghost rows is formed locally from stage slots the exchange only partly
     *  delivered, and the result depends on the partition. */
    static bool gXch() { return options().g_exchange; }
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
    /** DENDRO_ENUTS_NANCHK=1: per-block NaN checks of the stage data (debug;
     *  also serializes the threaded block loop). One parse at every site. */
    static bool nanChk() {
        static const bool v = []() {
            const char* e = std::getenv("DENDRO_ENUTS_NANCHK");
            return e && e[0] == '1';
        }();
        return v;
    }
    /** DENDRO_ENUTS_G_CARRY_STRICT=1 (test gates): abort when the iface-15 g
     *  history cold-starts although the octree looks unchanged (same global
     *  element count), or when a restart cannot read it */
    static bool gCarryStrict() {
        static const bool v = []() {
            const char* e = std::getenv("DENDRO_ENUTS_G_CARRY_STRICT");
            return e && e[0] == '1';
        }();
        return v;
    }
    static bool ifaceNL() { return (ifaceFix() & 7) == 7; }
    static bool ifaceShift() { return (ifaceFix() & 8) != 0; }
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
