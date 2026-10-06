/**
 * @file stepProf.h
 * @brief DENDRO_STEP_PROF=1 wall timers for the per-step (RK) path. Header-only;
 * when the knob is off every scope is a single predictable branch and nothing
 * is reduced or printed.
 */
#pragma once

#include <mpi.h>

#include <cstdlib>

namespace dendro {
namespace stepprof {

enum Id {
    EV_TOTAL = 0,
    EV_PRE_TS,
    EV_COPY,
    EV_AXPY_STAGE,
    EV_POST_TS_STAGE,
    EV_PRE_STAGE,
    EV_RHS,
    EV_POST_STAGE,
    EV_FINAL_AXPY,
    EV_SYNC_POSTAXPY,
    EV_BCAST_STEP,
    EV_POST_TS_FINAL,
    EV_WAITALL,
    CTX_UNZIP_TOTAL,
    CTX_ZIP_TOTAL,
    UZ_GHOST_BEGIN,
    UZ_GHOST_END,
    UZ_UNZIP_COMPUTE,
    GB_SYNC_PRE,
    GB_PACK_SEND,
    GE_WAIT,
    GE_SCATTER_RECV,
    GE_MIRROR,
    GE_SYNC_POST,
    SYNC_LOCALDUP,
    SYNC_PACK,
    SYNC_COMM,
    SYNC_UNPACK,
    NUM_IDS
};

inline const char* name(int id) {
    static const char* n[NUM_IDS] = {
        "evolve_total",     "pre_timestep",    "evec_copy",
        "axpy_stage",       "post_ts_stage",   "pre_stage",
        "rhs_total",        "post_stage",      "axpy_final",
        "sync_postaxpy",    "bcast_step",      "post_ts_final",
        "mesh_waitAll",     "ctx:unzip_total", "ctx:zip_total",
        "uz:ghost_begin",   "uz:ghost_end",    "uz:unzip_compute",
        "gb:sync_pre",      "gb:pack_send",    "ge:wait",
        "ge:scatter_recv",  "ge:mirror",       "ge:sync_post",
        "sync:local_dup",   "sync:pack",       "sync:comm",
        "sync:unpack"};
    return n[id];
}

inline bool enabled() {
    static const bool on = [] {
        const char* e = std::getenv("DENDRO_STEP_PROF");
        return e && e[0] == '1' && e[1] == '\0';
    }();
    return on;
}

inline double* acc() {
    static double a[NUM_IDS] = {0};
    return a;
}

inline long long* cnt() {
    static long long c[NUM_IDS] = {0};
    return c;
}

/** Always-on per-rank cumulative wall seconds (DENDRO_RANK_TABLE columns). Plain
 * MPI_Wtime pairs, never reset, no behaviour change.
 *  GHOST_WAIT: the two MPI_Waitall (send then recv) in readFromGhostEnd(ctx).
 *  GHOST_PACK: send-buffer pack + Irecv/Isend posting in readFromGhostBegin(ctx).
 *  SIDE_WAIT : zip-sync comm (Irecv/Isend/Waitall or Alltoallv) in
 *              syncZipNonPrimary + the Waitall in broadcastCgValuesByPhysPosStep. */
namespace rankprof {
// RHS_TOTAL: wall time inside the solver rhs call of ETS::evolve (includes GHOST_WAIT)
// EVOLVE_TOTAL: wall time of ETS::evolve; BARRIER_WAIT: its closing barrier
enum { GHOST_WAIT = 0, GHOST_PACK, SIDE_WAIT, RHS_TOTAL, EVOLVE_TOTAL, BARRIER_WAIT, NUM };
inline double* acc() {
    static double a[NUM] = {};
    return a;
}
}  // namespace rankprof

/** True only while ETS::evolve() is running, so init / output / remesh-time
 * calls into the same Mesh routines are not counted as step cost. */
inline bool& inEvolve() {
    static bool f = false;
    return f;
}

struct Scope {
    int id_;
    bool on_;
    double t0_;
    explicit Scope(int id) : id_(id), on_(enabled() && inEvolve()), t0_(0.0) {
        if (on_) t0_ = MPI_Wtime();
    }
    void stop() {
        if (!on_) return;
        acc()[id_] += MPI_Wtime() - t0_;
        cnt()[id_]++;
        on_ = false;
    }
    ~Scope() { stop(); }
};

/** Wraps one ETS::evolve() call: opens the counting window and times it. */
struct EvolveScope {
    Scope s_;
    double rpT0_;
    EvolveScope() : s_(EV_TOTAL), rpT0_(MPI_Wtime()) {
        if (enabled()) {
            inEvolve() = true;
            s_.on_ = true;
            s_.t0_ = MPI_Wtime();
        }
    }
    ~EvolveScope() {
        rankprof::acc()[rankprof::EVOLVE_TOTAL] += MPI_Wtime() - rpT0_;
        if (!enabled()) return;
        s_.stop();
        inEvolve() = false;
    }
};

}  // namespace stepprof
}  // namespace dendro
