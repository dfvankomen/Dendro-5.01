/**
 * @file ts.h
 * @author Milinda Fernando
 * @brief generic time integrator class for Dendro.
 * @version 0.1
 * @date 2019-10-18
 *
 * School of Computing, University of Utah
 * @copyright Copyright (c) 2019
 *
 */

#pragma once
#ifdef ETS_CHECK_GHOSTS
// verification scaffolding only; flag defaults OFF
#include <cstdio>
#include <set>
#endif

#include <algorithm>
#include <cfloat>
#include <cmath>

#include "ctx.h"
#include "dendro.h"
#include "dvec.h"
#include "logger.h"
#include "mesh.h"
#include "rkTableau.h"
#include "ts.h"

namespace ts {
/**time stepper type
 * UTS uniform time stepper.
 * UTS_ADAP: uniform over the grid but time step size changes over time.
 * NUTS: spatially adaptive time stepping.
 * NUTS_ADAP: NUTS where the smallest time step varies in time.
 *
 */
enum TimeStepperType { UTS = 0, UTS_ADAP, NUTS, NUTS_ADAP };

/**
 * @brief ETS Flags (currently not used)
 */
enum ETSFlags { FROM_T0 = 0, CHECKPT, CURR_STEP, CURR_TIME };

/**
 * @brief Embedded-pair error control: OFF (fixed dt), ESTIMATE (compute and
 * record the error norm, dt unchanged), ADAPTIVE (accept/reject + controller).
 */
enum class ETSErrCtrl { OFF = 0, ESTIMATE, ADAPTIVE };

/**@brief: per-step embedded error statistics (set when error control is on). */
struct ETSErrStats {
    /**@brief: scaled error norm of the accepted step (<= 1 meets tolerance)*/
    DendroScalar err                  = 0.0;
    /**@brief: step size actually taken*/
    DendroScalar dt_taken             = 0.0;
    /**@brief: controller proposal for the next step (capped when ADAPTIVE)*/
    DendroScalar dt_next              = 0.0;
    /**@brief: rejected attempts in the last step*/
    unsigned int rejects              = 0;
    /**@brief: rejected attempts since construction*/
    unsigned long long total_rejects  = 0;
    /**@brief: steps accepted with err > 1 after max_rejects*/
    unsigned long long forced_accepts = 0;
};

/**
 * @brief General explicit time stepper class for Dendro-5.0
 * @tparam T
 */

#ifdef __PROFILE_ETS__
// per-stage profiling slots (EVOLVE stays index 0 for dump_pt back-compat;
// STAGE_6 covers the 7-stage RK6).
enum ETSPROFILE {
    EVOLVE = 0,
    STAGE_0,
    STAGE_1,
    STAGE_2,
    STAGE_3,
    STAGE_4,
    STAGE_5,
    STAGE_6,
    ETS_LAST
};
#endif

// number of profiling slots; stages beyond this are just unprofiled.
#ifndef ETS_MAX_PROFILED_STAGES
#define ETS_MAX_PROFILED_STAGES 7
#endif

// hard cap on stage count; sizes the fused axpy scratch arrays (used whether or
// not profiling is on). must be >= the largest scheme's stage count (RK6 = 7).
#ifndef ETS_MAX_STAGES
#define ETS_MAX_STAGES 8
#endif

template <typename T, typename Ctx>
class ETS {
#ifdef __PROFILE_ETS__
   public:
    std::vector<profiler_t> m_uiCtxpt =
        std::vector<profiler_t>(static_cast<int>(ETSPROFILE::ETS_LAST));
    const char* ETSPROFILE_NAMES[static_cast<int>(ETSPROFILE::ETS_LAST)] = {
        "evolve", "stage_0", "stage_1", "stage_2", "stage_3",
        "stage_4", "stage_5", "stage_6"};

    void init_pt() {
        for (unsigned int i = 0; i < m_uiCtxpt.size(); i++)
            m_uiCtxpt[i].start();

        m_uiAppCtx->init_pt();
    }

    void reset_pt() {
        for (unsigned int i = 0; i < m_uiCtxpt.size(); i++)
            m_uiCtxpt[i].snapreset();

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
                outfile
                    << "step_ets\t act_npes\t glb_npes\t maxdepth\t numOcts\t "
                       "dof_cg\t dof_uz\t"
                    << "gele_min\t gele_mean\t gele_max\t"
                       "lele_min\t lele_mean\t lele_max\t"
                       "lnodes_min\t lnodes_mean\t lnodes_max\t"
                       "remsh_igt_min\t remesh_igt_mean\t remesh_igt_max\t"
                       "evolve_min\t evolve_mean\t evolve_max\t"
                       "unzip_async_min\t unzip_async_mean\t unzip_async_max\t"
                       "unzip_min\t unzip_mean\t unzip_max\t"
                       "rhs_min\t rhs_mean\t rhs_max\t"
                       "zip_async_min\t zip_async_mean\t zip_async_max\t"
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

        t_stat = m_uiCtxpt[ETSPROFILE::EVOLVE].snap;
        min_mean_max(&t_stat, t_stat_g, comm);
        if (!rank)
            outfile << t_stat_g[0] << "\t " << t_stat_g[1] << "\t "
                    << t_stat_g[2] << "\t ";

        t_stat = m_uiAppCtx->m_uiCtxpt[CTXPROFILE::UNZIP_WCOMM].snap;
        min_mean_max(&t_stat, t_stat_g, comm);
        if (!rank)
            outfile << t_stat_g[0] << "\t " << t_stat_g[1] << "\t "
                    << t_stat_g[2] << "\t ";

        t_stat = m_uiAppCtx->m_uiCtxpt[CTXPROFILE::UNZIP].snap;
        min_mean_max(&t_stat, t_stat_g, comm);
        if (!rank)
            outfile << t_stat_g[0] << "\t " << t_stat_g[1] << "\t "
                    << t_stat_g[2] << "\t ";

        t_stat = m_uiAppCtx->m_uiCtxpt[CTXPROFILE::RHS].snap;
        min_mean_max(&t_stat, t_stat_g, comm);
        if (!rank)
            outfile << t_stat_g[0] << "\t " << t_stat_g[1] << "\t "
                    << t_stat_g[2] << "\t ";

        t_stat = m_uiAppCtx->m_uiCtxpt[CTXPROFILE::ZIP].snap;
        min_mean_max(&t_stat, t_stat_g, comm);
        if (!rank)
            outfile << t_stat_g[0] << "\t " << t_stat_g[1] << "\t "
                    << t_stat_g[2] << "\t ";

        if (!rank) outfile << std::endl;
    }
#endif

   protected:
    /** @brief Application context. */
    Ctx* m_uiAppCtx;

    /**@brief: Time stepper type*/
    ETSType m_uiType;

    /**@brief: CFL factor*/
    DendroScalar m_uiCFL;

    /**@brief: time integrator coefficients for solution u*/
    DendroScalar* m_uiAij = NULL;

    /**@brief: time integrator coefficients for time */
    DendroScalar* m_uiBi  = NULL;

    /**@brief: time integrator weights*/
    DendroScalar* m_uiCi  = NULL;

    /**@brief: embedded error weights e_i = b_i - bhat_i (null if none)*/
    const DendroScalar* m_uiEi = nullptr;

    /**@brief: number of stages*/
    unsigned int m_uiNumStages;

    /**@brief: embedded error control settings*/
    ETSErrCtrl m_uiErrCtrl      = ETSErrCtrl::OFF;
    DendroScalar m_uiErrAtol    = 0.0;
    DendroScalar m_uiErrRtol    = 0.0;
    DendroScalar m_uiErrSafety  = 0.9;
    DendroScalar m_uiErrFacMin  = 0.2;
    DendroScalar m_uiErrFacMax  = 5.0;
    unsigned int m_uiMaxRejects = 10;

    /**@brief: step size cap for ADAPTIVE (<= 0: none)*/
    DendroScalar m_uiDtMax      = 0.0;

    /**@brief: embedded error statistics of the last step*/
    ETSErrStats m_uiErrStats;

    /**@brief: time step info*/
    TSInfo m_uiTimeInfo;

    /**@brief: evolution variables*/
    DVec m_uiEVar;

    /**@brief: stage vector*/
    std::vector<DVec> m_uiStVec;

    /**@brief: evolution temp vector*/
    DVec m_uiEVecTmp[2];

    /**@brief: state true if the internal variables are allocated. */
    bool m_uiIsInternalAlloc = false;

   protected:
    /**
     * @brief Allocates internal variables for the time stepper.
     * @return int
     */
    int allocate_internal_vars();

    /**@brief: Deallocate internal variables. */
    int deallocate_internal_vars();

    /**@brief: false for schemes that cannot use embedded error control. */
    virtual bool supports_error_control() const { return true; }

    /**@brief: run all stages of one step of size dt from m_uiEVar. */
    void stage_loop(double current_t, double dt);

    /**@brief: m_uiEVar += dt * sum_k b_k k_k (with post_timestep). */
    void final_combine(double dt);

    /**@brief: err = dt * sum_k e_k k_k into m_uiEVecTmp[0]; returns the global
     * max over owned nodes of |err| / (atol + rtol * max(|y_n|, |y_n+1|)). */
    DendroScalar embedded_error_norm(double dt);

    /**@brief: one step with the embedded error estimate (and control). */
    void evolve_embedded(double current_t, double dt);

   public:
    /**
     * @brief Construct a new ETS object
     * @param pMesh : underlying mesh data structure.
     */
    ETS(Ctx* appCtx);

    /**
     * @brief Destroy the ETS object
     */
    virtual ~ETS();

    /**
     * @brief Set the ets coefficients
     * $ k_i = f(u^{n} + \sum_{j=1}^{i-1} a_{i,j}*dt , t^{n} + b_i*th)$
     * $u^{n+1} = u^{n} + \sum_{m=1}^{num\_stages} k_m $
     * @param aij : time integrator coefficients for the f_rhs first term
     * @param bi : time integrator coefficients for the f_rhs second term
     * @param ci : time integrator weights
     * @param num_stages : number of stages
     * @return int : is success return zero.
     */
    int set_ets_coefficients(DendroScalar* aij, DendroScalar* bi,
                             DendroScalar* ci, unsigned int num_stages);

    /**
     * @brief : sets default ETS time integrator.
     * @param [in] type: time integrator type.
     */
    int set_ets_coefficients(ETSType type);

    /**
     * @brief Set the evolve vars for the ETS time stepper.
     *
     * @param eVar : evolution variables, multiple evolution variables should be
     * added as one vector with multiple dof.
     * @return int
     */
    int set_evolve_vars(DVec eVar);

    /**@brief: initialize the ETS solver*/
    virtual void init();

    /**@brief: returns the current time step*/
    inline DendroIntL curr_step() { return m_uiTimeInfo._m_uiStep; };

    /**@brief: returns the current time*/
    inline DendroScalar curr_time() { return m_uiTimeInfo._m_uiT; };

    /**@brief: */
    inline DendroScalar ts_size() const { return m_uiTimeInfo._m_uiTh; }

    /**@brief: returns the mesh is active. */
    inline bool is_active() const { return m_uiAppCtx->get_mesh()->isActive(); }

    /**@brief: returns the active rank*/
    unsigned int get_active_rank() const;

    /**@brief: returns the active npes*/
    unsigned int get_active_npes() const;

    /**@brief: returns the global rank*/
    unsigned int get_global_rank() const;

    /**@brief: return the global npes*/
    unsigned int get_global_npes() const;

    /**@brief: return the active communicator*/
    MPI_Comm get_active_comm() const;

    /**@brief: return the global communicator. */
    MPI_Comm get_global_comm() const;

    /**@brief: returns the underlying mesh data structure. */
    const ot::Mesh* get_mesh() const { return m_uiAppCtx->get_mesh(); }

    /**@brief: returns the evolution variables. */
    inline DVec get_evolve_vars() const { return m_uiEVar; }

    /**@brief: returns the time step info*/
    inline TSInfo get_timestep_info() const { return m_uiTimeInfo; }

    /**
     * @brief Enable embedded error estimation / adaptive step control. Needs a
     * single-step scheme with an embedded pair (RK45_CASH_KARP, RKF45,
     * RK45_DORMAND_PRINCE) and host evolution vars; call after
     * set_ets_coefficients and set_evolve_vars.
     * @param mode        OFF, ESTIMATE or ADAPTIVE
     * @param atol, rtol  per-point scale atol + rtol * max(|y_n|, |y_n+1|)
     * @param safety, facmin, facmax  h_new = h * min(facmax, max(facmin,
     * safety * err^(-1/5))); facmax is 1 right after a reject
     * @param max_rejects attempts beyond this are accepted with a warning
     * @return 0 on success, -1 (logged) if unsupported or invalid.
     */
    virtual int set_error_control(ETSErrCtrl mode, DendroScalar atol,
                                  DendroScalar rtol, DendroScalar safety = 0.9,
                                  DendroScalar facmin      = 0.2,
                                  DendroScalar facmax      = 5.0,
                                  unsigned int max_rejects = 10);

    /**@brief: cap the ADAPTIVE step size and clamp the current dt to it. */
    void set_dt_max(DendroScalar dt_max);

    /**@brief: returns the error control mode. */
    inline ETSErrCtrl error_control() const { return m_uiErrCtrl; }

    /**@brief: returns the embedded error statistics of the last step. */
    inline const ETSErrStats& get_err_stats() const { return m_uiErrStats; }

    /**@brief: perform synchronizations with correct variable allocations for
     * the new mesh: should be called after remeshing.  */
    virtual int sync_with_mesh();

    /**@brief: advance to next time step*/
    virtual void evolve();

    /**@brief: dump load statistics*/
    void dump_load_statistics(std::ostream& sout);
};

template <typename T, typename Ctx>
ETS<T, Ctx>::ETS(Ctx* appCtx) {
    m_uiAppCtx    = appCtx;
    m_uiAij       = NULL;
    m_uiBi        = NULL;
    m_uiCi        = NULL;
    m_uiNumStages = 0;
    m_uiTimeInfo  = appCtx->get_ts_info();

    m_uiEVar      = m_uiAppCtx->get_evolution_vars();

    dendro::logger::debug(dendro::logger::Scope{"ETS"},
                          "Explicit time stepper (ETS) created!");
}

template <typename T, typename Ctx>
ETS<T, Ctx>::~ETS() {
    return;
}

template <typename T, typename Ctx>
int ETS<T, Ctx>::set_ets_coefficients(DendroScalar* aij, DendroScalar* bi,
                                      DendroScalar* ci,
                                      unsigned int num_stages) {
    if (m_uiErrCtrl != ETSErrCtrl::OFF) {
        dendro::logger::error(dendro::logger::Scope{"ETS"},
                              "raw coefficients have no embedded pair; turn "
                              "error control off first");
        return -1;
    }
    m_uiEi        = nullptr;
    m_uiAij       = aij;
    m_uiBi        = bi;
    m_uiCi        = ci;

    m_uiNumStages = num_stages;
    return 0;
}

template <typename T, typename Ctx>
int ETS<T, Ctx>::allocate_internal_vars() {
    if (m_uiIsInternalAlloc) return 0;

    m_uiStVec.resize(m_uiNumStages);
    for (unsigned int i = 0; i < m_uiNumStages; i++)
        m_uiStVec[i].create_vector(m_uiAppCtx->get_mesh(), m_uiEVar.get_type(),
                                   m_uiEVar.get_loc(), m_uiEVar.get_dof(),
                                   m_uiEVar.is_ghost_allocated());

    m_uiEVecTmp[0].create_vector(m_uiAppCtx->get_mesh(), m_uiEVar.get_type(),
                                 m_uiEVar.get_loc(), m_uiEVar.get_dof(),
                                 m_uiEVar.is_ghost_allocated());
    m_uiEVecTmp[1].create_vector(m_uiAppCtx->get_mesh(), m_uiEVar.get_type(),
                                 m_uiEVar.get_loc(), m_uiEVar.get_dof(),
                                 m_uiEVar.is_ghost_allocated());

    m_uiIsInternalAlloc = true;
    return 0;
}

template <typename T, typename Ctx>
int ETS<T, Ctx>::deallocate_internal_vars() {
    if (!m_uiIsInternalAlloc) return 0;

    for (unsigned int i = 0; i < m_uiNumStages; i++)
        m_uiStVec[i].destroy_vector();

    m_uiStVec.clear();

    m_uiEVecTmp[0].destroy_vector();
    m_uiEVecTmp[1].destroy_vector();

    m_uiIsInternalAlloc = false;
    return 0;
}

template <typename T, typename Ctx>
int ETS<T, Ctx>::set_ets_coefficients(ETSType type) {
    m_uiType = type;

    // tableaux live in rkTableau.h; pointers are function-static (persist).
    const DendroScalar *nodes = nullptr, *weights = nullptr, *matrix = nullptr;
    unsigned int nstages      = 0;
    if (ts::get_rk_tableau(type, nstages, nodes, weights, matrix) != 0)
        return -1;
    const DendroScalar* err = nullptr;
    ts::get_rk_error_weights(type, err);
    if (m_uiErrCtrl != ETSErrCtrl::OFF && err == nullptr) {
        dendro::logger::error(dendro::logger::Scope{"ETS"},
                              "ETS type {} has no embedded pair; turn error "
                              "control off first",
                              static_cast<int>(type));
        return -1;
    }

    m_uiEi        = err;
    m_uiNumStages = nstages;
    m_uiCi        = (DendroScalar*)nodes;    // c_i abscissae
    m_uiBi        = (DendroScalar*)weights;  // b_i weights
    m_uiAij       = (DendroScalar*)matrix;   // a_ij matrix

    return 0;
}

template <typename T, typename Ctx>
void ETS<T, Ctx>::init() {
    dendro::logger::info(
        dendro::logger::Scope{"ETS"},
        "Now initializing the ETS (initializing app context, allocating "
        "internal vars, and synchronizing with mesh");
    m_uiAppCtx->initialize();
    m_uiTimeInfo = m_uiAppCtx->get_ts_info();
    allocate_internal_vars();
    // Ctx initialize might have changed the mesh i.e. converge untill mesh
    // adapted to the initial data.
    m_uiAppCtx->set_ets_synced(false);
    this->sync_with_mesh();
}

template <typename T, typename Ctx>
int ETS<T, Ctx>::set_evolve_vars(DVec eVars) {
    m_uiEVar = eVars;
    return 0;
}

template <typename T, typename Ctx>
void ETS<T, Ctx>::evolve() {
#ifdef __PROFILE_ETS__
    m_uiCtxpt[ETSPROFILE::EVOLVE].start();
#endif

    dendro::logger::debug(dendro::logger::Scope{"ETS"},
                          "Beginning ETS evolve (overall time step: {})",
                          m_uiTimeInfo._m_uiStep);

    const ot::Mesh* pMesh  = m_uiAppCtx->get_mesh();
    m_uiTimeInfo           = m_uiAppCtx->get_ts_info();
    const double current_t = m_uiTimeInfo._m_uiT;
    const double dt        = m_uiTimeInfo._m_uiTh;

#ifdef ETS_CHECK_GHOSTS
    // ONE-SHOT PRECONDITION CHECK for the "fuse copy_data into axpy_multi" change.
    //
    // Dropping copy_data leaves tmp's GHOST nodes stale; the change is only safe if
    // the ghost exchange overwrites every one of them before the RHS reads them.
    // readFromGhostEnd writes exactly vec[recvNodeSM[k]] for k in each recv proc's
    // [offset, offset+count) range (mesh.tcc:~857), and a comment there asserts
    // "recvNodeSM is a permutation". This checks that claim instead of trusting it:
    // it computes the WRITTEN index set and subtracts it from the actual ghost range.
    // Any node in ghosts \ written would keep stale data => fusion changes results.
    {
        static bool _chk_done = false;
        if (!_chk_done && pMesh->isActive()) {
            _chk_done                  = true;
            const auto& _recvNodeSM    = pMesh->getRecvNodeSM();
            const auto& _recvCount     = pMesh->getNodalRecvCounts();
            const auto& _recvOffset    = pMesh->getNodalRecvOffsets();
            const auto& _recvProcList  = pMesh->getRecvProcList();

            std::set<unsigned int> _W;
            for (unsigned int _pi = 0; _pi < _recvProcList.size(); _pi++) {
                const unsigned int _p = _recvProcList[_pi];
                for (unsigned int _k = _recvOffset[_p];
                     _k < _recvOffset[_p] + _recvCount[_p]; _k++)
                    _W.insert(_recvNodeSM[_k]);
            }
            const unsigned int _preB  = pMesh->getNodePreGhostBegin();
            const unsigned int _preE  = pMesh->getNodePreGhostEnd();
            const unsigned int _postB = pMesh->getNodePostGhostBegin();
            const unsigned int _postE = pMesh->getNodePostGhostEnd();
            unsigned int _missing = 0, _outside = 0;
            for (unsigned int _i = _preB; _i < _preE; _i++)
                if (!_W.count(_i)) _missing++;
            for (unsigned int _i = _postB; _i < _postE; _i++)
                if (!_W.count(_i)) _missing++;
            // written indices that are NOT ghosts would mean the exchange scribbles
            // on local nodes -- also fatal to the reasoning, so count them too.
            for (unsigned int _w : _W)
                if (!((_w >= _preB && _w < _preE) || (_w >= _postB && _w < _postE)))
                    _outside++;
            const unsigned int _nghost = (_preE - _preB) + (_postE - _postB);
            fprintf(stderr,
                    "[ghostchk] rank=%d ghosts=%u written=%zu missing=%u "
                    "written_outside_ghosts=%u  %s\n",
                    pMesh->getMPIRank(), _nghost, _W.size(), _missing, _outside,
                    (_missing == 0 && _outside == 0)
                        ? "OK: recvNodeSM covers exactly the ghosts -> fusion SAFE"
                        : "<<< PRECONDITION VIOLATED -> fusion UNSAFE");
        }
    }
#endif

    m_uiAppCtx->pre_timestep(m_uiEVar);

    if (m_uiErrCtrl != ETSErrCtrl::OFF) {
        evolve_embedded(current_t, dt);
    } else if (pMesh->isActive()) {
        stage_loop(current_t, dt);
        final_combine(dt);
    } else {
        m_uiAppCtx->post_timestep(m_uiEVar);
    }

    m_uiAppCtx->increment_ts_info();
    if (m_uiErrCtrl == ETSErrCtrl::ADAPTIVE)
        m_uiAppCtx->set_timestep(m_uiErrStats.dt_next);
    m_uiTimeInfo = m_uiAppCtx->get_ts_info();
    pMesh->waitAll();

#ifdef __PROFILE_ETS__
    m_uiCtxpt[ETSPROFILE::EVOLVE].stop();
#endif

    dendro::logger::debug(dendro::logger::Scope{"ETS"},
                          "ETS evolve step finished!");
}

template <typename T, typename Ctx>
void ETS<T, Ctx>::stage_loop(double current_t, double dt) {
    [[maybe_unused]] const ot::Mesh* pMesh = m_uiAppCtx->get_mesh();
    double current_t_adv                   = current_t;

    for (unsigned int stage = 0; stage < m_uiNumStages; stage++) {
        dendro::logger::debug(dendro::logger::Scope{"ETS"},
                              "Now executing ETS Evolve stage {}/{}", stage + 1,
                              m_uiNumStages);

#ifdef __PROFILE_ETS__
        if (stage < ETS_MAX_PROFILED_STAGES)
            m_uiCtxpt[ETSPROFILE::STAGE_0 + stage].start();
#endif

        // stage input tmp = EVar + sum_p (aip*dt)*StVec[p], post_timestep fused
        {
            DendroScalar a_cf[ETS_MAX_STAGES];
            const DVec* a_sp[ETS_MAX_STAGES];
            unsigned int a_n = 0;
            for (unsigned int p = 0; p < stage; p++) {
                const DendroScalar aip = m_uiAij[stage * m_uiNumStages + p];
                if (aip != 0.0) {
                    a_cf[a_n] = aip * dt;
                    a_sp[a_n] = &m_uiStVec[p];
                    a_n++;
                }
            }
#if defined(DENDRO_RK_FUSE)
            m_uiAppCtx->rk_stage_input(m_uiEVar, a_n, a_cf, a_sp,
                                       m_uiEVecTmp[0]);
#else
            m_uiEVecTmp[0].copy_data(m_uiEVar);
            if (a_n) DVec::axpy_multi(pMesh, a_n, a_cf, a_sp, m_uiEVecTmp[0]);
            m_uiAppCtx->post_timestep(m_uiEVecTmp[0]);
#endif
        }

        current_t_adv = current_t + m_uiCi[stage] * dt;
        m_uiAppCtx->pre_stage(m_uiStVec[stage]);
        m_uiAppCtx->rhs(&m_uiEVecTmp[0], &m_uiStVec[stage], 1, current_t_adv);
        m_uiAppCtx->post_stage(m_uiStVec[stage]);

#ifdef __PROFILE_ETS__
        if (stage < ETS_MAX_PROFILED_STAGES)
            m_uiCtxpt[ETSPROFILE::STAGE_0 + stage].stop();
#endif
    }
}

template <typename T, typename Ctx>
void ETS<T, Ctx>::final_combine(double dt) {
    [[maybe_unused]] const ot::Mesh* pMesh = m_uiAppCtx->get_mesh();

    // EVar += sum_k (bi*dt)*StVec[k], post_timestep fused.
    DendroScalar b_cf[ETS_MAX_STAGES];
    const DVec* b_sp[ETS_MAX_STAGES];
    for (unsigned int k = 0; k < m_uiNumStages; k++) {
        b_cf[k] = m_uiBi[k] * dt;
        b_sp[k] = &m_uiStVec[k];
    }
#if defined(DENDRO_RK_FUSE)
    m_uiAppCtx->rk_combine(m_uiNumStages, b_cf, b_sp, m_uiEVar);
#else
    DVec::axpy_multi(pMesh, m_uiNumStages, b_cf, b_sp, m_uiEVar);
    m_uiAppCtx->post_timestep(m_uiEVar);
#endif
}

template <typename T, typename Ctx>
DendroScalar ETS<T, Ctx>::embedded_error_norm(double dt) {
    const ot::Mesh* pMesh  = m_uiAppCtx->get_mesh();
    const unsigned int dof = m_uiEVar.get_dof();
    const unsigned int sz  = m_uiEVar.get_size() / dof;
    unsigned int nb, ne;
    if (m_uiEVar.get_type() == ot::DVEC_TYPE::OCT_SHARED_NODES) {
        nb = pMesh->getNodeLocalBegin();
        ne = pMesh->getNodeLocalEnd();
    } else {
        const unsigned int npe = pMesh->getNumNodesPerElement();
        nb                     = pMesh->getElementLocalBegin() * npe;
        ne                     = pMesh->getElementLocalEnd() * npe;
    }

    const unsigned int ns = m_uiNumStages;
    const T* sp[ETS_MAX_STAGES];
    DendroScalar b_cf[ETS_MAX_STAGES], e_cf[ETS_MAX_STAGES];
    for (unsigned int k = 0; k < ns; k++) {
        sp[k]   = m_uiStVec[k].get_vec_ptr();
        b_cf[k] = m_uiBi[k] * dt;
        e_cf[k] = m_uiEi[k] * dt;
    }
    const T* y              = m_uiEVar.get_vec_ptr();
    T* err                  = m_uiEVecTmp[0].get_vec_ptr();
    const DendroScalar atol = m_uiErrAtol;
    const DendroScalar rtol = m_uiErrRtol;

    double emax             = 0.0;
#ifdef DENDRO_HYBRID_OMP
#pragma omp parallel for collapse(2) reduction(max : emax)
#endif
    for (unsigned int v = 0; v < dof; v++)
        for (unsigned int node = nb; node < ne; node++) {
            const unsigned int idx = v * sz + node;
            T y1 = y[idx], e = 0.0;
            for (unsigned int k = 0; k < ns; k++) {
                y1 += b_cf[k] * sp[k][idx];
                e += e_cf[k] * sp[k][idx];
            }
            err[idx]        = e;
            const double sc = atol + rtol * std::max(std::fabs((double)y[idx]),
                                                     std::fabs((double)y1));
            double r        = std::fabs((double)e) / sc;
            // NaN/inf counts as an infinite error
            if (!(r <= DBL_MAX)) r = HUGE_VAL;
            emax = std::max(emax, r);
        }

    double emax_g = emax;
    par::Mpi_Allreduce(&emax, &emax_g, 1, MPI_MAX, pMesh->getMPICommunicator());
    return emax_g;
}

template <typename T, typename Ctx>
void ETS<T, Ctx>::evolve_embedded(double current_t, double dt) {
    const ot::Mesh* pMesh = m_uiAppCtx->get_mesh();
    const bool adapt      = (m_uiErrCtrl == ETSErrCtrl::ADAPTIVE);
    if (adapt && m_uiDtMax > 0.0) dt = std::min(dt, m_uiDtMax);

    // {err, dt_taken, dt_next, rejects, forced}; zeros on inactive ranks
    double st[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    if (pMesh->isActive()) {
        unsigned int nrej = 0;
        double err        = 0.0;
        for (;;) {
            stage_loop(current_t, dt);
            err = embedded_error_norm(dt);
            if (!adapt || err <= 1.0) break;
            if (nrej == m_uiMaxRejects) {
                st[4] = 1.0;
                break;
            }
            nrej++;
            const double dt_rej = dt;
            dt *= std::max((double)m_uiErrFacMin,
                           m_uiErrSafety * std::pow(err, -0.2));
            dendro::logger::debug(dendro::logger::Scope{"ETS"},
                                  "step rejected: err={} dt={} -> {}", err,
                                  dt_rej, dt);
        }
        final_combine(dt);

        const double fmax = nrej ? 1.0 : (double)m_uiErrFacMax;
        const double fac =
            err > 0.0 ? m_uiErrSafety * std::pow(err, -0.2) : fmax;
        double dt_next =
            dt * std::min(fmax, std::max((double)m_uiErrFacMin, fac));
        if (adapt && m_uiDtMax > 0.0) dt_next = std::min(dt_next, m_uiDtMax);

        st[0] = err;
        st[1] = dt;
        st[2] = dt_next;
        st[3] = nrej;
    } else {
        m_uiAppCtx->post_timestep(m_uiEVar);
    }

    double st_g[5];
    par::Mpi_Allreduce(st, st_g, 5, MPI_MAX, pMesh->getMPIGlobalCommunicator());
    m_uiErrStats.err      = st_g[0];
    m_uiErrStats.dt_taken = st_g[1];
    m_uiErrStats.dt_next  = st_g[2];
    m_uiErrStats.rejects  = (unsigned int)st_g[3];
    m_uiErrStats.total_rejects += m_uiErrStats.rejects;
    if (st_g[4] > 0.0) {
        m_uiErrStats.forced_accepts++;
        if (!pMesh->getMPIRankGlobal())
            dendro::logger::warn(
                dendro::logger::Scope{"ETS"},
                "step accepted with err={} after {} rejects (dt={})", st_g[0],
                m_uiMaxRejects, st_g[1]);
    }

    dendro::logger::debug(dendro::logger::Scope{"ETS"},
                          "embedded err={} dt={} next={} rejects={}", st_g[0],
                          st_g[1], st_g[2], m_uiErrStats.rejects);

    // time advances by the step actually taken
    m_uiAppCtx->set_timestep(st_g[1]);
}

template <typename T, typename Ctx>
int ETS<T, Ctx>::set_error_control(ETSErrCtrl mode, DendroScalar atol,
                                   DendroScalar rtol, DendroScalar safety,
                                   DendroScalar facmin, DendroScalar facmax,
                                   unsigned int max_rejects) {
    const dendro::logger::Scope scope{"ETS"};
    if (mode == ETSErrCtrl::OFF) {
        m_uiErrCtrl = mode;
        return 0;
    }
    if (!supports_error_control()) {
        dendro::logger::error(scope,
                              "embedded error control needs a single-step RK "
                              "scheme (not MSRK/TSRK)");
        return -1;
    }
    if (m_uiEi == nullptr) {
        dendro::logger::error(scope,
                              "embedded error control needs an embedded pair "
                              "(RK45_CASH_KARP, RKF45, RK45_DORMAND_PRINCE)");
        return -1;
    }
    if (m_uiEVar.get_loc() != ot::DVEC_LOC::HOST ||
        (m_uiEVar.get_type() != ot::DVEC_TYPE::OCT_SHARED_NODES &&
         m_uiEVar.get_type() != ot::DVEC_TYPE::OCT_LOCAL_NODES)) {
        dendro::logger::error(scope,
                              "embedded error control needs host nodal "
                              "evolution vars (no GPU support)");
        return -1;
    }
    if (!(atol > 0.0) || !(rtol >= 0.0) || !(safety > 0.0 && safety <= 1.0) ||
        !(facmin > 0.0 && facmin <= 1.0) || !(facmax >= 1.0)) {
        dendro::logger::error(scope,
                              "invalid error control settings: need atol > 0, "
                              "rtol >= 0, 0 < safety <= 1, 0 < facmin <= 1 <= "
                              "facmax");
        return -1;
    }
    m_uiErrCtrl    = mode;
    m_uiErrAtol    = atol;
    m_uiErrRtol    = rtol;
    m_uiErrSafety  = safety;
    m_uiErrFacMin  = facmin;
    m_uiErrFacMax  = facmax;
    m_uiMaxRejects = max_rejects;
    return 0;
}

template <typename T, typename Ctx>
void ETS<T, Ctx>::set_dt_max(DendroScalar dt_max) {
    m_uiDtMax = dt_max;
    if (m_uiErrCtrl == ETSErrCtrl::ADAPTIVE && dt_max > 0.0 &&
        m_uiAppCtx->get_ts_info()._m_uiTh > dt_max)
        m_uiAppCtx->set_timestep(dt_max);
    m_uiTimeInfo = m_uiAppCtx->get_ts_info();
}

template <typename T, typename Ctx>
unsigned int ETS<T, Ctx>::get_active_rank() const {
    if (is_active())
        return m_uiAppCtx->get_mesh()->getMPIRank();
    else
        return get_global_rank();
}

template <typename T, typename Ctx>
unsigned int ETS<T, Ctx>::get_active_npes() const {
    if (is_active())
        return m_uiAppCtx->get_mesh()->getMPICommSize();
    else
        return get_global_npes();
}

template <typename T, typename Ctx>
unsigned int ETS<T, Ctx>::get_global_rank() const {
    return m_uiAppCtx->get_mesh()->getMPIRankGlobal();
}

template <typename T, typename Ctx>
unsigned int ETS<T, Ctx>::get_global_npes() const {
    return m_uiAppCtx->get_mesh()->getMPICommSizeGlobal();
}

template <typename T, typename Ctx>
MPI_Comm ETS<T, Ctx>::get_active_comm() const {
    if (is_active())
        return m_uiAppCtx->get_mesh()->getMPICommunicator();
    else
        return MPI_COMM_NULL;
}

template <typename T, typename Ctx>
MPI_Comm ETS<T, Ctx>::get_global_comm() const {
    return m_uiAppCtx->get_mesh()->getMPIGlobalCommunicator();
}

template <typename T, typename Ctx>
int ETS<T, Ctx>::sync_with_mesh() {
    if (m_uiAppCtx->is_ets_synced()) return 0;

    dendro::logger::debug(
        dendro::logger::Scope{"ETS"},
        "Now syncing ETS with mesh (reallocating internal ets variables)");

    m_uiEVar = m_uiAppCtx->get_evolution_vars();
    deallocate_internal_vars();
    allocate_internal_vars();
    m_uiAppCtx->set_ets_synced(true);

    dendro::logger::debug(dendro::logger::Scope{"ETS"},
                          "Finished syncing ETS with mesh!");

    return 0;
}

template <typename T, typename Ctx>
void ETS<T, Ctx>::dump_load_statistics(std::ostream& sout) {
    const ot::Mesh* pMesh = m_uiAppCtx->get_mesh();

    if (pMesh->isActive()) {
        double local_weight = pMesh->getNumLocalMeshElements();
        // const ot::TreeNode* pNodes = pMesh->getAllElements().data();
        // for(unsigned int ele = pMesh->getElementLocalBegin(); ele <
        // pMesh->getElementLocalEnd(); ele++)
        //     local_weight+=getOctWeight(&pNodes[ele]);
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

}  // end of namespace ts
