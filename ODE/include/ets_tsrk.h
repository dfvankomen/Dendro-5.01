/**
 * @file ets_tsrk.h
 * @brief Two-step Runge-Kutta (TSRK) explicit time stepper (prototype).
 *
 * RK-like sibling of ETS_MSRK: single solution vector, reusing the previous
 * `depth` steps' stage-derivative clouds (no stage-value cloud). The order-6
 * method (s=4 -> 4 fresh evals/step = RK4 communication, depth-2, imag stab
 * 1.85) is from get_tsrk_o6_tableau(); coefficients are numerically optimized
 * (findings/14) -- a prototype. Startup bootstraps the clouds with RK6 micro-
 * integration (reusing base ETS::evolve at RK6); re-bootstraps on remesh/dt
 * change like ETS_MSRK.
 */
#pragma once
#include <vector>

#include "ets.h"
#include "rkTableau.h"

namespace ts {

template <typename T, typename Ctx>
class ETS_TSRK : public ETS<T, Ctx> {
    using ETS<T, Ctx>::m_uiAppCtx;
    using ETS<T, Ctx>::m_uiNumStages;
    using ETS<T, Ctx>::m_uiEVar;
    using ETS<T, Ctx>::m_uiStVec;
    using ETS<T, Ctx>::m_uiEVecTmp;
    using ETS<T, Ctx>::m_uiTimeInfo;

    static constexpr unsigned int S     = 4;   // stages / fresh evals per step
    static constexpr unsigned int DEPTH = 2;   // past steps reused

    // tableau (row-major A[i*S+j])
    DendroScalar m_c[S], m_A1[S * S], m_A2[S * S], m_r[S * S];
    DendroScalar m_b[S], m_v1[S], m_v2[S];

    // history clouds: m_hist[m*S + j] = f at stage j from (m+1) steps back.
    std::vector<DVec> m_hist;
    bool m_histAlloc = false;
    unsigned int m_bootstrapRemaining;
    DendroScalar m_prevDt;

    int allocate_history_vars() {
        if (m_histAlloc) return 0;
        m_hist.resize(DEPTH * S);
        for (auto& h : m_hist)
            h.create_vector(m_uiAppCtx->get_mesh(), m_uiEVar.get_type(),
                            m_uiEVar.get_loc(), m_uiEVar.get_dof(),
                            m_uiEVar.is_ghost_allocated());
        m_histAlloc = true;
        return 0;
    }
    int deallocate_history_vars() {
        if (!m_histAlloc) return 0;
        for (auto& h : m_hist) h.destroy_vector();
        m_hist.clear();
        m_histAlloc = false;
        return 0;
    }
    void invalidate_history() {
        m_bootstrapRemaining = DEPTH;
        m_prevDt             = -1.0;
    }

    // one RK6 micro-step of size (absc*h) from a saved base state, evaluating
    // the RHS at the endpoint into `outF`. Uses the base ETS::evolve at RK6.
    void micro_eval(const DVec& base, const TSInfo& ti0, double absc, double h,
                    DVec& outF) {
        m_uiEVar.copy_data(base);
        TSInfo tmi     = ti0;
        tmi._m_uiTh    = absc * h;
        m_uiAppCtx->set_ts_info(tmi);
        ETS<T, Ctx>::evolve();  // advances m_uiEVar by absc*h (RK6 coefficients)
        m_uiAppCtx->rhs(&m_uiEVar, &outF, 1, ti0._m_uiT + absc * h);
    }

   public:
    ETS_TSRK(Ctx* appCtx) : ETS<T, Ctx>(appCtx) {
        get_tsrk_o6_tableau(m_c, m_A1, m_A2, m_r, m_b, m_v1, m_v2);
        // base uses RK6 (Luther, 7 stages) for bootstrap micro-integration;
        // this also sizes the base stage-vector pool (m_uiStVec) to 7 >= S.
        this->set_ets_coefficients(ETSType::RK6);
        m_bootstrapRemaining = DEPTH;
        m_prevDt             = -1.0;
    }
    ~ETS_TSRK() { deallocate_history_vars(); }

    void init() {
        m_uiAppCtx->initialize();
        m_uiTimeInfo = m_uiAppCtx->get_ts_info();
        this->allocate_internal_vars();
        allocate_history_vars();
        m_uiAppCtx->set_ets_synced(false);
        this->sync_with_mesh();
    }

    int sync_with_mesh() {
        if (m_uiAppCtx->is_ets_synced()) return 0;
        m_uiEVar = m_uiAppCtx->get_evolution_vars();
        this->deallocate_internal_vars();
        deallocate_history_vars();
        this->allocate_internal_vars();
        allocate_history_vars();
        invalidate_history();
        m_uiAppCtx->set_ets_synced(true);
        return 0;
    }

    void evolve() {
        m_uiTimeInfo    = m_uiAppCtx->get_ts_info();
        const double dt = m_uiTimeInfo._m_uiTh;
        if (m_prevDt > 0.0 &&
            std::abs(dt - m_prevDt) > 1e-12 * std::abs(dt))
            invalidate_history();
        if (m_bootstrapRemaining > 0)
            evolve_bootstrap();
        else
            evolve_tsrk();
        m_prevDt = m_uiAppCtx->get_ts_info()._m_uiTh;
    }

    // build this step's stage cloud (RK6 micro-integration), age history, then
    // advance the solution one full RK6 step. After DEPTH of these, switch.
    void evolve_bootstrap() {
        const ot::Mesh* pMesh = m_uiAppCtx->get_mesh();
        const TSInfo ti0      = m_uiAppCtx->get_ts_info();
        const double h        = ti0._m_uiTh;
        m_uiEVecTmp[1].copy_data(m_uiEVar);  // save the true state

        // age: hist2 <- hist1
        for (unsigned int j = 0; j < S; j++)
            m_hist[S + j].copy_data(m_hist[j]);
        // build new cloud into hist1 (f at y(t_n + c_j*h))
        for (unsigned int j = 0; j < S; j++)
            micro_eval(m_uiEVecTmp[1], ti0, m_c[j], h, m_hist[j]);

        // advance the real solution by one full step (RK6)
        m_uiEVar.copy_data(m_uiEVecTmp[1]);
        TSInfo tfull    = ti0;
        m_uiAppCtx->set_ts_info(tfull);
        ETS<T, Ctx>::evolve();  // EVar -> y(t_n+h); ctx time advanced by h

        m_bootstrapRemaining--;
        pMesh->waitAll();
    }

    void evolve_tsrk() {
        const ot::Mesh* pMesh  = m_uiAppCtx->get_mesh();
        m_uiTimeInfo           = m_uiAppCtx->get_ts_info();
        const double t         = m_uiTimeInfo._m_uiT;
        const double dt        = m_uiTimeInfo._m_uiTh;
        m_uiAppCtx->pre_timestep(m_uiEVar);

        if (pMesh->isActive()) {
            DendroScalar cf[16];
            const DVec* sp[16];
            // fresh stages: StVec[i] = f(Y_i)
            for (unsigned int i = 0; i < S; i++) {
                m_uiEVecTmp[0].copy_data(m_uiEVar);
                unsigned int n = 0;
                for (unsigned int j = 0; j < S; j++) {
                    if (m_A1[i * S + j] != 0.0) {
                        cf[n] = m_A1[i * S + j] * dt; sp[n++] = &m_hist[j];
                    }
                    if (m_A2[i * S + j] != 0.0) {
                        cf[n] = m_A2[i * S + j] * dt; sp[n++] = &m_hist[S + j];
                    }
                }
                for (unsigned int j = 0; j < i; j++)
                    if (m_r[i * S + j] != 0.0) {
                        cf[n] = m_r[i * S + j] * dt; sp[n++] = &m_uiStVec[j];
                    }
                if (n) DVec::axpy_multi(pMesh, n, cf, sp, m_uiEVecTmp[0]);
                m_uiAppCtx->post_timestep(m_uiEVecTmp[0]);
                m_uiAppCtx->pre_stage(m_uiStVec[i]);
                m_uiAppCtx->rhs(&m_uiEVecTmp[0], &m_uiStVec[i], 1, t + m_c[i] * dt);
                m_uiAppCtx->post_stage(m_uiStVec[i]);
            }
            // update: EVar += dt*( b.F + v1.F1 + v2.F2 )
            unsigned int n = 0;
            for (unsigned int i = 0; i < S; i++) {
                if (m_b[i] != 0.0) { cf[n] = m_b[i] * dt; sp[n++] = &m_uiStVec[i]; }
                if (m_v1[i] != 0.0) { cf[n] = m_v1[i] * dt; sp[n++] = &m_hist[i]; }
                if (m_v2[i] != 0.0) { cf[n] = m_v2[i] * dt; sp[n++] = &m_hist[S + i]; }
            }
            if (n) DVec::axpy_multi(pMesh, n, cf, sp, m_uiEVar);
        }
        m_uiAppCtx->post_timestep(m_uiEVar);

        // rotate: hist2 <- hist1 ; hist1 <- this step's fresh stage derivs
        for (unsigned int j = 0; j < S; j++)
            m_hist[S + j].copy_data(m_hist[j]);
        for (unsigned int j = 0; j < S; j++)
            m_hist[j].copy_data(m_uiStVec[j]);

        m_uiAppCtx->increment_ts_info();
        m_uiTimeInfo = m_uiAppCtx->get_ts_info();
        pMesh->waitAll();
    }
};

}  // namespace ts
