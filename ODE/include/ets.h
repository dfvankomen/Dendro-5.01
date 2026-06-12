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
#include "ctx.h"
#include "dendro.h"
#include "dvec.h"
#include "logger.h"
#include "mesh.h"
#include "ts.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <tuple>

namespace ts {

// cg-state tracer: dumps cg values at target phys positions at each
// checkpoint in evolve(). gate by EM4_CG_TRACE_DIR + (EM4_CG_TRACE_PHYS
// or EM4_CG_TRACE_BBOX). phys list format: "x1,y1,z1;x2,y2,z2;..."
// (eOrder-scaled tree coords). bbox format: "xlo,xhi,ylo,yhi,zlo,zhi".
// Optionally restrict to a specific step via EM4_CG_TRACE_STEP=N.
// dumps to <dir>/<tag>_step<S>_sub<I>_r<R>.txt.
// DENDRO_NAN_SCAN=1: per-phase NaN detector. Scans all dof fields of a
// CG vector; on first hit prints rank/step/stage/tag, per-var counts and
// the first few offending cgs (LOCAL vs ghost). Cheap enough to leave in
// (one pass over the vector per call site, gated off by default).
template <typename T>
static void ets_nan_scan(const ot::Mesh* pMesh, const T* vec,
                         unsigned int dof, const char* tag, int step,
                         int substage) {
    // plain getenv on purpose: DENDRO_PROBE_GETENV compiles to nullptr
    // without the DENDRO_ENABLE_DEBUG_PROBES build option, which silently
    // disabled this scanner in default builds (lesson learned 2026-06-12).
    static const char* ns_env = std::getenv("DENDRO_NAN_SCAN");
    static const bool ns_on =
        ns_env && ns_env[0] == '1' && ns_env[1] == '\0';
    if (!ns_on || !pMesh->isActive() || vec == nullptr) return;
    // one-shot pointer identity print (first few calls): lets the tracker
    // probe's `in` pointer be matched against the scanned buffer.
    static int ptr_prints = 0;
    if (ptr_prints < 12) {
        std::fprintf(stderr, "[nan-scan r%d %s] scanning ptr=%p dof=%u\n",
                     pMesh->getMPIRank(), tag, (const void*)vec, dof);
        std::fflush(stderr);
        ptr_prints++;
    }
    const unsigned int cgSz = pMesh->getDegOfFreedom();
    const unsigned int nLB  = pMesh->getNodeLocalBegin();
    const unsigned int nLE  = pMesh->getNodeLocalEnd();
    // targeted value dump: DENDRO_NAN_SCAN_DUMP_CG=<cg> prints all dof
    // values at that cg at every scan point (signature identifies the
    // writer of unphysical-but-finite values).
    {
        static const char* dc_env =
            std::getenv("DENDRO_NAN_SCAN_DUMP_CG");
        static const long dump_cg = dc_env ? std::atol(dc_env) : -1;
        if (dump_cg >= 0 && (unsigned long)dump_cg < cgSz) {
            std::fprintf(stderr, "[cg-dump r%d step=%d %s cg=%ld]",
                         pMesh->getMPIRank(), step, tag, dump_cg);
            for (unsigned int v = 0; v < dof; v++)
                std::fprintf(stderr, " %.3e",
                             (double)vec[(size_t)v * cgSz + dump_cg]);
            std::fprintf(stderr, "\n");
            std::fflush(stderr);
        }
    }
    size_t total            = 0;
    for (unsigned int v = 0; v < dof; v++) {
        size_t cnt        = 0;
        unsigned int firstCg[3];
        for (unsigned int cg = 0; cg < cgSz; cg++) {
            if (!std::isfinite((double)vec[(size_t)v * cgSz + cg])) {
                if (cnt < 3) firstCg[cnt] = cg;
                cnt++;
            }
        }
        if (cnt) {
            std::fprintf(stderr,
                         "[nan-scan r%d step=%d stage=%d %s] var=%u count=%zu"
                         " ptr=%p first:",
                         pMesh->getMPIRank(), step, substage, tag, v,
                         cnt, (const void*)vec);
            for (size_t k = 0; k < cnt && k < 3; k++)
                std::fprintf(stderr, " cg=%u(%s)", firstCg[k],
                             (firstCg[k] >= nLB && firstCg[k] < nLE)
                                 ? "LOCAL"
                                 : "ghost");
            std::fprintf(stderr, "\n");
            std::fflush(stderr);
            total += cnt;
        }
    }
}

template <typename T>
static void em4_cg_trace(const ot::Mesh* pMesh, const T* vec,
                         unsigned int dof, const char* tag, int step,
                         int substage) {
    static const char* trace_dir   = DENDRO_PROBE_GETENV("EM4_CG_TRACE_DIR");
    static const char* trace_phys  = DENDRO_PROBE_GETENV("EM4_CG_TRACE_PHYS");
    static const char* trace_bbox  = DENDRO_PROBE_GETENV("EM4_CG_TRACE_BBOX");
    static const char* trace_step_env = DENDRO_PROBE_GETENV("EM4_CG_TRACE_STEP");
    static const int trace_step_only =
        trace_step_env ? std::atoi(trace_step_env) : -1;
    // step-range support: EM4_CG_TRACE_STEP_MIN / _MAX (inclusive).
    // Overrides EM4_CG_TRACE_STEP if both set. -1 disables.
    static const char* trace_step_min_env =
        DENDRO_PROBE_GETENV("EM4_CG_TRACE_STEP_MIN");
    static const char* trace_step_max_env =
        DENDRO_PROBE_GETENV("EM4_CG_TRACE_STEP_MAX");
    static const int trace_step_min =
        trace_step_min_env ? std::atoi(trace_step_min_env) : -1;
    static const int trace_step_max =
        trace_step_max_env ? std::atoi(trace_step_max_env) : -1;
    // optional: only dump when tag starts with this prefix (e.g. "50" for postBcast)
    static const char* trace_tag_prefix = DENDRO_PROBE_GETENV("EM4_CG_TRACE_TAG_PREFIX");
    if (trace_tag_prefix && trace_tag_prefix[0] != '\0') {
        const size_t plen = std::strlen(trace_tag_prefix);
        if (std::strncmp(tag, trace_tag_prefix, plen) != 0) return;
    }
    static bool parsed             = false;
    static std::vector<std::tuple<unsigned long long, unsigned long long,
                                  unsigned long long>>
        targets;
    static unsigned long long bb_xlo = 0, bb_xhi = 0, bb_ylo = 0,
                              bb_yhi = 0, bb_zlo = 0, bb_zhi = 0;
    static bool bb_on = false;
    if (!trace_dir) return;
    if (!trace_phys && !trace_bbox) return;
    if (!pMesh || !pMesh->isActive()) return;
    if (trace_step_only >= 0 && step != trace_step_only) return;
    if (trace_step_min >= 0 && step < trace_step_min) return;
    if (trace_step_max >= 0 && step > trace_step_max) return;
    if (!parsed) {
        if (trace_phys) {
            std::string s(trace_phys);
            size_t pos = 0;
            while (pos < s.size()) {
                unsigned long long x, y, z;
                if (std::sscanf(s.c_str() + pos, "%llu,%llu,%llu", &x, &y, &z)
                    == 3) {
                    targets.emplace_back(x, y, z);
                }
                size_t next = s.find(';', pos);
                if (next == std::string::npos) break;
                pos = next + 1;
            }
        }
        if (trace_bbox) {
            std::sscanf(trace_bbox,
                "%llu,%llu,%llu,%llu,%llu,%llu",
                &bb_xlo, &bb_xhi, &bb_ylo, &bb_yhi, &bb_zlo, &bb_zhi);
            bb_on = true;
        }
        parsed = true;
    }
    if (targets.empty() && !bb_on) return;

    const unsigned int npe         = pMesh->getNumNodesPerElement();
    const unsigned int eOrd        = pMesh->getElementOrder();
    const unsigned int maxD        = m_uiMaxDepth;
    const unsigned int nLB         = pMesh->getNodeLocalBegin();
    const unsigned int nLE         = pMesh->getNodeLocalEnd();
    const unsigned int nTotal      = pMesh->getDegOfFreedom();
    const auto& cg2dg              = pMesh->getCG2DGMap();
    const auto& allElements        = pMesh->getAllElements();

    char fn[1024];
    std::snprintf(fn, sizeof(fn), "%s/%s_step%d_sub%d_r%d.txt", trace_dir,
                  tag, step, substage, (int)pMesh->getMPIRank());
    FILE* fp = std::fopen(fn, "w");
    if (!fp) return;
    std::fprintf(fp,
                 "# tag=%s step=%d substage=%d rank=%d dof=%u\n"
                 "# loc cg phys_x phys_y phys_z v hex "
                 "owner_ele owner_lev owner_x owner_y owner_z owner_sub_n "
                 "owner_sub_i owner_sub_j owner_sub_k\n",
                 tag, step, substage, (int)pMesh->getMPIRank(), dof);

    for (unsigned int cg = 0; cg < nTotal; cg++) {
        if (cg >= cg2dg.size()) continue;
        const unsigned int dg = cg2dg[cg];
        if (dg == LOOK_UP_TABLE_DEFAULT) continue;
        const unsigned int e = dg / npe;
        const unsigned int n = dg % npe;
        if (e >= allElements.size()) continue;
        const ot::TreeNode& tn = allElements[e];
        const unsigned int lev = tn.getLevel();
        if (lev > maxD) continue;
        const unsigned long long len = 1ull << (maxD - lev);
        const unsigned int ni        = n % (eOrd + 1);
        const unsigned int nj        = (n / (eOrd + 1)) % (eOrd + 1);
        const unsigned int nk        = n / ((eOrd + 1) * (eOrd + 1));
        const unsigned long long px =
            (unsigned long long)tn.getX() * eOrd + (unsigned long long)ni * len;
        const unsigned long long py =
            (unsigned long long)tn.getY() * eOrd + (unsigned long long)nj * len;
        const unsigned long long pz =
            (unsigned long long)tn.getZ() * eOrd + (unsigned long long)nk * len;

        bool match = false;
        if (bb_on) {
            if (px >= bb_xlo && px <= bb_xhi && py >= bb_ylo
                && py <= bb_yhi && pz >= bb_zlo && pz <= bb_zhi)
                match = true;
        }
        if (!match) {
            for (auto& t : targets) {
                if (px == std::get<0>(t) && py == std::get<1>(t)
                    && pz == std::get<2>(t)) {
                    match = true;
                    break;
                }
            }
        }
        if (!match) continue;

        const char loc = (cg >= nLB && cg < nLE) ? 'L' : 'G';
        for (unsigned int v = 0; v < dof; v++) {
            const T val = vec[v * nTotal + cg];
            uint64_t hb = 0;
            std::memcpy(&hb, &val, sizeof(hb));
            std::fprintf(fp,
                         "%c %u %llu %llu %llu %u %lx %u %u %u %u %u %u %u %u %u\n",
                         loc, cg, px, py, pz, v, (unsigned long)hb,
                         e, lev,
                         (unsigned)tn.getX(), (unsigned)tn.getY(),
                         (unsigned)tn.getZ(),
                         n, ni, nj, nk);
        }
    }
    std::fclose(fp);
}

// hanging-node E2N_CG resolution dumper. for every LOCAL elem on this rank
// and every sub_n that's hanging, dumps a row with: elem TN, sub indices,
// geometric phys, resolved cg index, resolved cg's phys, is_local. format
// is partition-independent (keyed by elem TN + sub_n + geom phys) so two
// runs (graph and SFC) can be diffed at the (TN, sub_n) level to find
// hanging-node routing differences.
//
// gate: EM4_HANG_DUMP_DIR + EM4_HANG_DUMP_STEP=N. dumps once per step
// matching N. file: <dir>/hang_step<N>_r<R>.txt.
static inline void em4_hang_dump(const ot::Mesh* pMesh, int step) {
    static const char* dir   = DENDRO_PROBE_GETENV("EM4_HANG_DUMP_DIR");
    static const char* s_env = DENDRO_PROBE_GETENV("EM4_HANG_DUMP_STEP");
    static const int s_only  = s_env ? std::atoi(s_env) : -1;
    if (!dir) return;
    if (s_only >= 0 && step != s_only) return;
    if (!pMesh || !pMesh->isActive()) return;

    const unsigned int npe  = pMesh->getNumNodesPerElement();
    const unsigned int eOrd = pMesh->getElementOrder();
    const unsigned int maxD = m_uiMaxDepth;
    const unsigned int nLB  = pMesh->getNodeLocalBegin();
    const unsigned int nLE  = pMesh->getNodeLocalEnd();
    const auto& cg2dg       = pMesh->getCG2DGMap();
    const auto& e2n         = pMesh->getE2NMapping();
    const auto& allEle      = pMesh->getAllElements();
    const unsigned int LB   = pMesh->getElementLocalBegin();
    const unsigned int LE   = pMesh->getElementLocalEnd();

    char fn[1024];
    std::snprintf(fn, sizeof(fn), "%s/hang_step%d_r%d.txt", dir, step,
                  (int)pMesh->getMPIRank());
    FILE* fp = std::fopen(fn, "w");
    if (!fp) return;
    std::fprintf(fp,
        "# step=%d rank=%d eOrd=%u\n"
        "# elem_lev elem_x elem_y elem_z sub_n sub_i sub_j sub_k "
        "geom_px geom_py geom_pz cg cg_px cg_py cg_pz is_local\n",
        step, (int)pMesh->getMPIRank(), eOrd);

    for (unsigned int e = LB; e < LE; e++) {
        const ot::TreeNode& tn = allEle[e];
        const unsigned int elev = tn.getLevel();
        if (elev == 0) continue;
        const unsigned long long elen = 1ull << (maxD - elev);
        for (unsigned int n = 0; n < npe; n++) {
            const unsigned int ni = n % (eOrd + 1);
            const unsigned int nj = (n / (eOrd + 1)) % (eOrd + 1);
            const unsigned int nk = n / ((eOrd + 1) * (eOrd + 1));
            // pMesh is const; isNodeHanging is a const method
            if (!pMesh->isNodeHanging(e, ni, nj, nk)) continue;
            const unsigned long long gpx =
                (unsigned long long)tn.getX() * eOrd
                + (unsigned long long)ni * elen;
            const unsigned long long gpy =
                (unsigned long long)tn.getY() * eOrd
                + (unsigned long long)nj * elen;
            const unsigned long long gpz =
                (unsigned long long)tn.getZ() * eOrd
                + (unsigned long long)nk * elen;
            const unsigned int cg = e2n[e * npe + n];
            // resolve cg -> phys via cg2dg
            unsigned long long cpx = 0, cpy = 0, cpz = 0;
            if (cg < cg2dg.size()) {
                const unsigned int dg = cg2dg[cg];
                if (dg != LOOK_UP_TABLE_DEFAULT) {
                    const unsigned int oe = dg / npe;
                    const unsigned int on = dg % npe;
                    if (oe < allEle.size()) {
                        const ot::TreeNode& oTN = allEle[oe];
                        const unsigned int olev = oTN.getLevel();
                        if (olev <= maxD) {
                            const unsigned long long olen = 1ull
                                << (maxD - olev);
                            const unsigned int oni = on % (eOrd + 1);
                            const unsigned int onj =
                                (on / (eOrd + 1)) % (eOrd + 1);
                            const unsigned int onk =
                                on / ((eOrd + 1) * (eOrd + 1));
                            cpx = (unsigned long long)oTN.getX() * eOrd
                                + (unsigned long long)oni * olen;
                            cpy = (unsigned long long)oTN.getY() * eOrd
                                + (unsigned long long)onj * olen;
                            cpz = (unsigned long long)oTN.getZ() * eOrd
                                + (unsigned long long)onk * olen;
                        }
                    }
                }
            }
            const int is_local = (cg >= nLB && cg < nLE) ? 1 : 0;
            std::fprintf(fp,
                "%u %u %u %u %u %u %u %u "
                "%llu %llu %llu %u %llu %llu %llu %d\n",
                elev, (unsigned)tn.getX(), (unsigned)tn.getY(),
                (unsigned)tn.getZ(), n, ni, nj, nk,
                gpx, gpy, gpz, cg, cpx, cpy, cpz, is_local);
        }
    }
    std::fclose(fp);
}

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
 * @brief General explicit time stepper class for Dendro-5.0
 * @tparam T
 */

#ifdef __PROFILE_ETS__
enum ETSPROFILE { EVOLVE = 0, ETS_LAST };
#endif

template <typename T, typename Ctx>
class ETS {
#ifdef __PROFILE_ETS__
   public:
    std::vector<profiler_t> m_uiCtxpt =
        std::vector<profiler_t>(static_cast<int>(ETSPROFILE::ETS_LAST));
    const char* ETSPROFILE_NAMES[static_cast<int>(ETSPROFILE::ETS_LAST)] = {
        "evolve"};

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

    /**@brief: number of stages*/
    unsigned int m_uiNumStages;

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

   private:
    /**
     * @brief Allocates internal variables for the time stepper.
     * @return int
     */
    int allocate_internal_vars();

    /**@brief: Deallocate internal variables. */
    int deallocate_internal_vars();

   public:
    /**
     * @brief Construct a new ETS object
     * @param pMesh : underlying mesh data structure.
     */
    ETS(Ctx* appCtx);

    /**
     * @brief Destroy the ETS object
     */
    ~ETS();

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
    void init();

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

    /**@brief: perform synchronizations with correct variable allocations for
     * the new mesh: should be called after remeshing.  */
    int sync_with_mesh();

    /**@brief: advance to next time step*/
    void evolve();

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

    if (type == ETSType::RK3) {
        m_uiNumStages                     = 3;

        static const DendroScalar ETS_C[] = {1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0};
        static const DendroScalar ETS_T[] = {0.0, 1.0, 1.0 / 2.0};
        static const DendroScalar ETS_U[] = {
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0 / 4.0, 1.0 / 4.0, 0.0};

        m_uiCi  = (DendroScalar*)ETS_T;
        m_uiBi  = (DendroScalar*)ETS_C;
        m_uiAij = (DendroScalar*)ETS_U;

        dendro::logger::debug(dendro::logger::Scope{"ETS"},
                              "ETS Coefficients set for RK3");

    } else if (type == ETSType::RK4) {
        m_uiNumStages                     = 4;

        static const DendroScalar ETS_C[] = {1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0,
                                             1.0 / 6.0};
        static const DendroScalar ETS_T[] = {0, 1.0 / 2.0, 1.0 / 2.0, 1.0};
        static const DendroScalar ETS_U[] = {
            0.0, 0.0,       0.0, 0.0, 1.0 / 2.0, 0.0, 0.0, 0.0,
            0.0, 1.0 / 2.0, 0.0, 0.0, 0.0,       0.0, 1.0, 0.0};

        m_uiCi  = (DendroScalar*)ETS_T;
        m_uiBi  = (DendroScalar*)ETS_C;
        m_uiAij = (DendroScalar*)ETS_U;

        dendro::logger::debug(dendro::logger::Scope{"ETS"},
                              "ETS Coefficients set for RK4");

    } else if (type == ETSType::RK5) {
        return -1;
        // these coefficients looks wrong. - Milinda. (need to fix those and
        // enable the rk5 method. )
        m_uiNumStages                     = 5;
        static const DendroScalar ETS_C[] = {
            7.0 / 90.0, 0, 32 / 90.0, 12.0 / 90.0, 32.0 / 90.0, 7.0 / 90.0};
        static const DendroScalar ETS_T[] = {0,         1.0 / 4.0, 1.0 / 4.0,
                                             1.0 / 2.0, 3.0 / 4.0, 1.0};
        static const DendroScalar ETS_U[] = {
            0.0,        0.0,        0.0,        0.0,         0.0,
            1.0 / 8.0,  1.0 / 8.0,  0.0,        0.0,         0.0,
            0.0,        -1.0 / 2.0, 1.0,        0.0,         0.0,
            3.0 / 16.0, 0.0,        0.0,        9.0 / 16.0,  0.0,
            -3.0 / 7.0, 2.0 / 7.0,  12.0 / 7.0, -12.0 / 7.0, 8.0 / 7.0};

        m_uiCi  = (DendroScalar*)ETS_T;
        m_uiBi  = (DendroScalar*)ETS_C;
        m_uiAij = (DendroScalar*)ETS_U;

        dendro::logger::debug(dendro::logger::Scope{"ETS"},
                              "ETS Coefficients set for RK5");

    } else {
        dendro::logger::error(dendro::logger::Scope{"ETS"},
                              "UNKNOWN ETS TYPE (supports RK3, RK4, and RK5)");
        return -1;
    }

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

    // env-gated breadcrumb tracer. set DENDRO_EVOLVE_TRACE=1 to enable.
    // prints per-rank stderr line at each named checkpoint inside evolve().
    // used to bisect a stall between "Current Step: N" and "Current Step: N+1".
    static const bool _ev_trace = []() {
        const char* v = DENDRO_PROBE_GETENV("DENDRO_EVOLVE_TRACE");
        return v && v[0] == '1' && v[1] == '\0';
    }();
#define _EV_TRACE(tag) do { \
    if (_ev_trace) { \
        int _r = -1; MPI_Comm_rank(MPI_COMM_WORLD, &_r); \
        std::fprintf(stderr, "[ETV r%d step=%lu] %s\n", \
                     _r, (unsigned long)m_uiTimeInfo._m_uiStep, tag); \
        std::fflush(stderr); \
    } \
} while (0)
    _EV_TRACE("00_enter");

    dendro::logger::debug(dendro::logger::Scope{"ETS"},
                          "Beginning ETS evolve (overall time step: {})",
                          m_uiTimeInfo._m_uiStep);

    const ot::Mesh* pMesh  = m_uiAppCtx->get_mesh();

    // EARLIEST evolve() trace point — fires before pre_timestep, before
    // sync_with_mesh effects propagate, before any other state-touching
    // code. used to bisect divergence between end-of-remesh and step start.
    em4_cg_trace(pMesh, m_uiEVar.get_vec_ptr(), m_uiEVar.get_dof(),
                 "98_evolve_entry", (int)m_uiTimeInfo._m_uiStep, -1);
    m_uiTimeInfo           = m_uiAppCtx->get_ts_info();
    const double current_t = m_uiTimeInfo._m_uiT;
    double current_t_adv   = current_t;
    const double dt        = m_uiTimeInfo._m_uiTh;

    ets_nan_scan(pMesh, m_uiEVar.get_vec_ptr(), m_uiEVar.get_dof(),
                 "00_evolveEnter", (int)m_uiTimeInfo._m_uiStep, -1);
    _EV_TRACE("01_pre_timestep_start");
    m_uiAppCtx->pre_timestep(m_uiEVar);
    _EV_TRACE("02_pre_timestep_done");
    ets_nan_scan(pMesh, m_uiEVar.get_vec_ptr(), m_uiEVar.get_dof(),
                 "05_postPreTimestep", (int)m_uiTimeInfo._m_uiStep, -1);

    const unsigned int DOF    = m_uiEVar.get_dof();
    const unsigned int szPDof = pMesh->getDegOfFreedom();

    const int trace_step = (int)m_uiTimeInfo._m_uiStep;
    em4_cg_trace(pMesh, m_uiEVar.get_vec_ptr(), m_uiEVar.get_dof(),
                 "00_step_start", trace_step, -1);
    em4_hang_dump(pMesh, trace_step);

    // env-gated element-set dump at start of step. used to detect AMR
    // refinement decision drift between graph and skip partitioning.
    // gate: EM4_ELEMSET_DUMP_DIR=/path + EM4_ELEMSET_DUMP_STEPS="N1,N2,...".
    // for each matching step, every rank dumps its sorted LOCAL element
    // list (TN coords + level) to <dir>/elemset_step<N>_r<R>.txt.
    {
        static const char* esd_dir = DENDRO_PROBE_GETENV("EM4_ELEMSET_DUMP_DIR");
        static const char* esd_steps_env =
            DENDRO_PROBE_GETENV("EM4_ELEMSET_DUMP_STEPS");
        if (esd_dir && esd_steps_env && pMesh->isActive()) {
            static std::set<int> esd_steps;
            static bool esd_parsed = false;
            if (!esd_parsed) {
                std::string s(esd_steps_env);
                size_t p = 0;
                while (p < s.size()) {
                    size_t n = s.find(',', p);
                    if (n == std::string::npos) n = s.size();
                    if (n > p)
                        esd_steps.insert(std::atoi(
                            s.substr(p, n - p).c_str()));
                    p = n + 1;
                }
                esd_parsed = true;
            }
            if (esd_steps.count(trace_step)) {
                const auto& allEle = pMesh->getAllElements();
                const unsigned int LB = pMesh->getElementLocalBegin();
                const unsigned int LE = pMesh->getElementLocalEnd();
                const int rank = pMesh->getMPIRank();
                std::vector<std::tuple<unsigned int, unsigned int,
                                       unsigned int, unsigned int>> locals;
                for (unsigned int e = LB; e < LE; e++) {
                    const auto& t = allEle[e];
                    locals.emplace_back(t.getLevel(), t.getX(), t.getY(),
                                        t.getZ());
                }
                std::sort(locals.begin(), locals.end());
                char fn[1024];
                std::snprintf(fn, sizeof(fn),
                              "%s/elemset_step%d_r%d.txt",
                              esd_dir, trace_step, rank);
                FILE* fp = std::fopen(fn, "w");
                if (fp) {
                    std::fprintf(fp,
                        "# step=%d rank=%d numLocal=%u\n"
                        "# lev x y z\n",
                        trace_step, rank, LE - LB);
                    for (const auto& t : locals) {
                        std::fprintf(fp, "%u %u %u %u\n",
                                     std::get<0>(t), std::get<1>(t),
                                     std::get<2>(t), std::get<3>(t));
                    }
                    std::fclose(fp);
                }
            }
        }
    }
    _EV_TRACE("03_pre_stages_loop");

    if (pMesh->isActive()) {
        int rank                              = pMesh->getMPIRank();

        const unsigned int nodeLocalBegin     = pMesh->getNodeLocalBegin();
        const unsigned int nodeLocalEnd       = pMesh->getNodeLocalEnd();

        const std::vector<ot::Block>& blkList = pMesh->getLocalBlockList();
        unsigned int offset;
        double ptmin[3], ptmax[3];
        unsigned int sz[3];
        unsigned int bflag;
        double dx, dy, dz;

        for (int stage = 0; stage < m_uiNumStages; stage++) {
            dendro::logger::debug(dendro::logger::Scope{"ETS"},
                                  "Now executing ETS Evolve stage {}/{}",
                                  stage + 1, m_uiNumStages);
            if (_ev_trace) {
                int _r = -1; MPI_Comm_rank(MPI_COMM_WORLD, &_r);
                std::fprintf(stderr,
                    "[ETV r%d step=%lu stage=%d] 10_stage_enter\n", _r,
                    (unsigned long)m_uiTimeInfo._m_uiStep, stage);
                std::fflush(stderr);
            }

            m_uiEVecTmp[0].copy_data(m_uiEVar);

            for (int p = 0; p < stage; p++)
                DVec::axpy(m_uiAppCtx->get_mesh(),
                           m_uiAij[(stage)*m_uiNumStages + p] * dt,
                           m_uiStVec[p], m_uiEVecTmp[0]);
            if (_ev_trace) {
                int _r = -1; MPI_Comm_rank(MPI_COMM_WORLD, &_r);
                std::fprintf(stderr,
                    "[ETV r%d step=%lu stage=%d] 11_pre_post_timestep\n", _r,
                    (unsigned long)m_uiTimeInfo._m_uiStep, stage);
                std::fflush(stderr);
            }
            m_uiAppCtx->post_timestep(m_uiEVecTmp[0]);
            if (_ev_trace) {
                int _r = -1; MPI_Comm_rank(MPI_COMM_WORLD, &_r);
                std::fprintf(stderr,
                    "[ETV r%d step=%lu stage=%d] 12_post_post_timestep\n", _r,
                    (unsigned long)m_uiTimeInfo._m_uiStep, stage);
                std::fflush(stderr);
            }

            // per-substage sync: force m_uiEVecTmp[0] to be partition-
            // invariant before the RHS unzip reads it. without this, axpy
            // accumulates substage-internal drift from RHS+zip at non-
            // consensus cgs, producing 1-ULP off in the post-step state.
            current_t_adv = current_t + m_uiCi[stage] * dt;
            em4_cg_trace(pMesh, m_uiEVecTmp[0].get_vec_ptr(),
                         m_uiEVecTmp[0].get_dof(), "10_preRHS",
                         trace_step, stage);
            ets_nan_scan(pMesh, m_uiEVecTmp[0].get_vec_ptr(),
                         m_uiEVecTmp[0].get_dof(), "10_preRHS",
                         trace_step, stage);
            if (_ev_trace) {
                int _r = -1; MPI_Comm_rank(MPI_COMM_WORLD, &_r);
                std::fprintf(stderr,
                    "[ETV r%d step=%lu stage=%d] 13_pre_pre_stage\n", _r,
                    (unsigned long)m_uiTimeInfo._m_uiStep, stage);
                std::fflush(stderr);
            }
            m_uiAppCtx->pre_stage(m_uiStVec[stage]);
            if (_ev_trace) {
                int _r = -1; MPI_Comm_rank(MPI_COMM_WORLD, &_r);
                std::fprintf(stderr,
                    "[ETV r%d step=%lu stage=%d] 14_pre_rhs\n", _r,
                    (unsigned long)m_uiTimeInfo._m_uiStep, stage);
                std::fflush(stderr);
            }
            m_uiAppCtx->rhs(&m_uiEVecTmp[0], &m_uiStVec[stage], 1,
                            current_t_adv);
            if (_ev_trace) {
                int _r = -1; MPI_Comm_rank(MPI_COMM_WORLD, &_r);
                std::fprintf(stderr,
                    "[ETV r%d step=%lu stage=%d] 15_post_rhs\n", _r,
                    (unsigned long)m_uiTimeInfo._m_uiStep, stage);
                std::fflush(stderr);
            }
            m_uiAppCtx->post_stage(m_uiStVec[stage]);
            em4_cg_trace(pMesh, m_uiStVec[stage].get_vec_ptr(),
                         m_uiStVec[stage].get_dof(), "20_postRHS",
                         trace_step, stage);
            ets_nan_scan(pMesh, m_uiStVec[stage].get_vec_ptr(),
                         m_uiStVec[stage].get_dof(), "20_postRHS",
                         trace_step, stage);

            if (_ev_trace) {
                int _r = -1; MPI_Comm_rank(MPI_COMM_WORLD, &_r);
                std::fprintf(stderr,
                    "[ETV r%d step=%lu stage=%d] 16_stage_exit\n", _r,
                    (unsigned long)m_uiTimeInfo._m_uiStep, stage);
                std::fflush(stderr);
            }
        }

        dendro::logger::debug(dendro::logger::Scope{"ETS"},
                              "Calculating next step after stages");
        _EV_TRACE("20_pre_final_axpy");
        for (unsigned int k = 0; k < m_uiNumStages; k++)
            DVec::axpy(m_uiAppCtx->get_mesh(), m_uiBi[k] * dt, m_uiStVec[k],
                       m_uiEVar);
        _EV_TRACE("21_post_final_axpy");
        em4_cg_trace(pMesh, m_uiEVar.get_vec_ptr(), m_uiEVar.get_dof(),
                     "30_postFinalAxpy", trace_step, -1);
        ets_nan_scan(pMesh, m_uiEVar.get_vec_ptr(), m_uiEVar.get_dof(),
                     "30_postFinalAxpy", trace_step, -1);
    }
    _EV_TRACE("22_pre_post_axpy_sync");

    // post-axpy sync: the final axpy `m_uiEVar += dt*B*stVec` accumulates
    // RHS into local cgs, but plan-zip only fills RHS at PRIMARY cgs
    // (per phys position). Non-primary local cgs (duplicates created by
    // graph repartitioning) get RHS=0, so the axpy leaves them at last-
    // step values while primaries advance. The drift compounds per step
    // → 50× U_E2 residual at step 230. Force non-primary cgs to match
    // primary's post-axpy value via the side-channel Alltoallv. Gated
    // OFF by default to allow A/B testing — see project_corner_drift_*.
    // env DENDRO_DISABLE_POST_AXPY_SYNC=1 disables.
    {
        static const char* nps_env =
            std::getenv("DENDRO_DISABLE_POST_AXPY_SYNC");
        const bool skip_post_axpy_sync =
            nps_env && nps_env[0] == '1' && nps_env[1] == '\0';
        if (!skip_post_axpy_sync && pMesh->isActive()) {
            T* dvec_ptr = m_uiEVar.get_vec_ptr();
            const unsigned int DOFV = m_uiEVar.get_dof();
            ot::Mesh* pMeshMut = const_cast<ot::Mesh*>(pMesh);
            pMeshMut->syncZipNonPrimaryPublic(dvec_ptr, DOFV);
            em4_cg_trace(pMesh, m_uiEVar.get_vec_ptr(), m_uiEVar.get_dof(),
                         "40_postSync", trace_step, -1);
            ets_nan_scan(pMesh, m_uiEVar.get_vec_ptr(), m_uiEVar.get_dof(),
                         "40_postSync", trace_step, -1);
            // additional pass: position-keyed broadcast (gated by
            // DENDRO_FORCE_POS_BCAST=1). brings every cg at consensus
            // phys_pos into bit-identity across ranks, plus zeros out
            // cgs at phys positions with no canonical writer anywhere
            // (matches SFC's far-field hanging-position IC).
            pMeshMut->broadcastCgValuesByPhysPosPublic(dvec_ptr, DOFV);
            em4_cg_trace(pMesh, m_uiEVar.get_vec_ptr(), m_uiEVar.get_dof(),
                         "50_postBcast", trace_step, -1);
            ets_nan_scan(pMesh, m_uiEVar.get_vec_ptr(), m_uiEVar.get_dof(),
                         "50_postBcast", trace_step, -1);
        }
    }

    _EV_TRACE("30_pre_final_post_timestep");
    m_uiAppCtx->post_timestep(m_uiEVar);
    _EV_TRACE("31_post_final_post_timestep");
    ets_nan_scan(pMesh, m_uiEVar.get_vec_ptr(), m_uiEVar.get_dof(),
                 "60_postTimestep", trace_step, -1);

    m_uiAppCtx->increment_ts_info();
    m_uiTimeInfo = m_uiAppCtx->get_ts_info();
    _EV_TRACE("32_pre_waitAll");
    pMesh->waitAll();
    _EV_TRACE("33_post_waitAll");

    // probe: dump cg values at specific step boundaries to bisect the
    // 1-ULP partition-dependent seed. set EM4_STEP_DUMP_DIR=/path AND
    // EM4_STEP_DUMP_STEPS="8,9,10". dumps non-hanging local cgs keyed
    // by (phys_x, phys_y, phys_z). m_uiEVar at this point holds the
    // just-completed step's state.
    {
        const char* dump_dir = DENDRO_PROBE_GETENV("EM4_STEP_DUMP_DIR");
        const char* dump_steps_env = DENDRO_PROBE_GETENV("EM4_STEP_DUMP_STEPS");
        if (dump_dir && dump_steps_env && pMesh->isActive()) {
            std::set<long> dump_steps;
            std::string s_in(dump_steps_env);
            size_t spos = 0;
            while (spos < s_in.size()) {
                size_t nx = s_in.find(',', spos);
                if (nx == std::string::npos) nx = s_in.size();
                if (nx > spos)
                    dump_steps.insert(std::atol(
                        s_in.substr(spos, nx - spos).c_str()));
                spos = nx + 1;
            }
            // increment_ts_info bumped the step; we dump the value of
            // the just-completed step (the new step minus 1).
            const long cur_step = (long)m_uiTimeInfo._m_uiStep - 1;
            if (dump_steps.count(cur_step)) {
                T* dvec_ptr = m_uiEVar.get_vec_ptr();
                const unsigned int DOFV = m_uiEVar.get_dof();
                const unsigned int szPDofL = pMesh->getDegOfFreedom();
                const unsigned int* e2n =
                    &(*(pMesh->getE2NMapping().begin()));
                const unsigned int* e2n_dg_ =
                    &(*(pMesh->getE2NMapping_DG().begin()));
                const auto& pNodes = pMesh->getAllElements();
                const unsigned int NLB = pMesh->getNodeLocalBegin();
                const unsigned int NLE = pMesh->getNodeLocalEnd();
                const unsigned int ELB = pMesh->getElementLocalBegin();
                const unsigned int ELE = pMesh->getElementLocalEnd();
                const unsigned int eOrd = pMesh->getElementOrder();
                const unsigned int nPe = pMesh->getNumNodesPerElement();
                const unsigned int maxD = m_uiMaxDepth;
                // dump by CG INDEX (all local cgs), using m_uiCG2DG
                // to find canonical (oe, on) for phys position.
                const auto& cg2dg = pMesh->getCG2DGMap();
                for (unsigned int v = 0; v < DOFV; v++) {
                    T* vec_v = dvec_ptr + v * szPDofL;
                    char fn[1024];
                    std::snprintf(fn, sizeof(fn),
                        "%s/step%ld_v%u_r%d.txt",
                        dump_dir, cur_step, v,
                        (int)pMesh->getMPIRank());
                    FILE* fp = std::fopen(fn, "w");
                    if (!fp) continue;
                    std::fprintf(fp,
                        "# step=%ld v=%u rank=%d NLB=%u NLE=%u\n"
                        "# cg phys_x phys_y phys_z hex\n",
                        cur_step, v, (int)pMesh->getMPIRank(),
                        NLB, NLE);
                    for (unsigned int cg = NLB; cg < NLE; cg++) {
                        if (cg >= cg2dg.size()) continue;
                        const unsigned int dg = cg2dg[cg];
                        const unsigned int oe = dg / nPe;
                        const unsigned int on = dg % nPe;
                        if (oe >= pNodes.size()) continue;
                        const unsigned int oni = on % (eOrd+1);
                        const unsigned int onj =
                            (on/(eOrd+1)) % (eOrd+1);
                        const unsigned int onk =
                            on / ((eOrd+1)*(eOrd+1));
                        const ot::TreeNode& oTN = pNodes[oe];
                        const unsigned int olen =
                            (unsigned int)1u
                            << (maxD - oTN.getLevel());
                        const unsigned long long px =
                            (unsigned long long)oTN.getX() * eOrd
                            + (unsigned long long)oni * olen;
                        const unsigned long long py =
                            (unsigned long long)oTN.getY() * eOrd
                            + (unsigned long long)onj * olen;
                        const unsigned long long pz =
                            (unsigned long long)oTN.getZ() * eOrd
                            + (unsigned long long)onk * olen;
                        uint64_t hb = 0;
                        T val = vec_v[cg];
                        std::memcpy(&hb, &val, sizeof(hb));
                        std::fprintf(fp,
                            "%u %llu %llu %llu %lx\n",
                            cg, px, py, pz, (unsigned long)hb);
                    }
                    std::fclose(fp);
                }
            }
        }
    }

    _EV_TRACE("99_evolve_exit");
#undef _EV_TRACE

#ifdef __PROFILE_ETS__
    m_uiCtxpt[ETSPROFILE::EVOLVE].stop();
#endif

    dendro::logger::debug(dendro::logger::Scope{"ETS"},
                          "ETS evolve step finished!");
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
