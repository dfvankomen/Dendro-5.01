// Temporal order of every BSSN_RK_TYPE scheme on a real octree, through the
// actual ts::ETS / ts::ETS_MSRK / ts::ETS_TSRK path.
//   decay : du/dt = -lambda*u, error vs the analytic solution
//   adv   : du/dt = -a*du/dx,  error vs a fine-dt RK6 reference on the same mesh
// Run:  ./testMeshTimeIntegrator maxDepth eleOrder [decay|adv|both]
//       exit 0 = all pass, 1 = failure

#include <cmath>
#include <cstdio>
#include <functional>
#include <memory>
#include <vector>

#include "TreeNode.h"
#include "ctx.h"
#include "dendro.h"
#include "derivatives.h"
#include "dvec.h"
#include "ets.h"
#include "ets_msrk.h"
#include "ets_tsrk.h"
#include "hcurvedata.h"
#include "mesh.h"
#include "meshUtils.h"
#include "mpi.h"
#include "octUtils.h"
#include "oda.h"

using DVec = ot::DVector<DendroScalar, unsigned int>;

static const double LAMBDA        = 1.0;   // decay rate
static const double AVEL          = 1.0;   // advection speed
static const unsigned int ASYNC_K = 1;

static double ic_decay(double x, double y, double z) {
    return 1.0 + 0.5 * std::sin(2.0 * M_PI * x) * std::sin(2.0 * M_PI * y) *
                     std::sin(2.0 * M_PI * z);
}
static double ic_adv(double x) { return std::sin(2.0 * M_PI * x); }

// ---------------------------------------------------------------------------
class MeshTICtx : public ts::Ctx<MeshTICtx, DendroScalar, unsigned int> {
    DVec m_evar;                 // zipped evolution var
    DVec m_unz, m_dudx_unz;      // unzip work buffers (advection)
    std::vector<double> m_ic;    // per-node IC value (decay) / x-coord (adv)
    bool m_adv;
    std::unique_ptr<dendroderivs::DendroDerivatives> m_deriv;
    Point m_pmin{0.0, 0.0, 0.0}, m_pmax{1.0, 1.0, 1.0};

   public:
    MeshTICtx(ot::Mesh* pMesh, bool advection)
        : ts::Ctx<MeshTICtx, DendroScalar, unsigned int>(), m_adv(advection) {
        m_uiMesh = pMesh;
        m_evar.create_vector(m_uiMesh, ot::DVEC_TYPE::OCT_SHARED_NODES,
                             ot::DVEC_LOC::HOST, 1, true);
        m_uiTinfo = {0.0, 0.0, 0, 0.0, 0.0};
        ot::alloc_mpi_ctx<DendroScalar>(m_uiMesh, m_mpi_ctx, 1, ASYNC_K);
        if (m_adv) {
            m_unz.create_vector(m_uiMesh, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING,
                                ot::DVEC_LOC::HOST, 1, true);
            m_dudx_unz.create_vector(m_uiMesh,
                                     ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING,
                                     ot::DVEC_LOC::HOST, 1, true);
            m_deriv = std::make_unique<dendroderivs::DendroDerivatives>(
                "BL6", "JTT6", m_uiMesh->getElementOrder(),
                std::vector<double>(), std::vector<double>(), 1, 1);
            unsigned int mx = 0;
            for (const auto& b : m_uiMesh->getLocalBlockList())
                mx = std::max(mx, b.getAllocationSzX() * b.getAllocationSzY() *
                                      b.getAllocationSzZ());
            m_deriv->set_maximum_block_size(mx);
        }
    }
    ~MeshTICtx() {
        m_evar.destroy_vector();
        if (m_adv) {
            m_unz.destroy_vector();
            m_dudx_unz.destroy_vector();
        }
        ot::dealloc_mpi_ctx<DendroScalar>(m_uiMesh, m_mpi_ctx, 1, ASYNC_K);
    }

    int init_grid() {
        const ot::TreeNode* pNodes  = &(*(m_uiMesh->getAllElements().begin()));
        const unsigned int eo       = m_uiMesh->getElementOrder();
        const unsigned int* e2n_cg  = &(*(m_uiMesh->getE2NMapping().begin()));
        const unsigned int* e2n_dg  = &(*(m_uiMesh->getE2NMapping_DG().begin()));
        const unsigned int nPe      = m_uiMesh->getNumNodesPerElement();
        const unsigned int nB       = m_uiMesh->getNodeLocalBegin();
        const unsigned int nE       = m_uiMesh->getNodeLocalEnd();
        DendroScalar* u;
        m_evar.to_2d(&u);
        m_ic.assign(m_uiMesh->getDegOfFreedom(), 0.0);
        const double maxd = (double)(1u << m_uiMaxDepth);
        for (unsigned int e = m_uiMesh->getElementLocalBegin();
             e < m_uiMesh->getElementLocalEnd(); e++)
            for (unsigned int k = 0; k < eo + 1; k++)
                for (unsigned int j = 0; j < eo + 1; j++)
                    for (unsigned int i = 0; i < eo + 1; i++) {
                        const unsigned int cg =
                            e2n_cg[e * nPe + k * (eo + 1) * (eo + 1) +
                                   j * (eo + 1) + i];
                        if (cg < nB || cg >= nE) continue;
                        const unsigned int dg =
                            e2n_dg[e * nPe + k * (eo + 1) * (eo + 1) +
                                   j * (eo + 1) + i];
                        unsigned int own, ix, jy, kz;
                        m_uiMesh->dg2eijk(dg, own, ix, jy, kz);
                        const double len =
                            (double)(1u << (m_uiMaxDepth - pNodes[own].getLevel()));
                        const double x = (pNodes[own].getX() + ix * len / eo) / maxd;
                        const double y = (pNodes[own].getY() + jy * len / eo) / maxd;
                        const double z = (pNodes[own].getZ() + kz * len / eo) / maxd;
                        u[cg]     = m_adv ? ic_adv(x) : ic_decay(x, y, z);
                        m_ic[cg]  = m_adv ? x : u[cg];   // store x (adv) or IC
                    }
        return 0;
    }
    int initialize() { return init_grid(); }

    int rhs(DVec* in, DVec* out, unsigned int sz, DendroScalar time) {
        if (!m_adv) {
            DendroScalar *pin, *pout;
            in->to_2d(&pin);
            out->to_2d(&pout);
            for (unsigned int n = m_uiMesh->getNodeLocalBegin();
                 n < m_uiMesh->getNodeLocalEnd(); n++)
                pout[n] = -LAMBDA * pin[n];
            return 0;
        }
        // advection: out = -a * du/dx  via unzip -> grad_x -> scale -> zip
        this->unzip(*in, m_unz);
        DendroScalar *uUnz, *dUnz;
        m_unz.to_2d(&uUnz);
        m_dudx_unz.to_2d(&dUnz);
        for (const auto& b : m_uiMesh->getLocalBlockList()) {
            const DendroIntL off   = b.getOffset();
            unsigned int s[3] = {b.getAllocationSzX(), b.getAllocationSzY(),
                                 b.getAllocationSzZ()};
            const double dx = b.computeDx(m_pmin, m_pmax);
            m_deriv->grad_x(&dUnz[off], &uUnz[off], dx, s, b.getBlkNodeFlag());
            for (unsigned int q = 0; q < s[0] * s[1] * s[2]; q++)
                dUnz[off + q] *= -AVEL;
        }
        this->zip(m_dudx_unz, *out);
        return 0;
    }

    double max_error() {
        DendroScalar* u;
        m_evar.to_2d(&u);
        const double t = m_uiTinfo._m_uiT;
        double e       = 0.0;
        for (unsigned int n = m_uiMesh->getNodeLocalBegin();
             n < m_uiMesh->getNodeLocalEnd(); n++) {
            double ex;
            if (m_adv) {
                const double x = m_ic[n];
                if (x < 0.15 || x > 0.85) continue;   // skip boundary region
                ex = ic_adv(x - AVEL * t);
            } else {
                ex = m_ic[n] * std::exp(-LAMBDA * t);
            }
            e = std::max(e, std::fabs(u[n] - ex));
        }
        double eg = e;
        MPI_Allreduce(&e, &eg, 1, MPI_DOUBLE, MPI_MAX,
                      m_uiMesh->getMPIGlobalCommunicator());
        return eg;
    }

    // Masked identically to max_error(), so the two are comparable.
    std::vector<double> interior_state() {
        DendroScalar* u;
        m_evar.to_2d(&u);
        std::vector<double> out;
        for (unsigned int n = m_uiMesh->getNodeLocalBegin();
             n < m_uiMesh->getNodeLocalEnd(); n++) {
            if (m_adv) {
                const double x = m_ic[n];
                if (x < 0.15 || x > 0.85) continue;
            }
            out.push_back(u[n]);
        }
        return out;
    }

    double max_error_vs(const std::vector<double>& ref) {
        const std::vector<double> cur = interior_state();
        double e                      = 0.0;
        for (std::size_t i = 0; i < cur.size() && i < ref.size(); i++)
            e = std::max(e, std::fabs(cur[i] - ref[i]));
        double eg = e;
        MPI_Allreduce(&e, &eg, 1, MPI_DOUBLE, MPI_MAX,
                      m_uiMesh->getMPIGlobalCommunicator());
        return eg;
    }

    double max_abs() {
        DendroScalar* u;
        m_evar.to_2d(&u);
        double m = 0.0;
        for (unsigned int n = m_uiMesh->getNodeLocalBegin();
             n < m_uiMesh->getNodeLocalEnd(); n++)
            m = std::max(m, std::fabs(u[n]));
        double mg = m;
        MPI_Allreduce(&m, &mg, 1, MPI_DOUBLE, MPI_MAX,
                      m_uiMesh->getMPIGlobalCommunicator());
        return mg;
    }

    DVec& get_evolution_vars() { return m_evar; }
    DVec& get_constraint_vars() { return m_evar; }
    DVec& get_primitive_vars() { return m_evar; }

    // ---- required Ctx interface (stubs) ----
    int pre_stage(DVec) { return 0; }
    int post_stage(DVec) { return 0; }
    int pre_timestep(DVec) { return 0; }
    int post_timestep(DVec) { return 0; }
    int rhs_blk(const DendroScalar*, DendroScalar*, unsigned int, unsigned int,
                DendroScalar) { return 0; }
    int pre_stage_blk(DendroScalar*, unsigned int, unsigned int, DendroScalar) { return 0; }
    int post_stage_blk(DendroScalar*, unsigned int, unsigned int, DendroScalar) { return 0; }
    int pre_timestep_blk(DendroScalar*, unsigned int, unsigned int, DendroScalar) { return 0; }
    int post_timestep_blk(DendroScalar*, unsigned int, unsigned int, DendroScalar) { return 0; }
    bool is_remesh() { return false; }
    int write_vtu() { return 0; }
    int write_checkpt() { return 0; }
    int restore_checkpt() { return 0; }
    int finalize() { return 0; }
    int terminal_output() { return 0; }
    unsigned int get_async_batch_sz() { return ASYNC_K; }
    unsigned int get_num_refine_vars() { return 1; }
    const unsigned int* get_refine_var_ids() {
        static unsigned int ids[1] = {0};
        return ids;
    }
    int grid_transfer(const ot::Mesh*) { return 0; }
    static unsigned int getBlkTimestepFac(unsigned int, unsigned int, unsigned int) {
        return 1;
    }
};

// ---------------------------------------------------------------------------
// kind: 0 = single-step ETS, 1 = ETS_MSRK, 2 = ETS_TSRK
struct Scheme {
    ts::ETSType type;
    const char* name;
    int kind;
    int order;
    double imag_reach;  // ODE imag-axis reach, from testTimeIntegratorOrder
};

// imag_reach is reported by testTimeIntegratorOrder, not computed here.
static const std::vector<Scheme> SCHEMES = {
    {ts::ETSType::RK3, "RK3", 0, 3, 1.7321},
    {ts::ETSType::RK4, "RK4", 0, 4, 2.8284},
    {ts::ETSType::RK5, "RK5", 0, 5, 0.8523},
    {ts::ETSType::RK4_MSRK2_1, "RK4_MSRK2_1", 1, 4, 2.5387},
    {ts::ETSType::RK4_MSRK2_2, "RK4_MSRK2_2", 1, 4, 2.4620},
    {ts::ETSType::RK4_MSRK3, "RK4_MSRK3", 1, 4, 1.3071},
    {ts::ETSType::RK4_RALSTON, "RK4_RALSTON", 0, 4, 2.8284},
    {ts::ETSType::RK45_CASH_KARP, "RK45_CASH_KARP", 0, 5, 0.0430},
    {ts::ETSType::RKF45, "RKF45", 0, 5, 0.0320},
    {ts::ETSType::RK5_NYSTROM, "RK5_NYSTROM", 0, 5, 0.0290},
    {ts::ETSType::RK6, "RK6", 0, 6, 0.0790},
    {ts::ETSType::RK6_TSRK, "RK6_TSRK", 2, 6, 1.8500},
};

// blowup > 0 stops early and returns false once |u| exceeds it; NaN counts.
template <typename ETST>
static bool drive(ETST& ets, MeshTICtx& ctx, int nsteps, double blowup) {
    ets.init();
    for (int i = 0; i < nsteps; i++) {
        ets.evolve();
        if (blowup > 0.0 && !(ctx.max_abs() < blowup)) return false;
    }
    return true;
}

static bool evolve_steps(MeshTICtx& ctx, const Scheme& sc, double dt,
                         int nsteps, double blowup) {
    ctx.set_ts_info({0.0, dt * nsteps, 0, 0.0, dt});
    if (sc.kind == 1) {
        ts::ETS_MSRK<DendroScalar, MeshTICtx> ets(&ctx, sc.type);
        ets.set_ets_coefficients(ts::ETSType::RK4);
        return drive(ets, ctx, nsteps, blowup);
    }
    if (sc.kind == 2) {
        ts::ETS_TSRK<DendroScalar, MeshTICtx> ets(&ctx);
        return drive(ets, ctx, nsteps, blowup);
    }
    ts::ETS<DendroScalar, MeshTICtx> ets(&ctx);
    ets.set_ets_coefficients(sc.type);
    return drive(ets, ctx, nsteps, blowup);
}

static double run_decay(ot::Mesh* mesh, const Scheme& sc, double T,
                        std::vector<double>& errs_out) {
    const std::vector<int> Ns = {10, 20, 40, 80};
    errs_out.clear();
    for (int N : Ns) {
        MeshTICtx ctx(mesh, false);
        evolve_steps(ctx, sc, T / N, N, 0.0);
        errs_out.push_back(ctx.max_error());
    }
    double best = 0.0;
    for (std::size_t i = 0; i + 1 < errs_out.size(); i++)
        if (errs_out[i] > 1e-13 && errs_out[i + 1] > 1e-14)
            best = std::max(best, std::log2(errs_out[i] / errs_out[i + 1]));
    return best;
}

// Do not add a max-stable-dt probe here: this operator has its own growth mode
// (~2.9/unit time), so any such bisection measures the operator, not the scheme.
static const int NADV_LVL    = 4;
static const int NADV_STEPS0 = 8;
static const int REF_REFINE  = 16;
static const double BLOWUP   = 1e4;  // only catches genuine divergence/NaN

static std::vector<double> reference_state(ot::Mesh* mesh, double T,
                                           int nsteps) {
    const Scheme ref{ts::ETSType::RK6, "ref", 0, 6, 0.0790};
    MeshTICtx ctx(mesh, true);
    if (!evolve_steps(ctx, ref, T / nsteps, nsteps, BLOWUP)) return {};
    return ctx.interior_state();
}

static double adv_err(ot::Mesh* mesh, const Scheme& sc, double T, int nsteps,
                      const std::vector<double>& ref) {
    MeshTICtx ctx(mesh, true);
    if (!evolve_steps(ctx, sc, T / nsteps, nsteps, BLOWUP)) return -1.0;
    return ctx.max_error_vs(ref);
}

// Coarsest dt whose RK4 error lands in [1e-6, 1e-2], so four halvings resolve.
static double calibrate_dt(ot::Mesh* mesh, int& nref_out) {
    const Scheme rk4{ts::ETSType::RK4, "RK4", 0, 4, 2.8284};
    for (double dt = 0.05; dt > 1e-6; dt *= 0.5) {
        const double T   = NADV_STEPS0 * dt;
        const int nref   = NADV_STEPS0 * (1 << (NADV_LVL - 1)) * REF_REFINE;
        std::vector<double> ref = reference_state(mesh, T, nref);
        if (ref.empty()) continue;
        const double e = adv_err(mesh, rk4, T, NADV_STEPS0, ref);
        if (e > 1e-6 && e < 1e-2) {
            nref_out = nref;
            return dt;
        }
    }
    nref_out = 0;
    return 0.0;
}

static double run_adv_order(ot::Mesh* mesh, const Scheme& sc, double dt0,
                            int nref, std::vector<double>& errs_out) {
    const double T          = NADV_STEPS0 * dt0;
    std::vector<double> ref = reference_state(mesh, T, nref);
    errs_out.clear();
    if (ref.empty()) return -1.0;
    for (int l = 0; l < NADV_LVL; l++)
        errs_out.push_back(
            adv_err(mesh, sc, T, NADV_STEPS0 * (1 << l), ref));
    double best = 0.0;
    for (std::size_t i = 0; i + 1 < errs_out.size(); i++) {
        if (errs_out[i] <= 0.0 || errs_out[i + 1] <= 0.0) continue;
        if (errs_out[i] > 1e-1 || errs_out[i] < 1e-12) continue;
        if (errs_out[i + 1] < 1e-14) continue;
        best = std::max(best, std::log2(errs_out[i] / errs_out[i + 1]));
    }
    return best;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank;
    MPI_Comm_rank(comm, &rank);

    m_uiMaxDepth              = (argc > 1) ? atoi(argv[1]) : 4;
    const unsigned int eOrder = (argc > 2) ? atoi(argv[2]) : 6;
    const std::string mode    = (argc > 3) ? argv[3] : "decay";
    _InitializeHcurve(m_uiDim);

    std::function<double(double, double, double)> fr =
        [](double x, double y, double z) {
            double L = (double)(1u << m_uiMaxDepth);
            return ic_decay(x / L, y / L, z / L);
        };
    std::vector<ot::TreeNode> tmp;
    function2Octree(fr, tmp, m_uiMaxDepth, 1e-3, eOrder, comm);
    ot::Mesh* mesh = ot::createMesh(tmp.data(), tmp.size(), eOrder, comm, 1,
                                    ot::SM_TYPE::FDM, DENDRO_DEFAULT_GRAIN_SZ,
                                    0.3, DENDRO_DEFAULT_SF_K);
    mesh->setDomainBounds(Point(0.0, 0.0, 0.0), Point(1.0, 1.0, 1.0));

    int failures = 0;

    if (mode == "decay" || mode == "both") {
        if (!rank)
            std::printf(
                "\n=== DECAY  du/dt=-%.1f u  — temporal order vs analytic ===\n"
                "%-16s %6s %8s   %s\n",
                LAMBDA, "scheme", "theory", "order", "errors (N=10,20,40,80)");
        for (const auto& sc : SCHEMES) {
            std::vector<double> errs;
            const double obs    = run_decay(mesh, sc, 1.0, errs);
            const bool order_ok = obs >= (sc.order - 0.5);
            if (!order_ok) failures++;
            if (!rank) {
                std::printf("%-16s %6d %8.2f  ", sc.name, sc.order, obs);
                for (double e : errs) std::printf(" %.2e", e);
                std::printf("   %s\n", order_ok ? "PASS" : "FAIL");
            }
        }
    }

    if (mode == "adv" || mode == "both") {
        if (eOrder < 6) {
            if (!rank)
                std::printf(
                    "\nADVECTION skipped: BL6 needs eleOrder >= 6 (got %u); "
                    "try `%s %u 6 adv`.\n",
                    eOrder, argv[0], m_uiMaxDepth);
            delete mesh;
            MPI_Finalize();
            return 1;
        }

        int nref        = 0;
        const double dt0 = calibrate_dt(mesh, nref);
        if (dt0 <= 0.0) {
            if (!rank)
                std::printf(
                    "\nADVECTION: could not calibrate a dt whose RK4 error "
                    "lands in [1e-6, 1e-2]\n");
            failures++;
        } else {
            if (!rank)
                std::printf(
                    "\n=== ADVECTION  du/dt=-%.1f du/dx  (BL6/JTT6) ===\n"
                    "temporal order vs an RK6 reference at dt/%d on the same "
                    "mesh,\nwhich cancels the ~1e-3 BL6 spatial error. dt0=%.3e, "
                    "shared by\nevery scheme, so the error columns compare "
                    "directly.\n"
                    "%-16s %6s %8s %10s   %s\n",
                    AVEL, (1 << (NADV_LVL - 1)) * REF_REFINE, dt0, "scheme",
                    "theory", "order", "imag-reach",
                    "errors (dt0 -> dt0/8)");
            for (const auto& sc : SCHEMES) {
                std::vector<double> errs;
                const double obs    = run_adv_order(mesh, sc, dt0, nref, errs);
                const bool order_ok = obs >= (sc.order - 0.5);
                if (!order_ok) failures++;
                if (!rank) {
                    std::printf("%-16s %6d %8.2f %10.4f  ", sc.name, sc.order,
                                obs, sc.imag_reach);
                    for (double e : errs) std::printf(" %.2e", e);
                    std::printf("   %s\n", order_ok ? "PASS" : "FAIL");
                }
            }
            if (!rank)
                std::printf(
                    "  imag-reach is from testTimeIntegratorOrder's linear\n"
                    "  analysis: the order above is real, but a scheme with\n"
                    "  reach 0.03-0.08 can only cash it in at a dt ~35x smaller\n"
                    "  than RK4's on the same operator.\n");
        }
    }

    if (!rank)
        std::printf("\n%s (%d check%s failed)\n\n",
                    failures == 0 ? "ALL CHECKS PASS" : "FAILURES DETECTED",
                    failures, failures == 1 ? "" : "s");

    delete mesh;
    MPI_Finalize();
    return failures == 0 ? 0 : 1;
}
