/**
 * @file testEnutsOrder.cpp
 * @brief Temporal convergence order of the LTS stepper (ts::ExplicitNUTS)
 * across a level interface.
 *
 * du/dt = -a du/dx + A cos(w t) on a fixed two-level octree (x < 1/2 one
 * level finer), u(0) = sin(2 pi x): the wave crosses the interface and the
 * forcing is time-dependent, so both the interface stage conversion and the
 * stage time enter the error. The error is the max norm over every owned node
 * (interface included) against the global stepper (RK4) at dt/64 on the same
 * mesh and operator, so only the time error remains. The order is the mean of
 * log2(e(dt) / e(dt/2)) over four step sizes.
 *
 * Arms: iface 3, iface 15, ratio cap 1 (every block at the finest step: RK4).
 * Each arm passes when its order reaches the floor below; the floors come
 * from the measured baseline (lts_tidy_gates.md, T4).
 * Builds with DENDRO_ENUTS_TEST_SABOTAGE=1 (frozen stage time) or =2 (the
 * original first-order interface conversion) must fail.
 *
 * Usage: testEnutsOrder [arm: all | iface3 | iface15 | cap1]
 */

#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "TreeNode.h"
#include "ctx.h"
#include "dendro.h"
#include "derivatives.h"
#include "dvec.h"
#include "enuts.h"
#include "ets.h"
#include "lts.h"
#include "mesh.h"
#include "meshUtils.h"
#include "octUtils.h"

using DVec = ot::DVector<DendroScalar, unsigned int>;

namespace {

const double AVEL  = 1.0;             // advection speed
const double FAMP  = 1.0;             // forcing amplitude
const double FOMEG = 2.0 * M_PI * 4;  // forcing frequency
const unsigned int ASYNC_K = 1;

double forcing(double t) { return FAMP * std::cos(FOMEG * t); }

class AdvCtx : public ts::Ctx<AdvCtx, DendroScalar, unsigned int> {
    DVec m_evar, m_unz, m_dudx;
    std::unique_ptr<dendroderivs::DendroDerivatives> m_deriv;
    Point m_pmin{0.0, 0.0, 0.0}, m_pmax{1.0, 1.0, 1.0};
    std::vector<DendroScalar> m_blkScr;

   public:
    explicit AdvCtx(ot::Mesh* pMesh) {
        m_uiMesh = pMesh;
        m_evar.create_vector(m_uiMesh, ot::DVEC_TYPE::OCT_SHARED_NODES, ot::DVEC_LOC::HOST, 1,
                             true);
        m_unz.create_vector(m_uiMesh, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::HOST,
                            1, true);
        m_dudx.create_vector(m_uiMesh, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING,
                             ot::DVEC_LOC::HOST, 1, true);
        m_uiTinfo = {0.0, 0.0, 0, 0.0, 0.0};
        ot::alloc_mpi_ctx<DendroScalar>(m_uiMesh, m_mpi_ctx, 1, ASYNC_K);
        m_deriv = std::make_unique<dendroderivs::DendroDerivatives>(
            "BL6", "JTT6", m_uiMesh->getElementOrder(), std::vector<double>(),
            std::vector<double>(), 1, 1);
        unsigned int mx = 0;
        for (const auto& b : m_uiMesh->getLocalBlockList())
            mx = std::max(mx, b.getAllocationSzX() * b.getAllocationSzY() * b.getAllocationSzZ());
        m_deriv->set_maximum_block_size(mx);
    }
    ~AdvCtx() {
        m_evar.destroy_vector();
        m_unz.destroy_vector();
        m_dudx.destroy_vector();
        ot::dealloc_mpi_ctx<DendroScalar>(m_uiMesh, m_mpi_ctx, 1, ASYNC_K);
    }

    int initialize() {
        const ot::TreeNode* pNodes = m_uiMesh->getAllElements().data();
        const unsigned int eo      = m_uiMesh->getElementOrder();
        const unsigned int nPe     = m_uiMesh->getNumNodesPerElement();
        const unsigned int* e2n_cg = m_uiMesh->getE2NMapping().data();
        const unsigned int* e2n_dg = m_uiMesh->getE2NMapping_DG().data();
        DendroScalar* u;
        m_evar.to_2d(&u);
        const double maxd = (double)(1u << m_uiMaxDepth);
        for (unsigned int e = m_uiMesh->getElementLocalBegin(); e < m_uiMesh->getElementLocalEnd();
             e++)
            for (unsigned int n = 0; n < nPe; n++) {
                const unsigned int cg = e2n_cg[e * nPe + n];
                if (cg < m_uiMesh->getNodeLocalBegin() || cg >= m_uiMesh->getNodeLocalEnd())
                    continue;
                unsigned int own, ix, jy, kz;
                m_uiMesh->dg2eijk(e2n_dg[e * nPe + n], own, ix, jy, kz);
                const double len = (double)(1u << (m_uiMaxDepth - pNodes[own].getLevel()));
                const double x   = (pNodes[own].getX() + ix * len / eo) / maxd;
                u[cg]            = std::sin(2.0 * M_PI * x);
            }
        return 0;
    }

    // global stepper: unzip, -a d/dx per block, zip, plus the forcing
    int rhs(DVec* in, DVec* out, unsigned int, DendroScalar time) {
        this->unzip(*in, m_unz);
        DendroScalar *uU, *dU;
        m_unz.to_2d(&uU);
        m_dudx.to_2d(&dU);
        const double f = forcing(time);
        for (const auto& b : m_uiMesh->getLocalBlockList()) {
            const DendroIntL off = b.getOffset();
            unsigned int s[3]    = {b.getAllocationSzX(), b.getAllocationSzY(),
                                    b.getAllocationSzZ()};
            m_deriv->grad_x(&dU[off], &uU[off], b.computeDx(m_pmin, m_pmax), s,
                            b.getBlkNodeFlag());
            for (unsigned int q = 0; q < s[0] * s[1] * s[2]; q++)
                dU[off + q] = -AVEL * dU[off + q] + f;
        }
        this->zip(m_dudx, *out);
        return 0;
    }

    // LTS stepper: the same operator on one padded block at its stage time
    int rhs_blk(const DendroScalar* in, DendroScalar* out, unsigned int, unsigned int blk,
                DendroScalar time) {
        const ot::Block& b = m_uiMesh->getLocalBlockList()[blk];
        const ts::BlockView v(b, Point(0.0, 0.0, 0.0),
                              Point((double)(1u << m_uiMaxDepth), (double)(1u << m_uiMaxDepth),
                                    (double)(1u << m_uiMaxDepth)),
                              m_pmin, m_pmax, m_uiMesh->getElementOrder() / 2);
        unsigned int s[3] = {v.sz[0], v.sz[1], v.sz[2]};
        m_deriv->grad_x(out, in, b.computeDx(m_pmin, m_pmax), s, v.bflag);
        const double f = forcing(time);
        for (unsigned int q = 0; q < v.nn; q++) out[q] = -AVEL * out[q] + f;
        return 0;
    }

    unsigned int compute_lts_ts_offset() {
        unsigned int lmin, lmax;
        m_uiMesh->computeMinMaxLevel(lmin, lmax);
        return ts::level_offset(
            *m_uiMesh, lmax, m_uiTinfo._m_uiTh,
            [](unsigned int) { return std::numeric_limits<double>::infinity(); },
            m_uiMesh->getMPIGlobalCommunicator());
    }

    std::vector<double> state() {
        DendroScalar* u;
        m_evar.to_2d(&u);
        return std::vector<double>(u + m_uiMesh->getNodeLocalBegin(),
                                   u + m_uiMesh->getNodeLocalEnd());
    }

    DVec& get_evolution_vars() { return m_evar; }
    DVec& get_constraint_vars() { return m_evar; }
    DVec& get_primitive_vars() { return m_evar; }
    int pre_stage(DVec) { return 0; }
    int post_stage(DVec) { return 0; }
    int pre_timestep(DVec) { return 0; }
    int post_timestep(DVec) { return 0; }
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
};

// level L everywhere, one level finer for x < 1/2
ot::Mesh* twoLevelMesh(unsigned int L, unsigned int eOrder, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    std::vector<ot::TreeNode> oct;
    if (!rank) {
        const unsigned int n = 1u << L, len = 1u << (m_uiMaxDepth - L);
        for (unsigned int k = 0; k < n; k++)
            for (unsigned int j = 0; j < n; j++)
                for (unsigned int i = 0; i < n; i++) {
                    if (2 * i < n) {
                        const unsigned int h = len / 2;
                        for (unsigned int c = 0; c < 8; c++)
                            oct.emplace_back(i * len + (c & 1) * h, j * len + ((c >> 1) & 1) * h,
                                             k * len + ((c >> 2) & 1) * h, L + 1, m_uiDim,
                                             m_uiMaxDepth);
                    } else {
                        oct.emplace_back(i * len, j * len, k * len, L, m_uiDim, m_uiMaxDepth);
                    }
                }
    }
    ot::Mesh* m = ot::createMesh(oct.data(), oct.size(), eOrder, comm, 0, ot::SM_TYPE::FDM,
                                 DENDRO_DEFAULT_GRAIN_SZ, 0.1, DENDRO_DEFAULT_SF_K);
    m->setDomainBounds(Point(0.0, 0.0, 0.0), Point(1.0, 1.0, 1.0));
    return m;
}

std::vector<double> runGlobal(ot::Mesh* mesh, double dt, unsigned int nsteps) {
    AdvCtx ctx(mesh);
    ctx.set_ts_info({0.0, dt * nsteps, 0, 0.0, dt});
    ts::ETS<DendroScalar, AdvCtx> ets(&ctx);
    ets.set_ets_coefficients(ts::ETSType::RK4);
    ets.init();
    for (unsigned int s = 0; s < nsteps; s++) ets.evolve();
    return ctx.state();
}

// fine steps of dt up to t_end (a whole number of coarse steps)
std::vector<double> runLts(ot::Mesh* mesh, double dt, double t_end, const ts::LtsOptions& o) {
    auto ctx = std::make_unique<AdvCtx>(mesh);
    ctx->set_ts_info({0.0, t_end, 0, 0.0, dt});
    auto lts = std::make_unique<ts::ExplicitNUTS<DendroScalar, AdvCtx>>(ctx.get(),
                                                                         /*useWpart=*/false, o);
    lts->set_evolve_vars(ctx->get_evolution_vars());
    lts->set_ets_coefficients(ts::ETSType::RK4);
    for (lts->init(); lts->curr_time() < t_end - 0.5 * dt; lts->evolve()) {
    }
    return ctx->state();
}

// a non-finite value or a size mismatch counts as an infinite error (and is reported)
double maxDiff(const std::vector<double>& a, const std::vector<double>& b, MPI_Comm comm) {
    double e = (a.size() == b.size()) ? 0.0 : std::numeric_limits<double>::infinity();
    size_t nonfinite = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); i++) {
        const double d = std::fabs(a[i] - b[i]);
        if (!std::isfinite(d)) nonfinite++;
        e = std::isfinite(d) ? std::max(e, d) : std::numeric_limits<double>::infinity();
    }
    if (a.size() != b.size() || nonfinite)
        std::printf("    [rank-local] state size %zu vs reference %zu, non-finite %zu\n", a.size(),
                    b.size(), nonfinite);
    double g = 0.0;
    MPI_Allreduce(&e, &g, 1, MPI_DOUBLE, MPI_MAX, comm);
    return g;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    _InitializeHcurve(m_uiDim);
    m_uiMaxDepth          = 8;
    const std::string arm = (argc > 1) ? argv[1] : "all";

    const unsigned int L = 2, eOrder = 6;
    ot::Mesh* mesh       = twoLevelMesh(L, eOrder, MPI_COMM_WORLD);
    int fails            = 0;
    if (mesh->isActive()) {
        MPI_Comm comm = mesh->getMPICommunicator();
        // finest node spacing 1 / (2^(L+1) eOrder); CFL 0.25 of it at the coarsest dt
        const double h   = 1.0 / ((1u << (L + 1)) * eOrder);
        const double dt0 = 0.25 * h;
        const double T   = 16 * dt0;  // a whole number of coarse steps at every dt
        const std::vector<double> ref = runGlobal(mesh, dt0 / 64, 16 * 64);

        struct Arm {
            const char* name;
            unsigned int iface, cap;
            double floor;
        };
        const Arm arms[] = {{"iface3", 3, 0, 2.6}, {"iface15", 15, 0, 3.6}, {"cap1", 3, 1, 3.7}};
        for (const Arm& a : arms) {
            if (arm != "all" && arm != a.name) continue;
            ts::LtsOptions o;
            o.iface     = a.iface;
            o.ratio_cap = a.cap;
            std::vector<double> err;
            for (unsigned int l = 0; l < 4; l++)
                err.push_back(maxDiff(runLts(mesh, dt0 / (1u << l), T, o), ref, comm));
            double sum = 0.0;
            for (unsigned int l = 0; l + 1 < err.size(); l++) sum += std::log2(err[l] / err[l + 1]);
            const double order = sum / (err.size() - 1);
            const bool ok      = order >= a.floor && std::isfinite(order);
            if (!ok) fails++;
            if (!rank)
                std::printf("  %-8s err %.3e %.3e %.3e %.3e  order %.2f (floor %.1f) %s\n", a.name,
                            err[0], err[1], err[2], err[3], order, a.floor, ok ? "ok" : "FAIL");
        }
    }
    delete mesh;
    int all = 0;
    MPI_Allreduce(&fails, &all, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (!rank) std::printf("testEnutsOrder: %s\n", all ? "FAIL" : "PASS");
    MPI_Finalize();
    return all ? 1 : 0;
}
