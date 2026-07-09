// Mesh-based time-integrator validation: builds a real Dendro octree and
// evolves through the actual ts::ETS / ts::ETS_MSRK path vs an analytical
// solution. Modes:
//   decay : du/dt = -lambda*u (decoupled -> clean temporal order)
//   adv   : du/dt = -a*du/dx (BL6 compact-FD; interior error, spatial-limited)
// Run:  ./testMeshTimeIntegrator maxDepth eleOrder [decay|adv|both]

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
            const unsigned int off = b.getOffset();
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
static void run(ot::Mesh* mesh, const char* name, ts::ETSType type, bool msrk,
                bool adv, double T, MPI_Comm comm) {
    const std::vector<int> Ns = {10, 20, 40, 80};
    std::vector<double> errs;
    for (int N : Ns) {
        MeshTICtx ctx(mesh, adv);
        ctx.set_ts_info({0.0, T, 0, 0.0, T / N});
        if (msrk) {
            ts::ETS_MSRK<DendroScalar, MeshTICtx> ets(&ctx, type);
            ets.set_ets_coefficients(ts::ETSType::RK4);
            ets.init();
            for (int s = 0; s < N; s++) ets.evolve();
        } else {
            ts::ETS<DendroScalar, MeshTICtx> ets(&ctx);
            ets.set_ets_coefficients(type);
            ets.init();
            for (int s = 0; s < N; s++) ets.evolve();
        }
        errs.push_back(ctx.max_error());
    }
    double best = 0.0;
    for (std::size_t i = 0; i + 1 < errs.size(); i++)
        if (errs[i] > 1e-13)
            best = std::max(best, std::log2(errs[i] / errs[i + 1]));
    int rank;
    MPI_Comm_rank(comm, &rank);
    if (!rank) {
        std::printf("  %-14s errs =", name);
        for (double e : errs) std::printf(" %.3e", e);
        std::printf("   order ~ %.2f\n", best);
    }
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

    if ((mode == "decay" || mode == "both")) {
        if (!rank) std::printf("=== DECAY  du/dt=-%.1f u  (clean temporal order) ===\n", LAMBDA);
        run(mesh, "RK4", ts::ETSType::RK4, false, false, 1.0, comm);
        run(mesh, "RK6", ts::ETSType::RK6, false, false, 1.0, comm);
        run(mesh, "RK4_MSRK2_1", ts::ETSType::RK4_MSRK2_1, true, false, 1.0, comm);
    }
    if ((mode == "adv" || mode == "both")) {
        if (!rank) std::printf("=== ADVECTION  du/dt=-%.1f du/dx  (BL6, T=0.1, interior) ===\n", AVEL);
        run(mesh, "RK4", ts::ETSType::RK4, false, true, 0.1, comm);
        run(mesh, "RK6", ts::ETSType::RK6, false, true, 0.1, comm);
        run(mesh, "RK4_MSRK2_1", ts::ETSType::RK4_MSRK2_1, true, true, 0.1, comm);
    }

    delete mesh;
    MPI_Finalize();
    return 0;
}
