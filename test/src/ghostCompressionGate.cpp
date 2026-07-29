/**
 * @file ghostCompressionGate.cpp
 * @brief Does the compressed ghost exchange produce the right answer at all?
 *
 * Until this test existed the compressed path had NEVER EXECUTED. Three separate
 * things prevented it: Ctx::unzip gates on dendro_compress::COMPRESSION_OPTION
 * which nothing set, setUpCompressor()/set_compression_options() had zero
 * callers (the only one died with NLSigma), and nothing anywhere passed
 * use_compression=true. Every compression number the project had was the codec
 * measured in isolation; the two-phase count exchange, the MPI_Testsome drain
 * and the per-peer extract/compress/decompress/unextract had never run.
 *
 * The gate: with the DUMMY codec -- which is a memcpy in both directions -- the
 * compressed exchange must reproduce the uncompressed one BIT FOR BIT. Any
 * difference is a plumbing bug, not a codec tradeoff, and needs no physics or
 * error budget to interpret.
 *
 * It is also run with a LOSSY codec (quant16), which MUST differ. A gate that
 * can only pass proves nothing -- if the lossy arm reports zero difference then
 * the comparison itself is broken (e.g. compression silently not engaging) and
 * the dummy PASS is meaningless. That failure mode is exactly what this project
 * has been bitten by before, so it is checked rather than assumed.
 *
 * Usage: mpirun -np <N> ghostCompressionGate [maxDepth=6] [eleOrder=6] [dof=4] [wtol=1e-5]
 * Exit code 0 = all gates pass, 1 = failure.
 */

#include <mpi.h>

#include <algorithm>
#include <any>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#include "TreeNode.h"
#include "compression.h"
#include "ctx.h"
#include "dvec.h"
#include "mesh.h"
#include "meshUtils.h"
#include "octUtils.h"

using DVec = ot::DVector<DendroScalar, unsigned int>;

static const unsigned int ASYNC_K = 1;

/**
 * @brief Smallest Ctx that can drive Ctx::unzip -- no RHS, no timestepper.
 *
 * unzip() is the only thing exercised, so nothing else needs to be real.
 */
class GateCtx : public ts::Ctx<GateCtx, DendroScalar, unsigned int> {
    unsigned int m_dof;

   public:
    DVec m_evar;  // zipped, ghosted
    DVec m_unz;   // unzipped output

    GateCtx(ot::Mesh* pMesh, unsigned int dof)
        : ts::Ctx<GateCtx, DendroScalar, unsigned int>(), m_dof(dof) {
        m_uiMesh = pMesh;
        m_evar.create_vector(m_uiMesh, ot::DVEC_TYPE::OCT_SHARED_NODES,
                             ot::DVEC_LOC::HOST, m_dof, true);
        m_unz.create_vector(m_uiMesh, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING,
                            ot::DVEC_LOC::HOST, m_dof, true);
        m_uiTinfo = {0.0, 0.0, 0, 0.0, 0.0};
        ot::alloc_mpi_ctx<DendroScalar>(m_uiMesh, m_mpi_ctx, m_dof, ASYNC_K);
        // REQUIRED. Ctx::unzip unconditionally accumulates into
        // m_uiTotalBytesSend[j] / ...Recv[j] for j < getMPICommSize(), and those
        // vectors start empty -- so any Ctx that skips this segfaults on the
        // first unzip, before compression is even reached. Cost one debugging
        // round here.
        this->prepareBytesVectors();
    }
    ~GateCtx() {
        m_evar.destroy_vector();
        m_unz.destroy_vector();
        ot::dealloc_mpi_ctx<DendroScalar>(m_uiMesh, m_mpi_ctx, m_dof, ASYNC_K);
    }

    /**
     * @brief Deterministic, per-variable, spatially varying initial data.
     *
     * Deliberately NOT smooth-and-tiny: the halo must contain values a codec can
     * get wrong, and the per-variable amplitude spread (1 .. 10^(v)) is what
     * exercises the per-group absmax scaling.
     */
    void fill() {
        DendroScalar* p = m_evar.get_vec_ptr();
        const unsigned int nAll = m_uiMesh->getDegOfFreedom();
        const ot::TreeNode* pNodes = m_uiMesh->getAllElements().data();
        const unsigned int eo = m_uiMesh->getElementOrder();
        const unsigned int* e2n = m_uiMesh->getE2NMapping().data();
        std::vector<char> seen(nAll, 0);
        for (unsigned int e = m_uiMesh->getElementLocalBegin();
             e < m_uiMesh->getElementLocalEnd(); ++e) {
            for (unsigned int k = 0; k <= eo; ++k)
                for (unsigned int j = 0; j <= eo; ++j)
                    for (unsigned int i = 0; i <= eo; ++i) {
                        const unsigned int nid =
                            e2n[e * m_uiMesh->getNumNodesPerElement() +
                                (k * (eo + 1) + j) * (eo + 1) + i];
                        if (nid >= nAll || seen[nid]) continue;
                        seen[nid] = 1;
                        const unsigned int sz =
                            1u << (m_uiMaxDepth - pNodes[e].getLevel());
                        const double x = pNodes[e].getX() + i * sz / (double)eo;
                        const double y = pNodes[e].getY() + j * sz / (double)eo;
                        const double z = pNodes[e].getZ() + k * sz / (double)eo;
                        const double L = (double)(1u << m_uiMaxDepth);
                        for (unsigned int v = 0; v < m_dof; ++v) {
                            const double amp = std::pow(10.0, (double)v);
                            p[v * nAll + nid] =
                                amp * (std::sin(6.28318530718 * x / L) *
                                           std::cos(4.0 * y / L) +
                                       0.25 * std::sin(9.0 * z / L));
                        }
                    }
        }
    }

    int initialize() { return 0; }
};

/** @brief run one unzip and return the unzipped buffer. */
static std::vector<DendroScalar> run_unzip(ot::Mesh* mesh, unsigned int dof,
                                          bool use_compression) {
    GateCtx ctx(mesh, dof);
    ctx.fill();
    // sync the zipped vector's ghosts the ordinary way first, so both arms start
    // from identical zipped state -- the thing under test is the unzip path.
    ctx.unzip(ctx.m_evar, ctx.m_unz, ASYNC_K, use_compression);
    const DendroScalar* u = ctx.m_unz.get_vec_ptr();
    return std::vector<DendroScalar>(u, u + ctx.m_unz.get_size());
}

struct Diff {
    unsigned long ndiff = 0;
    double maxabs       = 0.0;
};

static Diff compare(const std::vector<DendroScalar>& a,
                    const std::vector<DendroScalar>& b) {
    Diff d;
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) {
            ++d.ndiff;
            d.maxabs = std::max(d.maxabs, std::fabs((double)a[i] - (double)b[i]));
        }
    }
    return d;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, npes;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);

    m_uiMaxDepth              = (argc > 1) ? atoi(argv[1]) : 6;
    const unsigned int eOrder = (argc > 2) ? atoi(argv[2]) : 6;
    const unsigned int dof    = (argc > 3) ? atoi(argv[3]) : 4;
    const double wtol         = (argc > 4) ? atof(argv[4]) : 1e-5;

    _InitializeHcurve(m_uiDim);

    std::function<double(double, double, double)> fr = [](double x, double y,
                                                         double z) {
        const double L = (double)(1u << m_uiMaxDepth);
        const double dx = x / L - 0.55, dy = y / L - 0.5, dz = z / L - 0.45;
        return std::exp(-(dx * dx + dy * dy + dz * dz) / (2.0 * 0.08 * 0.08));
    };
    std::vector<ot::TreeNode> tmp;
    function2Octree(fr, tmp, m_uiMaxDepth, wtol, eOrder, comm);
    ot::Mesh* mesh = ot::createMesh(tmp.data(), tmp.size(), eOrder, comm, 1,
                                    ot::SM_TYPE::FDM, DENDRO_DEFAULT_GRAIN_SZ,
                                    0.3, DENDRO_DEFAULT_SF_K);
    mesh->setDomainBounds(Point(0.0, 0.0, 0.0), Point(1.0, 1.0, 1.0));

    if (!rank)
        std::printf(
            "\n=== ghost compression gate ===\n"
            "npes=%d maxDepth=%u eleOrder=%u dof=%u wtol=%.1e  activeNpes=%d\n",
            npes, m_uiMaxDepth, eOrder, dof, wtol,
            mesh->isActive() ? mesh->getMPICommSize() : 0);

    int failures = 0;

    // ---- reference: compression OFF -------------------------------------
    dendro_compress::COMPRESSION_OPTION = dendro_compress::CompressionType::NONE;
    const std::vector<DendroScalar> ref = run_unzip(mesh, dof, false);

    // ---- ARM 1: DUMMY codec must be BIT IDENTICAL -----------------------
    {
        dendro_compress::setUpCompressor(
            dendrocompression::CompressionType::COMP_DUMMY,
            {eOrder, dof});
        const std::vector<DendroScalar> got = run_unzip(mesh, dof, true);
        Diff d = compare(ref, got);
        unsigned long gd = 0;
        double gm = 0.0;
        MPI_Reduce(&d.ndiff, &gd, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, comm);
        MPI_Reduce(&d.maxabs, &gm, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (!rank) {
            std::printf(
                "\n  [GATE 1] DUMMY codec vs compression OFF\n"
                "           differing values : %lu\n"
                "           max |diff|       : %.3e\n"
                "           %s\n",
                gd, gm,
                gd == 0 ? "PASS - compressed plumbing is bit-exact"
                        : "*** FAIL - plumbing bug (dummy is a memcpy) ***");
            if (gd != 0) failures++;
        }
    }

    // ---- ARM 2: a LOSSY codec MUST differ (proves the gate can fail) ----
    {
        dendro_compress::setUpCompressor(
            dendrocompression::CompressionType::COMP_QUANT,
            {eOrder, dof, 16u});
        const std::vector<DendroScalar> got = run_unzip(mesh, dof, true);
        Diff d = compare(ref, got);
        unsigned long gd = 0;
        double gm = 0.0;
        MPI_Reduce(&d.ndiff, &gd, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, comm);
        MPI_Reduce(&d.maxabs, &gm, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (!rank) {
            std::printf(
                "\n  [GATE 2] quant16 (lossy) vs compression OFF"
                "  -- MUST differ\n"
                "           differing values : %lu\n"
                "           max |diff|       : %.3e\n"
                "           %s\n",
                gd, gm,
                gd != 0
                    ? "PASS - the comparison can detect a difference"
                    : "*** FAIL - lossy codec changed nothing: compression is "
                      "NOT engaging, so GATE 1 is vacuous ***");
            if (gd == 0) failures++;
        }
    }

    dendro_compress::COMPRESSION_OPTION = dendro_compress::CompressionType::NONE;

    int rc = 0;
    if (!rank) {
        std::printf("\n  %s (%d failure(s))\n\n",
                    failures ? "GATE FAILED" : "ALL GATES PASSED", failures);
        rc = failures ? 1 : 0;
    }
    MPI_Bcast(&rc, 1, MPI_INT, 0, comm);

    delete mesh;
    MPI_Finalize();
    return rc;
}
