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
     * get wrong, and a per-variable amplitude spread is what exercises the
     * per-group absmax scaling.
     *
     * The spread is capped at 4 decades (10^(v mod 4)) rather than 10^v. An
     * unbounded 10^v reached 1e23 at dof=24, which made GATE 2's max|diff| read
     * 2.1e19 -- correct for the data but uninterpretable, and nothing like the
     * ~3-4 decades a BSSN state actually spans.
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
                            const double amp = std::pow(10.0, (double)(v % 4));
                            p[v * nAll + nid] =
                                amp * (std::sin(6.28318530718 * x / L) *
                                           std::cos(4.0 * y / L) +
                                       0.25 * std::sin(9.0 * z / L));
                        }
                    }
        }
    }

    int initialize() { return 0; }

    /**
     * @brief The UNCOMPRESSED ghost exchange on its own, no unzip.
     *
     * Needed to say what fraction of an unzip the exchange even is -- a % of the
     * whole unzip is meaningless otherwise, since compression cannot touch the
     * interpolation. Deliberately does NOT use the t_compression_* timers to
     * infer this: those nest (the Mesh-level start/stop sit inside the
     * Ctx-level ones) and t_compression_compress additionally brackets the
     * omp_par::scan calls, so their sum overcounts -- it read 274% of the unzip
     * when tried, which is how the double-count was noticed.
     */
    void raw_exchange() {
        m_uiMesh->readFromGhostBegin(m_mpi_ctx[0], m_evar.get_vec_ptr(), m_dof);
        m_uiMesh->readFromGhostEnd(m_mpi_ctx[0], m_evar.get_vec_ptr(), m_dof);
    }

    /**
     * @brief The COMPRESSED exchange on its own, no unzip.
     *
     * The counterpart to raw_exchange(), and the only honest way to measure a
     * change to the exchange PROTOCOL. The exchange is under 4% of an unzip, so
     * an effect worth ~0.1 ms sits inside the build-to-build code-layout noise
     * of the full-unzip number -- two builds differing only in a preprocessor
     * flag were seen 0.23 ms apart on the compression-OFF arm, which is
     * identical code in both. Compare exchanges to exchanges.
     */
    void compressed_exchange() {
        // Ctx::unzip's guard, which calling the exchange directly skips. Ranks
        // outside a small octree's active comm have no scatter map; without
        // this the harness segfaults at 8 ranks but not at 4, which reads like
        // a bug in the exchange and is not one.
        if (!m_uiMesh->isActive()) return;
        this->exchange_host_gathered_dof_compression(m_evar, m_unz, ASYNC_K);
    }
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

/**
 * @brief Wall time per unzip, max over ranks, in ms.
 *
 * Interleaving matters more than rep count here (shared machine), so the caller
 * alternates arms rather than running each to completion.
 */
static double time_unzip(ot::Mesh* mesh, unsigned int dof, bool use_compression,
                         unsigned int reps, MPI_Comm comm) {
    GateCtx ctx(mesh, dof);
    ctx.fill();
    for (unsigned int w = 0; w < 3; ++w)  // warm caches / first-touch
        ctx.unzip(ctx.m_evar, ctx.m_unz, ASYNC_K, use_compression);
    MPI_Barrier(comm);
    const double t0 = MPI_Wtime();
    for (unsigned int r = 0; r < reps; ++r)
        ctx.unzip(ctx.m_evar, ctx.m_unz, ASYNC_K, use_compression);
    const double local = (MPI_Wtime() - t0) / reps * 1e3;
    double mx = 0.0;
    MPI_Allreduce(&local, &mx, 1, MPI_DOUBLE, MPI_MAX, comm);
    return mx;
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

    // ---- ARM 2: a LOSSLESS codec must ALSO be bit identical --------------
    // Strictly stronger than ARM 1: dummy is a memcpy and never exercises a real
    // codec's framing, size accounting or fallback path. blosc does all three
    // (and its raw fallback fires often on the small dimensionalities), while
    // still being obliged to reproduce the input exactly.
    {
        dendro_compress::setUpCompressor(
            dendrocompression::CompressionType::COMP_BLOSC, {eOrder, dof});
        const std::vector<DendroScalar> got = run_unzip(mesh, dof, true);
        Diff d = compare(ref, got);
        unsigned long gd = 0;
        double gm = 0.0;
        MPI_Reduce(&d.ndiff, &gd, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, comm);
        MPI_Reduce(&d.maxabs, &gm, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (!rank) {
            std::printf(
                "\n  [GATE 2] BLOSC codec (lossless) vs compression OFF\n"
                "           differing values : %lu\n"
                "           max |diff|       : %.3e\n"
                "           %s\n",
                gd, gm,
                gd == 0
                    ? "PASS - a real codec round-trips bit-exactly through the "
                      "exchange"
                    : "*** FAIL - lossless codec lost data: framing, sizing or "
                      "the raw fallback is wrong ***");
            if (gd != 0) failures++;
        }
    }

    // ---- ARM 3: a LOSSY codec MUST differ (proves the gate can fail) ----
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
                "\n  [GATE 3] quant16 (lossy) vs compression OFF"
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

    // ---- COST FLOOR ------------------------------------------------------
    // On one node MPI is shared-memory memcpy, so there is essentially NO wire
    // time for compression to reclaim. Timing here therefore measures the pure
    // OVERHEAD -- codec, extra buffers, the extract/unextract passes, and the
    // count-exchange MPI_Waitall barrier -- with none of the benefit. That is
    // the floor a real fabric has to beat, and unlike an arithmetic estimate it
    // includes every cost actually present in the path.
    {
        const unsigned int reps = 30;
        // interleave the arms: never compare runs taken minutes apart
        double t_off = 0, t_dum = 0, t_qnt = 0;
        for (unsigned int pass = 0; pass < 3; ++pass) {
            dendro_compress::COMPRESSION_OPTION =
                dendro_compress::CompressionType::NONE;
            t_off += time_unzip(mesh, dof, false, reps, comm);

            dendro_compress::setUpCompressor(
                dendrocompression::CompressionType::COMP_DUMMY, {eOrder, dof});
            t_dum += time_unzip(mesh, dof, true, reps, comm);

            dendro_compress::setUpCompressor(
                dendrocompression::CompressionType::COMP_QUANT,
                {eOrder, dof, 16u});
            t_qnt += time_unzip(mesh, dof, true, reps, comm);
        }
        t_off /= 3.0; t_dum /= 3.0; t_qnt /= 3.0;

        // Bytes this rank sends per exchange: the scatter map's own send-buffer
        // size, which is exactly what the exchange moves.
        double send_bytes = 0.0;
        if (mesh->isActive()) {
            const unsigned int an = mesh->getMPICommSize();
            send_bytes = (double)(mesh->getNodalSendOffsets()[an - 1] +
                                  mesh->getNodalSendCounts()[an - 1]) *
                         dof * sizeof(DendroScalar);
        }
        {
            double mx = 0.0;
            MPI_Allreduce(&send_bytes, &mx, 1, MPI_DOUBLE, MPI_MAX, comm);
            send_bytes = mx;
        }

        // How much of an unzip is the EXCHANGE at all? Measured directly by
        // running the uncompressed exchange on its own (see raw_exchange).
        // Uncompressed vs compressed EXCHANGE, nothing else running. This is
        // the number to quote when the exchange PROTOCOL changes; the
        // full-unzip figures above are far too diluted to resolve it.
        //
        // Estimator: MINIMUM over interleaved passes, not the mean. The signal
        // is ~0.17 ms and the contamination on a shared laptop is one-sided --
        // a pass can only be made slower by a descheduled rank, never faster --
        // so the mean tracks the interference and the min tracks the machine.
        // With mean-of-one-pass the derived break-even bandwidth printed -5.20
        // and +10.83 GB/s on consecutive runs of the SAME binary. Exchanges are
        // cheap (~0.4 ms), so passes are affordable where whole unzips are not.
        double exch_ms = 0.0, cexch_dum_ms = 0.0, cexch_qnt_ms = 0.0;
        {
            const unsigned int passes    = 7;
            const unsigned int exch_reps = reps * 4;

            const dendro_compress::CompressionType arms[3] = {
                dendro_compress::CompressionType::NONE,
                dendro_compress::CompressionType::DUMMY,
                dendro_compress::CompressionType::QUANT};
            double* outs[3] = {&exch_ms, &cexch_dum_ms, &cexch_qnt_ms};
            for (int a = 0; a < 3; ++a) *outs[a] = 1e300;

            // One ctx per arm, built once: the constructor allocates the
            // compressed buffers and slot layout, which is setup, not exchange.
            for (int a = 0; a < 3; ++a) {
                dendro_compress::COMPRESSION_OPTION = arms[a];
                GateCtx cctx(mesh, dof);
                cctx.fill();
                for (unsigned int w = 0; w < 3; ++w)
                    a == 0 ? cctx.raw_exchange() : cctx.compressed_exchange();

                for (unsigned int p = 0; p < passes; ++p) {
                    MPI_Barrier(comm);
                    const double c0 = MPI_Wtime();
                    for (unsigned int r = 0; r < exch_reps; ++r)
                        a == 0 ? cctx.raw_exchange()
                               : cctx.compressed_exchange();
                    const double loc = (MPI_Wtime() - c0) / exch_reps * 1e3;
                    double mx        = 0.0;
                    MPI_Allreduce(&loc, &mx, 1, MPI_DOUBLE, MPI_MAX, comm);
                    if (mx < *outs[a]) *outs[a] = mx;
                }
            }
            dendro_compress::COMPRESSION_OPTION =
                dendro_compress::CompressionType::NONE;
        }

        if (!rank) {
            std::printf(
                "\n  [COST FLOOR] wall ms per unzip (max over ranks, %u reps x 3"
                " interleaved passes)\n"
                "    compression OFF      : %8.3f ms\n"
                "    DUMMY  (plumbing)    : %8.3f ms   %+7.1f%%\n"
                "    quant16              : %8.3f ms   %+7.1f%%\n",
                reps, t_off, t_dum, 100.0 * (t_dum - t_off) / t_off, t_qnt,
                100.0 * (t_qnt - t_off) / t_off);
            std::printf(
                "\n    uncompressed EXCHANGE alone : %8.3f ms  (%.1f%% of the OFF\n"
                "      unzip -- the rest is interpolation, which compression\n"
                "      cannot touch, so THIS is the slice in play)\n"
                "    overhead as a share of the exchange:  dummy %+.1f%%, "
                "quant16 %+.1f%%\n",
                exch_ms, 100.0 * exch_ms / t_off,
                100.0 * (t_dum - t_off) / exch_ms,
                100.0 * (t_qnt - t_off) / exch_ms);
            std::printf(
                "\n    [EXCHANGE ONLY] ms per exchange, no unzip -- this is the\n"
                "      number that moves when the exchange PROTOCOL changes\n"
                "      uncompressed        : %8.3f ms\n"
                "      compressed, DUMMY   : %8.3f ms   %+7.1f%%\n"
                "      compressed, quant16 : %8.3f ms   %+7.1f%%\n",
                exch_ms, cexch_dum_ms,
                100.0 * (cexch_dum_ms - exch_ms) / exch_ms, cexch_qnt_ms,
                100.0 * (cexch_qnt_ms - exch_ms) / exch_ms);
            // Bytes this rank puts on the wire per exchange, from the scatter map
            // directly (rather than the unsigned-int byte counters, which wrap
            // at 4 GB).
            const double MB = send_bytes / 1048576.0;
            const double saved_frac = 1.0 - 1.0 / 1.850;  // measured real-mix ratio
            // Overhead is taken from the EXCHANGE-ONLY arms, not from
            // (t_qnt - t_off). The full-unzip delta is a ~0.15 ms signal inside
            // a ~6 ms measurement, and it is not merely imprecise: it goes
            // NEGATIVE often enough to print a negative break-even bandwidth,
            // which is nonsense. Same two runs, same machine, gave -0.88 and
            // 5.64 GB/s off the unzip delta while the exchange-only delta held
            // steady. Measure the thing you changed.
            const double overhead_ms = cexch_qnt_ms - exch_ms;
            // fabric bandwidth at which saved wire time == measured overhead
            const double be_gbs =
                (send_bytes * saved_frac) / (overhead_ms * 1e-3) / 1e9;
            std::printf(
                "\n    per-rank send volume        : %8.3f MiB / exchange\n"
                "    quant16 overhead (exchange) : %8.3f ms\n"
                "    => break-even fabric bandwidth: %7.2f GB/s per rank\n",
                MB, overhead_ms, be_gbs);
            std::printf(
                "\n    COST FLOOR, read carefully. On one node MPI is a\n"
                "    shared-memory copy, so the deltas above are essentially all\n"
                "    cost. The break-even figure says: compression pays only if a\n"
                "    rank's share of fabric bandwidth is BELOW that number, since\n"
                "    slower wire means more time saved per byte removed.\n"
                "    Two caveats that matter more than the number:\n"
                "      - dummy and quant16 overheads are not cleanly separable at\n"
                "        this noise level (seen at +3.9%% and +7.0%% across runs).\n"
                "      - it assumes exchange time is proportional to BYTES. Hybrid\n"
                "        Parallelism measured the exchange is ~62%% MPI_Waitall,\n"
                "        which is mostly neighbour IMBALANCE, not wire -- and\n"
                "        imbalance does not shrink when bytes do. If only ~40%% of\n"
                "        exchange time is byte-proportional, divide the saving\n"
                "        accordingly.\n"
                "    Settling that split needs a cluster; it cannot be done here.\n");
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
