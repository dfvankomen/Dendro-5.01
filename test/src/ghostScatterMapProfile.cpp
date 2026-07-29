/**
 * @file ghostScatterMapProfile.cpp
 * @brief What does a REAL ghost payload look like, and what compression ratio
 *        is therefore achievable on it?
 *
 * Every codec number this project has (ZFP, quantize, interpolation) was
 * measured on the 3-D case: a 125-point group. But the nodal scatter map is a
 * mix of dimensionalities -- interior slabs (n^3), faces (n^2), edges (n) and
 * corners (1) -- and a per-group codec's ratio collapses as the group shrinks,
 * because the per-group header stops being amortized. With n=5 and float32,
 * 16-bit quantization gives 1.969x on a 3-D group but exactly nothing on a
 * corner.
 *
 * So the achievable ratio is a property of the MESH AND PARTITION, not of the
 * codec, and it has never been measured. This does that, and nothing else: it
 * builds a distributed octree, reads the scatter-map dimensional histogram that
 * Mesh already computes (m_uiScatterMapConfigDimCountsSend, exposed via
 * getSendNodeSMConfigDimCounts()), and reports the ratio ceiling implied by it.
 *
 * It does NOT compress anything and does not need compression enabled -- the
 * numbers here bound what any per-group codec can do before one is chosen.
 *
 * Also reported, because both decide how to parallelize the codec:
 *   - per-peer payload sizes  -> is there enough work to thread WITHIN a peer,
 *                                or should threading go ACROSS peers?
 *   - intra-node vs inter-node peer split -> how much of the halo is even worth
 *                                compressing (a message that never reaches the
 *                                fabric gains nothing from being smaller)
 *
 * Usage:  mpirun -np <N> ghostScatterMapProfile [maxDepth=6] [eleOrder=6] [wtol=1e-3]
 */

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

#include "mesh.h"
#include "meshUtils.h"
#include "octUtils.h"
#include "scattermapConfig.h"
#include "TreeNode.h"

namespace {

// Dimensionality index used by Mesh: arr[getNDim()], so 0 = corner ... 3 = interior.
constexpr const char* DIM_NAME[4] = {"0d corner", "1d edge  ", "2d face  ",
                                     "3d interior"};

/** @brief points in one group of the given dimensionality, for element order n. */
inline unsigned long pts_per_group(int ndim, unsigned int n) {
    switch (ndim) {
        case 3: return (unsigned long)n * n * n;
        case 2: return (unsigned long)n * n;
        case 1: return (unsigned long)n;
        default: return 1ul;
    }
}

/**
 * @brief Wire bytes for one group under per-group scaled quantization.
 *
 * Layout is [T scale][Q code * npts]; if that is not smaller than the raw
 * group, the codec stores the group raw instead (which is what
 * QuantizeCompressor decides in its constructor). Mirrors that logic exactly so
 * the ceiling reported here is the one the codec will actually achieve.
 */
inline unsigned long quant_group_bytes(unsigned long npts, unsigned int tsz,
                                       unsigned int qsz) {
    const unsigned long q = tsz + npts * qsz;
    const unsigned long raw = npts * tsz;
    return q < raw ? q : raw;
}

struct Totals {
    unsigned long groups[4] = {0, 0, 0, 0};
    unsigned long pts[4]    = {0, 0, 0, 0};
};

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, npes;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);

    m_uiMaxDepth               = (argc > 1) ? atoi(argv[1]) : 6;
    const unsigned int eOrder  = (argc > 2) ? atoi(argv[2]) : 6;
    const double wtol          = (argc > 3) ? atof(argv[3]) : 1e-3;
    // n_ as the compressor defines it: eleOrder - 1, NOT 2*eleOrder+1. These are
    // scatter-map groups, not unzip blocks.
    const unsigned int n       = eOrder - 1;

    // n == 0 silently zeroes every points-per-group and makes the whole report
    // read 1.000x, which looks like a real (null) result rather than bad input.
    // Caught exactly that way once already, from a mis-split shell argument.
    if (eOrder < 2) {
        if (!rank)
            std::fprintf(stderr,
                         "ghostScatterMapProfile: eleOrder must be >= 2 "
                         "(got %u; n = eleOrder-1 would be %u)\n",
                         eOrder, n);
        MPI_Abort(comm, 1);
    }

    _InitializeHcurve(m_uiDim);

    // A single off-centre Gaussian: cheap, and it produces a genuinely
    // non-uniform octree so the halo is not artificially regular.
    std::function<double(double, double, double)> fr = [](double x, double y,
                                                          double z) {
        const double L = (double)(1u << m_uiMaxDepth);
        const double dx = x / L - 0.55, dy = y / L - 0.5, dz = z / L - 0.45;
        return std::exp(-(dx * dx + dy * dy + dz * dz) / (2.0 * 0.08 * 0.08));
    };

    std::vector<ot::TreeNode> tmp;
    function2Octree(fr, tmp, m_uiMaxDepth, wtol, eOrder, comm);
    ot::Mesh* mesh =
        ot::createMesh(tmp.data(), tmp.size(), eOrder, comm, 1,
                       ot::SM_TYPE::FDM, DENDRO_DEFAULT_GRAIN_SZ, 0.3,
                       DENDRO_DEFAULT_SF_K);
    mesh->setDomainBounds(Point(0.0, 0.0, 0.0), Point(1.0, 1.0, 1.0));

    if (!rank) {
        std::printf(
            "\n=== ghost scatter-map profile ===\n"
            "npes=%d  maxDepth=%u  eleOrder=%u (n=%u)  wtol=%.1e\n",
            npes, m_uiMaxDepth, eOrder, n, wtol);
    }

    // ---- gather the local histogram -------------------------------------
    Totals t;
    unsigned long n_send_peers = 0, local_max_peer_pts = 0;
    std::vector<unsigned long> peer_pts;

    if (mesh->isActive()) {
        const auto& dimCounts = mesh->getSendNodeSMConfigDimCounts();
        const auto& sendProcList = mesh->getSendProcList();
        // dimCounts is indexed by ACTIVE rank, one entry per rank in the active
        // comm (zero-filled for non-peers), so walk the peer list rather than
        // the whole vector.
        for (const unsigned int p : sendProcList) {
            if (p >= dimCounts.size()) continue;
            unsigned long this_peer_pts = 0;
            for (int d = 0; d < 4; ++d) {
                const unsigned long g = dimCounts[p][d];
                const unsigned long pts = g * pts_per_group(d, n);
                t.groups[d] += g;
                t.pts[d] += pts;
                this_peer_pts += pts;
            }
            if (this_peer_pts) {
                ++n_send_peers;
                peer_pts.push_back(this_peer_pts);
                local_max_peer_pts =
                    std::max(local_max_peer_pts, this_peer_pts);
            }
        }
    }

    // ---- reduce -----------------------------------------------------------
    Totals g;
    MPI_Reduce(t.groups, g.groups, 4, MPI_UNSIGNED_LONG, MPI_SUM, 0, comm);
    MPI_Reduce(t.pts, g.pts, 4, MPI_UNSIGNED_LONG, MPI_SUM, 0, comm);

    unsigned long tot_peers = 0, max_peer_pts = 0;
    MPI_Reduce(&n_send_peers, &tot_peers, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, comm);
    MPI_Reduce(&local_max_peer_pts, &max_peer_pts, 1, MPI_UNSIGNED_LONG, MPI_MAX,
               0, comm);

    // min/mean peer payload, for the threading-granularity question
    unsigned long local_min_peer_pts =
        peer_pts.empty() ? ~0ul
                         : *std::min_element(peer_pts.begin(), peer_pts.end());
    unsigned long min_peer_pts = 0;
    MPI_Reduce(&local_min_peer_pts, &min_peer_pts, 1, MPI_UNSIGNED_LONG, MPI_MIN,
               0, comm);

    // ---- how many peers are on THIS node? --------------------------------
    // A message that never leaves the node gains nothing from compression, so
    // this bounds the fraction of the halo worth compressing at all.
    unsigned long intra = 0, inter = 0;
    {
        MPI_Comm shmem;
        MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                            &shmem);
        int shrank = 0, shsize = 0;
        MPI_Comm_rank(shmem, &shrank);
        MPI_Comm_size(shmem, &shsize);
        // global ranks that share this node
        std::vector<int> node_ranks(shsize);
        MPI_Allgather(&rank, 1, MPI_INT, node_ranks.data(), 1, MPI_INT, shmem);
        if (mesh->isActive()) {
            // NOTE: peer ids from getSendProcList() are ACTIVE-comm ranks. This
            // comparison is only exact when the active comm == world (the usual
            // case for these runs); flagged in the output when it is not.
            for (const unsigned int p : mesh->getSendProcList()) {
                const bool same_node =
                    std::find(node_ranks.begin(), node_ranks.end(), (int)p) !=
                    node_ranks.end();
                (same_node ? intra : inter)++;
            }
        }
        MPI_Comm_free(&shmem);
    }
    unsigned long tot_intra = 0, tot_inter = 0;
    MPI_Reduce(&intra, &tot_intra, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, comm);
    MPI_Reduce(&inter, &tot_inter, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, comm);

    int active_npes = mesh->isActive() ? mesh->getMPICommSize() : 0;
    int max_active  = 0;
    MPI_Reduce(&active_npes, &max_active, 1, MPI_INT, MPI_MAX, 0, comm);

    // ---- report -----------------------------------------------------------
    if (!rank) {
        const unsigned long tot_groups =
            g.groups[0] + g.groups[1] + g.groups[2] + g.groups[3];
        const unsigned long tot_pts =
            g.pts[0] + g.pts[1] + g.pts[2] + g.pts[3];
        if (!tot_groups) {
            std::printf("  no ghost groups (npes=1 or inactive mesh)\n");
        } else {
            std::printf(
                "\n  scatter-map groups (send side, summed over ranks)\n"
                "  %-12s %10s %8s %12s %8s %8s\n",
                "dim", "groups", "%grp", "points", "%pts", "pts/grp");
            for (int d = 3; d >= 0; --d) {
                std::printf("  %-12s %10lu %7.2f%% %12lu %7.2f%% %8lu\n",
                            DIM_NAME[d], g.groups[d],
                            100.0 * g.groups[d] / tot_groups, g.pts[d],
                            100.0 * g.pts[d] / tot_pts, pts_per_group(d, n));
            }
            std::printf("  %-12s %10lu %7s  %12lu\n", "TOTAL", tot_groups, "",
                        tot_pts);

            // ---- achievable ratio, per wire type ----
            std::printf(
                "\n  achievable ratio for per-group scaled quantization\n"
                "  (mirrors QuantizeCompressor: a group is stored raw when\n"
                "   [scale + codes] would not be smaller than the raw group)\n");
            struct WireT { const char* nm; unsigned int sz; };
            const WireT wires[2] = {{"float32 (CTX_FLOAT)", 4},
                                    {"float64 (CTX_DOUBLE)", 8}};
            for (const auto& w : wires) {
                for (unsigned int qsz : {2u, 1u}) {
                    unsigned long raw = 0, comp = 0;
                    unsigned long raw3 = 0, comp3 = 0;
                    for (int d = 0; d < 4; ++d) {
                        const unsigned long npg = pts_per_group(d, n);
                        raw += g.groups[d] * npg * w.sz;
                        comp += g.groups[d] * quant_group_bytes(npg, w.sz, qsz);
                        if (d == 3) {
                            raw3 = g.groups[d] * npg * w.sz;
                            comp3 = g.groups[d] * quant_group_bytes(npg, w.sz, qsz);
                        }
                    }
                    std::printf(
                        "    %-22s quant%-2u : REAL MIX %5.3fx   "
                        "(3d-only would be %5.3fx)\n",
                        w.nm, 8 * qsz, raw ? (double)raw / comp : 1.0,
                        comp3 ? (double)raw3 / comp3 : 1.0);
                }
            }

            // ---- upper bound: what if the header were free? ----
            std::printf(
                "\n  ceiling if the per-group header were FREE (i.e. the pure\n"
                "  bit-width gain, unreachable but bounds any such codec):\n"
                "    float32 -> quant16 2.000x   quant8 4.000x\n");

            std::printf(
                "\n  peer payload sizes (threading granularity)\n"
                "    send peers (summed over ranks) : %lu\n"
                "    points per peer  min / max     : %lu / %lu\n"
                "    -> bytes per peer (float32)    : %.1f KiB / %.1f KiB\n",
                tot_peers, min_peer_pts, max_peer_pts,
                min_peer_pts * 4.0 / 1024.0, max_peer_pts * 4.0 / 1024.0);
            std::printf(
                "\n  peer locality (compressing an intra-node message is a loss)\n"
                "    intra-node peers : %lu\n"
                "    inter-node peers : %lu\n",
                tot_intra, tot_inter);
            if (max_active != npes) {
                std::printf(
                    "    [!] active comm (%d) != world (%d): the intra/inter\n"
                    "        split compares active-comm peer ids against world\n"
                    "        ranks and is NOT reliable for this run.\n",
                    max_active, npes);
            }
        }
        std::printf("\n");
    }

    delete mesh;
    MPI_Finalize();
    return 0;
}
