/**
 * edgeCutCompare -- partition QUALITY comparison: SFC/Hilbert vs fastpart, on
 * the SAME octree. Emits JSON Lines for tools/partition_quality/analyze_*.py.
 *
 * WHY THIS EXISTS
 * ---------------
 * Measured 2026-07-15: graph mode moves ~1.87x SFC's ghost bytes and costs
 * ~+50% wall-clock at np=4. The surplus is NOT our plumbing (DOG(unzip) is
 * identical between modes -> block fragmentation refuted). What is left is raw
 * partition shape. But edge cut is the one thing a graph partitioner
 * *explicitly minimises* and a space-filling curve does not optimise at all --
 * so fastpart losing at it would be a statement about the algorithm.
 *
 * THE HYPOTHESIS THIS TESTS
 * -------------------------
 * fastpart's dgraph_from_octree builds edges from e2e[6] -- FACE neighbours
 * only. Dendro's real ghost traffic follows block padding, which reaches EDGE
 * and CORNER neighbours too. So fastpart may be minimising an objective that
 * is not the cost, while Hilbert -- being *geometric* -- is good at all
 * locality at once. We therefore score every labelling under BOTH graphs:
 *
 *   graph=face6      6 face neighbours   (what fastpart optimises)
 *   graph=stencil26  6 + 12 edge + 8 corner (what ~costs real money)
 *
 * win on face6 + loss on stencil26 => wrong objective; DENDRO_GRAPH_FULL_STENCIL
 *                                     is the existing lever.
 * loss on BOTH                     => partitioner is weaker than the curve
 *                                     (no multilevel coarsening); dendrolib
 *                                     cannot fix it.
 *
 * WHAT IT DELIBERATELY DOES NOT DO
 * --------------------------------
 * It does not call repartitionMeshGlobal. That would (a) rebuild every
 * downstream structure we don't need and (b) overwrite fastpart's labels with
 * the block-atomic vote, so we would be scoring the vote, not the partitioner.
 * We take the raw fastpart labels. `--variants fastpart_voted` is intentionally
 * absent -- if you want to know what the vote costs, that is a separate
 * experiment (and a good one).
 *
 * Edge cut is deterministic, so this needs no reps and no timing discipline.
 *
 * Usage:
 *   mpirun -np 4 ./edgeCutCompare --max-depth 7 --json out.jsonl
 *   mpirun -np 4 ./edgeCutCompare --variants sfc,fastpart,fastpart_fullstencil \
 *          --max-depth 8 --refine blob --json out.jsonl
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "TreeNode.h"
#include "dendro.h"
#include "mesh.h"
#include "meshUtils.h"
#include "mpi.h"
#include "octUtils.h"

extern "C" {
#include "fastpart.h"
}

namespace {

struct Opts {
    unsigned int max_depth = 7;
    double wavelet_tol = 1e-5;
    double partition_tol = 0.1;
    unsigned int eorder = 4;
    unsigned int grain = 50;
    std::string variants = "sfc,fastpart";
    std::string refine = "sine";
    std::string json_path;
    std::string label;
};

void usage(const char* argv0) {
    std::cout
        << "usage: " << argv0 << " [options]\n"
        << "  --max-depth N        max octree depth        (default 7)\n"
        << "  --wavelet-tol X      wavelet tolerance       (default 1e-5)\n"
        << "  --partition-tol X    load imbalance tol      (default 0.1)\n"
        << "  --eorder N           element order           (default 4)\n"
        << "  --grain N            dendro grain size       (default 50)\n"
        << "  --refine NAME        sine|blob|offblob|puncture|puncture2\n"
        << "                       (default sine)\n"
        << "  --variants LIST      comma list of:\n"
        << "                         sfc, sfc_weighted, sfc_levelwise, fastpart,\n"
        << "                         fastpart_fullstencil, fastpart_weighted\n"
        << "                       (default sfc,fastpart)\n"
        << "  --label S            free-form tag copied into config.label\n"
        << "  --json PATH          write JSONL here (rank 0); default stdout\n";
}

std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(item);
    }
    return out;
}

// ---------------------------------------------------------------- metrics

struct Metrics {
    long long edge_cut = 0;        // undirected, assuming symmetric adjacency
    long long edge_cut_directed = 0;
    long long total_edges = 0;     // undirected
    long long total_edges_directed = 0;
    long long boundary_vertices = 0;
    long long total_comm_volume = 0;  // sum_v |distinct remote parts touching v|
    double edge_cut_frac = 0.0;
    double ele_imbalance = 0.0;
    double work_imbalance = 0.0;
    double lts_eff = 0.0;        // per-sub-step ENUTS parallel efficiency
    double lts_eff_bound = 0.0;  // ceil-granularity ceiling on lts_eff
    double lts_comm_scoped_ratio = 0.0;  // scoped-exchange comms / full-every-substep
    long long n_elements = 0;
};

/** Per-sub-step ENUTS parallel efficiency of a labelling.
 *
 * work_imbalance above scores TOTAL LTS work per rank -- which any weighted
 * 1-D prefix split balances trivially and which does NOT predict LTS speedup.
 * What gates the ENUTS evolve is per-PARTIAL-STEP balance: partial_evolve(pt)
 * runs a collective DG ghost exchange per RK stage (enuts.h:1905), so every
 * rank waits on the busiest rank at every pt. A block at level L fires when
 * pt % 2^(lmax-L) == 0 (getBlkTimestepFac), each firing costing ~its element
 * count. So:
 *
 *   eff = sum_pt mean_rank(active work at pt) / sum_pt max_rank(active work)
 *
 * = the fraction of busiest-rank-gated capacity doing useful work, the same
 * accounting dump_est_speedup does (it reduces min/sum/max per pt but only
 * prints the sum). eff_bound is the ceil-granularity ceiling: even a perfect
 * partition cannot beat ceil(active_total/np) on the max at any pt -- the
 * "deepest levels are near-point" floor, made visible.
 */
void score_lts_substeps(const std::vector<unsigned int>& local_gids,
                        const std::vector<unsigned int>& label_of_gid,
                        const std::vector<unsigned int>& level_of_local,
                        unsigned int lmin, unsigned int lmax, unsigned int npes,
                        MPI_Comm comm, Metrics& m) {
    const unsigned int nlev = lmax - lmin + 1;
    std::vector<long long> cnt(npes * nlev, 0), cnt_g(npes * nlev, 0);
    for (size_t i = 0; i < local_gids.size(); ++i) {
        unsigned int lev = level_of_local[i];
        if (lev < lmin) lev = lmin;
        if (lev > lmax) lev = lmax;
        cnt[label_of_gid[local_gids[i]] * nlev + (lev - lmin)] += 1;
    }
    MPI_Allreduce(cnt.data(), cnt_g.data(), npes * nlev, MPI_LONG_LONG, MPI_SUM,
                  comm);

    const unsigned int coarset_t = 1u << (lmax - lmin);
    double sum_mean = 0.0, sum_max = 0.0, sum_ideal = 0.0;
    std::vector<long long> active(npes);
    for (unsigned int pt = 0; pt < coarset_t; ++pt) {
        long long total = 0, amax = 0;
        for (unsigned int p = 0; p < npes; ++p) {
            long long a = 0;
            for (unsigned int l = 0; l < nlev; ++l) {
                const unsigned int blk_dt = 1u << (lmax - (lmin + l));
                if (pt % blk_dt == 0) a += cnt_g[p * nlev + l];
            }
            active[p] = a;
            total += a;
            amax = std::max(amax, a);
        }
        if (total == 0) continue;
        sum_mean += (double)total / (double)npes;
        sum_max += (double)amax;
        sum_ideal += (double)((total + npes - 1) / npes);  // ceil(total/np)
    }
    m.lts_eff = sum_max > 0 ? sum_mean / sum_max : 0.0;
    m.lts_eff_bound = sum_ideal > 0 ? sum_mean / sum_ideal : 0.0;
}

/** Score one labelling under one adjacency graph.
 *
 * `label_of_gid` maps a global element id -> owning part.
 * `adj` yields, for local element index i, the global ids of its neighbours in
 * the chosen graph (LOOK_UP_TABLE_DEFAULT entries already stripped).
 *
 * Cut is counted DIRECTED (every (local elem, neighbour) pair) and halved. For
 * a symmetric graph that is the exact undirected count; if Dendro's E2E is
 * asymmetric across a level jump the halving is approximate -- which is why
 * edge_cut_frac (a ratio on the same graph, so the bias cancels between
 * partitioners) is the number to trust.
 */
Metrics score(const std::vector<std::vector<unsigned int>>& adj,
              const std::vector<unsigned int>& local_gids,
              const std::vector<unsigned int>& label_of_gid,
              const std::vector<unsigned int>& level_of_local,
              unsigned int lmin, unsigned int lmax, unsigned int npes,
              MPI_Comm comm) {
    Metrics m;
    long long cut_d = 0, edges_d = 0, bnd = 0, commvol = 0;
    std::vector<long long> ele_per_part(npes, 0), work_per_part(npes, 0);
    // comm volume binned by the OWNING element's level -> lets us model a
    // ghost exchange scoped to the blocks active at each LTS sub-step.
    const unsigned int nlev = lmax - lmin + 1;
    std::vector<long long> commvol_by_lev(nlev, 0);

    for (size_t i = 0; i < adj.size(); ++i) {
        const unsigned int gid = local_gids[i];
        const unsigned int mine = label_of_gid[gid];
        ele_per_part[mine] += 1;
        work_per_part[mine] +=
            (long long)ot::oct_work_weight(level_of_local[i], lmin, lmax);

        std::set<unsigned int> remote_parts;
        for (unsigned int ngid : adj[i]) {
            edges_d++;
            const unsigned int theirs = label_of_gid[ngid];
            if (theirs != mine) {
                cut_d++;
                remote_parts.insert(theirs);
            }
        }
        if (!remote_parts.empty()) {
            bnd++;
            commvol += (long long)remote_parts.size();
            unsigned int lev = level_of_local[i];
            if (lev < lmin) lev = lmin;
            if (lev > lmax) lev = lmax;
            commvol_by_lev[lev - lmin] += (long long)remote_parts.size();
        }
    }

    long long loc[4] = {cut_d, edges_d, bnd, commvol};
    long long sum[4] = {0, 0, 0, 0};
    MPI_Allreduce(loc, sum, 4, MPI_LONG_LONG, MPI_SUM, comm);

    std::vector<long long> ele_sum(npes, 0), work_sum(npes, 0);
    MPI_Allreduce(ele_per_part.data(), ele_sum.data(), npes, MPI_LONG_LONG,
                  MPI_SUM, comm);
    MPI_Allreduce(work_per_part.data(), work_sum.data(), npes, MPI_LONG_LONG,
                  MPI_SUM, comm);

    m.edge_cut_directed = sum[0];
    m.total_edges_directed = sum[1];
    m.edge_cut = sum[0] / 2;
    m.total_edges = sum[1] / 2;
    m.boundary_vertices = sum[2];
    m.total_comm_volume = sum[3];
    m.edge_cut_frac = sum[1] ? (double)sum[0] / (double)sum[1] : 0.0;

    // --- scoping prize: LTS comms with a per-active-block exchange vs the
    // current full-ghost-layer-every-sub-step exchange (enuts.h:1905, which does
    // NOT scope to m_uiActiveBlkIDs). Over one coarse step of coarset_t=2^(lmax-
    // lmin) sub-steps the current code pays coarset_t * full_volume (== GTS
    // comms; LTS saves compute but not comms). A scoped exchange sends a level-L
    // boundary element only on the 2^(L-lmin) sub-steps where L fires. Model:
    // attribute each send slot to its owning element's level (first-order --
    // ignores the coarser-neighbour pull a fine block makes; same-level
    // dominates). ratio = scoped/full in (0,1]; prize = 1-ratio.
    std::vector<long long> cvl(nlev, 0);
    MPI_Allreduce(commvol_by_lev.data(), cvl.data(), nlev, MPI_LONG_LONG,
                  MPI_SUM, comm);
    const double coarset_t = (double)(1u << (lmax - lmin));
    double scoped = 0.0, full_tot = 0.0;
    for (unsigned int l = 0; l < nlev; ++l) {
        const double fire = (double)(1u << l);  // 2^(L-lmin) firings/coarse step
        scoped += (double)cvl[l] * fire;
        full_tot += (double)cvl[l];
    }
    m.lts_comm_scoped_ratio =
        full_tot > 0 ? scoped / (coarset_t * full_tot) : 0.0;

    long long ele_tot = 0, work_tot = 0, ele_max = 0, work_max = 0;
    for (unsigned int p = 0; p < npes; ++p) {
        ele_tot += ele_sum[p];
        work_tot += work_sum[p];
        ele_max = std::max(ele_max, ele_sum[p]);
        work_max = std::max(work_max, work_sum[p]);
    }
    m.n_elements = ele_tot;
    const double ele_mean = (double)ele_tot / (double)npes;
    const double work_mean = (double)work_tot / (double)npes;
    m.ele_imbalance = ele_mean > 0 ? ele_max / ele_mean : 0.0;
    m.work_imbalance = work_mean > 0 ? work_max / work_mean : 0.0;
    return m;
}

void emit(std::ostream& os, const Opts& o, unsigned int npes,
          const std::string& partitioner, const std::string& graph,
          const Metrics& m) {
    os << "{"
       << "\"schema\":1"
       << ",\"config\":{"
       << "\"npes\":" << npes
       << ",\"max_depth\":" << o.max_depth
       << ",\"wavelet_tol\":" << o.wavelet_tol
       << ",\"grain_sz\":" << o.grain
       << ",\"partition_tol\":" << o.partition_tol
       << ",\"eorder\":" << o.eorder
       << ",\"refine\":\"" << o.refine << "\"";
    if (!o.label.empty()) os << ",\"label\":\"" << o.label << "\"";
    os << "}"
       << ",\"partitioner\":\"" << partitioner << "\""
       << ",\"graph\":\"" << graph << "\""
       << ",\"metrics\":{"
       << "\"n_elements\":" << m.n_elements
       << ",\"edge_cut\":" << m.edge_cut
       << ",\"edge_cut_directed\":" << m.edge_cut_directed
       << ",\"total_edges\":" << m.total_edges
       << ",\"edge_cut_frac\":" << m.edge_cut_frac
       << ",\"boundary_vertices\":" << m.boundary_vertices
       << ",\"total_comm_volume\":" << m.total_comm_volume
       << ",\"ele_imbalance\":" << m.ele_imbalance
       << ",\"work_imbalance\":" << m.work_imbalance
       << ",\"lts_eff\":" << m.lts_eff
       << ",\"lts_eff_bound\":" << m.lts_eff_bound
       << ",\"lts_comm_scoped_ratio\":" << m.lts_comm_scoped_ratio
       << "}}" << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, npes_i;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes_i);
    const unsigned int npes = (unsigned int)npes_i;

    Opts o;
    for (int i = 1; i < argc; ++i) {
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) {
                if (!rank) std::cerr << "missing value for " << what << "\n";
                MPI_Abort(comm, 2);
            }
            return argv[++i];
        };
        std::string a = argv[i];
        if (a == "--max-depth") o.max_depth = atoi(next("--max-depth"));
        else if (a == "--wavelet-tol") o.wavelet_tol = atof(next("--wavelet-tol"));
        else if (a == "--partition-tol") o.partition_tol = atof(next("--partition-tol"));
        else if (a == "--eorder") o.eorder = atoi(next("--eorder"));
        else if (a == "--grain") o.grain = atoi(next("--grain"));
        else if (a == "--variants") o.variants = next("--variants");
        else if (a == "--refine") o.refine = next("--refine");
        else if (a == "--label") o.label = next("--label");
        else if (a == "--json") o.json_path = next("--json");
        else if (a == "-h" || a == "--help") {
            if (!rank) usage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (!rank) { std::cerr << "unknown arg: " << a << "\n"; usage(argv[0]); }
            MPI_Abort(comm, 2);
        }
    }

    m_uiMaxDepth = o.max_depth;
    _InitializeHcurve(m_uiDim);

    const double d_min = -5.5, d_max = 5.5;
    std::function<double(double, double, double)> func;
    if (o.refine == "blob") {
        // gaussian blob -> genuine depth spread (deep at centre, coarse far
        // field). closer to EM4/BSSN, and the regime where work-imbalance --
        // graph partitioning's one measured win -- actually exists.
        func = [d_min, d_max](double x, double y, double z) {
            double xx = (x / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;
            double yy = (y / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;
            double zz = (z / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;
            return exp(-(xx * xx + yy * yy + zz * zz) / 2.0);
        };
    } else if (o.refine == "offblob") {
        // OFF-CENTRE gaussian in the +++ octant. This is the config the
        // work-balance study needs: the centred `blob` is symmetric across all 8
        // top-level octants, so an equal-COUNT SFC split gets equal WORK for free
        // and work_imbalance reads a meaningless ~1.0 (see sweep.sh note). Here
        // the deep refinement clusters in one spatial region -> one contiguous
        // stretch of the Hilbert curve -> it lands on a FEW ranks. SFC (which
        // balances element count) then piles the heavy sub-cycled elements onto
        // those ranks = high work_imbalance; a work-weighted partitioner is
        // supposed to spread them. This is the whole point of the experiment, so
        // the asymmetry is deliberate and load-bearing.
        const double cx = 3.5, cy = 3.5, cz = 3.5;  // squarely inside +++ octant
        func = [d_min, d_max, cx, cy, cz](double x, double y, double z) {
            double xx = (x / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;
            double yy = (y / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;
            double zz = (z / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;
            double r2 = (xx - cx) * (xx - cx) + (yy - cy) * (yy - cy) +
                        (zz - cz) * (zz - cz);
            return exp(-r2 / 2.0);
        };
    } else if (o.refine == "puncture" || o.refine == "puncture2") {
        // SHARP near-point spike(s) -- the LTS regime offblob cannot expose.
        // offblob's sigma=1 gaussian is diffuse: the deep levels occupy a
        // sizeable region and several ranks share them. A puncture (BBH) is the
        // opposite: the deepest levels live in a tiny volume = a SHORT stretch
        // of the Hilbert curve = ONE rank under any contiguous SFC split, even
        // weighted. That rank then gates every fine sub-step. puncture2 places
        // two spikes (BBH-like) off-axis so neither sits on an octant seam.
        const double sig2 = 2.0 * 0.25 * 0.25;  // sigma=0.25 -> near-point
        const bool two = (o.refine == "puncture2");
        func = [d_min, d_max, sig2, two](double x, double y, double z) {
            double xx = (x / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;
            double yy = (y / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;
            double zz = (z / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;
            double r2a = (xx - 2.1) * (xx - 2.1) + (yy - 0.3) * (yy - 0.3) +
                         (zz - 0.3) * (zz - 0.3);
            double v = exp(-r2a / sig2);
            if (two) {
                double r2b = (xx + 2.1) * (xx + 2.1) +
                             (yy + 0.3) * (yy + 0.3) + (zz + 0.3) * (zz + 0.3);
                v += exp(-r2b / sig2);
            }
            return v;
        };
    } else if (o.refine == "sine") {
        // matches partitioningMeshTests.cpp so results are comparable to the
        // existing harness.
        func = [d_min, d_max](double x, double y, double z) {
            double xx = (x / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;
            double yy = (y / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;
            double zz = (z / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;
            return sin(2 * M_PI * xx) * sin(2 * M_PI * yy) * sin(2 * M_PI * zz);
        };
    } else {
        if (!rank) std::cerr << "unknown --refine: " << o.refine << "\n";
        MPI_Abort(comm, 2);
    }

    std::vector<ot::TreeNode> tmpNodes;
    function2Octree(func, tmpNodes, m_uiMaxDepth, o.wavelet_tol, o.eorder, comm);

    // pBlockSetup=false: a cut-quality tool never unzips, and block decomposition
    // is the expensive part of mesh construction.
    ot::Mesh* mesh =
        ot::createMesh(tmpNodes.data(), tmpNodes.size(), o.eorder, comm, 1,
                       ot::SM_TYPE::FDM, o.grain, o.partition_tol, 256, NULL,
                       false);
    mesh->setDomainBounds(Point(d_min, d_min, d_min), Point(d_max, d_max, d_max));

    if (!mesh->isActive()) {
        // inactive ranks hold no elements; nothing to score.
        delete mesh;
        MPI_Finalize();
        return 0;
    }

    unsigned int lmin = 0, lmax = 0;
    mesh->computeMinMaxLevel(lmin, lmax);

    // one connectivity map feeds BOTH arms -> identical adjacency by construction
    auto [conn, local_to_global, ele_offsets, ele_counts] =
        mesh->buildOctantConnectivityMap<unsigned int>();

    const size_t nloc = conn.size();
    const unsigned int num_ele_global = ele_offsets.back();

    std::vector<unsigned int> local_gids(nloc), level_of_local(nloc);
    for (size_t i = 0; i < nloc; ++i) {
        local_gids[i] = conn[i].eid;
        level_of_local[i] = conn[i].level;
    }

    // adjacency in global ids, per graph
    std::vector<std::vector<unsigned int>> adj_face(nloc), adj_sten(nloc);
    for (size_t i = 0; i < nloc; ++i) {
        for (int k = 0; k < 6; ++k) {
            unsigned int n = conn[i].e2e[k];
            if (n != LOOK_UP_TABLE_DEFAULT && n < num_ele_global) {
                adj_face[i].push_back(n);
                adj_sten[i].push_back(n);
            }
        }
        for (int k = 0; k < 12; ++k) {
            unsigned int n = conn[i].edgeNeighbors[k];
            if (n != LOOK_UP_TABLE_DEFAULT && n < num_ele_global)
                adj_sten[i].push_back(n);
        }
        for (int k = 0; k < 8; ++k) {
            unsigned int n = conn[i].vertexNeighbors[k];
            if (n != LOOK_UP_TABLE_DEFAULT && n < num_ele_global)
                adj_sten[i].push_back(n);
        }
    }

    // ---- SFC labels: owner is implied by the element-offset ranges
    std::vector<unsigned int> sfc_label(num_ele_global, 0);
    for (unsigned int p = 0; p < npes; ++p)
        for (unsigned int g = ele_offsets[p]; g < ele_offsets[p + 1]; ++g)
            sfc_label[g] = p;

    // ---- fastpart input (shared by every fastpart variant)
    // field-by-field, mirroring repartitionMeshGlobal (src/mesh.cpp:18745) --
    // oct_data and oct_element are NOT layout-compatible, so no memcpy.
    std::vector<oct_element> oct_in(nloc);
    for (size_t i = 0; i < nloc; ++i) {
        oct_element& e = oct_in[i];
        e.rank = conn[i].rank;
        e.trank = conn[i].trank;
        e.eid = conn[i].eid;
        e.localid = conn[i].localid;
        e.level = conn[i].level;
        for (int k = 0; k < 3; ++k) e.coord[k] = conn[i].coord[k];
        for (int k = 0; k < 6; ++k) e.e2e[k] = conn[i].e2e[k];
        for (int k = 0; k < 12; ++k) e.edgeNeighbors[k] = conn[i].edgeNeighbors[k];
        for (int k = 0; k < 8; ++k) e.vertexNeighbors[k] = conn[i].vertexNeighbors[k];
    }
    std::vector<fastpart_uint_t> vtx_dist(ele_offsets.begin(), ele_offsets.end());

    std::vector<unsigned int> work_w(nloc, 1);
    for (size_t i = 0; i < nloc; ++i)
        work_w[i] = ot::oct_work_weight(conn[i].level, lmin, lmax);

    // ---- SFC-WEIGHTED labels: the honest LTS baseline. sfc_label balances
    // element COUNT; under LTS the sensible SFC choice balances WORK -- split
    // the SAME Hilbert order into equal-WEIGHT segments (idealised, element-
    // granular). This is what ENUTS useWpart does via Dendro's weighted flexible
    // partition (getOctWeight == oct_work_weight), and the tougher baseline graph
    // must beat: if sfc_weighted already balances work, graph's LTS advantage is
    // illusory -- the lever is WEIGHTING (available to both), not the graph.
    // (Confirmed empirically: nlsmNUTS SFC-weighted LD Bal was perfect 1.000.)
    std::vector<unsigned int> sfc_weighted_label(num_ele_global, 0);
    {
        std::vector<int> cnt(npes), disp(npes);
        for (unsigned int p = 0; p < npes; ++p) {
            cnt[p]  = (int)(ele_offsets[p + 1] - ele_offsets[p]);
            disp[p] = (int)ele_offsets[p];
        }
        std::vector<long long> gw(num_ele_global, 0), mine(nloc);
        for (size_t i = 0; i < nloc; ++i) mine[i] = (long long)work_w[i];
        MPI_Allgatherv(mine.data(), (int)nloc, MPI_LONG_LONG, gw.data(),
                       cnt.data(), disp.data(), MPI_LONG_LONG, comm);
        long long total = 0;
        for (unsigned int g = 0; g < num_ele_global; ++g) total += gw[g];
        // assign each element by the midpoint of its weight-interval -> balanced,
        // contiguous segments (SFC order preserved since the prefix is monotone).
        const double target = (double)total / (double)npes;
        long long prefix = 0;
        for (unsigned int g = 0; g < num_ele_global; ++g) {
            double mid    = (double)prefix + 0.5 * (double)gw[g];
            unsigned int p = target > 0 ? (unsigned int)(mid / target) : 0;
            sfc_weighted_label[g] = (p >= npes) ? npes - 1 : p;
            prefix += gw[g];
        }
    }

    // ---- SFC-LEVELWISE labels: split EACH LEVEL's elements into equal-count
    // contiguous Hilbert segments, independently per level. Every rank then
    // holds ~1/np of every level => near-perfect PER-SUB-STEP balance by
    // construction -- the thing neither sfc_weighted (contiguous prefix split;
    // total-weight balance only) nor single-constraint fastpart optimises.
    // This is the cheap stand-in for a multi-constraint partition: its lts_eff
    // is (approximately) the ceiling any partitioner could reach, and its edge
    // cut is the comms price of reaching it the curve-contiguous-per-level way.
    // Each rank owns up to nlev curve segments instead of 1, so expect the cut
    // to rise; whether it rises less than lts_eff gains is the experiment.
    std::vector<unsigned int> sfc_levelwise_label(num_ele_global, 0);
    {
        std::vector<int> cnt(npes), disp(npes);
        for (unsigned int p = 0; p < npes; ++p) {
            cnt[p]  = (int)(ele_offsets[p + 1] - ele_offsets[p]);
            disp[p] = (int)ele_offsets[p];
        }
        std::vector<unsigned int> glev(num_ele_global, 0);
        MPI_Allgatherv(level_of_local.data(), (int)nloc, MPI_UNSIGNED,
                       glev.data(), cnt.data(), disp.data(), MPI_UNSIGNED,
                       comm);
        std::vector<unsigned long long> n_of_lev(m_uiMaxDepth + 2, 0),
            k_of_lev(m_uiMaxDepth + 2, 0);
        for (unsigned int g = 0; g < num_ele_global; ++g) n_of_lev[glev[g]]++;
        for (unsigned int g = 0; g < num_ele_global; ++g) {
            const unsigned int lev = glev[g];
            // k-th of n_L elements at this level -> part floor(k*np/n_L)
            unsigned int p =
                (unsigned int)((k_of_lev[lev] * (unsigned long long)npes) /
                               n_of_lev[lev]);
            sfc_levelwise_label[g] = (p >= npes) ? npes - 1 : p;
            k_of_lev[lev]++;
        }
    }

    std::ofstream fout;
    std::ostream* os = &std::cout;
    if (!o.json_path.empty() && rank == 0) {
        fout.open(o.json_path, std::ios::app);
        if (!fout) {
            std::cerr << "cannot open " << o.json_path << "\n";
            MPI_Abort(comm, 2);
        }
        os = &fout;
    }

    for (const std::string& v : split_csv(o.variants)) {
        std::vector<unsigned int> label_of_gid;

        if (v == "sfc") {
            label_of_gid = sfc_label;
        } else if (v == "sfc_weighted") {
            label_of_gid = sfc_weighted_label;
        } else if (v == "sfc_levelwise") {
            label_of_gid = sfc_levelwise_label;
        } else if (v == "fastpart" || v == "fastpart_fullstencil" ||
                   v == "fastpart_weighted") {
            std::vector<fastpart_uint_t> parts(nloc, 0);
            std::vector<fastpart_uint_t> vwgt(nloc, 1);
            for (size_t i = 0; i < nloc; ++i) vwgt[i] = (fastpart_uint_t)work_w[i];

            fastpart_oct_partopts opts;
            fastpart_oct_partopts_init(&opts);
            opts.include_edge_corner = (v == "fastpart_fullstencil");
            opts.vwgt = (v == "fastpart_weighted") ? vwgt.data() : nullptr;

            MPI_Comm c = comm;
            fastpart_partgraph_octree_ex(vtx_dist.data(), oct_in.data(), &opts,
                                         parts.data(), &c);

            // gather every rank's labels so any global id resolves in O(1)
            std::vector<int> cnt(npes), disp(npes);
            for (unsigned int p = 0; p < npes; ++p) {
                cnt[p] = (int)(ele_offsets[p + 1] - ele_offsets[p]);
                disp[p] = (int)ele_offsets[p];
            }
            std::vector<unsigned int> mine(parts.begin(), parts.end());
            label_of_gid.assign(num_ele_global, 0);
            MPI_Allgatherv(mine.data(), (int)nloc, MPI_UNSIGNED,
                           label_of_gid.data(), cnt.data(), disp.data(),
                           MPI_UNSIGNED, comm);
        } else {
            if (!rank) std::cerr << "unknown variant: " << v << "\n";
            continue;
        }

        Metrics mf = score(adj_face, local_gids, label_of_gid, level_of_local,
                           lmin, lmax, npes, comm);
        Metrics ms = score(adj_sten, local_gids, label_of_gid, level_of_local,
                           lmin, lmax, npes, comm);
        // lts_eff is a property of the labelling alone (graph-independent);
        // compute once, report on both graph rows.
        score_lts_substeps(local_gids, label_of_gid, level_of_local, lmin, lmax,
                           npes, comm, ms);
        mf.lts_eff       = ms.lts_eff;
        mf.lts_eff_bound = ms.lts_eff_bound;
        if (rank == 0) {
            emit(*os, o, npes, v, "face6", mf);
            emit(*os, o, npes, v, "stencil26", ms);
        }
    }

    if (fout.is_open()) fout.close();
    delete mesh;
    MPI_Finalize();
    return 0;
}
