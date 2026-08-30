/**
 * @file mem_probe.h
 * @brief Opt-in per-rank resident-memory probe for the remesh path.
 *
 * Reads VmRSS/VmHWM/VmSize out of /proc/self/status and prints one `[mem]` CSV
 * line per rank to stderr. Gated at runtime by BSSN_MEM_PROBE=1, so an
 * instrumented binary is byte-for-byte the production binary in behaviour when
 * the variable is unset -- a branch and a return.
 *
 * With BSSN_MEM_PROBE_RESET=1 the probe clears the kernel high-water mark
 * (write "5" to /proc/self/clear_refs) AFTER printing, so each line's VmHWM is
 * the peak over the interval since the previous probe rather than the peak
 * since exec. Without it, VmHWM is monotone and only the final value is
 * meaningful.
 *
 * Header-only and allocation-free on the measured path.
 */
#pragma once

#include <mpi.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <ctime>

namespace dendro {

inline void mem_probe(const char* tag, int step) {
    static const int s_on = (std::getenv("BSSN_MEM_PROBE") != nullptr &&
                             std::atoi(std::getenv("BSSN_MEM_PROBE")) != 0)
                                ? 1
                                : 0;
    if (!s_on) return;
    static const int s_reset =
        (std::getenv("BSSN_MEM_PROBE_RESET") != nullptr &&
         std::atoi(std::getenv("BSSN_MEM_PROBE_RESET")) != 0)
            ? 1
            : 0;

    long vm_rss = -1, vm_hwm = -1, vm_size = -1;
    if (FILE* f = std::fopen("/proc/self/status", "r")) {
        char line[256];
        while (std::fgets(line, sizeof(line), f) != nullptr) {
            if (std::sscanf(line, "VmRSS: %ld kB", &vm_rss) == 1) continue;
            if (std::sscanf(line, "VmHWM: %ld kB", &vm_hwm) == 1) continue;
            if (std::sscanf(line, "VmSize: %ld kB", &vm_size) == 1) continue;
        }
        std::fclose(f);
    }

    char host[128];
    host[0] = '\0';
    gethostname(host, sizeof(host) - 1);
    host[sizeof(host) - 1] = '\0';

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // monotonic seconds: the gap between two probe points is the wall cost of
    // the phase between them, which is how the remesh path is A/B'd.
    struct timespec ts_;
    clock_gettime(CLOCK_MONOTONIC, &ts_);
    const double t_now = (double)ts_.tv_sec + 1e-9 * (double)ts_.tv_nsec;

    // same column order as bssn::bssn_mem_probe, with the mesh-derived columns
    // left at 0 so one parser reads both.
    std::fprintf(stderr, "[mem],%d,%s,%s,%d,%ld,%ld,%ld,0,0,0,0,0,0,0,0,%.6f\n",
                 rank, host, tag, step, vm_rss, vm_hwm, vm_size, t_now);
    std::fflush(stderr);

    if (s_reset) {
        if (FILE* c = std::fopen("/proc/self/clear_refs", "w")) {
            std::fputs("5\n", c);
            std::fclose(c);
        }
    }
}

}  // namespace dendro
