/**
 * @file breadcrumb.h
 * @brief Opt-in unbuffered per-rank breadcrumbs (DENDRO_BREADCRUMB_DIR).
 *
 * With DENDRO_BREADCRUMB_DIR=<dir> every rank appends "t=<MPI_Wtime> <tag>"
 * lines to <dir>/r<world rank>.txt with one write(2) per line (O_APPEND, no
 * stdio buffering), so the files survive a kill at a time limit. Records
 * written with rec() also go there; when the directory is unset they fall
 * back to <DENDRO_BREADCRUMB_REC_DIR or "first_xchg">/r<rank>.txt so a
 * profiler can stay reduce-free. tools/breadcrumbs.py aggregates the files.
 */
#pragma once

#include <fcntl.h>
#include <mpi.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace dendro {
namespace bc {

namespace detail {
inline int openRankFile(const char* dir) {
    int init = 0, fin = 0;
    MPI_Initialized(&init);
    MPI_Finalized(&fin);
    if (!init || fin) return -2;
    int r = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    mkdir(dir, 0755);
    const std::string p = std::string(dir) + "/r" + std::to_string(r) + ".txt";
    const int f = open(p.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
    return f < 0 ? -1 : f;
}

/** Lazily opened descriptor; -1 = off, -2 = MPI not up yet (retry later). */
inline int& fdSlot(bool rec) {
    static int f[2] = {-2, -2};
    return f[rec ? 1 : 0];
}

inline int fd(bool rec) {
    int& f = fdSlot(rec);
    if (f != -2) return f;
    const char* d = std::getenv("DENDRO_BREADCRUMB_DIR");
    if (d && *d) {
        f = rec ? fd(false) : openRankFile(d);
        return f;
    }
    if (!rec) return f = -1;
    const char* rd = std::getenv("DENDRO_BREADCRUMB_REC_DIR");
    f = openRankFile(rd && *rd ? rd : "first_xchg");
    return f;
}

inline void vwrite(int f, const char* fmt, va_list ap) {
    char buf[1024];
    int n = std::snprintf(buf, sizeof(buf), "t=%.6f ", MPI_Wtime());
    const int m = std::vsnprintf(buf + n, sizeof(buf) - n - 1, fmt, ap);
    n += m < 0 ? 0 : (m > (int)sizeof(buf) - n - 2 ? (int)sizeof(buf) - n - 2 : m);
    buf[n++] = '\n';
    ssize_t w = ::write(f, buf, (size_t)n);
    (void)w;
}
}  // namespace detail

/** True when DENDRO_BREADCRUMB_DIR is set (and MPI is initialized). */
inline bool on() { return detail::fd(false) >= 0; }

/** Append one breadcrumb line to this rank's file (no-op when off). */
__attribute__((format(printf, 1, 2))) inline void mark(const char* fmt, ...) {
    const int f = detail::fd(false);
    if (f < 0) return;
    va_list ap;
    va_start(ap, fmt);
    detail::vwrite(f, fmt, ap);
    va_end(ap);
}

/** Append one record line (profiler output) to this rank's file: the
 *  breadcrumb file when on, else the record directory. Never collective. */
__attribute__((format(printf, 1, 2))) inline void rec(const char* fmt, ...) {
    const int f = detail::fd(true);
    if (f < 0) return;
    va_list ap;
    va_start(ap, fmt);
    detail::vwrite(f, fmt, ap);
    va_end(ap);
}

/** A report line: to this rank's breadcrumb file when on, else to stdout
 *  (newline appended, flushed). */
__attribute__((format(printf, 1, 2))) inline void line(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    const int f = detail::fd(false);
    if (f >= 0) {
        detail::vwrite(f, fmt, ap);
    } else {
        std::vprintf(fmt, ap);
        std::fputc('\n', stdout);
        std::fflush(stdout);
    }
    va_end(ap);
}

/** Entry/exit pair for a scope: "<tag>+ id=<id>" on entry, "<tag>- id=<id>" on exit. */
struct Scope {
    const char* tag_;
    int id_;
    explicit Scope(const char* tag, int id = -1) : tag_(on() ? tag : nullptr), id_(id) {
        if (tag_) mark("%s+ id=%d", tag_, id_);
    }
    ~Scope() {
        if (tag_) mark("%s- id=%d", tag_, id_);
    }
};

}  // namespace bc
}  // namespace dendro
