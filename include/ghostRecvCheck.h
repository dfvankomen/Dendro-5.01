/**
 * @file ghostRecvCheck.h
 * @brief Opt-in check that a completed ghost receive was actually written.
 * With DENDRO_GHOST_RECV_CHECK=1, readFromGhostBegin fills each receive range
 * with a sentinel and readFromGhostEnd aborts if one survives MPI_Waitall.
 * Only float and double buffers are checked. */
#pragma once

#include <mpi.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ot {
namespace ghost_recv_check {

/// signaling NaNs (quiet bit clear, nonzero payload); arithmetic never makes them
constexpr std::uint64_t SENTINEL_F64 = 0x7ff4dead5eed0001ull;
constexpr std::uint32_t SENTINEL_F32 = 0x7fa5eed1u;

/// DENDRO_GHOST_RECV_CHECK, read once: 1/true/yes/on or 0/false/no/off, unset
/// or empty is off, else abort. Owner: dfvankomen. Remove once no target
/// cluster's MPI completes a receive unwritten (UCX under ASan).
inline bool enabled() {
    static const bool on = [] {
        const char* s = std::getenv("DENDRO_GHOST_RECV_CHECK");
        if (s == nullptr || s[0] == '\0') return false;
        const char* const on_words[]  = {"1", "true", "yes", "on"};
        const char* const off_words[] = {"0", "false", "no", "off"};
        for (const char* w : on_words)
            if (std::strcmp(s, w) == 0) return true;
        for (const char* w : off_words)
            if (std::strcmp(s, w) == 0) return false;
        std::fprintf(stderr,
                     "DENDRO_GHOST_RECV_CHECK=%s: expected 1/true/yes/on or "
                     "0/false/no/off\n",
                     s);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return false;
    }();
    return on;
}

namespace detail {
template <typename W>
inline void fill_bits(void* buf, std::size_t n, W w) {
    unsigned char* p = static_cast<unsigned char*>(buf);
    for (std::size_t i = 0; i < n; i++) std::memcpy(p + i * sizeof(W), &w, sizeof(W));
}

template <typename W>
inline std::size_t count_bits(const void* buf, std::size_t n, W w,
                              std::size_t* first, std::size_t* last) {
    const unsigned char* p = static_cast<const unsigned char*>(buf);
    std::size_t c          = 0;
    for (std::size_t i = 0; i < n; i++) {
        W v;
        std::memcpy(&v, p + i * sizeof(W), sizeof(W));
        if (v == w) {
            if (c == 0) *first = i;
            *last = i;
            c++;
        }
    }
    return c;
}
}  // namespace detail

/// fill n words of buf with the sentinel (no-op for unchecked types)
template <typename T>
inline void poison(T*, std::size_t) {}
inline void poison(double* buf, std::size_t n) { detail::fill_bits(buf, n, SENTINEL_F64); }
inline void poison(float* buf, std::size_t n) { detail::fill_bits(buf, n, SENTINEL_F32); }

/// number of sentinel words in buf[0, n); first/last index set when nonzero
template <typename T>
inline std::size_t count_unwritten(const T*, std::size_t, std::size_t*, std::size_t*) {
    return 0;
}
inline std::size_t count_unwritten(const double* buf, std::size_t n, std::size_t* first,
                                   std::size_t* last) {
    return detail::count_bits(buf, n, SENTINEL_F64, first, last);
}
inline std::size_t count_unwritten(const float* buf, std::size_t n, std::size_t* first,
                                   std::size_t* last) {
    return detail::count_bits(buf, n, SENTINEL_F32, first, last);
}

/**
 * @brief scan one peer's received range; print and abort if any word is unwritten
 * @param buf start of the peer's range (dof * count words)
 * @param dof variables in the range, each count words long
 * @param count nodes received from the peer (words per variable)
 * @param rank this rank, for the message
 * @param from the peer the range came from
 * @param comm communicator to abort
 */
template <typename T>
inline void verify_or_abort(const T* buf, unsigned int dof, unsigned int count,
                            int rank, unsigned int from, MPI_Comm comm) {
    const std::size_t len = (std::size_t)dof * count;
    std::size_t first = 0, last = 0;
    const std::size_t bad = count_unwritten(buf, len, &first, &last);
    if (bad == 0) return;
    std::fprintf(stderr,
                 "ghost recv check: rank %d from %u: %zu of %zu words unwritten after "
                 "MPI_Waitall (first %zu = var %zu node %zu, last %zu); the transport "
                 "completed the receive without delivering them\n",
                 rank, from, bad, len, first, first / count, first % count, last);
    std::fflush(stderr);
    MPI_Abort(comm, 1);
}

}  // namespace ghost_recv_check
}  // namespace ot
