/**
 * @file ghostRecvCheck.h
 * @brief Opt-in check that a completed ghost receive was actually written.
 *
 * With DENDRO_GHOST_RECV_CHECK=1 in the environment, Mesh::readFromGhostBegin
 * fills the receive buffer with a signaling-NaN sentinel before posting the
 * Irecvs, and Mesh::readFromGhostEnd scans each peer's range after MPI_Waitall.
 * A sentinel that survives means the transport reported the message complete
 * (full count) without writing those words; the rank prints the peer, the
 * range and the count, and aborts. Off by default: nothing is filled or
 * scanned, so results are unchanged.
 *
 * Why: on kingspeak (Open MPI 5.0.3 + UCX 1.15, ASan build run with
 * `--mca memory ^patcher`) 272,576 of 1,888,848 words of one ghost message
 * were never written while MPI_Waitall returned the full count, and the stale
 * words became NaN constraints at t = 0.
 *
 * Only float and double buffers are checked (a NaN sentinel cannot collide
 * with data the solver produces); other types pass through unchecked. The
 * sentinel is written and compared as raw bits, so no floating-point
 * operation touches it (safe under feenableexcept).
 */
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

/// true when DENDRO_GHOST_RECV_CHECK is set to a value other than "0" (read once)
inline bool enabled() {
    static const bool on = [] {
        const char* s = std::getenv("DENDRO_GHOST_RECV_CHECK");
        return s != nullptr && s[0] != '\0' && std::strcmp(s, "0") != 0;
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
 * @param count nodes received from the peer (words per variable)
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
