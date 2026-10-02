/**
 * @file asyncExchangeContex.h
 * @author Milinda Fernando (milinda@cs.utah.edu)
 * @brief Context information for asynchronous communication
 * @version 0.1
 * @date 2018-11-19
 *
 * @copyright Copyright (c) 2020
 *
 */

#ifndef DENDRO_5_0_UPDATECTX_H
#define DENDRO_5_0_UPDATECTX_H

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <cstring>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mpi.h"

namespace ot {

/**@brief Process-wide pool for ghost-exchange buffers (default ON,
 * DENDRO_XCHG_BUF_POOL=0 disables).
 * Buffers are kept across exchanges and meshes, so a new mesh's first exchange
 * reuses warm, already-faulted and transport-registered (RDMA reg-cache hit)
 * memory instead of fresh mmap'd pages. Blocks grow with headroom and are
 * prefaulted when created; a free block too small for a request is released,
 * so the block count stays at the peak number of buffers in flight.
 * DENDRO_XCHG_BUF_POOL_MAX_MB (default 512) caps the pooled bytes per rank;
 * requests that would exceed it fall back to plain malloc/free (never fails).
 * DENDRO_XCHG_BUF_POOL_CHECK=1 poisons every handed-out buffer with 0xff bytes
 * to prove no stale content is read. DENDRO_XCHG_BUF_POOL_REPORT=1 prints the
 * per-rank peak at exit; peak is also reduced in dumpCommStats.
 * Thread safety: guarded by a mutex (exchanges are normally issued from one
 * thread per rank; production is MPI-per-core with no OpenMP). */
namespace xbuf {
inline bool poolOn() {
    static const bool on = [] {
        const char* e = std::getenv("DENDRO_XCHG_BUF_POOL");
        return !(e && e[0] == '0');
    }();
    return on;
}
inline bool poolCheck() {
    static const bool on = [] {
        const char* e = std::getenv("DENDRO_XCHG_BUF_POOL_CHECK");
        return e && e[0] == '1';
    }();
    return on;
}
inline size_t poolCapBytes() {
    static const size_t cap = [] {
        const char* e = std::getenv("DENDRO_XCHG_BUF_POOL_MAX_MB");
        double mb     = (e && e[0]) ? std::atof(e) : 512.0;
        if (mb < 0) mb = 0;
        return (size_t)(mb * 1048576.0);
    }();
    return cap;
}
struct Pool {
    std::vector<std::pair<void*, size_t>> free;
    std::unordered_map<void*, size_t> used;
    std::mutex mtx;
    size_t total = 0;  // bytes held by the pool (free + in use)
    size_t peak  = 0;
    size_t fallbacks = 0;
    int rank     = -1;
};
inline void poolReport();
inline Pool& pool() {
    static Pool p;
    return p;
}
/**@brief peak pooled bytes on this rank (0 if the pool was never used). */
inline size_t poolPeakBytes() { return pool().peak; }
inline void poolReport() {
    Pool& P = pool();
    if (P.peak == 0 && P.fallbacks == 0) return;
    std::fprintf(stderr,
                 "[xbuf-pool] rank=%d peak_MB=%.2f cap_MB=%.0f fallbacks=%zu\n",
                 P.rank, P.peak / 1048576.0, poolCapBytes() / 1048576.0,
                 P.fallbacks);
}
inline void* acquire(size_t bytes) {
    if (!poolOn()) return malloc(bytes);
    Pool& P = pool();
    std::lock_guard<std::mutex> lk(P.mtx);
    if (P.rank < 0) {
        int init = 0;
        MPI_Initialized(&init);
        if (init) MPI_Comm_rank(MPI_COMM_WORLD, &P.rank);
        const char* r = std::getenv("DENDRO_XCHG_BUF_POOL_REPORT");
        if (r && r[0] == '1') std::atexit([] { poolReport(); });
    }
    long best  = -1;
    long large = -1;
    for (size_t i = 0; i < P.free.size(); i++) {
        if (P.free[i].second >= bytes &&
            (best < 0 || P.free[i].second < P.free[best].second))
            best = (long)i;
        if (large < 0 || P.free[i].second > P.free[large].second)
            large = (long)i;
    }
    void* p    = nullptr;
    size_t cap = 0;
    if (best >= 0) {
        p   = P.free[best].first;
        cap = P.free[best].second;
        P.free.erase(P.free.begin() + best);
    } else {
        const size_t pg = 4096;
        cap = ((bytes + bytes / 4 + pg) / pg) * pg;
        size_t dropped = (large >= 0) ? P.free[large].second : 0;
        if (P.total - dropped + cap > poolCapBytes()) {
            P.fallbacks++;
            return malloc(bytes);  // over the cap: untracked plain buffer
        }
        // nothing fits: drop the largest (too small) free block, grow
        if (large >= 0) {
            std::free(P.free[large].first);
            P.total -= dropped;
            P.free.erase(P.free.begin() + large);
        }
        if (posix_memalign(&p, pg, cap) != 0) return malloc(bytes);
        std::memset(p, 0, cap);
        P.total += cap;
        if (P.total > P.peak) P.peak = P.total;
    }
    P.used[p] = cap;
    if (poolCheck()) std::memset(p, 0xff, bytes);
    return p;
}
inline void release(void* p) {
    if (!p) return;
    if (!poolOn()) {
        std::free(p);
        return;
    }
    Pool& P = pool();
    std::lock_guard<std::mutex> lk(P.mtx);
    auto it = P.used.find(p);
    if (it == P.used.end()) {
        std::free(p);
        return;
    }
    P.free.emplace_back(p, it->second);
    P.used.erase(it);
}
}  // namespace xbuf

class AsyncExchangeContex {
   protected:
    /** pointer to the variable which perform the ghost exchange */
    void* m_uiBuffer  = NULL;

    /** pointer to the send buffer*/
    void* m_uiSendBuf = NULL;

    /** pointer to the send buffer*/
    void* m_uiRecvBuf = NULL;

    std::vector<MPI_Request*> m_uiRequests;

   public:
    /**@brief batched requests for send */
    std::vector<MPI_Request> m_send_req;

    /**@brief batched requests for recv */
    std::vector<MPI_Request> m_recv_req;

    AsyncExchangeContex() {};

    /**@brief creates an async ghost exchange contex*/
    AsyncExchangeContex(const void* var) {
        m_uiBuffer  = (void*)var;
        m_uiSendBuf = nullptr;
        m_uiRecvBuf = nullptr;
        m_uiRequests.clear();
        m_send_req.clear();
        m_recv_req.clear();
    }

    /**@brief : defaut destructor*/
    ~AsyncExchangeContex() {};

    /**@brief allocates send buffer for ghost exchange*/
    inline void allocateSendBuffer(size_t bytes) {
        m_uiSendBuf = xbuf::acquire(bytes);
    }

    /**@brief allocates recv buffer for ghost exchange*/
    inline void allocateRecvBuffer(size_t bytes) {
        m_uiRecvBuf = xbuf::acquire(bytes);
    }

    /**@brief allocates send buffer for ghost exchange*/
    inline void deAllocateSendBuffer() {
        xbuf::release(m_uiSendBuf);
        m_uiSendBuf = NULL;
    }

    /**@brief allocates recv buffer for ghost exchange*/
    inline void deAllocateRecvBuffer() {
        xbuf::release(m_uiRecvBuf);
        m_uiRecvBuf = NULL;
    }

    inline void* getSendBuffer() { return m_uiSendBuf; }
    inline void* getRecvBuffer() { return m_uiRecvBuf; }

    inline void getSendBuffer(void* ptr) { m_uiSendBuf = ptr; }
    inline void getRecvBuffer(void* ptr) { m_uiRecvBuf = ptr; }

    inline const void* getBuffer() { return m_uiBuffer; }

    inline std::vector<MPI_Request*>& getRequestList() { return m_uiRequests; }

    bool operator==(AsyncExchangeContex other) const {
        return (m_uiBuffer == other.m_uiBuffer);
    }
};

}  // namespace ot

#endif  // DENDRO_5_0_UPDATECTX_H
