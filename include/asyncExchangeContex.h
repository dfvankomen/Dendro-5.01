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

#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mpi.h"

namespace ot {

/**@brief Process-wide pool for ghost-exchange buffers (DENDRO_XCHG_BUF_POOL=1).
 * Buffers are kept across exchanges and meshes, so a new mesh's first exchange
 * reuses warm, already-faulted (and transport-registered) memory instead of
 * fresh malloc pages. Blocks grow with headroom and are prefaulted when created;
 * a free block too small for a request is released, so the block count stays at
 * the peak number of buffers in flight. DENDRO_XCHG_BUF_POOL_CHECK=1 poisons
 * every handed-out buffer with NaN bytes to prove no stale content is read. */
namespace xbuf {
inline bool poolOn() {
    static const bool on = [] {
        const char* e = std::getenv("DENDRO_XCHG_BUF_POOL");
        return e && e[0] == '1';
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
struct Pool {
    std::vector<std::pair<void*, size_t>> free;
    std::unordered_map<void*, size_t> used;
};
inline Pool& pool() {
    static Pool p;
    return p;
}
inline void* acquire(size_t bytes) {
    if (!poolOn()) return malloc(bytes);
    Pool& P    = pool();
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
        // nothing fits: drop the largest (too small) free block, grow
        if (large >= 0) {
            std::free(P.free[large].first);
            P.free.erase(P.free.begin() + large);
        }
        const size_t pg = 4096;
        cap = ((bytes + bytes / 4 + pg) / pg) * pg;
        if (posix_memalign(&p, pg, cap) != 0) return malloc(bytes);
        std::memset(p, 0, cap);
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
