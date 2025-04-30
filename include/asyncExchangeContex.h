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

#include <iostream>
#include <ostream>
#include <vector>

#include "mpi.h"

namespace ot {

/** @brief: type to send data through */
enum CTXSendType { CTX_FLOAT = 0, CTX_DOUBLE, CTX_INT };

static const char* CTX_SEND_TYPE_NAMES[] = {"CTX_FLOAT", "CTX_DOUBLE",
                                            "CTX_INT"};

inline std::ostream& operator<<(std::ostream& out, const CTXSendType t) {
    return out << "<CTXSendType: " << CTX_SEND_TYPE_NAMES[t] << ">";
}

constexpr size_t getCTXSendTypeSize(CTXSendType dt) {
    return (dt == CTX_FLOAT)    ? sizeof(float)
           : (dt == CTX_DOUBLE) ? sizeof(double)
           : (dt == CTX_INT)    ? sizeof(int)
                                : 0;
}

template <typename T>
struct SendTypeHelper {
    static constexpr T value = {};
};

template <>
struct SendTypeHelper<float> {
    static constexpr CTXSendType value = CTXSendType::CTX_FLOAT;
};

template <>
struct SendTypeHelper<double> {
    static constexpr CTXSendType value = CTXSendType::CTX_DOUBLE;
};

// TODO: send support for ints or other floating types

class AsyncExchangeContex {
   protected:
    /** pointer to the variable which perform the ghost exchange */
    void* m_uiBuffer  = NULL;

    /** pointer to the send buffer*/
    void* m_uiSendBuf = NULL;

    /** pointer to the send buffer*/
    void* m_uiRecvBuf = NULL;

    std::vector<MPI_Request*> m_uiRequests;

#ifdef DENDRO_ENABLE_GHOST_COMPRESSION
    /** list of the compressed send buffers, compress algos allocate their own
     * mem */
    std::vector<unsigned char*> m_uiCompSendBufs;

    void* m_uiCompSendBuf = NULL;

    /** pointer to the compressed receive buffer */
    void* m_uiCompRecvBuf = NULL;

   private:
    /**Send counts for the compressed data*/
    std::vector<unsigned int> m_uiSendCompressCounts;

    /**Receive counts for the compressed data*/
    std::vector<unsigned int> m_uiRecvCompressCounts;

    /** Send offsets for the compressed data */
    std::vector<unsigned int> m_uiSendCompressOffsets;

    /** Receive offsets for the compressed data */
    std::vector<unsigned int> m_uiRecvCompressOffsets;

    MPI_Request m_alltoallRequest;

#endif

    CTXSendType m_ctxSendType = CTXSendType::CTX_DOUBLE;

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
        m_uiSendBuf = malloc(bytes);
    }

    /**@brief allocates recv buffer for ghost exchange*/
    inline void allocateRecvBuffer(size_t bytes) {
        m_uiRecvBuf = malloc(bytes);
    }

    /**@brief allocates send buffer for ghost exchange*/
    inline void deAllocateSendBuffer() {
        if (m_uiSendBuf != NULL) {
            free(m_uiSendBuf);
            m_uiSendBuf = NULL;
        } else {
            std::cerr << "WARNING: ATTEMPTED TO DEALLOCATE A NULL SEND BUFFER"
                      << std::endl;
        }
    }

    /**@brief allocates recv buffer for ghost exchange*/
    inline void deAllocateRecvBuffer() {
        if (m_uiRecvBuf != NULL) {
            free(m_uiRecvBuf);
            m_uiRecvBuf = NULL;
        } else {
            std::cerr << "WARNING: ATTEMPTED TO DEALLOCATE A NULL RECV BUFFER"
                      << std::endl;
        }
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

#ifdef DENDRO_ENABLE_GHOST_COMPRESSION

    inline MPI_Request* getAllToAllRequest() { return &m_alltoallRequest; }

    inline void allocateCompressRecvBuffer(size_t bytes) {
        m_uiCompRecvBuf = malloc(bytes);
    }

    inline void allocateCompressSendBuffers(unsigned int num) {
        m_uiCompSendBufs.resize(num, NULL);
    }

    inline void allocateCompressSendBuffer(size_t bytes) {
        m_uiCompSendBuf = malloc(bytes);
    }

    inline std::vector<unsigned char*>& getCompressSendBuffers() {
        return m_uiCompSendBufs;
    }

    inline void clearCompressSendBuffers() {
        deleteCompressSendBuffers();
        m_uiCompSendBufs.clear();
    }

    inline void deleteCompressSendBuffers() {
        for (int ii = 0; ii < m_uiCompSendBufs.size(); ii++) {
            delete[] m_uiCompSendBufs[ii];
            m_uiCompSendBufs[ii] = NULL;
        }
    }

    inline void deallocateCompressRecvBuffer() {
        free(m_uiCompRecvBuf);
        m_uiCompRecvBuf = NULL;
    }

    inline void deallocateCompressSendBuffer() {
        free(m_uiCompSendBuf);
        m_uiCompRecvBuf = NULL;
    }

    inline void reallocateCompressSendBuffer(size_t bytes) {
        void* temp_ptr = realloc(m_uiCompSendBuf, bytes);

        if (temp_ptr == NULL) {
            printf(
                "\nReallocation of compressed recieve buffer failed (see "
                "asyncExchangeContext.h and mesh.tcc)!\nEXITING!\n");
            free(m_uiCompSendBuf);
            exit(0);
        } else {
            m_uiCompSendBuf = temp_ptr;
            temp_ptr        = NULL;
        }
    }

    inline void reallocateCompressRecvBuffer(size_t bytes) {
        void* temp_ptr = realloc(m_uiCompRecvBuf, bytes);

        if (temp_ptr == NULL) {
            printf(
                "\nReallocation of compressed recieve buffer failed (see "
                "asyncExchangeContext.h and mesh.tcc)!\nEXITING!\n");
            free(m_uiCompRecvBuf);
            exit(0);
        } else {
            m_uiCompRecvBuf = temp_ptr;
            temp_ptr        = NULL;
        }
    }

    inline void* getCompressSendBuffer() { return m_uiCompSendBuf; }

    inline void* getCompressRecvBuffer() { return m_uiCompRecvBuf; }

    inline void getCompressSendBuffer(void* ptr) { m_uiCompSendBuf = ptr; }

    inline void getCompressRecvBuffer(void* ptr) { m_uiCompRecvBuf = ptr; }

    /**@brief returns the send compress counts*/
    inline std::vector<unsigned int>& getSendCompressCounts() {
        return m_uiSendCompressCounts;
    }
    /**@brief returns the receive compress counts*/
    inline std::vector<unsigned int>& getReceiveCompressCounts() {
        return m_uiRecvCompressCounts;
    }

    /**@brief returns the send compress offsets */
    inline std::vector<unsigned int>& getSendCompressOffsets() {
        return m_uiSendCompressOffsets;
    }
    /**@brief returns the receive compress counts*/
    inline std::vector<unsigned int>& getReceiveCompressOffsets() {
        return m_uiRecvCompressOffsets;
    }

#endif

    void setCommDtype(CTXSendType inType) { m_ctxSendType = inType; }

    inline CTXSendType getCommDtype() const { return m_ctxSendType; }
};

}  // namespace ot

#endif  // DENDRO_5_0_UPDATECTX_H
