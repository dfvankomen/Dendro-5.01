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

#include "mpi.h"
#include <vector>

namespace ot {

    class AsyncExchangeContex {

        protected :
            /** pointer to the variable which perform the ghost exchange */
            void* m_uiBuffer=NULL;

            /** pointer to the send buffer*/
            void* m_uiSendBuf=NULL;

            /** pointer to the send buffer*/
            void* m_uiRecvBuf=NULL;

            std::vector<MPI_Request*>  m_uiRequests;

#ifdef DENDRO_ENABLE_GHOST_COMPRESSION
            /** list of the compressed send buffers, compress algos allocate their own mem */
            std::vector<unsigned char*> m_uiCompSendBuf;

            /** pointer to the compressed receive buffer */
            void* m_uiCompRecvBuf=NULL;

        private:
            /**Send counts for the compressed data*/
            std::vector<unsigned int> m_uiSendCompressCounts;

            /**Receive counts for the compressed data*/
            std::vector<unsigned int> m_uiRecvCompressCounts;

#endif

        public:
            /**@brief batched requests for send */
            std::vector<MPI_Request> m_send_req;
            
            /**@brief batched requests for recv */
            std::vector<MPI_Request> m_recv_req;

            AsyncExchangeContex(){};
            
            /**@brief creates an async ghost exchange contex*/
            AsyncExchangeContex(const void* var)
            {
                m_uiBuffer=(void*)var;
                m_uiSendBuf=nullptr;
                m_uiRecvBuf=nullptr;
                m_uiRequests.clear();
                m_send_req.clear();
                m_recv_req.clear();
            }

            /**@brief : defaut destructor*/
            ~AsyncExchangeContex() {};

            /**@brief allocates send buffer for ghost exchange*/
            inline void allocateSendBuffer(size_t bytes)
            {
                m_uiSendBuf=malloc(bytes);
            }

            /**@brief allocates recv buffer for ghost exchange*/
            inline void allocateRecvBuffer(size_t bytes)
            {
                m_uiRecvBuf=malloc(bytes);
            }

            /**@brief allocates send buffer for ghost exchange*/
            inline void deAllocateSendBuffer()
            {
                free(m_uiSendBuf);
                m_uiSendBuf=NULL;
            }

            /**@brief allocates recv buffer for ghost exchange*/
            inline void deAllocateRecvBuffer()
            {
                free(m_uiRecvBuf);
                m_uiRecvBuf=NULL;
            }

            inline void* getSendBuffer() { return m_uiSendBuf;}
            inline void* getRecvBuffer() { return m_uiRecvBuf;}

            inline void getSendBuffer(void* ptr) { m_uiSendBuf=ptr;}
            inline void getRecvBuffer(void* ptr) { m_uiRecvBuf=ptr;}

            inline const void* getBuffer() {return m_uiBuffer;}

            inline std::vector<MPI_Request*>& getRequestList(){ return m_uiRequests;}

            bool operator== (AsyncExchangeContex other) const{
                return( m_uiBuffer == other.m_uiBuffer );
            }

#ifdef DENDRO_ENABLE_GHOST_COMPRESSION

            inline void allocateCompressRecvBuffer(size_t bytes)
            {
                m_uiCompRecvBuf=malloc(bytes);
            }

            inline void allocateCompressSendBuffers(unsigned int num)
            {
                m_uiCompSendBuf.resize(num, NULL);
            }

            inline std::vector<unsigned char*> &getCompressSendBuffers()
            {
                return m_uiCompSendBuf;
            }

            inline void clearCompressSendBuffers()
            {
                deleteCompressSendBuffers();
                m_uiCompSendBuf.clear();
            }

            inline void deleteCompressSendBuffers()
            {
                for (int ii = 0; ii < m_uiCompSendBuf.size(); ii++) {
                    delete[] m_uiCompSendBuf[ii];
                    m_uiCompSendBuf[ii] = NULL;
                }
            }

            inline void deallocateCompressRecvBuffer()
            {
                free(m_uiCompRecvBuf);
                m_uiCompRecvBuf=NULL;
            }

            inline void reallocateCompressRecvBuffer(size_t bytes)
            {

                void * temp_ptr = realloc(m_uiCompRecvBuf, bytes);

                if (temp_ptr == NULL) {
                    printf("\nReallocation of compressed recieve buffer failed (see asyncExchangeContext.h and mesh.tcc)!\nEXITING!\n");
                    free(m_uiCompRecvBuf);
                    exit(0);
                } else {
                    m_uiCompRecvBuf = temp_ptr;
                    temp_ptr = NULL;
                }
            }

            inline void* getCompressRecvBuffer() { return m_uiCompRecvBuf;}

            inline void getCompressRecvBuffer(void* ptr) { m_uiCompRecvBuf=ptr;}

            /**@brief returns the send compress counts*/
            inline std::vector<unsigned int> &getSendCompressCounts() { return m_uiSendCompressCounts; }
            /**@brief returns the receive compress counts*/
            inline std::vector<unsigned int> &getReceiveCompressCounts() { return m_uiRecvCompressCounts; }

#endif
               

    };

} //end namespace

#endif //DENDRO_5_0_UPDATECTX_H
