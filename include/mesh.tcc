//
// Created by milinda on 9/22/17.
/**
 *@author Milinda Fernando
 *School of Computing, University of Utah
 *@brief Contains templated functions in the mesh class.
 * (code moved from mesh.h)
 */
//

namespace ot {
template <typename T>
T* Mesh::createVector() const {
    if (!m_uiIsActive) return NULL;

    T* vec = NULL;

    try {
        vec = new T[m_uiNumActualNodes];

    } catch (const std::bad_alloc& e) {
        std::cout << " rank: " << m_uiActiveRank << " func: " << __func__
                  << " bad allocation error " << std::endl;
        MPI_Abort(m_uiCommGlobal, 0);
    }

    return vec;
}

template <typename T>
T* Mesh::createCGVector(T initVal, unsigned int dof) const {
    if (!m_uiIsActive) return NULL;

    T* vec = NULL;
    try {
        vec = new T[m_uiNumActualNodes * dof];

    } catch (const std::bad_alloc& e) {
        std::cout << " rank: " << m_uiActiveRank << " func: " << __func__
                  << " bad allocation error " << std::endl;
        MPI_Abort(m_uiCommGlobal, 0);
    }

    for (unsigned int i = 0; i < m_uiNumActualNodes * dof; i++)
        vec[i] = initVal;

    return vec;
}

template <typename T>
T* Mesh::createCGVector(std::function<void(T, T, T, T*)> func,
                        unsigned int dof) const {
    if (!m_uiIsActive) return NULL;

    T* vec = NULL;
    try {
        vec = new T[m_uiNumActualNodes * dof];

    } catch (const std::bad_alloc& e) {
        std::cout << " rank: " << m_uiActiveRank << " func: " << __func__
                  << " bad allocation error " << std::endl;
        MPI_Abort(m_uiCommGlobal, 0);
    }

    for (unsigned int i = 0; i < m_uiNumActualNodes * dof; i++) vec[i] = (T)0;

    // initialize the vector to the function.
    T* fvar                    = new T[dof];

    const ot::TreeNode* pNodes = &(*(m_uiAllElements.begin()));
    for (unsigned int elem = m_uiElementLocalBegin; elem < m_uiElementLocalEnd;
         elem++) {
        for (unsigned int k = 0; k < (m_uiElementOrder + 1); k++)
            for (unsigned int j = 0; j < (m_uiElementOrder + 1); j++)
                for (unsigned int i = 0; i < (m_uiElementOrder + 1); i++) {
                    const unsigned int nodeLookUp_CG =
                        m_uiE2NMapping_CG[elem * m_uiNpE +
                                          k * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          j * (m_uiElementOrder + 1) + i];
                    if (nodeLookUp_CG >= m_uiNodeLocalBegin &&
                        nodeLookUp_CG < m_uiNodeLocalEnd) {
                        unsigned int ownerID, ii_x, jj_y, kk_z;
                        const unsigned int nodeLookUp_DG =
                            m_uiE2NMapping_DG[elem * m_uiNpE +
                                              k * (m_uiElementOrder + 1) *
                                                  (m_uiElementOrder + 1) +
                                              j * (m_uiElementOrder + 1) + i];
                        dg2eijk(nodeLookUp_DG, ownerID, ii_x, jj_y, kk_z);
                        const unsigned int len =
                            1u << (m_uiMaxDepth - pNodes[ownerID].getLevel());

                        const double x =
                            pNodes[ownerID].getX() +
                            ii_x * (len / ((double)m_uiElementOrder));
                        const double y =
                            pNodes[ownerID].getY() +
                            jj_y * (len / ((double)m_uiElementOrder));
                        const double z =
                            pNodes[ownerID].getZ() +
                            kk_z * (len / ((double)m_uiElementOrder));

                        Point physical_coord;
                        this->octCoordToDomainCoord(Point(x, y, z),
                                                    physical_coord);
                        func(physical_coord.x(), physical_coord.y(),
                             physical_coord.z(), fvar);

                        for (unsigned int v = 0; v < dof; v++)
                            vec[v * m_uiNumActualNodes + nodeLookUp_CG] =
                                fvar[v];
                    }
                }
    }

    delete[] fvar;
    return vec;
}

template <typename T>
T* Mesh::createElementVector(T initVal, unsigned int dof) const {
    if (!m_uiIsActive) return NULL;

    T* vec = NULL;

    try {
        vec = new T[m_uiNumTotalElements * dof];

    } catch (const std::bad_alloc& e) {
        std::cout << " rank: " << m_uiActiveRank << " func: " << __func__
                  << " bad allocation error " << std::endl;
        MPI_Abort(m_uiCommGlobal, 0);
    }

    for (unsigned int i = 0; i < m_uiNumTotalElements * dof; i++)
        vec[i] = initVal;

    return vec;
}

template <typename T>
T* Mesh::createDGVector(T initVal, unsigned int dof) const {
    if (!m_uiIsActive) return NULL;

    T* vec = NULL;
    try {
        vec = new T[m_uiNumTotalElements * m_uiNpE * dof];

    } catch (const std::bad_alloc& e) {
        std::cout << " rank: " << m_uiActiveRank << " func: " << __func__
                  << " bad allocation error " << std::endl;
        MPI_Abort(m_uiCommGlobal, 0);
    }

    for (unsigned int i = 0; i < m_uiNumTotalElements * m_uiNpE * dof; i++)
        vec[i] = initVal;

    return vec;
}

template <typename T>
T* Mesh::createDGVector(std::function<void(T, T, T, T*)> func,
                        unsigned int dof) const {
    if (!m_uiIsActive) return NULL;

    T* vec = NULL;
    try {
        vec = new T[m_uiNumTotalElements * m_uiNpE * dof];

    } catch (const std::bad_alloc& e) {
        std::cout << " rank: " << m_uiActiveRank << " func: " << __func__
                  << " bad allocation error " << std::endl;
        MPI_Abort(m_uiCommGlobal, 0);
    }

    for (unsigned int i = 0; i < m_uiNumTotalElements * m_uiNpE * dof; i++)
        vec[i] = (T)0;

    T* fvar                    = new T[dof];
    const ot::TreeNode* pNodes = &(*(m_uiAllElements.begin()));
    for (unsigned int elem = m_uiElementLocalBegin; elem < m_uiElementLocalEnd;
         elem++) {
        for (unsigned int k = 0; k < (m_uiElementOrder + 1); k++)
            for (unsigned int j = 0; j < (m_uiElementOrder + 1); j++)
                for (unsigned int i = 0; i < (m_uiElementOrder + 1); i++) {
                    const unsigned int dg_index =
                        elem * m_uiNpE +
                        k * (m_uiElementOrder + 1) * (m_uiElementOrder + 1) +
                        j * (m_uiElementOrder + 1) + i;
                    const unsigned int len =
                        1u << (m_uiMaxDepth - pNodes[elem].getLevel());

                    const double x = pNodes[elem].getX() +
                                     i * (len / ((double)m_uiElementOrder));
                    const double y = pNodes[elem].getY() +
                                     j * (len / ((double)m_uiElementOrder));
                    const double z = pNodes[elem].getZ() +
                                     k * (len / ((double)m_uiElementOrder));

                    Point physical_coord;
                    this->octCoordToDomainCoord(Point(x, y, z), physical_coord);
                    func(physical_coord.x(), physical_coord.y(),
                         physical_coord.z(), fvar);

                    for (unsigned int v = 0; v < dof; v++)
                        vec[v * m_uiNumTotalElements * m_uiNpE + dg_index] =
                            fvar[v];
                }
    }

    delete[] fvar;
    return vec;
}

template <typename T>
T* Mesh::createVector(const T initValue) const {
    if (!m_uiIsActive) return NULL;

    T* vec = NULL;

    try {
        vec = new T[m_uiNumActualNodes];
    } catch (const std::bad_alloc& e) {
        std::cout << " rank: " << m_uiActiveRank << " func: " << __func__
                  << " bad allocation error " << std::endl;
        MPI_Abort(m_uiCommGlobal, 0);
    }

    for (unsigned int k = 0; k < m_uiNumActualNodes; k++) vec[k] = initValue;

    return vec;
}

template <typename T>
T* Mesh::createVector(std::function<T(T, T, T)> func) const {
    if (!m_uiIsActive) return NULL;

    T* vec = NULL;

    try {
        vec = new T[m_uiNumActualNodes];

    } catch (const std::bad_alloc& e) {
        std::cout << " rank: " << m_uiActiveRank << " func: " << __func__
                  << " bad allocation error " << std::endl;
        MPI_Abort(m_uiCommGlobal, 0);
    }

    unsigned int nodeLookUp_CG;
    unsigned int nodeLookUp_DG;
    unsigned int len;
    double x, y, z;
    const ot::TreeNode* pNodes = &(*(m_uiAllElements.begin()));
    unsigned int ownerID, ii_x, jj_y, kk_z;

    for (unsigned int elem = m_uiElementLocalBegin; elem < m_uiElementLocalEnd;
         elem++) {
        for (unsigned int k = 0; k < (m_uiElementOrder + 1); k++)
            for (unsigned int j = 0; j < (m_uiElementOrder + 1); j++)
                for (unsigned int i = 0; i < (m_uiElementOrder + 1); i++) {
                    nodeLookUp_CG =
                        m_uiE2NMapping_CG[elem * m_uiNpE +
                                          k * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          j * (m_uiElementOrder + 1) + i];
                    if (nodeLookUp_CG >= m_uiNodeLocalBegin &&
                        nodeLookUp_CG < m_uiNodeLocalEnd) {
                        nodeLookUp_DG =
                            m_uiE2NMapping_DG[elem * m_uiNpE +
                                              k * (m_uiElementOrder + 1) *
                                                  (m_uiElementOrder + 1) +
                                              j * (m_uiElementOrder + 1) + i];
                        dg2eijk(nodeLookUp_DG, ownerID, ii_x, jj_y, kk_z);
                        len = 1u << (m_uiMaxDepth - pNodes[ownerID].getLevel());
                        x   = pNodes[ownerID].getX() +
                              ii_x * (len / ((double)m_uiElementOrder));
                        y   = pNodes[ownerID].getY() +
                              jj_y * (len / ((double)m_uiElementOrder));
                        z   = pNodes[ownerID].getZ() +
                              kk_z * (len / ((double)m_uiElementOrder));
                        vec[nodeLookUp_CG] = func(x, y, z);
                    }
                }
    }

    return vec;
}

template <typename T>
void Mesh::createVector(std::vector<T>& vec) const {
    if (!m_uiIsActive) {
        vec.clear();
        return;
    }
    vec.resize(m_uiNumActualNodes);
}

template <typename T>
void Mesh::createVector(std::vector<T>& vec, const T initValue) const {
    if (!m_uiIsActive) {
        vec.clear();
        return;
    }
    vec.resize(m_uiNumActualNodes, initValue);
}

template <typename T>
void Mesh::createVector(std::vector<T>& vec,
                        std::function<T(T, T, T)> func) const {
    if (!m_uiIsActive) {
        vec.clear();
        return;
    }
    vec.clear();
    vec.resize(m_uiNumActualNodes, 0);
    unsigned int nodeLookUp_CG;
    unsigned int nodeLookUp_DG;
    unsigned int len;
    double x, y, z;
    const ot::TreeNode* pNodes = &(*(m_uiAllElements.begin()));
    unsigned int ownerID, ii_x, jj_y, kk_z;

    for (unsigned int elem = m_uiElementLocalBegin; elem < m_uiElementLocalEnd;
         elem++) {
        for (unsigned int k = 0; k < (m_uiElementOrder + 1); k++)
            for (unsigned int j = 0; j < (m_uiElementOrder + 1); j++)
                for (unsigned int i = 0; i < (m_uiElementOrder + 1); i++) {
                    nodeLookUp_CG =
                        m_uiE2NMapping_CG[elem * m_uiNpE +
                                          k * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          j * (m_uiElementOrder + 1) + i];
                    if (nodeLookUp_CG >= m_uiNodeLocalBegin &&
                        nodeLookUp_CG < m_uiNodeLocalEnd) {
                        nodeLookUp_DG =
                            m_uiE2NMapping_DG[elem * m_uiNpE +
                                              k * (m_uiElementOrder + 1) *
                                                  (m_uiElementOrder + 1) +
                                              j * (m_uiElementOrder + 1) + i];
                        dg2eijk(nodeLookUp_DG, ownerID, ii_x, jj_y, kk_z);
                        len = 1u << (m_uiMaxDepth - pNodes[ownerID].getLevel());
                        x   = pNodes[ownerID].getX() +
                              ii_x * (len / ((double)m_uiElementOrder));
                        y   = pNodes[ownerID].getY() +
                              jj_y * (len / ((double)m_uiElementOrder));
                        z   = pNodes[ownerID].getZ() +
                              kk_z * (len / ((double)m_uiElementOrder));
                        vec[nodeLookUp_CG] = func(x, y, z);
                    }
                }
    }
}

template <typename T>
void Mesh::createUnZippedVector(std::vector<T>& uvec) const {
    if (!m_uiIsActive) {
        uvec.clear();
        return;
    }
    uvec.resize(m_uiUnZippedVecSz);
}

template <typename T>
void Mesh::createUnZippedVector(std::vector<T>& uvec, const T initValue) const {
    if (!m_uiIsActive) {
        uvec.clear();
        return;
    }
    uvec.resize(m_uiUnZippedVecSz, initValue);
}

template <typename T>
T* Mesh::createUnZippedVector(unsigned int dof) const {
    if (!m_uiIsActive) return NULL;

    T* uvec = NULL;

    try {
        uvec = new T[dof * m_uiUnZippedVecSz];

    } catch (const std::bad_alloc& e) {
        std::cout << " rank: " << m_uiActiveRank << " func: " << __func__
                  << " bad allocation error " << std::endl;
        MPI_Abort(m_uiCommGlobal, 0);
    }

    return uvec;
}

template <typename T>
T* Mesh::createUnZippedVector(const T initValue, unsigned int dof) const {
    if (!m_uiIsActive) return NULL;

    T* uvec = NULL;
    try {
        uvec = new T[dof * m_uiUnZippedVecSz];

    } catch (const std::bad_alloc& e) {
        std::cout << " rank: " << m_uiActiveRank << " func: " << __func__
                  << " bad allocation error " << std::endl;
        MPI_Abort(m_uiCommGlobal, 0);
    }

    for (unsigned int k = 0; k < (dof * m_uiUnZippedVecSz); k++)
        uvec[k] = initValue;

    return uvec;
}

template <typename T>
void Mesh::CG2DGVec(T* cg_vec, T* dg_vec, bool gsynced, unsigned int dof) {
    if (!m_uiIsActive) return;

    if (!gsynced) {
        this->readFromGhostBegin(cg_vec, dof);
        this->readFromGhostEnd(cg_vec, dof);
    }

    const unsigned int vsz_dg = m_uiNumTotalElements * m_uiNpE;
    const unsigned int vsz_cg = m_uiNumActualNodes;

    for (unsigned int v = 0; v < dof; v++)
        for (unsigned int ele = m_uiElementLocalBegin;
             ele < m_uiElementLocalEnd; ele++)
            this->getElementNodalValues(
                cg_vec + v * vsz_cg, dg_vec + v * vsz_dg + ele * m_uiNpE, ele);

    return;
}

template <typename T>
void Mesh::DG2CGVec(const T* dg_vec, T* cg_vec, unsigned int dof) const {
    if (!m_uiIsActive) return;

    const unsigned int vsz_dg = m_uiNumTotalElements * m_uiNpE;
    const unsigned int vsz_cg = m_uiNumActualNodes;

    bool isHanging;
    unsigned int cnum;
    for (unsigned int v = 0; v < dof; v++) {
        for (unsigned int ele = m_uiElementLocalBegin;
             ele < m_uiElementLocalEnd; ele++) {
            for (unsigned int k = 0; k < (m_uiElementOrder + 1); k++)
                for (unsigned int j = 0; j < (m_uiElementOrder + 1); j++)
                    for (unsigned int i = 0; i < (m_uiElementOrder + 1); i++) {
                        isHanging = isNodeHanging(ele, i, j, k);
                        if (!isHanging)
                            cg_vec[v * vsz_cg +
                                   m_uiE2NMapping_CG
                                       [ele * m_uiNpE +
                                        k * (m_uiElementOrder + 1) *
                                            (m_uiElementOrder + 1) +
                                        j * (m_uiElementOrder + 1) + i]] =
                                dg_vec[v * vsz_dg + ele * m_uiNpE +
                                       k * (m_uiElementOrder + 1) *
                                           (m_uiElementOrder + 1) +
                                       j * (m_uiElementOrder + 1) + i];
                        else {
                            cnum = m_uiAllElements[(ele)].getMortonIndex();
                            const unsigned int iix =
                                m_uiElementOrder * (int)(cnum & 1u) + i;
                            const unsigned int jjy =
                                m_uiElementOrder * (int)((cnum & 2u) >> 1u) + j;
                            const unsigned int kkz =
                                m_uiElementOrder * (int)((cnum & 4u) >> 2u) + k;

                            if ((iix % 2 == 0) && (jjy % 2 == 0) &&
                                (kkz % 2 == 0))
                                cg_vec
                                    [v * vsz_cg +
                                     m_uiE2NMapping_CG
                                         [ele * m_uiNpE +
                                          (kkz >> 1u) * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          (jjy >> 1u) * (m_uiElementOrder + 1) +
                                          (iix >> 1u)]] =
                                        dg_vec[v * vsz_dg + ele * m_uiNpE +
                                               k * (m_uiElementOrder + 1) *
                                                   (m_uiElementOrder + 1) +
                                               j * (m_uiElementOrder + 1) + i];
                        }
                    }
        }
    }

    return;
}

template <typename T>
void Mesh::DG2CGVec(const T* dg_vec, T*& cg_vec, bool isAllocated,
                    const unsigned int* eleIDs, unsigned int nEle,
                    unsigned int dof) const {
    if (!m_uiIsActive) return;

    if (!isAllocated) cg_vec = this->createCGVector((T)0, dof);

    const unsigned int vsz_dg = m_uiNumTotalElements * m_uiNpE;
    const unsigned int vsz_cg = m_uiNumActualNodes;

    bool isHanging;
    unsigned int cnum;
    for (unsigned int v = 0; v < dof; v++) {
        for (unsigned int i = 0; i < nEle; i++) {
            const unsigned int ele = eleIDs[i];
            assert(ele < m_uiAllElements.size());

            for (unsigned int k = 0; k < (m_uiElementOrder + 1); k++)
                for (unsigned int j = 0; j < (m_uiElementOrder + 1); j++)
                    for (unsigned int i = 0; i < (m_uiElementOrder + 1); i++) {
                        isHanging = isNodeHanging(ele, i, j, k);
                        if (!isHanging)
                            cg_vec[v * vsz_cg +
                                   m_uiE2NMapping_CG
                                       [ele * m_uiNpE +
                                        k * (m_uiElementOrder + 1) *
                                            (m_uiElementOrder + 1) +
                                        j * (m_uiElementOrder + 1) + i]] =
                                dg_vec[v * vsz_dg + ele * m_uiNpE +
                                       k * (m_uiElementOrder + 1) *
                                           (m_uiElementOrder + 1) +
                                       j * (m_uiElementOrder + 1) + i];
                        else {
                            cnum = m_uiAllElements[(ele)].getMortonIndex();
                            const unsigned int iix =
                                m_uiElementOrder * (int)(cnum & 1u) + i;
                            const unsigned int jjy =
                                m_uiElementOrder * (int)((cnum & 2u) >> 1u) + j;
                            const unsigned int kkz =
                                m_uiElementOrder * (int)((cnum & 4u) >> 2u) + k;

                            if ((iix % 2 == 0) && (jjy % 2 == 0) &&
                                (kkz % 2 == 0))
                                cg_vec
                                    [v * vsz_cg +
                                     m_uiE2NMapping_CG
                                         [ele * m_uiNpE +
                                          (kkz >> 1u) * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          (jjy >> 1u) * (m_uiElementOrder + 1) +
                                          (iix >> 1u)]] =
                                        dg_vec[v * vsz_dg + ele * m_uiNpE +
                                               k * (m_uiElementOrder + 1) *
                                                   (m_uiElementOrder + 1) +
                                               j * (m_uiElementOrder + 1) + i];
                        }
                    }
        }
    }
}

template <typename T>
void Mesh::performGhostExchange(std::vector<T>& vec) {
    if ((m_uiActiveNpes == 1) || (!m_uiIsActive)) return;

    for (unsigned int p = 0; p < m_uiActiveNpes; p++) {
        for (unsigned int k = m_uiSendNodeOffset[p];
             k < (m_uiSendNodeOffset[p] + m_uiSendNodeCount[p]); k++) {
            m_uiSendBufferNodes[k] = (T)vec[m_uiScatterMapActualNodeSend[k]];
        }
    }

#ifdef ALLTOALL_SPARSE
    par::Mpi_Alltoallv_sparse(&(*(m_uiSendBufferNodes.begin())),
                              (int*)(&(*(m_uiSendNodeCount.begin()))),
                              (int*)(&(*(m_uiSendNodeOffset.begin()))),
                              &(*(m_uiRecvBufferNodes.begin())),
                              (int*)(&(*(m_uiRecvNodeCount.begin()))),
                              (int*)(&(*(m_uiRecvNodeOffset.begin()))),
                              m_uiCommActive);
#else
    par::Mpi_Alltoallv(&(*(m_uiSendBufferNodes.begin())),
                       (int*)(&(*(m_uiSendNodeCount.begin()))),
                       (int*)(&(*(m_uiSendNodeOffset.begin()))),
                       &(*(m_uiRecvBufferNodes.begin())),
                       (int*)(&(*(m_uiRecvNodeCount.begin()))),
                       (int*)(&(*(m_uiRecvNodeOffset.begin()))),
                       m_uiCommActive);
#endif

    for (unsigned int p = 0; p < m_uiActiveNpes; p++) {
        for (unsigned int k = m_uiRecvNodeOffset[p];
             k < (m_uiRecvNodeOffset[p] + m_uiRecvNodeCount[p]); k++) {
            // if(fabs(vec[m_uiScatterMapActualNodeRecv[k]]-m_uiRecvBufferNodes[k])>1e-15)
            // std::cout<<"rank: "<<m_uiActiveRank<<" computed:
            // "<<vec[m_uiScatterMapActualNodeRecv[k]]<<" revieved:
            // "<<m_uiRecvBufferNodes[k]<<" recv: from : "<<p<<std::endl;
            vec[m_uiScatterMapActualNodeRecv[k]] = (T)m_uiRecvBufferNodes[k];
        }
    }
}

template <typename T>
void Mesh::performGhostExchange(T* vec) {
    if ((m_uiActiveNpes == 1) || (!m_uiIsActive)) return;

    for (unsigned int p = 0; p < m_uiActiveNpes; p++) {
        for (unsigned int k = m_uiSendNodeOffset[p];
             k < (m_uiSendNodeOffset[p] + m_uiSendNodeCount[p]); k++) {
            m_uiSendBufferNodes[k] = (T)vec[m_uiScatterMapActualNodeSend[k]];
        }
    }

#ifdef ALLTOALL_SPARSE
    par::Mpi_Alltoallv_sparse(&(*(m_uiSendBufferNodes.begin())),
                              (int*)(&(*(m_uiSendNodeCount.begin()))),
                              (int*)(&(*(m_uiSendNodeOffset.begin()))),
                              &(*(m_uiRecvBufferNodes.begin())),
                              (int*)(&(*(m_uiRecvNodeCount.begin()))),
                              (int*)(&(*(m_uiRecvNodeOffset.begin()))),
                              m_uiCommActive);
#else
    par::Mpi_Alltoallv(&(*(m_uiSendBufferNodes.begin())),
                       (int*)(&(*(m_uiSendNodeCount.begin()))),
                       (int*)(&(*(m_uiSendNodeOffset.begin()))),
                       &(*(m_uiRecvBufferNodes.begin())),
                       (int*)(&(*(m_uiRecvNodeCount.begin()))),
                       (int*)(&(*(m_uiRecvNodeOffset.begin()))),
                       m_uiCommActive);
#endif

    for (unsigned int p = 0; p < m_uiActiveNpes; p++) {
        for (unsigned int k = m_uiRecvNodeOffset[p];
             k < (m_uiRecvNodeOffset[p] + m_uiRecvNodeCount[p]); k++) {
            // if(/*fabs(vec[m_uiScatterMapActualNodeRecv[k]]-m_uiRecvBufferNodes[k])>1e-15*/
            // isnan(m_uiRecvBufferNodes[k])) std::cout<<"rank:
            // "<<m_uiActiveRank<<" computed:
            // "<<vec[m_uiScatterMapActualNodeRecv[k]]<<" revieved:
            // "<<m_uiRecvBufferNodes[k]<<" recv: from : "<<p<<std::endl;
            vec[m_uiScatterMapActualNodeRecv[k]] = (T)m_uiRecvBufferNodes[k];
        }
    }
}

template <typename T>
void Mesh::ghostExchangeStart(T* vec, T* sendNodeBuffer, T* recvNodeBuffer,
                              MPI_Request* send_reqs, MPI_Request* recv_reqs) {
    if ((m_uiActiveNpes == 1) || (!m_uiIsActive)) return;

    unsigned int proc_id;

    // active recv procs
    for (unsigned int recv_p = 0; recv_p < m_uiRecvProcList.size(); recv_p++) {
        proc_id           = m_uiRecvProcList[recv_p];
        recv_reqs[recv_p] = MPI_Request();
        par::Mpi_Irecv((recvNodeBuffer + m_uiRecvNodeOffset[proc_id]),
                       m_uiRecvNodeCount[proc_id], proc_id, 0, m_uiCommActive,
                       &recv_reqs[recv_p]);
    }

    for (unsigned int send_p = 0; send_p < m_uiSendProcList.size(); send_p++) {
        proc_id = m_uiSendProcList[send_p];
        for (unsigned int k = m_uiSendNodeOffset[proc_id];
             k < (m_uiSendNodeOffset[proc_id] + m_uiSendNodeCount[proc_id]);
             k++) {
            sendNodeBuffer[k] = (T)vec[m_uiScatterMapActualNodeSend[k]];
        }
    }
    // active send procs
    for (unsigned int send_p = 0; send_p < m_uiSendProcList.size(); send_p++) {
        proc_id           = m_uiSendProcList[send_p];
        send_reqs[send_p] = MPI_Request();
        par::Mpi_Isend((sendNodeBuffer + m_uiSendNodeOffset[proc_id]),
                       m_uiSendNodeCount[proc_id], proc_id, 0, m_uiCommActive,
                       &send_reqs[send_p]);
    }
}

template <typename T>
void Mesh::ghostExchangeRecvSync(T* vec, T* recvNodeBuffer,
                                 MPI_Request* recv_reqs, MPI_Status* recv_sts) {
    if ((m_uiActiveNpes == 1) || (!m_uiIsActive)) return;

    dendro::timer::t_unzip_async_comm.start();
    MPI_Waitall(m_uiRecvProcList.size(), recv_reqs, recv_sts);
    dendro::timer::t_unzip_async_comm.stop();

    unsigned int proc_id = 0;
    for (unsigned int recv_p = 0; recv_p < m_uiRecvProcList.size(); recv_p++) {
        proc_id = m_uiRecvProcList[recv_p];
        for (unsigned int k = m_uiRecvNodeOffset[proc_id];
             k < (m_uiRecvNodeOffset[proc_id] + m_uiRecvNodeCount[proc_id]);
             k++) {
            vec[m_uiScatterMapActualNodeRecv[k]] = (T)recvNodeBuffer[k];
        }
    }
}

template <typename T>
void Mesh::readFromGhostBegin(T* vec, unsigned int dof) {
    if (this->getMPICommSizeGlobal() == 1 || (!m_uiIsActive)) return;

    // send recv buffers.
    T* sendB = NULL;
    T* recvB = NULL;

    if (this->isActive()) {
        const std::vector<unsigned int>& nodeSendCount =
            this->getNodalSendCounts();
        const std::vector<unsigned int>& nodeSendOffset =
            this->getNodalSendOffsets();

        const std::vector<unsigned int>& nodeRecvCount =
            this->getNodalRecvCounts();
        const std::vector<unsigned int>& nodeRecvOffset =
            this->getNodalRecvOffsets();

        const std::vector<unsigned int>& sendProcList = this->getSendProcList();
        const std::vector<unsigned int>& recvProcList = this->getRecvProcList();

        const std::vector<unsigned int>& sendNodeSM   = this->getSendNodeSM();
        const std::vector<unsigned int>& recvNodeSM   = this->getRecvNodeSM();

        const unsigned int activeNpes                 = this->getMPICommSize();

        const unsigned int sendBSz =
            nodeSendOffset[activeNpes - 1] + nodeSendCount[activeNpes - 1];
        const unsigned int recvBSz =
            nodeRecvOffset[activeNpes - 1] + nodeRecvCount[activeNpes - 1];
        unsigned int proc_id;

        AsyncExchangeContex ctx(vec);
        MPI_Comm commActive = this->getMPICommunicator();

        if (recvBSz) {
            ctx.allocateRecvBuffer((sizeof(T) * recvBSz * dof));
            recvB = (T*)ctx.getRecvBuffer();

            // active recv procs -- post into the context's by-value request
            // vector (no per-exchange heap alloc; waited via MPI_Waitall).
            ctx.m_recv_req.resize(recvProcList.size());
            for (unsigned int recv_p = 0; recv_p < recvProcList.size();
                 recv_p++) {
                proc_id = recvProcList[recv_p];
                par::Mpi_Irecv((recvB + dof * nodeRecvOffset[proc_id]),
                               dof * nodeRecvCount[proc_id], proc_id,
                               m_uiCommTag, commActive, &ctx.m_recv_req[recv_p]);
            }
        }

        if (sendBSz) {
            ctx.allocateSendBuffer(sizeof(T) * dof * sendBSz);
            sendB = (T*)ctx.getSendBuffer();

            // threaded gather into the send buffer (disjoint writes per proc)
#ifdef DENDRO_HYBRID_OMP
#pragma omp parallel for private(proc_id)
#endif
            for (unsigned int send_p = 0; send_p < sendProcList.size();
                 send_p++) {
                proc_id = sendProcList[send_p];

                for (unsigned int var = 0; var < dof; var++) {
                    for (unsigned int k = nodeSendOffset[proc_id];
                         k < (nodeSendOffset[proc_id] + nodeSendCount[proc_id]);
                         k++) {
                        sendB[dof * (nodeSendOffset[proc_id]) +
                              (var * nodeSendCount[proc_id]) +
                              (k - nodeSendOffset[proc_id])] =
                            (vec + var * m_uiNumActualNodes)[sendNodeSM[k]];
                    }
                }
            }

            // active send procs -- post into the context's by-value request
            // vector (no per-exchange heap alloc; waited via MPI_Waitall).
            ctx.m_send_req.resize(sendProcList.size());
            for (unsigned int send_p = 0; send_p < sendProcList.size();
                 send_p++) {
                proc_id = sendProcList[send_p];
                par::Mpi_Isend(sendB + dof * nodeSendOffset[proc_id],
                               dof * nodeSendCount[proc_id], proc_id,
                               m_uiCommTag, commActive, &ctx.m_send_req[send_p]);
            }
        }

        m_uiCommTag++;
        m_uiMPIContexts.push_back(ctx);
    }

    return;
}

template <typename T>
void Mesh::readFromGhostEnd(T* vec, unsigned int dof) {
    if (this->getMPICommSizeGlobal() == 1 || (!m_uiIsActive)) return;

    // send recv buffers.
    T* sendB = NULL;
    T* recvB = NULL;

    if (this->isActive()) {
        const std::vector<unsigned int>& nodeSendCount =
            this->getNodalSendCounts();
        const std::vector<unsigned int>& nodeSendOffset =
            this->getNodalSendOffsets();

        const std::vector<unsigned int>& nodeRecvCount =
            this->getNodalRecvCounts();
        const std::vector<unsigned int>& nodeRecvOffset =
            this->getNodalRecvOffsets();

        const std::vector<unsigned int>& sendProcList = this->getSendProcList();
        const std::vector<unsigned int>& recvProcList = this->getRecvProcList();

        const std::vector<unsigned int>& sendNodeSM   = this->getSendNodeSM();
        const std::vector<unsigned int>& recvNodeSM   = this->getRecvNodeSM();

        const unsigned int activeNpes                 = this->getMPICommSize();

        const unsigned int sendBSz =
            nodeSendOffset[activeNpes - 1] + nodeSendCount[activeNpes - 1];
        const unsigned int recvBSz =
            nodeRecvOffset[activeNpes - 1] + nodeRecvCount[activeNpes - 1];
        unsigned int proc_id;

        int ctxIndex = -1;
        for (unsigned int i = 0; i < m_uiMPIContexts.size(); i++) {
            if (m_uiMPIContexts[i].getBuffer() == vec) {
                ctxIndex = i;
                break;
            }
        }

        if (ctxIndex == -1) {
            std::cout << "rank: " << m_uiActiveRank
                      << " async ctx not found for vec: " << &vec
                      << " in async comm end: " << __LINE__ << std::endl;
            MPI_Abort(m_uiCommActive, 0);
        }

        // wait for all posted recvs + sends (one MPI_Waitall each; the request
        // objects live in the context -- no per-request heap free needed).
        AsyncExchangeContex& cctx = m_uiMPIContexts[ctxIndex];
        if (!cctx.m_recv_req.empty())
            MPI_Waitall(cctx.m_recv_req.size(), cctx.m_recv_req.data(),
                        MPI_STATUSES_IGNORE);
        if (!cctx.m_send_req.empty())
            MPI_Waitall(cctx.m_send_req.size(), cctx.m_send_req.data(),
                        MPI_STATUSES_IGNORE);

        if (recvBSz) {
            // copy the recv data to the vec
            recvB = (T*)cctx.getRecvBuffer();

            // threaded scatter from recv buffer (recvNodeSM is a permutation)
#ifdef DENDRO_HYBRID_OMP
#pragma omp parallel for private(proc_id)
#endif
            for (unsigned int recv_p = 0; recv_p < recvProcList.size();
                 recv_p++) {
                proc_id = recvProcList[recv_p];

                for (unsigned int var = 0; var < dof; var++) {
                    for (unsigned int k = nodeRecvOffset[proc_id];
                         k < (nodeRecvOffset[proc_id] + nodeRecvCount[proc_id]);
                         k++) {
                        (vec + var * m_uiNumActualNodes)[recvNodeSM[k]] =
                            recvB[dof * (nodeRecvOffset[proc_id]) +
                                  (var * nodeRecvCount[proc_id]) +
                                  (k - nodeRecvOffset[proc_id])];
                    }
                }
            }
        }

        cctx.deAllocateSendBuffer();
        cctx.deAllocateRecvBuffer();
        cctx.m_recv_req.clear();
        cctx.m_send_req.clear();

        // remove the context ...
        m_uiMPIContexts.erase(m_uiMPIContexts.begin() + ctxIndex);
    }

    return;
}

template <typename T>
void Mesh::readFromGhostBegin(AsyncExchangeContex& ctx, T* vec,
                              unsigned int dof) {
    if (this->getMPICommSizeGlobal() == 1 || (!m_uiIsActive)) return;

    // send recv buffers.
    T* sendB = NULL;
    T* recvB = NULL;

    if (this->isActive()) {
        const std::vector<unsigned int>& nodeSendCount =
            this->getNodalSendCounts();
        const std::vector<unsigned int>& nodeSendOffset =
            this->getNodalSendOffsets();

        const std::vector<unsigned int>& nodeRecvCount =
            this->getNodalRecvCounts();
        const std::vector<unsigned int>& nodeRecvOffset =
            this->getNodalRecvOffsets();

        const std::vector<unsigned int>& sendProcList = this->getSendProcList();
        const std::vector<unsigned int>& recvProcList = this->getRecvProcList();

        const std::vector<unsigned int>& sendNodeSM   = this->getSendNodeSM();
        const std::vector<unsigned int>& recvNodeSM   = this->getRecvNodeSM();

        const unsigned int activeNpes                 = this->getMPICommSize();

        const unsigned int sendBSz =
            nodeSendOffset[activeNpes - 1] + nodeSendCount[activeNpes - 1];
        const unsigned int recvBSz =
            nodeRecvOffset[activeNpes - 1] + nodeRecvCount[activeNpes - 1];
        unsigned int proc_id;

        MPI_Comm commActive = this->getMPICommunicator();

        if (recvBSz) {
            recvB = (T*)ctx.getRecvBuffer();

            // active recv procs
            for (unsigned int recv_p = 0; recv_p < recvProcList.size();
                 recv_p++) {
                proc_id = recvProcList[recv_p];
                par::Mpi_Irecv((recvB + dof * nodeRecvOffset[proc_id]),
                               dof * nodeRecvCount[proc_id], proc_id,
                               m_uiCommTag, commActive,
                               &ctx.m_recv_req[recv_p]);
            }
        }

        if (sendBSz) {
            sendB = (T*)ctx.getSendBuffer();
#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
            dendro::timer::t_ghost_pack.start();
#endif
            // threaded gather into the send buffer (disjoint writes per proc)
#ifdef DENDRO_HYBRID_OMP
#pragma omp parallel for private(proc_id)
#endif
            for (unsigned int send_p = 0; send_p < sendProcList.size();
                 send_p++) {
                proc_id = sendProcList[send_p];

                for (unsigned int var = 0; var < dof; var++) {
                    for (unsigned int k = nodeSendOffset[proc_id];
                         k < (nodeSendOffset[proc_id] + nodeSendCount[proc_id]);
                         k++) {
                        sendB[dof * (nodeSendOffset[proc_id]) +
                              (var * nodeSendCount[proc_id]) +
                              (k - nodeSendOffset[proc_id])] =
                            (vec + var * m_uiNumActualNodes)[sendNodeSM[k]];
                    }
                }
            }
#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
            dendro::timer::t_ghost_pack.stop();
#endif

            // active send procs
            for (unsigned int send_p = 0; send_p < sendProcList.size();
                 send_p++) {
                proc_id = sendProcList[send_p];
                par::Mpi_Isend(sendB + dof * nodeSendOffset[proc_id],
                               dof * nodeSendCount[proc_id], proc_id,
                               m_uiCommTag, commActive,
                               &ctx.m_send_req[send_p]);
            }
        }

        m_uiCommTag++;
    }

    return;
}

template <typename T>
void Mesh::readFromGhostEnd(AsyncExchangeContex& ctx, T* vec,
                            unsigned int dof) {
    if (this->getMPICommSizeGlobal() == 1 || (!m_uiIsActive)) return;

    // send recv buffers.
    T* sendB = NULL;
    T* recvB = NULL;

    if (this->isActive()) {
        const std::vector<unsigned int>& nodeSendCount =
            this->getNodalSendCounts();
        const std::vector<unsigned int>& nodeSendOffset =
            this->getNodalSendOffsets();

        const std::vector<unsigned int>& nodeRecvCount =
            this->getNodalRecvCounts();
        const std::vector<unsigned int>& nodeRecvOffset =
            this->getNodalRecvOffsets();

        const std::vector<unsigned int>& sendProcList = this->getSendProcList();
        const std::vector<unsigned int>& recvProcList = this->getRecvProcList();

        const std::vector<unsigned int>& sendNodeSM   = this->getSendNodeSM();
        const std::vector<unsigned int>& recvNodeSM   = this->getRecvNodeSM();

        const unsigned int activeNpes                 = this->getMPICommSize();

        const unsigned int sendBSz =
            nodeSendOffset[activeNpes - 1] + nodeSendCount[activeNpes - 1];
        const unsigned int recvBSz =
            nodeRecvOffset[activeNpes - 1] + nodeRecvCount[activeNpes - 1];
        unsigned int proc_id;

        MPI_Status status;
        // need to wait for the commns to finish ...
        // t_ghost_wait is an UPPER BOUND on wire time, not a measurement of it:
        // a rank that reaches this Waitall early because its neighbours are
        // still computing blocks will bill that neighbour imbalance to comm.
        // Rising t_ghost_wait therefore does NOT imply the network got slower.
#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
        dendro::timer::t_ghost_wait.start();
#endif
        MPI_Waitall(sendProcList.size(), ctx.m_send_req.data(),
                    MPI_STATUSES_IGNORE);
        MPI_Waitall(recvProcList.size(), ctx.m_recv_req.data(),
                    MPI_STATUSES_IGNORE);
#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
        dendro::timer::t_ghost_wait.stop();
#endif

        if (recvBSz) {
            // copy the recv data to the vec
            recvB = (T*)ctx.getRecvBuffer();

#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
            dendro::timer::t_ghost_unpack.start();
#endif
            // threaded scatter from recv buffer (recvNodeSM is a permutation)
#ifdef DENDRO_HYBRID_OMP
#pragma omp parallel for private(proc_id)
#endif
            for (unsigned int recv_p = 0; recv_p < recvProcList.size();
                 recv_p++) {
                proc_id = recvProcList[recv_p];

                for (unsigned int var = 0; var < dof; var++) {
                    for (unsigned int k = nodeRecvOffset[proc_id];
                         k < (nodeRecvOffset[proc_id] + nodeRecvCount[proc_id]);
                         k++) {
                        (vec + var * m_uiNumActualNodes)[recvNodeSM[k]] =
                            recvB[dof * (nodeRecvOffset[proc_id]) +
                                  (var * nodeRecvCount[proc_id]) +
                                  (k - nodeRecvOffset[proc_id])];
                    }
                }
            }
#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
            dendro::timer::t_ghost_unpack.stop();
#endif
        }
    }

    return;
}

template <typename T>
void Mesh::readFromGhostBeginElementVec(T* vec, unsigned int dof) {
    if (this->getMPICommSizeGlobal() == 1 || (!m_uiIsActive)) return;

    // send recv buffers.
    T* sendB = NULL;
    T* recvB = NULL;

    if (this->isActive()) {
        const unsigned int activeNpes = m_uiActiveNpes;
        const unsigned int sendBSz    = m_uiSendEleOffset[activeNpes - 1] +
                                        m_uiSendEleCount[activeNpes - 1];
        const unsigned int recvBSz    = m_uiRecvEleOffset[activeNpes - 1] +
                                        m_uiRecvEleCount[activeNpes - 1];
        unsigned int proc_id;

        AsyncExchangeContex ctx(vec);
        MPI_Comm commActive = this->getMPICommunicator();

        if (recvBSz) {
            ctx.allocateRecvBuffer((sizeof(T) * recvBSz * dof));
            recvB = (T*)ctx.getRecvBuffer();

            // active recv procs
            for (unsigned int recv_p = 0;
                 recv_p < m_uiElementRecvProcList.size(); recv_p++) {
                proc_id          = m_uiElementRecvProcList[recv_p];
                MPI_Request* req = new MPI_Request();
                par::Mpi_Irecv((recvB + dof * m_uiRecvEleOffset[proc_id]),
                               dof * m_uiRecvEleCount[proc_id], proc_id,
                               m_uiCommTag, commActive, req);
                ctx.getRequestList().push_back(req);
            }
        }

        if (sendBSz) {
            ctx.allocateSendBuffer(sizeof(T) * dof * sendBSz);
            sendB = (T*)ctx.getSendBuffer();

            for (unsigned int send_p = 0;
                 send_p < m_uiElementSendProcList.size(); send_p++) {
                proc_id = m_uiElementSendProcList[send_p];

                for (unsigned int var = 0; var < dof; var++) {
                    for (unsigned int k = m_uiSendEleOffset[proc_id];
                         k < (m_uiSendEleOffset[proc_id] +
                              m_uiSendEleCount[proc_id]);
                         k++) {
                        sendB[dof * (m_uiSendEleOffset[proc_id]) +
                              (var * m_uiSendEleCount[proc_id]) +
                              (k - m_uiSendEleOffset[proc_id])] =
                            (vec + var * m_uiNumTotalElements)
                                [m_uiElementLocalBegin +
                                 m_uiScatterMapElementRound1[k]];
                    }
                }
            }

            // active send procs
            for (unsigned int send_p = 0;
                 send_p < m_uiElementSendProcList.size(); send_p++) {
                proc_id          = m_uiElementSendProcList[send_p];
                MPI_Request* req = new MPI_Request();
                par::Mpi_Isend(sendB + dof * m_uiSendEleOffset[proc_id],
                               dof * m_uiSendEleCount[proc_id], proc_id,
                               m_uiCommTag, commActive, req);
                ctx.getRequestList().push_back(req);
            }
        }

        m_uiCommTag++;
        m_uiMPIContexts.push_back(ctx);
    }

    return;
}

template <typename T>
void Mesh::readFromGhostEndElementVec(T* vec, unsigned int dof) {
    if (this->getMPICommSizeGlobal() == 1 || (!m_uiIsActive)) return;

    // send recv buffers.
    T* sendB = NULL;
    T* recvB = NULL;

    if (this->isActive()) {
        const unsigned int activeNpes = m_uiActiveNpes;
        const unsigned int sendBSz    = m_uiSendEleOffset[activeNpes - 1] +
                                        m_uiSendEleCount[activeNpes - 1];
        const unsigned int recvBSz    = m_uiRecvEleOffset[activeNpes - 1] +
                                        m_uiRecvEleCount[activeNpes - 1];
        unsigned int proc_id;

        int ctxIndex = -1;
        for (unsigned int i = 0; i < m_uiMPIContexts.size(); i++) {
            if (m_uiMPIContexts[i].getBuffer() == vec) {
                ctxIndex = i;
                break;
            }
        }

        if (ctxIndex == -1) {
            std::cout << "rank: " << m_uiActiveRank
                      << " async ctx not found for vec: " << &vec
                      << " in async comm end: " << __LINE__ << std::endl;
            MPI_Abort(m_uiCommActive, 0);
        }

        assert(m_uiMPIContexts[ctxIndex].getBuffer() == vec);

        MPI_Status status;
        // need to wait for the commns to finish ...
        for (unsigned int i = 0;
             i < m_uiMPIContexts[ctxIndex].getRequestList().size(); i++) {
            MPI_Wait(m_uiMPIContexts[ctxIndex].getRequestList()[i], &status);
        }

        if (recvBSz) {
            // copy the recv data to the vec
            recvB = (T*)m_uiMPIContexts[ctxIndex].getRecvBuffer();

            for (unsigned int recv_p = 0;
                 recv_p < m_uiElementRecvProcList.size(); recv_p++) {
                proc_id = m_uiElementRecvProcList[recv_p];
                for (unsigned int var = 0; var < dof; var++) {
                    for (unsigned int k = m_uiRecvEleOffset[proc_id];
                         k < (m_uiRecvEleOffset[proc_id] +
                              m_uiRecvEleCount[proc_id]);
                         k++)
                        (vec + var * m_uiNumTotalElements)
                            [m_uiGhostElementRound1Index[k]] =
                                recvB[dof * (m_uiRecvEleOffset[proc_id]) +
                                      (var * m_uiRecvEleCount[proc_id]) +
                                      (k - m_uiRecvEleOffset[proc_id])];
                }
            }
        }

        m_uiMPIContexts[ctxIndex].deAllocateSendBuffer();
        m_uiMPIContexts[ctxIndex].deAllocateRecvBuffer();

        for (unsigned int i = 0;
             i < m_uiMPIContexts[ctxIndex].getRequestList().size(); i++)
            delete m_uiMPIContexts[ctxIndex].getRequestList()[i];

        m_uiMPIContexts[ctxIndex].getRequestList().clear();

        // remove the context ...
        m_uiMPIContexts.erase(m_uiMPIContexts.begin() + ctxIndex);
    }

    return;
}

template <typename T>
void Mesh::readFromGhostBeginEleDGVec(T* vec, unsigned int dof) {
    if (this->getMPICommSizeGlobal() == 1 || (!m_uiIsActive)) return;

    // send recv buffers.
    T* sendB = NULL;
    T* recvB = NULL;

    if (this->isActive()) {
        const unsigned int activeNpes = m_uiActiveNpes;
        const unsigned int sendBSz    = (m_uiSendEleOffset[activeNpes - 1] +
                                         m_uiSendEleCount[activeNpes - 1]) *
                                        m_uiNpE;
        const unsigned int recvBSz    = (m_uiRecvEleOffset[activeNpes - 1] +
                                         m_uiRecvEleCount[activeNpes - 1]) *
                                        m_uiNpE;
        unsigned int proc_id;

        AsyncExchangeContex ctx(vec);
        MPI_Comm commActive = this->getMPICommunicator();

        if (recvBSz) {
            ctx.allocateRecvBuffer((sizeof(T) * recvBSz * dof));
            recvB = (T*)ctx.getRecvBuffer();

            // active recv procs
            for (unsigned int recv_p = 0;
                 recv_p < m_uiElementRecvProcList.size(); recv_p++) {
                proc_id          = m_uiElementRecvProcList[recv_p];
                MPI_Request* req = new MPI_Request();
                par::Mpi_Irecv(
                    (recvB + dof * m_uiNpE * m_uiRecvEleOffset[proc_id]),
                    dof * m_uiNpE * m_uiRecvEleCount[proc_id], proc_id,
                    m_uiCommTag, commActive, req);
                ctx.getRequestList().push_back(req);
            }
        }

        if (sendBSz) {
            ctx.allocateSendBuffer(sizeof(T) * dof * sendBSz);
            sendB = (T*)ctx.getSendBuffer();

            for (unsigned int send_p = 0;
                 send_p < m_uiElementSendProcList.size(); send_p++) {
                proc_id = m_uiElementSendProcList[send_p];

                for (unsigned int var = 0; var < dof; var++) {
                    for (unsigned int k = m_uiSendEleOffset[proc_id];
                         k < (m_uiSendEleOffset[proc_id] +
                              m_uiSendEleCount[proc_id]);
                         k++) {
                        for (unsigned int node = 0; node < m_uiNpE; node++)
                            sendB[dof * m_uiNpE * (m_uiSendEleOffset[proc_id]) +
                                  (var * m_uiNpE * m_uiSendEleCount[proc_id]) +
                                  (k - m_uiSendEleOffset[proc_id]) * m_uiNpE +
                                  node] =
                                (vec +
                                 var * m_uiNumTotalElements *
                                     m_uiNpE)[(m_uiElementLocalBegin +
                                               m_uiScatterMapElementRound1[k]) *
                                                  m_uiNpE +
                                              node];
                    }
                }
            }

            // active send procs
            for (unsigned int send_p = 0;
                 send_p < m_uiElementSendProcList.size(); send_p++) {
                proc_id          = m_uiElementSendProcList[send_p];
                MPI_Request* req = new MPI_Request();
                par::Mpi_Isend(
                    sendB + dof * m_uiNpE * m_uiSendEleOffset[proc_id],
                    dof * m_uiNpE * m_uiSendEleCount[proc_id], proc_id,
                    m_uiCommTag, commActive, req);
                ctx.getRequestList().push_back(req);
            }
        }

        m_uiCommTag++;
        m_uiMPIContexts.push_back(ctx);
    }

    return;
}

template <typename T>
void Mesh::readFromGhostEndEleDGVec(T* vec, unsigned int dof) {
    if (this->getMPICommSizeGlobal() == 1 || (!m_uiIsActive)) return;

    // send recv buffers.
    T* sendB = NULL;
    T* recvB = NULL;

    if (this->isActive()) {
        const unsigned int activeNpes = m_uiActiveNpes;
        const unsigned int sendBSz    = (m_uiSendEleOffset[activeNpes - 1] +
                                         m_uiSendEleCount[activeNpes - 1]) *
                                        m_uiNpE;
        const unsigned int recvBSz    = (m_uiRecvEleOffset[activeNpes - 1] +
                                         m_uiRecvEleCount[activeNpes - 1]) *
                                        m_uiNpE;
        unsigned int proc_id;

        int ctxIndex = -1;
        for (unsigned int i = 0; i < m_uiMPIContexts.size(); i++) {
            if (m_uiMPIContexts[i].getBuffer() == vec) {
                ctxIndex = i;
                break;
            }
        }

        if (ctxIndex == -1) {
            std::cout << "rank: " << m_uiActiveRank
                      << " async ctx not found for vec: " << &vec
                      << " in async comm end: " << __LINE__ << std::endl;
            MPI_Abort(m_uiCommActive, 0);
        }

        assert(m_uiMPIContexts[ctxIndex].getBuffer() == vec);

        MPI_Status status;
        // need to wait for the commns to finish ...
        for (unsigned int i = 0;
             i < m_uiMPIContexts[ctxIndex].getRequestList().size(); i++) {
            MPI_Wait(m_uiMPIContexts[ctxIndex].getRequestList()[i], &status);
        }

        if (recvBSz) {
            // copy the recv data to the vec
            recvB = (T*)m_uiMPIContexts[ctxIndex].getRecvBuffer();

            for (unsigned int recv_p = 0;
                 recv_p < m_uiElementRecvProcList.size(); recv_p++) {
                proc_id = m_uiElementRecvProcList[recv_p];
                for (unsigned int var = 0; var < dof; var++) {
                    for (unsigned int k = m_uiRecvEleOffset[proc_id];
                         k < (m_uiRecvEleOffset[proc_id] +
                              m_uiRecvEleCount[proc_id]);
                         k++)
                        for (unsigned int node = 0; node < m_uiNpE; node++)
                            (vec + var * m_uiNumTotalElements *
                                       m_uiNpE)[m_uiGhostElementRound1Index[k] *
                                                    m_uiNpE +
                                                node] = recvB
                                [dof * (m_uiRecvEleOffset[proc_id] * m_uiNpE) +
                                 (var * m_uiNpE * m_uiRecvEleCount[proc_id]) +
                                 (k - m_uiRecvEleOffset[proc_id]) * m_uiNpE +
                                 node];
                }
            }
        }

        m_uiMPIContexts[ctxIndex].deAllocateSendBuffer();
        m_uiMPIContexts[ctxIndex].deAllocateRecvBuffer();

        for (unsigned int i = 0;
             i < m_uiMPIContexts[ctxIndex].getRequestList().size(); i++)
            delete m_uiMPIContexts[ctxIndex].getRequestList()[i];

        m_uiMPIContexts[ctxIndex].getRequestList().clear();

        // remove the context ...
        m_uiMPIContexts.erase(m_uiMPIContexts.begin() + ctxIndex);
    }

    return;
}

template <typename T>
void Mesh::writeFromGhostBegin(T* vec, unsigned int dof) {
    if (this->getMPICommSizeGlobal() == 1 || (!m_uiIsActive)) return;

    // send recv buffers.
    T* sendB = NULL;
    T* recvB = NULL;

    if (this->isActive()) {
        // inverse direction of the read ghost
        const std::vector<unsigned int>& nodeSendCount =
            this->getNodalRecvCounts();
        const std::vector<unsigned int>& nodeSendOffset =
            this->getNodalRecvOffsets();

        const std::vector<unsigned int>& nodeRecvCount =
            this->getNodalSendCounts();
        const std::vector<unsigned int>& nodeRecvOffset =
            this->getNodalSendOffsets();

        const std::vector<unsigned int>& sendProcList = this->getRecvProcList();
        const std::vector<unsigned int>& recvProcList = this->getSendProcList();

        const std::vector<unsigned int>& sendNodeSM   = this->getRecvNodeSM();
        const std::vector<unsigned int>& recvNodeSM   = this->getSendNodeSM();

        const unsigned int activeNpes                 = this->getMPICommSize();

        const unsigned int sendBSz =
            nodeSendOffset[activeNpes - 1] + nodeSendCount[activeNpes - 1];
        const unsigned int recvBSz =
            nodeRecvOffset[activeNpes - 1] + nodeRecvCount[activeNpes - 1];
        unsigned int proc_id;

        AsyncExchangeContex ctx(vec);
        MPI_Comm commActive = this->getMPICommunicator();

        if (recvBSz) {
            ctx.allocateRecvBuffer((sizeof(T) * recvBSz * dof));
            recvB = (T*)ctx.getRecvBuffer();

            // active recv procs
            for (unsigned int recv_p = 0; recv_p < recvProcList.size();
                 recv_p++) {
                proc_id          = recvProcList[recv_p];
                MPI_Request* req = new MPI_Request();
                par::Mpi_Irecv((recvB + dof * nodeRecvOffset[proc_id]),
                               dof * nodeRecvCount[proc_id], proc_id,
                               m_uiCommTag, commActive, req);
                ctx.getRequestList().push_back(req);
            }
        }

        if (sendBSz) {
            ctx.allocateSendBuffer(sizeof(T) * dof * sendBSz);
            sendB = (T*)ctx.getSendBuffer();

            for (unsigned int send_p = 0; send_p < sendProcList.size();
                 send_p++) {
                proc_id = sendProcList[send_p];

                for (unsigned int var = 0; var < dof; var++) {
                    for (unsigned int k = nodeSendOffset[proc_id];
                         k < (nodeSendOffset[proc_id] + nodeSendCount[proc_id]);
                         k++) {
                        sendB[dof * (nodeSendOffset[proc_id]) +
                              (var * nodeSendCount[proc_id]) +
                              (k - nodeSendOffset[proc_id])] =
                            (vec + var * m_uiNumActualNodes)[sendNodeSM[k]];
                    }
                }
            }

            // active send procs
            for (unsigned int send_p = 0; send_p < sendProcList.size();
                 send_p++) {
                proc_id          = sendProcList[send_p];
                MPI_Request* req = new MPI_Request();
                par::Mpi_Isend(sendB + dof * nodeSendOffset[proc_id],
                               dof * nodeSendCount[proc_id], proc_id,
                               m_uiCommTag, commActive, req);
                ctx.getRequestList().push_back(req);
            }
        }

        m_uiCommTag++;
        m_uiMPIContexts.push_back(ctx);
    }

    return;
}

template <typename T>
void Mesh::writeFromGhostEnd(T* vec, ot::GWMode mode, unsigned int dof) {
    if (this->getMPICommSizeGlobal() == 1 || (!m_uiIsActive)) return;

    // send recv buffers.
    T* sendB = NULL;
    T* recvB = NULL;

    if (this->isActive()) {
        // inverse direction of the read ghost
        const std::vector<unsigned int>& nodeSendCount =
            this->getNodalRecvCounts();
        const std::vector<unsigned int>& nodeSendOffset =
            this->getNodalRecvOffsets();

        const std::vector<unsigned int>& nodeRecvCount =
            this->getNodalSendCounts();
        const std::vector<unsigned int>& nodeRecvOffset =
            this->getNodalSendOffsets();

        const std::vector<unsigned int>& sendProcList = this->getRecvProcList();
        const std::vector<unsigned int>& recvProcList = this->getSendProcList();

        const std::vector<unsigned int>& sendNodeSM   = this->getRecvNodeSM();
        const std::vector<unsigned int>& recvNodeSM   = this->getSendNodeSM();

        const unsigned int activeNpes                 = this->getMPICommSize();

        const unsigned int sendBSz =
            nodeSendOffset[activeNpes - 1] + nodeSendCount[activeNpes - 1];
        const unsigned int recvBSz =
            nodeRecvOffset[activeNpes - 1] + nodeRecvCount[activeNpes - 1];
        unsigned int proc_id;

        int ctxIndex = -1;
        for (unsigned int i = 0; i < m_uiMPIContexts.size(); i++) {
            if (m_uiMPIContexts[i].getBuffer() == vec) {
                ctxIndex = i;
                break;
            }
        }

        if (ctxIndex == -1) {
            std::cout << "rank: " << m_uiActiveRank
                      << " async ctx not found for vec: " << &vec
                      << " in async comm end: " << __LINE__ << std::endl;
            MPI_Abort(m_uiCommActive, 0);
        }

        MPI_Status status;
        // need to wait for the commns to finish ...
        for (unsigned int i = 0;
             i < m_uiMPIContexts[ctxIndex].getRequestList().size(); i++) {
            MPI_Wait(m_uiMPIContexts[ctxIndex].getRequestList()[i], &status);
        }

        if (mode == ot::GWMode::ACCUMILATE) {
            if (recvBSz) {
                // copy the recv data to the vec
                recvB = (T*)m_uiMPIContexts[ctxIndex].getRecvBuffer();

                for (unsigned int recv_p = 0; recv_p < recvProcList.size();
                     recv_p++) {
                    proc_id = recvProcList[recv_p];

                    for (unsigned int var = 0; var < dof; var++) {
                        for (unsigned int k = nodeRecvOffset[proc_id];
                             k <
                             (nodeRecvOffset[proc_id] + nodeRecvCount[proc_id]);
                             k++) {
                            (vec + var * m_uiNumActualNodes)[recvNodeSM[k]] +=
                                recvB[dof * (nodeRecvOffset[proc_id]) +
                                      (var * nodeRecvCount[proc_id]) +
                                      (k - nodeRecvOffset[proc_id])];
                        }
                    }
                }
            }
        } else {
            assert(mode == ot::GWMode::OVERWRITE);
            if (recvBSz) {
                // copy the recv data to the vec
                recvB = (T*)m_uiMPIContexts[ctxIndex].getRecvBuffer();

                for (unsigned int recv_p = 0; recv_p < recvProcList.size();
                     recv_p++) {
                    proc_id = recvProcList[recv_p];

                    for (unsigned int var = 0; var < dof; var++) {
                        for (unsigned int k = nodeRecvOffset[proc_id];
                             k <
                             (nodeRecvOffset[proc_id] + nodeRecvCount[proc_id]);
                             k++) {
                            (vec + var * m_uiNumActualNodes)[recvNodeSM[k]] =
                                recvB[dof * (nodeRecvOffset[proc_id]) +
                                      (var * nodeRecvCount[proc_id]) +
                                      (k - nodeRecvOffset[proc_id])];
                        }
                    }
                }
            }
        }

        m_uiMPIContexts[ctxIndex].deAllocateSendBuffer();
        m_uiMPIContexts[ctxIndex].deAllocateRecvBuffer();

        for (unsigned int i = 0;
             i < m_uiMPIContexts[ctxIndex].getRequestList().size(); i++)
            delete m_uiMPIContexts[ctxIndex].getRequestList()[i];

        m_uiMPIContexts[ctxIndex].getRequestList().clear();

        // remove the context ...
        m_uiMPIContexts.erase(m_uiMPIContexts.begin() + ctxIndex);
    }

    return;
}

template <typename T>
void Mesh::gatherFromGhostBegin(T* vec, unsigned int dof) {
    this->writeFromGhostBegin(vec, dof);
}

template <typename T>
void Mesh::gatherFromGhostEnd(T* vec, std::vector<std::vector<T> >& gatherV,
                              unsigned int dof) {
    if (this->getMPICommSizeGlobal() == 1 || (!m_uiIsActive)) return;

    // send recv buffers.
    T* sendB = NULL;
    T* recvB = NULL;

    if (this->isActive()) {
        // inverse direction of the read ghost
        const std::vector<unsigned int>& nodeSendCount =
            this->getNodalRecvCounts();
        const std::vector<unsigned int>& nodeSendOffset =
            this->getNodalRecvOffsets();

        const std::vector<unsigned int>& nodeRecvCount =
            this->getNodalSendCounts();
        const std::vector<unsigned int>& nodeRecvOffset =
            this->getNodalSendOffsets();

        const std::vector<unsigned int>& sendProcList = this->getRecvProcList();
        const std::vector<unsigned int>& recvProcList = this->getSendProcList();

        const std::vector<unsigned int>& sendNodeSM   = this->getRecvNodeSM();
        const std::vector<unsigned int>& recvNodeSM   = this->getSendNodeSM();

        const unsigned int activeNpes                 = this->getMPICommSize();

        const unsigned int sendBSz =
            nodeSendOffset[activeNpes - 1] + nodeSendCount[activeNpes - 1];
        const unsigned int recvBSz =
            nodeRecvOffset[activeNpes - 1] + nodeRecvCount[activeNpes - 1];
        unsigned int proc_id;

        unsigned int ctxIndex = 0;
        for (unsigned int i = 0; i < m_uiMPIContexts.size(); i++) {
            if (m_uiMPIContexts[i].getBuffer() == vec) {
                ctxIndex = i;
                break;
            }
        }

        MPI_Status status;
        // need to wait for the commns to finish ...
        for (unsigned int i = 0;
             i < m_uiMPIContexts[ctxIndex].getRequestList().size(); i++) {
            MPI_Wait(m_uiMPIContexts[ctxIndex].getRequestList()[i], &status);
        }

        if (recvBSz) {
            gatherV.resize(m_uiNumActualNodes);
            recvB = (T*)m_uiMPIContexts[ctxIndex].getRecvBuffer();
            std::vector<unsigned int> gcounts;
            gcounts.resize(m_uiNumActualNodes, 0);

            for (unsigned int p = 0; p < recvProcList.size(); p++) {
                const unsigned int proc_id = recvProcList[p];
                for (unsigned int k = nodeRecvOffset[proc_id];
                     k < (nodeRecvOffset[proc_id] + nodeRecvCount[proc_id]);
                     k++)
                    gcounts[recvNodeSM[k]]++;
            }

            for (unsigned int k = m_uiNodeLocalBegin; k < m_uiNodeLocalEnd;
                 k++) {
                if (gcounts[k] > 0) {
                    gatherV[k].resize(gcounts[k] * dof);
                    gcounts[k] = 0;
                }
            }

            for (unsigned int p = 0; p < recvProcList.size(); p++) {
                const unsigned int proc_id = recvProcList[p];
                for (unsigned int k = nodeRecvOffset[proc_id];
                     k < (nodeRecvOffset[proc_id] + nodeRecvCount[proc_id]);
                     k++) {
                    const unsigned int gsz = gatherV[recvNodeSM[k]].size();
                    for (unsigned int v = 0; v < dof; v++) {
                        gatherV[recvNodeSM[k]]
                               [v * gsz + gcounts[recvNodeSM[k]]] =
                                   recvB[dof * (nodeRecvOffset[proc_id]) +
                                         (v * nodeRecvCount[proc_id]) +
                                         (k - nodeRecvOffset[proc_id])];
                        gcounts[recvNodeSM[k]] += 1;
                    }
                }
            }

            for (unsigned int recv_p = 0; recv_p < recvProcList.size();
                 recv_p++) {
                const unsigned int proc_id = recvProcList[recv_p];
                for (unsigned int var = 0; var < dof; var++) {
                    for (unsigned int k = nodeRecvOffset[proc_id];
                         k < (nodeRecvOffset[proc_id] + nodeRecvCount[proc_id]);
                         k++)
                        (vec + var * m_uiNumActualNodes)[recvNodeSM[k]] =
                            recvB[dof * (nodeRecvOffset[proc_id]) +
                                  (var * nodeRecvCount[proc_id]) +
                                  (k - nodeRecvOffset[proc_id])];
                }
            }
        }

        m_uiMPIContexts[ctxIndex].deAllocateSendBuffer();
        m_uiMPIContexts[ctxIndex].deAllocateRecvBuffer();

        for (unsigned int i = 0;
             i < m_uiMPIContexts[ctxIndex].getRequestList().size(); i++)
            delete m_uiMPIContexts[ctxIndex].getRequestList()[i];

        m_uiMPIContexts[ctxIndex].getRequestList().clear();
        m_uiMPIContexts.erase(m_uiMPIContexts.begin() + ctxIndex);
    }
}

template <typename T, unsigned int length, unsigned int offsetCentered,
          unsigned int offsetBackward, unsigned int offsetForward>
void Mesh::applyStencil(const std::vector<T>& in, std::vector<T>& out,
                        const Stencil<T, length, offsetCentered>& centered,
                        const Stencil<T, length, offsetBackward>& backward,
                        const Stencil<T, length, offsetForward>& forward) {
    if (!m_uiIsActive) return;

    double t_uzip;
    double t_uzip_g[3];

    double t_zip;
    double t_zip_g[3];

    double t_stencil;
    double t_stencil_g[3];

    unsigned int blkNpe_1D;
    std::vector<T> unzipVec;
    createUnZippedVector(unzipVec);

    std::vector<T> unzipVec1;
    this->createUnZippedVector(unzipVec1, 0.0);

#ifdef PROFILE_APPLY_STENCIL
    auto t1 = std::chrono::high_resolution_clock::now();
#endif
    this->unzip(&(*(in.begin())), &(*(unzipVec.begin())));
    // std::cout<<"rank: "<<m_uiActiveRank<<" unzip completed "<<std::endl;

#ifdef PROFILE_APPLY_STENCIL
    auto t2 = std::chrono::high_resolution_clock::now();
    t_uzip =
        std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    par::Mpi_Reduce(&t_uzip, t_uzip_g, 1, MPI_MIN, 0, m_uiCommActive);
    par::Mpi_Reduce(&t_uzip, t_uzip_g + 1, 1, MPI_SUM, 0, m_uiCommActive);
    par::Mpi_Reduce(&t_uzip, t_uzip_g + 2, 1, MPI_MIN, 0, m_uiCommActive);
    t_uzip_g[1] = t_uzip_g[1] / (double)m_uiActiveNpes;
#endif

    unsigned int regLev = 0;
    ot::TreeNode blkNode;

    unsigned int centeredOffset = centered.getOffset();
    unsigned int backwardOffset = backward.getOffset();
    unsigned int forwardOffset  = forward.getOffset();

    // all the 3 stencil directions should be in the same.
    assert(centered.getStencilDirection() == forward.getStencilDirection());
    assert(centered.getStencilDirection() == backward.getStencilDirection());
    double h = 0.0;
    unsigned int lx, ly, lz, offset, paddWidth;
#ifdef DEBUG_UNZIP_OP
    double d_min = -0.5;
    double d_max = 0.5;
    std::function<double(double, double, double)> func =
        [d_min, d_max](const double x, const double y, const double z) {
            return (
                sin(2 * M_PI *
                    ((x / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min)) *
                sin(2 * M_PI *
                    ((y / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min)) *
                sin(2 * M_PI *
                    ((z / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min)));
        };
    std::function<double(double, double, double)> dx_func = [d_min, d_max](
                                                                const double x,
                                                                const double y,
                                                                const double
                                                                    z) {
        return (2 * M_PI * (1.0 / (1u << m_uiMaxDepth) * (d_max - d_min))) *
               (cos(2 * M_PI *
                    ((x / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min)) *
                sin(2 * M_PI *
                    ((y / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min)) *
                sin(2 * M_PI *
                    ((z / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min)));
    };
    unsigned int x, y, z, sz, regSz;

    for (unsigned int blk = 0; blk < m_uiLocalBlockList.size(); blk++) {
        blkNode   = m_uiLocalBlockList[blk].getBlockNode();
        regLev    = m_uiLocalBlockList[blk].getRegularGridLev();
        lx        = m_uiLocalBlockList[blk].getAllocationSzX();
        ly        = m_uiLocalBlockList[blk].getAllocationSzY();
        lz        = m_uiLocalBlockList[blk].getAllocationSzZ();
        offset    = m_uiLocalBlockList[blk].getOffset();
        paddWidth = m_uiLocalBlockList[blk].get1DPadWidth();
        // h=((1u<<(m_uiMaxDepth))*m_uiElementOrder)/((0.5-(-0.5)) *
        // ((1u<<(regLev-blkNode.getLevel()))));
        h         = ((blkNode.maxX() - blkNode.minX())) /
                    ((1u << (regLev - blkNode.getLevel())) * m_uiElementOrder);
        h         = 1.0 / h;
        blkNpe_1D = m_uiElementOrder * (1u << (regLev - blkNode.getLevel())) +
                    1 + 2 * paddWidth;
        assert(blkNpe_1D > paddWidth);

        for (unsigned int k = 0; k < (blkNpe_1D); k++)
            for (unsigned int j = 0; j < (blkNpe_1D); j++)
                for (unsigned int i = 0; i < (blkNpe_1D); i++) {
                    assert(((blkNode.maxX() - blkNode.minX())) %
                               ((1u << (regLev - blkNode.getLevel())) *
                                m_uiElementOrder) ==
                           0);
                    sz    = ((blkNode.maxX() - blkNode.minX())) /
                            ((1u << (regLev - blkNode.getLevel())) *
                             m_uiElementOrder);
                    regSz = 1u << (m_uiMaxDepth - regLev);

                    if ((i > paddWidth && i < (blkNpe_1D - paddWidth - 1)) &&
                        (j > paddWidth && j < (blkNpe_1D - paddWidth - 1)) &&
                        (k > paddWidth && k < (blkNpe_1D - paddWidth - 1))) {
                        x = blkNode.getX() + (i - paddWidth) * sz;
                        y = blkNode.getY() + (j - paddWidth) * sz;
                        z = blkNode.getZ() + (k - paddWidth) * sz;
                        if (fabs(func(x, y, z) -
                                 unzipVec[offset + k * (ly * lx) + j * (lx) +
                                          i]) > 1e-5)
                            std::cout
                                << " [internal node mismatch] blk: " << blk
                                << " blkNode: " << blkNode << " sz: " << sz
                                << " blkNode_1D: " << blkNpe_1D
                                << " (x,y,z): ( " << x << ", " << y << ", " << z
                                << ")  (i,j,k)= (" << i << "," << j << ", " << k
                                << " )"
                                << " ) func: " << func(x, y, z)
                                << " : read value "
                                << unzipVec[offset + k * (ly * lx) + j * (lx) +
                                            i]
                                << std::endl;
                    }

                    if ((blkNode.getX() >= regSz) &&
                        (i >= 0 && i <= (paddWidth)) &&
                        (j > paddWidth && j < (blkNpe_1D - paddWidth)) &&
                        (k > paddWidth && k < (blkNpe_1D - paddWidth))) {
                        x = blkNode.getX() - regSz + (i + paddWidth) * sz;
                        y = blkNode.getY() + (j - paddWidth) * sz;
                        z = blkNode.getZ() + (k - paddWidth) * sz;
                        if (fabs(func(x, y, z) -
                                 unzipVec[offset + k * (ly * lx) + j * (lx) +
                                          i]) > 1e-5)
                            std::cout
                                << " [left ghost layer mismatch] blk: " << blk
                                << " blkNode: " << blkNode << " sz: " << sz
                                << " blkNode_1D: " << blkNpe_1D
                                << " (x,y,z): ( " << x << ", " << y << ", " << z
                                << ")  (i,j,k)= (" << i << "," << j << ", " << k
                                << " )"
                                << " ) func: " << func(x, y, z)
                                << " : read value "
                                << unzipVec[offset + k * (ly * lx) + j * (lx) +
                                            i]
                                << std::endl;
                    }

                    if ((blkNode.getY() >= regSz) &&
                        (j >= 0 && j <= (paddWidth)) &&
                        (i > paddWidth && i < (blkNpe_1D - paddWidth)) &&
                        (k > paddWidth && k < (blkNpe_1D - paddWidth))) {
                        x = blkNode.getX() + (i - paddWidth) * sz;
                        y = blkNode.getY() - regSz + (j + paddWidth) * sz;
                        z = blkNode.getZ() + (k - paddWidth) * sz;
                        if (fabs(func(x, y, z) -
                                 unzipVec[offset + k * (ly * lx) + j * (lx) +
                                          i]) > 1e-5)
                            std::cout
                                << " [down ghost layer mismatch] blk: " << blk
                                << " blkNode: " << blkNode << " sz: " << sz
                                << " blkNode_1D: " << blkNpe_1D
                                << " (x,y,z): ( " << x << ", " << y << ", " << z
                                << ")  (i,j,k)= (" << i << "," << j << ", " << k
                                << " )"
                                << " ) func: " << func(x, y, z)
                                << " : read value "
                                << unzipVec[offset + k * (ly * lx) + j * (lx) +
                                            i]
                                << std::endl;
                    }

                    if ((blkNode.getZ() >= regSz) &&
                        (k >= 0 && k <= (paddWidth)) &&
                        (i > paddWidth && i < (blkNpe_1D - paddWidth)) &&
                        (j > paddWidth && j < (blkNpe_1D - paddWidth))) {
                        x = blkNode.getX() + (i - paddWidth) * sz;
                        y = blkNode.getY() + (j - paddWidth) * sz;
                        z = blkNode.getZ() - regSz + (k + paddWidth) * sz;
                        if (fabs(func(x, y, z) -
                                 unzipVec[offset + k * (ly * lx) + j * (lx) +
                                          i]) > 1e-5)
                            std::cout
                                << " [back ghost layer mismatch] blk: " << blk
                                << " blkNode: " << blkNode << " sz: " << sz
                                << " blkNode_1D: " << blkNpe_1D
                                << " (x,y,z): ( " << x << ", " << y << ", " << z
                                << ")  (i,j,k)= (" << i << "," << j << ", " << k
                                << " )"
                                << " ) func: " << func(x, y, z)
                                << " : read value "
                                << unzipVec[offset + k * (ly * lx) + j * (lx) +
                                            i]
                                << std::endl;
                    }

                    if ((blkNode.maxX() + regSz <= m_uiMeshDomain_max) &&
                        (i >= (blkNpe_1D - paddWidth) && i < (blkNpe_1D)) &&
                        (j > paddWidth && j < (blkNpe_1D - paddWidth)) &&
                        (k > paddWidth && k < (blkNpe_1D - paddWidth))) {
                        x = blkNode.getX() - regSz + (i + paddWidth) * sz;
                        y = blkNode.getY() + (j - paddWidth) * sz;
                        z = blkNode.getZ() + (k - paddWidth) * sz;
                        if (fabs(func(x, y, z) -
                                 unzipVec[offset + k * (ly * lx) + j * (lx) +
                                          i]) > 1e-5)
                            std::cout
                                << " [right ghost layer mismatch] blk: " << blk
                                << " blkNode: " << blkNode << " sz: " << sz
                                << " blkNode_1D: " << blkNpe_1D
                                << " (x,y,z): ( " << x << ", " << y << ", " << z
                                << ")  (i,j,k)= (" << i << "," << j << ", " << k
                                << " )"
                                << " ) func: " << func(x, y, z)
                                << " : read value "
                                << unzipVec[offset + k * (ly * lx) + j * (lx) +
                                            i]
                                << std::endl;
                    }

                    if ((blkNode.maxY() + regSz <= m_uiMeshDomain_max) &&
                        (j >= (blkNpe_1D - paddWidth) && j < (blkNpe_1D)) &&
                        (i > paddWidth && i < (blkNpe_1D - paddWidth)) &&
                        (k > paddWidth && k < (blkNpe_1D - paddWidth))) {
                        x = blkNode.getX() + (i - paddWidth) * sz;
                        y = blkNode.getY() - regSz + (j + paddWidth) * sz;
                        z = blkNode.getZ() + (k - paddWidth) * sz;
                        if (fabs(func(x, y, z) -
                                 unzipVec[offset + k * (ly * lx) + j * (lx) +
                                          i]) > 1e-5)
                            std::cout
                                << " [up ghost layer mismatch] blk: " << blk
                                << " blkNode: " << blkNode << " sz: " << sz
                                << " blkNode_1D: " << blkNpe_1D
                                << " (x,y,z): ( " << x << ", " << y << ", " << z
                                << ")  (i,j,k)= (" << i << "," << j << ", " << k
                                << " )"
                                << " ) func: " << func(x, y, z)
                                << " : read value "
                                << unzipVec[offset + k * (ly * lx) + j * (lx) +
                                            i]
                                << std::endl;
                    }

                    if ((blkNode.maxZ() + regSz <= m_uiMeshDomain_max) &&
                        (k >= (blkNpe_1D - paddWidth) && k < (blkNpe_1D)) &&
                        (i > paddWidth && i < (blkNpe_1D - paddWidth)) &&
                        (j > paddWidth && j < (blkNpe_1D - paddWidth))) {
                        x = blkNode.getX() + (i - paddWidth) * sz;
                        y = blkNode.getY() + (j - paddWidth) * sz;
                        z = blkNode.getZ() - regSz + (k + paddWidth) * sz;
                        if (fabs(func(x, y, z) -
                                 unzipVec[offset + k * (ly * lx) + j * (lx) +
                                          i]) > 1e-5)
                            std::cout
                                << " [front ghost layer mismatch] blk: " << blk
                                << " blkNode: " << blkNode << " sz: " << sz
                                << " blkNode_1D: " << blkNpe_1D
                                << " (x,y,z): ( " << x << ", " << y << ", " << z
                                << ")  (i,j,k)= (" << i << "," << j << ", " << k
                                << " )"
                                << " ) func: " << func(x, y, z)
                                << " : read value "
                                << unzipVec[offset + k * (ly * lx) + j * (lx) +
                                            i]
                                << std::endl;
                    }
                }
    }

#endif

#ifdef PROFILE_APPLY_STENCIL
    t1 = std::chrono::high_resolution_clock::now();
#endif

    if (centered.getStencilDirection() == StencilDirection::STENCIL_DIR_X) {
        for (unsigned int blk = 0; blk < m_uiLocalBlockList.size(); blk++) {
            blkNode   = m_uiLocalBlockList[blk].getBlockNode();
            regLev    = m_uiLocalBlockList[blk].getRegularGridLev();

            lx        = m_uiLocalBlockList[blk].getAllocationSzX();
            ly        = m_uiLocalBlockList[blk].getAllocationSzY();
            lz        = m_uiLocalBlockList[blk].getAllocationSzZ();
            offset    = m_uiLocalBlockList[blk].getOffset();
            paddWidth = m_uiLocalBlockList[blk].get1DPadWidth();

            // h=((1u<<(m_uiMaxDepth))*m_uiElementOrder)/((0.5-(-0.5)) *
            // ((1u<<(regLev-blkNode.getLevel()))));
            h         = ((blkNode.maxX() - blkNode.minX())) /
                        ((double)(1u << (regLev - blkNode.getLevel())) *
                         m_uiElementOrder);
            h         = 1.0 / h;
            blkNpe_1D =
                m_uiElementOrder * (1u << (regLev - blkNode.getLevel())) + 1 +
                2 * paddWidth;
            assert(blkNpe_1D > paddWidth);

            if (blkNode.minX() == m_uiMeshDomain_min) {
                // std::cout<<"rank: "<<m_uiActiveRank<<" applying forward
                // difference difference: "<<std::endl;
                for (unsigned int k = paddWidth; k < (blkNpe_1D - paddWidth);
                     k++)
                    for (unsigned int j = paddWidth;
                         j < (blkNpe_1D - paddWidth); j++)
                        for (unsigned int i = paddWidth; i < 2 * paddWidth;
                             i++) {
                            unzipVec1[offset + k * (ly * lx) + j * (lx) + i] =
                                0;
                            for (unsigned int index = 0;
                                 index < forward.getStencilLength(); index++)
                                unzipVec1[offset + k * (ly * lx) + j * (lx) +
                                          i] +=
                                    forward[index] *
                                    unzipVec[offset + k * (ly * lx) + j * (lx) +
                                             i + index - forwardOffset] *
                                    h;
                        }

                for (unsigned int k = paddWidth; k < (blkNpe_1D - paddWidth);
                     k++)
                    for (unsigned int j = paddWidth;
                         j < (blkNpe_1D - paddWidth); j++)
                        for (unsigned int i = 2 * paddWidth;
                             i < (blkNpe_1D - 2 * paddWidth); i++) {
                            unzipVec1[offset + k * (ly * lx) + j * (lx) + i] =
                                0;
                            for (unsigned int index = 0;
                                 index < centered.getStencilLength(); index++)
                                unzipVec1[offset + k * (ly * lx) + j * (lx) +
                                          i] +=
                                    centered[index] *
                                    unzipVec[offset + k * (ly * lx) + j * (lx) +
                                             i + index - centeredOffset] *
                                    h;
                        }

                if (blkNode.maxX() == m_uiMeshDomain_max) {
                    for (unsigned int k = paddWidth;
                         k < (blkNpe_1D - paddWidth); k++)
                        for (unsigned int j = paddWidth;
                             j < (blkNpe_1D - paddWidth); j++)
                            for (unsigned int i = (blkNpe_1D - 2 * paddWidth);
                                 i < (blkNpe_1D - paddWidth); i++) {
                                unzipVec1[offset + k * (ly * lx) + j * (lx) +
                                          i] = 0;
                                for (unsigned int index = 0;
                                     index < backward.getStencilLength();
                                     index++)
                                    unzipVec1[offset + k * (ly * lx) +
                                              j * (lx) + i] +=
                                        backward[index] *
                                        unzipVec[offset + k * (ly * lx) +
                                                 j * (lx) + i + index -
                                                 backwardOffset] *
                                        h;
                            }

                } else {
                    assert(blkNode.maxX() < m_uiMeshDomain_max);
                    for (unsigned int k = paddWidth;
                         k < (blkNpe_1D - paddWidth); k++)
                        for (unsigned int j = paddWidth;
                             j < (blkNpe_1D - paddWidth); j++)
                            for (unsigned int i = (blkNpe_1D - 2 * paddWidth);
                                 i < (blkNpe_1D - paddWidth); i++) {
                                unzipVec1[offset + k * (ly * lx) + j * (lx) +
                                          i] = 0;
                                for (unsigned int index = 0;
                                     index < centered.getStencilLength();
                                     index++)
                                    unzipVec1[offset + k * (ly * lx) +
                                              j * (lx) + i] +=
                                        centered[index] *
                                        unzipVec[offset + k * (ly * lx) +
                                                 j * (lx) + i + index -
                                                 centeredOffset] *
                                        h;
                            }
                }

            } else if (blkNode.maxX() == m_uiMeshDomain_max) {
                assert(blkNode.minX() > m_uiMeshDomain_min);
                assert((blkNpe_1D - 2 * paddWidth));
                // std::cout<<"rank: "<<m_uiActiveRank<<" applying backward
                // difference difference: "<<std::endl;
                for (unsigned int k = paddWidth; k < (blkNpe_1D - paddWidth);
                     k++)
                    for (unsigned int j = paddWidth;
                         j < (blkNpe_1D - paddWidth); j++)
                        for (unsigned int i = paddWidth;
                             i < (blkNpe_1D - 2 * paddWidth); i++) {
                            unzipVec1[offset + k * (ly * lx) + j * (lx) + i] =
                                0;
                            for (unsigned int index = 0;
                                 index < centered.getStencilLength(); index++)
                                unzipVec1[offset + k * (ly * lx) + j * (lx) +
                                          i] +=
                                    centered[index] *
                                    unzipVec[offset + k * (ly * lx) + j * (lx) +
                                             i + index - centeredOffset] *
                                    h;
                        }

                for (unsigned int k = paddWidth; k < (blkNpe_1D - paddWidth);
                     k++)
                    for (unsigned int j = paddWidth;
                         j < (blkNpe_1D - paddWidth); j++)
                        for (unsigned int i = (blkNpe_1D - 2 * paddWidth);
                             i < (blkNpe_1D - paddWidth); i++) {
                            unzipVec1[offset + k * (ly * lx) + j * (lx) + i] =
                                0;
                            for (unsigned int index = 0;
                                 index < backward.getStencilLength(); index++)
                                unzipVec1[offset + k * (ly * lx) + j * (lx) +
                                          i] +=
                                    backward[index] *
                                    unzipVec[offset + k * (ly * lx) + j * (lx) +
                                             i + index - backwardOffset] *
                                    h;
                        }

            } else {
                assert(blkNode.minX() > m_uiMeshDomain_min &&
                       blkNode.maxX() < m_uiMeshDomain_max);
                // std::cout<<"rank: "<<m_uiActiveRank<<" applying centered
                // difference difference: "<<std::endl;
                for (unsigned int k = paddWidth; k < (blkNpe_1D - paddWidth);
                     k++)
                    for (unsigned int j = paddWidth;
                         j < (blkNpe_1D - paddWidth); j++)
                        for (unsigned int i = paddWidth;
                             i < (blkNpe_1D - paddWidth); i++) {
                            unzipVec1[offset + k * (ly * lx) + j * (lx) + i] =
                                0;
                            for (unsigned int index = 0;
                                 index < centered.getStencilLength(); index++) {
                                unzipVec1[offset + k * (ly * lx) + j * (lx) +
                                          i] +=
                                    centered[index] *
                                    unzipVec[offset + k * (ly * lx) + j * (lx) +
                                             (i + index - centeredOffset)] *
                                    h;
                            }
                        }
            }
        }
    }

    if (centered.getStencilDirection() == StencilDirection::STENCIL_DIR_Y) {
        for (unsigned int blk = 0; blk < m_uiLocalBlockList.size(); blk++) {
            blkNode   = m_uiLocalBlockList[blk].getBlockNode();
            regLev    = m_uiLocalBlockList[blk].getRegularGridLev();
            // h=((1u<<(m_uiMaxDepth))*m_uiElementOrder)/((0.5-(-0.5)) *
            // ((1u<<(regLev-blkNode.getLevel()))));

            lx        = m_uiLocalBlockList[blk].getAllocationSzX();
            ly        = m_uiLocalBlockList[blk].getAllocationSzY();
            lz        = m_uiLocalBlockList[blk].getAllocationSzZ();
            offset    = m_uiLocalBlockList[blk].getOffset();
            paddWidth = m_uiLocalBlockList[blk].get1DPadWidth();

            h         = ((blkNode.maxY() - blkNode.minY())) /
                        ((double)(1u << (regLev - blkNode.getLevel())) *
                         m_uiElementOrder);
            h         = 1.0 / h;
            blkNpe_1D =
                m_uiElementOrder * (1u << (regLev - blkNode.getLevel())) + 1 +
                2 * paddWidth;
            assert(blkNpe_1D > paddWidth);

            if (blkNode.minY() == m_uiMeshDomain_min) {
                // std::cout<<"rank: "<<m_uiActiveRank<<" applying forward
                // difference difference: "<<std::endl;
                for (unsigned int k = paddWidth; k < (blkNpe_1D - paddWidth);
                     k++)
                    for (unsigned int i = paddWidth;
                         i < (blkNpe_1D - paddWidth); i++)
                        for (unsigned int j = paddWidth; j < 2 * paddWidth;
                             j++) {
                            unzipVec1[offset + k * (ly * lx) + j * (lx) + i] =
                                0;
                            for (unsigned int index = 0;
                                 index < forward.getStencilLength(); index++)
                                unzipVec1[offset + k * (ly * lx) + j * (lx) +
                                          i] +=
                                    forward[index] *
                                    unzipVec[offset + k * (ly * lx) +
                                             (j + index - forwardOffset) *
                                                 (lx) +
                                             i] *
                                    h;
                        }

                for (unsigned int k = paddWidth; k < (blkNpe_1D - paddWidth);
                     k++)
                    for (unsigned int i = paddWidth;
                         i < (blkNpe_1D - paddWidth); i++)
                        for (unsigned int j = 2 * paddWidth;
                             j < (blkNpe_1D - 2 * paddWidth); j++) {
                            unzipVec1[offset + k * (ly * lx) + j * (lx) + i] =
                                0;
                            for (unsigned int index = 0;
                                 index < centered.getStencilLength(); index++)
                                unzipVec1[offset + k * (ly * lx) + j * (lx) +
                                          i] +=
                                    centered[index] *
                                    unzipVec[offset + k * (ly * lx) +
                                             (j + index - centeredOffset) *
                                                 (lx) +
                                             i] *
                                    h;
                        }

                if (blkNode.maxY() == m_uiMeshDomain_max) {
                    for (unsigned int k = paddWidth;
                         k < (blkNpe_1D - paddWidth); k++)
                        for (unsigned int i = paddWidth;
                             i < (blkNpe_1D - paddWidth); i++)
                            for (unsigned int j = (blkNpe_1D - 2 * paddWidth);
                                 j < (blkNpe_1D - paddWidth); j++) {
                                unzipVec1[offset + k * (ly * lx) + j * (lx) +
                                          i] = 0;
                                for (unsigned int index = 0;
                                     index < backward.getStencilLength();
                                     index++)
                                    unzipVec1[offset + k * (ly * lx) +
                                              j * (lx) + i] +=
                                        backward[index] *
                                        unzipVec[offset + k * (ly * lx) +
                                                 (j + index - backwardOffset) *
                                                     (lx) +
                                                 i] *
                                        h;
                            }

                } else {
                    assert(blkNode.maxY() < m_uiMeshDomain_max);
                    for (unsigned int k = paddWidth;
                         k < (blkNpe_1D - paddWidth); k++)
                        for (unsigned int i = paddWidth;
                             i < (blkNpe_1D - paddWidth); i++)
                            for (unsigned int j = (blkNpe_1D - 2 * paddWidth);
                                 j < (blkNpe_1D - paddWidth); j++) {
                                unzipVec1[offset + k * (ly * lx) + j * (lx) +
                                          i] = 0;
                                for (unsigned int index = 0;
                                     index < centered.getStencilLength();
                                     index++)
                                    unzipVec1[offset + k * (ly * lx) +
                                              j * (lx) + i] +=
                                        centered[index] *
                                        unzipVec[offset + k * (ly * lx) +
                                                 (j + index - centeredOffset) *
                                                     (lx) +
                                                 i] *
                                        h;
                            }
                }

            } else if (blkNode.maxY() == m_uiMeshDomain_max) {
                assert(blkNode.minY() > m_uiMeshDomain_min);
                assert((blkNpe_1D - 2 * paddWidth));
                // std::cout<<"rank: "<<m_uiActiveRank<<" applying backward
                // difference difference: "<<std::endl;
                for (unsigned int k = paddWidth; k < (blkNpe_1D - paddWidth);
                     k++)
                    for (unsigned int i = paddWidth;
                         i < (blkNpe_1D - paddWidth); i++)
                        for (unsigned int j = paddWidth;
                             j < (blkNpe_1D - 2 * paddWidth); j++) {
                            unzipVec1[offset + k * (ly * lx) + j * (lx) + i] =
                                0;
                            for (unsigned int index = 0;
                                 index < centered.getStencilLength(); index++)
                                unzipVec1[offset + k * (ly * lx) + j * (lx) +
                                          i] +=
                                    centered[index] *
                                    unzipVec[offset + k * (ly * lx) +
                                             (j + index - centeredOffset) *
                                                 (lx) +
                                             i] *
                                    h;
                        }

                for (unsigned int k = paddWidth; k < (blkNpe_1D - paddWidth);
                     k++)
                    for (unsigned int i = paddWidth;
                         i < (blkNpe_1D - paddWidth); i++)
                        for (unsigned int j = (blkNpe_1D - 2 * paddWidth);
                             j < (blkNpe_1D - paddWidth); j++) {
                            unzipVec1[offset + k * (ly * lx) + j * (lx) + i] =
                                0;
                            for (unsigned int index = 0;
                                 index < backward.getStencilLength(); index++)
                                unzipVec1[offset + k * (ly * lx) + j * (lx) +
                                          i] +=
                                    backward[index] *
                                    unzipVec[offset + k * (ly * lx) +
                                             (j + index - backwardOffset) *
                                                 (lx) +
                                             i] *
                                    h;
                        }

            } else {
                assert(blkNode.minY() > m_uiMeshDomain_min &&
                       blkNode.maxY() < m_uiMeshDomain_max);
                // std::cout<<"rank: "<<m_uiActiveRank<<" applying centered
                // difference difference: "<<std::endl;
                for (unsigned int k = paddWidth; k < (blkNpe_1D - paddWidth);
                     k++)
                    for (unsigned int i = paddWidth;
                         i < (blkNpe_1D - paddWidth); i++)
                        for (unsigned int j = paddWidth;
                             j < (blkNpe_1D - paddWidth); j++) {
                            unzipVec1[offset + k * (ly * lx) + j * (lx) + i] =
                                0;
                            for (unsigned int index = 0;
                                 index < centered.getStencilLength(); index++) {
                                unzipVec1[offset + k * (ly * lx) + j * (lx) +
                                          i] +=
                                    centered[index] *
                                    unzipVec[offset + k * (ly * lx) +
                                             (j + index - centeredOffset) *
                                                 (lx) +
                                             (i)] *
                                    h;
                            }
                        }
            }
        }
    }

    if (centered.getStencilDirection() == StencilDirection::STENCIL_DIR_Z) {
        for (unsigned int blk = 0; blk < m_uiLocalBlockList.size(); blk++) {
            blkNode   = m_uiLocalBlockList[blk].getBlockNode();
            regLev    = m_uiLocalBlockList[blk].getRegularGridLev();
            // h=((1u<<(m_uiMaxDepth))*m_uiElementOrder)/((0.5-(-0.5)) *
            // ((1u<<(regLev-blkNode.getLevel()))));

            lx        = m_uiLocalBlockList[blk].getAllocationSzX();
            ly        = m_uiLocalBlockList[blk].getAllocationSzY();
            lz        = m_uiLocalBlockList[blk].getAllocationSzZ();
            offset    = m_uiLocalBlockList[blk].getOffset();
            paddWidth = m_uiLocalBlockList[blk].get1DPadWidth();

            h         = ((blkNode.maxZ() - blkNode.minZ())) /
                        ((double)(1u << (regLev - blkNode.getLevel())) *
                         m_uiElementOrder);
            h         = 1.0 / h;
            blkNpe_1D =
                m_uiElementOrder * (1u << (regLev - blkNode.getLevel())) + 1 +
                2 * paddWidth;
            assert(blkNpe_1D > paddWidth);

            if (blkNode.minZ() == m_uiMeshDomain_min) {
                // std::cout<<"rank: "<<m_uiActiveRank<<" applying forward
                // difference difference: "<<std::endl;
                for (unsigned int j = paddWidth; j < (blkNpe_1D - paddWidth);
                     j++)
                    for (unsigned int i = paddWidth;
                         i < (blkNpe_1D - paddWidth); i++)
                        for (unsigned int k = paddWidth; k < 2 * paddWidth;
                             k++) {
                            unzipVec1[offset + k * (ly * lx) + j * (lx) + i] =
                                0;
                            for (unsigned int index = 0;
                                 index < forward.getStencilLength(); index++)
                                unzipVec1[offset + k * (ly * lx) + j * (lx) +
                                          i] +=
                                    forward[index] *
                                    unzipVec[offset +
                                             (k + index - forwardOffset) *
                                                 (ly * lx) +
                                             (j) * (lx) + i] *
                                    h;
                        }

                for (unsigned int j = paddWidth; j < (blkNpe_1D - paddWidth);
                     j++)
                    for (unsigned int i = paddWidth;
                         i < (blkNpe_1D - paddWidth); i++)
                        for (unsigned int k = 2 * paddWidth;
                             k < (blkNpe_1D - 2 * paddWidth); k++) {
                            unzipVec1[offset + k * (ly * lx) + j * (lx) + i] =
                                0;
                            for (unsigned int index = 0;
                                 index < centered.getStencilLength(); index++)
                                unzipVec1[offset + k * (ly * lx) + j * (lx) +
                                          i] +=
                                    centered[index] *
                                    unzipVec[offset +
                                             (k + index - centeredOffset) *
                                                 (ly * lx) +
                                             (j) * (lx) + i] *
                                    h;
                        }

                if (blkNode.maxZ() == m_uiMeshDomain_max) {
                    for (unsigned int j = paddWidth;
                         j < (blkNpe_1D - paddWidth); j++)
                        for (unsigned int i = paddWidth;
                             i < (blkNpe_1D - paddWidth); i++)
                            for (unsigned int k = (blkNpe_1D - 2 * paddWidth);
                                 k < (blkNpe_1D - paddWidth); k++) {
                                unzipVec1[offset + k * (ly * lx) + j * (lx) +
                                          i] = 0;
                                for (unsigned int index = 0;
                                     index < backward.getStencilLength();
                                     index++)
                                    unzipVec1[offset + k * (ly * lx) +
                                              j * (lx) + i] +=
                                        backward[index] *
                                        unzipVec[offset +
                                                 (k + index - backwardOffset) *
                                                     (ly * lx) +
                                                 (j) * (lx) + i] *
                                        h;
                            }

                } else {
                    assert(blkNode.maxZ() < m_uiMeshDomain_max);
                    for (unsigned int j = paddWidth;
                         j < (blkNpe_1D - paddWidth); j++)
                        for (unsigned int i = paddWidth;
                             i < (blkNpe_1D - paddWidth); i++)
                            for (unsigned int k = (blkNpe_1D - 2 * paddWidth);
                                 k < (blkNpe_1D - paddWidth); k++) {
                                unzipVec1[offset + k * (ly * lx) + j * (lx) +
                                          i] = 0;
                                for (unsigned int index = 0;
                                     index < centered.getStencilLength();
                                     index++)
                                    unzipVec1[offset + k * (ly * lx) +
                                              j * (lx) + i] +=
                                        centered[index] *
                                        unzipVec[offset +
                                                 (k + index - centeredOffset) *
                                                     (ly * lx) +
                                                 (j) * (lx) + i] *
                                        h;
                            }
                }

            } else if (blkNode.maxZ() == m_uiMeshDomain_max) {
                assert(blkNode.minZ() > m_uiMeshDomain_min);
                assert((blkNpe_1D - 2 * paddWidth));
                // std::cout<<"rank: "<<m_uiActiveRank<<" applying backward
                // difference difference: "<<std::endl;
                for (unsigned int j = paddWidth; j < (blkNpe_1D - paddWidth);
                     j++)
                    for (unsigned int i = paddWidth;
                         i < (blkNpe_1D - paddWidth); i++)
                        for (unsigned int k = paddWidth;
                             k < (blkNpe_1D - 2 * paddWidth); k++) {
                            unzipVec1[offset + k * (ly * lx) + j * (lx) + i] =
                                0;
                            for (unsigned int index = 0;
                                 index < centered.getStencilLength(); index++)
                                unzipVec1[offset + k * (ly * lx) + j * (lx) +
                                          i] +=
                                    centered[index] *
                                    unzipVec[offset +
                                             (k + index - centeredOffset) *
                                                 (ly * lx) +
                                             (j) * (lx) + i] *
                                    h;
                        }

                for (unsigned int j = paddWidth; j < (blkNpe_1D - paddWidth);
                     j++)
                    for (unsigned int i = paddWidth;
                         i < (blkNpe_1D - paddWidth); i++)
                        for (unsigned int k = (blkNpe_1D - 2 * paddWidth);
                             k < (blkNpe_1D - paddWidth); k++) {
                            unzipVec1[offset + k * (ly * lx) + j * (lx) + i] =
                                0;
                            for (unsigned int index = 0;
                                 index < backward.getStencilLength(); index++)
                                unzipVec1[offset + k * (ly * lx) + j * (lx) +
                                          i] +=
                                    backward[index] *
                                    unzipVec[offset +
                                             (k + index - backwardOffset) *
                                                 (ly * lx) +
                                             (j) * (lx) + i] *
                                    h;
                        }

            } else {
                assert(blkNode.minZ() > m_uiMeshDomain_min &&
                       blkNode.maxZ() < m_uiMeshDomain_max);
                // std::cout<<"rank: "<<m_uiActiveRank<<" applying centered
                // difference difference: "<<std::endl;
                for (unsigned int j = paddWidth; j < (blkNpe_1D - paddWidth);
                     j++)
                    for (unsigned int i = paddWidth;
                         i < (blkNpe_1D - paddWidth); i++)
                        for (unsigned int k = paddWidth;
                             k < (blkNpe_1D - paddWidth); k++) {
                            unzipVec1[offset + k * (ly * lx) + j * (lx) + i] =
                                0;
                            for (unsigned int index = 0;
                                 index < centered.getStencilLength(); index++) {
                                unzipVec1[offset + k * (ly * lx) + j * (lx) +
                                          i] +=
                                    centered[index] *
                                    unzipVec[offset +
                                             (k + index - centeredOffset) *
                                                 (ly * lx) +
                                             (j) * (lx) + (i)] *
                                    h;
                            }
                        }
            }
        }
    }

#ifdef PROFILE_APPLY_STENCIL
    t2 = std::chrono::high_resolution_clock::now();
    t_stencil =
        std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    par::Mpi_Reduce(&t_stencil, t_stencil_g, 1, MPI_MIN, 0, m_uiCommActive);
    par::Mpi_Reduce(&t_stencil, t_stencil_g + 1, 1, MPI_SUM, 0, m_uiCommActive);
    par::Mpi_Reduce(&t_stencil, t_stencil_g + 2, 1, MPI_MIN, 0, m_uiCommActive);
    t_stencil_g[1] = t_stencil_g[1] / (double)m_uiActiveNpes;

    t1             = std::chrono::high_resolution_clock::now();
#endif
    this->createVector(out);
    this->zip(&(*(unzipVec1.begin())), &(*(out.begin())));

#ifdef PROFILE_APPLY_STENCIL
    t2 = std::chrono::high_resolution_clock::now();
    t_zip =
        std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    par::Mpi_Reduce(&t_zip, t_zip_g, 1, MPI_MIN, 0, m_uiCommActive);
    par::Mpi_Reduce(&t_zip, t_zip_g + 1, 1, MPI_SUM, 0, m_uiCommActive);
    par::Mpi_Reduce(&t_zip, t_zip_g + 2, 1, MPI_MIN, 0, m_uiCommActive);
    t_zip_g[1] = t_zip_g[1] / (double)m_uiActiveNpes;

    if (!m_uiActiveRank) {
        std::cout << "unzip_max \t stencil_max \t zip_max " << std::endl;
        std::cout << t_uzip_g[1] << " \t " << t_stencil_g[1] << " \t "
                  << t_zip_g[1] << std::endl;
    }

#endif
    unzipVec1.clear();
    unzipVec.clear();
}

template <typename pKey, typename pNode>
void Mesh::searchKeys(std::vector<pKey>& pKeys, std::vector<pNode>& pNodes) {
    assert(seq::test::isSorted(pNodes));

    std::vector<Key> pKeys_cpy;
    pKeys_cpy.resize(pKeys.size());

    for (unsigned int k = 0; k < pKeys.size(); k++) {
        pKeys_cpy[k] = pKeys[k];
        pKeys_cpy[k].addOwner(k);
        pKeys_cpy[k].setSearchResult(LOOK_UP_TABLE_DEFAULT);
    }

    SFC::seqSearch::SFC_treeSearch(
        &(*(pKeys_cpy.begin())), &(*(pNodes.begin())), 0, pKeys_cpy.size(), 0,
        pNodes.size(), m_uiMaxDepth, m_uiMaxDepth, ROOT_ROTATION);

    for (unsigned int k = 0; k < pKeys_cpy.size(); k++) {
        if ((pKeys_cpy[k].getFlag() & OCT_FOUND)) {
            pKeys[(*(pKeys_cpy[k].getOwnerList()))[0]].setSearchResult(
                pKeys_cpy[k].getSearchResult());
            pKeys[k].setFlag((pKeys[k].getFlag() | OCT_FOUND));
        }
    }

    pKeys_cpy.clear();
}

template <typename T>
bool Mesh::isReMeshUnzip(
    const T** unzippedVec, const unsigned int* varIds,
    const unsigned int numVars,
    std::function<double(double, double, double, double*)> wavelet_tol,
    double amr_coarse_fac, double coarsen_hx) {
    // This is the default isRMesh code that is used as refiment criteria. (if
    // needed some complicated application specific refinement routine please
    // have a look at the waveletAMR.h(tcc) file. )

    // new wavelet code goes here.
    bool isMeshGlobalChanged = false;
    bool isMeshLocalChanged  = false;
    // std::cout<<"calling amr"<<std::endl;
    const bool includeBdy =
        true;  // change this to false to exclude boundary from AMR.
    std::vector<unsigned int> refine_flags;

    if (this->isActive()) {
        RefElement* refEl                     = &m_uiRefEl;

        const std::vector<ot::Block>& blkList = this->getLocalBlockList();
        const unsigned int eOrder             = m_uiElementOrder;

        const unsigned int numLocalElements   = m_uiNumLocalElements;

        refine_flags.clear();
        refine_flags.resize(numLocalElements, OCT_NO_CHANGE);

        const ot::TreeNode* pNodes = m_uiAllElements.data();

        std::vector<double> eleWMax;
        eleWMax.resize(numLocalElements, 0);

        const unsigned int eleOfst       = m_uiElementLocalBegin;
        const size_t n_blocks            = blkList.size();
        const unsigned int nx_uniform    = (2 * eOrder + 1);
        const unsigned int sz_per_dof    = nx_uniform * nx_uniform * nx_uniform;
        const unsigned int isz_uniform[] = {nx_uniform, nx_uniform, nx_uniform};
        // size of im_vec1/im_vec2 the refEl needs for I3D_Parent2Child
        const unsigned int nPe = (eOrder + 1) * (eOrder + 1) * (eOrder + 1);

#if defined(DENDRO_UNZIP_OMP)
// Block-parallel: each thread owns its own WaveletEl (which has its
// own m_uiVIn/m_uiNVec/m_uiVOut workspaces) and its own im1/im2
// scratch for the underlying I3D_Parent2Child calls (passed via the
// thread-safe overload).
#pragma omp parallel
        {
            wavelet::WaveletEl wrefEl_tls(refEl);
            std::vector<double> im1_tls(nPe), im2_tls(nPe);
            std::vector<T> blkIn_tls(numVars * sz_per_dof);
            std::vector<double> wCout_tls(sz_per_dof);

#pragma omp for schedule(dynamic, 1)
            for (size_t blk = 0; blk < n_blocks; blk++) {
#else
        {
            // serial fallback — reuse the same per-block scratch vectors
            wavelet::WaveletEl wrefEl_tls(refEl);
            std::vector<double> im1_tls(nPe), im2_tls(nPe);
            std::vector<T> blkIn_tls(numVars * sz_per_dof);
            std::vector<double> wCout_tls(sz_per_dof);

            for (size_t blk = 0; blk < n_blocks; blk++) {
#endif
                const unsigned int pw = blkList[blk].get1DPadWidth();
                if ((eOrder >> 1u) != pw) {
                    std::cout
                        << " padding width should be half the eleOrder for "
                           "generic wavelet computations. "
                        << std::endl;
                    MPI_Abort(this->getMPICommunicator(), 0);
                }

                for (unsigned int ele = blkList[blk].getLocalElementBegin();
                     ele < blkList[blk].getLocalElementEnd(); ele++) {
                    const bool isBdyOct = this->isBoundaryOctant(ele);

                    const double oct_dx =
                        (1u << (m_uiMaxDepth - pNodes[ele].getLevel())) /
                        (double(m_uiElementOrder));
                    Point oct_pt1 =
                        Point(pNodes[ele].minX(), pNodes[ele].minY(),
                              pNodes[ele].minZ());
                    Point oct_pt2 = Point(pNodes[ele].minX() + oct_dx,
                                          pNodes[ele].minY() + oct_dx,
                                          pNodes[ele].minZ() + oct_dx);
                    Point domain_pt1, domain_pt2, dx_domain;
                    this->octCoordToDomainCoord(oct_pt1, domain_pt1);
                    this->octCoordToDomainCoord(oct_pt2, domain_pt2);
                    dx_domain            = domain_pt2 - domain_pt1;
                    double hx[3]         = {dx_domain.x(), dx_domain.y(),
                                            dx_domain.z()};
                    const double tol_ele = wavelet_tol(
                        domain_pt1.x(), domain_pt1.y(), domain_pt1.z(), hx);

                    if (!includeBdy && isBdyOct) {
                        eleWMax[ele - eleOfst] =
                            amr_coarse_fac * tol_ele + 1e-8;
                        continue;
                    }

                    for (unsigned int v = 0; v < numVars; v++) {
                        const unsigned int vid = varIds[v];
                        this->getUnzipElementalNodalValues(
                            unzippedVec[vid], blk, ele,
                            blkIn_tls.data() + v * sz_per_dof, true);
                    }

                    double wMax = 0.0;
                    for (unsigned int v = 0; v < numVars; v++) {
#if defined(DENDRO_UNZIP_OMP)
                        // thread-safe variant (uses external im1/im2)
                        wrefEl_tls.compute_wavelets_3D(
                            (double*)(blkIn_tls.data() + v * sz_per_dof),
                            isz_uniform, wCout_tls, isBdyOct, im1_tls.data(),
                            im2_tls.data());
#else
                        wrefEl_tls.compute_wavelets_3D(
                            (double*)(blkIn_tls.data() + v * sz_per_dof),
                            isz_uniform, wCout_tls, isBdyOct);
#endif
                        const double l_max =
                            (normL2(wCout_tls.data(), wCout_tls.size())) /
                            sqrt((double)wCout_tls.size());
                        if (wMax < l_max) wMax = l_max;
                        if (wMax > tol_ele) break;
                    }
                    eleWMax[ele - eleOfst] = wMax;
                }
            }
        }

        // mark elements for refinement first.
        for (unsigned int ele = m_uiElementLocalBegin;
             ele < m_uiElementLocalEnd; ele++) {
            const double oct_dx =
                (1u << (m_uiMaxDepth - pNodes[ele].getLevel())) /
                (double(m_uiElementOrder));
            Point oct_pt1 = Point(pNodes[ele].minX(), pNodes[ele].minY(),
                                  pNodes[ele].minZ());
            Point oct_pt2 =
                Point(pNodes[ele].minX() + oct_dx, pNodes[ele].minY() + oct_dx,
                      pNodes[ele].minZ() + oct_dx);
            Point domain_pt1, domain_pt2, dx_domain;
            this->octCoordToDomainCoord(oct_pt1, domain_pt1);
            this->octCoordToDomainCoord(oct_pt2, domain_pt2);
            dx_domain    = domain_pt2 - domain_pt1;
            double hx[3] = {dx_domain.x(), dx_domain.y(), dx_domain.z()};
            const double tol_ele =
                wavelet_tol(domain_pt1.x(), domain_pt1.y(), domain_pt1.z(), hx);

            const double l_max = eleWMax[ele - eleOfst];

            if (l_max > tol_ele) {
                refine_flags[(ele - eleOfst)] = OCT_SPLIT;
                isMeshLocalChanged            = true;

            } else if (l_max < amr_coarse_fac * tol_ele) {
                refine_flags[ele - eleOfst] = OCT_COARSE;
                isMeshLocalChanged          = true;

            } else {
                refine_flags[ele - eleOfst] = OCT_NO_CHANGE;
            }
        }

        if (isMeshLocalChanged)
            isMeshLocalChanged = this->setMeshRefinementFlags(refine_flags);
    }

    // par::Mpi_Allreduce(&isMeshLocalChanged,&isMeshGlobalChanged,1,MPI_LOR,this->getMPIGlobalCommunicator());
    MPI_Allreduce(&isMeshLocalChanged, &isMeshGlobalChanged, 1, MPI_CXX_BOOL,
                  MPI_LOR, this->getMPIGlobalCommunicator());
    return isMeshGlobalChanged;

// old remesh code hard coded oly for 4th order interp. for refine wavelets and
// 3rd order for coarsen wavelets. (not encouraged to use :) )
#if 0
        bool isOctChange=false;
        if(m_uiIsActive)
        {
            // remove all the previously set falgs if there is any.  THIS will change the all flags to no CHANGE
            for(unsigned int ele=m_uiElementLocalBegin;ele<m_uiElementLocalEnd;ele++)
                m_uiAllElements[ele].setFlag(((OCT_NO_CHANGE<<NUM_LEVEL_BITS)|m_uiAllElements[ele].getLevel()));

            ot::TreeNode blkNode;
            unsigned int sz[3];
            double dh[3];
            unsigned int bflag,offset;
            unsigned int regLev;
            unsigned int eIndex[3];
            double *  waveletR = NULL;
            double *  waveletC = NULL;
            unsigned int num_wr =0 ,num_wc =0;

            double * wsIn = new double[m_uiNpE];
            double * wsOut = new double[m_uiNpE];
            double ** ws = new double*[2];
            ws[0] = wsIn;
            ws[1] = wsOut;

            // upper bound for the refine and coarsen wavelets.     
            waveletR = new double[64];
            num_wr = 64;
            
            waveletC = new double[64];
            num_wc = 64 ;

            const unsigned int paddWidth=3;
            unsigned int eleIndexMin=0,eleIndexMax=0;

            double l_inf;
            double x,y,z,tol;


            // first pass to identify the refined elements.
            for(unsigned blk=0;blk<m_uiLocalBlockList.size();blk++)
            {

                blkNode=m_uiLocalBlockList[blk].getBlockNode();

                sz[0]=m_uiLocalBlockList[blk].getAllocationSzX();
                sz[1]=m_uiLocalBlockList[blk].getAllocationSzY();
                sz[2]=m_uiLocalBlockList[blk].getAllocationSzZ();

                bflag=m_uiLocalBlockList[blk].getBlkNodeFlag();
                offset=m_uiLocalBlockList[blk].getOffset();

                regLev=m_uiLocalBlockList[blk].getRegularGridLev();
                eleIndexMax=(1u<<(regLev-blkNode.getLevel()))-1;

                //if(bflag!=0) continue;

                for(unsigned int ele=m_uiLocalBlockList[blk].getLocalElementBegin();ele<m_uiLocalBlockList[blk].getLocalElementEnd();ele++)
                {

                    if((m_uiAllElements[ele].getLevel()+MAXDEAPTH_LEVEL_DIFF+1)>=m_uiMaxDepth) continue;

                    x=m_uiAllElements[ele].getX();
                    y=m_uiAllElements[ele].getY();
                    z=m_uiAllElements[ele].getZ();
                    tol=wavelet_tol(x,y,z);

                    eIndex[0]=(m_uiAllElements[ele].getX()-blkNode.getX())>>(m_uiMaxDepth-regLev);
                    eIndex[1]=(m_uiAllElements[ele].getY()-blkNode.getY())>>(m_uiMaxDepth-regLev);
                    eIndex[2]=(m_uiAllElements[ele].getZ()-blkNode.getZ())>>(m_uiMaxDepth-regLev);

                    if((bflag &(1u<<OCT_DIR_LEFT)) && eIndex[0]==eleIndexMin)   continue;
                    if((bflag &(1u<<OCT_DIR_DOWN)) && eIndex[1]==eleIndexMin)   continue;
                    if((bflag &(1u<<OCT_DIR_BACK)) && eIndex[2]==eleIndexMin)   continue;

                    if((bflag &(1u<<OCT_DIR_RIGHT)) && eIndex[0]==eleIndexMax)  continue;
                    if((bflag &(1u<<OCT_DIR_UP)) && eIndex[1]==eleIndexMax)     continue;
                    if((bflag &(1u<<OCT_DIR_FRONT)) && eIndex[2]==eleIndexMax)  continue;

                    for(unsigned int var=0;var<numVars;var++)
                    {

                         refine_wavelets(&unzippedVec[varIds[var]][offset],m_uiElementOrder,eIndex,paddWidth,sz,waveletR,num_wr,(double**)ws);
                        //  for(unsigned int k=0; k<4; k+=3)
                        //      for(unsigned int j=0; j<4; j+=3)
                        //       for(unsigned int i=0; i<4; i+=3)                                
                        //         waveletR[k*16 + j*4 + i] =0;

                         l_inf=normLInfty(waveletR,num_wr);
                         //l_inf = normL2(waveletR,num_wr)/num_wr;

                            // for(unsigned int k=1; k<3; k+=1)
                            //   for(unsigned int j=1; j<3; j+=1)
                            //    for(unsigned int i=1; i<3; i+=1)
                            //     std::cout<<"ref1: (i,j,k) : " << (i-1)<<" , "<<(j-1)<<" , "<<(k-1)<<": "<<waveletR[k*16 + j*4 + i]<<std::endl;
                            
                        
                       
                        // computeRefineWavelets(&unzippedVec[varIds[var]][offset],0,m_uiElementOrder,eIndex,paddWidth,sz,waveletR);
                        // l_inf=normLInfty(waveletR,NUM_REFINE_WAVELET_COEF);

                        //     for(unsigned int k=1; k<3; k+=1)
                        //       for(unsigned int j=1; j<3; j+=1)
                        //        for(unsigned int i=1; i<3; i+=1)
                        //         std::cout<<"ref2: (i,j,k) : " << (i-1)<<" , "<<(j-1)<<" , "<<(k-1)<<": "<<waveletR[(k-1)*4 + (j-1)*2 +i-1]<<std::endl;

                        if(l_inf>tol)
                        {
                            // for(unsigned int k=0;k<num_wr;k++)
                            //    std::cout<<"elem: "<<m_uiAllElements[ele]<<" wr["<<k<<"]: "<<waveletR[k]<<std::endl;
                            assert((m_uiAllElements[ele].getLevel()+MAXDEAPTH_LEVEL_DIFF+1)<m_uiMaxDepth);
                            //std::cout<<"rank: "<<m_uiActiveRank<<" element R: "<<m_uiAllElements[ele]<<" w_tol: "<<l_inf<<std::endl;
                            m_uiAllElements[ele].setFlag(((OCT_SPLIT<<NUM_LEVEL_BITS)|m_uiAllElements[ele].getLevel()));
                            assert((m_uiAllElements[ele].getFlag()>>NUM_LEVEL_BITS)==OCT_SPLIT);
                            break; // no point checking for the other variables since this octree needs to be refined.
                        }

                    }



                }


            }

            //second pass to identify the coarsening elements.
            for(unsigned blk=0;blk<m_uiLocalBlockList.size();blk++)
            {

                blkNode=m_uiLocalBlockList[blk].getBlockNode();

                sz[0]=m_uiLocalBlockList[blk].getAllocationSzX();
                sz[1]=m_uiLocalBlockList[blk].getAllocationSzY();
                sz[2]=m_uiLocalBlockList[blk].getAllocationSzZ();

                bflag=m_uiLocalBlockList[blk].getBlkNodeFlag();
                offset=m_uiLocalBlockList[blk].getOffset();

                regLev=m_uiLocalBlockList[blk].getRegularGridLev();
                eleIndexMax=(1u<<(regLev-blkNode.getLevel()))-1;

                dh[0]=coarsen_hx*(m_uiLocalBlockList[blk].computeGridDx());
                dh[1]=coarsen_hx*(m_uiLocalBlockList[blk].computeGridDy());
                dh[2]=coarsen_hx*(m_uiLocalBlockList[blk].computeGridDz());


                if((eleIndexMax==0) || (bflag!=0)) continue; // this implies the blocks with only 1 child and boundary blocks.

                bool isEligibleCoarsen=true;
                bool isCoarsen=true;
                ot::TreeNode tmpOct;

                for(unsigned int ele=m_uiLocalBlockList[blk].getLocalElementBegin();ele<m_uiLocalBlockList[blk].getLocalElementEnd();ele+=NUM_CHILDREN)
                {

                    assert(m_uiAllElements[ele].getParent()==m_uiAllElements[ele+NUM_CHILDREN-1].getParent());

                    isEligibleCoarsen=true;
                    for(unsigned int child=0;child<NUM_CHILDREN;child++)
                    {
                        if((m_uiAllElements[ele+child].getFlag()>>NUM_LEVEL_BITS)==OCT_SPLIT)
                        {
                            isEligibleCoarsen=false;
                            break;
                        }

                    }

                    if((isEligibleCoarsen) && (m_uiAllElements[ele].getLevel()>1))
                    {
                        tmpOct=m_uiAllElements[ele].getParent();
                        x=tmpOct.getX() + (1u<<(m_uiMaxDepth-tmpOct.getLevel()-1));
                        y=tmpOct.getY() + (1u<<(m_uiMaxDepth-tmpOct.getLevel()-1));
                        z=tmpOct.getZ() + (1u<<(m_uiMaxDepth-tmpOct.getLevel()-1));
                        tol=wavelet_tol(x,y,z);
                        tmpOct=ot::TreeNode(tmpOct.getX(),tmpOct.getY(),tmpOct.getZ(),tmpOct.getLevel()+1,m_uiDim,m_uiMaxDepth);

                        for(unsigned int child=0;child<NUM_CHILDREN;child++)
                        {
                            if(tmpOct==m_uiAllElements[ele+child])
                            {
                                eIndex[0]=(m_uiAllElements[ele+child].getX()-blkNode.getX())>>(m_uiMaxDepth-regLev);
                                eIndex[1]=(m_uiAllElements[ele+child].getY()-blkNode.getY())>>(m_uiMaxDepth-regLev);
                                eIndex[2]=(m_uiAllElements[ele+child].getZ()-blkNode.getZ())>>(m_uiMaxDepth-regLev);

                                break;
                            }

                        }

                        isCoarsen=true;

                        for(unsigned int var=0;var<numVars;var++)
                        {
                            coarsen_wavelets(&unzippedVec[varIds[var]][offset],m_uiElementOrder,eIndex,paddWidth,sz,waveletC,num_wc,(double**)ws);
                            //computeCoarsenWavelets(unzippedVec[varIds[var]],offset,m_uiElementOrder,eIndex,paddWidth,sz,waveletC);
                            l_inf=normLInfty(waveletC,NUM_COARSE_WAVELET_COEF);
                            //l_inf=normLInfty(waveletC,num_wc);
                            //l_inf = normL2(waveletC,num_wc)/num_wc;
                            if(l_inf>amr_coarse_fac*tol)
                            {
                                isCoarsen=false;
                                break;
                            }

                        }


                        if(isCoarsen)
                        {

                            for(unsigned int child=0;child<NUM_CHILDREN;child++)
                            {
                                m_uiAllElements[ele+child].setFlag(((OCT_COARSE<<NUM_LEVEL_BITS)|m_uiAllElements[ele].getLevel()));
                                assert((m_uiAllElements[ele].getFlag()>>NUM_LEVEL_BITS)==OCT_COARSE);
                                //std::cout<<"rank: "<<m_uiActiveRank<<" element C: "<<m_uiAllElements[ele]<<" is coarsening "<<l_inf<<std::endl;
                            }

                        }


                    }

                }


            }

            delete [] waveletR;
            delete [] waveletC;
            delete [] wsIn;
            delete [] wsOut;
            delete [] ws;
            
            isOctChange=false;
            for(unsigned int ele=m_uiElementLocalBegin;ele<m_uiElementLocalEnd;ele++)
                if((m_uiAllElements[ele].getFlag()>>NUM_LEVEL_BITS)!=OCT_NO_CHANGE)//if((m_uiAllElements[ele].getFlag()>>NUM_LEVEL_BITS)==OCT_SPLIT) // trigger remesh only when some refinement occurs (laid back remesh :)  ) //if((m_uiAllElements[ele].getFlag()>>NUM_LEVEL_BITS)!=OCT_NO_CHANGE)
                {
                    isOctChange=true;
                    break;
                }

        }

        bool isOctChanged_g;
        MPI_Allreduce(&isOctChange,&isOctChanged_g,1,MPI_CXX_BOOL,MPI_LOR,m_uiCommGlobal);
        //if(!m_uiGlobalRank) std::cout<<"is oct changed: "<<isOctChanged_g<<std::endl;
        return isOctChanged_g;
#endif
}

// Original-signature wrapper: routes through the thread-safe overload using
// RefElement's shared im_vec1/im_vec2 as scratch. NOT safe to call
// concurrently from multiple threads (use the explicit-scratch overload
// for that).
template <typename T>
void Mesh::getElementNodalValues(const T* vec, T* nodalValues,
                                 unsigned int elementID, bool isDGVec) const {
    this->getElementNodalValues(vec, nodalValues, elementID, isDGVec,
                                m_uiRefEl.getImVec1(), m_uiRefEl.getImVec2());
}

template <typename T>
void Mesh::getElementNodalValues(const T* vec, T* nodalValues,
                                 unsigned int elementID, bool isDGVec,
                                 double* im1, double* im2, bool allowWide,
                                 const T* allDg, size_t allDgEleStride) const {
    if (!m_uiIsActive) return;

    // handles the element get nodal values if the vec is an element DG vector.
    if (isDGVec) {
        for (unsigned int node = 0; node < m_uiNpE; node++)
            nodalValues[node] = vec[elementID * m_uiNpE + node];

        return;
    }

#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
    dendro::timer::t_unzip_sync_nodalval.start();
#endif
    // thread_local scratch: allocated once per thread, not per element (assign()
    // below reuses + zero-fills, so bit-identical to the old resize-from-empty).
    static thread_local std::vector<T> edgeInpIn;
    static thread_local std::vector<T> edgeInpOut;

    static thread_local std::vector<T> faceInpIn;
    static thread_local std::vector<T> faceInpOut;

    unsigned int cnum;
    bool isHanging;

    std::vector<unsigned int> edgeIndex;
    std::vector<unsigned int> faceIndex;

    bool nodeStatus[OCT_DIR_TOTAL];
    for (unsigned int w = 0; w < OCT_DIR_TOTAL; w++) nodeStatus[w] = false;

    edgeInpIn.assign((m_uiElementOrder + 1), T{});
    edgeInpOut.assign((m_uiElementOrder + 1), T{});

    faceInpIn.assign((m_uiElementOrder + 1) * (m_uiElementOrder + 1), T{});
    faceInpOut.assign((m_uiElementOrder + 1) * (m_uiElementOrder + 1), T{});

    for (unsigned int k = 1; k < (m_uiElementOrder); k++)
        for (unsigned int j = 1; j < (m_uiElementOrder); j++)
            for (unsigned int i = 1; i < (m_uiElementOrder); i++) {
                nodalValues[k * (m_uiElementOrder + 1) *
                                (m_uiElementOrder + 1) +
                            j * (m_uiElementOrder + 1) + i] =
                    vec[m_uiE2NMapping_CG[elementID * m_uiNpE +
                                          k * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          j * (m_uiElementOrder + 1) + i]];
                assert(!(this->isNodeHanging(
                    elementID, i, j,
                    k)));  // internal nodes cannot be hangging.
            }
    nodeStatus[OCT_DIR_INTERNAL] = true;

    // face interpolations
    // face : OCT_DIR_LEFT (1)
    isHanging = this->isFaceHanging(elementID, OCT_DIR_LEFT, cnum);
    if (isHanging) {
        faceNodesIndex(elementID, OCT_DIR_LEFT, faceIndex, false);
        for (unsigned int index = 0; index < faceIndex.size(); index++)
            faceInpIn[index] = vec[m_uiE2NMapping_CG[faceIndex[index]]];

        if (!(allowWide &&
              this->prolongateHangingFaceWide(vec, elementID, OCT_DIR_LEFT, cnum,
                                              &(*(faceInpOut.begin())), im1,
                                              im2, allDg, allDgEleStride)))
            this->parent2ChildInterpolation(&(*(faceInpIn.begin())),
                                            &(*(faceInpOut.begin())), cnum, 2,
                                            im1, im2);

        for (unsigned int k = 0; k < (m_uiElementOrder + 1); k++)
            for (unsigned int j = 0; j < (m_uiElementOrder + 1); j++)
                nodalValues[k * (m_uiElementOrder + 1) *
                                (m_uiElementOrder + 1) +
                            j * (m_uiElementOrder + 1) + 0] =
                    faceInpOut[k * (m_uiElementOrder + 1) + j];

        nodeStatus[OCT_DIR_LEFT_DOWN]       = true;
        nodeStatus[OCT_DIR_LEFT_UP]         = true;
        nodeStatus[OCT_DIR_LEFT_BACK]       = true;
        nodeStatus[OCT_DIR_LEFT_FRONT]      = true;

        nodeStatus[OCT_DIR_LEFT_DOWN_BACK]  = true;
        nodeStatus[OCT_DIR_LEFT_UP_BACK]    = true;
        nodeStatus[OCT_DIR_LEFT_UP_FRONT]   = true;
        nodeStatus[OCT_DIR_LEFT_DOWN_FRONT] = true;

    } else {
        for (unsigned int k = 1; k < m_uiElementOrder; k++)
            for (unsigned int j = 1; j < m_uiElementOrder; j++)
                nodalValues[k * (m_uiElementOrder + 1) *
                                (m_uiElementOrder + 1) +
                            j * (m_uiElementOrder + 1) + 0] =
                    vec[m_uiE2NMapping_CG[elementID * m_uiNpE +
                                          k * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          j * (m_uiElementOrder + 1) + 0]];
    }

    // face : OCT_DIR_RIGHT (2)
    isHanging = this->isFaceHanging(elementID, OCT_DIR_RIGHT, cnum);
    if (isHanging) {
        faceNodesIndex(elementID, OCT_DIR_RIGHT, faceIndex, false);
        for (unsigned int index = 0; index < faceIndex.size(); index++)
            faceInpIn[index] = vec[m_uiE2NMapping_CG[faceIndex[index]]];

        if (!(allowWide &&
              this->prolongateHangingFaceWide(vec, elementID, OCT_DIR_RIGHT, cnum,
                                              &(*(faceInpOut.begin())), im1,
                                              im2, allDg, allDgEleStride)))
            this->parent2ChildInterpolation(&(*(faceInpIn.begin())),
                                            &(*(faceInpOut.begin())), cnum, 2,
                                            im1, im2);

        for (unsigned int k = 0; k < (m_uiElementOrder + 1); k++)
            for (unsigned int j = 0; j < (m_uiElementOrder + 1); j++)
                nodalValues[k * (m_uiElementOrder + 1) *
                                (m_uiElementOrder + 1) +
                            j * (m_uiElementOrder + 1) + m_uiElementOrder] =
                    faceInpOut[k * (m_uiElementOrder + 1) + j];

        nodeStatus[OCT_DIR_RIGHT_DOWN]       = true;
        nodeStatus[OCT_DIR_RIGHT_UP]         = true;
        nodeStatus[OCT_DIR_RIGHT_BACK]       = true;
        nodeStatus[OCT_DIR_RIGHT_FRONT]      = true;

        nodeStatus[OCT_DIR_RIGHT_DOWN_BACK]  = true;
        nodeStatus[OCT_DIR_RIGHT_UP_BACK]    = true;
        nodeStatus[OCT_DIR_RIGHT_UP_FRONT]   = true;
        nodeStatus[OCT_DIR_RIGHT_DOWN_FRONT] = true;

    } else {
        for (unsigned int k = 1; k < m_uiElementOrder; k++)
            for (unsigned int j = 1; j < m_uiElementOrder; j++)
                nodalValues[k * (m_uiElementOrder + 1) *
                                (m_uiElementOrder + 1) +
                            j * (m_uiElementOrder + 1) + m_uiElementOrder] =
                    vec[m_uiE2NMapping_CG[elementID * m_uiNpE +
                                          k * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          j * (m_uiElementOrder + 1) +
                                          m_uiElementOrder]];
    }

    // face : OCT_DIR_DOWN (3)
    isHanging = this->isFaceHanging(elementID, OCT_DIR_DOWN, cnum);
    if (isHanging) {
        faceNodesIndex(elementID, OCT_DIR_DOWN, faceIndex, false);
        for (unsigned int index = 0; index < faceIndex.size(); index++)
            faceInpIn[index] = vec[m_uiE2NMapping_CG[faceIndex[index]]];

        if (!(allowWide &&
              this->prolongateHangingFaceWide(vec, elementID, OCT_DIR_DOWN, cnum,
                                              &(*(faceInpOut.begin())), im1,
                                              im2, allDg, allDgEleStride)))
            this->parent2ChildInterpolation(&(*(faceInpIn.begin())),
                                            &(*(faceInpOut.begin())), cnum, 2,
                                            im1, im2);

        for (unsigned int k = 0; k < (m_uiElementOrder + 1); k++)
            for (unsigned int i = 0; i < (m_uiElementOrder + 1); i++)
                nodalValues[k * (m_uiElementOrder + 1) *
                                (m_uiElementOrder + 1) +
                            0 * (m_uiElementOrder + 1) + i] =
                    faceInpOut[k * (m_uiElementOrder + 1) + i];

        nodeStatus[OCT_DIR_RIGHT_DOWN]       = true;
        nodeStatus[OCT_DIR_LEFT_DOWN]        = true;
        nodeStatus[OCT_DIR_DOWN_BACK]        = true;
        nodeStatus[OCT_DIR_DOWN_FRONT]       = true;

        nodeStatus[OCT_DIR_LEFT_DOWN_BACK]   = true;
        nodeStatus[OCT_DIR_RIGHT_DOWN_BACK]  = true;
        nodeStatus[OCT_DIR_RIGHT_DOWN_FRONT] = true;
        nodeStatus[OCT_DIR_LEFT_DOWN_FRONT]  = true;

    } else {
        for (unsigned int k = 1; k < m_uiElementOrder; k++)
            for (unsigned int i = 1; i < m_uiElementOrder; i++)
                nodalValues[k * (m_uiElementOrder + 1) *
                                (m_uiElementOrder + 1) +
                            0 * (m_uiElementOrder + 1) + i] =
                    vec[m_uiE2NMapping_CG[elementID * m_uiNpE +
                                          k * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          0 * (m_uiElementOrder + 1) + i]];
    }

    // face : OCT_DIR_UP (4)
    isHanging = this->isFaceHanging(elementID, OCT_DIR_UP, cnum);
    if (isHanging) {
        faceNodesIndex(elementID, OCT_DIR_UP, faceIndex, false);
        for (unsigned int index = 0; index < faceIndex.size(); index++)
            faceInpIn[index] = vec[m_uiE2NMapping_CG[faceIndex[index]]];

        if (!(allowWide &&
              this->prolongateHangingFaceWide(vec, elementID, OCT_DIR_UP, cnum,
                                              &(*(faceInpOut.begin())), im1,
                                              im2, allDg, allDgEleStride)))
            this->parent2ChildInterpolation(&(*(faceInpIn.begin())),
                                            &(*(faceInpOut.begin())), cnum, 2,
                                            im1, im2);

        for (unsigned int k = 0; k < (m_uiElementOrder + 1); k++)
            for (unsigned int i = 0; i < (m_uiElementOrder + 1); i++)
                nodalValues[k * (m_uiElementOrder + 1) *
                                (m_uiElementOrder + 1) +
                            m_uiElementOrder * (m_uiElementOrder + 1) + i] =
                    faceInpOut[k * (m_uiElementOrder + 1) + i];

        nodeStatus[OCT_DIR_RIGHT_UP]       = true;
        nodeStatus[OCT_DIR_LEFT_UP]        = true;
        nodeStatus[OCT_DIR_UP_BACK]        = true;
        nodeStatus[OCT_DIR_UP_FRONT]       = true;

        nodeStatus[OCT_DIR_LEFT_UP_BACK]   = true;
        nodeStatus[OCT_DIR_RIGHT_UP_BACK]  = true;
        nodeStatus[OCT_DIR_RIGHT_UP_FRONT] = true;
        nodeStatus[OCT_DIR_LEFT_UP_FRONT]  = true;

    } else {
        for (unsigned int k = 1; k < m_uiElementOrder; k++)
            for (unsigned int i = 1; i < m_uiElementOrder; i++)
                nodalValues[k * (m_uiElementOrder + 1) *
                                (m_uiElementOrder + 1) +
                            m_uiElementOrder * (m_uiElementOrder + 1) + i] =
                    vec[m_uiE2NMapping_CG
                            [elementID * m_uiNpE +
                             k * (m_uiElementOrder + 1) *
                                 (m_uiElementOrder + 1) +
                             m_uiElementOrder * (m_uiElementOrder + 1) + i]];
    }

    // face : OCT_DIR_BACK (5)
    isHanging = this->isFaceHanging(elementID, OCT_DIR_BACK, cnum);
    if (isHanging) {
        faceNodesIndex(elementID, OCT_DIR_BACK, faceIndex, false);
        for (unsigned int index = 0; index < faceIndex.size(); index++)
            faceInpIn[index] = vec[m_uiE2NMapping_CG[faceIndex[index]]];

        if (!(allowWide &&
              this->prolongateHangingFaceWide(vec, elementID, OCT_DIR_BACK, cnum,
                                              &(*(faceInpOut.begin())), im1,
                                              im2, allDg, allDgEleStride)))
            this->parent2ChildInterpolation(&(*(faceInpIn.begin())),
                                            &(*(faceInpOut.begin())), cnum, 2,
                                            im1, im2);

        for (unsigned int j = 0; j < (m_uiElementOrder + 1); j++)
            for (unsigned int i = 0; i < (m_uiElementOrder + 1); i++)
                nodalValues[0 * (m_uiElementOrder + 1) *
                                (m_uiElementOrder + 1) +
                            j * (m_uiElementOrder + 1) + i] =
                    faceInpOut[j * (m_uiElementOrder + 1) + i];

        nodeStatus[OCT_DIR_LEFT_BACK]       = true;
        nodeStatus[OCT_DIR_RIGHT_BACK]      = true;
        nodeStatus[OCT_DIR_UP_BACK]         = true;
        nodeStatus[OCT_DIR_DOWN_BACK]       = true;

        nodeStatus[OCT_DIR_LEFT_DOWN_BACK]  = true;
        nodeStatus[OCT_DIR_LEFT_UP_BACK]    = true;
        nodeStatus[OCT_DIR_RIGHT_DOWN_BACK] = true;
        nodeStatus[OCT_DIR_RIGHT_UP_BACK]   = true;

    } else {
        for (unsigned int j = 1; j < m_uiElementOrder; j++)
            for (unsigned int i = 1; i < m_uiElementOrder; i++)
                nodalValues[0 * (m_uiElementOrder + 1) *
                                (m_uiElementOrder + 1) +
                            j * (m_uiElementOrder + 1) + i] =
                    vec[m_uiE2NMapping_CG[elementID * m_uiNpE +
                                          0 * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          j * (m_uiElementOrder + 1) + i]];
    }

    // face : OCT_DIR_FRONT (6)
    isHanging = this->isFaceHanging(elementID, OCT_DIR_FRONT, cnum);
    if (isHanging) {
        faceNodesIndex(elementID, OCT_DIR_FRONT, faceIndex, false);
        for (unsigned int index = 0; index < faceIndex.size(); index++)
            faceInpIn[index] = vec[m_uiE2NMapping_CG[faceIndex[index]]];

        if (!(allowWide &&
              this->prolongateHangingFaceWide(vec, elementID, OCT_DIR_FRONT, cnum,
                                              &(*(faceInpOut.begin())), im1,
                                              im2, allDg, allDgEleStride)))
            this->parent2ChildInterpolation(&(*(faceInpIn.begin())),
                                            &(*(faceInpOut.begin())), cnum, 2,
                                            im1, im2);

        for (unsigned int j = 0; j < (m_uiElementOrder + 1); j++)
            for (unsigned int i = 0; i < (m_uiElementOrder + 1); i++)
                nodalValues[m_uiElementOrder * (m_uiElementOrder + 1) *
                                (m_uiElementOrder + 1) +
                            j * (m_uiElementOrder + 1) + i] =
                    faceInpOut[j * (m_uiElementOrder + 1) + i];

        nodeStatus[OCT_DIR_LEFT_FRONT]       = true;
        nodeStatus[OCT_DIR_RIGHT_FRONT]      = true;
        nodeStatus[OCT_DIR_UP_FRONT]         = true;
        nodeStatus[OCT_DIR_DOWN_FRONT]       = true;

        nodeStatus[OCT_DIR_LEFT_DOWN_FRONT]  = true;
        nodeStatus[OCT_DIR_LEFT_UP_FRONT]    = true;
        nodeStatus[OCT_DIR_RIGHT_DOWN_FRONT] = true;
        nodeStatus[OCT_DIR_RIGHT_UP_FRONT]   = true;

    } else {
        for (unsigned int j = 1; j < m_uiElementOrder; j++)
            for (unsigned int i = 1; i < m_uiElementOrder; i++)
                nodalValues[m_uiElementOrder * (m_uiElementOrder + 1) *
                                (m_uiElementOrder + 1) +
                            j * (m_uiElementOrder + 1) + i] =
                    vec[m_uiE2NMapping_CG[elementID * m_uiNpE +
                                          m_uiElementOrder *
                                              (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          j * (m_uiElementOrder + 1) + i]];
    }

    // edge: OCT_DIR_LEFT_DOWN (1)

    if ((!nodeStatus[OCT_DIR_LEFT_DOWN])) {
        isHanging = this->isEdgeHanging(elementID, OCT_DIR_LEFT_DOWN, cnum);
        if (isHanging) {
            edgeNodeIndex(elementID, OCT_DIR_LEFT, OCT_DIR_DOWN, edgeIndex,
                          false);
            for (unsigned int index = 0; index < edgeIndex.size(); index++)
                edgeInpIn[index] = vec[m_uiE2NMapping_CG[edgeIndex[index]]];

            if (!(allowWide &&
                  this->prolongateHangingEdgeWide(
                      vec, elementID, OCT_DIR_LEFT, OCT_DIR_DOWN, cnum,
                      &(*(edgeInpOut.begin())), im1, im2, allDg, allDgEleStride)))
                this->parent2ChildInterpolation(&(*(edgeInpIn.begin())),
                                                &(*(edgeInpOut.begin())),
                                                cnum, 1, im1, im2);

            for (unsigned int k = 0; k < (m_uiElementOrder + 1); k++)
                nodalValues[k * (m_uiElementOrder + 1) *
                                (m_uiElementOrder + 1) +
                            0 * (m_uiElementOrder + 1) + 0] = edgeInpOut[k];

            nodeStatus[OCT_DIR_LEFT_DOWN_BACK]  = true;
            nodeStatus[OCT_DIR_LEFT_DOWN_FRONT] = true;

        } else {
            for (unsigned int k = 1; k < (m_uiElementOrder); k++)
                nodalValues[k * (m_uiElementOrder + 1) *
                                (m_uiElementOrder + 1) +
                            0 * (m_uiElementOrder + 1) + 0] =
                    vec[m_uiE2NMapping_CG[elementID * m_uiNpE +
                                          k * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          0 * (m_uiElementOrder + 1) + 0]];
        }
    }

    // edge: OCT_DIR_LEFT_UP (2)

    if ((!nodeStatus[OCT_DIR_LEFT_UP])) {
        isHanging = this->isEdgeHanging(elementID, OCT_DIR_LEFT_UP, cnum);
        if (isHanging) {
            edgeNodeIndex(elementID, OCT_DIR_LEFT, OCT_DIR_UP, edgeIndex,
                          false);
            for (unsigned int index = 0; index < edgeIndex.size(); index++)
                edgeInpIn[index] = vec[m_uiE2NMapping_CG[edgeIndex[index]]];

            if (!(allowWide &&
                  this->prolongateHangingEdgeWide(
                      vec, elementID, OCT_DIR_LEFT, OCT_DIR_UP, cnum,
                      &(*(edgeInpOut.begin())), im1, im2, allDg, allDgEleStride)))
                this->parent2ChildInterpolation(&(*(edgeInpIn.begin())),
                                                &(*(edgeInpOut.begin())),
                                                cnum, 1, im1, im2);

            for (unsigned int k = 0; k < (m_uiElementOrder + 1); k++)
                nodalValues[k * (m_uiElementOrder + 1) *
                                (m_uiElementOrder + 1) +
                            m_uiElementOrder * (m_uiElementOrder + 1) + 0] =
                    edgeInpOut[k];

            nodeStatus[OCT_DIR_LEFT_UP_BACK]  = true;
            nodeStatus[OCT_DIR_LEFT_UP_FRONT] = true;

        } else {
            for (unsigned int k = 1; k < (m_uiElementOrder); k++)
                nodalValues[k * (m_uiElementOrder + 1) *
                                (m_uiElementOrder + 1) +
                            m_uiElementOrder * (m_uiElementOrder + 1) + 0] =
                    vec[m_uiE2NMapping_CG
                            [elementID * m_uiNpE +
                             k * (m_uiElementOrder + 1) *
                                 (m_uiElementOrder + 1) +
                             m_uiElementOrder * (m_uiElementOrder + 1) + 0]];
        }
    }

    // edge: OCT_DIR_LEFT_BACK (3)

    if ((!nodeStatus[OCT_DIR_LEFT_BACK])) {
        isHanging = this->isEdgeHanging(elementID, OCT_DIR_LEFT_BACK, cnum);
        if (isHanging) {
            edgeNodeIndex(elementID, OCT_DIR_LEFT, OCT_DIR_BACK, edgeIndex,
                          false);
            for (unsigned int index = 0; index < edgeIndex.size(); index++)
                edgeInpIn[index] = vec[m_uiE2NMapping_CG[edgeIndex[index]]];

            if (!(allowWide &&
                  this->prolongateHangingEdgeWide(
                      vec, elementID, OCT_DIR_LEFT, OCT_DIR_BACK, cnum,
                      &(*(edgeInpOut.begin())), im1, im2, allDg, allDgEleStride)))
                this->parent2ChildInterpolation(&(*(edgeInpIn.begin())),
                                                &(*(edgeInpOut.begin())),
                                                cnum, 1, im1, im2);

            for (unsigned int j = 0; j < (m_uiElementOrder + 1); j++)
                nodalValues[0 * (m_uiElementOrder + 1) *
                                (m_uiElementOrder + 1) +
                            j * (m_uiElementOrder + 1) + 0] = edgeInpOut[j];

            nodeStatus[OCT_DIR_LEFT_DOWN_BACK] = true;
            nodeStatus[OCT_DIR_LEFT_UP_BACK]   = true;

        } else {
            for (unsigned int j = 1; j < (m_uiElementOrder); j++)
                nodalValues[0 * (m_uiElementOrder + 1) *
                                (m_uiElementOrder + 1) +
                            j * (m_uiElementOrder + 1) + 0] =
                    vec[m_uiE2NMapping_CG[elementID * m_uiNpE +
                                          0 * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          j * (m_uiElementOrder + 1) + 0]];
        }
    }

    // edge: OCT_DIR_LEFT_FRONT(4)

    if ((!nodeStatus[OCT_DIR_LEFT_FRONT])) {
        isHanging = this->isEdgeHanging(elementID, OCT_DIR_LEFT_FRONT, cnum);
        if (isHanging) {
            edgeNodeIndex(elementID, OCT_DIR_LEFT, OCT_DIR_FRONT, edgeIndex,
                          false);
            for (unsigned int index = 0; index < edgeIndex.size(); index++)
                edgeInpIn[index] = vec[m_uiE2NMapping_CG[edgeIndex[index]]];

            if (!(allowWide &&
                  this->prolongateHangingEdgeWide(
                      vec, elementID, OCT_DIR_LEFT, OCT_DIR_FRONT, cnum,
                      &(*(edgeInpOut.begin())), im1, im2, allDg, allDgEleStride)))
                this->parent2ChildInterpolation(&(*(edgeInpIn.begin())),
                                                &(*(edgeInpOut.begin())),
                                                cnum, 1, im1, im2);

            for (unsigned int j = 0; j < (m_uiElementOrder + 1); j++)
                nodalValues[m_uiElementOrder * (m_uiElementOrder + 1) *
                                (m_uiElementOrder + 1) +
                            j * (m_uiElementOrder + 1) + 0] = edgeInpOut[j];

            nodeStatus[OCT_DIR_LEFT_DOWN_FRONT] = true;
            nodeStatus[OCT_DIR_LEFT_UP_FRONT]   = true;

        } else {
            for (unsigned int j = 1; j < (m_uiElementOrder); j++)
                nodalValues[m_uiElementOrder * (m_uiElementOrder + 1) *
                                (m_uiElementOrder + 1) +
                            j * (m_uiElementOrder + 1) + 0] =
                    vec[m_uiE2NMapping_CG[elementID * m_uiNpE +
                                          m_uiElementOrder *
                                              (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          j * (m_uiElementOrder + 1) + 0]];
        }
    }

    // edge: OCT_DIR_RIGHT_DOWN (5)

    if ((!nodeStatus[OCT_DIR_RIGHT_DOWN])) {
        isHanging = this->isEdgeHanging(elementID, OCT_DIR_RIGHT_DOWN, cnum);
        if (isHanging) {
            edgeNodeIndex(elementID, OCT_DIR_RIGHT, OCT_DIR_DOWN, edgeIndex,
                          false);
            for (unsigned int index = 0; index < edgeIndex.size(); index++)
                edgeInpIn[index] = vec[m_uiE2NMapping_CG[edgeIndex[index]]];

            if (!(allowWide &&
                  this->prolongateHangingEdgeWide(
                      vec, elementID, OCT_DIR_RIGHT, OCT_DIR_DOWN, cnum,
                      &(*(edgeInpOut.begin())), im1, im2, allDg, allDgEleStride)))
                this->parent2ChildInterpolation(&(*(edgeInpIn.begin())),
                                                &(*(edgeInpOut.begin())),
                                                cnum, 1, im1, im2);

            for (unsigned int k = 0; k < (m_uiElementOrder + 1); k++)
                nodalValues[k * (m_uiElementOrder + 1) *
                                (m_uiElementOrder + 1) +
                            0 * (m_uiElementOrder + 1) + m_uiElementOrder] =
                    edgeInpOut[k];

            nodeStatus[OCT_DIR_RIGHT_DOWN_BACK]  = true;
            nodeStatus[OCT_DIR_RIGHT_DOWN_FRONT] = true;

        } else {
            for (unsigned int k = 1; k < (m_uiElementOrder); k++)
                nodalValues[k * (m_uiElementOrder + 1) *
                                (m_uiElementOrder + 1) +
                            0 * (m_uiElementOrder + 1) + m_uiElementOrder] =
                    vec[m_uiE2NMapping_CG[elementID * m_uiNpE +
                                          k * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          0 * (m_uiElementOrder + 1) +
                                          m_uiElementOrder]];
        }
    }

    // edge: OCT_DIR_RIGHT_UP (6)

    if ((!nodeStatus[OCT_DIR_RIGHT_UP])) {
        isHanging = this->isEdgeHanging(elementID, OCT_DIR_RIGHT_UP, cnum);
        if (isHanging) {
            edgeNodeIndex(elementID, OCT_DIR_RIGHT, OCT_DIR_UP, edgeIndex,
                          false);
            for (unsigned int index = 0; index < edgeIndex.size(); index++)
                edgeInpIn[index] = vec[m_uiE2NMapping_CG[edgeIndex[index]]];

            if (!(allowWide &&
                  this->prolongateHangingEdgeWide(
                      vec, elementID, OCT_DIR_RIGHT, OCT_DIR_UP, cnum,
                      &(*(edgeInpOut.begin())), im1, im2, allDg, allDgEleStride)))
                this->parent2ChildInterpolation(&(*(edgeInpIn.begin())),
                                                &(*(edgeInpOut.begin())),
                                                cnum, 1, im1, im2);

            for (unsigned int k = 0; k < (m_uiElementOrder + 1); k++)
                nodalValues[k * (m_uiElementOrder + 1) *
                                (m_uiElementOrder + 1) +
                            m_uiElementOrder * (m_uiElementOrder + 1) +
                            m_uiElementOrder] = edgeInpOut[k];

            nodeStatus[OCT_DIR_RIGHT_UP_BACK]  = true;
            nodeStatus[OCT_DIR_RIGHT_UP_FRONT] = true;

        } else {
            for (unsigned int k = 1; k < (m_uiElementOrder); k++)
                nodalValues[k * (m_uiElementOrder + 1) *
                                (m_uiElementOrder + 1) +
                            m_uiElementOrder * (m_uiElementOrder + 1) +
                            m_uiElementOrder] =
                    vec[m_uiE2NMapping_CG[elementID * m_uiNpE +
                                          k * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          m_uiElementOrder *
                                              (m_uiElementOrder + 1) +
                                          m_uiElementOrder]];
        }
    }

    // edge: OCT_DIR_RIGHT_BACK (7)

    if ((!nodeStatus[OCT_DIR_RIGHT_BACK])) {
        isHanging = this->isEdgeHanging(elementID, OCT_DIR_RIGHT_BACK, cnum);
        if (isHanging) {
            edgeNodeIndex(elementID, OCT_DIR_RIGHT, OCT_DIR_BACK, edgeIndex,
                          false);
            for (unsigned int index = 0; index < edgeIndex.size(); index++)
                edgeInpIn[index] = vec[m_uiE2NMapping_CG[edgeIndex[index]]];

            if (!(allowWide &&
                  this->prolongateHangingEdgeWide(
                      vec, elementID, OCT_DIR_RIGHT, OCT_DIR_BACK, cnum,
                      &(*(edgeInpOut.begin())), im1, im2, allDg, allDgEleStride)))
                this->parent2ChildInterpolation(&(*(edgeInpIn.begin())),
                                                &(*(edgeInpOut.begin())),
                                                cnum, 1, im1, im2);

            for (unsigned int j = 0; j < (m_uiElementOrder + 1); j++)
                nodalValues[0 * (m_uiElementOrder + 1) *
                                (m_uiElementOrder + 1) +
                            j * (m_uiElementOrder + 1) + m_uiElementOrder] =
                    edgeInpOut[j];

            nodeStatus[OCT_DIR_RIGHT_DOWN_BACK] = true;
            nodeStatus[OCT_DIR_RIGHT_UP_BACK]   = true;

        } else {
            for (unsigned int j = 1; j < (m_uiElementOrder); j++)
                nodalValues[0 * (m_uiElementOrder + 1) *
                                (m_uiElementOrder + 1) +
                            j * (m_uiElementOrder + 1) + m_uiElementOrder] =
                    vec[m_uiE2NMapping_CG[elementID * m_uiNpE +
                                          0 * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          j * (m_uiElementOrder + 1) +
                                          m_uiElementOrder]];
        }
    }

    // edge: OCT_DIR_RIGHT_FRONT(8)

    if ((!nodeStatus[OCT_DIR_RIGHT_FRONT])) {
        isHanging = this->isEdgeHanging(elementID, OCT_DIR_RIGHT_FRONT, cnum);
        if (isHanging) {
            edgeNodeIndex(elementID, OCT_DIR_RIGHT, OCT_DIR_FRONT, edgeIndex,
                          false);
            for (unsigned int index = 0; index < edgeIndex.size(); index++)
                edgeInpIn[index] = vec[m_uiE2NMapping_CG[edgeIndex[index]]];

            if (!(allowWide &&
                  this->prolongateHangingEdgeWide(
                      vec, elementID, OCT_DIR_RIGHT, OCT_DIR_FRONT, cnum,
                      &(*(edgeInpOut.begin())), im1, im2, allDg, allDgEleStride)))
                this->parent2ChildInterpolation(&(*(edgeInpIn.begin())),
                                                &(*(edgeInpOut.begin())),
                                                cnum, 1, im1, im2);

            for (unsigned int j = 0; j < (m_uiElementOrder + 1); j++)
                nodalValues[m_uiElementOrder * (m_uiElementOrder + 1) *
                                (m_uiElementOrder + 1) +
                            j * (m_uiElementOrder + 1) + m_uiElementOrder] =
                    edgeInpOut[j];

            nodeStatus[OCT_DIR_RIGHT_DOWN_FRONT] = true;
            nodeStatus[OCT_DIR_RIGHT_UP_FRONT]   = true;

        } else {
            for (unsigned int j = 1; j < (m_uiElementOrder); j++)
                nodalValues[m_uiElementOrder * (m_uiElementOrder + 1) *
                                (m_uiElementOrder + 1) +
                            j * (m_uiElementOrder + 1) + m_uiElementOrder] =
                    vec[m_uiE2NMapping_CG
                            [elementID * m_uiNpE +
                             m_uiElementOrder * (m_uiElementOrder + 1) *
                                 (m_uiElementOrder + 1) +
                             j * (m_uiElementOrder + 1) + m_uiElementOrder]];
        }
    }

    // edge: OCT_DIR_DOWN_BACK (9)

    if ((!nodeStatus[OCT_DIR_DOWN_BACK])) {
        isHanging = this->isEdgeHanging(elementID, OCT_DIR_DOWN_BACK, cnum);
        if (isHanging) {
            edgeNodeIndex(elementID, OCT_DIR_DOWN, OCT_DIR_BACK, edgeIndex,
                          false);
            for (unsigned int index = 0; index < edgeIndex.size(); index++)
                edgeInpIn[index] = vec[m_uiE2NMapping_CG[edgeIndex[index]]];

            if (!(allowWide &&
                  this->prolongateHangingEdgeWide(
                      vec, elementID, OCT_DIR_DOWN, OCT_DIR_BACK, cnum,
                      &(*(edgeInpOut.begin())), im1, im2, allDg, allDgEleStride)))
                this->parent2ChildInterpolation(&(*(edgeInpIn.begin())),
                                                &(*(edgeInpOut.begin())),
                                                cnum, 1, im1, im2);

            for (unsigned int i = 0; i < (m_uiElementOrder + 1); i++)
                nodalValues[0 * (m_uiElementOrder + 1) *
                                (m_uiElementOrder + 1) +
                            0 * (m_uiElementOrder + 1) + i] = edgeInpOut[i];

            nodeStatus[OCT_DIR_LEFT_DOWN_BACK]  = true;
            nodeStatus[OCT_DIR_RIGHT_DOWN_BACK] = true;

        } else {
            for (unsigned int i = 1; i < (m_uiElementOrder); i++)
                nodalValues[0 * (m_uiElementOrder + 1) *
                                (m_uiElementOrder + 1) +
                            0 * (m_uiElementOrder + 1) + i] =
                    vec[m_uiE2NMapping_CG[elementID * m_uiNpE +
                                          0 * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          0 * (m_uiElementOrder + 1) + i]];
        }
    }

    // edge: OCT_DIR_DOWN_FRONT (10)

    if (!nodeStatus[OCT_DIR_DOWN_FRONT]) {
        isHanging = this->isEdgeHanging(elementID, OCT_DIR_DOWN_FRONT, cnum);
        if (isHanging) {
            edgeNodeIndex(elementID, OCT_DIR_DOWN, OCT_DIR_FRONT, edgeIndex,
                          false);
            for (unsigned int index = 0; index < edgeIndex.size(); index++)
                edgeInpIn[index] = vec[m_uiE2NMapping_CG[edgeIndex[index]]];

            if (!(allowWide &&
                  this->prolongateHangingEdgeWide(
                      vec, elementID, OCT_DIR_DOWN, OCT_DIR_FRONT, cnum,
                      &(*(edgeInpOut.begin())), im1, im2, allDg, allDgEleStride)))
                this->parent2ChildInterpolation(&(*(edgeInpIn.begin())),
                                                &(*(edgeInpOut.begin())),
                                                cnum, 1, im1, im2);

            for (unsigned int i = 0; i < (m_uiElementOrder + 1); i++)
                nodalValues[m_uiElementOrder * (m_uiElementOrder + 1) *
                                (m_uiElementOrder + 1) +
                            0 * (m_uiElementOrder + 1) + i] = edgeInpOut[i];

            nodeStatus[OCT_DIR_LEFT_DOWN_FRONT]  = true;
            nodeStatus[OCT_DIR_RIGHT_DOWN_FRONT] = true;

        } else {
            for (unsigned int i = 1; i < (m_uiElementOrder); i++)
                nodalValues[m_uiElementOrder * (m_uiElementOrder + 1) *
                                (m_uiElementOrder + 1) +
                            0 * (m_uiElementOrder + 1) + i] =
                    vec[m_uiE2NMapping_CG[elementID * m_uiNpE +
                                          m_uiElementOrder *
                                              (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          0 * (m_uiElementOrder + 1) + i]];
        }
    }

    // edge: OCT_DIR_UP_BACK (11)

    if (!nodeStatus[OCT_DIR_UP_BACK]) {
        isHanging = this->isEdgeHanging(elementID, OCT_DIR_UP_BACK, cnum);
        if (isHanging) {
            edgeNodeIndex(elementID, OCT_DIR_UP, OCT_DIR_BACK, edgeIndex,
                          false);
            for (unsigned int index = 0; index < edgeIndex.size(); index++)
                edgeInpIn[index] = vec[m_uiE2NMapping_CG[edgeIndex[index]]];

            if (!(allowWide &&
                  this->prolongateHangingEdgeWide(
                      vec, elementID, OCT_DIR_UP, OCT_DIR_BACK, cnum,
                      &(*(edgeInpOut.begin())), im1, im2, allDg, allDgEleStride)))
                this->parent2ChildInterpolation(&(*(edgeInpIn.begin())),
                                                &(*(edgeInpOut.begin())),
                                                cnum, 1, im1, im2);

            for (unsigned int i = 0; i < (m_uiElementOrder + 1); i++)
                nodalValues[0 * (m_uiElementOrder + 1) *
                                (m_uiElementOrder + 1) +
                            m_uiElementOrder * (m_uiElementOrder + 1) + i] =
                    edgeInpOut[i];

            nodeStatus[OCT_DIR_LEFT_UP_BACK]  = true;
            nodeStatus[OCT_DIR_RIGHT_UP_BACK] = true;

        } else {
            for (unsigned int i = 1; i < (m_uiElementOrder); i++)
                nodalValues[0 * (m_uiElementOrder + 1) *
                                (m_uiElementOrder + 1) +
                            m_uiElementOrder * (m_uiElementOrder + 1) + i] =
                    vec[m_uiE2NMapping_CG
                            [elementID * m_uiNpE +
                             0 * (m_uiElementOrder + 1) *
                                 (m_uiElementOrder + 1) +
                             m_uiElementOrder * (m_uiElementOrder + 1) + i]];
        }
    }

    // edge: OCT_DIR_UP_FRONT (12)

    if (!nodeStatus[OCT_DIR_UP_FRONT]) {
        isHanging = this->isEdgeHanging(elementID, OCT_DIR_UP_FRONT, cnum);
        if (isHanging) {
            edgeNodeIndex(elementID, OCT_DIR_UP, OCT_DIR_FRONT, edgeIndex,
                          false);
            for (unsigned int index = 0; index < edgeIndex.size(); index++)
                edgeInpIn[index] = vec[m_uiE2NMapping_CG[edgeIndex[index]]];

            if (!(allowWide &&
                  this->prolongateHangingEdgeWide(
                      vec, elementID, OCT_DIR_UP, OCT_DIR_FRONT, cnum,
                      &(*(edgeInpOut.begin())), im1, im2, allDg, allDgEleStride)))
                this->parent2ChildInterpolation(&(*(edgeInpIn.begin())),
                                                &(*(edgeInpOut.begin())),
                                                cnum, 1, im1, im2);

            for (unsigned int i = 0; i < (m_uiElementOrder + 1); i++)
                nodalValues[m_uiElementOrder * (m_uiElementOrder + 1) *
                                (m_uiElementOrder + 1) +
                            m_uiElementOrder * (m_uiElementOrder + 1) + i] =
                    edgeInpOut[i];

            nodeStatus[OCT_DIR_LEFT_UP_FRONT]  = true;
            nodeStatus[OCT_DIR_RIGHT_UP_FRONT] = true;

        } else {
            for (unsigned int i = 1; i < (m_uiElementOrder); i++)
                nodalValues[m_uiElementOrder * (m_uiElementOrder + 1) *
                                (m_uiElementOrder + 1) +
                            m_uiElementOrder * (m_uiElementOrder + 1) + i] =
                    vec[m_uiE2NMapping_CG
                            [elementID * m_uiNpE +
                             m_uiElementOrder * (m_uiElementOrder + 1) *
                                 (m_uiElementOrder + 1) +
                             m_uiElementOrder * (m_uiElementOrder + 1) + i]];
        }
    }

    // node: OCT_DIR_LEFT_DOWN_BACK
    if ((!(this->isNodeHanging(elementID, 0, 0, 0))) ||
        (!nodeStatus[OCT_DIR_LEFT_DOWN_BACK]))
        nodalValues[0 * (m_uiElementOrder + 1) * (m_uiElementOrder + 1) +
                    0 * (m_uiElementOrder + 1) + 0] =
            vec[m_uiE2NMapping_CG[elementID * m_uiNpE +
                                  0 * (m_uiElementOrder + 1) *
                                      (m_uiElementOrder + 1) +
                                  0 * (m_uiElementOrder + 1) + 0]];

    // node: OCT_DIR_RIGHT_DOWN_BACK
    if (!(this->isNodeHanging(elementID, m_uiElementOrder, 0, 0)) ||
        (!nodeStatus[OCT_DIR_RIGHT_DOWN_BACK]))
        nodalValues[0 * (m_uiElementOrder + 1) * (m_uiElementOrder + 1) +
                    0 * (m_uiElementOrder + 1) + m_uiElementOrder] =
            vec[m_uiE2NMapping_CG
                    [elementID * m_uiNpE +
                     0 * (m_uiElementOrder + 1) * (m_uiElementOrder + 1) +
                     0 * (m_uiElementOrder + 1) + m_uiElementOrder]];

    // node: OCT_DIR_LEFT_UP_BACK
    if (!(this->isNodeHanging(elementID, 0, m_uiElementOrder, 0)) ||
        (!nodeStatus[OCT_DIR_LEFT_UP_BACK]))
        nodalValues[0 * (m_uiElementOrder + 1) * (m_uiElementOrder + 1) +
                    m_uiElementOrder * (m_uiElementOrder + 1) + 0] =
            vec[m_uiE2NMapping_CG
                    [elementID * m_uiNpE +
                     0 * (m_uiElementOrder + 1) * (m_uiElementOrder + 1) +
                     m_uiElementOrder * (m_uiElementOrder + 1) + 0]];

    // node: OCT_DIR_RIGHT_UP_BACK
    if (!(this->isNodeHanging(elementID, m_uiElementOrder, m_uiElementOrder,
                              0)) ||
        (!nodeStatus[OCT_DIR_RIGHT_UP_BACK]))
        nodalValues[0 * (m_uiElementOrder + 1) * (m_uiElementOrder + 1) +
                    m_uiElementOrder * (m_uiElementOrder + 1) +
                    m_uiElementOrder] =
            vec[m_uiE2NMapping_CG[elementID * m_uiNpE +
                                  0 * (m_uiElementOrder + 1) *
                                      (m_uiElementOrder + 1) +
                                  m_uiElementOrder * (m_uiElementOrder + 1) +
                                  m_uiElementOrder]];

    // node: OCT_DIR_LEFT_DOWN_FRONT
    if (!(this->isNodeHanging(elementID, 0, 0, m_uiElementOrder)) ||
        (!nodeStatus[OCT_DIR_LEFT_DOWN_FRONT]))
        nodalValues[m_uiElementOrder * (m_uiElementOrder + 1) *
                        (m_uiElementOrder + 1) +
                    0 * (m_uiElementOrder + 1) + 0] =
            vec[m_uiE2NMapping_CG[elementID * m_uiNpE +
                                  m_uiElementOrder * (m_uiElementOrder + 1) *
                                      (m_uiElementOrder + 1) +
                                  0 * (m_uiElementOrder + 1) + 0]];

    // node: OCT_DIR_RIGHT_DOWN_FRONT
    if (!(this->isNodeHanging(elementID, m_uiElementOrder, 0,
                              m_uiElementOrder)) ||
        (!nodeStatus[OCT_DIR_RIGHT_DOWN_FRONT]))
        nodalValues[m_uiElementOrder * (m_uiElementOrder + 1) *
                        (m_uiElementOrder + 1) +
                    0 * (m_uiElementOrder + 1) + m_uiElementOrder] =
            vec[m_uiE2NMapping_CG[elementID * m_uiNpE +
                                  m_uiElementOrder * (m_uiElementOrder + 1) *
                                      (m_uiElementOrder + 1) +
                                  0 * (m_uiElementOrder + 1) +
                                  m_uiElementOrder]];

    // node: OCT_DIR_LEFT_UP_FRONT
    if (!(this->isNodeHanging(elementID, 0, m_uiElementOrder,
                              m_uiElementOrder)) ||
        (!nodeStatus[OCT_DIR_LEFT_UP_FRONT]))
        nodalValues[m_uiElementOrder * (m_uiElementOrder + 1) *
                        (m_uiElementOrder + 1) +
                    m_uiElementOrder * (m_uiElementOrder + 1) + 0] =
            vec[m_uiE2NMapping_CG[elementID * m_uiNpE +
                                  m_uiElementOrder * (m_uiElementOrder + 1) *
                                      (m_uiElementOrder + 1) +
                                  m_uiElementOrder * (m_uiElementOrder + 1) +
                                  0]];

    // node: OCT_DIR_RIGHT_UP_FRONT
    if (!(this->isNodeHanging(elementID, m_uiElementOrder, m_uiElementOrder,
                              m_uiElementOrder)) ||
        (!nodeStatus[OCT_DIR_RIGHT_UP_FRONT]))
        nodalValues[m_uiElementOrder * (m_uiElementOrder + 1) *
                        (m_uiElementOrder + 1) +
                    m_uiElementOrder * (m_uiElementOrder + 1) +
                    m_uiElementOrder] =
            vec[m_uiE2NMapping_CG[elementID * m_uiNpE +
                                  m_uiElementOrder * (m_uiElementOrder + 1) *
                                      (m_uiElementOrder + 1) +
                                  m_uiElementOrder * (m_uiElementOrder + 1) +
                                  m_uiElementOrder]];

#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
    dendro::timer::t_unzip_sync_nodalval.stop();
#endif
}

template <typename T>
void Mesh::computeElementalContribution(const T* in, T* out,
                                        unsigned int elementID) const {
    if (!m_uiIsActive) return;

    const unsigned int eleOrder = m_uiElementOrder;
    const unsigned int npe_1d   = eleOrder + 1;
    const unsigned int npe_2d   = (eleOrder + 1) * (eleOrder + 1);
    const unsigned int nPe = (eleOrder + 1) * (eleOrder + 1) * (eleOrder + 1);

    //@todo later move this to outer allocation and reuse the memeory.
    double* qMat           = new double[nPe * nPe];
    double* qTIn           = new double[nPe];

    this->getElementQMat(elementID, qMat, true);

    for (unsigned int i = 0; i < nPe; i++) {
        qTIn[i] = 0;

        for (unsigned int j = 0; j < nPe; j++) {
            qTIn[i] += qMat[j * nPe + i] * in[j];  // note the transpose.
        }
    }

    for (unsigned int i = 0; i < nPe; i++)
        out[m_uiE2NMapping_CG[elementID * nPe + i]] += qTIn[i];

    delete[] qMat;
    delete[] qTIn;

    return;
}

template <typename T>
void Mesh::interGridTransfer(std::vector<T>& vec, const ot::Mesh* pMesh,
                             INTERGRID_TRANSFER_MODE mode) {
    std::vector<T> tvec;
    pMesh->createVector<T>(tvec, 0);

    this->interGridTransfer(vec.data(), tvec.data(), pMesh, mode, 1);

    std::swap(vec, tvec);
    tvec.clear();
    return;
}

template <typename T>
void Mesh::interGridTransfer(T*& vec, const ot::Mesh* pMesh,
                             INTERGRID_TRANSFER_MODE mode, unsigned int dof) {
    T* tVec = pMesh->createCGVector<T>(0, dof);
    this->interGridTransfer(vec, tVec, pMesh, mode, dof);

    std::swap(vec, tVec);
    delete[] tVec;
    return;
}

template <typename T>
void Mesh::interGridTransfer(T* vecIn, T* vecOut, const ot::Mesh* pMesh,
                             INTERGRID_TRANSFER_MODE mode, unsigned int dof) {
    MPI_Comm comm = m_uiCommGlobal;
    int rank, npes;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);

    std::vector<unsigned int> sendC;
    std::vector<unsigned int> recvC;

    std::vector<unsigned int> sendOfst;
    std::vector<unsigned int> recvOfst;

    sendC.resize(npes);
    recvC.resize(npes);
    sendOfst.resize(npes);
    recvOfst.resize(npes);

    this->interGridTransferSendRecvCompute(pMesh);
    const unsigned int cg_sz_old = m_uiNumActualNodes;
    const unsigned int cg_sz_new = pMesh->getDegOfFreedom();
    const ot::TreeNode* m2prime  = m_uiM2Prime.data();

    // scale the elemental counts by m_uiNpE;
    for (unsigned int p = 0; p < npes; p++) {
        sendC[p]    = m_uiIGTSendC[p] * m_uiNpE;
        recvC[p]    = m_uiIGTRecvC[p] * m_uiNpE;

        sendOfst[p] = m_uiIGTSendOfst[p] * m_uiNpE;
        recvOfst[p] = m_uiIGTRecvOfst[p] * m_uiNpE;
    }

    std::vector<T> wVec;  // dg of m2prime;
    std::vector<T> nodalVals;
    nodalVals.resize(m_uiNpE);

    unsigned int cnum;
    bool isHanging;

    std::vector<double> vallchildren;
    std::vector<T> wVec_m2;

    vallchildren.resize((2 * m_uiElementOrder + 1) *
                        (2 * m_uiElementOrder + 1) *
                        (2 * m_uiElementOrder + 1));
    wVec_m2.resize(recvOfst[npes - 1] + recvC[npes - 1]);

#ifdef DENDRO_UNZIP_OMP
    // Resolve the running m2primeCount/ele prefix dependency ONCE (mesh topology
    // is var-independent) so the per-var transfer loop below can run threaded.
    // Each work item maps one source octant (group) to its m2prime output base.
    struct IGTWorkItem {
        unsigned int ele;
        unsigned int m2pBase;
    };
    std::vector<IGTWorkItem> igtItems;
    if (m_uiIsActive) {
        igtItems.reserve(m_uiElementLocalEnd - m_uiElementLocalBegin);
        unsigned int m2c = 0;
        for (unsigned int ele = m_uiElementLocalBegin;
             ele < m_uiElementLocalEnd;) {
            const unsigned int octType =
                (m_uiAllElements[ele].getFlag() >> NUM_LEVEL_BITS);
            igtItems.push_back({ele, m2c});
            if (octType == OCT_SPLIT) {
                m2c += NUM_CHILDREN;
                ele += 1;
            } else if (octType == OCT_COARSE) {
                m2c += 1;
                ele += NUM_CHILDREN;
            } else {
                m2c += 1;
                ele += 1;
            }
        }
    }
#endif

    for (unsigned int var = 0; var < dof; var++) {
        T* vec = vecIn + (var * cg_sz_old);
        T* out = vecOut + (var * cg_sz_new);

        if (m_uiIsActive) {
            const unsigned int npes1           = this->getMPICommSize();
            const unsigned int rank1           = this->getMPIRank();

            const unsigned int numM2PrimeElems = m_uiM2Prime.size();
            wVec.resize(numM2PrimeElems * m_uiNpE);

            // std::cout<<"rank1: "<<rank1<<" m2prime:
            // "<<m2prime.size()<<std::endl;

#ifdef DENDRO_UNZIP_OMP
            // Threaded transfer over the precomputed work items: each item owns
            // a disjoint wVec slab (write-disjoint -> race-free). Per-thread
            // scratch + the thread-safe RefElement overloads (im1/im2);
            // I3D_Children2Parent is scratch-free. Bit-identical to the serial
            // #else branch below (no cross-element reduction).
#pragma omp parallel
            {
                std::vector<T> nodalVals_t(m_uiNpE);
                std::vector<double> vallchildren_t((2 * m_uiElementOrder + 1) *
                                                   (2 * m_uiElementOrder + 1) *
                                                   (2 * m_uiElementOrder + 1));
                std::vector<double> im1_t(m_uiNpE), im2_t(m_uiNpE);
#pragma omp for schedule(dynamic, 16)
                for (size_t it = 0; it < igtItems.size(); it++) {
                    const unsigned int ele     = igtItems[it].ele;
                    const unsigned int m2pBase = igtItems[it].m2pBase;
                    const unsigned int octType =
                        (m_uiAllElements[ele].getFlag() >> NUM_LEVEL_BITS);
                    unsigned int cnum;
                    bool isHanging;

                    if (octType == OCT_SPLIT) {
                        this->getElementNodalValues(vec, nodalVals_t.data(), ele,
                                                    false, im1_t.data(),
                                                    im2_t.data());
                        for (unsigned int child = 0; child < NUM_CHILDREN;
                             child++) {
                            cnum = m2prime[m2pBase + child].getMortonIndex();
                            this->parent2ChildInterpolation(
                                nodalVals_t.data(),
                                &wVec[(m2pBase + child) * m_uiNpE], cnum, 3,
                                im1_t.data(), im2_t.data());
                        }
                    } else if (octType == OCT_COARSE) {
                        if (mode == INTERGRID_TRANSFER_MODE::P2CT) {
                            const unsigned int p1d = 2 * m_uiElementOrder + 1;
                            for (unsigned int child = 0; child < NUM_CHILDREN;
                                 child++) {
                                this->getElementNodalValues(
                                    vec, nodalVals_t.data(), ele + child, false,
                                    im1_t.data(), im2_t.data());
                                for (unsigned int k = 0; k < m_uiElementOrder + 1;
                                     k++)
                                    for (unsigned int j = 0;
                                         j < m_uiElementOrder + 1; j++)
                                        for (unsigned int i = 0;
                                             i < m_uiElementOrder + 1; i++) {
                                            cnum = m_uiAllElements[(ele + child)]
                                                       .getMortonIndex();
                                            const unsigned int iix =
                                                m_uiElementOrder *
                                                    (int)(cnum & 1u) +
                                                i;
                                            const unsigned int jjy =
                                                m_uiElementOrder *
                                                    (int)((cnum & 2u) >> 1u) +
                                                j;
                                            const unsigned int kkz =
                                                m_uiElementOrder *
                                                    (int)((cnum & 4u) >> 2u) +
                                                k;
                                            vallchildren_t[kkz * p1d * p1d +
                                                           jjy * p1d + iix] =
                                                nodalVals_t[k *
                                                                (m_uiElementOrder +
                                                                 1) *
                                                                (m_uiElementOrder +
                                                                 1) +
                                                            j * (m_uiElementOrder +
                                                                 1) +
                                                            i];
                                        }
                            }
                            m_uiRefEl.I3D_Children2Parent(
                                vallchildren_t.data(), &wVec[m2pBase * m_uiNpE]);
                        } else {
                            assert(mode == INTERGRID_TRANSFER_MODE::INJECTION);
                            for (unsigned int child = 0; child < NUM_CHILDREN;
                                 child++) {
                                for (unsigned int k = 0; k < m_uiElementOrder + 1;
                                     k++)
                                    for (unsigned int j = 0;
                                         j < m_uiElementOrder + 1; j++)
                                        for (unsigned int i = 0;
                                             i < m_uiElementOrder + 1; i++) {
                                            isHanging = this->isNodeHanging(
                                                (ele + child), i, j, k);
                                            if (isHanging) {
                                                wVec[m2pBase * m_uiNpE +
                                                     k * (m_uiElementOrder + 1) *
                                                         (m_uiElementOrder + 1) +
                                                     j * (m_uiElementOrder + 1) +
                                                     i] =
                                                    vec[m_uiE2NMapping_CG
                                                            [(ele + child) *
                                                                 m_uiNpE +
                                                             k * (m_uiElementOrder +
                                                                  1) *
                                                                 (m_uiElementOrder +
                                                                  1) +
                                                             j * (m_uiElementOrder +
                                                                  1) +
                                                             i]];
                                            } else {
                                                cnum =
                                                    m_uiAllElements[(ele + child)]
                                                        .getMortonIndex();
                                                const unsigned int iix =
                                                    m_uiElementOrder *
                                                        (int)(cnum & 1u) +
                                                    i;
                                                const unsigned int jjy =
                                                    m_uiElementOrder *
                                                        (int)((cnum & 2u) >> 1u) +
                                                    j;
                                                const unsigned int kkz =
                                                    m_uiElementOrder *
                                                        (int)((cnum & 4u) >> 2u) +
                                                    k;
                                                if ((iix % 2 == 0) &&
                                                    (jjy % 2 == 0) &&
                                                    (kkz % 2 == 0)) {
                                                    wVec[m2pBase * m_uiNpE +
                                                         (kkz >> 1u) *
                                                             (m_uiElementOrder +
                                                              1) *
                                                             (m_uiElementOrder +
                                                              1) +
                                                         (jjy >> 1u) *
                                                             (m_uiElementOrder +
                                                              1) +
                                                         (iix >> 1u)] =
                                                        vec[m_uiE2NMapping_CG
                                                                [(ele + child) *
                                                                     m_uiNpE +
                                                                 k * (m_uiElementOrder +
                                                                      1) *
                                                                     (m_uiElementOrder +
                                                                      1) +
                                                                 j * (m_uiElementOrder +
                                                                      1) +
                                                                 i]];
                                                }
                                            }
                                        }
                            }
                        }
                    } else {
                        // OCT_NO_CHANGE
                        this->getElementNodalValues(vec,
                                                    &wVec[m2pBase * m_uiNpE], ele,
                                                    false, im1_t.data(),
                                                    im2_t.data());
                    }
                }
            }
#else
            unsigned int m2primeCount = 0;
            for (unsigned int ele = m_uiElementLocalBegin;
                 ele < m_uiElementLocalEnd; ele++) {
                // std::cout<<" m2primeCount: "<<m2primeCount<<" wvec offset :
                // "<<m2primeCount*m_uiNpE<< " bound:" <<
                // (m2primeCount+1)*m_uiNpE <<" wvec size :
                // "<<wVec.size()<<std::endl;

                if ((m_uiAllElements[ele].getFlag() >> NUM_LEVEL_BITS) ==
                    OCT_SPLIT) {
                    this->getElementNodalValues(vec, &(*(nodalVals.begin())),
                                                ele);
                    for (unsigned int child = 0; child < NUM_CHILDREN;
                         child++) {
                        cnum = m2prime[m2primeCount + child].getMortonIndex();
                        this->parent2ChildInterpolation(
                            &(*(nodalVals.begin())),
                            &(*(wVec.begin() +
                                (m2primeCount + child) * m_uiNpE)),
                            cnum, 3);
                    }

                    m2primeCount += NUM_CHILDREN;

                } else if ((m_uiAllElements[ele].getFlag() >> NUM_LEVEL_BITS) ==
                           OCT_COARSE) {
                    assert(m_uiAllElements[ele].getParent() ==
                           m_uiAllElements[ele + NUM_CHILDREN - 1].getParent());

                    if (mode == INTERGRID_TRANSFER_MODE::P2CT) {
                        const unsigned int p1d = 2 * m_uiElementOrder + 1;
                        for (unsigned int child = 0; child < NUM_CHILDREN;
                             child++) {
                            this->getElementNodalValues(vec, nodalVals.data(),
                                                        ele + child);
                            for (unsigned int k = 0; k < m_uiElementOrder + 1;
                                 k++)
                                for (unsigned int j = 0;
                                     j < m_uiElementOrder + 1; j++)
                                    for (unsigned int i = 0;
                                         i < m_uiElementOrder + 1; i++) {
                                        cnum = m_uiAllElements[(ele + child)]
                                                   .getMortonIndex();
                                        const unsigned int iix =
                                            m_uiElementOrder *
                                                (int)(cnum & 1u) +
                                            i;
                                        const unsigned int jjy =
                                            m_uiElementOrder *
                                                (int)((cnum & 2u) >> 1u) +
                                            j;
                                        const unsigned int kkz =
                                            m_uiElementOrder *
                                                (int)((cnum & 4u) >> 2u) +
                                            k;

                                        vallchildren[kkz * p1d * p1d +
                                                     jjy * p1d +
                                                     iix] = nodalVals
                                            [k * (m_uiElementOrder + 1) *
                                                 (m_uiElementOrder + 1) +
                                             j * (m_uiElementOrder + 1) +
                                             i];  // vec[m_uiE2NMapping_CG[(ele+child)*m_uiNpE+k*(m_uiElementOrder+1)*(m_uiElementOrder+1)+j*(m_uiElementOrder+1)+i]];
                                    }
                        }

                        m_uiRefEl.I3D_Children2Parent(
                            vallchildren.data(), &wVec[m2primeCount * m_uiNpE]);

                    } else {
                        assert(mode == INTERGRID_TRANSFER_MODE::INJECTION);
                        for (unsigned int child = 0; child < NUM_CHILDREN;
                             child++) {
                            for (unsigned int k = 0; k < m_uiElementOrder + 1;
                                 k++)
                                for (unsigned int j = 0;
                                     j < m_uiElementOrder + 1; j++)
                                    for (unsigned int i = 0;
                                         i < m_uiElementOrder + 1; i++) {
                                        isHanging = this->isNodeHanging(
                                            (ele + child), i, j, k);
                                        if (isHanging) {
                                            wVec[m2primeCount * m_uiNpE +
                                                 k * (m_uiElementOrder + 1) *
                                                     (m_uiElementOrder + 1) +
                                                 j * (m_uiElementOrder + 1) +
                                                 i] =
                                                vec[m_uiE2NMapping_CG
                                                        [(ele + child) *
                                                             m_uiNpE +
                                                         k *
                                                             (m_uiElementOrder +
                                                              1) *
                                                             (m_uiElementOrder +
                                                              1) +
                                                         j * (m_uiElementOrder +
                                                              1) +
                                                         i]];

                                        } else {
                                            cnum =
                                                m_uiAllElements[(ele + child)]
                                                    .getMortonIndex();
                                            const unsigned int iix =
                                                m_uiElementOrder *
                                                    (int)(cnum & 1u) +
                                                i;
                                            const unsigned int jjy =
                                                m_uiElementOrder *
                                                    (int)((cnum & 2u) >> 1u) +
                                                j;
                                            const unsigned int kkz =
                                                m_uiElementOrder *
                                                    (int)((cnum & 4u) >> 2u) +
                                                k;
                                            // std::cout<<" iix: "<<iix<<" jjy:
                                            // "<<jjy<<" kkz: "<<kkz<<std::endl;

                                            if ((iix % 2 == 0) &&
                                                (jjy % 2 == 0) &&
                                                (kkz % 2 == 0)) {
                                                wVec[m2primeCount * m_uiNpE +
                                                     (kkz >> 1u) *
                                                         (m_uiElementOrder +
                                                          1) *
                                                         (m_uiElementOrder +
                                                          1) +
                                                     (jjy >> 1u) *
                                                         (m_uiElementOrder +
                                                          1) +
                                                     (iix >> 1u)] = vec
                                                    [m_uiE2NMapping_CG
                                                         [(ele + child) *
                                                              m_uiNpE +
                                                          k *
                                                              (m_uiElementOrder +
                                                               1) *
                                                              (m_uiElementOrder +
                                                               1) +
                                                          j * (m_uiElementOrder +
                                                               1) +
                                                          i]];
                                            }
                                        }
                                    }
                        }
                    }

                    ele += (NUM_CHILDREN - 1);
                    m2primeCount += 1;

                } else {
                    assert((m_uiAllElements[ele].getFlag() >> NUM_LEVEL_BITS) ==
                           OCT_NO_CHANGE);

                    this->getElementNodalValues(
                        vec, &(*(wVec.begin() + (m2primeCount * m_uiNpE))),
                        ele);
                    m2primeCount += 1;
                }
            }
#endif

            if (npes1 == 1 && pMesh->isActive() &&
                pMesh->getMPICommSize() == 1) {
                // sequential case.

                if (numM2PrimeElems != pMesh->getNumLocalMeshElements()) {
                    std::cout << " seq::[Inter-grid Transfer error ]: Recvn DG "
                                 "elements: "
                              << numM2PrimeElems << " m2 num local elements "
                              << pMesh->getNumLocalMeshElements() << std::endl;
                    MPI_Abort(comm, 0);
                }

                const unsigned int* e2n = &(*(pMesh->getE2NMapping().begin()));

                const unsigned int m2LocalElemBegin =
                    pMesh->getElementLocalBegin();
                const unsigned int m2LocalElemEnd = pMesh->getElementLocalEnd();

                const unsigned int m2LocalNodeBegin =
                    pMesh->getNodeLocalBegin();
                const unsigned int m2LocalNodeEnd = pMesh->getNodeLocalEnd();

                unsigned int lookUp;
                const unsigned int eleOrder = pMesh->getElementOrder();

                for (unsigned int ele = m2LocalElemBegin; ele < m2LocalElemEnd;
                     ele++) {
                    // std::cout<<"ele: "<<ele<<"data copied "<<std::endl;
                    for (unsigned int k = 0; k < eleOrder + 1; k++)
                        for (unsigned int j = 0; j < eleOrder + 1; j++)
                            for (unsigned int i = 0; i < eleOrder + 1; i++) {
                                if (!(pMesh->isNodeHanging(ele, i, j, k))) {
                                    lookUp = e2n[ele * m_uiNpE +
                                                 k * (eleOrder + 1) *
                                                     (eleOrder + 1) +
                                                 j * (eleOrder + 1) + i];
                                    if ((lookUp >= m2LocalNodeBegin &&
                                         lookUp < m2LocalNodeEnd))
                                        out[lookUp] =
                                            wVec[(ele - m2LocalElemBegin) *
                                                     m_uiNpE +
                                                 k * (eleOrder + 1) *
                                                     (eleOrder + 1) +
                                                 j * (eleOrder + 1) + i];
                                }
                            }
                }

                continue;
            }
        }

        par::Mpi_Alltoallv_sparse(&(*(wVec.begin())), (int*)sendC.data(),
                                  (int*)sendOfst.data(), &(*(wVec_m2.begin())),
                                  (int*)recvC.data(), (int*)recvOfst.data(),
                                  comm);
        if (pMesh->isActive()) {
            const unsigned int* e2n = &(*(pMesh->getE2NMapping().begin()));

            const unsigned int m2LocalElemBegin = pMesh->getElementLocalBegin();
            const unsigned int m2LocalElemEnd   = pMesh->getElementLocalEnd();

            const unsigned int m2LocalNodeBegin = pMesh->getNodeLocalBegin();
            const unsigned int m2LocalNodeEnd   = pMesh->getNodeLocalEnd();

            unsigned int lookUp;
            const unsigned int eleOrder = pMesh->getElementOrder();

            for (unsigned int ele = m2LocalElemBegin; ele < m2LocalElemEnd;
                 ele++) {
                for (unsigned int k = 0; k < eleOrder + 1; k++)
                    for (unsigned int j = 0; j < eleOrder + 1; j++)
                        for (unsigned int i = 0; i < eleOrder + 1; i++) {
                            if (!(pMesh->isNodeHanging(ele, i, j, k))) {
                                lookUp =
                                    e2n[ele * m_uiNpE +
                                        k * (eleOrder + 1) * (eleOrder + 1) +
                                        j * (eleOrder + 1) + i];
                                if ((lookUp >= m2LocalNodeBegin &&
                                     lookUp < m2LocalNodeEnd))
                                    out[lookUp] =
                                        wVec_m2[(ele - m2LocalElemBegin) *
                                                    m_uiNpE +
                                                k * (eleOrder + 1) *
                                                    (eleOrder + 1) +
                                                j * (eleOrder + 1) + i];
                            }
                        }
            }
        }
    }

    return;
}

template <typename T>
void Mesh::interGridTransfer_DG(T* vecIn, T* vecOut, const ot::Mesh* pMesh,
                                unsigned int dof) {
    // Note that this is the intergrid transfer for the DG representation of the
    // vector, In DG / octant local representation there is no hanging nodes,
    // each octant has it's own shared nodes.

    MPI_Comm comm = m_uiCommGlobal;
    int rank, npes;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);

    std::vector<unsigned int> sendC;
    std::vector<unsigned int> recvC;

    std::vector<unsigned int> sendOfst;
    std::vector<unsigned int> recvOfst;

    sendC.resize(npes);
    recvC.resize(npes);
    sendOfst.resize(npes);
    recvOfst.resize(npes);

    std::vector<T> wVec;  // dg of m2prime;

    this->interGridTransferSendRecvCompute(pMesh);

    const unsigned int dg_sz_old = getDegOfFreedomDG();
    const unsigned int dg_sz_new = pMesh->getDegOfFreedomDG();

    const ot::TreeNode* m2prime  = m_uiM2Prime.data();
    // scale the elemental counts by m_uiNpE;
    for (unsigned int p = 0; p < npes; p++) {
        sendC[p]    = m_uiIGTSendC[p] * m_uiNpE;
        recvC[p]    = m_uiIGTRecvC[p] * m_uiNpE;

        sendOfst[p] = m_uiIGTSendOfst[p] * m_uiNpE;
        recvOfst[p] = m_uiIGTRecvOfst[p] * m_uiNpE;
    }

    std::vector<T> wVec_m2;
    wVec_m2.resize(recvOfst[npes - 1] + recvC[npes - 1]);

    std::vector<T> nodalVals;
    nodalVals.resize(m_uiNpE);

    unsigned int cnum;
    bool isHanging;

    std::vector<double> vallchildren;
    vallchildren.resize((2 * m_uiElementOrder + 1) *
                        (2 * m_uiElementOrder + 1) *
                        (2 * m_uiElementOrder + 1));

    for (unsigned int v = 0; v < dof; v++) {
        T* vec = vecIn + v * dg_sz_old;
        T* out = vecOut + v * dg_sz_new;

        if (m_uiIsActive) {
            const unsigned int npes1           = this->getMPICommSize();
            const unsigned int rank1           = this->getMPIRank();

            const unsigned int numM2PrimeElems = m_uiM2Prime.size();
            wVec.resize(numM2PrimeElems * m_uiNpE);

            // std::cout<<"rank1: "<<rank1<<" m2prime:
            // "<<m2prime.size()<<std::endl;

            unsigned int m2primeCount = 0;
            for (unsigned int ele = m_uiElementLocalBegin;
                 ele < m_uiElementLocalEnd; ele++) {
                if ((m_uiAllElements[ele].getFlag() >> NUM_LEVEL_BITS) ==
                    OCT_SPLIT) {
                    for (unsigned int child = 0; child < NUM_CHILDREN;
                         child++) {
                        cnum = m2prime[m2primeCount + child].getMortonIndex();
                        this->parent2ChildInterpolation(
                            vec + ele * m_uiNpE,
                            &(*(wVec.begin() +
                                (m2primeCount + child) * m_uiNpE)),
                            cnum, 3);
                    }

                    m2primeCount += NUM_CHILDREN;

                } else if ((m_uiAllElements[ele].getFlag() >> NUM_LEVEL_BITS) ==
                           OCT_COARSE) {
                    assert(m_uiAllElements[ele].getParent() ==
                           m_uiAllElements[ele + NUM_CHILDREN - 1].getParent());

                    // for DG we use only one mode for the coarsening,
                    // pure injection
                    for (unsigned int child = 0; child < NUM_CHILDREN;
                         child++) {
                        for (unsigned int k = 0; k < m_uiElementOrder + 1; k++)
                            for (unsigned int j = 0; j < m_uiElementOrder + 1;
                                 j++)
                                for (unsigned int i = 0;
                                     i < m_uiElementOrder + 1; i++) {
                                    cnum = m_uiAllElements[(ele + child)]
                                               .getMortonIndex();
                                    const unsigned int iix =
                                        m_uiElementOrder * (int)(cnum & 1u) + i;
                                    const unsigned int jjy =
                                        m_uiElementOrder *
                                            (int)((cnum & 2u) >> 1u) +
                                        j;
                                    const unsigned int kkz =
                                        m_uiElementOrder *
                                            (int)((cnum & 4u) >> 2u) +
                                        k;
                                    // std::cout<<" iix: "<<iix<<" jjy:
                                    // "<<jjy<<" kkz: "<<kkz<<std::endl;

                                    if ((iix % 2 == 0) && (jjy % 2 == 0) &&
                                        (kkz % 2 == 0)) {
                                        wVec[m2primeCount * m_uiNpE +
                                             (kkz >> 1u) *
                                                 (m_uiElementOrder + 1) *
                                                 (m_uiElementOrder + 1) +
                                             (jjy >> 1u) *
                                                 (m_uiElementOrder + 1) +
                                             (iix >> 1u)] =
                                            vec[(ele + child) * m_uiNpE +
                                                k * (m_uiElementOrder + 1) *
                                                    (m_uiElementOrder + 1) +
                                                j * (m_uiElementOrder + 1) + i];
                                    }
                                }
                    }

                    ele += (NUM_CHILDREN - 1);
                    m2primeCount += 1;

                } else {
                    assert((m_uiAllElements[ele].getFlag() >> NUM_LEVEL_BITS) ==
                           OCT_NO_CHANGE);

                    for (unsigned int node = 0; node < m_uiNpE; node++)
                        wVec[(m2primeCount * m_uiNpE) + node] =
                            vec[ele * m_uiNpE + node];

                    m2primeCount += 1;
                }
            }

            if (npes1 == 1 && pMesh->isActive() &&
                pMesh->getMPICommSize() == 1) {
                // sequential case.

                if (numM2PrimeElems != pMesh->getNumLocalMeshElements()) {
                    std::cout << " seq::[Inter-grid Transfer error ]: Recvn DG "
                                 "elements: "
                              << numM2PrimeElems << " m2 num local elements "
                              << pMesh->getNumLocalMeshElements() << std::endl;
                    MPI_Abort(comm, 0);
                }

                const unsigned int* e2n = &(*(pMesh->getE2NMapping().begin()));

                const unsigned int m2LocalElemBegin =
                    pMesh->getElementLocalBegin();
                const unsigned int m2LocalElemEnd = pMesh->getElementLocalEnd();

                const unsigned int m2LocalNodeBegin =
                    m2LocalElemBegin * m_uiNpE;
                const unsigned int m2LocalNodeEnd = m2LocalElemEnd * m_uiNpE;

                unsigned int lookUp;
                const unsigned int eleOrder = pMesh->getElementOrder();

                for (unsigned int ele = m2LocalElemBegin; ele < m2LocalElemEnd;
                     ele++) {
                    for (unsigned int k = 0; k < eleOrder + 1; k++)
                        for (unsigned int j = 0; j < eleOrder + 1; j++)
                            for (unsigned int i = 0; i < eleOrder + 1; i++) {
                                lookUp = ele * m_uiNpE +
                                         k * (eleOrder + 1) * (eleOrder + 1) +
                                         j * (eleOrder + 1) + i;
                                if ((lookUp >= m2LocalNodeBegin &&
                                     lookUp < m2LocalNodeEnd))
                                    out[lookUp] =
                                        wVec[(ele - m2LocalElemBegin) *
                                                 m_uiNpE +
                                             k * (eleOrder + 1) *
                                                 (eleOrder + 1) +
                                             j * (eleOrder + 1) + i];
                            }
                }

                continue;
            }
        }

        par::Mpi_Alltoallv_sparse(&(*(wVec.begin())), (int*)sendC.data(),
                                  (int*)sendOfst.data(), &(*(wVec_m2.begin())),
                                  (int*)recvC.data(), (int*)recvOfst.data(),
                                  comm);

        if (pMesh->isActive()) {
            const unsigned int m2LocalElemBegin = pMesh->getElementLocalBegin();
            const unsigned int m2LocalElemEnd   = pMesh->getElementLocalEnd();

            const unsigned int m2LocalNodeBegin = m2LocalElemBegin * m_uiNpE;
            const unsigned int m2LocalNodeEnd   = m2LocalElemEnd * m_uiNpE;

            unsigned int lookUp;
            const unsigned int eleOrder = pMesh->getElementOrder();

            for (unsigned int ele = m2LocalElemBegin; ele < m2LocalElemEnd;
                 ele++) {
                for (unsigned int k = 0; k < eleOrder + 1; k++)
                    for (unsigned int j = 0; j < eleOrder + 1; j++)
                        for (unsigned int i = 0; i < eleOrder + 1; i++) {
                            lookUp = ele * m_uiNpE +
                                     k * (eleOrder + 1) * (eleOrder + 1) +
                                     j * (eleOrder + 1) + i;

                            if ((lookUp >= m2LocalNodeBegin &&
                                 lookUp < m2LocalNodeEnd))
                                out[lookUp] =
                                    wVec_m2[(ele - m2LocalElemBegin) * m_uiNpE +
                                            k * (eleOrder + 1) *
                                                (eleOrder + 1) +
                                            j * (eleOrder + 1) + i];
                        }
            }
        }
    }

    return;
}

template <typename T>
void Mesh::interGridTransferCellVec(T* vecIn, T* vecOut, const ot::Mesh* pMesh,
                                    unsigned int dof,
                                    INTERGRID_TRANSFER_MODE mode) {
    MPI_Comm comm = m_uiCommGlobal;
    int rank, npes;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);

    std::vector<T> wVec;  // dg of m2prime;
    this->interGridTransferSendRecvCompute(pMesh);

    // currently hard coded to the cell vec copy.
    assert(mode == INTERGRID_TRANSFER_MODE::CELLVEC_CPY);

    const unsigned int cell_sz_old = m_uiAllElements.size();
    const unsigned int cell_sz_new = pMesh->getAllElements().size();

    const ot::TreeNode* m2prime    = m_uiM2Prime.data();

    const unsigned int* sendC      = m_uiIGTSendC.data();
    const unsigned int* recvC      = m_uiIGTRecvC.data();
    const unsigned int* sendOfst   = m_uiIGTSendOfst.data();
    const unsigned int* recvOfst   = m_uiIGTRecvOfst.data();

    std::vector<T> wVec_m2;
    wVec_m2.resize(recvOfst[npes - 1] + recvC[npes - 1]);

    for (unsigned int v = 0; v < dof; v++) {
        T* vec = vecIn + v * cell_sz_old;
        T* out = vecOut + v * cell_sz_new;

        if (m_uiIsActive) {
            const unsigned int npes1           = this->getMPICommSize();
            const unsigned int rank1           = this->getMPIRank();

            const unsigned int numM2PrimeElems = m_uiM2Prime.size();
            wVec.resize(numM2PrimeElems);

            // std::cout<<"rank1: "<<rank1<<" m2prime:
            // "<<m2prime.size()<<std::endl;

            unsigned int m2primeCount = 0;
            for (unsigned int ele = m_uiElementLocalBegin;
                 ele < m_uiElementLocalEnd; ele++) {
                if ((m_uiAllElements[ele].getFlag() >> NUM_LEVEL_BITS) ==
                    OCT_SPLIT) {
                    for (unsigned int child = 0; child < NUM_CHILDREN; child++)
                        wVec[m2primeCount + child] = vec[ele];

                    m2primeCount += NUM_CHILDREN;

                } else if ((m_uiAllElements[ele].getFlag() >> NUM_LEVEL_BITS) ==
                           OCT_COARSE) {
                    assert(m_uiAllElements[ele].getParent() ==
                           m_uiAllElements[ele + NUM_CHILDREN - 1].getParent());
                    // check if the cell vector child cells agrees with the
                    // value.
                    assert(vec[ele] == vec[ele + NUM_CHILDREN - 1]);
                    wVec[m2primeCount] = vec[ele];
                    ele += (NUM_CHILDREN - 1);
                    m2primeCount += 1;

                } else {
                    assert((m_uiAllElements[ele].getFlag() >> NUM_LEVEL_BITS) ==
                           OCT_NO_CHANGE);
                    wVec[m2primeCount] = vec[ele];
                    m2primeCount += 1;
                }
            }

            if (npes1 == 1 && pMesh->isActive() &&
                pMesh->getMPICommSize() == 1) {
                // sequential case.

                if (numM2PrimeElems != pMesh->getNumLocalMeshElements()) {
                    std::cout << " seq::[Inter-grid Transfer error ]: Recvn DG "
                                 "elements: "
                              << numM2PrimeElems << " m2 num local elements "
                              << pMesh->getNumLocalMeshElements() << std::endl;
                    MPI_Abort(comm, 0);
                }

                const unsigned int* e2n = &(*(pMesh->getE2NMapping().begin()));

                const unsigned int m2LocalElemBegin =
                    pMesh->getElementLocalBegin();
                const unsigned int m2LocalElemEnd = pMesh->getElementLocalEnd();

                const unsigned int m2LocalNodeBegin =
                    m2LocalElemBegin * m_uiNpE;
                const unsigned int m2LocalNodeEnd = m2LocalElemEnd * m_uiNpE;

                unsigned int lookUp;
                const unsigned int eleOrder = pMesh->getElementOrder();

                for (unsigned int ele = m2LocalElemBegin; ele < m2LocalElemEnd;
                     ele++) {
                    out[ele] = wVec[(ele - m2LocalElemBegin)];
                }

                continue;
            }
        }

        par::Mpi_Alltoallv_sparse(&(*(wVec.begin())), (int*)sendC,
                                  (int*)sendOfst, &(*(wVec_m2.begin())),
                                  (int*)recvC, (int*)recvOfst, comm);

        if (pMesh->isActive()) {
            const unsigned int m2LocalElemBegin = pMesh->getElementLocalBegin();
            const unsigned int m2LocalElemEnd   = pMesh->getElementLocalEnd();

            const unsigned int m2LocalNodeBegin = m2LocalElemBegin * m_uiNpE;
            const unsigned int m2LocalNodeEnd   = m2LocalElemEnd * m_uiNpE;

            unsigned int lookUp;
            const unsigned int eleOrder = pMesh->getElementOrder();

            for (unsigned int ele = m2LocalElemBegin; ele < m2LocalElemEnd;
                 ele++)
                out[ele] = wVec_m2[(ele - m2LocalElemBegin)];
        }
    }

    return;
}

template <typename T>
void Mesh::zip(const T* unzippedVec, T* zippedVec) {
    if (!m_uiIsActive) return;

    const ot::TreeNode* pNodes = m_uiAllElements.data();
    const ot::Block* blkList   = m_uiLocalBlockList.data();
    const size_t n_blocks      = m_uiLocalBlockList.size();
    const unsigned int eO      = m_uiElementOrder;
    const unsigned int npE     = m_uiNpE;
    const unsigned int eOp1    = eO + 1;
    const unsigned int eOp1Sq  = eOp1 * eOp1;

    // Zip is naturally parallel over blocks: each block's local elements own
    // a DISJOINT set of CG nodes (the dg_idx/npE == elem ownership filter
    // guarantees one writer per CG node across all blocks), and each block
    // reads from its own slice of unzippedVec. So #pragma omp parallel for
    // over blocks is race-free without any extra bookkeeping.
#if defined(DENDRO_UNZIP_OMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (size_t blk = 0; blk < n_blocks; blk++) {
        const ot::TreeNode blkNode          = blkList[blk].getBlockNode();
        const unsigned int regLev           = blkList[blk].getRegularGridLev();
        const unsigned int lx               = blkList[blk].getAllocationSzX();
        const unsigned int ly               = blkList[blk].getAllocationSzY();
        const unsigned int /*lz*/ lz_unused = blkList[blk].getAllocationSzZ();
        (void)lz_unused;
        const unsigned int offset    = blkList[blk].getOffset();
        const unsigned int paddWidth = blkList[blk].get1DPadWidth();

        for (unsigned int elem = blkList[blk].getLocalElementBegin();
             elem < blkList[blk].getLocalElementEnd(); elem++) {
            const unsigned int ei = (pNodes[elem].getX() - blkNode.getX()) >>
                                    (m_uiMaxDepth - regLev);
            const unsigned int ej = (pNodes[elem].getY() - blkNode.getY()) >>
                                    (m_uiMaxDepth - regLev);
            const unsigned int ek = (pNodes[elem].getZ() - blkNode.getZ()) >>
                                    (m_uiMaxDepth - regLev);

            assert(pNodes[elem].getLevel() == regLev);

            for (unsigned int k = 0; k < eOp1; k++)
                for (unsigned int j = 0; j < eOp1; j++)
                    for (unsigned int i = 0; i < eOp1; i++) {
                        const unsigned int dg_lookup =
                            m_uiE2NMapping_DG[elem * npE + k * eOp1Sq +
                                              j * eOp1 + i];
                        if ((dg_lookup / npE) == elem) {
                            zippedVec[m_uiE2NMapping_CG[elem * npE +
                                                        k * eOp1Sq + j * eOp1 +
                                                        i]] =
                                unzippedVec[offset +
                                            (ek * eO + k + paddWidth) *
                                                (ly * lx) +
                                            (ej * eO + j + paddWidth) * lx +
                                            (ei * eO + i + paddWidth)];
                        }
                    }
        }
    }
}

template <typename T>
void Mesh::OCT_DIR_LEFT_DOWN_Unzip(const ot::Block& blk, const T* zippedVec,
                                   T* unzippedVec, T* eleDGVec,
                                   bool* eleDGValid) {
    const unsigned int* blk2diagMap = blk.getBlk2DiagMap();
    unsigned int lookUp;  // first OCT_DIR_LEFT_DOWN element.

    const unsigned int rank       = getMPIRank();
    const unsigned int regLev     = blk.getRegularGridLev();
    const unsigned int blkElem_1D = 1u
                                    << (regLev - blk.getBlockNode().getLevel());
    const unsigned int paddWidth  = blk.get1DPadWidth();
    const unsigned int offset     = blk.getOffset();
    const unsigned int lx         = blk.getAllocationSzX();
    const unsigned int ly         = blk.getAllocationSzY();
    const unsigned int lz         = blk.getAllocationSzZ();

    unsigned int edgeCount        = 0;
    const unsigned int edgeDir    = (OCT_DIR_LEFT_DOWN - EDGE_OFFSET);
    lookUp = blk2diagMap[edgeDir * (2 * blkElem_1D) + 2 * edgeCount];
    if (lookUp == LOOK_UP_TABLE_DEFAULT) return;

    const unsigned int ei      = 0;
    const unsigned int ej      = 0;
    unsigned int ek            = 0;

    const unsigned int dir1    = OCT_DIR_LEFT;
    const unsigned int dir2    = OCT_DIR_DOWN;
    const unsigned int dir3    = OCT_DIR_FRONT;
    const unsigned int dir4    = OCT_DIR_RIGHT;
    const ot::TreeNode blkNode = blk.getBlockNode();
    const unsigned int sz      = 1u << (m_uiMaxDepth - regLev);

    const ot::TreeNode* pNodes = &(*(m_uiAllElements.begin()));

    std::vector<T> interpIn;
    interpIn.resize(m_uiNpE);

    std::vector<T> interpOut;
    interpOut.resize(m_uiNpE);

    const int i_offset       = -(m_uiElementOrder - paddWidth);
    const int j_offset       = -(m_uiElementOrder - paddWidth);
    const int k_offset       = (paddWidth);

    const unsigned int kb    = 0;
    const unsigned int ke    = (m_uiElementOrder + 1);
    const unsigned int jb    = (m_uiElementOrder - paddWidth);
    const unsigned int je    = (m_uiElementOrder + 1);
    const unsigned int ib    = (m_uiElementOrder - paddWidth);
    const unsigned int ie    = (m_uiElementOrder + 1);

    const unsigned int cnum1 = 3;
    const unsigned int cnum2 = 7;
    unsigned int cnum;

    unsigned int nodeLookUp_CG;
    bool isHanging;

    unsigned int bflag = blk.getBlkNodeFlag();

    // std::cout<<" lookup : "<<pNodes[lookUp]<<" blkNode:
    // "<<blk.getBlockNode()<<std::endl;

    while (edgeCount < blkElem_1D) {
        ek     = edgeCount;
        lookUp = blk2diagMap[edgeDir * (2 * blkElem_1D) + 2 * edgeCount];
        assert(lookUp != LOOK_UP_TABLE_DEFAULT);
        if (pNodes[lookUp].getLevel() == regLev) {
            T* lookUpVec = &eleDGVec[lookUp * m_uiNpE];
            if (!eleDGValid[lookUp]) {
                getElementNodalValues(zippedVec, lookUpVec, lookUp);
                eleDGValid[lookUp] = true;
            }
            for (unsigned int k = kb; k < ke; k++)
                for (unsigned int j = jb; j < je; j++)
                    for (unsigned int i = ib; i < ie; i++) {
                        unzippedVec[offset +
                                    (ek * m_uiElementOrder + k + k_offset) *
                                        (ly * lx) +
                                    (ej * m_uiElementOrder + j + j_offset) *
                                        (lx) +
                                    (ei * m_uiElementOrder + i + i_offset)] =
                            lookUpVec[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i];
                    }

            edgeCount += 1;

        } else if (pNodes[lookUp].getLevel() < regLev) {
            if ((pNodes[lookUp].getLevel() + 1) != regLev)
                std::cout << "LEFT_DOWN_DIAG_UNIZIP ERROR: 2:1 balance error "
                          << std::endl;

            T* lookUpVec = &eleDGVec[lookUp * m_uiNpE];
            if (!eleDGValid[lookUp]) {
                getElementNodalValues(zippedVec, lookUpVec, lookUp);
                eleDGValid[lookUp] = true;
            }
            // note this might not be the cnum1 cnum2.
            cnum = ot::TreeNode(blkNode.minX() - sz, blkNode.minY() - sz,
                                blkNode.minZ() + ek * sz, regLev, m_uiDim,
                                m_uiMaxDepth)
                       .getMortonIndex();
            parent2ChildInterpolation(lookUpVec, &(*(interpOut.begin())), cnum,
                                      3);
            for (unsigned int k = kb; k < ke; k++)
                for (unsigned int j = jb; j < je; j++)
                    for (unsigned int i = ib; i < ie; i++) {
                        unzippedVec[offset +
                                    (ek * m_uiElementOrder + k + k_offset) *
                                        (ly * lx) +
                                    (ej * m_uiElementOrder + j + j_offset) *
                                        (lx) +
                                    (ei * m_uiElementOrder + i + i_offset)] =
                            interpOut[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i];
                    }

            ek = edgeCount + 1;
            if (ek < blkElem_1D) {
                cnum = ot::TreeNode(blkNode.minX() - sz, blkNode.minY() - sz,
                                    blkNode.minZ() + ek * sz, regLev, m_uiDim,
                                    m_uiMaxDepth)
                           .getMortonIndex();
                parent2ChildInterpolation(lookUpVec, &(*(interpOut.begin())),
                                          cnum, 3);
                for (unsigned int k = kb; k < ke; k++)
                    for (unsigned int j = jb; j < je; j++)
                        for (unsigned int i = ib; i < ie; i++) {
                            unzippedVec[offset +
                                        (ek * m_uiElementOrder + k + k_offset) *
                                            (ly * lx) +
                                        (ej * m_uiElementOrder + j + j_offset) *
                                            (lx) +
                                        (ei * m_uiElementOrder + i +
                                         i_offset)] =
                                interpOut[k * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          j * (m_uiElementOrder + 1) + i];
                        }
            }

            edgeCount += 2;

        } else {
            assert(pNodes[lookUp].getLevel() == (regLev + 1));
            if (pNodes[lookUp].getLevel() != (regLev + 1))
                std::cout << "LEFT_DOWN_DIAG_UNIZIP ERROR: 2:1 balance error "
                          << std::endl;

            cnum = pNodes[lookUp].getMortonIndex();
            assert(cnum == cnum1);
            for (unsigned int k = 0; k < m_uiElementOrder + 1; k++)
                for (unsigned int j = 0; j < m_uiElementOrder + 1; j++)
                    for (unsigned int i = 0; i < m_uiElementOrder + 1; i++) {
                        isHanging = isNodeHanging(lookUp, i, j, k);
                        nodeLookUp_CG =
                            m_uiE2NMapping_CG[lookUp * m_uiNpE +
                                              k * (m_uiElementOrder + 1) *
                                                  (m_uiElementOrder + 1) +
                                              j * (m_uiElementOrder + 1) + i];
                        if (isHanging) {
                            interpOut[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i] =
                                zippedVec[nodeLookUp_CG];
                        } else {
                            const unsigned int iix =
                                m_uiElementOrder * (int)(cnum & 1u) + i;
                            const unsigned int jjy =
                                m_uiElementOrder * (int)((cnum & 2u) >> 1u) + j;
                            const unsigned int kkz =
                                m_uiElementOrder * (int)((cnum & 4u) >> 2u) + k;
                            // std::cout<<" iix: "<<iix<<" jjy: "<<jjy<<" kkz:
                            // "<<kkz<<std::endl;

                            if ((iix % 2 == 0) && (jjy % 2 == 0) &&
                                (kkz % 2 == 0)) {
                                interpOut[(kkz >> 1u) * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          (jjy >> 1u) * (m_uiElementOrder + 1) +
                                          (iix >> 1u)] =
                                    zippedVec[nodeLookUp_CG];
                            }
                        }
                    }

            lookUp =
                blk2diagMap[edgeDir * (2 * blkElem_1D) + 2 * edgeCount + 1];

            assert(lookUp != LOOK_UP_TABLE_DEFAULT);
            assert(pNodes[lookUp].getLevel() == (regLev + 1));
            cnum = pNodes[lookUp].getMortonIndex();
            assert(cnum == cnum2);
            for (unsigned int k = 0; k < m_uiElementOrder + 1; k++)
                for (unsigned int j = 0; j < m_uiElementOrder + 1; j++)
                    for (unsigned int i = 0; i < m_uiElementOrder + 1; i++) {
                        isHanging = isNodeHanging(lookUp, i, j, k);
                        nodeLookUp_CG =
                            m_uiE2NMapping_CG[lookUp * m_uiNpE +
                                              k * (m_uiElementOrder + 1) *
                                                  (m_uiElementOrder + 1) +
                                              j * (m_uiElementOrder + 1) + i];
                        if (isHanging) {
                            interpOut[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i] =
                                zippedVec[nodeLookUp_CG];
                        } else {
                            const unsigned int iix =
                                m_uiElementOrder * (int)(cnum & 1u) + i;
                            const unsigned int jjy =
                                m_uiElementOrder * (int)((cnum & 2u) >> 1u) + j;
                            const unsigned int kkz =
                                m_uiElementOrder * (int)((cnum & 4u) >> 2u) + k;
                            // std::cout<<" iix: "<<iix<<" jjy: "<<jjy<<" kkz:
                            // "<<kkz<<std::endl;

                            if ((iix % 2 == 0) && (jjy % 2 == 0) &&
                                (kkz % 2 == 0)) {
                                interpOut[(kkz >> 1u) * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          (jjy >> 1u) * (m_uiElementOrder + 1) +
                                          (iix >> 1u)] =
                                    zippedVec[nodeLookUp_CG];
                            }
                        }
                    }

            for (unsigned int k = kb; k < ke; k++)
                for (unsigned int j = jb; j < je; j++)
                    for (unsigned int i = ib; i < ie; i++) {
                        unzippedVec[offset +
                                    (ek * m_uiElementOrder + k + k_offset) *
                                        (ly * lx) +
                                    (ej * m_uiElementOrder + j + j_offset) *
                                        (lx) +
                                    (ei * m_uiElementOrder + i + i_offset)] =
                            interpOut[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i];
                    }

            edgeCount += 1;
        }
    }
}

template <typename T>
void Mesh::OCT_DIR_LEFT_UP_Unzip(const ot::Block& blk, const T* zippedVec,
                                 T* unzippedVec, T* eleDGVec,
                                 bool* eleDGValid) {
    const unsigned int* blk2diagMap = blk.getBlk2DiagMap();
    unsigned int lookUp;  // first OCT_DIR_LEFT_DOWN element.

    const unsigned int rank       = getMPIRank();
    const unsigned int regLev     = blk.getRegularGridLev();
    const unsigned int blkElem_1D = 1u
                                    << (regLev - blk.getBlockNode().getLevel());
    const unsigned int paddWidth  = blk.get1DPadWidth();
    const unsigned int offset     = blk.getOffset();
    const unsigned int lx         = blk.getAllocationSzX();
    const unsigned int ly         = blk.getAllocationSzY();
    const unsigned int lz         = blk.getAllocationSzZ();

    unsigned int edgeCount        = 0;
    const unsigned int edgeDir    = (OCT_DIR_LEFT_UP - EDGE_OFFSET);
    lookUp = blk2diagMap[edgeDir * (2 * blkElem_1D) + 2 * edgeCount];
    if (lookUp == LOOK_UP_TABLE_DEFAULT) return;

    const unsigned int ei      = 0;
    const unsigned int ej      = blkElem_1D - 1;
    unsigned int ek            = 0;

    const unsigned int dir1    = OCT_DIR_LEFT;
    const unsigned int dir2    = OCT_DIR_UP;
    const unsigned int dir3    = OCT_DIR_FRONT;
    const unsigned int dir4    = OCT_DIR_RIGHT;
    const ot::TreeNode blkNode = blk.getBlockNode();
    const unsigned int sz      = 1u << (m_uiMaxDepth - regLev);

    const ot::TreeNode* pNodes = &(*(m_uiAllElements.begin()));

    std::vector<T> interpIn;
    interpIn.resize(m_uiNpE);

    std::vector<T> interpOut;
    interpOut.resize(m_uiNpE);

    const int i_offset       = -(m_uiElementOrder - paddWidth);
    const int j_offset       = (m_uiElementOrder + paddWidth);
    const int k_offset       = (paddWidth);

    const unsigned int kb    = 0;
    const unsigned int ke    = (m_uiElementOrder + 1);
    const unsigned int jb    = 0;
    const unsigned int je    = paddWidth + 1;
    const unsigned int ib    = (m_uiElementOrder - paddWidth);
    const unsigned int ie    = (m_uiElementOrder + 1);

    const unsigned int cnum1 = 1;
    const unsigned int cnum2 = 5;
    unsigned int cnum;

    unsigned int nodeLookUp_CG;
    bool isHanging;

    while (edgeCount < blkElem_1D) {
        ek     = edgeCount;
        lookUp = blk2diagMap[edgeDir * (2 * blkElem_1D) + 2 * edgeCount];
        assert(lookUp != LOOK_UP_TABLE_DEFAULT);
        if (pNodes[lookUp].getLevel() == regLev) {
            T* lookUpVec = &eleDGVec[lookUp * m_uiNpE];
            if (!eleDGValid[lookUp]) {
                getElementNodalValues(zippedVec, lookUpVec, lookUp);
                eleDGValid[lookUp] = true;
            }
            for (unsigned int k = kb; k < ke; k++)
                for (unsigned int j = jb; j < je; j++)
                    for (unsigned int i = ib; i < ie; i++) {
                        unzippedVec[offset +
                                    (ek * m_uiElementOrder + k + k_offset) *
                                        (ly * lx) +
                                    (ej * m_uiElementOrder + j + j_offset) *
                                        (lx) +
                                    (ei * m_uiElementOrder + i + i_offset)] =
                            lookUpVec[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i];
                    }

            edgeCount += 1;

        } else if (pNodes[lookUp].getLevel() < regLev) {
            if ((pNodes[lookUp].getLevel() + 1) != regLev)
                std::cout << "LEFT_UP_DIAG_UNIZIP ERROR: 2:1 balance error "
                          << std::endl;

            T* lookUpVec = &eleDGVec[lookUp * m_uiNpE];
            if (!eleDGValid[lookUp]) {
                getElementNodalValues(zippedVec, lookUpVec, lookUp);
                eleDGValid[lookUp] = true;
            }
            cnum = ot::TreeNode(blkNode.minX() - sz, blkNode.maxY(),
                                blkNode.minZ() + ek * sz, regLev, m_uiDim,
                                m_uiMaxDepth)
                       .getMortonIndex();
            parent2ChildInterpolation(lookUpVec, &(*(interpOut.begin())), cnum,
                                      3);
            for (unsigned int k = kb; k < ke; k++)
                for (unsigned int j = jb; j < je; j++)
                    for (unsigned int i = ib; i < ie; i++) {
                        unzippedVec[offset +
                                    (ek * m_uiElementOrder + k + k_offset) *
                                        (ly * lx) +
                                    (ej * m_uiElementOrder + j + j_offset) *
                                        (lx) +
                                    (ei * m_uiElementOrder + i + i_offset)] =
                            interpOut[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i];
                    }

            ek = edgeCount + 1;

            if (ek < blkElem_1D) {
                cnum = ot::TreeNode(blkNode.minX() - sz, blkNode.maxY(),
                                    blkNode.minZ() + ek * sz, regLev, m_uiDim,
                                    m_uiMaxDepth)
                           .getMortonIndex();
                parent2ChildInterpolation(lookUpVec, &(*(interpOut.begin())),
                                          cnum, 3);
                for (unsigned int k = kb; k < ke; k++)
                    for (unsigned int j = jb; j < je; j++)
                        for (unsigned int i = ib; i < ie; i++) {
                            unzippedVec[offset +
                                        (ek * m_uiElementOrder + k + k_offset) *
                                            (ly * lx) +
                                        (ej * m_uiElementOrder + j + j_offset) *
                                            (lx) +
                                        (ei * m_uiElementOrder + i +
                                         i_offset)] =
                                interpOut[k * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          j * (m_uiElementOrder + 1) + i];
                        }
            }

            edgeCount += 2;

        } else {
            assert(pNodes[lookUp].getLevel() == (regLev + 1));
            if (pNodes[lookUp].getLevel() != (regLev + 1))
                std::cout << "LEFT_UP_DIAG_UNIZIP ERROR: 2:1 balance error "
                          << std::endl;

            cnum = pNodes[lookUp].getMortonIndex();
            assert(cnum == cnum1);
            for (unsigned int k = 0; k < m_uiElementOrder + 1; k++)
                for (unsigned int j = 0; j < m_uiElementOrder + 1; j++)
                    for (unsigned int i = 0; i < m_uiElementOrder + 1; i++) {
                        isHanging = isNodeHanging(lookUp, i, j, k);
                        nodeLookUp_CG =
                            m_uiE2NMapping_CG[lookUp * m_uiNpE +
                                              k * (m_uiElementOrder + 1) *
                                                  (m_uiElementOrder + 1) +
                                              j * (m_uiElementOrder + 1) + i];
                        if (isHanging) {
                            interpOut[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i] =
                                zippedVec[nodeLookUp_CG];
                        } else {
                            const unsigned int iix =
                                m_uiElementOrder * (int)(cnum & 1u) + i;
                            const unsigned int jjy =
                                m_uiElementOrder * (int)((cnum & 2u) >> 1u) + j;
                            const unsigned int kkz =
                                m_uiElementOrder * (int)((cnum & 4u) >> 2u) + k;
                            // std::cout<<" iix: "<<iix<<" jjy: "<<jjy<<" kkz:
                            // "<<kkz<<std::endl;

                            if ((iix % 2 == 0) && (jjy % 2 == 0) &&
                                (kkz % 2 == 0)) {
                                interpOut[(kkz >> 1u) * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          (jjy >> 1u) * (m_uiElementOrder + 1) +
                                          (iix >> 1u)] =
                                    zippedVec[nodeLookUp_CG];
                            }
                        }
                    }

            lookUp =
                blk2diagMap[edgeDir * (2 * blkElem_1D) + 2 * edgeCount + 1];

            assert(lookUp != LOOK_UP_TABLE_DEFAULT);
            assert(pNodes[lookUp].getLevel() == (regLev + 1));
            cnum = pNodes[lookUp].getMortonIndex();
            assert(cnum == cnum2);
            for (unsigned int k = 0; k < m_uiElementOrder + 1; k++)
                for (unsigned int j = 0; j < m_uiElementOrder + 1; j++)
                    for (unsigned int i = 0; i < m_uiElementOrder + 1; i++) {
                        isHanging = isNodeHanging(lookUp, i, j, k);
                        nodeLookUp_CG =
                            m_uiE2NMapping_CG[lookUp * m_uiNpE +
                                              k * (m_uiElementOrder + 1) *
                                                  (m_uiElementOrder + 1) +
                                              j * (m_uiElementOrder + 1) + i];
                        if (isHanging) {
                            interpOut[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i] =
                                zippedVec[nodeLookUp_CG];
                        } else {
                            const unsigned int iix =
                                m_uiElementOrder * (int)(cnum & 1u) + i;
                            const unsigned int jjy =
                                m_uiElementOrder * (int)((cnum & 2u) >> 1u) + j;
                            const unsigned int kkz =
                                m_uiElementOrder * (int)((cnum & 4u) >> 2u) + k;
                            // std::cout<<" iix: "<<iix<<" jjy: "<<jjy<<" kkz:
                            // "<<kkz<<std::endl;

                            if ((iix % 2 == 0) && (jjy % 2 == 0) &&
                                (kkz % 2 == 0)) {
                                interpOut[(kkz >> 1u) * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          (jjy >> 1u) * (m_uiElementOrder + 1) +
                                          (iix >> 1u)] =
                                    zippedVec[nodeLookUp_CG];
                            }
                        }
                    }

            for (unsigned int k = kb; k < ke; k++)
                for (unsigned int j = jb; j < je; j++)
                    for (unsigned int i = ib; i < ie; i++) {
                        unzippedVec[offset +
                                    (ek * m_uiElementOrder + k + k_offset) *
                                        (ly * lx) +
                                    (ej * m_uiElementOrder + j + j_offset) *
                                        (lx) +
                                    (ei * m_uiElementOrder + i + i_offset)] =
                            interpOut[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i];
                    }

            edgeCount += 1;
        }
    }
}

template <typename T>
void Mesh::OCT_DIR_LEFT_BACK_Unzip(const ot::Block& blk, const T* zippedVec,
                                   T* unzippedVec, T* eleDGVec,
                                   bool* eleDGValid) {
    const unsigned int* blk2diagMap = blk.getBlk2DiagMap();
    unsigned int lookUp;  // first OCT_DIR_LEFT_DOWN element.

    const unsigned int rank       = getMPIRank();
    const unsigned int regLev     = blk.getRegularGridLev();
    const unsigned int blkElem_1D = 1u
                                    << (regLev - blk.getBlockNode().getLevel());
    const unsigned int paddWidth  = blk.get1DPadWidth();
    const unsigned int offset     = blk.getOffset();
    const unsigned int lx         = blk.getAllocationSzX();
    const unsigned int ly         = blk.getAllocationSzY();
    const unsigned int lz         = blk.getAllocationSzZ();

    unsigned int edgeCount        = 0;
    const unsigned int edgeDir    = (OCT_DIR_LEFT_BACK - EDGE_OFFSET);
    lookUp = blk2diagMap[edgeDir * (2 * blkElem_1D) + 2 * edgeCount];
    if (lookUp == LOOK_UP_TABLE_DEFAULT) return;

    const unsigned int ei      = 0;
    unsigned int ej            = 0;
    const unsigned int ek      = 0;

    const unsigned int dir1    = OCT_DIR_LEFT;
    const unsigned int dir2    = OCT_DIR_BACK;
    const unsigned int dir3    = OCT_DIR_UP;
    const unsigned int dir4    = OCT_DIR_RIGHT;
    const unsigned int dir5    = OCT_DIR_FRONT;
    const ot::TreeNode blkNode = blk.getBlockNode();
    const unsigned int sz      = 1u << (m_uiMaxDepth - regLev);

    const ot::TreeNode* pNodes = &(*(m_uiAllElements.begin()));

    std::vector<T> interpIn;
    interpIn.resize(m_uiNpE);

    std::vector<T> interpOut;
    interpOut.resize(m_uiNpE);

    const int i_offset       = -(m_uiElementOrder - paddWidth);
    const int j_offset       = paddWidth;
    const int k_offset       = -(m_uiElementOrder - paddWidth);

    const unsigned int kb    = (m_uiElementOrder - paddWidth);
    const unsigned int ke    = (m_uiElementOrder + 1);

    const unsigned int jb    = 0;
    const unsigned int je    = (m_uiElementOrder + 1);

    const unsigned int ib    = (m_uiElementOrder - paddWidth);
    const unsigned int ie    = (m_uiElementOrder + 1);

    const unsigned int cnum1 = 5;
    const unsigned int cnum2 = 7;
    unsigned int cnum;

    unsigned int nodeLookUp_CG;
    bool isHanging;

    while (edgeCount < blkElem_1D) {
        ej     = edgeCount;
        lookUp = blk2diagMap[edgeDir * (2 * blkElem_1D) + 2 * edgeCount];
        assert(lookUp != LOOK_UP_TABLE_DEFAULT);
        if (pNodes[lookUp].getLevel() == regLev) {
            T* lookUpVec = &eleDGVec[lookUp * m_uiNpE];
            if (!eleDGValid[lookUp]) {
                getElementNodalValues(zippedVec, lookUpVec, lookUp);
                eleDGValid[lookUp] = true;
            }
            for (unsigned int k = kb; k < ke; k++)
                for (unsigned int j = jb; j < je; j++)
                    for (unsigned int i = ib; i < ie; i++) {
                        unzippedVec[offset +
                                    (ek * m_uiElementOrder + k + k_offset) *
                                        (ly * lx) +
                                    (ej * m_uiElementOrder + j + j_offset) *
                                        (lx) +
                                    (ei * m_uiElementOrder + i + i_offset)] =
                            lookUpVec[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i];
                    }

            edgeCount += 1;

        } else if (pNodes[lookUp].getLevel() < regLev) {
            if ((pNodes[lookUp].getLevel() + 1) != regLev)
                std::cout << "LEFT_BACK_DIAG_UNIZIP ERROR: 2:1 balance error "
                          << std::endl;

            T* lookUpVec = &eleDGVec[lookUp * m_uiNpE];
            if (!eleDGValid[lookUp]) {
                getElementNodalValues(zippedVec, lookUpVec, lookUp);
                eleDGValid[lookUp] = true;
            }
            cnum =
                ot::TreeNode(blkNode.minX() - sz, blkNode.minY() + ej * sz,
                             blkNode.minZ() - sz, regLev, m_uiDim, m_uiMaxDepth)
                    .getMortonIndex();
            parent2ChildInterpolation(lookUpVec, &(*(interpOut.begin())), cnum,
                                      3);
            for (unsigned int k = kb; k < ke; k++)
                for (unsigned int j = jb; j < je; j++)
                    for (unsigned int i = ib; i < ie; i++) {
                        unzippedVec[offset +
                                    (ek * m_uiElementOrder + k + k_offset) *
                                        (ly * lx) +
                                    (ej * m_uiElementOrder + j + j_offset) *
                                        (lx) +
                                    (ei * m_uiElementOrder + i + i_offset)] =
                            interpOut[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i];
                    }

            ej = edgeCount + 1;
            if (ej < blkElem_1D) {
                cnum = ot::TreeNode(
                           blkNode.minX() - sz, blkNode.minY() + ej * sz,
                           blkNode.minZ() - sz, regLev, m_uiDim, m_uiMaxDepth)
                           .getMortonIndex();
                parent2ChildInterpolation(lookUpVec, &(*(interpOut.begin())),
                                          cnum, 3);
                for (unsigned int k = kb; k < ke; k++)
                    for (unsigned int j = jb; j < je; j++)
                        for (unsigned int i = ib; i < ie; i++) {
                            unzippedVec[offset +
                                        (ek * m_uiElementOrder + k + k_offset) *
                                            (ly * lx) +
                                        (ej * m_uiElementOrder + j + j_offset) *
                                            (lx) +
                                        (ei * m_uiElementOrder + i +
                                         i_offset)] =
                                interpOut[k * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          j * (m_uiElementOrder + 1) + i];
                        }
            }

            edgeCount += 2;

        } else {
            assert(pNodes[lookUp].getLevel() == (regLev + 1));
            if (pNodes[lookUp].getLevel() != (regLev + 1))
                std::cout << "LEFT_FRONT_BACK_UNIZIP ERROR: 2:1 balance error "
                          << std::endl;

            cnum = pNodes[lookUp].getMortonIndex();
            assert(cnum == cnum1);
            for (unsigned int k = 0; k < m_uiElementOrder + 1; k++)
                for (unsigned int j = 0; j < m_uiElementOrder + 1; j++)
                    for (unsigned int i = 0; i < m_uiElementOrder + 1; i++) {
                        isHanging = isNodeHanging(lookUp, i, j, k);
                        nodeLookUp_CG =
                            m_uiE2NMapping_CG[lookUp * m_uiNpE +
                                              k * (m_uiElementOrder + 1) *
                                                  (m_uiElementOrder + 1) +
                                              j * (m_uiElementOrder + 1) + i];
                        if (isHanging) {
                            interpOut[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i] =
                                zippedVec[nodeLookUp_CG];
                        } else {
                            const unsigned int iix =
                                m_uiElementOrder * (int)(cnum & 1u) + i;
                            const unsigned int jjy =
                                m_uiElementOrder * (int)((cnum & 2u) >> 1u) + j;
                            const unsigned int kkz =
                                m_uiElementOrder * (int)((cnum & 4u) >> 2u) + k;
                            // std::cout<<" iix: "<<iix<<" jjy: "<<jjy<<" kkz:
                            // "<<kkz<<std::endl;

                            if ((iix % 2 == 0) && (jjy % 2 == 0) &&
                                (kkz % 2 == 0)) {
                                interpOut[(kkz >> 1u) * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          (jjy >> 1u) * (m_uiElementOrder + 1) +
                                          (iix >> 1u)] =
                                    zippedVec[nodeLookUp_CG];
                            }
                        }
                    }

            lookUp =
                blk2diagMap[edgeDir * (2 * blkElem_1D) + 2 * edgeCount + 1];
            assert(lookUp != LOOK_UP_TABLE_DEFAULT);
            assert(pNodes[lookUp].getLevel() == (regLev + 1));
            cnum = pNodes[lookUp].getMortonIndex();
            assert(cnum == cnum2);
            for (unsigned int k = 0; k < m_uiElementOrder + 1; k++)
                for (unsigned int j = 0; j < m_uiElementOrder + 1; j++)
                    for (unsigned int i = 0; i < m_uiElementOrder + 1; i++) {
                        isHanging = isNodeHanging(lookUp, i, j, k);
                        nodeLookUp_CG =
                            m_uiE2NMapping_CG[lookUp * m_uiNpE +
                                              k * (m_uiElementOrder + 1) *
                                                  (m_uiElementOrder + 1) +
                                              j * (m_uiElementOrder + 1) + i];
                        if (isHanging) {
                            interpOut[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i] =
                                zippedVec[nodeLookUp_CG];
                        } else {
                            const unsigned int iix =
                                m_uiElementOrder * (int)(cnum & 1u) + i;
                            const unsigned int jjy =
                                m_uiElementOrder * (int)((cnum & 2u) >> 1u) + j;
                            const unsigned int kkz =
                                m_uiElementOrder * (int)((cnum & 4u) >> 2u) + k;
                            // std::cout<<" iix: "<<iix<<" jjy: "<<jjy<<" kkz:
                            // "<<kkz<<std::endl;

                            if ((iix % 2 == 0) && (jjy % 2 == 0) &&
                                (kkz % 2 == 0)) {
                                interpOut[(kkz >> 1u) * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          (jjy >> 1u) * (m_uiElementOrder + 1) +
                                          (iix >> 1u)] =
                                    zippedVec[nodeLookUp_CG];
                            }
                        }
                    }

            for (unsigned int k = kb; k < ke; k++)
                for (unsigned int j = jb; j < je; j++)
                    for (unsigned int i = ib; i < ie; i++) {
                        unzippedVec[offset +
                                    (ek * m_uiElementOrder + k + k_offset) *
                                        (ly * lx) +
                                    (ej * m_uiElementOrder + j + j_offset) *
                                        (lx) +
                                    (ei * m_uiElementOrder + i + i_offset)] =
                            interpOut[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i];
                    }

            edgeCount += 1;
        }
    }
}

template <typename T>
void Mesh::OCT_DIR_LEFT_FRONT_Unzip(const ot::Block& blk, const T* zippedVec,
                                    T* unzippedVec, T* eleDGVec,
                                    bool* eleDGValid) {
    const unsigned int* blk2diagMap = blk.getBlk2DiagMap();
    unsigned int lookUp;  // first OCT_DIR_LEFT_DOWN element.

    const unsigned int rank       = getMPIRank();
    const unsigned int regLev     = blk.getRegularGridLev();
    const unsigned int blkElem_1D = 1u
                                    << (regLev - blk.getBlockNode().getLevel());
    const unsigned int paddWidth  = blk.get1DPadWidth();
    const unsigned int offset     = blk.getOffset();
    const unsigned int lx         = blk.getAllocationSzX();
    const unsigned int ly         = blk.getAllocationSzY();
    const unsigned int lz         = blk.getAllocationSzZ();

    unsigned int edgeCount        = 0;
    const unsigned int edgeDir    = (OCT_DIR_LEFT_FRONT - EDGE_OFFSET);
    lookUp = blk2diagMap[edgeDir * (2 * blkElem_1D) + 2 * edgeCount];
    if (lookUp == LOOK_UP_TABLE_DEFAULT) return;

    const unsigned int ei      = 0;
    unsigned int ej            = 0;
    const unsigned int ek      = (blkElem_1D - 1);

    const unsigned int dir1    = OCT_DIR_LEFT;
    const unsigned int dir2    = OCT_DIR_FRONT;
    const unsigned int dir3    = OCT_DIR_UP;
    const unsigned int dir4    = OCT_DIR_RIGHT;

    const ot::TreeNode blkNode = blk.getBlockNode();
    const unsigned int sz      = 1u << (m_uiMaxDepth - regLev);

    const ot::TreeNode* pNodes = &(*(m_uiAllElements.begin()));

    std::vector<T> interpIn;
    interpIn.resize(m_uiNpE);

    std::vector<T> interpOut;
    interpOut.resize(m_uiNpE);

    const int i_offset       = -(m_uiElementOrder - paddWidth);
    const int j_offset       = paddWidth;
    const int k_offset       = (m_uiElementOrder + paddWidth);

    const unsigned int kb    = 0;
    const unsigned int ke    = (paddWidth + 1);

    const unsigned int jb    = 0;
    const unsigned int je    = (m_uiElementOrder + 1);

    const unsigned int ib    = (m_uiElementOrder - paddWidth);
    const unsigned int ie    = (m_uiElementOrder + 1);

    const unsigned int cnum1 = 1;
    const unsigned int cnum2 = 3;
    unsigned int cnum;

    unsigned int nodeLookUp_CG;
    bool isHanging;

    while (edgeCount < blkElem_1D) {
        ej     = edgeCount;
        lookUp = blk2diagMap[edgeDir * (2 * blkElem_1D) + 2 * edgeCount];
        assert(lookUp != LOOK_UP_TABLE_DEFAULT);
        if (pNodes[lookUp].getLevel() == regLev) {
            T* lookUpVec = &eleDGVec[lookUp * m_uiNpE];
            if (!eleDGValid[lookUp]) {
                getElementNodalValues(zippedVec, lookUpVec, lookUp);
                eleDGValid[lookUp] = true;
            }
            for (unsigned int k = kb; k < ke; k++)
                for (unsigned int j = jb; j < je; j++)
                    for (unsigned int i = ib; i < ie; i++) {
                        unzippedVec[offset +
                                    (ek * m_uiElementOrder + k + k_offset) *
                                        (ly * lx) +
                                    (ej * m_uiElementOrder + j + j_offset) *
                                        (lx) +
                                    (ei * m_uiElementOrder + i + i_offset)] =
                            lookUpVec[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i];
                    }

            edgeCount += 1;

        } else if (pNodes[lookUp].getLevel() < regLev) {
            if ((pNodes[lookUp].getLevel() + 1) != regLev)
                std::cout << "LEFT_FRONT_DIAG_UNIZIP ERROR: 2:1 balance error "
                          << std::endl;

            T* lookUpVec = &eleDGVec[lookUp * m_uiNpE];
            if (!eleDGValid[lookUp]) {
                getElementNodalValues(zippedVec, lookUpVec, lookUp);
                eleDGValid[lookUp] = true;
            }
            cnum = ot::TreeNode(blkNode.minX() - sz, blkNode.minY() + ej * sz,
                                blkNode.maxZ(), regLev, m_uiDim, m_uiMaxDepth)
                       .getMortonIndex();
            parent2ChildInterpolation(lookUpVec, &(*(interpOut.begin())), cnum,
                                      3);
            for (unsigned int k = kb; k < ke; k++)
                for (unsigned int j = jb; j < je; j++)
                    for (unsigned int i = ib; i < ie; i++) {
                        unzippedVec[offset +
                                    (ek * m_uiElementOrder + k + k_offset) *
                                        (ly * lx) +
                                    (ej * m_uiElementOrder + j + j_offset) *
                                        (lx) +
                                    (ei * m_uiElementOrder + i + i_offset)] =
                            interpOut[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i];
                    }

            ej = edgeCount + 1;

            if (ej < blkElem_1D) {
                cnum =
                    ot::TreeNode(blkNode.minX() - sz, blkNode.minY() + ej * sz,
                                 blkNode.maxZ(), regLev, m_uiDim, m_uiMaxDepth)
                        .getMortonIndex();
                parent2ChildInterpolation(lookUpVec, &(*(interpOut.begin())),
                                          cnum, 3);
                for (unsigned int k = kb; k < ke; k++)
                    for (unsigned int j = jb; j < je; j++)
                        for (unsigned int i = ib; i < ie; i++) {
                            unzippedVec[offset +
                                        (ek * m_uiElementOrder + k + k_offset) *
                                            (ly * lx) +
                                        (ej * m_uiElementOrder + j + j_offset) *
                                            (lx) +
                                        (ei * m_uiElementOrder + i +
                                         i_offset)] =
                                interpOut[k * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          j * (m_uiElementOrder + 1) + i];
                        }
            }

            edgeCount += 2;

        } else {
            assert(pNodes[lookUp].getLevel() == (regLev + 1));
            if (pNodes[lookUp].getLevel() != (regLev + 1))
                std::cout << "LEFT_FRONT_DIAG_UNIZIP ERROR: 2:1 balance error "
                          << std::endl;

            cnum = pNodes[lookUp].getMortonIndex();
            assert(cnum == cnum1);
            for (unsigned int k = 0; k < m_uiElementOrder + 1; k++)
                for (unsigned int j = 0; j < m_uiElementOrder + 1; j++)
                    for (unsigned int i = 0; i < m_uiElementOrder + 1; i++) {
                        isHanging = isNodeHanging(lookUp, i, j, k);
                        nodeLookUp_CG =
                            m_uiE2NMapping_CG[lookUp * m_uiNpE +
                                              k * (m_uiElementOrder + 1) *
                                                  (m_uiElementOrder + 1) +
                                              j * (m_uiElementOrder + 1) + i];
                        if (isHanging) {
                            interpOut[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i] =
                                zippedVec[nodeLookUp_CG];
                        } else {
                            const unsigned int iix =
                                m_uiElementOrder * (int)(cnum & 1u) + i;
                            const unsigned int jjy =
                                m_uiElementOrder * (int)((cnum & 2u) >> 1u) + j;
                            const unsigned int kkz =
                                m_uiElementOrder * (int)((cnum & 4u) >> 2u) + k;
                            // std::cout<<" iix: "<<iix<<" jjy: "<<jjy<<" kkz:
                            // "<<kkz<<std::endl;

                            if ((iix % 2 == 0) && (jjy % 2 == 0) &&
                                (kkz % 2 == 0)) {
                                interpOut[(kkz >> 1u) * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          (jjy >> 1u) * (m_uiElementOrder + 1) +
                                          (iix >> 1u)] =
                                    zippedVec[nodeLookUp_CG];
                            }
                        }
                    }

            lookUp =
                blk2diagMap[edgeDir * (2 * blkElem_1D) + 2 * edgeCount + 1];
            assert(lookUp != LOOK_UP_TABLE_DEFAULT);
            assert(pNodes[lookUp].getLevel() == (regLev + 1));
            cnum = pNodes[lookUp].getMortonIndex();
            assert(cnum == cnum2);
            for (unsigned int k = 0; k < m_uiElementOrder + 1; k++)
                for (unsigned int j = 0; j < m_uiElementOrder + 1; j++)
                    for (unsigned int i = 0; i < m_uiElementOrder + 1; i++) {
                        isHanging = isNodeHanging(lookUp, i, j, k);
                        nodeLookUp_CG =
                            m_uiE2NMapping_CG[lookUp * m_uiNpE +
                                              k * (m_uiElementOrder + 1) *
                                                  (m_uiElementOrder + 1) +
                                              j * (m_uiElementOrder + 1) + i];
                        if (isHanging) {
                            interpOut[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i] =
                                zippedVec[nodeLookUp_CG];
                        } else {
                            const unsigned int iix =
                                m_uiElementOrder * (int)(cnum & 1u) + i;
                            const unsigned int jjy =
                                m_uiElementOrder * (int)((cnum & 2u) >> 1u) + j;
                            const unsigned int kkz =
                                m_uiElementOrder * (int)((cnum & 4u) >> 2u) + k;
                            // std::cout<<" iix: "<<iix<<" jjy: "<<jjy<<" kkz:
                            // "<<kkz<<std::endl;

                            if ((iix % 2 == 0) && (jjy % 2 == 0) &&
                                (kkz % 2 == 0)) {
                                interpOut[(kkz >> 1u) * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          (jjy >> 1u) * (m_uiElementOrder + 1) +
                                          (iix >> 1u)] =
                                    zippedVec[nodeLookUp_CG];
                            }
                        }
                    }

            for (unsigned int k = kb; k < ke; k++)
                for (unsigned int j = jb; j < je; j++)
                    for (unsigned int i = ib; i < ie; i++) {
                        unzippedVec[offset +
                                    (ek * m_uiElementOrder + k + k_offset) *
                                        (ly * lx) +
                                    (ej * m_uiElementOrder + j + j_offset) *
                                        (lx) +
                                    (ei * m_uiElementOrder + i + i_offset)] =
                            interpOut[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i];
                    }

            edgeCount += 1;
        }
    }
}

template <typename T>
void Mesh::OCT_DIR_RIGHT_DOWN_Unzip(const ot::Block& blk, const T* zippedVec,
                                    T* unzippedVec, T* eleDGVec,
                                    bool* eleDGValid) {
    const unsigned int* blk2diagMap = blk.getBlk2DiagMap();
    unsigned int lookUp;  // first OCT_DIR_LEFT_DOWN element.

    const unsigned int rank       = getMPIRank();
    const unsigned int regLev     = blk.getRegularGridLev();
    const unsigned int blkElem_1D = 1u
                                    << (regLev - blk.getBlockNode().getLevel());
    const unsigned int paddWidth  = blk.get1DPadWidth();
    const unsigned int offset     = blk.getOffset();
    const unsigned int lx         = blk.getAllocationSzX();
    const unsigned int ly         = blk.getAllocationSzY();
    const unsigned int lz         = blk.getAllocationSzZ();

    unsigned int edgeCount        = 0;
    const unsigned int edgeDir    = (OCT_DIR_RIGHT_DOWN - EDGE_OFFSET);
    lookUp = blk2diagMap[edgeDir * (2 * blkElem_1D) + 2 * edgeCount];
    if (lookUp == LOOK_UP_TABLE_DEFAULT) return;

    const unsigned int ei      = (blkElem_1D - 1);
    const unsigned int ej      = 0;
    unsigned int ek            = 0;

    const unsigned int dir1    = OCT_DIR_RIGHT;
    const unsigned int dir2    = OCT_DIR_DOWN;
    const unsigned int dir3    = OCT_DIR_FRONT;
    const unsigned int dir4    = OCT_DIR_UP;
    ot::TreeNode blkNode       = blk.getBlockNode();
    unsigned int sz            = 1u << (m_uiMaxDepth - regLev);

    const ot::TreeNode* pNodes = &(*(m_uiAllElements.begin()));

    std::vector<T> interpIn;
    interpIn.resize(m_uiNpE);

    std::vector<T> interpOut;
    interpOut.resize(m_uiNpE);

    const int i_offset       = (m_uiElementOrder + paddWidth);
    const int j_offset       = -(m_uiElementOrder - paddWidth);
    const int k_offset       = (paddWidth);

    const unsigned int kb    = 0;
    const unsigned int ke    = (m_uiElementOrder + 1);
    const unsigned int jb    = (m_uiElementOrder - paddWidth);
    const unsigned int je    = (m_uiElementOrder + 1);
    const unsigned int ib    = 0;
    const unsigned int ie    = paddWidth + 1;

    const unsigned int cnum1 = 2;
    const unsigned int cnum2 = 6;
    unsigned int cnum;

    unsigned int nodeLookUp_CG;
    bool isHanging;

    unsigned int bflag = blk.getBlkNodeFlag();

    // std::cout<<" lookup : "<<pNodes[lookUp]<<" blkNode:
    // "<<blk.getBlockNode()<<std::endl;

    while (edgeCount < blkElem_1D) {
        ek     = edgeCount;
        lookUp = blk2diagMap[edgeDir * (2 * blkElem_1D) + 2 * edgeCount];
        assert(lookUp != LOOK_UP_TABLE_DEFAULT);
        if (pNodes[lookUp].getLevel() == regLev) {
            T* lookUpVec = &eleDGVec[lookUp * m_uiNpE];
            if (!eleDGValid[lookUp]) {
                getElementNodalValues(zippedVec, lookUpVec, lookUp);
                eleDGValid[lookUp] = true;
            }
            for (unsigned int k = kb; k < ke; k++)
                for (unsigned int j = jb; j < je; j++)
                    for (unsigned int i = ib; i < ie; i++) {
                        unzippedVec[offset +
                                    (ek * m_uiElementOrder + k + k_offset) *
                                        (ly * lx) +
                                    (ej * m_uiElementOrder + j + j_offset) *
                                        (lx) +
                                    (ei * m_uiElementOrder + i + i_offset)] =
                            lookUpVec[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i];
                    }

            edgeCount += 1;

        } else if (pNodes[lookUp].getLevel() < regLev) {
            if ((pNodes[lookUp].getLevel() + 1) != regLev)
                std::cout << "RIGHT_DOWN_DIAG_UNIZIP ERROR: 2:1 balance error "
                          << std::endl;

            T* lookUpVec = &eleDGVec[lookUp * m_uiNpE];
            if (!eleDGValid[lookUp]) {
                getElementNodalValues(zippedVec, lookUpVec, lookUp);
                eleDGValid[lookUp] = true;
            }
            cnum = ot::TreeNode(blkNode.maxX(), blkNode.minY() - sz,
                                blkNode.minZ() + ek * sz, regLev, m_uiDim,
                                m_uiMaxDepth)
                       .getMortonIndex();
            parent2ChildInterpolation(lookUpVec, &(*(interpOut.begin())), cnum,
                                      3);
            for (unsigned int k = kb; k < ke; k++)
                for (unsigned int j = jb; j < je; j++)
                    for (unsigned int i = ib; i < ie; i++) {
                        unzippedVec[offset +
                                    (ek * m_uiElementOrder + k + k_offset) *
                                        (ly * lx) +
                                    (ej * m_uiElementOrder + j + j_offset) *
                                        (lx) +
                                    (ei * m_uiElementOrder + i + i_offset)] =
                            interpOut[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i];
                    }

            ek = edgeCount + 1;
            if (ek < blkElem_1D) {
                cnum = ot::TreeNode(blkNode.maxX(), blkNode.minY() - sz,
                                    blkNode.minZ() + ek * sz, regLev, m_uiDim,
                                    m_uiMaxDepth)
                           .getMortonIndex();
                parent2ChildInterpolation(lookUpVec, &(*(interpOut.begin())),
                                          cnum, 3);
                for (unsigned int k = kb; k < ke; k++)
                    for (unsigned int j = jb; j < je; j++)
                        for (unsigned int i = ib; i < ie; i++) {
                            unzippedVec[offset +
                                        (ek * m_uiElementOrder + k + k_offset) *
                                            (ly * lx) +
                                        (ej * m_uiElementOrder + j + j_offset) *
                                            (lx) +
                                        (ei * m_uiElementOrder + i +
                                         i_offset)] =
                                interpOut[k * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          j * (m_uiElementOrder + 1) + i];
                        }
            }

            edgeCount += 2;

        } else {
            assert(pNodes[lookUp].getLevel() == (regLev + 1));
            if (pNodes[lookUp].getLevel() != (regLev + 1))
                std::cout << "RIGHT_DOWN_DIAG_UNIZIP ERROR: 2:1 balance error "
                          << std::endl;

            cnum = pNodes[lookUp].getMortonIndex();
            assert(cnum == cnum1);
            for (unsigned int k = 0; k < m_uiElementOrder + 1; k++)
                for (unsigned int j = 0; j < m_uiElementOrder + 1; j++)
                    for (unsigned int i = 0; i < m_uiElementOrder + 1; i++) {
                        isHanging = isNodeHanging(lookUp, i, j, k);
                        nodeLookUp_CG =
                            m_uiE2NMapping_CG[lookUp * m_uiNpE +
                                              k * (m_uiElementOrder + 1) *
                                                  (m_uiElementOrder + 1) +
                                              j * (m_uiElementOrder + 1) + i];
                        if (isHanging) {
                            interpOut[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i] =
                                zippedVec[nodeLookUp_CG];
                        } else {
                            const unsigned int iix =
                                m_uiElementOrder * (int)(cnum & 1u) + i;
                            const unsigned int jjy =
                                m_uiElementOrder * (int)((cnum & 2u) >> 1u) + j;
                            const unsigned int kkz =
                                m_uiElementOrder * (int)((cnum & 4u) >> 2u) + k;
                            // std::cout<<" iix: "<<iix<<" jjy: "<<jjy<<" kkz:
                            // "<<kkz<<std::endl;

                            if ((iix % 2 == 0) && (jjy % 2 == 0) &&
                                (kkz % 2 == 0)) {
                                interpOut[(kkz >> 1u) * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          (jjy >> 1u) * (m_uiElementOrder + 1) +
                                          (iix >> 1u)] =
                                    zippedVec[nodeLookUp_CG];
                            }
                        }
                    }

            lookUp =
                blk2diagMap[edgeDir * (2 * blkElem_1D) + 2 * edgeCount + 1];
            assert(lookUp != LOOK_UP_TABLE_DEFAULT);
            assert(pNodes[lookUp].getLevel() == (regLev + 1));
            cnum = pNodes[lookUp].getMortonIndex();
            assert(cnum == cnum2);
            for (unsigned int k = 0; k < m_uiElementOrder + 1; k++)
                for (unsigned int j = 0; j < m_uiElementOrder + 1; j++)
                    for (unsigned int i = 0; i < m_uiElementOrder + 1; i++) {
                        isHanging = isNodeHanging(lookUp, i, j, k);
                        nodeLookUp_CG =
                            m_uiE2NMapping_CG[lookUp * m_uiNpE +
                                              k * (m_uiElementOrder + 1) *
                                                  (m_uiElementOrder + 1) +
                                              j * (m_uiElementOrder + 1) + i];
                        if (isHanging) {
                            interpOut[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i] =
                                zippedVec[nodeLookUp_CG];
                        } else {
                            const unsigned int iix =
                                m_uiElementOrder * (int)(cnum & 1u) + i;
                            const unsigned int jjy =
                                m_uiElementOrder * (int)((cnum & 2u) >> 1u) + j;
                            const unsigned int kkz =
                                m_uiElementOrder * (int)((cnum & 4u) >> 2u) + k;
                            // std::cout<<" iix: "<<iix<<" jjy: "<<jjy<<" kkz:
                            // "<<kkz<<std::endl;

                            if ((iix % 2 == 0) && (jjy % 2 == 0) &&
                                (kkz % 2 == 0)) {
                                interpOut[(kkz >> 1u) * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          (jjy >> 1u) * (m_uiElementOrder + 1) +
                                          (iix >> 1u)] =
                                    zippedVec[nodeLookUp_CG];
                            }
                        }
                    }

            for (unsigned int k = kb; k < ke; k++)
                for (unsigned int j = jb; j < je; j++)
                    for (unsigned int i = ib; i < ie; i++) {
                        unzippedVec[offset +
                                    (ek * m_uiElementOrder + k + k_offset) *
                                        (ly * lx) +
                                    (ej * m_uiElementOrder + j + j_offset) *
                                        (lx) +
                                    (ei * m_uiElementOrder + i + i_offset)] =
                            interpOut[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i];
                    }

            edgeCount += 1;
        }
    }
}

template <typename T>
void Mesh::OCT_DIR_RIGHT_UP_Unzip(const ot::Block& blk, const T* zippedVec,
                                  T* unzippedVec, T* eleDGVec,
                                  bool* eleDGValid) {
    const unsigned int* blk2diagMap = blk.getBlk2DiagMap();
    unsigned int lookUp;  // first OCT_DIR_LEFT_DOWN element.

    const unsigned int rank       = getMPIRank();
    const unsigned int regLev     = blk.getRegularGridLev();
    const unsigned int blkElem_1D = 1u
                                    << (regLev - blk.getBlockNode().getLevel());
    const unsigned int paddWidth  = blk.get1DPadWidth();
    const unsigned int offset     = blk.getOffset();
    const unsigned int lx         = blk.getAllocationSzX();
    const unsigned int ly         = blk.getAllocationSzY();
    const unsigned int lz         = blk.getAllocationSzZ();

    unsigned int edgeCount        = 0;
    const unsigned int edgeDir    = (OCT_DIR_RIGHT_UP - EDGE_OFFSET);
    lookUp = blk2diagMap[edgeDir * (2 * blkElem_1D) + 2 * edgeCount];
    if (lookUp == LOOK_UP_TABLE_DEFAULT) return;

    const unsigned int ei      = blkElem_1D - 1;
    const unsigned int ej      = blkElem_1D - 1;
    unsigned int ek            = 0;

    const unsigned int dir1    = OCT_DIR_RIGHT;
    const unsigned int dir2    = OCT_DIR_UP;
    const unsigned int dir3    = OCT_DIR_FRONT;
    const unsigned int dir4    = OCT_DIR_UP;
    const ot::TreeNode blkNode = blk.getBlockNode();
    const unsigned int sz      = 1u << (m_uiMaxDepth - regLev);

    const ot::TreeNode* pNodes = &(*(m_uiAllElements.begin()));

    std::vector<T> interpIn;
    interpIn.resize(m_uiNpE);

    std::vector<T> interpOut;
    interpOut.resize(m_uiNpE);

    const int i_offset       = (m_uiElementOrder + paddWidth);
    const int j_offset       = (m_uiElementOrder + paddWidth);
    const int k_offset       = (paddWidth);

    const unsigned int kb    = 0;
    const unsigned int ke    = (m_uiElementOrder + 1);
    const unsigned int jb    = 0;
    const unsigned int je    = paddWidth + 1;
    const unsigned int ib    = 0;
    const unsigned int ie    = paddWidth + 1;

    const unsigned int cnum1 = 0;
    const unsigned int cnum2 = 4;
    unsigned int cnum;

    unsigned int nodeLookUp_CG;
    bool isHanging;

    while (edgeCount < blkElem_1D) {
        ek     = edgeCount;
        lookUp = blk2diagMap[edgeDir * (2 * blkElem_1D) + 2 * edgeCount];
        assert(lookUp != LOOK_UP_TABLE_DEFAULT);
        if (pNodes[lookUp].getLevel() == regLev) {
            T* lookUpVec = &eleDGVec[lookUp * m_uiNpE];
            if (!eleDGValid[lookUp]) {
                getElementNodalValues(zippedVec, lookUpVec, lookUp);
                eleDGValid[lookUp] = true;
            }
            for (unsigned int k = kb; k < ke; k++)
                for (unsigned int j = jb; j < je; j++)
                    for (unsigned int i = ib; i < ie; i++) {
                        unzippedVec[offset +
                                    (ek * m_uiElementOrder + k + k_offset) *
                                        (ly * lx) +
                                    (ej * m_uiElementOrder + j + j_offset) *
                                        (lx) +
                                    (ei * m_uiElementOrder + i + i_offset)] =
                            lookUpVec[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i];
                    }

            edgeCount += 1;

        } else if (pNodes[lookUp].getLevel() < regLev) {
            if ((pNodes[lookUp].getLevel() + 1) != regLev)
                std::cout << "RIGHT_UP_DIAG_UNIZIP ERROR: 2:1 balance error "
                          << std::endl;

            T* lookUpVec = &eleDGVec[lookUp * m_uiNpE];
            if (!eleDGValid[lookUp]) {
                getElementNodalValues(zippedVec, lookUpVec, lookUp);
                eleDGValid[lookUp] = true;
            }
            cnum = ot::TreeNode(blkNode.maxX(), blkNode.maxY(),
                                blkNode.minZ() + ek * sz, regLev, m_uiDim,
                                m_uiMaxDepth)
                       .getMortonIndex();
            parent2ChildInterpolation(lookUpVec, &(*(interpOut.begin())), cnum,
                                      3);
            for (unsigned int k = kb; k < ke; k++)
                for (unsigned int j = jb; j < je; j++)
                    for (unsigned int i = ib; i < ie; i++) {
                        unzippedVec[offset +
                                    (ek * m_uiElementOrder + k + k_offset) *
                                        (ly * lx) +
                                    (ej * m_uiElementOrder + j + j_offset) *
                                        (lx) +
                                    (ei * m_uiElementOrder + i + i_offset)] =
                            interpOut[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i];
                    }

            ek = edgeCount + 1;
            if (ek < blkElem_1D) {
                cnum = ot::TreeNode(blkNode.maxX(), blkNode.maxY(),
                                    blkNode.minZ() + ek * sz, regLev, m_uiDim,
                                    m_uiMaxDepth)
                           .getMortonIndex();
                parent2ChildInterpolation(lookUpVec, &(*(interpOut.begin())),
                                          cnum, 3);
                for (unsigned int k = kb; k < ke; k++)
                    for (unsigned int j = jb; j < je; j++)
                        for (unsigned int i = ib; i < ie; i++) {
                            unzippedVec[offset +
                                        (ek * m_uiElementOrder + k + k_offset) *
                                            (ly * lx) +
                                        (ej * m_uiElementOrder + j + j_offset) *
                                            (lx) +
                                        (ei * m_uiElementOrder + i +
                                         i_offset)] =
                                interpOut[k * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          j * (m_uiElementOrder + 1) + i];
                        }
            }

            edgeCount += 2;

        } else {
            assert(pNodes[lookUp].getLevel() == (regLev + 1));
            if (pNodes[lookUp].getLevel() != (regLev + 1))
                std::cout << "RIGHT_UP_DIAG_UNIZIP ERROR: 2:1 balance error "
                          << std::endl;

            cnum = pNodes[lookUp].getMortonIndex();
            assert(cnum == cnum1);
            for (unsigned int k = 0; k < m_uiElementOrder + 1; k++)
                for (unsigned int j = 0; j < m_uiElementOrder + 1; j++)
                    for (unsigned int i = 0; i < m_uiElementOrder + 1; i++) {
                        isHanging = isNodeHanging(lookUp, i, j, k);
                        nodeLookUp_CG =
                            m_uiE2NMapping_CG[lookUp * m_uiNpE +
                                              k * (m_uiElementOrder + 1) *
                                                  (m_uiElementOrder + 1) +
                                              j * (m_uiElementOrder + 1) + i];
                        if (isHanging) {
                            interpOut[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i] =
                                zippedVec[nodeLookUp_CG];
                        } else {
                            const unsigned int iix =
                                m_uiElementOrder * (int)(cnum & 1u) + i;
                            const unsigned int jjy =
                                m_uiElementOrder * (int)((cnum & 2u) >> 1u) + j;
                            const unsigned int kkz =
                                m_uiElementOrder * (int)((cnum & 4u) >> 2u) + k;
                            // std::cout<<" iix: "<<iix<<" jjy: "<<jjy<<" kkz:
                            // "<<kkz<<std::endl;

                            if ((iix % 2 == 0) && (jjy % 2 == 0) &&
                                (kkz % 2 == 0)) {
                                interpOut[(kkz >> 1u) * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          (jjy >> 1u) * (m_uiElementOrder + 1) +
                                          (iix >> 1u)] =
                                    zippedVec[nodeLookUp_CG];
                            }
                        }
                    }

            lookUp =
                blk2diagMap[edgeDir * (2 * blkElem_1D) + 2 * edgeCount + 1];
            assert(lookUp != LOOK_UP_TABLE_DEFAULT);
            assert(pNodes[lookUp].getLevel() == (regLev + 1));
            cnum = pNodes[lookUp].getMortonIndex();
            assert(cnum == cnum2);
            for (unsigned int k = 0; k < m_uiElementOrder + 1; k++)
                for (unsigned int j = 0; j < m_uiElementOrder + 1; j++)
                    for (unsigned int i = 0; i < m_uiElementOrder + 1; i++) {
                        isHanging = isNodeHanging(lookUp, i, j, k);
                        nodeLookUp_CG =
                            m_uiE2NMapping_CG[lookUp * m_uiNpE +
                                              k * (m_uiElementOrder + 1) *
                                                  (m_uiElementOrder + 1) +
                                              j * (m_uiElementOrder + 1) + i];
                        if (isHanging) {
                            interpOut[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i] =
                                zippedVec[nodeLookUp_CG];
                        } else {
                            const unsigned int iix =
                                m_uiElementOrder * (int)(cnum & 1u) + i;
                            const unsigned int jjy =
                                m_uiElementOrder * (int)((cnum & 2u) >> 1u) + j;
                            const unsigned int kkz =
                                m_uiElementOrder * (int)((cnum & 4u) >> 2u) + k;
                            // std::cout<<" iix: "<<iix<<" jjy: "<<jjy<<" kkz:
                            // "<<kkz<<std::endl;

                            if ((iix % 2 == 0) && (jjy % 2 == 0) &&
                                (kkz % 2 == 0)) {
                                interpOut[(kkz >> 1u) * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          (jjy >> 1u) * (m_uiElementOrder + 1) +
                                          (iix >> 1u)] =
                                    zippedVec[nodeLookUp_CG];
                            }
                        }
                    }

            for (unsigned int k = kb; k < ke; k++)
                for (unsigned int j = jb; j < je; j++)
                    for (unsigned int i = ib; i < ie; i++) {
                        unzippedVec[offset +
                                    (ek * m_uiElementOrder + k + k_offset) *
                                        (ly * lx) +
                                    (ej * m_uiElementOrder + j + j_offset) *
                                        (lx) +
                                    (ei * m_uiElementOrder + i + i_offset)] =
                            interpOut[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i];
                    }

            edgeCount += 1;
        }
    }
}

template <typename T>
void Mesh::OCT_DIR_RIGHT_BACK_Unzip(const ot::Block& blk, const T* zippedVec,
                                    T* unzippedVec, T* eleDGVec,
                                    bool* eleDGValid) {
    const unsigned int* blk2diagMap = blk.getBlk2DiagMap();
    unsigned int lookUp;  // first OCT_DIR_LEFT_DOWN element.

    const unsigned int rank       = getMPIRank();
    const unsigned int regLev     = blk.getRegularGridLev();
    const unsigned int blkElem_1D = 1u
                                    << (regLev - blk.getBlockNode().getLevel());
    const unsigned int paddWidth  = blk.get1DPadWidth();
    const unsigned int offset     = blk.getOffset();
    const unsigned int lx         = blk.getAllocationSzX();
    const unsigned int ly         = blk.getAllocationSzY();
    const unsigned int lz         = blk.getAllocationSzZ();

    unsigned int edgeCount        = 0;
    const unsigned int edgeDir    = (OCT_DIR_RIGHT_BACK - EDGE_OFFSET);
    lookUp = blk2diagMap[edgeDir * (2 * blkElem_1D) + 2 * edgeCount];
    if (lookUp == LOOK_UP_TABLE_DEFAULT) return;

    const unsigned int ei      = blkElem_1D - 1;
    unsigned int ej            = 0;
    const unsigned int ek      = 0;

    const unsigned int dir1    = OCT_DIR_RIGHT;
    const unsigned int dir2    = OCT_DIR_BACK;
    const unsigned int dir3    = OCT_DIR_UP;
    const unsigned int dir4    = OCT_DIR_FRONT;
    const unsigned int dir5    = OCT_DIR_FRONT;
    const ot::TreeNode blkNode = blk.getBlockNode();
    const unsigned int sz      = 1u << (m_uiMaxDepth - regLev);

    const ot::TreeNode* pNodes = &(*(m_uiAllElements.begin()));

    std::vector<T> interpIn;
    interpIn.resize(m_uiNpE);

    std::vector<T> interpOut;
    interpOut.resize(m_uiNpE);

    const int i_offset       = (m_uiElementOrder + paddWidth);
    const int j_offset       = paddWidth;
    const int k_offset       = -(m_uiElementOrder - paddWidth);

    const unsigned int kb    = (m_uiElementOrder - paddWidth);
    const unsigned int ke    = (m_uiElementOrder + 1);

    const unsigned int jb    = 0;
    const unsigned int je    = (m_uiElementOrder + 1);

    const unsigned int ib    = 0;
    const unsigned int ie    = paddWidth + 1;

    const unsigned int cnum1 = 4;
    const unsigned int cnum2 = 6;
    unsigned int cnum;

    unsigned int nodeLookUp_CG;
    bool isHanging;

    while (edgeCount < blkElem_1D) {
        ej     = edgeCount;
        lookUp = blk2diagMap[edgeDir * (2 * blkElem_1D) + 2 * edgeCount];
        assert(lookUp != LOOK_UP_TABLE_DEFAULT);
        if (pNodes[lookUp].getLevel() == regLev) {
            T* lookUpVec = &eleDGVec[lookUp * m_uiNpE];
            if (!eleDGValid[lookUp]) {
                getElementNodalValues(zippedVec, lookUpVec, lookUp);
                eleDGValid[lookUp] = true;
            }
            for (unsigned int k = kb; k < ke; k++)
                for (unsigned int j = jb; j < je; j++)
                    for (unsigned int i = ib; i < ie; i++) {
                        unzippedVec[offset +
                                    (ek * m_uiElementOrder + k + k_offset) *
                                        (ly * lx) +
                                    (ej * m_uiElementOrder + j + j_offset) *
                                        (lx) +
                                    (ei * m_uiElementOrder + i + i_offset)] =
                            lookUpVec[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i];
                    }

            edgeCount += 1;

        } else if (pNodes[lookUp].getLevel() < regLev) {
            if ((pNodes[lookUp].getLevel() + 1) != regLev)
                std::cout << "LEFT_BACK_DIAG_UNIZIP ERROR: 2:1 balance error "
                          << std::endl;

            T* lookUpVec = &eleDGVec[lookUp * m_uiNpE];
            if (!eleDGValid[lookUp]) {
                getElementNodalValues(zippedVec, lookUpVec, lookUp);
                eleDGValid[lookUp] = true;
            }
            cnum =
                ot::TreeNode(blkNode.maxX(), blkNode.minY() + ej * sz,
                             blkNode.minZ() - sz, regLev, m_uiDim, m_uiMaxDepth)
                    .getMortonIndex();
            parent2ChildInterpolation(lookUpVec, &(*(interpOut.begin())), cnum,
                                      3);
            for (unsigned int k = kb; k < ke; k++)
                for (unsigned int j = jb; j < je; j++)
                    for (unsigned int i = ib; i < ie; i++) {
                        unzippedVec[offset +
                                    (ek * m_uiElementOrder + k + k_offset) *
                                        (ly * lx) +
                                    (ej * m_uiElementOrder + j + j_offset) *
                                        (lx) +
                                    (ei * m_uiElementOrder + i + i_offset)] =
                            interpOut[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i];
                    }

            ej = edgeCount + 1;
            if (ej < blkElem_1D) {
                cnum = ot::TreeNode(blkNode.maxX(), blkNode.minY() + ej * sz,
                                    blkNode.minZ() - sz, regLev, m_uiDim,
                                    m_uiMaxDepth)
                           .getMortonIndex();
                parent2ChildInterpolation(lookUpVec, &(*(interpOut.begin())),
                                          cnum, 3);
                for (unsigned int k = kb; k < ke; k++)
                    for (unsigned int j = jb; j < je; j++)
                        for (unsigned int i = ib; i < ie; i++) {
                            unzippedVec[offset +
                                        (ek * m_uiElementOrder + k + k_offset) *
                                            (ly * lx) +
                                        (ej * m_uiElementOrder + j + j_offset) *
                                            (lx) +
                                        (ei * m_uiElementOrder + i +
                                         i_offset)] =
                                interpOut[k * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          j * (m_uiElementOrder + 1) + i];
                        }
            }

            edgeCount += 2;

        } else {
            assert(pNodes[lookUp].getLevel() == (regLev + 1));
            if (pNodes[lookUp].getLevel() != (regLev + 1))
                std::cout << "LEFT_FRONT_BACK_UNIZIP ERROR: 2:1 balance error "
                          << std::endl;

            cnum = pNodes[lookUp].getMortonIndex();
            assert(cnum == cnum1);
            for (unsigned int k = 0; k < m_uiElementOrder + 1; k++)
                for (unsigned int j = 0; j < m_uiElementOrder + 1; j++)
                    for (unsigned int i = 0; i < m_uiElementOrder + 1; i++) {
                        isHanging = isNodeHanging(lookUp, i, j, k);
                        nodeLookUp_CG =
                            m_uiE2NMapping_CG[lookUp * m_uiNpE +
                                              k * (m_uiElementOrder + 1) *
                                                  (m_uiElementOrder + 1) +
                                              j * (m_uiElementOrder + 1) + i];
                        if (isHanging) {
                            interpOut[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i] =
                                zippedVec[nodeLookUp_CG];
                        } else {
                            const unsigned int iix =
                                m_uiElementOrder * (int)(cnum & 1u) + i;
                            const unsigned int jjy =
                                m_uiElementOrder * (int)((cnum & 2u) >> 1u) + j;
                            const unsigned int kkz =
                                m_uiElementOrder * (int)((cnum & 4u) >> 2u) + k;
                            // std::cout<<" iix: "<<iix<<" jjy: "<<jjy<<" kkz:
                            // "<<kkz<<std::endl;

                            if ((iix % 2 == 0) && (jjy % 2 == 0) &&
                                (kkz % 2 == 0)) {
                                interpOut[(kkz >> 1u) * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          (jjy >> 1u) * (m_uiElementOrder + 1) +
                                          (iix >> 1u)] =
                                    zippedVec[nodeLookUp_CG];
                            }
                        }
                    }

            lookUp =
                blk2diagMap[edgeDir * (2 * blkElem_1D) + 2 * edgeCount + 1];
            assert(lookUp != LOOK_UP_TABLE_DEFAULT);
            assert(pNodes[lookUp].getLevel() == (regLev + 1));
            cnum = pNodes[lookUp].getMortonIndex();
            assert(cnum == cnum2);
            for (unsigned int k = 0; k < m_uiElementOrder + 1; k++)
                for (unsigned int j = 0; j < m_uiElementOrder + 1; j++)
                    for (unsigned int i = 0; i < m_uiElementOrder + 1; i++) {
                        isHanging = isNodeHanging(lookUp, i, j, k);
                        nodeLookUp_CG =
                            m_uiE2NMapping_CG[lookUp * m_uiNpE +
                                              k * (m_uiElementOrder + 1) *
                                                  (m_uiElementOrder + 1) +
                                              j * (m_uiElementOrder + 1) + i];
                        if (isHanging) {
                            interpOut[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i] =
                                zippedVec[nodeLookUp_CG];
                        } else {
                            const unsigned int iix =
                                m_uiElementOrder * (int)(cnum & 1u) + i;
                            const unsigned int jjy =
                                m_uiElementOrder * (int)((cnum & 2u) >> 1u) + j;
                            const unsigned int kkz =
                                m_uiElementOrder * (int)((cnum & 4u) >> 2u) + k;
                            // std::cout<<" iix: "<<iix<<" jjy: "<<jjy<<" kkz:
                            // "<<kkz<<std::endl;

                            if ((iix % 2 == 0) && (jjy % 2 == 0) &&
                                (kkz % 2 == 0)) {
                                interpOut[(kkz >> 1u) * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          (jjy >> 1u) * (m_uiElementOrder + 1) +
                                          (iix >> 1u)] =
                                    zippedVec[nodeLookUp_CG];
                            }
                        }
                    }

            for (unsigned int k = kb; k < ke; k++)
                for (unsigned int j = jb; j < je; j++)
                    for (unsigned int i = ib; i < ie; i++) {
                        unzippedVec[offset +
                                    (ek * m_uiElementOrder + k + k_offset) *
                                        (ly * lx) +
                                    (ej * m_uiElementOrder + j + j_offset) *
                                        (lx) +
                                    (ei * m_uiElementOrder + i + i_offset)] =
                            interpOut[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i];
                    }

            edgeCount += 1;
        }
    }
}

template <typename T>
void Mesh::OCT_DIR_RIGHT_FRONT_Unzip(const ot::Block& blk, const T* zippedVec,
                                     T* unzippedVec, T* eleDGVec,
                                     bool* eleDGValid) {
    const unsigned int* blk2diagMap = blk.getBlk2DiagMap();
    unsigned int lookUp;  // first OCT_DIR_LEFT_DOWN element.

    const unsigned int rank       = getMPIRank();
    const unsigned int regLev     = blk.getRegularGridLev();
    const unsigned int blkElem_1D = 1u
                                    << (regLev - blk.getBlockNode().getLevel());
    const unsigned int paddWidth  = blk.get1DPadWidth();
    const unsigned int offset     = blk.getOffset();
    const unsigned int lx         = blk.getAllocationSzX();
    const unsigned int ly         = blk.getAllocationSzY();
    const unsigned int lz         = blk.getAllocationSzZ();

    unsigned int edgeCount        = 0;
    const unsigned int edgeDir    = (OCT_DIR_RIGHT_FRONT - EDGE_OFFSET);
    lookUp = blk2diagMap[edgeDir * (2 * blkElem_1D) + 2 * edgeCount];
    if (lookUp == LOOK_UP_TABLE_DEFAULT) return;

    const unsigned int ei      = (blkElem_1D - 1);
    unsigned int ej            = 0;
    const unsigned int ek      = (blkElem_1D - 1);

    const unsigned int dir1    = OCT_DIR_RIGHT;
    const unsigned int dir2    = OCT_DIR_FRONT;
    const unsigned int dir3    = OCT_DIR_UP;
    const unsigned int dir4    = OCT_DIR_RIGHT;

    const ot::TreeNode blkNode = blk.getBlockNode();
    const unsigned int sz      = 1u << (m_uiMaxDepth - regLev);

    const ot::TreeNode* pNodes = &(*(m_uiAllElements.begin()));

    std::vector<T> interpIn;
    interpIn.resize(m_uiNpE);

    std::vector<T> interpOut;
    interpOut.resize(m_uiNpE);

    const int i_offset       = (m_uiElementOrder + paddWidth);
    const int j_offset       = paddWidth;
    const int k_offset       = (m_uiElementOrder + paddWidth);

    const unsigned int kb    = 0;
    const unsigned int ke    = (paddWidth + 1);

    const unsigned int jb    = 0;
    const unsigned int je    = (m_uiElementOrder + 1);

    const unsigned int ib    = 0;
    const unsigned int ie    = paddWidth + 1;

    const unsigned int cnum1 = 0;
    const unsigned int cnum2 = 2;
    unsigned int cnum;

    unsigned int nodeLookUp_CG;
    bool isHanging;

    while (edgeCount < blkElem_1D) {
        ej     = edgeCount;
        lookUp = blk2diagMap[edgeDir * (2 * blkElem_1D) + 2 * edgeCount];
        assert(lookUp != LOOK_UP_TABLE_DEFAULT);
        if (pNodes[lookUp].getLevel() == regLev) {
            T* lookUpVec = &eleDGVec[lookUp * m_uiNpE];
            if (!eleDGValid[lookUp]) {
                getElementNodalValues(zippedVec, lookUpVec, lookUp);
                eleDGValid[lookUp] = true;
            }
            for (unsigned int k = kb; k < ke; k++)
                for (unsigned int j = jb; j < je; j++)
                    for (unsigned int i = ib; i < ie; i++) {
                        unzippedVec[offset +
                                    (ek * m_uiElementOrder + k + k_offset) *
                                        (ly * lx) +
                                    (ej * m_uiElementOrder + j + j_offset) *
                                        (lx) +
                                    (ei * m_uiElementOrder + i + i_offset)] =
                            lookUpVec[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i];
                    }

            edgeCount += 1;

        } else if (pNodes[lookUp].getLevel() < regLev) {
            if ((pNodes[lookUp].getLevel() + 1) != regLev)
                std::cout << "LEFT_FRONT_DIAG_UNIZIP ERROR: 2:1 balance error "
                          << std::endl;

            T* lookUpVec = &eleDGVec[lookUp * m_uiNpE];
            if (!eleDGValid[lookUp]) {
                getElementNodalValues(zippedVec, lookUpVec, lookUp);
                eleDGValid[lookUp] = true;
            }
            cnum = ot::TreeNode(blkNode.maxX(), blkNode.minY() + ej * sz,
                                blkNode.maxZ(), regLev, m_uiDim, m_uiMaxDepth)
                       .getMortonIndex();
            parent2ChildInterpolation(lookUpVec, &(*(interpOut.begin())), cnum,
                                      3);
            for (unsigned int k = kb; k < ke; k++)
                for (unsigned int j = jb; j < je; j++)
                    for (unsigned int i = ib; i < ie; i++) {
                        unzippedVec[offset +
                                    (ek * m_uiElementOrder + k + k_offset) *
                                        (ly * lx) +
                                    (ej * m_uiElementOrder + j + j_offset) *
                                        (lx) +
                                    (ei * m_uiElementOrder + i + i_offset)] =
                            interpOut[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i];
                    }

            ej = edgeCount + 1;
            if (ej < blkElem_1D) {
                cnum =
                    ot::TreeNode(blkNode.maxX(), blkNode.minY() + ej * sz,
                                 blkNode.maxZ(), regLev, m_uiDim, m_uiMaxDepth)
                        .getMortonIndex();
                parent2ChildInterpolation(lookUpVec, &(*(interpOut.begin())),
                                          cnum, 3);
                for (unsigned int k = kb; k < ke; k++)
                    for (unsigned int j = jb; j < je; j++)
                        for (unsigned int i = ib; i < ie; i++) {
                            unzippedVec[offset +
                                        (ek * m_uiElementOrder + k + k_offset) *
                                            (ly * lx) +
                                        (ej * m_uiElementOrder + j + j_offset) *
                                            (lx) +
                                        (ei * m_uiElementOrder + i +
                                         i_offset)] =
                                interpOut[k * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          j * (m_uiElementOrder + 1) + i];
                        }
            }

            edgeCount += 2;

        } else {
            assert(pNodes[lookUp].getLevel() == (regLev + 1));
            if (pNodes[lookUp].getLevel() != (regLev + 1))
                std::cout << "LEFT_FRONT_DIAG_UNIZIP ERROR: 2:1 balance error "
                          << std::endl;

            cnum = pNodes[lookUp].getMortonIndex();
            assert(cnum == cnum1);
            for (unsigned int k = 0; k < m_uiElementOrder + 1; k++)
                for (unsigned int j = 0; j < m_uiElementOrder + 1; j++)
                    for (unsigned int i = 0; i < m_uiElementOrder + 1; i++) {
                        isHanging = isNodeHanging(lookUp, i, j, k);
                        nodeLookUp_CG =
                            m_uiE2NMapping_CG[lookUp * m_uiNpE +
                                              k * (m_uiElementOrder + 1) *
                                                  (m_uiElementOrder + 1) +
                                              j * (m_uiElementOrder + 1) + i];
                        if (isHanging) {
                            interpOut[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i] =
                                zippedVec[nodeLookUp_CG];
                        } else {
                            const unsigned int iix =
                                m_uiElementOrder * (int)(cnum & 1u) + i;
                            const unsigned int jjy =
                                m_uiElementOrder * (int)((cnum & 2u) >> 1u) + j;
                            const unsigned int kkz =
                                m_uiElementOrder * (int)((cnum & 4u) >> 2u) + k;
                            // std::cout<<" iix: "<<iix<<" jjy: "<<jjy<<" kkz:
                            // "<<kkz<<std::endl;

                            if ((iix % 2 == 0) && (jjy % 2 == 0) &&
                                (kkz % 2 == 0)) {
                                interpOut[(kkz >> 1u) * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          (jjy >> 1u) * (m_uiElementOrder + 1) +
                                          (iix >> 1u)] =
                                    zippedVec[nodeLookUp_CG];
                            }
                        }
                    }

            lookUp =
                blk2diagMap[edgeDir * (2 * blkElem_1D) + 2 * edgeCount + 1];
            assert(lookUp != LOOK_UP_TABLE_DEFAULT);
            assert(pNodes[lookUp].getLevel() == (regLev + 1));
            cnum = pNodes[lookUp].getMortonIndex();
            assert(cnum == cnum2);
            for (unsigned int k = 0; k < m_uiElementOrder + 1; k++)
                for (unsigned int j = 0; j < m_uiElementOrder + 1; j++)
                    for (unsigned int i = 0; i < m_uiElementOrder + 1; i++) {
                        isHanging = isNodeHanging(lookUp, i, j, k);
                        nodeLookUp_CG =
                            m_uiE2NMapping_CG[lookUp * m_uiNpE +
                                              k * (m_uiElementOrder + 1) *
                                                  (m_uiElementOrder + 1) +
                                              j * (m_uiElementOrder + 1) + i];
                        if (isHanging) {
                            interpOut[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i] =
                                zippedVec[nodeLookUp_CG];
                        } else {
                            const unsigned int iix =
                                m_uiElementOrder * (int)(cnum & 1u) + i;
                            const unsigned int jjy =
                                m_uiElementOrder * (int)((cnum & 2u) >> 1u) + j;
                            const unsigned int kkz =
                                m_uiElementOrder * (int)((cnum & 4u) >> 2u) + k;
                            // std::cout<<" iix: "<<iix<<" jjy: "<<jjy<<" kkz:
                            // "<<kkz<<std::endl;

                            if ((iix % 2 == 0) && (jjy % 2 == 0) &&
                                (kkz % 2 == 0)) {
                                interpOut[(kkz >> 1u) * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          (jjy >> 1u) * (m_uiElementOrder + 1) +
                                          (iix >> 1u)] =
                                    zippedVec[nodeLookUp_CG];
                            }
                        }
                    }

            for (unsigned int k = kb; k < ke; k++)
                for (unsigned int j = jb; j < je; j++)
                    for (unsigned int i = ib; i < ie; i++) {
                        unzippedVec[offset +
                                    (ek * m_uiElementOrder + k + k_offset) *
                                        (ly * lx) +
                                    (ej * m_uiElementOrder + j + j_offset) *
                                        (lx) +
                                    (ei * m_uiElementOrder + i + i_offset)] =
                            interpOut[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i];
                    }

            edgeCount += 1;
        }
    }
}

template <typename T>
void Mesh::OCT_DIR_DOWN_BACK_Unzip(const ot::Block& blk, const T* zippedVec,
                                   T* unzippedVec, T* eleDGVec,
                                   bool* eleDGValid) {
    const unsigned int* blk2diagMap = blk.getBlk2DiagMap();
    unsigned int lookUp;  // first OCT_DIR_LEFT_DOWN element.

    const unsigned int rank       = getMPIRank();
    const unsigned int regLev     = blk.getRegularGridLev();
    const unsigned int blkElem_1D = 1u
                                    << (regLev - blk.getBlockNode().getLevel());
    const unsigned int paddWidth  = blk.get1DPadWidth();
    const unsigned int offset     = blk.getOffset();
    const unsigned int lx         = blk.getAllocationSzX();
    const unsigned int ly         = blk.getAllocationSzY();
    const unsigned int lz         = blk.getAllocationSzZ();

    unsigned int edgeCount        = 0;
    const unsigned int edgeDir    = (OCT_DIR_DOWN_BACK - EDGE_OFFSET);
    lookUp = blk2diagMap[edgeDir * (2 * blkElem_1D) + 2 * edgeCount];
    if (lookUp == LOOK_UP_TABLE_DEFAULT) return;

    unsigned int ei            = 0;
    const unsigned int ej      = 0;
    const unsigned int ek      = 0;

    const unsigned int dir1    = OCT_DIR_DOWN;
    const unsigned int dir2    = OCT_DIR_BACK;
    const unsigned int dir3    = OCT_DIR_RIGHT;
    const unsigned int dir4    = OCT_DIR_LEFT;
    const unsigned int dir5    = OCT_DIR_UP;

    const ot::TreeNode blkNode = blk.getBlockNode();
    const unsigned int sz      = 1u << (m_uiMaxDepth - regLev);

    const ot::TreeNode* pNodes = &(*(m_uiAllElements.begin()));

    std::vector<T> interpIn;
    interpIn.resize(m_uiNpE);

    std::vector<T> interpOut;
    interpOut.resize(m_uiNpE);

    const int i_offset       = paddWidth;
    const int j_offset       = -(m_uiElementOrder - paddWidth);
    const int k_offset       = -(m_uiElementOrder - paddWidth);

    const unsigned int kb    = (m_uiElementOrder - paddWidth);
    const unsigned int ke    = (m_uiElementOrder + 1);

    const unsigned int jb    = (m_uiElementOrder - paddWidth);
    const unsigned int je    = (m_uiElementOrder + 1);

    const unsigned int ib    = 0;
    const unsigned int ie    = (m_uiElementOrder + 1);

    const unsigned int cnum1 = 6;
    const unsigned int cnum2 = 7;
    unsigned int cnum;

    unsigned int nodeLookUp_CG;
    bool isHanging;

    while (edgeCount < blkElem_1D) {
        ei     = edgeCount;
        lookUp = blk2diagMap[edgeDir * (2 * blkElem_1D) + 2 * edgeCount];
        assert(lookUp != LOOK_UP_TABLE_DEFAULT);
        if (pNodes[lookUp].getLevel() == regLev) {
            T* lookUpVec = &eleDGVec[lookUp * m_uiNpE];
            if (!eleDGValid[lookUp]) {
                getElementNodalValues(zippedVec, lookUpVec, lookUp);
                eleDGValid[lookUp] = true;
            }
            for (unsigned int k = kb; k < ke; k++)
                for (unsigned int j = jb; j < je; j++)
                    for (unsigned int i = ib; i < ie; i++) {
                        unzippedVec[offset +
                                    (ek * m_uiElementOrder + k + k_offset) *
                                        (ly * lx) +
                                    (ej * m_uiElementOrder + j + j_offset) *
                                        (lx) +
                                    (ei * m_uiElementOrder + i + i_offset)] =
                            lookUpVec[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i];
                    }

            edgeCount += 1;

        } else if (pNodes[lookUp].getLevel() < regLev) {
            if ((pNodes[lookUp].getLevel() + 1) != regLev)
                std::cout << "DOWN_BACK_DIAG_UNIZIP ERROR: 2:1 balance error "
                          << std::endl;

            T* lookUpVec = &eleDGVec[lookUp * m_uiNpE];
            if (!eleDGValid[lookUp]) {
                getElementNodalValues(zippedVec, lookUpVec, lookUp);
                eleDGValid[lookUp] = true;
            }
            cnum =
                ot::TreeNode(blkNode.minX() + ei * sz, blkNode.minY() - sz,
                             blkNode.minZ() - sz, regLev, m_uiDim, m_uiMaxDepth)
                    .getMortonIndex();
            parent2ChildInterpolation(lookUpVec, &(*(interpOut.begin())), cnum,
                                      3);
            for (unsigned int k = kb; k < ke; k++)
                for (unsigned int j = jb; j < je; j++)
                    for (unsigned int i = ib; i < ie; i++) {
                        unzippedVec[offset +
                                    (ek * m_uiElementOrder + k + k_offset) *
                                        (ly * lx) +
                                    (ej * m_uiElementOrder + j + j_offset) *
                                        (lx) +
                                    (ei * m_uiElementOrder + i + i_offset)] =
                            interpOut[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i];
                    }

            ei = edgeCount + 1;
            if (ei < blkElem_1D) {
                cnum = ot::TreeNode(blkNode.minX() + ei * sz,
                                    blkNode.minY() - sz, blkNode.minZ() - sz,
                                    regLev, m_uiDim, m_uiMaxDepth)
                           .getMortonIndex();
                parent2ChildInterpolation(lookUpVec, &(*(interpOut.begin())),
                                          cnum, 3);
                for (unsigned int k = kb; k < ke; k++)
                    for (unsigned int j = jb; j < je; j++)
                        for (unsigned int i = ib; i < ie; i++) {
                            unzippedVec[offset +
                                        (ek * m_uiElementOrder + k + k_offset) *
                                            (ly * lx) +
                                        (ej * m_uiElementOrder + j + j_offset) *
                                            (lx) +
                                        (ei * m_uiElementOrder + i +
                                         i_offset)] =
                                interpOut[k * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          j * (m_uiElementOrder + 1) + i];
                        }
            }

            edgeCount += 2;

        } else {
            assert(pNodes[lookUp].getLevel() == (regLev + 1));
            if (pNodes[lookUp].getLevel() != (regLev + 1))
                std::cout << "DOWN_BACK_DIAG_UNIZIP ERROR: 2:1 balance error "
                          << std::endl;

            cnum = pNodes[lookUp].getMortonIndex();
            assert(cnum == cnum1);
            for (unsigned int k = 0; k < m_uiElementOrder + 1; k++)
                for (unsigned int j = 0; j < m_uiElementOrder + 1; j++)
                    for (unsigned int i = 0; i < m_uiElementOrder + 1; i++) {
                        isHanging = isNodeHanging(lookUp, i, j, k);
                        nodeLookUp_CG =
                            m_uiE2NMapping_CG[lookUp * m_uiNpE +
                                              k * (m_uiElementOrder + 1) *
                                                  (m_uiElementOrder + 1) +
                                              j * (m_uiElementOrder + 1) + i];
                        if (isHanging) {
                            interpOut[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i] =
                                zippedVec[nodeLookUp_CG];
                        } else {
                            const unsigned int iix =
                                m_uiElementOrder * (int)(cnum & 1u) + i;
                            const unsigned int jjy =
                                m_uiElementOrder * (int)((cnum & 2u) >> 1u) + j;
                            const unsigned int kkz =
                                m_uiElementOrder * (int)((cnum & 4u) >> 2u) + k;
                            // std::cout<<" iix: "<<iix<<" jjy: "<<jjy<<" kkz:
                            // "<<kkz<<std::endl;

                            if ((iix % 2 == 0) && (jjy % 2 == 0) &&
                                (kkz % 2 == 0)) {
                                interpOut[(kkz >> 1u) * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          (jjy >> 1u) * (m_uiElementOrder + 1) +
                                          (iix >> 1u)] =
                                    zippedVec[nodeLookUp_CG];
                            }
                        }
                    }

            lookUp =
                blk2diagMap[edgeDir * (2 * blkElem_1D) + 2 * edgeCount + 1];
            assert(lookUp != LOOK_UP_TABLE_DEFAULT);
            assert(pNodes[lookUp].getLevel() == (regLev + 1));
            cnum = pNodes[lookUp].getMortonIndex();
            assert(cnum == cnum2);
            for (unsigned int k = 0; k < m_uiElementOrder + 1; k++)
                for (unsigned int j = 0; j < m_uiElementOrder + 1; j++)
                    for (unsigned int i = 0; i < m_uiElementOrder + 1; i++) {
                        isHanging = isNodeHanging(lookUp, i, j, k);
                        nodeLookUp_CG =
                            m_uiE2NMapping_CG[lookUp * m_uiNpE +
                                              k * (m_uiElementOrder + 1) *
                                                  (m_uiElementOrder + 1) +
                                              j * (m_uiElementOrder + 1) + i];
                        if (isHanging) {
                            interpOut[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i] =
                                zippedVec[nodeLookUp_CG];
                        } else {
                            const unsigned int iix =
                                m_uiElementOrder * (int)(cnum & 1u) + i;
                            const unsigned int jjy =
                                m_uiElementOrder * (int)((cnum & 2u) >> 1u) + j;
                            const unsigned int kkz =
                                m_uiElementOrder * (int)((cnum & 4u) >> 2u) + k;
                            // std::cout<<" iix: "<<iix<<" jjy: "<<jjy<<" kkz:
                            // "<<kkz<<std::endl;

                            if ((iix % 2 == 0) && (jjy % 2 == 0) &&
                                (kkz % 2 == 0)) {
                                interpOut[(kkz >> 1u) * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          (jjy >> 1u) * (m_uiElementOrder + 1) +
                                          (iix >> 1u)] =
                                    zippedVec[nodeLookUp_CG];
                            }
                        }
                    }

            for (unsigned int k = kb; k < ke; k++)
                for (unsigned int j = jb; j < je; j++)
                    for (unsigned int i = ib; i < ie; i++) {
                        unzippedVec[offset +
                                    (ek * m_uiElementOrder + k + k_offset) *
                                        (ly * lx) +
                                    (ej * m_uiElementOrder + j + j_offset) *
                                        (lx) +
                                    (ei * m_uiElementOrder + i + i_offset)] =
                            interpOut[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i];
                    }

            edgeCount += 1;
        }
    }
}

template <typename T>
void Mesh::OCT_DIR_DOWN_FRONT_Unzip(const ot::Block& blk, const T* zippedVec,
                                    T* unzippedVec, T* eleDGVec,
                                    bool* eleDGValid) {
    const unsigned int* blk2diagMap = blk.getBlk2DiagMap();
    unsigned int lookUp;  // first OCT_DIR_LEFT_DOWN element.

    const unsigned int rank       = getMPIRank();
    const unsigned int regLev     = blk.getRegularGridLev();
    const unsigned int blkElem_1D = 1u
                                    << (regLev - blk.getBlockNode().getLevel());
    const unsigned int paddWidth  = blk.get1DPadWidth();
    const unsigned int offset     = blk.getOffset();
    const unsigned int lx         = blk.getAllocationSzX();
    const unsigned int ly         = blk.getAllocationSzY();
    const unsigned int lz         = blk.getAllocationSzZ();

    unsigned int edgeCount        = 0;
    const unsigned int edgeDir    = (OCT_DIR_DOWN_FRONT - EDGE_OFFSET);
    lookUp = blk2diagMap[edgeDir * (2 * blkElem_1D) + 2 * edgeCount];
    if (lookUp == LOOK_UP_TABLE_DEFAULT) return;

    unsigned int ei            = 0;
    const unsigned int ej      = 0;
    const unsigned int ek      = (blkElem_1D - 1);

    const unsigned int dir1    = OCT_DIR_DOWN;
    const unsigned int dir2    = OCT_DIR_FRONT;
    const unsigned int dir3    = OCT_DIR_RIGHT;
    const unsigned int dir4    = OCT_DIR_UP;

    const ot::TreeNode blkNode = blk.getBlockNode();
    const unsigned int sz      = 1u << (m_uiMaxDepth - regLev);

    const ot::TreeNode* pNodes = &(*(m_uiAllElements.begin()));

    std::vector<T> interpIn;
    interpIn.resize(m_uiNpE);

    std::vector<T> interpOut;
    interpOut.resize(m_uiNpE);

    const int i_offset       = paddWidth;
    const int j_offset       = -(m_uiElementOrder - paddWidth);
    const int k_offset       = (m_uiElementOrder + paddWidth);

    const unsigned int kb    = 0;
    const unsigned int ke    = paddWidth + 1;

    const unsigned int jb    = (m_uiElementOrder - paddWidth);
    const unsigned int je    = (m_uiElementOrder + 1);

    const unsigned int ib    = 0;
    const unsigned int ie    = (m_uiElementOrder + 1);

    const unsigned int cnum1 = 2;
    const unsigned int cnum2 = 3;
    unsigned int cnum;

    unsigned int nodeLookUp_CG;
    bool isHanging;

    while (edgeCount < blkElem_1D) {
        ei     = edgeCount;
        lookUp = blk2diagMap[edgeDir * (2 * blkElem_1D) + 2 * edgeCount];
        assert(lookUp != LOOK_UP_TABLE_DEFAULT);
        if (pNodes[lookUp].getLevel() == regLev) {
            T* lookUpVec = &eleDGVec[lookUp * m_uiNpE];
            if (!eleDGValid[lookUp]) {
                getElementNodalValues(zippedVec, lookUpVec, lookUp);
                eleDGValid[lookUp] = true;
            }
            for (unsigned int k = kb; k < ke; k++)
                for (unsigned int j = jb; j < je; j++)
                    for (unsigned int i = ib; i < ie; i++) {
                        unzippedVec[offset +
                                    (ek * m_uiElementOrder + k + k_offset) *
                                        (ly * lx) +
                                    (ej * m_uiElementOrder + j + j_offset) *
                                        (lx) +
                                    (ei * m_uiElementOrder + i + i_offset)] =
                            lookUpVec[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i];
                    }

            edgeCount += 1;

        } else if (pNodes[lookUp].getLevel() < regLev) {
            if ((pNodes[lookUp].getLevel() + 1) != regLev)
                std::cout << "DOWN_FRONT_DIAG_UNIZIP ERROR: 2:1 balance error "
                          << std::endl;

            T* lookUpVec = &eleDGVec[lookUp * m_uiNpE];
            if (!eleDGValid[lookUp]) {
                getElementNodalValues(zippedVec, lookUpVec, lookUp);
                eleDGValid[lookUp] = true;
            }
            cnum = ot::TreeNode(blkNode.minX() + ei * sz, blkNode.minY() - sz,
                                blkNode.maxZ(), regLev, m_uiDim, m_uiMaxDepth)
                       .getMortonIndex();
            parent2ChildInterpolation(lookUpVec, &(*(interpOut.begin())), cnum,
                                      3);
            for (unsigned int k = kb; k < ke; k++)
                for (unsigned int j = jb; j < je; j++)
                    for (unsigned int i = ib; i < ie; i++) {
                        unzippedVec[offset +
                                    (ek * m_uiElementOrder + k + k_offset) *
                                        (ly * lx) +
                                    (ej * m_uiElementOrder + j + j_offset) *
                                        (lx) +
                                    (ei * m_uiElementOrder + i + i_offset)] =
                            interpOut[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i];
                    }

            ei = edgeCount + 1;
            if (ei < blkElem_1D) {
                cnum =
                    ot::TreeNode(blkNode.minX() + ei * sz, blkNode.minY() - sz,
                                 blkNode.maxZ(), regLev, m_uiDim, m_uiMaxDepth)
                        .getMortonIndex();
                parent2ChildInterpolation(lookUpVec, &(*(interpOut.begin())),
                                          cnum, 3);
                for (unsigned int k = kb; k < ke; k++)
                    for (unsigned int j = jb; j < je; j++)
                        for (unsigned int i = ib; i < ie; i++) {
                            unzippedVec[offset +
                                        (ek * m_uiElementOrder + k + k_offset) *
                                            (ly * lx) +
                                        (ej * m_uiElementOrder + j + j_offset) *
                                            (lx) +
                                        (ei * m_uiElementOrder + i +
                                         i_offset)] =
                                interpOut[k * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          j * (m_uiElementOrder + 1) + i];
                        }
            }

            edgeCount += 2;

        } else {
            assert(pNodes[lookUp].getLevel() == (regLev + 1));
            if (pNodes[lookUp].getLevel() != (regLev + 1))
                std::cout << "DOWN_FRONT_DIAG_UNIZIP ERROR: 2:1 balance error "
                          << std::endl;

            cnum = pNodes[lookUp].getMortonIndex();
            assert(cnum == cnum1);
            for (unsigned int k = 0; k < m_uiElementOrder + 1; k++)
                for (unsigned int j = 0; j < m_uiElementOrder + 1; j++)
                    for (unsigned int i = 0; i < m_uiElementOrder + 1; i++) {
                        isHanging = isNodeHanging(lookUp, i, j, k);
                        nodeLookUp_CG =
                            m_uiE2NMapping_CG[lookUp * m_uiNpE +
                                              k * (m_uiElementOrder + 1) *
                                                  (m_uiElementOrder + 1) +
                                              j * (m_uiElementOrder + 1) + i];
                        if (isHanging) {
                            interpOut[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i] =
                                zippedVec[nodeLookUp_CG];
                        } else {
                            const unsigned int iix =
                                m_uiElementOrder * (int)(cnum & 1u) + i;
                            const unsigned int jjy =
                                m_uiElementOrder * (int)((cnum & 2u) >> 1u) + j;
                            const unsigned int kkz =
                                m_uiElementOrder * (int)((cnum & 4u) >> 2u) + k;
                            // std::cout<<" iix: "<<iix<<" jjy: "<<jjy<<" kkz:
                            // "<<kkz<<std::endl;

                            if ((iix % 2 == 0) && (jjy % 2 == 0) &&
                                (kkz % 2 == 0)) {
                                interpOut[(kkz >> 1u) * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          (jjy >> 1u) * (m_uiElementOrder + 1) +
                                          (iix >> 1u)] =
                                    zippedVec[nodeLookUp_CG];
                            }
                        }
                    }

            lookUp =
                blk2diagMap[edgeDir * (2 * blkElem_1D) + 2 * edgeCount + 1];
            assert(lookUp != LOOK_UP_TABLE_DEFAULT);
            assert(pNodes[lookUp].getLevel() == (regLev + 1));
            cnum = pNodes[lookUp].getMortonIndex();
            assert(cnum == cnum2);
            for (unsigned int k = 0; k < m_uiElementOrder + 1; k++)
                for (unsigned int j = 0; j < m_uiElementOrder + 1; j++)
                    for (unsigned int i = 0; i < m_uiElementOrder + 1; i++) {
                        isHanging = isNodeHanging(lookUp, i, j, k);
                        nodeLookUp_CG =
                            m_uiE2NMapping_CG[lookUp * m_uiNpE +
                                              k * (m_uiElementOrder + 1) *
                                                  (m_uiElementOrder + 1) +
                                              j * (m_uiElementOrder + 1) + i];
                        if (isHanging) {
                            interpOut[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i] =
                                zippedVec[nodeLookUp_CG];
                        } else {
                            const unsigned int iix =
                                m_uiElementOrder * (int)(cnum & 1u) + i;
                            const unsigned int jjy =
                                m_uiElementOrder * (int)((cnum & 2u) >> 1u) + j;
                            const unsigned int kkz =
                                m_uiElementOrder * (int)((cnum & 4u) >> 2u) + k;
                            // std::cout<<" iix: "<<iix<<" jjy: "<<jjy<<" kkz:
                            // "<<kkz<<std::endl;

                            if ((iix % 2 == 0) && (jjy % 2 == 0) &&
                                (kkz % 2 == 0)) {
                                interpOut[(kkz >> 1u) * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          (jjy >> 1u) * (m_uiElementOrder + 1) +
                                          (iix >> 1u)] =
                                    zippedVec[nodeLookUp_CG];
                            }
                        }
                    }

            for (unsigned int k = kb; k < ke; k++)
                for (unsigned int j = jb; j < je; j++)
                    for (unsigned int i = ib; i < ie; i++) {
                        unzippedVec[offset +
                                    (ek * m_uiElementOrder + k + k_offset) *
                                        (ly * lx) +
                                    (ej * m_uiElementOrder + j + j_offset) *
                                        (lx) +
                                    (ei * m_uiElementOrder + i + i_offset)] =
                            interpOut[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i];
                    }

            edgeCount += 1;
        }
    }
}

template <typename T>
void Mesh::OCT_DIR_UP_BACK_Unzip(const ot::Block& blk, const T* zippedVec,
                                 T* unzippedVec, T* eleDGVec,
                                 bool* eleDGValid) {
    const unsigned int* blk2diagMap = blk.getBlk2DiagMap();
    unsigned int lookUp;  // first OCT_DIR_LEFT_DOWN element.

    const unsigned int rank       = getMPIRank();
    const unsigned int regLev     = blk.getRegularGridLev();
    const unsigned int blkElem_1D = 1u
                                    << (regLev - blk.getBlockNode().getLevel());
    const unsigned int paddWidth  = blk.get1DPadWidth();
    const unsigned int offset     = blk.getOffset();
    const unsigned int lx         = blk.getAllocationSzX();
    const unsigned int ly         = blk.getAllocationSzY();
    const unsigned int lz         = blk.getAllocationSzZ();

    unsigned int edgeCount        = 0;
    const unsigned int edgeDir    = (OCT_DIR_UP_BACK - EDGE_OFFSET);
    lookUp = blk2diagMap[edgeDir * (2 * blkElem_1D) + 2 * edgeCount];
    if (lookUp == LOOK_UP_TABLE_DEFAULT) return;

    unsigned int ei            = 0;
    const unsigned int ej      = (blkElem_1D - 1);
    const unsigned int ek      = 0;

    const unsigned int dir1    = OCT_DIR_UP;
    const unsigned int dir2    = OCT_DIR_BACK;
    const unsigned int dir3    = OCT_DIR_RIGHT;
    const unsigned int dir4    = OCT_DIR_LEFT;
    const unsigned int dir5    = OCT_DIR_UP;

    const ot::TreeNode blkNode = blk.getBlockNode();
    const unsigned int sz      = 1u << (m_uiMaxDepth - regLev);

    const ot::TreeNode* pNodes = &(*(m_uiAllElements.begin()));

    std::vector<T> interpIn;
    interpIn.resize(m_uiNpE);

    std::vector<T> interpOut;
    interpOut.resize(m_uiNpE);

    const int i_offset       = paddWidth;
    const int j_offset       = (m_uiElementOrder + paddWidth);
    const int k_offset       = -(m_uiElementOrder - paddWidth);

    const unsigned int kb    = (m_uiElementOrder - paddWidth);
    const unsigned int ke    = (m_uiElementOrder + 1);

    const unsigned int jb    = 0;
    const unsigned int je    = paddWidth + 1;

    const unsigned int ib    = 0;
    const unsigned int ie    = (m_uiElementOrder + 1);

    const unsigned int cnum1 = 4;
    const unsigned int cnum2 = 5;
    unsigned int cnum;

    unsigned int nodeLookUp_CG;
    bool isHanging;

    while (edgeCount < blkElem_1D) {
        ei     = edgeCount;
        lookUp = blk2diagMap[edgeDir * (2 * blkElem_1D) + 2 * edgeCount];
        assert(lookUp != LOOK_UP_TABLE_DEFAULT);
        if (pNodes[lookUp].getLevel() == regLev) {
            T* lookUpVec = &eleDGVec[lookUp * m_uiNpE];
            if (!eleDGValid[lookUp]) {
                getElementNodalValues(zippedVec, lookUpVec, lookUp);
                eleDGValid[lookUp] = true;
            }
            for (unsigned int k = kb; k < ke; k++)
                for (unsigned int j = jb; j < je; j++)
                    for (unsigned int i = ib; i < ie; i++) {
                        unzippedVec[offset +
                                    (ek * m_uiElementOrder + k + k_offset) *
                                        (ly * lx) +
                                    (ej * m_uiElementOrder + j + j_offset) *
                                        (lx) +
                                    (ei * m_uiElementOrder + i + i_offset)] =
                            lookUpVec[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i];
                    }

            edgeCount += 1;

        } else if (pNodes[lookUp].getLevel() < regLev) {
            if ((pNodes[lookUp].getLevel() + 1) != regLev)
                std::cout << "UP_BACK_DIAG_UNIZIP ERROR: 2:1 balance error "
                          << std::endl;

            T* lookUpVec = &eleDGVec[lookUp * m_uiNpE];
            if (!eleDGValid[lookUp]) {
                getElementNodalValues(zippedVec, lookUpVec, lookUp);
                eleDGValid[lookUp] = true;
            }
            cnum =
                ot::TreeNode(blkNode.minX() + ei * sz, blkNode.maxY(),
                             blkNode.minZ() - sz, regLev, m_uiDim, m_uiMaxDepth)
                    .getMortonIndex();
            parent2ChildInterpolation(lookUpVec, &(*(interpOut.begin())), cnum,
                                      3);
            for (unsigned int k = kb; k < ke; k++)
                for (unsigned int j = jb; j < je; j++)
                    for (unsigned int i = ib; i < ie; i++) {
                        unzippedVec[offset +
                                    (ek * m_uiElementOrder + k + k_offset) *
                                        (ly * lx) +
                                    (ej * m_uiElementOrder + j + j_offset) *
                                        (lx) +
                                    (ei * m_uiElementOrder + i + i_offset)] =
                            interpOut[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i];
                    }

            ei = edgeCount + 1;
            if (ei < blkElem_1D) {
                cnum = ot::TreeNode(blkNode.minX() + ei * sz, blkNode.maxY(),
                                    blkNode.minZ() - sz, regLev, m_uiDim,
                                    m_uiMaxDepth)
                           .getMortonIndex();
                parent2ChildInterpolation(lookUpVec, &(*(interpOut.begin())),
                                          cnum, 3);
                for (unsigned int k = kb; k < ke; k++)
                    for (unsigned int j = jb; j < je; j++)
                        for (unsigned int i = ib; i < ie; i++) {
                            unzippedVec[offset +
                                        (ek * m_uiElementOrder + k + k_offset) *
                                            (ly * lx) +
                                        (ej * m_uiElementOrder + j + j_offset) *
                                            (lx) +
                                        (ei * m_uiElementOrder + i +
                                         i_offset)] =
                                interpOut[k * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          j * (m_uiElementOrder + 1) + i];
                        }
            }

            edgeCount += 2;

        } else {
            assert(pNodes[lookUp].getLevel() == (regLev + 1));
            if (pNodes[lookUp].getLevel() != (regLev + 1))
                std::cout << "UP_BACK_DIAG_UNIZIP ERROR: 2:1 balance error "
                          << std::endl;

            cnum = pNodes[lookUp].getMortonIndex();
            assert(cnum == cnum1);
            for (unsigned int k = 0; k < m_uiElementOrder + 1; k++)
                for (unsigned int j = 0; j < m_uiElementOrder + 1; j++)
                    for (unsigned int i = 0; i < m_uiElementOrder + 1; i++) {
                        isHanging = isNodeHanging(lookUp, i, j, k);
                        nodeLookUp_CG =
                            m_uiE2NMapping_CG[lookUp * m_uiNpE +
                                              k * (m_uiElementOrder + 1) *
                                                  (m_uiElementOrder + 1) +
                                              j * (m_uiElementOrder + 1) + i];
                        if (isHanging) {
                            interpOut[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i] =
                                zippedVec[nodeLookUp_CG];
                        } else {
                            const unsigned int iix =
                                m_uiElementOrder * (int)(cnum & 1u) + i;
                            const unsigned int jjy =
                                m_uiElementOrder * (int)((cnum & 2u) >> 1u) + j;
                            const unsigned int kkz =
                                m_uiElementOrder * (int)((cnum & 4u) >> 2u) + k;
                            // std::cout<<" iix: "<<iix<<" jjy: "<<jjy<<" kkz:
                            // "<<kkz<<std::endl;

                            if ((iix % 2 == 0) && (jjy % 2 == 0) &&
                                (kkz % 2 == 0)) {
                                interpOut[(kkz >> 1u) * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          (jjy >> 1u) * (m_uiElementOrder + 1) +
                                          (iix >> 1u)] =
                                    zippedVec[nodeLookUp_CG];
                            }
                        }
                    }

            lookUp =
                blk2diagMap[edgeDir * (2 * blkElem_1D) + 2 * edgeCount + 1];
            assert(lookUp != LOOK_UP_TABLE_DEFAULT);
            assert(pNodes[lookUp].getLevel() == (regLev + 1));
            cnum = pNodes[lookUp].getMortonIndex();
            assert(cnum == cnum2);
            for (unsigned int k = 0; k < m_uiElementOrder + 1; k++)
                for (unsigned int j = 0; j < m_uiElementOrder + 1; j++)
                    for (unsigned int i = 0; i < m_uiElementOrder + 1; i++) {
                        isHanging = isNodeHanging(lookUp, i, j, k);
                        nodeLookUp_CG =
                            m_uiE2NMapping_CG[lookUp * m_uiNpE +
                                              k * (m_uiElementOrder + 1) *
                                                  (m_uiElementOrder + 1) +
                                              j * (m_uiElementOrder + 1) + i];
                        if (isHanging) {
                            interpOut[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i] =
                                zippedVec[nodeLookUp_CG];
                        } else {
                            const unsigned int iix =
                                m_uiElementOrder * (int)(cnum & 1u) + i;
                            const unsigned int jjy =
                                m_uiElementOrder * (int)((cnum & 2u) >> 1u) + j;
                            const unsigned int kkz =
                                m_uiElementOrder * (int)((cnum & 4u) >> 2u) + k;
                            // std::cout<<" iix: "<<iix<<" jjy: "<<jjy<<" kkz:
                            // "<<kkz<<std::endl;

                            if ((iix % 2 == 0) && (jjy % 2 == 0) &&
                                (kkz % 2 == 0)) {
                                interpOut[(kkz >> 1u) * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          (jjy >> 1u) * (m_uiElementOrder + 1) +
                                          (iix >> 1u)] =
                                    zippedVec[nodeLookUp_CG];
                            }
                        }
                    }

            for (unsigned int k = kb; k < ke; k++)
                for (unsigned int j = jb; j < je; j++)
                    for (unsigned int i = ib; i < ie; i++) {
                        unzippedVec[offset +
                                    (ek * m_uiElementOrder + k + k_offset) *
                                        (ly * lx) +
                                    (ej * m_uiElementOrder + j + j_offset) *
                                        (lx) +
                                    (ei * m_uiElementOrder + i + i_offset)] =
                            interpOut[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i];
                    }

            edgeCount += 1;
        }
    }
}

template <typename T>
void Mesh::OCT_DIR_UP_FRONT_Unzip(const ot::Block& blk, const T* zippedVec,
                                  T* unzippedVec, T* eleDGVec,
                                  bool* eleDGValid) {
    const unsigned int* blk2diagMap = blk.getBlk2DiagMap();
    unsigned int lookUp;  // first OCT_DIR_LEFT_DOWN element.

    const unsigned int rank       = getMPIRank();
    const unsigned int regLev     = blk.getRegularGridLev();
    const unsigned int blkElem_1D = 1u
                                    << (regLev - blk.getBlockNode().getLevel());
    const unsigned int paddWidth  = blk.get1DPadWidth();
    const unsigned int offset     = blk.getOffset();
    const unsigned int lx         = blk.getAllocationSzX();
    const unsigned int ly         = blk.getAllocationSzY();
    const unsigned int lz         = blk.getAllocationSzZ();

    unsigned int edgeCount        = 0;
    const unsigned int edgeDir    = (OCT_DIR_UP_FRONT - EDGE_OFFSET);
    lookUp = blk2diagMap[edgeDir * (2 * blkElem_1D) + 2 * edgeCount];
    if (lookUp == LOOK_UP_TABLE_DEFAULT) return;

    unsigned int ei            = 0;
    const unsigned int ej      = (blkElem_1D - 1);
    const unsigned int ek      = (blkElem_1D - 1);

    const unsigned int dir1    = OCT_DIR_UP;
    const unsigned int dir2    = OCT_DIR_FRONT;
    const unsigned int dir3    = OCT_DIR_RIGHT;
    const unsigned int dir4    = OCT_DIR_UP;

    const ot::TreeNode blkNode = blk.getBlockNode();
    const unsigned int sz      = 1u << (m_uiMaxDepth - regLev);

    const ot::TreeNode* pNodes = &(*(m_uiAllElements.begin()));

    std::vector<T> interpIn;
    interpIn.resize(m_uiNpE);

    std::vector<T> interpOut;
    interpOut.resize(m_uiNpE);

    const int i_offset       = paddWidth;
    const int j_offset       = (m_uiElementOrder + paddWidth);
    const int k_offset       = (m_uiElementOrder + paddWidth);

    const unsigned int kb    = 0;
    const unsigned int ke    = paddWidth + 1;

    const unsigned int jb    = 0;
    const unsigned int je    = paddWidth + 1;

    const unsigned int ib    = 0;
    const unsigned int ie    = (m_uiElementOrder + 1);

    const unsigned int cnum1 = 0;
    const unsigned int cnum2 = 1;
    unsigned int cnum;

    unsigned int nodeLookUp_CG;
    bool isHanging;

    while (edgeCount < blkElem_1D) {
        ei     = edgeCount;
        lookUp = blk2diagMap[edgeDir * (2 * blkElem_1D) + 2 * edgeCount];
        assert(lookUp != LOOK_UP_TABLE_DEFAULT);
        if (pNodes[lookUp].getLevel() == regLev) {
            T* lookUpVec = &eleDGVec[lookUp * m_uiNpE];
            if (!eleDGValid[lookUp]) {
                getElementNodalValues(zippedVec, lookUpVec, lookUp);
                eleDGValid[lookUp] = true;
            }
            for (unsigned int k = kb; k < ke; k++)
                for (unsigned int j = jb; j < je; j++)
                    for (unsigned int i = ib; i < ie; i++) {
                        unzippedVec[offset +
                                    (ek * m_uiElementOrder + k + k_offset) *
                                        (ly * lx) +
                                    (ej * m_uiElementOrder + j + j_offset) *
                                        (lx) +
                                    (ei * m_uiElementOrder + i + i_offset)] =
                            lookUpVec[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i];
                    }

            edgeCount += 1;

        } else if (pNodes[lookUp].getLevel() < regLev) {
            if ((pNodes[lookUp].getLevel() + 1) != regLev)
                std::cout << "UP_FRONT_DIAG_UNIZIP ERROR: 2:1 balance error "
                          << std::endl;

            T* lookUpVec = &eleDGVec[lookUp * m_uiNpE];
            if (!eleDGValid[lookUp]) {
                getElementNodalValues(zippedVec, lookUpVec, lookUp);
                eleDGValid[lookUp] = true;
            }
            cnum = ot::TreeNode(blkNode.minX() + ei * sz, blkNode.maxY(),
                                blkNode.maxZ(), regLev, m_uiDim, m_uiMaxDepth)
                       .getMortonIndex();
            parent2ChildInterpolation(lookUpVec, &(*(interpOut.begin())), cnum,
                                      3);
            for (unsigned int k = kb; k < ke; k++)
                for (unsigned int j = jb; j < je; j++)
                    for (unsigned int i = ib; i < ie; i++) {
                        unzippedVec[offset +
                                    (ek * m_uiElementOrder + k + k_offset) *
                                        (ly * lx) +
                                    (ej * m_uiElementOrder + j + j_offset) *
                                        (lx) +
                                    (ei * m_uiElementOrder + i + i_offset)] =
                            interpOut[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i];
                    }

            ei = edgeCount + 1;
            if (ei < blkElem_1D) {
                cnum =
                    ot::TreeNode(blkNode.minX() + ei * sz, blkNode.maxY(),
                                 blkNode.maxZ(), regLev, m_uiDim, m_uiMaxDepth)
                        .getMortonIndex();
                parent2ChildInterpolation(lookUpVec, &(*(interpOut.begin())),
                                          cnum, 3);
                for (unsigned int k = kb; k < ke; k++)
                    for (unsigned int j = jb; j < je; j++)
                        for (unsigned int i = ib; i < ie; i++) {
                            unzippedVec[offset +
                                        (ek * m_uiElementOrder + k + k_offset) *
                                            (ly * lx) +
                                        (ej * m_uiElementOrder + j + j_offset) *
                                            (lx) +
                                        (ei * m_uiElementOrder + i +
                                         i_offset)] =
                                interpOut[k * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          j * (m_uiElementOrder + 1) + i];
                        }
            }

            edgeCount += 2;

        } else {
            assert(pNodes[lookUp].getLevel() == (regLev + 1));
            if (pNodes[lookUp].getLevel() != (regLev + 1))
                std::cout
                    << "DOWN_UP_FRONT_DIAG_UNIZIP ERROR: 2:1 balance error "
                    << std::endl;

            cnum = pNodes[lookUp].getMortonIndex();
            assert(cnum == cnum1);
            for (unsigned int k = 0; k < m_uiElementOrder + 1; k++)
                for (unsigned int j = 0; j < m_uiElementOrder + 1; j++)
                    for (unsigned int i = 0; i < m_uiElementOrder + 1; i++) {
                        isHanging = isNodeHanging(lookUp, i, j, k);
                        nodeLookUp_CG =
                            m_uiE2NMapping_CG[lookUp * m_uiNpE +
                                              k * (m_uiElementOrder + 1) *
                                                  (m_uiElementOrder + 1) +
                                              j * (m_uiElementOrder + 1) + i];
                        if (isHanging) {
                            interpOut[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i] =
                                zippedVec[nodeLookUp_CG];
                        } else {
                            const unsigned int iix =
                                m_uiElementOrder * (int)(cnum & 1u) + i;
                            const unsigned int jjy =
                                m_uiElementOrder * (int)((cnum & 2u) >> 1u) + j;
                            const unsigned int kkz =
                                m_uiElementOrder * (int)((cnum & 4u) >> 2u) + k;
                            // std::cout<<" iix: "<<iix<<" jjy: "<<jjy<<" kkz:
                            // "<<kkz<<std::endl;

                            if ((iix % 2 == 0) && (jjy % 2 == 0) &&
                                (kkz % 2 == 0)) {
                                interpOut[(kkz >> 1u) * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          (jjy >> 1u) * (m_uiElementOrder + 1) +
                                          (iix >> 1u)] =
                                    zippedVec[nodeLookUp_CG];
                            }
                        }
                    }

            lookUp =
                blk2diagMap[edgeDir * (2 * blkElem_1D) + 2 * edgeCount + 1];
            assert(lookUp != LOOK_UP_TABLE_DEFAULT);
            assert(pNodes[lookUp].getLevel() == (regLev + 1));
            cnum = pNodes[lookUp].getMortonIndex();
            assert(cnum == cnum2);
            for (unsigned int k = 0; k < m_uiElementOrder + 1; k++)
                for (unsigned int j = 0; j < m_uiElementOrder + 1; j++)
                    for (unsigned int i = 0; i < m_uiElementOrder + 1; i++) {
                        isHanging = isNodeHanging(lookUp, i, j, k);
                        nodeLookUp_CG =
                            m_uiE2NMapping_CG[lookUp * m_uiNpE +
                                              k * (m_uiElementOrder + 1) *
                                                  (m_uiElementOrder + 1) +
                                              j * (m_uiElementOrder + 1) + i];
                        if (isHanging) {
                            interpOut[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i] =
                                zippedVec[nodeLookUp_CG];
                        } else {
                            const unsigned int iix =
                                m_uiElementOrder * (int)(cnum & 1u) + i;
                            const unsigned int jjy =
                                m_uiElementOrder * (int)((cnum & 2u) >> 1u) + j;
                            const unsigned int kkz =
                                m_uiElementOrder * (int)((cnum & 4u) >> 2u) + k;
                            // std::cout<<" iix: "<<iix<<" jjy: "<<jjy<<" kkz:
                            // "<<kkz<<std::endl;

                            if ((iix % 2 == 0) && (jjy % 2 == 0) &&
                                (kkz % 2 == 0)) {
                                interpOut[(kkz >> 1u) * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          (jjy >> 1u) * (m_uiElementOrder + 1) +
                                          (iix >> 1u)] =
                                    zippedVec[nodeLookUp_CG];
                            }
                        }
                    }

            for (unsigned int k = kb; k < ke; k++)
                for (unsigned int j = jb; j < je; j++)
                    for (unsigned int i = ib; i < ie; i++) {
                        unzippedVec[offset +
                                    (ek * m_uiElementOrder + k + k_offset) *
                                        (ly * lx) +
                                    (ej * m_uiElementOrder + j + j_offset) *
                                        (lx) +
                                    (ei * m_uiElementOrder + i + i_offset)] =
                            interpOut[k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i];
                    }

            edgeCount += 1;
        }
    }
}

template <typename T>
void Mesh::OCT_DIR_LEFT_DOWN_BACK_Unzip(const ot::Block& blk,
                                        const T* zippedVec, T* unzippedVec,
                                        T* eleDGVec, bool* eleDGValid) {
    const unsigned int rank           = getMPIRank();
    const unsigned int dir            = OCT_DIR_LEFT_DOWN_BACK;
    const unsigned int* blk2VertexMap = blk.getBlk2VertexMap();
    const unsigned int lookUp         = blk2VertexMap[dir - VERTEX_OFFSET];

    if (lookUp == LOOK_UP_TABLE_DEFAULT) return;

    const unsigned int regLev    = blk.getRegularGridLev();
    const ot::TreeNode* pNodes   = &(*(m_uiAllElements.begin()));
    const unsigned int offset    = blk.getOffset();

    const unsigned int paddWidth = blk.get1DPadWidth();
    const unsigned int sz        = 1u << (m_uiMaxDepth - regLev);
    const ot::TreeNode blkNode   = blk.getBlockNode();

    const unsigned int ei        = 0;
    const unsigned int ej        = 0;
    const unsigned int ek        = 0;

    const unsigned int lx        = blk.getAllocationSzX();
    const unsigned int ly        = blk.getAllocationSzY();
    const unsigned int lz        = blk.getAllocationSzZ();

    const int i_offset           = -(m_uiElementOrder - paddWidth);
    const int j_offset           = -(m_uiElementOrder - paddWidth);
    const int k_offset           = -(m_uiElementOrder - paddWidth);

    const unsigned int kb        = (m_uiElementOrder - paddWidth);
    const unsigned int ke        = (m_uiElementOrder + 1);

    const unsigned int jb        = (m_uiElementOrder - paddWidth);
    const unsigned int je        = (m_uiElementOrder + 1);

    const unsigned int ib        = (m_uiElementOrder - paddWidth);
    const unsigned int ie        = (m_uiElementOrder + 1);

    unsigned int cnum;
    bool isHanging;
    unsigned int nodeLookUp_CG;

    std::vector<T> interpIn;
    interpIn.resize(m_uiNpE);

    std::vector<T> interpOut;
    interpOut.resize(m_uiNpE);

    if (pNodes[lookUp].getLevel() == regLev) {
        T* lookUpVec = &eleDGVec[lookUp * m_uiNpE];
        if (!eleDGValid[lookUp]) {
            getElementNodalValues(zippedVec, lookUpVec, lookUp);
            eleDGValid[lookUp] = true;
        }
        for (unsigned int k = kb; k < ke; k++)
            for (unsigned int j = jb; j < je; j++)
                for (unsigned int i = ib; i < ie; i++) {
                    unzippedVec[offset +
                                (ek * m_uiElementOrder + k + k_offset) *
                                    (ly * lx) +
                                (ej * m_uiElementOrder + j + j_offset) * (lx) +
                                (ei * m_uiElementOrder + i + i_offset)] =
                        lookUpVec[k * (m_uiElementOrder + 1) *
                                      (m_uiElementOrder + 1) +
                                  j * (m_uiElementOrder + 1) + i];
                }

    } else if (pNodes[lookUp].getLevel() < regLev) {
        if (pNodes[lookUp].getLevel() != (regLev - 1)) {
            std::cout
                << "rank: " << rank
                << " [LEFT_DOWN_BACK Unzip]: 2:1 balance violation blk node: "
                << blkNode << " lookup : " << pNodes[lookUp] << std::endl;
            exit(0);
        }
        assert(pNodes[lookUp].getLevel() == (regLev - 1));
        T* lookUpVec = &eleDGVec[lookUp * m_uiNpE];
        if (!eleDGValid[lookUp]) {
            getElementNodalValues(zippedVec, lookUpVec, lookUp);
            eleDGValid[lookUp] = true;
        }
        cnum = ot::TreeNode(blkNode.minX() - sz, blkNode.minY() - sz,
                            blkNode.minZ() - sz, regLev, m_uiDim, m_uiMaxDepth)
                   .getMortonIndex();

        parent2ChildInterpolation(lookUpVec, &(*(interpOut.begin())), cnum, 3);
        for (unsigned int k = kb; k < ke; k++)
            for (unsigned int j = jb; j < je; j++)
                for (unsigned int i = ib; i < ie; i++) {
                    unzippedVec[offset +
                                (ek * m_uiElementOrder + k + k_offset) *
                                    (ly * lx) +
                                (ej * m_uiElementOrder + j + j_offset) * (lx) +
                                (ei * m_uiElementOrder + i + i_offset)] =
                        interpOut[k * (m_uiElementOrder + 1) *
                                      (m_uiElementOrder + 1) +
                                  j * (m_uiElementOrder + 1) + i];
                }

    } else {
        if ((pNodes[lookUp].getLevel()) != regLev + 1) {
            std::cout
                << "rank: " << rank
                << " [LEFT_DOWN_BACK Unzip]: 2:1 balance violation blk node: "
                << blkNode << " lookup : " << pNodes[lookUp] << std::endl;
            exit(0);
        }
        assert((pNodes[lookUp].getLevel()) == regLev + 1);

        cnum = pNodes[lookUp].getMortonIndex();
        for (unsigned int k = 0; k < m_uiElementOrder + 1; k++)
            for (unsigned int j = 0; j < m_uiElementOrder + 1; j++)
                for (unsigned int i = 0; i < m_uiElementOrder + 1; i++) {
                    isHanging = isNodeHanging(lookUp, i, j, k);
                    nodeLookUp_CG =
                        m_uiE2NMapping_CG[lookUp * m_uiNpE +
                                          k * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          j * (m_uiElementOrder + 1) + i];
                    if (isHanging) {
                        interpOut[k * (m_uiElementOrder + 1) *
                                      (m_uiElementOrder + 1) +
                                  j * (m_uiElementOrder + 1) + i] =
                            zippedVec[nodeLookUp_CG];
                    } else {
                        const unsigned int iix =
                            m_uiElementOrder * (int)(cnum & 1u) + i;
                        const unsigned int jjy =
                            m_uiElementOrder * (int)((cnum & 2u) >> 1u) + j;
                        const unsigned int kkz =
                            m_uiElementOrder * (int)((cnum & 4u) >> 2u) + k;
                        // std::cout<<" iix: "<<iix<<" jjy: "<<jjy<<" kkz:
                        // "<<kkz<<std::endl;

                        if ((iix % 2 == 0) && (jjy % 2 == 0) &&
                            (kkz % 2 == 0)) {
                            interpOut[(kkz >> 1u) * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      (jjy >> 1u) * (m_uiElementOrder + 1) +
                                      (iix >> 1u)] = zippedVec[nodeLookUp_CG];
                        }
                    }
                }

        for (unsigned int k = kb; k < ke; k++)
            for (unsigned int j = jb; j < je; j++)
                for (unsigned int i = ib; i < ie; i++) {
                    unzippedVec[offset +
                                (ek * m_uiElementOrder + k + k_offset) *
                                    (ly * lx) +
                                (ej * m_uiElementOrder + j + j_offset) * (lx) +
                                (ei * m_uiElementOrder + i + i_offset)] =
                        interpOut[k * (m_uiElementOrder + 1) *
                                      (m_uiElementOrder + 1) +
                                  j * (m_uiElementOrder + 1) + i];
                }
    }
}

template <typename T>
void Mesh::OCT_DIR_RIGHT_DOWN_BACK_Unzip(const ot::Block& blk,
                                         const T* zippedVec, T* unzippedVec,
                                         T* eleDGVec, bool* eleDGValid) {
    const unsigned int rank           = getMPIRank();
    const unsigned int dir            = OCT_DIR_RIGHT_DOWN_BACK;
    const unsigned int* blk2VertexMap = blk.getBlk2VertexMap();
    const unsigned int lookUp         = blk2VertexMap[dir - VERTEX_OFFSET];

    if (lookUp == LOOK_UP_TABLE_DEFAULT) return;

    const unsigned int regLev     = blk.getRegularGridLev();
    const ot::TreeNode* pNodes    = &(*(m_uiAllElements.begin()));
    const unsigned int offset     = blk.getOffset();
    const unsigned int blkElem_1D = 1u
                                    << (regLev - blk.getBlockNode().getLevel());

    const unsigned int paddWidth  = blk.get1DPadWidth();
    const unsigned int sz         = 1u << (m_uiMaxDepth - regLev);
    const ot::TreeNode blkNode    = blk.getBlockNode();

    const unsigned int ei         = blkElem_1D - 1;
    const unsigned int ej         = 0;
    const unsigned int ek         = 0;

    const unsigned int lx         = blk.getAllocationSzX();
    const unsigned int ly         = blk.getAllocationSzY();
    const unsigned int lz         = blk.getAllocationSzZ();

    const int i_offset            = (m_uiElementOrder + paddWidth);
    const int j_offset            = -(m_uiElementOrder - paddWidth);
    const int k_offset            = -(m_uiElementOrder - paddWidth);

    const unsigned int kb         = (m_uiElementOrder - paddWidth);
    const unsigned int ke         = (m_uiElementOrder + 1);

    const unsigned int jb         = (m_uiElementOrder - paddWidth);
    const unsigned int je         = (m_uiElementOrder + 1);

    const unsigned int ib         = 0;
    const unsigned int ie         = paddWidth + 1;

    unsigned int cnum;
    bool isHanging;
    unsigned int nodeLookUp_CG;

    std::vector<T> interpIn;
    interpIn.resize(m_uiNpE);

    std::vector<T> interpOut;
    interpOut.resize(m_uiNpE);

    if (pNodes[lookUp].getLevel() == regLev) {
        T* lookUpVec = &eleDGVec[lookUp * m_uiNpE];
        if (!eleDGValid[lookUp]) {
            getElementNodalValues(zippedVec, lookUpVec, lookUp);
            eleDGValid[lookUp] = true;
        }
        for (unsigned int k = kb; k < ke; k++)
            for (unsigned int j = jb; j < je; j++)
                for (unsigned int i = ib; i < ie; i++) {
                    unzippedVec[offset +
                                (ek * m_uiElementOrder + k + k_offset) *
                                    (ly * lx) +
                                (ej * m_uiElementOrder + j + j_offset) * (lx) +
                                (ei * m_uiElementOrder + i + i_offset)] =
                        lookUpVec[k * (m_uiElementOrder + 1) *
                                      (m_uiElementOrder + 1) +
                                  j * (m_uiElementOrder + 1) + i];
                }

    } else if (pNodes[lookUp].getLevel() < regLev) {
        if (pNodes[lookUp].getLevel() != (regLev - 1)) {
            std::cout
                << "rank: " << rank
                << " [RIGHT_DOWN_BACK Unzip]: 2:1 balance violation blk node: "
                << blkNode << " lookup : " << pNodes[lookUp] << std::endl;
            exit(0);
        }
        assert(pNodes[lookUp].getLevel() == (regLev - 1));
        T* lookUpVec = &eleDGVec[lookUp * m_uiNpE];
        if (!eleDGValid[lookUp]) {
            getElementNodalValues(zippedVec, lookUpVec, lookUp);
            eleDGValid[lookUp] = true;
        }
        cnum = ot::TreeNode(blkNode.maxX(), blkNode.minY() - sz,
                            blkNode.minZ() - sz, regLev, m_uiDim, m_uiMaxDepth)
                   .getMortonIndex();

        parent2ChildInterpolation(lookUpVec, &(*(interpOut.begin())), cnum, 3);
        for (unsigned int k = kb; k < ke; k++)
            for (unsigned int j = jb; j < je; j++)
                for (unsigned int i = ib; i < ie; i++) {
                    unzippedVec[offset +
                                (ek * m_uiElementOrder + k + k_offset) *
                                    (ly * lx) +
                                (ej * m_uiElementOrder + j + j_offset) * (lx) +
                                (ei * m_uiElementOrder + i + i_offset)] =
                        interpOut[k * (m_uiElementOrder + 1) *
                                      (m_uiElementOrder + 1) +
                                  j * (m_uiElementOrder + 1) + i];
                }

    } else {
        if ((pNodes[lookUp].getLevel()) != regLev + 1) {
            std::cout
                << "rank: " << rank
                << " [RIGHT_DOWN_BACK Unzip]: 2:1 balance violation blk node: "
                << blkNode << " lookup : " << pNodes[lookUp] << std::endl;
            exit(0);
        }
        assert((pNodes[lookUp].getLevel()) == regLev + 1);

        cnum = pNodes[lookUp].getMortonIndex();
        for (unsigned int k = 0; k < m_uiElementOrder + 1; k++)
            for (unsigned int j = 0; j < m_uiElementOrder + 1; j++)
                for (unsigned int i = 0; i < m_uiElementOrder + 1; i++) {
                    isHanging = isNodeHanging(lookUp, i, j, k);
                    nodeLookUp_CG =
                        m_uiE2NMapping_CG[lookUp * m_uiNpE +
                                          k * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          j * (m_uiElementOrder + 1) + i];
                    if (isHanging) {
                        interpOut[k * (m_uiElementOrder + 1) *
                                      (m_uiElementOrder + 1) +
                                  j * (m_uiElementOrder + 1) + i] =
                            zippedVec[nodeLookUp_CG];
                    } else {
                        const unsigned int iix =
                            m_uiElementOrder * (int)(cnum & 1u) + i;
                        const unsigned int jjy =
                            m_uiElementOrder * (int)((cnum & 2u) >> 1u) + j;
                        const unsigned int kkz =
                            m_uiElementOrder * (int)((cnum & 4u) >> 2u) + k;
                        // std::cout<<" iix: "<<iix<<" jjy: "<<jjy<<" kkz:
                        // "<<kkz<<std::endl;

                        if ((iix % 2 == 0) && (jjy % 2 == 0) &&
                            (kkz % 2 == 0)) {
                            interpOut[(kkz >> 1u) * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      (jjy >> 1u) * (m_uiElementOrder + 1) +
                                      (iix >> 1u)] = zippedVec[nodeLookUp_CG];
                        }
                    }
                }

        for (unsigned int k = kb; k < ke; k++)
            for (unsigned int j = jb; j < je; j++)
                for (unsigned int i = ib; i < ie; i++) {
                    unzippedVec[offset +
                                (ek * m_uiElementOrder + k + k_offset) *
                                    (ly * lx) +
                                (ej * m_uiElementOrder + j + j_offset) * (lx) +
                                (ei * m_uiElementOrder + i + i_offset)] =
                        interpOut[k * (m_uiElementOrder + 1) *
                                      (m_uiElementOrder + 1) +
                                  j * (m_uiElementOrder + 1) + i];
                }
    }
}

template <typename T>
void Mesh::OCT_DIR_LEFT_UP_BACK_Unzip(const ot::Block& blk, const T* zippedVec,
                                      T* unzippedVec, T* eleDGVec,
                                      bool* eleDGValid) {
    const unsigned int rank           = getMPIRank();
    const unsigned int dir            = OCT_DIR_LEFT_UP_BACK;
    const unsigned int* blk2VertexMap = blk.getBlk2VertexMap();
    const unsigned int lookUp         = blk2VertexMap[dir - VERTEX_OFFSET];

    if (lookUp == LOOK_UP_TABLE_DEFAULT) return;

    const unsigned int regLev     = blk.getRegularGridLev();
    const ot::TreeNode* pNodes    = &(*(m_uiAllElements.begin()));
    const unsigned int offset     = blk.getOffset();
    const unsigned int blkElem_1D = 1u
                                    << (regLev - blk.getBlockNode().getLevel());

    const unsigned int paddWidth  = blk.get1DPadWidth();
    const unsigned int sz         = 1u << (m_uiMaxDepth - regLev);
    const ot::TreeNode blkNode    = blk.getBlockNode();

    const unsigned int ei         = 0;
    const unsigned int ej         = blkElem_1D - 1;
    const unsigned int ek         = 0;

    const unsigned int lx         = blk.getAllocationSzX();
    const unsigned int ly         = blk.getAllocationSzY();
    const unsigned int lz         = blk.getAllocationSzZ();

    const int i_offset            = -(m_uiElementOrder - paddWidth);
    const int j_offset            = (m_uiElementOrder + paddWidth);
    const int k_offset            = -(m_uiElementOrder - paddWidth);

    const unsigned int kb         = (m_uiElementOrder - paddWidth);
    const unsigned int ke         = (m_uiElementOrder + 1);

    const unsigned int jb         = 0;
    const unsigned int je         = paddWidth + 1;

    const unsigned int ib         = (m_uiElementOrder - paddWidth);
    const unsigned int ie         = (m_uiElementOrder + 1);

    unsigned int cnum;
    bool isHanging;
    unsigned int nodeLookUp_CG;

    std::vector<T> interpIn;
    interpIn.resize(m_uiNpE);

    std::vector<T> interpOut;
    interpOut.resize(m_uiNpE);

    if (pNodes[lookUp].getLevel() == regLev) {
        T* lookUpVec = &eleDGVec[lookUp * m_uiNpE];
        if (!eleDGValid[lookUp]) {
            getElementNodalValues(zippedVec, lookUpVec, lookUp);
            eleDGValid[lookUp] = true;
        }
        for (unsigned int k = kb; k < ke; k++)
            for (unsigned int j = jb; j < je; j++)
                for (unsigned int i = ib; i < ie; i++) {
                    unzippedVec[offset +
                                (ek * m_uiElementOrder + k + k_offset) *
                                    (ly * lx) +
                                (ej * m_uiElementOrder + j + j_offset) * (lx) +
                                (ei * m_uiElementOrder + i + i_offset)] =
                        lookUpVec[k * (m_uiElementOrder + 1) *
                                      (m_uiElementOrder + 1) +
                                  j * (m_uiElementOrder + 1) + i];
                }

    } else if (pNodes[lookUp].getLevel() < regLev) {
        if (pNodes[lookUp].getLevel() != (regLev - 1)) {
            std::cout
                << "rank: " << rank
                << " [LEFT_UP_BACK Unzip]: 2:1 balance violation blk node: "
                << blkNode << " lookup : " << pNodes[lookUp] << std::endl;
            exit(0);
        }
        assert(pNodes[lookUp].getLevel() == (regLev - 1));
        T* lookUpVec = &eleDGVec[lookUp * m_uiNpE];
        if (!eleDGValid[lookUp]) {
            getElementNodalValues(zippedVec, lookUpVec, lookUp);
            eleDGValid[lookUp] = true;
        }
        cnum = ot::TreeNode(blkNode.minX() - sz, blkNode.maxY(),
                            blkNode.minZ() - sz, regLev, m_uiDim, m_uiMaxDepth)
                   .getMortonIndex();

        parent2ChildInterpolation(lookUpVec, &(*(interpOut.begin())), cnum, 3);
        for (unsigned int k = kb; k < ke; k++)
            for (unsigned int j = jb; j < je; j++)
                for (unsigned int i = ib; i < ie; i++) {
                    unzippedVec[offset +
                                (ek * m_uiElementOrder + k + k_offset) *
                                    (ly * lx) +
                                (ej * m_uiElementOrder + j + j_offset) * (lx) +
                                (ei * m_uiElementOrder + i + i_offset)] =
                        interpOut[k * (m_uiElementOrder + 1) *
                                      (m_uiElementOrder + 1) +
                                  j * (m_uiElementOrder + 1) + i];
                }

    } else {
        if ((pNodes[lookUp].getLevel()) != regLev + 1) {
            std::cout
                << "rank: " << rank
                << " [LEFT_UP_BACK Unzip]: 2:1 balance violation blk node: "
                << blkNode << " lookup : " << pNodes[lookUp] << std::endl;
            exit(0);
        }
        assert((pNodes[lookUp].getLevel()) == regLev + 1);

        cnum = pNodes[lookUp].getMortonIndex();
        for (unsigned int k = 0; k < m_uiElementOrder + 1; k++)
            for (unsigned int j = 0; j < m_uiElementOrder + 1; j++)
                for (unsigned int i = 0; i < m_uiElementOrder + 1; i++) {
                    isHanging = isNodeHanging(lookUp, i, j, k);
                    nodeLookUp_CG =
                        m_uiE2NMapping_CG[lookUp * m_uiNpE +
                                          k * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          j * (m_uiElementOrder + 1) + i];
                    if (isHanging) {
                        interpOut[k * (m_uiElementOrder + 1) *
                                      (m_uiElementOrder + 1) +
                                  j * (m_uiElementOrder + 1) + i] =
                            zippedVec[nodeLookUp_CG];
                    } else {
                        const unsigned int iix =
                            m_uiElementOrder * (int)(cnum & 1u) + i;
                        const unsigned int jjy =
                            m_uiElementOrder * (int)((cnum & 2u) >> 1u) + j;
                        const unsigned int kkz =
                            m_uiElementOrder * (int)((cnum & 4u) >> 2u) + k;
                        // std::cout<<" iix: "<<iix<<" jjy: "<<jjy<<" kkz:
                        // "<<kkz<<std::endl;

                        if ((iix % 2 == 0) && (jjy % 2 == 0) &&
                            (kkz % 2 == 0)) {
                            interpOut[(kkz >> 1u) * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      (jjy >> 1u) * (m_uiElementOrder + 1) +
                                      (iix >> 1u)] = zippedVec[nodeLookUp_CG];
                        }
                    }
                }

        for (unsigned int k = kb; k < ke; k++)
            for (unsigned int j = jb; j < je; j++)
                for (unsigned int i = ib; i < ie; i++) {
                    unzippedVec[offset +
                                (ek * m_uiElementOrder + k + k_offset) *
                                    (ly * lx) +
                                (ej * m_uiElementOrder + j + j_offset) * (lx) +
                                (ei * m_uiElementOrder + i + i_offset)] =
                        interpOut[k * (m_uiElementOrder + 1) *
                                      (m_uiElementOrder + 1) +
                                  j * (m_uiElementOrder + 1) + i];
                }
    }
}

template <typename T>
void Mesh::OCT_DIR_RIGHT_UP_BACK_Unzip(const ot::Block& blk, const T* zippedVec,
                                       T* unzippedVec, T* eleDGVec,
                                       bool* eleDGValid) {
    const unsigned int rank           = getMPIRank();
    const unsigned int dir            = OCT_DIR_RIGHT_UP_BACK;
    const unsigned int* blk2VertexMap = blk.getBlk2VertexMap();
    const unsigned int lookUp         = blk2VertexMap[dir - VERTEX_OFFSET];

    if (lookUp == LOOK_UP_TABLE_DEFAULT) return;

    const unsigned int regLev     = blk.getRegularGridLev();
    const ot::TreeNode* pNodes    = &(*(m_uiAllElements.begin()));
    const unsigned int offset     = blk.getOffset();
    const unsigned int blkElem_1D = 1u
                                    << (regLev - blk.getBlockNode().getLevel());

    const unsigned int paddWidth  = blk.get1DPadWidth();
    const unsigned int sz         = 1u << (m_uiMaxDepth - regLev);
    const ot::TreeNode blkNode    = blk.getBlockNode();

    const unsigned int ei         = blkElem_1D - 1;
    const unsigned int ej         = blkElem_1D - 1;
    const unsigned int ek         = 0;

    const unsigned int lx         = blk.getAllocationSzX();
    const unsigned int ly         = blk.getAllocationSzY();
    const unsigned int lz         = blk.getAllocationSzZ();

    const int i_offset            = (m_uiElementOrder + paddWidth);
    const int j_offset            = (m_uiElementOrder + paddWidth);
    const int k_offset            = -(m_uiElementOrder - paddWidth);

    const unsigned int kb         = (m_uiElementOrder - paddWidth);
    const unsigned int ke         = (m_uiElementOrder + 1);

    const unsigned int jb         = 0;
    const unsigned int je         = paddWidth + 1;

    const unsigned int ib         = 0;
    const unsigned int ie         = paddWidth + 1;

    unsigned int cnum;
    bool isHanging;
    unsigned int nodeLookUp_CG;

    std::vector<T> interpIn;
    interpIn.resize(m_uiNpE);

    std::vector<T> interpOut;
    interpOut.resize(m_uiNpE);

    if (pNodes[lookUp].getLevel() == regLev) {
        T* lookUpVec = &eleDGVec[lookUp * m_uiNpE];
        if (!eleDGValid[lookUp]) {
            getElementNodalValues(zippedVec, lookUpVec, lookUp);
            eleDGValid[lookUp] = true;
        }
        for (unsigned int k = kb; k < ke; k++)
            for (unsigned int j = jb; j < je; j++)
                for (unsigned int i = ib; i < ie; i++) {
                    unzippedVec[offset +
                                (ek * m_uiElementOrder + k + k_offset) *
                                    (ly * lx) +
                                (ej * m_uiElementOrder + j + j_offset) * (lx) +
                                (ei * m_uiElementOrder + i + i_offset)] =
                        lookUpVec[k * (m_uiElementOrder + 1) *
                                      (m_uiElementOrder + 1) +
                                  j * (m_uiElementOrder + 1) + i];
                }

    } else if (pNodes[lookUp].getLevel() < regLev) {
        if (pNodes[lookUp].getLevel() != (regLev - 1)) {
            std::cout
                << "rank: " << rank
                << " [RIGHT_UP_BACK Unzip]: 2:1 balance violation blk node: "
                << blkNode << " lookup : " << pNodes[lookUp] << std::endl;
            exit(0);
        }
        assert(pNodes[lookUp].getLevel() == (regLev - 1));
        T* lookUpVec = &eleDGVec[lookUp * m_uiNpE];
        if (!eleDGValid[lookUp]) {
            getElementNodalValues(zippedVec, lookUpVec, lookUp);
            eleDGValid[lookUp] = true;
        }
        cnum = ot::TreeNode(blkNode.maxX(), blkNode.maxY(), blkNode.minZ() - sz,
                            regLev, m_uiDim, m_uiMaxDepth)
                   .getMortonIndex();

        parent2ChildInterpolation(lookUpVec, &(*(interpOut.begin())), cnum, 3);
        for (unsigned int k = kb; k < ke; k++)
            for (unsigned int j = jb; j < je; j++)
                for (unsigned int i = ib; i < ie; i++) {
                    unzippedVec[offset +
                                (ek * m_uiElementOrder + k + k_offset) *
                                    (ly * lx) +
                                (ej * m_uiElementOrder + j + j_offset) * (lx) +
                                (ei * m_uiElementOrder + i + i_offset)] =
                        interpOut[k * (m_uiElementOrder + 1) *
                                      (m_uiElementOrder + 1) +
                                  j * (m_uiElementOrder + 1) + i];
                }

    } else {
        if ((pNodes[lookUp].getLevel()) != regLev + 1) {
            std::cout
                << "rank: " << rank
                << " [RIGHT_UP_BACK Unzip]: 2:1 balance violation blk node: "
                << blkNode << " lookup : " << pNodes[lookUp] << std::endl;
            exit(0);
        }
        assert((pNodes[lookUp].getLevel()) == regLev + 1);

        cnum = pNodes[lookUp].getMortonIndex();
        for (unsigned int k = 0; k < m_uiElementOrder + 1; k++)
            for (unsigned int j = 0; j < m_uiElementOrder + 1; j++)
                for (unsigned int i = 0; i < m_uiElementOrder + 1; i++) {
                    isHanging = isNodeHanging(lookUp, i, j, k);
                    nodeLookUp_CG =
                        m_uiE2NMapping_CG[lookUp * m_uiNpE +
                                          k * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          j * (m_uiElementOrder + 1) + i];
                    if (isHanging) {
                        interpOut[k * (m_uiElementOrder + 1) *
                                      (m_uiElementOrder + 1) +
                                  j * (m_uiElementOrder + 1) + i] =
                            zippedVec[nodeLookUp_CG];
                    } else {
                        const unsigned int iix =
                            m_uiElementOrder * (int)(cnum & 1u) + i;
                        const unsigned int jjy =
                            m_uiElementOrder * (int)((cnum & 2u) >> 1u) + j;
                        const unsigned int kkz =
                            m_uiElementOrder * (int)((cnum & 4u) >> 2u) + k;
                        // std::cout<<" iix: "<<iix<<" jjy: "<<jjy<<" kkz:
                        // "<<kkz<<std::endl;

                        if ((iix % 2 == 0) && (jjy % 2 == 0) &&
                            (kkz % 2 == 0)) {
                            interpOut[(kkz >> 1u) * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      (jjy >> 1u) * (m_uiElementOrder + 1) +
                                      (iix >> 1u)] = zippedVec[nodeLookUp_CG];
                        }
                    }
                }

        for (unsigned int k = kb; k < ke; k++)
            for (unsigned int j = jb; j < je; j++)
                for (unsigned int i = ib; i < ie; i++) {
                    unzippedVec[offset +
                                (ek * m_uiElementOrder + k + k_offset) *
                                    (ly * lx) +
                                (ej * m_uiElementOrder + j + j_offset) * (lx) +
                                (ei * m_uiElementOrder + i + i_offset)] =
                        interpOut[k * (m_uiElementOrder + 1) *
                                      (m_uiElementOrder + 1) +
                                  j * (m_uiElementOrder + 1) + i];
                }
    }
}

template <typename T>
void Mesh::OCT_DIR_LEFT_DOWN_FRONT_Unzip(const ot::Block& blk,
                                         const T* zippedVec, T* unzippedVec,
                                         T* eleDGVec, bool* eleDGValid) {
    const unsigned int rank           = getMPIRank();
    const unsigned int dir            = OCT_DIR_LEFT_DOWN_FRONT;
    const unsigned int* blk2VertexMap = blk.getBlk2VertexMap();
    const unsigned int lookUp         = blk2VertexMap[dir - VERTEX_OFFSET];

    if (lookUp == LOOK_UP_TABLE_DEFAULT) return;

    const unsigned int regLev     = blk.getRegularGridLev();
    const ot::TreeNode* pNodes    = &(*(m_uiAllElements.begin()));
    const unsigned int offset     = blk.getOffset();

    const unsigned int paddWidth  = blk.get1DPadWidth();
    const unsigned int sz         = 1u << (m_uiMaxDepth - regLev);
    const ot::TreeNode blkNode    = blk.getBlockNode();

    const unsigned int blkElem_1D = 1u
                                    << (regLev - blk.getBlockNode().getLevel());

    const unsigned int ei         = 0;
    const unsigned int ej         = 0;
    const unsigned int ek         = blkElem_1D - 1;

    const unsigned int lx         = blk.getAllocationSzX();
    const unsigned int ly         = blk.getAllocationSzY();
    const unsigned int lz         = blk.getAllocationSzZ();

    const int i_offset            = -(m_uiElementOrder - paddWidth);
    const int j_offset            = -(m_uiElementOrder - paddWidth);
    const int k_offset            = (m_uiElementOrder + paddWidth);

    const unsigned int kb         = 0;
    const unsigned int ke         = paddWidth + 1;

    const unsigned int jb         = (m_uiElementOrder - paddWidth);
    const unsigned int je         = (m_uiElementOrder + 1);

    const unsigned int ib         = (m_uiElementOrder - paddWidth);
    const unsigned int ie         = (m_uiElementOrder + 1);

    unsigned int cnum;
    bool isHanging;
    unsigned int nodeLookUp_CG;

    std::vector<T> interpIn;
    interpIn.resize(m_uiNpE);

    std::vector<T> interpOut;
    interpOut.resize(m_uiNpE);

    if (pNodes[lookUp].getLevel() == regLev) {
        T* lookUpVec = &eleDGVec[lookUp * m_uiNpE];
        if (!eleDGValid[lookUp]) {
            getElementNodalValues(zippedVec, lookUpVec, lookUp);
            eleDGValid[lookUp] = true;
        }
        for (unsigned int k = kb; k < ke; k++)
            for (unsigned int j = jb; j < je; j++)
                for (unsigned int i = ib; i < ie; i++) {
                    unzippedVec[offset +
                                (ek * m_uiElementOrder + k + k_offset) *
                                    (ly * lx) +
                                (ej * m_uiElementOrder + j + j_offset) * (lx) +
                                (ei * m_uiElementOrder + i + i_offset)] =
                        lookUpVec[k * (m_uiElementOrder + 1) *
                                      (m_uiElementOrder + 1) +
                                  j * (m_uiElementOrder + 1) + i];
                }

    } else if (pNodes[lookUp].getLevel() < regLev) {
        if (pNodes[lookUp].getLevel() != (regLev - 1)) {
            std::cout
                << "rank: " << rank
                << " [LEFT_DOWN_FRONT Unzip]: 2:1 balance violation blk node: "
                << blkNode << " lookup : " << pNodes[lookUp] << std::endl;
            exit(0);
        }
        assert(pNodes[lookUp].getLevel() == (regLev - 1));
        T* lookUpVec = &eleDGVec[lookUp * m_uiNpE];
        if (!eleDGValid[lookUp]) {
            getElementNodalValues(zippedVec, lookUpVec, lookUp);
            eleDGValid[lookUp] = true;
        }
        cnum = ot::TreeNode(blkNode.minX() - sz, blkNode.minY() - sz,
                            blkNode.maxZ(), regLev, m_uiDim, m_uiMaxDepth)
                   .getMortonIndex();

        parent2ChildInterpolation(lookUpVec, &(*(interpOut.begin())), cnum, 3);
        for (unsigned int k = kb; k < ke; k++)
            for (unsigned int j = jb; j < je; j++)
                for (unsigned int i = ib; i < ie; i++) {
                    unzippedVec[offset +
                                (ek * m_uiElementOrder + k + k_offset) *
                                    (ly * lx) +
                                (ej * m_uiElementOrder + j + j_offset) * (lx) +
                                (ei * m_uiElementOrder + i + i_offset)] =
                        interpOut[k * (m_uiElementOrder + 1) *
                                      (m_uiElementOrder + 1) +
                                  j * (m_uiElementOrder + 1) + i];
                }

    } else {
        if ((pNodes[lookUp].getLevel()) != regLev + 1) {
            std::cout
                << "rank: " << rank
                << " [LEFT_DOWN_FRONT Unzip]: 2:1 balance violation blk node: "
                << blkNode << " lookup : " << pNodes[lookUp] << std::endl;
            exit(0);
        }
        assert((pNodes[lookUp].getLevel()) == regLev + 1);

        cnum = pNodes[lookUp].getMortonIndex();
        for (unsigned int k = 0; k < m_uiElementOrder + 1; k++)
            for (unsigned int j = 0; j < m_uiElementOrder + 1; j++)
                for (unsigned int i = 0; i < m_uiElementOrder + 1; i++) {
                    isHanging = isNodeHanging(lookUp, i, j, k);
                    nodeLookUp_CG =
                        m_uiE2NMapping_CG[lookUp * m_uiNpE +
                                          k * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          j * (m_uiElementOrder + 1) + i];
                    if (isHanging) {
                        interpOut[k * (m_uiElementOrder + 1) *
                                      (m_uiElementOrder + 1) +
                                  j * (m_uiElementOrder + 1) + i] =
                            zippedVec[nodeLookUp_CG];
                    } else {
                        const unsigned int iix =
                            m_uiElementOrder * (int)(cnum & 1u) + i;
                        const unsigned int jjy =
                            m_uiElementOrder * (int)((cnum & 2u) >> 1u) + j;
                        const unsigned int kkz =
                            m_uiElementOrder * (int)((cnum & 4u) >> 2u) + k;
                        // std::cout<<" iix: "<<iix<<" jjy: "<<jjy<<" kkz:
                        // "<<kkz<<std::endl;

                        if ((iix % 2 == 0) && (jjy % 2 == 0) &&
                            (kkz % 2 == 0)) {
                            interpOut[(kkz >> 1u) * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      (jjy >> 1u) * (m_uiElementOrder + 1) +
                                      (iix >> 1u)] = zippedVec[nodeLookUp_CG];
                        }
                    }
                }

        for (unsigned int k = kb; k < ke; k++)
            for (unsigned int j = jb; j < je; j++)
                for (unsigned int i = ib; i < ie; i++) {
                    unzippedVec[offset +
                                (ek * m_uiElementOrder + k + k_offset) *
                                    (ly * lx) +
                                (ej * m_uiElementOrder + j + j_offset) * (lx) +
                                (ei * m_uiElementOrder + i + i_offset)] =
                        interpOut[k * (m_uiElementOrder + 1) *
                                      (m_uiElementOrder + 1) +
                                  j * (m_uiElementOrder + 1) + i];
                }
    }
}

template <typename T>
void Mesh::OCT_DIR_RIGHT_DOWN_FRONT_Unzip(const ot::Block& blk,
                                          const T* zippedVec, T* unzippedVec,
                                          T* eleDGVec, bool* eleDGValid) {
    const unsigned int rank           = getMPIRank();
    const unsigned int dir            = OCT_DIR_RIGHT_DOWN_FRONT;
    const unsigned int* blk2VertexMap = blk.getBlk2VertexMap();
    const unsigned int lookUp         = blk2VertexMap[dir - VERTEX_OFFSET];

    if (lookUp == LOOK_UP_TABLE_DEFAULT) return;

    const unsigned int regLev     = blk.getRegularGridLev();
    const ot::TreeNode* pNodes    = &(*(m_uiAllElements.begin()));
    const unsigned int offset     = blk.getOffset();
    const unsigned int blkElem_1D = 1u
                                    << (regLev - blk.getBlockNode().getLevel());

    const unsigned int paddWidth  = blk.get1DPadWidth();
    const unsigned int sz         = 1u << (m_uiMaxDepth - regLev);
    const ot::TreeNode blkNode    = blk.getBlockNode();

    const unsigned int ei         = blkElem_1D - 1;
    const unsigned int ej         = 0;
    const unsigned int ek         = blkElem_1D - 1;

    const unsigned int lx         = blk.getAllocationSzX();
    const unsigned int ly         = blk.getAllocationSzY();
    const unsigned int lz         = blk.getAllocationSzZ();

    const int i_offset            = (m_uiElementOrder + paddWidth);
    const int j_offset            = -(m_uiElementOrder - paddWidth);
    const int k_offset            = (m_uiElementOrder + paddWidth);

    const unsigned int kb         = 0;
    const unsigned int ke         = paddWidth + 1;

    const unsigned int jb         = (m_uiElementOrder - paddWidth);
    const unsigned int je         = (m_uiElementOrder + 1);

    const unsigned int ib         = 0;
    const unsigned int ie         = paddWidth + 1;

    unsigned int cnum;
    bool isHanging;
    unsigned int nodeLookUp_CG;

    std::vector<T> interpIn;
    interpIn.resize(m_uiNpE);

    std::vector<T> interpOut;
    interpOut.resize(m_uiNpE);

    if (pNodes[lookUp].getLevel() == regLev) {
        T* lookUpVec = &eleDGVec[lookUp * m_uiNpE];
        if (!eleDGValid[lookUp]) {
            getElementNodalValues(zippedVec, lookUpVec, lookUp);
            eleDGValid[lookUp] = true;
        }
        for (unsigned int k = kb; k < ke; k++)
            for (unsigned int j = jb; j < je; j++)
                for (unsigned int i = ib; i < ie; i++) {
                    unzippedVec[offset +
                                (ek * m_uiElementOrder + k + k_offset) *
                                    (ly * lx) +
                                (ej * m_uiElementOrder + j + j_offset) * (lx) +
                                (ei * m_uiElementOrder + i + i_offset)] =
                        lookUpVec[k * (m_uiElementOrder + 1) *
                                      (m_uiElementOrder + 1) +
                                  j * (m_uiElementOrder + 1) + i];
                }

    } else if (pNodes[lookUp].getLevel() < regLev) {
        if (pNodes[lookUp].getLevel() != (regLev - 1)) {
            std::cout
                << "rank: " << rank
                << " [RIGHT_DOWN_FRONT Unzip]: 2:1 balance violation blk node: "
                << blkNode << " lookup : " << pNodes[lookUp] << std::endl;
            exit(0);
        }
        assert(pNodes[lookUp].getLevel() == (regLev - 1));
        T* lookUpVec = &eleDGVec[lookUp * m_uiNpE];
        if (!eleDGValid[lookUp]) {
            getElementNodalValues(zippedVec, lookUpVec, lookUp);
            eleDGValid[lookUp] = true;
        }
        cnum = ot::TreeNode(blkNode.maxX(), blkNode.minY() - sz, blkNode.maxZ(),
                            regLev, m_uiDim, m_uiMaxDepth)
                   .getMortonIndex();

        parent2ChildInterpolation(lookUpVec, &(*(interpOut.begin())), cnum, 3);
        for (unsigned int k = kb; k < ke; k++)
            for (unsigned int j = jb; j < je; j++)
                for (unsigned int i = ib; i < ie; i++) {
                    unzippedVec[offset +
                                (ek * m_uiElementOrder + k + k_offset) *
                                    (ly * lx) +
                                (ej * m_uiElementOrder + j + j_offset) * (lx) +
                                (ei * m_uiElementOrder + i + i_offset)] =
                        interpOut[k * (m_uiElementOrder + 1) *
                                      (m_uiElementOrder + 1) +
                                  j * (m_uiElementOrder + 1) + i];
                }

    } else {
        if ((pNodes[lookUp].getLevel()) != regLev + 1) {
            std::cout
                << "rank: " << rank
                << " [RIGHT_DOWN_FRONT Unzip]: 2:1 balance violation blk node: "
                << blkNode << " lookup : " << pNodes[lookUp] << std::endl;
            exit(0);
        }
        assert((pNodes[lookUp].getLevel()) == regLev + 1);

        cnum = pNodes[lookUp].getMortonIndex();
        for (unsigned int k = 0; k < m_uiElementOrder + 1; k++)
            for (unsigned int j = 0; j < m_uiElementOrder + 1; j++)
                for (unsigned int i = 0; i < m_uiElementOrder + 1; i++) {
                    isHanging = isNodeHanging(lookUp, i, j, k);
                    nodeLookUp_CG =
                        m_uiE2NMapping_CG[lookUp * m_uiNpE +
                                          k * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          j * (m_uiElementOrder + 1) + i];
                    if (isHanging) {
                        interpOut[k * (m_uiElementOrder + 1) *
                                      (m_uiElementOrder + 1) +
                                  j * (m_uiElementOrder + 1) + i] =
                            zippedVec[nodeLookUp_CG];
                    } else {
                        const unsigned int iix =
                            m_uiElementOrder * (int)(cnum & 1u) + i;
                        const unsigned int jjy =
                            m_uiElementOrder * (int)((cnum & 2u) >> 1u) + j;
                        const unsigned int kkz =
                            m_uiElementOrder * (int)((cnum & 4u) >> 2u) + k;
                        // std::cout<<" iix: "<<iix<<" jjy: "<<jjy<<" kkz:
                        // "<<kkz<<std::endl;

                        if ((iix % 2 == 0) && (jjy % 2 == 0) &&
                            (kkz % 2 == 0)) {
                            interpOut[(kkz >> 1u) * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      (jjy >> 1u) * (m_uiElementOrder + 1) +
                                      (iix >> 1u)] = zippedVec[nodeLookUp_CG];
                        }
                    }
                }

        for (unsigned int k = kb; k < ke; k++)
            for (unsigned int j = jb; j < je; j++)
                for (unsigned int i = ib; i < ie; i++) {
                    unzippedVec[offset +
                                (ek * m_uiElementOrder + k + k_offset) *
                                    (ly * lx) +
                                (ej * m_uiElementOrder + j + j_offset) * (lx) +
                                (ei * m_uiElementOrder + i + i_offset)] =
                        interpOut[k * (m_uiElementOrder + 1) *
                                      (m_uiElementOrder + 1) +
                                  j * (m_uiElementOrder + 1) + i];
                }
    }
}

template <typename T>
void Mesh::OCT_DIR_LEFT_UP_FRONT_Unzip(const ot::Block& blk, const T* zippedVec,
                                       T* unzippedVec, T* eleDGVec,
                                       bool* eleDGValid) {
    const unsigned int rank           = getMPIRank();
    const unsigned int dir            = OCT_DIR_LEFT_UP_FRONT;
    const unsigned int* blk2VertexMap = blk.getBlk2VertexMap();
    const unsigned int lookUp         = blk2VertexMap[dir - VERTEX_OFFSET];

    if (lookUp == LOOK_UP_TABLE_DEFAULT) return;

    const unsigned int regLev     = blk.getRegularGridLev();
    const ot::TreeNode* pNodes    = &(*(m_uiAllElements.begin()));
    const unsigned int offset     = blk.getOffset();
    const unsigned int blkElem_1D = 1u
                                    << (regLev - blk.getBlockNode().getLevel());

    const unsigned int paddWidth  = blk.get1DPadWidth();
    const unsigned int sz         = 1u << (m_uiMaxDepth - regLev);
    const ot::TreeNode blkNode    = blk.getBlockNode();

    const unsigned int ei         = 0;
    const unsigned int ej         = blkElem_1D - 1;
    const unsigned int ek         = blkElem_1D - 1;

    const unsigned int lx         = blk.getAllocationSzX();
    const unsigned int ly         = blk.getAllocationSzY();
    const unsigned int lz         = blk.getAllocationSzZ();

    const int i_offset            = -(m_uiElementOrder - paddWidth);
    const int j_offset            = (m_uiElementOrder + paddWidth);
    const int k_offset            = (m_uiElementOrder + paddWidth);

    const unsigned int kb         = 0;
    const unsigned int ke         = paddWidth + 1;

    const unsigned int jb         = 0;
    const unsigned int je         = paddWidth + 1;

    const unsigned int ib         = (m_uiElementOrder - paddWidth);
    const unsigned int ie         = (m_uiElementOrder + 1);

    unsigned int cnum;
    bool isHanging;
    unsigned int nodeLookUp_CG;

    std::vector<T> interpIn;
    interpIn.resize(m_uiNpE);

    std::vector<T> interpOut;
    interpOut.resize(m_uiNpE);

    if (pNodes[lookUp].getLevel() == regLev) {
        T* lookUpVec = &eleDGVec[lookUp * m_uiNpE];
        if (!eleDGValid[lookUp]) {
            getElementNodalValues(zippedVec, lookUpVec, lookUp);
            eleDGValid[lookUp] = true;
        }
        for (unsigned int k = kb; k < ke; k++)
            for (unsigned int j = jb; j < je; j++)
                for (unsigned int i = ib; i < ie; i++) {
                    unzippedVec[offset +
                                (ek * m_uiElementOrder + k + k_offset) *
                                    (ly * lx) +
                                (ej * m_uiElementOrder + j + j_offset) * (lx) +
                                (ei * m_uiElementOrder + i + i_offset)] =
                        lookUpVec[k * (m_uiElementOrder + 1) *
                                      (m_uiElementOrder + 1) +
                                  j * (m_uiElementOrder + 1) + i];
                }

    } else if (pNodes[lookUp].getLevel() < regLev) {
        if (pNodes[lookUp].getLevel() != (regLev - 1)) {
            std::cout
                << "rank: " << rank
                << " [LEFT_UP_FRONT Unzip]: 2:1 balance violation blk node: "
                << blkNode << " lookup : " << pNodes[lookUp] << std::endl;
            exit(0);
        }
        assert(pNodes[lookUp].getLevel() == (regLev - 1));
        T* lookUpVec = &eleDGVec[lookUp * m_uiNpE];
        if (!eleDGValid[lookUp]) {
            getElementNodalValues(zippedVec, lookUpVec, lookUp);
            eleDGValid[lookUp] = true;
        }
        cnum = ot::TreeNode(blkNode.minX() - sz, blkNode.maxY(), blkNode.maxZ(),
                            regLev, m_uiDim, m_uiMaxDepth)
                   .getMortonIndex();

        parent2ChildInterpolation(lookUpVec, &(*(interpOut.begin())), cnum, 3);
        for (unsigned int k = kb; k < ke; k++)
            for (unsigned int j = jb; j < je; j++)
                for (unsigned int i = ib; i < ie; i++) {
                    unzippedVec[offset +
                                (ek * m_uiElementOrder + k + k_offset) *
                                    (ly * lx) +
                                (ej * m_uiElementOrder + j + j_offset) * (lx) +
                                (ei * m_uiElementOrder + i + i_offset)] =
                        interpOut[k * (m_uiElementOrder + 1) *
                                      (m_uiElementOrder + 1) +
                                  j * (m_uiElementOrder + 1) + i];
                }

    } else {
        if ((pNodes[lookUp].getLevel()) != regLev + 1) {
            std::cout
                << "rank: " << rank
                << " [LEFT_UP_FRONT Unzip]: 2:1 balance violation blk node: "
                << blkNode << " lookup : " << pNodes[lookUp] << std::endl;
            exit(0);
        }
        assert((pNodes[lookUp].getLevel()) == regLev + 1);

        cnum = pNodes[lookUp].getMortonIndex();
        for (unsigned int k = 0; k < m_uiElementOrder + 1; k++)
            for (unsigned int j = 0; j < m_uiElementOrder + 1; j++)
                for (unsigned int i = 0; i < m_uiElementOrder + 1; i++) {
                    isHanging = isNodeHanging(lookUp, i, j, k);
                    nodeLookUp_CG =
                        m_uiE2NMapping_CG[lookUp * m_uiNpE +
                                          k * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          j * (m_uiElementOrder + 1) + i];
                    if (isHanging) {
                        interpOut[k * (m_uiElementOrder + 1) *
                                      (m_uiElementOrder + 1) +
                                  j * (m_uiElementOrder + 1) + i] =
                            zippedVec[nodeLookUp_CG];
                    } else {
                        const unsigned int iix =
                            m_uiElementOrder * (int)(cnum & 1u) + i;
                        const unsigned int jjy =
                            m_uiElementOrder * (int)((cnum & 2u) >> 1u) + j;
                        const unsigned int kkz =
                            m_uiElementOrder * (int)((cnum & 4u) >> 2u) + k;
                        // std::cout<<" iix: "<<iix<<" jjy: "<<jjy<<" kkz:
                        // "<<kkz<<std::endl;

                        if ((iix % 2 == 0) && (jjy % 2 == 0) &&
                            (kkz % 2 == 0)) {
                            interpOut[(kkz >> 1u) * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      (jjy >> 1u) * (m_uiElementOrder + 1) +
                                      (iix >> 1u)] = zippedVec[nodeLookUp_CG];
                        }
                    }
                }

        for (unsigned int k = kb; k < ke; k++)
            for (unsigned int j = jb; j < je; j++)
                for (unsigned int i = ib; i < ie; i++) {
                    unzippedVec[offset +
                                (ek * m_uiElementOrder + k + k_offset) *
                                    (ly * lx) +
                                (ej * m_uiElementOrder + j + j_offset) * (lx) +
                                (ei * m_uiElementOrder + i + i_offset)] =
                        interpOut[k * (m_uiElementOrder + 1) *
                                      (m_uiElementOrder + 1) +
                                  j * (m_uiElementOrder + 1) + i];
                }
    }
}

template <typename T>
void Mesh::OCT_DIR_RIGHT_UP_FRONT_Unzip(const ot::Block& blk,
                                        const T* zippedVec, T* unzippedVec,
                                        T* eleDGVec, bool* eleDGValid) {
    const unsigned int rank           = getMPIRank();
    const unsigned int dir            = OCT_DIR_RIGHT_UP_FRONT;
    const unsigned int* blk2VertexMap = blk.getBlk2VertexMap();
    const unsigned int lookUp         = blk2VertexMap[dir - VERTEX_OFFSET];

    if (lookUp == LOOK_UP_TABLE_DEFAULT) return;

    const unsigned int regLev     = blk.getRegularGridLev();
    const ot::TreeNode* pNodes    = &(*(m_uiAllElements.begin()));
    const unsigned int offset     = blk.getOffset();
    const unsigned int blkElem_1D = 1u
                                    << (regLev - blk.getBlockNode().getLevel());

    const unsigned int paddWidth  = blk.get1DPadWidth();
    const unsigned int sz         = 1u << (m_uiMaxDepth - regLev);
    const ot::TreeNode blkNode    = blk.getBlockNode();

    const unsigned int ei         = blkElem_1D - 1;
    const unsigned int ej         = blkElem_1D - 1;
    const unsigned int ek         = blkElem_1D - 1;

    const unsigned int lx         = blk.getAllocationSzX();
    const unsigned int ly         = blk.getAllocationSzY();
    const unsigned int lz         = blk.getAllocationSzZ();

    const int i_offset            = (m_uiElementOrder + paddWidth);
    const int j_offset            = (m_uiElementOrder + paddWidth);
    const int k_offset            = (m_uiElementOrder + paddWidth);

    const unsigned int kb         = 0;
    const unsigned int ke         = paddWidth + 1;

    const unsigned int jb         = 0;
    const unsigned int je         = paddWidth + 1;

    const unsigned int ib         = 0;
    const unsigned int ie         = paddWidth + 1;

    unsigned int cnum;
    bool isHanging;
    unsigned int nodeLookUp_CG;

    std::vector<T> interpIn;
    interpIn.resize(m_uiNpE);

    std::vector<T> interpOut;
    interpOut.resize(m_uiNpE);

    if (pNodes[lookUp].getLevel() == regLev) {
        T* lookUpVec = &eleDGVec[lookUp * m_uiNpE];
        if (!eleDGValid[lookUp]) {
            getElementNodalValues(zippedVec, lookUpVec, lookUp);
            eleDGValid[lookUp] = true;
        }
        for (unsigned int k = kb; k < ke; k++)
            for (unsigned int j = jb; j < je; j++)
                for (unsigned int i = ib; i < ie; i++) {
                    unzippedVec[offset +
                                (ek * m_uiElementOrder + k + k_offset) *
                                    (ly * lx) +
                                (ej * m_uiElementOrder + j + j_offset) * (lx) +
                                (ei * m_uiElementOrder + i + i_offset)] =
                        lookUpVec[k * (m_uiElementOrder + 1) *
                                      (m_uiElementOrder + 1) +
                                  j * (m_uiElementOrder + 1) + i];
                }

    } else if (pNodes[lookUp].getLevel() < regLev) {
        if (pNodes[lookUp].getLevel() != (regLev - 1)) {
            std::cout
                << "rank: " << rank
                << " [RIGHT_UP_FRONT Unzip]: 2:1 balance violation blk node: "
                << blkNode << " lookup : " << pNodes[lookUp] << std::endl;
            exit(0);
        }
        assert(pNodes[lookUp].getLevel() == (regLev - 1));
        T* lookUpVec = &eleDGVec[lookUp * m_uiNpE];
        if (!eleDGValid[lookUp]) {
            getElementNodalValues(zippedVec, lookUpVec, lookUp);
            eleDGValid[lookUp] = true;
        }
        cnum = ot::TreeNode(blkNode.maxX(), blkNode.maxY(), blkNode.maxZ(),
                            regLev, m_uiDim, m_uiMaxDepth)
                   .getMortonIndex();

        parent2ChildInterpolation(lookUpVec, &(*(interpOut.begin())), cnum, 3);
        for (unsigned int k = kb; k < ke; k++)
            for (unsigned int j = jb; j < je; j++)
                for (unsigned int i = ib; i < ie; i++) {
                    unzippedVec[offset +
                                (ek * m_uiElementOrder + k + k_offset) *
                                    (ly * lx) +
                                (ej * m_uiElementOrder + j + j_offset) * (lx) +
                                (ei * m_uiElementOrder + i + i_offset)] =
                        interpOut[k * (m_uiElementOrder + 1) *
                                      (m_uiElementOrder + 1) +
                                  j * (m_uiElementOrder + 1) + i];
                }

    } else {
        if ((pNodes[lookUp].getLevel()) != regLev + 1) {
            std::cout
                << "rank: " << rank
                << " [RIGHT_UP_FRONT Unzip]: 2:1 balance violation blk node: "
                << blkNode << " lookup : " << pNodes[lookUp] << std::endl;
            exit(0);
        }
        assert((pNodes[lookUp].getLevel()) == regLev + 1);

        cnum = pNodes[lookUp].getMortonIndex();
        for (unsigned int k = 0; k < m_uiElementOrder + 1; k++)
            for (unsigned int j = 0; j < m_uiElementOrder + 1; j++)
                for (unsigned int i = 0; i < m_uiElementOrder + 1; i++) {
                    isHanging = isNodeHanging(lookUp, i, j, k);
                    nodeLookUp_CG =
                        m_uiE2NMapping_CG[lookUp * m_uiNpE +
                                          k * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          j * (m_uiElementOrder + 1) + i];
                    if (isHanging) {
                        interpOut[k * (m_uiElementOrder + 1) *
                                      (m_uiElementOrder + 1) +
                                  j * (m_uiElementOrder + 1) + i] =
                            zippedVec[nodeLookUp_CG];
                    } else {
                        const unsigned int iix =
                            m_uiElementOrder * (int)(cnum & 1u) + i;
                        const unsigned int jjy =
                            m_uiElementOrder * (int)((cnum & 2u) >> 1u) + j;
                        const unsigned int kkz =
                            m_uiElementOrder * (int)((cnum & 4u) >> 2u) + k;
                        // std::cout<<" iix: "<<iix<<" jjy: "<<jjy<<" kkz:
                        // "<<kkz<<std::endl;

                        if ((iix % 2 == 0) && (jjy % 2 == 0) &&
                            (kkz % 2 == 0)) {
                            interpOut[(kkz >> 1u) * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      (jjy >> 1u) * (m_uiElementOrder + 1) +
                                      (iix >> 1u)] = zippedVec[nodeLookUp_CG];
                        }
                    }
                }

        for (unsigned int k = kb; k < ke; k++)
            for (unsigned int j = jb; j < je; j++)
                for (unsigned int i = ib; i < ie; i++) {
                    unzippedVec[offset +
                                (ek * m_uiElementOrder + k + k_offset) *
                                    (ly * lx) +
                                (ej * m_uiElementOrder + j + j_offset) * (lx) +
                                (ei * m_uiElementOrder + i + i_offset)] =
                        interpOut[k * (m_uiElementOrder + 1) *
                                      (m_uiElementOrder + 1) +
                                  j * (m_uiElementOrder + 1) + i];
                }
    }
}

template <typename T>
void Mesh::blockDiagonalUnZip(const ot::Block& blk, const T* zippedVec,
                              T* unzippedVec, T* eleDGVec, bool* eleDGValid) {
    OCT_DIR_LEFT_DOWN_Unzip(blk, zippedVec, unzippedVec, eleDGVec, eleDGValid);
    OCT_DIR_LEFT_UP_Unzip(blk, zippedVec, unzippedVec, eleDGVec, eleDGValid);
    OCT_DIR_LEFT_BACK_Unzip(blk, zippedVec, unzippedVec, eleDGVec, eleDGValid);
    OCT_DIR_LEFT_FRONT_Unzip(blk, zippedVec, unzippedVec, eleDGVec, eleDGValid);
    OCT_DIR_RIGHT_DOWN_Unzip(blk, zippedVec, unzippedVec, eleDGVec, eleDGValid);
    OCT_DIR_RIGHT_UP_Unzip(blk, zippedVec, unzippedVec, eleDGVec, eleDGValid);
    OCT_DIR_RIGHT_BACK_Unzip(blk, zippedVec, unzippedVec, eleDGVec, eleDGValid);
    OCT_DIR_RIGHT_FRONT_Unzip(blk, zippedVec, unzippedVec, eleDGVec,
                              eleDGValid);
    OCT_DIR_DOWN_BACK_Unzip(blk, zippedVec, unzippedVec, eleDGVec, eleDGValid);
    OCT_DIR_DOWN_FRONT_Unzip(blk, zippedVec, unzippedVec, eleDGVec, eleDGValid);
    OCT_DIR_UP_BACK_Unzip(blk, zippedVec, unzippedVec, eleDGVec, eleDGValid);
    OCT_DIR_UP_FRONT_Unzip(blk, zippedVec, unzippedVec, eleDGVec, eleDGValid);
}

template <typename T>
void Mesh::blockVertexUnZip(const ot::Block& blk, const T* zippedVec,
                            T* unzippedVec, T* eleDGVec, bool* eleDGValid) {
    OCT_DIR_LEFT_DOWN_BACK_Unzip(blk, zippedVec, unzippedVec, eleDGVec,
                                 eleDGValid);
    OCT_DIR_RIGHT_DOWN_BACK_Unzip(blk, zippedVec, unzippedVec, eleDGVec,
                                  eleDGValid);
    OCT_DIR_LEFT_UP_BACK_Unzip(blk, zippedVec, unzippedVec, eleDGVec,
                               eleDGValid);
    OCT_DIR_RIGHT_UP_BACK_Unzip(blk, zippedVec, unzippedVec, eleDGVec,
                                eleDGValid);

    OCT_DIR_LEFT_DOWN_FRONT_Unzip(blk, zippedVec, unzippedVec, eleDGVec,
                                  eleDGValid);
    OCT_DIR_RIGHT_DOWN_FRONT_Unzip(blk, zippedVec, unzippedVec, eleDGVec,
                                   eleDGValid);
    OCT_DIR_LEFT_UP_FRONT_Unzip(blk, zippedVec, unzippedVec, eleDGVec,
                                eleDGValid);
    OCT_DIR_RIGHT_UP_FRONT_Unzip(blk, zippedVec, unzippedVec, eleDGVec,
                                 eleDGValid);
}

template <typename T>
void Mesh::child2ParentInjection(const T* in, T* out, unsigned int* child,
                                 unsigned int lev) const {
    for (unsigned int cnum = 0; cnum < NUM_CHILDREN; cnum++) {
        if (child[cnum] == LOOK_UP_TABLE_DEFAULT ||
            m_uiAllElements[child[cnum]].getLevel() != lev ||
            !m_uiIsNodalMapValid[child[cnum]])
            continue;

        for (unsigned int k = 0; k < m_uiElementOrder + 1; k++)
            for (unsigned int j = 0; j < m_uiElementOrder + 1; j++)
                for (unsigned int i = 0; i < m_uiElementOrder + 1; i++) {
                    const bool isHanging =
                        this->isNodeHanging(child[cnum], i, j, k);
                    if (isHanging) {
                        out[k * (m_uiElementOrder + 1) *
                                (m_uiElementOrder + 1) +
                            j * (m_uiElementOrder + 1) + i] =
                            in[m_uiE2NMapping_CG[child[cnum] * m_uiNpE +
                                                 k * (m_uiElementOrder + 1) *
                                                     (m_uiElementOrder + 1) +
                                                 j * (m_uiElementOrder + 1) +
                                                 i]];
                    } else {
                        const unsigned int iix =
                            m_uiElementOrder * (int)(cnum & 1u) + i;
                        const unsigned int jjy =
                            m_uiElementOrder * (int)((cnum & 2u) >> 1u) + j;
                        const unsigned int kkz =
                            m_uiElementOrder * (int)((cnum & 4u) >> 2u) + k;
                        // std::cout<<" iix: "<<iix<<" jjy: "<<jjy<<" kkz:
                        // "<<kkz<<std::endl;

                        if ((iix % 2 == 0) && (jjy % 2 == 0) &&
                            (kkz % 2 == 0)) {
                            out[(kkz >> 1u) * (m_uiElementOrder + 1) *
                                    (m_uiElementOrder + 1) +
                                (jjy >> 1u) * (m_uiElementOrder + 1) +
                                (iix >> 1u)] =
                                in[m_uiE2NMapping_CG
                                       [child[cnum] * m_uiNpE +
                                        k * (m_uiElementOrder + 1) *
                                            (m_uiElementOrder + 1) +
                                        j * (m_uiElementOrder + 1) + i]];
                        }
                    }
                }
    }
}

template <typename T>
void Mesh::unzip(const T* in, T* out, const unsigned int* blkIDs,
                 unsigned int numblks, unsigned int dof) {
    if ((!m_uiIsActive) || (m_uiLocalBlockList.empty())) return;

    ot::TreeNode blkNode;
    unsigned int ei, ej, ek;  // element wise xyz coordinates.
    const ot::TreeNode* pNodes = &(*(m_uiAllElements.begin()));
    unsigned int regLev;
    // unsigned int blkNpe_1D;

    unsigned int lookUp;
    unsigned int lookUp1;
    unsigned int cnum;
    unsigned int faceCnum;

    unsigned int faceNeighCnum1[4] = {0, 0, 0, 0};  // immidiate neighbors
    unsigned int faceNeighCnum2[4] = {0, 0, 0, 0};  // neighbor's neighbors

    DendroRegister unsigned int nodeLookUp_CG;
    DendroRegister unsigned int nodeLookUp_DG;

    std::vector<T> interpOrInjectionOut;  // interpolation or injection output.
    std::vector<T> injectionInput;  // input for the injection (values from all
                                    // the 8 children) (This should be put in
                                    // the order of the morton ordering. )
    std::vector<T> interpolationInput;

    std::vector<T> edgeInterpIn;
    std::vector<T> edgeInterpOut;

    std::vector<T> faceInterpIn;
    std::vector<T> faceInterpOut;

    std::vector<unsigned int> edgeIndex;
    std::vector<unsigned int> faceIndex;
    std::vector<unsigned int> child;
    child.resize(NUM_CHILDREN);

    interpOrInjectionOut.resize(m_uiNpE);
    interpolationInput.resize(m_uiNpE);
    // injectionInput.resize(m_uiNpE*NUM_CHILDREN);

    std::vector<T> injectionTest;
    injectionTest.resize(m_uiNpE * NUM_CHILDREN);

    edgeIndex.resize((m_uiElementOrder + 1));
    faceIndex.resize((m_uiElementOrder + 1) * (m_uiElementOrder + 1));

    edgeInterpIn.resize((m_uiElementOrder + 1));
    edgeInterpOut.resize((m_uiElementOrder + 1));

    faceInterpIn.resize((m_uiElementOrder + 1) * (m_uiElementOrder + 1));
    faceInterpOut.resize((m_uiElementOrder + 1) * (m_uiElementOrder + 1));

    unsigned int mid_bit = 0;
    unsigned int sz;
    bool isHanging;
    unsigned int ownerID, ii_x, jj_y, kk_z;
    unsigned int eleIndexMin = 0;
    unsigned int eleIndexMax = 0;
    bool edgeHanging;
    bool faceHanging;

    unsigned int lx, ly, lz, offset, paddWidth;
    bool isParentValue = false;

    unsigned int fid[(NUM_CHILDREN >> 1u)];
    unsigned int cid[(NUM_CHILDREN >> 1u)];

/*if(!rank) std::cout<<"begin unzip "<<std::endl;*/
#ifdef DEBUG_UNZIP_OP
    double d_min, d_max;
    d_min = -0.5;
    d_max = 0.5;
    double x, y, z;
    unsigned int x1, y1, z1;
    std::function<double(double, double, double)> func =
        [d_min, d_max](const double x, const double y, const double z) {
            return (
                sin(2 * M_PI *
                    ((x / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min)) *
                sin(2 * M_PI *
                    ((y / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min)) *
                sin(2 * M_PI *
                    ((z / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min)));
        };
#endif

    // NOTE: Be careful when you access ghost elements for padding. (You should
    // only access the level 1 ghost elements. You should not access the level 2
    // ghost elements at any time. )
    paddWidth = m_uiLocalBlockList[0].get1DPadWidth();

    if ((m_uiElementOrder + 1) / 2 < paddWidth) {
        std::cout << "rank: " << m_uiActiveRank
                  << " paddiging with size : " << paddWidth
                  << " is too large for element order : " << m_uiElementOrder
                  << std::endl;
        MPI_Abort(m_uiCommGlobal, 0);
    }

    assert(numblks <= m_uiLocalBlockList.size());

    std::vector<T> ele_dg_vec;
    ele_dg_vec.resize(m_uiNumTotalElements * m_uiNpE, (T)0);
    bool* eleVec_valid = new bool[m_uiAllElements.size()];

    for (unsigned int v = 0; v < dof; v++) {
        const T* zippedVec = in + v * m_uiNumActualNodes;
        T* unzippedVec     = out + v * m_uiUnZippedVecSz;

        for (unsigned int ii = 0; ii < m_uiAllElements.size(); ii++)
            eleVec_valid[ii] = false;

        for (unsigned int b = 0; b < numblks; b++) {
            const unsigned int blk = blkIDs[b];
            blkNode                = m_uiLocalBlockList[blk].getBlockNode();
            assert(blkNode.maxX() <= m_uiMeshDomain_max &&
                   blkNode.minX() >= m_uiMeshDomain_min);
            regLev      = m_uiLocalBlockList[blk].getRegularGridLev();
            // blkNpe_1D=m_uiElementOrder*(1u<<(regLev-blkNode.getLevel()))+1+2*GHOST_WIDTH;
            // std::cout<<"rank: "<<m_uiActiveRank<<" -- blkNpw_1D:
            // "<<blkNpe_1D<<" blkNode: "<<blkNode<<" regLev:
            // "<<regLev<<std::endl;

            sz          = 1u << (m_uiMaxDepth - regLev);
            eleIndexMax = (1u << (regLev - blkNode.getLevel())) - 1;
            assert(eleIndexMax >= eleIndexMin);

            lx        = m_uiLocalBlockList[blk].getAllocationSzX();
            ly        = m_uiLocalBlockList[blk].getAllocationSzY();
            lz        = m_uiLocalBlockList[blk].getAllocationSzZ();
            offset    = m_uiLocalBlockList[blk].getOffset();
            paddWidth = m_uiLocalBlockList[blk].get1DPadWidth();

            for (unsigned int elem =
                     m_uiLocalBlockList[blk].getLocalElementBegin();
                 elem < m_uiLocalBlockList[blk].getLocalElementEnd(); elem++) {
                ei = (pNodes[elem].getX() - blkNode.getX()) >>
                     (m_uiMaxDepth - regLev);
                ej = (pNodes[elem].getY() - blkNode.getY()) >>
                     (m_uiMaxDepth - regLev);
                ek = (pNodes[elem].getZ() - blkNode.getZ()) >>
                     (m_uiMaxDepth - regLev);

                // std::cout<<"blk: "<<blk<<" : "<<blkNode<<" ek: "<<(ek)<<" ej:
                // "<<(ej)<<" ei: "<<(ei)<<" elem:
                // "<<m_uiAllElements[elem]<<std::endl;
                assert(pNodes[elem].getLevel() ==
                       regLev);  // this is enforced by block construction
#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                dendro::timer::t_unzip_sync_internal.start();
#endif

                T* lookUpElementVec = &ele_dg_vec[elem * m_uiNpE];
                if (!eleVec_valid[elem]) {
                    this->getElementNodalValues(zippedVec, lookUpElementVec,
                                                elem);
                    eleVec_valid[elem] = true;
                }

// (1). local nodes copy. Not need to interpolate or inject values. By block
// construction local octants in the block has is the same level as regular
// grid.
#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                dendro::timer::t_unzip_sync_cpy.start();
#endif
                for (unsigned int k = 0; k < (m_uiElementOrder + 1); k++)
                    for (unsigned int j = 0; j < (m_uiElementOrder + 1); j++)
                        for (unsigned int i = 0; i < (m_uiElementOrder + 1);
                             i++)
                            unzippedVec[offset +
                                        (ek * m_uiElementOrder + k +
                                         paddWidth) *
                                            (ly * lx) +
                                        (ej * m_uiElementOrder + j +
                                         paddWidth) *
                                            (lx) +
                                        (ei * m_uiElementOrder + i +
                                         paddWidth)] =
                                lookUpElementVec[k * (m_uiElementOrder + 1) *
                                                     (m_uiElementOrder + 1) +
                                                 j * (m_uiElementOrder + 1) +
                                                 i];

#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                dendro::timer::t_unzip_sync_cpy.stop();
                dendro::timer::t_unzip_sync_internal.stop();
#endif
// (2). copy the ghost layer (we only copy GHOST_WIDTH amounts of data from the
// zipped array )z`
//---------------------------------------------------------X direction padding
//--------------------------------------------------------------------------------------------------------------------
#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                dendro::timer::t_unzip_sync_face[0].start();
#endif
                if ((pNodes[elem].minX() == blkNode.minX())) {
                    assert(ei == eleIndexMin);

                    lookUp =
                        m_uiE2EMapping[elem * m_uiNumDirections + OCT_DIR_LEFT];
                    if (lookUp != LOOK_UP_TABLE_DEFAULT) {
                        if (pNodes[lookUp].getLevel() ==
                            pNodes[elem].getLevel()) {
#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_f_c1.start();
#endif
                            assert(paddWidth < (m_uiElementOrder + 1));

                            T* lookUpElementVec = &ele_dg_vec[lookUp * m_uiNpE];
                            if (!eleVec_valid[lookUp]) {
                                this->getElementNodalValues(
                                    zippedVec, lookUpElementVec, lookUp);
                                eleVec_valid[lookUp] = true;
                            }

#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_cpy.start();
#endif
                            for (unsigned int k = 0; k < (m_uiElementOrder + 1);
                                 k++)
                                for (unsigned int j = 0;
                                     j < (m_uiElementOrder + 1); j++)
                                    for (unsigned int i =
                                             (m_uiElementOrder - paddWidth);
                                         i < (m_uiElementOrder + 1); i++)
                                        unzippedVec[offset +
                                                    (ek * m_uiElementOrder + k +
                                                     paddWidth) *
                                                        (ly * lx) +
                                                    (ej * m_uiElementOrder + j +
                                                     paddWidth) *
                                                        (lx) +
                                                    (ei * m_uiElementOrder + i -
                                                     (m_uiElementOrder -
                                                      paddWidth))] =
                                            lookUpElementVec
                                                [k * (m_uiElementOrder + 1) *
                                                     (m_uiElementOrder + 1) +
                                                 j * (m_uiElementOrder + 1) +
                                                 i];

#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_cpy.stop();
                            dendro::timer::t_unzip_sync_f_c1.stop();
#endif

                        } else if (pNodes[lookUp].getLevel() <
                                   pNodes[elem].getLevel()) {
#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_f_c2.start();
#endif
                            assert(pNodes[lookUp].getLevel() + 1 == regLev);
                            mid_bit =
                                m_uiMaxDepth - pNodes[lookUp].getLevel() - 1;
                            cnum = (((((pNodes[elem].getZ()) >> mid_bit) & 1u)
                                     << 2u) |
                                    ((((pNodes[elem].getY()) >> mid_bit) & 1u)
                                     << 1u) |
                                    ((((pNodes[elem].getX() - sz)) >> mid_bit) &
                                     1u));
                            // std::cout<<"elem: "<<elem<<" :
                            // "<<m_uiAllElements[elem]<<" lookup:
                            // "<<m_uiAllElements[lookUp]<<" child:
                            // "<<ot::TreeNode(pNodes[elem].getX()-sz,pNodes[elem].getY(),pNodes[elem].getZ(),pNodes[elem].getLevel(),m_uiDim,m_uiMaxDepth)<<"
                            // cnum: "<<cnum<<std::endl;

#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_cpy.start();
#endif

#ifdef USE_FD_INTERP_FOR_UNZIP
                            const int st = this->getBlkBdyParentCNums(
                                blk, elem, OCT_DIR_LEFT, child.data(), fid,
                                cid);
                            if (st > 0) {
                                const unsigned int NUM_CHILDREN_BY2 =
                                    (NUM_CHILDREN >> 1u);
                                this->getBlkBoundaryParentNodes(
                                    zippedVec, lookUpElementVec.data(),
                                    interpolationInput.data(),
                                    interpOrInjectionOut.data(), lookUp, fid,
                                    cid, child.data());
                                for (unsigned int w = 0; w < NUM_CHILDREN_BY2;
                                     w++) {
                                    assert(pNodes[lookUp] ==
                                           pNodes[m_uiE2EMapping
                                                      [child[fid[w]] *
                                                           m_uiNumDirections +
                                                       OCT_DIR_LEFT]]);
                                    assert(child[fid[w]] !=
                                           LOOK_UP_TABLE_DEFAULT);
                                    // assert(m_uiE2BlkMap[(child[fid[w]] -
                                    // m_uiElementLocalBegin) ] == blk);

                                    if (child[fid[w]] < m_uiElementLocalBegin ||
                                        child[fid[w]] >= m_uiElementLocalEnd)
                                        continue;

                                    this->parent2ChildInterpolation(
                                        lookUpElementVec.data(),
                                        interpOrInjectionOut.data(), cid[w],
                                        m_uiDim);

                                    const ot::Block blk_fd =
                                        m_uiLocalBlockList[m_uiE2BlkMap[(
                                            child[fid[w]] -
                                            m_uiElementLocalBegin)]];
                                    const ot::TreeNode blkNode_fd =
                                        blk_fd.getBlockNode();
                                    const unsigned int regL_fd =
                                        blk_fd.getRegularGridLev();

                                    const unsigned int lx_fd =
                                        blk_fd.getAllocationSzX();
                                    const unsigned int ly_fd =
                                        blk_fd.getAllocationSzY();
                                    const unsigned int lz_fd =
                                        blk_fd.getAllocationSzZ();

                                    const unsigned int offset_fd =
                                        blk_fd.getOffset();

                                    const unsigned int ei_fd =
                                        (pNodes[child[fid[w]]].getX() -
                                         blkNode_fd.getX()) >>
                                        (m_uiMaxDepth - regL_fd);
                                    const unsigned int ej_fd =
                                        (pNodes[child[fid[w]]].getY() -
                                         blkNode_fd.getY()) >>
                                        (m_uiMaxDepth - regL_fd);
                                    const unsigned int ek_fd =
                                        (pNodes[child[fid[w]]].getZ() -
                                         blkNode_fd.getZ()) >>
                                        (m_uiMaxDepth - regL_fd);

                                    assert(paddWidth < (m_uiElementOrder + 1));
                                    for (unsigned int k = 0;
                                         k < (m_uiElementOrder + 1); k++)
                                        for (unsigned int j = 0;
                                             j < (m_uiElementOrder + 1); j++)
                                            for (unsigned int i =
                                                     (m_uiElementOrder -
                                                      paddWidth);
                                                 i < (m_uiElementOrder + 1);
                                                 i++)
                                                unzippedVec
                                                    [offset_fd +
                                                     (ek_fd * m_uiElementOrder +
                                                      k + paddWidth) *
                                                         (ly_fd * lx_fd) +
                                                     (ej_fd * m_uiElementOrder +
                                                      j + paddWidth) *
                                                         (lx_fd) +
                                                     (ei_fd * m_uiElementOrder +
                                                      i -
                                                      (m_uiElementOrder -
                                                       paddWidth))] = interpOrInjectionOut
                                                        [k *
                                                             (m_uiElementOrder +
                                                              1) *
                                                             (m_uiElementOrder +
                                                              1) +
                                                         j * (m_uiElementOrder +
                                                              1) +
                                                         i];
                                }
                            }
#else

                            T* lookUpElementVec = &ele_dg_vec[lookUp * m_uiNpE];
                            if (!eleVec_valid[lookUp]) {
                                this->getElementNodalValues(
                                    zippedVec, lookUpElementVec, lookUp);
                                eleVec_valid[lookUp] = true;
                            }
                            this->parent2ChildInterpolation(
                                lookUpElementVec,
                                &(*(interpOrInjectionOut.begin())), cnum);

                            assert(paddWidth < (m_uiElementOrder + 1));
                            for (unsigned int k = 0; k < (m_uiElementOrder + 1);
                                 k++)
                                for (unsigned int j = 0;
                                     j < (m_uiElementOrder + 1); j++)
                                    for (unsigned int i =
                                             (m_uiElementOrder - paddWidth);
                                         i < (m_uiElementOrder + 1); i++)
                                        unzippedVec[offset +
                                                    (ek * m_uiElementOrder + k +
                                                     paddWidth) *
                                                        (ly * lx) +
                                                    (ej * m_uiElementOrder + j +
                                                     paddWidth) *
                                                        (lx) +
                                                    (ei * m_uiElementOrder + i -
                                                     (m_uiElementOrder -
                                                      paddWidth))] =
                                            interpOrInjectionOut
                                                [k * (m_uiElementOrder + 1) *
                                                     (m_uiElementOrder + 1) +
                                                 j * (m_uiElementOrder + 1) +
                                                 i];
#endif

#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_cpy.stop();
                            dendro::timer::t_unzip_sync_f_c2.stop();
#endif

                        } else if (pNodes[lookUp].getLevel() >
                                   pNodes[elem].getLevel()) {
#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_f_c3.start();
#endif
                            assert(pNodes[lookUp].getLevel() == (regLev + 1));
                            // child.resize(NUM_CHILDREN,LOOK_UP_TABLE_DEFAULT);
                            //  get the immediate neighbours. These cannot be
                            //  LOOK_UP_TABLE_DEFAULT.
                            child[1] = lookUp;
                            child[3] =
                                m_uiE2EMapping[child[1] * m_uiNumDirections +
                                               OCT_DIR_UP];
                            assert(child[3] != LOOK_UP_TABLE_DEFAULT);
                            child[5] =
                                m_uiE2EMapping[child[1] * m_uiNumDirections +
                                               OCT_DIR_FRONT];
                            assert(child[5] != LOOK_UP_TABLE_DEFAULT);
                            child[7] =
                                m_uiE2EMapping[child[3] * m_uiNumDirections +
                                               OCT_DIR_FRONT];
                            assert(child[7] != LOOK_UP_TABLE_DEFAULT);

                            if (m_uiElementOrder == 4 && paddWidth == 3) {
                                // we need to search for the additional points.
                                child[0] =
                                    m_uiE2EMapping[child[1] *
                                                       m_uiNumDirections +
                                                   OCT_DIR_LEFT];
                                child[2] =
                                    m_uiE2EMapping[child[3] *
                                                       m_uiNumDirections +
                                                   OCT_DIR_LEFT];
                                child[4] =
                                    m_uiE2EMapping[child[5] *
                                                       m_uiNumDirections +
                                                   OCT_DIR_LEFT];
                                child[6] =
                                    m_uiE2EMapping[child[7] *
                                                       m_uiNumDirections +
                                                   OCT_DIR_LEFT];

                            } else {
                                child[0] =
                                    LOOK_UP_TABLE_DEFAULT;  // m_uiE2EMapping[child[1]*m_uiNumDirections+OCT_DIR_LEFT];
                                child[2] =
                                    LOOK_UP_TABLE_DEFAULT;  // m_uiE2EMapping[child[3]*m_uiNumDirections+OCT_DIR_LEFT];
                                child[4] =
                                    LOOK_UP_TABLE_DEFAULT;  // m_uiE2EMapping[child[5]*m_uiNumDirections+OCT_DIR_LEFT];
                                child[6] =
                                    LOOK_UP_TABLE_DEFAULT;  // m_uiE2EMapping[child[7]*m_uiNumDirections+OCT_DIR_LEFT];
                            }

                            this->child2ParentInjection(
                                zippedVec, interpOrInjectionOut.data(),
                                child.data(), pNodes[lookUp].getLevel());

#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_cpy.start();
#endif
                            for (unsigned int k = 0; k < (m_uiElementOrder + 1);
                                 k++)
                                for (unsigned int j = 0;
                                     j < (m_uiElementOrder + 1); j++)
                                    for (unsigned int i =
                                             (m_uiElementOrder - paddWidth);
                                         i < (m_uiElementOrder + 1); i++)
                                        unzippedVec[offset +
                                                    (ek * m_uiElementOrder + k +
                                                     paddWidth) *
                                                        (ly * lx) +
                                                    (ej * m_uiElementOrder + j +
                                                     paddWidth) *
                                                        (lx) +
                                                    (ei * m_uiElementOrder + i -
                                                     (m_uiElementOrder -
                                                      paddWidth))] =
                                            interpOrInjectionOut
                                                [k * (m_uiElementOrder + 1) *
                                                     (m_uiElementOrder + 1) +
                                                 j * (m_uiElementOrder + 1) +
                                                 i];

#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_cpy.stop();
                            dendro::timer::t_unzip_sync_f_c3.stop();
#endif
                        }
                    }
                }

#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                dendro::timer::t_unzip_sync_face[0].stop();
#endif

#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                dendro::timer::t_unzip_sync_face[1].start();
#endif

                if ((pNodes[elem].maxX() == blkNode.maxX())) {
                    assert(ei == eleIndexMax);
                    lookUp = m_uiE2EMapping[elem * m_uiNumDirections +
                                            OCT_DIR_RIGHT];
                    if (lookUp != LOOK_UP_TABLE_DEFAULT) {
                        if (pNodes[lookUp].getLevel() ==
                            pNodes[elem].getLevel()) {
#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_f_c1.start();
#endif
                            assert(paddWidth < (m_uiElementOrder + 1));

                            T* lookUpElementVec = &ele_dg_vec[lookUp * m_uiNpE];
                            if (!eleVec_valid[lookUp]) {
                                this->getElementNodalValues(
                                    zippedVec, lookUpElementVec, lookUp);
                                eleVec_valid[lookUp] = true;
                            }

#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_cpy.start();
#endif
                            for (unsigned int k = 0; k < (m_uiElementOrder + 1);
                                 k++)
                                for (unsigned int j = 0;
                                     j < (m_uiElementOrder + 1); j++)
                                    for (unsigned int i = 0;
                                         i < (paddWidth + 1); i++)
                                        unzippedVec[offset +
                                                    (ek * m_uiElementOrder + k +
                                                     paddWidth) *
                                                        (ly * lx) +
                                                    (ej * m_uiElementOrder + j +
                                                     paddWidth) *
                                                        (lx) +
                                                    ((ei + 1) *
                                                         m_uiElementOrder +
                                                     paddWidth +
                                                     i)] = lookUpElementVec
                                            [k * (m_uiElementOrder + 1) *
                                                 (m_uiElementOrder + 1) +
                                             j * (m_uiElementOrder + 1) + i];

#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_cpy.stop();
                            dendro::timer::t_unzip_sync_f_c1.stop();
#endif

                        } else if (pNodes[lookUp].getLevel() <
                                   pNodes[elem].getLevel()) {
#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_f_c2.start();
#endif
                            assert(pNodes[lookUp].getLevel() + 1 == regLev);
                            mid_bit =
                                m_uiMaxDepth - pNodes[lookUp].getLevel() - 1;
                            cnum = (((((pNodes[elem].getZ()) >> mid_bit) & 1u)
                                     << 2u) |
                                    ((((pNodes[elem].getY()) >> mid_bit) & 1u)
                                     << 1u) |
                                    ((((pNodes[elem].getX() + sz)) >> mid_bit) &
                                     1u));
                            // std::cout<<"elem: "<<elem<<" :
                            // "<<m_uiAllElements[elem]<<" lookup:
                            // "<<m_uiAllElements[lookUp]<<" child:
                            // "<<ot::TreeNode(pNodes[elem].getX()+sz,pNodes[elem].getY(),pNodes[elem].getZ(),pNodes[elem].getLevel(),m_uiDim,m_uiMaxDepth)<<"
                            // cnum: "<<cnum<<std::endl;

#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_cpy.start();
#endif

#ifdef USE_FD_INTERP_FOR_UNZIP
                            const int st = this->getBlkBdyParentCNums(
                                blk, elem, OCT_DIR_RIGHT, child.data(), fid,
                                cid);
                            if (st > 0) {
                                const unsigned int NUM_CHILDREN_BY2 =
                                    (NUM_CHILDREN >> 1u);
                                this->getBlkBoundaryParentNodes(
                                    zippedVec, lookUpElementVec.data(),
                                    interpolationInput.data(),
                                    interpOrInjectionOut.data(), lookUp, fid,
                                    cid, child.data());
                                for (unsigned int w = 0; w < NUM_CHILDREN_BY2;
                                     w++) {
                                    assert(pNodes[lookUp] ==
                                           pNodes[m_uiE2EMapping
                                                      [child[fid[w]] *
                                                           m_uiNumDirections +
                                                       OCT_DIR_RIGHT]]);
                                    assert(child[fid[w]] !=
                                           LOOK_UP_TABLE_DEFAULT);

                                    if (child[fid[w]] < m_uiElementLocalBegin ||
                                        child[fid[w]] >= m_uiElementLocalEnd)
                                        continue;

                                    this->parent2ChildInterpolation(
                                        lookUpElementVec.data(),
                                        interpOrInjectionOut.data(), cid[w],
                                        m_uiDim);

                                    const ot::Block blk_fd =
                                        m_uiLocalBlockList[m_uiE2BlkMap[(
                                            child[fid[w]] -
                                            m_uiElementLocalBegin)]];
                                    const ot::TreeNode blkNode_fd =
                                        blk_fd.getBlockNode();
                                    const unsigned int regL_fd =
                                        blk_fd.getRegularGridLev();

                                    const unsigned int lx_fd =
                                        blk_fd.getAllocationSzX();
                                    const unsigned int ly_fd =
                                        blk_fd.getAllocationSzY();
                                    const unsigned int lz_fd =
                                        blk_fd.getAllocationSzZ();

                                    const unsigned int offset_fd =
                                        blk_fd.getOffset();

                                    const unsigned int ei_fd =
                                        (pNodes[child[fid[w]]].getX() -
                                         blkNode_fd.getX()) >>
                                        (m_uiMaxDepth - regL_fd);
                                    const unsigned int ej_fd =
                                        (pNodes[child[fid[w]]].getY() -
                                         blkNode_fd.getY()) >>
                                        (m_uiMaxDepth - regL_fd);
                                    const unsigned int ek_fd =
                                        (pNodes[child[fid[w]]].getZ() -
                                         blkNode_fd.getZ()) >>
                                        (m_uiMaxDepth - regL_fd);

                                    for (unsigned int k = 0;
                                         k < (m_uiElementOrder + 1); k++)
                                        for (unsigned int j = 0;
                                             j < (m_uiElementOrder + 1); j++)
                                            for (unsigned int i = 0;
                                                 i < (paddWidth + 1); i++)
                                                unzippedVec
                                                    [offset_fd +
                                                     (ek_fd * m_uiElementOrder +
                                                      k + paddWidth) *
                                                         (ly_fd * lx_fd) +
                                                     (ej_fd * m_uiElementOrder +
                                                      j + paddWidth) *
                                                         (lx_fd) +
                                                     ((ei_fd + 1) *
                                                          m_uiElementOrder +
                                                      paddWidth +
                                                      i)] = interpOrInjectionOut
                                                        [k *
                                                             (m_uiElementOrder +
                                                              1) *
                                                             (m_uiElementOrder +
                                                              1) +
                                                         j * (m_uiElementOrder +
                                                              1) +
                                                         i];
                                }
                            }
#else

                            T* lookUpElementVec = &ele_dg_vec[lookUp * m_uiNpE];
                            if (!eleVec_valid[lookUp]) {
                                this->getElementNodalValues(
                                    zippedVec, lookUpElementVec, lookUp);
                                eleVec_valid[lookUp] = true;
                            }

                            this->parent2ChildInterpolation(
                                lookUpElementVec,
                                &(*(interpOrInjectionOut.begin())), cnum);
                            assert(paddWidth < (m_uiElementOrder + 1));
                            for (unsigned int k = 0; k < (m_uiElementOrder + 1);
                                 k++)
                                for (unsigned int j = 0;
                                     j < (m_uiElementOrder + 1); j++)
                                    for (unsigned int i = 0;
                                         i < (paddWidth + 1); i++)
                                        unzippedVec[offset +
                                                    (ek * m_uiElementOrder + k +
                                                     paddWidth) *
                                                        (ly * lx) +
                                                    (ej * m_uiElementOrder + j +
                                                     paddWidth) *
                                                        (lx) +
                                                    ((ei + 1) *
                                                         m_uiElementOrder +
                                                     paddWidth +
                                                     i)] = interpOrInjectionOut
                                            [k * (m_uiElementOrder + 1) *
                                                 (m_uiElementOrder + 1) +
                                             j * (m_uiElementOrder + 1) + i];
#endif

#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_cpy.stop();
                            dendro::timer::t_unzip_sync_f_c2.stop();
#endif

                        } else if (pNodes[lookUp].getLevel() >
                                   pNodes[elem].getLevel()) {
#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_f_c3.start();
#endif
                            // get the immediate neighbours. These cannot be
                            // LOOK_UP_TABLE_DEFAULT.
                            child[0] = lookUp;
                            child[2] =
                                m_uiE2EMapping[child[0] * m_uiNumDirections +
                                               OCT_DIR_UP];
                            assert(child[2] != LOOK_UP_TABLE_DEFAULT);
                            child[4] =
                                m_uiE2EMapping[child[0] * m_uiNumDirections +
                                               OCT_DIR_FRONT];
                            assert(child[4] != LOOK_UP_TABLE_DEFAULT);
                            child[6] =
                                m_uiE2EMapping[child[2] * m_uiNumDirections +
                                               OCT_DIR_FRONT];
                            assert(child[6] != LOOK_UP_TABLE_DEFAULT);

                            if (m_uiElementOrder == 4 && paddWidth == 3) {
                                child[1] =
                                    m_uiE2EMapping[child[0] *
                                                       m_uiNumDirections +
                                                   OCT_DIR_RIGHT];
                                child[3] =
                                    m_uiE2EMapping[child[2] *
                                                       m_uiNumDirections +
                                                   OCT_DIR_RIGHT];
                                child[5] =
                                    m_uiE2EMapping[child[4] *
                                                       m_uiNumDirections +
                                                   OCT_DIR_RIGHT];
                                child[7] =
                                    m_uiE2EMapping[child[6] *
                                                       m_uiNumDirections +
                                                   OCT_DIR_RIGHT];

                            } else {
                                child[1] =
                                    LOOK_UP_TABLE_DEFAULT;  // m_uiE2EMapping[child[0]*m_uiNumDirections+OCT_DIR_RIGHT];
                                child[3] =
                                    LOOK_UP_TABLE_DEFAULT;  // m_uiE2EMapping[child[2]*m_uiNumDirections+OCT_DIR_RIGHT];
                                child[5] =
                                    LOOK_UP_TABLE_DEFAULT;  // m_uiE2EMapping[child[4]*m_uiNumDirections+OCT_DIR_RIGHT];
                                child[7] =
                                    LOOK_UP_TABLE_DEFAULT;  // m_uiE2EMapping[child[6]*m_uiNumDirections+OCT_DIR_RIGHT];
                            }

                            this->child2ParentInjection(
                                zippedVec, interpOrInjectionOut.data(),
                                child.data(), pNodes[lookUp].getLevel());

#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_cpy.start();
#endif

                            for (unsigned int k = 0; k < (m_uiElementOrder + 1);
                                 k++)
                                for (unsigned int j = 0;
                                     j < (m_uiElementOrder + 1); j++)
                                    for (unsigned int i = 0;
                                         i < (paddWidth + 1); i++)
                                        unzippedVec[offset +
                                                    (ek * m_uiElementOrder + k +
                                                     paddWidth) *
                                                        (ly * lx) +
                                                    (ej * m_uiElementOrder + j +
                                                     paddWidth) *
                                                        (lx) +
                                                    ((ei + 1) *
                                                         m_uiElementOrder +
                                                     paddWidth +
                                                     i)] = interpOrInjectionOut
                                            [k * (m_uiElementOrder + 1) *
                                                 (m_uiElementOrder + 1) +
                                             j * (m_uiElementOrder + 1) + i];

#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_cpy.stop();
                            dendro::timer::t_unzip_sync_f_c3.stop();
#endif
                        }
                    }
                }

#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                dendro::timer::t_unzip_sync_face[1].stop();
#endif
#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                dendro::timer::t_unzip_sync_face[2].start();
#endif

                //---------------------------------------------------------------------------------------------------
                // Y
                // Direction----------------------------------------------------------------------------------
                if ((pNodes[elem].minY() == blkNode.minY())) {
                    assert(ej == 0);

                    lookUp =
                        m_uiE2EMapping[elem * m_uiNumDirections + OCT_DIR_DOWN];
                    if (lookUp != LOOK_UP_TABLE_DEFAULT) {
                        if (pNodes[lookUp].getLevel() ==
                            pNodes[elem].getLevel()) {
#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_f_c1.start();
#endif
                            assert(paddWidth < (m_uiElementOrder + 1));
                            T* lookUpElementVec = &ele_dg_vec[lookUp * m_uiNpE];
                            if (!eleVec_valid[lookUp]) {
                                this->getElementNodalValues(
                                    zippedVec, lookUpElementVec, lookUp);
                                eleVec_valid[lookUp] = true;
                            }

#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_cpy.start();
#endif
                            for (unsigned int k = 0; k < (m_uiElementOrder + 1);
                                 k++)
                                for (unsigned int i = 0;
                                     i < (m_uiElementOrder + 1); i++)
                                    for (unsigned int j =
                                             (m_uiElementOrder - paddWidth);
                                         j < (m_uiElementOrder + 1); j++)
                                        unzippedVec[offset +
                                                    (ek * m_uiElementOrder + k +
                                                     paddWidth) *
                                                        (ly * lx) +
                                                    (ej * m_uiElementOrder + j -
                                                     (m_uiElementOrder -
                                                      paddWidth)) *
                                                        (lx) +
                                                    (ei * m_uiElementOrder + i +
                                                     paddWidth)] =
                                            lookUpElementVec
                                                [k * (m_uiElementOrder + 1) *
                                                     (m_uiElementOrder + 1) +
                                                 j * (m_uiElementOrder + 1) +
                                                 i];

#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_cpy.stop();
                            dendro::timer::t_unzip_sync_f_c1.stop();
#endif

                        } else if (pNodes[lookUp].getLevel() <
                                   pNodes[elem].getLevel()) {
#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_f_c2.start();
#endif
                            assert(pNodes[lookUp].getLevel() + 1 == regLev);
                            mid_bit =
                                m_uiMaxDepth - pNodes[lookUp].getLevel() - 1;
                            cnum =
                                (((((pNodes[elem].getZ()) >> mid_bit) & 1u)
                                  << 2u) |
                                 (((((pNodes[elem].getY() - sz)) >> mid_bit) &
                                   1u)
                                  << 1u) |
                                 (((pNodes[elem].getX()) >> mid_bit) & 1u));

                            // std::cout<<"elem: "<<elem<<" :
                            // "<<m_uiAllElements[elem]<<" lookup:
                            // "<<m_uiAllElements[lookUp]<<" child:
                            // "<<ot::TreeNode(pNodes[elem].getX()-sz,pNodes[elem].getY(),pNodes[elem].getZ(),pNodes[elem].getLevel(),m_uiDim,m_uiMaxDepth)<<"
                            // cnum: "<<cnum<<std::endl;

                            // std::cout<<"m_uiActiveRank :
                            // "<<m_uiActiveRank<<"parent to child interpolation
                            // executed"<<std::endl;
                            assert(paddWidth < (m_uiElementOrder + 1));
#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_cpy.start();
#endif

#ifdef USE_FD_INTERP_FOR_UNZIP
                            const int st = this->getBlkBdyParentCNums(
                                blk, elem, OCT_DIR_DOWN, child.data(), fid,
                                cid);
                            if (st > 0) {
                                const unsigned int NUM_CHILDREN_BY2 =
                                    (NUM_CHILDREN >> 1u);
                                this->getBlkBoundaryParentNodes(
                                    zippedVec, lookUpElementVec.data(),
                                    interpolationInput.data(),
                                    interpOrInjectionOut.data(), lookUp, fid,
                                    cid, child.data());
                                for (unsigned int w = 0; w < NUM_CHILDREN_BY2;
                                     w++) {
                                    assert(pNodes[lookUp] ==
                                           pNodes[m_uiE2EMapping
                                                      [child[fid[w]] *
                                                           m_uiNumDirections +
                                                       OCT_DIR_DOWN]]);
                                    assert(child[fid[w]] !=
                                           LOOK_UP_TABLE_DEFAULT);

                                    if (child[fid[w]] < m_uiElementLocalBegin ||
                                        child[fid[w]] >= m_uiElementLocalEnd)
                                        continue;

                                    this->parent2ChildInterpolation(
                                        lookUpElementVec.data(),
                                        interpOrInjectionOut.data(), cid[w],
                                        m_uiDim);

                                    const ot::Block blk_fd =
                                        m_uiLocalBlockList[m_uiE2BlkMap[(
                                            child[fid[w]] -
                                            m_uiElementLocalBegin)]];
                                    const ot::TreeNode blkNode_fd =
                                        blk_fd.getBlockNode();
                                    const unsigned int regL_fd =
                                        blk_fd.getRegularGridLev();

                                    const unsigned int lx_fd =
                                        blk_fd.getAllocationSzX();
                                    const unsigned int ly_fd =
                                        blk_fd.getAllocationSzY();
                                    const unsigned int lz_fd =
                                        blk_fd.getAllocationSzZ();

                                    const unsigned int offset_fd =
                                        blk_fd.getOffset();

                                    const unsigned int ei_fd =
                                        (pNodes[child[fid[w]]].getX() -
                                         blkNode_fd.getX()) >>
                                        (m_uiMaxDepth - regL_fd);
                                    const unsigned int ej_fd =
                                        (pNodes[child[fid[w]]].getY() -
                                         blkNode_fd.getY()) >>
                                        (m_uiMaxDepth - regL_fd);
                                    const unsigned int ek_fd =
                                        (pNodes[child[fid[w]]].getZ() -
                                         blkNode_fd.getZ()) >>
                                        (m_uiMaxDepth - regL_fd);

                                    for (unsigned int k = 0;
                                         k < (m_uiElementOrder + 1); k++)
                                        for (unsigned int i = 0;
                                             i < (m_uiElementOrder + 1); i++)
                                            for (unsigned int j =
                                                     (m_uiElementOrder -
                                                      paddWidth);
                                                 j < (m_uiElementOrder + 1);
                                                 j++)
                                                unzippedVec
                                                    [offset_fd +
                                                     (ek_fd * m_uiElementOrder +
                                                      k + paddWidth) *
                                                         (ly_fd * lx_fd) +
                                                     (ej_fd * m_uiElementOrder +
                                                      j -
                                                      (m_uiElementOrder -
                                                       paddWidth)) *
                                                         (lx_fd) +
                                                     (ei_fd * m_uiElementOrder +
                                                      i +
                                                      paddWidth)] = interpOrInjectionOut
                                                        [k *
                                                             (m_uiElementOrder +
                                                              1) *
                                                             (m_uiElementOrder +
                                                              1) +
                                                         j * (m_uiElementOrder +
                                                              1) +
                                                         i];
                                }
                            }
#else
                            T* lookUpElementVec = &ele_dg_vec[lookUp * m_uiNpE];
                            if (!eleVec_valid[lookUp]) {
                                this->getElementNodalValues(
                                    zippedVec, lookUpElementVec, lookUp);
                                eleVec_valid[lookUp] = true;
                            }
                            this->parent2ChildInterpolation(
                                lookUpElementVec,
                                &(*(interpOrInjectionOut.begin())), cnum);

                            for (unsigned int k = 0; k < (m_uiElementOrder + 1);
                                 k++)
                                for (unsigned int i = 0;
                                     i < (m_uiElementOrder + 1); i++)
                                    for (unsigned int j =
                                             (m_uiElementOrder - paddWidth);
                                         j < (m_uiElementOrder + 1); j++)
                                        unzippedVec[offset +
                                                    (ek * m_uiElementOrder + k +
                                                     paddWidth) *
                                                        (ly * lx) +
                                                    (ej * m_uiElementOrder + j -
                                                     (m_uiElementOrder -
                                                      paddWidth)) *
                                                        (lx) +
                                                    (ei * m_uiElementOrder + i +
                                                     paddWidth)] =
                                            interpOrInjectionOut
                                                [k * (m_uiElementOrder + 1) *
                                                     (m_uiElementOrder + 1) +
                                                 j * (m_uiElementOrder + 1) +
                                                 i];

#endif

#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_cpy.stop();
                            dendro::timer::t_unzip_sync_f_c2.stop();
#endif

                        } else if (pNodes[lookUp].getLevel() >
                                   pNodes[elem].getLevel()) {
// get the immediate neighbours. These cannot be LOOK_UP_TABLE_DEFAULT.
#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_f_c3.start();
#endif
                            child[2] = lookUp;
                            child[3] =
                                m_uiE2EMapping[child[2] * m_uiNumDirections +
                                               OCT_DIR_RIGHT];
                            assert(child[3] != LOOK_UP_TABLE_DEFAULT);
                            child[6] =
                                m_uiE2EMapping[child[2] * m_uiNumDirections +
                                               OCT_DIR_FRONT];
                            assert(child[6] != LOOK_UP_TABLE_DEFAULT);
                            child[7] =
                                m_uiE2EMapping[child[3] * m_uiNumDirections +
                                               OCT_DIR_FRONT];
                            assert(child[7] != LOOK_UP_TABLE_DEFAULT);

                            if (m_uiElementOrder == 4 && paddWidth == 3) {
                                child[0] =
                                    m_uiE2EMapping[child[2] *
                                                       m_uiNumDirections +
                                                   OCT_DIR_DOWN];
                                child[1] =
                                    m_uiE2EMapping[child[3] *
                                                       m_uiNumDirections +
                                                   OCT_DIR_DOWN];
                                child[4] =
                                    m_uiE2EMapping[child[6] *
                                                       m_uiNumDirections +
                                                   OCT_DIR_DOWN];
                                child[5] =
                                    m_uiE2EMapping[child[7] *
                                                       m_uiNumDirections +
                                                   OCT_DIR_DOWN];

                            } else {
                                child[0] =
                                    LOOK_UP_TABLE_DEFAULT;  // m_uiE2EMapping[child[2]*m_uiNumDirections+OCT_DIR_DOWN];
                                child[1] =
                                    LOOK_UP_TABLE_DEFAULT;  // m_uiE2EMapping[child[3]*m_uiNumDirections+OCT_DIR_DOWN];
                                child[4] =
                                    LOOK_UP_TABLE_DEFAULT;  // m_uiE2EMapping[child[6]*m_uiNumDirections+OCT_DIR_DOWN];
                                child[5] =
                                    LOOK_UP_TABLE_DEFAULT;  // m_uiE2EMapping[child[7]*m_uiNumDirections+OCT_DIR_DOWN];
                            }

                            this->child2ParentInjection(
                                zippedVec, interpOrInjectionOut.data(),
                                child.data(), pNodes[lookUp].getLevel());

// std::cout<<"m_uiActiveRank : "<<m_uiActiveRank<<"child to parent
// interpolation executed"<<std::endl;
#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_cpy.start();
#endif
                            for (unsigned int k = 0; k < (m_uiElementOrder + 1);
                                 k++)
                                for (unsigned int i = 0;
                                     i < (m_uiElementOrder + 1); i++)
                                    for (unsigned int j =
                                             (m_uiElementOrder - paddWidth);
                                         j < (m_uiElementOrder + 1); j++)
                                        unzippedVec[offset +
                                                    (ek * m_uiElementOrder + k +
                                                     paddWidth) *
                                                        (ly * lx) +
                                                    (ej * m_uiElementOrder + j -
                                                     (m_uiElementOrder -
                                                      paddWidth)) *
                                                        (lx) +
                                                    (ei * m_uiElementOrder + i +
                                                     paddWidth)] =
                                            interpOrInjectionOut
                                                [k * (m_uiElementOrder + 1) *
                                                     (m_uiElementOrder + 1) +
                                                 j * (m_uiElementOrder + 1) +
                                                 i];

#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_cpy.stop();
                            dendro::timer::t_unzip_sync_f_c3.stop();
#endif
                        }
                    }
                }

#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                dendro::timer::t_unzip_sync_face[2].stop();
#endif
#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                dendro::timer::t_unzip_sync_face[3].start();
#endif
                if ((pNodes[elem].maxY() == blkNode.maxY())) {
                    assert(ej == (1u << (regLev - blkNode.getLevel())) - 1);
                    lookUp =
                        m_uiE2EMapping[elem * m_uiNumDirections + OCT_DIR_UP];
                    if (lookUp != LOOK_UP_TABLE_DEFAULT) {
                        if (pNodes[lookUp].getLevel() ==
                            pNodes[elem].getLevel()) {
#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_f_c1.start();
#endif
                            assert(paddWidth < (m_uiElementOrder + 1));
                            T* lookUpElementVec = &ele_dg_vec[lookUp * m_uiNpE];
                            if (!eleVec_valid[lookUp]) {
                                this->getElementNodalValues(
                                    zippedVec, lookUpElementVec, lookUp);
                                eleVec_valid[lookUp] = true;
                            }

#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_cpy.start();
#endif
                            for (unsigned int k = 0; k < (m_uiElementOrder + 1);
                                 k++)
                                for (unsigned int i = 0;
                                     i < (m_uiElementOrder + 1); i++)
                                    for (unsigned int j = 0;
                                         j < (paddWidth + 1); j++)
                                        unzippedVec[offset +
                                                    (ek * m_uiElementOrder + k +
                                                     paddWidth) *
                                                        (ly * lx) +
                                                    ((ej + 1) *
                                                         m_uiElementOrder +
                                                     paddWidth + j) *
                                                        (lx) +
                                                    (ei * m_uiElementOrder + i +
                                                     paddWidth)] =
                                            lookUpElementVec
                                                [k * (m_uiElementOrder + 1) *
                                                     (m_uiElementOrder + 1) +
                                                 j * (m_uiElementOrder + 1) +
                                                 i];

#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_cpy.stop();
                            dendro::timer::t_unzip_sync_f_c1.stop();
#endif

                        } else if (pNodes[lookUp].getLevel() <
                                   pNodes[elem].getLevel()) {
#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_f_c2.start();
#endif
                            assert(pNodes[lookUp].getLevel() + 1 == regLev);
                            mid_bit =
                                m_uiMaxDepth - pNodes[lookUp].getLevel() - 1;
                            cnum =
                                (((((pNodes[elem].getZ()) >> mid_bit) & 1u)
                                  << 2u) |
                                 (((((pNodes[elem].getY() + sz)) >> mid_bit) &
                                   1u)
                                  << 1u) |
                                 (((pNodes[elem].getX()) >> mid_bit) & 1u));
                            // std::cout<<"elem: "<<elem<<" :
                            // "<<m_uiAllElements[elem]<<" lookup:
                            // "<<m_uiAllElements[lookUp]<<" child:
                            // "<<ot::TreeNode(pNodes[elem].getX()+sz,pNodes[elem].getY(),pNodes[elem].getZ(),pNodes[elem].getLevel(),m_uiDim,m_uiMaxDepth)<<"
                            // cnum: "<<cnum<<std::endl;

#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_cpy.start();
#endif

#ifdef USE_FD_INTERP_FOR_UNZIP
                            const int st = this->getBlkBdyParentCNums(
                                blk, elem, OCT_DIR_UP, child.data(), fid, cid);
                            if (st > 0) {
                                const unsigned int NUM_CHILDREN_BY2 =
                                    (NUM_CHILDREN >> 1u);
                                this->getBlkBoundaryParentNodes(
                                    zippedVec, lookUpElementVec.data(),
                                    interpolationInput.data(),
                                    interpOrInjectionOut.data(), lookUp, fid,
                                    cid, child.data());
                                for (unsigned int w = 0; w < NUM_CHILDREN_BY2;
                                     w++) {
                                    assert(pNodes[lookUp] ==
                                           pNodes[m_uiE2EMapping
                                                      [child[fid[w]] *
                                                           m_uiNumDirections +
                                                       OCT_DIR_UP]]);
                                    assert(child[fid[w]] !=
                                           LOOK_UP_TABLE_DEFAULT);

                                    if (child[fid[w]] < m_uiElementLocalBegin ||
                                        child[fid[w]] >= m_uiElementLocalEnd)
                                        continue;

                                    this->parent2ChildInterpolation(
                                        lookUpElementVec.data(),
                                        interpOrInjectionOut.data(), cid[w],
                                        m_uiDim);

                                    const ot::Block blk_fd =
                                        m_uiLocalBlockList[m_uiE2BlkMap[(
                                            child[fid[w]] -
                                            m_uiElementLocalBegin)]];
                                    const ot::TreeNode blkNode_fd =
                                        blk_fd.getBlockNode();
                                    const unsigned int regL_fd =
                                        blk_fd.getRegularGridLev();

                                    const unsigned int lx_fd =
                                        blk_fd.getAllocationSzX();
                                    const unsigned int ly_fd =
                                        blk_fd.getAllocationSzY();
                                    const unsigned int lz_fd =
                                        blk_fd.getAllocationSzZ();

                                    const unsigned int offset_fd =
                                        blk_fd.getOffset();

                                    const unsigned int ei_fd =
                                        (pNodes[child[fid[w]]].getX() -
                                         blkNode_fd.getX()) >>
                                        (m_uiMaxDepth - regL_fd);
                                    const unsigned int ej_fd =
                                        (pNodes[child[fid[w]]].getY() -
                                         blkNode_fd.getY()) >>
                                        (m_uiMaxDepth - regL_fd);
                                    const unsigned int ek_fd =
                                        (pNodes[child[fid[w]]].getZ() -
                                         blkNode_fd.getZ()) >>
                                        (m_uiMaxDepth - regL_fd);

                                    for (unsigned int k = 0;
                                         k < (m_uiElementOrder + 1); k++)
                                        for (unsigned int i = 0;
                                             i < (m_uiElementOrder + 1); i++)
                                            for (unsigned int j = 0;
                                                 j < (paddWidth + 1); j++)
                                                unzippedVec
                                                    [offset_fd +
                                                     (ek_fd * m_uiElementOrder +
                                                      k + paddWidth) *
                                                         (ly_fd * lx_fd) +
                                                     ((ej_fd + 1) *
                                                          m_uiElementOrder +
                                                      paddWidth + j) *
                                                         (lx_fd) +
                                                     (ei_fd * m_uiElementOrder +
                                                      i +
                                                      paddWidth)] = interpOrInjectionOut
                                                        [k *
                                                             (m_uiElementOrder +
                                                              1) *
                                                             (m_uiElementOrder +
                                                              1) +
                                                         j * (m_uiElementOrder +
                                                              1) +
                                                         i];
                                }
                            }
#else
                            T* lookUpElementVec = &ele_dg_vec[lookUp * m_uiNpE];
                            if (!eleVec_valid[lookUp]) {
                                this->getElementNodalValues(
                                    zippedVec, lookUpElementVec, lookUp);
                                eleVec_valid[lookUp] = true;
                            }
                            this->parent2ChildInterpolation(
                                lookUpElementVec,
                                &(*(interpOrInjectionOut.begin())), cnum);

                            assert(paddWidth < (m_uiElementOrder + 1));
                            for (unsigned int k = 0; k < (m_uiElementOrder + 1);
                                 k++)
                                for (unsigned int i = 0;
                                     i < (m_uiElementOrder + 1); i++)
                                    for (unsigned int j = 0;
                                         j < (paddWidth + 1); j++)
                                        unzippedVec[offset +
                                                    (ek * m_uiElementOrder + k +
                                                     paddWidth) *
                                                        (ly * lx) +
                                                    ((ej + 1) *
                                                         m_uiElementOrder +
                                                     paddWidth + j) *
                                                        (lx) +
                                                    (ei * m_uiElementOrder + i +
                                                     paddWidth)] =
                                            interpOrInjectionOut
                                                [k * (m_uiElementOrder + 1) *
                                                     (m_uiElementOrder + 1) +
                                                 j * (m_uiElementOrder + 1) +
                                                 i];

#endif

#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_cpy.stop();
                            dendro::timer::t_unzip_sync_f_c2.stop();
#endif

                        } else if (pNodes[lookUp].getLevel() >
                                   pNodes[elem].getLevel()) {
#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_f_c3.start();
#endif
                            // get the immediate neighbours. These cannot be
                            // LOOK_UP_TABLE_DEFAULT.
                            child[0] = lookUp;
                            child[1] =
                                m_uiE2EMapping[child[0] * m_uiNumDirections +
                                               OCT_DIR_RIGHT];
                            assert(child[1] != LOOK_UP_TABLE_DEFAULT);
                            child[4] =
                                m_uiE2EMapping[child[0] * m_uiNumDirections +
                                               OCT_DIR_FRONT];
                            assert(child[4] != LOOK_UP_TABLE_DEFAULT);
                            child[5] =
                                m_uiE2EMapping[child[1] * m_uiNumDirections +
                                               OCT_DIR_FRONT];
                            assert(child[5] != LOOK_UP_TABLE_DEFAULT);

                            if (m_uiElementOrder == 4 && paddWidth == 3) {
                                child[2] =
                                    m_uiE2EMapping[child[0] *
                                                       m_uiNumDirections +
                                                   OCT_DIR_UP];
                                child[3] =
                                    m_uiE2EMapping[child[1] *
                                                       m_uiNumDirections +
                                                   OCT_DIR_UP];
                                child[6] =
                                    m_uiE2EMapping[child[4] *
                                                       m_uiNumDirections +
                                                   OCT_DIR_UP];
                                child[7] =
                                    m_uiE2EMapping[child[5] *
                                                       m_uiNumDirections +
                                                   OCT_DIR_UP];

                            } else {
                                child[2] =
                                    LOOK_UP_TABLE_DEFAULT;  // m_uiE2EMapping[child[0]*m_uiNumDirections+OCT_DIR_UP];
                                child[3] =
                                    LOOK_UP_TABLE_DEFAULT;  // m_uiE2EMapping[child[1]*m_uiNumDirections+OCT_DIR_UP];
                                child[6] =
                                    LOOK_UP_TABLE_DEFAULT;  // m_uiE2EMapping[child[4]*m_uiNumDirections+OCT_DIR_UP];
                                child[7] =
                                    LOOK_UP_TABLE_DEFAULT;  // m_uiE2EMapping[child[5]*m_uiNumDirections+OCT_DIR_UP];
                            }

                            this->child2ParentInjection(
                                zippedVec, interpOrInjectionOut.data(),
                                child.data(), pNodes[lookUp].getLevel());

#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_cpy.start();
#endif
                            for (unsigned int k = 0; k < (m_uiElementOrder + 1);
                                 k++)
                                for (unsigned int i = 0;
                                     i < (m_uiElementOrder + 1); i++)
                                    for (unsigned int j = 0;
                                         j < (paddWidth + 1); j++)
                                        unzippedVec[offset +
                                                    (ek * m_uiElementOrder + k +
                                                     paddWidth) *
                                                        (ly * lx) +
                                                    ((ej + 1) *
                                                         m_uiElementOrder +
                                                     paddWidth + j) *
                                                        (lx) +
                                                    (ei * m_uiElementOrder + i +
                                                     paddWidth)] =
                                            interpOrInjectionOut
                                                [k * (m_uiElementOrder + 1) *
                                                     (m_uiElementOrder + 1) +
                                                 j * (m_uiElementOrder + 1) +
                                                 i];

#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_cpy.stop();
                            dendro::timer::t_unzip_sync_f_c3.stop();
#endif
                        }
                    }
                }

#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                dendro::timer::t_unzip_sync_face[3].stop();
#endif
#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                dendro::timer::t_unzip_sync_face[4].start();
#endif
                //---------------------------------------------------------------------
                // Z direction padding.
                //-------------------------------------------------------------------------------------------------------

                if ((pNodes[elem].minZ() == blkNode.minZ())) {
                    assert(ek == 0);

                    lookUp =
                        m_uiE2EMapping[elem * m_uiNumDirections + OCT_DIR_BACK];
                    if (lookUp != LOOK_UP_TABLE_DEFAULT) {
                        if (pNodes[lookUp].getLevel() ==
                            pNodes[elem].getLevel()) {
#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_f_c1.start();
#endif
                            assert(paddWidth < (m_uiElementOrder + 1));
                            T* lookUpElementVec = &ele_dg_vec[lookUp * m_uiNpE];
                            if (!eleVec_valid[lookUp]) {
                                this->getElementNodalValues(
                                    zippedVec, lookUpElementVec, lookUp);
                                eleVec_valid[lookUp] = true;
                            }

#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_cpy.start();
#endif
                            for (unsigned int j = 0; j < (m_uiElementOrder + 1);
                                 j++)
                                for (unsigned int i = 0;
                                     i < (m_uiElementOrder + 1); i++)
                                    for (unsigned int k =
                                             (m_uiElementOrder - paddWidth);
                                         k < (m_uiElementOrder + 1); k++)
                                        unzippedVec[offset +
                                                    (ek * m_uiElementOrder + k -
                                                     (m_uiElementOrder -
                                                      paddWidth)) *
                                                        (ly * lx) +
                                                    (ej * m_uiElementOrder + j +
                                                     paddWidth) *
                                                        (lx) +
                                                    (ei * m_uiElementOrder + i +
                                                     paddWidth)] =
                                            lookUpElementVec
                                                [k * (m_uiElementOrder + 1) *
                                                     (m_uiElementOrder + 1) +
                                                 j * (m_uiElementOrder + 1) +
                                                 i];

#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_cpy.stop();
                            dendro::timer::t_unzip_sync_f_c1.stop();
#endif

                        } else if (pNodes[lookUp].getLevel() <
                                   pNodes[elem].getLevel()) {
#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_f_c2.start();
#endif
                            assert(pNodes[lookUp].getLevel() + 1 == regLev);
                            mid_bit =
                                m_uiMaxDepth - pNodes[lookUp].getLevel() - 1;
                            cnum =
                                ((((((pNodes[elem].getZ() - sz)) >> mid_bit) &
                                   1u)
                                  << 2u) |
                                 ((((pNodes[elem].getY()) >> mid_bit) & 1u)
                                  << 1u) |
                                 (((pNodes[elem].getX()) >> mid_bit) & 1u));
                            // std::cout<<"elem: "<<elem<<" :
                            // "<<m_uiAllElements[elem]<<" lookup:
                            // "<<m_uiAllElements[lookUp]<<" child:
                            // "<<ot::TreeNode(pNodes[elem].getX()-sz,pNodes[elem].getY(),pNodes[elem].getZ(),pNodes[elem].getLevel(),m_uiDim,m_uiMaxDepth)<<"
                            // cnum: "<<cnum<<std::endl;

                            // std::cout<<"m_uiActiveRank :
                            // "<<m_uiActiveRank<<"parent to child interpolation
                            // executed"<<std::endl;
                            assert(paddWidth < (m_uiElementOrder + 1));
#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_cpy.start();
#endif

#ifdef USE_FD_INTERP_FOR_UNZIP
                            const int st = this->getBlkBdyParentCNums(
                                blk, elem, OCT_DIR_BACK, child.data(), fid,
                                cid);
                            if (st > 0) {
                                const unsigned int NUM_CHILDREN_BY2 =
                                    (NUM_CHILDREN >> 1u);
                                this->getBlkBoundaryParentNodes(
                                    zippedVec, lookUpElementVec.data(),
                                    interpolationInput.data(),
                                    interpOrInjectionOut.data(), lookUp, fid,
                                    cid, child.data());
                                for (unsigned int w = 0; w < NUM_CHILDREN_BY2;
                                     w++) {
                                    assert(pNodes[lookUp] ==
                                           pNodes[m_uiE2EMapping
                                                      [child[fid[w]] *
                                                           m_uiNumDirections +
                                                       OCT_DIR_BACK]]);
                                    assert(child[fid[w]] !=
                                           LOOK_UP_TABLE_DEFAULT);

                                    if (child[fid[w]] < m_uiElementLocalBegin ||
                                        child[fid[w]] >= m_uiElementLocalEnd)
                                        continue;

                                    this->parent2ChildInterpolation(
                                        lookUpElementVec.data(),
                                        interpOrInjectionOut.data(), cid[w],
                                        m_uiDim);
                                    const ot::Block blk_fd =
                                        m_uiLocalBlockList[m_uiE2BlkMap[(
                                            child[fid[w]] -
                                            m_uiElementLocalBegin)]];
                                    const ot::TreeNode blkNode_fd =
                                        blk_fd.getBlockNode();
                                    const unsigned int regL_fd =
                                        blk_fd.getRegularGridLev();

                                    const unsigned int lx_fd =
                                        blk_fd.getAllocationSzX();
                                    const unsigned int ly_fd =
                                        blk_fd.getAllocationSzY();
                                    const unsigned int lz_fd =
                                        blk_fd.getAllocationSzZ();

                                    const unsigned int offset_fd =
                                        blk_fd.getOffset();

                                    const unsigned int ei_fd =
                                        (pNodes[child[fid[w]]].getX() -
                                         blkNode_fd.getX()) >>
                                        (m_uiMaxDepth - regL_fd);
                                    const unsigned int ej_fd =
                                        (pNodes[child[fid[w]]].getY() -
                                         blkNode_fd.getY()) >>
                                        (m_uiMaxDepth - regL_fd);
                                    const unsigned int ek_fd =
                                        (pNodes[child[fid[w]]].getZ() -
                                         blkNode_fd.getZ()) >>
                                        (m_uiMaxDepth - regL_fd);

                                    for (unsigned int j = 0;
                                         j < (m_uiElementOrder + 1); j++)
                                        for (unsigned int i = 0;
                                             i < (m_uiElementOrder + 1); i++)
                                            for (unsigned int k =
                                                     (m_uiElementOrder -
                                                      paddWidth);
                                                 k < (m_uiElementOrder + 1);
                                                 k++)
                                                unzippedVec
                                                    [offset_fd +
                                                     (ek_fd * m_uiElementOrder +
                                                      k -
                                                      (m_uiElementOrder -
                                                       paddWidth)) *
                                                         (ly_fd * lx_fd) +
                                                     (ej_fd * m_uiElementOrder +
                                                      j + paddWidth) *
                                                         (lx_fd) +
                                                     (ei_fd * m_uiElementOrder +
                                                      i +
                                                      paddWidth)] = interpOrInjectionOut
                                                        [k *
                                                             (m_uiElementOrder +
                                                              1) *
                                                             (m_uiElementOrder +
                                                              1) +
                                                         j * (m_uiElementOrder +
                                                              1) +
                                                         i];
                                }
                            }
#else
                            T* lookUpElementVec = &ele_dg_vec[lookUp * m_uiNpE];
                            if (!eleVec_valid[lookUp]) {
                                this->getElementNodalValues(
                                    zippedVec, lookUpElementVec, lookUp);
                                eleVec_valid[lookUp] = true;
                            }
                            this->parent2ChildInterpolation(
                                lookUpElementVec,
                                &(*(interpOrInjectionOut.begin())), cnum);

                            for (unsigned int j = 0; j < (m_uiElementOrder + 1);
                                 j++)
                                for (unsigned int i = 0;
                                     i < (m_uiElementOrder + 1); i++)
                                    for (unsigned int k =
                                             (m_uiElementOrder - paddWidth);
                                         k < (m_uiElementOrder + 1); k++)
                                        unzippedVec[offset +
                                                    (ek * m_uiElementOrder + k -
                                                     (m_uiElementOrder -
                                                      paddWidth)) *
                                                        (ly * lx) +
                                                    (ej * m_uiElementOrder + j +
                                                     paddWidth) *
                                                        (lx) +
                                                    (ei * m_uiElementOrder + i +
                                                     paddWidth)] =
                                            interpOrInjectionOut
                                                [k * (m_uiElementOrder + 1) *
                                                     (m_uiElementOrder + 1) +
                                                 j * (m_uiElementOrder + 1) +
                                                 i];

#endif

#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_cpy.stop();
                            dendro::timer::t_unzip_sync_f_c2.stop();
#endif

                        } else if (pNodes[lookUp].getLevel() >
                                   pNodes[elem].getLevel()) {
#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_f_c3.start();
#endif
                            // get the immediate neighbours. These cannot be
                            // LOOK_UP_TABLE_DEFAULT.
                            child[4] = lookUp;
                            child[5] =
                                m_uiE2EMapping[child[4] * m_uiNumDirections +
                                               OCT_DIR_RIGHT];
                            assert(child[5] != LOOK_UP_TABLE_DEFAULT);
                            child[6] =
                                m_uiE2EMapping[child[4] * m_uiNumDirections +
                                               OCT_DIR_UP];
                            assert(child[6] != LOOK_UP_TABLE_DEFAULT);
                            child[7] =
                                m_uiE2EMapping[child[5] * m_uiNumDirections +
                                               OCT_DIR_UP];
                            assert(child[7] != LOOK_UP_TABLE_DEFAULT);

                            if (m_uiElementOrder == 4 && paddWidth == 3) {
                                child[0] =
                                    m_uiE2EMapping[child[4] *
                                                       m_uiNumDirections +
                                                   OCT_DIR_BACK];
                                child[1] =
                                    m_uiE2EMapping[child[5] *
                                                       m_uiNumDirections +
                                                   OCT_DIR_BACK];
                                child[2] =
                                    m_uiE2EMapping[child[6] *
                                                       m_uiNumDirections +
                                                   OCT_DIR_BACK];
                                child[3] =
                                    m_uiE2EMapping[child[7] *
                                                       m_uiNumDirections +
                                                   OCT_DIR_BACK];

                            } else {
                                child[0] =
                                    LOOK_UP_TABLE_DEFAULT;  // m_uiE2EMapping[child[4]*m_uiNumDirections+OCT_DIR_BACK];
                                child[1] =
                                    LOOK_UP_TABLE_DEFAULT;  // m_uiE2EMapping[child[5]*m_uiNumDirections+OCT_DIR_BACK];
                                child[2] =
                                    LOOK_UP_TABLE_DEFAULT;  // m_uiE2EMapping[child[6]*m_uiNumDirections+OCT_DIR_BACK];
                                child[3] =
                                    LOOK_UP_TABLE_DEFAULT;  // m_uiE2EMapping[child[7]*m_uiNumDirections+OCT_DIR_BACK];
                            }

                            this->child2ParentInjection(
                                zippedVec, interpOrInjectionOut.data(),
                                child.data(), pNodes[lookUp].getLevel());

// std::cout<<"m_uiActiveRank : "<<m_uiActiveRank<<"child to parent
// interpolation executed"<<std::endl;
#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_cpy.start();
#endif
                            for (unsigned int j = 0; j < (m_uiElementOrder + 1);
                                 j++)
                                for (unsigned int i = 0;
                                     i < (m_uiElementOrder + 1); i++)
                                    for (unsigned int k =
                                             (m_uiElementOrder - paddWidth);
                                         k < (m_uiElementOrder + 1); k++)
                                        unzippedVec[offset +
                                                    (ek * m_uiElementOrder + k -
                                                     (m_uiElementOrder -
                                                      paddWidth)) *
                                                        (ly * lx) +
                                                    (ej * m_uiElementOrder + j +
                                                     paddWidth) *
                                                        (lx) +
                                                    (ei * m_uiElementOrder + i +
                                                     paddWidth)] =
                                            interpOrInjectionOut
                                                [k * (m_uiElementOrder + 1) *
                                                     (m_uiElementOrder + 1) +
                                                 j * (m_uiElementOrder + 1) +
                                                 i];
#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_cpy.stop();
                            dendro::timer::t_unzip_sync_f_c3.stop();
#endif
                        }
                    }
                }

#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                dendro::timer::t_unzip_sync_face[4].stop();
#endif
#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                dendro::timer::t_unzip_sync_face[5].start();
#endif

                if ((pNodes[elem].maxZ() == blkNode.maxZ())) {
                    assert(ek == (1u << (regLev - blkNode.getLevel())) - 1);
                    lookUp = m_uiE2EMapping[elem * m_uiNumDirections +
                                            OCT_DIR_FRONT];
                    if (lookUp != LOOK_UP_TABLE_DEFAULT) {
                        if (pNodes[lookUp].getLevel() ==
                            pNodes[elem].getLevel()) {
#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_f_c1.start();
#endif
                            assert(paddWidth < (m_uiElementOrder + 1));
                            T* lookUpElementVec = &ele_dg_vec[lookUp * m_uiNpE];
                            if (!eleVec_valid[lookUp]) {
                                this->getElementNodalValues(
                                    zippedVec, lookUpElementVec, lookUp);
                                eleVec_valid[lookUp] = true;
                            }

#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_cpy.start();
#endif
                            for (unsigned int j = 0; j < (m_uiElementOrder + 1);
                                 j++)
                                for (unsigned int i = 0;
                                     i < (m_uiElementOrder + 1); i++)
                                    for (unsigned int k = 0;
                                         k < (paddWidth + 1); k++)
                                        unzippedVec[offset +
                                                    ((ek + 1) *
                                                         m_uiElementOrder +
                                                     paddWidth + k) *
                                                        (ly * lx) +
                                                    (ej * m_uiElementOrder + j +
                                                     paddWidth) *
                                                        (lx) +
                                                    (ei * m_uiElementOrder + i +
                                                     paddWidth)] =
                                            lookUpElementVec
                                                [k * (m_uiElementOrder + 1) *
                                                     (m_uiElementOrder + 1) +
                                                 j * (m_uiElementOrder + 1) +
                                                 i];

#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_cpy.stop();
                            dendro::timer::t_unzip_sync_f_c1.stop();
#endif

                        } else if (pNodes[lookUp].getLevel() <
                                   pNodes[elem].getLevel()) {
                            assert(pNodes[lookUp].getLevel() + 1 == regLev);
#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_f_c2.start();
#endif
                            mid_bit =
                                m_uiMaxDepth - pNodes[lookUp].getLevel() - 1;
                            cnum =
                                ((((((pNodes[elem].getZ() + sz)) >> mid_bit) &
                                   1u)
                                  << 2u) |
                                 ((((pNodes[elem].getY()) >> mid_bit) & 1u)
                                  << 1u) |
                                 (((pNodes[elem].getX()) >> mid_bit) & 1u));
                            // std::cout<<"elem: "<<elem<<" :
                            // "<<m_uiAllElements[elem]<<" lookup:
                            // "<<m_uiAllElements[lookUp]<<" child:
                            // "<<ot::TreeNode(pNodes[elem].getX()+sz,pNodes[elem].getY(),pNodes[elem].getZ(),pNodes[elem].getLevel(),m_uiDim,m_uiMaxDepth)<<"
                            // cnum: "<<cnum<<std::endl;

#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_cpy.start();
#endif

#ifdef USE_FD_INTERP_FOR_UNZIP
                            const int st = this->getBlkBdyParentCNums(
                                blk, elem, OCT_DIR_FRONT, child.data(), fid,
                                cid);
                            if (st > 0) {
                                const unsigned int NUM_CHILDREN_BY2 =
                                    (NUM_CHILDREN >> 1u);
                                this->getBlkBoundaryParentNodes(
                                    zippedVec, lookUpElementVec.data(),
                                    interpolationInput.data(),
                                    interpOrInjectionOut.data(), lookUp, fid,
                                    cid, child.data());
                                for (unsigned int w = 0; w < NUM_CHILDREN_BY2;
                                     w++) {
                                    assert(pNodes[lookUp] ==
                                           pNodes[m_uiE2EMapping
                                                      [child[fid[w]] *
                                                           m_uiNumDirections +
                                                       OCT_DIR_FRONT]]);
                                    assert(child[fid[w]] !=
                                           LOOK_UP_TABLE_DEFAULT);

                                    if (child[fid[w]] < m_uiElementLocalBegin ||
                                        child[fid[w]] >= m_uiElementLocalEnd)
                                        continue;

                                    this->parent2ChildInterpolation(
                                        lookUpElementVec.data(),
                                        interpOrInjectionOut.data(), cid[w],
                                        m_uiDim);

                                    const ot::Block blk_fd =
                                        m_uiLocalBlockList[m_uiE2BlkMap[(
                                            child[fid[w]] -
                                            m_uiElementLocalBegin)]];
                                    const ot::TreeNode blkNode_fd =
                                        blk_fd.getBlockNode();
                                    const unsigned int regL_fd =
                                        blk_fd.getRegularGridLev();

                                    const unsigned int lx_fd =
                                        blk_fd.getAllocationSzX();
                                    const unsigned int ly_fd =
                                        blk_fd.getAllocationSzY();
                                    const unsigned int lz_fd =
                                        blk_fd.getAllocationSzZ();

                                    const unsigned int offset_fd =
                                        blk_fd.getOffset();

                                    const unsigned int ei_fd =
                                        (pNodes[child[fid[w]]].getX() -
                                         blkNode_fd.getX()) >>
                                        (m_uiMaxDepth - regL_fd);
                                    const unsigned int ej_fd =
                                        (pNodes[child[fid[w]]].getY() -
                                         blkNode_fd.getY()) >>
                                        (m_uiMaxDepth - regL_fd);
                                    const unsigned int ek_fd =
                                        (pNodes[child[fid[w]]].getZ() -
                                         blkNode_fd.getZ()) >>
                                        (m_uiMaxDepth - regL_fd);

                                    for (unsigned int j = 0;
                                         j < (m_uiElementOrder + 1); j++)
                                        for (unsigned int i = 0;
                                             i < (m_uiElementOrder + 1); i++)
                                            for (unsigned int k = 0;
                                                 k < (paddWidth + 1); k++)
                                                unzippedVec
                                                    [offset_fd +
                                                     ((ek_fd + 1) *
                                                          m_uiElementOrder +
                                                      paddWidth + k) *
                                                         (ly_fd * lx_fd) +
                                                     (ej_fd * m_uiElementOrder +
                                                      j + paddWidth) *
                                                         (lx_fd) +
                                                     (ei_fd * m_uiElementOrder +
                                                      i +
                                                      paddWidth)] = interpOrInjectionOut
                                                        [k *
                                                             (m_uiElementOrder +
                                                              1) *
                                                             (m_uiElementOrder +
                                                              1) +
                                                         j * (m_uiElementOrder +
                                                              1) +
                                                         i];
                                }
                            }
#else
                            T* lookUpElementVec = &ele_dg_vec[lookUp * m_uiNpE];
                            if (!eleVec_valid[lookUp]) {
                                this->getElementNodalValues(
                                    zippedVec, lookUpElementVec, lookUp);
                                eleVec_valid[lookUp] = true;
                            }
                            this->parent2ChildInterpolation(
                                lookUpElementVec,
                                &(*(interpOrInjectionOut.begin())), cnum);
                            assert(paddWidth < (m_uiElementOrder + 1));

                            for (unsigned int j = 0; j < (m_uiElementOrder + 1);
                                 j++)
                                for (unsigned int i = 0;
                                     i < (m_uiElementOrder + 1); i++)
                                    for (unsigned int k = 0;
                                         k < (paddWidth + 1); k++)
                                        unzippedVec[offset +
                                                    ((ek + 1) *
                                                         m_uiElementOrder +
                                                     paddWidth + k) *
                                                        (ly * lx) +
                                                    (ej * m_uiElementOrder + j +
                                                     paddWidth) *
                                                        (lx) +
                                                    (ei * m_uiElementOrder + i +
                                                     paddWidth)] =
                                            interpOrInjectionOut
                                                [k * (m_uiElementOrder + 1) *
                                                     (m_uiElementOrder + 1) +
                                                 j * (m_uiElementOrder + 1) +
                                                 i];

#endif

#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_cpy.stop();
                            dendro::timer::t_unzip_sync_f_c2.stop();
#endif

                        } else if (pNodes[lookUp].getLevel() >
                                   pNodes[elem].getLevel()) {
// get the immediate neighbours. These cannot be LOOK_UP_TABLE_DEFAULT.
#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_f_c3.start();
#endif
                            child[0] = lookUp;
                            child[1] =
                                m_uiE2EMapping[child[0] * m_uiNumDirections +
                                               OCT_DIR_RIGHT];
                            assert(child[1] != LOOK_UP_TABLE_DEFAULT);
                            child[2] =
                                m_uiE2EMapping[child[0] * m_uiNumDirections +
                                               OCT_DIR_UP];
                            assert(child[2] != LOOK_UP_TABLE_DEFAULT);
                            child[3] =
                                m_uiE2EMapping[child[1] * m_uiNumDirections +
                                               OCT_DIR_UP];
                            assert(child[3] != LOOK_UP_TABLE_DEFAULT);

                            if (m_uiElementOrder == 4 && paddWidth == 3) {
                                child[4] =
                                    m_uiE2EMapping[child[0] *
                                                       m_uiNumDirections +
                                                   OCT_DIR_FRONT];
                                child[5] =
                                    m_uiE2EMapping[child[1] *
                                                       m_uiNumDirections +
                                                   OCT_DIR_FRONT];
                                child[6] =
                                    m_uiE2EMapping[child[2] *
                                                       m_uiNumDirections +
                                                   OCT_DIR_FRONT];
                                child[7] =
                                    m_uiE2EMapping[child[3] *
                                                       m_uiNumDirections +
                                                   OCT_DIR_FRONT];

                            } else {
                                child[4] =
                                    LOOK_UP_TABLE_DEFAULT;  // m_uiE2EMapping[child[0]*m_uiNumDirections+OCT_DIR_FRONT];
                                child[5] =
                                    LOOK_UP_TABLE_DEFAULT;  // m_uiE2EMapping[child[1]*m_uiNumDirections+OCT_DIR_FRONT];
                                child[6] =
                                    LOOK_UP_TABLE_DEFAULT;  // m_uiE2EMapping[child[2]*m_uiNumDirections+OCT_DIR_FRONT];
                                child[7] =
                                    LOOK_UP_TABLE_DEFAULT;  // m_uiE2EMapping[child[3]*m_uiNumDirections+OCT_DIR_FRONT];
                            }

                            this->child2ParentInjection(
                                zippedVec, interpOrInjectionOut.data(),
                                child.data(), pNodes[lookUp].getLevel());

#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_cpy.start();
#endif
                            for (unsigned int j = 0; j < (m_uiElementOrder + 1);
                                 j++)
                                for (unsigned int i = 0;
                                     i < (m_uiElementOrder + 1); i++)
                                    for (unsigned int k = 0;
                                         k < (paddWidth + 1); k++)
                                        unzippedVec[offset +
                                                    ((ek + 1) *
                                                         m_uiElementOrder +
                                                     paddWidth + k) *
                                                        (ly * lx) +
                                                    (ej * m_uiElementOrder + j +
                                                     paddWidth) *
                                                        (lx) +
                                                    (ei * m_uiElementOrder + i +
                                                     paddWidth)] =
                                            interpOrInjectionOut
                                                [k * (m_uiElementOrder + 1) *
                                                     (m_uiElementOrder + 1) +
                                                 j * (m_uiElementOrder + 1) +
                                                 i];

#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                            dendro::timer::t_unzip_sync_cpy.stop();
                            dendro::timer::t_unzip_sync_f_c3.stop();
#endif
                        }
                    }
                }

#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
                dendro::timer::t_unzip_sync_face[5].stop();
#endif
            }

#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
            dendro::timer::t_unzip_sync_edge.start();
#endif
            blockDiagonalUnZip(m_uiLocalBlockList[blk], zippedVec, unzippedVec,
                               ele_dg_vec.data(), eleVec_valid);

#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
            dendro::timer::t_unzip_sync_edge.stop();
#endif

#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
            dendro::timer::t_unzip_sync_vtex.start();
#endif
            blockVertexUnZip(m_uiLocalBlockList[blk], zippedVec, unzippedVec,
                             ele_dg_vec.data(), eleVec_valid);

#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
            dendro::timer::t_unzip_sync_vtex.stop();
#endif
        }
    }

    delete[] eleVec_valid;
}

template <typename T>
void Mesh::unzip_scatter(const T* in, T* out, unsigned int dof,
                         int blk_filter) {
    if (!m_uiIsActive) return;

#ifdef DENDRO_WIDE_PROLONGATION
    // new DG arrays this call: retire every wpx child-memo entry
    m_uiWpxMemoEpoch = wpxMemoClock();
#endif

    const ot::TreeNode* pNodes = m_uiAllElements.data();
    const ot::Block* blkList   = m_uiLocalBlockList.data();
    const unsigned int eOrder  = m_uiElementOrder;
    const unsigned int nPe     = m_uiNpE;

    const unsigned int cgSz    = this->getDegOfFreedom();
    const unsigned int unSz    = this->getDegOfFreedomUnZip();

    const unsigned int* e2n    = this->getE2NMapping().data();
    const unsigned int* e2e    = this->getE2EMapping().data();

    const unsigned int dgSz    = nPe;

    T* uzWVec                  = out;

    const double d_compar_tol  = 1e-10;

#if defined(DENDRO_UNZIP_OMP)
    // OpenMP block-parallel path: parallelize over blocks (disjoint uzWVec
    // output regions, so race-free even at wavelet boundaries). b2e CSR map is
    // cached on the mesh (buildE2BlockMap); alias it to keep the loop unchanged.
    const size_t n_blocks                       = m_uiLocalBlockList.size();
    const std::vector<unsigned int>& b2e_offset = m_b2e_unzip_offset;
    const std::vector<unsigned int>& b2e_map    = m_b2e_unzip_map;

    // Pre-compute DG values for every element ONCE, instead of redoing
    // getElementNodalValues per (ele, blk) pair (which would be ~8x more
    // calls for typical meshes). Memory: m_uiNumTotalElements * dof * dgSz.
    //
    // Threaded: each element writes only its own disjoint all_dg slice, and we
    // use the thread-safe getElementNodalValues overload with per-thread scratch
    // (im1_pre/im2_pre) so the hanging-face parent2ChildInterpolation no longer
    // races on RefElement's shared im_vec1/im_vec2. This precompute used to be
    // pinned serial -- an Amdahl bottleneck that dominated the threaded unzip on
    // large meshes (its cost scales with element count, not block count, so it
    // capped unzip thread-scaling). Element work is uneven (only hanging-face
    // elements interpolate), hence the dynamic schedule.
    std::vector<T> all_dg((std::size_t)m_uiNumTotalElements * dof * dgSz);
#ifdef DENDRO_WIDE_PROLONGATION
    // Pass 0 of two. A widened hanging face needs the FACE OWNER's
    // neighbourhood, which is two hops away and so out of reach for any
    // element near a partition edge -- including LOCAL ones, whose face owner
    // may itself be a ghost. That is a genuine circular dependency: this
    // rank cannot finish such an element before the exchange, and the
    // exchange has to carry it.
    //
    // Breaking it costs one extra materialisation. Pass 0 builds every local
    // element with hanging faces NARROW, which needs round-1 data only and so
    // always succeeds; after the exchange every ghost slice holds a finished
    // narrow value, and pass 1 redoes the faces wide out of that array.
    //
    // The values are unchanged from the single-pass np=1 path because the
    // widening always DID read narrow neighbours -- gatherExtendedCoarseNodesCG
    // re-materialises them with allowWide=false. Pass 0 just precomputes what
    // that gather used to recompute per call.
    std::vector<T> dg_narrow((std::size_t)m_uiNumTotalElements * dof * dgSz);
#endif
#pragma omp parallel
    {
        std::vector<double> im1_pre(nPe), im2_pre(nPe);
        double* const im1_p = im1_pre.data();
        double* const im2_p = im2_pre.data();
#pragma omp for schedule(dynamic, 16)
        for (unsigned int ele = 0; ele < m_uiNumTotalElements; ele++) {
#ifdef DENDRO_WIDE_PROLONGATION
            // Ghost slices come from the rank that owns them. Locals are ALL
            // materialised -- no e2b or block filter -- because another rank
            // may request any of them and exchangeWideProlongDG packs
            // straight out of this array.
            if (ele < m_uiElementLocalBegin || ele >= m_uiElementLocalEnd)
                continue;
            T* base0 = dg_narrow.data() + (std::size_t)ele * dof * dgSz;
            for (unsigned int v = 0; v < dof; v++)
                this->getElementNodalValues(in + v * cgSz, base0 + v * dgSz,
                                            ele, false, im1_p, im2_p,
                                            /*allowWide=*/false);
            continue;
#else
            if (m_e2b_unzip_counts[ele] == 0) continue;
            if (blk_filter >= 0) {
                // Overlap path: precompute DG only for elements that feed a
                // filter-matching block (interior or boundary). Equivalent to
                // the old block-iterating dedup, but expressed per-element so it
                // parallelizes without a shared dg_done array. Interior blocks
                // read only local elements, so their DG values are valid even
                // before the ghost exchange completes.
                bool feeds            = false;
                const unsigned int eo = m_e2b_unzip_offset[ele];
                for (unsigned int i = 0; i < m_e2b_unzip_counts[ele]; i++)
                    if ((int)blkList[m_e2b_unzip_map[eo + i]].getBlockType() ==
                        blk_filter) {
                        feeds = true;
                        break;
                    }
                if (!feeds) continue;
            }
#endif
            T* base = all_dg.data() + (std::size_t)ele * dof * dgSz;
            for (unsigned int v = 0; v < dof; v++)
                this->getElementNodalValues(in + v * cgSz, base + v * dgSz, ele,
                                            false, im1_p, im2_p);
        }
    }
#ifdef DENDRO_WIDE_PROLONGATION
    this->exchangeWideProlongDG(in, cgSz, dg_narrow.data(),
                                (size_t)dof * dgSz, dof, dgSz);

    // Pass 1: same elements, faces widened out of the now-complete narrow
    // array. It must not read what it writes, or the answer would depend on
    // element visit order, hence the separate destination.
#pragma omp parallel
    {
        std::vector<double> im1_w(nPe), im2_w(nPe);
#pragma omp for schedule(dynamic, 16)
        for (unsigned int ele = m_uiElementLocalBegin;
             ele < m_uiElementLocalEnd; ele++) {
            T* base = all_dg.data() + (std::size_t)ele * dof * dgSz;
            for (unsigned int v = 0; v < dof; v++)
                this->getElementNodalValues(
                    in + v * cgSz, base + v * dgSz, ele, false, im1_w.data(),
                    im2_w.data(), /*allowWide=*/true,
                    dg_narrow.data() + (std::size_t)v * dgSz,
                    (size_t)dof * dgSz);
        }
    }
    this->exchangeWideProlongDG(in, cgSz, all_dg.data(),
                                (size_t)dof * dgSz, dof, dgSz);

    // Diagnostic: exact-input mode. Overwrite every DG value -- local and
    // ghost, hanging nodes included -- with the analytic field, so the pad
    // fill below runs on perfect inputs. See m_uiWpxAnalyticDebug.
    if (m_uiWpxAnalyticDebug) {
        // DENDRO_WPX_EXACT_CLASS limits the overwrite to one node class so
        // the input error can be attributed: "face" = at most one extreme
        // coordinate, "edge" = two or more (element edge or vertex).
        // Unset = every node. Non-hanging nodes are exact already, so
        // overwriting them is a no-op either way.
        const char* cls_dbg = std::getenv("DENDRO_WPX_EXACT_CLASS");
        const int want_cls =
            (!cls_dbg) ? -1 : ((cls_dbg[0] == 'f') ? 0 : 1);
        const unsigned int eo_dbg = m_uiElementOrder;
        for (unsigned int e_dbg = 0; e_dbg < m_uiNumTotalElements; e_dbg++) {
            const ot::TreeNode& tn_dbg = m_uiAllElements[e_dbg];
            const double h_dbg =
                (double)(1u << (m_uiMaxDepth - tn_dbg.getLevel())) /
                (double)eo_dbg;
            for (unsigned int kk = 0; kk <= eo_dbg; kk++)
                for (unsigned int jj = 0; jj <= eo_dbg; jj++)
                    for (unsigned int ii = 0; ii <= eo_dbg; ii++) {
                        const int next_dbg =
                            (ii == 0 || ii == eo_dbg ? 1 : 0) +
                            (jj == 0 || jj == eo_dbg ? 1 : 0) +
                            (kk == 0 || kk == eo_dbg ? 1 : 0);
                        if (want_cls == 0 && next_dbg >= 2) continue;
                        if (want_cls == 1 && next_dbg < 2) continue;
                        const double val_dbg = m_uiWpxAnalyticDebug(
                            (double)tn_dbg.getX() + ii * h_dbg,
                            (double)tn_dbg.getY() + jj * h_dbg,
                            (double)tn_dbg.getZ() + kk * h_dbg);
                        const unsigned int n_dbg =
                            (kk * (eo_dbg + 1) + jj) * (eo_dbg + 1) + ii;
                        for (unsigned int v_dbg = 0; v_dbg < dof; v_dbg++) {
                            const std::size_t at_dbg =
                                (std::size_t)e_dbg * dof * dgSz +
                                (std::size_t)v_dbg * dgSz + n_dbg;
                            all_dg[at_dbg] = (T)val_dbg;
                            dg_narrow[at_dbg] = (T)val_dbg;
                        }
                    }
        }
    }
#endif

#pragma omp parallel
    {
        std::vector<T> p2cI_all_tls(NUM_CHILDREN * dof * nPe);
        std::vector<double> im1_tls(nPe), im2_tls(nPe);
        std::vector<ot::TreeNode> childOct_tls;
        childOct_tls.reserve(NUM_CHILDREN);
        bool p2c_interp_valid_tls[NUM_CHILDREN];
        unsigned int last_ele_tls = UINT_MAX;

        T* p2cI_base_t            = p2cI_all_tls.data();
        double* const im1_t       = im1_tls.data();
        double* const im2_t       = im2_tls.data();

#pragma omp for schedule(dynamic, 1)
        for (size_t blk_idx = 0; blk_idx < n_blocks; blk_idx++) {
            const unsigned int blk = (unsigned int)blk_idx;
            // Overlap path: process only interior or only boundary blocks.
            if (blk_filter >= 0 &&
                (int)blkList[blk].getBlockType() != blk_filter)
                continue;
            const ot::TreeNode blkNode = blkList[blk].getBlockNode();
            const unsigned int PW      = blkList[blk].get1DPadWidth();
            const unsigned int lx      = blkList[blk].getAllocationSzX();
            const unsigned int ly      = blkList[blk].getAllocationSzY();
            const unsigned int lz      = blkList[blk].getAllocationSzZ();
            const unsigned int offset  = blkList[blk].getOffset();
            const unsigned int bLev =
                pNodes[blkList[blk].getLocalElementBegin()].getLevel();

            const unsigned int e_start = b2e_offset[blk];
            const unsigned int e_end   = b2e_offset[blk + 1];

            for (unsigned int idx = e_start; idx < e_end; idx++) {
                const unsigned int ele = b2e_map[idx];
                // dgWVec_t is just a pointer into the precomputed all_dg.
                T* dgWVec_t = all_dg.data() + (std::size_t)ele * dof * dgSz;

                if (ele != last_ele_tls) {
                    last_ele_tls = ele;
                    for (int c = 0; c < NUM_CHILDREN; c++)
                        p2c_interp_valid_tls[c] = false;
                }

                if (pNodes[ele].getLevel() == bLev) {
                    // same-level fast path (same as the serial integer path)
                    const uint64_t sz_morton =
                        ((uint64_t)1u << (m_uiMaxDepth - bLev));
                    const int64_t ddx =
                        (int64_t)pNodes[ele].getX() - (int64_t)blkNode.getX();
                    const int64_t ddy =
                        (int64_t)pNodes[ele].getY() - (int64_t)blkNode.getY();
                    const int64_t ddz =
                        (int64_t)pNodes[ele].getZ() - (int64_t)blkNode.getZ();
                    const int ei = (int)(ddx / (int64_t)sz_morton);
                    const int ej = (int)(ddy / (int64_t)sz_morton);
                    const int ek = (int)(ddz / (int64_t)sz_morton);
                    const int i0 = ei * (int)eOrder + (int)PW;
                    const int j0 = ej * (int)eOrder + (int)PW;
                    const int k0 = ek * (int)eOrder + (int)PW;
                    dendro::unzip::scatter_same_level_dispatch<T>(
                        dgWVec_t, uzWVec, eOrder, dof, (std::size_t)unSz,
                        (std::size_t)dgSz, (std::size_t)offset, lx, ly, lz, i0,
                        j0, k0);
                } else if (pNodes[ele].getLevel() > bLev) {
                    // fine -> coarse (even eOrder only; odd falls back below)
                    if ((eOrder % 2u) == 0) {
                        const uint64_t sz_ele =
                            ((uint64_t)1u
                             << (m_uiMaxDepth - pNodes[ele].getLevel()));
                        const int64_t ddx = (int64_t)pNodes[ele].getX() -
                                            (int64_t)blkNode.getX();
                        const int64_t ddy = (int64_t)pNodes[ele].getY() -
                                            (int64_t)blkNode.getY();
                        const int64_t ddz = (int64_t)pNodes[ele].getZ() -
                                            (int64_t)blkNode.getZ();
                        const int ei      = (int)(ddx / (int64_t)sz_ele);
                        const int ej      = (int)(ddy / (int64_t)sz_ele);
                        const int ek      = (int)(ddz / (int64_t)sz_ele);
                        const int half_eO = (int)eOrder / 2;
                        const int i0      = ei * half_eO + (int)PW;
                        const int j0      = ej * half_eO + (int)PW;
                        const int k0      = ek * half_eO + (int)PW;
                        dendro::unzip::scatter_fine_to_coarse_dispatch<T>(
                            dgWVec_t, uzWVec, eOrder, dof, (std::size_t)unSz,
                            (std::size_t)dgSz, (std::size_t)offset, lx, ly, lz,
                            i0, j0, k0);
                    }
                    // odd eOrder fall-through skipped in OMP path — not
                    // supported here. (Set DENDRO_UNZIP_OMP=OFF for odd
                    // orders.)
                } else {
                    // coarse -> fine: wavelet interpolation
                    childOct_tls.clear();
                    pNodes[ele].addChildren(childOct_tls);
                    // FP bounds matching the serial path exactly so that the
                    // SAME set of children is processed regardless of OMP.
                    const double hx =
                        (1u << (m_uiMaxDepth - bLev)) / (double)eOrder;
                    const double xmin = blkNode.minX() - PW * hx;
                    const double xmax = blkNode.maxX() + PW * hx;
                    const double ymin = blkNode.minY() - PW * hx;
                    const double ymax = blkNode.maxY() + PW * hx;
                    const double zmin = blkNode.minZ() - PW * hx;
                    const double zmax = blkNode.maxZ() + PW * hx;
                    for (unsigned int child = 0; child < NUM_CHILDREN;
                         child++) {
                        if ((childOct_tls[child].maxX() < xmin ||
                             childOct_tls[child].minX() >= xmax) ||
                            (childOct_tls[child].maxY() < ymin ||
                             childOct_tls[child].minY() >= ymax) ||
                            (childOct_tls[child].maxZ() < zmin ||
                             childOct_tls[child].minZ() >= zmax))
                            continue;

                        const unsigned int cnum =
                            childOct_tls[child].getMortonIndex();
                        if (!p2c_interp_valid_tls[cnum]) {
                            this->prolongateChildNodes(
                                in, cgSz, dgWVec_t, dgSz, ele, cnum, dof,
                                p2cI_base_t + cnum * dof * nPe, im1_t, im2_t,
                                all_dg.data(), (size_t)dof * dgSz);
                            p2c_interp_valid_tls[cnum] = true;
                        }

                        // child is at block's level — same-level reindex
                        const T* p2cI_base_child =
                            p2cI_base_t + cnum * dof * nPe;
                        const uint64_t sz_ele_child =
                            ((uint64_t)1u << (m_uiMaxDepth -
                                              childOct_tls[child].getLevel()));
                        const int64_t ddx =
                            (int64_t)childOct_tls[child].getX() -
                            (int64_t)blkNode.getX();
                        const int64_t ddy =
                            (int64_t)childOct_tls[child].getY() -
                            (int64_t)blkNode.getY();
                        const int64_t ddz =
                            (int64_t)childOct_tls[child].getZ() -
                            (int64_t)blkNode.getZ();
                        const int ei = (int)(ddx / (int64_t)sz_ele_child);
                        const int ej = (int)(ddy / (int64_t)sz_ele_child);
                        const int ek = (int)(ddz / (int64_t)sz_ele_child);
                        const int i0 = ei * (int)eOrder + (int)PW;
                        const int j0 = ej * (int)eOrder + (int)PW;
                        const int k0 = ek * (int)eOrder + (int)PW;
                        dendro::unzip::scatter_same_level_dispatch<T>(
                            p2cI_base_child, uzWVec, eOrder, dof,
                            (std::size_t)unSz, (std::size_t)nPe,
                            (std::size_t)offset, lx, ly, lz, i0, j0, k0);
                    }
                }
            }
        }
    }  // omp parallel
    return;
#endif

    std::vector<T> dg_ele_vec;
    dg_ele_vec.resize(dof * dgSz);

    T* dgWVec = dg_ele_vec.data();

#ifdef DENDRO_WIDE_PROLONGATION
    // Same contract as the OMP path above: a whole-mesh DG array whose ghost
    // slices are supplied by the ranks that own those elements. Two reasons it
    // has to be whole-mesh here rather than one element at a time: a ghost
    // element's hanging faces cannot be reconstructed on this rank (they need
    // round 2), and prolongateChildNodes must be handed a DG array or it falls
    // back to gathering through CG, which re-materialises neighbour elements --
    // ghosts included -- and hits the same wall.
    std::vector<T> all_dg_s((std::size_t)m_uiNumTotalElements * dof * dgSz);
    // Pass 0 of two -- see the OMP path for why the widening cannot be done in
    // a single sweep. Narrow values only, which never reach past round 1.
    std::vector<T> dg_narrow_s((std::size_t)m_uiNumTotalElements * dof * dgSz);
    {
        std::vector<double> im1_s(nPe), im2_s(nPe);
        auto materialise_narrow = [&](unsigned int e) {
            T* base = dg_narrow_s.data() + (std::size_t)e * dof * dgSz;
            for (unsigned int v = 0; v < dof; v++)
                this->getElementNodalValues(in + v * cgSz, base + v * dgSz, e,
                                            false, im1_s.data(), im2_s.data(),
                                            /*allowWide=*/false);
        };
        auto materialise_wide = [&](unsigned int e) {
            T* base = all_dg_s.data() + (std::size_t)e * dof * dgSz;
            for (unsigned int v = 0; v < dof; v++)
                this->getElementNodalValues(
                    in + v * cgSz, base + v * dgSz, e, false, im1_s.data(),
                    im2_s.data(), /*allowWide=*/true,
                    dg_narrow_s.data() + (std::size_t)v * dgSz,
                    (std::size_t)dof * dgSz);
        };

        // Overlap on the MATERIALISATION, not on the scatter. The scatter
        // sweep must keep its element order: m_uiAllElements is SFC-sorted, so
        // sweeping by index is SFC order at every rank count, which is what
        // makes the output bit-identical across np. Reordering it moved 384
        // pad points by up to 3.4e-08 -- shared points written by more than
        // one element, where the last writer changes. Materialisation has no
        // such constraint, and it is the expensive half anyway (it carries the
        // wide hanging-face reconstruction).
        this->buildWideProlongGhostMap();
        const bool wpx_prof = (std::getenv("DENDRO_WPX_PROFILE") != nullptr);
        double t_a = 0, t_b = 0, t_c = 0, t_d = 0, t_e = 0;
        if (wpx_prof) t_a = MPI_Wtime();

        // Each pass overlaps its own exchange the same way: build what other
        // ranks are waiting on, post, build the rest, land it.
        auto sweep = [&](const std::function<void(unsigned int)>& mat,
                         T* dst) {
            for (size_t i = 0; i < m_uiWpxSendEle.size(); i++)
                if (!m_uiWpxSendDone[m_uiWpxSendEle[i]]) {
                    mat(m_uiWpxSendEle[i]);
                    m_uiWpxSendDone[m_uiWpxSendEle[i]] = 1;
                }
            if (wpx_prof) t_b = MPI_Wtime();

            this->exchangeWideProlongDGBegin(dst, (std::size_t)dof * dgSz, dof,
                                             dgSz);
            if (wpx_prof) t_c = MPI_Wtime();

            // No m_e2b_unzip_counts filter: a local element feeding no block
            // of ours may still have been requested, and the pack reads this
            // array.
            for (unsigned int e = m_uiElementLocalBegin;
                 e < m_uiElementLocalEnd; e++)
                if (!m_uiWpxSendDone[e]) mat(e);
            if (wpx_prof) t_d = MPI_Wtime();

            this->exchangeWideProlongDGEnd(dst, (std::size_t)dof * dgSz, dof,
                                           dgSz);
            std::fill(m_uiWpxSendDone.begin(), m_uiWpxSendDone.end(), 0);
        };

        sweep(materialise_narrow, dg_narrow_s.data());
        sweep(materialise_wide, all_dg_s.data());

        // Diagnostic: exact-input mode. Overwrite every DG value -- local and
        // ghost, hanging nodes included -- with the analytic field, so the pad
        // fill below runs on perfect inputs. See m_uiWpxAnalyticDebug.
        if (m_uiWpxAnalyticDebug) {
            // DENDRO_WPX_EXACT_CLASS limits the overwrite to one node class
            // so the input error can be attributed: "face" = at most one
            // extreme coordinate, "edge" = two or more (element edge or
            // vertex). Unset = every node. Non-hanging nodes are exact
            // already, so overwriting them is a no-op either way.
            const char* cls_dbg = std::getenv("DENDRO_WPX_EXACT_CLASS");
            const int want_cls =
                (!cls_dbg) ? -1 : ((cls_dbg[0] == 'f') ? 0 : 1);
            const unsigned int eo_dbg = m_uiElementOrder;
            for (unsigned int e_dbg = 0; e_dbg < m_uiNumTotalElements; e_dbg++) {
                const ot::TreeNode& tn_dbg = m_uiAllElements[e_dbg];
                const double h_dbg =
                    (double)(1u << (m_uiMaxDepth - tn_dbg.getLevel())) /
                    (double)eo_dbg;
                for (unsigned int kk = 0; kk <= eo_dbg; kk++)
                    for (unsigned int jj = 0; jj <= eo_dbg; jj++)
                        for (unsigned int ii = 0; ii <= eo_dbg; ii++) {
                            const int next_dbg =
                                (ii == 0 || ii == eo_dbg ? 1 : 0) +
                                (jj == 0 || jj == eo_dbg ? 1 : 0) +
                                (kk == 0 || kk == eo_dbg ? 1 : 0);
                            if (want_cls == 0 && next_dbg >= 2) continue;
                            if (want_cls == 1 && next_dbg < 2) continue;
                            const double val_dbg = m_uiWpxAnalyticDebug(
                                (double)tn_dbg.getX() + ii * h_dbg,
                                (double)tn_dbg.getY() + jj * h_dbg,
                                (double)tn_dbg.getZ() + kk * h_dbg);
                            const unsigned int n_dbg =
                                (kk * (eo_dbg + 1) + jj) * (eo_dbg + 1) + ii;
                            for (unsigned int v_dbg = 0; v_dbg < dof; v_dbg++) {
                                const std::size_t at_dbg =
                                    (std::size_t)e_dbg * dof * dgSz +
                                    (std::size_t)v_dbg * dgSz + n_dbg;
                                all_dg_s[at_dbg] = (T)val_dbg;
                                dg_narrow_s[at_dbg] = (T)val_dbg;
                            }
                        }
            }
        }

        if (wpx_prof) {
            t_e = MPI_Wtime();
            std::printf(
                "[wpx-prof rank %d] send-set materialise %.2f ms | post %.2f "
                "ms | rest materialise %.2f ms | wait+unpack %.2f ms\n",
                m_uiActiveRank, (t_b - t_a) * 1e3, (t_c - t_b) * 1e3,
                (t_d - t_c) * 1e3, (t_e - t_d) * 1e3);
        }
    }
#endif

    std::vector<T> p2cI_all;
    p2cI_all.resize(NUM_CHILDREN * dof * nPe);
    bool p2c_interp_valid[NUM_CHILDREN];

    std::vector<ot::TreeNode> childOct;
    childOct.reserve(NUM_CHILDREN);

    for (unsigned int ele = 0; ele < m_uiNumTotalElements; ele++) {
        if (m_e2b_unzip_counts[ele] == 0) continue;

        for (unsigned int ii = 0; ii < NUM_CHILDREN; ii++)
            p2c_interp_valid[ii] = false;

        // get the elemental_local(dg) values
#ifdef DENDRO_WIDE_PROLONGATION
        dgWVec = all_dg_s.data() + (std::size_t)ele * dof * dgSz;
#else
        for (unsigned int v = 0; v < dof; v++)
            this->getElementNodalValues(in + v * cgSz, dgWVec + v * dgSz, ele,
                                        false);
#endif

        for (unsigned int i = 0; i < m_e2b_unzip_counts[ele]; i++) {
            const unsigned int e2b_offset = m_e2b_unzip_offset[ele];
            const unsigned int blk        = m_e2b_unzip_map[e2b_offset + i];
            assert(blk != LOOK_UP_TABLE_DEFAULT &&
                   blk < m_uiLocalBlockList.size());
            // Overlap path: scatter only into filter-matching blocks.
            if (blk_filter >= 0 &&
                (int)blkList[blk].getBlockType() != blk_filter)
                continue;

            const unsigned int regLevel = blkList[blk].getRegularGridLev();
            const ot::TreeNode blkNode  = blkList[blk].getBlockNode();
            const unsigned int PW       = blkList[blk].get1DPadWidth();

            const unsigned int lx       = blkList[blk].getAllocationSzX();
            const unsigned int ly       = blkList[blk].getAllocationSzY();
            const unsigned int lz       = blkList[blk].getAllocationSzZ();
            const unsigned int offset   = blkList[blk].getOffset();

            const unsigned int bLev =
                pNodes[blkList[blk].getLocalElementBegin()].getLevel();
            assert(regLevel == bLev);

            const double hx   = (1u << (m_uiMaxDepth - bLev)) / (double)eOrder;
            const double xmin = blkNode.minX() - PW * hx;
            const double xmax = blkNode.maxX() + PW * hx;
            const double ymin = blkNode.minY() - PW * hx;
            const double ymax = blkNode.maxY() + PW * hx;
            const double zmin = blkNode.minZ() - PW * hx;
            const double zmax = blkNode.maxZ() + PW * hx;

            // no interpolation needed just copy.
            if (pNodes[ele].getLevel() == bLev) {
#if defined(DENDRO_UNZIP_SCATTER_FAST)
                // Fast path: integer-index reindex + contiguous row memcpy.
                // Same-level scatter is a deterministic re-indexing — the
                // original FP coord math + std::round/fabs/tolerance snapping
                // is unnecessary work. See
                // include/mesh_unzip_scatter_kernels.h.
                const uint64_t sz_morton =
                    ((uint64_t)1u << (m_uiMaxDepth - bLev));
                const int64_t ddx =
                    (int64_t)pNodes[ele].getX() - (int64_t)blkNode.getX();
                const int64_t ddy =
                    (int64_t)pNodes[ele].getY() - (int64_t)blkNode.getY();
                const int64_t ddz =
                    (int64_t)pNodes[ele].getZ() - (int64_t)blkNode.getZ();
                // exact divisions at same level (verified)
                const int ei = (int)(ddx / (int64_t)sz_morton);
                const int ej = (int)(ddy / (int64_t)sz_morton);
                const int ek = (int)(ddz / (int64_t)sz_morton);
                const int i0 = ei * (int)eOrder + (int)PW;
                const int j0 = ej * (int)eOrder + (int)PW;
                const int k0 = ek * (int)eOrder + (int)PW;
                dendro::unzip::scatter_same_level_dispatch<T>(
                    dgWVec, uzWVec, eOrder, dof, (std::size_t)unSz,
                    (std::size_t)dgSz, (std::size_t)offset, lx, ly, lz, i0, j0,
                    k0);
#else
                const double hh =
                    (1u << (m_uiMaxDepth - pNodes[ele].getLevel())) /
                    (double)eOrder;
                const double invhh = 1.0 / hh;

                for (unsigned int k = 0; k < eOrder + 1; k++) {
                    double zz = pNodes[ele].minZ() + k * hh;

                    if (fabs(zz - zmin) < d_compar_tol) zz = zmin;
                    if (fabs(zz - zmax) < d_compar_tol) zz = zmax;

                    if (zz < zmin || zz > zmax) continue;
                    const int kkz = std::round((zz - zmin) * invhh);
                    assert(std::fabs(zz - zmin - kkz * hh) < d_compar_tol);
                    assert(kkz >= 0 && kkz < lz);

                    for (unsigned int j = 0; j < eOrder + 1; j++) {
                        double yy = pNodes[ele].minY() + j * hh;

                        if (fabs(yy - ymin) < d_compar_tol) yy = ymin;
                        if (fabs(yy - ymax) < d_compar_tol) yy = ymax;

                        if (yy < ymin || yy > ymax) continue;
                        const int jjy = std::round((yy - ymin) * invhh);
                        assert(std::fabs(yy - ymin - jjy * hh) < d_compar_tol);
                        assert(jjy >= 0 && jjy < ly);

                        for (unsigned int i = 0; i < eOrder + 1; i++) {
                            double xx = pNodes[ele].minX() + i * hh;

                            if (fabs(xx - xmin) < d_compar_tol) xx = xmin;
                            if (fabs(xx - xmax) < d_compar_tol) xx = xmax;

                            if (xx < xmin || xx > xmax) continue;
                            const int iix = std::round((xx - xmin) * invhh);
                            assert(std::fabs(xx - xmin - iix * hh) <
                                   d_compar_tol);
                            assert(iix >= 0 && iix < lx);

                            for (unsigned int v = 0; v < dof; v++)
                                uzWVec[v * unSz + offset + kkz * lx * ly +
                                       jjy * lx + iix] =
                                    dgWVec[v * dgSz +
                                           k * (eOrder + 1) * (eOrder + 1) +
                                           j * (eOrder + 1) + i];
                        }
                    }
                }
#endif

            } else if (pNodes[ele].getLevel() > bLev) {
                assert((bLev + 1) == pNodes[ele].getLevel());
                const unsigned int cnum = pNodes[ele].getMortonIndex();
                ot::TreeNode tmpParent  = pNodes[ele].getParent();

#if defined(DENDRO_UNZIP_SCATTER_FAST)
                // Fast path for even eOrder (BSSN eO=6 hits this). Element at
                // level bLev+1; for even eOrder the every-other element CG
                // node maps to a block CG node via integer arithmetic:
                //   iix = ei*(eOrder/2) + PW + i/2,   i in {0,2,...,eOrder}
                // where ei = (eleX - blkX) / S_ele (signed). Reads are
                // stride-2 in the source so no memcpy; gains come from
                // removing std::round/fabs/tolerance from the inner loop.
                if ((eOrder % 2u) == 0) {
                    const uint64_t sz_ele =
                        ((uint64_t)1u
                         << (m_uiMaxDepth - pNodes[ele].getLevel()));
                    const int64_t ddx =
                        (int64_t)pNodes[ele].getX() - (int64_t)blkNode.getX();
                    const int64_t ddy =
                        (int64_t)pNodes[ele].getY() - (int64_t)blkNode.getY();
                    const int64_t ddz =
                        (int64_t)pNodes[ele].getZ() - (int64_t)blkNode.getZ();
                    const int ei      = (int)(ddx / (int64_t)sz_ele);
                    const int ej      = (int)(ddy / (int64_t)sz_ele);
                    const int ek      = (int)(ddz / (int64_t)sz_ele);
                    const int half_eO = (int)eOrder / 2;
                    const int i0      = ei * half_eO + (int)PW;
                    const int j0      = ej * half_eO + (int)PW;
                    const int k0      = ek * half_eO + (int)PW;
                    dendro::unzip::scatter_fine_to_coarse_dispatch<T>(
                        dgWVec, uzWVec, eOrder, dof, (std::size_t)unSz,
                        (std::size_t)dgSz, (std::size_t)offset, lx, ly, lz, i0,
                        j0, k0);
                    continue;  // skip the FP fallback for this (ele,blk)
                }
                // Odd eOrder: fall through to FP path below.
#endif
                const double hh =
                    (1u << (m_uiMaxDepth - pNodes[ele].getLevel())) /
                    (double)eOrder;
                const double invhh = 1.0 / (2 * hh);

                assert(eOrder > 1);
                const unsigned int cb = (eOrder % 2 == 0) ? 0 : 1;

                for (unsigned int k = cb; k < eOrder + 1; k += 2) {
                    double zz = (pNodes[ele].minZ() + k * hh);
                    if (fabs(zz - zmin) < d_compar_tol) zz = zmin;
                    if (fabs(zz - zmax) < d_compar_tol) zz = zmax;

                    if (zz < zmin || zz > zmax) continue;
                    const int kkz = std::round((zz - zmin) * invhh);
                    assert(kkz >= 0 && kkz < lz);

                    for (unsigned int j = cb; j < eOrder + 1; j += 2) {
                        double yy = pNodes[ele].minY() + j * hh;

                        if (fabs(yy - ymin) < d_compar_tol) yy = ymin;
                        if (fabs(yy - ymax) < d_compar_tol) yy = ymax;

                        if (yy < ymin || yy > ymax) continue;

                        const int jjy = std::round((yy - ymin) * invhh);
                        assert(jjy >= 0 && jjy < ly);

                        for (unsigned int i = cb; i < eOrder + 1; i += 2) {
                            double xx = pNodes[ele].minX() + i * hh;

                            if (fabs(xx - xmin) < d_compar_tol) xx = xmin;
                            if (fabs(xx - xmax) < d_compar_tol) xx = xmax;

                            if (xx < xmin || xx > xmax) continue;
                            const int iix = std::round((xx - xmin) * invhh);
                            assert(iix >= 0 && iix < lx);

                            // std::cout<<"blk: "<<blk<<" blk copy : (i,j,k):
                            // ("<<iix<<" , "<<jjy<<", "<<kkz<<")"<<" of :
                            // "<<lx<<" xx: "<<xx<<" yy: "<<yy<<" zz:"<<zz<<"
                            // xmin: "<<xmin<<" ymin: "<<ymin<<" zmin:
                            // "<<zmin<<" hh : "<<hh<<" hhx : "<<hx<<std::endl;
                            for (unsigned int v = 0; v < dof; v++)
                                uzWVec[v * unSz + offset + kkz * lx * ly +
                                       jjy * lx + iix] =
                                    dgWVec[v * dgSz +
                                           k * (eOrder + 1) * (eOrder + 1) +
                                           j * (eOrder + 1) + i];
                        }
                    }
                }

            } else {
                assert((bLev) == (pNodes[ele].getLevel() + 1));
                childOct.clear();
                pNodes[ele].addChildren(
                    childOct);  // note this is the ordering of SFC (depends on
                                // Hilbert or Morton. )

                for (unsigned int child = 0; child < NUM_CHILDREN; child++) {
                    if ((childOct[child].maxX() < xmin ||
                         childOct[child].minX() >= xmax) ||
                        (childOct[child].maxY() < ymin ||
                         childOct[child].minY() >= ymax) ||
                        (childOct[child].maxZ() < zmin ||
                         childOct[child].minZ() >= zmax))
                        continue;

                    // std::cout<<"blk: "<<blk<<" blkNode: "<<blkNode<<" child:
                    // "<<child<<" child node "<<childOct[child]<<" parent :
                    // "<<pNodes[ele]<<std::endl;
                    const double hh =
                        (1u << (m_uiMaxDepth - childOct[child].getLevel())) /
                        (double)eOrder;
                    const double invhh      = 1.0 / hh;

                    const unsigned int cnum = childOct[child].getMortonIndex();
                    if (!p2c_interp_valid[cnum]) {
                        this->prolongateChildNodes(
                            in, cgSz, dgWVec, dgSz, ele, cnum, dof,
                            p2cI_all.data() + cnum * dof * nPe, nullptr,
                            nullptr
#ifdef DENDRO_WIDE_PROLONGATION
                            ,
                            all_dg_s.data(), (std::size_t)dof * dgSz
#endif
                        );

                        p2c_interp_valid[cnum] = true;
                    }

#if defined(DENDRO_UNZIP_SCATTER_FAST)
                    // Child is at the SAME level as the block, so the scatter
                    // from p2cI_all into uzWVec is just a same-level reindex.
                    // Re-use scatter_same_level_dispatch with p2cI_all as the
                    // virtual DG source (layout: cnum*dof*nPe + v*nPe).
                    {
                        const T* p2cI_base = p2cI_all.data() + cnum * dof * nPe;
                        const uint64_t sz_ele_child =
                            ((uint64_t)1u
                             << (m_uiMaxDepth - childOct[child].getLevel()));
                        const int64_t ddx = (int64_t)childOct[child].getX() -
                                            (int64_t)blkNode.getX();
                        const int64_t ddy = (int64_t)childOct[child].getY() -
                                            (int64_t)blkNode.getY();
                        const int64_t ddz = (int64_t)childOct[child].getZ() -
                                            (int64_t)blkNode.getZ();
                        const int ei      = (int)(ddx / (int64_t)sz_ele_child);
                        const int ej      = (int)(ddy / (int64_t)sz_ele_child);
                        const int ek      = (int)(ddz / (int64_t)sz_ele_child);
                        const int i0      = ei * (int)eOrder + (int)PW;
                        const int j0      = ej * (int)eOrder + (int)PW;
                        const int k0      = ek * (int)eOrder + (int)PW;
                        dendro::unzip::scatter_same_level_dispatch<T>(
                            p2cI_base, uzWVec, eOrder, dof, (std::size_t)unSz,
                            (std::size_t)nPe, (std::size_t)offset, lx, ly, lz,
                            i0, j0, k0);
                    }
#else
                    for (unsigned int v = 0; v < dof; v++) {
                        const T* const p2cI =
                            p2cI_all.data() + cnum * dof * nPe + v * nPe;
                        for (unsigned int k = 0; k < eOrder + 1; k++) {
                            double zz = childOct[child].minZ() + k * hh;

                            if (fabs(zz - zmin) < d_compar_tol) zz = zmin;
                            if (fabs(zz - zmax) < d_compar_tol) zz = zmax;

                            if (zz < zmin || zz > zmax) continue;
                            const int kkz = std::round((zz - zmin) * invhh);
                            assert(kkz >= 0 && kkz < lz);

                            for (unsigned int j = 0; j < eOrder + 1; j++) {
                                double yy = childOct[child].minY() + j * hh;

                                if (fabs(yy - ymin) < d_compar_tol) yy = ymin;
                                if (fabs(yy - ymax) < d_compar_tol) yy = ymax;

                                if (yy < ymin || yy > ymax) continue;

                                const int jjy = std::round((yy - ymin) * invhh);
                                assert(jjy >= 0 && jjy < ly);

                                for (unsigned int i = 0; i < eOrder + 1; i++) {
                                    double xx = childOct[child].minX() + i * hh;

                                    if (fabs(xx - xmin) < d_compar_tol)
                                        xx = xmin;
                                    if (fabs(xx - xmax) < d_compar_tol)
                                        xx = xmax;

                                    if (xx < xmin || xx > xmax) continue;
                                    const int iix =
                                        std::round((xx - xmin) * invhh);
                                    assert(iix >= 0 && iix < lx);

                                    uzWVec[v * unSz + offset + kkz * lx * ly +
                                           jjy * lx + iix] =
                                        p2cI[k * (eOrder + 1) * (eOrder + 1) +
                                             j * (eOrder + 1) + i];
                                }
                            }
                        }
                    }
#endif
                }
            }
        }
    }
}

template <typename T>
void Mesh::unzip_scatter_batch(const T* const* ins, T* const* outs,
                               unsigned int n_vars) {
    // Same guard as Mesh::unzip: callers use these interchangeably, and with an
    // empty block list the scatter has nothing to do but the DG precompute would
    // still allocate all_dg and walk every element.
    if ((!m_uiIsActive) || (m_uiLocalBlockList.empty())) return;
#if !defined(DENDRO_UNZIP_OMP)
    // Non-OMP build: just loop. No win to amortize. unzip_scatter ticks the
    // wpx memo epoch itself.
    for (unsigned int v = 0; v < n_vars; v++)
        this->unzip_scatter(ins[v], outs[v], 1);
    return;
#else
#ifdef DENDRO_WIDE_PROLONGATION
    // new DG arrays this call: retire every wpx child-memo entry
    m_uiWpxMemoEpoch = wpxMemoClock();
#endif
    const ot::TreeNode* pNodes = m_uiAllElements.data();
    const ot::Block* blkList   = m_uiLocalBlockList.data();
    const unsigned int eOrder  = m_uiElementOrder;
    const unsigned int nPe     = m_uiNpE;
    const unsigned int cgSz    = this->getDegOfFreedom();
    const unsigned int unSz    = this->getDegOfFreedomUnZip();
    const unsigned int dgSz    = nPe;

    // Build b2e map ONCE (mesh structure doesn't change across variables)
    const size_t n_blocks      = m_uiLocalBlockList.size();
    std::vector<unsigned int> b2e_count(n_blocks, 0);
    for (unsigned int ele = 0; ele < m_uiNumTotalElements; ele++) {
        if (m_e2b_unzip_counts[ele] == 0) continue;
        const unsigned int eo = m_e2b_unzip_offset[ele];
        for (unsigned int i = 0; i < m_e2b_unzip_counts[ele]; i++) {
            b2e_count[m_e2b_unzip_map[eo + i]]++;
        }
    }
    std::vector<unsigned int> b2e_offset(n_blocks + 1, 0);
    for (size_t b = 0; b < n_blocks; b++)
        b2e_offset[b + 1] = b2e_offset[b] + b2e_count[b];
    std::vector<unsigned int> b2e_map(b2e_offset[n_blocks]);
    std::vector<unsigned int> b2e_cur(n_blocks, 0);
    for (unsigned int ele = 0; ele < m_uiNumTotalElements; ele++) {
        if (m_e2b_unzip_counts[ele] == 0) continue;
        const unsigned int eo = m_e2b_unzip_offset[ele];
        for (unsigned int i = 0; i < m_e2b_unzip_counts[ele]; i++) {
            const unsigned int blk                    = m_e2b_unzip_map[eo + i];
            b2e_map[b2e_offset[blk] + b2e_cur[blk]++] = ele;
        }
    }

    // Per-variable scratch (reused across vars, size for ONE variable's DG).
    std::vector<T> all_dg((std::size_t)m_uiNumTotalElements * dgSz);

// ONE parallel region for ALL variables. The fork/join cost is paid once
// total, not n_vars times.
#pragma omp parallel
    {
        std::vector<T> p2cI_all_tls(NUM_CHILDREN * nPe);  // dof=1
        std::vector<double> im1_tls(nPe), im2_tls(nPe);
        std::vector<ot::TreeNode> childOct_tls;
        childOct_tls.reserve(NUM_CHILDREN);
        bool p2c_interp_valid_tls[NUM_CHILDREN];
        unsigned int last_ele_tls = UINT_MAX;

        T* p2cI_base_t            = p2cI_all_tls.data();
        double* const im1_t       = im1_tls.data();
        double* const im2_t       = im2_tls.data();

        // Per-thread scratch for the parallel precompute (separate from the
        // wavelet scratch above so the precompute can run concurrently).
        std::vector<double> pre_im1_tls(nPe), pre_im2_tls(nPe);
        double* const pre_im1_t = pre_im1_tls.data();
        double* const pre_im2_t = pre_im2_tls.data();

        for (unsigned int v = 0; v < n_vars; v++) {
            const T* in_v = ins[v];
            T* uzWVec     = outs[v];

// PARALLEL precompute (now thread-safe via the explicit-scratch
// overload of getElementNodalValues — it routes parent2Child
// calls through the im1_t/im2_t scratch instead of RefElement's
// shared im_vec1/im_vec2).
#pragma omp for schedule(static)
            for (unsigned int ele = 0; ele < m_uiNumTotalElements; ele++) {
                if (m_e2b_unzip_counts[ele] == 0) continue;
#ifdef DENDRO_WIDE_PROLONGATION
                // see the note in unzip_scatter: ghost slices come from the
                // owning rank, not from a local recompute
                if (ele < m_uiElementLocalBegin || ele >= m_uiElementLocalEnd)
                    continue;
#endif
                this->getElementNodalValues(
                    in_v, all_dg.data() + (std::size_t)ele * dgSz, ele, false,
                    pre_im1_t, pre_im2_t);
            }
// implicit barrier at end of `omp for`
#ifdef DENDRO_WIDE_PROLONGATION
#pragma omp single
            {
                this->exchangeWideProlongDG(in_v, (size_t)0,
                                            all_dg.data(), dgSz, 1u, dgSz);
            }
#endif

// PARALLEL scatter — reuse the same logic as the OMP path in
// unzip_scatter, but with dof=1 fixed.
#pragma omp for schedule(dynamic, 1)
            for (size_t blk_idx = 0; blk_idx < n_blocks; blk_idx++) {
                const unsigned int blk     = (unsigned int)blk_idx;
                const ot::TreeNode blkNode = blkList[blk].getBlockNode();
                const unsigned int PW      = blkList[blk].get1DPadWidth();
                const unsigned int lx      = blkList[blk].getAllocationSzX();
                const unsigned int ly      = blkList[blk].getAllocationSzY();
                const unsigned int lz      = blkList[blk].getAllocationSzZ();
                const unsigned int offset  = blkList[blk].getOffset();
                const unsigned int bLev =
                    pNodes[blkList[blk].getLocalElementBegin()].getLevel();

                const unsigned int e_start = b2e_offset[blk];
                const unsigned int e_end   = b2e_offset[blk + 1];

                // Reset cache for a new variable's worth of work.
                if (v == 0) last_ele_tls = UINT_MAX;
                // Actually reset on each variable regardless — the p2cI
                // contents depend on this variable's DG values, not previous
                // variable's. So invalidate on every new variable's iter.
                // (last_ele_tls is also per-variable scoped.)

                for (unsigned int idx = e_start; idx < e_end; idx++) {
                    const unsigned int ele = b2e_map[idx];
                    const T* dgWVec_t = all_dg.data() + (std::size_t)ele * dgSz;

                    if (ele != last_ele_tls) {
                        last_ele_tls = ele;
                        for (int c = 0; c < NUM_CHILDREN; c++)
                            p2c_interp_valid_tls[c] = false;
                    }

                    if (pNodes[ele].getLevel() == bLev) {
                        const uint64_t sz_morton =
                            ((uint64_t)1u << (m_uiMaxDepth - bLev));
                        const int64_t ddx = (int64_t)pNodes[ele].getX() -
                                            (int64_t)blkNode.getX();
                        const int64_t ddy = (int64_t)pNodes[ele].getY() -
                                            (int64_t)blkNode.getY();
                        const int64_t ddz = (int64_t)pNodes[ele].getZ() -
                                            (int64_t)blkNode.getZ();
                        const int ei      = (int)(ddx / (int64_t)sz_morton);
                        const int ej      = (int)(ddy / (int64_t)sz_morton);
                        const int ek      = (int)(ddz / (int64_t)sz_morton);
                        const int i0      = ei * (int)eOrder + (int)PW;
                        const int j0      = ej * (int)eOrder + (int)PW;
                        const int k0      = ek * (int)eOrder + (int)PW;
                        dendro::unzip::scatter_same_level_dispatch<T>(
                            dgWVec_t, uzWVec, eOrder, 1u, (std::size_t)unSz,
                            (std::size_t)dgSz, (std::size_t)offset, lx, ly, lz,
                            i0, j0, k0);
                    } else if (pNodes[ele].getLevel() > bLev) {
                        if ((eOrder % 2u) == 0) {
                            const uint64_t sz_ele =
                                ((uint64_t)1u
                                 << (m_uiMaxDepth - pNodes[ele].getLevel()));
                            const int64_t ddx = (int64_t)pNodes[ele].getX() -
                                                (int64_t)blkNode.getX();
                            const int64_t ddy = (int64_t)pNodes[ele].getY() -
                                                (int64_t)blkNode.getY();
                            const int64_t ddz = (int64_t)pNodes[ele].getZ() -
                                                (int64_t)blkNode.getZ();
                            const int ei      = (int)(ddx / (int64_t)sz_ele);
                            const int ej      = (int)(ddy / (int64_t)sz_ele);
                            const int ek      = (int)(ddz / (int64_t)sz_ele);
                            const int half_eO = (int)eOrder / 2;
                            const int i0      = ei * half_eO + (int)PW;
                            const int j0      = ej * half_eO + (int)PW;
                            const int k0      = ek * half_eO + (int)PW;
                            dendro::unzip::scatter_fine_to_coarse_dispatch<T>(
                                dgWVec_t, uzWVec, eOrder, 1u, (std::size_t)unSz,
                                (std::size_t)dgSz, (std::size_t)offset, lx, ly,
                                lz, i0, j0, k0);
                        }
                    } else {
                        childOct_tls.clear();
                        pNodes[ele].addChildren(childOct_tls);
                        const double hx =
                            (1u << (m_uiMaxDepth - bLev)) / (double)eOrder;
                        const double xmin = blkNode.minX() - PW * hx;
                        const double xmax = blkNode.maxX() + PW * hx;
                        const double ymin = blkNode.minY() - PW * hx;
                        const double ymax = blkNode.maxY() + PW * hx;
                        const double zmin = blkNode.minZ() - PW * hx;
                        const double zmax = blkNode.maxZ() + PW * hx;
                        for (unsigned int child = 0; child < NUM_CHILDREN;
                             child++) {
                            if ((childOct_tls[child].maxX() < xmin ||
                                 childOct_tls[child].minX() >= xmax) ||
                                (childOct_tls[child].maxY() < ymin ||
                                 childOct_tls[child].minY() >= ymax) ||
                                (childOct_tls[child].maxZ() < zmin ||
                                 childOct_tls[child].minZ() >= zmax))
                                continue;
                            const unsigned int cnum =
                                childOct_tls[child].getMortonIndex();
                            if (!p2c_interp_valid_tls[cnum]) {
                                // batch path walks one variable at a time
                                this->prolongateChildNodes(
                                    in_v, cgSz, dgWVec_t, dgSz, ele, cnum, 1u,
                                    p2cI_base_t + cnum * nPe, im1_t, im2_t,
                                    all_dg.data(), (size_t)dgSz);
                                p2c_interp_valid_tls[cnum] = true;
                            }
                            const T* p2cI_base_child = p2cI_base_t + cnum * nPe;
                            const uint64_t sz_ele_child =
                                ((uint64_t)1u
                                 << (m_uiMaxDepth -
                                     childOct_tls[child].getLevel()));
                            const int64_t ddx =
                                (int64_t)childOct_tls[child].getX() -
                                (int64_t)blkNode.getX();
                            const int64_t ddy =
                                (int64_t)childOct_tls[child].getY() -
                                (int64_t)blkNode.getY();
                            const int64_t ddz =
                                (int64_t)childOct_tls[child].getZ() -
                                (int64_t)blkNode.getZ();
                            const int ei = (int)(ddx / (int64_t)sz_ele_child);
                            const int ej = (int)(ddy / (int64_t)sz_ele_child);
                            const int ek = (int)(ddz / (int64_t)sz_ele_child);
                            const int i0 = ei * (int)eOrder + (int)PW;
                            const int j0 = ej * (int)eOrder + (int)PW;
                            const int k0 = ek * (int)eOrder + (int)PW;
                            dendro::unzip::scatter_same_level_dispatch<T>(
                                p2cI_base_child, uzWVec, eOrder, 1u,
                                (std::size_t)unSz, (std::size_t)nPe,
                                (std::size_t)offset, lx, ly, lz, i0, j0, k0);
                        }
                    }
                }
            }
            // implicit barrier at end of omp for; reset cache for next var.
            last_ele_tls = UINT_MAX;
        }
    }  // omp parallel
#endif
}

template <typename T>
void Mesh::unzip(const T* in, T* out, unsigned int dof, int blk_filter) {
#ifdef DENDRO_WIDE_PROLONGATION
    // Do NOT skip a rank that happens to own no blocks: unzip_scatter now
    // performs a collective DG ghost exchange, and dropping out here would
    // hang the ranks that do have blocks. With no blocks every
    // m_e2b_unzip_counts entry is zero, so the rest of the routine is a no-op.
    if (!m_uiIsActive) return;
#else
    if ((!m_uiIsActive) || (m_uiLocalBlockList.empty())) return;
#endif
    this->unzip_scatter(in, out, dof, blk_filter);
}

#if 0
    template<typename T>
    void Mesh::readSpecialPtsBegin(const T* in)
    {

        if(m_uiGlobalNpes==1)
            return;


         // send recv buffers.
        T* sendB = NULL;
        T* recvB = NULL;

        std::vector<T> eVec;
        eVec.resize(m_uiNpE);

        if(m_uiIsActive)
        {
            const unsigned int sendBSz=m_uiSendOffsetRePt[m_uiActiveNpes-1] + m_uiSendCountRePt[m_uiActiveNpes-1];
            const unsigned int recvBSz=m_uiRecvOffsetRePt[m_uiActiveNpes-1] + m_uiRecvCountRePt[m_uiActiveNpes-1];

            AsyncExchangeContex ctx(in);
            MPI_Comm commActive= m_uiCommActive;
            unsigned int proc_id;

            if(recvBSz)
            {
                ctx.allocateRecvBuffer((sizeof(T)*recvBSz));
                recvB=(T*)ctx.getRecvBuffer();

                // active recv procs
                for(unsigned int recv_p=0;recv_p<m_uiReqRecvProcList.size();recv_p++)
                {
                    proc_id=m_uiReqRecvProcList[recv_p];
                    MPI_Request* req=new MPI_Request();
                    par::Mpi_Irecv((recvB+m_uiRecvOffsetRePt[proc_id]),m_uiRecvCountRePt[proc_id],proc_id,m_uiCommTag,commActive,req);
                    ctx.getRequestList().push_back(req);

                }

            }

            if(sendBSz)
            {
                ctx.allocateSendBuffer(sizeof(T)*sendBSz);
                sendB=(T*)ctx.getSendBuffer();


                const unsigned int nx = m_uiElementOrder + 1;
                const unsigned int ny = m_uiElementOrder + 1;
                const unsigned int nz = m_uiElementOrder + 1;

                std::vector<unsigned int>* ownerList;
                unsigned int ownerID, ii_x, jj_y, kk_z;

                
                for(unsigned int i=0; i< m_uiUnzip_3pt_ele.size(); i++)
                {
                    ot::Key tmpEleKey= m_uiUnzip_3pt_ele[i];
                    assert((tmpEleKey.getFlag() & OCT_FOUND));
                    const unsigned int eleID = tmpEleKey.getSearchResult();
                    this->getElementNodalValues(in,&(*(eVec.begin())),eleID);
                    
                    const unsigned int step_sz = ((1u<< (m_uiMaxDepth - m_uiAllElements[eleID].getLevel()))/m_uiElementOrder);
                    ownerList = tmpEleKey.getOwnerList();
                    for(unsigned int w=0; w< ownerList->size();w++)
                    {
                        
                        const unsigned int ii = (m_uiUnzip_3pt_recv_keys[(*ownerList)[w]].minX() - m_uiAllElements[eleID].minX())/(step_sz); 
                        const unsigned int jj = (m_uiUnzip_3pt_recv_keys[(*ownerList)[w]].minY() - m_uiAllElements[eleID].minY())/(step_sz); 
                        const unsigned int kk = (m_uiUnzip_3pt_recv_keys[(*ownerList)[w]].minZ() - m_uiAllElements[eleID].minZ())/(step_sz);
                        const std::vector<unsigned int > * ownerList1 = m_uiUnzip_3pt_recv_keys[(*ownerList)[w]].getOwnerList();

                        for(unsigned int w1 = 0; w1 < ownerList1->size() ; w1++)
                        {
                            // if(m_uiActiveRank==1 && (*ownerList1)[w1]<18 )
                            //     std::cout<<" rank: "<<m_uiActiveRank<<" putting : "<<m_uiUnzip_3pt_recv_keys[(*ownerList)[w]]<<" to send buf loc: "<<(*ownerList1)[w1]<<std::endl;

                            sendB[(*ownerList1)[w1]] = eVec[kk * ny * nx + jj * nx + ii];
                        }
                            
                        
                    }
                }


                // active send procs
                for(unsigned int send_p=0;send_p<m_uiReqSendProcList.size();send_p++)
                {
                    proc_id=m_uiReqSendProcList[send_p];
                    MPI_Request * req=new MPI_Request();
                    par::Mpi_Isend(sendB+m_uiSendOffsetRePt[proc_id],m_uiSendCountRePt[proc_id],proc_id,m_uiCommTag,commActive,req);
                    ctx.getRequestList().push_back(req);

                }


            }

            m_uiCommTag++;
            m_uiMPIContexts.push_back(ctx);



        }

    }

    template <typename T>
    void Mesh::readSpecialPtsEnd(const T *in, T* out)
    {
        if(m_uiGlobalNpes == 1)
            return;

        // send recv buffers.
        T* sendB = NULL;
        T* recvB = NULL;

        if(m_uiIsActive)
        {
            const unsigned int sendBSz=m_uiSendOffsetRePt[m_uiActiveNpes-1] + m_uiSendCountRePt[m_uiActiveNpes-1];
            const unsigned int recvBSz=m_uiRecvOffsetRePt[m_uiActiveNpes-1] + m_uiRecvCountRePt[m_uiActiveNpes-1];

            //std::cout<<"rank: "<<m_uiActiveRank<<" recv sz: "<<recvBSz<<std::endl;

            unsigned int proc_id;
            unsigned int ctxIndex=0;

            for(unsigned int i=0;i<m_uiMPIContexts.size();i++)
            {
                if(m_uiMPIContexts[i].getBuffer()==in)
                {
                    ctxIndex=i;
                    break;
                }

            }

            MPI_Status status;
            // need to wait for the commns to finish ...
            for (unsigned int i = 0; i < m_uiMPIContexts[ctxIndex].getRequestList().size(); i++) {
                MPI_Wait(m_uiMPIContexts[ctxIndex].getRequestList()[i], &status);
            }

            if(recvBSz)
            {
                // copy the recv data to the vec
                recvB=(T*)m_uiMPIContexts[ctxIndex].getRecvBuffer();
                std::memcpy(out,recvB,sizeof(T)*recvBSz);
                
                // for(unsigned int i=0; i<recvBSz ; i++ )
                //     out[i] = recvB[i];
            }



            m_uiMPIContexts[ctxIndex].deAllocateSendBuffer();
            m_uiMPIContexts[ctxIndex].deAllocateRecvBuffer();

            for (unsigned int i = 0; i < m_uiMPIContexts[ctxIndex].getRequestList().size(); i++)
                delete m_uiMPIContexts[ctxIndex].getRequestList()[i];

            m_uiMPIContexts[ctxIndex].getRequestList().clear();

            // remove the context ...
            m_uiMPIContexts.erase(m_uiMPIContexts.begin() + ctxIndex);


        }

        return;
    }
#endif

template <typename T>
int Mesh::getFaceNeighborValues(unsigned int eleID, const T* in, T* out,
                                T* coords, unsigned int* neighID,
                                unsigned int face,
                                NeighbourLevel& level) const {
    if (!m_uiIsActive) return (0);

    const unsigned int lookUp =
        m_uiE2EMapping[eleID * m_uiNumDirections + face];
    if (lookUp == LOOK_UP_TABLE_DEFAULT) return (0);

    const unsigned int l1 = m_uiAllElements[eleID].getLevel();
    const unsigned int l2 = m_uiAllElements[lookUp].getLevel();

    for (unsigned int i = 0; i < (NUM_CHILDREN >> 1); i++)
        neighID[i] = LOOK_UP_TABLE_DEFAULT;

    int num_face_neighbours = 1;
    if (l1 == l2) {
        // both elements are in the same level.
        level      = NeighbourLevel::SAME;
        neighID[0] = lookUp;
        this->getElementNodalValues(in, out, lookUp);

        // coordinate computation
        const ot::TreeNode lookUpOct = m_uiAllElements[lookUp];
        const unsigned int sz = 1u << (m_uiMaxDepth - lookUpOct.getLevel());

        for (unsigned int k = 0; k < (m_uiElementOrder + 1); k++)
            for (unsigned int j = 0; j < (m_uiElementOrder + 1); j++)
                for (unsigned int i = 0; i < (m_uiElementOrder + 1); i++) {
                    coords[m_uiDim * (k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i) +
                           0] =
                        lookUpOct.minX() + i * (sz / (T)m_uiElementOrder);
                    coords[m_uiDim * (k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i) +
                           1] =
                        lookUpOct.minY() + j * (sz / (T)m_uiElementOrder);
                    coords[m_uiDim * (k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i) +
                           2] =
                        lookUpOct.minZ() + k * (sz / (T)m_uiElementOrder);
                }
    } else if (l2 < l1) {
        level      = NeighbourLevel::COARSE;
        // lookUp octant is coaser than eleID.
        neighID[0] = lookUp;
        this->getElementNodalValues(in, out + m_uiNpE, lookUp);

        unsigned int sz1 =
            1u << (m_uiMaxDepth - m_uiAllElements[eleID].getLevel());

        unsigned int x = m_uiAllElements[eleID].minX();
        unsigned int y = m_uiAllElements[eleID].minY();
        unsigned int z = m_uiAllElements[eleID].minZ();

        ot::TreeNode tmpOct;
        unsigned int cnum;
        switch (face) {
            case OCT_DIR_LEFT:
                tmpOct = ot::TreeNode(x - sz1, y, z, l1, m_uiDim, m_uiMaxDepth);
                cnum   = tmpOct.getMortonIndex();
                break;

            case OCT_DIR_RIGHT:
                tmpOct = ot::TreeNode(x + sz1, y, z, l1, m_uiDim, m_uiMaxDepth);
                cnum   = tmpOct.getMortonIndex();
                break;

            case OCT_DIR_DOWN:
                tmpOct = ot::TreeNode(x, y - sz1, z, l1, m_uiDim, m_uiMaxDepth);
                cnum   = tmpOct.getMortonIndex();
                break;

            case OCT_DIR_UP:
                tmpOct = ot::TreeNode(x, y + sz1, z, l1, m_uiDim, m_uiMaxDepth);
                cnum   = tmpOct.getMortonIndex();
                break;

            case OCT_DIR_BACK:
                tmpOct = ot::TreeNode(x, y, z - sz1, l1, m_uiDim, m_uiMaxDepth);
                cnum   = tmpOct.getMortonIndex();
                break;

            case OCT_DIR_FRONT:
                tmpOct = ot::TreeNode(x, y, z + sz1, l1, m_uiDim, m_uiMaxDepth);
                cnum   = tmpOct.getMortonIndex();
                break;

            default:
                std::cout << "global rank : " << m_uiGlobalRank
                          << " dir: " << face
                          << " is invalid. Function : " << __func__
                          << std::endl;
                MPI_Abort(m_uiCommGlobal, 0);
                break;
        }

        this->parent2ChildInterpolation(out + m_uiNpE, out, cnum, m_uiDim);

        // coordinate computation
        const ot::TreeNode lookUpOct = tmpOct;
        const unsigned int sz = 1u << (m_uiMaxDepth - lookUpOct.getLevel());

        for (unsigned int k = 0; k < (m_uiElementOrder + 1); k++)
            for (unsigned int j = 0; j < (m_uiElementOrder + 1); j++)
                for (unsigned int i = 0; i < (m_uiElementOrder + 1); i++) {
                    coords[m_uiDim * (k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i) +
                           0] =
                        lookUpOct.minX() + i * (sz / (T)m_uiElementOrder);
                    coords[m_uiDim * (k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i) +
                           1] =
                        lookUpOct.minY() + j * (sz / (T)m_uiElementOrder);
                    coords[m_uiDim * (k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i) +
                           2] =
                        lookUpOct.minZ() + k * (sz / (T)m_uiElementOrder);
                }

    } else {
        // lookUp octant is finer than eleID.
        assert(l2 > l1);

        unsigned int dir, dirOp, dir1, dir2;
        num_face_neighbours = 4;
        level               = NeighbourLevel::REFINE;
        switch (face) {
            case OCT_DIR_LEFT:

                dir        = OCT_DIR_LEFT;
                dirOp      = OCT_DIR_RIGHT;
                dir1       = OCT_DIR_FRONT;
                dir2       = OCT_DIR_UP;

                neighID[0] = lookUp;
                neighID[1] = m_uiE2EMapping[neighID[0] * NUM_FACES + dir2];
                neighID[2] = m_uiE2EMapping[neighID[0] * NUM_FACES + dir1];
                neighID[3] = m_uiE2EMapping[neighID[1] * NUM_FACES + dir1];

                break;

            case OCT_DIR_RIGHT:

                dir        = OCT_DIR_RIGHT;
                dirOp      = OCT_DIR_LEFT;

                dir1       = OCT_DIR_FRONT;
                dir2       = OCT_DIR_UP;

                neighID[0] = lookUp;
                neighID[1] = m_uiE2EMapping[neighID[0] * NUM_FACES + dir2];
                neighID[2] = m_uiE2EMapping[neighID[0] * NUM_FACES + dir1];
                neighID[3] = m_uiE2EMapping[neighID[1] * NUM_FACES + dir1];

                break;

            case OCT_DIR_DOWN:

                dir        = OCT_DIR_DOWN;
                dirOp      = OCT_DIR_UP;

                dir1       = OCT_DIR_FRONT;
                dir2       = OCT_DIR_RIGHT;

                neighID[0] = lookUp;
                neighID[1] = m_uiE2EMapping[neighID[0] * NUM_FACES + dir2];
                neighID[2] = m_uiE2EMapping[neighID[0] * NUM_FACES + dir1];
                neighID[3] = m_uiE2EMapping[neighID[1] * NUM_FACES + dir1];

                break;

            case OCT_DIR_UP:

                dir        = OCT_DIR_UP;
                dirOp      = OCT_DIR_DOWN;

                dir1       = OCT_DIR_FRONT;
                dir2       = OCT_DIR_RIGHT;

                neighID[0] = lookUp;
                neighID[1] = m_uiE2EMapping[neighID[0] * NUM_FACES + dir2];
                neighID[2] = m_uiE2EMapping[neighID[0] * NUM_FACES + dir1];
                neighID[3] = m_uiE2EMapping[neighID[1] * NUM_FACES + dir1];

                break;

            case OCT_DIR_BACK:

                dir        = OCT_DIR_BACK;
                dirOp      = OCT_DIR_FRONT;

                dir1       = OCT_DIR_UP;
                dir2       = OCT_DIR_RIGHT;

                neighID[0] = lookUp;
                neighID[1] = m_uiE2EMapping[neighID[0] * NUM_FACES + dir2];
                neighID[2] = m_uiE2EMapping[neighID[0] * NUM_FACES + dir1];
                neighID[3] = m_uiE2EMapping[neighID[1] * NUM_FACES + dir1];

                break;

            case OCT_DIR_FRONT:

                dir        = OCT_DIR_FRONT;
                dirOp      = OCT_DIR_BACK;

                dir1       = OCT_DIR_UP;
                dir2       = OCT_DIR_RIGHT;

                neighID[0] = lookUp;
                neighID[1] = m_uiE2EMapping[neighID[0] * NUM_FACES + dir2];
                neighID[2] = m_uiE2EMapping[neighID[0] * NUM_FACES + dir1];
                neighID[3] = m_uiE2EMapping[neighID[1] * NUM_FACES + dir1];

                break;

            default:
                std::cout << "global rank : " << m_uiGlobalRank
                          << " dir: " << face
                          << " is invalid. Function : " << __func__
                          << std::endl;
                MPI_Abort(m_uiCommGlobal, 0);
        }

        for (unsigned int child = 0; child < (NUM_CHILDREN >> 1); child++) {
            this->getElementNodalValues(in, out + child * m_uiNpE,
                                        neighID[child]);

            // coordinate computation
            const ot::TreeNode lookUpOct = m_uiAllElements[neighID[child]];
            const unsigned int sz = 1u << (m_uiMaxDepth - lookUpOct.getLevel());

            for (unsigned int k = 0; k < (m_uiElementOrder + 1); k++)
                for (unsigned int j = 0; j < (m_uiElementOrder + 1); j++)
                    for (unsigned int i = 0; i < (m_uiElementOrder + 1); i++) {
                        coords[child * m_uiNpE * m_uiDim +
                               m_uiDim * (k * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          j * (m_uiElementOrder + 1) + i) +
                               0] =
                            lookUpOct.minX() + i * (sz / (T)m_uiElementOrder);
                        coords[child * m_uiNpE * m_uiDim +
                               m_uiDim * (k * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          j * (m_uiElementOrder + 1) + i) +
                               1] =
                            lookUpOct.minY() + j * (sz / (T)m_uiElementOrder);
                        coords[child * m_uiNpE * m_uiDim +
                               m_uiDim * (k * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          j * (m_uiElementOrder + 1) + i) +
                               2] =
                            lookUpOct.minZ() + k * (sz / (T)m_uiElementOrder);
                    }
        }
    }

    return num_face_neighbours;
}

template <typename T>
void Mesh::getUnzipElementalNodalValues(const T* uzipVec, unsigned int blkID,
                                        unsigned int ele, T* out,
                                        bool isPadded) const {
    const ot::Block block       = m_uiLocalBlockList[blkID];
    ot::TreeNode blkNode        = m_uiLocalBlockList[blkID].getBlockNode();
    const unsigned int eleBegin = block.getLocalElementBegin();
    const unsigned int eleEnd   = block.getLocalElementEnd();

    assert(eleBegin <= ele && ele < eleEnd);
    const unsigned int regLev    = block.getRegularGridLev();
    const unsigned int lx        = block.getAllocationSzX();
    const unsigned int ly        = block.getAllocationSzY();
    const unsigned int lz        = block.getAllocationSzZ();
    const unsigned int offset    = block.getOffset();
    const unsigned int paddWidth = block.get1DPadWidth();

    const unsigned int ei = (m_uiAllElements[ele].getX() - blkNode.getX()) >>
                            (m_uiMaxDepth - regLev);
    const unsigned int ej = (m_uiAllElements[ele].getY() - blkNode.getY()) >>
                            (m_uiMaxDepth - regLev);
    const unsigned int ek = (m_uiAllElements[ele].getZ() - blkNode.getZ()) >>
                            (m_uiMaxDepth - regLev);
    const unsigned int eleIDMax = m_uiLocalBlockList[blkID].getElemSz1D();

    if (isPadded) {
        const unsigned int ib = ei * m_uiElementOrder;
        const unsigned int ie =
            ei * m_uiElementOrder + (m_uiElementOrder + 1) + 2 * paddWidth;

        const unsigned int jb = ej * m_uiElementOrder;
        const unsigned int je =
            ej * m_uiElementOrder + (m_uiElementOrder + 1) + 2 * paddWidth;

        const unsigned int kb = ek * m_uiElementOrder;
        const unsigned int ke =
            ek * m_uiElementOrder + (m_uiElementOrder + 1) + 2 * paddWidth;

        const unsigned int en[3] = {(m_uiElementOrder + 1) + 2 * paddWidth,
                                    (m_uiElementOrder + 1) + 2 * paddWidth,
                                    (m_uiElementOrder + 1) + 2 * paddWidth};

        for (unsigned int k = kb; k < ke; k++)
            for (unsigned int j = jb; j < je; j++)
                for (unsigned int i = ib; i < ie; i++)
                    out[(k - kb) * en[1] * en[0] + (j - jb) * en[1] +
                        (i - ib)] = uzipVec[offset + k * ly * lx + j * lx + i];

        // copy the unzip element last point to the padding region.
        if (m_uiAllElements[ele].minX() == 0) {
            assert(ei == 0);

            for (unsigned int k = kb; k < ke; k++)
                for (unsigned int j = jb; j < je; j++)
                    for (unsigned int i = ib; i < paddWidth; i++)
                        out[(k - kb) * en[1] * en[0] + (j - jb) * en[1] +
                            (i - ib)] =
                            uzipVec[offset + k * ly * lx + j * lx + paddWidth];
        }

        if (m_uiAllElements[ele].minY() == 0) {
            assert(ej == 0);

            for (unsigned int k = kb; k < ke; k++)
                for (unsigned int j = jb; j < paddWidth; j++)
                    for (unsigned int i = ib; i < ie; i++)
                        out[(k - kb) * en[1] * en[0] + (j - jb) * en[1] +
                            (i - ib)] =
                            uzipVec[offset + k * ly * lx + paddWidth * lx + i];
        }

        if (m_uiAllElements[ele].minZ() == 0) {
            assert(ek == 0);

            for (unsigned int k = kb; k < paddWidth; k++)
                for (unsigned int j = jb; j < je; j++)
                    for (unsigned int i = ib; i < ie; i++)
                        out[(k - kb) * en[1] * en[0] + (j - jb) * en[1] +
                            (i - ib)] =
                            uzipVec[offset + paddWidth * ly * lx + j * lx + i];
        }

        if (m_uiAllElements[ele].maxX() == (1u << m_uiMaxDepth)) {
            assert(ei == (eleIDMax - 1));

            for (unsigned int k = kb; k < ke; k++)
                for (unsigned int j = jb; j < je; j++)
                    for (unsigned int i = (ie - paddWidth); i < ie; i++)
                        out[(k - kb) * en[1] * en[0] + (j - jb) * en[1] +
                            (i - ib)] = uzipVec[offset + k * ly * lx + j * lx +
                                                (ie - paddWidth - 1)];
        }

        if (m_uiAllElements[ele].maxY() == (1u << m_uiMaxDepth)) {
            assert(ej == (eleIDMax - 1));

            for (unsigned int k = kb; k < ke; k++)
                for (unsigned int j = (je - paddWidth); j < je; j++)
                    for (unsigned int i = ib; i < ie; i++)
                        out[(k - kb) * en[1] * en[0] + (j - jb) * en[1] +
                            (i - ib)] = uzipVec[offset + k * ly * lx +
                                                (je - paddWidth - 1) * lx + i];
        }

        if (m_uiAllElements[ele].maxZ() == (1u << m_uiMaxDepth)) {
            assert(ek == (eleIDMax - 1));

            for (unsigned int k = (ke - paddWidth); k < ke; k++)
                for (unsigned int j = jb; j < je; j++)
                    for (unsigned int i = ib; i < ie; i++)
                        out[(k - kb) * en[1] * en[0] + (j - jb) * en[1] +
                            (i - ib)] =
                            uzipVec[offset + (ke - paddWidth - 1) * ly * lx +
                                    j * lx + i];
        }

    } else {
        const unsigned int ib = ei * m_uiElementOrder + paddWidth;
        const unsigned int ie = ei * m_uiElementOrder + (m_uiElementOrder + 1);

        const unsigned int jb = ej * m_uiElementOrder + paddWidth;
        const unsigned int je = ej * m_uiElementOrder + (m_uiElementOrder + 1);

        const unsigned int kb = ek * m_uiElementOrder + paddWidth;
        const unsigned int ke = ek * m_uiElementOrder + (m_uiElementOrder + 1);

        const unsigned int en[3] = {(m_uiElementOrder + 1),
                                    (m_uiElementOrder + 1),
                                    (m_uiElementOrder + 1)};

        for (unsigned int k = kb; k < ke; k++)
            for (unsigned int j = jb; j < je; j++)
                for (unsigned int i = ib; i < ie; i++)
                    out[(k - kb) * en[1] * en[0] + (j - jb) * en[1] +
                        (i - ib)] = uzipVec[offset + k * ly * lx + j * lx + i];
    }
}

/**
 * Report the first narrow fallback and stay quiet after that. A fallback is
 * legitimate -- there is simply no same-level coarse neighbour to widen into --
 * but it must not be invisible, because those nodes keep the original order.
 */
inline void wpxLogFallbackOnce(const char *what, unsigned int got,
                               unsigned int want) {
    static std::atomic<bool> logged{false};
    bool expected = false;
    if (logged.compare_exchange_strong(expected, true))
        std::cout << "[wide prolongation] stencil clipped at " << what
                  << ": reached " << got << " of " << want
                  << " extra coarse nodes on some axis, so those nodes use a "
                     "narrower stencil and keep a lower order. This is "
                     "expected at domain boundaries and level jumps. Logged "
                     "once per run."
                  << std::endl;
}

/**
 * Resolve the element `steps` face hops away from `ele`, requiring every
 * intermediate to be a usable same-level coarse element. Sets `bad_ghost`
 * when the walk is blocked by a round-2 ghost specifically, since that case
 * is rank dependent and must not be silently absorbed.
 */
inline unsigned int Mesh::wpxWalk(unsigned int ele, const unsigned int *dirs,
                                  unsigned int steps, bool &bad_ghost,
                                  unsigned int levelMask,
                                  bool dgGhostOk) const {
    unsigned int cur         = ele;
    const unsigned int lev   = m_uiAllElements[ele].getLevel();

    for (unsigned int s = 0; s < steps; s++) {
        const unsigned int nxt =
            m_uiE2EMapping[cur * m_uiNumDirections + dirs[s]];

        if (nxt == LOOK_UP_TABLE_DEFAULT) return LOOK_UP_TABLE_DEFAULT;
        if (nxt >= m_uiAllElements.size()) return LOOK_UP_TABLE_DEFAULT;
        {
            const unsigned int nl = m_uiAllElements[nxt].getLevel();
            unsigned int bit;
            if (nl == lev)
                bit = WPX_LVL_SAME;
            else if (nl == lev + 1) {
                // A finer neighbour spans half our extent, so its p intervals
                // cover p*H/2 -- exactly the 3H the stencil asks for at p=6,
                // and less below that. gatherExtendedCoarseImpl's decimated
                // mapax would then index from p-6 to +6 and walk off both
                // ends of the element (measured: SIGSEGV at p=4). Refuse the
                // direction instead; the operator degrades to same-level
                // reach, which is correct, just narrower.
                if (m_uiElementOrder < 6u) return LOOK_UP_TABLE_DEFAULT;
                bit = WPX_LVL_FINER;
            }
            else if (nl + 1 == lev)
                bit = WPX_LVL_COARSER;
            else
                return LOOK_UP_TABLE_DEFAULT;  // >1 level apart: not 2:1
            if (!(levelMask & bit)) return LOOK_UP_TABLE_DEFAULT;
        }
        if (!m_uiIsNodalMapValid[nxt] && !dgGhostOk) {
            bad_ghost = true;
            return LOOK_UP_TABLE_DEFAULT;
        }
        cur = nxt;
    }
    return cur;
}

/** Element at signed element offset (ox,oy,oz) from `ele`, or default. */
inline unsigned int Mesh::wpxNeighbour(unsigned int ele, int ox, int oy,
                                       int oz, bool &bad_ghost,
                                       unsigned int levelMask,
                                       bool dgGhostOk) const {
    unsigned int dirs[3];
    unsigned int n = 0;
    if (ox < 0)
        dirs[n++] = OCT_DIR_LEFT;
    else if (ox > 0)
        dirs[n++] = OCT_DIR_RIGHT;
    if (oy < 0)
        dirs[n++] = OCT_DIR_DOWN;
    else if (oy > 0)
        dirs[n++] = OCT_DIR_UP;
    if (oz < 0)
        dirs[n++] = OCT_DIR_BACK;
    else if (oz > 0)
        dirs[n++] = OCT_DIR_FRONT;

    return wpxWalk(ele, dirs, n, bad_ghost, levelMask, dgGhostOk);
}

inline unsigned int Mesh::probeCoarseExtension(unsigned int ele,
                                               unsigned int want_ext,
                                               unsigned int ext[6],
                                               unsigned int levelMask,
                                               unsigned char *mode,
                                               bool dgGhostOk) const {
    for (unsigned int d = 0; d < 6; d++) {
        ext[d] = 0;
        if (mode) mode[d] = WPX_EXT_NONE;
    }
    if (!m_uiIsActive || want_ext == 0) return WPX_OK;

    // a neighbour shares its touching node plane, so it can only add p nodes
    const unsigned int cap =
        (want_ext < m_uiElementOrder) ? want_ext : m_uiElementOrder;

    unsigned int status = WPX_OK;
    bool bad_ghost      = false;

    static const int off[6][3] = {{-1, 0, 0}, {1, 0, 0}, {0, -1, 0},
                                  {0, 1, 0},  {0, 0, -1}, {0, 0, 1}};
    const unsigned int lev = m_uiAllElements[ele].getLevel();
    for (unsigned int d = 0; d < 6; d++) {
        bool bg              = false;
        const unsigned int q =
            wpxNeighbour(ele, off[d][0], off[d][1], off[d][2], bg, levelMask,
                         dgGhostOk);
        if (q != LOOK_UP_TABLE_DEFAULT) {
            ext[d] = cap;
            if (mode) {
                const unsigned int ql = m_uiAllElements[q].getLevel();
                if (ql + 1 == lev)
                    mode[d] = WPX_EXT_GRADED;  // coarser: read at 2H
                else if (ql == lev + 1 && !WPX_DECIMATE_FINER)
                    mode[d] = WPX_EXT_STRADDLE;
                else
                    mode[d] = WPX_EXT_COARSE;
            }
        } else {
            status |= bg ? WPX_CLIPPED_GHOST : WPX_CLIPPED_GEOMETRY;
        }
        bad_ghost = bad_ghost || bg;
    }

    // Promote to GRADED any granted direction that needs coarse data
    // anywhere in its neighbourhood.
    //
    // A direction is read at ONE spacing, so if any offset feeding it can only
    // come from a coarser element, the whole direction has to be read at 2H --
    // that is the only spacing a coarser element can land on. A same-level
    // partner is perfectly happy read at 2H too (its nodes at S+2H, S+4H, S+6H
    // are exact DOFs, just sparser), so promotion costs nothing where the
    // neighbourhood is uniform and is what lets mixed-level neighbourhoods be
    // expressed at all. Without it a graded direction could essentially never
    // coexist with the same-level directions it shares offsets with, and the
    // whole mechanism stayed inert.
    if (mode && WPX_GRADE_COARSER) {
        for (unsigned int d = 0; d < 6; d++) {
            if (!ext[d] || mode[d] == WPX_EXT_GRADED) continue;
            if (mode[d] != WPX_EXT_COARSE) continue;
            const int ax = (int)d / 2, sgn = (d % 2) ? 1 : -1;
            bool needsCoarse = false;
            for (int oz = -1; oz <= 1 && !needsCoarse; oz++)
                for (int oy = -1; oy <= 1 && !needsCoarse; oy++)
                    for (int ox = -1; ox <= 1 && !needsCoarse; ox++) {
                        const int oo[3] = {ox, oy, oz};
                        if (oo[ax] != sgn) continue;
                        bool skip = false;
                        for (int a = 0; a < 3; a++) {
                            if (!oo[a]) continue;
                            const unsigned int dd =
                                (unsigned int)(2 * a + (oo[a] > 0 ? 1 : 0));
                            if (!ext[dd]) skip = true;
                        }
                        if (skip) continue;
                        bool bg2             = false;
                        const unsigned int q2 =
                            wpxNeighbour(ele, ox, oy, oz, bg2, levelMask,
                                         dgGhostOk);
                        if (q2 == LOOK_UP_TABLE_DEFAULT) continue;
                        if (m_uiAllElements[q2].getLevel() + 1 == lev)
                            needsCoarse = true;
                    }
            if (needsCoarse) mode[d] = WPX_EXT_GRADED;
        }
    }

    // Extending two or three axes at once also needs the edge and corner
    // elements. When one is missing, give up only the single direction that
    // corner depends on -- dropping the whole axis (both directions) throws
    // away usable same-level neighbours, and measurement showed that
    // accounted for a quarter of all refused directions.
    //
    // The offending direction is chosen highest-axis-first so the outcome
    // does not depend on traversal order. At most six directions can be
    // retired, hence the bound.
    for (int guard = 0; guard < 6; guard++) {
        int bad_dir = -1;

        for (int oz = -1; oz <= 1 && bad_dir < 0; oz++) {
            if ((oz < 0 && !ext[4]) || (oz > 0 && !ext[5])) continue;
            for (int oy = -1; oy <= 1 && bad_dir < 0; oy++) {
                if ((oy < 0 && !ext[2]) || (oy > 0 && !ext[3])) continue;
                for (int ox = -1; ox <= 1 && bad_dir < 0; ox++) {
                    if ((ox < 0 && !ext[0]) || (ox > 0 && !ext[1])) continue;
                    bool bg = false;
                    const unsigned int q =
                        wpxNeighbour(ele, ox, oy, oz, bg, levelMask, dgGhostOk);

                    bool ok_here = (q != LOOK_UP_TABLE_DEFAULT) &&
                                   wpxOffsetOk(ele, ox, oy, oz, ext, q, mode);

                    // A direction's mode is taken from its face neighbour,
                    // but an edge or corner offset along the same direction
                    // can land on an element at a different level. Reading it
                    // at the wrong spacing puts inconsistent coordinates into
                    // one cube, so require every offset a direction takes
                    // part in to agree with that direction's mode.
                    if (ok_here && mode) {
                        const unsigned int ql = m_uiAllElements[q].getLevel();
                        const int oo[3] = {ox, oy, oz};
                        for (int a = 0; a < 3 && ok_here; a++) {
                            if (oo[a] == 0) continue;
                            const unsigned char md =
                                mode[2 * a + (oo[a] > 0 ? 1 : 0)];
                            if (md == WPX_EXT_STRADDLE && ql != lev + 1)
                                ok_here = false;
                            // A graded direction reads its axis at 2H. Both
                            // a same-level partner (take every second node of
                            // it) and a coarser one (take every second node of
                            // its prolongated child) land on that lattice, so
                            // either will do. Requiring coarse here was too
                            // strict and retired directions the same-level
                            // path would otherwise have granted -- it took
                            // full reach 18.6% -> 8.1%.
                            if (md == WPX_EXT_GRADED && ql != lev &&
                                ql + 1 != lev)
                                ok_here = false;
                            // A decimated direction reads every partner onto
                            // this element's lattice, so it does not care
                            // whether the partner is same-level or one finer
                            // -- both land on the same spacing. That is the
                            // whole point of decimating rather than
                            // straddling, and it is what lets a direction
                            // survive partners at mixed levels.
                            if (md == WPX_EXT_COARSE) {
                                const bool okl =
                                    (ql == lev) ||
                                    (WPX_DECIMATE_FINER && ql == lev + 1);
                                if (!okl) ok_here = false;
                            }
                        }
                    }
                    if (ok_here) continue;

                    status |= bg ? WPX_CLIPPED_GHOST : WPX_CLIPPED_GEOMETRY;
                    // Retire a GRADED direction before a same-level one.
                    // Granting a coarser direction adds offsets to check, and
                    // when one of those fails the cascade would otherwise drop
                    // a same-level direction that was perfectly good -- which
                    // made enabling coarser extension LOWER total reach
                    // (18.6% -> 8.1%) rather than raise it. Preferring the
                    // graded direction here keeps the same-level result as a
                    // floor: coarser extension can only ever add.
                    const int cand[3] = {oz != 0 ? (oz < 0 ? 4 : 5) : -1,
                                         oy != 0 ? (oy < 0 ? 2 : 3) : -1,
                                         ox != 0 ? (ox < 0 ? 0 : 1) : -1};
                    if (mode)
                        for (int c = 0; c < 3 && bad_dir < 0; c++)
                            if (cand[c] >= 0 &&
                                mode[cand[c]] == WPX_EXT_GRADED)
                                bad_dir = cand[c];
                    for (int c = 0; c < 3 && bad_dir < 0; c++)
                        if (cand[c] >= 0) bad_dir = cand[c];
                }
            }
        }

        if (bad_dir < 0) break;
        ext[bad_dir] = 0;
        if (mode) mode[bad_dir] = WPX_EXT_NONE;
    }

    return status;
}

/** Coordinate of element e along axis a (0=x,1=y,2=z). */
inline unsigned int wpxCoord(const ot::TreeNode &t, int a) {
    return (a == 0) ? t.getX() : ((a == 1) ? t.getY() : t.getZ());
}

inline unsigned int Mesh::wpxFinerPartner(unsigned int base, unsigned int ele,
                                          int a) const {
    const ot::TreeNode &E = m_uiAllElements[ele];
    const unsigned int S  = 1u << (m_uiMaxDepth - E.getLevel());
    const unsigned int c0 = wpxCoord(E, a);
    const unsigned int lv = m_uiAllElements[base].getLevel();

    for (int sgn = 1; sgn >= -1; sgn -= 2) {
        const unsigned int d  = (unsigned int)(2 * a + (sgn > 0 ? 1 : 0));
        const unsigned int nb = m_uiE2EMapping[base * m_uiNumDirections + d];
        if (nb == LOOK_UP_TABLE_DEFAULT || nb >= m_uiAllElements.size())
            continue;
        if (m_uiAllElements[nb].getLevel() != lv) continue;
        if (!m_uiIsNodalMapValid[nb]) continue;
        const unsigned int cn = wpxCoord(m_uiAllElements[nb], a);
        if (cn >= c0 && cn < c0 + S) return nb;
    }
    return LOOK_UP_TABLE_DEFAULT;
}

inline bool Mesh::wpxOffsetOk(unsigned int ele, int ox, int oy, int oz,
                              const unsigned int ext[6], unsigned int q,
                              const unsigned char *mode) const {
    if (q == LOOK_UP_TABLE_DEFAULT || q >= m_uiAllElements.size()) return false;

    const ot::TreeNode &E = m_uiAllElements[ele];
    const ot::TreeNode &Q = m_uiAllElements[q];
    const long S  = 1l << (m_uiMaxDepth - E.getLevel());
    const long Sq = 1l << (m_uiMaxDepth - Q.getLevel());
    const int o[3] = {ox, oy, oz};

    // does this offset feed a graded axis at all?
    bool graded_off = false;
    if (mode)
        for (int a = 0; a < 3; a++)
            if (o[a] &&
                mode[2 * a + (o[a] > 0 ? 1 : 0)] == WPX_EXT_GRADED)
                graded_off = true;

    for (int a = 0; a < 3; a++) {
        const long c0 = (long)wpxCoord(E, a);
        const long cq = (long)wpxCoord(Q, a);

        // a graded direction reads its neighbour at 2H, so it reaches twice
        // as far for the same node count
        const long gl =
            (mode && mode[2 * a] == WPX_EXT_GRADED) ? 2l : 1l;
        const long gh =
            (mode && mode[2 * a + 1] == WPX_EXT_GRADED) ? 2l : 1l;

        long lo, hi;  // region this offset must supply, on this axis
        if (o[a] < 0) {
            const long reach =
                (gl * S * (long)ext[2 * a]) / (long)m_uiElementOrder;
            lo = c0 - reach;
            hi = c0;
        } else if (o[a] > 0) {
            const long reach =
                (gh * S * (long)ext[2 * a + 1]) / (long)m_uiElementOrder;
            lo = c0 + S;
            hi = c0 + S + reach;
        } else {
            lo = c0;
            hi = c0 + S;
        }
        if (hi <= lo) return false;
        // q must overlap the region, not merely be reachable by a hop chain
        if (cq + Sq <= lo || cq >= hi) return false;
        // A graded read takes a whole S-cube from this offset -- either the
        // partner's own nodes at stride 2, or its prolongated child's. Overlap
        // is not enough there: q must CONTAIN that cube, on every axis. This
        // is also what rejects a multi-hop walk that overshot, which is easy
        // once a coarser intermediate makes a hop move 2S instead of S.
        if (mode && graded_off) {
            const long cubeLo = c0 + (long)o[a] * S;
            if (cq > cubeLo || cq + Sq < cubeLo + S) return false;
        }
    }
    return true;
}

/**
 * Shared slab copy behind both gather wrappers. `fetch(e, buf)` must return a
 * pointer to element e's m_uiNpE nodal values, using buf as scratch if it has
 * to materialise them.
 *
 * Walks the (up to) 27 contributing elements and copies each one's slab,
 * rather than resolving a neighbour per node, so an element's values are
 * fetched at most once.
 */
template <typename T, typename FetchFn>
void Mesh::gatherExtendedCoarseImpl(unsigned int ele, const unsigned int ext[6],
                                    T *out, T *eleScratch, FetchFn fetch,
                                    const unsigned char *mode, bool dgGhostOk,
                                    const T *gslab,
                                    const unsigned char *gvalid) const {
    const unsigned int p   = m_uiElementOrder;
    const unsigned int nrp = p + 1;

    const unsigned int nx  = nrp + ext[0] + ext[1];
    const unsigned int ny  = nrp + ext[2] + ext[3];
    const unsigned int nz  = nrp + ext[4] + ext[5];

    for (int oz = -1; oz <= 1; oz++) {
        int gz0, gz1;
        if (oz < 0) {
            if (!ext[4]) continue;
            gz0 = -(int)ext[4]; gz1 = -1;
        } else if (oz > 0) {
            if (!ext[5]) continue;
            gz0 = (int)p + 1; gz1 = (int)p + (int)ext[5];
        } else {
            gz0 = 0; gz1 = (int)p;
        }

        for (int oy = -1; oy <= 1; oy++) {
            int gy0, gy1;
            if (oy < 0) {
                if (!ext[2]) continue;
                gy0 = -(int)ext[2]; gy1 = -1;
            } else if (oy > 0) {
                if (!ext[3]) continue;
                gy0 = (int)p + 1; gy1 = (int)p + (int)ext[3];
            } else {
                gy0 = 0; gy1 = (int)p;
            }

            for (int ox = -1; ox <= 1; ox++) {
                int gx0, gx1;
                if (ox < 0) {
                    if (!ext[0]) continue;
                    gx0 = -(int)ext[0]; gx1 = -1;
                } else if (ox > 0) {
                    if (!ext[1]) continue;
                    gx0 = (int)p + 1; gx1 = (int)p + (int)ext[1];
                } else {
                    gx0 = 0; gx1 = (int)p;
                }

                // A coarser partner cannot be read directly -- it samples
                // our lattice every other node. Its contribution arrives
                // pre-materialised as its own prolongated child, on our
                // lattice, and we take every second node of that (stride 2),
                // which is what puts the extension at 2H.
                const int sidx = (oz + 1) * 9 + (oy + 1) * 3 + (ox + 1);
                if (gslab && gvalid && gvalid[sidx]) {
                    const T *src = gslab + (size_t)sidx * m_uiNpE;
                    auto gmap    = [&](int g, int o, unsigned char md) {
                        const int st = (md == WPX_EXT_GRADED) ? 2 : 1;
                        if (o < 0) return (int)p + st * g;
                        if (o > 0) return st * (g - (int)p);
                        return g;
                    };
                    const unsigned char mdx =
                        mode ? mode[ox < 0 ? 0 : 1] : WPX_EXT_COARSE;
                    const unsigned char mdy =
                        mode ? mode[oy < 0 ? 2 : 3] : WPX_EXT_COARSE;
                    const unsigned char mdz =
                        mode ? mode[oz < 0 ? 4 : 5] : WPX_EXT_COARSE;
                    for (int gz = gz0; gz <= gz1; gz++) {
                        const int lk = gmap(gz, oz, mdz);
                        for (int gy = gy0; gy <= gy1; gy++) {
                            const int lj = gmap(gy, oy, mdy);
                            for (int gx = gx0; gx <= gx1; gx++) {
                                const int li = gmap(gx, ox, mdx);
                                out[(size_t)((gz + (int)ext[4]) * ny +
                                             (gy + (int)ext[2])) *
                                        nx +
                                    (size_t)(gx + (int)ext[0])] =
                                    src[(size_t)(lk * nrp + lj) * nrp + li];
                            }
                        }
                    }
                    continue;
                }

                bool bg              = false;
                const unsigned int e =
                    // The gather resolves whatever the probe was willing to
                    // grant; policy lives in the probe's level mask, not here.
                    // Harmless when ext is zero for this direction, since the
                    // offset is skipped above.
                    wpxNeighbour(ele, ox, oy, oz, bg,
                                 WPX_LVL_SAME | WPX_LVL_FINER, dgGhostOk);
                if (e == LOOK_UP_TABLE_DEFAULT) continue;
                if (!wpxOffsetOk(ele, ox, oy, oz, ext, e, mode)) continue;

                const unsigned int lev  = m_uiAllElements[ele].getLevel();
                const bool finer = (m_uiAllElements[e].getLevel() == lev + 1);

                if (!finer) {
                    const T *src = fetch(e, eleScratch);
                    for (int gz = gz0; gz <= gz1; gz++) {
                        const unsigned int lk =
                            (unsigned int)(gz - oz * (int)p);
                        for (int gy = gy0; gy <= gy1; gy++) {
                            const unsigned int lj =
                                (unsigned int)(gy - oy * (int)p);
                            for (int gx = gx0; gx <= gx1; gx++) {
                                const unsigned int li =
                                    (unsigned int)(gx - ox * (int)p);
                                out[(size_t)((gz + (int)ext[4]) * ny +
                                             (gy + (int)ext[2])) *
                                        nx +
                                    (size_t)(gx + (int)ext[0])] =
                                    src[(size_t)(lk * nrp + lj) * nrp + li];
                            }
                        }
                    }
                    continue;
                }

                // Finer neighbour: its node p-2i (or 2i on the far side) sits
                // exactly on this element's lattice position i*H, so the
                // decimation is an index map and introduces no interpolation.
                // It spans only half of this element per axis though, so the
                // axes with a zero offset need a second sub-element each.
                const int half = (int)p / 2;
                unsigned int sub[2][2][2];
                for (int a = 0; a < 8; a++)
                    (&sub[0][0][0])[a] = LOOK_UP_TABLE_DEFAULT;

                {
                    // index the base by where it sits inside this element
                    const ot::TreeNode &E = m_uiAllElements[ele];
                    const unsigned int S =
                        1u << (m_uiMaxDepth - E.getLevel());
                    auto slot = [&](unsigned int q, int a) {
                        if ((a == 0 && ox) || (a == 1 && oy) || (a == 2 && oz))
                            return 0;
                        return (wpxCoord(m_uiAllElements[q], a) >=
                                wpxCoord(E, a) + S / 2)
                                   ? 1
                                   : 0;
                    };

                    std::vector<unsigned int> seeds;
                    seeds.push_back(e);
                    for (int a = 0; a < 3; a++) {
                        if ((a == 0 && ox) || (a == 1 && oy) ||
                            (a == 2 && oz))
                            continue;
                        const size_t n0 = seeds.size();
                        for (size_t t = 0; t < n0; t++) {
                            const unsigned int q =
                                wpxFinerPartner(seeds[t], ele, a);
                            if (q != LOOK_UP_TABLE_DEFAULT)
                                seeds.push_back(q);
                        }
                    }
                    for (unsigned int q : seeds)
                        sub[slot(q, 2)][slot(q, 1)][slot(q, 0)] = q;
                }

                bool complete = true;
                for (int sz = 0; sz <= (oz ? 0 : 1) && complete; sz++)
                    for (int sy = 0; sy <= (oy ? 0 : 1) && complete; sy++)
                        for (int sx = 0; sx <= (ox ? 0 : 1); sx++)
                            if (sub[sz][sy][sx] == LOOK_UP_TABLE_DEFAULT)
                                complete = false;
                if (!complete) continue;

                // materialise each sub-element once
                const T *ssrc[2][2][2] = {};
                {
                    int slotn = 0;
                    for (int sz = 0; sz <= (oz ? 0 : 1); sz++)
                        for (int sy = 0; sy <= (oy ? 0 : 1); sy++)
                            for (int sx = 0; sx <= (ox ? 0 : 1); sx++) {
                                ssrc[sz][sy][sx] = fetch(
                                    sub[sz][sy][sx],
                                    eleScratch
                                        ? eleScratch + (size_t)slotn * m_uiNpE
                                        : (T *)nullptr);
                                slotn++;
                            }
                }

                // g -> (sub index, local index) on one axis.
                //
                // Along an extended axis the finer neighbour is read either
                // decimated (every other node, landing on this element's
                // lattice) or at its own fine spacing for a straddle, where
                // the extra nodes sit at H/2 and centre the stencil. Across
                // the transverse axes it is always decimated, which is what
                // keeps the gathered cube rectangular.
                auto mapax = [&](int g, int o, unsigned char md) {
                    int si = 0, li = 0;
                    const int step = (md == WPX_EXT_STRADDLE) ? 1 : 2;
                    if (o < 0)
                        li = (int)p - step * (-g);
                    else if (o > 0)
                        li = step * (g - (int)p);
                    else {
                        si = (g > half) ? 1 : 0;
                        li = 2 * (g - si * half);
                    }
                    return std::make_pair(si, li);
                };
                const unsigned char mdx =
                    mode ? mode[ox < 0 ? 0 : 1] : WPX_EXT_COARSE;
                const unsigned char mdy =
                    mode ? mode[oy < 0 ? 2 : 3] : WPX_EXT_COARSE;
                const unsigned char mdz =
                    mode ? mode[oz < 0 ? 4 : 5] : WPX_EXT_COARSE;

                for (int gz = gz0; gz <= gz1; gz++) {
                    const auto mz = mapax(gz, oz, mdz);
                    for (int gy = gy0; gy <= gy1; gy++) {
                        const auto my = mapax(gy, oy, mdy);
                        for (int gx = gx0; gx <= gx1; gx++) {
                            const auto mx = mapax(gx, ox, mdx);
                            out[(size_t)((gz + (int)ext[4]) * ny +
                                         (gy + (int)ext[2])) *
                                    nx +
                                (size_t)(gx + (int)ext[0])] =
                                ssrc[mz.first][my.first][mx.first]
                                    [(size_t)(mz.second * nrp + my.second) *
                                         nrp +
                                     mx.second];
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
void Mesh::exchangeWideProlongDGBegin(const T *allDg, size_t eleStride,
                                      unsigned int dof, size_t dgSz) const {
    this->buildWideProlongGhostMap();
    m_uiWpxInFlight = false;
    if (!m_uiIsActive || m_uiActiveNpes == 1) return;

    const unsigned int npes = m_uiActiveNpes;
    const size_t nPe        = m_uiNpE;
    const size_t pay        = (size_t)dof * nPe;
    const size_t esz        = sizeof(T);

    m_uiWpxSendCntB.assign(npes, 0);
    m_uiWpxSendOffB.assign(npes, 0);
    m_uiWpxRecvCntB.assign(npes, 0);
    m_uiWpxRecvOffB.assign(npes, 0);
    for (unsigned int p = 0; p < npes; p++) {
        m_uiWpxSendCntB[p] = (int)(m_uiWpxSendCount[p] * pay * esz);
        m_uiWpxRecvCntB[p] = (int)(m_uiWpxRecvCount[p] * pay * esz);
    }
    for (unsigned int p = 1; p < npes; p++) {
        m_uiWpxSendOffB[p] = m_uiWpxSendOffB[p - 1] + m_uiWpxSendCntB[p - 1];
        m_uiWpxRecvOffB[p] = m_uiWpxRecvOffB[p - 1] + m_uiWpxRecvCntB[p - 1];
    }

    m_uiWpxSendBuf.resize(m_uiWpxSendEle.size() * pay * esz);
    m_uiWpxRecvBuf.resize(m_uiWpxRecvEle.size() * pay * esz);

    // pack out of allDg -- the caller has already materialised every local
    // element, and that call (with its wide hanging-face reconstruction) is
    // the expensive one
    T *sb = reinterpret_cast<T *>(m_uiWpxSendBuf.data());
#pragma omp parallel for schedule(static)
    for (long i = 0; i < (long)m_uiWpxSendEle.size(); i++) {
        T *dst       = sb + (size_t)i * pay;
        const T *src = allDg + (size_t)m_uiWpxSendEle[i] * eleStride;
        for (unsigned int v = 0; v < dof; v++)
            for (size_t k = 0; k < nPe; k++)
                dst[(size_t)v * nPe + k] = src[(size_t)v * dgSz + k];
    }

    const bool wpx_prof2 = (std::getenv("DENDRO_WPX_PROFILE") != nullptr);
    const double tp0     = wpx_prof2 ? MPI_Wtime() : 0.0;
    MPI_Ialltoallv(m_uiWpxSendBuf.empty() ? nullptr : m_uiWpxSendBuf.data(),
                   m_uiWpxSendCntB.data(), m_uiWpxSendOffB.data(), MPI_BYTE,
                   m_uiWpxRecvBuf.empty() ? nullptr : m_uiWpxRecvBuf.data(),
                   m_uiWpxRecvCntB.data(), m_uiWpxRecvOffB.data(), MPI_BYTE,
                   m_uiCommActive, &m_uiWpxReq);
    m_uiWpxInFlight = true;
    if (wpx_prof2)
        std::printf(
            "[wpx-prof rank %d]   pack: sendEle %zu (%.2f MiB) recvEle %zu | "
            "MPI_Ialltoallv call %.2f ms\n",
            m_uiActiveRank, m_uiWpxSendEle.size(),
            (double)m_uiWpxSendBuf.size() / (1024.0 * 1024.0),
            m_uiWpxRecvEle.size(), (MPI_Wtime() - tp0) * 1e3);
}

template <typename T>
void Mesh::exchangeWideProlongDGEnd(T *allDg, size_t eleStride,
                                    unsigned int dof, size_t dgSz) const {
    if (!m_uiWpxInFlight) return;
    MPI_Wait(&m_uiWpxReq, MPI_STATUS_IGNORE);
    m_uiWpxInFlight = false;

    const size_t nPe = m_uiNpE;
    const size_t pay = (size_t)dof * nPe;
    const T *rb      = reinterpret_cast<const T *>(m_uiWpxRecvBuf.data());
#pragma omp parallel for schedule(static)
    for (long i = 0; i < (long)m_uiWpxRecvEle.size(); i++) {
        const T *src = rb + (size_t)i * pay;
        T *dst       = allDg + (size_t)m_uiWpxRecvEle[i] * eleStride;
        for (unsigned int v = 0; v < dof; v++)
            for (size_t k = 0; k < nPe; k++)
                dst[(size_t)v * dgSz + k] = src[(size_t)v * nPe + k];
    }
}

template <typename T>
void Mesh::exchangeWideProlongDG(const T *cg, size_t cgSz, T *allDg,
                                 size_t eleStride, unsigned int dof,
                                 size_t dgSz) const {
    this->buildWideProlongGhostMap();
    if (!m_uiIsActive || m_uiActiveNpes == 1) return;

    const unsigned int npes = m_uiActiveNpes;
    const size_t nPe        = m_uiNpE;
    const size_t pay        = (size_t)dof * nPe;  // doubles per element

    std::vector<int> sc(npes), so(npes), rc(npes), ro(npes);
    for (unsigned int p = 0; p < npes; p++) {
        sc[p] = (int)(m_uiWpxSendCount[p] * pay);
        rc[p] = (int)(m_uiWpxRecvCount[p] * pay);
    }
    so[0] = 0;
    ro[0] = 0;
    for (unsigned int p = 1; p < npes; p++) {
        so[p] = so[p - 1] + sc[p - 1];
        ro[p] = ro[p - 1] + rc[p - 1];
    }

    std::vector<T> sbuf((size_t)m_uiWpxSendEle.size() * pay);
    std::vector<T> rbuf((size_t)m_uiWpxRecvEle.size() * pay);

    // Pack on the owner, where the element is LOCAL and so the wide
    // hanging-face reconstruction has everything it needs -- that is why the
    // exchange terminates rather than needing round 3.
    //
    // Copy out of allDg rather than recomputing: the caller has already run
    // getElementNodalValues over every local element, and that call is the
    // expensive one (it does the wide hanging-face reconstruction). Packing
    // by recomputation doubled the work for nothing.
    (void)cg;
    (void)cgSz;
#pragma omp parallel for schedule(static)
    for (long i = 0; i < (long)m_uiWpxSendEle.size(); i++) {
        T *dst       = sbuf.data() + (size_t)i * pay;
        const T *src = allDg + (size_t)m_uiWpxSendEle[i] * eleStride;
        for (unsigned int v = 0; v < dof; v++)
            for (size_t k = 0; k < nPe; k++)
                dst[(size_t)v * nPe + k] = src[(size_t)v * dgSz + k];
    }

    par::Mpi_Alltoallv(sbuf.empty() ? nullptr : sbuf.data(), sc.data(),
                       so.data(), rbuf.empty() ? nullptr : rbuf.data(),
                       rc.data(), ro.data(), m_uiCommActive);

    for (size_t i = 0; i < m_uiWpxRecvEle.size(); i++) {
        const T *src = rbuf.data() + i * pay;
        T *dst       = allDg + (size_t)m_uiWpxRecvEle[i] * eleStride;
        for (unsigned int v = 0; v < dof; v++)
            for (size_t k = 0; k < nPe; k++)
                dst[(size_t)v * dgSz + k] = src[(size_t)v * nPe + k];
    }
}

template <typename T>
void Mesh::gatherExtendedCoarseNodes(const T *dgVec, unsigned int ele,
                                     const unsigned int ext[6], T *out,
                                     const unsigned char *mode) const {
    gatherExtendedCoarseImpl<T>(
        ele, ext, out, (T *)nullptr,
        [&](unsigned int e, T *) { return dgVec + (size_t)e * m_uiNpE; }, mode);
}

template <typename T>
void Mesh::gatherExtendedCoarseNodesDG(const T *dgVec, size_t ele_stride,
                                       size_t var_offset, unsigned int ele,
                                       const unsigned int ext[6], T *out,
                                       const unsigned char *mode,
                                       const T *gslab,
                                       const unsigned char *gvalid) const {
    gatherExtendedCoarseImpl<T>(ele, ext, out, (T *)nullptr,
                                [&](unsigned int e, T *) {
                                    return dgVec + (size_t)e * ele_stride +
                                           var_offset;
                                },
                                mode, /*dgGhostOk=*/true, gslab, gvalid);
}

template <typename T>
void Mesh::gatherExtendedCoarseNodesCG(const T *cgVec, unsigned int ele,
                                       const unsigned int ext[6], T *out,
                                       T *eleScratch, double *im1, double *im2,
                                       bool allowWide,
                                       const unsigned char *mode) const {
    // NOTE: a neighbour with hanging faces has those faces filled by the
    // narrow operator inside getElementNodalValues, so widening the pad does
    // not by itself fix a neighbour's own hanging nodes. That is the separate
    // getElementNodalValues work item.
    gatherExtendedCoarseImpl<T>(ele, ext, out, eleScratch,
                                [&](unsigned int e, T *buf) {
                                    this->getElementNodalValues(cgVec, buf, e,
                                                                false, im1,
                                                                im2, allowWide);
                                    return (const T *)buf;
                                },
                                mode);
}

/**
 * Fill one child's (m_uiElementOrder+1)^3 nodes, for every dof, from coarse
 * element `ele`.
 *
 * Without DENDRO_WIDE_PROLONGATION this is exactly today's call and nothing
 * else runs. With it, the child is built from a stencil that reaches into the
 * neighbouring coarse elements wherever they exist, and falls back to the
 * narrow operator per axis where they do not.
 *
 * @param out per-dof child nodes, variable v at out + v*m_uiNpE.
 */
/**
 * Extended node coordinates for one axis, in parent-element units (the parent
 * spans [0,1], its own nodes at j/p).
 *
 * A coarse extension adds nodes one element-spacing apart; a straddle adds
 * them at half that, because they come from a finer neighbour. Mixing the two
 * makes the array graded, which is exactly what build_1d_at exists for.
 */
inline void wpxAxisCoords(unsigned int p, unsigned int lo, unsigned char mlo,
                          unsigned int hi, unsigned char mhi,
                          std::vector<double> &xs) {
    xs.clear();
    // A finer neighbour is read at half our spacing (straddle); a coarser one
    // is read at twice it (graded, every second node of its prolongated
    // child). See the WPX_EXT_GRADED comment for why graded beats uniform
    // here even though both are degree-9 exact.
    auto spacing = [](unsigned char m) {
        if (m == ot::Mesh::WPX_EXT_STRADDLE) return 0.5;
        if (m == ot::Mesh::WPX_EXT_GRADED) return 2.0;
        return 1.0;
    };
    const double dl = spacing(mlo);
    const double dh = spacing(mhi);

    for (int i = (int)lo; i >= 1; i--)
        xs.push_back(-dl * (double)i / (double)p);
    for (unsigned int j = 0; j <= p; j++)
        xs.push_back((double)j / (double)p);
    for (unsigned int i = 1; i <= hi; i++)
        xs.push_back(1.0 + dh * (double)i / (double)p);
}

template <typename T>
void Mesh::prolongateChildNodesNarrow(const T *dgEle, size_t dgSz,
                                      unsigned int cnum, unsigned int dof,
                                      T *out, double *im1, double *im2) const {
    // A null scratch pair means the caller wants RefElement's shared
    // im_vec1/im_vec2, as the serial path did before it routed through here.
    for (unsigned int v = 0; v < dof; v++) {
        if (im1 != nullptr && im2 != nullptr)
            this->parent2ChildInterpolation(dgEle + v * dgSz,
                                            out + v * m_uiNpE, cnum, m_uiDim,
                                            im1, im2);
        else
            this->parent2ChildInterpolation(dgEle + v * dgSz,
                                            out + v * m_uiNpE, cnum, m_uiDim);
    }
}

template <typename T>
const T *Mesh::wpxProlongChildMemo(const T *in, size_t cgSz, unsigned int ele,
                                   unsigned int cnum, unsigned int dof,
                                   size_t dgSz, const T *allDg,
                                   size_t allDgEleStride,
                                   unsigned int lvlMask, double *im1,
                                   double *im2) const {
    // key on the array's identity plus everything the value depends on; the
    // epoch (ticked by unzip) retires entries when contents may have changed
    struct Key {
        const void *dg;
        std::uint64_t k;
        bool operator==(const Key &o) const { return dg == o.dg && k == o.k; }
    };
    struct KeyHash {
        size_t operator()(const Key &x) const {
            return std::hash<const void *>()(x.dg) ^
                   (std::hash<std::uint64_t>()(x.k) * 0x9e3779b97f4a7c15ull);
        }
    };
    static thread_local std::unordered_map<Key, std::vector<T>, KeyHash> memo;
    static thread_local std::uint64_t memo_epoch = 0;
    static thread_local size_t memo_elems       = 0;
    // ~16 MiB of cached values per thread; past that, clear and refill. SFC
    // locality keeps the working set of nearby owners hot either way, and a
    // miss only costs the recompute the memo exists to skip.
    constexpr size_t MEMO_ELEM_CAP = (size_t)2 * 1024 * 1024;

    if (memo_epoch != m_uiWpxMemoEpoch) {
        memo.clear();
        memo_elems = 0;
        memo_epoch = m_uiWpxMemoEpoch;
    }

    const Key key{allDg, ((std::uint64_t)ele << 12) |
                             ((std::uint64_t)(cnum & 7u) << 9) |
                             ((std::uint64_t)(lvlMask & 7u) << 6) |
                             (std::uint64_t)(dof & 63u)};
    auto it = memo.find(key);
    if (it != memo.end()) return it->second.data();

    std::vector<T> buf((size_t)dof * m_uiNpE);
    this->prolongateChildNodes(in, cgSz,
                               allDg + (size_t)ele * allDgEleStride, dgSz,
                               ele, cnum, dof, buf.data(), im1, im2, allDg,
                               allDgEleStride, lvlMask);

    if (memo_elems + buf.size() > MEMO_ELEM_CAP) {
        memo.clear();
        memo_elems = 0;
    }
    memo_elems += buf.size();
    return memo.emplace(key, std::move(buf)).first->second.data();
}

template <typename T>
void Mesh::prolongateChildNodes(const T *in, size_t cgSz, const T *dgEle,
                                size_t dgSz, unsigned int ele,
                                unsigned int cnum, unsigned int dof, T *out,
                                double *im1, double *im2, const T *allDg,
                                size_t allDgEleStride,
                                unsigned int lvlMask) const {
#ifdef DENDRO_WIDE_PROLONGATION
    if (!m_uiWpxRuntimeEnabled) {
        this->prolongateChildNodesNarrow(dgEle, dgSz, cnum, dof, out, im1, im2);
        return;
    }
    const unsigned int nrp   = m_uiElementOrder + 1;
    const unsigned int width = dendro::wideprolong::stencil_width(
        m_uiElementOrder);
    const unsigned int want  = (width > nrp) ? (width - nrp) : 0u;

    unsigned int ext[6];
    unsigned char emode[6];
    // With a whole-mesh DG array in hand, an element without a valid nodal
    // map is still readable -- its slice came from the rank that owns it.
    const bool dgGhostOk = (allDg != nullptr);
    const unsigned int st = this->probeCoarseExtension(
        ele, want, ext, lvlMask, emode, dgGhostOk);

    if (st & WPX_CLIPPED_GHOST) {
        // The stencil would have reached a round-2 ghost, so whether it is
        // available depends on the partition. Degrading quietly here would
        // make the answer a function of the rank count, so refuse instead.
        std::cerr << "[wide prolongation] element " << ele
                  << " needs a round-2 ghost neighbour; the nodal ghost layer "
                     "only covers round 1, so the wide stencil is not "
                     "available under this partition. Rebuild with "
                     "DENDRO_WIDE_PROLONGATION=OFF or run on fewer ranks."
                  << std::endl;
        MPI_Abort(m_uiCommGlobal, 1);
    }

    for (int a = 0; a < 3; a++) {
        const unsigned int tot = ext[2 * a] + ext[2 * a + 1];
        if (tot < want) {
            wpxLogFallbackOnce("unzip coarse->fine", tot, want);
            break;
        }
    }

    if (ext[0] || ext[1] || ext[2] || ext[3] || ext[4] || ext[5]) {
        static thread_local std::vector<double> opx, opy, opz;
        static thread_local std::vector<T> cube, eleScratch;
        static thread_local std::vector<double> w1, w2;

        // Materialise the coarse partners' prolongated children. A coarser
        // element samples our lattice every other node, so it cannot be read
        // directly; its own child lands ON our lattice, and taking every
        // second node of that child gives the graded 2H extension. The nested
        // call is given a mask WITHOUT COARSER, so it cannot recurse.
        // Per-call, NOT thread_local: the graded path calls this function
        // recursively (once per coarse partner, to prolongate its child), and
        // a shared buffer would let the nested call stomp the outer one's
        // slabs. It did exactly that -- the nested call, which never has a
        // graded direction, ran the clear below and emptied the outer gvalid,
        // so the gather saw no slabs and left the extension unfilled.
        std::vector<T> gslab;
        std::vector<unsigned char> gvalid;
        bool anyGraded = false;
        for (int d = 0; d < 6; d++)
            if (ext[d] && emode[d] == WPX_EXT_GRADED) anyGraded = true;

        // Clear it when this element has no graded direction. gslab/gvalid are
        // thread_local and reused, so leaving a previous element's flags set
        // makes the gather below read ITS slabs -- which is exactly what broke
        // the linear-field control (1.03e-01 at a coarsest-level element that
        // cannot have a graded direction at all).
        if (!(anyGraded && allDg != nullptr)) gvalid.clear();

        if (anyGraded && allDg != nullptr) {
            // Per dof: the slab is DATA, and reusing variable 0's slab for
            // every v silently mixed variables at any graded pad. Layout
            // [v][sidx][node], so the per-v gather below can be handed a
            // contiguous 27-slab block.
            gslab.assign((size_t)dof * 27 * m_uiNpE, T(0));
            gvalid.assign(27, 0);
            const ot::TreeNode &E = m_uiAllElements[ele];
            const long S = 1l << (m_uiMaxDepth - E.getLevel());
            for (int oz = -1; oz <= 1; oz++)
                for (int oy = -1; oy <= 1; oy++)
                    for (int ox = -1; ox <= 1; ox++) {
                        if (!ox && !oy && !oz) continue;
                        // An offset needs this path as soon as ANY axis it
                        // feeds is graded: that axis is read at stride 2, and
                        // the normal gather only knows stride 1.
                        const int oo[3] = {ox, oy, oz};
                        bool need = false, ok = true;
                        for (int a = 0; a < 3 && ok; a++) {
                            if (!oo[a]) continue;
                            const unsigned int d =
                                (unsigned int)(2 * a + (oo[a] > 0 ? 1 : 0));
                            if (!ext[d]) ok = false;
                            else if (emode[d] == WPX_EXT_GRADED) need = true;
                        }
                        if (!ok || !need) continue;
                        bool bg = false;
                        const unsigned int q = wpxNeighbour(
                            ele, ox, oy, oz, bg, lvlMask, dgGhostOk);
                        static const bool gdbg =
                            (std::getenv("DENDRO_WPX_HOLES") != nullptr);
                        if (q == LOOK_UP_TABLE_DEFAULT) {
                            if (gdbg)
                                std::printf("[gmat] ele %u off(%d,%d,%d): "
                                            "neighbour unresolved\n",
                                            ele, ox, oy, oz);
                            continue;
                        }

                        // The S-cube this offset must supply, on our lattice.
                        const ot::TreeNode &Q = m_uiAllElements[q];
                        long cube[3];
                        for (int a = 0; a < 3; a++)
                            cube[a] = (long)wpxCoord(E, a) + (long)oo[a] * S;

                        const unsigned int ql = Q.getLevel();
                        const int sidx =
                            (oz + 1) * 9 + (oy + 1) * 3 + (ox + 1);

                        if (ql == E.getLevel()) {
                            // Same level: the partner IS that cube, so its own
                            // nodes are already on our lattice and the stride-2
                            // read happens in the gather. Verify geometrically
                            // -- a multi-hop walk through a coarser
                            // intermediate moves 2S per hop and overshoots.
                            bool aligned = true;
                            for (int a = 0; a < 3; a++)
                                if ((long)wpxCoord(Q, a) != cube[a])
                                    aligned = false;
                            if (!aligned) continue;
                            const T *src =
                                allDg + (size_t)q * allDgEleStride;
                            for (unsigned int v = 0; v < dof; v++)
                                std::copy(src + (size_t)v * dgSz,
                                          src + (size_t)v * dgSz + m_uiNpE,
                                          gslab.begin() +
                                              ((size_t)v * 27 + sidx) *
                                                  m_uiNpE);
                            gvalid[sidx] = 1;
                            continue;
                        }

                        if (ql + 1 != E.getLevel()) {
                            if (gdbg)
                                std::printf("[gmat] ele %u off(%d,%d,%d): "
                                            "partner lvl %u vs ours %u\n",
                                            ele, ox, oy, oz, ql,
                                            E.getLevel());
                            continue;
                        }

                        // Coarser: its nodes sample our lattice every OTHER
                        // node, so read its prolongated child instead -- that
                        // lands on our lattice, and every second node of it is
                        // the graded 2H extension.
                        unsigned int cn = 0;
                        bool inside     = true;
                        for (int a = 0; a < 3; a++) {
                            const long rel = cube[a] - (long)wpxCoord(Q, a);
                            if (rel != 0 && rel != S) { inside = false; break; }
                            if (rel == S) cn |= (1u << a);
                        }
                        if (!inside) {
                            if (gdbg)
                                std::printf("[gmat] ele %u off(%d,%d,%d): cube "
                                            "not a child of q (E %u,%u,%u S %ld"
                                            " | Q %u,%u,%u)\n",
                                            ele, ox, oy, oz, E.getX(), E.getY(),
                                            E.getZ(), S, Q.getX(), Q.getY(),
                                            Q.getZ());
                            continue;
                        }
                        // memoised: the same partner child is requested by
                        // every fine element that borders it. Copy out
                        // immediately -- the pointer dies on the next call.
                        const T *cb = this->wpxProlongChildMemo(
                            in, cgSz, q, cn, dof, dgSz, allDg,
                            allDgEleStride,
                            lvlMask & ~(unsigned int)WPX_LVL_COARSER, im1,
                            im2);
                        for (unsigned int v = 0; v < dof; v++)
                            std::copy(cb + (size_t)v * m_uiNpE,
                                      cb + (size_t)(v + 1) * m_uiNpE,
                                      gslab.begin() +
                                          ((size_t)v * 27 + sidx) * m_uiNpE);
                        gvalid[sidx] = 1;
                    }
        }

        static thread_local std::vector<double> cx, cy, cz;
        wpxAxisCoords(m_uiElementOrder, ext[0], emode[0], ext[1], emode[1], cx);
        wpxAxisCoords(m_uiElementOrder, ext[2], emode[2], ext[3], emode[3], cy);
        wpxAxisCoords(m_uiElementOrder, ext[4], emode[4], ext[5], emode[5], cz);

        const unsigned int nx_in = (unsigned int)cx.size();
        const unsigned int ny_in = (unsigned int)cy.size();
        const unsigned int nz_in = (unsigned int)cz.size();

        dendro::wideprolong::build_1d_at(m_uiElementOrder, (cnum >> 0u) & 1u,
                                         cx, width, opx);
        dendro::wideprolong::build_1d_at(m_uiElementOrder, (cnum >> 1u) & 1u,
                                         cy, width, opy);
        dendro::wideprolong::build_1d_at(m_uiElementOrder, (cnum >> 2u) & 1u,
                                         cz, width, opz);

        const size_t ss = dendro::wideprolong::scratch_size(
            m_uiElementOrder, nx_in, ny_in, nz_in);
        // Zeroed, not merely resized: this buffer is thread_local and reused
        // across calls, so any region the gather cannot fill would otherwise
        // be read as stale data from a previous element. A hole is a bug in
        // the probe, but it must not present as plausible garbage.
        static const bool holes_dbg =
            (std::getenv("DENDRO_WPX_HOLES") != nullptr);
        cube.assign((size_t)nx_in * ny_in * nz_in,
                    holes_dbg ? std::numeric_limits<T>::quiet_NaN() : T(0));
        eleScratch.resize((size_t)8 * m_uiNpE);
        w1.resize(ss);
        w2.resize(ss);

        static thread_local std::vector<double> im1_own, im2_own;
        if (im1 == nullptr || im2 == nullptr) {
            im1_own.resize(m_uiNpE);
            im2_own.resize(m_uiNpE);
        }
        double *const g_im1 = (im1 != nullptr) ? im1 : im1_own.data();
        double *const g_im2 = (im2 != nullptr) ? im2 : im2_own.data();

        for (unsigned int v = 0; v < dof; v++) {
            // Prefer the caller's whole-mesh DG array: it was filled by
            // getElementNodalValues with widening enabled, so its hanging
            // faces are already corrected. Re-gathering from CG here would
            // both redo that work and feed the stencil narrow face values.
            if (allDg != nullptr)
                this->gatherExtendedCoarseNodesDG(
                    allDg, allDgEleStride, (size_t)v * dgSz, ele, ext,
                    cube.data(), emode,
                    gvalid.empty()
                        ? nullptr
                        : gslab.data() + (size_t)v * 27 * m_uiNpE,
                    gvalid.empty() ? nullptr : gvalid.data());
            else
                this->gatherExtendedCoarseNodesCG(in + v * cgSz, ele, ext,
                                                  cube.data(),
                                                  eleScratch.data(), g_im1,
                                                  g_im2, true, emode);
if (anyGraded && std::getenv("DENDRO_WPX_DEBUG_GRADED")) {
                static thread_local int shown = 0;
                int axesExt = 0;
                    for (int a = 0; a < 3; a++)
                        if (ext[2*a] || ext[2*a+1]) axesExt++;
                    if (shown < 2 && axesExt >= 2) {
                    shown++;
                    const ot::TreeNode &E = m_uiAllElements[ele];
                    const double Sd =
                        (double)(1u << (m_uiMaxDepth - E.getLevel()));
                    std::printf("[graded dbg] ele %u lvl %u S %.0f at (%u,%u,%u)\n",
                                ele, E.getLevel(), Sd, E.getX(), E.getY(),
                                E.getZ());
                    std::printf("  ext [%u %u %u %u %u %u] mode [%u %u %u %u %u %u]\n",
                                ext[0], ext[1], ext[2], ext[3], ext[4], ext[5],
                                emode[0], emode[1], emode[2], emode[3], emode[4],
                                emode[5]);
                    std::printf("  cx (n=%zu):", cx.size());
                    for (size_t t = 0; t < cx.size(); t++)
                        std::printf(" %.4f", cx[t]);
                    std::printf("\n  x-line of the gathered cube at the element's"
                                " own transverse origin:\n   ");
                    const unsigned int jj = ext[2], kk = ext[4];
                    for (unsigned int t = 0; t < nx_in; t++)
                        std::printf(" %.6f",
                                    (double)cube[(size_t)(kk * ny_in + jj) *
                                                     nx_in + t]);
                    std::printf("\n  implied x coords (octree units):");
                    for (size_t t = 0; t < cx.size(); t++)
                        std::printf(" %.1f", (double)E.getX() + cx[t] * Sd);
                    std::printf("\n  cy:");
                    for (size_t t = 0; t < cy.size(); t++)
                        std::printf(" %.4f", cy[t]);
                    std::printf("\n  y-line:");
                    for (unsigned int t = 0; t < ny_in; t++)
                        std::printf(" %.6f", (double)cube[(size_t)(kk*ny_in+t)*nx_in + ext[0]]);
                    std::printf("\n  implied y:");
                    for (size_t t = 0; t < cy.size(); t++)
                        std::printf(" %.1f", (double)E.getY() + cy[t]*Sd);
                    std::printf("\n");
                }
            }

            // Holes are the failure mode to rule out first: if the probe
            // granted a direction the gather cannot fill, the cube keeps
            // whatever it was seeded with. Seed with NaN and count.
            if (anyGraded && holes_dbg) {
                long holes = 0;
                for (size_t t = 0; t < cube.size(); t++)
                    if (cube[t] != cube[t]) holes++;
                if (holes) {
                    static thread_local long reported = 0;
                    if (reported < 5) {
                        reported++;
                        std::printf("[graded] ele %u: %ld of %zu cube entries "
                                    "UNFILLED (ext %u %u %u %u %u %u, mode "
                                    "%u %u %u %u %u %u) allDg=%s gvalid=%zu\n",
                                    ele, holes, cube.size(), ext[0], ext[1],
                                    ext[2], ext[3], ext[4], ext[5], emode[0],
                                    emode[1], emode[2], emode[3], emode[4],
                                    emode[5],
                                    allDg ? "yes" : "NULL", gvalid.size());
                    }
                }
            }

                        dendro::wideprolong::apply_3d(m_uiElementOrder, opx.data(), nx_in,
                                          opy.data(), ny_in, opz.data(), nz_in,
                                          cube.data(), out + v * m_uiNpE,
                                          w1.data(), w2.data());
        }
        return;
    }
    wpxLogFallbackOnce("unzip coarse->fine", 0, want);
#else
    (void)in;
    (void)cgSz;
#endif

    this->prolongateChildNodesNarrow(dgEle, dgSz, cnum, dof, out, im1, im2);
}

/** Diagnostic counters for the wide edge path. */
inline std::atomic<long> &wpxFaceCalls() {
    static std::atomic<long> c{0};
    return c;
}
inline std::atomic<long> &wpxFaceWins() {
    static std::atomic<long> c{0};
    return c;
}
inline std::atomic<long> &wpxEdgeCalls() {
    static std::atomic<long> c{0};
    return c;
}
inline std::atomic<long> &wpxEdgeWins() {
    static std::atomic<long> c{0};
    return c;
}

inline unsigned int Mesh::wpxEdgeOwner(unsigned int ele, unsigned int d1,
                                       unsigned int d2) const {
    const ot::TreeNode &E  = m_uiAllElements[ele];
    const unsigned int lev = E.getLevel();
    if (lev == 0) return LOOK_UP_TABLE_DEFAULT;
    const long S = 1l << (m_uiMaxDepth - lev);

    const unsigned int order[2][2] = {{d1, d2}, {d2, d1}};
    for (int t = 0; t < 2; t++) {
        unsigned int cur = ele;
        bool ok          = true;
        for (int h = 0; h < 2 && ok; h++) {
            const unsigned int nb =
                m_uiE2EMapping[cur * m_uiNumDirections + order[t][h]];
            if (nb == LOOK_UP_TABLE_DEFAULT || nb >= m_uiAllElements.size())
                ok = false;
            else
                cur = nb;
        }
        if (!ok) continue;
        if (m_uiAllElements[cur].getLevel() + 1 != lev) continue;
        if (!m_uiIsNodalMapValid[cur]) continue;

        // The owner is a diagonal NEIGHBOUR, so it contains the edge, not
        // this element's volume. Check the edge segment: along the axis the
        // edge runs, the owner must span it; on the other two axes the edge
        // sits on the owner's boundary, so the test is inclusive.
        const ot::TreeNode &Q = m_uiAllElements[cur];
        const long Sq         = 1l << (m_uiMaxDepth - Q.getLevel());
        const int a1          = (int)(d1 >> 1u), a2 = (int)(d2 >> 1u);
        if (a1 == a2) continue;
        const int ax = 3 - a1 - a2;

        bool contains = true;
        for (int a = 0; a < 3 && contains; a++) {
            const long cq = (long)wpxCoord(Q, a);
            if (a == ax) {
                const long c = (long)wpxCoord(E, a);
                if (c < cq || c + S > cq + Sq) contains = false;
            } else {
                long c = (long)wpxCoord(E, a);
                if (d1 == (unsigned int)(2 * a + 1) ||
                    d2 == (unsigned int)(2 * a + 1))
                    c += S;  // edge on this element's high side
                if (c < cq || c > cq + Sq) contains = false;
            }
        }
        if (contains) return cur;
    }
    return LOOK_UP_TABLE_DEFAULT;
}

template <typename T>
bool Mesh::prolongateHangingEdgeWide(const T *vec, unsigned int elementID,
                                     unsigned int d1, unsigned int d2,
                                     unsigned int cnum, T *out, double *im1,
                                     double *im2, const T *allDg,
                                     size_t allDgEleStride) const {
#ifndef DENDRO_WIDE_PROLONGATION
    (void)vec; (void)elementID; (void)d1; (void)d2; (void)cnum; (void)out;
    (void)im1; (void)im2; (void)allDg; (void)allDgEleStride;
    return false;
#else
    // Checked before the counter so the call counts report actual wide work.
    if (!m_uiWpxRuntimeEnabled) return false;

    const unsigned int p   = m_uiElementOrder;
    const unsigned int nrp = p + 1;

    wpxEdgeCalls()++;
    // see prolongateHangingFaceWide: without a DG array the volume gather
    // would recurse through its own hanging-face widening, so fall back.
    if (allDg == nullptr) return false;
    const unsigned int owner = this->wpxEdgeOwner(elementID, d1, d2);
    if (owner == LOOK_UP_TABLE_DEFAULT) return false;

    // the edge runs along the axis neither direction names
    const int a1 = (int)(d1 >> 1u), a2 = (int)(d2 >> 1u);
    if (a1 == a2) return false;
    const int ax = 3 - a1 - a2;

    const unsigned int width = dendro::wideprolong::stencil_width(p);
    const unsigned int want  = (width > nrp) ? (width - nrp) : 0u;

    // Like the hanging face: the edge values are the owner's prolongated
    // child restricted to the edge line, so build them with the volume
    // operator, which can extend graded into coarser neighbours where the
    // along-axis neighbour is not at the owner's level.
    const unsigned int mask = WPX_LVL_DEFAULT;

    unsigned int ext[6];
    unsigned char emode[6];
    const unsigned int st = this->probeCoarseExtension(
        owner, want, ext, mask, emode, true);
    if (st & WPX_CLIPPED_GHOST) {
        std::cerr << "[wide prolongation] hanging edge of element "
                  << elementID
                  << " needs a round-2 ghost neighbour; the nodal ghost layer "
                     "only covers round 1."
                  << std::endl;
        MPI_Abort(m_uiCommGlobal, 1);
    }

    // values on the line depend only on data along it, so with no room on
    // the edge axis the volume operator cannot beat the narrow 1D
    // interpolation -- keep today's fallback
    const unsigned int e_tot = ext[2 * ax] + ext[2 * ax + 1];
    if (!e_tot) {
        wpxLogFallbackOnce("hanging edge", 0, want);
        return false;
    }
    if (e_tot < want) wpxLogFallbackOnce("hanging edge", e_tot, want);

    // where this element's edge sits on the owner: for the two axes normal
    // to the edge it lies on the owner's boundary (node 0 or p), and along
    // the edge this element covers the half given by the 1D cnum bit
    const ot::TreeNode &E = m_uiAllElements[elementID];
    const ot::TreeNode &Q = m_uiAllElements[owner];
    const long Sq         = 1l << (m_uiMaxDepth - Q.getLevel());
    unsigned int loc[3]   = {0, 0, 0};
    for (int a = 0; a < 3; a++) {
        if (a == ax) continue;
        long c = (long)wpxCoord(E, a) - (long)wpxCoord(Q, a);
        if (d1 == (unsigned int)(2 * a + 1) || d2 == (unsigned int)(2 * a + 1))
            c += (1l << (m_uiMaxDepth - E.getLevel()));  // high-side edge
        loc[a] = (unsigned int)((c * (long)p) / Sq);
        if (loc[a] != 0 && loc[a] != p) return false;  // not on Q's edge
    }

    unsigned int cb[3];
    cb[ax] = cnum & 1u;
    for (int a = 0; a < 3; a++) {
        if (a == ax) continue;
        cb[a] = (loc[a] == p) ? 1u : 0u;
    }
    const unsigned int cnum3 = cb[0] | (cb[1] << 1u) | (cb[2] << 2u);

    // memoised, same lifetime rule as the face path
    const T *child = this->wpxProlongChildMemo(
        vec, (size_t)0, owner, cnum3, 1u, (size_t)m_uiNpE, allDg,
        allDgEleStride, mask, im1, im2);

    for (unsigned int t = 0; t < nrp; t++) {
        unsigned int id3[3];
        id3[0]  = loc[0];
        id3[1]  = loc[1];
        id3[2]  = loc[2];
        id3[ax] = t;
        out[t]  = child[(id3[2] * nrp + id3[1]) * nrp + id3[0]];
    }

    wpxEdgeWins()++;
    return true;
#endif
}

template <typename T>
bool Mesh::prolongateHangingFaceWide(const T *vec, unsigned int elementID,
                                     unsigned int dir, unsigned int cnum,
                                     T *out, double *im1, double *im2,
                                     const T *allDg,
                                     size_t allDgEleStride) const {
#ifndef DENDRO_WIDE_PROLONGATION
    (void)vec; (void)elementID; (void)dir; (void)cnum; (void)out;
    (void)im1; (void)im2; (void)allDg; (void)allDgEleStride;
    return false;
#else
    // Checked before the counter so the call counts report actual wide work.
    if (!m_uiWpxRuntimeEnabled) return false;

    const unsigned int p   = m_uiElementOrder;
    const unsigned int nrp = p + 1;

    wpxFaceCalls()++;
    // The volume operator below gathers through the whole-mesh DG array. A
    // CG gather is not an option here: its inner fetches widen their own
    // hanging faces, which re-enters this function on the owner's
    // neighbourhood and recurses without bound. Callers without a DG array
    // (grid transfer, oda, the CG diagnostics) fall back to narrow, exactly
    // as they did before allowWide became opt-in.
    if (allDg == nullptr) return false;
    const unsigned int owner =
        m_uiE2EMapping[elementID * m_uiNumDirections + dir];
    if (owner == LOOK_UP_TABLE_DEFAULT || owner >= m_uiAllElements.size())
        return false;
    if (m_uiAllElements[owner].getLevel() + 1 !=
        m_uiAllElements[elementID].getLevel())
        return false;

    // normal axis plus the two in-plane tangentials; the 2D face cnum encodes
    // the tangentials as bit 0 -> axA, bit 1 -> axB
    int axN, axA, axB;
    if (dir == OCT_DIR_LEFT || dir == OCT_DIR_RIGHT) {
        axN = 0; axA = 1; axB = 2;
    } else if (dir == OCT_DIR_DOWN || dir == OCT_DIR_UP) {
        axN = 1; axA = 0; axB = 2;
    } else {
        axN = 2; axA = 0; axB = 1;
    }

    const unsigned int width = dendro::wideprolong::stencil_width(p);
    const unsigned int want  = (width > nrp) ? (width - nrp) : 0u;

    // These face values are the owner's prolongated child restricted to the
    // shared plane, so build them with the VOLUME operator and read the plane
    // off the child. That is the one path that already knows how to extend
    // graded into a coarser neighbour, and reusing it keeps face and pad
    // values consistent by construction. The in-plane-only widening that used
    // to live here was the single largest input-error source when clipped:
    // rms 1.5e-07 on clipped faces against 6.7e-13 on unclipped ones, with
    // 43% of hanging-face nodes clipped on the wtol=1e-6 puncture mesh.
    //
    const unsigned int mask = WPX_LVL_DEFAULT;

    unsigned int ext[6];
    unsigned char emode[6];
    const unsigned int st =
        this->probeCoarseExtension(owner, want, ext, mask, emode, true);
    if (st & WPX_CLIPPED_GHOST) {
        std::cerr << "[wide prolongation] rank " << m_uiActiveRank
                  << ": hanging face of element " << elementID << " ("
                  << ((elementID >= m_uiElementLocalBegin &&
                       elementID < m_uiElementLocalEnd)
                          ? "LOCAL"
                          : "GHOST")
                  << ", face owner " << owner << " "
                  << ((owner >= m_uiElementLocalBegin &&
                       owner < m_uiElementLocalEnd)
                          ? "LOCAL"
                          : "GHOST")
                  << ") needs a round-2 ghost neighbour; the nodal ghost "
                     "layer only covers round 1."
                  << std::endl;
        MPI_Abort(m_uiCommGlobal, 1);
    }

    // values on the plane depend only on in-plane data, so with no
    // tangential room the volume operator cannot beat the narrow 2D
    // interpolation -- keep today's fallback
    const unsigned int a_tot = ext[2 * axA] + ext[2 * axA + 1];
    const unsigned int b_tot = ext[2 * axB] + ext[2 * axB + 1];
    if (!(a_tot || b_tot)) {
        wpxLogFallbackOnce("hanging face", 0, want);
        return false;
    }
    if (a_tot < want || b_tot < want)
        wpxLogFallbackOnce("hanging face", std::min(a_tot, b_tot), want);

    // the owner's virtual child sharing this face: tangential bits from the
    // 2D face cnum, normal bit on the side abutting this element
    unsigned int cb[3] = {0, 0, 0};
    cb[axA]            = cnum & 1u;
    cb[axB]            = (cnum >> 1u) & 1u;
    cb[axN]            = (dir & 1u) ? 0u : 1u;
    const unsigned int cnum3 = cb[0] | (cb[1] << 1u) | (cb[2] << 2u);
    // that child's face toward this element
    const unsigned int nIdx = (dir & 1u) ? 0u : p;

    // memoised: the owner's other exterior faces and edges want this same
    // child. Copy the plane out before anything can call the memo again.
    const T *child = this->wpxProlongChildMemo(
        vec, (size_t)0, owner, cnum3, 1u, (size_t)m_uiNpE, allDg,
        allDgEleStride, mask, im1, im2);

    for (unsigned int b = 0; b < nrp; b++)
        for (unsigned int a = 0; a < nrp; a++) {
            unsigned int id3[3];
            id3[axN]         = nIdx;
            id3[axA]         = a;
            id3[axB]         = b;
            out[b * nrp + a] =
                child[(id3[2] * nrp + id3[1]) * nrp + id3[0]];
        }

    wpxFaceWins()++;
    return true;
#endif
}

template <typename T>
void Mesh::getBlkBoundaryParentNodes(const T* zipVec, T* out, T* w1, T* w2,
                                     unsigned int lookUp,
                                     const unsigned int* fid,
                                     const unsigned int* cid,
                                     const unsigned int* child) {
    const unsigned int NUM_CHILDREN_BY2 = (NUM_CHILDREN >> 1u);
    const unsigned int eorder_by2       = (m_uiElementOrder + 1) >> 1u;
    const unsigned int nx               = m_uiElementOrder + 1;
    const unsigned int ny               = m_uiElementOrder + 1;
    const unsigned int nz               = m_uiElementOrder + 1;

    unsigned char bit[3];

    // finner elements.
    for (unsigned int w = 0; w < NUM_CHILDREN_BY2; w++) {
        this->getElementNodalValues(zipVec, w1, child[fid[w]]);
        // std::cout<<" cnum : "<<fid[w]<<std::endl;
        bit[0]                = binOp::getBit(fid[w], 0);
        bit[1]                = binOp::getBit(fid[w], 1);
        bit[2]                = binOp::getBit(fid[w], 2);

        const unsigned int kb = bit[2] * eorder_by2;
        unsigned int ke       = kb + eorder_by2 + 1;

        const unsigned int jb = bit[1] * eorder_by2;
        unsigned int je       = jb + eorder_by2 + 1;

        const unsigned int ib = bit[0] * eorder_by2;
        unsigned int ie       = ib + eorder_by2 + 1;

        for (unsigned int k = 0; k < nz; k += 2)
            for (unsigned int j = 0; j < ny; j += 2)
                for (unsigned int i = 0; i < nx; i += 2)
                    out[(kb + (k >> 1u)) * ny * nx + (jb + (j >> 1u)) * nx +
                        (ib + (i >> 1u))] = w1[k * ny * nx + j * nx + i];
    }

    this->getElementNodalValues(zipVec, w1, lookUp);
    // coarser elements.
    for (unsigned int w = 0; w < NUM_CHILDREN_BY2; w++) {
        this->parent2ChildInterpolation(w1, w2, fid[w], m_uiDim);

        // std::cout<<" cnum : "<<fid[w]<<std::endl;
        bit[0]                = binOp::getBit(cid[w], 0);
        bit[1]                = binOp::getBit(cid[w], 1);
        bit[2]                = binOp::getBit(cid[w], 2);

        const unsigned int kb = bit[2] * eorder_by2;
        unsigned int ke       = kb + eorder_by2 + 1;

        const unsigned int jb = bit[1] * eorder_by2;
        unsigned int je       = jb + eorder_by2 + 1;

        const unsigned int ib = bit[0] * eorder_by2;
        unsigned int ie       = ib + eorder_by2 + 1;

        for (unsigned int k = 0; k < nz; k += 2)
            for (unsigned int j = 0; j < ny; j += 2)
                for (unsigned int i = 0; i < nx; i += 2) {
                    // std::cout<< " cnum : "<<fid[w]<< " left lookup value:
                    // ijk: "<<i<<j<<k<<" "<<lookUpElementVec[(kb +
                    // (k>>1u))*ny*nx + (jb + (j>>1u))*nx + (ib + (i>>1u))]<< "
                    // inject value: "<<w2[k*ny*nx + j*nx + i]<<"";
                    // printf("lookup idx (%d,%d,%d)\n",(ib + (i>>1u)),(jb +
                    // (j>>1u)), (kb + (k>>1u)));
                    out[(kb + (k >> 1u)) * ny * nx + (jb + (j >> 1u)) * nx +
                        (ib + (i >> 1u))] = w2[k * ny * nx + j * nx + i];
                }
    }
}

template <typename T>
void Mesh::unzipDG(const T* in, T* out, const unsigned int* blkIDs,
                   unsigned int numblks, unsigned int dof) {
    if (!m_uiIsActive) return;

    const ot::TreeNode* pNodes = m_uiAllElements.data();
    const ot::Block* blkList   = m_uiLocalBlockList.data();
    const unsigned int eOrder  = m_uiElementOrder;
    const unsigned int nPe     = m_uiNpE;

    const unsigned int dgSz    = m_uiAllElements.size() * nPe;
    const unsigned int cgSz    = this->getDegOfFreedom();
    const unsigned int unSz    = this->getDegOfFreedomUnZip();

    const unsigned int* e2n    = this->getE2NMapping().data();
    const unsigned int* e2e    = this->getE2EMapping().data();

    const T* dgWVec            = in;
    T* uzWVec                  = out;

    for (unsigned int bid = 0; bid < numblks; bid++) {
        const unsigned int blk      = blkIDs[bid];
        const unsigned int regLevel = blkList[blk].getRegularGridLev();
        const ot::TreeNode blkNode  = blkList[blk].getBlockNode();
        const unsigned int PW       = blkList[blk].get1DPadWidth();

        const unsigned int lx       = blkList[blk].getAllocationSzX();
        const unsigned int ly       = blkList[blk].getAllocationSzY();
        const unsigned int lz       = blkList[blk].getAllocationSzZ();
        const unsigned int offset   = blkList[blk].getOffset();

        const unsigned int bLev =
            pNodes[blkList[blk].getLocalElementBegin()].getLevel();

        std::vector<unsigned int> eid;
        eid.reserve((NUM_CHILDREN + NUM_FACES + NUM_EDGES + 1) * 4);
        this->blkUnzipElementIDs(blk, eid);

        // now need to copy to the block unzip/ block asyncVector
        const double hx   = (1u << (m_uiMaxDepth - bLev)) / (double)eOrder;

        const double xmin = blkNode.minX() - PW * hx;
        const double xmax = blkNode.maxX() + PW * hx;
        const double ymin = blkNode.minY() - PW * hx;
        const double ymax = blkNode.maxY() + PW * hx;
        const double zmin = blkNode.minZ() - PW * hx;
        const double zmax = blkNode.maxZ() + PW * hx;

        std::vector<ot::TreeNode> childOct;
        childOct.reserve(NUM_CHILDREN);

        std::vector<T> p2cI;
        p2cI.resize(nPe);

        const double d_compar_tol = 1e-10;

        for (unsigned int m = 0; m < eid.size(); m++) {
            const unsigned int ele = eid[m];

            // no interpolation needed just copy.
            if (pNodes[ele].getLevel() == bLev) {
                const double hh =
                    (1u << (m_uiMaxDepth - pNodes[ele].getLevel())) /
                    (double)eOrder;
                const double invhh = 1.0 / hh;

                for (unsigned int k = 0; k < eOrder + 1; k++) {
                    double zz = pNodes[ele].minZ() + k * hh;

                    if (fabs(zz - zmin) < d_compar_tol) zz = zmin;
                    if (fabs(zz - zmax) < d_compar_tol) zz = zmax;

                    if (zz < zmin || zz > zmax) continue;
                    const int kkz = std::round((zz - zmin) * invhh);
                    assert(std::fabs(zz - zmin - kkz * hh) < d_compar_tol);
                    assert(kkz >= 0 && kkz < lz);

                    for (unsigned int j = 0; j < eOrder + 1; j++) {
                        double yy = pNodes[ele].minY() + j * hh;

                        if (fabs(yy - ymin) < d_compar_tol) yy = ymin;
                        if (fabs(yy - ymax) < d_compar_tol) yy = ymax;

                        if (yy < ymin || yy > ymax) continue;
                        const int jjy = std::round((yy - ymin) * invhh);
                        // std::cout<<"yy: "<<yy<<" (ymin + hh*jjy): "<<(ymin +
                        // hh*jjy)<<std::endl;
                        assert(std::fabs(yy - ymin - jjy * hh) < d_compar_tol);
                        assert(jjy >= 0 && jjy < ly);

                        for (unsigned int i = 0; i < eOrder + 1; i++) {
                            double xx = pNodes[ele].minX() + i * hh;

                            if (fabs(xx - xmin) < d_compar_tol) xx = xmin;
                            if (fabs(xx - xmax) < d_compar_tol) xx = xmax;

                            if (xx < xmin || xx > xmax) continue;
                            const int iix = std::round((xx - xmin) * invhh);
                            assert(std::fabs(xx - xmin - iix * hh) <
                                   d_compar_tol);
                            assert(iix >= 0 && iix < lx);

                            // std::cout<<"blk: "<<blk<<" copy : (i,j,k):
                            // ("<<kkz<<" , "<<jjy<<", "<<iix<<")"<<" of :
                            // "<<lx<<std::endl;

                            for (unsigned int v = 0; v < dof; v++)
                                uzWVec[v * unSz + offset + kkz * lx * ly +
                                       jjy * lx + iix] =
                                    dgWVec[v * dgSz + ele * nPe +
                                           k * (eOrder + 1) * (eOrder + 1) +
                                           j * (eOrder + 1) + i];
                        }
                    }
                }

            } else if (pNodes[ele].getLevel() > bLev) {
                assert((bLev + 1) == pNodes[ele].getLevel());
                const unsigned int cnum = pNodes[ele].getMortonIndex();
                ot::TreeNode tmpParent  = pNodes[ele].getParent();

                const double hh =
                    (1u << (m_uiMaxDepth - pNodes[ele].getLevel())) /
                    (double)eOrder;
                const double invhh = 1.0 / (2 * hh);

                assert(eOrder > 1);
                const unsigned int cb = (eOrder % 2 == 0) ? 0 : 1;

                for (unsigned int k = cb; k < eOrder + 1; k += 2) {
                    double zz = (pNodes[ele].minZ() + k * hh);
                    if (fabs(zz - zmin) < d_compar_tol) zz = zmin;
                    if (fabs(zz - zmax) < d_compar_tol) zz = zmax;

                    if (zz < zmin || zz > zmax) continue;
                    const int kkz = std::round((zz - zmin) * invhh);
                    assert(kkz >= 0 && kkz < lz);

                    for (unsigned int j = cb; j < eOrder + 1; j += 2) {
                        double yy = pNodes[ele].minY() + j * hh;

                        if (fabs(yy - ymin) < d_compar_tol) yy = ymin;
                        if (fabs(yy - ymax) < d_compar_tol) yy = ymax;

                        if (yy < ymin || yy > ymax) continue;

                        const int jjy = std::round((yy - ymin) * invhh);
                        assert(jjy >= 0 && jjy < ly);

                        for (unsigned int i = cb; i < eOrder + 1; i += 2) {
                            double xx = pNodes[ele].minX() + i * hh;

                            if (fabs(xx - xmin) < d_compar_tol) xx = xmin;
                            if (fabs(xx - xmax) < d_compar_tol) xx = xmax;

                            if (xx < xmin || xx > xmax) continue;
                            const int iix = std::round((xx - xmin) * invhh);
                            assert(iix >= 0 && iix < lx);

                            // std::cout<<"blk: "<<blk<<" blk copy : (i,j,k):
                            // ("<<iix<<" , "<<jjy<<", "<<kkz<<")"<<" of :
                            // "<<lx<<" xx: "<<xx<<" yy: "<<yy<<" zz:"<<zz<<"
                            // xmin: "<<xmin<<" ymin: "<<ymin<<" zmin:
                            // "<<zmin<<" hh : "<<hh<<" hhx : "<<hx<<std::endl;
                            for (unsigned int v = 0; v < dof; v++)
                                uzWVec[v * unSz + offset + kkz * lx * ly +
                                       jjy * lx + iix] =
                                    dgWVec[v * dgSz + ele * nPe +
                                           k * (eOrder + 1) * (eOrder + 1) +
                                           j * (eOrder + 1) + i];
                        }
                    }
                }

            } else {
                assert((bLev) == (pNodes[ele].getLevel() + 1));
                childOct.clear();
                pNodes[ele].addChildren(
                    childOct);  // note this is the ordering of SFC (depends on
                                // Hilbert or Morton. )

                for (unsigned int child = 0; child < NUM_CHILDREN; child++) {
                    if ((childOct[child].maxX() < xmin ||
                         childOct[child].minX() >= xmax) ||
                        (childOct[child].maxY() < ymin ||
                         childOct[child].minY() >= ymax) ||
                        (childOct[child].maxZ() < zmin ||
                         childOct[child].minZ() >= zmax))
                        continue;

                    // std::cout<<"blk: "<<blk<<" blkNode: "<<blkNode<<" child:
                    // "<<child<<" child node "<<childOct[child]<<" parent :
                    // "<<pNodes[ele]<<std::endl;
                    const double hh =
                        (1u << (m_uiMaxDepth - childOct[child].getLevel())) /
                        (double)eOrder;
                    const double invhh = 1.0 / hh;

                    for (unsigned int v = 0; v < dof; v++) {
                        const unsigned int cnum =
                            childOct[child].getMortonIndex();
                        this->parent2ChildInterpolation(
                            &dgWVec[v * dgSz + ele * nPe], p2cI.data(), cnum,
                            m_uiDim);

                        for (unsigned int k = 0; k < eOrder + 1; k++) {
                            double zz = childOct[child].minZ() + k * hh;

                            if (fabs(zz - zmin) < d_compar_tol) zz = zmin;
                            if (fabs(zz - zmax) < d_compar_tol) zz = zmax;

                            if (zz < zmin || zz > zmax) continue;
                            const int kkz = std::round((zz - zmin) * invhh);
                            assert(kkz >= 0 && kkz < lz);

                            for (unsigned int j = 0; j < eOrder + 1; j++) {
                                double yy = childOct[child].minY() + j * hh;

                                if (fabs(yy - ymin) < d_compar_tol) yy = ymin;
                                if (fabs(yy - ymax) < d_compar_tol) yy = ymax;

                                if (yy < ymin || yy > ymax) continue;

                                const int jjy = std::round((yy - ymin) * invhh);
                                assert(jjy >= 0 && jjy < ly);

                                for (unsigned int i = 0; i < eOrder + 1; i++) {
                                    double xx = childOct[child].minX() + i * hh;

                                    if (fabs(xx - xmin) < d_compar_tol)
                                        xx = xmin;
                                    if (fabs(xx - xmax) < d_compar_tol)
                                        xx = xmax;

                                    if (xx < xmin || xx > xmax) continue;
                                    const int iix =
                                        std::round((xx - xmin) * invhh);
                                    assert(iix >= 0 && iix < lx);

                                    uzWVec[v * unSz + offset + kkz * lx * ly +
                                           jjy * lx + iix] =
                                        p2cI[k * (eOrder + 1) * (eOrder + 1) +
                                             j * (eOrder + 1) + i];
                                }
                            }
                        }
                    }
                }
            }
        }

        // internal copy.
        for (unsigned int elem = blkList[blk].getLocalElementBegin();
             elem < blkList[blk].getLocalElementEnd(); elem++) {
            const unsigned int ei   = (pNodes[elem].getX() - blkNode.getX()) >>
                                      (m_uiMaxDepth - regLevel);
            const unsigned int ej   = (pNodes[elem].getY() - blkNode.getY()) >>
                                      (m_uiMaxDepth - regLevel);
            const unsigned int ek   = (pNodes[elem].getZ() - blkNode.getZ()) >>
                                      (m_uiMaxDepth - regLevel);

            const unsigned int emin = 0;
            const unsigned int emax =
                (1u << (regLevel - blkNode.getLevel())) - 1;

            // #pragma unroll
            // for(unsigned int v=0; v < dof; v++)
            //     std::memcpy(dgWVec + v*dgSz + elem * nPe , dgStages[rk_s-1] +
            //     v*dgSz + elem * nPe, sizeof(T)*nPe );

            for (unsigned int v = 0; v < dof; v++)
                for (unsigned int k = 0; k < (eOrder + 1); k++)
                    for (unsigned int j = 0; j < (eOrder + 1); j++)
                        for (unsigned int i = 0; i < (eOrder + 1); i++)
                            uzWVec[v * unSz + offset +
                                   (ek * eOrder + k + PW) * (ly * lx) +
                                   (ej * eOrder + j + PW) * (lx) +
                                   (ei * eOrder + i + PW)] =
                                dgWVec[v * dgSz + elem * nPe +
                                       k * (eOrder + 1) * (eOrder + 1) +
                                       j * (eOrder + 1) + i];
            ;
        }
    }
}

template <typename T>
void Mesh::unzipDG_scatter(const T* in, T* out, unsigned int dof) {
    if (!m_uiIsActive) return;

    const ot::TreeNode* pNodes = m_uiAllElements.data();
    const ot::Block* blkList   = m_uiLocalBlockList.data();
    const unsigned int eOrder  = m_uiElementOrder;
    const unsigned int nPe     = m_uiNpE;

    const unsigned int dgSz    = m_uiAllElements.size() * nPe;
    const unsigned int cgSz    = this->getDegOfFreedom();
    const unsigned int unSz    = this->getDegOfFreedomUnZip();

    const unsigned int* e2n    = this->getE2NMapping().data();
    const unsigned int* e2e    = this->getE2EMapping().data();

    const T* dgWVec            = in;
    T* uzWVec                  = out;

    std::vector<T> p2cI_all;
    p2cI_all.resize(NUM_CHILDREN * dof * nPe);
    bool p2c_interp_valid[NUM_CHILDREN];

    const double d_compar_tol = 1e-10;

    std::vector<ot::TreeNode> childOct;
    childOct.reserve(NUM_CHILDREN);

    for (unsigned int ele = 0; ele < m_uiNumTotalElements; ele++) {
        if (m_e2b_unzip_counts[ele] == 0) continue;

        for (unsigned int ii = 0; ii < NUM_CHILDREN; ii++)
            p2c_interp_valid[ii] = false;

        for (unsigned int i = 0; i < m_e2b_unzip_counts[ele]; i++) {
            const unsigned int e2b_offset = m_e2b_unzip_offset[ele];
            const unsigned int blk        = m_e2b_unzip_map[e2b_offset + i];
            assert(blk != LOOK_UP_TABLE_DEFAULT &&
                   blk < m_uiLocalBlockList.size());

            const unsigned int regLevel = blkList[blk].getRegularGridLev();
            const ot::TreeNode blkNode  = blkList[blk].getBlockNode();
            const unsigned int PW       = blkList[blk].get1DPadWidth();

            const unsigned int lx       = blkList[blk].getAllocationSzX();
            const unsigned int ly       = blkList[blk].getAllocationSzY();
            const unsigned int lz       = blkList[blk].getAllocationSzZ();
            const unsigned int offset   = blkList[blk].getOffset();

            const unsigned int bLev =
                pNodes[blkList[blk].getLocalElementBegin()].getLevel();

            const double hx   = (1u << (m_uiMaxDepth - bLev)) / (double)eOrder;
            const double xmin = blkNode.minX() - PW * hx;
            const double xmax = blkNode.maxX() + PW * hx;
            const double ymin = blkNode.minY() - PW * hx;
            const double ymax = blkNode.maxY() + PW * hx;
            const double zmin = blkNode.minZ() - PW * hx;
            const double zmax = blkNode.maxZ() + PW * hx;

            // no interpolation needed just copy.
            if (pNodes[ele].getLevel() == bLev) {
                const double hh =
                    (1u << (m_uiMaxDepth - pNodes[ele].getLevel())) /
                    (double)eOrder;
                const double invhh = 1.0 / hh;

                for (unsigned int k = 0; k < eOrder + 1; k++) {
                    double zz = pNodes[ele].minZ() + k * hh;

                    if (fabs(zz - zmin) < d_compar_tol) zz = zmin;
                    if (fabs(zz - zmax) < d_compar_tol) zz = zmax;

                    if (zz < zmin || zz > zmax) continue;
                    const int kkz = std::round((zz - zmin) * invhh);
                    assert(std::fabs(zz - zmin - kkz * hh) < d_compar_tol);
                    assert(kkz >= 0 && kkz < lz);

                    for (unsigned int j = 0; j < eOrder + 1; j++) {
                        double yy = pNodes[ele].minY() + j * hh;

                        if (fabs(yy - ymin) < d_compar_tol) yy = ymin;
                        if (fabs(yy - ymax) < d_compar_tol) yy = ymax;

                        if (yy < ymin || yy > ymax) continue;
                        const int jjy = std::round((yy - ymin) * invhh);
                        // std::cout<<"yy: "<<yy<<" (ymin + hh*jjy): "<<(ymin +
                        // hh*jjy)<<std::endl;
                        assert(std::fabs(yy - ymin - jjy * hh) < d_compar_tol);
                        assert(jjy >= 0 && jjy < ly);

                        for (unsigned int i = 0; i < eOrder + 1; i++) {
                            double xx = pNodes[ele].minX() + i * hh;

                            if (fabs(xx - xmin) < d_compar_tol) xx = xmin;
                            if (fabs(xx - xmax) < d_compar_tol) xx = xmax;

                            if (xx < xmin || xx > xmax) continue;
                            const int iix = std::round((xx - xmin) * invhh);
                            assert(std::fabs(xx - xmin - iix * hh) <
                                   d_compar_tol);
                            assert(iix >= 0 && iix < lx);

                            // std::cout<<"blk: "<<blk<<" copy : (i,j,k):
                            // ("<<kkz<<" , "<<jjy<<", "<<iix<<")"<<" of :
                            // "<<lx<<std::endl;

                            for (unsigned int v = 0; v < dof; v++)
                                uzWVec[v * unSz + offset + kkz * lx * ly +
                                       jjy * lx + iix] =
                                    dgWVec[v * dgSz + ele * nPe +
                                           k * (eOrder + 1) * (eOrder + 1) +
                                           j * (eOrder + 1) + i];
                        }
                    }
                }

            } else if (pNodes[ele].getLevel() > bLev) {
                assert((bLev + 1) == pNodes[ele].getLevel());
                const unsigned int cnum = pNodes[ele].getMortonIndex();
                ot::TreeNode tmpParent  = pNodes[ele].getParent();

                const double hh =
                    (1u << (m_uiMaxDepth - pNodes[ele].getLevel())) /
                    (double)eOrder;
                const double invhh = 1.0 / (2 * hh);

                assert(eOrder > 1);
                const unsigned int cb = (eOrder % 2 == 0) ? 0 : 1;

                for (unsigned int k = cb; k < eOrder + 1; k += 2) {
                    double zz = (pNodes[ele].minZ() + k * hh);
                    if (fabs(zz - zmin) < d_compar_tol) zz = zmin;
                    if (fabs(zz - zmax) < d_compar_tol) zz = zmax;

                    if (zz < zmin || zz > zmax) continue;
                    const int kkz = std::round((zz - zmin) * invhh);
                    assert(kkz >= 0 && kkz < lz);

                    for (unsigned int j = cb; j < eOrder + 1; j += 2) {
                        double yy = pNodes[ele].minY() + j * hh;

                        if (fabs(yy - ymin) < d_compar_tol) yy = ymin;
                        if (fabs(yy - ymax) < d_compar_tol) yy = ymax;

                        if (yy < ymin || yy > ymax) continue;

                        const int jjy = std::round((yy - ymin) * invhh);
                        assert(jjy >= 0 && jjy < ly);

                        for (unsigned int i = cb; i < eOrder + 1; i += 2) {
                            double xx = pNodes[ele].minX() + i * hh;

                            if (fabs(xx - xmin) < d_compar_tol) xx = xmin;
                            if (fabs(xx - xmax) < d_compar_tol) xx = xmax;

                            if (xx < xmin || xx > xmax) continue;
                            const int iix = std::round((xx - xmin) * invhh);
                            assert(iix >= 0 && iix < lx);

                            // std::cout<<"blk: "<<blk<<" blk copy : (i,j,k):
                            // ("<<iix<<" , "<<jjy<<", "<<kkz<<")"<<" of :
                            // "<<lx<<" xx: "<<xx<<" yy: "<<yy<<" zz:"<<zz<<"
                            // xmin: "<<xmin<<" ymin: "<<ymin<<" zmin:
                            // "<<zmin<<" hh : "<<hh<<" hhx : "<<hx<<std::endl;
                            for (unsigned int v = 0; v < dof; v++)
                                uzWVec[v * unSz + offset + kkz * lx * ly +
                                       jjy * lx + iix] =
                                    dgWVec[v * dgSz + ele * nPe +
                                           k * (eOrder + 1) * (eOrder + 1) +
                                           j * (eOrder + 1) + i];
                        }
                    }
                }

            } else {
                assert((bLev) == (pNodes[ele].getLevel() + 1));
                childOct.clear();
                pNodes[ele].addChildren(
                    childOct);  // note this is the ordering of SFC (depends on
                                // Hilbert or Morton. )

                for (unsigned int child = 0; child < NUM_CHILDREN; child++) {
                    if ((childOct[child].maxX() < xmin ||
                         childOct[child].minX() >= xmax) ||
                        (childOct[child].maxY() < ymin ||
                         childOct[child].minY() >= ymax) ||
                        (childOct[child].maxZ() < zmin ||
                         childOct[child].minZ() >= zmax))
                        continue;

                    // std::cout<<"blk: "<<blk<<" blkNode: "<<blkNode<<" child:
                    // "<<child<<" child node "<<childOct[child]<<" parent :
                    // "<<pNodes[ele]<<std::endl;
                    const double hh =
                        (1u << (m_uiMaxDepth - childOct[child].getLevel())) /
                        (double)eOrder;
                    const double invhh      = 1.0 / hh;

                    const unsigned int cnum = childOct[child].getMortonIndex();
                    if (!p2c_interp_valid[cnum]) {
                        for (unsigned int v = 0; v < dof; v++)
                            this->parent2ChildInterpolation(
                                &dgWVec[v * dgSz + ele * nPe],
                                p2cI_all.data() + cnum * dof * nPe + v * nPe,
                                cnum, m_uiDim);

                        p2c_interp_valid[cnum] = true;
                    }

                    for (unsigned int v = 0; v < dof; v++) {
                        const T* const p2cI =
                            p2cI_all.data() + cnum * dof * nPe + v * nPe;
                        for (unsigned int k = 0; k < eOrder + 1; k++) {
                            double zz = childOct[child].minZ() + k * hh;

                            if (fabs(zz - zmin) < d_compar_tol) zz = zmin;
                            if (fabs(zz - zmax) < d_compar_tol) zz = zmax;

                            if (zz < zmin || zz > zmax) continue;
                            const int kkz = std::round((zz - zmin) * invhh);
                            assert(kkz >= 0 && kkz < lz);

                            for (unsigned int j = 0; j < eOrder + 1; j++) {
                                double yy = childOct[child].minY() + j * hh;

                                if (fabs(yy - ymin) < d_compar_tol) yy = ymin;
                                if (fabs(yy - ymax) < d_compar_tol) yy = ymax;

                                if (yy < ymin || yy > ymax) continue;

                                const int jjy = std::round((yy - ymin) * invhh);
                                assert(jjy >= 0 && jjy < ly);

                                for (unsigned int i = 0; i < eOrder + 1; i++) {
                                    double xx = childOct[child].minX() + i * hh;

                                    if (fabs(xx - xmin) < d_compar_tol)
                                        xx = xmin;
                                    if (fabs(xx - xmax) < d_compar_tol)
                                        xx = xmax;

                                    if (xx < xmin || xx > xmax) continue;
                                    const int iix =
                                        std::round((xx - xmin) * invhh);
                                    assert(iix >= 0 && iix < lx);

                                    uzWVec[v * unSz + offset + kkz * lx * ly +
                                           jjy * lx + iix] =
                                        p2cI[k * (eOrder + 1) * (eOrder + 1) +
                                             j * (eOrder + 1) + i];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
void Mesh::unzipDG(const T* in, T* out, unsigned int dof) {
    if ((!m_uiIsActive) || (m_uiLocalBlockList.empty())) return;

    std::vector<unsigned int> blkIDs;
    blkIDs.resize(m_uiLocalBlockList.size());

    for (unsigned int i = 0; i < m_uiLocalBlockList.size(); i++) blkIDs[i] = i;

    // unzip all the blocks.
    this->unzipDG(in, out, blkIDs.data(), blkIDs.size(), dof);
}

}  // namespace ot
