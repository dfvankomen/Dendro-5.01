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

// shared counter for unzip-scatter call id, used by probes in
// unzip_scatter and getElementNodalValues. -1 = no probing.
inline int g_dendro_unzip_call_id = -1;

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
                        y = pNodes[ownerID].getY() +
                            jj_y * (len / ((double)m_uiElementOrder));
                        z = pNodes[ownerID].getZ() +
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

    // populate every CG slot any local element references, including
    // ghost CGs that the element's own canonical owner points to. same
    // physical position → same value from func, so redundant writes are
    // idempotent. also populate m_uiLocalNodalDG with f(element's OWN
    // sub physical position) for each local element, used by
    // performGhostExchange to deliver R2-boundary values that can't be
    // routed through the E2N-based CG path (R2's canonical disagrees
    // across ranks because R2's 26-neighborhood isn't ghost-fetched).
    const unsigned int numLocal =
        m_uiElementLocalEnd - m_uiElementLocalBegin;
    m_uiLocalNodalDG.assign(numLocal * m_uiNpE, (double)0);

    // Pass 1: iterate ALL elements (local + ghosts), but only write to
    // CG slots that are LOCAL on this rank. This back-fills LOCAL CG
    // slots referenced exclusively by ghost elements (rare edge case
    // where master's cascade chained a local element's own DG slot
    // away, leaving its CG referenced only by ghost neighbors).
    // Decode owner from E2N_DG so the value is f(canonical owner's
    // physical position) regardless of which element references it.
    for (unsigned int elem = m_uiElementPreGhostBegin;
         elem < m_uiElementPostGhostEnd; elem++) {
        if (elem >= m_uiElementLocalBegin && elem < m_uiElementLocalEnd)
            continue;  // skip locals here, pass 2 handles them fully
        for (unsigned int k = 0; k < (m_uiElementOrder + 1); k++)
            for (unsigned int j = 0; j < (m_uiElementOrder + 1); j++)
                for (unsigned int i = 0; i < (m_uiElementOrder + 1); i++) {
                    const unsigned int subIdx =
                        k * (m_uiElementOrder + 1) * (m_uiElementOrder + 1) +
                        j * (m_uiElementOrder + 1) + i;
                    nodeLookUp_CG =
                        m_uiE2NMapping_CG[elem * m_uiNpE + subIdx];
                    if (nodeLookUp_CG < m_uiNodeLocalBegin ||
                        nodeLookUp_CG >= m_uiNodeLocalEnd)
                        continue;  // only fill LOCAL CG slots from ghosts
                    nodeLookUp_DG =
                        m_uiE2NMapping_DG[elem * m_uiNpE + subIdx];
                    dg2eijk(nodeLookUp_DG, ownerID, ii_x, jj_y, kk_z);
                    len = 1u << (m_uiMaxDepth - pNodes[ownerID].getLevel());
                    x   = pNodes[ownerID].getX() +
                        ii_x * (len / ((double)m_uiElementOrder));
                    y = pNodes[ownerID].getY() +
                        jj_y * (len / ((double)m_uiElementOrder));
                    z = pNodes[ownerID].getZ() +
                        kk_z * (len / ((double)m_uiElementOrder));
                    vec[nodeLookUp_CG] = func(x, y, z);
                }
    }

    // Pass 2: iterate LOCAL elements, write all referenced CGs (local
    // and ghost on this rank) plus populate m_uiLocalNodalDG.
    for (unsigned int elem = m_uiElementLocalBegin; elem < m_uiElementLocalEnd;
         elem++) {
        const unsigned int elemLocalIdx = elem - m_uiElementLocalBegin;
        const unsigned int elemLen =
            1u << (m_uiMaxDepth - pNodes[elem].getLevel());
        const double eleHx = elemLen / ((double)m_uiElementOrder);
        for (unsigned int k = 0; k < (m_uiElementOrder + 1); k++)
            for (unsigned int j = 0; j < (m_uiElementOrder + 1); j++)
                for (unsigned int i = 0; i < (m_uiElementOrder + 1); i++) {
                    const unsigned int subIdx =
                        k * (m_uiElementOrder + 1) * (m_uiElementOrder + 1) +
                        j * (m_uiElementOrder + 1) + i;

                    // DG fill: f at element's own sub position
                    const double ex = pNodes[elem].getX() + i * eleHx;
                    const double ey = pNodes[elem].getY() + j * eleHx;
                    const double ez = pNodes[elem].getZ() + k * eleHx;
                    m_uiLocalNodalDG[elemLocalIdx * m_uiNpE + subIdx] =
                        (double)func(ex, ey, ez);

                    // CG fill: f at canonical owner's physical (existing
                    // behavior; hanging nodes land on owner's sub)
                    nodeLookUp_CG =
                        m_uiE2NMapping_CG[elem * m_uiNpE + subIdx];
                    if (nodeLookUp_CG >= m_uiNumActualNodes) continue;
                    nodeLookUp_DG =
                        m_uiE2NMapping_DG[elem * m_uiNpE + subIdx];
                    dg2eijk(nodeLookUp_DG, ownerID, ii_x, jj_y, kk_z);
                    len = 1u << (m_uiMaxDepth - pNodes[ownerID].getLevel());
                    x   = pNodes[ownerID].getX() +
                        ii_x * (len / ((double)m_uiElementOrder));
                    y = pNodes[ownerID].getY() +
                        jj_y * (len / ((double)m_uiElementOrder));
                    z = pNodes[ownerID].getZ() +
                        kk_z * (len / ((double)m_uiElementOrder));
                    vec[nodeLookUp_CG] = func(x, y, z);
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

    const bool useDG = (m_uiScatterMapSendIsDG.size() ==
                        m_uiScatterMapActualNodeSend.size());

    // Refresh per-element DG values so the DG-tagged send path sees
    // current CG values. Raw copy (not hanging interp): for CG, the
    // value at a shared position IS the canonical CG value, so reading
    // vec[E2N_CG[e][sub]] gives the correct value for every sub.
    if (useDG) this->syncLocalNodalDGFromCG(vec.data());

    for (unsigned int p = 0; p < m_uiActiveNpes; p++) {
        for (unsigned int k = m_uiSendNodeOffset[p];
             k < (m_uiSendNodeOffset[p] + m_uiSendNodeCount[p]); k++) {
            if (useDG && m_uiScatterMapSendIsDG[k]) {
                // DG path: sendSM encodes (localEle - localBegin) * NpE + sub
                m_uiSendBufferNodes[k] =
                    (T)m_uiLocalNodalDG[m_uiScatterMapActualNodeSend[k]];
            } else {
                m_uiSendBufferNodes[k] =
                    (T)vec[m_uiScatterMapActualNodeSend[k]];
            }
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
            vec[m_uiScatterMapActualNodeRecv[k]] = (T)m_uiRecvBufferNodes[k];
        }
    }
}

template <typename T>
void Mesh::performGhostExchange(T* vec) {
    if ((m_uiActiveNpes == 1) || (!m_uiIsActive)) return;

    // Side-channel non-primary sync runs FIRST (before the standard
    // ghost exchange) so that updated non-primary local cgs are
    // captured by the subsequent send-buffer pack. The standard
    // ghost exchange then propagates primary values to ghost cgs on
    // every rank transitively (each rank's local cgs at the
    // phys_pos hold the primary value, so all ghost views also do).
    syncZipNonPrimary(vec, 1u);

    const bool useDG = (m_uiScatterMapSendIsDG.size() ==
                        m_uiScatterMapActualNodeSend.size());

    // Refresh per-element DG values from `vec`'s CG layout when the
    // DG scatter path is active (graph-partitioned meshes with some
    // R2 boundary sends that route via m_uiLocalNodalDG).
    if (useDG) this->syncLocalNodalDGFromCG(vec);

    for (unsigned int p = 0; p < m_uiActiveNpes; p++) {
        for (unsigned int k = m_uiSendNodeOffset[p];
             k < (m_uiSendNodeOffset[p] + m_uiSendNodeCount[p]); k++) {
            if (useDG && m_uiScatterMapSendIsDG[k]) {
                m_uiSendBufferNodes[k] =
                    (T)m_uiLocalNodalDG[m_uiScatterMapActualNodeSend[k]];
            } else {
                m_uiSendBufferNodes[k] =
                    (T)vec[m_uiScatterMapActualNodeSend[k]];
            }
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

    // Sync non-primary local cgs from ghost. Each non-primary local
    // cg is at a phys_pos where another rank holds the primary; the
    // primary writes its local cg via zip and ghost-exchanges to us
    // on a specific ghost cg. We mirror that ghost value into the
    // non-primary local slot so VTU output and analytical-error
    // checks see the correct value at the non-primary slot.
    //
    // Stage 3 uses m_uiZipNonPrimaryToGhostCg, populated by
    // buildZipPlan() via an inverse-scatter-map lookup that picks
    // EXACTLY the ghost cg whose source is the primary's local cg
    // on the primary's rank. The legacy m_uiPassDDemotedToGhostCg
    // (built by Pass D's "first-found ghost cg at phys_pos" walk)
    // is still applied for compatibility — it will be removed in
    // Stage 4 when Pass A/D/E are deleted.
    static const char* nopd_env_a =
        std::getenv("DENDRO_DISABLE_PASSD_MIRROR");
    static const bool skip_passd_a =
        nopd_env_a && nopd_env_a[0] == '1' && nopd_env_a[1] == '\0';
    if (!skip_passd_a && !m_uiPassDDemotedToGhostCg.empty()) {
        for (auto& kv : m_uiPassDDemotedToGhostCg) {
            vec[kv.first] = vec[kv.second];
        }
    }
    if (!m_uiZipNonPrimaryToGhostCg.empty()) {
        for (auto& kv : m_uiZipNonPrimaryToGhostCg) {
            vec[kv.first] = vec[kv.second];
        }
    }
    static const char* erfix_env_a =
        std::getenv("DENDRO_ENABLE_ELEM_READ_FIXUP");
    static const bool erfix_a =
        erfix_env_a && erfix_env_a[0] == '1' && erfix_env_a[1] == '\0';
    if (erfix_a && !m_uiZipGhostToLocalAtConsensus.empty()) {
        for (auto& kv : m_uiZipGhostToLocalAtConsensus) {
            vec[kv.first] = vec[kv.second];
        }
    }
}

/**@brief Side-channel sync: Alltoallv that delivers each primary
 * cg's value (on the primary's rank) to the corresponding non-primary
 * local cg slot on every other rank that has a duplicate at the same
 * phys_pos. Built by buildZipPlan() from the global allgathered
 * claims. dof = number of variables packed in vec (each var occupies
 * m_uiNumActualNodes contiguous slots). */
template <typename T>
void Mesh::syncZipNonPrimary(T* vec, unsigned int dof) {
    // intra-rank duplicate sync: copy primary value to local duplicate
    // cgs. covers the case where the same rank has two local cgs at
    // the same phys_pos (e.g. hanging-face neighborhood with multiple
    // local writers). cross-rank Alltoallv below skips intra-rank
    // pairs; this loop fills that gap. runs unconditionally (no MPI).
    if (!m_uiZipLocalDupSrc.empty()) {
        const size_t Nd = m_uiZipLocalDupSrc.size();
        for (unsigned int v = 0; v < dof; v++) {
            T* vp = vec + v * m_uiNumActualNodes;
            for (size_t k = 0; k < Nd; k++)
                vp[m_uiZipLocalDupDst[k]] = vp[m_uiZipLocalDupSrc[k]];
        }
    }
    if (m_uiActiveNpes <= 1) return;
    if (m_uiZipSyncSendCg.empty() && m_uiZipSyncRecvCg.empty()) return;

    const size_t sendN = m_uiZipSyncSendCg.size();
    const size_t recvN = m_uiZipSyncRecvCg.size();
    std::vector<T> sendBuf(sendN * dof);
    std::vector<T> recvBuf(recvN * dof);

    // Single batched Alltoallv over all dof (was dof separate calls).
    // m_uiZipSyncSendCg is already ordered by destination rank, so cg
    // entry k belongs to a contiguous per-rank block. Interleave the
    // dof values at each k: sendBuf[k*dof + v]. Counts/offsets scale by
    // dof. One latency-bound collective instead of dof of them — the
    // post-axpy sync runs every RK substage, so this matters.
    for (size_t k = 0; k < sendN; k++) {
        const unsigned int cg = m_uiZipSyncSendCg[k];
        T* dst = sendBuf.data() + k * dof;
        for (unsigned int v = 0; v < dof; v++)
            dst[v] = vec[v * m_uiNumActualNodes + cg];
    }
    std::vector<int> sCnt(m_uiActiveNpes), sOff(m_uiActiveNpes),
        rCnt(m_uiActiveNpes), rOff(m_uiActiveNpes);
    for (int p = 0; p < m_uiActiveNpes; p++) {
        sCnt[p] = (int)(m_uiZipSyncSendCounts[p] * dof);
        sOff[p] = (int)(m_uiZipSyncSendOffsets[p] * dof);
        rCnt[p] = (int)(m_uiZipSyncRecvCounts[p] * dof);
        rOff[p] = (int)(m_uiZipSyncRecvOffsets[p] * dof);
    }
    par::Mpi_Alltoallv(sendBuf.data(), sCnt.data(), sOff.data(),
                       recvBuf.data(), rCnt.data(), rOff.data(),
                       m_uiCommActive);
    for (size_t k = 0; k < recvN; k++) {
        const unsigned int cg = m_uiZipSyncRecvCg[k];
        const T* src = recvBuf.data() + k * dof;
        for (unsigned int v = 0; v < dof; v++)
            vec[v * m_uiNumActualNodes + cg] = src[v];
    }
}

template <typename T>
void Mesh::broadcastCgValuesByPhysPos(T* vec, unsigned int dof) {
    // gate: DENDRO_FORCE_POS_BCAST=1 enables. Disabled by default because
    // it costs an extra allgather + alltoallv per call. enabled by
    // em4_partitioning.h::redistributeDVec right after the dstMesh ghost
    // exchange to bring graph-mode's ghost cgs into bit-identity with SFC
    // at consensus phys positions post-AMR sandwich.
    if (!m_uiIsActive) return;
    if (m_uiActiveNpes <= 1) return;
    static const char* gate_env = std::getenv("DENDRO_FORCE_POS_BCAST");
    static const bool gate_on =
        gate_env && gate_env[0] == '1' && gate_env[1] == '\0';
    if (!gate_on) return;

    const unsigned int npe = m_uiNpE;
    const unsigned int eOrd = m_uiElementOrder;
    const unsigned int nTotal_cg = m_uiNumActualNodes;
    const unsigned int nLB = m_uiNodeLocalBegin;
    const unsigned int nLE = m_uiNodeLocalEnd;

    struct Claim {
        unsigned long long x, y, z;
        int rank;
        unsigned int cg;
        unsigned char is_local;
        // canonical-owner tie-break fields (gated DENDRO_BCAST_TIEBREAK_SUBN=1).
        // sub_n is the node-within-element linear index; lower = corner closer
        // to the element's anchor (sub=(0,0,0)), which matches the SFC canonical-
        // owner for a shared corner. lev is the owner element's level — coarser
        // (lower lev) wins ties.
        unsigned char lev;
        unsigned short sub_n;
    };

    // step 1: walk every cg on this rank, compute its phys_pos via cg2dg.
    std::vector<Claim> myClaims;
    myClaims.reserve(nTotal_cg);
    for (unsigned int cg = 0; cg < nTotal_cg; cg++) {
        if (cg >= m_uiCG2DG.size()) continue;
        const unsigned int dg = m_uiCG2DG[cg];
        if (dg == LOOK_UP_TABLE_DEFAULT) continue;
        const unsigned int e = dg / npe;
        const unsigned int n = dg % npe;
        if (e >= m_uiAllElements.size()) continue;
        const ot::TreeNode& tn = m_uiAllElements[e];
        const unsigned int lev = tn.getLevel();
        if (lev > m_uiMaxDepth) continue;
        const unsigned long long len =
            (unsigned long long)1 << (m_uiMaxDepth - lev);
        const unsigned int ni = n % (eOrd + 1);
        const unsigned int nj = (n / (eOrd + 1)) % (eOrd + 1);
        const unsigned int nk = n / ((eOrd + 1) * (eOrd + 1));
        Claim c;
        c.x = (unsigned long long)tn.getX() * eOrd
              + (unsigned long long)ni * len;
        c.y = (unsigned long long)tn.getY() * eOrd
              + (unsigned long long)nj * len;
        c.z = (unsigned long long)tn.getZ() * eOrd
              + (unsigned long long)nk * len;
        c.rank = m_uiActiveRank;
        c.cg = cg;
        c.is_local = (cg >= nLB && cg < nLE) ? 1u : 0u;
        c.lev = (unsigned char)lev;
        c.sub_n = (unsigned short)n;
        myClaims.push_back(c);
    }

    // step 2: allgather claims across ranks. each rank produces the same
    // global ordering by sorting; we use this to pick a deterministic
    // canonical (rank, cg) per phys_pos.
    const int myCount = (int)myClaims.size();
    std::vector<int> counts(m_uiActiveNpes), offs(m_uiActiveNpes, 0);
    MPI_Allgather(&myCount, 1, MPI_INT, counts.data(), 1, MPI_INT,
                  m_uiCommActive);
    int total = 0;
    for (int p = 0; p < m_uiActiveNpes; p++) {
        offs[p] = total;
        total += counts[p];
    }
    std::vector<unsigned long long> allX(total), allY(total), allZ(total);
    std::vector<unsigned int> allCg(total);
    std::vector<unsigned char> allLocal(total);
    std::vector<unsigned char> allLev(total);
    std::vector<unsigned short> allSubN(total);
    std::vector<unsigned long long> myX(myCount), myY(myCount), myZ(myCount);
    std::vector<unsigned int> myCg(myCount);
    std::vector<unsigned char> myLocalFlag(myCount);
    std::vector<unsigned char> myLev(myCount);
    std::vector<unsigned short> mySubN(myCount);
    for (int i = 0; i < myCount; i++) {
        myX[i] = myClaims[i].x;
        myY[i] = myClaims[i].y;
        myZ[i] = myClaims[i].z;
        myCg[i] = myClaims[i].cg;
        myLocalFlag[i] = myClaims[i].is_local;
        myLev[i] = myClaims[i].lev;
        mySubN[i] = myClaims[i].sub_n;
    }
    MPI_Allgatherv(myX.data(), myCount, MPI_UINT64_T, allX.data(),
                   counts.data(), offs.data(), MPI_UINT64_T,
                   m_uiCommActive);
    MPI_Allgatherv(myY.data(), myCount, MPI_UINT64_T, allY.data(),
                   counts.data(), offs.data(), MPI_UINT64_T,
                   m_uiCommActive);
    MPI_Allgatherv(myZ.data(), myCount, MPI_UINT64_T, allZ.data(),
                   counts.data(), offs.data(), MPI_UINT64_T,
                   m_uiCommActive);
    MPI_Allgatherv(myCg.data(), myCount, MPI_UNSIGNED, allCg.data(),
                   counts.data(), offs.data(), MPI_UNSIGNED,
                   m_uiCommActive);
    MPI_Allgatherv(myLocalFlag.data(), myCount, MPI_UNSIGNED_CHAR,
                   allLocal.data(), counts.data(), offs.data(),
                   MPI_UNSIGNED_CHAR, m_uiCommActive);
    MPI_Allgatherv(myLev.data(), myCount, MPI_UNSIGNED_CHAR,
                   allLev.data(), counts.data(), offs.data(),
                   MPI_UNSIGNED_CHAR, m_uiCommActive);
    MPI_Allgatherv(mySubN.data(), myCount, MPI_UNSIGNED_SHORT,
                   allSubN.data(), counts.data(), offs.data(),
                   MPI_UNSIGNED_SHORT, m_uiCommActive);

    // step 3: build phys_pos -> canonical claim. canonical rule:
    //   1. prefer LOCAL claims (is_local=1) over GHOST claims.
    //   2. among locals: lowest rank wins; tie by lowest cg index.
    //   3. if no local exists at this phys: lowest rank's ghost wins.
    struct PhysKey3 {
        unsigned long long x, y, z;
        bool operator==(const PhysKey3& o) const {
            return x == o.x && y == o.y && z == o.z;
        }
    };
    struct PhysKey3Hash {
        size_t operator()(const PhysKey3& k) const {
            size_t h = std::hash<unsigned long long>()(k.x);
            h ^= std::hash<unsigned long long>()(k.y)
                + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            h ^= std::hash<unsigned long long>()(k.z)
                + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            return h;
        }
    };
    struct Winner {
        int rank;
        unsigned int cg;
        unsigned char is_local;
        unsigned char lev;
        unsigned short sub_n;
    };
    // bcast-allgather probe: dump every (rank, cg, advertised_phys,
    // is_local, value) record matching target phys. set
    // EM4_BCAST_PROBE_PHYS="x,y,z" and EM4_BCAST_PROBE_DIR to enable.
    // EM4_BCAST_PROBE_MIN_CALL / _MAX_CALL gate which calls to dump.
    static const char* bcp_dir = DENDRO_PROBE_GETENV("EM4_BCAST_PROBE_DIR");
    static const char* bcp_phys = DENDRO_PROBE_GETENV("EM4_BCAST_PROBE_PHYS");
    static const char* bcp_min_env = DENDRO_PROBE_GETENV("EM4_BCAST_PROBE_MIN_CALL");
    static const char* bcp_max_env = DENDRO_PROBE_GETENV("EM4_BCAST_PROBE_MAX_CALL");
    static const int bcp_min = bcp_min_env ? std::atoi(bcp_min_env) : 0;
    static const int bcp_max = bcp_max_env ? std::atoi(bcp_max_env) : 999999;
    static int bcp_call = 0;
    static unsigned long long bcp_x = 0, bcp_y = 0, bcp_z = 0;
    static bool bcp_phys_on = false;
    if (bcp_phys && !bcp_phys_on) {
        std::sscanf(bcp_phys, "%llu,%llu,%llu", &bcp_x, &bcp_y, &bcp_z);
        bcp_phys_on = true;
    }
    const bool bcp_in_range = (bcp_call >= bcp_min && bcp_call < bcp_max);
    FILE* bcp_fp = nullptr;
    if (bcp_dir && bcp_phys_on && bcp_in_range) {
        char fn[1024];
        std::snprintf(fn, sizeof(fn), "%s/bcast_call%d_r%d.txt",
                      bcp_dir, bcp_call, (int)m_uiActiveRank);
        bcp_fp = std::fopen(fn, "w");
        if (bcp_fp) {
            std::fprintf(bcp_fp,
                "# call=%d rank=%d target_phys=(%llu,%llu,%llu) dof=%u\n"
                "# advertised_rank advertised_cg advertised_phys is_local "
                "value_hex value_dec\n",
                bcp_call, (int)m_uiActiveRank, bcp_x, bcp_y, bcp_z, dof);
        }
    }

    std::unordered_map<PhysKey3, Winner, PhysKey3Hash> winners;
    winners.reserve(total);
    for (int p = 0; p < m_uiActiveNpes; p++) {
        for (int i = offs[p]; i < offs[p] + counts[p]; i++) {
            PhysKey3 k{allX[i], allY[i], allZ[i]};

            if (bcp_fp && allX[i] == bcp_x && allY[i] == bcp_y
                && allZ[i] == bcp_z) {
                // read this rank's cg value (for var=0) if this rec is on us.
                T val_to_log = (T)0;
                if (p == (int)m_uiActiveRank
                    && allCg[i] < (unsigned int)nTotal_cg) {
                    val_to_log = vec[allCg[i]];
                }
                uint64_t hb = 0;
                std::memcpy(&hb, &val_to_log, sizeof(hb));
                std::fprintf(bcp_fp,
                    "%d %u (%llu,%llu,%llu) %d %lx %g\n",
                    p, allCg[i], allX[i], allY[i], allZ[i],
                    (int)allLocal[i], (unsigned long)hb,
                    (double)val_to_log);
            }

            auto it = winners.find(k);
            if (it == winners.end()) {
                winners[k] = Winner{p, allCg[i], allLocal[i],
                                    allLev[i], allSubN[i]};
                continue;
            }
            Winner& w = it->second;
            // prefer local over ghost
            if (allLocal[i] && !w.is_local) {
                w = Winner{p, allCg[i], allLocal[i], allLev[i], allSubN[i]};
                continue;
            }
            if (!allLocal[i] && w.is_local) continue;
            // same is_local class: lower rank then lower cg
            bool take = (p < w.rank
                         || (p == w.rank && allCg[i] < w.cg));
            if (take) {
                w = Winner{p, allCg[i], allLocal[i], allLev[i], allSubN[i]};
            }
        }
    }

    // log canonical winner for target phys.
    if (bcp_fp && bcp_phys_on) {
        PhysKey3 k{bcp_x, bcp_y, bcp_z};
        auto wit = winners.find(k);
        if (wit != winners.end()) {
            const Winner& w = wit->second;
            std::fprintf(bcp_fp,
                "# CANONICAL WINNER: rank=%d cg=%u is_local=%d\n",
                w.rank, w.cg, (int)w.is_local);
        } else {
            std::fprintf(bcp_fp,
                "# NO winner found for target phys (no advertise records)\n");
        }
        std::fclose(bcp_fp);
    }
    if (bcp_dir) bcp_call++;

    // step 4: build per-rank send/recv lists.
    //   if I am the canonical rank for a phys_pos, I send the value to
    //   every (rank != me) that has a cg at this phys_pos.
    //   if I am NOT canonical, I expect a recv from the canonical rank
    //   into my cg at this phys_pos.
    std::vector<int> sendCounts(m_uiActiveNpes, 0);
    std::vector<int> recvCounts(m_uiActiveNpes, 0);
    for (int p = 0; p < m_uiActiveNpes; p++) {
        for (int i = offs[p]; i < offs[p] + counts[p]; i++) {
            PhysKey3 k{allX[i], allY[i], allZ[i]};
            auto wit = winners.find(k);
            if (wit == winners.end()) continue;
            const Winner& w = wit->second;
            if (p == w.rank) continue;  // intra-rank (incl canonical itself);
                                        // handled by local fixup below
            // (rank p, cg allCg[i]) is a cross-rank non-canonical recipient.
            if (m_uiActiveRank == w.rank) sendCounts[p]++;
            if (m_uiActiveRank == p) recvCounts[w.rank]++;
        }
    }
    std::vector<int> sendOffsets(m_uiActiveNpes, 0);
    std::vector<int> recvOffsets(m_uiActiveNpes, 0);
    for (int p = 1; p < m_uiActiveNpes; p++) {
        sendOffsets[p] = sendOffsets[p - 1] + sendCounts[p - 1];
        recvOffsets[p] = recvOffsets[p - 1] + recvCounts[p - 1];
    }
    const int totalSend = sendOffsets[m_uiActiveNpes - 1]
                          + sendCounts[m_uiActiveNpes - 1];
    const int totalRecv = recvOffsets[m_uiActiveNpes - 1]
                          + recvCounts[m_uiActiveNpes - 1];
    std::vector<unsigned int> sendSrcCg(totalSend);  // my local cg to read
    std::vector<unsigned int> recvDstCg(totalRecv);  // my cg to write
    std::vector<int> sendPos(m_uiActiveNpes), recvPos(m_uiActiveNpes);
    for (int p = 0; p < m_uiActiveNpes; p++) {
        sendPos[p] = sendOffsets[p];
        recvPos[p] = recvOffsets[p];
    }
    for (int p = 0; p < m_uiActiveNpes; p++) {
        for (int i = offs[p]; i < offs[p] + counts[p]; i++) {
            PhysKey3 k{allX[i], allY[i], allZ[i]};
            auto wit = winners.find(k);
            if (wit == winners.end()) continue;
            const Winner& w = wit->second;
            if (p == w.rank) continue;
            if (m_uiActiveRank == w.rank) {
                sendSrcCg[sendPos[p]++] = w.cg;
            }
            if (m_uiActiveRank == p) {
                recvDstCg[recvPos[w.rank]++] = allCg[i];
            }
        }
    }

    // step 5: Alltoallv per dof.
    std::vector<T> sendBuf(totalSend * dof);
    std::vector<T> recvBuf(totalRecv * dof);
    for (unsigned int v = 0; v < dof; v++) {
        const T* vp = vec + v * nTotal_cg;
        T* sb = sendBuf.data() + v * totalSend;
        for (int k = 0; k < totalSend; k++)
            sb[k] = vp[sendSrcCg[k]];
    }
    for (unsigned int v = 0; v < dof; v++) {
        par::Mpi_Alltoallv(
            sendBuf.data() + v * totalSend,
            sendCounts.data(), sendOffsets.data(),
            recvBuf.data() + v * totalRecv,
            recvCounts.data(), recvOffsets.data(),
            m_uiCommActive);
    }
    for (unsigned int v = 0; v < dof; v++) {
        T* vp = vec + v * nTotal_cg;
        const T* rb = recvBuf.data() + v * totalRecv;
        for (int k = 0; k < totalRecv; k++)
            vp[recvDstCg[k]] = rb[k];
    }

    // intra-rank duplicates: if same rank has multiple cgs at the same
    // phys_pos and we're the canonical rank, the Alltoallv loop above
    // skipped them (p == m_uiActiveRank && sendCounts/recvCounts only
    // counts cross-rank pairs). pick them up locally.
    for (int p = 0; p < m_uiActiveNpes; p++) {
        for (int i = offs[p]; i < offs[p] + counts[p]; i++) {
            if (p != m_uiActiveRank) continue;
            PhysKey3 k{allX[i], allY[i], allZ[i]};
            auto wit = winners.find(k);
            if (wit == winners.end()) continue;
            const Winner& w = wit->second;
            if (w.rank != m_uiActiveRank) continue;
            if ((int)allCg[i] == (int)w.cg) continue;
            for (unsigned int v = 0; v < dof; v++) {
                T* vp = vec + v * nTotal_cg;
                vp[allCg[i]] = vp[w.cg];
            }
        }
    }

    // post-pass: for phys positions where NO rank has a LOCAL writer
    // (winner.is_local == 0), zero out the local cg at this rank. these
    // are "fully hanging" positions — SFC's cascade never zips a value
    // here, so the cg stays at the IC value (0 in the far-field for EM4).
    // graph mode's redistribute may have deposited an interp residual
    // at these positions; zero them to match SFC.
    for (int i = 0; i < (int)myCount; i++) {
        if (myClaims[i].rank != m_uiActiveRank) continue;
        PhysKey3 k{myClaims[i].x, myClaims[i].y, myClaims[i].z};
        auto wit = winners.find(k);
        if (wit == winners.end()) continue;
        if (wit->second.is_local) continue;  // some rank owns it; keep value
        // no local writer anywhere — zero out my cg at this position
        const unsigned int my_cg = myClaims[i].cg;
        for (unsigned int v = 0; v < dof; v++) {
            T* vp = vec + v * nTotal_cg;
            vp[my_cg] = (T)0;
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

    // Side-channel non-primary sync runs before the standard ghost
    // exchange so updated non-primary local cgs are captured by the
    // send-buffer pack and propagated to ghost cgs transitively.
    syncZipNonPrimary(vec, dof);

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

            // active recv procs
            for (unsigned int recv_p = 0; recv_p < recvProcList.size();
                 recv_p++) {
                proc_id          = recvProcList[recv_p];
                MPI_Request* req = new MPI_Request();
                par::Mpi_Irecv((recvB + dof * nodeRecvOffset[proc_id]),
                               dof * nodeRecvCount[proc_id], proc_id,
                               m_uiCommTag, commActive, req);
                ctx.getRequestList().push_back(req);
                // std::cout << this->getMPIRank() << ": receiving from proc "
                // << proc_id << " a total of " << dof * nodeRecvCount[proc_id]
                // << std::endl;
            }
        }

        if (sendBSz) {
            ctx.allocateSendBuffer(sizeof(T) * dof * sendBSz);
            sendB = (T*)ctx.getSendBuffer();

            // Graph-partitioned meshes tag some R2-boundary send
            // entries as DG — sendNodeSM[k] then encodes
            // (localEle-localBegin)*NpE + sub rather than a CG index.
            // Read the current CG value via the sender's local E2N_CG
            // mapping so we don't rely on the stale m_uiLocalNodalDG
            // buffer (only populated by createVector(vec, func)).
            const bool useDG =
                (m_uiScatterMapSendIsDG.size() ==
                 m_uiScatterMapActualNodeSend.size());

            for (unsigned int send_p = 0; send_p < sendProcList.size();
                 send_p++) {
                proc_id = sendProcList[send_p];

                for (unsigned int var = 0; var < dof; var++) {
                    for (unsigned int k = nodeSendOffset[proc_id];
                         k < (nodeSendOffset[proc_id] + nodeSendCount[proc_id]);
                         k++) {
                        unsigned int idx = sendNodeSM[k];
                        if (useDG && m_uiScatterMapSendIsDG[k]) {
                            const unsigned int eLocal = idx / m_uiNpE;
                            const unsigned int sub    = idx % m_uiNpE;
                            const unsigned int e =
                                eLocal + m_uiElementLocalBegin;
                            idx =
                                m_uiE2NMapping_CG[e * m_uiNpE + sub];
                        }
                        sendB[dof * (nodeSendOffset[proc_id]) +
                              (var * nodeSendCount[proc_id]) +
                              (k - nodeSendOffset[proc_id])] =
                            (vec + var * m_uiNumActualNodes)[idx];
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
                // std::cout << this->getMPIRank() << ": sending from proc " <<
                // proc_id << " a total of " << dof * nodeSendOffset[proc_id] <<
                // std::endl;
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

        MPI_Status status;
        // need to wait for the commns to finish ...
        for (unsigned int i = 0;
             i < m_uiMPIContexts[ctxIndex].getRequestList().size(); i++) {
            MPI_Wait(m_uiMPIContexts[ctxIndex].getRequestList()[i], &status);
        }

        if (recvBSz) {
            // copy the recv data to the vec
            recvB = (T*)m_uiMPIContexts[ctxIndex].getRecvBuffer();

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

        // Sync non-primary local cgs from ghost (Stage 3 + legacy
        // Pass D path). See performGhostExchange for rationale.
        static const char* nopd_env_b =
            std::getenv("DENDRO_DISABLE_PASSD_MIRROR");
        static const bool skip_passd_b =
            nopd_env_b && nopd_env_b[0] == '1' && nopd_env_b[1] == '\0';
        if (!skip_passd_b && !m_uiPassDDemotedToGhostCg.empty()) {
            for (unsigned int var = 0; var < dof; var++) {
                T* vptr = vec + var * m_uiNumActualNodes;
                for (auto& kv : m_uiPassDDemotedToGhostCg) {
                    vptr[kv.first] = vptr[kv.second];
                }
            }
        }
        if (!m_uiZipNonPrimaryToGhostCg.empty()) {
            for (unsigned int var = 0; var < dof; var++) {
                T* vptr = vec + var * m_uiNumActualNodes;
                for (auto& kv : m_uiZipNonPrimaryToGhostCg) {
                    vptr[kv.first] = vptr[kv.second];
                }
            }
        }
        static const char* erfix_env_b =
            std::getenv("DENDRO_ENABLE_ELEM_READ_FIXUP");
        static const bool erfix_b =
            erfix_env_b && erfix_env_b[0] == '1' && erfix_env_b[1] == '\0';
        if (erfix_b && !m_uiZipGhostToLocalAtConsensus.empty()) {
            for (unsigned int var = 0; var < dof; var++) {
                T* vptr = vec + var * m_uiNumActualNodes;
                for (auto& kv : m_uiZipGhostToLocalAtConsensus) {
                    vptr[kv.first] = vptr[kv.second];
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
void Mesh::readFromGhostBegin(AsyncExchangeContex& ctx, T* vec,
                              unsigned int dof) {
    if (this->getMPICommSizeGlobal() == 1 || (!m_uiIsActive)) return;

    // Side-channel non-primary sync runs before the standard ghost
    // exchange (see performGhostExchange comment for rationale).
    syncZipNonPrimary(vec, dof);

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
            // Graph-partitioned meshes tag some send entries as DG —
            // sendNodeSM[k] then encodes (eLocal*NpE + sub) rather than
            // a CG index. Translate via E2N_CG on the fly so we read the
            // current vec value, same as the non-ctx variant does.
            // Without this, async unzip (Ctx::unzip) reads garbage from
            // vec[DG_encoded_idx] for every R2-boundary send, which
            // corrupts ghost values and poisons every RK step on NLSM.
            const bool useDG =
                (m_uiScatterMapSendIsDG.size() ==
                 m_uiScatterMapActualNodeSend.size());
            for (unsigned int send_p = 0; send_p < sendProcList.size();
                 send_p++) {
                proc_id = sendProcList[send_p];

                for (unsigned int var = 0; var < dof; var++) {
                    for (unsigned int k = nodeSendOffset[proc_id];
                         k < (nodeSendOffset[proc_id] + nodeSendCount[proc_id]);
                         k++) {
                        unsigned int idx = sendNodeSM[k];
                        if (useDG && m_uiScatterMapSendIsDG[k]) {
                            const unsigned int eLocal = idx / m_uiNpE;
                            const unsigned int sub    = idx % m_uiNpE;
                            const unsigned int e =
                                eLocal + m_uiElementLocalBegin;
                            idx =
                                m_uiE2NMapping_CG[e * m_uiNpE + sub];
                        }
                        sendB[dof * (nodeSendOffset[proc_id]) +
                              (var * nodeSendCount[proc_id]) +
                              (k - nodeSendOffset[proc_id])] =
                            (vec + var * m_uiNumActualNodes)[idx];
                    }
                }
            }

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
        MPI_Waitall(sendProcList.size(), ctx.m_send_req.data(),
                    MPI_STATUSES_IGNORE);
        MPI_Waitall(recvProcList.size(), ctx.m_recv_req.data(),
                    MPI_STATUSES_IGNORE);

        if (recvBSz) {
            // copy the recv data to the vec
            recvB = (T*)ctx.getRecvBuffer();

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

            // diagnostic: log ghost-recv events to find the (phys, var)
            // that first diverges by 1 ULP between graph and SFC. tracks
            // the per-step drift documented in docs/findings_2026-05-11.md.
            // off by default; gated on DENDRO_GHOST_RECV_PROBE=1.
            static const char* grp_env =
                DENDRO_PROBE_GETENV("DENDRO_GHOST_RECV_PROBE");
            static const bool grp_on =
                grp_env && grp_env[0] == '1' && grp_env[1] == '\0';
            if (grp_on) {
                static const char* grp_rank_env =
                    DENDRO_PROBE_GETENV("DENDRO_GHOST_RECV_PROBE_RANK");
                static const int grp_rank =
                    grp_rank_env ? std::atoi(grp_rank_env) : 2;
                static const char* grp_max_env =
                    DENDRO_PROBE_GETENV("DENDRO_GHOST_RECV_PROBE_CALL_MAX");
                static const long long grp_max =
                    grp_max_env ? std::atoll(grp_max_env) : -1;
                static const char* grp_call_min_env =
                    DENDRO_PROBE_GETENV("DENDRO_GHOST_RECV_PROBE_CALL_MIN");
                static const long long grp_call_min =
                    grp_call_min_env ? std::atoll(grp_call_min_env) : 0;
                // tbox filter: "cx,cy,cz,r[;cx,cy,cz,r]..."
                // entries are logged only if tkey is within r of any
                // center (axis-aligned cube; r=0 means exact match).
                // when unset, no filter (log all).
                struct Tbox {
                    long long cx, cy, cz, r;
                };
                static std::vector<Tbox> grp_tboxes = []() {
                    std::vector<Tbox> out;
                    const char* env =
                        DENDRO_PROBE_GETENV("DENDRO_GHOST_RECV_PROBE_TBOX");
                    if (!env || env[0] == '\0') return out;
                    std::string s(env);
                    size_t pos = 0;
                    while (pos < s.size()) {
                        size_t semi = s.find(';', pos);
                        std::string item = s.substr(pos,
                            (semi == std::string::npos)
                                ? std::string::npos : (semi - pos));
                        long long cx = 0, cy = 0, cz = 0, r = 0;
                        if (sscanf(item.c_str(),
                                   "%lld,%lld,%lld,%lld",
                                   &cx, &cy, &cz, &r) == 4) {
                            out.push_back({cx, cy, cz, r});
                        }
                        if (semi == std::string::npos) break;
                        pos = semi + 1;
                    }
                    return out;
                }();
                static long long grp_call = 0;
                grp_call++;
                if ((int)m_uiActiveRank == grp_rank &&
                    grp_call >= grp_call_min &&
                    (grp_max < 0 || grp_call <= grp_max)) {
                    const unsigned int npe_loc      = m_uiNpE;
                    const unsigned int eOrd_loc     = m_uiElementOrder;
                    const ot::TreeNode* pNodes_loc  =
                        m_uiAllElements.data();
                    const size_t allElemSize_loc    =
                        m_uiAllElements.size();
                    for (unsigned int recv_p = 0;
                         recv_p < recvProcList.size(); recv_p++) {
                        unsigned int grp_proc =
                            recvProcList[recv_p];
                        for (unsigned int var = 0; var < dof; var++) {
                            for (unsigned int k =
                                     nodeRecvOffset[grp_proc];
                                 k < (nodeRecvOffset[grp_proc]
                                      + nodeRecvCount[grp_proc]);
                                 k++) {
                                const unsigned int recv_cg =
                                    recvNodeSM[k];
                                const T val = (vec
                                    + var * m_uiNumActualNodes)[recv_cg];
                                unsigned long long tx = 0, ty = 0, tz = 0;
                                unsigned int elem_p = 0;
                                unsigned int sub_p  = 0;
                                unsigned int lev_p  = 0;
                                if (recv_cg < m_uiNumActualNodes) {
                                    const unsigned int dg_idx =
                                        m_uiCG2DG[recv_cg];
                                    elem_p = dg_idx / npe_loc;
                                    sub_p  = dg_idx % npe_loc;
                                    if (elem_p < allElemSize_loc) {
                                        lev_p =
                                            pNodes_loc[elem_p].getLevel();
                                        if (lev_p <= m_uiMaxDepth) {
                                            const unsigned long long len =
                                                (unsigned long long)1
                                                << (m_uiMaxDepth - lev_p);
                                            const unsigned int ni =
                                                sub_p % (eOrd_loc + 1);
                                            const unsigned int nj =
                                                (sub_p / (eOrd_loc + 1))
                                                % (eOrd_loc + 1);
                                            const unsigned int nk =
                                                sub_p
                                                / ((eOrd_loc + 1)
                                                   * (eOrd_loc + 1));
                                            tx = (unsigned long long)
                                                     pNodes_loc[elem_p]
                                                         .getX()
                                                 * eOrd_loc
                                                 + (unsigned long long)ni
                                                       * len;
                                            ty = (unsigned long long)
                                                     pNodes_loc[elem_p]
                                                         .getY()
                                                 * eOrd_loc
                                                 + (unsigned long long)nj
                                                       * len;
                                            tz = (unsigned long long)
                                                     pNodes_loc[elem_p]
                                                         .getZ()
                                                 * eOrd_loc
                                                 + (unsigned long long)nk
                                                       * len;
                                        }
                                    }
                                }
                                bool pass_box = grp_tboxes.empty();
                                if (!pass_box) {
                                    for (const auto& bx : grp_tboxes) {
                                        if (std::llabs((long long)tx
                                                       - bx.cx) <= bx.r
                                            && std::llabs((long long)ty
                                                          - bx.cy) <= bx.r
                                            && std::llabs((long long)tz
                                                          - bx.cz) <= bx.r) {
                                            pass_box = true;
                                            break;
                                        }
                                    }
                                }
                                if (!pass_box) continue;
                                uint64_t hexBits = 0;
                                std::memcpy(&hexBits, &val,
                                            sizeof(hexBits));
                                std::cout << "[ghost-recv] call="
                                          << grp_call
                                          << " dof=" << dof
                                          << " var=" << var
                                          << " recv_cg=" << recv_cg
                                          << " elem=" << elem_p
                                          << " lev=" << lev_p
                                          << " sub=" << sub_p
                                          << " tkey=" << tx << ","
                                          << ty << "," << tz
                                          << " sender_rank=" << grp_proc
                                          << " sender_idx="
                                          << (k - nodeRecvOffset[grp_proc])
                                          << " hex=0x" << std::hex
                                          << hexBits << std::dec
                                          << std::endl;
                            }
                        }
                    }
                }
            }
        }

        // Sync non-primary local cgs from ghost (Stage 3 + legacy
        // Pass D path). See performGhostExchange for rationale.
        //
        // The legacy PassD mirror picks "smallest-indexed ghost cg at
        // phys_pos" which doesn't always correspond to the cascade's
        // actual ship source for the canonical primary. When wrong, it
        // OVERWRITES the correct value that syncZipNonPrimary placed
        // into the demoted local cg at the start of readFromGhost.
        // Env gate `DENDRO_DISABLE_PASSD_MIRROR=1` skips it; Stage 3
        // mirror (which uses inverse scatter map lookup, correct
        // source) handles the cases it covers, and syncZipNonPrimary
        // handles the rest via direct Alltoallv.
        static const char* nopd_env =
            std::getenv("DENDRO_DISABLE_PASSD_MIRROR");
        static const bool skip_passd_mirror =
            nopd_env && nopd_env[0] == '1' && nopd_env[1] == '\0';
        if (!skip_passd_mirror && !m_uiPassDDemotedToGhostCg.empty()) {
            for (unsigned int var = 0; var < dof; var++) {
                T* vptr = vec + var * m_uiNumActualNodes;
                for (auto& kv : m_uiPassDDemotedToGhostCg) {
                    vptr[kv.first] = vptr[kv.second];
                }
            }
        }
        if (!m_uiZipNonPrimaryToGhostCg.empty()) {
            for (unsigned int var = 0; var < dof; var++) {
                T* vptr = vec + var * m_uiNumActualNodes;
                for (auto& kv : m_uiZipNonPrimaryToGhostCg) {
                    vptr[kv.first] = vptr[kv.second];
                }
            }
        }

        // Re-run syncZipNonPrimary AFTER the standard ghost exchange.
        // The first call (in readFromGhostBegin) delivers primary's
        // value to non-primary local AND ghost cgs at consensus
        // phys_pos (with the ghost-inclusive advertisement). But the
        // standard ghost exchange runs LATER and writes to ghost cgs
        // via the legacy scatter map — which may source from a wrong
        // rank's local cg with a stale (IC) value. Re-running sync
        // after the standard scatter restores the primary's value.
        // The send half of the alltoallv is unchanged; the recv side
        // overwrites stale standard-scatter writes.
        //
        // Env gate `DENDRO_DISABLE_POST_SYNC=1` skips this for A/B.
        // The drift investigation needs to test whether the
        // post-scatter overwrite is the source of the per-step
        // ~2.5e-12 noise on rank 2's +X edge.
        static const char* nopost_env =
            std::getenv("DENDRO_DISABLE_POST_SYNC");
        static const bool skip_post_sync =
            nopost_env && nopost_env[0] == '1' && nopost_env[1] == '\0';
        if (!skip_post_sync) {
            this->syncZipNonPrimary(vec, dof);
        }

        static const char* erfix_env_c =
            std::getenv("DENDRO_ENABLE_ELEM_READ_FIXUP");
        static const bool erfix_c =
            erfix_env_c && erfix_env_c[0] == '1' && erfix_env_c[1] == '\0';
        if (erfix_c && !m_uiZipGhostToLocalAtConsensus.empty()) {
            for (unsigned int var = 0; var < dof; var++) {
                T* vptr = vec + var * m_uiNumActualNodes;
                for (auto& kv : m_uiZipGhostToLocalAtConsensus) {
                    vptr[kv.first] = vptr[kv.second];
                }
            }
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
        const unsigned int recvBSz = m_uiRecvEleOffset[activeNpes - 1] +
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
        const unsigned int recvBSz = m_uiRecvEleOffset[activeNpes - 1] +
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
        const unsigned int recvBSz = (m_uiRecvEleOffset[activeNpes - 1] +
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
        const unsigned int recvBSz = (m_uiRecvEleOffset[activeNpes - 1] +
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

            // Graph-partitioned meshes tag some R2-boundary send
            // entries as DG — sendNodeSM[k] then encodes
            // (localEle-localBegin)*NpE + sub rather than a CG index.
            // Read the current CG value via the sender's local E2N_CG
            // mapping so we don't rely on the stale m_uiLocalNodalDG
            // buffer (only populated by createVector(vec, func)).
            const bool useDG =
                (m_uiScatterMapSendIsDG.size() ==
                 m_uiScatterMapActualNodeSend.size());

            for (unsigned int send_p = 0; send_p < sendProcList.size();
                 send_p++) {
                proc_id = sendProcList[send_p];

                for (unsigned int var = 0; var < dof; var++) {
                    for (unsigned int k = nodeSendOffset[proc_id];
                         k < (nodeSendOffset[proc_id] + nodeSendCount[proc_id]);
                         k++) {
                        unsigned int idx = sendNodeSM[k];
                        if (useDG && m_uiScatterMapSendIsDG[k]) {
                            const unsigned int eLocal = idx / m_uiNpE;
                            const unsigned int sub    = idx % m_uiNpE;
                            const unsigned int e =
                                eLocal + m_uiElementLocalBegin;
                            idx =
                                m_uiE2NMapping_CG[e * m_uiNpE + sub];
                        }
                        sendB[dof * (nodeSendOffset[proc_id]) +
                              (var * nodeSendCount[proc_id]) +
                              (k - nodeSendOffset[proc_id])] =
                            (vec + var * m_uiNumActualNodes)[idx];
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
                    sz = ((blkNode.maxX() - blkNode.minX())) /
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
            h = 1.0 / h;
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
            h = 1.0 / h;
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
            h = 1.0 / h;
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
        wavelet::WaveletEl* wrefEl            = new wavelet::WaveletEl(refEl);

        const std::vector<ot::Block>& blkList = this->getLocalBlockList();
        const unsigned int eOrder             = m_uiElementOrder;

        const unsigned int numLocalElements   = m_uiNumLocalElements;

        refine_flags.clear();
        refine_flags.resize(numLocalElements, OCT_NO_CHANGE);

        std::vector<T> blkIn;
        std::vector<double> wCout;
        const ot::TreeNode* pNodes = m_uiAllElements.data();

        std::vector<double> eleWMax;
        eleWMax.resize(numLocalElements, 0);

        const unsigned int eleOfst = m_uiElementLocalBegin;

        for (unsigned int blk = 0; blk < blkList.size(); blk++) {
            const unsigned int pw = blkList[blk].get1DPadWidth();
            if ((eOrder >> 1u) != pw) {
                std::cout << " padding width should be half the eleOrder for "
                             "generic wavelet computations. "
                          << std::endl;
                MPI_Abort(this->getMPICommunicator(), 0);
            }

            const unsigned int nx = (2 * eOrder + 1);
            const unsigned int ny = (2 * eOrder + 1);
            const unsigned int nz = (2 * eOrder + 1);

            // std::cout<<"nx "<<nx<<std::endl;

            blkIn.resize(numVars * nx * ny * nz);
            const unsigned int isz[] = {nx, ny, nz};
            const unsigned int bflag = blkList[blk].getBlkNodeFlag();

            for (unsigned int ele : blkList[blk]) {
                const unsigned int pw = blkList[blk].get1DPadWidth();
                const bool isBdyOct   = this->isBoundaryOctant(ele);

                const double oct_dx =
                    (1u << (m_uiMaxDepth - pNodes[ele].getLevel())) /
                    (double(m_uiElementOrder));
                Point oct_pt1 = Point(pNodes[ele].minX(), pNodes[ele].minY(),
                                      pNodes[ele].minZ());
                Point oct_pt2 = Point(pNodes[ele].minX() + oct_dx,
                                      pNodes[ele].minY() + oct_dx,
                                      pNodes[ele].minZ() + oct_dx);
                Point domain_pt1, domain_pt2, dx_domain;
                this->octCoordToDomainCoord(oct_pt1, domain_pt1);
                this->octCoordToDomainCoord(oct_pt2, domain_pt2);
                dx_domain    = domain_pt2 - domain_pt1;
                double hx[3] = {dx_domain.x(), dx_domain.y(), dx_domain.z()};
                const double tol_ele = wavelet_tol(
                    domain_pt1.x(), domain_pt1.y(), domain_pt1.z(), hx);

                if (!includeBdy && isBdyOct) {
                    // tol small enough to not refine but not to coarsen .
                    eleWMax[ele - eleOfst] = amr_coarse_fac * tol_ele + 1e-8;
                    continue;
                }

                for (unsigned int v = 0; v < numVars; v++) {
                    const unsigned int vid = varIds[v];
                    this->getUnzipElementalNodalValues(
                        unzippedVec[vid], blk, ele,
                        blkIn.data() + v * (nx * ny * nz), true);
                }

                // eleWMax[ele -
                // eleOfst]=wavelet::compute_element_wavelet(this,(const
                // wavelet::WaveletEl*)&wrefEl,blkIn.data(),tol_ele,numVars,isBdyOct);
                //  compute the wavelet
                {
                    double wMax           = 0.0;

                    const unsigned int nx = (2 * eOrder + 1);
                    const unsigned int ny = (2 * eOrder + 1);
                    const unsigned int nz = (2 * eOrder + 1);
                    assert(pw == (eOrder >> 1u));

                    const unsigned int sz_per_dof = nx * ny * nz;
                    const unsigned int isz[]      = {nx, ny, nz};
                    wCout.resize(sz_per_dof);

                    const unsigned int dof = numVars;
                    for (unsigned int v = 0; v < dof; v++) {
                        wrefEl->compute_wavelets_3D(
                            (double*)(blkIn.data() + v * sz_per_dof), isz,
                            wCout, isBdyOct);
                        const double l_max =
                            (normL2(wCout.data(), wCout.size())) /
                            sqrt(wCout.size());

                        if (wMax < l_max) wMax = l_max;

                        // for early bail out.
                        if (wMax > tol_ele) break;
                    }

                    eleWMax[ele - eleOfst] = wMax;
                }

                // if(isBdyOct)
                // std::cout<<"ele :  "<<ele<<" eleWMax:
                // "<<eleWMax[ele-eleOfst]<<std::endl;
            }
        }

        // delete the wavelet reference element.
        delete wrefEl;

        // amr-decision probe: dump per-element (TN, wMax, tol_ele,
        // decision) so we can diff AMR decisions across modes. Gated by
        // EM4_AMR_PROBE_DIR. Writes one file per remesh call per rank.
        static const char* amrp_dir = DENDRO_PROBE_GETENV("EM4_AMR_PROBE_DIR");
        static int amrp_call = 0;
        FILE* amrp_fp = nullptr;
        if (amrp_dir) {
            char fn[1024];
            std::snprintf(fn, sizeof(fn),
                          "%s/amr_call%d_r%d.txt",
                          amrp_dir, amrp_call, (int)m_uiActiveRank);
            amrp_fp = std::fopen(fn, "w");
            if (amrp_fp) {
                std::fprintf(amrp_fp,
                    "# call=%d rank=%d numLocal=%u amr_fac=%g\n"
                    "# ele lev tn_x tn_y tn_z phys_x phys_y phys_z "
                    "wMax_hex tol_ele_hex decision wMax tol_ele\n",
                    amrp_call, (int)m_uiActiveRank, m_uiNumLocalElements,
                    amr_coarse_fac);
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
            int decision = 0;  // 0 = no change, 1 = split, -1 = coarse

            if (l_max > tol_ele) {
                refine_flags[(ele - eleOfst)] = OCT_SPLIT;
                isMeshLocalChanged            = true;
                decision = 1;

            } else if (l_max < amr_coarse_fac * tol_ele) {
                refine_flags[ele - eleOfst] = OCT_COARSE;
                isMeshLocalChanged          = true;
                decision = -1;

            } else {
                refine_flags[ele - eleOfst] = OCT_NO_CHANGE;
            }

            if (amrp_fp) {
                uint64_t wmax_hb = 0, tol_hb = 0;
                std::memcpy(&wmax_hb, &l_max, sizeof(wmax_hb));
                std::memcpy(&tol_hb, &tol_ele, sizeof(tol_hb));
                std::fprintf(amrp_fp,
                    "%u %u %u %u %u %g %g %g %lx %lx %d %.17e %.17e\n",
                    ele, pNodes[ele].getLevel(),
                    (unsigned)pNodes[ele].getX(),
                    (unsigned)pNodes[ele].getY(),
                    (unsigned)pNodes[ele].getZ(),
                    domain_pt1.x(), domain_pt1.y(), domain_pt1.z(),
                    (unsigned long)wmax_hb, (unsigned long)tol_hb,
                    decision, l_max, tol_ele);
            }
        }

        if (amrp_fp) std::fclose(amrp_fp);
        if (amrp_dir) amrp_call++;

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

template <typename T>
void Mesh::getElementNodalValues(const T* vec, T* nodalValues,
                                 unsigned int elementID, bool isDGVec) const {
    if (!m_uiIsActive) return;

    // handles the element get nodal values if the vec is an element DG vector.
    if (isDGVec) {
        for (unsigned int node = 0; node < m_uiNpE; node++)
            nodalValues[node] = vec[elementID * m_uiNpE + node];

        return;
    }

    // hypothesis-2 probe: dump (cnum, faceIndex, e2n_cg, vec) per hanging
    // face for a target element TN, gated by DENDRO_GENV_PROBE_TN. probe
    // fires every call to this function for matching ele; combine with
    // DENDRO_UNZIP_PROBE_MIN_CALL/MAX_CALLS to limit total invocations.
    static const char* genvprobe_dir =
        DENDRO_PROBE_GETENV("DENDRO_GENV_PROBE_DIR");
    static const char* genvprobe_tn =
        DENDRO_PROBE_GETENV("DENDRO_GENV_PROBE_TN");
    static unsigned int gp_lev = 0, gp_x = 0, gp_y = 0, gp_z = 0;
    static bool gp_parsed = false;
    if (genvprobe_tn && !gp_parsed) {
        std::sscanf(genvprobe_tn, "%u,%u,%u,%u",
                    &gp_lev, &gp_x, &gp_y, &gp_z);
        gp_parsed = true;
    }
    static int gp_v_counter = 0;
    static const char* gp_max_env = DENDRO_PROBE_GETENV("DENDRO_GENV_PROBE_MAX");
    static const int gp_v_max = gp_max_env ? std::atoi(gp_max_env) : 16;
    static const char* gp_call_env = DENDRO_PROBE_GETENV("DENDRO_GENV_PROBE_CALL");
    static const int gp_call_target =
        gp_call_env ? std::atoi(gp_call_env) : -1;
    bool gp_match = (genvprobe_dir && gp_parsed && gp_v_counter < gp_v_max
                     && (gp_call_target < 0
                         || g_dendro_unzip_call_id == gp_call_target)
                     && m_uiAllElements[elementID].getLevel() == gp_lev
                     && m_uiAllElements[elementID].minX() == gp_x
                     && m_uiAllElements[elementID].minY() == gp_y
                     && m_uiAllElements[elementID].minZ() == gp_z);
    FILE* gp_fp = nullptr;
    if (gp_match) {
        char fn[1024];
        std::snprintf(fn, sizeof(fn),
                      "%s/genv_inv%d_r%d.txt",
                      genvprobe_dir, gp_v_counter,
                      (int)m_uiActiveRank);
        gp_fp = std::fopen(fn, "w");
        if (gp_fp) {
            std::fprintf(gp_fp,
                "# inv=%d rank=%d ele=%u TN(lev=%u,%u,%u,%u) eOrder=%u "
                "localBegin=%u localEnd=%u\n",
                gp_v_counter, (int)m_uiActiveRank, elementID,
                gp_lev, gp_x, gp_y, gp_z, m_uiElementOrder,
                m_uiNodeLocalBegin, m_uiNodeLocalEnd);
        }
        gp_v_counter++;
    }
    auto dump_face = [&](FILE* fp, const char* fname, bool isH,
                         unsigned int cn,
                         const std::vector<unsigned int>& fIdx,
                         const std::vector<T>& fIn) {
        if (!fp) return;
        std::fprintf(fp, "FACE %s isHanging=%d cnum=%u nIdx=%zu\n",
                     fname, (int)isH, cn, fIdx.size());
        if (isH) {
            for (size_t i = 0; i < fIdx.size(); i++) {
                unsigned int fi = fIdx[i];
                unsigned int cg = m_uiE2NMapping_CG[fi];
                uint64_t hb = 0;
                T v = fIn[i];
                std::memcpy(&hb, &v, sizeof(hb));
                std::fprintf(fp,
                    "  i=%zu faceIdx=%u (ele=%u sub=%u) cg=%u vhex=%lx\n",
                    i, fi, fi / m_uiNpE, fi % m_uiNpE,
                    cg, (unsigned long)hb);
            }
        }
    };

#ifdef ENABLE_DENDRO_PROFILE_COUNTERS
    dendro::timer::t_unzip_sync_nodalval.start();
#endif
    std::vector<T> edgeInpIn;
    std::vector<T> edgeInpOut;

    std::vector<T> faceInpIn;
    std::vector<T> faceInpOut;

    unsigned int cnum;
    bool isHanging;

    std::vector<unsigned int> edgeIndex;
    std::vector<unsigned int> faceIndex;

    bool nodeStatus[OCT_DIR_TOTAL];
    for (unsigned int w = 0; w < OCT_DIR_TOTAL; w++) nodeStatus[w] = false;

    edgeInpIn.resize((m_uiElementOrder + 1));
    edgeInpOut.resize((m_uiElementOrder + 1));

    faceInpIn.resize((m_uiElementOrder + 1) * (m_uiElementOrder + 1));
    faceInpOut.resize((m_uiElementOrder + 1) * (m_uiElementOrder + 1));

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
        dump_face(gp_fp, "LEFT", isHanging, cnum, faceIndex, faceInpIn);

        this->parent2ChildInterpolation(&(*(faceInpIn.begin())),
                                        &(*(faceInpOut.begin())), cnum, 2);

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
        dump_face(gp_fp, "RIGHT", isHanging, cnum, faceIndex, faceInpIn);

        this->parent2ChildInterpolation(&(*(faceInpIn.begin())),
                                        &(*(faceInpOut.begin())), cnum, 2);

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
        dump_face(gp_fp, "DOWN", isHanging, cnum, faceIndex, faceInpIn);

        this->parent2ChildInterpolation(&(*(faceInpIn.begin())),
                                        &(*(faceInpOut.begin())), cnum, 2);

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
        dump_face(gp_fp, "UP", isHanging, cnum, faceIndex, faceInpIn);

        this->parent2ChildInterpolation(&(*(faceInpIn.begin())),
                                        &(*(faceInpOut.begin())), cnum, 2);

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
        dump_face(gp_fp, "BACK", isHanging, cnum, faceIndex, faceInpIn);

        this->parent2ChildInterpolation(&(*(faceInpIn.begin())),
                                        &(*(faceInpOut.begin())), cnum, 2);

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
        dump_face(gp_fp, "FRONT", isHanging, cnum, faceIndex, faceInpIn);

        this->parent2ChildInterpolation(&(*(faceInpIn.begin())),
                                        &(*(faceInpOut.begin())), cnum, 2);

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

            this->parent2ChildInterpolation(&(*(edgeInpIn.begin())),
                                            &(*(edgeInpOut.begin())), cnum, 1);

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

            this->parent2ChildInterpolation(&(*(edgeInpIn.begin())),
                                            &(*(edgeInpOut.begin())), cnum, 1);

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

            this->parent2ChildInterpolation(&(*(edgeInpIn.begin())),
                                            &(*(edgeInpOut.begin())), cnum, 1);

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

            this->parent2ChildInterpolation(&(*(edgeInpIn.begin())),
                                            &(*(edgeInpOut.begin())), cnum, 1);

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

            this->parent2ChildInterpolation(&(*(edgeInpIn.begin())),
                                            &(*(edgeInpOut.begin())), cnum, 1);

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

            this->parent2ChildInterpolation(&(*(edgeInpIn.begin())),
                                            &(*(edgeInpOut.begin())), cnum, 1);

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

            this->parent2ChildInterpolation(&(*(edgeInpIn.begin())),
                                            &(*(edgeInpOut.begin())), cnum, 1);

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

            this->parent2ChildInterpolation(&(*(edgeInpIn.begin())),
                                            &(*(edgeInpOut.begin())), cnum, 1);

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

            this->parent2ChildInterpolation(&(*(edgeInpIn.begin())),
                                            &(*(edgeInpOut.begin())), cnum, 1);

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

            this->parent2ChildInterpolation(&(*(edgeInpIn.begin())),
                                            &(*(edgeInpOut.begin())), cnum, 1);

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

            this->parent2ChildInterpolation(&(*(edgeInpIn.begin())),
                                            &(*(edgeInpOut.begin())), cnum, 1);

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

            this->parent2ChildInterpolation(&(*(edgeInpIn.begin())),
                                            &(*(edgeInpOut.begin())), cnum, 1);

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
    if (gp_fp) std::fclose(gp_fp);
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
    // probe (off by default): dump post-transfer cg values keyed by
    // (phys_pos, dof_var) for the NEW mesh, one file per rank. used to
    // diff graph vs SFC after a remesh. set DENDRO_IGT_DUMP_DIR=/path.
    // dump index increments per call to this function.
    static const char* igt_dump_dir =
        DENDRO_PROBE_GETENV("DENDRO_IGT_DUMP_DIR");
    static int igt_call_id = 0;
    static const char* igt_max_env =
        DENDRO_PROBE_GETENV("DENDRO_IGT_DUMP_MAX_CALLS");
    static const int igt_dump_max =
        igt_max_env ? std::atoi(igt_max_env) : 2;
    const bool igt_dump_on =
        (igt_dump_dir != nullptr) && (igt_call_id < igt_dump_max);

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

    // probe: dump INPUT cg values (vec, the OLD mesh's cg array) keyed
    // by physical position. one file per (rank, call_id, var). dumped
    // alongside the output dump below.
    if (igt_dump_on && m_uiIsActive) {
        const unsigned int* e2n_o = m_uiE2NMapping_CG.data();
        const unsigned int* e2n_dg_o = m_uiE2NMapping_DG.data();
        const auto& pNodes_o = m_uiAllElements;
        const unsigned int oNLB = m_uiNodeLocalBegin;
        const unsigned int oNLE = m_uiNodeLocalEnd;
        const unsigned int oELB = m_uiElementLocalBegin;
        const unsigned int oELE = m_uiElementLocalEnd;
        const unsigned int eOrd_o = m_uiElementOrder;
        const unsigned int nPe_o = m_uiNpE;
        const unsigned int maxD_o = m_uiMaxDepth;
        for (unsigned int var = 0; var < dof; var++) {
            T* invv = vecIn + (var * m_uiNumActualNodes);
            char fn[1024];
            std::snprintf(fn, sizeof(fn),
                          "%s/igt_in_call%d_v%d_r%d.txt",
                          igt_dump_dir, igt_call_id, var,
                          (int)m_uiActiveRank);
            FILE* fp = std::fopen(fn, "w");
            if (!fp) continue;
            std::fprintf(fp,
                "# call=%d var=%u rank=%d INPUT localBegin=%u localEnd=%u\n"
                "# cg owner_ele owner_lev ownerX ownerY ownerZ ii jj kk phys_x phys_y phys_z hex\n",
                igt_call_id, var, (int)m_uiActiveRank,
                oNLB, oNLE);
            for (unsigned int ele = oELB; ele < oELE; ele++) {
                for (unsigned int k = 0; k <= eOrd_o; k++)
                for (unsigned int j = 0; j <= eOrd_o; j++)
                for (unsigned int i = 0; i <= eOrd_o; i++) {
                    if (this->isNodeHanging(ele, i, j, k)) continue;
                    const unsigned int sub = k*(eOrd_o+1)*(eOrd_o+1)
                        + j*(eOrd_o+1) + i;
                    const unsigned int slot = ele * nPe_o + sub;
                    const unsigned int cg = e2n_o[slot];
                    if (cg < oNLB || cg >= oNLE) continue;
                    const unsigned int dg = e2n_dg_o[slot];
                    const unsigned int oe = dg / nPe_o;
                    const unsigned int on = dg % nPe_o;
                    const unsigned int oni = on % (eOrd_o+1);
                    const unsigned int onj = (on / (eOrd_o+1)) % (eOrd_o+1);
                    const unsigned int onk = on / ((eOrd_o+1)*(eOrd_o+1));
                    const ot::TreeNode& oTN = pNodes_o[oe];
                    const unsigned int olen =
                        (unsigned int)1u << (maxD_o - oTN.getLevel());
                    const unsigned long long phys_x =
                        (unsigned long long)oTN.getX() * eOrd_o
                        + (unsigned long long)oni * olen;
                    const unsigned long long phys_y =
                        (unsigned long long)oTN.getY() * eOrd_o
                        + (unsigned long long)onj * olen;
                    const unsigned long long phys_z =
                        (unsigned long long)oTN.getZ() * eOrd_o
                        + (unsigned long long)onk * olen;
                    uint64_t hb = 0;
                    T v = invv[cg];
                    std::memcpy(&hb, &v, sizeof(hb));
                    std::fprintf(fp,
                        "%u %u %u %u %u %u %u %u %u %llu %llu %llu %lx\n",
                        cg, oe, oTN.getLevel(),
                        oTN.getX(), oTN.getY(), oTN.getZ(),
                        oni, onj, onk,
                        phys_x, phys_y, phys_z,
                        (unsigned long)hb);
                }
            }
            std::fclose(fp);
        }
    }

    // probe: dump LOCAL cg values keyed by physical position. one file
    // per (rank, call_id, var). only NON-HANGING cgs are dumped (those
    // are the ones that got values from intergrid transfer).
    if (igt_dump_on && pMesh->isActive()) {
        const unsigned int* e2n_p = &(*(pMesh->getE2NMapping().begin()));
        const unsigned int* e2n_dg_p =
            &(*(pMesh->getE2NMapping_DG().begin()));
        const auto& pNodes_p = pMesh->getAllElements();
        const unsigned int m2NLB = pMesh->getNodeLocalBegin();
        const unsigned int m2NLE = pMesh->getNodeLocalEnd();
        const unsigned int m2ELB = pMesh->getElementLocalBegin();
        const unsigned int m2ELE = pMesh->getElementLocalEnd();
        const unsigned int eOrd_p = pMesh->getElementOrder();
        const unsigned int nPe_p = pMesh->getNumNodesPerElement();
        const unsigned int maxD_p = m_uiMaxDepth;
        for (unsigned int var = 0; var < dof; var++) {
            T* outv = vecOut + (var * pMesh->getDegOfFreedom());
            char fn[1024];
            std::snprintf(fn, sizeof(fn),
                          "%s/igt_call%d_v%d_r%d.txt",
                          igt_dump_dir, igt_call_id, var,
                          (int)pMesh->getMPIRank());
            FILE* fp = std::fopen(fn, "w");
            if (!fp) continue;
            std::fprintf(fp,
                "# call=%d var=%u rank=%d localBegin=%u localEnd=%u\n"
                "# cg owner_ele owner_lev ownerX ownerY ownerZ ii jj kk phys_x phys_y phys_z hex\n",
                igt_call_id, var, (int)pMesh->getMPIRank(),
                m2NLB, m2NLE);
            // walk all (ele, sub) on new mesh; for each non-hanging
            // sub whose cg is local, dump it.
            for (unsigned int ele = m2ELB; ele < m2ELE; ele++) {
                for (unsigned int k = 0; k <= eOrd_p; k++)
                for (unsigned int j = 0; j <= eOrd_p; j++)
                for (unsigned int i = 0; i <= eOrd_p; i++) {
                    if (pMesh->isNodeHanging(ele, i, j, k)) continue;
                    const unsigned int sub = k*(eOrd_p+1)*(eOrd_p+1)
                        + j*(eOrd_p+1) + i;
                    const unsigned int slot = ele * nPe_p + sub;
                    const unsigned int cg = e2n_p[slot];
                    if (cg < m2NLB || cg >= m2NLE) continue;
                    const unsigned int dg = e2n_dg_p[slot];
                    const unsigned int oe = dg / nPe_p;
                    const unsigned int on = dg % nPe_p;
                    const unsigned int oni = on % (eOrd_p+1);
                    const unsigned int onj = (on / (eOrd_p+1)) % (eOrd_p+1);
                    const unsigned int onk = on / ((eOrd_p+1)*(eOrd_p+1));
                    const ot::TreeNode& oTN = pNodes_p[oe];
                    const unsigned int olen =
                        (unsigned int)1u << (maxD_p - oTN.getLevel());
                    const unsigned long long phys_x =
                        (unsigned long long)oTN.getX() * eOrd_p
                        + (unsigned long long)oni * olen;
                    const unsigned long long phys_y =
                        (unsigned long long)oTN.getY() * eOrd_p
                        + (unsigned long long)onj * olen;
                    const unsigned long long phys_z =
                        (unsigned long long)oTN.getZ() * eOrd_p
                        + (unsigned long long)onk * olen;
                    uint64_t hb = 0;
                    T v = outv[cg];
                    std::memcpy(&hb, &v, sizeof(hb));
                    std::fprintf(fp,
                        "%u %u %u %u %u %u %u %u %u %llu %llu %llu %lx\n",
                        cg, oe, oTN.getLevel(),
                        oTN.getX(), oTN.getY(), oTN.getZ(),
                        oni, onj, onk,
                        phys_x, phys_y, phys_z,
                        (unsigned long)hb);
                }
            }
            std::fclose(fp);
        }
    }
    if (igt_dump_dir) igt_call_id++;

    // post-IGT sync: the destination mesh may have cross-rank or
    // intra-rank duplicate cgs (same phys, multiple ranks/cgs claim
    // it). IGT writes via the scatter-map path, which doesn't sync
    // duplicates. without this call, the FIRST stepping remesh's IGT
    // leaves duplicate cgs at partition-dependent interpolated values,
    // seeding the U_E2 noise that grows step-by-step. env
    // DENDRO_DISABLE_POST_IGT_SYNC=1 disables.
    {
        static const char* dis_env =
            std::getenv("DENDRO_DISABLE_POST_IGT_SYNC");
        const bool skip =
            dis_env && dis_env[0] == '1' && dis_env[1] == '\0';
        if (!skip && pMesh != nullptr && pMesh->isActive()) {
            ot::Mesh* pMeshMut = const_cast<ot::Mesh*>(pMesh);
            pMeshMut->syncZipNonPrimary(vecOut, dof);
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
void Mesh::orphanPreGather(T* vec) {
    // Fix orphan local CG slots: slots in the local CG range that are
    // NOT referenced by any local element on this rank. Such slots can
    // arise in graph partitions where buildE2NMap allocates local-
    // range CGs at positions only referenced by ghost elements; their
    // values are frozen since zip never writes to them via a local
    // element. This pulls up-to-date values from ranks that DO have a
    // local element at the same physical position.
    if (!m_uiIsActive) return;
    MPI_Comm comm = m_uiCommGlobal;
    int rank, npes;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);

    const unsigned int npe  = m_uiNpE;
    const unsigned int eOrd = m_uiElementOrder;
    const unsigned int nLB  = m_uiNodeLocalBegin;
    const unsigned int nLE  = m_uiNodeLocalEnd;
    const auto* pN          = m_uiAllElements.data();

    // Identify orphan local cgs on me.
    //
    // Phys-pos keys use SCALED-integer coords: `getX()*eOrder + sub*len`.
    // This is always integer (getX, eOrder, sub, len are integers) and
    // agrees across levels: two subs at the same physical position from
    // different-level elements produce the same scaled-int. The naive
    // `getX() + sub*(len/eOrder)` truncates for non-divisible len/eOrder
    // (e.g. eOrder=6 at level-7+ where len/eOrder integer-divides to 0),
    // which would collapse distinct subs onto the same key.
    std::vector<unsigned char> refByLocal(nLE - nLB, 0);
    for (unsigned int e = m_uiElementLocalBegin;
         e < m_uiElementLocalEnd; e++) {
        for (unsigned int n = 0; n < npe; n++) {
            unsigned int cg = m_uiE2NMapping_CG[e * npe + n];
            if (cg >= nLB && cg < nLE) refByLocal[cg - nLB] = 1;
        }
    }
    struct OrphanEntry {
        int          ownerRank;
        unsigned int cg;
        uint64_t     x, y, z;  // scaled-int (octree * eOrder)
    };
    std::vector<OrphanEntry> myOrphans;
    for (unsigned int cg = nLB; cg < nLE; cg++) {
        if (!refByLocal[cg - nLB]) {
            unsigned int dg = m_uiCG2DG[cg];
            unsigned int ce = dg / npe;
            unsigned int cn = dg % npe;
            uint64_t len =
                (uint64_t)1 << (m_uiMaxDepth - pN[ce].getLevel());
            uint64_t x =
                (uint64_t)pN[ce].getX() * eOrd +
                (uint64_t)(cn % (eOrd + 1)) * len;
            uint64_t y =
                (uint64_t)pN[ce].getY() * eOrd +
                (uint64_t)((cn / (eOrd + 1)) % (eOrd + 1)) * len;
            uint64_t z =
                (uint64_t)pN[ce].getZ() * eOrd +
                (uint64_t)(cn / ((eOrd + 1) * (eOrd + 1))) * len;
            myOrphans.push_back({rank, cg, x, y, z});
        }
    }

    int myCount = (int)myOrphans.size();
    std::vector<int> counts(npes), offs(npes, 0);
    MPI_Allgather(&myCount, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
    int total = 0;
    for (int p = 0; p < npes; p++) {
        offs[p] = total;
        total += counts[p];
    }
    if (total == 0) return;  // no orphans anywhere

    std::vector<int>          allOwner(total);
    std::vector<unsigned int> allCG(total);
    std::vector<uint64_t>     allX(total), allY(total), allZ(total);
    {
        std::vector<int>          myO(myCount);
        std::vector<unsigned int> myCG(myCount);
        std::vector<uint64_t>     myX(myCount), myY(myCount), myZ(myCount);
        for (int k = 0; k < myCount; k++) {
            myO[k]  = myOrphans[k].ownerRank;
            myCG[k] = myOrphans[k].cg;
            myX[k]  = myOrphans[k].x;
            myY[k]  = myOrphans[k].y;
            myZ[k]  = myOrphans[k].z;
        }
        MPI_Allgatherv(myO.data(), myCount, MPI_INT, allOwner.data(),
                       counts.data(), offs.data(), MPI_INT, comm);
        MPI_Allgatherv(myCG.data(), myCount, MPI_UNSIGNED, allCG.data(),
                       counts.data(), offs.data(), MPI_UNSIGNED, comm);
        MPI_Allgatherv(myX.data(), myCount, MPI_UINT64_T, allX.data(),
                       counts.data(), offs.data(), MPI_UINT64_T, comm);
        MPI_Allgatherv(myY.data(), myCount, MPI_UINT64_T, allY.data(),
                       counts.data(), offs.data(), MPI_UINT64_T, comm);
        MPI_Allgatherv(myZ.data(), myCount, MPI_UINT64_T, allZ.data(),
                       counts.data(), offs.data(), MPI_UINT64_T, comm);
    }

    // Build phys_pos → (local element, sub) for my local elements.
    // Same scaled-int encoding as above.
    std::map<std::tuple<uint64_t, uint64_t, uint64_t>,
             std::pair<unsigned int, unsigned int>>
        posToES;
    for (unsigned int e = m_uiElementLocalBegin;
         e < m_uiElementLocalEnd; e++) {
        for (unsigned int n = 0; n < npe; n++) {
            uint64_t len =
                (uint64_t)1 << (m_uiMaxDepth - pN[e].getLevel());
            uint64_t x =
                (uint64_t)pN[e].getX() * eOrd +
                (uint64_t)(n % (eOrd + 1)) * len;
            uint64_t y =
                (uint64_t)pN[e].getY() * eOrd +
                (uint64_t)((n / (eOrd + 1)) % (eOrd + 1)) * len;
            uint64_t z =
                (uint64_t)pN[e].getZ() * eOrd +
                (uint64_t)(n / ((eOrd + 1) * (eOrd + 1))) * len;
            posToES.emplace(std::make_tuple(x, y, z),
                            std::make_pair(e, n));
        }
    }

    std::vector<std::vector<T>>            sendVal(npes);
    std::vector<std::vector<unsigned int>> sendDstCG(npes);
    for (int t = 0; t < total; t++) {
        int target = allOwner[t];
        if (target == rank) continue;
        auto it = posToES.find(std::make_tuple(allX[t], allY[t], allZ[t]));
        if (it == posToES.end()) continue;
        unsigned int e       = it->second.first;
        unsigned int n       = it->second.second;
        unsigned int cg_here = m_uiE2NMapping_CG[e * npe + n];
        sendVal[target].push_back(vec[cg_here]);
        sendDstCG[target].push_back(allCG[t]);
    }

    std::vector<int> sendCnt(npes, 0), recvCnt(npes, 0);
    for (int p = 0; p < npes; p++) sendCnt[p] = (int)sendVal[p].size();
    MPI_Alltoall(sendCnt.data(), 1, MPI_INT, recvCnt.data(), 1, MPI_INT,
                 comm);
    std::vector<int> sendOff(npes, 0), recvOff(npes, 0);
    int totS = 0, totR = 0;
    for (int p = 0; p < npes; p++) {
        sendOff[p] = totS;
        totS += sendCnt[p];
        recvOff[p] = totR;
        totR += recvCnt[p];
    }
    std::vector<T>            flatVal;
    std::vector<unsigned int> flatCG;
    flatVal.reserve(totS);
    flatCG.reserve(totS);
    for (int p = 0; p < npes; p++) {
        flatVal.insert(flatVal.end(), sendVal[p].begin(),
                       sendVal[p].end());
        flatCG.insert(flatCG.end(), sendDstCG[p].begin(),
                      sendDstCG[p].end());
    }
    std::vector<T>            recvValBuf(totR);
    std::vector<unsigned int> recvCGBuf(totR);
    MPI_Alltoallv(flatVal.data(), sendCnt.data(), sendOff.data(),
                  par::Mpi_datatype<T>::value(), recvValBuf.data(),
                  recvCnt.data(), recvOff.data(),
                  par::Mpi_datatype<T>::value(), comm);
    MPI_Alltoallv(flatCG.data(), sendCnt.data(), sendOff.data(),
                  MPI_UNSIGNED, recvCGBuf.data(), recvCnt.data(),
                  recvOff.data(), MPI_UNSIGNED, comm);

    for (int i = 0; i < totR; i++) {
        vec[recvCGBuf[i]] = recvValBuf[i];
    }
}

template <typename T>
void Mesh::syncLocalNodalDGFromCG(const T* vec) {
    if (!m_uiIsActive) return;
    const unsigned int numLocal =
        m_uiElementLocalEnd - m_uiElementLocalBegin;
    m_uiLocalNodalDG.assign((size_t)numLocal * m_uiNpE, 0.0);
    // Raw CG copy: for CG (continuous Galerkin) the value at a given
    // physical position is shared by all elements touching it.
    // vec[E2N_CG[e][n]] gives that shared value regardless of whether
    // n is a hanging sub on element e (in which case E2N_CG routes to
    // the parent's canonical CG slot).
    //
    // We deliberately do NOT use getElementNodalValues here, which
    // applies parent2ChildInterpolation for hanging sub-indices — that
    // would conflict with createVector(vec, func)'s f(physical) writes
    // and with the value actually stored at the canonical CG slot.
    for (unsigned int e = m_uiElementLocalBegin;
         e < m_uiElementLocalEnd; e++) {
        const unsigned int eLocal = e - m_uiElementLocalBegin;
        for (unsigned int n = 0; n < m_uiNpE; n++) {
            const unsigned int cg =
                m_uiE2NMapping_CG[e * m_uiNpE + n];
            m_uiLocalNodalDG[eLocal * m_uiNpE + n] = (double)vec[cg];
        }
    }
}

template <typename T>
void Mesh::redistributeVec(ot::Mesh* dstMesh, const T* vecIn, T* vecOut) const {
    // Redistribute a CG vector across a partition change. Assumes `this`
    // (source) and `dstMesh` (target) contain the same global element
    // set and dof=1 per element. See mesh.h for full contract.

    MPI_Comm comm = m_uiCommGlobal;
    int rank, npes;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);

    const unsigned int npe = dstMesh->getNumNodesPerElement();

    // ---- Step 1: Allgatherv each rank's local TreeNodes on dstMesh, build
    //              a global TreeNode -> rank map (hashed for speed).
    std::vector<ot::TreeNode> myLocal;
    if (dstMesh->isActive()) {
        const auto* pN = dstMesh->getAllElements().data();
        for (unsigned int e = dstMesh->getElementLocalBegin();
             e < dstMesh->getElementLocalEnd(); e++)
            myLocal.push_back(pN[e]);
    }
    int myCount = (int)myLocal.size();

    std::vector<int> counts(npes), offs(npes, 0);
    MPI_Allgather(&myCount, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
    int total = 0;
    for (int p = 0; p < npes; p++) {
        offs[p] = total;
        total += counts[p];
    }

    std::vector<ot::TreeNode> allLocal(total);
    MPI_Allgatherv(myLocal.data(), myCount,
                   par::Mpi_datatype<ot::TreeNode>::value(), allLocal.data(),
                   counts.data(), offs.data(),
                   par::Mpi_datatype<ot::TreeNode>::value(), comm);

    struct TNHash {
        size_t operator()(const ot::TreeNode& t) const noexcept {
            uint64_t h = (uint64_t)t.getX();
            h = (h << 21) ^ (uint64_t)t.getY();
            h = (h << 21) ^ (uint64_t)t.getZ();
            h = (h << 6) ^ (uint64_t)t.getLevel();
            h ^= h >> 33;
            h *= 0xff51afd7ed558ccdULL;
            h ^= h >> 33;
            h *= 0xc4ceb9fe1a85ec53ULL;
            h ^= h >> 33;
            return (size_t)h;
        }
    };
    std::unordered_map<ot::TreeNode, int, TNHash> tnToRank;
    tnToRank.reserve((size_t)total);
    for (int p = 0; p < npes; p++)
        for (int i = offs[p]; i < offs[p] + counts[p]; i++)
            tnToRank.emplace(allLocal[i], p);

    // ---- Step 2: walk this mesh's local elements, pack per-element DG
    //              values via E2N_CG (canonical owner's CG value) into
    //              per-rank send buffers keyed by TreeNode. Also pack
    //              a per-sub "local-flag" byte: 1 = E2N_CG resolves to
    //              a LOCAL cg on src (authoritative value), 0 = ghost
    //              on src (possibly stale, do not trust on dst).
    //
    //              This matters because dst may have the same physical
    //              position as LOCAL on multiple ranks (cg duplicates
    //              are common in graph partitions). Each dst local cg
    //              gets touched by writes from every src element that
    //              references it. Only ONE of those writes comes from
    //              the canonical-owner src rank with E2N_CG local. The
    //              others read from ghost cgs on src — which may be
    //              stale (e.g., case-B holes in the scatter map on
    //              graph-partitioned src). Without the flag the write
    //              order determines the final value (last-write-wins
    //              across different (d_e, n) writes to the same Q).
    //              With the flag, stale writes are skipped on dst so
    //              only the canonical value lands.
    // E2N_CG mis-routing probe: log every src LOCAL element where a node's
    // geometric phys position matches EM4_REDIST_PROBE_PHYS BUT the
    // E2N_CG-resolved cg points to a different phys, OR where the value
    // sent differs from neighbors. helps find the rogue cg that contaminates
    // the canonical bcast pick.
    static const char* rvp_dir = DENDRO_PROBE_GETENV("EM4_REDIST_PROBE_DIR");
    static const char* rvp_phys = DENDRO_PROBE_GETENV("EM4_REDIST_PROBE_PHYS");
    static const char* rvp_min_env = DENDRO_PROBE_GETENV("EM4_REDIST_PROBE_MIN_CALL");
    static const char* rvp_max_env = DENDRO_PROBE_GETENV("EM4_REDIST_PROBE_MAX_CALL");
    static const int rvp_min = rvp_min_env ? std::atoi(rvp_min_env) : 0;
    static const int rvp_max = rvp_max_env ? std::atoi(rvp_max_env) : 999999;
    static int rvp_call = 0;
    static unsigned long long rvp_x = 0, rvp_y = 0, rvp_z = 0;
    static bool rvp_phys_on = false;
    if (rvp_phys && !rvp_phys_on) {
        std::sscanf(rvp_phys, "%llu,%llu,%llu", &rvp_x, &rvp_y, &rvp_z);
        rvp_phys_on = true;
    }
    const bool rvp_in_range = (rvp_call >= rvp_min && rvp_call < rvp_max);
    FILE* rvp_fp = nullptr;
    if (rvp_dir && rvp_phys_on && rvp_in_range && m_uiIsActive) {
        char fn[1024];
        std::snprintf(fn, sizeof(fn), "%s/redist_send_call%d_r%d.txt",
                      rvp_dir, rvp_call, rank);
        rvp_fp = std::fopen(fn, "w");
        if (rvp_fp) {
            std::fprintf(rvp_fp,
                "# redistributeVec send-side probe, call=%d rank=%d\n"
                "# target_phys=(%llu,%llu,%llu)\n"
                "# format: srcEle srcTN(L,X,Y,Z) sub_n(i,j,k) "
                "node_geom_phys e2nCG_cg cg_resolved_phys "
                "vecIn_value match? target_rank\n",
                rvp_call, rank, rvp_x, rvp_y, rvp_z);
        }
    }

    std::vector<std::vector<ot::TreeNode>> sendTN(npes);
    std::vector<std::vector<T>> sendDG(npes);
    std::vector<std::vector<unsigned char>> sendLocalFlag(npes);
    if (m_uiIsActive) {
        const auto* pNO = m_uiAllElements.data();
        const auto& e2nO = m_uiE2NMapping_CG;
        const auto& cg2dgO = m_uiCG2DG;
        const unsigned int eOrdO = m_uiElementOrder;
        for (unsigned int e = m_uiElementLocalBegin;
             e < m_uiElementLocalEnd; e++) {
            auto it = tnToRank.find(pNO[e]);
            if (it == tnToRank.end()) continue;
            int target = it->second;
            sendTN[target].push_back(pNO[e]);
            for (unsigned int n = 0; n < npe; n++) {
                unsigned int cg = e2nO[e * npe + n];
                bool isLocal = (cg >= m_uiNodeLocalBegin &&
                                cg < m_uiNodeLocalEnd);
                sendDG[target].push_back((T)vecIn[cg]);
                sendLocalFlag[target].push_back(isLocal ? 1 : 0);

                if (rvp_fp) {
                    // compute geometric phys of this (e, n) on src.
                    const unsigned int ni = n % (eOrdO + 1);
                    const unsigned int nj = (n/(eOrdO+1)) % (eOrdO+1);
                    const unsigned int nk = n / ((eOrdO+1)*(eOrdO+1));
                    const unsigned long long lenE =
                        (unsigned long long)1u
                        << (m_uiMaxDepth - pNO[e].getLevel());
                    const unsigned long long gx =
                        (unsigned long long)pNO[e].getX()*eOrdO
                        + (unsigned long long)ni*lenE;
                    const unsigned long long gy =
                        (unsigned long long)pNO[e].getY()*eOrdO
                        + (unsigned long long)nj*lenE;
                    const unsigned long long gz =
                        (unsigned long long)pNO[e].getZ()*eOrdO
                        + (unsigned long long)nk*lenE;
                    // compute resolved cg's phys via cg2dg.
                    unsigned long long cx = 0, cy = 0, cz = 0;
                    bool ok = false;
                    if (cg < cg2dgO.size()) {
                        unsigned int dg = cg2dgO[cg];
                        if (dg != LOOK_UP_TABLE_DEFAULT) {
                            unsigned int oe = dg / npe;
                            unsigned int on = dg % npe;
                            if (oe < m_uiAllElements.size()) {
                                const auto& oTN = m_uiAllElements[oe];
                                const unsigned int oni = on%(eOrdO+1);
                                const unsigned int onj =
                                    (on/(eOrdO+1))%(eOrdO+1);
                                const unsigned int onk =
                                    on/((eOrdO+1)*(eOrdO+1));
                                const unsigned long long olen =
                                    (unsigned long long)1u
                                    << (m_uiMaxDepth - oTN.getLevel());
                                cx = (unsigned long long)oTN.getX()
                                    *eOrdO + (unsigned long long)oni
                                    *olen;
                                cy = (unsigned long long)oTN.getY()
                                    *eOrdO + (unsigned long long)onj
                                    *olen;
                                cz = (unsigned long long)oTN.getZ()
                                    *eOrdO + (unsigned long long)onk
                                    *olen;
                                ok = true;
                            }
                        }
                    }
                    const bool geom_match =
                        (gx == rvp_x && gy == rvp_y && gz == rvp_z);
                    const bool cg_match =
                        ok && (cx == rvp_x) && (cy == rvp_y) && (cz == rvp_z);
                    if (geom_match || cg_match) {
                        uint64_t hb = 0;
                        T vsend = (T)vecIn[cg];
                        std::memcpy(&hb, &vsend, sizeof(hb));
                        std::fprintf(rvp_fp,
                            "%u (L%u,%u,%u,%u) %u(%u,%u,%u) "
                            "(%llu,%llu,%llu) %u (%llu,%llu,%llu) "
                            "%lx=%g %s%s %d\n",
                            e, (unsigned)pNO[e].getLevel(),
                            (unsigned)pNO[e].getX(),
                            (unsigned)pNO[e].getY(),
                            (unsigned)pNO[e].getZ(),
                            n, ni, nj, nk,
                            gx, gy, gz,
                            cg, cx, cy, cz,
                            (unsigned long)hb, (double)vsend,
                            geom_match ? "GEOM" : "geom",
                            cg_match ? "_CG" : "_cg",
                            target);
                    }
                }
            }
        }
    }
    if (rvp_fp) std::fclose(rvp_fp);
    if (rvp_dir) rvp_call++;

    // ---- Step 3: alltoallv TreeNodes + per-element DG values + flags
    std::vector<int> sendCntTN(npes), sendCntDG(npes);
    std::vector<int> sendOffTN(npes, 0), sendOffDG(npes, 0);
    for (int p = 0; p < npes; p++) {
        sendCntTN[p] = (int)sendTN[p].size();
        sendCntDG[p] = (int)sendDG[p].size();
    }
    for (int p = 1; p < npes; p++) {
        sendOffTN[p] = sendOffTN[p - 1] + sendCntTN[p - 1];
        sendOffDG[p] = sendOffDG[p - 1] + sendCntDG[p - 1];
    }

    std::vector<int> recvCntTN(npes), recvCntDG(npes);
    std::vector<int> recvOffTN(npes, 0), recvOffDG(npes, 0);
    MPI_Alltoall(sendCntTN.data(), 1, MPI_INT, recvCntTN.data(), 1, MPI_INT,
                 comm);
    MPI_Alltoall(sendCntDG.data(), 1, MPI_INT, recvCntDG.data(), 1, MPI_INT,
                 comm);
    int totRecvTN = 0, totRecvDG = 0;
    for (int p = 0; p < npes; p++) {
        recvOffTN[p] = totRecvTN;
        recvOffDG[p] = totRecvDG;
        totRecvTN += recvCntTN[p];
        totRecvDG += recvCntDG[p];
    }

    std::vector<ot::TreeNode> flatTN;
    std::vector<T> flatDG;
    std::vector<unsigned char> flatFlag;
    flatTN.reserve(sendOffTN[npes - 1] + sendCntTN[npes - 1]);
    flatDG.reserve(sendOffDG[npes - 1] + sendCntDG[npes - 1]);
    flatFlag.reserve(sendOffDG[npes - 1] + sendCntDG[npes - 1]);
    for (int p = 0; p < npes; p++) {
        flatTN.insert(flatTN.end(), sendTN[p].begin(), sendTN[p].end());
        flatDG.insert(flatDG.end(), sendDG[p].begin(), sendDG[p].end());
        flatFlag.insert(flatFlag.end(), sendLocalFlag[p].begin(),
                        sendLocalFlag[p].end());
    }

    std::vector<ot::TreeNode> recvTN(totRecvTN);
    std::vector<T> recvDG(totRecvDG);
    std::vector<unsigned char> recvFlag(totRecvDG);
    MPI_Alltoallv(flatTN.data(), sendCntTN.data(), sendOffTN.data(),
                  par::Mpi_datatype<ot::TreeNode>::value(), recvTN.data(),
                  recvCntTN.data(), recvOffTN.data(),
                  par::Mpi_datatype<ot::TreeNode>::value(), comm);
    MPI_Alltoallv(flatDG.data(), sendCntDG.data(), sendOffDG.data(),
                  par::Mpi_datatype<T>::value(), recvDG.data(),
                  recvCntDG.data(), recvOffDG.data(),
                  par::Mpi_datatype<T>::value(), comm);
    MPI_Alltoallv(flatFlag.data(), sendCntDG.data(), sendOffDG.data(),
                  MPI_UNSIGNED_CHAR, recvFlag.data(), recvCntDG.data(),
                  recvOffDG.data(), MPI_UNSIGNED_CHAR, comm);

    // ---- Step 4: write into dstMesh's CG via E2N_CG and refresh
    //              m_uiLocalNodalDG (used by graph-partition DG ghost path).
    if (!dstMesh->isActive()) return;

    std::unordered_map<ot::TreeNode, unsigned int, TNHash> tnToLocal;
    tnToLocal.reserve(
        (size_t)(dstMesh->getElementLocalEnd() -
                 dstMesh->getElementLocalBegin()));
    {
        const auto* pNN = dstMesh->getAllElements().data();
        for (unsigned int e = dstMesh->getElementLocalBegin();
             e < dstMesh->getElementLocalEnd(); e++)
            tnToLocal.emplace(pNN[e], e);
    }

    // Two-pass write:
    //   Pass 1: only authoritative writes (recvFlag=1). Mark each
    //           local cg slot that received one.
    //   Pass 2: non-authoritative writes (recvFlag=0) but only where
    //           pass 1 didn't write. This handles positions where no
    //           src rank's local element ends up on dst's owning rank
    //           (fallback to a potentially-stale value is better
    //           than leaving the slot at zero).
    const auto& e2n = dstMesh->getE2NMapping();
    const unsigned int dstNodeLocalBegin = dstMesh->getNodeLocalBegin();
    const unsigned int dstNodeLocalEnd   = dstMesh->getNodeLocalEnd();
    const unsigned int dstEleBegin = dstMesh->getElementLocalBegin();
    const unsigned int numLocalEle = dstMesh->getElementLocalEnd() - dstEleBegin;
    std::vector<unsigned char> cgWrittenAuth(
        dstNodeLocalEnd - dstNodeLocalBegin, 0);
    // resolve tnToLocal once per recvTN entry (was 3x: pass 1, pass 2,
    // and the separate localDG-fill pass).
    std::vector<int> eAbs(recvTN.size(), -1);
    for (size_t i = 0; i < recvTN.size(); i++) {
        auto it = tnToLocal.find(recvTN[i]);
        if (it != tnToLocal.end()) eAbs[i] = (int)it->second;
    }

    // fused authoritative-write + localDG refresh (was two separate
    // O(recvTN * npe) passes).
    std::vector<double>& localDG = dstMesh->getLocalNodalDGRef();
    localDG.assign(numLocalEle * npe, (double)0);
    for (size_t i = 0; i < recvTN.size(); i++) {
        if (eAbs[i] < 0) continue;
        const unsigned int e      = (unsigned int)eAbs[i];
        const unsigned int eLocal = e - dstEleBegin;
        for (unsigned int n = 0; n < npe; n++) {
            const T val = recvDG[i * npe + n];
            localDG[eLocal * npe + n] = (double)val;
            if (recvFlag[i * npe + n]) {
                const unsigned int cg = e2n[e * npe + n];
                vecOut[cg] = val;
                if (cg >= dstNodeLocalBegin && cg < dstNodeLocalEnd)
                    cgWrittenAuth[cg - dstNodeLocalBegin] = 1;
            }
        }
    }

    // non-authoritative fill: stays separate because it depends on
    // cgWrittenAuth set by the auth pass for the same cg.
    for (size_t i = 0; i < recvTN.size(); i++) {
        if (eAbs[i] < 0) continue;
        const unsigned int e = (unsigned int)eAbs[i];
        for (unsigned int n = 0; n < npe; n++) {
            if (!recvFlag[i * npe + n]) {
                const unsigned int cg = e2n[e * npe + n];
                if (cg >= dstNodeLocalBegin && cg < dstNodeLocalEnd) {
                    if (!cgWrittenAuth[cg - dstNodeLocalBegin])
                        vecOut[cg] = recvDG[i * npe + n];
                } else {
                    vecOut[cg] = recvDG[i * npe + n];
                }
            }
        }
    }

    // recv-side probe: log every dst cg at target phys with the value
    // we just wrote, plus all recvTN entries that target that phys.
#ifdef DENDRO_ENABLE_DEBUG_PROBES
    if (rvp_dir && rvp_phys_on && rvp_in_range && dstMesh->isActive()) {
        char fn[1024];
        std::snprintf(fn, sizeof(fn), "%s/redist_recv_call%d_r%d.txt",
                      rvp_dir, rvp_call - 1, rank);  // rvp_call already inc'd
        FILE* fp = std::fopen(fn, "w");
        if (fp) {
            std::fprintf(fp,
                "# redistributeVec recv-side probe, call=%d rank=%d\n"
                "# target_phys=(%llu,%llu,%llu)\n",
                rvp_call - 1, rank, rvp_x, rvp_y, rvp_z);
            const auto* dstPN_p = dstMesh->getAllElements().data();
            const auto& dst_e2n = dstMesh->getE2NMapping();
            const unsigned int dstEOrd_p = dstMesh->getElementOrder();
            // walk dst LOCAL elements; for any node matching target phys,
            // dump (ele, sub, cg, value, authWritten).
            std::fprintf(fp, "## dst-LOCAL writes at target phys:\n");
            for (unsigned int e = dstMesh->getElementLocalBegin();
                 e < dstMesh->getElementLocalEnd(); e++) {
                const auto& tn = dstPN_p[e];
                for (unsigned int n = 0; n < npe; n++) {
                    const unsigned int ni = n%(dstEOrd_p+1);
                    const unsigned int nj = (n/(dstEOrd_p+1))%(dstEOrd_p+1);
                    const unsigned int nk =
                        n/((dstEOrd_p+1)*(dstEOrd_p+1));
                    const unsigned long long lenE =
                        (unsigned long long)1u
                        << (m_uiMaxDepth - tn.getLevel());
                    const unsigned long long gx =
                        (unsigned long long)tn.getX()*dstEOrd_p
                        + (unsigned long long)ni*lenE;
                    const unsigned long long gy =
                        (unsigned long long)tn.getY()*dstEOrd_p
                        + (unsigned long long)nj*lenE;
                    const unsigned long long gz =
                        (unsigned long long)tn.getZ()*dstEOrd_p
                        + (unsigned long long)nk*lenE;
                    if (gx != rvp_x || gy != rvp_y || gz != rvp_z) continue;
                    const unsigned int cg = dst_e2n[e * npe + n];
                    uint64_t hb = 0;
                    T val = vecOut[cg];
                    std::memcpy(&hb, &val, sizeof(hb));
                    const bool is_local = (cg >= dstNodeLocalBegin
                                           && cg < dstNodeLocalEnd);
                    int auth =
                        (is_local
                         && (cg - dstNodeLocalBegin)
                             < cgWrittenAuth.size())
                            ? (int)cgWrittenAuth[cg - dstNodeLocalBegin]
                            : -1;
                    // resolve cg's cg2dg-derived (oe, on, TN, phys)
                    unsigned long long cgPx = 0, cgPy = 0, cgPz = 0;
                    unsigned int oe2 = 0, on2 = 0;
                    unsigned int oTN2L = 0, oTN2X = 0, oTN2Y = 0, oTN2Z = 0;
                    const char* cgPhysOk = "DEFAULT";
                    if (cg < dstMesh->getCG2DGMap().size()) {
                        const auto& cg2dg2 = dstMesh->getCG2DGMap();
                        unsigned int dg2 = cg2dg2[cg];
                        if (dg2 != LOOK_UP_TABLE_DEFAULT) {
                            oe2 = dg2 / npe;
                            on2 = dg2 % npe;
                            if (oe2 < dstMesh->getAllElements().size()) {
                                const auto& oTN2 =
                                    dstMesh->getAllElements()[oe2];
                                oTN2L = oTN2.getLevel();
                                oTN2X = oTN2.getX();
                                oTN2Y = oTN2.getY();
                                oTN2Z = oTN2.getZ();
                                const unsigned int oni2 =
                                    on2 % (dstEOrd_p+1);
                                const unsigned int onj2 =
                                    (on2/(dstEOrd_p+1)) % (dstEOrd_p+1);
                                const unsigned int onk2 =
                                    on2 / ((dstEOrd_p+1)*(dstEOrd_p+1));
                                const unsigned long long olen2 =
                                    (unsigned long long)1u
                                    << (m_uiMaxDepth - oTN2.getLevel());
                                cgPx = (unsigned long long)oTN2.getX()
                                    * dstEOrd_p
                                    + (unsigned long long)oni2 * olen2;
                                cgPy = (unsigned long long)oTN2.getY()
                                    * dstEOrd_p
                                    + (unsigned long long)onj2 * olen2;
                                cgPz = (unsigned long long)oTN2.getZ()
                                    * dstEOrd_p
                                    + (unsigned long long)onk2 * olen2;
                                cgPhysOk =
                                    (cgPx == rvp_x && cgPy == rvp_y
                                     && cgPz == rvp_z) ? "CG_TGT" : "CG_OTHER";
                            } else {
                                cgPhysOk = "OE_OOR";
                            }
                        }
                    }
                    std::fprintf(fp,
                        "  dstEle=%u TN(L%u,%u,%u,%u) n=%u(%u,%u,%u) "
                        "cg=%u %s vecOut=%lx=%g auth=%d "
                        "cg2dg=(oe=%u,on=%u) cg2dgTN=(L%u,%u,%u,%u) "
                        "cg2dgPhys=(%llu,%llu,%llu) %s\n",
                        e, (unsigned)tn.getLevel(),
                        (unsigned)tn.getX(),
                        (unsigned)tn.getY(),
                        (unsigned)tn.getZ(),
                        n, ni, nj, nk,
                        cg, is_local ? "L" : "G",
                        (unsigned long)hb, (double)val, auth,
                        oe2, on2,
                        oTN2L, oTN2X, oTN2Y, oTN2Z,
                        cgPx, cgPy, cgPz, cgPhysOk);
                }
            }
            // also dump recvDG entries targeting this phys (per recv TN).
            std::fprintf(fp, "## recvDG entries with TN-node matching target phys:\n");
            for (size_t i = 0; i < recvTN.size(); i++) {
                const auto& tn = recvTN[i];
                auto itL = tnToLocal.find(tn);
                if (itL == tnToLocal.end()) continue;
                for (unsigned int n = 0; n < npe; n++) {
                    const unsigned int ni = n%(dstEOrd_p+1);
                    const unsigned int nj = (n/(dstEOrd_p+1))%(dstEOrd_p+1);
                    const unsigned int nk =
                        n/((dstEOrd_p+1)*(dstEOrd_p+1));
                    const unsigned long long lenE =
                        (unsigned long long)1u
                        << (m_uiMaxDepth - tn.getLevel());
                    const unsigned long long gx =
                        (unsigned long long)tn.getX()*dstEOrd_p
                        + (unsigned long long)ni*lenE;
                    const unsigned long long gy =
                        (unsigned long long)tn.getY()*dstEOrd_p
                        + (unsigned long long)nj*lenE;
                    const unsigned long long gz =
                        (unsigned long long)tn.getZ()*dstEOrd_p
                        + (unsigned long long)nk*lenE;
                    if (gx != rvp_x || gy != rvp_y || gz != rvp_z) continue;
                    T recv = recvDG[i * npe + n];
                    uint64_t hb = 0;
                    std::memcpy(&hb, &recv, sizeof(hb));
                    std::fprintf(fp,
                        "  recvTN(L%u,%u,%u,%u) n=%u(%u,%u,%u) "
                        "recvDG=%lx=%g flag=%d\n",
                        (unsigned)tn.getLevel(),
                        (unsigned)tn.getX(),
                        (unsigned)tn.getY(),
                        (unsigned)tn.getZ(),
                        n, ni, nj, nk,
                        (unsigned long)hb, (double)recv,
                        (int)recvFlag[i * npe + n]);
                }
            }
            // Explicit dump of specific cg ids the user is tracking.
            // (workaround for any LOOK_UP_TABLE_DEFAULT filtering issue
            // in the next loop.)
            std::fprintf(fp, "## explicit cg state dump:\n");
            for (unsigned int ccc : std::vector<unsigned int>{10785, 12141, 11229}) {
                if (ccc >= dstMesh->getDegOfFreedom()) {
                    std::fprintf(fp, "  cg=%u OUT-OF-RANGE NACT=%u\n",
                                 ccc, dstMesh->getDegOfFreedom());
                    continue;
                }
                if (ccc >= dstMesh->getCG2DGMap().size()) {
                    std::fprintf(fp, "  cg=%u OUT-OF-RANGE-CG2DG sz=%zu\n",
                                 ccc, dstMesh->getCG2DGMap().size());
                    continue;
                }
                unsigned int dgC = dstMesh->getCG2DGMap()[ccc];
                std::fprintf(fp, "  cg=%u dg=%u (npe=%u → oe=%u on=%u)",
                             ccc, dgC, npe, dgC/npe, dgC%npe);
                if (dgC == LOOK_UP_TABLE_DEFAULT) {
                    std::fprintf(fp, " LOOK_UP_TABLE_DEFAULT\n");
                    continue;
                }
                unsigned int oeC = dgC / npe;
                if (oeC >= dstMesh->getAllElements().size()) {
                    std::fprintf(fp, " OE-OOR (#eles=%zu)\n",
                                 dstMesh->getAllElements().size());
                    continue;
                }
                const auto& oTNC = dstMesh->getAllElements()[oeC];
                unsigned int onC = dgC % npe;
                const unsigned int oniC = onC % (dstEOrd_p+1);
                const unsigned int onjC =
                    (onC/(dstEOrd_p+1)) % (dstEOrd_p+1);
                const unsigned int onkC =
                    onC / ((dstEOrd_p+1)*(dstEOrd_p+1));
                const unsigned long long olenC =
                    (unsigned long long)1u
                    << (m_uiMaxDepth - oTNC.getLevel());
                const unsigned long long ppx =
                    (unsigned long long)oTNC.getX() * dstEOrd_p
                    + (unsigned long long)oniC * olenC;
                const unsigned long long ppy =
                    (unsigned long long)oTNC.getY() * dstEOrd_p
                    + (unsigned long long)onjC * olenC;
                const unsigned long long ppz =
                    (unsigned long long)oTNC.getZ() * dstEOrd_p
                    + (unsigned long long)onkC * olenC;
                T vv = vecOut[ccc];
                uint64_t hh = 0;
                std::memcpy(&hh, &vv, sizeof(hh));
                const bool tcgLocal2 =
                    (ccc >= dstNodeLocalBegin && ccc < dstNodeLocalEnd);
                std::fprintf(fp,
                    " TN(L%u,%u,%u,%u) (i,j,k)=(%u,%u,%u) phys=(%llu,%llu,%llu)"
                    " val=%lx=%g loc=%c\n",
                    (unsigned)oTNC.getLevel(),
                    (unsigned)oTNC.getX(),
                    (unsigned)oTNC.getY(),
                    (unsigned)oTNC.getZ(),
                    oniC, onjC, onkC,
                    ppx, ppy, ppz,
                    (unsigned long)hh, (double)vv,
                    tcgLocal2 ? 'L' : 'G');
            }
            // Find every dst LOCAL element-node referencing cgs whose
            // cg2dgPhys==target. Reveals where the "orphan" cgs are
            // actually referenced (may not be at target geom).
            std::fprintf(fp,
                "## E2N back-refs for cgs with cg2dgPhys==target:\n");
            // first collect cgs in dst whose cg2dg-phys matches target
            std::vector<unsigned int> targetCgs;
            {
                const auto& cg2dg3 = dstMesh->getCG2DGMap();
                for (unsigned int ccc = 0;
                     ccc < dstMesh->getDegOfFreedom(); ccc++) {
                    if (ccc >= cg2dg3.size()) continue;
                    unsigned int dgC = cg2dg3[ccc];
                    if (dgC == LOOK_UP_TABLE_DEFAULT) continue;
                    unsigned int oeC = dgC / npe;
                    unsigned int onC = dgC % npe;
                    if (oeC >= dstMesh->getAllElements().size()) continue;
                    const auto& oTNC = dstMesh->getAllElements()[oeC];
                    const unsigned int oniC = onC % (dstEOrd_p+1);
                    const unsigned int onjC =
                        (onC/(dstEOrd_p+1)) % (dstEOrd_p+1);
                    const unsigned int onkC =
                        onC / ((dstEOrd_p+1)*(dstEOrd_p+1));
                    const unsigned long long olenC =
                        (unsigned long long)1u
                        << (m_uiMaxDepth - oTNC.getLevel());
                    if ((unsigned long long)oTNC.getX() * dstEOrd_p
                        + (unsigned long long)oniC * olenC != rvp_x) continue;
                    if ((unsigned long long)oTNC.getY() * dstEOrd_p
                        + (unsigned long long)onjC * olenC != rvp_y) continue;
                    if ((unsigned long long)oTNC.getZ() * dstEOrd_p
                        + (unsigned long long)onkC * olenC != rvp_z) continue;
                    targetCgs.push_back(ccc);
                }
            }
            std::fprintf(fp, "  targetCgs count=%zu\n", targetCgs.size());
            for (unsigned int tcg : targetCgs) {
                const bool tcgLocal =
                    (tcg >= dstNodeLocalBegin && tcg < dstNodeLocalEnd);
                std::fprintf(fp, "  ## cg=%u %s — backrefs:\n",
                             tcg, tcgLocal ? "L" : "G");
                bool any = false;
                for (unsigned int e = 0;
                     e < dstMesh->getAllElements().size(); e++) {
                    for (unsigned int n = 0; n < npe; n++) {
                        if (e2n[e*npe + n] != tcg) continue;
                        const unsigned int niB = n%(dstEOrd_p+1);
                        const unsigned int njB =
                            (n/(dstEOrd_p+1))%(dstEOrd_p+1);
                        const unsigned int nkB =
                            n/((dstEOrd_p+1)*(dstEOrd_p+1));
                        const ot::TreeNode& bTN = dstPN_p[e];
                        const unsigned long long lenB =
                            (unsigned long long)1u
                            << (m_uiMaxDepth - bTN.getLevel());
                        const unsigned long long gxB =
                            (unsigned long long)bTN.getX()*dstEOrd_p
                            + (unsigned long long)niB*lenB;
                        const unsigned long long gyB =
                            (unsigned long long)bTN.getY()*dstEOrd_p
                            + (unsigned long long)njB*lenB;
                        const unsigned long long gzB =
                            (unsigned long long)bTN.getZ()*dstEOrd_p
                            + (unsigned long long)nkB*lenB;
                        const bool isLocalE =
                            (e >= dstMesh->getElementLocalBegin()
                             && e < dstMesh->getElementLocalEnd());
                        std::fprintf(fp,
                            "    e=%u %s TN(L%u,%u,%u,%u) n=%u(%u,%u,%u) "
                            "geom=(%llu,%llu,%llu)\n",
                            e, isLocalE ? "L" : "G",
                            (unsigned)bTN.getLevel(),
                            (unsigned)bTN.getX(),
                            (unsigned)bTN.getY(),
                            (unsigned)bTN.getZ(),
                            n, niB, njB, nkB,
                            gxB, gyB, gzB);
                        any = true;
                    }
                }
                if (!any) std::fprintf(fp, "    (no E2N back-refs found)\n");
            }
            // Extra: walk recvTN entries; for each (i, n), if dst's
            // e2n-resolved cg has cg2dgPhys == target, log it. Catches
            // writes from element-nodes whose GEOMETRIC phys != target
            // but whose E2N → cg → cg2dgPhys IS target (E2N misrouting).
            std::fprintf(fp,
                "## E2N-resolved writes targeting cg2dgPhys=target (any geom):\n");
            for (size_t i = 0; i < recvTN.size(); i++) {
                const auto& tnR = recvTN[i];
                auto itL = tnToLocal.find(tnR);
                if (itL == tnToLocal.end()) continue;
                unsigned int dstE = itL->second;
                for (unsigned int n = 0; n < npe; n++) {
                    unsigned int cgW = e2n[dstE * npe + n];
                    if (cgW >= dstMesh->getCG2DGMap().size()) continue;
                    unsigned int dgW = dstMesh->getCG2DGMap()[cgW];
                    if (dgW == LOOK_UP_TABLE_DEFAULT) continue;
                    unsigned int oeW = dgW / npe;
                    unsigned int onW = dgW % npe;
                    if (oeW >= dstMesh->getAllElements().size()) continue;
                    const auto& oTNW = dstMesh->getAllElements()[oeW];
                    const unsigned int oniW = onW % (dstEOrd_p+1);
                    const unsigned int onjW = (onW/(dstEOrd_p+1)) % (dstEOrd_p+1);
                    const unsigned int onkW = onW / ((dstEOrd_p+1)*(dstEOrd_p+1));
                    const unsigned long long olenW =
                        (unsigned long long)1u
                        << (m_uiMaxDepth - oTNW.getLevel());
                    const unsigned long long cgPxW =
                        (unsigned long long)oTNW.getX() * dstEOrd_p
                        + (unsigned long long)oniW * olenW;
                    const unsigned long long cgPyW =
                        (unsigned long long)oTNW.getY() * dstEOrd_p
                        + (unsigned long long)onjW * olenW;
                    const unsigned long long cgPzW =
                        (unsigned long long)oTNW.getZ() * dstEOrd_p
                        + (unsigned long long)onkW * olenW;
                    if (cgPxW != rvp_x || cgPyW != rvp_y
                        || cgPzW != rvp_z) continue;
                    // log the write
                    const unsigned int niR = n%(dstEOrd_p+1);
                    const unsigned int njR = (n/(dstEOrd_p+1))%(dstEOrd_p+1);
                    const unsigned int nkR = n/((dstEOrd_p+1)*(dstEOrd_p+1));
                    const unsigned long long lenR =
                        (unsigned long long)1u
                        << (m_uiMaxDepth - tnR.getLevel());
                    const unsigned long long gxR =
                        (unsigned long long)tnR.getX()*dstEOrd_p
                        + (unsigned long long)niR*lenR;
                    const unsigned long long gyR =
                        (unsigned long long)tnR.getY()*dstEOrd_p
                        + (unsigned long long)njR*lenR;
                    const unsigned long long gzR =
                        (unsigned long long)tnR.getZ()*dstEOrd_p
                        + (unsigned long long)nkR*lenR;
                    T recvR = recvDG[i * npe + n];
                    uint64_t hbR = 0;
                    std::memcpy(&hbR, &recvR, sizeof(hbR));
                    const bool is_localW =
                        (cgW >= dstNodeLocalBegin && cgW < dstNodeLocalEnd);
                    std::fprintf(fp,
                        "  recvTN(L%u,%u,%u,%u) n=%u(%u,%u,%u) "
                        "geomR=(%llu,%llu,%llu) → cg=%u %s "
                        "cg2dgTN=(L%u,%u,%u,%u) recvDG=%lx=%g flag=%d\n",
                        (unsigned)tnR.getLevel(),
                        (unsigned)tnR.getX(),
                        (unsigned)tnR.getY(),
                        (unsigned)tnR.getZ(),
                        n, niR, njR, nkR,
                        gxR, gyR, gzR,
                        cgW, is_localW ? "L" : "G",
                        (unsigned)oTNW.getLevel(),
                        (unsigned)oTNW.getX(),
                        (unsigned)oTNW.getY(),
                        (unsigned)oTNW.getZ(),
                        (unsigned long)hbR, (double)recvR,
                        (int)recvFlag[i * npe + n]);
                }
            }
            std::fclose(fp);
        }
    }
#endif

    // ---- Step 5: orphan fill. Dst local CGs NOT referenced by any dst
    //              local element ("orphans") don't get written by the
    //              element-matching pass above, since every write lands
    //              via dst-local-element E2N_CG. On graph partitions
    //              these exist; on SFC they don't (so this step is a
    //              no-op for SFC).
    //
    //              Without this fill, orphan dst CGs carry whatever
    //              vecOut held pre-call: zero on first call, the IC
    //              (populated in init_grid) on the first remesh, stale
    //              values thereafter — i.e., orphans stay frozen while
    //              the field evolves. Here we pull the correct current
    //              value from src by phys_pos match against src local
    //              CGs. Uses the same scaled-int (getX()*eOrder+sub*len)
    //              phys_pos keys as orphanPreGather to avoid level-
    //              dependent truncation.
    //
    //              Only src-LOCAL cgs are trusted (ghost cgs on src may
    //              be stale — e.g., case-B holes). If no src rank has a
    //              local cg at an orphan's phys_pos, the orphan is left
    //              unfilled (rare residual; separate case-B fix needed).
    const unsigned int dstEOrd  = dstMesh->getElementOrder();
    const auto* dstPN           = dstMesh->getAllElements().data();
    const auto& dstCG2DG        = dstMesh->getCG2DGMap();
    std::vector<unsigned char> dstRefByLocal(
        dstNodeLocalEnd - dstNodeLocalBegin, 0);
    for (unsigned int e = dstMesh->getElementLocalBegin();
         e < dstMesh->getElementLocalEnd(); e++) {
        for (unsigned int n = 0; n < npe; n++) {
            unsigned int cg = e2n[e * npe + n];
            if (cg >= dstNodeLocalBegin && cg < dstNodeLocalEnd)
                dstRefByLocal[cg - dstNodeLocalBegin] = 1;
        }
    }
    struct OrphanKey {
        int          rank;
        unsigned int cg;
        uint64_t     x, y, z;
    };
    std::vector<OrphanKey> myOrphans;
    for (unsigned int cg = dstNodeLocalBegin; cg < dstNodeLocalEnd; cg++) {
        if (dstRefByLocal[cg - dstNodeLocalBegin]) continue;
        unsigned int dg = dstCG2DG[cg];
        unsigned int ce = dg / npe;
        unsigned int cn = dg % npe;
        uint64_t len =
            (uint64_t)1 << (m_uiMaxDepth - dstPN[ce].getLevel());
        uint64_t x =
            (uint64_t)dstPN[ce].getX() * dstEOrd +
            (uint64_t)(cn % (dstEOrd + 1)) * len;
        uint64_t y =
            (uint64_t)dstPN[ce].getY() * dstEOrd +
            (uint64_t)((cn / (dstEOrd + 1)) % (dstEOrd + 1)) * len;
        uint64_t z =
            (uint64_t)dstPN[ce].getZ() * dstEOrd +
            (uint64_t)(cn / ((dstEOrd + 1) * (dstEOrd + 1))) * len;
        myOrphans.push_back({rank, cg, x, y, z});
    }

    int myOrphCount = (int)myOrphans.size();
    std::vector<int> oCounts(npes), oOffs(npes, 0);
    MPI_Allgather(&myOrphCount, 1, MPI_INT, oCounts.data(), 1, MPI_INT,
                  comm);
    int oTotal = 0;
    for (int p = 0; p < npes; p++) {
        oOffs[p] = oTotal;
        oTotal += oCounts[p];
    }
    if (oTotal == 0) return;  // no orphans anywhere, done

    std::vector<int>          oAllRank(oTotal);
    std::vector<unsigned int> oAllCG(oTotal);
    std::vector<uint64_t>     oAllX(oTotal), oAllY(oTotal), oAllZ(oTotal);
    {
        std::vector<int>          myR(myOrphCount);
        std::vector<unsigned int> myC(myOrphCount);
        std::vector<uint64_t>     myX(myOrphCount), myY(myOrphCount),
                                  myZ(myOrphCount);
        for (int k = 0; k < myOrphCount; k++) {
            myR[k] = myOrphans[k].rank;
            myC[k] = myOrphans[k].cg;
            myX[k] = myOrphans[k].x;
            myY[k] = myOrphans[k].y;
            myZ[k] = myOrphans[k].z;
        }
        MPI_Allgatherv(myR.data(), myOrphCount, MPI_INT, oAllRank.data(),
                       oCounts.data(), oOffs.data(), MPI_INT, comm);
        MPI_Allgatherv(myC.data(), myOrphCount, MPI_UNSIGNED,
                       oAllCG.data(), oCounts.data(), oOffs.data(),
                       MPI_UNSIGNED, comm);
        MPI_Allgatherv(myX.data(), myOrphCount, MPI_UINT64_T,
                       oAllX.data(), oCounts.data(), oOffs.data(),
                       MPI_UINT64_T, comm);
        MPI_Allgatherv(myY.data(), myOrphCount, MPI_UINT64_T,
                       oAllY.data(), oCounts.data(), oOffs.data(),
                       MPI_UINT64_T, comm);
        MPI_Allgatherv(myZ.data(), myOrphCount, MPI_UINT64_T,
                       oAllZ.data(), oCounts.data(), oOffs.data(),
                       MPI_UINT64_T, comm);
    }

    // Build phys_pos → src local cg map. Only LOCAL src cgs: ghost cgs
    // on src can be stale (case-B). By filtering to local-only, each
    // phys_pos has at most one entry across all ranks (the canonical
    // owner), so we avoid any ordering ambiguity on the dst side.
    //
    // Key by CG2DG-derived phys (canonical phys) instead of element-node
    // geometric phys. For non-hanging meshes the two are equal. For
    // hanging-face neighborhoods, an element's (e,n) geometric phys can
    // differ from its E2N-resolved cg's cg2dg phys: the hanging child
    // node's value lives at the coarse parent's canonical cg. Keying by
    // geometric would map a hanging position to the coarse parent's cg
    // value — orphan fill would then write that coarse-parent value (at
    // a *different* phys) into the dst orphan, corrupting state at the
    // target phys (see docs/findings_2026-05-26_orphan_fill_geom_key_bug).
    // Keying by canonical phys ensures each src LOCAL cg appears once
    // at its true phys, so orphan fill reads the correct value.
    //
    // env DENDRO_ORPHAN_FILL_GEOM_KEY=1 reverts to the buggy geometric
    // keying (for A/B comparison only; do not enable in production).
    static const char* ofgk_env = std::getenv("DENDRO_ORPHAN_FILL_GEOM_KEY");
    static const bool ofgk_on =
        ofgk_env && ofgk_env[0] == '1' && ofgk_env[1] == '\0';
    std::map<std::tuple<uint64_t, uint64_t, uint64_t>, unsigned int>
        srcPosToLocalCG;
    if (m_uiIsActive) {
        const unsigned int srcEOrd = m_uiElementOrder;
        const auto* srcPN          = m_uiAllElements.data();
        if (ofgk_on) {
            // legacy buggy geometric-phys keying (for A/B only)
            for (unsigned int e = m_uiElementLocalBegin;
                 e < m_uiElementLocalEnd; e++) {
                for (unsigned int n = 0; n < npe; n++) {
                    unsigned int cg = m_uiE2NMapping_CG[e * npe + n];
                    if (!(cg >= m_uiNodeLocalBegin &&
                          cg < m_uiNodeLocalEnd)) continue;
                    uint64_t len = (uint64_t)1
                                   << (m_uiMaxDepth - srcPN[e].getLevel());
                    uint64_t x =
                        (uint64_t)srcPN[e].getX() * srcEOrd +
                        (uint64_t)(n % (srcEOrd + 1)) * len;
                    uint64_t y =
                        (uint64_t)srcPN[e].getY() * srcEOrd +
                        (uint64_t)((n / (srcEOrd + 1)) % (srcEOrd + 1)) * len;
                    uint64_t z =
                        (uint64_t)srcPN[e].getZ() * srcEOrd +
                        (uint64_t)(n / ((srcEOrd + 1) * (srcEOrd + 1))) * len;
                    srcPosToLocalCG.emplace(std::make_tuple(x, y, z), cg);
                }
            }
        } else {
            const auto& srcCG2DG       = m_uiCG2DG;
            for (unsigned int cg = m_uiNodeLocalBegin;
                 cg < m_uiNodeLocalEnd; cg++) {
                if (cg >= srcCG2DG.size()) continue;
                unsigned int dg = srcCG2DG[cg];
                if (dg == LOOK_UP_TABLE_DEFAULT) continue;
                unsigned int oe = dg / npe;
                unsigned int on = dg % npe;
                if (oe >= m_uiAllElements.size()) continue;
                uint64_t len = (uint64_t)1
                               << (m_uiMaxDepth - srcPN[oe].getLevel());
                uint64_t x =
                    (uint64_t)srcPN[oe].getX() * srcEOrd +
                    (uint64_t)(on % (srcEOrd + 1)) * len;
                uint64_t y =
                    (uint64_t)srcPN[oe].getY() * srcEOrd +
                    (uint64_t)((on / (srcEOrd + 1)) % (srcEOrd + 1)) * len;
                uint64_t z =
                    (uint64_t)srcPN[oe].getZ() * srcEOrd +
                    (uint64_t)(on / ((srcEOrd + 1) * (srcEOrd + 1))) * len;
                srcPosToLocalCG.emplace(std::make_tuple(x, y, z), cg);
            }
        }
    }

    // Same-rank orphans: if src and dst share this rank, write locally
    // without going through MPI. (Orphans are defined per dst-rank, but
    // src may still be the canonical owner on the same rank.)
    for (int t = oOffs[rank]; t < oOffs[rank] + oCounts[rank]; t++) {
        auto it = srcPosToLocalCG.find(
            std::make_tuple(oAllX[t], oAllY[t], oAllZ[t]));
        if (it == srcPosToLocalCG.end()) continue;
        vecOut[oAllCG[t]] = vecIn[it->second];
    }

    // Cross-rank: pack (dst_cg, value) per dst rank and alltoallv.
    std::vector<std::vector<T>>            sendValO(npes);
    std::vector<std::vector<unsigned int>> sendDstCGO(npes);
    for (int t = 0; t < oTotal; t++) {
        int dstRank = oAllRank[t];
        if (dstRank == rank) continue;
        auto it = srcPosToLocalCG.find(
            std::make_tuple(oAllX[t], oAllY[t], oAllZ[t]));
        if (it == srcPosToLocalCG.end()) continue;
        sendValO[dstRank].push_back(vecIn[it->second]);
        sendDstCGO[dstRank].push_back(oAllCG[t]);
    }

    std::vector<int> sCntO(npes, 0), rCntO(npes, 0);
    for (int p = 0; p < npes; p++) sCntO[p] = (int)sendValO[p].size();
    MPI_Alltoall(sCntO.data(), 1, MPI_INT, rCntO.data(), 1, MPI_INT, comm);
    std::vector<int> sOffO(npes, 0), rOffO(npes, 0);
    int totSO = 0, totRO = 0;
    for (int p = 0; p < npes; p++) {
        sOffO[p] = totSO;
        totSO += sCntO[p];
        rOffO[p] = totRO;
        totRO += rCntO[p];
    }
    std::vector<T>            flatValO;
    std::vector<unsigned int> flatCGO;
    flatValO.reserve(totSO);
    flatCGO.reserve(totSO);
    for (int p = 0; p < npes; p++) {
        flatValO.insert(flatValO.end(), sendValO[p].begin(),
                        sendValO[p].end());
        flatCGO.insert(flatCGO.end(), sendDstCGO[p].begin(),
                       sendDstCGO[p].end());
    }
    std::vector<T>            recvValO(totRO);
    std::vector<unsigned int> recvCGO(totRO);
    MPI_Alltoallv(flatValO.data(), sCntO.data(), sOffO.data(),
                  par::Mpi_datatype<T>::value(), recvValO.data(),
                  rCntO.data(), rOffO.data(),
                  par::Mpi_datatype<T>::value(), comm);
    MPI_Alltoallv(flatCGO.data(), sCntO.data(), sOffO.data(),
                  MPI_UNSIGNED, recvCGO.data(), rCntO.data(),
                  rOffO.data(), MPI_UNSIGNED, comm);

    for (int i = 0; i < totRO; i++) {
        vecOut[recvCGO[i]] = recvValO[i];
    }
}

template <typename T>
void Mesh::zip(const T* unzippedVec, T* zippedVec) {
    if (!m_uiIsActive) return;

    // Plan-driven zip path (default). Iterates the precomputed
    // m_uiZipPlanCg / m_uiZipPlanUnzipIdx vectors built by
    // buildZipPlan() after mesh construction. Replaces the legacy
    // implicit "E2N_DG self-owned scan" inside zip with an O(N)
    // data-driven loop.
    //
    // Env gate: DENDRO_USE_LEGACY_ZIP=1 falls back to the legacy
    // implementation. Used to A/B test plan-zip vs cascade-zip
    // during the partitioning rewrite.
    static const char* legacy_env = std::getenv("DENDRO_USE_LEGACY_ZIP");
    static const bool use_legacy =
        legacy_env && legacy_env[0] == '1' && legacy_env[1] == '\0';
    if (!use_legacy) {
        const size_t N = m_uiZipPlanCg.size();
        const unsigned int* __restrict__ cgs   = m_uiZipPlanCg.data();
        const unsigned int* __restrict__ idxs  = m_uiZipPlanUnzipIdx.data();
        for (size_t i = 0; i < N; i++)
            zippedVec[cgs[i]] = unzippedVec[idxs[i]];

        // probe: dump (cg, unzip_idx, source_value, block_id, slot) for
        // target cg list. gated by EM4_ZIP_PROBE_DIR + EM4_ZIP_PROBE_CGS
        // ("cg1,cg2,..."). also resolves which block + (i,j,k) the
        // unzip_idx falls into for comparison across modes.
        static const char* zp_dir = DENDRO_PROBE_GETENV("EM4_ZIP_PROBE_DIR");
        static const char* zp_cgs = DENDRO_PROBE_GETENV("EM4_ZIP_PROBE_CGS");
        static int zp_call = 0;
        if (zp_dir && zp_cgs) {
            static std::vector<unsigned int> tgt_cgs;
            static bool tgt_parsed = false;
            if (!tgt_parsed) {
                std::string s(zp_cgs);
                size_t p = 0;
                while (p < s.size()) {
                    unsigned int c;
                    if (std::sscanf(s.c_str() + p, "%u", &c) == 1)
                        tgt_cgs.push_back(c);
                    size_t n = s.find(',', p);
                    if (n == std::string::npos) break;
                    p = n + 1;
                }
                tgt_parsed = true;
            }
            char fn[1024];
            std::snprintf(fn, sizeof(fn), "%s/zipplan_call%d_r%d.txt",
                          zp_dir, zp_call, (int)m_uiActiveRank);
            FILE* fp = std::fopen(fn, "w");
            if (fp) {
                std::fprintf(fp,
                    "# call=%d rank=%d N_plan=%zu nblk=%zu\n"
                    "# cg unzip_idx src_hex block_id blk_x blk_y blk_z "
                    "blev rlev pw lx ly lz bi bj bk phys_x phys_y phys_z\n",
                    zp_call, (int)m_uiActiveRank, N,
                    m_uiLocalBlockList.size());
                for (size_t i = 0; i < N; i++) {
                    bool match = false;
                    for (unsigned int c : tgt_cgs)
                        if (cgs[i] == c) { match = true; break; }
                    if (!match) continue;
                    const unsigned int uzi = idxs[i];
                    T sv = unzippedVec[uzi];
                    uint64_t hb = 0;
                    std::memcpy(&hb, &sv, sizeof(hb));
                    // resolve block + (bi, bj, bk) from uzi
                    int blk_id = -1;
                    unsigned int bi = 0, bj = 0, bk = 0;
                    unsigned int lx = 0, ly = 0, lz = 0;
                    unsigned int bx = 0, by = 0, bz = 0;
                    unsigned int blev = 0, rlev = 0, pw = 0;
                    long long phys_x = 0, phys_y = 0, phys_z = 0;
                    for (size_t b = 0; b < m_uiLocalBlockList.size(); b++) {
                        const auto& bk_obj = m_uiLocalBlockList[b];
                        unsigned int off = bk_obj.getOffset();
                        unsigned int Lx = bk_obj.getAllocationSzX();
                        unsigned int Ly = bk_obj.getAllocationSzY();
                        unsigned int Lz = bk_obj.getAllocationSzZ();
                        if (uzi >= off && uzi < off + Lx * Ly * Lz) {
                            blk_id = (int)b;
                            unsigned int loc = uzi - off;
                            bi = loc % Lx;
                            bj = (loc / Lx) % Ly;
                            bk = loc / (Lx * Ly);
                            lx = Lx; ly = Ly; lz = Lz;
                            bx = bk_obj.getBlockNode().getX();
                            by = bk_obj.getBlockNode().getY();
                            bz = bk_obj.getBlockNode().getZ();
                            blev = bk_obj.getBlockNode().getLevel();
                            rlev = bk_obj.getRegularGridLev();
                            pw = bk_obj.get1DPadWidth();
                            const unsigned int eOrd_p = m_uiElementOrder;
                            const unsigned int h =
                                (unsigned int)1u << (m_uiMaxDepth - rlev);
                            phys_x = (long long)bx * eOrd_p
                                + ((long long)bi - (long long)pw) * h;
                            phys_y = (long long)by * eOrd_p
                                + ((long long)bj - (long long)pw) * h;
                            phys_z = (long long)bz * eOrd_p
                                + ((long long)bk - (long long)pw) * h;
                            break;
                        }
                    }
                    std::fprintf(fp,
                        "%u %u %lx %d %u %u %u %u %u %u %u %u %u %u %u %u "
                        "%lld %lld %lld\n",
                        cgs[i], uzi, (unsigned long)hb,
                        blk_id, bx, by, bz, blev, rlev, pw,
                        lx, ly, lz, bi, bj, bk,
                        phys_x, phys_y, phys_z);
                }
                std::fclose(fp);
            }
        }
        if (zp_dir) zp_call++;
        return;
    }

    // ---- legacy zip path (kept for A/B testing) ----
    ot::TreeNode blkNode;
    unsigned int ei, ej, ek;
    unsigned int regLev;
    const ot::TreeNode* pNodes = &(*(m_uiAllElements.begin()));

    unsigned int lx, ly, lz, offset, paddWidth;

    for (unsigned int blk = 0; blk < m_uiLocalBlockList.size(); blk++) {
        blkNode   = m_uiLocalBlockList[blk].getBlockNode();
        regLev    = m_uiLocalBlockList[blk].getRegularGridLev();

        lx        = m_uiLocalBlockList[blk].getAllocationSzX();
        ly        = m_uiLocalBlockList[blk].getAllocationSzY();
        lz        = m_uiLocalBlockList[blk].getAllocationSzZ();
        offset    = m_uiLocalBlockList[blk].getOffset();
        paddWidth = m_uiLocalBlockList[blk].get1DPadWidth();

        for (unsigned int elem : m_uiLocalBlockList[blk]) {
            ei = (pNodes[elem].getX() - blkNode.getX()) >>
                 (m_uiMaxDepth - regLev);
            ej = (pNodes[elem].getY() - blkNode.getY()) >>
                 (m_uiMaxDepth - regLev);
            ek = (pNodes[elem].getZ() - blkNode.getZ()) >>
                 (m_uiMaxDepth - regLev);

            assert(pNodes[elem].getLevel() == regLev);

            for (unsigned int k = 0; k < m_uiElementOrder + 1; k++)
                for (unsigned int j = 0; j < m_uiElementOrder + 1; j++)
                    for (unsigned int i = 0; i < m_uiElementOrder + 1; i++) {
                        const unsigned int sub =
                            k * (m_uiElementOrder + 1) *
                                (m_uiElementOrder + 1) +
                            j * (m_uiElementOrder + 1) + i;
                        bool doWrite =
                            (m_uiE2NMapping_DG[elem * m_uiNpE + sub]
                             / m_uiNpE) == elem;
                        if (doWrite)
                            zippedVec[m_uiE2NMapping_CG
                                          [elem * m_uiNpE + sub]] =
                                unzippedVec
                                    [offset +
                                     (ek * m_uiElementOrder + k + paddWidth) *
                                         (ly * lx) +
                                     (ej * m_uiElementOrder + j + paddWidth) *
                                         (lx) +
                                     (ei * m_uiElementOrder + i + paddWidth)];
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
    const unsigned int paddWidth = blk.get1DPadWidth();
    const unsigned int offset    = blk.getOffset();
    const unsigned int lx        = blk.getAllocationSzX();
    const unsigned int ly        = blk.getAllocationSzY();
    const unsigned int lz        = blk.getAllocationSzZ();

    unsigned int edgeCount       = 0;
    const unsigned int edgeDir   = (OCT_DIR_LEFT_DOWN - EDGE_OFFSET);
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
    const unsigned int paddWidth = blk.get1DPadWidth();
    const unsigned int offset    = blk.getOffset();
    const unsigned int lx        = blk.getAllocationSzX();
    const unsigned int ly        = blk.getAllocationSzY();
    const unsigned int lz        = blk.getAllocationSzZ();

    unsigned int edgeCount       = 0;
    const unsigned int edgeDir   = (OCT_DIR_LEFT_UP - EDGE_OFFSET);
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
    const unsigned int paddWidth = blk.get1DPadWidth();
    const unsigned int offset    = blk.getOffset();
    const unsigned int lx        = blk.getAllocationSzX();
    const unsigned int ly        = blk.getAllocationSzY();
    const unsigned int lz        = blk.getAllocationSzZ();

    unsigned int edgeCount       = 0;
    const unsigned int edgeDir   = (OCT_DIR_LEFT_BACK - EDGE_OFFSET);
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
    const unsigned int paddWidth = blk.get1DPadWidth();
    const unsigned int offset    = blk.getOffset();
    const unsigned int lx        = blk.getAllocationSzX();
    const unsigned int ly        = blk.getAllocationSzY();
    const unsigned int lz        = blk.getAllocationSzZ();

    unsigned int edgeCount       = 0;
    const unsigned int edgeDir   = (OCT_DIR_LEFT_FRONT - EDGE_OFFSET);
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
    const unsigned int paddWidth = blk.get1DPadWidth();
    const unsigned int offset    = blk.getOffset();
    const unsigned int lx        = blk.getAllocationSzX();
    const unsigned int ly        = blk.getAllocationSzY();
    const unsigned int lz        = blk.getAllocationSzZ();

    unsigned int edgeCount       = 0;
    const unsigned int edgeDir   = (OCT_DIR_RIGHT_DOWN - EDGE_OFFSET);
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
    const unsigned int paddWidth = blk.get1DPadWidth();
    const unsigned int offset    = blk.getOffset();
    const unsigned int lx        = blk.getAllocationSzX();
    const unsigned int ly        = blk.getAllocationSzY();
    const unsigned int lz        = blk.getAllocationSzZ();

    unsigned int edgeCount       = 0;
    const unsigned int edgeDir   = (OCT_DIR_RIGHT_UP - EDGE_OFFSET);
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
    const unsigned int paddWidth = blk.get1DPadWidth();
    const unsigned int offset    = blk.getOffset();
    const unsigned int lx        = blk.getAllocationSzX();
    const unsigned int ly        = blk.getAllocationSzY();
    const unsigned int lz        = blk.getAllocationSzZ();

    unsigned int edgeCount       = 0;
    const unsigned int edgeDir   = (OCT_DIR_RIGHT_BACK - EDGE_OFFSET);
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
    const unsigned int paddWidth = blk.get1DPadWidth();
    const unsigned int offset    = blk.getOffset();
    const unsigned int lx        = blk.getAllocationSzX();
    const unsigned int ly        = blk.getAllocationSzY();
    const unsigned int lz        = blk.getAllocationSzZ();

    unsigned int edgeCount       = 0;
    const unsigned int edgeDir   = (OCT_DIR_RIGHT_FRONT - EDGE_OFFSET);
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
    const unsigned int paddWidth = blk.get1DPadWidth();
    const unsigned int offset    = blk.getOffset();
    const unsigned int lx        = blk.getAllocationSzX();
    const unsigned int ly        = blk.getAllocationSzY();
    const unsigned int lz        = blk.getAllocationSzZ();

    unsigned int edgeCount       = 0;
    const unsigned int edgeDir   = (OCT_DIR_DOWN_BACK - EDGE_OFFSET);
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
    const unsigned int paddWidth = blk.get1DPadWidth();
    const unsigned int offset    = blk.getOffset();
    const unsigned int lx        = blk.getAllocationSzX();
    const unsigned int ly        = blk.getAllocationSzY();
    const unsigned int lz        = blk.getAllocationSzZ();

    unsigned int edgeCount       = 0;
    const unsigned int edgeDir   = (OCT_DIR_DOWN_FRONT - EDGE_OFFSET);
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
    const unsigned int paddWidth = blk.get1DPadWidth();
    const unsigned int offset    = blk.getOffset();
    const unsigned int lx        = blk.getAllocationSzX();
    const unsigned int ly        = blk.getAllocationSzY();
    const unsigned int lz        = blk.getAllocationSzZ();

    unsigned int edgeCount       = 0;
    const unsigned int edgeDir   = (OCT_DIR_UP_BACK - EDGE_OFFSET);
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
    const unsigned int paddWidth = blk.get1DPadWidth();
    const unsigned int offset    = blk.getOffset();
    const unsigned int lx        = blk.getAllocationSzX();
    const unsigned int ly        = blk.getAllocationSzY();
    const unsigned int lz        = blk.getAllocationSzZ();

    unsigned int edgeCount       = 0;
    const unsigned int edgeDir   = (OCT_DIR_UP_FRONT - EDGE_OFFSET);
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

    const unsigned int paddWidth = blk.get1DPadWidth();
    const unsigned int sz        = 1u << (m_uiMaxDepth - regLev);
    const ot::TreeNode blkNode   = blk.getBlockNode();

    const unsigned int ei        = blkElem_1D - 1;
    const unsigned int ej        = 0;
    const unsigned int ek        = 0;

    const unsigned int lx        = blk.getAllocationSzX();
    const unsigned int ly        = blk.getAllocationSzY();
    const unsigned int lz        = blk.getAllocationSzZ();

    const int i_offset           = (m_uiElementOrder + paddWidth);
    const int j_offset           = -(m_uiElementOrder - paddWidth);
    const int k_offset           = -(m_uiElementOrder - paddWidth);

    const unsigned int kb        = (m_uiElementOrder - paddWidth);
    const unsigned int ke        = (m_uiElementOrder + 1);

    const unsigned int jb        = (m_uiElementOrder - paddWidth);
    const unsigned int je        = (m_uiElementOrder + 1);

    const unsigned int ib        = 0;
    const unsigned int ie        = paddWidth + 1;

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

    const unsigned int paddWidth = blk.get1DPadWidth();
    const unsigned int sz        = 1u << (m_uiMaxDepth - regLev);
    const ot::TreeNode blkNode   = blk.getBlockNode();

    const unsigned int ei        = 0;
    const unsigned int ej        = blkElem_1D - 1;
    const unsigned int ek        = 0;

    const unsigned int lx        = blk.getAllocationSzX();
    const unsigned int ly        = blk.getAllocationSzY();
    const unsigned int lz        = blk.getAllocationSzZ();

    const int i_offset           = -(m_uiElementOrder - paddWidth);
    const int j_offset           = (m_uiElementOrder + paddWidth);
    const int k_offset           = -(m_uiElementOrder - paddWidth);

    const unsigned int kb        = (m_uiElementOrder - paddWidth);
    const unsigned int ke        = (m_uiElementOrder + 1);

    const unsigned int jb        = 0;
    const unsigned int je        = paddWidth + 1;

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

    const unsigned int paddWidth = blk.get1DPadWidth();
    const unsigned int sz        = 1u << (m_uiMaxDepth - regLev);
    const ot::TreeNode blkNode   = blk.getBlockNode();

    const unsigned int ei        = blkElem_1D - 1;
    const unsigned int ej        = blkElem_1D - 1;
    const unsigned int ek        = 0;

    const unsigned int lx        = blk.getAllocationSzX();
    const unsigned int ly        = blk.getAllocationSzY();
    const unsigned int lz        = blk.getAllocationSzZ();

    const int i_offset           = (m_uiElementOrder + paddWidth);
    const int j_offset           = (m_uiElementOrder + paddWidth);
    const int k_offset           = -(m_uiElementOrder - paddWidth);

    const unsigned int kb        = (m_uiElementOrder - paddWidth);
    const unsigned int ke        = (m_uiElementOrder + 1);

    const unsigned int jb        = 0;
    const unsigned int je        = paddWidth + 1;

    const unsigned int ib        = 0;
    const unsigned int ie        = paddWidth + 1;

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

    const unsigned int ei = 0;
    const unsigned int ej = 0;
    const unsigned int ek = blkElem_1D - 1;

    const unsigned int lx = blk.getAllocationSzX();
    const unsigned int ly = blk.getAllocationSzY();
    const unsigned int lz = blk.getAllocationSzZ();

    const int i_offset    = -(m_uiElementOrder - paddWidth);
    const int j_offset    = -(m_uiElementOrder - paddWidth);
    const int k_offset    = (m_uiElementOrder + paddWidth);

    const unsigned int kb = 0;
    const unsigned int ke = paddWidth + 1;

    const unsigned int jb = (m_uiElementOrder - paddWidth);
    const unsigned int je = (m_uiElementOrder + 1);

    const unsigned int ib = (m_uiElementOrder - paddWidth);
    const unsigned int ie = (m_uiElementOrder + 1);

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

    const unsigned int paddWidth = blk.get1DPadWidth();
    const unsigned int sz        = 1u << (m_uiMaxDepth - regLev);
    const ot::TreeNode blkNode   = blk.getBlockNode();

    const unsigned int ei        = blkElem_1D - 1;
    const unsigned int ej        = 0;
    const unsigned int ek        = blkElem_1D - 1;

    const unsigned int lx        = blk.getAllocationSzX();
    const unsigned int ly        = blk.getAllocationSzY();
    const unsigned int lz        = blk.getAllocationSzZ();

    const int i_offset           = (m_uiElementOrder + paddWidth);
    const int j_offset           = -(m_uiElementOrder - paddWidth);
    const int k_offset           = (m_uiElementOrder + paddWidth);

    const unsigned int kb        = 0;
    const unsigned int ke        = paddWidth + 1;

    const unsigned int jb        = (m_uiElementOrder - paddWidth);
    const unsigned int je        = (m_uiElementOrder + 1);

    const unsigned int ib        = 0;
    const unsigned int ie        = paddWidth + 1;

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

    const unsigned int paddWidth = blk.get1DPadWidth();
    const unsigned int sz        = 1u << (m_uiMaxDepth - regLev);
    const ot::TreeNode blkNode   = blk.getBlockNode();

    const unsigned int ei        = 0;
    const unsigned int ej        = blkElem_1D - 1;
    const unsigned int ek        = blkElem_1D - 1;

    const unsigned int lx        = blk.getAllocationSzX();
    const unsigned int ly        = blk.getAllocationSzY();
    const unsigned int lz        = blk.getAllocationSzZ();

    const int i_offset           = -(m_uiElementOrder - paddWidth);
    const int j_offset           = (m_uiElementOrder + paddWidth);
    const int k_offset           = (m_uiElementOrder + paddWidth);

    const unsigned int kb        = 0;
    const unsigned int ke        = paddWidth + 1;

    const unsigned int jb        = 0;
    const unsigned int je        = paddWidth + 1;

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

    const unsigned int paddWidth = blk.get1DPadWidth();
    const unsigned int sz        = 1u << (m_uiMaxDepth - regLev);
    const ot::TreeNode blkNode   = blk.getBlockNode();

    const unsigned int ei        = blkElem_1D - 1;
    const unsigned int ej        = blkElem_1D - 1;
    const unsigned int ek        = blkElem_1D - 1;

    const unsigned int lx        = blk.getAllocationSzX();
    const unsigned int ly        = blk.getAllocationSzY();
    const unsigned int lz        = blk.getAllocationSzZ();

    const int i_offset           = (m_uiElementOrder + paddWidth);
    const int j_offset           = (m_uiElementOrder + paddWidth);
    const int k_offset           = (m_uiElementOrder + paddWidth);

    const unsigned int kb        = 0;
    const unsigned int ke        = paddWidth + 1;

    const unsigned int jb        = 0;
    const unsigned int je        = paddWidth + 1;

    const unsigned int ib        = 0;
    const unsigned int ie        = paddWidth + 1;

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
void Mesh::unzip_scatter(const T* in, T* out, unsigned int dof) {
    // probe: dump INPUT cg values at start of unzip. gates same as bbuf
    // dump (DENDRO_BBUF_DUMP_DIR / MIN_CALL / MAX_CALL). file name uses
    // "ucgin" prefix to distinguish from bbuf dumps.
    {
        static const char* ucg_dir = DENDRO_PROBE_GETENV("DENDRO_BBUF_DUMP_DIR");
        static const char* ucg_min_env =
            DENDRO_PROBE_GETENV("DENDRO_BBUF_DUMP_MIN_CALL");
        static const char* ucg_max_env =
            DENDRO_PROBE_GETENV("DENDRO_BBUF_DUMP_MAX_CALL");
        static const int ucg_min = ucg_min_env ? std::atoi(ucg_min_env) : 0;
        static const int ucg_max = ucg_max_env ? std::atoi(ucg_max_env) : 4;
        static int ucg_call_id = 0;
        if (ucg_dir && ucg_call_id >= ucg_min && ucg_call_id < ucg_max) {
            const ot::TreeNode* pN = m_uiAllElements.data();
            const unsigned int eOrd_p = m_uiElementOrder;
            const unsigned int nPe_p = m_uiNpE;
            const unsigned int maxD = m_uiMaxDepth;
            const unsigned int NLB = m_uiNodeLocalBegin;
            const unsigned int NLE = m_uiNodeLocalEnd;
            const unsigned int NACT = this->getDegOfFreedom();
            const auto& cg2dg = m_uiCG2DG;
            for (unsigned int v = 0; v < 1 && v < dof; v++) {
                const T* in_v = in + v * NACT;
                char fn[1024];
                std::snprintf(fn, sizeof(fn),
                              "%s/ucgin_call%d_v%u_r%d.txt",
                              ucg_dir, ucg_call_id, v,
                              (int)m_uiActiveRank);
                FILE* fp = std::fopen(fn, "w");
                if (!fp) continue;
                std::fprintf(fp,
                    "# call=%d v=%u rank=%d NLB=%u NLE=%u NACT=%u\n"
                    "# loc cg phys_x phys_y phys_z hex\n",
                    ucg_call_id, v, (int)m_uiActiveRank, NLB, NLE, NACT);
                for (unsigned int cg = 0; cg < NACT; cg++) {
                    if (cg >= cg2dg.size()) continue;
                    unsigned int dg = cg2dg[cg];
                    unsigned int oe = dg / nPe_p;
                    unsigned int on = dg % nPe_p;
                    if (oe >= m_uiAllElements.size()) continue;
                    const unsigned int oni = on % (eOrd_p+1);
                    const unsigned int onj = (on/(eOrd_p+1)) % (eOrd_p+1);
                    const unsigned int onk =
                        on / ((eOrd_p+1)*(eOrd_p+1));
                    const ot::TreeNode& oTN = pN[oe];
                    const unsigned int olen =
                        (unsigned int)1u << (maxD - oTN.getLevel());
                    const unsigned long long px =
                        (unsigned long long)oTN.getX() * eOrd_p
                        + (unsigned long long)oni * olen;
                    const unsigned long long py =
                        (unsigned long long)oTN.getY() * eOrd_p
                        + (unsigned long long)onj * olen;
                    const unsigned long long pz =
                        (unsigned long long)oTN.getZ() * eOrd_p
                        + (unsigned long long)onk * olen;
                    uint64_t hb = 0;
                    T val = in_v[cg];
                    std::memcpy(&hb, &val, sizeof(hb));
                    const char loc = (cg >= NLB && cg < NLE) ? 'L' : 'G';
                    std::fprintf(fp, "%c %u %llu %llu %llu %lx\n",
                                 loc, cg, px, py, pz, (unsigned long)hb);
                }
                std::fclose(fp);
            }
        }
        if (ucg_dir) ucg_call_id++;
    }

    if (!m_uiIsActive) return;

    const ot::TreeNode* pNodes = m_uiAllElements.data();
    const ot::Block* blkList   = m_uiLocalBlockList.data();
    const unsigned int eOrder  = m_uiElementOrder;
    const unsigned int nPe     = m_uiNpE;

    // diagnostic write probe: dump every uzWVec write whose Morton phys
    // is in DENDRO_UNZIP_PROBE_MBOX (default "off"). used to detect
    // multi-writer last-writer-wins races at hanging-face boundaries.
    // see docs/findings_2026-05-12.md.
    static const char* uzprobe_dir =
        DENDRO_PROBE_GETENV("DENDRO_UNZIP_PROBE");
    static const char* uzprobe_mbox =
        DENDRO_PROBE_GETENV("DENDRO_UNZIP_PROBE_MBOX");
    static const char* uzprobe_max_env =
        DENDRO_PROBE_GETENV("DENDRO_UNZIP_PROBE_MAX_CALLS");
    static const int uzprobe_max =
        uzprobe_max_env ? std::atoi(uzprobe_max_env) : 2;
    static const char* uzprobe_min_env =
        DENDRO_PROBE_GETENV("DENDRO_UNZIP_PROBE_MIN_CALL");
    static const int uzprobe_min =
        uzprobe_min_env ? std::atoi(uzprobe_min_env) : 0;
    static int uzprobe_call_id = 0;
    static double uzprobe_mxlo = 0, uzprobe_mxhi = 0;
    static double uzprobe_mylo = 0, uzprobe_myhi = 0;
    static double uzprobe_mzlo = 0, uzprobe_mzhi = 0;
    static bool uzprobe_parsed = false;
    if (uzprobe_dir && uzprobe_mbox && !uzprobe_parsed) {
        sscanf(uzprobe_mbox, "%lf,%lf,%lf,%lf,%lf,%lf",
               &uzprobe_mxlo, &uzprobe_mxhi,
               &uzprobe_mylo, &uzprobe_myhi,
               &uzprobe_mzlo, &uzprobe_mzhi);
        uzprobe_parsed = true;
    }
    bool uzprobe_on = (uzprobe_dir != nullptr) && uzprobe_parsed
                      && (uzprobe_call_id >= uzprobe_min)
                      && (uzprobe_call_id < uzprobe_max);
    // expose current call_id to getElementNodalValues probe.
    if (uzprobe_dir) g_dendro_unzip_call_id = uzprobe_call_id;
    FILE* uzprobe_fp = nullptr;
    if (uzprobe_on) {
        char fname[1024];
        std::snprintf(fname, sizeof(fname),
                      "%s/unzwr_call%d_r%d.txt",
                      uzprobe_dir, uzprobe_call_id, (int)m_uiActiveRank);
        uzprobe_fp = std::fopen(fname, "w");
        if (uzprobe_fp) {
            std::fprintf(uzprobe_fp,
                "# call=%d rank=%d mbox=%g..%g,%g..%g,%g..%g\n"
                "# path blk blev bx by bz iix jjy kkz mxx myy mzz "
                "ele elev ex ey ez var hex_value\n",
                uzprobe_call_id, (int)m_uiActiveRank,
                uzprobe_mxlo, uzprobe_mxhi,
                uzprobe_mylo, uzprobe_myhi,
                uzprobe_mzlo, uzprobe_mzhi);
        }
    }

    const unsigned int cgSz    = this->getDegOfFreedom();
    const unsigned int unSz    = this->getDegOfFreedomUnZip();

    const unsigned int* e2n    = this->getE2NMapping().data();
    const unsigned int* e2e    = this->getE2EMapping().data();

    const unsigned int dgSz    = nPe;
    std::vector<T> dg_ele_vec;
    dg_ele_vec.resize(dof * dgSz);

    T* dgWVec = dg_ele_vec.data();
    T* uzWVec = out;

    std::vector<T> p2cI_all;
    p2cI_all.resize(NUM_CHILDREN * dof * nPe);
    bool p2c_interp_valid[NUM_CHILDREN];

    const double d_compar_tol = 1e-10;

    std::vector<ot::TreeNode> childOct;
    childOct.reserve(NUM_CHILDREN);

    // EXPERIMENT (REVERTED — default OFF): partition-deterministic
    // last-writer fix at hanging-face block positions. forcing same-
    // level to apply LAST (pass 1) makes the IC block-buffer partition-
    // invariant, but over the long-haul run the +X corner drift
    // WORSENS from 2.3e-12 to 2.9e-9 because the lev-5 element's same-
    // level write uses p2c-interpolated dgWVec values at hanging
    // faces — less correct than the lev-4 ancestor's coarser-path
    // write. gate kept for further experimentation.
    // see docs/findings_2026-05-12.md.
    static const char* sw_env = std::getenv("DENDRO_UNZIP_SAME_WINS");
    static const bool same_wins_on =
        sw_env && sw_env[0] == '1' && sw_env[1] == '\0';

    // EXPERIMENT (default OFF): source-aware wins. two-pass scheme
    // where pass 0 is the standard last-writer-wins (same as no-fix),
    // and pass 1 RE-WRITES only positions whose source in the writing
    // element is NON-HANGING (i.e., a direct CG read). this makes
    // the owner element's direct read the final value at any position
    // it touches, eliminating partition-dependent winner selection at
    // CG positions. positions that are hanging in ALL writers keep
    // their pass-0 value.
    // see docs/handoff_2026-05-12.md (option 2).
    static const char* sa_env = std::getenv("DENDRO_UNZIP_SRC_AWARE");
    static const bool src_aware_on =
        sa_env && sa_env[0] == '1' && sa_env[1] == '\0';
    const int n_phases = (same_wins_on || src_aware_on) ? 2 : 1;

    // canonical TN-sorted iteration of the outer element loop:
    // (level desc, x asc, y asc, z asc) — coarser writes last.
    // graph and SFC iterate elements in the same global order, so the
    // same element wins last-writer at every multi-writer block-buffer
    // position regardless of partition. validated 2026-05-12: reduces
    // post-refinement-IGT block-buffer drift at hanging faces from
    // 8.25e-9 to 2.05e-9 (4×). other sort orders (finer-last, Morton,
    // reverse-Morton) tested and refuted — see
    // findings_2026-05-13_iter_order_exhausted.md.
    std::vector<unsigned int> ele_order(m_uiNumTotalElements);
    std::iota(ele_order.begin(), ele_order.end(), 0u);
    std::sort(ele_order.begin(), ele_order.end(),
        [&pNodes](unsigned int a, unsigned int b) {
            const ot::TreeNode& na = pNodes[a];
            const ot::TreeNode& nb = pNodes[b];
            if (na.getLevel() != nb.getLevel())
                return na.getLevel() > nb.getLevel();
            if (na.getX() != nb.getX()) return na.getX() < nb.getX();
            if (na.getY() != nb.getY()) return na.getY() < nb.getY();
            return na.getZ() < nb.getZ();
        });

    // canonical-writer gating: when m_uiUnzipCanonWriter[idx] is set
    // (multi-writer slot), only the elected writer's value is committed.
    // makes block-padding output partition-invariant — see
    // findings_2026-05-13_real_bug.md / project_real_bug_unzip_writer_order.
    const bool canon_writer_on =
        m_uiUnzipCanonWriterBuilt &&
        m_uiUnzipCanonWriter.size() == (size_t)unSz;
    const unsigned int* canon_writer_tbl =
        canon_writer_on ? m_uiUnzipCanonWriter.data() : nullptr;

    for (int phase = 0; phase < n_phases; phase++)
    for (unsigned int e_idx = 0; e_idx < m_uiNumTotalElements; e_idx++) {
        const unsigned int ele = ele_order[e_idx];
        if (m_e2b_unzip_counts[ele] == 0) continue;

        for (unsigned int ii = 0; ii < NUM_CHILDREN; ii++)
            p2c_interp_valid[ii] = false;

        // get the elemental_local(dg) values
        for (unsigned int v = 0; v < dof; v++)
            this->getElementNodalValues(in + v * cgSz, dgWVec + v * dgSz, ele,
                                        false);

        // hypothesis-2 probe: dump dgWVec for a target element TN at a
        // specific call_id. format DENDRO_DUMP_DGWVEC_TN="lev,x,y,z,call".
        // dumps to <UNZIP_PROBE>/dgWVec_call<C>_r<R>.txt; writes once per
        // call when ele matches.
        static const char* dgw_tn_env =
            DENDRO_PROBE_GETENV("DENDRO_DUMP_DGWVEC_TN");
        static unsigned int dgw_lev = 0, dgw_x = 0, dgw_y = 0, dgw_z = 0;
        static int dgw_call_target  = -1;
        static bool dgw_tn_parsed   = false;
        if (dgw_tn_env && !dgw_tn_parsed) {
            std::sscanf(dgw_tn_env, "%u,%u,%u,%u,%d", &dgw_lev, &dgw_x,
                        &dgw_y, &dgw_z, &dgw_call_target);
            dgw_tn_parsed = true;
        }
        if (dgw_tn_env && uzprobe_dir
            && uzprobe_call_id == dgw_call_target
            && pNodes[ele].getLevel() == dgw_lev
            && pNodes[ele].minX() == dgw_x
            && pNodes[ele].minY() == dgw_y
            && pNodes[ele].minZ() == dgw_z) {
            char fn[1024];
            std::snprintf(fn, sizeof(fn),
                          "%s/dgWVec_call%d_r%d.txt",
                          uzprobe_dir, uzprobe_call_id,
                          (int)m_uiActiveRank);
            FILE* dgf = std::fopen(fn, "w");
            if (dgf) {
                std::fprintf(dgf,
                    "# call=%d rank=%d ele=%u TN(lev=%u,%u,%u,%u) dof=%u nPe=%u\n"
                    "# v i j k hex hanging cg cg_x cg_y cg_z cgval_hex\n",
                    uzprobe_call_id, (int)m_uiActiveRank, ele,
                    dgw_lev, dgw_x, dgw_y, dgw_z, dof, nPe);
                const unsigned int dgw_npe_p1 = eOrder + 1;
                for (unsigned int v = 0; v < dof; v++) {
                    const T* vec_v = in + v * cgSz;
                    for (unsigned int k = 0; k < eOrder + 1; k++)
                    for (unsigned int j = 0; j < eOrder + 1; j++)
                    for (unsigned int i = 0; i < eOrder + 1; i++) {
                        uint64_t hb = 0;
                        T val = dgWVec[v * dgSz
                                       + k * dgw_npe_p1 * dgw_npe_p1
                                       + j * dgw_npe_p1 + i];
                        std::memcpy(&hb, &val, sizeof(hb));
                        int hang = this->isNodeHanging(ele, i, j, k) ? 1 : 0;
                        const unsigned int node_idx =
                            k * dgw_npe_p1 * dgw_npe_p1
                            + j * dgw_npe_p1 + i;
                        const unsigned int cg =
                            m_uiE2NMapping_CG[ele * nPe + node_idx];
                        // physical position of dgWVec target node (in TN
                        // coords * eOrder for comparison with bbuf probe).
                        const unsigned int level_ele =
                            m_uiAllElements[ele].getLevel();
                        const unsigned int h_ele =
                            (1u << (m_uiMaxDepth - level_ele));
                        const long long ex =
                            (long long)m_uiAllElements[ele].minX()
                            * eOrder + (long long)i * h_ele;
                        const long long ey =
                            (long long)m_uiAllElements[ele].minY()
                            * eOrder + (long long)j * h_ele;
                        const long long ez =
                            (long long)m_uiAllElements[ele].minZ()
                            * eOrder + (long long)k * h_ele;
                        uint64_t cgvhb = 0;
                        if (cg != LOOK_UP_TABLE_DEFAULT && cg < cgSz) {
                            T cgv = vec_v[cg];
                            std::memcpy(&cgvhb, &cgv, sizeof(cgvhb));
                        }
                        std::fprintf(dgf,
                            "%u %u %u %u %lx %d %u %lld %lld %lld %lx\n",
                            v, i, j, k,
                            (unsigned long)hb, hang,
                            cg, ex, ey, ez,
                            (unsigned long)cgvhb);
                    }
                }
                std::fclose(dgf);
            }
        }


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

            // Use the block's regular grid level. Older code read the
            // first element's level, which is correct for SFC blocks
            // (all elements are at regLev by construction) but wrong
            // for non-SFC blocks from repartitioned meshes, where
            // m_elementIndices[0] can be at any level present in the
            // block.
            const unsigned int bLev = blkList[blk].getRegularGridLev();

            const double hx   = (1u << (m_uiMaxDepth - bLev)) / (double)eOrder;
            const double xmin = blkNode.minX() - PW * hx;
            const double xmax = blkNode.maxX() + PW * hx;
            const double ymin = blkNode.minY() - PW * hx;
            const double ymax = blkNode.maxY() + PW * hx;
            const double zmin = blkNode.minZ() - PW * hx;
            const double zmax = blkNode.maxZ() + PW * hx;

            // no interpolation needed just copy.
            if (pNodes[ele].getLevel() == bLev) {
                if (same_wins_on && phase != 1) continue;
                // src_aware: both phases run same-level; pass 1 filters
                // out hanging-source writes inside the (i,j,k) loop.
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

                            // src_aware: in pass 1 only write at non-hanging
                            // source nodes (direct CG reads); skip hanging
                            // nodes so pass-0's chosen value persists.
                            if (src_aware_on && phase == 1) {
                                if (this->isNodeHanging(ele, i, j, k))
                                    continue;
                            }

                            const unsigned int slot_idx =
                                offset + kkz * lx * ly + jjy * lx + iix;
                            if (canon_writer_tbl) {
                                const unsigned int canon =
                                    canon_writer_tbl[slot_idx];
                                if (canon != LOOK_UP_TABLE_DEFAULT &&
                                    canon != ele)
                                    continue;
                            }

                            for (unsigned int v = 0; v < dof; v++) {
                                T val_to_write =
                                    dgWVec[v * dgSz +
                                           k * (eOrder + 1) * (eOrder + 1) +
                                           j * (eOrder + 1) + i];
                                uzWVec[v * unSz + offset + kkz * lx * ly +
                                       jjy * lx + iix] = val_to_write;
                                if (uzprobe_fp && xx >= uzprobe_mxlo
                                    && xx <= uzprobe_mxhi
                                    && yy >= uzprobe_mylo
                                    && yy <= uzprobe_myhi
                                    && zz >= uzprobe_mzlo
                                    && zz <= uzprobe_mzhi) {
                                    uint64_t hb = 0;
                                    std::memcpy(&hb, &val_to_write,
                                                sizeof(hb));
                                    std::fprintf(uzprobe_fp,
                                        "same %u %u %u %u %u %d %d %d %g %g %g "
                                        "%u %u %u %u %u %u %lx\n",
                                        blk, (unsigned)bLev,
                                        (unsigned)blkNode.minX(),
                                        (unsigned)blkNode.minY(),
                                        (unsigned)blkNode.minZ(),
                                        iix, jjy, (int)kkz,
                                        xx, yy, zz,
                                        ele, (unsigned)pNodes[ele].getLevel(),
                                        (unsigned)pNodes[ele].minX(),
                                        (unsigned)pNodes[ele].minY(),
                                        (unsigned)pNodes[ele].minZ(),
                                        v,
                                        (unsigned long)hb);
                                }
                            }
                        }
                    }
                }

            } else if (pNodes[ele].getLevel() > bLev) {
                if (same_wins_on && phase != 0) continue;
                // src_aware: both phases run finer; pass 1 filters
                // out hanging-source writes inside the (i,j,k) loop.
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
                            // src_aware: pass 1 skips hanging-source writes.
                            if (src_aware_on && phase == 1) {
                                if (this->isNodeHanging(ele, i, j, k))
                                    continue;
                            }
                            {
                                const unsigned int slot_idx =
                                    offset + kkz * lx * ly + jjy * lx + iix;
                                if (canon_writer_tbl) {
                                    const unsigned int canon =
                                        canon_writer_tbl[slot_idx];
                                    if (canon != LOOK_UP_TABLE_DEFAULT &&
                                        canon != ele)
                                        continue;
                                }
                            }
                            for (unsigned int v = 0; v < dof; v++) {
                                T val_to_write =
                                    dgWVec[v * dgSz +
                                           k * (eOrder + 1) * (eOrder + 1) +
                                           j * (eOrder + 1) + i];
                                uzWVec[v * unSz + offset + kkz * lx * ly +
                                       jjy * lx + iix] = val_to_write;
                                if (uzprobe_fp && xx >= uzprobe_mxlo
                                    && xx <= uzprobe_mxhi
                                    && yy >= uzprobe_mylo
                                    && yy <= uzprobe_myhi
                                    && zz >= uzprobe_mzlo
                                    && zz <= uzprobe_mzhi) {
                                    uint64_t hb = 0;
                                    std::memcpy(&hb, &val_to_write,
                                                sizeof(hb));
                                    std::fprintf(uzprobe_fp,
                                        "finer %u %u %u %u %u %d %d %d %g %g %g "
                                        "%u %u %u %u %u %u %lx\n",
                                        blk, (unsigned)bLev,
                                        (unsigned)blkNode.minX(),
                                        (unsigned)blkNode.minY(),
                                        (unsigned)blkNode.minZ(),
                                        iix, jjy, (int)kkz,
                                        xx, yy, zz,
                                        ele, (unsigned)pNodes[ele].getLevel(),
                                        (unsigned)pNodes[ele].minX(),
                                        (unsigned)pNodes[ele].minY(),
                                        (unsigned)pNodes[ele].minZ(),
                                        v,
                                        (unsigned long)hb);
                                }
                            }
                        }
                    }
                }

            } else {
                if (same_wins_on && phase != 0) continue;
                // src_aware: coarser writes are inherently
                // p2c-interpolated (no direct CG source), so pass 1
                // skips coarser entirely.
                if (src_aware_on && phase == 1) continue;
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
                                &dgWVec[v * dgSz],
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

                                    if (canon_writer_tbl) {
                                        const unsigned int slot_idx =
                                            offset + kkz * lx * ly +
                                            jjy * lx + iix;
                                        const unsigned int canon =
                                            canon_writer_tbl[slot_idx];
                                        if (canon != LOOK_UP_TABLE_DEFAULT &&
                                            canon != ele)
                                            continue;
                                    }

                                    T val_to_write =
                                        p2cI[k * (eOrder + 1) * (eOrder + 1) +
                                             j * (eOrder + 1) + i];
                                    uzWVec[v * unSz + offset + kkz * lx * ly +
                                           jjy * lx + iix] = val_to_write;
                                    if (uzprobe_fp && xx >= uzprobe_mxlo
                                        && xx <= uzprobe_mxhi
                                        && yy >= uzprobe_mylo
                                        && yy <= uzprobe_myhi
                                        && zz >= uzprobe_mzlo
                                        && zz <= uzprobe_mzhi) {
                                        uint64_t hb = 0;
                                        std::memcpy(&hb, &val_to_write,
                                                    sizeof(hb));
                                        std::fprintf(uzprobe_fp,
                                            "coarser %u %u %u %u %u %d %d %d "
                                            "%g %g %g %u %u %u %u %u %u %lx\n",
                                            blk, (unsigned)bLev,
                                            (unsigned)blkNode.minX(),
                                            (unsigned)blkNode.minY(),
                                            (unsigned)blkNode.minZ(),
                                            iix, jjy, (int)kkz,
                                            xx, yy, zz,
                                            ele,
                                            (unsigned)pNodes[ele].getLevel(),
                                            (unsigned)pNodes[ele].minX(),
                                            (unsigned)pNodes[ele].minY(),
                                            (unsigned)pNodes[ele].minZ(),
                                            v,
                                            (unsigned long)hb);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if (uzprobe_fp) {
        std::fclose(uzprobe_fp);
    }
    // bump unconditionally so DENDRO_UNZIP_PROBE_MIN_CALL gates on
    // absolute call index, not just on calls that produced output.
    if (uzprobe_dir) uzprobe_call_id++;

    // probe: dump the FINAL block buffer state keyed by phys position.
    // gate by DENDRO_BBUF_DUMP_DIR. fires once per unzip call_id.
    // limit by DENDRO_BBUF_DUMP_MAX_CALLS.
    {
        static const char* bbdir = DENDRO_PROBE_GETENV("DENDRO_BBUF_DUMP_DIR");
        static const char* bbmin_env =
            DENDRO_PROBE_GETENV("DENDRO_BBUF_DUMP_MIN_CALL");
        static const char* bbmax_env =
            DENDRO_PROBE_GETENV("DENDRO_BBUF_DUMP_MAX_CALL");
        static const int bbmin = bbmin_env ? std::atoi(bbmin_env) : 0;
        static const int bbmax = bbmax_env ? std::atoi(bbmax_env) : 4;
        static int bb_call_id = 0;
        if (bbdir && bb_call_id >= bbmin && bb_call_id < bbmax) {
            const unsigned int npes_skip_unused = 0; (void)npes_skip_unused;
            const ot::TreeNode* pNodes_p = m_uiAllElements.data();
            (void)pNodes_p;
            const unsigned int maxD = m_uiMaxDepth;
            const unsigned int eOrd_p = m_uiElementOrder;
            const unsigned int unSz_p = this->getDegOfFreedomUnZip();
            // dump ALL variables — RHS uses cross-variable stencils, so
            // divergence in any variable may seed a 1-ULP off in another.
            for (unsigned int v = 0; v < dof; v++) {
                T* out_v = out + v * unSz_p;
                char fn[1024];
                std::snprintf(fn, sizeof(fn),
                              "%s/bbuf_call%d_v%u_r%d.txt",
                              bbdir, bb_call_id, v, (int)m_uiActiveRank);
                FILE* fp = std::fopen(fn, "w");
                if (!fp) continue;
                std::fprintf(fp,
                    "# call=%d v=%u rank=%d nblk=%zu\n"
                    "# blk bx by bz blev rlev paddWidth bi bj bk "
                    "phys_x phys_y phys_z hex\n",
                    bb_call_id, v, (int)m_uiActiveRank,
                    m_uiLocalBlockList.size());
                for (size_t b = 0; b < m_uiLocalBlockList.size(); b++) {
                    const auto& blk = m_uiLocalBlockList[b];
                    const ot::TreeNode bn = blk.getBlockNode();
                    const unsigned int rLev = blk.getRegularGridLev();
                    const unsigned int lx = blk.getAllocationSzX();
                    const unsigned int ly = blk.getAllocationSzY();
                    const unsigned int lz = blk.getAllocationSzZ();
                    const unsigned int off = blk.getOffset();
                    const unsigned int PW = blk.get1DPadWidth();
                    const unsigned int h =
                        (unsigned int)1u << (maxD - rLev);
                    for (unsigned int k = 0; k < lz; k++) {
                        for (unsigned int j = 0; j < ly; j++) {
                            for (unsigned int i = 0; i < lx; i++) {
                                const unsigned int idx =
                                    off + (k * ly + j) * lx + i;
                                T val = out_v[idx];
                                uint64_t hb = 0;
                                std::memcpy(&hb, &val, sizeof(hb));
                                // phys position: block origin shifted by
                                // (i - PW) * h * eOrd_p / eOrd_p? in the
                                // block's regularization frame, each cg
                                // node spans h / eOrd_p. Position i in
                                // buffer is (i - PW) * (h / eOrd_p) from
                                // block origin. PhysKey uses eOrd_p as
                                // scale.
                                const long long px =
                                    (long long)bn.getX() * eOrd_p
                                    + ((long long)i - (long long)PW) * h;
                                const long long py =
                                    (long long)bn.getY() * eOrd_p
                                    + ((long long)j - (long long)PW) * h;
                                const long long pz =
                                    (long long)bn.getZ() * eOrd_p
                                    + ((long long)k - (long long)PW) * h;
                                std::fprintf(fp,
                                    "%zu %u %u %u %u %u %u "
                                    "%u %u %u %lld %lld %lld %lx\n",
                                    b, bn.getX(), bn.getY(), bn.getZ(),
                                    bn.getLevel(), rLev, PW,
                                    i, j, k,
                                    px, py, pz, (unsigned long)hb);
                            }
                        }
                    }
                }
                std::fclose(fp);
            }
        }
        // always increment so MIN_CALL/MAX_CALL gate on absolute call
        // index regardless of whether files were written this call.
        if (bbdir) bb_call_id++;
    }
}

template <typename T>
void Mesh::unzip(const T* in, T* out, unsigned int dof) {
    if ((!m_uiIsActive) || (m_uiLocalBlockList.empty())) return;

    // std::vector<unsigned int > blkIDs;
    // blkIDs.resize(m_uiLocalBlockList.size());

    // for(unsigned int i=0; i< m_uiLocalBlockList.size(); i++)
    //     blkIDs[i] = i ;
    // unzip all the blocks.
    // this->unzip(in,out,blkIDs.data(),blkIDs.size(),dof);
    this->unzip_scatter(in, out, dof);
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
            const unsigned int ei = (pNodes[elem].getX() - blkNode.getX()) >>
                                    (m_uiMaxDepth - regLevel);
            const unsigned int ej = (pNodes[elem].getY() - blkNode.getY()) >>
                                    (m_uiMaxDepth - regLevel);
            const unsigned int ek = (pNodes[elem].getZ() - blkNode.getZ()) >>
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

            // Use the block's regular grid level. Older code read the
            // first element's level, which is correct for SFC blocks
            // (all elements are at regLev by construction) but wrong
            // for non-SFC blocks from repartitioned meshes, where
            // m_elementIndices[0] can be at any level present in the
            // block.
            const unsigned int bLev = blkList[blk].getRegularGridLev();

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
