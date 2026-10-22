/**
 * @file enuts_sync.tcc
 * @brief ExplicitNUTS block time sync (sync plans, parent-to-child store, sync_blk_timestep).
 * (code moved from enuts.h, which includes this file)
 */

namespace ts {

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::build_sync_plan(unsigned int blk,
                                           const std::vector<unsigned int>& eid,
                                           SyncPlan& pl) {
    ot::Mesh* pMesh = (ot::Mesh*)m_uiAppCtx->get_mesh();
    const ot::TreeNode* pNodes = pMesh->getAllElements().data();
    const ot::Block* blkList   = pMesh->getLocalBlockList().data();
    const ot::TreeNode blkNode = blkList[blk].getBlockNode();
    const unsigned int PW      = blkList[blk].get1DPadWidth();
    const unsigned int eOrder  = pMesh->getElementOrder();
    const unsigned int lx      = blkList[blk].getAllocationSzX();
    const unsigned int ly      = blkList[blk].getAllocationSzY();
    const unsigned int lz      = blkList[blk].getAllocationSzZ();
    const unsigned int bLev = pNodes[blkList[blk].firstElementID()].getLevel();

    const double hx   = (1u << (m_uiMaxDepth - bLev)) / (double)eOrder;
    const double xmin = blkNode.minX() - PW * hx;
    const double xmax = blkNode.maxX() + PW * hx;
    const double ymin = blkNode.minY() - PW * hx;
    const double ymax = blkNode.maxY() + PW * hx;
    const double zmin = blkNode.minZ() - PW * hx;
    const double zmax = blkNode.maxZ() + PW * hx;
    const double d_compar_tol = 1e-6;
    const unsigned int n1     = eOrder + 1;

    pl.ents.clear();
    pl.src.clear();
    pl.dst.clear();

    // one node's target index in the block, or false when it lies outside
    auto clampIdx = [&](double c, double cmin, double cmax, double invhh,
                        unsigned int& idx) -> bool {
        if (fabs(c - cmin) < d_compar_tol) c = cmin;
        if (fabs(c - cmax) < d_compar_tol) c = cmax;
        if (c < cmin || c > cmax) return false;
        idx = std::round((c - cmin) * invhh);
        return true;
    };

    std::vector<ot::TreeNode> childOct;
    childOct.reserve(NUM_CHILDREN);

    for (unsigned int m = 0; m < eid.size(); m++) {
        const unsigned int ele = eid[m];
        const unsigned int lev = pNodes[ele].getLevel();

        if (lev == bLev || lev > bLev) {
            const bool fine = lev > bLev;
            const double hh = (1u << (m_uiMaxDepth - lev)) / (double)eOrder;
            const double invhh = fine ? 1.0 / (2 * hh) : 1.0 / hh;
            const unsigned int cb   = fine ? ((eOrder % 2 == 0) ? 0 : 1) : 0;
            const unsigned int step = fine ? 2 : 1;
            const unsigned int b0   = pl.src.size();
            for (unsigned int k = cb; k < n1; k += step) {
                unsigned int kkz;
                if (!clampIdx(pNodes[ele].minZ() + k * hh, zmin, zmax, invhh, kkz)) continue;
                for (unsigned int j = cb; j < n1; j += step) {
                    unsigned int jjy;
                    if (!clampIdx(pNodes[ele].minY() + j * hh, ymin, ymax, invhh, jjy)) continue;
                    for (unsigned int i = cb; i < n1; i += step) {
                        unsigned int iix;
                        if (!clampIdx(pNodes[ele].minX() + i * hh, xmin, xmax, invhh, iix)) continue;
                        pl.src.push_back(k * n1 * n1 + j * n1 + i);
                        pl.dst.push_back(kkz * lx * ly + jjy * lx + iix);
                    }
                }
            }
            if (pl.src.size() > b0)
                pl.ents.push_back({ele, -1, b0, (unsigned int)pl.src.size(), 0u});
        } else {
            childOct.clear();
            pNodes[ele].addChildren(childOct);
            for (unsigned int child = 0; child < NUM_CHILDREN; child++) {
                if ((childOct[child].maxX() < xmin || childOct[child].minX() >= xmax) ||
                    (childOct[child].maxY() < ymin || childOct[child].minY() >= ymax) ||
                    (childOct[child].maxZ() < zmin || childOct[child].minZ() >= zmax))
                    continue;
                const double hh = (1u << (m_uiMaxDepth - childOct[child].getLevel())) / (double)eOrder;
                const double invhh = 1.0 / hh;
                const unsigned int b0 = pl.src.size();
                for (unsigned int k = 0; k < n1; k++) {
                    unsigned int kkz;
                    if (!clampIdx(childOct[child].minZ() + k * hh, zmin, zmax, invhh, kkz)) continue;
                    for (unsigned int j = 0; j < n1; j++) {
                        unsigned int jjy;
                        if (!clampIdx(childOct[child].minY() + j * hh, ymin, ymax, invhh, jjy)) continue;
                        for (unsigned int i = 0; i < n1; i++) {
                            unsigned int iix;
                            if (!clampIdx(childOct[child].minX() + i * hh, xmin, xmax, invhh, iix)) continue;
                            pl.src.push_back(k * n1 * n1 + j * n1 + i);
                            pl.dst.push_back(kkz * lx * ly + jjy * lx + iix);
                        }
                    }
                }
                if (pl.src.size() > b0)
                    pl.ents.push_back(
                        {ele, (int)childOct[child].getMortonIndex(), b0,
                         (unsigned int)pl.src.size(),
                         p2c_index(ele, childOct[child].getMortonIndex())});
            }
        }
    }
    pl.built = true;
}

template <typename T, typename Ctx>
unsigned int ExplicitNUTS<T, Ctx>::p2c_index(unsigned int ele,
                                             unsigned int cnum) {
    const ot::Mesh* pMesh = m_uiAppCtx->get_mesh();
    const size_t n        = (size_t)pMesh->getAllElements().size() * NUM_CHILDREN;
    if (m_uiP2cIdx.size() != n) {
        m_uiP2cIdx.assign(n, -1);
        m_uiP2cPassOf.clear();
        m_uiP2cSlotOf.clear();
        m_uiP2cTagOf.clear();
        m_uiP2cStepOf.clear();
    }
    int& i = m_uiP2cIdx[(size_t)ele * NUM_CHILDREN + cnum];
    if (i < 0) {
        i = (int)m_uiP2cPassOf.size();
        m_uiP2cPassOf.push_back(0);
        m_uiP2cSlotOf.push_back(0);
        m_uiP2cTagOf.push_back(SyncTag());
    }
    return (unsigned int)i;
}

template <typename T, typename Ctx>
T* ExplicitNUTS<T, Ctx>::p2c_slot(unsigned int idx, const SyncTag& key,
                                  size_t stride, bool& hit) {
    // [ENUTS_OMP] worker threads keep their own pass store (see SyncThr)
    SyncThr* const thr_ =
        ts::EnutsProf::tid() ? &m_uiSyncThr[ts::EnutsProf::tid()] : nullptr;
    unsigned long long& m_uiP2cPoolPass = thr_ ? thr_->p2cPoolPass : this->m_uiP2cPoolPass;
    size_t& m_uiP2cUsed                 = thr_ ? thr_->p2cUsed : this->m_uiP2cUsed;
    std::vector<unsigned long long>& m_uiP2cPassOf = thr_ ? thr_->p2cPassOf : this->m_uiP2cPassOf;
    std::vector<unsigned int>& m_uiP2cSlotOf       = thr_ ? thr_->p2cSlotOf : this->m_uiP2cSlotOf;
    std::vector<SyncTag>& m_uiP2cTagOf             = thr_ ? thr_->p2cTagOf : this->m_uiP2cTagOf;
    std::vector<T>& m_uiP2cPool                    = thr_ ? thr_->p2cPool : this->m_uiP2cPool;
    hit = false;
    if (m_uiP2cSharedOn) {
        // the pre-pass wrote every child of this pass into its reader blocks
        if (this->m_uiP2cPassOf[idx] == m_uiSyncPass && this->m_uiP2cTagOf[idx] == key) {
            hit = true;
            return nullptr;
        }
        if (ts::EnutsProf::on()) ts::EnutsProf::get().t[ts::EnutsProf::CN_P2C_OVF] += 1;
        return nullptr;  // not listed by the pre-pass: computed, not stored
    }
    if (m_uiP2cPoolPass != m_uiSyncPass) {
        m_uiP2cPoolPass = m_uiSyncPass;
        m_uiP2cUsed     = 0;
    }
    if (m_uiP2cPassOf[idx] == m_uiSyncPass) {
        // same (element, child) seen in this pass: same data unless the
        // source differs (its tag)
        if (m_uiP2cTagOf[idx] == key)
            hit = true;
        else
            m_uiP2cTagOf[idx] = key;
        return m_uiP2cPool.data() + (size_t)m_uiP2cSlotOf[idx] * stride;
    }
    const size_t cap =
        std::max<size_t>(64, ts::EnutsProf::p2cStoreBytes() / (stride * sizeof(T)));
    if (m_uiP2cUsed >= cap) {
        if (ts::EnutsProf::on()) ts::EnutsProf::get().t[ts::EnutsProf::CN_P2C_OVF] += 1;
        return nullptr;
    }
    if ((m_uiP2cUsed + 1) * stride > m_uiP2cPool.size())
        m_uiP2cPool.resize(std::min(
            cap * stride,
            std::max<size_t>(2 * m_uiP2cPool.size(), 64 * stride)));
    m_uiP2cSlotOf[idx] = (unsigned int)m_uiP2cUsed++;
    m_uiP2cPassOf[idx] = m_uiSyncPass;
    m_uiP2cTagOf[idx]  = key;
    return m_uiP2cPool.data() + (size_t)m_uiP2cSlotOf[idx] * stride;
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::sync_lazy_build(bool ownEid) {
    ot::Mesh* pMesh          = (ot::Mesh*)m_uiAppCtx->get_mesh();
    const ot::Block* blkList = pMesh->getLocalBlockList().data();
    const unsigned int nPe   = pMesh->getNumNodesPerElement();
    if (m_uiBlkDepEid.size() != pMesh->getLocalBlockList().size()) {
        m_uiBlkDepEid.assign(pMesh->getLocalBlockList().size(), {});
        for (unsigned int b = 0; b < m_uiBlkDepEid.size(); b++)
            computeBlockUnzipDepElements(pMesh, b, m_uiBlkDepEid[b]);
    }
    if (ownEid && m_uiBlkOwnEid.size() != m_uiBlkDepEid.size()) {
        const unsigned int* e2n_dg = pMesh->getE2NMapping_DG().data();
        m_uiBlkOwnEid.assign(m_uiBlkDepEid.size(), {});
        std::vector<unsigned char> seen(pMesh->getAllElements().size(), 0);
        for (unsigned int b = 0; b < m_uiBlkOwnEid.size(); b++) {
            std::vector<unsigned int>& oe = m_uiBlkOwnEid[b];
            oe = m_uiBlkDepEid[b];
            for (const unsigned int e : oe) seen[e] = 1;
            for (const unsigned int e : blkList[b]) seen[e] = 1;
            const size_t nd = oe.size();
            auto addOwners = [&](unsigned int e) {
                for (unsigned int q = 0; q < nPe; q++) {
                    const unsigned int o = e2n_dg[e * nPe + q] / nPe;
                    if (!seen[o]) {
                        seen[o] = 1;
                        oe.push_back(o);
                    }
                }
            };
            for (size_t m = 0; m < nd; m++) addOwners(oe[m]);
            for (const unsigned int e : blkList[b]) addOwners(e);
            for (const unsigned int e : oe) seen[e] = 0;
            for (const unsigned int e : blkList[b]) seen[e] = 0;
        }
    }
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::sync_prebuild(bool ownPlan, bool ownEid,
                                         bool syncFast) {
    ot::Mesh* pMesh         = (ot::Mesh*)m_uiAppCtx->get_mesh();
    const unsigned int nBlk = pMesh->getLocalBlockList().size();
    sync_lazy_build(ownEid);
    // building a plan appends to tables all blocks share (p2c store indices,
    // rows of the coarser dependents): all blocks here, once per mesh, so a
    // block call inside the parallel region only reads them
    if (ownPlan) {
        if (m_uiOwnPlan.size() != nBlk) m_uiOwnPlan.assign(nBlk, OwnPlan());
        for (unsigned int b = 0; b < nBlk; b++)
            if (!m_uiOwnPlan[b].built)
                build_own_plan(b, m_uiBlkDepEid[b], m_uiOwnPlan[b]);
    } else if (syncFast) {
        if (m_uiSyncPlan.size() != nBlk) m_uiSyncPlan.assign(nBlk, SyncPlan());
        for (unsigned int b = 0; b < nBlk; b++)
            if (!m_uiSyncPlan[b].built)
                build_sync_plan(b, m_uiBlkDepEid[b], m_uiSyncPlan[b]);
    }
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::free_sync_threads() {
    for (SyncThr& st : m_uiSyncThr) delete st.op;
    m_uiSyncThr.clear();
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::sync_thread_setup(bool cgScratch) {
#ifdef DENDRO_ENUTS_HAS_OMP
    ot::Mesh* pMesh = (ot::Mesh*)m_uiAppCtx->get_mesh();
    const size_t nt = (size_t)omp_get_max_threads();
    const size_t dof = m_uiEVar.get_dof();
    const size_t nPe = pMesh->getNumNodesPerElement();
    const size_t dgN = dof * pMesh->getAllElements().size() * nPe;
    const size_t cgN = dof * pMesh->getDegOfFreedom();
    if (m_uiSyncThr.size() != nt) {
        free_sync_threads();
        m_uiSyncThr.resize(nt);
    }
    // slot 0 stays empty: thread 0 uses the shared members
    for (size_t t = 1; t < nt; t++) {
        SyncThr& st = m_uiSyncThr[t];
        if (!st.op) st.op = new ENUTSOp(m_uiType);
        // zeroed like the shared scratch they mirror
        if (!st.dgW) st.dgW.reset(new T[dgN]());
        if (cgScratch && !st.cgW) st.cgW.reset(new T[cgN]());
        // mirrors of the shared per-pass caches, sized like them
        if (st.p2cPassOf.size() != m_uiP2cPassOf.size()) {
            st.p2cPassOf.assign(m_uiP2cPassOf.size(), 0);
            st.p2cSlotOf.assign(m_uiP2cPassOf.size(), 0);
            st.p2cTagOf.assign(m_uiP2cPassOf.size(), SyncTag());
            st.p2cUsed     = 0;
            st.p2cPoolPass = 0;
        }
        if (st.ownConv.size() != m_uiOwnConv.size()) {
            st.ownConv.assign(m_uiOwnConv.size(), OwnConv());
            st.ownConvPass.assign(m_uiOwnConvPass.size(), 0);
            st.ownRowBufRow = ~0u;
        }
        if (st.ownScr.size() != m_uiOwnScr.size())
            st.ownScr.assign(m_uiOwnScr.size(), 0);
    }
#endif
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::sync_blk_timestep(unsigned int blk,
                                             unsigned int rk_s,
                                             bool isIGTSync) {
#ifdef __PROFILE_ENUTS__
    m_uiPt[ENUTSPROFILE::BLK_SYNC].start();
#endif

    ot::Mesh* pMesh = (ot::Mesh*)m_uiAppCtx->get_mesh();
    if ((!(pMesh->isActive())) || m_uiBVec[blk]._vec[stage_slot(rk_s)].isSynced()) return;
    const bool sprof = ts::EnutsProf::on();
    double tcorrCf = 0, tcorrFc = 0, tsy0 = 0;
    if (sprof) tsy0 = ts::EnutsProf::now();
    if (sprof) ts::EnutsProf::get().t[ts::EnutsProf::CN_CALLS] += 1;

    const ot::TreeNode* pNodes  = pMesh->getAllElements().data();
    const ot::Block* blkList    = pMesh->getLocalBlockList().data();

    const unsigned int regLevel = blkList[blk].getRegularGridLev();
    const ot::TreeNode blkNode  = blkList[blk].getBlockNode();
    const unsigned int PW       = blkList[blk].get1DPadWidth();

    const unsigned int eOrder   = pMesh->getElementOrder();
    const unsigned int nPe      = pMesh->getNumNodesPerElement();

    const unsigned int* etVec   = m_uiEleTime.data();
    const unsigned int* dtELev  = m_uiEleDT.data();

    MPI_Comm comm               = pMesh->getMPICommunicator();

    const unsigned int lx       = blkList[blk].getAllocationSzX();
    const unsigned int ly       = blkList[blk].getAllocationSzY();
    const unsigned int lz       = blkList[blk].getAllocationSzZ();
    const DendroIntL offset   = blkList[blk].getOffset();

    const unsigned int dgSz =
        pMesh->getAllElements().size() * pMesh->getNumNodesPerElement();
    const unsigned int cgSz = pMesh->getDegOfFreedom();
    const unsigned int unSz = pMesh->getDegOfFreedomUnZip();

    const unsigned int* e2n = pMesh->getE2NMapping().data();
    const unsigned int* e2e = pMesh->getE2EMapping().data();

    const unsigned int bLev =
        pNodes[blkList[blk].firstElementID()].getLevel();
    const unsigned int bTime = etVec[blkList[blk].firstElementID()];
    const unsigned int bDT   = dtELev[blkList[blk].firstElementID()];

    T dt;
    unsigned int tl  = 0;
    unsigned int dtl = 0;  // dt time level.
    unsigned int lookUp;
    const unsigned int cSz[3] = {eOrder + 1, eOrder + 1, eOrder + 1};

    unsigned int fchild[4];
    unsigned int echild[2];

    const unsigned int dof = m_uiEVar.get_dof();
    const bool ownSync  = rk_s >= 1 && !isIGTSync;
    // ownSync runs the plan (below, once the dependents are known), or the
    // per-element form where the plan is unavailable
    // the per-element sync takes the bit-identical shortcuts (no same-time DG
    // copy, per-stage correction reuse across blocks, padding-fill plan,
    // per-pass store and shared sweeps of the parent-to-child interpolations)
    // except in the per-element owner-sync form
    const bool syncFast = !ownSync;
    const bool ifxEq  = (ts::EnutsProf::ifaceFix() & 1) && rk_s >= 1 && !isIGTSync;
    const bool ifxAll = (ts::EnutsProf::ifaceFix() & 2) && rk_s >= 1 && !isIGTSync;
    // [ENUTS_OMP edit 1/5 begin] worker threads (tid >= 1) of the threaded sync
    // loop get their own scratch: the locals below shadow the shared members
    // for the rest of this function (p2c_slot and own_sync_fill do the same).
    // thr_ == nullptr = serial / thread 0.
    SyncThr* const thr_ =
        ts::EnutsProf::tid() ? &m_uiSyncThr[ts::EnutsProf::tid()] : nullptr;
    ENUTSOp* const m_uiECOp           = thr_ ? thr_->op : this->m_uiECOp;
    std::vector<T>& m_uiSyncCVec      = thr_ ? thr_->cVec : this->m_uiSyncCVec;
    std::vector<SyncTag>& m_uiSyncTag = thr_ ? thr_->tag : this->m_uiSyncTag;
    std::vector<T>& m_uiOwnSweep      = thr_ ? thr_->ownSweep : this->m_uiOwnSweep;
    std::vector<unsigned char>& m_uiOwnSweepOk = thr_ ? thr_->ownSweepOk : this->m_uiOwnSweepOk;
    // [ENUTS_OMP edit 1/5 end]
    std::vector<T> cVecLocal;
    std::vector<T>& cVec = syncFast ? m_uiSyncCVec : cVecLocal;
    if (cVec.size() < (size_t)dof * nPe * (m_uiNumStages + 1))
        cVec.resize(dof * nPe * (m_uiNumStages + 1));
    if (syncFast && m_uiSyncTag.size() != pMesh->getAllElements().size())
        m_uiSyncTag.assign(pMesh->getAllElements().size(), SyncTag());

    T* cVin[(m_uiNumStages + 1) * dof];   // correction vec. pointer in,
    T* cVout[(m_uiNumStages + 1) * dof];  // correction vec. pointer out,

    // [ENUTS_OMP edit 2/5] per-thread stand-in for the shared work vector
    T* dgWVec  = thr_ ? thr_->dgW.get() : m_uiDGTmp0.get_vec_ptr();
    // T * cgWVec = m_uiEVecTmp[0].get_vec_ptr();
    T* uzWVec  = m_uiEvarUzip.get_vec_ptr();

    T* m_uiVec = (T*)m_uiBVec[blk]._vec[stage_slot(rk_s)].data();

    T* dgStages[m_uiNumStages];
    for (unsigned int i = 0; i < m_uiNumStages; i++)
        dgStages[i] = m_uiStVec[i].get_vec_ptr();

    T* dgEVar = m_uiEvarDG.get_vec_ptr();

    // [ENUTS_OMP edit 3/5] the m_uiBlkDepEid build (here) and the
    // m_uiBlkOwnEid build (below) moved verbatim to sync_lazy_build(); the
    // threaded sync loop runs sync_prebuild() before the parallel region, so
    // in a worker these are read-only size checks. any new table or plan
    // built on first use belongs in sync_prebuild().
    sync_lazy_build(false);
    const std::vector<unsigned int>& eid = m_uiBlkDepEid[blk];
    if (sprof) ts::EnutsProf::get().t[ts::EnutsProf::CN_EID] += eid.size();
    if (ownSync && m_uiSyncPassOpen && prepare_owner_sync(pMesh)) {
        own_sync_fill(blk, rk_s, eid);
        if (sprof) ts::EnutsProf::get().t[ts::EnutsProf::SY_BLK] += ts::EnutsProf::now() - tsy0;
#ifdef __PROFILE_ENUTS__
        m_uiPt[ENUTSPROFILE::BLK_SYNC].stop();
#endif
        return;
    }
    if (ownSync) sync_lazy_build(true);  // [ENUTS_OMP edit 3/5, second part]
    if (!thr_ && m_uiEvarDG.get_vec_ptr() == nullptr) ensure_evar_dg();  // serial path
    // elements whose stage is brought to this block's time frame
    const std::vector<unsigned int>& cel = ownSync ? m_uiBlkOwnEid[blk] : eid;

    if (rk_s == 0) {  // Sync operation for U(t,.)

        for (unsigned int m = 0; m < eid.size(); m++) {
            const unsigned int elem = eid[m];
            tl  = etVec[elem];   // this is the current time the block is at, in
                                 // the block loop time.
            dtl = dtELev[elem];  // this is time step size level used to get to
                                 // this time point.
            // You can use the element/block level to compute the future time
            // step size level.

            T t_blk = m_uiTimeInfo._m_uiT + bTime * m_uiTimeInfo._m_uiTh;
            T t_ele = m_uiTimeInfo._m_uiT + tl * m_uiTimeInfo._m_uiTh;

            if (tl == bTime) {
                // both elements are in the same time.
                dt = 0;
                if (!syncFast) {
#pragma unroll
                    for (unsigned int v = 0; v < dof; v++)
                        std::memcpy(dgWVec + v * dgSz + elem * nPe,
                                    dgEVar + v * dgSz + elem * nPe,
                                    sizeof(T) * nPe);
                }

            } else if (tl > bTime) {
                assert(dtl >= bDT);
                // elem is in the future.
                const T dt_c = dtl * (m_uiTimeInfo._m_uiTh);
                const T dt_f = bDT * (m_uiTimeInfo._m_uiTh);
                dt           = (tl - bTime) * (m_uiTimeInfo._m_uiTh);

                for (unsigned int v = 0; v < dof; v++) {
                    for (unsigned int s = 0; s < m_uiNumStages; s++) {
                        cVin[v * (m_uiNumStages + 1) + s] =
                            dgStages[s] + v * dgSz + elem * nPe;
                        cVout[v * (m_uiNumStages + 1) + s] =
                            cVec.data() + v * (m_uiNumStages + 1) * nPe +
                            s * nPe;
                    }

                    cVin[v * (m_uiNumStages + 1) + m_uiNumStages] =
                        dgEVar + v * dgSz + elem * nPe;
                    cVout[v * (m_uiNumStages + 1) + m_uiNumStages] =
                        cVec.data() + v * (m_uiNumStages + 1) * nPe +
                        m_uiNumStages * nPe;
                }

#ifdef __PROFILE_ENUTS__
                m_uiPt[ENUTSPROFILE::NUTS_CORRECTION].start();
#endif

                { const double tc0_ = sprof ? ts::EnutsProf::now() : 0.0;
 m_uiECOp->coarser_finer_ut_correction(cVout, (const T**)cVin,
                                                      cSz, dt_c, dt_f, dt, dof);
 if (sprof) tcorrCf += ts::EnutsProf::now() - tc0_; }

#ifdef __PROFILE_ENUTS__
                m_uiPt[ENUTSPROFILE::NUTS_CORRECTION].stop();
#endif

#pragma unroll
                for (unsigned int v = 0; v < dof; v++)
                    std::memcpy(dgWVec + v * dgSz + elem * nPe,
                                cVout[v * (m_uiNumStages + 1) + m_uiNumStages],
                                sizeof(T) * nPe);

            } else {
                // elem is in the past.
                assert(tl < bTime);
                assert(dtl <= bDT);
                // elem is in the future.
                const T dt_c = bDT * (m_uiTimeInfo._m_uiTh);
                const T dt_f = dtl * (m_uiTimeInfo._m_uiTh);

                dt           = (bTime - tl) * (m_uiTimeInfo._m_uiTh);

                for (unsigned int v = 0; v < dof; v++) {
                    for (unsigned int s = 0; s < m_uiNumStages; s++) {
                        cVin[v * (m_uiNumStages + 1) + s] =
                            dgStages[s] + v * dgSz + elem * nPe;
                        cVout[v * (m_uiNumStages + 1) + s] =
                            cVec.data() + v * (m_uiNumStages + 1) * nPe +
                            s * nPe;
                    }

                    cVin[v * (m_uiNumStages + 1) + m_uiNumStages] =
                        dgEVar + v * dgSz + elem * nPe;
                    cVout[v * (m_uiNumStages + 1) + m_uiNumStages] =
                        cVec.data() + v * (m_uiNumStages + 1) * nPe +
                        m_uiNumStages * nPe;
                }

#ifdef __PROFILE_ENUTS__
                m_uiPt[ENUTSPROFILE::NUTS_CORRECTION].start();
#endif

                { const double tc0_ = sprof ? ts::EnutsProf::now() : 0.0;
 m_uiECOp->finer_coarser_ut_correction(cVout, (const T**)cVin,
                                                      cSz, dt_c, dt_f, dt, dof);
 if (sprof) tcorrFc += ts::EnutsProf::now() - tc0_; }

#ifdef __PROFILE_ENUTS__
                m_uiPt[ENUTSPROFILE::NUTS_CORRECTION].stop();
#endif

#pragma unroll
                for (unsigned int v = 0; v < dof; v++)
                    std::memcpy(dgWVec + v * dgSz + elem * nPe,
                                cVout[v * (m_uiNumStages + 1) + m_uiNumStages],
                                sizeof(T) * nPe);
            }
        }

    } else {
        assert(rk_s >= 1 && rk_s <= m_uiNumStages);
        assert(PW > 0);

        // apply time interpolation correction operators.
        for (unsigned int m = 0; m < cel.size(); m++) {
            const unsigned int elem = cel[m];
            tl  = etVec[elem];   // this is the current time the block is at, in
                                 // the block loop time.
            dtl = dtELev[elem];  // this is time step size level used to get to
                                 // this time point.
            // You can use the element/block level to compute the future time
            // step size level.

            T t_blk = m_uiTimeInfo._m_uiT + bTime * m_uiTimeInfo._m_uiTh;
            T t_ele = m_uiTimeInfo._m_uiT + tl * m_uiTimeInfo._m_uiTh;

            const T t_blk_prev = m_uiTimeInfo._m_uiT +
                                 bTime * m_uiTimeInfo._m_uiTh -
                                 bDT * m_uiTimeInfo._m_uiTh;
            const T t_ele_prev = m_uiTimeInfo._m_uiT +
                                 tl * m_uiTimeInfo._m_uiTh -
                                 dtl * m_uiTimeInfo._m_uiTh;

            if (isIGTSync) {
                // update where these stages are computed.
                t_blk = t_blk_prev;
                t_ele = t_ele_prev;

                if (fabs(t_ele - t_blk) <= __LTS_BLK_SYNC_TIME_COMP_TOL__) {
// no interpolation is needed.
#pragma unroll
                    for (unsigned int v = 0; v < dof; v++)
                        std::memcpy(dgWVec + v * dgSz + elem * nPe,
                                    dgStages[rk_s - 1] + v * dgSz + elem * nPe,
                                    sizeof(T) * nPe);

                } else if ((t_ele - t_blk) > __LTS_BLK_SYNC_TIME_COMP_TOL__) {
                    // elem is in future.
                    T dt_c = dtl * (m_uiTimeInfo._m_uiTh);
                    T dt_f = bDT * (m_uiTimeInfo._m_uiTh);

                    if (dt_c < dt_f) std::swap(dt_c, dt_f);

                    dt = (t_ele - t_blk);

                    for (unsigned int v = 0; v < dof; v++)
                        for (unsigned int s = 1; s <= rk_s; s++) {
                            cVin[v * rk_s + (s - 1)] =
                                dgStages[s - 1] + v * dgSz + elem * nPe;
                            cVout[v * rk_s + (s - 1)] =
                                cVec.data() + v * rk_s * nPe + (s - 1) * nPe;
                        }

#ifdef __PROFILE_ENUTS__
                    m_uiPt[ENUTSPROFILE::NUTS_CORRECTION].start();
#endif

                    { const double tc0_ = sprof ? ts::EnutsProf::now() : 0.0;
 m_uiECOp->Ccf(cVout, (const T**)cVin, cSz, rk_s, dt_c, dt_f,
                                  dt, dof);
 if (sprof) tcorrCf += ts::EnutsProf::now() - tc0_; }

#ifdef __PROFILE_ENUTS__
                    m_uiPt[ENUTSPROFILE::NUTS_CORRECTION].stop();
#endif

#pragma unroll
                    for (unsigned int v = 0; v < dof; v++)
                        std::memcpy(dgWVec + v * dgSz + elem * nPe,
                                    cVout[v * rk_s + (rk_s - 1)],
                                    sizeof(T) * nPe);
                } else if ((t_blk - t_ele) > __LTS_BLK_SYNC_TIME_COMP_TOL__) {
                    // blk is in future
                    T dt_c = (bDT) * (m_uiTimeInfo._m_uiTh);
                    T dt_f = (dtl) * (m_uiTimeInfo._m_uiTh);

                    if (dt_c < dt_f) std::swap(dt_c, dt_f);

                    dt = (t_blk - t_ele);

                    for (unsigned int v = 0; v < dof; v++)
                        for (unsigned int s = 1; s <= rk_s; s++) {
                            cVin[v * rk_s + (s - 1)] =
                                dgStages[s - 1] + v * dgSz + elem * nPe;
                            cVout[v * rk_s + (s - 1)] =
                                cVec.data() + v * rk_s * nPe + (s - 1) * nPe;
                        }

#ifdef __PROFILE_ENUTS__
                    m_uiPt[ENUTSPROFILE::NUTS_CORRECTION].start();
#endif

                    { const double tc0_ = sprof ? ts::EnutsProf::now() : 0.0;
 m_uiECOp->Cfc(cVout, (const T**)cVin, cSz, rk_s, dt_c, dt_f,
                                  dt, dof);
 if (sprof) tcorrFc += ts::EnutsProf::now() - tc0_; }

#ifdef __PROFILE_ENUTS__
                    m_uiPt[ENUTSPROFILE::NUTS_CORRECTION].stop();
#endif

#pragma unroll
                    for (unsigned int v = 0; v < dof; v++)
                        std::memcpy(dgWVec + v * dgSz + elem * nPe,
                                    cVout[v * rk_s + (rk_s - 1)],
                                    sizeof(T) * nPe);
                }

            } else {
                if (bTime == tl && !(ifxEq && dtl != bDT)) {
                    // fast: the padding fill below reads the stage vector directly
                    if (!syncFast) {
#pragma unroll
                        for (unsigned int v = 0; v < dof; v++)
                            std::memcpy(dgWVec + v * dgSz + elem * nPe,
                                        dgStages[rk_s - 1] + v * dgSz + elem * nPe,
                                        sizeof(T) * nPe);
                    }

                } else {
                    // tl > bTime: elem is in the future (Ccf), else blk is (Cfc)
                    // eqT (ifxEq only): same time but a different step, so the
                    // element's stage k_s was built with its own dt; convert it
                    // with a zero time shift (Ccf from a coarser element)
                    const bool eqT    = (tl == bTime);
                    const bool eleFut = eqT ? (dtl > bDT) : (tl > bTime);
                    assert(eqT || eleFut || tl < bTime);
                    // ifxAll: an element ahead of the block has finished its
                    // step, so all of its stages enter the time shift
                    const unsigned int nsIn =
                        (ifxAll && eleFut && !eqT) ? m_uiNumStages : rk_s;
                    T dt_c = (eleFut ? dtl : bDT) * (m_uiTimeInfo._m_uiTh);
                    T dt_f = (eleFut ? bDT : dtl) * (m_uiTimeInfo._m_uiTh);

                    if (dt_c < dt_f) std::swap(dt_c, dt_f);

                    dt = eqT ? 0 : (eleFut ? (t_ele - t_blk) : (t_blk - t_ele));
                    if (ts::EnutsProf::ifaceShift() && eleFut && !eqT)
                        dt = (T)(bTime + dtl - tl) * m_uiTimeInfo._m_uiTh;

                    if (syncFast && m_uiSyncPassOpen &&
                        m_uiSyncTag[elem].pass == m_uiSyncPass &&
                        m_uiSyncTag[elem].rk == rk_s &&
                        m_uiSyncTag[elem].fut == (unsigned char)eleFut &&
                        m_uiSyncTag[elem].dt_c == dt_c &&
                        m_uiSyncTag[elem].dt_f == dt_f &&
                        m_uiSyncTag[elem].dt == dt) {
                        // another block of this stage pass already left the
                        // same correction in dgWVec
                        if (sprof) ts::EnutsProf::get().t[ts::EnutsProf::CN_HIT] += 1;
                    } else {
                        for (unsigned int v = 0; v < dof; v++)
                            for (unsigned int s = 1; s <= nsIn; s++) {
                                cVin[v * nsIn + (s - 1)] =
                                    dgStages[s - 1] + v * dgSz + elem * nPe;
                                cVout[v * nsIn + (s - 1)] =
                                    cVec.data() + v * nsIn * nPe + (s - 1) * nPe;
                            }

                        if (sprof) ts::EnutsProf::get().t[ts::EnutsProf::CN_CORR] += 1;
                        {
                            const double tc0_ = sprof ? ts::EnutsProf::now() : 0.0;
                            if (m_uiStGOn && (eqT || nsIn == m_uiNumStages)) {
                                // nonlinear form; g of an element's previous
                                // step where its k3, k4 do not exist yet
                                T row[5], m5[5];
                                const T* in5[5];
                                const bool useG = m_uiECOp->stageRowNL(
                                    eleFut, row, rk_s, dt_c, dt_f, dt, nsIn);
                                for (unsigned int k = 0; k < nsIn; k++) m5[k] = row[k];
                                if (useG) m5[nsIn] = row[4];
                                for (unsigned int v = 0; v < dof; v++) {
                                    for (unsigned int k = 0; k < nsIn; k++)
                                        in5[k] = cVin[v * nsIn + k];
                                    if (useG)
                                        in5[nsIn] = m_uiStG.get_vec_ptr() +
                                                    (size_t)v * dgSz + (size_t)elem * nPe;
                                    m_uiECOp->applyRow(cVout[v * nsIn + (rk_s - 1)], in5,
                                                       m5, nsIn + (useG ? 1 : 0), nPe);
                                }
                            } else if (eleFut)
                                m_uiECOp->Ccf(cVout, (const T**)cVin, cSz, rk_s,
                                              dt_c, dt_f, dt, dof, nsIn);
                            else
                                m_uiECOp->Cfc(cVout, (const T**)cVin, cSz, rk_s,
                                              dt_c, dt_f, dt, dof, nsIn);
                            if (sprof) (eleFut ? tcorrCf : tcorrFc) += ts::EnutsProf::now() - tc0_;
                        }

#pragma unroll
                        for (unsigned int v = 0; v < dof; v++)
                            std::memcpy(dgWVec + v * dgSz + elem * nPe,
                                        cVout[v * nsIn + (rk_s - 1)],
                                        sizeof(T) * nPe);
                        if (syncFast && m_uiSyncPassOpen)
                            m_uiSyncTag[elem] = {m_uiSyncPass, rk_s,
                                                 (unsigned char)eleFut, dt_c,
                                                 dt_f, dt};
                    }
                }
            }
        }

        // for(unsigned int elem = blkList[blk].getLocalElementBegin(); elem <
        // blkList[blk].getLocalElementEnd(); elem++)
        // {
        //     const unsigned int
        //     ei=(pNodes[elem].getX()-blkNode.getX())>>(m_uiMaxDepth-regLevel);
        //     const unsigned int
        //     ej=(pNodes[elem].getY()-blkNode.getY())>>(m_uiMaxDepth-regLevel);
        //     const unsigned int
        //     ek=(pNodes[elem].getZ()-blkNode.getZ())>>(m_uiMaxDepth-regLevel);

        //     const unsigned int emin = 0;
        //     const unsigned int emax = (1u<<(regLevel-blkNode.getLevel()))-1;

        //     assert(etVec[elem] == bTime );

        //     #pragma unroll
        //     for(unsigned int v=0; v < dof; v++)
        //         std::memcpy(dgWVec + v*dgSz + elem * nPe , dgStages[rk_s-1] +
        //         v*dgSz + elem * nPe, sizeof(T)*nPe );

        // }

        // for(unsigned int elem = blkList[blk].getLocalElementBegin(); elem <
        // blkList[blk].getLocalElementEnd(); elem++)
        //     eid.push_back(elem);
        // pMesh->DG2CGVec(dgWVec,cgWVec,true,eid.data(),eid.size(),dof);
        // pMesh->unzip(cgWVec,uzWVec, &blk, 1, dof);

        // #pragma unroll
        // for(unsigned int v=0; v < dof; v++)
        //     std::memcpy(m_uiVec + v*lx*ly*lz , uzWVec + v* unSz + offset,
        //     sizeof(T)*(lx*ly*lz));

        // return;
    }

    if (ownSync) {
        // dgWVec holds the stage of the dependents and of the owners of their
        // nodes in this block's time frame; add the block's own elements, give
        // every cg node its owner's value and read each element back through
        // the cg vector (what zip + unzip does for a stage in ETS)
        const unsigned int* e2n_dg = pMesh->getE2NMapping_DG().data();
        // [ENUTS_OMP edit 4/5] per-thread stand-in for the shared cg scratch
        if (!thr_) ensure_cg_tmp();  // thread 0 / serial only
        T* cgW = thr_ ? thr_->cgW.get() : m_uiEVecTmp[0].get_vec_ptr();
        std::vector<unsigned int> es(eid.begin(), eid.end());
        for (const unsigned int elem : blkList[blk]) {
            es.push_back(elem);
            for (unsigned int v = 0; v < dof; v++)
                std::memcpy(dgWVec + v * dgSz + elem * nPe,
                            dgStages[rk_s - 1] + v * dgSz + elem * nPe,
                            sizeof(T) * nPe);
        }
        for (const unsigned int ele : es)
            for (unsigned int q = 0; q < nPe; q++) {
                const unsigned int cg = e2n[ele * nPe + q];
                const unsigned int dg = e2n_dg[ele * nPe + q];
                for (unsigned int v = 0; v < dof; v++)
                    cgW[v * cgSz + cg] = dgWVec[v * dgSz + dg];
            }
        for (const unsigned int ele : es)
            for (unsigned int v = 0; v < dof; v++)
                pMesh->getElementNodalValues(cgW + v * cgSz,
                                             dgWVec + v * dgSz + ele * nPe, ele);
    }

    double tun0 = 0;
    if (sprof) {
        tun0 = ts::EnutsProf::now();
        ts::EnutsProf::get().t[ts::EnutsProf::SY_CORR] += tcorrCf;
        ts::EnutsProf::get().t[ts::EnutsProf::SY_CORR_FC] += tcorrFc;
        ts::EnutsProf::get().t[ts::EnutsProf::SY_SAME] += (tun0 - tsy0) - tcorrCf - tcorrFc;
    }
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

    const double d_compar_tol = 1e-6;

    if (syncFast) {
        if (m_uiSyncPlan.size() != pMesh->getLocalBlockList().size())
            m_uiSyncPlan.assign(pMesh->getLocalBlockList().size(), SyncPlan());
        SyncPlan& pl = m_uiSyncPlan[blk];
        if (!pl.built) build_sync_plan(blk, eid, pl);
        const size_t lxyz = (size_t)lx * ly * lz;
        if (sprof) {
            ts::EnutsProf::get().t[ts::EnutsProf::CN_ENT] += pl.ents.size();
            ts::EnutsProf::get().t[ts::EnutsProf::CN_PAIRS] += pl.src.size();
            unsigned int nc = 0;
            for (const typename SyncPlan::Ent& en : pl.ents) nc += (en.cnum >= 0);
            ts::EnutsProf::get().t[ts::EnutsProf::CN_P2C] += nc;
        }
        const size_t p2cStride = (size_t)dof * nPe;
        const bool useP2cCache = m_uiSyncPassOpen && !isIGTSync && p2cStride > 0;
        // sweeps of the 3D interpolation shared by the children of one source row
        if (m_uiOwnSweep.size() < 6 * p2cStride) m_uiOwnSweep.resize(6 * p2cStride);
        if (m_uiOwnSweepOk.size() != (size_t)6 * dof) m_uiOwnSweepOk.assign((size_t)6 * dof, 0);
        unsigned int swEle = ~0u;
        const T* swBase    = nullptr;
        for (const typename SyncPlan::Ent& en : pl.ents) {
            const T* srcBase = dgWVec;
            const bool sameTime = !isIGTSync && etVec[en.ele] == bTime &&
                                  !(ifxEq && dtELev[en.ele] != bDT);
            if (sameTime)
                srcBase = (rk_s == 0) ? (const T*)dgEVar : (const T*)dgStages[rk_s - 1];
            const unsigned int* sp = pl.src.data();
            const unsigned int* dp = pl.dst.data();
            if (en.cnum < 0) {
                for (unsigned int v = 0; v < dof; v++) {
                    T* dv       = m_uiVec + v * lxyz;
                    const T* sv = srcBase + (size_t)v * dgSz + (size_t)en.ele * nPe;
                    for (unsigned int q = en.b; q < en.e; q++) dv[dp[q]] = sv[sp[q]];
                }
                continue;
            }
            T* slotData = nullptr;
            bool hit    = false;
            if (useP2cCache) {
                SyncTag key;
                if (sameTime) {
                    key.pass = m_uiSyncPass;
                    key.rk   = rk_s;
                    key.fut  = 2;
                } else {
                    key = m_uiSyncTag[en.ele];
                }
                // nullptr: the pass store is full, interpolate into scratch
                slotData = p2c_slot(en.store, key, p2cStride, hit);
            }
            if (sprof) ts::EnutsProf::get().t[ts::EnutsProf::CN_P2C_HIT] += hit;
            for (unsigned int v = 0; v < dof; v++) {
                T* dv       = m_uiVec + v * lxyz;
                const T* sv = srcBase + (size_t)v * dgSz + (size_t)en.ele * nPe;
                T* pv       = slotData ? slotData + (size_t)v * nPe : p2cI.data();
                if (!hit) {
                    const double tp2 = sprof ? ts::EnutsProf::now() : 0.0;
                    if (m_uiDim == 3) {
                        if (v == 0 && (swEle != en.ele || swBase != srcBase)) {
                            std::fill(m_uiOwnSweepOk.begin(), m_uiOwnSweepOk.end(), 0);
                            swEle  = en.ele;
                            swBase = srcBase;
                        }
                        p2c3_shared(pMesh, sv, pv, en.cnum,
                                    m_uiOwnSweep.data() + (size_t)v * 6 * nPe,
                                    m_uiOwnSweepOk.data() + (size_t)v * 6);
                    } else {
                        pMesh->parent2ChildInterpolation(sv, pv, en.cnum, m_uiDim);
                    }
                    if (sprof) ts::EnutsProf::get().t[ts::EnutsProf::SY_P2C] += ts::EnutsProf::now() - tp2;
                }
                for (unsigned int q = en.b; q < en.e; q++) dv[dp[q]] = pv[sp[q]];
            }
        }
    } else
    for (unsigned int m = 0; m < eid.size(); m++) {
        const unsigned int ele = eid[m];
        const T* srcv          = dgWVec;
        if (syncFast && !isIGTSync && etVec[ele] == bTime &&
            !(ifxEq && dtELev[ele] != bDT))
            srcv = (rk_s == 0) ? (const T*)dgEVar : (const T*)dgStages[rk_s - 1];

        // no interpolation needed just copy.
        if (pNodes[ele].getLevel() == bLev) {
            const double hh = (1u << (m_uiMaxDepth - pNodes[ele].getLevel())) /
                              (double)eOrder;
            const double invhh = 1.0 / hh;

            for (unsigned int k = 0; k < eOrder + 1; k++) {
                double zz = pNodes[ele].minZ() + k * hh;

                if (fabs(zz - zmin) < d_compar_tol) zz = zmin;
                if (fabs(zz - zmax) < d_compar_tol) zz = zmax;

                if (zz < zmin || zz > zmax) continue;
                const unsigned int kkz = std::round((zz - zmin) * invhh);
                assert(std::fabs(zz - zmin - kkz * hh) < 1e-6);
                assert(kkz >= 0 && kkz < lz);

                for (unsigned int j = 0; j < eOrder + 1; j++) {
                    double yy = pNodes[ele].minY() + j * hh;

                    if (fabs(yy - ymin) < d_compar_tol) yy = ymin;
                    if (fabs(yy - ymax) < d_compar_tol) yy = ymax;

                    if (yy < ymin || yy > ymax) continue;
                    const unsigned int jjy = std::round((yy - ymin) * invhh);
                    // std::cout<<"yy: "<<yy<<" (ymin + hh*jjy): "<<(ymin +
                    // hh*jjy)<<std::endl;
                    assert(std::fabs(yy - ymin - jjy * hh) < 1e-6);
                    assert(jjy >= 0 && jjy < ly);

                    for (unsigned int i = 0; i < eOrder + 1; i++) {
                        double xx = pNodes[ele].minX() + i * hh;

                        if (fabs(xx - xmin) < d_compar_tol) xx = xmin;
                        if (fabs(xx - xmax) < d_compar_tol) xx = xmax;

                        if (xx < xmin || xx > xmax) continue;
                        const unsigned int iix =
                            std::round((xx - xmin) * invhh);
                        assert(std::fabs(xx - xmin - iix * hh) < 1e-6);
                        assert(iix >= 0 && iix < lx);

                        // std::cout<<"blk: "<<blk<<" copy : (i,j,k): ("<<kkz<<"
                        // , "<<jjy<<", "<<iix<<")"<<" of : "<<lx<<std::endl;

                        // if(blkNode.isAncestor(pNodes[ele]))
                        // {
                        //     const unsigned int
                        //     ei=(pNodes[ele].getX()-blkNode.getX())>>(m_uiMaxDepth-regLevel);
                        //     const unsigned int
                        //     ej=(pNodes[ele].getY()-blkNode.getY())>>(m_uiMaxDepth-regLevel);
                        //     const unsigned int
                        //     ek=(pNodes[ele].getZ()-blkNode.getZ())>>(m_uiMaxDepth-regLevel);

                        //     std::cout<<"blk: "<<blk<<" copy : (i,j,k):
                        //     ("<<iix<<" , "<<jjy<<", "<<kkz<<")"<<" of :
                        //     "<<lx<<" ei: "<<ei<<" ej: "<<ej<<" ek: "<<ek << "
                        //     xx: "<<xx<<" yy: "<<yy<<" zz:"<<zz<<" xmin:
                        //     "<<xmin<<" ymin: "<<ymin<<" zmin: "<<zmin<<" hh :
                        //     "<<hh<<" hhx : "<<hx<<std::endl;

                        // }

                        for (unsigned int v = 0; v < dof; v++) {
                            // double v1 = m_uiVec[(v*lx*ly*lz) + kkz*lx*ly +
                            // jjy*lx + iix]; double v2 = srcv[v*dgSz +
                            // ele*nPe + k*(eOrder+1)*(eOrder+1)+ j*(eOrder+1) +
                            // i]; if(rk_s==0 && fabs(v2-v1)>1e-3)
                            // {
                            //     std::cout<<"var : "<<v<<" equal level copy:
                            //     "<<blk<<" old : "<<v1<<" new :
                            //     "<<v2<<std::endl;
                            // }
                            m_uiVec[(v * lx * ly * lz) + kkz * lx * ly +
                                    jjy * lx + iix] =
                                srcv[v * dgSz + ele * nPe +
                                       k * (eOrder + 1) * (eOrder + 1) +
                                       j * (eOrder + 1) + i];
                        }
                    }
                }
            }

        } else if (pNodes[ele].getLevel() > bLev) {
            assert((bLev + 1) == pNodes[ele].getLevel());
            const unsigned int cnum = pNodes[ele].getMortonIndex();
            ot::TreeNode tmpParent  = pNodes[ele].getParent();

            const double hh = (1u << (m_uiMaxDepth - pNodes[ele].getLevel())) /
                              (double)eOrder;
            const double invhh = 1.0 / (2 * hh);

            assert(eOrder > 1);
            const unsigned int cb = (eOrder % 2 == 0) ? 0 : 1;

            for (unsigned int k = cb; k < eOrder + 1; k += 2) {
                double zz = (pNodes[ele].minZ() + k * hh);
                if (fabs(zz - zmin) < d_compar_tol) zz = zmin;
                if (fabs(zz - zmax) < d_compar_tol) zz = zmax;

                if (zz < zmin || zz > zmax) continue;
                const unsigned int kkz = std::round((zz - zmin) * invhh);
                assert(kkz >= 0 && kkz < lz);

                for (unsigned int j = cb; j < eOrder + 1; j += 2) {
                    double yy = pNodes[ele].minY() + j * hh;

                    if (fabs(yy - ymin) < d_compar_tol) yy = ymin;
                    if (fabs(yy - ymax) < d_compar_tol) yy = ymax;

                    if (yy < ymin || yy > ymax) continue;

                    const unsigned int jjy = std::round((yy - ymin) * invhh);
                    assert(jjy >= 0 && jjy < ly);

                    for (unsigned int i = cb; i < eOrder + 1; i += 2) {
                        double xx = pNodes[ele].minX() + i * hh;

                        if (fabs(xx - xmin) < d_compar_tol) xx = xmin;
                        if (fabs(xx - xmax) < d_compar_tol) xx = xmax;

                        if (xx < xmin || xx > xmax) continue;
                        const unsigned int iix =
                            std::round((xx - xmin) * invhh);
                        assert(iix >= 0 && iix < lx);

                        // if(blkNode.isAncestor(pNodes[ele]))
                        // {
                        //     const unsigned int
                        //     ei=(pNodes[ele].getX()-blkNode.getX())>>(m_uiMaxDepth-regLevel);
                        //     const unsigned int
                        //     ej=(pNodes[ele].getY()-blkNode.getY())>>(m_uiMaxDepth-regLevel);
                        //     const unsigned int
                        //     ek=(pNodes[ele].getZ()-blkNode.getZ())>>(m_uiMaxDepth-regLevel);

                        //     std::cout<<"blk: "<<blk<<" copy : (i,j,k):
                        //     ("<<iix<<" , "<<jjy<<", "<<kkz<<")"<<" of :
                        //     "<<lx<<" ei: "<<ei<<" ej: "<<ej<<" ek: "<<ek << "
                        //     xx: "<<xx<<" yy: "<<yy<<" zz:"<<zz<<" xmin:
                        //     "<<xmin<<" ymin: "<<ymin<<" zmin: "<<zmin<<" hh :
                        //     "<<hh<<" hhx : "<<hx<<std::endl;

                        // }

                        // std::cout<<"blk: "<<blk<<" blk copy : (i,j,k):
                        // ("<<iix<<" , "<<jjy<<", "<<kkz<<")"<<" of : "<<lx<<"
                        // xx: "<<xx<<" yy: "<<yy<<" zz:"<<zz<<" xmin:
                        // "<<xmin<<" ymin: "<<ymin<<" zmin: "<<zmin<<" hh :
                        // "<<hh<<" hhx : "<<hx<<std::endl;
                        for (unsigned int v = 0; v < dof; v++) {
                            // double v1 = m_uiVec[(v*lx*ly*lz) + kkz*lx*ly +
                            // jjy*lx + iix]; double v2 = srcv[v*dgSz +
                            // ele*nPe + k*(eOrder+1)*(eOrder+1)+ j*(eOrder+1) +
                            // i]; if(rk_s==0 && fabs(v2-v1)>1e-3)
                            // {
                            //     std::cout<<"var : "<<v<<" copy from finer
                            //     octant: "<<blk<<" old : "<<v1<<" new :
                            //     "<<v2<<std::endl;
                            // }
                            m_uiVec[(v * lx * ly * lz) + kkz * lx * ly +
                                    jjy * lx + iix] =
                                srcv[v * dgSz + ele * nPe +
                                       k * (eOrder + 1) * (eOrder + 1) +
                                       j * (eOrder + 1) + i];
                        }
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
                    const unsigned int cnum = childOct[child].getMortonIndex();
                    pMesh->parent2ChildInterpolation(
                        &srcv[v * dgSz + ele * nPe], p2cI.data(), cnum,
                        m_uiDim);

                    for (unsigned int k = 0; k < eOrder + 1; k++) {
                        double zz = childOct[child].minZ() + k * hh;

                        if (fabs(zz - zmin) < d_compar_tol) zz = zmin;
                        if (fabs(zz - zmax) < d_compar_tol) zz = zmax;

                        if (zz < zmin || zz > zmax) continue;
                        const unsigned int kkz =
                            std::round((zz - zmin) * invhh);
                        assert(kkz >= 0 && kkz < lz);

                        for (unsigned int j = 0; j < eOrder + 1; j++) {
                            double yy = childOct[child].minY() + j * hh;

                            if (fabs(yy - ymin) < d_compar_tol) yy = ymin;
                            if (fabs(yy - ymax) < d_compar_tol) yy = ymax;

                            if (yy < ymin || yy > ymax) continue;

                            const unsigned int jjy =
                                std::round((yy - ymin) * invhh);
                            assert(jjy >= 0 && jjy < ly);

                            for (unsigned int i = 0; i < eOrder + 1; i++) {
                                double xx = childOct[child].minX() + i * hh;

                                if (fabs(xx - xmin) < d_compar_tol) xx = xmin;
                                if (fabs(xx - xmax) < d_compar_tol) xx = xmax;

                                if (xx < xmin || xx > xmax) continue;
                                const unsigned int iix =
                                    std::round((xx - xmin) * invhh);
                                assert(iix >= 0 && iix < lx);
                                // std::cout<<"blk: "<<blk<<" blk copy :
                                // (i,j,k): ("<<iix<<" , "<<jjy<<",
                                // "<<kkz<<")"<<" of : "<<lx<<" xx: "<<xx<<" yy:
                                // "<<yy<<" zz:"<<zz<<" xmin: "<<xmin<<" ymin:
                                // "<<ymin<<" zmin: "<<zmin<<" hh : "<<hh<<" hhx
                                // : "<<hx<<" child: "<<child<<std::endl;
                                //  double v1 = m_uiVec[(v*lx*ly*lz) + kkz*lx*ly
                                //  + jjy*lx + iix]; double v2 =
                                //  p2cI[k*(eOrder+1)*(eOrder+1)+ j*(eOrder+1) +
                                //  i]; if(rk_s==0 && fabs(v2-v1)>1e-3)
                                //  {
                                //      std::cout<<"var : "<<v<<" copy from
                                //      coaser octant(p2c): "<<blk<<" old :
                                //      "<<v1<<" new : "<<v2<<std::endl;
                                //  }
                                m_uiVec[(v * lx * ly * lz) + kkz * lx * ly +
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

    // block internal copy.
    double town0 = 0;
    if (sprof) {
        town0 = ts::EnutsProf::now();
        ts::EnutsProf::get().t[ts::EnutsProf::SY_UNZIP] += town0 - tun0;
    }
    if (syncFast) {
        const T* ownSrc = (rk_s == 0) ? (const T*)dgEVar : (const T*)dgStages[rk_s - 1];
        const unsigned int n1 = eOrder + 1;
        for (const unsigned int elem : blkList[blk]) {
            const unsigned int ei = (pNodes[elem].getX() - blkNode.getX()) >>
                                    (m_uiMaxDepth - regLevel);
            const unsigned int ej = (pNodes[elem].getY() - blkNode.getY()) >>
                                    (m_uiMaxDepth - regLevel);
            const unsigned int ek = (pNodes[elem].getZ() - blkNode.getZ()) >>
                                    (m_uiMaxDepth - regLevel);
            assert(etVec[elem] == bTime);
            for (unsigned int v = 0; v < dof; v++) {
                const T* sv = ownSrc + (size_t)v * dgSz + (size_t)elem * nPe;
                T* dv       = m_uiVec + (size_t)v * lx * ly * lz;
                for (unsigned int k = 0; k < n1; k++)
                    for (unsigned int j = 0; j < n1; j++)
                        std::memcpy(dv + (size_t)(ek * eOrder + k + PW) * (ly * lx) +
                                        (size_t)(ej * eOrder + j + PW) * lx +
                                        (ei * eOrder + PW),
                                    sv + k * n1 * n1 + j * n1, sizeof(T) * n1);
            }
        }
    } else if (rk_s == 0) {
        for (const unsigned int elem : blkList[blk]) {
            const unsigned int ei = (pNodes[elem].getX() - blkNode.getX()) >>
                                    (m_uiMaxDepth - regLevel);
            const unsigned int ej = (pNodes[elem].getY() - blkNode.getY()) >>
                                    (m_uiMaxDepth - regLevel);
            const unsigned int ek = (pNodes[elem].getZ() - blkNode.getZ()) >>
                                    (m_uiMaxDepth - regLevel);

            const unsigned int emin = 0;
            const unsigned int emax =
                (1u << (regLevel - blkNode.getLevel())) - 1;

            assert(etVec[elem] == bTime);

            // #pragma unroll
            // for(unsigned int v=0; v < dof; v++)
            //     std::memcpy(dgWVec + v*dgSz + elem * nPe , dgStages[rk_s-1] +
            //     v*dgSz + elem * nPe, sizeof(T)*nPe );

            for (unsigned int v = 0; v < dof; v++)
                for (unsigned int k = 0; k < (eOrder + 1); k++)
                    for (unsigned int j = 0; j < (eOrder + 1); j++)
                        for (unsigned int i = 0; i < (eOrder + 1); i++) {
                            // double v1 = m_uiVec[v*lx*ly*lz +
                            // (ek*eOrder+k+PW)*(ly*lx)+(ej*eOrder+j+PW)*(lx)+(ei*eOrder+i+PW)];
                            // double v2 = dgEVar[ v*dgSz + elem * nPe +
                            // k*(eOrder+1)*(eOrder+1) + j*(eOrder+1) + i];

                            // if(fabs(v1-v2)>1e-3)
                            //     std::cout<<"i: "<<i<< "j: "<<j<<" k: "<<k<<"
                            //     [internal]: v1(old): "<<v1<<" v2:(new):
                            //     "<<v2<<std::endl;

                            m_uiVec[v * lx * ly * lz +
                                    (ek * eOrder + k + PW) * (ly * lx) +
                                    (ej * eOrder + j + PW) * (lx) +
                                    (ei * eOrder + i + PW)] =
                                dgEVar[v * dgSz + elem * nPe +
                                       k * (eOrder + 1) * (eOrder + 1) +
                                       j * (eOrder + 1) + i];
                        }
        }

    } else {
        for (const unsigned int elem : blkList[blk]) {
            const unsigned int ei = (pNodes[elem].getX() - blkNode.getX()) >>
                                    (m_uiMaxDepth - regLevel);
            const unsigned int ej = (pNodes[elem].getY() - blkNode.getY()) >>
                                    (m_uiMaxDepth - regLevel);
            const unsigned int ek = (pNodes[elem].getZ() - blkNode.getZ()) >>
                                    (m_uiMaxDepth - regLevel);

            const unsigned int emin = 0;
            const unsigned int emax =
                (1u << (regLevel - blkNode.getLevel())) - 1;

            assert(etVec[elem] == bTime);

            // #pragma unroll
            // for(unsigned int v=0; v < dof; v++)
            //     std::memcpy(dgWVec + v*dgSz + elem * nPe , dgStages[rk_s-1] +
            //     v*dgSz + elem * nPe, sizeof(T)*nPe );

            for (unsigned int v = 0; v < dof; v++)
                for (unsigned int k = 0; k < (eOrder + 1); k++)
                    for (unsigned int j = 0; j < (eOrder + 1); j++)
                        for (unsigned int i = 0; i < (eOrder + 1); i++)
                            m_uiVec[v * lx * ly * lz +
                                    (ek * eOrder + k + PW) * (ly * lx) +
                                    (ej * eOrder + j + PW) * (lx) +
                                    (ei * eOrder + i + PW)] =
                                (ownSync ? dgWVec : dgStages[rk_s - 1])
                                        [v * dgSz + elem * nPe +
                                         k * (eOrder + 1) * (eOrder + 1) +
                                         j * (eOrder + 1) + i];
        }
    }

    if (sprof) {
        const double tend = ts::EnutsProf::now();
        ts::EnutsProf::get().t[ts::EnutsProf::SY_OWN] += tend - town0;
        ts::EnutsProf::get().t[ts::EnutsProf::SY_BLK] += tend - tsy0;
    }
#ifdef __PROFILE_ENUTS__
    m_uiPt[ENUTSPROFILE::BLK_SYNC].stop();
#endif
}

}  // namespace ts
