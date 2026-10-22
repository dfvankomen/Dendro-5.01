/**
 * @file enuts_remesh.tcc
 * @brief ExplicitNUTS per-mesh setup and teardown (internal vectors, sync_with_mesh, remesh test).
 * (code moved from enuts.h, which includes this file)
 */

namespace ts {

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::init_data_structures() {
    // identify dependent and independent blocks.
    const ot::Mesh* pMesh = m_uiAppCtx->get_mesh();
    const bool isActive   = pMesh->isActive();

    pMesh->computeMinMaxLevel(m_uiLevMin, m_uiLevMax);
    assert((m_uiLevMin > 0) && (m_uiLevMax <= m_uiMaxDepth));

    update_wpart_metadata();
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::free_data_structures() {
    return;
}

template <typename T, typename Ctx>
int ExplicitNUTS<T, Ctx>::allocate_internal_vars() {
    assert(m_uiNumStages > 0);
    const unsigned int DOF = m_uiEVar.get_dof();

    if (m_uiIsInternalAlloc)
        return 0;  // no need to allocated again if the internal vars are
                   // allocated.

    m_uiStVec.resize(m_uiNumStages);

    for (unsigned int i = 0; i < m_uiNumStages; i++)
        m_uiStVec[i].create_vector(
            m_uiAppCtx->get_mesh(), ot::DVEC_TYPE::OCT_LOCAL_NODES,
            ot::DVEC_LOC::HOST, m_uiEVar.get_dof(), true);

    // a work array: its rows are written before they are read
    m_uiDGTmp0.create_vector(m_uiAppCtx->get_mesh(),
                             ot::DVEC_TYPE::OCT_LOCAL_NODES, ot::DVEC_LOC::HOST,
                             m_uiEVar.get_dof(), true, false);

    m_uiStGOn = ts::EnutsProf::ifaceNL() && m_uiType == ETSType::RK4;
    if (m_uiStGOn) {
        m_uiStG.create_vector(m_uiAppCtx->get_mesh(),
                              ot::DVEC_TYPE::OCT_LOCAL_NODES, ot::DVEC_LOC::HOST,
                              m_uiEVar.get_dof(), true);
        std::fill(m_uiStG.get_vec_ptr(),
                  m_uiStG.get_vec_ptr() + m_uiStG.get_size(), (T)0);
        m_uiStGValid.assign(m_uiAppCtx->get_mesh()->getAllElements().size(), 0);
        m_uiStGValidCls.assign(m_uiOwnDTs.size(), 0);
        if (ts::EnutsProf::on()) ts::EnutsProf::get().t[ts::EnutsProf::CN_GRESET] += 1;
    }

    m_uiEvarUzip.create_vector(m_uiAppCtx->get_mesh(),
                               ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING,
                               ot::DVEC_LOC::HOST, m_uiEVar.get_dof(), true);
    m_uiCgOwnerMesh = nullptr;
    m_uiBlkDepEid.clear();
    m_uiBlkOwnEid.clear();
    m_uiSyncPlan.clear();
    own_reset();
    free_sync_threads();
    m_uiScopedMesh = nullptr;
    m_uiScopedOk   = false;
    // the DG copy of the solution (element-wise sync path, labelled zip) is
    // allocated where those paths start (ensure_evar_dg)

    const ot::Mesh* pMesh = m_uiAppCtx->get_mesh();
    if (pMesh->isActive()) {
        m_uiEleTime.resize(pMesh->getAllElements().size(), 0);
        m_uiEleDT.resize(pMesh->getAllElements().size(), 0);

        const std::vector<ot::Block>& blkList = pMesh->getLocalBlockList();
        m_uiBVec.resize(blkList.size());

        // the ring needs a_{s,j} = 0 for j < s-1 (stage s reads k_{s-1} only)
        m_uiStageRing = true;
        for (unsigned int st = 3; st <= m_uiNumStages; st++)
            for (unsigned int j = 1; j < st - 1; j++)
                if (m_uiAij[(st - 1) * m_uiNumStages + (j - 1)] != (T)0)
                    m_uiStageRing = false;
        // slots: solution, NS stages (ring: one stage and the accumulator)
        const unsigned int nvec = m_uiStageRing ? 3 : m_uiNumStages + 1;
        unsigned int maxNN = 0;
        for (unsigned int blk = 0; blk < blkList.size(); blk++) {
            const unsigned int sz[3] = {blkList[blk].getAllocationSzX(),
                                        blkList[blk].getAllocationSzY(),
                                        blkList[blk].getAllocationSzZ()};
            m_uiBVec[blk].alloc_vec(nvec, blk, sz, DOF);
            maxNN = std::max(maxNN, sz[0] * sz[1] * sz[2]);
        }
        unsigned int nt = 1;
#ifdef DENDRO_ENUTS_HAS_OMP
        nt = (unsigned int)omp_get_max_threads();
#endif
        m_uiStageIn.resize(nt);
        const unsigned int ssz[3] = {maxNN, 1, 1};
        for (unsigned int t = 0; t < nt; t++)
            m_uiStageIn[t].createVec(0, ssz, false,
                                     BLK_ASYNC_VEC_MODE::BLK_UNZIP, DOF);
        if (m_uiStageRing) {
            m_uiStageSgn.resize(blkList.size());
            for (unsigned int blk = 0; blk < blkList.size(); blk++)
                m_uiStageSgn[blk].assign(
                    ((size_t)DOF * m_uiBVec[blk]._vec[0].getSz() + 63) / 64, 0);
        }
    }

    m_uiECOp            = new ENUTSOp(m_uiType);
    m_uiIsInternalAlloc = true;
    m_uiBlkTimeLevMinMax.resize(2 * m_uiMaxDepth, 0);

    m_uiActiveBlkIDs.reserve(pMesh->getLocalBlockList().size());
    m_uiDependentActiveBlkIDs.reserve(pMesh->getLocalBlockList().size());
    m_uiIndependentActiveBlkIDs.reserve(pMesh->getLocalBlockList().size());
    if (ts::ts_mem_info()) mem_report();
    return 0;
}

/** DENDRO_TS_MEM_INFO=1: bytes per buffer family (collective, rank 0 prints).
 *  blk_*: the per-block padded vectors (_vec[0] solution, [1..NS] stages,
 *  [NS+1] stage input / output scratch); thr_dgW_omp: the per-thread sync
 *  scratch sync_thread_setup would allocate at this thread count. */
template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::mem_report() {
    static int ncall = 0;
    char tag[32];
    std::snprintf(tag, sizeof(tag), ncall ? "remesh%d" : "init", ncall);
    ncall++;
    const ot::Mesh* pMesh = m_uiAppCtx->get_mesh();
    const bool a          = pMesh->isActive();
    const double B        = (double)sizeof(T);
    const unsigned int DOF = m_uiEVar.get_dof();
    double blkN = 0;  // padded nodes over all local blocks (one copy, one dof)
    double stageSlots = 0, scratchSlots = 0;
    for (unsigned int blk = 0; blk < m_uiBVec.size(); blk++) {
        const double nn = (double)m_uiBVec[blk]._vec[0].getSz();
        blkN += nn;
        const unsigned int nv = (unsigned int)m_uiBVec[blk]._vec.size();
        stageSlots += nn * (nv - 1);
    }
    for (unsigned int t = 0; t < m_uiStageIn.size(); t++)
        scratchSlots += (double)m_uiStageIn[t].getSz();
    double thr = 0;
#ifdef DENDRO_ENUTS_HAS_OMP
    if (a) {
        const double nt = (double)omp_get_max_threads();
        thr = (nt > 1 ? nt - 1 : 0) * B * DOF * pMesh->getAllElements().size() *
              pMesh->getNumNodesPerElement();
    }
#endif
    const char* nm[] = {"ctx_evar",   "blk_sol",   "blk_stages", "blk_scratch",
                        "evar_uzip",  "st_dg",     "dg_tmp0",    "evar_dg",
                        "cg_tmp_x2",  "stG",       "ele_time_dt", "thr_dgW_omp",
                        "ctx_unzip_x2_est"};
    double v[] = {
        a ? B * m_uiEVar.get_size() : 0,
        B * DOF * blkN,
        B * DOF * stageSlots,
        B * DOF * scratchSlots,
        a ? B * m_uiEvarUzip.get_size() : 0,
        a ? B * (double)m_uiNumStages * m_uiStVec[0].get_size() : 0,
        a ? B * m_uiDGTmp0.get_size() : 0,
        (a && m_uiEvarDG.get_vec_ptr()) ? B * m_uiEvarDG.get_size() : 0,
        a ? B * ((m_uiEVecTmp[0].get_vec_ptr() ? m_uiEVecTmp[0].get_size() : 0) +
                 (m_uiEVecTmp[1].get_vec_ptr() ? m_uiEVecTmp[1].get_size() : 0))
          : 0,
        (a && m_uiStGOn) ? B * m_uiStG.get_size() : 0,
        (double)sizeof(unsigned int) * (m_uiEleTime.size() + m_uiEleDT.size()),
        thr,
        a ? B * 2.0 * pMesh->getDegOfFreedomUnZip() * DOF : 0};
    ts::ts_mem_counts<T>(pMesh, DOF, "enuts", tag);
    ts::ts_mem_print("enuts", tag, 13, nm, v, true);

    // stage locality census: which blocks sit next to a time-level change and
    // which element rows of m_uiStVec are read at all / read by a block with a
    // smaller step (those must survive past the sub-step they were written in)
    double c[7] = {0, 0, 0, 0, 0, 0, 0};
    const char* cn[] = {"blocks",        "blocks_jump",  "blk_nodes",
                        "blk_nodes_jump", "elems_all",    "elems_live",
                        "elems_persist"};
    if (a && m_uiLevMin >= 1 && m_uiLevMax <= m_uiMaxDepth &&
        m_uiLevMin <= m_uiLevMax) {
        const std::vector<ot::TreeNode>& all  = pMesh->getAllElements();
        const std::vector<ot::Block>& blkList = pMesh->getLocalBlockList();
        sync_lazy_build(true);  // same lists partial_evolve builds lazily
        std::vector<unsigned int> dtOf(all.size(), 0);
        for (size_t e = 0; e < all.size(); e++)
            dtOf[e] = step_ratio(all[e].getLevel());
        std::vector<unsigned char> live(all.size(), 0), persist(all.size(), 0);
        for (unsigned int b = 0; b < blkList.size(); b++) {
            const unsigned int bdt = dtOf[blkList[b].firstElementID()];
            const std::vector<unsigned int>& R =
                (b < m_uiBlkOwnEid.size()) ? m_uiBlkOwnEid[b]
                                                  : m_uiBlkDepEid[b];
            bool jump = false;
            for (const unsigned int e : R) {
                live[e] = 1;
                if (dtOf[e] != bdt) jump = true;
                if (dtOf[e] > bdt) persist[e] = 1;
            }
            const double nn = (double)m_uiBVec[b]._vec[0].getSz();
            c[0] += 1;
            c[2] += nn;
            if (jump) {
                c[1] += 1;
                c[3] += nn;
            }
        }
        const std::vector<unsigned int>& sm = pMesh->getSendElementSM();
        const unsigned int lb                = pMesh->getElementLocalBegin();
        for (const unsigned int e : sm) live[e + lb] = 1;
        c[4] = (double)all.size();
        for (size_t e = 0; e < all.size(); e++) {
            c[5] += live[e];
            c[6] += persist[e];
        }
    }
    ts::ts_mem_print("enuts", tag, 7, cn, c, false);
}

template <typename T, typename Ctx>
int ExplicitNUTS<T, Ctx>::deallocate_internal_vars() {
    if (!m_uiIsInternalAlloc) return 0;

    for (unsigned int i = 0; i < m_uiNumStages; i++) {
        this->m_uiStVec[i].destroy_vector();
    }

    this->m_uiStVec.clear();
    this->m_uiEVecTmp[0].destroy_vector();
    this->m_uiEVecTmp[1].destroy_vector();

    // deallocate DG tmp.
    m_uiDGTmp0.destroy_vector();
    if (m_uiStGOn) {
        m_uiStG.destroy_vector();
        m_uiStGOn = false;
    }

    m_uiEvarUzip.destroy_vector();
    m_uiEvarDG.destroy_vector();

    m_uiEleTime.clear();
    m_uiEleDT.clear();

    for (unsigned int k = 0; k < m_uiBVec.size(); k++) {
        for (unsigned int j = 0; j < m_uiBVec[k]._vec.size(); j++)
            m_uiBVec[k]._vec[j].destroyVec();

        m_uiBVec[k]._vec.clear();
    }

    m_uiBVec.clear();
    for (unsigned int t = 0; t < m_uiStageIn.size(); t++)
        m_uiStageIn[t].destroyVec();
    m_uiStageIn.clear();
    m_uiStageSgn.clear();
    m_uiBlkDepEid.clear();
    m_uiBlkOwnEid.clear();
    m_uiSyncPlan.clear();
    own_reset();
    free_sync_threads();
    m_uiScopedMesh = nullptr;
    m_uiScopedOk   = false;

    m_uiBlkTimeLevMinMax.clear();

    m_uiActiveBlkIDs.clear();
    m_uiIndependentActiveBlkIDs.clear();
    m_uiDependentActiveBlkIDs.clear();

    delete m_uiECOp;
    m_uiIsInternalAlloc = false;
    return 0;
}

template <typename T, typename Ctx>
int ExplicitNUTS<T, Ctx>::sync_with_mesh() {
    if (m_uiAppCtx->is_ets_synced()) return 0;

    // the level offset belongs to the new mesh, and the ladder built below reads it
    m_uiLevOffset = m_uiAppCtx->compute_lts_ts_offset();

    this->deallocate_internal_vars();
    this->free_data_structures();

    m_uiEVar = m_uiAppCtx->get_evolution_vars();
    this->init_data_structures();
    this->allocate_internal_vars();
    {
        const ot::Mesh* pMesh = m_uiAppCtx->get_mesh();
        if (pMesh->isActive() && !pMesh->getMPIRank())
            std::cout << "[LTS] level offset " << m_uiLevOffset
                      << ", coarsest step ratio " << step_ratio(m_uiLevMin)
                      << std::endl;
    }

    m_uiAppCtx->set_ets_synced(true);

    return 0;
}

template <typename T, typename Ctx>
bool ExplicitNUTS<T, Ctx>::isRemeshEvars(
    const unsigned int* varIds, unsigned int numvars,
    std::function<double(double, double, double, double*)> wavelet_tol,
    double amr_coarsen_fac) {
    ot::Mesh* pMesh = (ot::Mesh*)m_uiAppCtx->get_mesh();
    std::vector<unsigned int> refine_flags;
    bool isMeshGlobalChanged = false;
    bool isMeshLocalChanged  = false;

    if (pMesh->isActive()) {
        const std::vector<ot::Block>& blkList = pMesh->getLocalBlockList();
        const unsigned int eOrder             = pMesh->getElementOrder();

        const unsigned int unz_ele =
            (2 * eOrder + 1) * (2 * eOrder + 1) * (2 * eOrder + 1);
        const unsigned int DOF = m_uiEVar.get_dof();

        std::vector<T> uEleVec;
        uEleVec.resize(unz_ele * DOF);

        std::vector<T> uEleRefIn;
        uEleRefIn.resize(unz_ele * numvars);

        std::vector<double> wMax;
        wMax.resize(pMesh->getNumLocalMeshElements(), 0.0);

        const unsigned int eleOfst = pMesh->getElementLocalBegin();
        RefElement* refEl          = (RefElement*)pMesh->getReferenceElement();
        wavelet::WaveletEl wRefEl(refEl);

        const ot::TreeNode* pNodes = pMesh->getAllElements().data();

        for (unsigned int blk = 0; blk < blkList.size(); blk++) {
            for (const unsigned int ele : blkList[blk]) {
                bool isBdyEle = pMesh->isBoundaryOctant(ele);
                m_uiBVec[blk]._vec[0].getUnzipElementalValues(pMesh, ele,
                                                              uEleVec.data());

                for (unsigned int v = 0; v < numvars; v++) {
                    const unsigned int vid = varIds[v];
                    std::memcpy(uEleRefIn.data() + v * unz_ele,
                                uEleVec.data() + vid * unz_ele,
                                sizeof(T) * unz_ele);
                }

                const double oct_dx =
                    (1u << (m_uiMaxDepth - pNodes[ele].getLevel())) /
                    (double(pMesh->getElementOrder()));
                Point oct_pt1 = Point(pNodes[ele].minX(), pNodes[ele].minY(),
                                      pNodes[ele].minZ());
                Point oct_pt2 = Point(pNodes[ele].minX() + oct_dx,
                                      pNodes[ele].minY() + oct_dx,
                                      pNodes[ele].minZ() + oct_dx);
                Point domain_pt1, domain_pt2, dx_domain;
                pMesh->octCoordToDomainCoord(oct_pt1, domain_pt1);
                pMesh->octCoordToDomainCoord(oct_pt2, domain_pt2);
                dx_domain    = domain_pt2 - domain_pt1;
                double hx[3] = {dx_domain.x(), dx_domain.y(), dx_domain.z()};
                const double tol_ele = wavelet_tol(
                    domain_pt1.x(), domain_pt1.y(), domain_pt1.z(), hx);

                wMax[ele - eleOfst] = wavelet::compute_element_wavelet(
                    pMesh, (const wavelet::WaveletEl*)&wRefEl, uEleRefIn.data(),
                    tol_ele, numvars, isBdyEle);

                // if(isBdyEle)
                // std::cout<<"ele : "<<ele<<" comp. coefficient :
                // "<<wMax[ele-eleOfst]<<std::endl;
            }
        }

        refine_flags.clear();
        refine_flags.resize(pMesh->getNumLocalMeshElements(), OCT_NO_CHANGE);

        for (unsigned int ele = pMesh->getElementLocalBegin();
             ele < pMesh->getElementLocalEnd(); ele++) {
            const double oct_dx =
                (1u << (m_uiMaxDepth - pNodes[ele].getLevel())) /
                (double(pMesh->getElementOrder()));
            Point oct_pt1 = Point(pNodes[ele].minX(), pNodes[ele].minY(),
                                  pNodes[ele].minZ());
            Point oct_pt2 =
                Point(pNodes[ele].minX() + oct_dx, pNodes[ele].minY() + oct_dx,
                      pNodes[ele].minZ() + oct_dx);
            Point domain_pt1, domain_pt2, dx_domain;
            pMesh->octCoordToDomainCoord(oct_pt1, domain_pt1);
            pMesh->octCoordToDomainCoord(oct_pt2, domain_pt2);
            dx_domain    = domain_pt2 - domain_pt1;
            double hx[3] = {dx_domain.x(), dx_domain.y(), dx_domain.z()};
            const double tol_ele =
                wavelet_tol(domain_pt1.x(), domain_pt1.y(), domain_pt1.z(), hx);

            const double l_max = wMax[ele - eleOfst];

            if (pNodes[ele].getLevel() < m_uiLevMax && l_max > tol_ele) {
                refine_flags[(ele - eleOfst)] = OCT_SPLIT;
                isMeshLocalChanged            = true;

            } else if (pNodes[ele].getLevel() > m_uiLevMin &&
                       l_max < amr_coarsen_fac * tol_ele) {
                // 08/17/2020 : time sub-cycling mess up when we coarsen during
                // the LTS remesh. (currently coarsen enabled at synced remesh.
                // )
                refine_flags[(ele - eleOfst)] = OCT_NO_CHANGE;  // OCT_COARSE;
                // isMeshLocalChanged =true;
            } else {
                refine_flags[(ele - eleOfst)] = OCT_NO_CHANGE;
            }

            // std::cout<<"ele: "<<ele<<" refinement flag:
            // "<<refine_flags[(ele-eleOfst)]<<" wc :
            // "<<wMax[ele-eleOfst]<<std::endl;
        }
    }
    // std::cout<<"grank : "<<pMesh->getMPIRankGlobal()<<std::endl;
    MPI_Allreduce(&isMeshLocalChanged, &isMeshGlobalChanged, 1, MPI_CXX_BOOL,
                  MPI_LOR, pMesh->getMPIGlobalCommunicator());

    if (isMeshGlobalChanged) pMesh->setMeshRefinementFlags(refine_flags);

    return isMeshGlobalChanged;
}

}  // namespace ts
