/**
 * @file enuts_owner.tcc
 * @brief ExplicitNUTS owner sync (cg owner map, owner-sync plan, fill, owner-node exchange).
 * (code moved from enuts.h, which includes this file)
 */

namespace ts {

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::build_cg_owner_map() {
    ot::Mesh* pMesh = m_uiAppCtx->get_mesh();
    if (m_uiCgOwnerMesh == pMesh && m_uiCgOwnerDG.size() == pMesh->getDegOfFreedom()) return;
    m_uiCgOwnerMesh = pMesh;
    const unsigned int eo = pMesh->getElementOrder(), nPe = pMesh->getNumNodesPerElement();
    const std::vector<ot::TreeNode>& all = pMesh->getAllElements();
    const unsigned int* e2n = pMesh->getE2NMapping().data();
    m_uiCgOwnerDG.assign(pMesh->getDegOfFreedom(), LOOK_UP_TABLE_DEFAULT);
    // sharers of a cg = every (element, non-hanging node) mapped to it; the
    // SFC build makes the coarsest sharer the owner, ties by TreeNode order
    for (unsigned int e = pMesh->getElementPreGhostBegin(); e < pMesh->getElementPostGhostEnd(); e++) {
        if (e < pMesh->getElementLocalBegin() || e >= pMesh->getElementLocalEnd()) {
            bool touchesLocal = false;
            for (unsigned int nd = 0; nd < nPe && !touchesLocal; nd++) {
                const unsigned int cg = e2n[e * nPe + nd];
                touchesLocal = cg >= pMesh->getNodeLocalBegin() && cg < pMesh->getNodeLocalEnd();
            }
            if (!touchesLocal) continue;
        }
        for (unsigned int k = 0; k <= eo; k++)
            for (unsigned int j = 0; j <= eo; j++)
                for (unsigned int i = 0; i <= eo; i++) {
                    const unsigned int nd = k * (eo + 1) * (eo + 1) + j * (eo + 1) + i;
                    const unsigned int cg = e2n[e * nPe + nd];
                    if (cg < pMesh->getNodeLocalBegin() || cg >= pMesh->getNodeLocalEnd()) continue;
                    if (pMesh->isNodeHanging(e, i, j, k)) continue;
                    unsigned int& cur = m_uiCgOwnerDG[cg];
                    if (cur == LOOK_UP_TABLE_DEFAULT) { cur = e * nPe + nd; continue; }
                    const unsigned int ce = cur / nPe;
                    if (all[e].getLevel() < all[ce].getLevel() ||
                        (all[e].getLevel() == all[ce].getLevel() && all[e] < all[ce]))
                        cur = e * nPe + nd;
                }
    }
    // pick the owner per grid POSITION over every local and received-ghost
    // element, not per cg slot (the slot pick above stays where no position
    // pick lands). on a labelled mesh the sharers of one position are split
    // across a primary slot, its duplicate slots and ghost slots, so the
    // per-slot pick can land on a finer sharer's copy than the SFC owner (and
    // differ between a primary and its dups); after sub-cycling those copies
    // differ well above rounding
    const std::vector<unsigned int>& cg2dg = pMesh->getCG2DGMap();
    {
        const unsigned int md = m_uiMaxDepth;
        struct KeyHash {
            size_t operator()(const unsigned __int128& k) const {
                const uint64_t a = (uint64_t)k, b = (uint64_t)(k >> 64);
                return std::hash<uint64_t>()(a ^ (b * 0x9E3779B97F4A7C15ull));
            }
        };
        auto key = [&](unsigned int e, unsigned int i, unsigned int j, unsigned int k) {
            const unsigned long long len = 1ull << (md - all[e].getLevel());
            const unsigned long long x = (unsigned long long)all[e].getX() * eo + i * len;
            const unsigned long long y = (unsigned long long)all[e].getY() * eo + j * len;
            const unsigned long long z = (unsigned long long)all[e].getZ() * eo + k * len;
            return ((unsigned __int128)x << 84) | ((unsigned __int128)y << 42) | (unsigned __int128)z;
        };
        std::unordered_map<unsigned __int128, unsigned int, KeyHash> best;
        best.reserve(pMesh->getNodeLocalEnd() - pMesh->getNodeLocalBegin());
        for (unsigned int cg = pMesh->getNodeLocalBegin(); cg < pMesh->getNodeLocalEnd(); cg++) {
            if (cg >= cg2dg.size() || cg2dg[cg] == LOOK_UP_TABLE_DEFAULT) continue;
            unsigned int e, i, j, k;
            pMesh->dg2eijk(cg2dg[cg], e, i, j, k);
            best.emplace(key(e, i, j, k), LOOK_UP_TABLE_DEFAULT);
        }
        auto visit = [&](unsigned int e) {
            for (unsigned int k = 0; k <= eo; k++)
                for (unsigned int j = 0; j <= eo; j++)
                    for (unsigned int i = 0; i <= eo; i++) {
                        if (pMesh->isNodeHanging(e, i, j, k)) continue;
                        auto it = best.find(key(e, i, j, k));
                        if (it == best.end()) continue;
                        const unsigned int dg = e * nPe + k * (eo + 1) * (eo + 1) + j * (eo + 1) + i;
                        unsigned int& cur = it->second;
                        if (cur == LOOK_UP_TABLE_DEFAULT) { cur = dg; continue; }
                        const unsigned int ce = cur / nPe;
                        if (all[e].getLevel() < all[ce].getLevel() ||
                            (all[e].getLevel() == all[ce].getLevel() && all[e] < all[ce]))
                            cur = dg;
                    }
        };
        for (unsigned int e = pMesh->getElementLocalBegin(); e < pMesh->getElementLocalEnd(); e++) visit(e);
        for (unsigned int e : pMesh->getRecvElementSM()) visit(e);  // ghosts the DG exchange fills
        for (unsigned int cg = pMesh->getNodeLocalBegin(); cg < pMesh->getNodeLocalEnd(); cg++) {
            if (cg >= cg2dg.size() || cg2dg[cg] == LOOK_UP_TABLE_DEFAULT) continue;
            unsigned int e, i, j, k;
            pMesh->dg2eijk(cg2dg[cg], e, i, j, k);
            const unsigned int w = best[key(e, i, j, k)];
            if (w != LOOK_UP_TABLE_DEFAULT) m_uiCgOwnerDG[cg] = w;
        }
    }
    // slots no element references (or only hanging ones): keep the mesh's own canonical
    for (unsigned int cg = pMesh->getNodeLocalBegin(); cg < pMesh->getNodeLocalEnd(); cg++)
        if (m_uiCgOwnerDG[cg] == LOOK_UP_TABLE_DEFAULT && cg < cg2dg.size()) m_uiCgOwnerDG[cg] = cg2dg[cg];
    if (pMesh->canonKeysBuilt()) {
        // labelled owner plan: it reads owners through cg2dg; the zip must
        // take the same copy. count where the by-position pick differs
        size_t diff = 0;
        for (unsigned int cg = pMesh->getNodeLocalBegin(); cg < pMesh->getNodeLocalEnd() && cg < cg2dg.size(); cg++) {
            if (m_uiCgOwnerDG[cg] != cg2dg[cg]) diff++;
            m_uiCgOwnerDG[cg] = cg2dg[cg];
        }
        std::cout << "[enuts-owner-sync r" << pMesh->getMPIRankGlobal()
                  << "] end-of-step owner pinned to cg2dg; by-position pick differed at "
                  << diff << " local cgs" << std::endl;
    }
    if (ts::EnutsProf::nanChk()) {
        const unsigned int* e2nDG = pMesh->getE2NMapping_DG().data();
        unsigned int set = 0, agree = 0, disagree = 0;
        for (unsigned int cg = pMesh->getNodeLocalBegin(); cg < pMesh->getNodeLocalEnd(); cg++) {
            if (m_uiCgOwnerDG[cg] == LOOK_UP_TABLE_DEFAULT) continue;
            set++;
            unsigned int oe, oi, oj, ok;
            pMesh->dg2eijk(m_uiCgOwnerDG[cg], oe, oi, oj, ok);
            const unsigned int nd = ok * (eo + 1) * (eo + 1) + oj * (eo + 1) + oi;
            if (e2nDG[oe * nPe + nd] == m_uiCgOwnerDG[cg] && e2n[oe * nPe + nd] == cg) { agree++; continue; }
            disagree++;
            if (disagree <= 8) {
                const unsigned int dgE = pMesh->getCG2DGMap()[cg];
                unsigned int de, di, dj, dk; pMesh->dg2eijk(dgE, de, di, dj, dk);
                const bool oeLocal = oe >= pMesh->getElementLocalBegin() && oe < pMesh->getElementLocalEnd();
                const bool deLocal = de >= pMesh->getElementLocalBegin() && de < pMesh->getElementLocalEnd();
                std::cout << "[cg-owner r" << pMesh->getMPIRankGlobal() << "]   cg " << cg << " mine " << all[oe] << " (" << oi << ',' << oj << ',' << ok << ") local=" << oeLocal
                          << "  e2nDG " << all[de] << " (" << di << ',' << dj << ',' << dk << ") local=" << deLocal
                          << " e2nDG-self=" << (e2nDG[de * nPe + dk * (eo + 1) * (eo + 1) + dj * (eo + 1) + di] == dgE)
                          << " tnless(mine<e2n)=" << (all[oe] < all[de]) << std::endl;
            }
        }
        std::cout << "[cg-owner r" << pMesh->getMPIRankGlobal() << "] local="
                  << (pMesh->getNodeLocalEnd() - pMesh->getNodeLocalBegin()) << " set=" << set
                  << " agreeE2NDG=" << agree << " disagree=" << disagree << std::endl;
    }
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::zip_labelled_via_dg() {
    ot::Mesh* pMesh = m_uiAppCtx->get_mesh();
    const unsigned int DOF  = m_uiEVar.get_dof();
    const unsigned int cgSz = pMesh->getDegOfFreedom();
    const unsigned int dgSz = pMesh->getDegOfFreedomDG();
    T* cg       = m_uiEVar.get_vec_ptr();
    ensure_evar_dg();
    const T* dg = m_uiEvarDG.get_vec_ptr();
    for (unsigned int v = 0; v < DOF; v++)
        for (unsigned int c = pMesh->getNodeLocalBegin(); c < pMesh->getNodeLocalEnd(); c++)
            if (m_uiCgOwnerDG[c] != LOOK_UP_TABLE_DEFAULT)
                cg[v * cgSz + c] = dg[v * dgSz + m_uiCgOwnerDG[c]];
}

template <typename T, typename Ctx>
bool ExplicitNUTS<T, Ctx>::prepare_owner_sync(ot::Mesh* pMesh) {
    if (m_uiOwnMesh == pMesh) return m_uiOwnOk;
    m_uiOwnPlan.clear();
    m_uiOwnRowBufRow = ~0u;
    m_uiOwnOk        = false;
    m_uiOwnMesh      = pMesh;
    if (!pMesh->isActive()) return false;

    const unsigned int nPe  = pMesh->getNumNodesPerElement();
    const unsigned int eO   = pMesh->getElementOrder();
    const unsigned int eo1  = eO + 1;
    const unsigned int nEle = pMesh->getAllElements().size();
    const unsigned int cgSz = pMesh->getDegOfFreedom();
    const int npes          = pMesh->getMPICommSize();
    const ot::TreeNode* pNodes          = pMesh->getAllElements().data();
    const unsigned int* e2n             = pMesh->getE2NMapping().data();
    const unsigned int* e2nDG           = pMesh->getE2NMapping_DG().data();
    const std::vector<unsigned int>& c2d = pMesh->getCG2DGMap();

    // labelled meshes build the plan too. Allowed only when the labelled
    // rebuild joined on the SFC source's canonical ownership keys (K6): then
    // every node has one owner element (cg2dg), locally owned nodes have a
    // local owner element and the send map holds cg indices, which is the
    // model the plan assumes. On a labelled mesh every check runs and records
    // (probe), and the outcome is agreed collectively (the per-element fallback
    // must be taken by every rank or none: the two paths use different exchanges)
    const bool probe = pMesh->useLabelledUnzip();
    bool probeBad    = false;
    // one rank: fall back to the per-element form; more: there is no fallback
    auto fail = [&](const char* why, size_t n) -> bool {
        std::cout << "[enuts-owner-sync r" << pMesh->getMPIRankGlobal()
                  << "] plan unavailable: " << why << " (" << n << ")"
                  << (probe ? "; probe" : npes > 1 ? "; aborting" : "; per-element owner sync")
                  << std::endl;
        if (probe) {
            probeBad = true;
            return false;
        }
        if (npes > 1) MPI_Abort(pMesh->getMPICommunicator(), 11);
        return false;
    };
    if (probe && !pMesh->canonKeysBuilt()) fail("labelled mesh without canonical ownership keys", 0);
    if (probe) {
        // the plan's model needs one owner slot per node and cg-indexed sends
        pMesh->ensureZipPlanBuiltPublic();
        const size_t nonPrim = pMesh->getLocalNonPrimaryNodes().size();
        if (nonPrim) fail("non-primary duplicate cg slots", nonPrim);
        size_t dgTag = 0;
        for (const unsigned char f : pMesh->getScatterMapSendIsDG()) dgTag += f;
        if (dgTag) fail("DG-tagged send-map entries", dgTag);
        size_t notInc = 0;
        for (size_t n = 1; n < c2d.size(); n++) notInc += (c2d[n] <= c2d[n - 1]);
        if (notInc) fail("cg2dg not strictly increasing", notInc);
    }
    if (c2d.size() < cgSz) return fail("cg2dg map smaller than the cg vector", c2d.size());
    if (eo1 * eo1 > 65535u) return fail("element order", eO);

    // the plan reads a node through cg2dg; it must be the owner slot the
    // element-wise zip uses (e2n_dg), for every element
    {
        size_t bad = 0;
        for (size_t i = 0; i < (size_t)nEle * nPe; i++)
            if (e2n[i] >= cgSz || c2d[e2n[i]] != e2nDG[i]) bad++;
        if (bad && !probe) return fail("e2n_dg is not cg2dg o e2n", bad);
        if (bad) fail("e2n_dg is not cg2dg o e2n", bad);
    }

    m_uiOwnDTs.clear();
    for (unsigned int l = m_uiLevMin; l <= m_uiLevMax; l++)
        m_uiOwnDTs.push_back(
            step_ratio(l));
    std::sort(m_uiOwnDTs.begin(), m_uiOwnDTs.end());
    m_uiOwnDTs.erase(std::unique(m_uiOwnDTs.begin(), m_uiOwnDTs.end()),
                     m_uiOwnDTs.end());
    const unsigned int nCls = m_uiOwnDTs.size();
    if (nCls == 0 || nCls > 250 || m_uiOwnDTs[0] < 1)
        return fail("time-step classes", nCls);
    m_uiOwnChain = true;
    for (unsigned int i = 1; i < nCls; i++)
        if (m_uiOwnDTs[i] % m_uiOwnDTs[i - 1] != 0) m_uiOwnChain = false;

    auto clsOfEle = [&](unsigned int e) -> unsigned int {
        const unsigned int l = pNodes[e].getLevel();
        if (e >= nEle || l < m_uiLevMin || l > m_uiLevMax) return 255u;
        const unsigned int dt =
            step_ratio(l);
        return (unsigned int)(std::lower_bound(m_uiOwnDTs.begin(),
                                               m_uiOwnDTs.end(), dt) -
                              m_uiOwnDTs.begin());
    };
    m_uiNodeCls.assign(cgSz, 255);
    for (unsigned int n = 0; n < cgSz; n++)
        if (c2d[n] / nPe < nEle) m_uiNodeCls[n] = (unsigned char)clsOfEle(c2d[n] / nPe);

    m_uiOwnNodeRecv.clear();
    m_uiOwnSendSlot.clear();
    m_uiOwnRecvSlot.clear();
    m_uiOwnSendOff.clear();
    m_uiOwnRecvOff.clear();
    m_uiOwnSendCum.clear();
    m_uiOwnRecvCum.clear();
    if (npes > 1) {
        const unsigned int nb = pMesh->getNodeLocalBegin();
        const unsigned int ne = pMesh->getNodeLocalEnd();
        const unsigned int eb = pMesh->getElementLocalBegin();
        const unsigned int ee = pMesh->getElementLocalEnd();
        // a node owned by this rank has its owner element here: its stage
        // value is local data, sent to the ranks that hold it as a ghost
        size_t badOwn = 0;
        std::vector<T> tmp(cgSz, (T)-1);
        for (unsigned int n = nb; n < ne; n++) {
            const unsigned int oe = c2d[n] / nPe;
            if (oe < eb || oe >= ee || m_uiNodeCls[n] == 255) badOwn++;
            tmp[n] = (T)m_uiNodeCls[n];
        }
        pMesh->readFromGhostBegin(tmp.data(), 1);
        pMesh->readFromGhostEnd(tmp.data(), 1);

        const std::vector<unsigned int>& sOff = pMesh->getNodalSendOffsets();
        const std::vector<unsigned int>& sCnt = pMesh->getNodalSendCounts();
        const std::vector<unsigned int>& rOff = pMesh->getNodalRecvOffsets();
        const std::vector<unsigned int>& rCnt = pMesh->getNodalRecvCounts();
        const std::vector<unsigned int>& sPr  = pMesh->getSendProcList();
        const std::vector<unsigned int>& rPr  = pMesh->getRecvProcList();
        const std::vector<unsigned int>& sSM  = pMesh->getSendNodeSM();
        const std::vector<unsigned int>& rSM  = pMesh->getRecvNodeSM();

        m_uiOwnNodeRecv.assign(cgSz, 0);
        size_t badSend = 0, badRecv = 0, badCls = 0, clsDiff = 0;
        std::vector<unsigned int> ord;
        // per neighbor: nodes in scatter-map order, stably sorted by class
        // (both sides sort the same sequence of classes)
        auto build = [&](const std::vector<unsigned int>& pr,
                         const std::vector<unsigned int>& off,
                         const std::vector<unsigned int>& cnt,
                         const std::vector<unsigned int>& sm, bool recv,
                         std::vector<unsigned int>& slot,
                         std::vector<unsigned int>& poff,
                         std::vector<unsigned int>& cum) {
            poff.assign(pr.size() + 1, 0);
            cum.assign(pr.size() * nCls, 0);
            for (size_t pi = 0; pi < pr.size(); pi++) {
                const unsigned int p = pr[pi];
                ord.resize(cnt[p]);
                for (unsigned int k = 0; k < cnt[p]; k++) ord[k] = sm[off[p] + k];
                for (unsigned int k = 0; k < cnt[p]; k++) {
                    const unsigned int n = ord[k];
                    if (recv) {
                        if (n >= cgSz || (n >= nb && n < ne)) {
                            badRecv++;
                            continue;
                        }
                        const T c = tmp[n];
                        if (!(c >= 0 && c < (T)nCls)) {
                            badCls++;
                            continue;
                        }
                        if (m_uiNodeCls[n] != (unsigned char)c) clsDiff++;
                        m_uiNodeCls[n]     = (unsigned char)c;
                        m_uiOwnNodeRecv[n] = 1;
                        const unsigned int oe = c2d[n] / nPe;
                        // the slot that keeps the owner's value must not be
                        // one a local block writes
                        if (oe >= eb && oe < ee) badRecv++;
                    } else if (n < nb || n >= ne) {
                        badSend++;
                    }
                }
                if (badRecv || badSend || badCls) continue;
                std::stable_sort(ord.begin(), ord.end(),
                                 [&](unsigned int a, unsigned int b) {
                                     return m_uiNodeCls[a] < m_uiNodeCls[b];
                                 });
                for (unsigned int k = 0; k < cnt[p]; k++) {
                    slot.push_back(c2d[ord[k]]);
                    cum[pi * nCls + m_uiNodeCls[ord[k]]]++;
                }
                for (unsigned int c = 1; c < nCls; c++)
                    cum[pi * nCls + c] += cum[pi * nCls + c - 1];
                poff[pi + 1] = (unsigned int)slot.size();
            }
        };
        build(sPr, sOff, sCnt, sSM, false, m_uiOwnSendSlot, m_uiOwnSendOff,
              m_uiOwnSendCum);
        build(rPr, rOff, rCnt, rSM, true, m_uiOwnRecvSlot, m_uiOwnRecvOff,
              m_uiOwnRecvCum);
        if (badOwn && !probe) return fail("local nodes whose owner element is not local", badOwn);
        if (badSend && !probe) return fail("send-map entries that are not local cg nodes", badSend);
        if (badRecv && !probe) return fail("ghost nodes kept in a local element's slot", badRecv);
        if (badCls && !probe) return fail("ghost nodes without an owner class", badCls);
        if (probe) {
            if (badOwn) fail("local nodes whose owner element is not local", badOwn);
            if (badSend) fail("send-map entries that are not local cg nodes", badSend);
            if (badRecv) fail("ghost nodes kept in a local element's slot", badRecv);
            if (badCls) fail("ghost nodes without an owner class", badCls);
            if (clsDiff) fail("ghost node class differs from the owner's", clsDiff);
        }
        if (probe) {
            int okl = probeBad ? 0 : 1, okg = 0;
            MPI_Allreduce(&okl, &okg, 1, MPI_INT, MPI_MIN, pMesh->getMPICommunicator());
            if (!okg) {
                if (!pMesh->getMPIRank())
                    std::cout << "[enuts-owner-sync] labelled mesh: plan not buildable on some rank; "
                                 "per-element sync on every rank" << std::endl;
                m_uiOwnPlan.clear();
                return false;
            }
        }
        if (probe && npes == 1 && probeBad) {
            m_uiOwnPlan.clear();
            return false;
        }
        if (std::getenv("DENDRO_ENUTS_OWNER_SYNC_INFO"))
            std::cout << "[enuts-owner-sync r" << pMesh->getMPIRankGlobal()
                      << "] classes " << nCls << " chain " << m_uiOwnChain
                      << " send nodes " << m_uiOwnSendSlot.size()
                      << " recv nodes " << m_uiOwnRecvSlot.size()
                      << " ghost nodes whose local owner had another class "
                      << clsDiff << std::endl;
    }

    // hanging face / edge interpolations per element, in the order of
    // Mesh::getElementNodalValues (same construction as Mesh::buildUnzipPlan)
    static const unsigned int faceAx[6] = {0, 0, 1, 1, 2, 2};
    static const unsigned int faceHi[6] = {0, 1, 0, 1, 0, 1};
    static const unsigned int faceDir[6] = {OCT_DIR_LEFT, OCT_DIR_RIGHT,
                                            OCT_DIR_DOWN, OCT_DIR_UP,
                                            OCT_DIR_BACK, OCT_DIR_FRONT};
    static const unsigned int edgeFree[12] = {2, 2, 1, 1, 2, 2, 1, 1, 0, 0, 0, 0};
    static const unsigned int edgeF[12][2] = {
        {0, 2}, {0, 3}, {0, 4}, {0, 5}, {1, 2}, {1, 3},
        {1, 4}, {1, 5}, {2, 4}, {2, 5}, {3, 4}, {3, 5}};
    static const unsigned int edgeDir[12] = {
        OCT_DIR_LEFT_DOWN,  OCT_DIR_LEFT_UP,     OCT_DIR_LEFT_BACK,
        OCT_DIR_LEFT_FRONT, OCT_DIR_RIGHT_DOWN,  OCT_DIR_RIGHT_UP,
        OCT_DIR_RIGHT_BACK, OCT_DIR_RIGHT_FRONT, OCT_DIR_DOWN_BACK,
        OCT_DIR_DOWN_FRONT, OCT_DIR_UP_BACK,     OCT_DIR_UP_FRONT};
    m_uiOwnSlots.clear();
    auto slot = [eo1](const unsigned int* x) {
        return x[2] * eo1 * eo1 + x[1] * eo1 + x[0];
    };
    for (unsigned int f = 0; f < 6; f++) {
        const unsigned int a  = faceAx[f];
        const unsigned int lo = (a == 0) ? 1 : 0, hi = (a == 2) ? 1 : 2;
        unsigned int x[3];
        x[a] = faceHi[f] * eO;
        for (x[hi] = 0; x[hi] < eo1; x[hi]++)
            for (x[lo] = 0; x[lo] < eo1; x[lo]++) m_uiOwnSlots.push_back(slot(x));
    }
    for (unsigned int e = 0; e < 12; e++) {
        unsigned int x[3];
        for (unsigned int s = 0; s < 2; s++)
            x[faceAx[edgeF[e][s]]] = faceHi[edgeF[e][s]] * eO;
        const unsigned int a = edgeFree[e];
        for (x[a] = 0; x[a] < eo1; x[a]++) m_uiOwnSlots.push_back(slot(x));
    }
    m_uiOwnOpOff.assign(nEle + 1, 0);
    m_uiOwnOps.clear();
    unsigned int cnum;
    for (unsigned int ele = 0; ele < nEle; ele++) {
        m_uiOwnOpOff[ele] = (unsigned int)m_uiOwnOps.size();
        bool faceHang[6];
        unsigned char cornerSet = 0;
        for (unsigned int f = 0; f < 6; f++) {
            faceHang[f] = pMesh->isFaceHanging(ele, faceDir[f], cnum);
            if (!faceHang[f]) continue;
            m_uiOwnOps.push_back((unsigned char)(f | (cnum << 5)));
            for (unsigned int c = 0; c < 8; c++)
                if (((c >> faceAx[f]) & 1u) == faceHi[f]) cornerSet |= 1u << c;
        }
        for (unsigned int e = 0; e < 12; e++) {
            if (faceHang[edgeF[e][0]] || faceHang[edgeF[e][1]]) continue;
            if (!pMesh->isEdgeHanging(ele, edgeDir[e], cnum)) continue;
            m_uiOwnOps.push_back((unsigned char)((6 + e) | (cnum << 5)));
            for (unsigned int c = 0; c < 8; c++) {
                bool on = true;
                for (unsigned int s = 0; s < 2; s++)
                    on = on && (((c >> faceAx[edgeF[e][s]]) & 1u) ==
                                faceHi[edgeF[e][s]]);
                if (on) cornerSet |= 1u << c;
            }
        }
        // an interpolated corner that is not hanging is gathered again
        for (unsigned int c = 0; c < 8; c++)
            if (((cornerSet >> c) & 1u) &&
                !pMesh->isNodeHanging(ele, (c & 1u) * eO, ((c >> 1) & 1u) * eO,
                                      ((c >> 2) & 1u) * eO))
                m_uiOwnOps.push_back((unsigned char)(18 + c));
    }
    m_uiOwnOpOff[nEle] = (unsigned int)m_uiOwnOps.size();

    m_uiOwnRowOf.assign(nEle, -1);
    m_uiOwnRowEle.clear();
    m_uiOwnRowSrc.clear();
    m_uiOwnRowN.clear();
    m_uiOwnRowOff.clear();
    m_uiOwnRowPc.clear();
    m_uiOwnConv.assign((size_t)nCls * nCls, OwnConv());
    m_uiOwnConvPass.assign(nCls, 0);
    m_uiOwnScr.assign((size_t)5 * nPe, 0);

    if (ts::EnutsProf::ownerSyncCheck()) {
        // recipe == Mesh::getElementNodalValues, bit for bit, on arbitrary data
        std::vector<T> cg(cgSz), g(nPe), a(nPe), b(nPe), pin(eo1 * eo1), pout(eo1 * eo1);
        std::vector<double> im1(nPe), im2(nPe);
        unsigned long long x = 88172645463325252ull;
        for (unsigned int n = 0; n < cgSz; n++) {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            cg[n] = (T)((double)(x >> 11) / 9007199254740992.0 - 0.5);
        }
        size_t bad = 0, hang = 0;
        for (unsigned int ele = 0; ele < nEle; ele++) {
            for (unsigned int n = 0; n < nPe; n++) g[n] = cg[e2n[(size_t)ele * nPe + n]];
            own_row_ops(ele, g.data(), a.data(), pin.data(), pout.data(), im1.data(), im2.data());
            pMesh->getElementNodalValues(cg.data(), b.data(), ele);
            hang += (m_uiOwnOpOff[ele + 1] > m_uiOwnOpOff[ele]);
            if (std::memcmp(a.data(), b.data(), sizeof(T) * nPe) != 0) bad++;
        }
        std::cout << "[enuts-owner-sync-check r" << pMesh->getMPIRankGlobal()
                  << "] elements " << nEle << " with hanging ops " << hang
                  << " recipe mismatches " << bad << std::endl;
        if (bad) MPI_Abort(pMesh->getMPICommunicator(), 12);
    }
    if (ts::EnutsProf::xchCensus()) {
        const size_t dgSz = (size_t)nEle * nPe;
        m_uiXchDg         = dgSz;
        m_uiXchFlag.assign((size_t)m_uiNumStages * dgSz, 0);
        m_uiXchInPlan.assign(dgSz, 0);
    }
    if (ts::EnutsProf::xchTrace() && npes > 1) xch_calibrate(pMesh);
    m_uiOwnOk = true;
    return true;
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::own_row_ops(unsigned int ele, const T* g, T* out,
                                       T* pin, T* pout, double* im1,
                                       double* im2) const {
    const ot::Mesh* pMesh  = m_uiAppCtx->get_mesh();
    const unsigned int nPe = pMesh->getNumNodesPerElement();
    const unsigned int eO  = pMesh->getElementOrder();
    const unsigned int eo1 = eO + 1;
    if (out != g) std::memcpy(out, g, sizeof(T) * nPe);
    for (unsigned int o = m_uiOwnOpOff[ele]; o < m_uiOwnOpOff[ele + 1]; o++) {
        const unsigned int id = m_uiOwnOps[o] & 31u;
        if (id >= 18) {
            const unsigned int c = id - 18;
            const unsigned int n = (c >> 2) * eO * eo1 * eo1 +
                                   ((c >> 1) & 1u) * eO * eo1 + (c & 1u) * eO;
            out[n] = g[n];
            continue;
        }
        const bool face        = id < 6;
        const unsigned int len = face ? eo1 * eo1 : eo1;
        const unsigned int* s  = m_uiOwnSlots.data() +
                                (face ? id * eo1 * eo1 : 6 * eo1 * eo1 + (id - 6) * eo1);
        for (unsigned int t = 0; t < len; t++) pin[t] = g[s[t]];
        pMesh->parent2ChildInterpolation(pin, pout, m_uiOwnOps[o] >> 5,
                                         face ? 2 : 1, im1, im2);
        for (unsigned int t = 0; t < len; t++) out[s[t]] = pout[t];
    }
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::build_own_plan(unsigned int blk,
                                          const std::vector<unsigned int>& eid,
                                          OwnPlan& op) {
    ot::Mesh* pMesh            = (ot::Mesh*)m_uiAppCtx->get_mesh();
    const ot::TreeNode* pNodes = pMesh->getAllElements().data();
    const ot::Block* blkList   = pMesh->getLocalBlockList().data();
    const ot::TreeNode blkNode = blkList[blk].getBlockNode();
    const unsigned int regLevel = blkList[blk].getRegularGridLev();
    const unsigned int PW      = blkList[blk].get1DPadWidth();
    const unsigned int eOrder  = pMesh->getElementOrder();
    const unsigned int n1      = eOrder + 1;
    const unsigned int nPe     = pMesh->getNumNodesPerElement();
    const unsigned int lx      = blkList[blk].getAllocationSzX();
    const unsigned int ly      = blkList[blk].getAllocationSzY();
    const unsigned int lz      = blkList[blk].getAllocationSzZ();
    const size_t lxyz          = (size_t)lx * ly * lz;
    const unsigned int cgSz    = pMesh->getDegOfFreedom();
    const unsigned int nCls    = m_uiOwnDTs.size();
    const unsigned int* e2n    = pMesh->getE2NMapping().data();
    const std::vector<unsigned int>& c2d = pMesh->getCG2DGMap();
    const bool par             = pMesh->getMPICommSize() > 1;
    const unsigned int nb      = pMesh->getNodeLocalBegin();
    const unsigned int ne      = pMesh->getNodeLocalEnd();

    if (m_uiSyncPlan.size() != pMesh->getLocalBlockList().size())
        m_uiSyncPlan.assign(pMesh->getLocalBlockList().size(), SyncPlan());
    SyncPlan& pl = m_uiSyncPlan[blk];
    if (!pl.built) build_sync_plan(blk, eid, pl);

    // last writer of every block slot: the padding fill in plan order, then
    // the block's own elements
    const unsigned int NONE = ~0u;
    std::vector<unsigned int> wEle(lxyz, NONE), wNode(lxyz, 0);
    std::vector<signed char> wC(lxyz, -1);
    std::vector<unsigned char> wOwn(lxyz, 0);  // slot of one of the block's own elements
    for (const typename SyncPlan::Ent& en : pl.ents)
        for (unsigned int q = en.b; q < en.e; q++) {
            const unsigned int d = pl.dst[q];
            wEle[d]  = en.ele;
            wC[d]    = (signed char)en.cnum;
            wNode[d] = pl.src[q];
        }
    for (const unsigned int elem : blkList[blk]) {
        const unsigned int ei = (pNodes[elem].getX() - blkNode.getX()) >> (m_uiMaxDepth - regLevel);
        const unsigned int ej = (pNodes[elem].getY() - blkNode.getY()) >> (m_uiMaxDepth - regLevel);
        const unsigned int ek = (pNodes[elem].getZ() - blkNode.getZ()) >> (m_uiMaxDepth - regLevel);
        for (unsigned int k = 0; k < n1; k++)
            for (unsigned int j = 0; j < n1; j++)
                for (unsigned int i = 0; i < n1; i++) {
                    const size_t d = (size_t)(ek * eOrder + k + PW) * (ly * lx) +
                                     (size_t)(ej * eOrder + j + PW) * lx +
                                     (ei * eOrder + i + PW);
                    wEle[d]  = elem;
                    wC[d]    = -1;
                    wNode[d] = k * n1 * n1 + j * n1 + i;
                    wOwn[d]  = 1;
                }
    }
    size_t nSelf      = 0;  // self-copy slots dropped (reported with OWNER_SYNC_INFO)

    size_t uncovered = 0, badNode = 0;
    // class of a source node; it must be a value this rank has
    auto need = [&](unsigned int cg) -> unsigned int {
        if (cg >= cgSz || m_uiNodeCls[cg] >= nCls) {
            badNode++;
            return 0;
        }
        if (par && (cg < nb || cg >= ne) && !m_uiOwnNodeRecv[cg]) uncovered++;
        return m_uiNodeCls[cg];
    };

    std::vector<std::vector<unsigned int>> bd(nCls), bs(nCls);
    struct Inst {
        unsigned int ele, o;
        std::vector<std::pair<unsigned short, unsigned int>> out;
    };
    std::vector<Inst> opI, chI;
    std::unordered_map<unsigned long long, unsigned int> opMap, chMap;
    // per element with hanging ops: for each node the op that writes it last
    // (-1: the gathered value) and its index in that op's slot list
    std::unordered_map<unsigned int, std::vector<int>> lastOp;
    auto lastOpOf = [&](unsigned int e) -> const std::vector<int>& {
        auto it = lastOp.find(e);
        if (it != lastOp.end()) return it->second;
        std::vector<int>& lo = lastOp[e];
        lo.assign(2 * nPe, -1);
        for (unsigned int o = m_uiOwnOpOff[e]; o < m_uiOwnOpOff[e + 1]; o++) {
            const unsigned int id = m_uiOwnOps[o] & 31u;
            if (id >= 18) {
                const unsigned int c = id - 18;
                lo[2 * ((c >> 2) * eOrder * n1 * n1 + ((c >> 1) & 1u) * eOrder * n1 +
                        (c & 1u) * eOrder)] = -1;
                continue;
            }
            const unsigned int len = (id < 6) ? n1 * n1 : n1;
            const unsigned int* s  = m_uiOwnSlots.data() +
                                    ((id < 6) ? id * n1 * n1 : 6 * n1 * n1 + (id - 6) * n1);
            for (unsigned int t = 0; t < len; t++) {
                lo[2 * s[t]]     = (int)(o - m_uiOwnOpOff[e]);
                lo[2 * s[t] + 1] = (int)t;
            }
        }
        return lo;
    };

    for (size_t d = 0; d < lxyz; d++) {
        const unsigned int e = wEle[d];
        if (e == NONE) continue;
        const unsigned int n = wNode[d];
        if (wC[d] >= 0) {
            const unsigned long long key = (unsigned long long)e * NUM_CHILDREN + wC[d];
            auto it = chMap.find(key);
            if (it == chMap.end()) {
                it = chMap.emplace(key, (unsigned int)chI.size()).first;
                chI.push_back({e, (unsigned int)wC[d], {}});
            }
            chI[it->second].out.emplace_back((unsigned short)n, (unsigned int)d);
            continue;
        }
        int o = -1, t = 0;
        if (m_uiOwnOpOff[e + 1] > m_uiOwnOpOff[e]) {
            const std::vector<int>& lo = lastOpOf(e);
            o = lo[2 * n];
            t = lo[2 * n + 1];
        }
        if (o < 0) {
            const unsigned int cg = e2n[(size_t)e * nPe + n];
            // the element owns this node, so the source is its own DG row =
            // this block's value (zipped from it): skip
            if (wOwn[d] && cg < cgSz && c2d[cg] == (size_t)e * nPe + n) {
                nSelf++;
                continue;
            }
            const unsigned int c  = need(cg);
            bd[c].push_back((unsigned int)d);
            bs[c].push_back(cg < cgSz ? c2d[cg] : 0);
        } else {
            const unsigned long long key = (unsigned long long)e * 32 + o;
            auto it = opMap.find(key);
            if (it == opMap.end()) {
                it = opMap.emplace(key, (unsigned int)opI.size()).first;
                opI.push_back({e, (unsigned int)o, {}});
            }
            opI[it->second].out.emplace_back((unsigned short)t, (unsigned int)d);
        }
    }

    op = OwnPlan();
    op.clsOff.assign(nCls + 1, 0);
    op.runOff.assign(nCls + 1, 0);
    op.celeOff.assign(nCls + 1, 0);
    // an owner element that feeds at least this many slots of the block is
    // converted whole (shared between blocks of a pass), fewer: per node
    const unsigned int wholeMin = std::max(8u, nPe / 6);
    std::unordered_map<unsigned int, unsigned int> cnt;
    for (unsigned int c = 0; c < nCls; c++) {
        cnt.clear();
        for (const unsigned int sl : bs[c]) cnt[sl / nPe]++;
        for (const auto& kv : cnt)
            if (kv.second >= wholeMin) op.cele.push_back(kv.first);
        std::sort(op.cele.begin() + op.celeOff[c], op.cele.end());
        op.celeOff[c + 1] = (unsigned int)op.cele.size();
        bool open = false;
        for (size_t q = 0; q < bd[c].size(); q++) {
            if (cnt[bs[c][q] / nPe] < wholeMin) {
                op.pdst.push_back(bd[c][q]);
                op.psrc.push_back(bs[c][q]);
                continue;
            }
            if (open && op.rlen.back() < 65535 &&
                op.rdst.back() + op.rlen.back() == bd[c][q] &&
                op.rsrc.back() + op.rlen.back() == bs[c][q] &&
                op.rsrc.back() / nPe == bs[c][q] / nPe) {
                op.rlen.back()++;
            } else {
                op.rdst.push_back(bd[c][q]);
                op.rsrc.push_back(bs[c][q]);
                op.rlen.push_back(1);
                open = true;
            }
        }
        op.clsOff[c + 1] = (unsigned int)op.pdst.size();
        op.runOff[c + 1] = (unsigned int)op.rdst.size();
        if (c + 1 == nCls && nSelf && std::getenv("DENDRO_ENUTS_OWNER_SYNC_INFO"))
            std::cout << "[enuts-owner-sync] blk " << blk << ": " << nSelf
                      << " self-copy slots dropped" << std::endl;
    }
    for (const Inst& in : opI) {
        const unsigned char code = m_uiOwnOps[m_uiOwnOpOff[in.ele] + in.o];
        const unsigned int id    = code & 31u;
        const unsigned int len   = (id < 6) ? n1 * n1 : n1;
        const unsigned int* s    = m_uiOwnSlots.data() +
                                  ((id < 6) ? id * n1 * n1 : 6 * n1 * n1 + (id - 6) * n1);
        typename OwnPlan::Op w;
        w.inB  = (unsigned int)op.opSrc.size();
        w.outB = (unsigned int)op.oDst.size();
        w.dim  = (id < 6) ? 2 : 1;
        w.cnum = code >> 5;
        w.segB = (unsigned int)op.opSegCls.size();
        {
            std::vector<unsigned char> ic(len);
            for (unsigned int t = 0; t < len; t++)
                ic[t] = (unsigned char)need(e2n[(size_t)in.ele * nPe + s[t]]);
            for (unsigned int c = 0; c < nCls; c++) {
                unsigned int k = 0;
                for (unsigned int t = 0; t < len; t++) {
                    if (ic[t] != c) continue;
                    const unsigned int cg = e2n[(size_t)in.ele * nPe + s[t]];
                    op.opT.push_back(t);
                    op.opSrc.push_back(cg < cgSz ? c2d[cg] : 0);
                    k++;
                }
                if (!k) continue;
                op.opSegCls.push_back((unsigned char)c);
                op.opSegEnd.push_back((unsigned short)(op.opSrc.size() - w.inB));
            }
        }
        w.segE = (unsigned int)op.opSegCls.size();
        for (const auto& pr : in.out) {
            op.oT.push_back(pr.first);
            op.oDst.push_back(pr.second);
        }
        w.outE = (unsigned int)op.oDst.size();
        op.ops.push_back(w);
    }
    // children of one parent together: its reconciled row is built once
    std::sort(chI.begin(), chI.end(), [](const Inst& a, const Inst& b) {
        return a.ele != b.ele ? a.ele < b.ele : a.o < b.o;
    });
    for (const Inst& in : chI) {
        if (m_uiOwnRowOf[in.ele] < 0) {
            m_uiOwnRowOf[in.ele] = (int)m_uiOwnRowEle.size();
            m_uiOwnRowEle.push_back(in.ele);
            std::vector<unsigned char> rc(nPe);
            for (unsigned int n = 0; n < nPe; n++)
                rc[n] = (unsigned char)need(e2n[(size_t)in.ele * nPe + n]);
            {
                const unsigned int dt = step_ratio(pNodes[in.ele].getLevel());
                m_uiOwnRowPc.push_back((unsigned char)(
                    std::lower_bound(m_uiOwnDTs.begin(), m_uiOwnDTs.end(), dt) -
                    m_uiOwnDTs.begin()));
            }
            for (unsigned int c = 0; c < nCls; c++) {
                m_uiOwnRowOff.push_back((unsigned int)m_uiOwnRowN.size());
                for (unsigned int n = 0; n < nPe; n++) {
                    if (rc[n] != c) continue;
                    const unsigned int cg = e2n[(size_t)in.ele * nPe + n];
                    // its own node: part of the element's DG row
                    if (cg < cgSz && c2d[cg] == in.ele * nPe + n &&
                        c == m_uiOwnRowPc.back())
                        continue;
                    m_uiOwnRowN.push_back(n);
                    m_uiOwnRowSrc.push_back(cg < cgSz ? c2d[cg] : 0);
                }
            }
            m_uiOwnRowOff.push_back((unsigned int)m_uiOwnRowN.size());
        }
        typename OwnPlan::Child w;
        w.row   = (unsigned int)m_uiOwnRowOf[in.ele];
        w.store = p2c_index(in.ele, in.o);
        w.cnum  = (unsigned char)in.o;
        w.outB  = (unsigned int)op.cDst.size();
        for (const auto& pr : in.out) {
            if (op.cDst.size() > w.outB && op.cLen.back() < 65535 &&
                op.cT.back() + op.cLen.back() == pr.first &&
                op.cDst.back() + op.cLen.back() == pr.second) {
                op.cLen.back()++;
            } else {
                op.cT.push_back(pr.first);
                op.cDst.push_back(pr.second);
                op.cLen.push_back(1);
            }
        }
        w.outE = (unsigned int)op.cDst.size();
        op.ch.push_back(w);
    }
    if (badNode || uncovered) {
        std::cout << "[enuts-owner-sync r" << pMesh->getMPIRankGlobal()
                  << "] blk " << blk << ": source nodes without a cg index or "
                  << "class " << badNode << ", ghost nodes no rank sends "
                  << uncovered << "; aborting" << std::endl;
        MPI_Abort(pMesh->getMPICommunicator(), 13);
    }
    if (ts::EnutsProf::xchCensus() && !m_uiXchInPlan.empty()) {
        unsigned char* ip = m_uiXchInPlan.data();
        const unsigned int bDT = m_uiEleDT[pMesh->getLocalBlockList()[blk].firstElementID()];
        const unsigned int bc  = (unsigned int)(
            std::lower_bound(m_uiOwnDTs.begin(), m_uiOwnDTs.end(), bDT) - m_uiOwnDTs.begin());
        for (const unsigned int e : op.cele)
            if (xch_whole_row(e, bc)) std::memset(ip + (size_t)e * nPe, 1, nPe);
        for (size_t r = 0; r < op.rsrc.size(); r++)
            std::memset(ip + op.rsrc[r], 1, op.rlen[r]);
        for (const unsigned int q : op.psrc) ip[q] = 1;
        for (const unsigned int q : op.opSrc) ip[q] = 1;
        for (const typename OwnPlan::Child& w : op.ch) {
            std::memset(ip + (size_t)m_uiOwnRowEle[w.row] * nPe, 1, nPe);
            const unsigned int* ro = m_uiOwnRowOff.data() + (size_t)w.row * (nCls + 1);
            for (unsigned int q = ro[0]; q < ro[nCls]; q++) ip[m_uiOwnRowSrc[q]] = 1;
        }
    }
    op.built = true;
}


template <typename T, typename Ctx>
const typename ExplicitNUTS<T, Ctx>::OwnConv* ExplicitNUTS<T, Ctx>::own_conv_rows(
    unsigned int bc, unsigned int rk_s, unsigned int bTime, unsigned int bDT) {
    SyncThr* const thr_ =
        ts::EnutsProf::tid() ? &m_uiSyncThr[ts::EnutsProf::tid()] : nullptr;
    ENUTSOp* const m_uiECOp           = thr_ ? thr_->op : this->m_uiECOp;
    std::vector<OwnConv>& m_uiOwnConv = thr_ ? thr_->ownConv : this->m_uiOwnConv;
    std::vector<unsigned long long>& m_uiOwnConvPass = thr_ ? thr_->ownConvPass : this->m_uiOwnConvPass;
    const unsigned int nCls = m_uiOwnDTs.size();
    OwnConv* cv = m_uiOwnConv.data() + (size_t)bc * nCls;
    if (m_uiOwnConvPass[bc] != m_uiSyncPass) {
        const bool ifxEq  = (ts::EnutsProf::ifaceFix() & 1);
        const bool ifxAll = (ts::EnutsProf::ifaceFix() & 2);
        for (unsigned int c = 0; c < nCls; c++) {
            const unsigned int dtl = m_uiOwnDTs[c];
            // every level steps at the multiples of its dt
            const unsigned int tl = ((m_uiOwnPt + dtl - 1) / dtl) * dtl;
            OwnConv& w            = cv[c];
            w                     = OwnConv();
            if (bTime == tl && !(ifxEq && dtl != bDT)) continue;
            const T t_blk     = m_uiTimeInfo._m_uiT + bTime * m_uiTimeInfo._m_uiTh;
            const T t_ele     = m_uiTimeInfo._m_uiT + tl * m_uiTimeInfo._m_uiTh;
            const bool eqT    = (tl == bTime);
            const bool eleFut = eqT ? (dtl > bDT) : (tl > bTime);
            const unsigned int nsIn =
                (ifxAll && eleFut && !eqT) ? m_uiNumStages : rk_s;
            T dt_c = (eleFut ? dtl : bDT) * (m_uiTimeInfo._m_uiTh);
            T dt_f = (eleFut ? bDT : dtl) * (m_uiTimeInfo._m_uiTh);
            if (dt_c < dt_f) std::swap(dt_c, dt_f);
            T dt = eqT ? 0 : (eleFut ? (t_ele - t_blk) : (t_blk - t_ele));
            if (ts::EnutsProf::ifaceShift() && eleFut && !eqT)
                dt = (T)(bTime + dtl - tl) * m_uiTimeInfo._m_uiTh;
            w.same     = 0;
            w.ns       = (unsigned char)nsIn;
            if (m_uiStGOn && (eqT || nsIn == m_uiNumStages)) {
                T row[5];
                if (m_uiECOp->stageRowNL(eleFut, row, rk_s, dt_c, dt_f, dt, nsIn)) {
                    w.g       = 1;
                    row[nsIn] = row[4];
                    w.ns      = (unsigned char)(nsIn + 1);
                }
                for (unsigned int k = 0; k < w.ns; k++) w.m[k] = row[k];
            } else
            m_uiECOp->stageRow(eleFut, w.m, rk_s, dt_c, dt_f, dt);
        }
        m_uiOwnConvPass[bc] = m_uiSyncPass;
    }
    return cv;
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::own_row_build(unsigned int row_, unsigned int rk_s,
                                         const OwnConv* cv, T* rowBuf, T* g, T* pin,
                                         T* pout, double* im1, double* im2) {
    SyncThr* const thr_ =
        ts::EnutsProf::tid() ? &m_uiSyncThr[ts::EnutsProf::tid()] : nullptr;
    ENUTSOp* const m_uiECOp = thr_ ? thr_->op : this->m_uiECOp;
    const ot::Mesh* pMesh   = m_uiAppCtx->get_mesh();
    const unsigned int nPe  = pMesh->getNumNodesPerElement();
    const size_t dgSz       = (size_t)pMesh->getAllElements().size() * nPe;
    const unsigned int dof  = m_uiEVar.get_dof();
    const unsigned int nCls = m_uiOwnDTs.size();
    const T* S[8];
    const T* SG[8];
    const T* stG = m_uiStGOn ? m_uiStG.get_vec_ptr() : nullptr;
    struct { unsigned int row; } w = {row_};
    const unsigned int ele  = m_uiOwnRowEle[w.row];
    const unsigned int* ro  = m_uiOwnRowOff.data() + (size_t)w.row * (nCls + 1);
    const bool hasOps       = m_uiOwnOpOff[ele + 1] > m_uiOwnOpOff[ele];
    if (ts::EnutsProf::xchCensus() && !m_uiXchFlag.empty()) {
        xch_mark_ele(ele, nPe, cv[m_uiOwnRowPc[w.row]], rk_s);
        for (unsigned int c = 0; c < nCls; c++)
            for (unsigned int q = ro[c]; q < ro[c + 1]; q++)
                xch_mark(m_uiOwnRowSrc[q], cv[c], rk_s);
    }
    for (unsigned int v = 0; v < dof; v++) {
        for (unsigned int k = 0; k < m_uiNumStages; k++)
            SG[k] = S[k] = m_uiStVec[k].get_vec_ptr() + (size_t)v * dgSz;
        if (stG) SG[rk_s] = stG + (size_t)v * dgSz;
        T* row = rowBuf + (size_t)v * nPe;
        T* gg  = hasOps ? g : row;
        {
            // the element's own DG row in the block's frame, then the
            // nodes other elements own
            const OwnConv& pc = cv[m_uiOwnRowPc[w.row]];
            if (pc.same) {
                std::memcpy(gg, S[rk_s - 1] + (size_t)ele * nPe, sizeof(T) * nPe);
            } else {
                const T* Se[8];
                for (unsigned int k = 0; k < m_uiNumStages; k++)
                    Se[k] = S[k] + (size_t)ele * nPe;
                if (pc.g) Se[pc.ns - 1] = stG + (size_t)v * dgSz + (size_t)ele * nPe;
                if (ts::EnutsProf::on() && pc.g && !v) {
                    ts::EnutsProf::get().t[ts::EnutsProf::CN_GALL] += nPe;
                    if (!gValidCls(m_uiOwnRowPc[w.row])) ts::EnutsProf::get().t[ts::EnutsProf::CN_G0] += nPe;
                }
                m_uiECOp->applyRow(gg, Se, pc.m, pc.ns, nPe);
            }
        }
        for (unsigned int c = 0; c < nCls; c++) {
            const unsigned int n = ro[c + 1] - ro[c];
            if (!n) continue;
            const unsigned int* rn = m_uiOwnRowN.data() + ro[c];
            const unsigned int* rs = m_uiOwnRowSrc.data() + ro[c];
            if (cv[c].same) {
                const T* sv = S[rk_s - 1];
                for (unsigned int q = 0; q < n; q++) gg[rn[q]] = sv[rs[q]];
            } else {
                m_uiECOp->gatherRow(gg, rn, rs, n, cv[c].g ? SG : S, cv[c].m, cv[c].ns);
                if (ts::EnutsProf::on() && cv[c].g && !v) {
                    ts::EnutsProf::get().t[ts::EnutsProf::CN_GALL] += n;
                    if (!gValidCls(c)) ts::EnutsProf::get().t[ts::EnutsProf::CN_G0] += n;
                }
            }
        }
        if (hasOps) {
            const bool rprof = ts::EnutsProf::on();
            const double to0 = rprof ? ts::EnutsProf::now() : 0.0;
            own_row_ops(ele, g, row, pin, pout, im1, im2);
            if (rprof) ts::EnutsProf::get().t[ts::EnutsProf::SY_ROWOPS] += ts::EnutsProf::now() - to0;
        }
    }
    if (ts::EnutsProf::on()) {
        ts::EnutsProf::get().t[ts::EnutsProf::CN_ROWS] += 1;
        ts::EnutsProf::get().t[ts::EnutsProf::CN_ROWS_OPS] += hasOps;
    }
}
template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::own_p2c_prepass(unsigned int rk_s, const unsigned int* actIds,
                                           unsigned int nAct, bool thr) {
    ot::Mesh* pMesh          = (ot::Mesh*)m_uiAppCtx->get_mesh();
    const ot::Block* blkList = pMesh->getLocalBlockList().data();
    const unsigned int nPe   = pMesh->getNumNodesPerElement();
    const unsigned int dof   = m_uiEVar.get_dof();
    const size_t stride      = (size_t)dof * nPe;
    const unsigned int nCls  = m_uiOwnDTs.size();
    m_uiP2cSharedOn = false;
    if (m_uiOwnPlan.size() != pMesh->getLocalBlockList().size())
        m_uiOwnPlan.assign(pMesh->getLocalBlockList().size(), OwnPlan());
    SyncTag key;
    key.pass = m_uiSyncPass;
    key.rk   = rk_s;
    key.fut  = 3;
    // 1. the children of the active blocks, one entry per reader block, grouped by parent row
    struct Item { unsigned int row, store, blk, outB, outE; unsigned char cnum; };
    std::vector<Item> items;
    size_t slots = 0;
    for (unsigned int bb = 0; bb < nAct; bb++) {
        const unsigned int blk = actIds[bb];
        OwnPlan& op            = m_uiOwnPlan[blk];
        if (!op.built) {
            sync_lazy_build(true);
            build_own_plan(blk, m_uiBlkDepEid[blk], op);
        }
        for (const typename OwnPlan::Child& w : op.ch) {
            if (m_uiP2cPassOf[w.store] != m_uiSyncPass) {
                m_uiP2cPassOf[w.store] = m_uiSyncPass;
                m_uiP2cSlotOf[w.store] = ~0u;
                m_uiP2cTagOf[w.store]  = key;
                slots++;
            }
            items.push_back({w.row, w.store, blk, w.outB, w.outE, w.cnum});
        }
    }
    if (ts::EnutsProf::on()) {
        // distinct children of this coarse step and the store a hoisted p2c
        // (one interpolated child per contributing class and stage) would need
        if (m_uiP2cStepOf.size() != m_uiP2cPassOf.size()) m_uiP2cStepOf.assign(m_uiP2cPassOf.size(), 0);
        double nd = 0, dbl = 0;
        for (const Item& it : items) {
            if (m_uiP2cStepOf[it.store] == m_uiCoarseStep) continue;
            m_uiP2cStepOf[it.store] = m_uiCoarseStep;
            const unsigned int* ro = m_uiOwnRowOff.data() + (size_t)it.row * (nCls + 1);
            unsigned int cls = 0;
            for (unsigned int c = 0; c < nCls; c++)
                cls += (ro[c + 1] > ro[c] || c == m_uiOwnRowPc[it.row]);
            nd += 1;
            dbl += (double)cls * (m_uiNumStages + 1) * stride;
        }
        ts::EnutsProf::get().t[ts::EnutsProf::CN_P2C_DISTINCT] += nd;
        ts::EnutsProf::get().t[ts::EnutsProf::CN_HOIST_DBL] += dbl;
    }
    if (!slots) return;
    // by row, then by child: the readers of one child are contiguous
    std::stable_sort(items.begin(), items.end(), [](const Item& a, const Item& b) {
        return a.row != b.row ? a.row < b.row : a.store < b.store;
    });
    m_uiP2cWorkRow.clear();
    m_uiP2cWorkBlk.clear();
    m_uiP2cWorkOff.assign(1, 0);
    m_uiP2cWorkStore.resize(items.size());
    m_uiP2cWorkCnum.resize(items.size());
    m_uiP2cWorkRdr.resize(items.size());
    m_uiP2cWorkOutB.resize(items.size());
    m_uiP2cWorkOutE.resize(items.size());
    for (size_t i = 0; i < items.size(); i++) {
        if (i == 0 || items[i].row != items[i - 1].row) {
            m_uiP2cWorkRow.push_back(items[i].row);
            m_uiP2cWorkBlk.push_back(items[i].blk);
            if (i) m_uiP2cWorkOff.push_back((unsigned int)i);
        }
        m_uiP2cWorkStore[i] = items[i].store;
        m_uiP2cWorkCnum[i]  = items[i].cnum;
        m_uiP2cWorkRdr[i]   = items[i].blk;
        m_uiP2cWorkOutB[i]  = items[i].outB;
        m_uiP2cWorkOutE[i]  = items[i].outE;
    }
    m_uiP2cWorkOff.push_back((unsigned int)items.size());
    m_uiP2cPoolPass  = m_uiSyncPass;
    m_uiP2cUsed      = 0;
    m_uiP2cSharedOn  = true;
    const unsigned int nRows = (unsigned int)m_uiP2cWorkRow.size();
    // 2. each unique row once: its reconciled row, then its children into the reader blocks
#ifdef DENDRO_ENUTS_HAS_OMP
#pragma omp parallel for schedule(dynamic, 1) if (thr)
#endif
    for (unsigned int r = 0; r < nRows; r++) {
        SyncThr* const thr_ =
            ts::EnutsProf::tid() ? &m_uiSyncThr[ts::EnutsProf::tid()] : nullptr;
        std::vector<T>& rowBufV   = thr_ ? thr_->ownRowBuf : this->m_uiOwnRowBuf;
        std::vector<T>& scrV      = thr_ ? thr_->ownScr : this->m_uiOwnScr;
        std::vector<T>& sweepV    = thr_ ? thr_->ownSweep : this->m_uiOwnSweep;
        std::vector<unsigned char>& sweepOkV = thr_ ? thr_->ownSweepOk : this->m_uiOwnSweepOk;
        if (rowBufV.size() < stride) rowBufV.resize(stride);
        if (sweepV.size() < 6 * stride) sweepV.resize(6 * stride);
        if (sweepOkV.size() != (size_t)6 * dof) sweepOkV.assign((size_t)6 * dof, 0);
        T* scr      = scrV.data();
        T* g        = scr;
        T* pin      = scr + nPe;
        T* pout     = scr + 2 * nPe;
        double* im1 = (double*)(scr + 3 * nPe);
        double* im2 = (double*)(scr + 4 * nPe);
        const unsigned int blk   = m_uiP2cWorkBlk[r];
        const unsigned int bTime = m_uiEleTime[blkList[blk].firstElementID()];
        const unsigned int bDT   = m_uiEleDT[blkList[blk].firstElementID()];
        const unsigned int bc    = (unsigned int)(
            std::lower_bound(m_uiOwnDTs.begin(), m_uiOwnDTs.end(), bDT) - m_uiOwnDTs.begin());
        const OwnConv* cv = own_conv_rows(bc, rk_s, bTime, bDT);
        const bool pprof  = ts::EnutsProf::on();
        const double tr0  = pprof ? ts::EnutsProf::now() : 0.0;
        own_row_build(m_uiP2cWorkRow[r], rk_s, cv, rowBufV.data(), g, pin, pout, im1, im2);
        if (pprof) ts::EnutsProf::get().t[ts::EnutsProf::SY_ROW] += ts::EnutsProf::now() - tr0;
        std::fill(sweepOkV.begin(), sweepOkV.end(), 0);
        // one child at a time into pout, then its output runs into every reader block
        std::vector<T>& childV = thr_ ? thr_->ownRowBuf2 : this->m_uiOwnRowBuf2;
        if (childV.size() < stride) childV.resize(stride);
        unsigned int i = m_uiP2cWorkOff[r];
        while (i < m_uiP2cWorkOff[r + 1]) {
            const unsigned int st = m_uiP2cWorkStore[i];
            const double tp0 = pprof ? ts::EnutsProf::now() : 0.0;
            for (unsigned int v = 0; v < dof; v++)
                p2c3_shared(pMesh, rowBufV.data() + (size_t)v * nPe, childV.data() + (size_t)v * nPe,
                            m_uiP2cWorkCnum[i], sweepV.data() + (size_t)v * 6 * nPe,
                            sweepOkV.data() + (size_t)v * 6);
            const double tc0 = pprof ? ts::EnutsProf::now() : 0.0;
            if (pprof) ts::EnutsProf::get().t[ts::EnutsProf::SY_P2C] += tc0 - tp0;
            for (; i < m_uiP2cWorkOff[r + 1] && m_uiP2cWorkStore[i] == st; i++) {
                const unsigned int rb = m_uiP2cWorkRdr[i];
                const OwnPlan& rop    = m_uiOwnPlan[rb];
                T* blkVec = (T*)m_uiBVec[rb]._vec[stage_slot(rk_s)].data();
                const size_t lxyz = (size_t)blkList[rb].getAllocationSzX() *
                                    blkList[rb].getAllocationSzY() *
                                    blkList[rb].getAllocationSzZ();
                for (unsigned int v = 0; v < dof; v++) {
                    T* dv       = blkVec + (size_t)v * lxyz;
                    const T* pv = childV.data() + (size_t)v * nPe;
                    for (unsigned int q = m_uiP2cWorkOutB[i]; q < m_uiP2cWorkOutE[i]; q++) {
                        const T* sp = pv + rop.cT[q];
                        T* dp       = dv + rop.cDst[q];
                        for (unsigned int k = 0; k < rop.cLen[q]; k++) dp[k] = sp[k];
                    }
                }
            }
            if (pprof) ts::EnutsProf::get().t[ts::EnutsProf::SY_CPY] += ts::EnutsProf::now() - tc0;
        }
        (void)nCls;
    }
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::own_sync_fill(unsigned int blk, unsigned int rk_s,
                                         const std::vector<unsigned int>& eid) {
    // [ENUTS_OMP begin] worker threads (tid >= 1) of the threaded sync loop:
    // the locals below shadow the shared scratch / per-pass caches for the
    // rest of this function (see SyncThr). the plans were built by
    // sync_prebuild(). thr_ == nullptr = serial / thread 0.
    SyncThr* const thr_ =
        ts::EnutsProf::tid() ? &m_uiSyncThr[ts::EnutsProf::tid()] : nullptr;
    ENUTSOp* const m_uiECOp           = thr_ ? thr_->op : this->m_uiECOp;
    std::vector<OwnConv>& m_uiOwnConv = thr_ ? thr_->ownConv : this->m_uiOwnConv;
    std::vector<unsigned long long>& m_uiOwnConvPass = thr_ ? thr_->ownConvPass : this->m_uiOwnConvPass;
    std::vector<T>& m_uiOwnScr        = thr_ ? thr_->ownScr : this->m_uiOwnScr;
    std::vector<SyncTag>& m_uiSyncTag = thr_ ? thr_->tag : this->m_uiSyncTag;
    std::vector<T>& m_uiOwnSweep      = thr_ ? thr_->ownSweep : this->m_uiOwnSweep;
    std::vector<unsigned char>& m_uiOwnSweepOk = thr_ ? thr_->ownSweepOk : this->m_uiOwnSweepOk;
    std::vector<T>& m_uiOwnRowBuf     = thr_ ? thr_->ownRowBuf : this->m_uiOwnRowBuf;
    unsigned int& m_uiOwnRowBufRow    = thr_ ? thr_->ownRowBufRow : this->m_uiOwnRowBufRow;
    unsigned long long& m_uiOwnRowBufPass = thr_ ? thr_->ownRowBufPass : this->m_uiOwnRowBufPass;
    // [ENUTS_OMP end]
    ot::Mesh* pMesh          = (ot::Mesh*)m_uiAppCtx->get_mesh();
    const ot::Block* blkList = pMesh->getLocalBlockList().data();
    const unsigned int nPe   = pMesh->getNumNodesPerElement();
    const unsigned int n1    = pMesh->getElementOrder() + 1;
    const size_t dgSz        = (size_t)pMesh->getAllElements().size() * nPe;
    const size_t lxyz        = (size_t)blkList[blk].getAllocationSzX() *
                               blkList[blk].getAllocationSzY() *
                               blkList[blk].getAllocationSzZ();
    const unsigned int dof   = m_uiEVar.get_dof();
    const unsigned int nCls  = m_uiOwnDTs.size();
    const bool sprof         = ts::EnutsProf::on();
    const double t0          = sprof ? ts::EnutsProf::now() : 0.0;
    double tp2c              = 0;

    if (m_uiOwnPlan.size() != pMesh->getLocalBlockList().size())
        m_uiOwnPlan.assign(pMesh->getLocalBlockList().size(), OwnPlan());
    OwnPlan& op = m_uiOwnPlan[blk];
    if (!op.built) build_own_plan(blk, eid, op);

    const unsigned int bTime = m_uiEleTime[blkList[blk].firstElementID()];
    const unsigned int bDT   = m_uiEleDT[blkList[blk].firstElementID()];
    const unsigned int bc    = (unsigned int)(
        std::lower_bound(m_uiOwnDTs.begin(), m_uiOwnDTs.end(), bDT) - m_uiOwnDTs.begin());
    assert(bc < nCls && m_uiOwnDTs[bc] == bDT);

    // one conversion row per owner class for this block class and stage pass:
    // what the per-element loop decides from (element time, element dt)
    const OwnConv* cv = own_conv_rows(bc, rk_s, bTime, bDT);
    if (ts::EnutsProf::xchCensus() && !m_uiXchFlag.empty()) {
        // the stage slots this call reads (rows: in own_row_build)
        for (unsigned int c = 0; c < nCls; c++) {
            const OwnConv& w = cv[c];
            if (!w.same)
                for (unsigned int i = op.celeOff[c]; i < op.celeOff[c + 1]; i++)
                    xch_mark_ele(op.cele[i], nPe, w, rk_s);
            else
                for (unsigned int r = op.runOff[c]; r < op.runOff[c + 1]; r++)
                    for (unsigned int i = 0; i < op.rlen[r]; i++)
                        xch_mark(op.rsrc[r] + i, w, rk_s);
            for (unsigned int q = op.clsOff[c]; q < op.clsOff[c + 1]; q++)
                xch_mark(op.psrc[q], w, rk_s);
        }
        for (const typename OwnPlan::Op& w : op.ops) {
            unsigned int b = 0;
            for (unsigned int sg = w.segB; sg < w.segE; sg++) {
                const OwnConv& c     = cv[op.opSegCls[sg]];
                const unsigned int e = op.opSegEnd[sg];
                for (unsigned int t = b; t < e; t++) xch_mark(op.opSrc[w.inB + t], c, rk_s);
                b = e;
            }
        }
    }

    T* blkVec = (T*)m_uiBVec[blk]._vec[stage_slot(rk_s)].data();
    // SG: the inputs of a class that reads g (k1 .. k_rk_s, g)
    const T* S[8];
    const T* SG[8];
    const T* stG = m_uiStGOn ? m_uiStG.get_vec_ptr() : nullptr;
    T* scr      = m_uiOwnScr.data();  // g, pin, pout | im1, im2
    T* g        = scr;
    T* pin      = scr + nPe;
    T* pout     = scr + 2 * nPe;
    double* im1 = (double*)(scr + 3 * nPe);
    double* im2 = (double*)(scr + 4 * nPe);

    // owner elements of a converted class that feed many slots: the whole
    // element into m_uiDGTmp0, unless a block of this pass and class did it
    // [ENUTS_OMP] per-thread stand-in for the shared work vector
    T* dgW = thr_ ? thr_->dgW.get() : m_uiDGTmp0.get_vec_ptr();
    if (m_uiSyncTag.size() != pMesh->getAllElements().size())
        m_uiSyncTag.assign(pMesh->getAllElements().size(), SyncTag());
    SyncTag etag;
    etag.pass = m_uiSyncPass;
    etag.rk   = rk_s;
    etag.fut  = 4;
    etag.dt_c = (T)bc;
    unsigned int nconv = 0, nreuse = 0;
    for (unsigned int c = 0; c < nCls; c++) {
        if (cv[c].same) continue;
        for (unsigned int i = op.celeOff[c]; i < op.celeOff[c + 1]; i++) {
            const unsigned int ele = op.cele[i];
            if (m_uiSyncPassOpen && m_uiSyncTag[ele] == etag) {
                nreuse++;
                continue;
            }
            nconv++;
            for (unsigned int v = 0; v < dof; v++) {
                for (unsigned int k = 0; k < m_uiNumStages; k++)
                    S[k] = m_uiStVec[k].get_vec_ptr() + (size_t)v * dgSz + (size_t)ele * nPe;
                if (cv[c].g) S[cv[c].ns - 1] = stG + (size_t)v * dgSz + (size_t)ele * nPe;
                if (sprof && cv[c].g && !v) {
                    ts::EnutsProf::get().t[ts::EnutsProf::CN_GALL] += nPe;
                    if (!gValidCls(c)) ts::EnutsProf::get().t[ts::EnutsProf::CN_G0] += nPe;
                }
                m_uiECOp->applyRow(dgW + (size_t)v * dgSz + (size_t)ele * nPe, S,
                                   cv[c].m, cv[c].ns, nPe);
            }
            m_uiSyncTag[ele] = m_uiSyncPassOpen ? etag : SyncTag();
        }
    }
    const double t1 = sprof ? ts::EnutsProf::now() : 0.0;
    double tops     = 0;

    for (unsigned int v = 0; v < dof; v++) {
        for (unsigned int k = 0; k < m_uiNumStages; k++)
            SG[k] = S[k] = m_uiStVec[k].get_vec_ptr() + (size_t)v * dgSz;
        if (stG) SG[rk_s] = stG + (size_t)v * dgSz;
        T* dv = blkVec + (size_t)v * lxyz;
        for (unsigned int c = 0; c < nCls; c++) {
            if (op.runOff[c + 1] > op.runOff[c]) {
                const T* base = cv[c].same ? S[rk_s - 1]
                                           : (const T*)(dgW + (size_t)v * dgSz);
                const unsigned int* rd   = op.rdst.data();
                const unsigned int* rs   = op.rsrc.data();
                const unsigned short* rl = op.rlen.data();
                for (unsigned int r = op.runOff[c]; r < op.runOff[c + 1]; r++) {
                    const T* sp = base + rs[r];
                    T* dp       = dv + rd[r];
                    for (unsigned int i = 0; i < rl[r]; i++) dp[i] = sp[i];
                }
            }
            const unsigned int b = op.clsOff[c], n = op.clsOff[c + 1] - b;
            if (!n) continue;
            const unsigned int* dp = op.pdst.data() + b;
            const unsigned int* sp = op.psrc.data() + b;
            if (cv[c].same) {
                const T* sv = S[rk_s - 1];
                for (unsigned int q = 0; q < n; q++) dv[dp[q]] = sv[sp[q]];
            } else {
                m_uiECOp->gatherRow(dv, dp, sp, n, cv[c].g ? SG : S, cv[c].m, cv[c].ns);
                if (sprof && cv[c].g && !v) {
                    ts::EnutsProf::get().t[ts::EnutsProf::CN_GALL] += n;
                    if (!gValidCls(c)) ts::EnutsProf::get().t[ts::EnutsProf::CN_G0] += n;
                }
            }
        }
        const double to0 = sprof ? ts::EnutsProf::now() : 0.0;
        for (const typename OwnPlan::Op& w : op.ops) {
            const unsigned int len    = (w.dim == 2) ? n1 * n1 : n1;
            const unsigned int* src = op.opSrc.data() + w.inB;
            const unsigned int* ti  = op.opT.data() + w.inB;
            unsigned int b          = 0;
            for (unsigned int sg = w.segB; sg < w.segE; sg++) {
                const OwnConv& c     = cv[op.opSegCls[sg]];
                const unsigned int e = op.opSegEnd[sg];
                if (c.same) {
                    const T* sv = S[rk_s - 1];
                    for (unsigned int t = b; t < e; t++) pin[ti[t]] = sv[src[t]];
                } else {
                    m_uiECOp->gatherRow(pin, ti + b, src + b, e - b, c.g ? SG : S, c.m, c.ns);
                    if (sprof && c.g && !v) {
                        ts::EnutsProf::get().t[ts::EnutsProf::CN_GALL] += e - b;
                        if (!gValidCls(op.opSegCls[sg])) ts::EnutsProf::get().t[ts::EnutsProf::CN_G0] += e - b;
                    }
                }
                b = e;
            }
            pMesh->parent2ChildInterpolation(pin, pout, w.cnum, w.dim, im1, im2);
            for (unsigned int q = w.outB; q < w.outE; q++)
                dv[op.oDst[q]] = pout[op.oT[q]];
        }
        if (sprof) tops += ts::EnutsProf::now() - to0;
    }
    const double t2 = sprof ? ts::EnutsProf::now() : 0.0;

    const size_t stride = (size_t)dof * nPe;
    if (m_uiOwnSweep.size() < 6 * stride) m_uiOwnSweep.resize(6 * stride);
    if (m_uiOwnSweepOk.size() != (size_t)6 * dof) m_uiOwnSweepOk.assign((size_t)6 * dof, 0);
    SyncTag key;
    key.pass = m_uiSyncPass;
    key.rk   = rk_s;
    key.fut  = 3;
    unsigned int nhit = 0;
    double trow = 0, tcpy = 0;
    for (const typename OwnPlan::Child& w : op.ch) {
        bool hit = false;
        T* sd    = m_uiSyncPassOpen ? p2c_slot(w.store, key, stride, hit) : nullptr;
        nhit += hit;
        if (hit && !sd) continue;  // the pre-pass wrote this child
        if (!hit && !(m_uiOwnRowBufRow == w.row && m_uiOwnRowBufPass == m_uiSyncPass &&
                      m_uiSyncPassOpen)) {
            const double tr0 = sprof ? ts::EnutsProf::now() : 0.0;
            // the dependent's row as its zip + gather gives it: owner values,
            // then its own hanging faces and edges
            if (m_uiOwnRowBuf.size() < stride) m_uiOwnRowBuf.resize(stride);
            own_row_build(w.row, rk_s, cv, m_uiOwnRowBuf.data(), g, pin, pout, im1, im2);
            m_uiOwnRowBufRow  = w.row;
            m_uiOwnRowBufPass = m_uiSyncPass;
            std::fill(m_uiOwnSweepOk.begin(), m_uiOwnSweepOk.end(), 0);
            if (sprof) trow += ts::EnutsProf::now() - tr0;
        }
        if (!hit) {
            const double tp = sprof ? ts::EnutsProf::now() : 0.0;
            for (unsigned int v = 0; v < dof; v++) {
                T* dv = blkVec + (size_t)v * lxyz;
                T* pv = sd ? sd + (size_t)v * nPe : pout;
                p2c3_shared(pMesh, m_uiOwnRowBuf.data() + (size_t)v * nPe, pv, w.cnum,
                            m_uiOwnSweep.data() + (size_t)v * 6 * nPe,
                            m_uiOwnSweepOk.data() + (size_t)v * 6);
                if (!sd)
                    for (unsigned int q = w.outB; q < w.outE; q++) {
                        const T* sp = pv + op.cT[q];
                        T* dp       = dv + op.cDst[q];
                        for (unsigned int i = 0; i < op.cLen[q]; i++) dp[i] = sp[i];
                    }
            }
            if (sprof) tp2c += ts::EnutsProf::now() - tp;
            if (!sd) continue;
        }
        const double tc0 = sprof ? ts::EnutsProf::now() : 0.0;
        for (unsigned int v = 0; v < dof; v++) {
            T* dv       = blkVec + (size_t)v * lxyz;
            const T* pv = sd + (size_t)v * nPe;
            for (unsigned int q = w.outB; q < w.outE; q++) {
                const T* sp = pv + op.cT[q];
                T* dp       = dv + op.cDst[q];
                for (unsigned int i = 0; i < op.cLen[q]; i++) dp[i] = sp[i];
            }
        }
        if (sprof) tcpy += ts::EnutsProf::now() - tc0;
    }

    if (sprof) {
        ts::EnutsProf& P = ts::EnutsProf::get();
        const double t3 = ts::EnutsProf::now();
        // plan path: same_copy = plan + rows + whole-element conversions,
        // corr_coarse_fut = node copies, corr_fine_past = hanging faces/edges,
        // unzip_pad = children of coarser dependents (p2c inside)
        P.t[ts::EnutsProf::SY_SAME] += t1 - t0;
        P.t[ts::EnutsProf::SY_CORR] += (t2 - t1) - tops;
        P.t[ts::EnutsProf::SY_CORR_FC] += tops;
        P.t[ts::EnutsProf::SY_UNZIP] += t3 - t2;
        P.t[ts::EnutsProf::SY_P2C] += tp2c;
        P.t[ts::EnutsProf::SY_ROW] += trow;
        P.t[ts::EnutsProf::SY_CPY] += tcpy;
        P.t[ts::EnutsProf::CN_ENT] += op.ops.size();
        P.t[ts::EnutsProf::CN_CORR] += nconv;
        P.t[ts::EnutsProf::CN_HIT] += nreuse;
        size_t nrun = 0;
        for (const unsigned short l : op.rlen) nrun += l;
        size_t ncr = 0;
        for (const unsigned short l : op.cLen) ncr += l;
        P.t[ts::EnutsProf::CN_PAIRS] += nrun + op.pdst.size() + op.oDst.size() + ncr;
        P.t[ts::EnutsProf::CN_P2C] += op.ch.size();
        P.t[ts::EnutsProf::CN_P2C_HIT] += nhit;
    }
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::own_node_exchange_begin(T* stv, unsigned int dof,
                                                   unsigned int pt) {
    ot::Mesh* pMesh = (ot::Mesh*)m_uiAppCtx->get_mesh();
    if (!pMesh->isActive() || pMesh->getMPICommSize() < 2) return;
    const unsigned int nCls = m_uiOwnDTs.size();
    const size_t dgSz = (size_t)pMesh->getAllElements().size() * pMesh->getNumNodesPerElement();
    // the owners that computed this stage are the classes whose dt divides pt:
    // a prefix of the ascending classes when the dts form a divisibility chain
    int j = -1;
    if (!m_uiOwnChain)
        j = (int)nCls - 1;
    else
        for (unsigned int i = 0; i < nCls; i++)
            if (pt % m_uiOwnDTs[i] == 0) j = (int)i;
    const std::vector<unsigned int>& sPr = pMesh->getSendProcList();
    const std::vector<unsigned int>& rPr = pMesh->getRecvProcList();
    MPI_Comm comm = pMesh->getMPICommunicator();
    const int tag = 7153;
    m_uiOwnReq.assign(sPr.size() + rPr.size(), MPI_REQUEST_NULL);
    if (j < 0) return;
    if (!m_uiXchScoped && m_uiOwnChain) xch_scope_build(pMesh);
    auto nn = [&](const unsigned int* cum) { return cum[j]; };
    if (ts::EnutsProf::xchCensus() && !m_uiXchFlag.empty()) {
        // what became of the previous message into these slots (this stage):
        // read by a plan since, or overwritten unread; never-read slots no
        // plan lists are the dead weight of the map itself
        unsigned int s = 0;
        for (unsigned int k = 0; k < m_uiNumStages; k++)
            if (m_uiStVec[k].get_vec_ptr() == stv) s = k;
        const unsigned char* fl = m_uiXchFlag.data() + (size_t)s * dgSz;
        double nRd = 0, nUn = 0, nNo = 0;
        std::vector<double> unC(nCls, 0), rdC(nCls, 0);
        for (size_t i = 0; i < rPr.size(); i++) {
            const unsigned int cnt = m_uiOwnRecvCum[i * nCls + j];
            const unsigned int* sl = m_uiOwnRecvSlot.data() + m_uiOwnRecvOff[i];
            const unsigned int* cum = m_uiOwnRecvCum.data() + i * nCls;
            unsigned int c = 0;
            for (unsigned int q = 0; q < cnt; q++) {
                while (q >= cum[c]) c++;
                const unsigned char f = fl[sl[q]];
                if (!f) continue;
                if (!m_uiXchInPlan[sl[q]])
                    nNo++;
                else if (f == 1) {
                    nUn++;
                    unC[c]++;
                } else {
                    nRd++;
                    rdC[c]++;
                }
            }
        }
        if (ts::EnutsProf::xchCensusLvl() >= 2 && !pMesh->getMPIRankGlobal()) {
            std::cout << "[xch-census r0] pt " << pt << " s " << s + 1 << " j " << j << " read " << nRd
                      << " unread " << nUn << " noplan " << nNo << " | per class read/unread:";
            for (unsigned int c = 0; c < nCls; c++) std::cout << " " << rdC[c] << "/" << unC[c];
            std::cout << std::endl;
        }
        ts::EnutsProf& P = ts::EnutsProf::get();
        P.t[ts::EnutsProf::CN_XCH_NOPLAN] += nNo * dof;
        P.t[ts::EnutsProf::CN_XCH_UNREAD] += nUn * dof;
        P.t[ts::EnutsProf::CN_XCH_READ] += nRd * dof;
    }
    // XCH_TRACE: one extra double per partner message, the sender's stamp
    const size_t xt = ts::EnutsProf::xchTrace() ? 1 : 0;
    size_t rTot = 0, sTot = 0;
    for (size_t i = 0; i < rPr.size(); i++) rTot += nn(m_uiOwnRecvCum.data() + i * nCls);
    for (size_t i = 0; i < sPr.size(); i++) sTot += nn(m_uiOwnSendCum.data() + i * nCls);
    if (m_uiOwnRecvBuf.size() < rTot * dof + xt * rPr.size())
        m_uiOwnRecvBuf.resize(rTot * dof + xt * rPr.size());
    if (m_uiOwnSendBuf.size() < sTot * dof + xt * sPr.size())
        m_uiOwnSendBuf.resize(sTot * dof + xt * sPr.size());
    size_t off = 0;
    for (size_t i = 0; i < rPr.size(); i++) {
        const unsigned int cnt = nn(m_uiOwnRecvCum.data() + i * nCls);
        if (!cnt) continue;
        par::Mpi_Irecv(m_uiOwnRecvBuf.data() + off, (int)(cnt * dof + xt), rPr[i], tag,
                       comm, &m_uiOwnReq[i]);
        off += (size_t)cnt * dof + xt;
    }
    off = 0;
    for (size_t i = 0; i < sPr.size(); i++) {
        const unsigned int* cum = m_uiOwnSendCum.data() + i * nCls;
        const unsigned int cnt  = nn(cum);
        if (!cnt) continue;
        const unsigned int* sl = m_uiOwnSendSlot.data() + m_uiOwnSendOff[i];
        T* b                   = m_uiOwnSendBuf.data() + off + xt;
        const bool thr         = ts::EnutsProf::ompOn();
#ifdef DENDRO_ENUTS_HAS_OMP
#pragma omp parallel for schedule(static) if (thr && !omp_in_parallel())
#endif
        for (unsigned int v = 0; v < dof; v++) {
            const T* sv = stv + (size_t)v * dgSz;
            for (unsigned int q = 0; q < cnt; q++) b[(size_t)v * cnt + q] = sv[sl[q]];
        }
        if (xt) b[-1] = (T)ts::EnutsProf::now();
        par::Mpi_Isend(b - xt, (int)(cnt * dof + xt), sPr[i], tag, comm,
                       &m_uiOwnReq[rPr.size() + i]);
        off += (size_t)cnt * dof + xt;
    }
    if (ts::EnutsProf::on()) {
        ts::EnutsProf::get().t[ts::EnutsProf::CN_XCH_SEND] += (double)sTot * dof;
        ts::EnutsProf::get().t[ts::EnutsProf::CN_XCH_RECV] += (double)rTot * dof;
        ts::EnutsProf::get().t[ts::EnutsProf::CN_XCH_CALLS] += 1;
    }
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::own_node_exchange_end(T* stv, unsigned int dof,
                                                 unsigned int pt) {
    ot::Mesh* pMesh = (ot::Mesh*)m_uiAppCtx->get_mesh();
    if (!pMesh->isActive() || pMesh->getMPICommSize() < 2) return;
    const unsigned int nCls = m_uiOwnDTs.size();
    const size_t dgSz = (size_t)pMesh->getAllElements().size() * pMesh->getNumNodesPerElement();
    int j = -1;
    if (!m_uiOwnChain)
        j = (int)nCls - 1;
    else
        for (unsigned int i = 0; i < nCls; i++)
            if (pt % m_uiOwnDTs[i] == 0) j = (int)i;
    if (j < 0) return;
    auto nn = [&](const unsigned int* cum) { return cum[j]; };
    const bool xprof = ts::EnutsProf::on();
    const double tw0 = xprof ? ts::EnutsProf::now() : 0.0;
    const std::vector<unsigned int>& rPr = pMesh->getRecvProcList();
    const std::vector<unsigned int>& sPr = pMesh->getSendProcList();
    const size_t xt = ts::EnutsProf::xchTrace() ? 1 : 0;
    if (xt) {
        // XCH_TRACE: completion order of every request; the last one is the
        // partner this call waited on. its lateness = its send stamp in my
        // clock minus the time I started waiting (a send request completes
        // when the receiver posted its receive, in its own exchange begin,
        // so its stamp to me stands in for that too)
        XchRec r{};
        r.pt = pt;
        r.j  = (unsigned int)j;
        for (unsigned int k = 0; k < m_uiNumStages; k++)
            if (m_uiStVec[k].get_vec_ptr() == stv) r.s = k;
        r.tw   = tw0 - m_uiXchT0;
        r.late = r.lateMax = std::nan("");
        r.arg = r.lateArg = ~0u;
        const int nReq = (int)m_uiOwnReq.size();
        int argIdx = -1, argR = -1, argS = -1;
        double tlast = tw0, tfirst = -1, tlr = tw0, tls = tw0;
        for (int k = 0; k < nReq; k++) {
            int idx = MPI_UNDEFINED;
            MPI_Waitany(nReq, m_uiOwnReq.data(), &idx, MPI_STATUS_IGNORE);
            if (idx == MPI_UNDEFINED) break;
            const double tn = ts::EnutsProf::now();
            if (idx < (int)rPr.size()) {
                if (tfirst < 0) tfirst = tn;
                tlr  = tn;
                argR = idx;
            } else {
                tls  = tn;
                argS = idx;
            }
            tlast = tn;
        }
        // the last receive is the partner waited on, unless a send (the
        // partner's receive posted late) outlasted it by more than a poll
        argIdx    = (argS >= 0 && tls > tlr + 5e-5) ? argS : argR;
        r.wait    = tlast - tw0;
        r.rwait   = tlr - tw0;
        r.first   = tfirst < 0 ? 0 : tfirst - tw0;
        r.argSend = (argIdx >= (int)rPr.size());
        if (argIdx >= 0) r.arg = r.argSend ? sPr[argIdx - rPr.size()] : rPr[argIdx];
        size_t off = 0;
        for (size_t i = 0; i < rPr.size(); i++) {
            const unsigned int cnt = m_uiOwnRecvCum[i * nCls + j];
            if (!cnt) continue;
            r.nrecv += (double)cnt * dof;
            const double L = std::isnan(m_uiXchOff[i])
                                 ? std::nan("")
                                 : (double)m_uiOwnRecvBuf[off] + m_uiXchOff[i] - tw0;
            if (rPr[i] == r.arg) {
                r.late = L;
                r.argN = (double)cnt * dof;
            }
            if (!std::isnan(L) && (std::isnan(r.lateMax) || L > r.lateMax)) {
                r.lateMax = L;
                r.lateArg = rPr[i];
            }
            off += (size_t)cnt * dof + xt;
        }
        if (r.argSend && argIdx >= 0)
            r.argN = (double)m_uiOwnSendCum[(argIdx - rPr.size()) * nCls + j] * dof;
        ts::EnutsProf& P = ts::EnutsProf::get();
        P.t[ts::EnutsProf::EX_FIRST] += r.first;
        if (!std::isnan(r.late)) P.t[ts::EnutsProf::EX_LATE] += std::max(0.0, r.late);
        if (!std::isnan(r.lateMax)) P.t[ts::EnutsProf::EX_LATEMAX] += std::max(0.0, r.lateMax);
        P.t[ts::EnutsProf::CN_XCH_ARGSEND] += r.argSend;
        P.t[ts::EnutsProf::CN_XCH_NOOFF] += (argIdx >= 0 && std::isnan(r.late));
        P.t[ts::EnutsProf::CN_XCH_LATEBAD] += (!std::isnan(r.lateMax) && r.lateMax > r.wait + 1e-4);
        m_uiXchRec.push_back(r);
    } else {
    if (xprof && !rPr.empty()) {
        // split of the wait: until the first receive completes (a late
        // partner) vs the rest (transfer / the slowest partner)
        int idx = MPI_UNDEFINED;
        MPI_Waitany((int)rPr.size(), m_uiOwnReq.data(), &idx, MPI_STATUS_IGNORE);
        ts::EnutsProf::get().t[ts::EnutsProf::EX_FIRST] += ts::EnutsProf::now() - tw0;
    }
    if (!m_uiOwnReq.empty())
        MPI_Waitall((int)m_uiOwnReq.size(), m_uiOwnReq.data(), MPI_STATUSES_IGNORE);
    }
    const double tu0 = xprof ? ts::EnutsProf::now() : 0.0;
    if (xprof) ts::EnutsProf::get().t[ts::EnutsProf::EX_WAIT] += tu0 - tw0;
    const bool cen = ts::EnutsProf::xchCensus() && !m_uiXchFlag.empty();
    unsigned char* fl = nullptr;
    if (cen) {
        unsigned int s = 0;
        for (unsigned int k = 0; k < m_uiNumStages; k++)
            if (m_uiStVec[k].get_vec_ptr() == stv) s = k;
        fl = m_uiXchFlag.data() + (size_t)s * dgSz;
    }
    size_t off = 0;
    for (size_t i = 0; i < rPr.size(); i++) {
        const unsigned int* cum = m_uiOwnRecvCum.data() + i * nCls;
        const unsigned int cnt  = nn(cum);
        if (!cnt) continue;
        const unsigned int* sl = m_uiOwnRecvSlot.data() + m_uiOwnRecvOff[i];
        const T* b             = m_uiOwnRecvBuf.data() + off + xt;
        const bool thr         = ts::EnutsProf::ompOn();
#ifdef DENDRO_ENUTS_HAS_OMP
#pragma omp parallel for schedule(static) if (thr && !omp_in_parallel())
#endif
        for (unsigned int v = 0; v < dof; v++) {
            T* sv = stv + (size_t)v * dgSz;
            for (unsigned int q = 0; q < cnt; q++) sv[sl[q]] = b[(size_t)v * cnt + q];
        }
        if (cen)
            for (unsigned int q = 0; q < cnt; q++) fl[sl[q]] = 1;
        off += (size_t)cnt * dof + xt;
    }
    if (xprof) ts::EnutsProf::get().t[ts::EnutsProf::EX_UNPACK] += ts::EnutsProf::now() - tu0;
}

}  // namespace ts
