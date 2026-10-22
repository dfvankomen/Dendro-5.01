/**
 * @file enuts_exchange.tcc
 * @brief ExplicitNUTS scoped stage exchange (scope lists, calibration, trace dump, check).
 * (code moved from enuts.h, which includes this file)
 */

namespace ts {

template <typename T, typename Ctx>
bool ExplicitNUTS<T, Ctx>::xch_scope_build(ot::Mesh* pMesh) {
    // scoped exchange: every rank must have every block's plan (sync_prebuild
    // builds them all in the first sync pass); until then, unscoped
    const unsigned int nBlk = pMesh->getLocalBlockList().size();
    int ok = (m_uiOwnPlan.size() == nBlk);
    for (unsigned int b = 0; ok && b < nBlk; b++) ok = m_uiOwnPlan[b].built;
    MPI_Comm comm = pMesh->getMPICommunicator();
    int g         = 0;
    MPI_Allreduce(&ok, &g, 1, MPI_INT, MPI_MIN, comm);
    if (!g) return false;
    const ot::Block* blkList = pMesh->getLocalBlockList().data();
    const unsigned int nPe   = pMesh->getNumNodesPerElement();
    const size_t dgSz        = (size_t)pMesh->getAllElements().size() * nPe;
    const unsigned int nCls  = m_uiOwnDTs.size();
    // per DG slot, the class of the finest local block whose plan reads it
    std::vector<unsigned char> rcls(dgSz, 255);
    auto low = [&](size_t slot, unsigned char c) {
        if (c < rcls[slot]) rcls[slot] = c;
    };
    for (unsigned int b = 0; b < nBlk; b++) {
        const OwnPlan& op = m_uiOwnPlan[b];
        const unsigned int bDT = m_uiEleDT[blkList[b].firstElementID()];
        const unsigned char bc = (unsigned char)(
            std::lower_bound(m_uiOwnDTs.begin(), m_uiOwnDTs.end(), bDT) - m_uiOwnDTs.begin());
        for (const unsigned int e : op.cele)
            if (xch_whole_row(e, bc))
                for (unsigned int n = 0; n < nPe; n++) low((size_t)e * nPe + n, bc);
        for (size_t r = 0; r < op.rsrc.size(); r++)
            for (unsigned int i = 0; i < op.rlen[r]; i++) low((size_t)op.rsrc[r] + i, bc);
        for (const unsigned int q : op.psrc) low(q, bc);
        for (const unsigned int q : op.opSrc) low(q, bc);
        for (const typename OwnPlan::Child& w : op.ch) {
            const size_t e = m_uiOwnRowEle[w.row];
            for (unsigned int n = 0; n < nPe; n++) low(e * nPe + n, bc);
            const unsigned int* ro = m_uiOwnRowOff.data() + (size_t)w.row * (nCls + 1);
            for (unsigned int q = ro[0]; q < ro[nCls]; q++) low(m_uiOwnRowSrc[q], bc);
        }
    }
    // per receive partner, the effective class of each slot in list order
    // (max of the owner class the list is sorted by and the reader class);
    // tell the sender, hear ours from each receiver
    const std::vector<unsigned int>& sPr = pMesh->getSendProcList();
    const std::vector<unsigned int>& rPr = pMesh->getRecvProcList();
    std::vector<unsigned char> rEff(m_uiOwnRecvSlot.size(), 255), sEff(m_uiOwnSendSlot.size(), 255);
    for (size_t i = 0; i < rPr.size(); i++) {
        const unsigned int* cum = m_uiOwnRecvCum.data() + i * nCls;
        for (unsigned int c = 0; c < nCls; c++)
            for (unsigned int q = (c ? cum[c - 1] : 0); q < cum[c]; q++) {
                const unsigned char r = rcls[m_uiOwnRecvSlot[m_uiOwnRecvOff[i] + q]];
                rEff[m_uiOwnRecvOff[i] + q] = (r == 255) ? 255 : std::max((unsigned char)c, r);
            }
    }
    std::vector<MPI_Request> rq(sPr.size() + rPr.size(), MPI_REQUEST_NULL);
    for (size_t i = 0; i < sPr.size(); i++) {
        const int n = (int)(m_uiOwnSendOff[i + 1] - m_uiOwnSendOff[i]);
        if (n) MPI_Irecv(sEff.data() + m_uiOwnSendOff[i], n, MPI_UNSIGNED_CHAR, sPr[i], 7156, comm, &rq[i]);
    }
    for (size_t i = 0; i < rPr.size(); i++) {
        const int n = (int)(m_uiOwnRecvOff[i + 1] - m_uiOwnRecvOff[i]);
        if (n) MPI_Isend(rEff.data() + m_uiOwnRecvOff[i], n, MPI_UNSIGNED_CHAR, rPr[i], 7156, comm, &rq[sPr.size() + i]);
    }
    MPI_Waitall((int)rq.size(), rq.data(), MPI_STATUSES_IGNORE);
    // both sides: the same stable sort of the same sequence by effective
    // class; the prefix counts now count effective classes, 255 never sent
    auto resort = [&](size_t np, std::vector<unsigned int>& slot, const std::vector<unsigned int>& off,
                      std::vector<unsigned int>& cum, const std::vector<unsigned char>& eff) {
        std::vector<unsigned int> idx;
        std::vector<unsigned int> tmp;
        for (size_t i = 0; i < np; i++) {
            const unsigned int b = off[i], n = off[i + 1] - off[i];
            idx.resize(n);
            for (unsigned int q = 0; q < n; q++) idx[q] = q;
            std::stable_sort(idx.begin(), idx.end(), [&](unsigned int a, unsigned int c) {
                return eff[b + a] < eff[b + c];
            });
            tmp.assign(slot.begin() + b, slot.begin() + b + n);
            for (unsigned int q = 0; q < n; q++) slot[b + q] = tmp[idx[q]];
            for (unsigned int c = 0; c < nCls; c++) cum[i * nCls + c] = 0;
            for (unsigned int q = 0; q < n; q++)
                if (eff[b + q] < nCls) cum[i * nCls + eff[b + q]]++;
            for (unsigned int c = 1; c < nCls; c++) cum[i * nCls + c] += cum[i * nCls + c - 1];
        }
    };
    if (ts::EnutsProf::gXch()) {
        m_uiOwnRecvSlotU = m_uiOwnRecvSlot;
        m_uiOwnRecvCumU  = m_uiOwnRecvCum;
        m_uiOwnSendSlotU = m_uiOwnSendSlot;
        m_uiOwnSendCumU  = m_uiOwnSendCum;
    }
    resort(rPr.size(), m_uiOwnRecvSlot, m_uiOwnRecvOff, m_uiOwnRecvCum, rEff);
    resort(sPr.size(), m_uiOwnSendSlot, m_uiOwnSendOff, m_uiOwnSendCum, sEff);
    if (std::getenv("DENDRO_ENUTS_OWNER_SYNC_INFO")) {
        size_t never = 0;
        for (const unsigned char e : rEff) never += (e == 255);
        std::cout << "[enuts-owner-sync r" << pMesh->getMPIRankGlobal() << "] scoped send lists: recv nodes "
                  << rEff.size() << " never read " << never << std::endl;
    }
    m_uiXchScoped = true;
    return true;
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::xch_calibrate(ot::Mesh* pMesh) {
    // XCH_TRACE: per receive partner the offset of its clock from mine, from
    // round trips (stamp each way, then the measured one-way values each
    // way; the sample with the least latency wins). partners I only receive
    // from stay NaN.
    const std::vector<unsigned int>& rPr = pMesh->getRecvProcList();
    const std::vector<unsigned int>& sPr = pMesh->getSendProcList();
    MPI_Comm comm                        = pMesh->getMPICommunicator();
    const size_t nr                      = rPr.size();
    m_uiXchOff.assign(nr, std::nan(""));
    m_uiXchLat.assign(nr, std::nan(""));
    m_uiXchSpread.assign(nr, std::nan(""));
    m_uiXchOffRank.assign(rPr.begin(), rPr.end());
    std::vector<int> both(nr, 0);
    for (size_t i = 0; i < nr; i++)
        both[i] = std::find(sPr.begin(), sPr.end(), rPr[i]) != sPr.end();
    std::vector<double> st(nr, 0), mine(nr, 0), tr(nr, 0), dm(nr, 0), dp(nr, 0), best(nr, 1e300);
    const int nRound = 64;
    std::vector<double> latS(nr * nRound, 1e300), offS(nr * nRound, 0);
    std::vector<MPI_Request> rq(2 * nr, MPI_REQUEST_NULL);
    for (int round = 0; round < nRound; round++) {
        for (size_t i = 0; i < nr; i++)
            if (both[i]) MPI_Irecv(&st[i], 1, MPI_DOUBLE, rPr[i], 7154, comm, &rq[i]);
        for (size_t i = 0; i < nr; i++)
            if (both[i]) {
                mine[i] = ts::EnutsProf::now();
                MPI_Isend(&mine[i], 1, MPI_DOUBLE, rPr[i], 7154, comm, &rq[nr + i]);
            }
        for (size_t k = 0; k < 2 * nr; k++) {
            int idx = MPI_UNDEFINED;
            MPI_Waitany((int)(2 * nr), rq.data(), &idx, MPI_STATUS_IGNORE);
            if (idx == MPI_UNDEFINED) break;
            if (idx < (int)nr) tr[idx] = ts::EnutsProf::now();
        }
        for (size_t i = 0; i < nr; i++) dm[i] = tr[i] - st[i];  // latency - offset
        for (size_t i = 0; i < nr; i++)
            if (both[i]) {
                MPI_Irecv(&dp[i], 1, MPI_DOUBLE, rPr[i], 7155, comm, &rq[i]);
                MPI_Isend(&dm[i], 1, MPI_DOUBLE, rPr[i], 7155, comm, &rq[nr + i]);
            }
        MPI_Waitall((int)(2 * nr), rq.data(), MPI_STATUSES_IGNORE);
        for (size_t i = 0; i < nr; i++) {
            if (!both[i]) continue;
            const double lat          = 0.5 * (dm[i] + dp[i]);
            latS[i * nRound + round] = lat;
            offS[i * nRound + round] = 0.5 * (dp[i] - dm[i]);
            if (lat < best[i]) best[i] = lat;
        }
    }
    // the offset: median over the samples within 2 us of the least latency
    // (a descheduled poll inflates one direction); spread = their range
    for (size_t i = 0; i < nr; i++) {
        if (!both[i]) continue;
        std::vector<double> good;
        for (int k = 0; k < nRound; k++)
            if (latS[i * nRound + k] <= best[i] + 2e-6) good.push_back(offS[i * nRound + k]);
        std::sort(good.begin(), good.end());
        m_uiXchOff[i]    = good[good.size() / 2];
        m_uiXchLat[i]    = best[i];
        m_uiXchSpread[i] = good.back() - good.front();
    }
    m_uiXchT0  = ts::EnutsProf::now();
    m_uiXchCal = true;
    m_uiXchRec.clear();
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::xch_dump() {
    const char* dir = ts::EnutsProf::xchTraceDir();
    if (!dir || m_uiXchRec.empty()) return;
    int rk = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rk);
    std::ofstream f(std::string(dir) + "/enuts_xch_r" + std::to_string(rk) + ".txt");
    // XCH_TRACE: the calibration instant in this rank's clock, so entry times (t_wait_enter, since
    // it) chain into one clock across ranks through the partner offsets
    f << "# t0 " << std::setprecision(12) << m_uiXchT0 << "\n";
    f << "# clock offsets (partner rank: partner clock - mine, s; nan = no round trip; then the best\n"
         "# one-way latency of the 64 samples and the range of the offsets among the low-latency\n"
         "# samples, s; then the largest lateness - wait seen with that partner as arg-max (> 0:\n"
         "# the offset is at least that much too large; the corrected sums subtract it)\n";
    std::map<unsigned int, double> bad;
    for (const XchRec& r : m_uiXchRec)
        if (!std::isnan(r.late) && r.late - r.wait > bad[r.arg]) bad[r.arg] = r.late - r.wait;
    for (size_t i = 0; i < m_uiXchOff.size(); i++)
        f << "# off " << m_uiXchOffRank[i] << " " << std::setprecision(9) << m_uiXchOff[i] << " "
          << m_uiXchLat[i] << " " << m_uiXchSpread[i] << " " << bad[m_uiXchOffRank[i]] << "\n";
    f << "# call pt stage j t_wait_enter nrecv_doubles wait_s last_recv_s first_recv_s argmax_partner "
         "argmax_kind(r/s) argmax_late_s argmax_doubles max_late_s max_late_partner\n";
    std::map<unsigned int, std::pair<unsigned int, double>> hist;
    double sw = 0, sl = 0, slm = 0, slc = 0;
    unsigned int nbad = 0;
    for (size_t k = 0; k < m_uiXchRec.size(); k++) {
        const XchRec& r = m_uiXchRec[k];
        f << k << " " << r.pt << " " << r.s + 1 << " " << r.j << " " << std::setprecision(9)
          << r.tw << " " << std::setprecision(6) << r.nrecv << " " << r.wait << " " << r.rwait << " " << r.first
          << " " << (int)r.arg << " " << (r.argSend ? 's' : 'r') << " " << r.late << " "
          << r.argN << " " << r.lateMax << " " << (int)r.lateArg << "\n";
        hist[r.arg].first++;
        hist[r.arg].second += r.wait;
        sw += r.wait;
        if (!std::isnan(r.late)) {
            sl += std::max(0.0, r.late);
            slc += std::max(0.0, std::min(r.late - bad[r.arg], r.wait));
            nbad += (r.late > r.wait + 1e-4);
        }
        if (!std::isnan(r.lateMax)) slm += std::max(0.0, r.lateMax);
    }
    std::vector<std::pair<unsigned int, std::pair<unsigned int, double>>> hv(hist.begin(), hist.end());
    std::sort(hv.begin(), hv.end(), [](const auto& a, const auto& b) {
        return a.second.second > b.second.second;
    });
    f << "# summary calls " << m_uiXchRec.size() << " wait " << sw << " argmax_late_pos " << sl
      << " argmax_late_corrected " << slc << " max_late_pos " << slm << " late_gt_wait_calls " << nbad
      << "\n";
    f << "# argmax partner: calls wait_s (by wait)\n";
    for (size_t i = 0; i < hv.size() && i < 12; i++)
        f << "# arg " << (int)hv[i].first << " " << hv[i].second.first << " " << hv[i].second.second
          << "\n";
}

template <typename T, typename Ctx>
bool ExplicitNUTS<T, Ctx>::prepare_scoped_exchange(ot::Mesh* pMesh) {
    if (m_uiScopedMesh == pMesh) return m_uiScopedOk;
    m_uiScopedMesh = pMesh;
    m_uiScopedOk   = false;
    m_uiScopedPlan = ot::EleDGScopedPlan();
    m_uiScopedDTs.clear();
    if (!pMesh->isActive()) return false;
    // one active rank: no element ghost scatter maps exist, nothing to scope
    if (pMesh->getMPICommSize() < 2) return false;

    // ascending distinct timestep factors; they must form a divisibility chain
    // so the elements firing at pt (DT | pt) are a prefix of the sorted segment
    for (unsigned int l = m_uiLevMin; l <= m_uiLevMax; l++)
        m_uiScopedDTs.push_back(
            step_ratio(l));
    std::sort(m_uiScopedDTs.begin(), m_uiScopedDTs.end());
    m_uiScopedDTs.erase(
        std::unique(m_uiScopedDTs.begin(), m_uiScopedDTs.end()),
        m_uiScopedDTs.end());
    bool ok = !m_uiScopedDTs.empty() && m_uiScopedDTs[0] >= 1;
    for (size_t i = 1; ok && i < m_uiScopedDTs.size(); i++)
        ok = (m_uiScopedDTs[i] % m_uiScopedDTs[i - 1] == 0);

    // element -> DT by its own level; must be in range, and every local block
    // must be single-level (activity is decided by the block's first element)
    const std::vector<ot::TreeNode>& all = pMesh->getAllElements();
    std::vector<unsigned int> dtOf(all.size(), 0);
    for (size_t e = 0; ok && e < all.size(); e++) {
        const unsigned int l = all[e].getLevel();
        if (l < m_uiLevMin || l > m_uiLevMax) {
            ok = false;
            break;
        }
        dtOf[e] = step_ratio(l);
    }
    const std::vector<ot::Block>& blkList = pMesh->getLocalBlockList();
    for (unsigned int b = 0; ok && b < blkList.size(); b++) {
        const unsigned int bl = all[blkList[b].firstElementID()].getLevel();
        for (const unsigned int e : blkList[b])
            if (all[e].getLevel() != bl) {
                ok = false;
                break;
            }
    }
    // all ranks must agree (Begin/End are collective in effect)
    int loc = ok ? 1 : 0, glob = 1;
    MPI_Allreduce(&loc, &glob, 1, MPI_INT, MPI_MIN, pMesh->getMPICommunicator());
    ok = glob == 1;
    if (!ok) {
        if (!pMesh->getMPIRankGlobal())
            std::cout << "[enuts-scoped] preconditions failed (DT chain / level "
                         "range / multi-level block): full exchange"
                      << std::endl;
        return false;
    }
    pMesh->buildEleDGScopedPlan(dtOf, m_uiScopedDTs, m_uiScopedPlan);
    m_uiScopedZero.assign(std::max(1u, m_uiScopedPlan.npes), 0u);
    m_uiScopedOk = true;
    return true;
}

template <typename T, typename Ctx>
void ExplicitNUTS<T, Ctx>::scoped_check_compare(ot::Mesh* pMesh, const T* real,
                                                const T* shadow,
                                                unsigned int DOF,
                                                unsigned int pt,
                                                unsigned int rk) {
    if (m_uiBlkDepEid.size() != pMesh->getLocalBlockList().size()) {
        m_uiBlkDepEid.assign(pMesh->getLocalBlockList().size(), {});
        for (unsigned int b = 0; b < m_uiBlkDepEid.size(); b++)
            computeBlockUnzipDepElements(pMesh, b, m_uiBlkDepEid[b]);
    }
    const unsigned int nPe  = pMesh->getNumNodesPerElement();
    const size_t dgSz       = (size_t)pMesh->getAllElements().size() * nPe;
    const unsigned int lb   = pMesh->getElementLocalBegin();
    const unsigned int le   = pMesh->getElementLocalEnd();
    std::set<unsigned int> seen;
    unsigned long long bad = 0;
    for (const unsigned int blk : m_uiActiveBlkIDs)
        for (const unsigned int e : m_uiBlkDepEid[blk]) {
            if (e >= lb && e < le) continue;
            if (!seen.insert(e).second) continue;
            ts::EnutsProf::get().chk_cmp++;
            bool diff = false;
            for (unsigned int v = 0; v < DOF; v++)
                if (std::memcmp(real + v * dgSz + (size_t)e * nPe,
                                shadow + v * dgSz + (size_t)e * nPe,
                                sizeof(T) * nPe) != 0)
                    diff = true;
            if (diff) {
                bad++;
                if (bad <= 5)
                    std::cout << "[enuts-scoped-check r"
                              << pMesh->getMPIRankGlobal() << "] MISMATCH pt=" << pt
                              << " rk=" << rk << " ghost elem " << e << " "
                              << pMesh->getAllElements()[e] << " etime="
                              << m_uiEleTime[e] << " edt=" << m_uiEleDT[e]
                              << std::endl;
            }
        }
    ts::EnutsProf::get().chk_bad += bad;
    if (bad) {
        std::cout << "[enuts-scoped-check r" << pMesh->getMPIRankGlobal()
                  << "] " << bad << " mismatching dependent ghost elems at pt="
                  << pt << " rk=" << rk << "; aborting" << std::endl;
        MPI_Abort(pMesh->getMPICommunicator(), 9);
    }
}

}  // namespace ts
