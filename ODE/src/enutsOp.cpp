#include "enutsOp.h"
namespace ts {

ENUTSOp::ENUTSOp(ETSType type) {
    m_uiType = type;

    if (type == ETSType::RK3) {
        m_uiNumStages                 = 3;
        const unsigned int d1         = m_uiNumStages;
        const unsigned int d2         = (m_uiNumStages * m_uiNumStages);

        const DendroScalar ETS_U[]    = {0.0, 0.0,       0.0,       1.0, 0.0,
                                         0.0, 1.0 / 4.0, 1.0 / 4.0, 0.0};

        // coefficient matrix, (picewise linear approx)
        const DendroScalar ETS_C[]    = {1.0, 0.0, 0.0,       1.0,      1.0,
                                         0.0, 1.0, 1.0 / 2.0, 1.0 / 4.0};

        const DendroScalar ETS_InvC[] = {1.0, 0.0,  0.0,  -1.0, 1.0,
                                         0.0, -2.0, -2.0, 4.0};

        m_uiAij.resize(d2);
        m_uiBij.resize(d2);
        m_uiCij.resize(d2);
        m_uiInvCij.resize(d2);
        m_uiPi.resize(d1);
        m_uiInvPi.resize(d1);
        m_uiMij.resize(d2);
        m_uiVin.resize(d1);
        m_uiVout.resize(d1);

        std::memcpy(m_uiAij.data(), ETS_U, sizeof(DendroScalar) * d2);
        std::memcpy(m_uiCij.data(), ETS_C, sizeof(DendroScalar) * d2);
        std::memcpy(m_uiInvCij.data(), ETS_InvC, sizeof(DendroScalar) * d2);

    } else if (type == ETSType::RK4) {
        m_uiNumStages              = 4;
        const unsigned int d1      = m_uiNumStages;
        const unsigned int d2      = (m_uiNumStages * m_uiNumStages);

        const DendroScalar ETS_U[] = {0.0, 0.0, 0.0, 0.0,       1.0 / 2.0, 0.0,
                                      0.0, 0.0, 0.0, 1.0 / 2.0, 0.0,       0.0,
                                      0.0, 0.0, 1.0, 0.0};

        // coefficient matrix, (picewise linear approx)
        const DendroScalar ETS_C[] = {
            1.0,       0.0,       0.0, 0.0, 1.0, 1.0 / 2.0, 0.0,      0.0, 1.0,
            1.0 / 2.0, 1.0 / 4.0, 0.0, 1.0, 1.0, 1.0 / 2.0, 1.0 / 4.0};

        const DendroScalar ETS_InvC[] = {1.0, 0.0, 0.0,  0.0,  -2.0, 2.0,
                                         0.0, 0.0, 0.0,  -4.0, 4.0,  0.0,
                                         4.0, 0.0, -8.0, 4.0};

        m_uiAij.resize(d2);
        m_uiBij.resize(d2);
        m_uiCij.resize(d2);
        m_uiInvCij.resize(d2);
        m_uiPi.resize(d1);
        m_uiInvPi.resize(d1);
        m_uiMij.resize(d2);
        m_uiVin.resize(d1);
        m_uiVout.resize(d1);

        std::memcpy(m_uiAij.data(), ETS_U, sizeof(DendroScalar) * d2);
        std::memcpy(m_uiCij.data(), ETS_C, sizeof(DendroScalar) * d2);
        std::memcpy(m_uiInvCij.data(), ETS_InvC, sizeof(DendroScalar) * d2);

    } else if (type == ETSType::RK5) {
        // these coefficients looks wrong. - Milinda. (need to fix those and
        // enable the rk5 method. )
        return;
        /*m_uiNumStages = 5;
        const unsigned int d1 = m_uiNumStages;
        const unsigned int d2 = (m_uiNumStages*m_uiNumStages);

        const DendroScalar ETS_U[] =        {   0.0       ,  0.0        , 0.0 ,
        0.0        , 0.0, 1.0/8.0   ,  1.0/8.0    , 0.0       , 0.0        ,
        0.0, 0.0       , -1.0/2.0    , 1.0       , 0.0        , 0.0, 3.0/16.0  ,
        0.0        , 0.0       , 9.0/16.0   , 0.0, -3.0/7.0  ,  2.0/7.0
        , 12.0/7.0  , -12.0/7.0  , 8.0/7.0};

        m_uiAij.resize(d2);
        m_uiBij.resize(d2);
        m_uiCij.resize(d2);
        m_uiPi.resize(d1);

        std::memcpy(m_uiAij.data(), ETS_U, sizeof(DendroScalar)*(d2));*/
    }

    // std::cout<<" m_uinum stages : "<<m_uiNumStages<<std::endl;
    return;
}

void ENUTSOp::Pdt(DendroScalar dt, unsigned int rk_s) {
    m_uiPi[0]    = 1.0;
    m_uiInvPi[0] = 1.0;
    for (unsigned int s = 2; s <= rk_s; s++) {
        m_uiPi[s - 1]    = m_uiPi[s - 2] * dt;
        m_uiInvPi[s - 1] = 1.0 / m_uiPi[s - 1];
    }
}

void ENUTSOp::Bdt(DendroScalar dt, unsigned int rk_s) {
    if (m_uiType == ETSType::RK3) {
        /*const DendroScalar ETS_B[] =        { 1.0     , 1.0      , 1.0/2.0,
                                              0.0     , 1.0      , 1.0,
                                              0.0     , 0.0      , 1.0  };*/

        m_uiBij[0 * m_uiNumStages + 0] = 1.0;
        m_uiBij[0 * m_uiNumStages + 1] = 1.0 * dt;
        m_uiBij[0 * m_uiNumStages + 2] = (1.0 / 2.0) * dt * dt;

        m_uiBij[1 * m_uiNumStages + 0] = 0.0;
        m_uiBij[1 * m_uiNumStages + 1] = 1.0;
        m_uiBij[1 * m_uiNumStages + 2] = 1.0 * dt;

        m_uiBij[2 * m_uiNumStages + 0] = 0.0;
        m_uiBij[2 * m_uiNumStages + 1] = 0.0;
        m_uiBij[2 * m_uiNumStages + 2] = 1.0;

    } else if (m_uiType == ETSType::RK4) {
        /*const DendroScalar ETS_B[] =       { 1.0     , 1.0      , 1.0/2.0
           , 1.0/6.0, 0.0     , 1.0      , 1.0     , 1.0/2.0, 0.0     , 0.0
           , 1.0     , 1.0    , 0.0     , 0.0      , 0.0     , 1.0    };*/

        m_uiBij[0 * m_uiNumStages + 0] = 1.0;
        m_uiBij[0 * m_uiNumStages + 1] = 1.0 * dt;
        m_uiBij[0 * m_uiNumStages + 2] = (1.0 / 2.0) * dt * dt;
        m_uiBij[0 * m_uiNumStages + 3] = (1.0 / 6.0) * dt * dt * dt;

        m_uiBij[1 * m_uiNumStages + 0] = 0.0;
        m_uiBij[1 * m_uiNumStages + 1] = 1.0;
        m_uiBij[1 * m_uiNumStages + 2] = 1.0 * dt;
        m_uiBij[1 * m_uiNumStages + 3] = (1.0 / 2.0) * dt * dt;

        m_uiBij[2 * m_uiNumStages + 0] = 0.0;
        m_uiBij[2 * m_uiNumStages + 1] = 0.0;
        m_uiBij[2 * m_uiNumStages + 2] = 1.0;
        m_uiBij[2 * m_uiNumStages + 3] = 1.0 * dt;

        m_uiBij[3 * m_uiNumStages + 0] = 0.0;
        m_uiBij[3 * m_uiNumStages + 1] = 0.0;
        m_uiBij[3 * m_uiNumStages + 2] = 0.0;
        m_uiBij[3 * m_uiNumStages + 3] = 1.0;
    }
}

void ENUTSOp::buildM(bool cf, DendroScalar* M, unsigned int rk_s,
                     DendroScalar dt_c, DendroScalar dt_f, DendroScalar dt) {
    // cf: coarser -> finer (Ccf), else finer -> coarser (Cfc). Same arithmetic
    // as the original in-place code; the result is copied out of m_uiMij.
    const unsigned int d = m_uiNumStages;
    Bdt(dt, rk_s);
    const double inv_in = 1.0 / (cf ? dt_c : dt_f);
    const double dt_out = cf ? dt_f : dt_c;

    // B x P^{-1} x C^{-1}
    for (unsigned int i = 0; i < d; i++)
        for (unsigned int j = 0; j < d; j++) {
            m_uiMij[i * d + j] = 0.0;
            double tmp         = 1.0;
            for (unsigned int k = 0; k < d; k++) {
                m_uiMij[i * d + j] +=
                    m_uiBij[i * d + k] * m_uiInvCij[k * d + j] * tmp;
                tmp = tmp * inv_in;
            }
        }

    std::swap(m_uiMij, m_uiBij);

    // C x P x B x P^{-1} x C^{-1}
    for (unsigned int i = 0; i < d; i++)
        for (unsigned int j = 0; j < d; j++) {
            m_uiMij[i * d + j] = 0.0;
            double tmp         = 1.0;
            for (unsigned int k = 0; k < d; k++) {
                m_uiMij[i * d + j] +=
                    m_uiCij[i * d + k] * m_uiBij[k * d + j] * tmp;
                tmp = tmp * dt_out;
            }
        }
    for (unsigned int i = 0; i < d * d; i++) M[i] = m_uiMij[i];
}

const DendroScalar* ENUTSOp::cachedM(bool cf, unsigned int rk_s,
                                     DendroScalar dt_c, DendroScalar dt_f,
                                     DendroScalar dt) {
    // the operator depends only on (direction, rk_s, dt_c, dt_f, dt): a few
    // distinct ones recur for every element of a stage pass
    for (unsigned int i = 0; i < OP_CACHE; i++) {
        const OpCache& c = m_uiOpCache[i];
        if (c.valid && c.cf == cf && c.rk == rk_s && c.dt_c == dt_c &&
            c.dt_f == dt_f && c.dt == dt)
            return c.M;
    }
    OpCache& c  = m_uiOpCache[m_uiOpCacheNext];
    m_uiOpCacheNext = (m_uiOpCacheNext + 1) % OP_CACHE;
    c.valid     = false;
    buildM(cf, c.M, rk_s, dt_c, dt_f, dt);
    c.cf = cf; c.rk = rk_s; c.dt_c = dt_c; c.dt_f = dt_f; c.dt = dt;
    c.valid = true;
    return c.M;
}

void ENUTSOp::stageRow(bool cf, DendroScalar* row, unsigned int rk_s,
                       DendroScalar dt_c, DendroScalar dt_f, DendroScalar dt) {
    const DendroScalar* M = cachedM(cf, rk_s, dt_c, dt_f, dt);
    for (unsigned int k = 0; k < m_uiNumStages; k++)
        row[k] = M[(rk_s - 1) * m_uiNumStages + k];
}

// out[n] = sum_k m[k] * in[k][n], accumulated from 0 in the order k = 0, 1, ..
// (the summation order of the original per-node loop)
static void enuts_apply_row(DendroScalar* __restrict__ o,
                            const DendroScalar* const* in,
                            const DendroScalar* m, unsigned int ns,
                            unsigned int NN) {
    switch (ns) {
        case 1: {
            const DendroScalar m0 = m[0];
            const DendroScalar* a = in[0];
            for (unsigned int n = 0; n < NN; n++) {
                DendroScalar t = 0;
                t += m0 * a[n];
                o[n] = t;
            }
        } break;
        case 2: {
            const DendroScalar m0 = m[0], m1 = m[1];
            const DendroScalar *a = in[0], *b = in[1];
            for (unsigned int n = 0; n < NN; n++) {
                DendroScalar t = 0;
                t += m0 * a[n];
                t += m1 * b[n];
                o[n] = t;
            }
        } break;
        case 3: {
            const DendroScalar m0 = m[0], m1 = m[1], m2 = m[2];
            const DendroScalar *a = in[0], *b = in[1], *c = in[2];
            for (unsigned int n = 0; n < NN; n++) {
                DendroScalar t = 0;
                t += m0 * a[n];
                t += m1 * b[n];
                t += m2 * c[n];
                o[n] = t;
            }
        } break;
        case 4: {
            const DendroScalar m0 = m[0], m1 = m[1], m2 = m[2], m3 = m[3];
            const DendroScalar *a = in[0], *b = in[1], *c = in[2], *e = in[3];
            for (unsigned int n = 0; n < NN; n++) {
                DendroScalar t = 0;
                t += m0 * a[n];
                t += m1 * b[n];
                t += m2 * c[n];
                t += m3 * e[n];
                o[n] = t;
            }
        } break;
        default:
            for (unsigned int n = 0; n < NN; n++) {
                DendroScalar t = 0;
                for (unsigned int k = 0; k < ns; k++) t += m[k] * in[k][n];
                o[n] = t;
            }
    }
}

void ENUTSOp::applyRow(DendroScalar* out, const DendroScalar* const* in,
                       const DendroScalar* m, unsigned int ns,
                       unsigned int NN) const {
    enuts_apply_row(out, in, m, ns, NN);
}

void ENUTSOp::gatherRow(DendroScalar* blk, const unsigned int* dst,
                        const unsigned int* src, unsigned int n,
                        const DendroScalar* const* in, const DendroScalar* m,
                        unsigned int ns) const {
    switch (ns) {
        case 1: {
            const DendroScalar m0 = m[0];
            const DendroScalar* a = in[0];
            for (unsigned int q = 0; q < n; q++) {
                DendroScalar t = 0;
                t += m0 * a[src[q]];
                blk[dst[q]] = t;
            }
        } break;
        case 2: {
            const DendroScalar m0 = m[0], m1 = m[1];
            const DendroScalar *a = in[0], *b = in[1];
            for (unsigned int q = 0; q < n; q++) {
                const unsigned int s = src[q];
                DendroScalar t = 0;
                t += m0 * a[s];
                t += m1 * b[s];
                blk[dst[q]] = t;
            }
        } break;
        case 3: {
            const DendroScalar m0 = m[0], m1 = m[1], m2 = m[2];
            const DendroScalar *a = in[0], *b = in[1], *c = in[2];
            for (unsigned int q = 0; q < n; q++) {
                const unsigned int s = src[q];
                DendroScalar t = 0;
                t += m0 * a[s];
                t += m1 * b[s];
                t += m2 * c[s];
                blk[dst[q]] = t;
            }
        } break;
        case 4: {
            const DendroScalar m0 = m[0], m1 = m[1], m2 = m[2], m3 = m[3];
            const DendroScalar *a = in[0], *b = in[1], *c = in[2], *e = in[3];
            for (unsigned int q = 0; q < n; q++) {
                const unsigned int s = src[q];
                DendroScalar t = 0;
                t += m0 * a[s];
                t += m1 * b[s];
                t += m2 * c[s];
                t += m3 * e[s];
                blk[dst[q]] = t;
            }
        } break;
        default:
            for (unsigned int q = 0; q < n; q++) {
                DendroScalar t = 0;
                for (unsigned int k = 0; k < ns; k++) t += m[k] * in[k][src[q]];
                blk[dst[q]] = t;
            }
    }
}

void ENUTSOp::Cfc(DendroScalar** out, const DendroScalar** in,
                  const unsigned int* sz, unsigned int rk_s, DendroScalar dt_c,
                  DendroScalar dt_f, DendroScalar dt, unsigned int dof,
                  unsigned int nav) {
    // nav source stages are available (and set the in/out stride); 0 = rk_s,
    // the original form, which truncates the time shift to rk_s stages
    const unsigned int ns = nav ? nav : rk_s;
    const unsigned int NN = sz[0] * sz[1] * sz[2];
    const unsigned int d  = m_uiNumStages;
    // M = C x P x B x P^{-1} x C^{-1}
    const DendroScalar* M = cachedM(false, rk_s, dt_c, dt_f, dt);
    for (unsigned int v = 0; v < dof; v++)
        enuts_apply_row(out[v * ns + rk_s - 1], in + v * ns,
                        M + (rk_s - 1) * d, ns, NN);
}

void ENUTSOp::Ccf(DendroScalar** out, const DendroScalar** in,
                  const unsigned int* sz, unsigned int rk_s, DendroScalar dt_c,
                  DendroScalar dt_f, DendroScalar dt, unsigned int dof,
                  unsigned int nav) {
    // nav source stages are available (and set the in/out stride); 0 = rk_s,
    // the original form, which truncates the time shift to rk_s stages
    const unsigned int ns = nav ? nav : rk_s;
    const unsigned int NN = sz[0] * sz[1] * sz[2];
    const unsigned int d  = m_uiNumStages;
    const DendroScalar* M = cachedM(true, rk_s, dt_c, dt_f, dt);
    for (unsigned int v = 0; v < dof; v++)
        enuts_apply_row(out[v * ns + rk_s - 1], in + v * ns,
                        M + (rk_s - 1) * d, ns, NN);
}

void ENUTSOp::coarser_finer_ut_correction(DendroScalar** out,
                                          const DendroScalar** in,
                                          const unsigned int* sz,
                                          DendroScalar dt_c, DendroScalar dt_f,
                                          DendroScalar dt, unsigned int dof) {
    const unsigned int nx        = sz[0];
    const unsigned int ny        = sz[1];
    const unsigned int nz        = sz[2];
    const unsigned int NN        = nx * ny * nz;
    const unsigned int d         = m_uiNumStages;
    const unsigned int totalVars = d + 1;

    // note: when dt=0, then the B = I;
    Bdt(dt, m_uiNumStages);

    const double inv_dtc = (1.0 / dt_c);

    // B x P^{-1} x C^{-1}
    for (unsigned int i = 0; i < d; i++)
        for (unsigned int j = 0; j < d; j++) {
            m_uiMij[i * d + j] = 0.0;
            double tmp         = 1.0;
            for (unsigned int k = 0; k < d; k++) {
                m_uiMij[i * d + j] +=
                    m_uiBij[i * d + k] * m_uiInvCij[k * d + j] * tmp;
                tmp = tmp * inv_dtc;
            }
        }

    // M_ij has the correct operator.
    // Vin  : 0, 1, 2, .. m_uiNumStages-1 these are RK stages, m_uiNumStages
    // contains the U_f or U_c. Vout : 0, 1, 2, .. m_uiNumStages-1 contains the
    // correct stages to compute correct U form U_f or U_c
    for (unsigned int v = 0; v < dof; v++) {
        for (unsigned int n = 0; n < NN; n++) {
            for (unsigned int s = 0; s < d; s++) {
                m_uiVin[s]  = in[v * (d + 1) + (s)][n];
                m_uiVout[s] = 0;
            }

            for (unsigned int si = 0; si < m_uiNumStages; si++)
                for (unsigned int sj = 0; sj < m_uiNumStages; sj++)
                    m_uiVout[si] += m_uiMij[si * d + sj] * m_uiVin[sj];

            double dt_by_fac_p = 1.0;

            out[v * (d + 1) + m_uiNumStages][n] =
                in[v * (d + 1) + m_uiNumStages][n];

            for (unsigned int si = 0; si < m_uiNumStages; si++) {
                dt_by_fac_p *= (dt_f / ((double)(si + 1)));
                out[v * (d + 1) + m_uiNumStages][n] -=
                    (dt_by_fac_p * m_uiVout[si]);
            }

            // printf("out s %d n %d  val %f \n", rk_s-1,n, out[v*rk_s +
            // rk_s-1][n]); printf("coarset to finer: in value: %f out value: %f
            // \n",in[v*(d+1) + m_uiNumStages][n],out[v*(d+1) +
            // m_uiNumStages][n]);
        }
    }

    return;
}

void ENUTSOp::finer_coarser_ut_correction(DendroScalar** out,
                                          const DendroScalar** in,
                                          const unsigned int* sz,
                                          DendroScalar dt_c, DendroScalar dt_f,
                                          DendroScalar dt, unsigned int dof) {
    // std::cout<<"caling finer_coarser"<<std::endl;
    const unsigned int nx        = sz[0];
    const unsigned int ny        = sz[1];
    const unsigned int nz        = sz[2];
    const unsigned int NN        = nx * ny * nz;
    const unsigned int d         = m_uiNumStages;
    const unsigned int totalVars = d + 1;

    // note: when dt=0, then the B = I;
    Bdt(dt, m_uiNumStages);

    const double inv_dtf = (1.0 / dt_f);

    // B x P^{-1} x C^{-1}
    for (unsigned int i = 0; i < d; i++)
        for (unsigned int j = 0; j < d; j++) {
            m_uiMij[i * d + j] = 0.0;
            double tmp         = 1.0;
            for (unsigned int k = 0; k < d; k++) {
                m_uiMij[i * d + j] +=
                    m_uiBij[i * d + k] * m_uiInvCij[k * d + j] * tmp;
                tmp = tmp * inv_dtf;
            }
        }

    // M_ij has the correct operator.
    // Vin  : 0, 1, 2, .. m_uiNumStages-1 these are RK stages, m_uiNumStages
    // contains the U_f or U_c. Vout : 0, 1, 2, .. m_uiNumStages-1 contains the
    // correct stages to compute correct U form U_f or U_c
    for (unsigned int v = 0; v < dof; v++) {
        for (unsigned int n = 0; n < NN; n++) {
            for (unsigned int s = 0; s < d; s++) {
                m_uiVin[s]  = in[v * (d + 1) + (s)][n];
                m_uiVout[s] = 0;
            }

            for (unsigned int si = 0; si < m_uiNumStages; si++)
                for (unsigned int sj = 0; sj < m_uiNumStages; sj++)
                    m_uiVout[si] += m_uiMij[si * d + sj] * m_uiVin[sj];

            double dt_by_fac_p = 1.0;

            out[v * (d + 1) + m_uiNumStages][n] =
                in[v * (d + 1) + m_uiNumStages][n];

            for (unsigned int si = 0; si < m_uiNumStages; si++) {
                dt_by_fac_p *= (dt_f / ((double)(si + 1)));
                out[v * (d + 1) + m_uiNumStages][n] +=
                    (dt_by_fac_p * m_uiVout[si]);
            }
            // printf("out s %d n %d  val %f \n", rk_s-1,n, out[v*rk_s +
            // rk_s-1][n]); printf("finer to coarser: in value: %f out value: %f
            // \n",in[v*(d+1) + m_uiNumStages][n],out[v*(d+1) +
            // m_uiNumStages][n]);
        }
    }

    return;
}

}  // namespace ts
