#include "rhs.h"

#include "compact_derivs.h"

using namespace std;
using namespace nlsm;
using namespace dendro_cfd;

/*----------------------------------------------------------------------;
 *
 * RHS for non-linear sigma model
 *
 *----------------------------------------------------------------------*/
void nlsmRhs(double **unzipVarsRHS, const double **uZipVars,
             const unsigned int &offset, const double *pmin, const double *pmax,
             const unsigned int *sz, const unsigned int &bflag) {
    double *const deriv_base = nlsm::NLSM_DERIV_WORKSPACE;

    const double *chi = &uZipVars[VAR::U_CHI][offset];
    const double *phi = &uZipVars[VAR::U_PHI][offset];

    double *chi_rhs = &unzipVarsRHS[VAR::U_CHI][offset];
    double *phi_rhs = &unzipVarsRHS[VAR::U_PHI][offset];

    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];

    double hx = (pmax[0] - pmin[0]) / (nx - 1);
    double hy = (pmax[1] - pmin[1]) / (ny - 1);
    double hz = (pmax[2] - pmin[2]) / (nz - 1);

    int idx[3];

    unsigned int n = sz[0] * sz[1] * sz[2];

    const unsigned int BLK_SZ = n;

    nlsm::timer::t_deriv.start();

    double *grad_0_chi = deriv_base + 0 * BLK_SZ;
    double *grad_1_chi = deriv_base + 1 * BLK_SZ;
    double *grad_2_chi = deriv_base + 2 * BLK_SZ;

    double *grad_0_phi = deriv_base + 3 * BLK_SZ;
    double *grad_1_phi = deriv_base + 4 * BLK_SZ;
    double *grad_2_phi = deriv_base + 5 * BLK_SZ;

    double *grad2_0_0_chi = deriv_base + 6 * BLK_SZ;
    double *grad2_1_1_chi = deriv_base + 7 * BLK_SZ;
    double *grad2_2_2_chi = deriv_base + 8 * BLK_SZ;

    // double *grad_0_chi = new double[n];
    // double *grad_1_chi = new double[n];
    // double *grad_2_chi = new double[n];

    // double *grad_0_phi = new double[n];
    // double *grad_1_phi = new double[n];
    // double *grad_2_phi = new double[n];

    // double *grad2_0_0_chi = new double[n];
    // double *grad2_1_1_chi = new double[n];
    // double *grad2_2_2_chi = new double[n];

    deriv_xx(grad2_0_0_chi, chi, hx, sz, bflag);
    deriv_yy(grad2_1_1_chi, chi, hy, sz, bflag);
    deriv_zz(grad2_2_2_chi, chi, hz, sz, bflag);

    nlsm::timer::t_deriv.stop();

    register double x;
    register double y;
    register double z;
    register unsigned int pp;

    double r;
    double eta;
    const unsigned int PW = nlsm::NLSM_PADDING_WIDTH;

    // cout << "begin loop" << endl;
    for (unsigned int k = PW; k < nz - PW; k++) {
        z = pmin[2] + k * hz;

        for (unsigned int j = PW; j < ny - PW; j++) {
            y = pmin[1] + j * hy;

            for (unsigned int i = PW; i < nx - PW; i++) {
                x = pmin[0] + i * hx;
                pp = i + nx * (j + ny * k);
                r = sqrt(x * x + y * y + z * z);

                nlsm::timer::t_rhs.start();
#ifdef NLSM_NONLINEAR
                if (r > 1.0e-6) {
                    phi_rhs[pp] = NLSM_WAVE_SPEED_X * grad2_0_0_chi[pp] +
                                  NLSM_WAVE_SPEED_Y * grad2_1_1_chi[pp] +
                                  NLSM_WAVE_SPEED_Z * grad2_2_2_chi[pp] -
                                  sin(2 * chi[pp]) / pow(r, 2);
                    chi_rhs[pp] = phi[pp];
                } else {
                    chi_rhs[pp] = 0.0;
                    phi_rhs[pp] = 0.0;
                }
#else
                phi_rhs[pp] = NLSM_WAVE_SPEED_X * grad2_0_0_chi[pp] +
                              NLSM_WAVE_SPEED_Y * grad2_1_1_chi[pp] +
                              NLSM_WAVE_SPEED_Z * grad2_2_2_chi[pp];
                chi_rhs[pp] = phi[pp];
#endif
                nlsm::timer::t_rhs.stop();
            }
        }
    }

    if (bflag != 0) {
        nlsm::timer::t_bdyc.start();

        deriv_x(grad_0_chi, chi, hx, sz, bflag);
        deriv_y(grad_1_chi, chi, hy, sz, bflag);
        deriv_z(grad_2_chi, chi, hz, sz, bflag);

        deriv_x(grad_0_phi, phi, hx, sz, bflag);
        deriv_y(grad_1_phi, phi, hy, sz, bflag);
        deriv_z(grad_2_phi, phi, hz, sz, bflag);

        nlsm_bcs(chi_rhs, chi, grad_0_chi, grad_1_chi, grad_2_chi, pmin, pmax,
                 1.0, 0.0, sz, bflag);
        nlsm_bcs(phi_rhs, phi, grad_0_phi, grad_1_phi, grad_2_phi, pmin, pmax,
                 1.0, 0.0, sz, bflag);
        nlsm::timer::t_bdyc.stop();
    }

    nlsm::timer::t_deriv.start();
    ko_deriv_x(grad_0_chi, chi, hx, sz, bflag);
    ko_deriv_y(grad_1_chi, chi, hy, sz, bflag);
    ko_deriv_z(grad_2_chi, chi, hz, sz, bflag);

    ko_deriv_x(grad_0_phi, phi, hx, sz, bflag);
    ko_deriv_y(grad_1_phi, phi, hy, sz, bflag);
    ko_deriv_z(grad_2_phi, phi, hz, sz, bflag);
    nlsm::timer::t_deriv.stop();

    nlsm::timer::t_rhs.start();

    const double sigma = KO_DISS_SIGMA;

    for (unsigned int k = PW; k < nz - PW; k++) {
        for (unsigned int j = PW; j < ny - PW; j++) {
            for (unsigned int i = PW; i < nx - PW; i++) {
                pp = i + nx * (j + ny * k);

                chi_rhs[pp] +=
                    sigma * (grad_0_chi[pp] + grad_1_chi[pp] + grad_2_chi[pp]);
                phi_rhs[pp] +=
                    sigma * (grad_0_phi[pp] + grad_1_phi[pp] + grad_2_phi[pp]);
            }
        }
    }

    nlsm::timer::t_rhs.stop();

    nlsm::timer::t_deriv.start();

    // delete[] grad2_0_0_chi;
    // delete[] grad2_1_1_chi;
    // delete[] grad2_2_2_chi;

    // delete[] grad_0_chi;
    // delete[] grad_1_chi;
    // delete[] grad_2_chi;

    // delete[] grad_0_phi;
    // delete[] grad_1_phi;
    // delete[] grad_2_phi;

    nlsm::timer::t_deriv.stop();

#if 0
    for (unsigned int m = 0; m < 24; m++) {
      std::cout<<"  || dtu("<<m<<")|| = "<<normLInfty(unzipVarsRHS[m] + offset, n)<<std::endl;
    }
#endif
}

/*----------------------------------------------------------------------;
 *
 * RHS for non-linear sigma model
 *
 *----------------------------------------------------------------------*/
void nlsmRhs_COMPACT(double **unzipVarsRHS, double **uZipVars,
                     const unsigned int &offset, const double *pmin,
                     const double *pmax, const unsigned int *sz,
                     const unsigned int &bflag) {
    double *const deriv_base = nlsm::NLSM_DERIV_WORKSPACE;

    double *const chi = &uZipVars[VAR::U_CHI][offset];
    double *const phi = &uZipVars[VAR::U_PHI][offset];

    double *const chi_rhs = &unzipVarsRHS[VAR::U_CHI][offset];
    double *const phi_rhs = &unzipVarsRHS[VAR::U_PHI][offset];

    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];

    double hx = (pmax[0] - pmin[0]) / (nx - 1);
    double hy = (pmax[1] - pmin[1]) / (ny - 1);
    double hz = (pmax[2] - pmin[2]) / (nz - 1);

    int idx[3];

    unsigned int n = sz[0] * sz[1] * sz[2];

    const unsigned int BLK_SZ = n;

    nlsm::timer::t_deriv.start();

    double *grad_0_chi = deriv_base + 0 * BLK_SZ;
    double *grad_1_chi = deriv_base + 1 * BLK_SZ;
    double *grad_2_chi = deriv_base + 2 * BLK_SZ;

    double *grad_0_phi = deriv_base + 3 * BLK_SZ;
    double *grad_1_phi = deriv_base + 4 * BLK_SZ;
    double *grad_2_phi = deriv_base + 5 * BLK_SZ;

    double *grad2_0_0_chi = deriv_base + 6 * BLK_SZ;
    double *grad2_1_1_chi = deriv_base + 7 * BLK_SZ;
    double *grad2_2_2_chi = deriv_base + 8 * BLK_SZ;

    // double *grad_0_chi = new double[n];
    // double *grad_1_chi = new double[n];
    // double *grad_2_chi = new double[n];

    // double *grad_0_phi = new double[n];
    // double *grad_1_phi = new double[n];
    // double *grad_2_phi = new double[n];

    // double *grad2_0_0_chi = new double[n];
    // double *grad2_1_1_chi = new double[n];
    // double *grad2_2_2_chi = new double[n];

    // compute the derivatives

    if (nlsm::NLSM_2ND_DERIV_TYPE == CFD2ND_NONE) {
        deriv_xx(grad2_0_0_chi, chi, hx, sz, bflag);
        deriv_yy(grad2_1_1_chi, chi, hy, sz, bflag);
        deriv_zz(grad2_2_2_chi, chi, hz, sz, bflag);
    } else {
        cfd.cfd_xx(grad2_0_0_chi, chi, hx, sz, bflag);
        cfd.cfd_yy(grad2_1_1_chi, chi, hy, sz, bflag);
        cfd.cfd_zz(grad2_2_2_chi, chi, hz, sz, bflag);
    }

    // std::cout << "Grad 2 0 0" << std::endl;
    // dendro_cfd::print_square_mat(grad2_0_0_chi, sz[0]);

    nlsm::timer::t_deriv.stop();

    register double x;
    register double y;
    register double z;
    register unsigned int pp;

    double r;
    double eta;
    const unsigned int PW = nlsm::NLSM_PADDING_WIDTH;

    // cout << "begin loop" << endl;
    for (unsigned int k = PW; k < nz - PW; k++) {
        z = pmin[2] + k * hz;

        for (unsigned int j = PW; j < ny - PW; j++) {
            y = pmin[1] + j * hy;

            for (unsigned int i = PW; i < nx - PW; i++) {
                x = pmin[0] + i * hx;
                pp = i + nx * (j + ny * k);
                r = sqrt(x * x + y * y + z * z);

                nlsm::timer::t_rhs.start();
#ifdef NLSM_NONLINEAR
                if (r > 1.0e-6) {
                    phi_rhs[pp] = NLSM_WAVE_SPEED_X * grad2_0_0_chi[pp] +
                                  NLSM_WAVE_SPEED_Y * grad2_1_1_chi[pp] +
                                  NLSM_WAVE_SPEED_Z * grad2_2_2_chi[pp] -
                                  sin(2 * chi[pp]) / pow(r, 2);
                    chi_rhs[pp] = phi[pp];
                } else {
                    chi_rhs[pp] = 0.0;
                    phi_rhs[pp] = 0.0;
                }
#else
                phi_rhs[pp] = NLSM_WAVE_SPEED_X * grad2_0_0_chi[pp] +
                              NLSM_WAVE_SPEED_Y * grad2_1_1_chi[pp] +
                              NLSM_WAVE_SPEED_Z * grad2_2_2_chi[pp];
                chi_rhs[pp] = phi[pp];
#endif
                nlsm::timer::t_rhs.stop();
            }
        }
    }

    if (bflag != 0) {
        nlsm::timer::t_bdyc.start();

        if (nlsm::NLSM_DERIV_TYPE == CFD_NONE) {
            // std::cout << "doing not cfd deriv for bdy" << std::endl;
            deriv_x(grad_0_chi, chi, hx, sz, bflag);
            deriv_y(grad_1_chi, chi, hy, sz, bflag);
            deriv_z(grad_2_chi, chi, hz, sz, bflag);

            deriv_x(grad_0_phi, phi, hx, sz, bflag);
            deriv_y(grad_1_phi, phi, hy, sz, bflag);
            deriv_z(grad_2_phi, phi, hz, sz, bflag);
        } else {
            // std::cout << "doing CFD** deriv for bdy" << std::endl;
            cfd.cfd_x(grad_0_chi, chi, hx, sz, bflag);
            cfd.cfd_y(grad_1_chi, chi, hy, sz, bflag);
            cfd.cfd_z(grad_2_chi, chi, hz, sz, bflag);

            cfd.cfd_x(grad_0_phi, phi, hx, sz, bflag);
            cfd.cfd_y(grad_1_phi, phi, hy, sz, bflag);
            cfd.cfd_z(grad_2_phi, phi, hz, sz, bflag);
        }

        nlsm_bcs(chi_rhs, chi, grad_0_chi, grad_1_chi, grad_2_chi, pmin, pmax,
                 1.0, 0.0, sz, bflag);
        nlsm_bcs(phi_rhs, phi, grad_0_phi, grad_1_phi, grad_2_phi, pmin, pmax,
                 1.0, 0.0, sz, bflag);
        nlsm::timer::t_bdyc.stop();
    }

    
    if (nlsm::NLSM_FILTER_TYPE == FILT_KO_DISS) {
        // std::cout << "DOING KO DISS" << std::endl;
        nlsm::timer::t_deriv.start();
        ko_deriv_x(grad_0_chi, chi, hx, sz, bflag);
        ko_deriv_y(grad_1_chi, chi, hy, sz, bflag);
        ko_deriv_z(grad_2_chi, chi, hz, sz, bflag);

        ko_deriv_x(grad_0_phi, phi, hx, sz, bflag);
        ko_deriv_y(grad_1_phi, phi, hy, sz, bflag);
        ko_deriv_z(grad_2_phi, phi, hz, sz, bflag);

        nlsm::timer::t_deriv.stop();

        nlsm::timer::t_rhs.start();

        const double sigma = KO_DISS_SIGMA;

        for (unsigned int k = PW; k < nz - PW; k++) {
            for (unsigned int j = PW; j < ny - PW; j++) {
                for (unsigned int i = PW; i < nx - PW; i++) {
                    pp = i + nx * (j + ny * k);

                    chi_rhs[pp] += sigma * (grad_0_chi[pp] + grad_1_chi[pp] +
                                            grad_2_chi[pp]);
                    phi_rhs[pp] += sigma * (grad_0_phi[pp] + grad_1_phi[pp] +
                                            grad_2_phi[pp]);
                }
            }
        }

        nlsm::timer::t_rhs.stop();

    } else {
        nlsm::timer::t_deriv.start();
        nlsm::timer::t_rhs.start();
        cfd.filter_cfd_x(chi, grad_0_chi, hx, sz, bflag);
        cfd.filter_cfd_y(chi, grad_1_chi, hy, sz, bflag);
        cfd.filter_cfd_z(chi, grad_2_chi, hz, sz, bflag);

        cfd.filter_cfd_x(phi, grad_0_phi, hx, sz, bflag);
        cfd.filter_cfd_y(phi, grad_1_phi, hy, sz, bflag);
        cfd.filter_cfd_z(phi, grad_2_phi, hz, sz, bflag);

        nlsm::timer::t_deriv.stop();
        nlsm::timer::t_rhs.stop();
    }

    

    nlsm::timer::t_deriv.start();

    // delete[] grad2_0_0_chi;
    // delete[] grad2_1_1_chi;
    // delete[] grad2_2_2_chi;

    // delete[] grad_0_chi;
    // delete[] grad_1_chi;
    // delete[] grad_2_chi;

    // delete[] grad_0_phi;
    // delete[] grad_1_phi;
    // delete[] grad_2_phi;

    nlsm::timer::t_deriv.stop();

#if 0
    for (unsigned int m = 0; m < 24; m++) {
      std::cout<<"  || dtu("<<m<<")|| = "<<normLInfty(unzipVarsRHS[m] + offset, n)<<std::endl;
    }
#endif
}

/*----------------------------------------------------------------------;
 *
 *
 *
 *----------------------------------------------------------------------*/
void nlsm_bcs(double *f_rhs, const double *f, const double *dxf,
              const double *dyf, const double *dzf, const double *pmin,
              const double *pmax, const double f_falloff,
              const double f_asymptotic, const unsigned int *sz,
              const unsigned int &bflag) {
    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];

    double hx = (pmax[0] - pmin[0]) / (nx - 1);
    double hy = (pmax[1] - pmin[1]) / (ny - 1);
    double hz = (pmax[2] - pmin[2]) / (nz - 1);

    const unsigned int PW = nlsm::NLSM_PADDING_WIDTH;
    unsigned int ib = PW;
    unsigned int jb = PW;
    unsigned int kb = PW;
    unsigned int ie = sz[0] - PW;
    unsigned int je = sz[1] - PW;
    unsigned int ke = sz[2] - PW;

    double x, y, z;
    unsigned int pp;
    double inv_r;

    if (bflag & (1u << OCT_DIR_LEFT)) {
        double x = pmin[0] + ib * hx;
        for (unsigned int k = kb; k < ke; k++) {
            z = pmin[2] + k * hz;
            for (unsigned int j = jb; j < je; j++) {
                y = pmin[1] + j * hy;
                pp = IDX(ib, j, k);
                inv_r = 1.0 / sqrt(x * x + y * y + z * z);

#ifdef NLSM_DIRICHLET_BDY
                f_rhs[pp] = 0.0;
#else

                f_rhs[pp] = -inv_r * (x * dxf[pp] + y * dyf[pp] + z * dzf[pp] +
                                      f_falloff * (f[pp] - f_asymptotic));
#endif
            }
        }
    }

    if (bflag & (1u << OCT_DIR_RIGHT)) {
        x = pmin[0] + (ie - 1) * hx;
        for (unsigned int k = kb; k < ke; k++) {
            z = pmin[2] + k * hz;
            for (unsigned int j = jb; j < je; j++) {
                y = pmin[1] + j * hy;
                pp = IDX((ie - 1), j, k);
                inv_r = 1.0 / sqrt(x * x + y * y + z * z);

#ifdef NLSM_DIRICHLET_BDY
                f_rhs[pp] = 0.0;
#else

                f_rhs[pp] = -inv_r * (x * dxf[pp] + y * dyf[pp] + z * dzf[pp] +
                                      f_falloff * (f[pp] - f_asymptotic));
#endif
            }
        }
    }

    if (bflag & (1u << OCT_DIR_DOWN)) {
        y = pmin[1] + jb * hy;
        for (unsigned int k = kb; k < ke; k++) {
            z = pmin[2] + k * hz;
            for (unsigned int i = ib; i < ie; i++) {
                x = pmin[0] + i * hx;
                inv_r = 1.0 / sqrt(x * x + y * y + z * z);
                pp = IDX(i, jb, k);

#ifdef NLSM_DIRICHLET_BDY
                f_rhs[pp] = 0.0;
#else

                f_rhs[pp] = -inv_r * (x * dxf[pp] + y * dyf[pp] + z * dzf[pp] +
                                      f_falloff * (f[pp] - f_asymptotic));
#endif
            }
        }
    }

    if (bflag & (1u << OCT_DIR_UP)) {
        y = pmin[1] + (je - 1) * hy;
        for (unsigned int k = kb; k < ke; k++) {
            z = pmin[2] + k * hz;
            for (unsigned int i = ib; i < ie; i++) {
                x = pmin[0] + i * hx;
                inv_r = 1.0 / sqrt(x * x + y * y + z * z);
                pp = IDX(i, (je - 1), k);

#ifdef NLSM_DIRICHLET_BDY
                f_rhs[pp] = 0.0;
#else

                f_rhs[pp] = -inv_r * (x * dxf[pp] + y * dyf[pp] + z * dzf[pp] +
                                      f_falloff * (f[pp] - f_asymptotic));
#endif
            }
        }
    }

    if (bflag & (1u << OCT_DIR_BACK)) {
        z = pmin[2] + kb * hz;
        for (unsigned int j = jb; j < je; j++) {
            y = pmin[1] + j * hy;
            for (unsigned int i = ib; i < ie; i++) {
                x = pmin[0] + i * hx;
                inv_r = 1.0 / sqrt(x * x + y * y + z * z);
                pp = IDX(i, j, kb);

#ifdef NLSM_DIRICHLET_BDY
                f_rhs[pp] = 0.0;
#else

                f_rhs[pp] = -inv_r * (x * dxf[pp] + y * dyf[pp] + z * dzf[pp] +
                                      f_falloff * (f[pp] - f_asymptotic));
#endif
            }
        }
    }

    if (bflag & (1u << OCT_DIR_FRONT)) {
        z = pmin[2] + (ke - 1) * hz;
        for (unsigned int j = jb; j < je; j++) {
            y = pmin[1] + j * hy;
            for (unsigned int i = ib; i < ie; i++) {
                x = pmin[0] + i * hx;
                inv_r = 1.0 / sqrt(x * x + y * y + z * z);
                pp = IDX(i, j, (ke - 1));

#ifdef NLSM_DIRICHLET_BDY
                f_rhs[pp] = 0.0;
#else

                f_rhs[pp] = -inv_r * (x * dxf[pp] + y * dyf[pp] + z * dzf[pp] +
                                      f_falloff * (f[pp] - f_asymptotic));
#endif
            }
        }
    }
}

/*----------------------------------------------------------------------;
 *
 *
 *
 *----------------------------------------------------------------------*/
