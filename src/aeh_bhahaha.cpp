#include "aeh_bhahaha.h"

#include "BHaHAHA.h"
#include "daUtils.h"

namespace dendro_aeh {
    struct HorizonMassSpinCharge {
    double area = 0.0;
    double Mirr = 0.0;
    double Q = 0.0;
    double Jx = 0.0;
    double Jy = 0.0;
    double Jz = 0.0;
    double Jmag = 0.0;
    double J = 0.0;
    double M = 0.0;
    double chi = 0.0;
    double D = 0.0;
};

HorizonMassSpinCharge compute_mass_spin_charge(
    const int which_horizon,
    const bhahaha_params_and_data_struct& bha,
    const bhahaha_diagnostics_struct& bhahaha_diags,
    const std::vector<double>& prev_horizon_m1,
    const double* emda_horizon_data,
    const int ntheta,
    const int nphi
) {
    HorizonMassSpinCharge q;

    const double dtheta = M_PI / static_cast<double>(ntheta);
    const double dphi   = 2.0 * M_PI / static_cast<double>(nphi);

    const size_t horizon_offset =
        static_cast<size_t>(which_horizon) *
        static_cast<size_t>(ntheta) *
        static_cast<size_t>(nphi);

    const int nr = bha.Nr_external_input;
    const double rmin_grid = bha.r_min_external_input;
    const double dr = bha.dr_external_input;
    int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    const bool debug_dilaton_charge =
        (mpi_rank == 0 && which_horizon == 0);

    const int total_pts = nr * ntheta * nphi;

    auto get_metric_data = [&](int field, int ir, int itheta, int iphi) {
        const int idx = idx3_spherical(ir, itheta, iphi, ntheta, nr);
        return bha.input_metric_data[field * total_pts + idx];
    };

    auto interp_metric_data = [&](int field, double r, int itheta, int iphi) {
        // BHaHAHA radial grid is cell-centered:
        // r_i = r_min + (i + 0.5) dr
        const double first_r =
            (rmin_grid > 0.0) ? rmin_grid + (0.5 - BHAHAHA_NGHOSTS) * dr
                              : rmin_grid + 0.5 * dr;
        double x = (r - first_r) / dr;

        int i0 = static_cast<int>(std::floor(x));
        double a = x - static_cast<double>(i0);

        if (i0 < 0) {
            i0 = 0;
            a = 0.0;
        }

        if (i0 >= nr - 1) {
            i0 = nr - 2;
            a = 1.0;
        }

        const double f0 = get_metric_data(field, i0,     itheta, iphi);
        const double f1 = get_metric_data(field, i0 + 1, itheta, iphi);

        return (1.0 - a) * f0 + a * f1;
    };

    auto get_emda_data = [&](int field, int ir, int itheta, int iphi) {
        if (!emda_horizon_data) return 0.0;
        const int idx = idx3_spherical(ir, itheta, iphi, ntheta, nr);
        return emda_horizon_data[field * total_pts + idx];
    };

    auto interp_emda_data = [&](int field, double r, int itheta, int iphi) {
        if (!emda_horizon_data) return 0.0;

        const double first_r =
            (rmin_grid > 0.0) ? rmin_grid + (0.5 - BHAHAHA_NGHOSTS) * dr
                              : rmin_grid + 0.5 * dr;
        double x = (r - first_r) / dr;

        int i0 = static_cast<int>(std::floor(x));
        double a = x - static_cast<double>(i0);

        if (i0 < 0) {
            i0 = 0;
            a = 0.0;
        }

        if (i0 >= nr - 1) {
            i0 = nr - 2;
            a = 1.0;
        }

        const double f0 = get_emda_data(field, i0,     itheta, iphi);
        const double f1 = get_emda_data(field, i0 + 1, itheta, iphi);

        return (1.0 - a) * f0 + a * f1;
    };

    // ------------------------------------------------------------------
    // Proper area and irreducible mass from BHaHAHA
    // ------------------------------------------------------------------
    q.area = bhahaha_diags.area;
    q.Mirr = std::sqrt(q.area / (16.0 * M_PI));

    // ------------------------------------------------------------------
    // Approximate spin integral
    //
    // Assumed input_metric_data layout:
    //   0: g_xx
    //   1: g_xy
    //   2: g_xz
    //   3: g_yy
    //   4: g_yz
    //   5: g_zz
    //   6: K_xx
    //   7: K_xy
    //   8: K_xz
    //   9: K_yy
    //  10: K_yz
    //  11: K_zz
    //
    // If transform_ uses a different order, fix the indices below.
    // ------------------------------------------------------------------
    double Q_integral = 0.0;
    double Jx_integral = 0.0;
    double Jy_integral = 0.0;
    double Jz_integral = 0.0;
    double D_integral = 0.0;
    double computed_area_from_dA = 0.0;
    double phi_sum = 0.0;
    double phi_min = 1.0e300;
    double phi_max = -1.0e300;
    double dphi_dr_sum = 0.0;
    double dphi_dr_min = 1.0e300;
    double dphi_dr_max = -1.0e300;
    int dilaton_debug_count = 0;
    int dilaton_debug_printed = 0;

    double rmin = 1.0e300;
    double rmax = -1.0e300;
    double rsum = 0.0;
    int count = 0;

    const double weights_2nd_order[1] = {1.0};
    const double weights_4th_order[4] = {
        13.0 / 12.0, 11.0 / 12.0, 11.0 / 12.0, 13.0 / 12.0
    };
    const double weights_8th_order[8] = {
        295627.0 / 241920.0, 71329.0 / 241920.0,
        17473.0 / 8960.0,   128953.0 / 241920.0,
        128953.0 / 241920.0, 17473.0 / 8960.0,
        71329.0 / 241920.0,  295627.0 / 241920.0
    };

    const double* integration_weights = weights_2nd_order;
    int weight_stencil_size = 1;
    if (ntheta % 8 == 0 && nphi % 8 == 0) {
        integration_weights = weights_8th_order;
        weight_stencil_size = 8;
    } else if (ntheta % 4 == 0 && nphi % 4 == 0) {
        integration_weights = weights_4th_order;
        weight_stencil_size = 4;
    }

    const auto horizon_r = [&](int theta_idx, int phi_idx) {
        while (theta_idx < 0 || theta_idx >= ntheta) {
            if (theta_idx < 0) {
                theta_idx = -theta_idx - 1;
            } else {
                theta_idx = 2 * ntheta - theta_idx - 1;
            }
            phi_idx += nphi / 2;
        }

        phi_idx %= nphi;
        if (phi_idx < 0) {
            phi_idx += nphi;
        }

        const size_t idx =
            horizon_offset +
            static_cast<size_t>(phi_idx) * static_cast<size_t>(ntheta) +
            static_cast<size_t>(theta_idx);
        return prev_horizon_m1[idx];
    };

    for (int iphi = 0; iphi < nphi; ++iphi) {
        const double phi =
            -M_PI + (static_cast<double>(iphi) + 0.5) * dphi;

        const double sinphi = std::sin(phi);
        const double cosphi = std::cos(phi);

        for (int itheta = 0; itheta < ntheta; ++itheta) {
            const double theta =
                (static_cast<double>(itheta) + 0.5) * dtheta;

            const double sintheta = std::sin(theta);
            const double costheta = std::cos(theta);

            const double r = horizon_r(itheta, iphi);

            const double r_theta =
                ((1.0 / 60.0) *
                     (-horizon_r(itheta - 3, iphi) +
                       horizon_r(itheta + 3, iphi)) +
                 (3.0 / 20.0) *
                     ( horizon_r(itheta - 2, iphi) -
                       horizon_r(itheta + 2, iphi)) +
                 (3.0 / 4.0) *
                     (-horizon_r(itheta - 1, iphi) +
                       horizon_r(itheta + 1, iphi))) /
                dtheta;

            const double r_phi =
                ((1.0 / 60.0) *
                     (-horizon_r(itheta, iphi - 3) +
                       horizon_r(itheta, iphi + 3)) +
                 (3.0 / 20.0) *
                     ( horizon_r(itheta, iphi - 2) -
                       horizon_r(itheta, iphi + 2)) +
                 (3.0 / 4.0) *
                     (-horizon_r(itheta, iphi - 1) +
                       horizon_r(itheta, iphi + 1))) /
                dphi;

            rmin = std::min(rmin, r);
            rmax = std::max(rmax, r);
            rsum += r;
            count++;

            const double sx = sintheta * cosphi;
            const double sy = sintheta * sinphi;
            const double sz = costheta;

            const double ethetax = costheta * cosphi;
            const double ethetay = costheta * sinphi;
            const double ethetaz = -sintheta;

            const double ephix = -sintheta * sinphi;
            const double ephiy = sintheta * cosphi;
            const double ephiz = 0.0;

            // Cartesian position relative to the horizon center.
            const double xrel = r * sx;
            const double yrel = r * sy;
            const double zrel = r * sz;

            const double Xtheta_x = r_theta * sx + r * ethetax;
            const double Xtheta_y = r_theta * sy + r * ethetay;
            const double Xtheta_z = r_theta * sz + r * ethetaz;

            const double Xphi_x = r_phi * sx + r * ephix;
            const double Xphi_y = r_phi * sy + r * ephiy;
            const double Xphi_z = r_phi * sz + r * ephiz;

            const double gxx = interp_metric_data(0, r, itheta, iphi);
            const double gxy = interp_metric_data(1, r, itheta, iphi);
            const double gxz = interp_metric_data(2, r, itheta, iphi);
            const double gyy = interp_metric_data(3, r, itheta, iphi);
            const double gyz = interp_metric_data(4, r, itheta, iphi);
            const double gzz = interp_metric_data(5, r, itheta, iphi);

            // Extrinsic curvature interpolated to the horizon surface.
            const double Kxx = interp_metric_data(6,  r, itheta, iphi);
            const double Kxy = interp_metric_data(7,  r, itheta, iphi);
            const double Kxz = interp_metric_data(8,  r, itheta, iphi);
            const double Kyy = interp_metric_data(9,  r, itheta, iphi);
            const double Kyz = interp_metric_data(10, r, itheta, iphi);
            const double Kzz = interp_metric_data(11, r, itheta, iphi);

            const double Ex = interp_emda_data(0, r, itheta, iphi);
            const double Ey = interp_emda_data(1, r, itheta, iphi);
            const double Ez = interp_emda_data(2, r, itheta, iphi);

            // Approximate covariant normal s_i = gamma_ij s^j.
            // TODO: replace this radial normal and uniform dA with true
            // surface geometry from angular derivatives of r(theta,phi).
            const double scovx = gxx * sx + gxy * sy + gxz * sz;
            const double scovy = gxy * sx + gyy * sy + gyz * sz;
            const double scovz = gxz * sx + gyz * sy + gzz * sz;

            const double detg =
                gxx * (gyy * gzz - gyz * gyz) -
                gxy * (gxy * gzz - gyz * gxz) +
                gxz * (gxy * gyz - gyy * gxz);

            double Ktrace = 0.0;
            double igxx = 0.0;
            double igxy = 0.0;
            double igxz = 0.0;
            double igyy = 0.0;
            double igyz = 0.0;
            double igzz = 0.0;
            if (std::abs(detg) > 0.0) {
                igxx = (gyy * gzz - gyz * gyz) / detg;
                igxy = (gxz * gyz - gxy * gzz) / detg;
                igxz = (gxy * gyz - gxz * gyy) / detg;
                igyy = (gxx * gzz - gxz * gxz) / detg;
                igyz = (gxy * gxz - gxx * gyz) / detg;
                igzz = (gxx * gyy - gxy * gxy) / detg;

                Ktrace = igxx * Kxx + 2.0 * igxy * Kxy +
                         2.0 * igxz * Kxz + igyy * Kyy +
                         2.0 * igyz * Kyz + igzz * Kzz;
            }

            const auto gamma_dot = [&](double ax, double ay, double az,
                                       double bx, double by, double bz) {
                return ax * (gxx * bx + gxy * by + gxz * bz) +
                       ay * (gxy * bx + gyy * by + gyz * bz) +
                       az * (gxz * bx + gyz * by + gzz * bz);
            };

            const double qtt = gamma_dot(Xtheta_x, Xtheta_y, Xtheta_z,
                                         Xtheta_x, Xtheta_y, Xtheta_z);
            const double qtp = gamma_dot(Xtheta_x, Xtheta_y, Xtheta_z,
                                         Xphi_x, Xphi_y, Xphi_z);
            const double qpp = gamma_dot(Xphi_x, Xphi_y, Xphi_z,
                                         Xphi_x, Xphi_y, Xphi_z);
            const double qdet = qtt * qpp - qtp * qtp;
            const double weight_theta =
                integration_weights[itheta % weight_stencil_size];
            const double weight_phi =
                integration_weights[iphi % weight_stencil_size];
            const double dA = (qdet > 0.0)
                                  ? std::sqrt(qdet) * weight_theta *
                                        weight_phi * dtheta * dphi
                                  : 0.0;
            computed_area_from_dA += dA;

            // Surface normal from finite-difference tangents. This replaces
            // the old flat radial normal but still depends on finite-difference
            // angular derivatives of r(theta,phi).
            double ncovx = Xtheta_y * Xphi_z - Xtheta_z * Xphi_y;
            double ncovy = Xtheta_z * Xphi_x - Xtheta_x * Xphi_z;
            double ncovz = Xtheta_x * Xphi_y - Xtheta_y * Xphi_x;

            double nconx = igxx * ncovx + igxy * ncovy + igxz * ncovz;
            double ncony = igxy * ncovx + igyy * ncovy + igyz * ncovz;
            double nconz = igxz * ncovx + igyz * ncovy + igzz * ncovz;
            double nnorm2 = ncovx * nconx + ncovy * ncony + ncovz * nconz;

            if (nnorm2 > 0.0) {
                const double inv_norm = 1.0 / std::sqrt(nnorm2);
                ncovx *= inv_norm;
                ncovy *= inv_norm;
                ncovz *= inv_norm;
                nconx *= inv_norm;
                ncony *= inv_norm;
                nconz *= inv_norm;
            } else {
                ncovx = scovx;
                ncovy = scovy;
                ncovz = scovz;
                nconx = sx;
                ncony = sy;
                nconz = sz;
            }

            const double Pxx = Kxx - Ktrace * gxx;
            const double Pxy = Kxy - Ktrace * gxy;
            const double Pxz = Kxz - Ktrace * gxz;
            const double Pyy = Kyy - Ktrace * gyy;
            const double Pyz = Kyz - Ktrace * gyz;
            const double Pzz = Kzz - Ktrace * gzz;

            const double Psx = Pxx * nconx + Pxy * ncony + Pxz * nconz;
            const double Psy = Pxy * nconx + Pyy * ncony + Pyz * nconz;
            const double Psz = Pxz * nconx + Pyz * ncony + Pzz * nconz;

            // Approximate rotational vector fields about the horizon centroid.
            const double phix_x = 0.0;
            const double phix_y = -zrel;
            const double phix_z = yrel;
            const double phiy_x = zrel;
            const double phiy_y = 0.0;
            const double phiy_z = -xrel;
            const double phiz_x = -yrel;
            const double phiz_y = xrel;
            const double phiz_z = 0.0;

            Q_integral += (Ex * ncovx + Ey * ncovy + Ez * ncovz) * dA;
            {
                const double phi_minus =
                    interp_emda_data(6, r - dr, itheta, iphi);
                const double phi_center =
                    interp_emda_data(6, r, itheta, iphi);
                const double phi_plus =
                    interp_emda_data(6, r + dr, itheta, iphi);
                const double dphi_dr =
                    (phi_plus - phi_minus) / (2.0 * dr);
                D_integral += dphi_dr * dA;

                phi_sum += phi_center;
                phi_min = std::min(phi_min, phi_center);
                phi_max = std::max(phi_max, phi_center);
                dphi_dr_sum += dphi_dr;
                dphi_dr_min = std::min(dphi_dr_min, dphi_dr);
                dphi_dr_max = std::max(dphi_dr_max, dphi_dr);
                dilaton_debug_count++;

                if (debug_dilaton_charge && dilaton_debug_printed < 5) {
                    const double contribution =
                        -(1.0 / (4.0 * M_PI)) * dphi_dr * dA;
                    std::cout << "[AEH DEBUG] dilaton point="
                              << dilaton_debug_printed
                              << " phi_minus=" << phi_minus
                              << " phi_center=" << phi_center
                              << " phi_plus=" << phi_plus
                              << " dphi_dr=" << dphi_dr
                              << " dA=" << dA
                              << " contribution=" << contribution
                              << std::endl;
                    dilaton_debug_printed++;
                }
            }
            Jx_integral +=
                (phix_x * Psx + phix_y * Psy + phix_z * Psz) * dA;
            Jy_integral +=
                (phiy_x * Psx + phiy_y * Psy + phiy_z * Psz) * dA;
            Jz_integral +=
                (phiz_x * Psx + phiz_y * Psy + phiz_z * Psz) * dA;
        }
    }

    const double rmean =
        (count > 0) ? (rsum / static_cast<double>(count)) : 0.0;

    std::cout << "[DEBUG AH RADIUS] horizon=" << which_horizon
              << " rmin=" << rmin
              << " rmax=" << rmax
              << " rmean=" << rmean
              << std::endl;

    q.Q = (1.0 / (4.0 * M_PI)) * Q_integral;
    q.D = -(1.0 / (4.0 * M_PI)) * D_integral;
    q.Jx = (1.0 / (8.0 * M_PI)) * Jx_integral;
    q.Jy = (1.0 / (8.0 * M_PI)) * Jy_integral;
    q.Jz = (1.0 / (8.0 * M_PI)) * Jz_integral;
    q.Jmag = std::sqrt(q.Jx * q.Jx + q.Jy * q.Jy + q.Jz * q.Jz);
    q.J = q.Jmag;

    // Magnetic charge is intentionally left at zero for now.

    // ------------------------------------------------------------------
    // Mass and dimensionless spin
    //
    // Temporary spinning, uncharged Christodoulou formula:
    // M^2 = M_irr^2 + J^2 / (4 M_irr^2)
    // ------------------------------------------------------------------
    q.M = std::sqrt(
        q.Mirr * q.Mirr +
        (q.Jmag * q.Jmag) / (4.0 * q.Mirr * q.Mirr)
    );

    q.chi = q.Jmag / (q.M * q.M);

    if (debug_dilaton_charge) {
        const double inv_count =
            (dilaton_debug_count > 0)
                ? 1.0 / static_cast<double>(dilaton_debug_count)
                : 0.0;
        std::cout << "[AEH DEBUG] dilaton summary"
                  << " avg_phi=" << phi_sum * inv_count
                  << " min_phi=" << phi_min
                  << " max_phi=" << phi_max
                  << " avg_dphi_dr=" << dphi_dr_sum * inv_count
                  << " min_dphi_dr=" << dphi_dr_min
                  << " max_dphi_dr=" << dphi_dr_max
                  << " sum_dA=" << computed_area_from_dA
                  << " final_D=" << q.D
                  << std::endl;
    }

    std::cout << "[AEH DEBUG] computed_area_from_dA="
              << computed_area_from_dA
              << " bhahaha_area=" << q.area
              << " QE=" << q.Q
              << " Jx=" << q.Jx
              << " Jy=" << q.Jy
              << " Jz=" << q.Jz
              << " Jmag=" << q.Jmag
              << " chi=" << q.chi
              << " dilaton_charge=" << q.D
              << std::endl;

    if (q.area > 0.0) {
        const double rel_area_diff =
            std::abs(computed_area_from_dA - q.area) / q.area;
        if (rel_area_diff > 0.05) {
            std::cout << "[AEH WARNING] computed_area_from_dA differs from "
                      << "BHaHAHA area by " << rel_area_diff
                      << " for horizon=" << which_horizon << std::endl;
        }
    }

    return q;
}
void AEH_BHaHAHA::find_horizons(
    const ot::Mesh* mesh, const double** var, const unsigned int current_step,
    const double current_time, const std::vector<Point> tracked_location_data) {
    // get the active comm and information about it
    unsigned int rankActive  = mesh->getMPIRank();
    unsigned int npesActive  = mesh->getMPICommSize();
    MPI_Comm commActive      = mesh->getMPICommunicator();
    unsigned int globalRank  = mesh->getMPIRankGlobal();
    // this is the function where we're going to find everything

    // then indices for BH1, BH2 (inspiral horizons) and common horizon
    const int inspiral_bh1   = 0;
    const int inspiral_bh2   = 1;
    const int common_horizon = 2;

    if (!mesh->isActive()) return;

    if (!globalRank)
        std::cout << "Now beginning BHaHAHA_find_horizons..." << std::endl;

    // Function flow is:
    // -------------------------
    // 1: Initialization and Parameter Setup

    // TODO: exit early? and validate maximum number of horizons (this can be
    // done in par reading)

    // start the profiling
    bhahaha_profiler_.start();

    if (!initialized_) {
        throw std::runtime_error("BHaHAHA Was not initialized correctly!");
    }

    // start by reconstructing the bha_param_data_ vector and populate with
    // input parameters
    create_bhahaha_param_structs();

    // then update with the current times and steps

    // need to update the data structures with the current step and time
    for (unsigned int which_horizon = 0; which_horizon < num_horizons_;
         which_horizon++) {
        bha_param_data_[which_horizon].time_external_input      = current_time;
        bha_param_data_[which_horizon].iteration_external_input = current_step;
    }
    // ------- END STEP 1 --------

    // 2: Horizon guess extrapolation
    // -- use Lagrange exptrapolation based on data from up to the last three
    // times
    // -- if horizon was never found, force a full-sphere guess
    // -- guesses restrict the volume searched for horizon
    for (int which_horizon = 0; which_horizon < num_horizons_;
         which_horizon++) {
        bhahaha_params_and_data_struct* bah_params_and_data =
            &bha_param_data_[which_horizon];

        // predict via Lagrange extrapolation or default to last-known data
        double x_center, y_center, z_center, r_min, r_max;

#if 0
        if (rankActive == 0) {
            std::cout << which_horizon << ": bah_params_and_data x centers: "
                      << bah_params_and_data->x_center_m1 << " "
                      << bah_params_and_data->x_center_m2 << " "
                      << bah_params_and_data->x_center_m3 << " " << std::endl;

            std::cout << which_horizon << ": bah_params_and_data y centers: "
                      << bah_params_and_data->y_center_m1 << " "
                      << bah_params_and_data->y_center_m2 << " "
                      << bah_params_and_data->y_center_m3 << " " << std::endl;

            std::cout << which_horizon << ": bah_params_and_data z centers: "
                      << bah_params_and_data->z_center_m1 << " "
                      << bah_params_and_data->z_center_m2 << " "
                      << bah_params_and_data->z_center_m3 << " " << std::endl;

            std::cout << which_horizon << ": bah_params_and_data t: "
                      << bah_params_and_data->t_m1 << " "
                      << bah_params_and_data->t_m2 << " "
                      << bah_params_and_data->t_m3 << " " << std::endl;
        }
#endif

        bah_xyz_center_r_minmax(bah_params_and_data, &x_center, &y_center,
                                &z_center, &r_min, &r_max);

        // need to make sure that we force a full guess if the parameters
        // request (or enforce) it
        if (bah_params_and_data->use_fixed_radius_guess_on_full_sphere) {
            r_min = 0.0;
            r_max = bah_max_search_radius_[which_horizon];
        }

#if 0
        if (rankActive == 0) {
            std::cout << which_horizon
                      << ": XCENTER, YCENTER, ZCENTER, RMIN, RMAX " << x_center
                      << " " << y_center << " " << z_center << ", " << r_min
                      << " " << r_max << std::endl;
        }
#endif

        // guesses need to be stored
        x_guess_[which_horizon]     = x_center;
        y_guess_[which_horizon]     = y_center;
        z_guess_[which_horizon]     = z_center;
        r_min_guess_[which_horizon] = r_min;
        r_max_guess_[which_horizon] = r_max;
    }

    // 3: BH Mode Synchronization (if enabled) - sync horizon guess across ranks
    // - activate or deactivate indivisdual or common horizon searches based on
    // BBH criteria
    if (is_bbh_) {
        // time output for starting BBH mode

        // 3.a: only one rank keeps the "real" data, all others zero out
        for (int which_horizon = 0; which_horizon < num_horizons_;
             which_horizon++) {
            const int which_MPI_rank = which_horizon % npesActive;
            if (rankActive != which_MPI_rank) {
                bah_use_fixed_radius_guess_on_full_sphere_[which_horizon] = 0;
                bah_horizon_active_[which_horizon]                        = 0;

                x_center_m1_[which_horizon]                               = 0.0;
                y_center_m1_[which_horizon]                               = 0.0;
                z_center_m1_[which_horizon]                               = 0.0;

                x_guess_[which_horizon]                                   = 0.0;
                y_guess_[which_horizon]                                   = 0.0;
                z_guess_[which_horizon]                                   = 0.0;

                // NOTE: r_min_guess doesn't need to be set to zero to trigger
                // common horizon search

                r_max_m1_[which_horizon]                                  = 0.0;
                t_m1_[which_horizon]                                      = 0.0;
                r_max_guess_[which_horizon]                               = 0.0;
                r_min_guess_[which_horizon]                               = 0.0;

                // also zero out the failed last find here
                failed_last_find_[which_horizon]     = false;
                failed_last_find_int_[which_horizon] = 0;

                // also be sure to clear out the prev_horizon buffers if we
                // don't own it
                const size_t horizon_data_size =
                    static_cast<size_t>(max_ntheta_ * max_nphi_);
                const size_t horizon_offset =
                    static_cast<size_t>(which_horizon) * horizon_data_size;

                std::fill(prev_horizon_m1_.begin() + horizon_offset,
                          prev_horizon_m1_.begin() + horizon_offset +
                              horizon_data_size,
                          0.0);
                std::fill(prev_horizon_m2_.begin() + horizon_offset,
                          prev_horizon_m2_.begin() + horizon_offset +
                              horizon_data_size,
                          0.0);
                std::fill(prev_horizon_m3_.begin() + horizon_offset,
                          prev_horizon_m3_.begin() + horizon_offset +
                              horizon_data_size,
                          0.0);

            }
        }

        // 3.b: MPI reduction to sum arrays across ranks which makes them
        // consistent
        bah_sum_shared_arrays(mesh);
#if 0
        if (rankActive == 0) {
            for (int which_horizon = 0; which_horizon < num_horizons;
                 which_horizon++) {
                std::cout
                    << "SYNCED DATA FOR HORIZONS: "
                    << BAH_USE_FIXED_RADIUS_GUESS_ON_FULL_SPHERE[which_horizon]
                    << " " << bah_horizon_active[which_horizon] << " | "
                    << x_center_m1[which_horizon] << " "
                    << y_center_m1[which_horizon] << " "
                    << z_center_m1[which_horizon] << " | - guess "
                    << x_guess[which_horizon] << " " << y_guess[which_horizon]
                    << " " << z_guess[which_horizon] << " "
                    << r_max_guess[which_horizon] << std::endl;
            }
        }
#endif

        // if use_fixed_radius_guess_on_full_sphere=0 then we've found a common
        // horizon
        const bool found_common =
            (bah_use_fixed_radius_guess_on_full_sphere_[common_horizon] == 0);

        // are the black holes active?
        const bool bh1_active    = (bah_horizon_active_[inspiral_bh1] != 0);
        const bool bh2_active    = (bah_horizon_active_[inspiral_bh2] != 0);
        const bool common_active = (bah_horizon_active_[common_horizon] != 0);

        // step 3.c: deactivate individual BH horizons if common is found
        if (common_active && found_common && bh1_active && bh2_active) {
            bah_horizon_active_[inspiral_bh1] = 0;
            bah_horizon_active_[inspiral_bh2] = 0;
            // disabled the search!
        }

        // 3.d: we activate common horizon based on proximity of last known good
        // states
        if (bh1_active && bh2_active && !common_active) {
            const double x_bh1_last          = x_center_m1_[inspiral_bh1];
            const double y_bh1_last          = y_center_m1_[inspiral_bh1];
            const double z_bh1_last          = z_center_m1_[inspiral_bh1];
            const double rmax_bh1_last       = r_max_m1_[inspiral_bh1];
            const double time_bh1_last_found = t_m1_[inspiral_bh1];

            const double x_bh2_last          = x_center_m1_[inspiral_bh2];
            const double y_bh2_last          = y_center_m1_[inspiral_bh2];
            const double z_bh2_last          = z_center_m1_[inspiral_bh2];
            const double rmax_bh2_last       = r_max_m1_[inspiral_bh2];
            const double time_bh2_last_found = t_m1_[inspiral_bh2];

            // only trigger if they've been found *before
            const bool bh1_found_before      = (time_bh1_last_found >= 0.0);
            const bool bh2_found_before      = (time_bh2_last_found >= 0.0);

            // check last found times
            if (!globalRank && bah_verbosity_level_ > 1) {
                // BH1
                std::cout << "[BAH]: last found BH1 time: " << time_bh1_last_found << std::endl;
                std::cout << "[BAH]: last found BH1 position: (" 
                  << x_bh1_last << ',' << y_bh1_last << ',' << z_bh1_last 
                  << ')' << std::endl;
                // BH2
                std::cout << "[BAH]: last found BH2 time: " << time_bh2_last_found << std::endl;
                std::cout << "[BAH]: last found BH2 position: (" 
                  << x_bh2_last << ',' << y_bh2_last << ',' << z_bh2_last 
                  << ')' << std::endl;
            }

            // if they've both been found, then we can check their centers
            if (bh1_found_before && bh2_found_before) {
                const double dist_between_centers =
                    dist(x_bh1_last, x_bh2_last, y_bh1_last, y_bh2_last,
                         z_bh1_last, z_bh2_last);

                const double threshold_diameter =
                    2.0 * bah_max_search_radius_[common_horizon];

                if (bah_verbosity_level_ > 0) {
                    if (rankActive == 0) {
                        std::cout
                            << GRN << "BBH Trigger " << "sep=" << std::fixed
                            << std::setprecision(6) << dist_between_centers
                            << ", "
                            << "rmax1=" << rmax_bh1_last << ", "
                            << "rmax2=" << rmax_bh2_last << ", "
                            << "sum="
                            << (dist_between_centers + rmax_bh1_last +
                                rmax_bh2_last)
                            << " <= "
                            << "thresh=" << threshold_diameter << "?" << NRM
                            << std::endl;
                    }
                }

                if (dist_between_centers + rmax_bh1_last + rmax_bh2_last <=
                    threshold_diameter) {
                    // activate common horizon! hooray
                    bah_horizon_active_[common_horizon] = 1;

                    // compute center of mass from last good centers
                    const double m1 = bah_m_scale_[inspiral_bh1];
                    const double m2 = bah_m_scale_[inspiral_bh2];

                    double x_center_common =
                        (m1 * x_bh1_last + m2 * x_bh2_last) / (m1 + m2);
                    double y_center_common =
                        (m1 * y_bh1_last + m2 * y_bh2_last) / (m1 + m2);
                    double z_center_common =
                        (m1 * z_bh1_last + m2 * z_bh2_last) / (m1 + m2);

                    // update the common-horizon center
                    x_center_m1_[common_horizon] = x_center_common;
                    y_center_m1_[common_horizon] = y_center_common;
                    z_center_m1_[common_horizon] = z_center_common;

                    // force the next extrapolation to use full-sphere radius!
                    t_m1_[common_horizon]        = t_m2_[common_horizon] =
                        t_m3_[common_horizon]    = -1.0;

                    // and working guesses for iteration
                    x_guess_[common_horizon]     = x_center_common;
                    y_guess_[common_horizon]     = y_center_common;
                    z_guess_[common_horizon]     = z_center_common;

                    // and a naive initial guess
                    r_min_guess_[common_horizon] = 0.0;
                    r_max_guess_[common_horizon] =
                        bah_max_search_radius_[common_horizon];

                    if (bah_verbosity_level_ > 0) {
                        if (rankActive == 0) {
                            std::cout
                                << GRN
                                << "BBH COMMON HORIZON ACTIVATED: center=("
                                << std::fixed << std::setprecision(6)
                                << x_center_common << "," << y_center_common
                                << "," << z_center_common << ") r_max_guess="
                                << r_max_guess_[common_horizon] << NRM
                                << std::endl;
                        }
                    }
                }
            } else {
                // sanity check: make sure horizons found when expected
                if (rankActive == 0 && bah_verbosity_level_ > 1) {
                    std::cout << "[BAH]: Horizon never found before!" << std::endl;
                }
            }
        }
    }
    // ------- END STEP 3 --------

    // 4: Metric interpolation
    // -- Allocate memory and interpolate metric data onto the BHaHAHA grid
    // using guesses
    std::vector<double*> emda_horizon_data(num_horizons_, nullptr);
    for (int which_horizon = 0; which_horizon < num_horizons_;
         which_horizon++) {
        // if we're in inactive and in Binary Black HOle mode, skip
        if (is_bbh_ && !bah_horizon_active_[which_horizon]) {
            continue;
        }

        bhahaha_params_and_data_struct* bah_params_and_data =
            &bha_param_data_[which_horizon];

        // set up the radial grid
        std::vector<double> radii(bah_nr_interp_max_[which_horizon]);

        bah_radial_grid_cell_centered_set_up(
            bah_nr_interp_max_[which_horizon],
            bah_max_search_radius_[which_horizon], r_min_guess_[which_horizon],
            r_max_guess_[which_horizon],
            &bah_params_and_data->Nr_external_input,
            &bah_params_and_data->r_min_external_input,
            &bah_params_and_data->dr_external_input, radii.data());

        // round robin assignment for the horizon finding
        const int which_MPI_rank = which_horizon % npesActive;

        // allocate the param_data[which_horizon] input_metric_data array on
        // that rank
        if (rankActive == which_MPI_rank) {
            bha_param_data_[which_horizon].input_metric_data =
                (double*)malloc(NUM_EXT_INPUT_CARTESIAN_GFS *
                                bah_params_and_data->Nr_external_input *
                                max_ntheta_ * max_nphi_ * sizeof(double));

            if (!bha_param_data_[which_horizon].input_metric_data) {
                std::cerr << "ERROR ALLOCATING MEMORY FOR INPUT METRIC DATA!"
                          << std::endl;
                exit(EXIT_FAILURE);
            }

            emda_horizon_data[which_horizon] =
                (double*)malloc(NUM_EMDA_HORIZON_FIELDS *
                                bah_params_and_data->Nr_external_input *
                                max_ntheta_ * max_nphi_ * sizeof(double));

            if (!emda_horizon_data[which_horizon]) {
                std::cerr << "ERROR ALLOCATING MEMORY FOR EMDA HORIZON DATA!"
                          << std::endl;
                exit(EXIT_FAILURE);
            }
        }

        // then interpolate the metric data!
        interpolate_metric_data(
            mesh, var, which_horizon, which_MPI_rank,
            bah_params_and_data->Nr_external_input, max_ntheta_, max_nphi_,
            radii.data(), x_guess_[which_horizon], y_guess_[which_horizon],
            z_guess_[which_horizon],
            bha_param_data_[which_horizon].input_metric_data,
            emda_horizon_data[which_horizon]);
    }

    // 5: Horizon Finding and Diagnostics
    // -- Execute algorithm for each active horizon
    // -- Output diagnostic information and update persistent data
    for (int which_horizon = 0; which_horizon < num_horizons_;
         which_horizon++) {
        const int inspiral_bh1   = 0;
        const int inspiral_bh2   = 1;
        const int common_horizon = 2;

        // only proceed if horizon is active
        if (is_bbh_ && !bah_horizon_active_[which_horizon]) continue;

        // if a common horizon hasn't been found, set lower multigrid resolution
        // higher to increase the chance a common horizon will be found.
        if (is_bbh_ && which_horizon == common_horizon &&
            r_max_guess_[common_horizon] ==
                bah_max_search_radius_[common_horizon]) {
            if (num_resolutions_multigrid_ > 1) {
                bha_param_data_[common_horizon].Ntheta_array_multigrid[0] =
                    ntheta_array_multigrid_[1];
                bha_param_data_[common_horizon].Nphi_array_multigrid[0] =
                    nphi_array_multigrid_[1];
            }
        }

        // round-robin assign ranks for horizon finding
        const int which_rank = which_horizon % npesActive;
        if (rankActive == which_rank) {
            // check for poisoning before solving
            bah_poisoning_check_inputs(&bha_param_data_[which_horizon]);

            bhahaha_diagnostics_struct bhahaha_diags;

#if 0
            if (which_horizon == common_horizon) {
                print_bha_param_data(&bha_param_data_[which_horizon],
                                     which_horizon);
            }
#endif

            const int bah_return_code = bah_find_horizon(
                &bha_param_data_[which_horizon], &bhahaha_diags);

// if found successfully, we update the "persistent" storage
if (bah_return_code == BHAHAHA_SUCCESS) {

    // Copy newly found horizon data into persistent storage
    transfer_to_persistent_from_bhahaha(
        &bha_param_data_[which_horizon]);

    // ------------------------------------------------------------------
    // Compute horizon physical quantities (area, irreducible mass, etc.)
    // Uses the stored horizon surface prev_horizon_m1_
    // ------------------------------------------------------------------
HorizonMassSpinCharge hq = compute_mass_spin_charge(
    which_horizon,
    bha_param_data_[which_horizon],
    bhahaha_diags,
    prev_horizon_m1_,
    emda_horizon_data[which_horizon],
    max_ntheta_,
    max_nphi_
);

    // Print to stdout for quick debugging/monitoring
    std::cout << "[AH QUANTS] horizon=" << which_horizon
              << " area=" << hq.area
              << " Mirr=" << hq.Mirr
              << " M=" << hq.M
              << " J=" << hq.J
              << " Q=" << hq.Q
              << " chi=" << hq.chi
              << " D=" << hq.D
              << std::endl;

    // ------------------------------------------------------------------
    // File output (same cadence as other diagnostics)
    // This ensures consistency with existing BHaHAHA outputs
    // ------------------------------------------------------------------
    if (current_step % file_output_freq_ == 0) {

        // Append horizon quantities to file
        std::ofstream fout(
            out_dir_ + "/emda_prof_HorizonMassSpinCharge.dat",
            std::ios::app
        );

        if (current_step == 0 && rankActive == 0) {
            fout << "# step time horizon area Mirr M J Q chi D\n";
        }

        fout << current_step << " "
             << current_time << " "
             << which_horizon << " "
             << hq.area << " "
             << hq.Mirr << " "
             << hq.M << " "
             << hq.J << " "
             << hq.Q << " "
             << hq.chi << " "
             << hq.D << "\n";

        fout.close();

        // Existing BHaHAHA diagnostics output
        bah_diagnostics_file_output(
            &bhahaha_diags, &bha_param_data_[which_horizon],
            num_horizons_, x_guess_[which_horizon],
            y_guess_[which_horizon], z_guess_[which_horizon],
            out_dir_.c_str());
    }

    // ------------------------------------------------------------------
    // Reset failure flags and allow next iteration to use refined guesses
    // ------------------------------------------------------------------
    bah_use_fixed_radius_guess_on_full_sphere_[which_horizon] = 0;
    failed_last_find_[which_horizon]     = false;
    failed_last_find_int_[which_horizon] = 0;
} else {
                std::cerr << "ERROR[HORIZON]: Failure to find horizon "
                          << which_horizon << " with error code "
                          << bah_return_code << ": "
                          << bah_error_message(
                                 (bhahaha_error_codes)bah_return_code)
                          << std::endl;

                // update failure flags
                failed_last_find_[which_horizon]     = true;
                failed_last_find_int_[which_horizon] = 1;

                // revert to full-sphere guess
                bah_use_fixed_radius_guess_on_full_sphere_[which_horizon] = 1;
                r_min_guess_[which_horizon] = 0.0;
                r_max_guess_[which_horizon] = bah_max_search_radius_[which_horizon];

                // update the guess with the BH data for this horizon
                // taken from the bh history if possible
                if (!tracked_location_data.empty() &&
                    which_horizon != common_horizon) {
                    if (bah_verbosity_level_ > 0) {
                        std::cout << "\tAH NOTICE rank " << std::setw(4)
                                  << rankActive << " horizon " << std::setw(4)
                                  << (which_horizon + 1)
                                  << ": Last find failed, overwriting guess points "
                                     "to tracked puncture location: "
                                  << tracked_location_data[which_horizon]
                                  << std::endl;
                    }
                    // update center location
                    x_guess_[which_horizon] = tracked_location_data[which_horizon].x();
                    y_guess_[which_horizon] = tracked_location_data[which_horizon].y();
                    z_guess_[which_horizon] = tracked_location_data[which_horizon].z();
                }
            }
        }

        // free metric data
        free(bha_param_data_[which_horizon].input_metric_data);
        free(emda_horizon_data[which_horizon]);
    }

    // now that we're done we can delete the whole param_data
    bha_param_data_.clear();

    bhahaha_profiler_.stop();
    // TODO: print info

    MPI_Barrier(commActive);
}

void AEH_BHaHAHA::bah_sum_shared_arrays(const ot::Mesh* mesh) {
    unsigned int rankActive = mesh->getMPIRank();
    unsigned int npesActive = mesh->getMPICommSize();
    MPI_Comm commActive     = mesh->getMPICommunicator();
    unsigned int globalRank = mesh->getMPIRankGlobal();

    // integer arrays that need to be synchronized
    std::vector<int> combined_int_buffer;
    combined_int_buffer.insert(
        combined_int_buffer.end(),
        bah_use_fixed_radius_guess_on_full_sphere_.begin(),
        bah_use_fixed_radius_guess_on_full_sphere_.end());

    combined_int_buffer.insert(combined_int_buffer.end(),
                               bah_horizon_active_.begin(),
                               bah_horizon_active_.end());
    combined_int_buffer.insert(combined_int_buffer.end(),
                               failed_last_find_int_.begin(),
                               failed_last_find_int_.end());

    MPI_Allreduce(MPI_IN_PLACE, combined_int_buffer.data(),
                  combined_int_buffer.size(), MPI_INT, MPI_SUM, commActive);

    // then unpack the integer arrays
    auto int_it = combined_int_buffer.begin();
    std::copy(int_it,
              int_it + bah_use_fixed_radius_guess_on_full_sphere_.size(),
              bah_use_fixed_radius_guess_on_full_sphere_.begin());
    int_it += bah_use_fixed_radius_guess_on_full_sphere_.size();
    std::copy(int_it, int_it + bah_horizon_active_.size(),
              bah_horizon_active_.begin());
    int_it += bah_horizon_active_.size();
    std::copy(int_it, int_it + failed_last_find_int_.size(),
              failed_last_find_int_.begin());

    // make sure the failed_last_find is converted back to booleans
    for (size_t i = 0; i < num_horizons_; ++i) {
        failed_last_find_[i] = (failed_last_find_int_[i] == 1);
    }

    // double arrays need to also be synchronized
    std::vector<double> combined_double_buffer;
    combined_double_buffer.insert(combined_double_buffer.end(),
                                  x_center_m1_.begin(), x_center_m1_.end());
    combined_double_buffer.insert(combined_double_buffer.end(),
                                  y_center_m1_.begin(), y_center_m1_.end());
    combined_double_buffer.insert(combined_double_buffer.end(),
                                  z_center_m1_.begin(), z_center_m1_.end());
    combined_double_buffer.insert(combined_double_buffer.end(),
                                  x_guess_.begin(), x_guess_.end());
    combined_double_buffer.insert(combined_double_buffer.end(),
                                  y_guess_.begin(), y_guess_.end());
    combined_double_buffer.insert(combined_double_buffer.end(),
                                  z_guess_.begin(), z_guess_.end());
    combined_double_buffer.insert(combined_double_buffer.end(),
                                  r_max_m1_.begin(), r_max_m1_.end());
    combined_double_buffer.insert(combined_double_buffer.end(), t_m1_.begin(),
                                  t_m1_.end());
    combined_double_buffer.insert(combined_double_buffer.end(),
                                  r_max_guess_.begin(), r_max_guess_.end());
    combined_double_buffer.insert(combined_double_buffer.end(),
                                  r_min_guess_.begin(), r_min_guess_.end());

    // make sure the prev_horizon arrays are added to the buffer
    combined_double_buffer.insert(combined_double_buffer.end(),
                                  prev_horizon_m1_.begin(),
                                  prev_horizon_m1_.end());
    combined_double_buffer.insert(combined_double_buffer.end(),
                                  prev_horizon_m2_.begin(),
                                  prev_horizon_m2_.end());
    combined_double_buffer.insert(combined_double_buffer.end(),
                                  prev_horizon_m3_.begin(),
                                  prev_horizon_m3_.end());

    MPI_Allreduce(MPI_IN_PLACE, combined_double_buffer.data(),
                  combined_double_buffer.size(), MPI_DOUBLE, MPI_SUM,
                  commActive);

    // then unpack
    auto it = combined_double_buffer.begin();
    std::copy(it, it + x_center_m1_.size(), x_center_m1_.begin());
    it += x_center_m1_.size();
    std::copy(it, it + y_center_m1_.size(), y_center_m1_.begin());
    it += y_center_m1_.size();
    std::copy(it, it + z_center_m1_.size(), z_center_m1_.begin());
    it += z_center_m1_.size();
    std::copy(it, it + x_guess_.size(), x_guess_.begin());
    it += x_guess_.size();
    std::copy(it, it + y_guess_.size(), y_guess_.begin());
    it += y_guess_.size();
    std::copy(it, it + z_guess_.size(), z_guess_.begin());
    it += z_guess_.size();
    std::copy(it, it + r_max_m1_.size(), r_max_m1_.begin());
    it += r_max_m1_.size();
    std::copy(it, it + t_m1_.size(), t_m1_.begin());
    it += t_m1_.size();
    std::copy(it, it + r_max_guess_.size(), r_max_guess_.begin());
    it += r_max_guess_.size();
    std::copy(it, it + r_min_guess_.size(), r_min_guess_.begin());
    // and unpack the prev_horizon stuff
    it += r_min_guess_.size();
    std::copy(it, it + prev_horizon_m1_.size(), prev_horizon_m1_.begin());
    it += prev_horizon_m1_.size();
    std::copy(it, it + prev_horizon_m2_.size(), prev_horizon_m2_.begin());
    it += prev_horizon_m2_.size();
    std::copy(it, it + prev_horizon_m3_.size(), prev_horizon_m3_.begin());
}

void AEH_BHaHAHA::fill_domain_coords(const int which_horizon, const int n_r,
                                     const int n_theta, const int n_phi,
                                     const double* radii, const double x_center,
                                     const double y_center,
                                     const double z_center,
                                     std::vector<double>& domain_coords) {
    const double dtheta = M_PI / ((double)n_theta);
    const double dphi   = 2.0 * M_PI / ((double)n_phi);

    const int n_pts     = n_r * n_theta * n_phi;
    int pt_idx          = 0;

    for (int iphi = 0; iphi < n_phi; iphi++) {
        const double phi = -M_PI + ((double)iphi + 0.5) *
                                       dphi;  // aximuthal angle at cell center
        const double sinphi = sin(phi);
        const double cosphi = cos(phi);

        for (int itheta = 0; itheta < n_theta; itheta++) {
            const double theta =
                ((double)itheta + 0.5) * dtheta;  // polar angle at cell center
            const double sintheta = sin(theta);
            const double costheta = cos(theta);

            for (int ir = 0; ir < n_r; ir++) {
                const double r = radii[ir];
                const int idx3 = idx3_spherical(ir, itheta, iphi, n_theta, n_r);

                // fill x, y, z
                domain_coords[3 * pt_idx + 0] =
                    x_center + r * sintheta * cosphi;
                domain_coords[3 * pt_idx + 1] =
                    y_center + r * sintheta * sinphi;
                domain_coords[3 * pt_idx + 2] = z_center + r * costheta;

                pt_idx++;
            }
        }
    }
}

void AEH_BHaHAHA::interpolate_metric_data(
    const ot::Mesh* mesh, const double** varData, const int which_horizon,
    const int which_rank, const int n_r, const int n_theta, const int n_phi,
    const double* radii, const double x_center, const double y_center,
    const double z_center, double* input_metric_data,
    double* emda_horizon_data) {
    // get the active comm and information about it
    unsigned int rankActive  = mesh->getMPIRank();
    unsigned int npesActive  = mesh->getMPICommSize();
    MPI_Comm commActive      = mesh->getMPICommunicator();
    unsigned int globalRank  = mesh->getMPIRankGlobal();

    // calculate the total elements we need
    const int total_elements = n_r * n_theta * n_phi;

    // build up the extracted buffer, which needs to be large
    std::vector<double> extracted_buffer(total_elements *
                                         variable_indices_.size());

    // vector for coordinates that are going to be needed
    std::vector<double> domain_coords;
    domain_coords.resize(3 * total_elements);

    // fills the domain_coords array with the spherical coordinates (in
    // cartesian) that are going to be used
    fill_domain_coords(which_horizon, n_r, n_theta, n_phi, radii, x_center,
                       y_center, z_center, domain_coords);

    // vector that the interpolation routine indicates are "valid" i.e. on this
    // partition
    std::vector<unsigned int> validIndex;
    validIndex.clear();

    // for each of the variables that are indicated by the user, we interpolate
    // them
    for (int idx_extract = 0; idx_extract < variable_indices_.size();
         idx_extract++) {
        validIndex.clear();
        int to_extract = variable_indices_[idx_extract];

#if 0
        if (rankActive == 0)
            std::cout << rankActive << ": EXTRACTING: " << to_extract
                      << std::endl;
#endif

        // perform the interpolation at the points, store to an offset of the
        // extracted buffer
        ot::da::interpolateToCoords(
            mesh, varData[to_extract], domain_coords.data(),
            domain_coords.size(), grid_limits_, domain_limits_,
            &extracted_buffer[idx_extract * total_elements], validIndex);
    }

#if 0
    std::cout << rankActive << ": finished extracing data, I had "
              << validIndex.size() << " there were " << total_elements
              << " points requested... " << std::endl;
#endif

    int local_num_valid = validIndex.size();
    std::vector<double> transformed_buffer(
        local_num_valid * NUM_HORIZON_INTERP_FIELDS, 0.0);

    // temporary buffers to fill for the transformation defined by the user
    std::vector<double> input_pts(variable_indices_.size());

    // transform these variables to gd and Kd forms from before...
    for (unsigned int index = 0; index < local_num_valid; index++) {
        unsigned int idx = validIndex[index];

        for (unsigned int i = 0; i < variable_indices_.size(); i++) {
            // extract out the points needed to fill our buffer for this
            // location
            input_pts[i] = extracted_buffer[total_elements * i + idx];
        }

        // then call the transformation function
        std::vector<double> transformed_points = transform_(input_pts);

        if (transformed_points.size() < NUM_EXT_INPUT_CARTESIAN_GFS) {
            throw std::runtime_error(
                "AEH transform_ returned fewer than 12 BHaHAHA fields");
        }

        // The first 12 entries are the BHaHAHA-compatible gamma_ij/K_ij
        // interface. Entries 12..18 are optional EMDA sidecar data.
        const size_t num_to_copy = std::min<size_t>(
            transformed_points.size(), NUM_HORIZON_INTERP_FIELDS);
        for (size_t i = 0; i < num_to_copy; i++) {
            transformed_buffer[local_num_valid * i + index] =
                transformed_points[i];
        }
    }
// now the local values are in the "extracted"
#if 0
    std::cout << "TRANSFORMED BUFFER SIZE: " << transformed_buffer.size()
              << std::endl;
#endif

    // adjust local_num_valid to multiply by 12 to account for full buffer
    std::vector<int> receive_idxs;

    // so we can do recv_counts and do an MPI_Gather to the rank we are using
    std::vector<int> recv_counts(npesActive), displs(npesActive);
    MPI_Gather(&local_num_valid, 1, MPI_INT, recv_counts.data(), 1, MPI_INT,
               which_rank, commActive);

    if (rankActive == which_rank) {
        displs[0] = 0;
        for (int i = 1; i < npesActive; ++i) {
            displs[i] = displs[i - 1] + recv_counts[i - 1];
        }

        receive_idxs.resize(total_elements);

#if 0
        std::cout << "WHITCH RANK: " << which_rank << std::endl;
        print_vec(recv_counts, "RECV_COUNTS");
        print_vec(displs, "DISPLS");
#endif
    }

    // two rounds of communication, need to send the indices for where the data
    // goes in the final array of points
    MPI_Gatherv(validIndex.data(), local_num_valid, MPI_INT,
                receive_idxs.data(), recv_counts.data(), displs.data(), MPI_INT,
                which_rank, commActive);

    std::vector<double> receive_buffer;
    if (rankActive == which_rank) {
        receive_buffer.resize(total_elements * NUM_HORIZON_INTERP_FIELDS);

        // modify receive counts based on the number of fields coming in
        for (int i = 0; i < npesActive; ++i) {
            displs[i]      = displs[i] * NUM_HORIZON_INTERP_FIELDS;
            recv_counts[i] = recv_counts[i] * NUM_HORIZON_INTERP_FIELDS;
        }

#if 0
        std::cout << "WHITCH RANK: " << which_rank << std::endl;
        print_vec(recv_counts, "UPDATED RECV_COUNTS");
        print_vec(displs, "UPDATED DISPLS");
#endif
    }
    int local_num_valid_total =
        local_num_valid * NUM_HORIZON_INTERP_FIELDS;

    MPI_Gatherv(transformed_buffer.data(), local_num_valid_total, MPI_DOUBLE,
                receive_buffer.data(), recv_counts.data(), displs.data(),
                MPI_DOUBLE, which_rank, commActive);

    // now we unpack the data, following displacements on target processor
    if (rankActive == which_rank) {
        // the received data is stored essentially as a flattened array of the
        // following shape: process_id -> variable_id -> valid_point. This means
        // that we need to index in by full displacement, then by vidx * current
        // receive count,a nd then by the index that we've received.
        for (int i = 0; i < npesActive; ++i) {
            int recv_count =
                recv_counts[i] / NUM_HORIZON_INTERP_FIELDS;
            int buffer_offset = displs[i];
            int point_offset =
                displs[i] / NUM_HORIZON_INTERP_FIELDS;
            for (int vidx = 0; vidx < NUM_HORIZON_INTERP_FIELDS; ++vidx) {
                for (int j = 0; j < recv_count; ++j) {
                    int idx = receive_idxs[point_offset + j];
                    const double value =
                        receive_buffer[buffer_offset + vidx * recv_count + j];

                    if (vidx < NUM_EXT_INPUT_CARTESIAN_GFS) {
                        input_metric_data[vidx * total_elements + idx] =
                            value;
                    } else if (emda_horizon_data) {
                        const int emda_vidx =
                            vidx - NUM_EXT_INPUT_CARTESIAN_GFS;
                        emda_horizon_data[emda_vidx * total_elements + idx] =
                            value;
                    }
                }
            }
        }
    }
    if (rankActive == which_rank && rankActive == 0 && emda_horizon_data) {
        const int debug_points = std::min(total_elements, 5);
        for (int idx = 0; idx < debug_points; ++idx) {
            std::cout << "[AEH DEBUG] emda_horizon_data point=" << idx
                      << " Ex=" << emda_horizon_data[0 * total_elements + idx]
                      << " Ey=" << emda_horizon_data[1 * total_elements + idx]
                      << " Ez=" << emda_horizon_data[2 * total_elements + idx]
                      << " Bx=" << emda_horizon_data[3 * total_elements + idx]
                      << " By=" << emda_horizon_data[4 * total_elements + idx]
                      << " Bz=" << emda_horizon_data[5 * total_elements + idx]
                      << " dilaton="
                      << emda_horizon_data[6 * total_elements + idx]
                      << std::endl;
        }
    }
    // data should now be unpacked!
}

void AEH_BHaHAHA::synchronize_to_root(const ot::Mesh* mesh,
                                      const int targetProc) {
    unsigned int rankActive    = mesh->getMPIRank();
    unsigned int npesActive    = mesh->getMPICommSize();
    MPI_Comm commActive        = mesh->getMPICommunicator();
    unsigned int globalRank    = mesh->getMPIRankGlobal();

    // need to synchronize the x/y/z center arrays, the t arrays, the r
    // min/max arrays, and the prev_horizon arrays (as well as the active
    // arrays), that's 18 arrays
    constexpr int num_standard = 20;

    const size_t data_to_send_per_horizon =
        1 + num_standard + 3 * max_nphi_ * max_ntheta_;

    size_t local_horizons = 0;
    for (unsigned int which_horizon = 0; which_horizon < num_horizons_;
         ++which_horizon) {
        if (which_horizon % npesActive == rankActive) {
            local_horizons++;
        }
    }

    // then we can preallocate the send buffer since we know how many there
    // are
    std::vector<double> sendBuffer(local_horizons * data_to_send_per_horizon);

    // offset starts at 0, because sendBuffer 0 is wanted for num horizons
    // for easy bookkeeping
    size_t curr_offset = 0;

    for (unsigned int which_horizon = 0; which_horizon < num_horizons_;
         ++which_horizon) {
        // round-robin is our strategy for which process calculates for the
        // horizon, it's consistent with the find_horizons calculations
        if (which_horizon % npesActive != rankActive) continue;

        // now we can collect the data into the sendBuffer
        // update the send buffer for how many horizons there are to send
        // and then set which horizon this is!
        sendBuffer[curr_offset++] = static_cast<double>(which_horizon);

        // then fill with the values that need to be synchronized
        sendBuffer[curr_offset++] = x_center_m1_[which_horizon];
        sendBuffer[curr_offset++] = y_center_m1_[which_horizon];
        sendBuffer[curr_offset++] = z_center_m1_[which_horizon];
        sendBuffer[curr_offset++] = x_center_m2_[which_horizon];
        sendBuffer[curr_offset++] = y_center_m2_[which_horizon];
        sendBuffer[curr_offset++] = z_center_m2_[which_horizon];
        sendBuffer[curr_offset++] = x_center_m3_[which_horizon];
        sendBuffer[curr_offset++] = y_center_m3_[which_horizon];
        sendBuffer[curr_offset++] = z_center_m3_[which_horizon];
        sendBuffer[curr_offset++] = t_m1_[which_horizon];
        sendBuffer[curr_offset++] = t_m2_[which_horizon];
        sendBuffer[curr_offset++] = t_m3_[which_horizon];
        sendBuffer[curr_offset++] = r_min_m1_[which_horizon];
        sendBuffer[curr_offset++] = r_min_m2_[which_horizon];
        sendBuffer[curr_offset++] = r_min_m3_[which_horizon];
        sendBuffer[curr_offset++] = r_max_m1_[which_horizon];
        sendBuffer[curr_offset++] = r_max_m2_[which_horizon];
        sendBuffer[curr_offset++] = r_max_m3_[which_horizon];
        sendBuffer[curr_offset++] = bah_horizon_active_[which_horizon];
        sendBuffer[curr_offset++] =
            failed_last_find_[which_horizon] ? 1.0 : 0.0;

        // make sure curr_offset is updated
        // curr_offset += num_standard;

        const size_t size_send          = max_ntheta_ * max_nphi_;
        const size_t extract_offset     = which_horizon * size_send;
        const size_t extract_offset_end = (which_horizon + 1) * size_send;

        std::copy(prev_horizon_m1_.begin() + extract_offset,
                  prev_horizon_m1_.begin() + extract_offset_end,
                  sendBuffer.begin() + curr_offset);
        curr_offset += size_send;
        std::copy(prev_horizon_m2_.begin() + extract_offset,
                  prev_horizon_m2_.begin() + extract_offset_end,
                  sendBuffer.begin() + curr_offset);
        curr_offset += size_send;
        std::copy(prev_horizon_m3_.begin() + extract_offset,
                  prev_horizon_m3_.begin() + extract_offset_end,
                  sendBuffer.begin() + curr_offset);
        curr_offset += size_send;
    }

    std::vector<int> recv_counts(npesActive), displs(npesActive);
    int send_count = sendBuffer.size();
    MPI_Gather(&send_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT,
               targetProc, commActive);

    std::vector<double> recv_buffer;

    if (rankActive == targetProc) {
        displs[0] = 0;
        for (int i = 1; i < npesActive; ++i) {
            displs[i] = displs[i - 1] + recv_counts[i - 1];
        }

        recv_buffer.resize(displs.back() + recv_counts.back());
    }

    // then gatherv
    MPI_Gatherv(sendBuffer.data(), send_count, MPI_DOUBLE, recv_buffer.data(),
                recv_counts.data(), displs.data(), MPI_DOUBLE, targetProc,
                commActive);

    // now we have the data, we need to iterate through the horizons
    if (rankActive == targetProc) {
        size_t read_offset = 0;
        while (read_offset < recv_buffer.size()) {
            const unsigned int which_horizon =
                static_cast<unsigned int>(recv_buffer[read_offset++]);

            x_center_m1_[which_horizon]        = recv_buffer[read_offset + 0];
            y_center_m1_[which_horizon]        = recv_buffer[read_offset + 1];
            z_center_m1_[which_horizon]        = recv_buffer[read_offset + 2];
            x_center_m2_[which_horizon]        = recv_buffer[read_offset + 3];
            y_center_m2_[which_horizon]        = recv_buffer[read_offset + 4];
            z_center_m2_[which_horizon]        = recv_buffer[read_offset + 5];
            x_center_m3_[which_horizon]        = recv_buffer[read_offset + 6];
            y_center_m3_[which_horizon]        = recv_buffer[read_offset + 7];
            z_center_m3_[which_horizon]        = recv_buffer[read_offset + 8];
            t_m1_[which_horizon]               = recv_buffer[read_offset + 9];
            t_m2_[which_horizon]               = recv_buffer[read_offset + 10];
            t_m3_[which_horizon]               = recv_buffer[read_offset + 11];
            r_min_m1_[which_horizon]           = recv_buffer[read_offset + 12];
            r_min_m2_[which_horizon]           = recv_buffer[read_offset + 13];
            r_min_m3_[which_horizon]           = recv_buffer[read_offset + 14];
            r_max_m1_[which_horizon]           = recv_buffer[read_offset + 15];
            r_max_m2_[which_horizon]           = recv_buffer[read_offset + 16];
            r_max_m3_[which_horizon]           = recv_buffer[read_offset + 17];
            bah_horizon_active_[which_horizon] = recv_buffer[read_offset + 18];
            failed_last_find_[which_horizon] =
                (recv_buffer[read_offset + 19] == 1.0);
            failed_last_find_int_[which_horizon] =
                failed_last_find_[which_horizon] ? 1 : 0;

            // make sure curr_offset is updated
            read_offset += num_standard;

            // then copy the rest of the data in
            const size_t size_send      = max_ntheta_ * max_nphi_;
            const size_t extract_offset = which_horizon * size_send;

            std::copy(recv_buffer.begin() + read_offset,
                      recv_buffer.begin() + read_offset + size_send,
                      prev_horizon_m1_.begin() + extract_offset);
            read_offset += size_send;

            std::copy(recv_buffer.begin() + read_offset,
                      recv_buffer.begin() + read_offset + size_send,
                      prev_horizon_m2_.begin() + extract_offset);
            read_offset += size_send;

            std::copy(recv_buffer.begin() + read_offset,
                      recv_buffer.begin() + read_offset + size_send,
                      prev_horizon_m3_.begin() + extract_offset);
            read_offset += size_send;
        }
        // and we should be done?
    }
}

void AEH_BHaHAHA::create_checkpoint(const ot::Mesh* mesh,
                                    const std::string& checkpoint_output) {
    // return if we're not active...
    if (!mesh->isActive()) return;

    unsigned int rankActive    = mesh->getMPIRank();
    unsigned int npesActive    = mesh->getMPICommSize();
    MPI_Comm commActive        = mesh->getMPICommunicator();
    unsigned int globalRank    = mesh->getMPIRankGlobal();

    // use rank 3 to try and free up other procs?
    const unsigned int procUse = npesActive < 3 ? 0 : 3;

    // for (unsigned int i = 0; i < num_horizons_; ++i) {
    //     transfer_to_persistent_from_bhahaha(&bha_param_data_[i]);
    // }

    // make sure we synchronize to root
    this->synchronize_to_root(mesh, procUse);

    if (rankActive == procUse) {
        nlohmann::json checkPoint;

        // make sure the n_horizons is achieved
        checkPoint["N_HORIZONS"]       = num_horizons_;
        checkPoint["IS_BBH"]           = is_bbh_;

        // using base91 for good compression and high-accuracy checkpointing
        checkPoint["X_CENTER_M1"]      = b91_encode(x_center_m1_);
        checkPoint["Y_CENTER_M1"]      = b91_encode(y_center_m1_);
        checkPoint["Z_CENTER_M1"]      = b91_encode(z_center_m1_);
        checkPoint["X_CENTER_M2"]      = b91_encode(x_center_m2_);
        checkPoint["Y_CENTER_M2"]      = b91_encode(y_center_m2_);
        checkPoint["Z_CENTER_M2"]      = b91_encode(z_center_m2_);
        checkPoint["X_CENTER_M3"]      = b91_encode(x_center_m3_);
        checkPoint["Y_CENTER_M3"]      = b91_encode(y_center_m3_);
        checkPoint["Z_CENTER_M3"]      = b91_encode(z_center_m3_);
        checkPoint["T_M1"]             = b91_encode(t_m1_);
        checkPoint["T_M2"]             = b91_encode(t_m2_);
        checkPoint["T_M3"]             = b91_encode(t_m3_);
        checkPoint["R_MIN_M1"]         = b91_encode(r_min_m1_);
        checkPoint["R_MIN_M2"]         = b91_encode(r_min_m2_);
        checkPoint["R_MIN_M3"]         = b91_encode(r_min_m3_);
        checkPoint["R_MAX_M1"]         = b91_encode(r_max_m1_);
        checkPoint["R_MAX_M2"]         = b91_encode(r_max_m2_);
        checkPoint["R_MAX_M3"]         = b91_encode(r_max_m3_);
        checkPoint["PREV_HORIZON_M1"]  = b91_encode(prev_horizon_m1_);
        checkPoint["PREV_HORIZON_M2"]  = b91_encode(prev_horizon_m2_);
        checkPoint["PREV_HORIZON_M3"]  = b91_encode(prev_horizon_m3_);
        checkPoint["HORIZON_ACTIVE"]   = bah_horizon_active_;
        checkPoint["FAILED_LAST_FIND"] = failed_last_find_;
        checkPoint["USE_FIXED_RADIUS_GUESS"] =
            bah_use_fixed_radius_guess_on_full_sphere_;

        std::ofstream outfile(checkpoint_output);
        if (!outfile) {
            std::cout << checkpoint_output
                      << " file open failed for BAH checkpoint" << std::endl;
            return;
        }

        outfile << std::setw(4) << checkPoint << std::endl;
        outfile.close();
    }
}

void AEH_BHaHAHA::restore_checkpoint(const ot::Mesh* mesh,
                                     const std::string& checkpoint_file) {
    if (!mesh->isActive()) return;

    unsigned int rankActive = mesh->getMPIRank();
    unsigned int npesActive = mesh->getMPICommSize();
    MPI_Comm commActive     = mesh->getMPICommunicator();
    unsigned int globalRank = mesh->getMPIRankGlobal();

    // all processes can just read the json file

    nlohmann::json checkPoint;

    std::ifstream infile(checkpoint_file);
    if (!infile) {
        std::cout << checkpoint_file
                  << " file open failed! Could not restore AH solver!"
                  << std::endl;
        return;
    }

    infile >> checkPoint;

    // read in the strings
    std::string temp             = checkPoint["X_CENTER_M1"].get<std::string>();
    std::vector<double> temp_vec = b91_decode<double>(temp);

    // then we fill in the data
    restore_vector(x_center_m1_, temp_vec, num_horizons_);

    // repeat for the rest of them...
    temp     = checkPoint["Y_CENTER_M1"];
    temp_vec = b91_decode<double>(temp);
    restore_vector(y_center_m1_, temp_vec, num_horizons_);
    temp     = checkPoint["Z_CENTER_M1"];
    temp_vec = b91_decode<double>(temp);
    restore_vector(z_center_m1_, temp_vec, num_horizons_);
    temp     = checkPoint["X_CENTER_M2"];
    temp_vec = b91_decode<double>(temp);
    restore_vector(x_center_m2_, temp_vec, num_horizons_);
    temp     = checkPoint["Y_CENTER_M2"];
    temp_vec = b91_decode<double>(temp);
    restore_vector(y_center_m2_, temp_vec, num_horizons_);
    temp     = checkPoint["Z_CENTER_M2"];
    temp_vec = b91_decode<double>(temp);
    restore_vector(z_center_m2_, temp_vec, num_horizons_);
    temp     = checkPoint["X_CENTER_M3"];
    temp_vec = b91_decode<double>(temp);
    restore_vector(x_center_m3_, temp_vec, num_horizons_);
    temp     = checkPoint["Y_CENTER_M3"];
    temp_vec = b91_decode<double>(temp);
    restore_vector(y_center_m3_, temp_vec, num_horizons_);
    temp     = checkPoint["Z_CENTER_M3"];
    temp_vec = b91_decode<double>(temp);
    restore_vector(z_center_m3_, temp_vec, num_horizons_);
    // times
    temp     = checkPoint["T_M1"];
    temp_vec = b91_decode<double>(temp);
    restore_vector(t_m1_, temp_vec, num_horizons_);
    temp     = checkPoint["T_M2"];
    temp_vec = b91_decode<double>(temp);
    restore_vector(t_m2_, temp_vec, num_horizons_);
    temp     = checkPoint["T_M3"];
    temp_vec = b91_decode<double>(temp);
    restore_vector(t_m3_, temp_vec, num_horizons_);

    // radii
    temp     = checkPoint["R_MIN_M1"];
    temp_vec = b91_decode<double>(temp);
    restore_vector(r_min_m1_, temp_vec, num_horizons_);
    temp     = checkPoint["R_MIN_M2"];
    temp_vec = b91_decode<double>(temp);
    restore_vector(r_min_m2_, temp_vec, num_horizons_);
    temp     = checkPoint["R_MIN_M3"];
    temp_vec = b91_decode<double>(temp);
    restore_vector(r_min_m3_, temp_vec, num_horizons_);
    temp     = checkPoint["R_MAX_M1"];
    temp_vec = b91_decode<double>(temp);
    restore_vector(r_max_m1_, temp_vec, num_horizons_);
    temp     = checkPoint["R_MAX_M2"];
    temp_vec = b91_decode<double>(temp);
    restore_vector(r_max_m2_, temp_vec, num_horizons_);
    temp     = checkPoint["R_MAX_M3"];
    temp_vec = b91_decode<double>(temp);
    restore_vector(r_max_m3_, temp_vec, num_horizons_);

    // then the big ones are the prev_horizon values, these are of size
    // num_horizons_ * n_theta * n_phi
    const size_t prev_horizon_size = num_horizons_ * max_ntheta_ * max_nphi_;
    temp                           = checkPoint["PREV_HORIZON_M1"];
    temp_vec                       = b91_decode<double>(temp);
    restore_vector(prev_horizon_m1_, temp_vec, prev_horizon_size);
    temp     = checkPoint["PREV_HORIZON_M2"];
    temp_vec = b91_decode<double>(temp);
    restore_vector(prev_horizon_m2_, temp_vec, prev_horizon_size);
    temp     = checkPoint["PREV_HORIZON_M3"];
    temp_vec = b91_decode<double>(temp);
    restore_vector(prev_horizon_m3_, temp_vec, prev_horizon_size);

    // then the currently active ones
    std::vector<int> temp_active = checkPoint["HORIZON_ACTIVE"];
    restore_vector(bah_horizon_active_, temp_active, num_horizons_);

    // check to make sure FAILED_LAST_FIND and USE_FIXED_RADIUS_GUESS are set
    // up:
    if (checkPoint.contains("FAILED_LAST_FIND")) {
        std::vector<bool> temp_failed = checkPoint["FAILED_LAST_FIND"];
        restore_vector(failed_last_find_, temp_failed, num_horizons_);

        // and then sync the new integer version
        for (size_t i = 0; i < num_horizons_; ++i) {
            failed_last_find_int_[i] = failed_last_find_[i] ? 1 : 0;
        }
    }

    if (checkPoint.contains("USE_FIXED_RADIUS_GUESS")) {
        std::vector<int> temp_fixed_guess =
            checkPoint["USE_FIXED_RADIUS_GUESS"];
        restore_vector(bah_use_fixed_radius_guess_on_full_sphere_,
                       temp_fixed_guess, num_horizons_);
    }

    // unpack back into the bah struct to ensure we're all on the same page
    // here
    // for (unsigned int which_horizon = 0; which_horizon < num_horizons_;
    //      ++which_horizon) {
    //     transfer_to_bhahaha_from_persistent(&bha_param_data_[which_horizon]);
    // }
}

}  // namespace dendro_aeh
