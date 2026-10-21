//
// Created by milinda on 4/27/18.
//
/**
 * @author Milinda Fernando
 * School of Computing, University of Utah.
 * @brief contains utilitity functions of mesh based distributed array(da)
 * */
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "TreeNode.h"
#include "mesh.h"
#include "sfcSearch.h"
#include "sfcSort.h"
namespace ot {

namespace da {

template <typename T>
T lagrangeInterpElementToCoord(const ot::Mesh* mesh, const T* in,
                               double* domain_coord, const Point& pt_min,
                               const Point& pt_max, unsigned int elementID,
                               unsigned int interpOrder) {
    if (!mesh->isActive()) return (T)0.0;

    const unsigned int localBegin   = mesh->getElementLocalBegin();
    const unsigned int localEnd     = mesh->getElementLocalEnd();

    const unsigned int eleOrder     = mesh->getElementOrder();
    const unsigned int nPe          = mesh->getNumNodesPerElement();

    const unsigned int rankActive   = mesh->getMPIRank();
    const unsigned int npesActive   = mesh->getMPICommSize();
    const ot::TreeNode* allElements = &(*(mesh->getAllElements().begin()));

    if (interpOrder > eleOrder) {
        std::cout << " Error in " << __func__ << " interpOrder " << interpOrder
                  << " > eleOrder " << eleOrder << std::endl;
        return (T)0.0;
    }
    if (eleOrder % interpOrder != 0) {
        std::cout << " Error in " << __func__ << " eleOrder " << eleOrder
                  << " not perfectly divisible by interpOrder " << interpOrder
                  << std::endl;
        return (T)0.0;
    }

    Point d_pt(domain_coord[0], domain_coord[1], domain_coord[2]);

    //!!! Note: This can only happens due to floating point rounding off issues,
    //!-------

    if (fabs(domain_coord[0] - pt_min.x()) < 1e-6) domain_coord[0] = pt_min.x();

    if (fabs(domain_coord[1] - pt_min.y()) < 1e-6) domain_coord[1] = pt_min.y();

    if (fabs(domain_coord[2] - pt_min.z()) < 1e-6) domain_coord[2] = pt_min.z();

    if (fabs(domain_coord[0] - pt_max.x()) < 1e-6) domain_coord[0] = pt_max.x();

    if (fabs(domain_coord[1] - pt_max.y()) < 1e-6) domain_coord[1] = pt_max.y();

    if (fabs(domain_coord[2] - pt_max.z()) < 1e-6) domain_coord[2] = pt_max.z();

    // !!!
    // -------------------------------------------------------------------------------

    if (domain_coord[0] < pt_min.x() || domain_coord[0] > pt_max.x()) {
        std::cout << "Error in " << __func__
                  << " x coord is not lies in the element for Lagrange "
                     "interpolation with the specified element "
                  << allElements[elementID] << std::endl;
        std::cout << "d_point : " << d_pt << " pt_min : " << pt_min
                  << " pt_max: " << pt_max << std::endl;
        exit(0);
    }

    if (domain_coord[1] < pt_min.y() || domain_coord[1] > pt_max.y()) {
        std::cout << "Error in " << __func__
                  << " y coord is not lies in the element for Lagrange "
                     "interpolation with the specified element "
                  << allElements[elementID] << std::endl;
        std::cout << "d_point : " << d_pt << " pt_min : " << pt_min
                  << " pt_max: " << pt_max << std::endl;
        exit(0);
    }

    if (domain_coord[2] < pt_min.z() || domain_coord[2] > pt_max.z()) {
        std::cout << "Error in " << __func__
                  << " z coord is not lies in the element for Lagrange "
                     "interpolation with the specified element "
                  << allElements[elementID] << std::endl;
        std::cout << "d_point : " << d_pt << " pt_min : " << pt_min
                  << " pt_max: " << pt_max << std::endl;
        exit(0);
    }

    std::vector<T> nodalValues;
    nodalValues.resize(nPe);

    mesh->getElementNodalValues(in, &(*(nodalValues.begin())), elementID);

    const double x              = domain_coord[0];
    const double y              = domain_coord[1];
    const double z              = domain_coord[2];

    const unsigned int shiftSzX = eleOrder / interpOrder;
    const unsigned int shiftSzY = eleOrder / interpOrder;
    const unsigned int shiftSzZ = eleOrder / interpOrder;

    const double eleX           = pt_min.x();
    const double eleY           = pt_min.y();
    const double eleZ           = pt_min.z();

    // step sizes for the interpolation order.
    const double hx             = (pt_max.x() - pt_min.x()) / (interpOrder);
    const double hy             = (pt_max.y() - pt_min.y()) / (interpOrder);
    const double hz             = (pt_max.z() - pt_min.z()) / (interpOrder);

    std::vector<T>
        interp_yz;  // x=coord[0] plane (yz) plane interpolated values.
    std::vector<T> interp_z;  // (line x=coord[0], and y=cood[1])
    interp_yz.resize((eleOrder + 1) * (eleOrder + 1), (T)0.0);
    interp_z.resize(eleOrder + 1, T(0.0));

    // computation of the lagrange polynomials in the x direction.
    double Lpoly[interpOrder +
                 1];  // Lagrange polynomials evaluated in in the x direction.
    for (unsigned int li = 0; li < (eleOrder + 1); li += shiftSzX) {
        Lpoly[li / shiftSzX] = 1.0;
        for (unsigned int lj = 0; lj < (eleOrder + 1); lj += shiftSzX) {
            if (lj == li) continue;
            Lpoly[li / shiftSzX] *= ((x - (eleX + hx * lj))) /
                                    ((eleX + hx * li) - (eleX + hx * lj));
        }
    }

    // filling the yz plane.
    for (unsigned int k = 0; k < (eleOrder + 1); k++)
        for (unsigned int j = 0; j < (eleOrder + 1); j++) {
            interp_yz[k * (eleOrder + 1) + j] = (T)0.0;
            for (unsigned int li = 0; li < (eleOrder + 1); li += shiftSzX)
                interp_yz[k * (eleOrder + 1) + j] +=
                    nodalValues[k * (eleOrder + 1) * (eleOrder + 1) +
                                j * (eleOrder + 1) + li] *
                    Lpoly[li / shiftSzX];
        }

    // Lagrange polynomials evaluated in the y direction.
    for (unsigned int li = 0; li < (eleOrder + 1); li += shiftSzY) {
        Lpoly[li / shiftSzY] = 1.0;
        for (unsigned int lj = 0; lj < (eleOrder + 1); lj += shiftSzY) {
            if (lj == li) continue;
            Lpoly[li / shiftSzY] *= ((y - (eleY + hy * lj))) /
                                    ((eleY + hy * li) - (eleY + hy * lj));
        }
    }

    for (unsigned int k = 0; k < (eleOrder + 1);
         k++)  // (line x=coord[0], and y=cood[1])
    {
        interp_z[k] = (T)0.0;
        for (unsigned int li = 0; li < eleOrder + 1;
             li += shiftSzY) {  // lagrange basis polynomial
            interp_z[k] +=
                interp_yz[k * (eleOrder + 1) + li] * Lpoly[li / shiftSzY];
        }
    }

    // Lagrange polynomials evaluated in the z direction.
    for (unsigned int li = 0; li < (eleOrder + 1); li += shiftSzZ) {
        Lpoly[li / shiftSzZ] = 1.0;
        for (unsigned int lj = 0; lj < (eleOrder + 1); lj += shiftSzZ) {
            if (lj == li) continue;
            Lpoly[li / shiftSzY] *= ((z - (eleZ + hz * lj))) /
                                    ((eleZ + hz * li) - (eleZ + hz * lj));
        }
    }

    T val = (T)0.0;
    for (unsigned int li = 0; li < eleOrder + 1;
         li +=
         shiftSzX)  // interpolating to the point (coord[0],coord[1],coord[2])
    {               // lagrange basis polynomial
        val += interp_z[li] * Lpoly[li / shiftSzZ];
    }

    return val;
}

template <typename T>
T linear_lagrange(const ot::Mesh* mesh, const T* in, double* domain_coord,
                  const Point& pt_min, const Point& pt_max,
                  unsigned int elementID) {
    if (!mesh->isActive()) return (T)0.0;

    const unsigned int localBegin   = mesh->getElementLocalBegin();
    const unsigned int localEnd     = mesh->getElementLocalEnd();

    const unsigned int eleOrder     = mesh->getElementOrder();
    const unsigned int nPe          = mesh->getNumNodesPerElement();

    const unsigned int rankActive   = mesh->getMPIRank();
    const unsigned int npesActive   = mesh->getMPICommSize();
    const ot::TreeNode* allElements = &(*(mesh->getAllElements().begin()));

    Point d_pt(domain_coord[0], domain_coord[1], domain_coord[2]);

    //!!! Note: This can only happens due to floating point rounding off issues,
    //!-------

    if (fabs(domain_coord[0] - pt_min.x()) < 1e-6) domain_coord[0] = pt_min.x();

    if (fabs(domain_coord[1] - pt_min.y()) < 1e-6) domain_coord[1] = pt_min.y();

    if (fabs(domain_coord[2] - pt_min.z()) < 1e-6) domain_coord[2] = pt_min.z();

    if (fabs(domain_coord[0] - pt_max.x()) < 1e-6) domain_coord[0] = pt_max.x();

    if (fabs(domain_coord[1] - pt_max.y()) < 1e-6) domain_coord[1] = pt_max.y();

    if (fabs(domain_coord[2] - pt_max.z()) < 1e-6) domain_coord[2] = pt_max.z();

    // !!!
    // -------------------------------------------------------------------------------

    if (domain_coord[0] < pt_min.x() || domain_coord[0] > pt_max.x()) {
        std::cout << "Error in " << __func__
                  << " x coord is not lies in the element for Lagrange "
                     "interpolation with the specified element "
                  << allElements[elementID] << std::endl;
        std::cout << "d_point : " << d_pt << " pt_min : " << pt_min
                  << " pt_max: " << pt_max << std::endl;
        exit(0);
    }

    if (domain_coord[1] < pt_min.y() || domain_coord[1] > pt_max.y()) {
        std::cout << "Error in " << __func__
                  << " y coord is not lies in the element for Lagrange "
                     "interpolation with the specified element "
                  << allElements[elementID] << std::endl;
        std::cout << "d_point : " << d_pt << " pt_min : " << pt_min
                  << " pt_max: " << pt_max << std::endl;
        exit(0);
    }

    if (domain_coord[2] < pt_min.z() || domain_coord[2] > pt_max.z()) {
        std::cout << "Error in " << __func__
                  << " z coord is not lies in the element for Lagrange "
                     "interpolation with the specified element "
                  << allElements[elementID] << std::endl;
        std::cout << "d_point : " << d_pt << " pt_min : " << pt_min
                  << " pt_max: " << pt_max << std::endl;
        exit(0);
    }

    std::vector<T> nodalValues;
    nodalValues.resize(nPe);

    mesh->getElementNodalValues(in, &(*(nodalValues.begin())), elementID);

    const double x          = domain_coord[0];
    const double y          = domain_coord[1];
    const double z          = domain_coord[2];

    const double eleX       = pt_min.x();
    const double eleY       = pt_min.y();
    const double eleZ       = pt_min.z();

    // step sizes for the interpolation order.
    const double hx         = (pt_max.x() - pt_min.x()) / (eleOrder);
    const double hy         = (pt_max.y() - pt_min.y()) / (eleOrder);
    const double hz         = (pt_max.z() - pt_min.z()) / (eleOrder);

    const unsigned int iix  = (unsigned int)std::floor((x - eleX) / hx);
    const unsigned int jjy  = (unsigned int)std::floor((y - eleY) / hy);
    const unsigned int kkz  = (unsigned int)std::floor((z - eleZ) / hz);

    unsigned int k_index[2] = {kkz, kkz + 1};
    unsigned int j_index[2] = {jjy, jjy + 1};
    unsigned int i_index[2] = {iix, iix + 1};

    assert(iix < (eleOrder + 1) && iix >= 0);
    assert(jjy < (eleOrder + 1) && jjy >= 0);
    assert(kkz < (eleOrder + 1) && kkz >= 0);

    std::vector<T>
        interp_yz;  // x=coord[0] plane (yz) plane interpolated values.
    std::vector<T> interp_z;  // (line x=coord[0], and y=cood[1])

    const unsigned int interp_order = 1;

    interp_yz.resize((interp_order + 1) * (interp_order + 1), (T)0.0);
    interp_z.resize((interp_order + 1), T(0.0));

    /*
        Point simplex_coord[NUM_CHILDREN];

        simplex_coord [ 0 ] = Point( eleX +      iix * hx ,   eleY +  jjy * hy
       ,   eleZ + kkz * hz); simplex_coord [ 1 ] = Point( eleX +  (iix+1) * hx ,
       eleY +  jjy * hy      ,   eleZ + kkz * hz); simplex_coord [ 2 ] = Point(
       eleX +      iix * hx ,   eleY + (jjy + 1) * hy ,   eleZ + kkz * hz);
        simplex_coord [ 3 ] = Point( eleX +  (iix+1) * hx ,   eleY + (jjy + 1) *
       hy ,   eleZ + kkz * hz);

        simplex_coord [ 4 ] = Point( eleX +      iix * hx ,   eleY +  jjy * hy
       ,   eleZ + (kkz+1) * hz); simplex_coord [ 5 ] = Point( eleX +  (iix+1) *
       hx ,   eleY +  jjy * hy      ,   eleZ + (kkz+1) * hz); simplex_coord [ 6
       ] = Point( eleX +      iix * hx ,   eleY + (jjy + 1) * hy ,   eleZ +
       (kkz+1) * hz); simplex_coord [ 7 ] = Point( eleX +  (iix+1) * hx ,   eleY
       + (jjy + 1) * hy ,   eleZ + (kkz+1) * hz);
    */

    double Lpoly[2];  // Lagrange polynomials evaluated in in the x direction.
    double sc[2];

    sc[0] = eleX + (iix)*hx;
    sc[1] = eleX + (iix + 1) * hx;

    assert(x >= sc[0] && x <= sc[1]);

    for (unsigned int li = 0; li < (interp_order + 1); li += 1) {
        Lpoly[li] = 1.0;
        for (unsigned int lj = 0; lj < (interp_order + 1); lj += 1) {
            if (lj == li) continue;
            Lpoly[li] *= ((x - sc[lj])) / (sc[li] - sc[lj]);
        }
    }

    // -- filling the yz plane.
    for (unsigned int k = 0; k < (interp_order + 1); k++)
        for (unsigned int j = 0; j < (interp_order + 1); j++) {
            interp_yz[k * (interp_order + 1) + j] = (T)0.0;
            for (unsigned int li = 0; li < (interp_order + 1); li += 1)
                interp_yz[k * (interp_order + 1) + j] +=
                    nodalValues[k_index[k] * (eleOrder + 1) * (eleOrder + 1) +
                                j_index[j] * (eleOrder + 1) + i_index[li]] *
                    Lpoly[li];
        }

    sc[0] = eleY + (jjy)*hy;
    sc[1] = eleY + (jjy + 1) * hy;

    assert(y >= sc[0] && y <= sc[1]);

    // -- Lagrange polynomials evaluated in the y direction.
    for (unsigned int li = 0; li < (interp_order + 1); li += 1) {
        Lpoly[li] = 1.0;
        for (unsigned int lj = 0; lj < (interp_order + 1); lj += 1) {
            if (lj == li) continue;
            Lpoly[li] *= ((y - sc[lj])) / (sc[li] - sc[lj]);
        }
    }

    for (unsigned int k = 0; k < (interp_order + 1); k++) {
        interp_z[k] = (T)0.0;
        for (unsigned int li = 0; li < interp_order + 1; li += 1) {
            // -- lagrange basis polynomial
            interp_z[k] += interp_yz[k * (interp_order + 1) + li] * Lpoly[li];
        }
    }

    sc[0] = eleZ + (kkz)*hz;
    sc[1] = eleZ + (kkz + 1) * hz;

    assert(z >= sc[0] && z <= sc[1]);

    // -- Lagrange polynomials evaluated in the z direction.
    for (unsigned int li = 0; li < (interp_order + 1); li += 1) {
        Lpoly[li] = 1.0;
        for (unsigned int lj = 0; lj < (interp_order + 1); lj += 1) {
            if (lj == li) continue;
            Lpoly[li] *= ((z - sc[lj])) / (sc[li] - sc[lj]);
        }
    }

    T val = (T)0.0;
    for (unsigned int li = 0; li < interp_order + 1; li += 1)
        val += interp_z[li] * Lpoly[li];

    return val;
}

template <typename T, typename CoordT>
void interpolateToCoords(const ot::Mesh* mesh, const T* in,
                         const CoordT* domain_coords, unsigned int length,
                         const Point* const grid_limit,
                         const Point* const domain_limit, T* out,
                         std::vector<unsigned int>& validIndices,
                         std::vector<int>* outLev) {
    unsigned int rankGlobal = mesh->getMPIRankGlobal();
    unsigned int npesGlobal = mesh->getMPICommSizeGlobal();

    dendro::logger::debug(dendro::logger::Scope{"daUtils"},
                          "Now running interpolateToCoords...");

    if (mesh->isActive()) {
        unsigned int rankActive = mesh->getMPIRank();
        unsigned int npesActive = mesh->getMPICommSize();

        const double gridRangeX =
            grid_limit[1].x() - grid_limit[0].x();  // x range of the grid.
        const double gridRangeY =
            grid_limit[1].y() - grid_limit[0].y();  // y range of the grid.
        const double gridRangeZ =
            grid_limit[1].z() - grid_limit[0].z();  // z range of the grid.

        const double domainRangeX =
            domain_limit[1].x() -
            domain_limit[0].x();  // x range of the domain.
        const double domainRangeY =
            domain_limit[1].y() -
            domain_limit[0].y();  // y range of the domain.
        const double domainRangeZ =
            domain_limit[1].z() -
            domain_limit[0].z();  // z range of the domain.

        const ot::TreeNode* allElements = &(*(mesh->getAllElements().begin()));

        std::vector<ot::TreeNode> meshOctree;
        meshOctree.resize(mesh->getAllElements().size());

        for (unsigned int ele = 0; ele < meshOctree.size(); ele++)
            meshOctree[ele] = allElements[ele];

        const unsigned int localElementBegin = mesh->getElementLocalBegin();
        const unsigned int localElementEnd   = mesh->getElementLocalEnd();

        const unsigned int eleOrder          = mesh->getElementOrder();
        const unsigned int nPe               = mesh->getNumNodesPerElement();

        std::vector<T> nodalValues;
        nodalValues.resize(nPe);

        if (length % m_uiDim != 0) {
            std::cout << "Invalid use of " << __func__
                      << " length of coords should be perfectly divisible by "
                         "m_uiDim: "
                      << m_uiDim << std::endl;
            return;
        }

        // DENDRO_INTERP_CHECK=1: collective over the active comm after the
        // pass: every in-domain point found on exactly one rank, else print
        static const bool check = []() {
            const char* e = std::getenv("DENDRO_INTERP_CHECK");
            return e && e[0] == '1';
        }();
        // DENDRO_INTERP_PROF=1: rank 0 prints per-call timings and counts; 2: every rank
        static const int iprof = []() {
            const char* e = std::getenv("DENDRO_INTERP_PROF");
            return e ? std::atoi(e) : 0;
        }();
        static unsigned long long ncall = 0;
        ncall++;
        double tq0 = 0, tq2 = 0;
        if (iprof) tq0 = MPI_Wtime();

        // 1. convert coords to octants
        const unsigned int numPts = length / m_uiDim;
        if (outLev) outLev->assign(numPts, -1);
        // out-of-domain points are skipped below: a negative offset cast to
        // unsigned int wraps and would match a bogus element
        unsigned int n_skipped = 0;
        for (unsigned int i = 0; i < numPts; i++) {
            const double cx = domain_coords[m_uiDim * i + 0];
            const double cy = domain_coords[m_uiDim * i + 1];
            const double cz = domain_coords[m_uiDim * i + 2];
            if (cx < domain_limit[0].x() || cx > domain_limit[1].x() ||
                cy < domain_limit[0].y() || cy > domain_limit[1].y() ||
                cz < domain_limit[0].z() || cz > domain_limit[1].z())
                n_skipped++;
        }

        if (n_skipped > 0 && rankActive == 0) {
            std::cerr << "Warning[" << __func__ << "]: skipped " << n_skipped
                      << " of " << numPts << " points outside domain ["
                      << domain_limit[0].x() << "," << domain_limit[0].y()
                      << "," << domain_limit[0].z() << "] to ["
                      << domain_limit[1].x() << "," << domain_limit[1].y()
                      << "," << domain_limit[1].z() << "]" << std::endl;
        }

        validIndices.clear();

        // every point is located by containment in the local elements, from
        // the deepest one that contains it (the true leaf): an SFC splitter
        // search assumes one SFC range per rank, and a labelled local range
        // can hold coarse duplicate instances that overlap the fine leaves,
        // whose cgs have no canonical writer (the BSSN tracker NaN, see
        // docs/graph_partitioning_correctness.md). outLev (when provided)
        // reports the chosen element level per point so callers can resolve
        // cross-rank duplicate-instance contributions by depth.
        {
            // the local elements' bounding box; a point outside it is in no
            // local element
            unsigned int bx0 = std::numeric_limits<unsigned int>::max(), bx1 = 0;
            unsigned int by0 = bx0, by1 = 0, bz0 = bx0, bz1 = 0;
            for (unsigned int e = localElementBegin; e < localElementEnd; e++) {
                const ot::TreeNode& el = meshOctree[e];
                bx0 = std::min(bx0, el.minX()); bx1 = std::max(bx1, el.maxX());
                by0 = std::min(by0, el.minY()); by1 = std::max(by1, el.maxY());
                bz0 = std::min(bz0, el.minZ()); bz1 = std::max(bz1, el.maxZ());
            }
            for (unsigned int i = 0; i < numPts; i++) {
                const unsigned int ox = (unsigned int)(grid_limit[0].x() +
                    (domain_coords[m_uiDim * i + 0] - domain_limit[0].x()) *
                        (gridRangeX / domainRangeX));
                const unsigned int oy = (unsigned int)(grid_limit[0].y() +
                    (domain_coords[m_uiDim * i + 1] - domain_limit[0].y()) *
                        (gridRangeY / domainRangeY));
                const unsigned int oz = (unsigned int)(grid_limit[0].z() +
                    (domain_coords[m_uiDim * i + 2] - domain_limit[0].z()) *
                        (gridRangeZ / domainRangeZ));
                if (ox < bx0 || ox >= bx1 || oy < by0 || oy >= by1 || oz < bz0 ||
                    oz >= bz1)
                    continue;
                unsigned int bestE   = LOOK_UP_TABLE_DEFAULT;
                unsigned int bestLev = 0;
                for (unsigned int e = localElementBegin; e < localElementEnd;
                     e++) {
                    const ot::TreeNode& el = meshOctree[e];
                    if (ox < el.minX() || ox >= el.maxX()) continue;
                    if (oy < el.minY() || oy >= el.maxY()) continue;
                    if (oz < el.minZ() || oz >= el.maxZ()) continue;
                    if (bestE == LOOK_UP_TABLE_DEFAULT ||
                        el.getLevel() > bestLev) {
                        bestE   = e;
                        bestLev = el.getLevel();
                    }
                }
                if (bestE == LOOK_UP_TABLE_DEFAULT) continue;

                const ot::TreeNode& el = meshOctree[bestE];
                Point r_min(
                    domain_limit[0].x() + ((el.minX() - grid_limit[0].x()) *
                                           (domainRangeX / gridRangeX)),
                    domain_limit[0].y() + ((el.minY() - grid_limit[0].y()) *
                                           (domainRangeY / gridRangeY)),
                    domain_limit[0].z() + ((el.minZ() - grid_limit[0].z()) *
                                           (domainRangeZ / gridRangeZ)));
                Point r_max(
                    domain_limit[0].x() + ((el.maxX() - grid_limit[0].x()) *
                                           (domainRangeX / gridRangeX)),
                    domain_limit[0].y() + ((el.maxY() - grid_limit[0].y()) *
                                           (domainRangeY / gridRangeY)),
                    domain_limit[0].z() + ((el.maxZ() - grid_limit[0].z()) *
                                           (domainRangeZ / gridRangeZ)));

                double c[3];
                c[0]   = domain_coords[m_uiDim * i];
                c[1]   = domain_coords[m_uiDim * i + 1];
                c[2]   = domain_coords[m_uiDim * i + 2];
                out[i] = lagrangeInterpElementToCoord(
                    mesh, in, c, r_min, r_max, bestE,
                    mesh->getElementOrder());
                if (outLev) (*outLev)[i] = (int)bestLev;
                validIndices.push_back(i);
            }
        }

        std::sort(validIndices.begin(), validIndices.end());
        meshOctree.clear();

        if (iprof) {
            tq2 = MPI_Wtime();
            if (iprof >= 2 || rankActive == 0)
                std::printf(
                    "[interp-prof r%u] call %llu pts %u local_elems %u "
                    "%.3e s found %zu\n",
                    rankActive, ncall, numPts,
                    localElementEnd - localElementBegin, tq2 - tq0,
                    validIndices.size());
        }
        if (check) {
            // every in-domain point on exactly one rank: the callers sum the
            // per-rank contributions, so a miss drops a point and a duplicate
            // counts it twice
            std::vector<int> cnt(numPts, 0);
            for (unsigned int v : validIndices)
                if (v < numPts) cnt[v] = 1;
            MPI_Allreduce(MPI_IN_PLACE, cnt.data(), (int)numPts, MPI_INT,
                          MPI_SUM, mesh->getMPICommunicator());
            unsigned int nzero = 0, ndup = 0, first = numPts;
            for (unsigned int i = 0; i < numPts; i++) {
                const double cx = domain_coords[m_uiDim * i + 0];
                const double cy = domain_coords[m_uiDim * i + 1];
                const double cz = domain_coords[m_uiDim * i + 2];
                if (cx < domain_limit[0].x() || cx > domain_limit[1].x() ||
                    cy < domain_limit[0].y() || cy > domain_limit[1].y() ||
                    cz < domain_limit[0].z() || cz > domain_limit[1].z())
                    continue;
                if (cnt[i] == 0) { nzero++; if (first == numPts) first = i; }
                if (cnt[i] > 1) { ndup++; if (first == numPts) first = i; }
            }
            if (rankActive == 0 && (nzero || ndup))
                std::printf(
                    "[interp-check] call %llu: %u of %u points found on no "
                    "rank, %u on more than one (first bad point %u at %.17g "
                    "%.17g %.17g)\n",
                    ncall, nzero, numPts, ndup, first,
                    domain_coords[m_uiDim * first + 0],
                    domain_coords[m_uiDim * first + 1],
                    domain_coords[m_uiDim * first + 2]);
        }
    }

    dendro::logger::debug(dendro::logger::Scope{"daUtils"},
                          "Finished calling interpolate coordinates!");
}

template <typename T, typename CoordT>
void interpolateToCoordsAndGather(const ot::Mesh* mesh, const T* in,
                                  const CoordT* domain_coords,
                                  unsigned int length,
                                  const Point* const grid_limit,
                                  const Point* const domain_limit, T* out,
                                  unsigned int root, unsigned int dof) {
    if (mesh->isActive()) {
        MPI_Comm comm_active     = mesh->getMPICommunicator();
        unsigned int rank_active = mesh->getMPIRank();
        unsigned int npes_active = mesh->getMPICommSize();

        std::vector<unsigned int> v_index;
        v_index.reserve(length);

        unsigned int in_sz  = mesh->getDegOfFreedom();
        unsigned int out_sz = length;

        interpolateToCoords(mesh, in + 0 * in_sz, domain_coords, length,
                            grid_limit, domain_limit, out + 0 * out_sz,
                            v_index);
        int s_count = v_index.size() * dof;
        std::vector<int> r_count;
        std::vector<int> r_offset;

        if (rank_active == root) {
            r_count.resize(npes_active);
            r_offset.resize(npes_active);
        }

        par::Mpi_Gather(&s_count, r_count.data(), 1, root, comm_active);

        if (rank_active == root) {
            r_offset[0] = 0;
            omp_par::scan(r_count.data(), r_offset.data(), r_count.size());
        }

        for (unsigned int v_id = 1; v_id < dof; v_id++) {
            v_index.clear();
            interpolateToCoords(mesh, in + v_id * in_sz, domain_coords, length,
                                grid_limit, domain_limit, out + v_id * out_sz,
                                v_index);
        }

        std::vector<T> send_buff;
        send_buff.resize(s_count);

        for (unsigned int w = 0; w < v_index.size(); w++)
            for (unsigned int v_id = 0; v_id < dof; v_id++)
                send_buff[w * dof + v_id] = out[v_id * out_sz + v_index[w]];

        par::Mpi_Gatherv(send_buff.data(), s_count, out, r_count.data(),
                         r_offset.data(), root, comm_active);
    }
    return;
}

}  // end of namespace da

}  // end of namespace ot
