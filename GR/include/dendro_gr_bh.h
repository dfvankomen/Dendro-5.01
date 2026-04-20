/**
 * @file bh.h
 * @brief Basic structure to store black hole parameters.
 *
 * This is a generic BH struct for use by any GR solver. If your theory
 * needs additional fields (e.g., electric/magnetic charge for EMDA),
 * create a derived struct in your solver project.
 */

#ifndef DENDRO_GR_BH_H
#define DENDRO_GR_BH_H

#include "point.h"

namespace dendro_gr {

struct BH {
   private:
    double m_mass;
    Point m_coord;
    Point m_velocity;
    double m_spin;
    double m_spinTheta;
    double m_spinPhi;

   public:
    BH() : m_mass(0), m_coord(0, 0, 0), m_velocity(0, 0, 0),
           m_spin(0), m_spinTheta(0), m_spinPhi(0) {}

    BH(double mass, double cx, double cy, double cz,
       double vx, double vy, double vz,
       double spin, double spinTheta, double spinPhi)
        : m_mass(mass), m_coord(cx, cy, cz), m_velocity(vx, vy, vz),
          m_spin(spin), m_spinTheta(spinTheta), m_spinPhi(spinPhi) {}

    double getBHMass() const { return m_mass; }
    Point getBHCoord() const { return m_coord; }
    double getBHCoordX() const { return m_coord.x(); }
    double getBHCoordY() const { return m_coord.y(); }
    double getBHCoordZ() const { return m_coord.z(); }
    Point getBHVelocity() const { return m_velocity; }
    double getBHSpin() const { return m_spin; }
    double getBHSpinTheta() const { return m_spinTheta; }
    double getBHSpinPhi() const { return m_spinPhi; }

    void setBHMass(double mass) { m_mass = mass; }
    void setBHCoord(Point coord) { m_coord = coord; }
    void setBHCoord(double x, double y, double z) { m_coord = Point(x, y, z); }
    void setBHVelocity(Point v) { m_velocity = v; }
    void setBHVelocity(double vx, double vy, double vz) { m_velocity = Point(vx, vy, vz); }
    void setBHSpin(double spin) { m_spin = spin; }
    void setBHSpinTheta(double t) { m_spinTheta = t; }
    void setBHSpinPhi(double p) { m_spinPhi = p; }
};

}  // namespace dendro_gr

#endif  // DENDRO_GR_BH_H
