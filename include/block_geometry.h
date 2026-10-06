// block_geometry.h  --  proposed dendrolib API (prototype)
//
// Collapses the per-block geometry preamble that every "compute over blocks"
// kernel (rhs, constraints, psi4, ...) copy-pastes today into one range-for.
#pragma once

#include <cstddef>
#include <vector>

#include "block.h"
#include "point.h"

namespace ot {

// Everything a per-block kernel needs. POD -> cheap to materialize on the host
// and trivially copyable to a device.
struct BlockGeometry {
    unsigned int index;
    DendroIntL   offset;
    unsigned int sz[3];
    unsigned int bflag;
    unsigned int pw;
    double       dx[3];
    double       ptmin[3];
    double       ptmax[3];
    const Block* raw;

    DendroIntL n() const { return (DendroIntL)sz[0] * sz[1] * sz[2]; }
};

class BlockGeometryRange {
   public:
    BlockGeometryRange(const Block* blocks, std::size_t numBlocks,
                       const Point& dmin, const Point& dmax,
                       const Point& octMin, const Point& octMax)
        : m_blocks(blocks),
          m_n(numBlocks),
          m_dmin(dmin),
          m_dmax(dmax),
          m_octMin(octMin),
          m_sx((dmax.x() - dmin.x()) / (octMax.x() - octMin.x())),
          m_sy((dmax.y() - dmin.y()) / (octMax.y() - octMin.y())),
          m_sz((dmax.z() - dmin.z()) / (octMax.z() - octMin.z())) {}

    // convenience: from a mesh's local block list
    BlockGeometryRange(const std::vector<Block>& blocks, const Point& dmin,
                       const Point& dmax, const Point& octMin, const Point& octMax)
        : BlockGeometryRange(blocks.data(), blocks.size(), dmin, dmax, octMin,
                             octMax) {}

    BlockGeometry at(std::size_t i) const {
        const Block& b = m_blocks[i];
        BlockGeometry g;
        g.index  = (unsigned int)i;
        g.offset = b.getOffset();
        g.sz[0]  = b.getAllocationSzX();
        g.sz[1]  = b.getAllocationSzY();
        g.sz[2]  = b.getAllocationSzZ();
        g.bflag  = b.getBlkNodeFlag();
        g.pw     = b.get1DPadWidth();
        g.raw    = &b;
        g.dx[0]  = b.computeDx(m_dmin, m_dmax);
        g.dx[1]  = b.computeDy(m_dmin, m_dmax);
        g.dx[2]  = b.computeDz(m_dmin, m_dmax);
        const TreeNode nd = b.getBlockNode();
        g.ptmin[0] = m_sx * (nd.minX() - m_octMin.x()) + m_dmin.x() - g.pw * g.dx[0];
        g.ptmin[1] = m_sy * (nd.minY() - m_octMin.y()) + m_dmin.y() - g.pw * g.dx[1];
        g.ptmin[2] = m_sz * (nd.minZ() - m_octMin.z()) + m_dmin.z() - g.pw * g.dx[2];
        g.ptmax[0] = m_sx * (nd.maxX() - m_octMin.x()) + m_dmin.x() + g.pw * g.dx[0];
        g.ptmax[1] = m_sy * (nd.maxY() - m_octMin.y()) + m_dmin.y() + g.pw * g.dx[1];
        g.ptmax[2] = m_sz * (nd.maxZ() - m_octMin.z()) + m_dmin.z() + g.pw * g.dx[2];
        return g;
    }

    struct iterator {
        const BlockGeometryRange* r;
        std::size_t               i;
        bool operator!=(const iterator& o) const { return i != o.i; }
        iterator& operator++() { ++i; return *this; }
        BlockGeometry operator*() const { return r->at(i); }
    };
    iterator    begin() const { return {this, 0}; }
    iterator    end() const { return {this, m_n}; }
    std::size_t size() const { return m_n; }

   private:
    const Block* m_blocks;
    std::size_t  m_n;
    Point        m_dmin, m_dmax, m_octMin;
    double       m_sx, m_sy, m_sz;
};

}  // namespace ot
