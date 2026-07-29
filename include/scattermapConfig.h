
#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <sstream>
#include <string>

namespace sm_config {
struct SMConfig {
    // cfg encoded single value for size of i, j, and k
    // NOTE: all three can be either 0, 1, or 2, so bitwise we get 00, 01, 10 as
    // possible options
    unsigned char cfg;
    // offset for indexing into the scattermap
    std::size_t sm_offset;

    // TODO: we can probably ignore this, since sm_offset ensures that the data
    // is put back no matter if the sort order "differs" slightly between
    // processes, but currently in mesh.cpp, we're extracting data this way
    // unsigned int nodeOffsetID;
    // yes, it can be ignored, we can sort by sm_offset as our secondary field

    void setCFG(unsigned char cfg_in) { cfg = cfg_in; }

    void setCFG(unsigned int ii, unsigned int jj, unsigned int kk) {
        // encodes the data
        cfg = (ii << 6) | (jj << 3) | kk;
    }

    void setSMOffset(std::size_t sm_offset_in) { sm_offset = sm_offset_in; }

    const unsigned int getNDim() const {
        return (((cfg >> 6) & 7u) == 1) + (((cfg >> 3) & 7u) == 1) +
               (((cfg) & 7u) == 1);
    }

    const bool getX() const { return (((cfg >> 6) & 7u) == 1); }
    const bool getY() const { return (((cfg >> 3) & 7u) == 1); }
    const bool getZ() const { return (((cfg) & 7u) == 1); }

    std::string cfgStringBits() const {
        std::stringstream ss;
        std::for_each(
            std::make_reverse_iterator(std::begin({0, 1, 2, 3, 4, 5, 6, 7})),
            std::make_reverse_iterator(std::end({0, 1, 2, 3, 4, 5, 6, 7})),
            [&](int i) { ss << ((cfg >> i) & 1); });
        return ss.str();
    }

    std::string toBinaryString() const {
        std::stringstream ss;
        const unsigned int cfg_cast = static_cast<unsigned int>(cfg);
        // int num_bits                = std::numeric_limits<unsigned
        // int>::digits;
        int num_bits                = 9;
        for (int i = num_bits - 1; i >= 0; --i) {
            ss << ((cfg_cast >> i) & 1);
        }
        return ss.str();
    }

    unsigned int cfgAsInt() const { return static_cast<unsigned int>(cfg); }
};
}  // namespace sm_config
