#pragma once
#include <any>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "compression_base.hpp"

namespace dendrocompression {

template <typename T>
class DendroCompression {
    using Creator =
        std::function<std::unique_ptr<Compression<T>>(std::vector<std::any>)>;
    std::unordered_map<dendrocompression::CompressionType, Creator> creators;

   public:
    void register_compressor(
        const dendrocompression::CompressionType &comp_type, Creator creator) {
        creators[comp_type] = std::move(creator);
    }

    std::unique_ptr<Compression<T>> create(
        const dendrocompression::CompressionType comp_type,
        std::vector<std::any> args) {
        return creators.at(comp_type)(std::move(args));
    }
};

extern DendroCompression<float> floatCompressor;
extern DendroCompression<double> doubleCompressor;

void register_compressors();

}  // namespace dendrocompression
