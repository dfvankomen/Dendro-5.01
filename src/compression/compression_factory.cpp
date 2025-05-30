#include "compression/compression_factory.hpp"

#include "compression/compression_base.hpp"
#include "compression/compression_wrapper.hpp"

// compressors
#include "compression/compressor_dummy.hpp"
#include "compression/compressor_interpolation.hpp"
#include "compression/compressor_onnx.hpp"
#include "compression/compressor_torchscript.hpp"
#include "compression/compressor_zfp.hpp"

namespace dendrocompression {

DendroCompression<float> floatCompressor;
DendroCompression<double> doubleCompressor;

void register_compressors() {
    // ----
    // DummyCompressor Registration, used for testing
    floatCompressor.register_compressor(
        dendrocompression::CompressionType::COMP_DUMMY,
        [](const std::vector<std::any>& args) {
            return std::make_unique<DummyCompressor<float>>(
                // dummy compressor takes in two unsigned ints
                std::any_cast<unsigned int>(args[0]),
                std::any_cast<unsigned int>(args[1]));
        });
    doubleCompressor.register_compressor(
        dendrocompression::CompressionType::COMP_DUMMY,
        [](const std::vector<std::any>& args) {
            return std::make_unique<DummyCompressor<double>>(
                // dummy compressor takes in two unsigned ints
                std::any_cast<unsigned int>(args[0]),
                std::any_cast<unsigned int>(args[1]));
        });

    // ZFP Compressor Registration
    floatCompressor.register_compressor(
        dendrocompression::CompressionType::COMP_ZFP,
        [](const std::vector<std::any>& args) {
            if (args.size() != 4 || !args[0].has_value() ||
                !args[1].has_value() || !args[2].has_value() ||
                !args[3].has_value()) {
                throw std::runtime_error(
                    "Invalid argument count or unset values");
            }
            return std::make_unique<ZFPCompressor<float>>(
                // dummy compressor takes in two unsigned ints
                std::any_cast<unsigned int>(args[0]),
                std::any_cast<unsigned int>(args[1]),
                std::any_cast<std::string>(args[2]),
                std::any_cast<double>(args[3]));
        });

    doubleCompressor.register_compressor(
        dendrocompression::CompressionType::COMP_ZFP,
        [](const std::vector<std::any>& args) {
            return std::make_unique<ZFPCompressor<double>>(
                // dummy compressor takes in two unsigned ints
                std::any_cast<unsigned int>(args[0]),
                std::any_cast<unsigned int>(args[1]),
                std::any_cast<std::string>(args[2]),
                std::any_cast<double>(args[3]));
        });

    // TorchScript Compressor Registration
    floatCompressor.register_compressor(
        dendrocompression::CompressionType::COMP_TORCH_SCRIPT,
        [](const std::vector<std::any>& args) {
            if (args.size() != 10 || !args[0].has_value() ||
                !args[1].has_value() || !args[2].has_value() ||
                !args[3].has_value()) {
                throw std::runtime_error(
                    "Invalid number of inputs for setting up TOrchScript "
                    "Compressor");
            }
            return std::make_unique<TorchScriptCompressor<float>>(
                // dummy compressor takes in two unsigned ints
                std::any_cast<unsigned int>(args[0]),
                std::any_cast<unsigned int>(args[1]),
                std::any_cast<std::string>(args[2]),
                std::any_cast<std::string>(args[3]),
                std::any_cast<std::string>(args[4]),
                std::any_cast<std::string>(args[5]),
                std::any_cast<std::string>(args[6]),
                std::any_cast<std::string>(args[7]),
                std::any_cast<std::string>(args[8]),
                std::any_cast<std::string>(args[9]));
        });

    doubleCompressor.register_compressor(
        dendrocompression::CompressionType::COMP_TORCH_SCRIPT,
        [](const std::vector<std::any>& args) {
            return std::make_unique<TorchScriptCompressor<double>>(
                // dummy compressor takes in two unsigned ints
                std::any_cast<unsigned int>(args[0]),
                std::any_cast<unsigned int>(args[1]),
                std::any_cast<std::string>(args[2]),
                std::any_cast<std::string>(args[3]),
                std::any_cast<std::string>(args[4]),
                std::any_cast<std::string>(args[5]),
                std::any_cast<std::string>(args[6]),
                std::any_cast<std::string>(args[7]),
                std::any_cast<std::string>(args[8]),
                std::any_cast<std::string>(args[9]));
        });

    // ONNX Compressor Registration
    floatCompressor.register_compressor(
        dendrocompression::CompressionType::COMP_ONNX_MODEL,
        [](const std::vector<std::any>& args) {
            if (args.size() != 10 || !args[0].has_value() ||
                !args[1].has_value() || !args[2].has_value() ||
                !args[3].has_value()) {
                throw std::runtime_error(
                    "Invalid number of inputs for setting up TOrchScript "
                    "Compressor");
            }
            return std::make_unique<ONNXCompressor<float>>(
                // dummy compressor takes in two unsigned ints
                std::any_cast<unsigned int>(args[0]),
                std::any_cast<unsigned int>(args[1]),
                std::any_cast<std::string>(args[2]),
                std::any_cast<std::string>(args[3]),
                std::any_cast<std::string>(args[4]),
                std::any_cast<std::string>(args[5]),
                std::any_cast<std::string>(args[6]),
                std::any_cast<std::string>(args[7]),
                std::any_cast<std::string>(args[8]),
                std::any_cast<std::string>(args[9]));
        });

    doubleCompressor.register_compressor(
        dendrocompression::CompressionType::COMP_ONNX_MODEL,
        [](const std::vector<std::any>& args) {
            return std::make_unique<ONNXCompressor<double>>(
                // dummy compressor takes in two unsigned ints
                std::any_cast<unsigned int>(args[0]),
                std::any_cast<unsigned int>(args[1]),
                std::any_cast<std::string>(args[2]),
                std::any_cast<std::string>(args[3]),
                std::any_cast<std::string>(args[4]),
                std::any_cast<std::string>(args[5]),
                std::any_cast<std::string>(args[6]),
                std::any_cast<std::string>(args[7]),
                std::any_cast<std::string>(args[8]),
                std::any_cast<std::string>(args[9]));
        });

    // Interpolation Compressor Registration
    floatCompressor.register_compressor(
        dendrocompression::CompressionType::COMP_INTERP,
        [](const std::vector<std::any>& args) {
            if (args.size() != 3 || !args[0].has_value() ||
                !args[1].has_value() || !args[2].has_value()) {
                throw std::runtime_error(
                    "Invalid number of inputs for setting up Interpolation "
                    "Compressor");
            }
            return std::make_unique<InterpolationCompressor<float>>(
                // dummy compressor takes in two unsigned ints
                std::any_cast<unsigned int>(args[0]),
                std::any_cast<unsigned int>(args[1]),
                std::any_cast<unsigned int>(args[2]));
        });

    doubleCompressor.register_compressor(
        dendrocompression::CompressionType::COMP_INTERP,
        [](const std::vector<std::any>& args) {
            if (args.size() != 3 || !args[0].has_value() ||
                !args[1].has_value() || !args[2].has_value()) {
                throw std::runtime_error(
                    "Invalid number of inputs for setting up Interpolation "
                    "Compressor");
            }

            return std::make_unique<InterpolationCompressor<double>>(
                // dummy compressor takes in two unsigned ints
                std::any_cast<unsigned int>(args[0]),
                std::any_cast<unsigned int>(args[1]),
                std::any_cast<unsigned int>(args[2]));
        });

    // do all of the compression registering here...
}

}  // namespace dendrocompression
