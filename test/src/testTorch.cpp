#include <torch/script.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

template <typename T>
torch::Tensor convertDataToModelType(std::vector<T>& in,
                                     std::string module_type) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "T must be float or double for conversion!");

    if constexpr (std::is_same_v<T, float>) {
        if (module_type == "float") {
            return torch::from_blob(const_cast<float*>(in.data()),
                                    {1, static_cast<long>(in.size())},
                                    torch::kFloat);
        } else if (module_type == "double") {
            std::vector<double> double_output(in.size());
            std::transform(in.begin(), in.end(), double_output.begin(),
                           [](float d) { return static_cast<double>(d); });
            return torch::from_blob(double_output.data(),
                                    {1, static_cast<long>(in.size())},
                                    torch::kDouble);
        } else {
            std::cerr << "Model data type not currently supported!"
                      << std::endl;
            exit(0);
        }

    } else if constexpr (std::is_same_v<T, double>) {
        if (module_type == "float") {
            std::vector<float> float_output(in.size());
            std::transform(in.begin(), in.end(), float_output.begin(),
                           [](double d) { return static_cast<float>(d); });
            return torch::from_blob(float_output.data(),
                                    {1, static_cast<long>(in.size())},
                                    torch::kFloat);
        } else if (module_type == "double") {
            return torch::from_blob(
                in.data(), {1, static_cast<long>(in.size())}, torch::kDouble);
        } else {
            std::cerr << "Model data type not currently supported!"
                      << std::endl;
            exit(0);
        }
    } else {
        std::cerr << "Internal error: T managed to not be a float or a double "
                     "in data conversion wrapper."
                  << std::endl;
        exit(0);
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: torchTest [exported-script-module]\n";
        return -1;
    }

    torch::jit::Module module;
    try {
        module = torch::jit::load(argv[1]);
    } catch (const c10::Error& e) {
        std::cerr << "Error loading pytorch model!\n";
        return -1;
    }

    std::vector<torch::jit::IValue> inputs;

    unsigned int points_per_dim = 5;
    unsigned int n_vars         = 2;

    unsigned int total_points =
        points_per_dim * points_per_dim * points_per_dim * n_vars;

    std::vector<float> input_data(total_points, 23000.0);

    torch::Tensor input_tensor = convertDataToModelType(input_data, "float");

    std::cout << input_tensor << std::endl;

    // bool use_cuda                 = torch::cuda::is_available();
    // torch::Device device          = use_cuda ? torch::kCUDA : torch::kCPU;
    //
    // module.to(device);

    // input_tensor = input_tensor.to(device);
    inputs.push_back(input_tensor);

    torch::Tensor output;

    try {
        output = module.forward(inputs).toTensor();
    } catch (const std::runtime_error& e) {
        std::cerr << "Something went wrong while attempting to run the model, "
                     "make sure things are correct!"
                  << std::endl;
    }

    std::cout << output << std::endl;

    std::cout << output.sizes() << std::endl;

    unsigned int output_size = output.sizes()[1];

    std::cout << output_size << std::endl;

    return 0;
}
