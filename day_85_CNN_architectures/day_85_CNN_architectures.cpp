#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

/*
 * CNN Repository and Model Governance Engine
 *
 * C++17 implementation of architecture specifications,
 * tensor shape calculations, parameter estimation and
 * model deployment eligibility.
 *
 * Compile:
 * g++ -std=c++17 -O2 -Wall -Wextra -pedantic cnn_architectures.cpp -o cnn
 *
 * Run:
 * ./cnn
 */

enum class Architecture {
    LENET,
    ALEXNET,
    VGG16,
    INCEPTION
};

enum class LayerType {
    CONVOLUTION,
    POOLING,
    DENSE,
    BRANCH_CONCATENATION,
    ACTIVATION
};

enum class DeploymentState {
    DRAFT,
    VALIDATED,
    APPROVED,
    DEPLOYED,
    REJECTED
};

struct TensorShape {
    int channels;
    int height;
    int width;

    void validate() const {
        if (channels <= 0 || height <= 0 || width <= 0) {
            throw std::invalid_argument("Invalid tensor dimensions");
        }
    }

    std::string toString() const {
        return std::to_string(channels) + "x" +
               std::to_string(height) + "x" +
               std::to_string(width);
    }
};

struct LayerSpec {
    std::string name;
    LayerType type;
    int inputChannels;
    int outputChannels;
    int kernel;
    int stride;
    int padding;
};

struct ArchitectureSpec {
    Architecture architecture;
    std::string name;
    int year;
    int expectedChannels;
    int minimumInput;
    std::vector<LayerSpec> layers;
    std::string innovation;
};

struct AnalysisResult {
    std::string architecture;
    std::size_t convolutionParameters;
    std::size_t denseParameters;
    std::vector<std::string> observations;
};

struct DeploymentRequest {
    std::string modelId;
    Architecture architecture;
    TensorShape input;
    int classes;
    bool evaluationPassed;
    bool securityReviewPassed;
    bool reproducibilityVerified;
    DeploymentState state = DeploymentState::DRAFT;
};

class ArchitectureError : public std::runtime_error {
public:
    explicit ArchitectureError(const std::string& message)
        : std::runtime_error(message) {}
};

class ArchitectureRegistry {
private:
    std::map<Architecture, ArchitectureSpec> specifications;

    static LayerSpec conv(
        const std::string& name,
        int input,
        int output,
        int kernel,
        int stride = 1,
        int padding = 0
    ) {
        return {
            name,
            LayerType::CONVOLUTION,
            input,
            output,
            kernel,
            stride,
            padding
        };
    }

public:
    ArchitectureRegistry() {
        specifications.emplace(
            Architecture::LENET,
            ArchitectureSpec{
                Architecture::LENET,
                "LeNet-5",
                1998,
                1,
                32,
                {
                    conv("conv1", 1, 6, 5),
                    conv("conv2", 6, 16, 5),
                    conv("conv3", 16, 120, 5)
                },
                "Small convolutional layers and average pooling"
            }
        );

        specifications.emplace(
            Architecture::ALEXNET,
            ArchitectureSpec{
                Architecture::ALEXNET,
                "AlexNet",
                2012,
                3,
                224,
                {
                    conv("conv1", 3, 64, 11, 4, 2),
                    conv("conv2", 64, 192, 5, 1, 2),
                    conv("conv3", 192, 384, 3, 1, 1),
                    conv("conv4", 384, 256, 3, 1, 1),
                    conv("conv5", 256, 256, 3, 1, 1)
                },
                "ReLU, dropout and GPU-based large-scale training"
            }
        );

        specifications.emplace(
            Architecture::VGG16,
            ArchitectureSpec{
                Architecture::VGG16,
                "VGG16",
                2014,
                3,
                224,
                {
                    conv("conv1_1", 3, 64, 3, 1, 1),
                    conv("conv1_2", 64, 64, 3, 1, 1),
                    conv("conv2_1", 64, 128, 3, 1, 1),
                    conv("conv2_2", 128, 128, 3, 1, 1),
                    conv("conv3_1", 128, 256, 3, 1, 1),
                    conv("conv3_2", 256, 256, 3, 1, 1),
                    conv("conv3_3", 256, 256, 3, 1, 1),
                    conv("conv4_1", 256, 512, 3, 1, 1),
                    conv("conv4_2", 512, 512, 3, 1, 1),
                    conv("conv4_3", 512, 512, 3, 1, 1),
                    conv("conv5_1", 512, 512, 3, 1, 1),
                    conv("conv5_2", 512, 512, 3, 1, 1),
                    conv("conv5_3", 512, 512, 3, 1, 1)
                },
                "Thirteen 3x3 convolutions and five pooling stages"
            }
        );

        specifications.emplace(
            Architecture::INCEPTION,
            ArchitectureSpec{
                Architecture::INCEPTION,
                "Inception v1",
                2014,
                3,
                224,
                {
                    conv("branch1", 192, 64, 1),
                    conv("branch2_reduce", 192, 96, 1),
                    conv("branch2", 96, 128, 3, 1, 1),
                    conv("branch3_reduce", 192, 16, 1),
                    conv("branch3", 16, 32, 5, 1, 2),
                    conv("branch4", 192, 32, 1)
                },
                "Parallel branches with channel reduction"
            }
        );
    }

    const ArchitectureSpec& get(Architecture architecture) const {
        auto iterator = specifications.find(architecture);

        if (iterator == specifications.end()) {
            throw ArchitectureError("Architecture not registered");
        }

        return iterator->second;
    }

    const std::map<Architecture, ArchitectureSpec>& all() const {
        return specifications;
    }
};

class ModelAnalyzer {
private:
    static std::size_t convolutionParameters(
        int inputChannels,
        int outputChannels,
        int kernel
    ) {
        return static_cast<std::size_t>(
            inputChannels * outputChannels * kernel * kernel +
            outputChannels
        );
    }

    static std::size_t denseParameters(int input, int output) {
        return static_cast<std::size_t>(input) * output + output;
    }

public:
    AnalysisResult analyze(
        const ArchitectureSpec& specification,
        const TensorShape& input,
        int classes
    ) const {
        input.validate();

        if (input.channels != specification.expectedChannels) {
            throw ArchitectureError("Input channel mismatch");
        }

        if (input.height < specification.minimumInput ||
            input.width < specification.minimumInput) {
            throw ArchitectureError("Input resolution is too small");
        }

        if (classes < 2) {
            throw ArchitectureError("At least two classes are required");
        }

        std::size_t convolutionTotal = 0;
        int currentChannels = specification.expectedChannels;

        AnalysisResult result{
            specification.name,
            0,
            0,
            {}
        };

        for (const auto& layer : specification.layers) {
            if (layer.inputChannels != currentChannels) {
                // Inception branches begin from the same source tensor.
                if (specification.architecture != Architecture::INCEPTION) {
                    throw ArchitectureError(
                        "Sequential channel mismatch at " + layer.name
                    );
                }
            }

            const std::size_t parameters = convolutionParameters(
                layer.inputChannels,
                layer.outputChannels,
                layer.kernel
            );

            convolutionTotal += parameters;

            result.observations.push_back(
                layer.name + ": " +
                std::to_string(parameters) + " parameters"
            );

            if (specification.architecture != Architecture::INCEPTION) {
                currentChannels = layer.outputChannels;
            }
        }

        if (specification.architecture == Architecture::LENET) {
            result.denseParameters += denseParameters(84, classes);
        } else if (specification.architecture == Architecture::ALEXNET) {
            result.denseParameters += denseParameters(256 * 6 * 6, 4096);
            result.denseParameters += denseParameters(4096, 4096);
            result.denseParameters += denseParameters(4096, classes);
        } else if (specification.architecture == Architecture::VGG16) {
            result.denseParameters += denseParameters(512 * 7 * 7, 4096);
            result.denseParameters += denseParameters(4096, 4096);
            result.denseParameters += denseParameters(4096, classes);
        } else {
            result.denseParameters += denseParameters(1024, classes);
        }

        result.convolutionParameters = convolutionTotal;

        return result;
    }
};

class DeploymentGovernance {
private:
    static void transition(
        DeploymentRequest& request,
        DeploymentState expected,
        DeploymentState next
    ) {
        if (request.state != expected) {
            throw ArchitectureError("Invalid deployment state transition");
        }

        request.state = next;
    }

public:
    void validate(
        DeploymentRequest& request,
        const ArchitectureRegistry& registry
    ) const {
        if (request.modelId.empty()) {
            throw ArchitectureError("Model identifier is required");
        }

        if (request.classes < 2) {
            throw ArchitectureError("Invalid class count");
        }

        const auto& specification = registry.get(request.architecture);

        request.input.validate();

        if (request.input.channels != specification.expectedChannels ||
            request.input.height < specification.minimumInput ||
            request.input.width < specification.minimumInput) {
            throw ArchitectureError("Model input contract violation");
        }

        transition(
            request,
            DeploymentState::DRAFT,
            DeploymentState::VALIDATED
        );
    }

    bool eligible(const DeploymentRequest& request) const {
        return request.state == DeploymentState::APPROVED &&
               request.evaluationPassed &&
               request.securityReviewPassed &&
               request.reproducibilityVerified;
    }

    void approve(DeploymentRequest& request) const {
        if (request.state != DeploymentState::VALIDATED) {
            throw ArchitectureError("Model must be validated before approval");
        }

        if (!request.evaluationPassed ||
            !request.securityReviewPassed ||
            !request.reproducibilityVerified) {
            throw ArchitectureError(
                "Evaluation, security and reproducibility gates are required"
            );
        }

        transition(
            request,
            DeploymentState::VALIDATED,
            DeploymentState::APPROVED
        );
    }

    void deploy(DeploymentRequest& request) const {
        if (!eligible(request)) {
            throw ArchitectureError("Deployment requirements not satisfied");
        }

        transition(
            request,
            DeploymentState::APPROVED,
            DeploymentState::DEPLOYED
        );
    }
};

std::string architectureName(Architecture architecture) {
    switch (architecture) {
        case Architecture::LENET: return "LeNet-5";
        case Architecture::ALEXNET: return "AlexNet";
        case Architecture::VGG16: return "VGG16";
        case Architecture::INCEPTION: return "Inception v1";
    }

    throw ArchitectureError("Unknown architecture");
}

void printResult(const AnalysisResult& result) {
    std::cout << "\n" << result.architecture << "\n";
    std::cout << "-------------------------------\n";
    std::cout << "Convolution parameters: "
              << result.convolutionParameters << "\n";
    std::cout << "Dense parameters: "
              << result.denseParameters << "\n";
    std::cout << "Total parameters: "
              << result.convolutionParameters +
                 result.denseParameters << "\n";
}

int main() {
    try {
        ArchitectureRegistry registry;
        ModelAnalyzer analyzer;

        std::cout << "CNN ARCHITECTURE ENGINEERING LAB\n";
        std::cout << "================================\n";

        for (const auto& [architecture, specification] : registry.all()) {
            std::cout
                << specification.name
                << " | " << specification.year
                << " | " << specification.innovation
                << "\n";

            TensorShape input{
                specification.expectedChannels,
                specification.minimumInput,
                specification.minimumInput
            };

            AnalysisResult result = analyzer.analyze(
                specification,
                input,
                10
            );

            printResult(result);
        }

        std::cout << "\nDEPLOYMENT GOVERNANCE\n";
        std::cout << "=====================\n";

        DeploymentRequest request{
            "cnn-classifier-001",
            Architecture::INCEPTION,
            {3, 224, 224},
            10,
            true,
            true,
            true,
            DeploymentState::DRAFT
        };

        DeploymentGovernance governance;

        governance.validate(request, registry);

        std::cout << "Model validated.\n";

        governance.approve(request);

        std::cout << "Evaluation and security gates approved.\n";

        governance.deploy(request);

        std::cout << "Model deployment authorized.\n";

        if (request.state != DeploymentState::DEPLOYED) {
            throw ArchitectureError("Unexpected deployment state");
        }

        // Invalid configurations must be rejected.
        try {
            DeploymentRequest invalid{
                "invalid-model",
                Architecture::LENET,
                {3, 32, 32},
                10,
                false,
                false,
                false,
                DeploymentState::DRAFT
            };

            governance.validate(invalid, registry);
        } catch (const ArchitectureError& error) {
            std::cout << "\nInvalid model rejected: "
                      << error.what() << "\n";
        }

        std::cout << "\nAll engineering checks completed.\n";
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
