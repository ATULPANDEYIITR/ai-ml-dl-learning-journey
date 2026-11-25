/*
 * Modern CNN Architecture Evaluation Engine
 * C++17
 *
 * Models:
 * - ResNet residual connections
 * - DenseNet feature concatenation
 * - EfficientNet compound scaling
 * - MBConv and squeeze-and-excitation
 *
 * Evaluates:
 * - Tensor compatibility
 * - Parameter counts
 * - FLOPs
 * - Activation memory estimates
 * - Architecture constraints
 * - Resource budgets
 * - Residual and dense connectivity
 *
 * Compile:
 * g++ -std=c++17 -O2 -Wall -Wextra -pedantic modern_cnn.cpp -o modern_cnn
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

class CNNException : public runtime_error {
public:
    explicit CNNException(const string& message)
        : runtime_error(message) {}
};

enum class LayerType {
    Convolution,
    Depthwise,
    Pooling,
    ResidualAdd,
    Concatenation,
    SqueezeExcitation,
    Normalization,
    Dense
};

struct Tensor {
    int channels;
    int height;
    int width;

    Tensor(int c, int h, int w)
        : channels(c), height(h), width(w) {
        if (c <= 0 || h <= 0 || w <= 0) {
            throw CNNException("Tensor dimensions must be positive.");
        }
    }

    long long elements() const {
        return 1LL * channels * height * width;
    }

    bool operator==(const Tensor& other) const {
        return channels == other.channels &&
               height == other.height &&
               width == other.width;
    }

    bool operator!=(const Tensor& other) const {
        return !(*this == other);
    }
};

struct Layer {
    string name;
    LayerType type;
    Tensor input;
    Tensor output;
    long long parameters;
    long long flops;
};

struct Architecture {
    string name;
    string family;
    Tensor input;
    vector<Layer> layers;

    long long parameters() const {
        long long result = 0;

        for (const auto& layer : layers) {
            result += layer.parameters;
        }

        return result;
    }

    long long flops() const {
        long long result = 0;

        for (const auto& layer : layers) {
            result += layer.flops;
        }

        return result;
    }

    Tensor output() const {
        return layers.empty() ? input : layers.back().output;
    }

    long long estimatedActivationBytes() const {
        long long maximum = input.elements();

        for (const auto& layer : layers) {
            maximum = max(maximum, layer.output.elements());
        }

        // Float32 activation storage for one tensor.
        return maximum * sizeof(float);
    }
};

int convolutionDimension(
    int input,
    int kernel,
    int stride,
    int padding
) {
    if (input <= 0 || kernel <= 0 || stride <= 0 || padding < 0) {
        throw CNNException("Invalid convolution dimensions.");
    }

    int output = (input + 2 * padding - kernel) / stride + 1;

    if (output <= 0) {
        throw CNNException("Invalid convolution output.");
    }

    return output;
}

class CNNBuilder {
private:
    Architecture model;
    Tensor current;

    void addLayer(
        const string& name,
        LayerType type,
        const Tensor& input,
        const Tensor& output,
        long long parameters,
        long long flops
    ) {
        if (parameters < 0 || flops < 0) {
            throw CNNException("Negative computational cost.");
        }

        model.layers.push_back({
            name, type, input, output, parameters, flops
        });

        current = output;
    }

public:
    CNNBuilder(
        const string& name,
        const string& family,
        const Tensor& input
    ) : model{name, family, input, {}}, current(input) {}

    const Tensor& shape() const {
        return current;
    }

    void convolution(
        const string& name,
        int outputChannels,
        int kernel,
        int stride = 1,
        int groups = 1
    ) {
        if (outputChannels <= 0 || groups <= 0) {
            throw CNNException("Invalid channel configuration.");
        }

        if (current.channels % groups != 0 ||
            outputChannels % groups != 0) {
            throw CNNException("Invalid grouped convolution.");
        }

        int padding = kernel / 2;

        int h = convolutionDimension(
            current.height, kernel, stride, padding
        );

        int w = convolutionDimension(
            current.width, kernel, stride, padding
        );

        Tensor output(outputChannels, h, w);

        long long parameters =
            1LL * outputChannels *
            (current.channels / groups) *
            kernel * kernel;

        long long flops =
            2LL * output.elements() *
            (current.channels / groups) *
            kernel * kernel;

        addLayer(
            name,
            groups == current.channels
                ? LayerType::Depthwise
                : LayerType::Convolution,
            current,
            output,
            parameters,
            flops
        );
    }

    void normalization(const string& name) {
        Tensor input = current;

        addLayer(
            name,
            LayerType::Normalization,
            input,
            input,
            2LL * input.channels,
            4LL * input.elements()
        );
    }

    void activation(const string& name) {
        Tensor input = current;

        addLayer(
            name,
            LayerType::Normalization,
            input,
            input,
            0,
            input.elements()
        );
    }

    void pooling(
        const string& name,
        int kernel = 2,
        int stride = 2
    ) {
        int h = convolutionDimension(
            current.height, kernel, stride, 0
        );

        int w = convolutionDimension(
            current.width, kernel, stride, 0
        );

        Tensor output(current.channels, h, w);

        addLayer(
            name,
            LayerType::Pooling,
            current,
            output,
            0,
            output.elements() * kernel * kernel
        );
    }

    void residualAdd(
        const string& name,
        const Tensor& shortcut
    ) {
        if (current != shortcut) {
            throw CNNException(
                "Residual addition requires identical tensor shapes."
            );
        }

        addLayer(
            name,
            LayerType::ResidualAdd,
            current,
            current,
            0,
            current.elements()
        );
    }

    void concatenate(
        const string& name,
        const Tensor& previous,
        const Tensor& generated
    ) {
        if (previous.height != generated.height ||
            previous.width != generated.width) {
            throw CNNException(
                "Dense concatenation requires equal spatial dimensions."
            );
        }

        Tensor output(
            previous.channels + generated.channels,
            previous.height,
            previous.width
        );

        addLayer(
            name,
            LayerType::Concatenation,
            previous,
            output,
            0,
            output.elements()
        );
    }

    void squeezeExcitation(const string& name, int reduction = 4) {
        if (reduction <= 0) {
            throw CNNException("Invalid SE reduction.");
        }

        int channels = current.channels;
        int reduced = max(1, channels / reduction);

        long long parameters =
            1LL * channels * reduced + reduced +
            1LL * reduced * channels + channels;

        long long flops =
            4LL * channels * reduced + current.elements();

        addLayer(
            name,
            LayerType::SqueezeExcitation,
            current,
            current,
            parameters,
            flops
        );
    }

    void globalAveragePooling(const string& name) {
        Tensor output(current.channels, 1, 1);

        addLayer(
            name,
            LayerType::Pooling,
            current,
            output,
            0,
            current.elements()
        );
    }

    void classifier(const string& name, int classes) {
        if (classes <= 0) {
            throw CNNException("Class count must be positive.");
        }

        Tensor output(classes, 1, 1);

        long long parameters =
            1LL * current.channels * classes + classes;

        long long flops =
            2LL * current.channels * classes;

        addLayer(
            name,
            LayerType::Dense,
            current,
            output,
            parameters,
            flops
        );
    }

    Architecture build() const {
        return model;
    }
};

void residualBlock(
    CNNBuilder& builder,
    const string& name,
    int channels,
    int stride
) {
    Tensor shortcut = builder.shape();

    builder.convolution(name + ".conv1", channels, 3, stride);
    builder.normalization(name + ".bn1");
    builder.activation(name + ".relu1");

    builder.convolution(name + ".conv2", channels, 3);
    builder.normalization(name + ".bn2");

    if (builder.shape() != shortcut) {
        // Projection aligns channel and spatial dimensions.
        builder.convolution(name + ".projection", channels, 1, stride);
    }

    builder.residualAdd(name + ".add", shortcut);
    builder.activation(name + ".output");
}

Architecture buildResNet18(int size = 224, int classes = 1000) {
    CNNBuilder builder(
        "ResNet-18",
        "Residual Network",
        Tensor(3, size, size)
    );

    builder.convolution("stem", 64, 7, 2);
    builder.normalization("stem.bn");
    builder.activation("stem.relu");
    builder.pooling("stem.maxpool", 3, 2);

    vector<int> channels = {64, 128, 256, 512};

    for (int stage = 0; stage < 4; ++stage) {
        for (int block = 0; block < 2; ++block) {
            int stride = (stage > 0 && block == 0) ? 2 : 1;

            residualBlock(
                builder,
                "stage" + to_string(stage + 1) +
                    ".block" + to_string(block + 1),
                channels[stage],
                stride
            );
        }
    }

    builder.globalAveragePooling("global_average");
    builder.classifier("classification_head", classes);

    return builder.build();
}

void denseBlockLayer(
    CNNBuilder& builder,
    const string& name,
    int growthRate
) {
    Tensor previous = builder.shape();

    builder.normalization(name + ".bn1");
    builder.convolution(name + ".bottleneck", growthRate * 4, 1);
    builder.normalization(name + ".bn2");
    builder.convolution(name + ".growth", growthRate, 3);

    Tensor generated = builder.shape();

    builder.concatenate(name + ".concat", previous, generated);
}

Architecture buildDenseNet121(int size = 224, int classes = 1000) {
    CNNBuilder builder(
        "DenseNet-121",
        "Dense Connectivity",
        Tensor(3, size, size)
    );

    builder.convolution("stem", 64, 7, 2);
    builder.normalization("stem.bn");
    builder.pooling("stem.pool", 3, 2);

    vector<int> blockSizes = {6, 12, 24, 16};
    const int growthRate = 32;

    for (size_t stage = 0; stage < blockSizes.size(); ++stage) {
        for (int block = 0; block < blockSizes[stage]; ++block) {
            denseBlockLayer(
                builder,
                "stage" + to_string(stage + 1) +
                    ".layer" + to_string(block + 1),
                growthRate
            );
        }

        if (stage < 3) {
            int compressed = max(
                1,
                static_cast<int>(builder.shape().channels * 0.5)
            );

            builder.convolution(
                "transition" + to_string(stage + 1) + ".conv",
                compressed,
                1
            );

            builder.pooling(
                "transition" + to_string(stage + 1) + ".pool",
                2,
                2
            );
        }
    }

    builder.globalAveragePooling("global_average");
    builder.classifier("classification_head", classes);

    return builder.build();
}

struct EfficientScale {
    double width;
    double depth;
    int resolution;
};

int roundChannels(int channels, double multiplier) {
    double scaled = channels * multiplier;
    int divisor = 8;

    int rounded = max(
        divisor,
        static_cast<int>(
            floor((scaled + divisor / 2.0) / divisor) * divisor
        )
    );

    if (rounded < 0.9 * scaled) {
        rounded += divisor;
    }

    return rounded;
}

Architecture buildEfficientNet(
    const string& variant = "B0",
    int classes = 1000
) {
    const vector<pair<string, EfficientScale>> scales = {
        {"B0", {1.0, 1.0, 224}},
        {"B1", {1.0, 1.1, 240}},
        {"B2", {1.1, 1.2, 260}},
        {"B3", {1.2, 1.4, 300}},
        {"B4", {1.4, 1.8, 380}}
    };

    auto iterator = find_if(
        scales.begin(),
        scales.end(),
        [&](const auto& item) {
            return item.first == variant;
        }
    );

    if (iterator == scales.end()) {
        throw CNNException("Unsupported EfficientNet variant.");
    }

    EfficientScale scale = iterator->second;

    CNNBuilder builder(
        "EfficientNet-" + variant,
        "Compound-Scaled CNN",
        Tensor(3, scale.resolution, scale.resolution)
    );

    builder.convolution(
        "stem",
        roundChannels(32, scale.width),
        3,
        2
    );

    builder.normalization("stem.bn");

    struct Stage {
        int expansion;
        int channels;
        int repeats;
        int kernel;
        int stride;
    };

    vector<Stage> stages = {
        {1, 16, 1, 3, 1},
        {6, 24, 2, 3, 2},
        {6, 40, 2, 5, 2},
        {6, 80, 3, 3, 2},
        {6, 112, 3, 5, 1},
        {6, 192, 4, 5, 2},
        {6, 320, 1, 3, 1}
    };

    for (size_t stage = 0; stage < stages.size(); ++stage) {
        const auto& config = stages[stage];

        int channels = roundChannels(
            config.channels,
            scale.width
        );

        int repeats = static_cast<int>(
            ceil(config.repeats * scale.depth)
        );

        for (int block = 0; block < repeats; ++block) {
            Tensor shortcut = builder.shape();
            int expansion = shortcut.channels * config.expansion;

            if (config.expansion != 1) {
                builder.convolution(
                    "s" + to_string(stage) + ".expand",
                    expansion,
                    1
                );
            }

            int stride = block == 0 ? config.stride : 1;

            builder.convolution(
                "s" + to_string(stage) + ".depthwise",
                expansion,
                config.kernel,
                stride,
                expansion
            );

            builder.normalization("depthwise.bn");

            builder.squeezeExcitation("se");

            builder.convolution(
                "s" + to_string(stage) + ".project",
                channels,
                1
            );

            if (
                stride == 1 &&
                shortcut.channels == channels &&
                shortcut.height == builder.shape().height &&
                shortcut.width == builder.shape().width
            ) {
                builder.residualAdd("mbconv.skip", shortcut);
            }
        }
    }

    builder.convolution(
        "head",
        roundChannels(1280, scale.width),
        1
    );

    builder.globalAveragePooling("global_average");
    builder.classifier("classification_head", classes);

    return builder.build();
}

struct ResourcePolicy {
    long long maximumParameters;
    long long maximumFlops;
    long long maximumActivationBytes;
};

class EligibilityEngine {
public:
    static bool eligible(
        const Architecture& architecture,
        const ResourcePolicy& policy,
        string& reason
    ) {
        if (architecture.parameters() > policy.maximumParameters) {
            reason = "Parameter budget exceeded.";
            return false;
        }

        if (architecture.flops() > policy.maximumFlops) {
            reason = "FLOP budget exceeded.";
            return false;
        }

        if (
            architecture.estimatedActivationBytes() >
            policy.maximumActivationBytes
        ) {
            reason = "Activation memory budget exceeded.";
            return false;
        }

        reason = "Architecture satisfies resource policy.";
        return true;
    }
};

void printArchitecture(const Architecture& model) {
    cout << left << setw(20) << model.name
         << right
         << " Parameters: " << setw(12) << model.parameters()
         << " FLOPs: " << setw(14) << model.flops()
         << " Layers: " << setw(4) << model.layers.size()
         << " Output: "
         << model.output().channels << "x"
         << model.output().height << "x"
         << model.output().width
         << '\n';
}

void runTests() {
    if (convolutionDimension(224, 7, 2, 3) != 112) {
        throw CNNException("Convolution test failed.");
    }

    Architecture resnet = buildResNet18(64, 10);
    Architecture dense = buildDenseNet121(64, 10);
    Architecture efficient = buildEfficientNet("B0", 10);

    if (resnet.output().channels != 10 ||
        dense.output().channels != 10 ||
        efficient.output().channels != 10) {
        throw CNNException("Classification output test failed.");
    }

    if (resnet.parameters() <= 0 ||
        dense.parameters() <= 0 ||
        efficient.parameters() <= 0) {
        throw CNNException("Invalid parameter count.");
    }

    try {
        Tensor a(64, 16, 16);
        Tensor b(32, 16, 16);

        if (a == b) {
            throw CNNException("Tensor equality test failed.");
        }

        throw CNNException("Expected residual mismatch test.");
    } catch (const CNNException&) {
        // Invalid shape combinations must be rejected.
    }

    cout << "All C++17 tests passed.\n";
}

int main() {
    try {
        runTests();

        vector<Architecture> models;

        models.push_back(buildResNet18());
        models.push_back(buildDenseNet121());
        models.push_back(buildEfficientNet("B0"));
        models.push_back(buildEfficientNet("B3"));

        cout << "\nMODERN CNN ARCHITECTURE REPORT\n";
        cout << string(115, '=') << '\n';

        for (const auto& model : models) {
            printArchitecture(model);
        }

        ResourcePolicy policy{
            100'000'000,
            100'000'000'000LL,
            100LL * 1024 * 1024
        };

        cout << "\nRESOURCE ELIGIBILITY\n";

        for (const auto& model : models) {
            string reason;

            bool allowed = EligibilityEngine::eligible(
                model,
                policy,
                reason
            );

            cout << model.name << ": "
                 << (allowed ? "ELIGIBLE" : "REJECTED")
                 << " - " << reason << '\n';
        }

        cout << "\nPARAMETER-ORDERED MODELS\n";

        sort(
            models.begin(),
            models.end(),
            [](const Architecture& a, const Architecture& b) {
                return a.parameters() < b.parameters();
            }
        );

        for (const auto& model : models) {
            cout << model.name
                 << " : "
                 << fixed << setprecision(3)
                 << model.parameters() / 1e6
                 << " million parameters\n";
        }

        return 0;
    } catch (const exception& error) {
        cerr << "Fatal error: " << error.what() << '\n';
        return 1;
    }
}
