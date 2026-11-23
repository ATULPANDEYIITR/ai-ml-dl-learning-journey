#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

enum class Padding {
    VALID,
    SAME
};

struct Tensor {
    int channels;
    int height;
    int width;
    vector<double> data;

    Tensor(int c, int h, int w)
        : channels(c), height(h), width(w) {

        if (c <= 0 || h <= 0 || w <= 0) {
            throw invalid_argument("Tensor dimensions must be positive.");
        }

        data.assign(
            static_cast<size_t>(c) * h * w,
            0.0
        );
    }

    double& at(int c, int r, int col) {
        return data[
            static_cast<size_t>(c) * height * width +
            static_cast<size_t>(r) * width +
            col
        ];
    }

    double at(int c, int r, int col) const {
        return data[
            static_cast<size_t>(c) * height * width +
            static_cast<size_t>(r) * width +
            col
        ];
    }

    void validate() const {
        if (
            channels <= 0 ||
            height <= 0 ||
            width <= 0 ||
            data.size() != static_cast<size_t>(channels) * height * width
        ) {
            throw invalid_argument("Invalid tensor structure.");
        }

        for (double value : data) {
            if (!isfinite(value)) {
                throw invalid_argument("Tensor contains non-finite values.");
            }
        }
    }
};


struct Kernel {
    int inputChannels;
    int height;
    int width;
    vector<double> weights;
    double bias;

    Kernel(int channels, int h, int w, double b = 0.0)
        : inputChannels(channels),
          height(h),
          width(w),
          bias(b) {

        if (channels <= 0 || h <= 0 || w <= 0) {
            throw invalid_argument("Invalid kernel dimensions.");
        }

        if (!isfinite(bias)) {
            throw invalid_argument("Kernel bias must be finite.");
        }

        weights.assign(
            static_cast<size_t>(channels) * h * w,
            0.0
        );
    }

    double& at(int c, int r, int col) {
        return weights[
            static_cast<size_t>(c) * height * width +
            static_cast<size_t>(r) * width +
            col
        ];
    }

    double at(int c, int r, int col) const {
        return weights[
            static_cast<size_t>(c) * height * width +
            static_cast<size_t>(r) * width +
            col
        ];
    }

    void validate() const {
        if (
            weights.size() !=
            static_cast<size_t>(inputChannels) * height * width
        ) {
            throw invalid_argument("Invalid kernel weight storage.");
        }

        for (double value : weights) {
            if (!isfinite(value)) {
                throw invalid_argument("Kernel contains non-finite weights.");
            }
        }
    }
};


struct PaddingInfo {
    int top = 0;
    int bottom = 0;
    int left = 0;
    int right = 0;
};


struct ConvResult {
    Tensor output;
    PaddingInfo padding;
    int stride;
};


class ConvolutionEngine {
private:

    static int outputDimension(
        int input,
        int kernel,
        int stride,
        Padding padding
    ) {
        if (padding == Padding::SAME) {
            return (input + stride - 1) / stride;
        }

        if (kernel > input) {
            throw invalid_argument("Kernel exceeds VALID input dimensions.");
        }

        return (input - kernel) / stride + 1;
    }

    static PaddingInfo calculatePadding(
        int height,
        int width,
        int kernelHeight,
        int kernelWidth,
        int stride,
        Padding padding
    ) {
        PaddingInfo result;

        if (padding == Padding::VALID) {
            return result;
        }

        int outputHeight = (height + stride - 1) / stride;
        int outputWidth = (width + stride - 1) / stride;

        int totalHeight = max(
            0,
            (outputHeight - 1) * stride + kernelHeight - height
        );

        int totalWidth = max(
            0,
            (outputWidth - 1) * stride + kernelWidth - width
        );

        result.top = totalHeight / 2;
        result.bottom = totalHeight - result.top;

        result.left = totalWidth / 2;
        result.right = totalWidth - result.left;

        return result;
    }

public:

    static ConvResult apply(
        const Tensor& input,
        const vector<Kernel>& filters,
        int stride,
        Padding padding
    ) {
        input.validate();

        if (stride <= 0) {
            throw invalid_argument("Stride must be positive.");
        }

        if (filters.empty()) {
            throw invalid_argument("At least one filter is required.");
        }

        const Kernel& reference = filters.front();

        reference.validate();

        for (const Kernel& kernel : filters) {
            kernel.validate();

            if (
                kernel.inputChannels != input.channels ||
                kernel.height != reference.height ||
                kernel.width != reference.width
            ) {
                throw invalid_argument("Filter dimensions do not match.");
            }
        }

        int outputHeight = outputDimension(
            input.height,
            reference.height,
            stride,
            padding
        );

        int outputWidth = outputDimension(
            input.width,
            reference.width,
            stride,
            padding
        );

        PaddingInfo pad = calculatePadding(
            input.height,
            input.width,
            reference.height,
            reference.width,
            stride,
            padding
        );

        Tensor output(
            static_cast<int>(filters.size()),
            outputHeight,
            outputWidth
        );

        /*
         * Each filter generates one output feature map.
         * Every input channel contributes to the same output map.
         */
        for (size_t f = 0; f < filters.size(); f++) {
            const Kernel& kernel = filters[f];

            for (int outRow = 0; outRow < outputHeight; outRow++) {
                for (int outCol = 0; outCol < outputWidth; outCol++) {

                    double sum = kernel.bias;

                    for (int c = 0; c < input.channels; c++) {
                        for (int kr = 0; kr < kernel.height; kr++) {
                            for (int kc = 0; kc < kernel.width; kc++) {

                                int inputRow =
                                    outRow * stride + kr - pad.top;

                                int inputCol =
                                    outCol * stride + kc - pad.left;

                                double inputValue = 0.0;

                                if (
                                    inputRow >= 0 &&
                                    inputRow < input.height &&
                                    inputCol >= 0 &&
                                    inputCol < input.width
                                ) {
                                    inputValue = input.at(
                                        c,
                                        inputRow,
                                        inputCol
                                    );
                                }

                                sum += inputValue * kernel.at(c, kr, kc);
                            }
                        }
                    }

                    output.at(
                        static_cast<int>(f),
                        outRow,
                        outCol
                    ) = sum;
                }
            }
        }

        return {
            move(output),
            pad,
            stride
        };
    }
};


Tensor applyReLU(const Tensor& input) {
    input.validate();

    Tensor output = input;

    for (double& value : output.data) {
        value = max(0.0, value);
    }

    return output;
}


Tensor applyMaxPooling(
    const Tensor& input,
    int poolSize,
    int stride
) {
    input.validate();

    if (poolSize <= 0 || stride <= 0) {
        throw invalid_argument("Pooling parameters must be positive.");
    }

    if (
        poolSize > input.height ||
        poolSize > input.width
    ) {
        throw invalid_argument("Pooling window exceeds input dimensions.");
    }

    int outputHeight =
        (input.height - poolSize) / stride + 1;

    int outputWidth =
        (input.width - poolSize) / stride + 1;

    Tensor output(
        input.channels,
        outputHeight,
        outputWidth
    );

    for (int c = 0; c < input.channels; c++) {
        for (int r = 0; r < outputHeight; r++) {
            for (int col = 0; col < outputWidth; col++) {

                double maximum = numeric_limits<double>::lowest();

                for (int kr = 0; kr < poolSize; kr++) {
                    for (int kc = 0; kc < poolSize; kc++) {

                        maximum = max(
                            maximum,
                            input.at(
                                c,
                                r * stride + kr,
                                col * stride + kc
                            )
                        );
                    }
                }

                output.at(c, r, col) = maximum;
            }
        }
    }

    return output;
}


size_t countParameters(const vector<Kernel>& filters) {
    size_t total = 0;

    for (const Kernel& filter : filters) {
        total += filter.weights.size() + 1;
    }

    return total;
}


void printTensor(const string& name, const Tensor& tensor) {
    cout << "\n" << name << "\n";

    for (int c = 0; c < tensor.channels; c++) {
        cout << "Feature map " << c + 1 << ":\n";

        for (int r = 0; r < tensor.height; r++) {
            for (int col = 0; col < tensor.width; col++) {
                cout << setw(9)
                     << fixed
                     << setprecision(2)
                     << tensor.at(c, r, col);
            }

            cout << "\n";
        }
    }
}


void printConfiguration(
    const Tensor& input,
    const ConvResult& result
) {
    cout << "Input shape: "
         << input.channels << " x "
         << input.height << " x "
         << input.width << "\n";

    cout << "Output shape: "
         << result.output.channels << " x "
         << result.output.height << " x "
         << result.output.width << "\n";

    cout << "Stride: " << result.stride << "\n";

    cout << "Padding (top, bottom, left, right): "
         << result.padding.top << ", "
         << result.padding.bottom << ", "
         << result.padding.left << ", "
         << result.padding.right << "\n";
}


void demonstrateConvolution() {
    cout << "\n1. BASIC CONVOLUTION\n";

    Tensor image(1, 5, 5);

    double values[5][5] = {
        {1, 1, 1, 0, 0},
        {1, 1, 1, 0, 0},
        {1, 1, 1, 0, 0},
        {0, 0, 0, 1, 1},
        {0, 0, 0, 1, 1}
    };

    for (int r = 0; r < 5; r++) {
        for (int c = 0; c < 5; c++) {
            image.at(0, r, c) = values[r][c];
        }
    }

    Kernel edge(1, 3, 3);

    double weights[3][3] = {
        {-1, 0, 1},
        {-1, 0, 1},
        {-1, 0, 1}
    };

    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            edge.at(0, r, c) = weights[r][c];
        }
    }

    auto result = ConvolutionEngine::apply(
        image,
        {edge},
        1,
        Padding::VALID
    );

    printConfiguration(image, result);
    printTensor("Vertical edge detection", result.output);
}


void demonstrateStrideAndPadding() {
    cout << "\n2. STRIDE AND PADDING\n";

    Tensor image(1, 5, 5);

    for (int r = 0; r < 5; r++) {
        for (int c = 0; c < 5; c++) {
            image.at(0, r, c) = r * 5 + c + 1;
        }
    }

    Kernel kernel(1, 3, 3);

    double values[3][3] = {
        {1, 0, -1},
        {1, 0, -1},
        {1, 0, -1}
    };

    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            kernel.at(0, r, c) = values[r][c];
        }
    }

    for (Padding padding : {Padding::VALID, Padding::SAME}) {
        for (int stride : {1, 2}) {

            auto result = ConvolutionEngine::apply(
                image,
                {kernel},
                stride,
                padding
            );

            cout << "\nMode: "
                 << (padding == Padding::VALID ? "VALID" : "SAME")
                 << ", stride: " << stride << "\n";

            printConfiguration(image, result);
            printTensor("Feature map", result.output);
        }
    }
}


void demonstrateMultiChannel() {
    cout << "\n3. MULTI-CHANNEL CONVOLUTION\n";

    Tensor rgb(3, 4, 4);

    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            rgb.at(0, r, c) = r * 4 + c + 1;
            rgb.at(1, r, c) = 16 - (r * 4 + c);
            rgb.at(2, r, c) = (r + c) % 3;
        }
    }

    Kernel first(3, 2, 2, 0.5);
    Kernel second(3, 2, 2, -1.0);

    double firstWeights[3][2][2] = {
        {{1, 0}, {0, -1}},
        {{0, 1}, {-1, 0}},
        {{1, 1}, {1, 1}}
    };

    double secondWeights[3][2][2] = {
        {{-1, 0}, {0, 1}},
        {{1, 1}, {0, 0}},
        {{0, -1}, {1, 0}}
    };

    for (int ch = 0; ch < 3; ch++) {
        for (int r = 0; r < 2; r++) {
            for (int c = 0; c < 2; c++) {
                first.at(ch, r, c) = firstWeights[ch][r][c];
                second.at(ch, r, c) = secondWeights[ch][r][c];
            }
        }
    }

    vector<Kernel> filters = {first, second};

    auto result = ConvolutionEngine::apply(
        rgb,
        filters,
        1,
        Padding::VALID
    );

    printConfiguration(rgb, result);

    cout << "Trainable parameters: "
         << countParameters(filters) << "\n";

    printTensor("Filter bank output", result.output);
}


void demonstratePooling() {
    cout << "\n4. ACTIVATION AND MAX POOLING\n";

    Tensor feature(1, 4, 4);

    double values[4][4] = {
        {-2, 1, 3, 4},
        {5, -1, 2, 8},
        {1, 7, -3, 2},
        {4, 6, 5, -2}
    };

    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            feature.at(0, r, c) = values[r][c];
        }
    }

    Tensor activated = applyReLU(feature);
    Tensor pooled = applyMaxPooling(activated, 2, 2);

    printTensor("Original feature map", feature);
    printTensor("After ReLU", activated);
    printTensor("After max pooling", pooled);
}


void demonstrateParameterCalculation() {
    cout << "\n5. TRAINABLE PARAMETERS\n";

    int inputChannels = 3;
    int filterCount = 16;
    int kernelHeight = 3;
    int kernelWidth = 3;

    size_t weights =
        static_cast<size_t>(inputChannels) *
        filterCount *
        kernelHeight *
        kernelWidth;

    size_t biases = filterCount;

    cout << "Weight parameters: " << weights << "\n";
    cout << "Bias parameters: " << biases << "\n";
    cout << "Total parameters: " << weights + biases << "\n";
}


void runTests() {
    cout << "\n6. VALIDATION TESTS\n";

    Tensor input(1, 2, 2);

    input.at(0, 0, 0) = 1;
    input.at(0, 0, 1) = 2;
    input.at(0, 1, 0) = 3;
    input.at(0, 1, 1) = 4;

    Kernel identity(1, 2, 2);

    identity.at(0, 0, 0) = 1;
    identity.at(0, 0, 1) = 0;
    identity.at(0, 1, 0) = 0;
    identity.at(0, 1, 1) = 1;

    auto result = ConvolutionEngine::apply(
        input,
        {identity},
        1,
        Padding::VALID
    );

    if (result.output.at(0, 0, 0) != 5) {
        throw runtime_error("Identity-kernel test failed.");
    }

    cout << "Identity kernel: PASSED\n";

    bool rejected = false;

    try {
        ConvolutionEngine::apply(
            input,
            {identity},
            0,
            Padding::VALID
        );
    } catch (const invalid_argument&) {
        rejected = true;
    }

    if (!rejected) {
        throw runtime_error("Invalid stride was not rejected.");
    }

    cout << "Invalid stride: PASSED\n";

    rejected = false;

    try {
        Kernel oversized(1, 3, 3);

        ConvolutionEngine::apply(
            input,
            {oversized},
            1,
            Padding::VALID
        );
    } catch (const invalid_argument&) {
        rejected = true;
    }

    if (!rejected) {
        throw runtime_error("Oversized kernel was not rejected.");
    }

    cout << "Oversized VALID kernel: PASSED\n";

    rejected = false;

    try {
        Tensor invalid(1, 1, 1);

        invalid.at(0, 0, 0) =
            numeric_limits<double>::quiet_NaN();

        invalid.validate();
    } catch (const invalid_argument&) {
        rejected = true;
    }

    if (!rejected) {
        throw runtime_error("Non-finite tensor was not rejected.");
    }

    cout << "Non-finite input: PASSED\n";
}


int main() {
    try {
        cout << "CNN FUNDAMENTALS: C++17\n";
        cout << "=======================\n";

        demonstrateConvolution();
        demonstrateStrideAndPadding();
        demonstrateMultiChannel();
        demonstratePooling();
        demonstrateParameterCalculation();
        runTests();

        cout << "\nAll demonstrations completed successfully.\n";

    } catch (const exception& error) {
        cerr << "CNN execution error: "
             << error.what() << "\n";

        return 1;
    }

    return 0;
}
