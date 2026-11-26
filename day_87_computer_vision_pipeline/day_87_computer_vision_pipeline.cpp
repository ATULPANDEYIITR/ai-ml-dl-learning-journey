#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace std;

class PipelineError : public runtime_error {
public:
    explicit PipelineError(const string& message)
        : runtime_error(message) {}
};

struct Pixel {
    uint16_t r, g, b;
};

struct Image {
    int width;
    int height;
    int maxValue;
    vector<Pixel> pixels;

    Image(int w, int h, int maximum, vector<Pixel> data)
        : width(w), height(h), maxValue(maximum),
          pixels(move(data)) {
        validate();
    }

    void validate() const {
        if (width <= 0 || height <= 0 ||
            width > 4096 || height > 4096) {
            throw PipelineError("Invalid image dimensions.");
        }

        if (maxValue < 1 || maxValue > 65535) {
            throw PipelineError("Invalid maximum pixel value.");
        }

        if (pixels.size() !=
            static_cast<size_t>(width) * height) {
            throw PipelineError("Pixel count does not match dimensions.");
        }

        for (const auto& p : pixels) {
            if (p.r > maxValue ||
                p.g > maxValue ||
                p.b > maxValue) {
                throw PipelineError("Pixel channel outside permitted range.");
            }
        }
    }

    Pixel& at(int x, int y) {
        if (x < 0 || x >= width || y < 0 || y >= height) {
            throw PipelineError("Pixel coordinate outside image.");
        }

        return pixels[static_cast<size_t>(y) * width + x];
    }

    const Pixel& at(int x, int y) const {
        if (x < 0 || x >= width || y < 0 || y >= height) {
            throw PipelineError("Pixel coordinate outside image.");
        }

        return pixels[static_cast<size_t>(y) * width + x];
    }
};

class PPMCodec {
public:
    static Image read(const string& filePath) {
        ifstream input(filePath);

        if (!input) {
            throw PipelineError("Cannot open input image.");
        }

        string magic;
        input >> magic;

        if (magic != "P3") {
            throw PipelineError("Only P3 PPM is supported.");
        }

        int width, height, maximum;

        if (!(input >> width >> height >> maximum)) {
            throw PipelineError("Invalid PPM header.");
        }

        if (width <= 0 || height <= 0 ||
            width > 4096 || height > 4096 ||
            maximum < 1 || maximum > 65535) {
            throw PipelineError("Invalid PPM dimensions or maximum.");
        }

        vector<Pixel> pixels;
        pixels.reserve(static_cast<size_t>(width) * height);

        for (int i = 0; i < width * height; ++i) {
            int r, g, b;

            if (!(input >> r >> g >> b)) {
                throw PipelineError("Incomplete PPM pixel data.");
            }

            if (r < 0 || r > maximum ||
                g < 0 || g > maximum ||
                b < 0 || b > maximum) {
                throw PipelineError("Invalid pixel channel.");
            }

            pixels.push_back({
                static_cast<uint16_t>(r),
                static_cast<uint16_t>(g),
                static_cast<uint16_t>(b)
            });
        }

        string extra;

        if (input >> extra) {
            throw PipelineError("Unexpected data after PPM pixels.");
        }

        return Image(width, height, maximum, move(pixels));
    }

    static void write(const Image& image, const string& filePath) {
        image.validate();

        ofstream output(filePath);

        if (!output) {
            throw PipelineError("Cannot create output image.");
        }

        output << "P3\n"
               << image.width << " "
               << image.height << "\n"
               << image.maxValue << "\n";

        for (const auto& pixel : image.pixels) {
            output << pixel.r << " "
                   << pixel.g << " "
                   << pixel.b << "\n";
        }

        if (!output) {
            throw PipelineError("Failed writing image.");
        }
    }
};

class ImagePreprocessor {
public:
    static Image resize(
        const Image& source,
        int targetWidth,
        int targetHeight
    ) {
        source.validate();

        if (targetWidth <= 0 || targetHeight <= 0 ||
            targetWidth > 4096 || targetHeight > 4096) {
            throw PipelineError("Invalid target dimensions.");
        }

        vector<Pixel> output;
        output.reserve(
            static_cast<size_t>(targetWidth) * targetHeight
        );

        for (int y = 0; y < targetHeight; ++y) {
            int sourceY = min(
                source.height - 1,
                y * source.height / targetHeight
            );

            for (int x = 0; x < targetWidth; ++x) {
                int sourceX = min(
                    source.width - 1,
                    x * source.width / targetWidth
                );

                output.push_back(source.at(sourceX, sourceY));
            }
        }

        return Image(
            targetWidth,
            targetHeight,
            source.maxValue,
            move(output)
        );
    }

    static vector<array<double, 3>> normalize(
        const Image& image,
        bool minusOneToOne = false
    ) {
        vector<array<double, 3>> result;
        result.reserve(image.pixels.size());

        for (const auto& pixel : image.pixels) {
            array<double, 3> values = {
                static_cast<double>(pixel.r) / image.maxValue,
                static_cast<double>(pixel.g) / image.maxValue,
                static_cast<double>(pixel.b) / image.maxValue
            };

            if (minusOneToOne) {
                for (double& value : values) {
                    value = value * 2.0 - 1.0;
                }
            }

            result.push_back(values);
        }

        return result;
    }

    static vector<double> grayscale(const Image& image) {
        vector<double> result;
        result.reserve(image.pixels.size());

        for (const auto& pixel : image.pixels) {
            result.push_back(
                (
                    0.2126 * pixel.r +
                    0.7152 * pixel.g +
                    0.0722 * pixel.b
                ) / image.maxValue
            );
        }

        return result;
    }

    static vector<double> standardize(
        const vector<double>& values
    ) {
        if (values.empty()) {
            throw PipelineError("Cannot standardize empty data.");
        }

        double mean = accumulate(
            values.begin(),
            values.end(),
            0.0
        ) / values.size();

        double variance = 0.0;

        for (double value : values) {
            if (!isfinite(value)) {
                throw PipelineError("Non-finite feature value.");
            }

            variance += (value - mean) * (value - mean);
        }

        variance /= values.size();

        double deviation = sqrt(variance);

        vector<double> result;

        for (double value : values) {
            result.push_back(
                deviation == 0.0
                    ? 0.0
                    : (value - mean) / deviation
            );
        }

        return result;
    }
};

class ImageAugmentor {
private:
    mt19937 generator;

public:
    explicit ImageAugmentor(uint32_t seed = 42)
        : generator(seed) {}

    Image horizontalFlip(const Image& image) const {
        vector<Pixel> output;

        for (int y = 0; y < image.height; ++y) {
            for (int x = image.width - 1; x >= 0; --x) {
                output.push_back(image.at(x, y));
            }
        }

        return Image(
            image.width,
            image.height,
            image.maxValue,
            move(output)
        );
    }

    Image verticalFlip(const Image& image) const {
        vector<Pixel> output;

        for (int y = image.height - 1; y >= 0; --y) {
            for (int x = 0; x < image.width; ++x) {
                output.push_back(image.at(x, y));
            }
        }

        return Image(
            image.width,
            image.height,
            image.maxValue,
            move(output)
        );
    }

    Image rotate90(const Image& image) const {
        vector<Pixel> output;
        output.reserve(image.pixels.size());

        for (int x = 0; x < image.width; ++x) {
            for (int y = image.height - 1; y >= 0; --y) {
                output.push_back(image.at(x, y));
            }
        }

        return Image(
            image.height,
            image.width,
            image.maxValue,
            move(output)
        );
    }

    Image brightness(
        const Image& image,
        double factor
    ) const {
        if (!isfinite(factor) || factor < 0.0 || factor > 3.0) {
            throw PipelineError("Invalid brightness factor.");
        }

        vector<Pixel> output;

        for (const auto& pixel : image.pixels) {
            auto adjust = [&](uint16_t channel) {
                return static_cast<uint16_t>(
                    clamp(
                        lround(channel * factor),
                        0L,
                        static_cast<long>(image.maxValue)
                    )
                );
            };

            output.push_back({
                adjust(pixel.r),
                adjust(pixel.g),
                adjust(pixel.b)
            });
        }

        return Image(
            image.width,
            image.height,
            image.maxValue,
            move(output)
        );
    }

    Image randomTransform(const Image& image) {
        bernoulli_distribution flip(0.5);
        uniform_real_distribution<double> brightnessRange(0.8, 1.2);

        Image result = image;

        if (flip(generator)) {
            result = horizontalFlip(result);
        }

        if (flip(generator)) {
            result = verticalFlip(result);
        }

        return brightness(
            result,
            brightnessRange(generator)
        );
    }
};

struct FeatureVector {
    array<double, 9> values{};

    static const array<string, 9>& names() {
        static const array<string, 9> featureNames = {
            "red_mean",
            "green_mean",
            "blue_mean",
            "red_std",
            "green_std",
            "blue_std",
            "red_ratio",
            "green_ratio",
            "blue_ratio"
        };

        return featureNames;
    }
};

class FeatureExtractor {
public:
    static FeatureVector extract(const Image& image) {
        image.validate();

        FeatureVector features;
        array<double, 3> sums{};
        array<double, 3> squares{};

        for (const auto& pixel : image.pixels) {
            array<double, 3> values = {
                static_cast<double>(pixel.r) / image.maxValue,
                static_cast<double>(pixel.g) / image.maxValue,
                static_cast<double>(pixel.b) / image.maxValue
            };

            for (size_t i = 0; i < 3; ++i) {
                sums[i] += values[i];
                squares[i] += values[i] * values[i];
            }
        }

        double total = static_cast<double>(image.pixels.size());
        double sumMeans = 0.0;

        for (size_t i = 0; i < 3; ++i) {
            double mean = sums[i] / total;

            double deviation = sqrt(
                max(
                    0.0,
                    squares[i] / total - mean * mean
                )
            );

            features.values[i] = mean;
            features.values[i + 3] = deviation;

            sumMeans += mean;
        }

        sumMeans = max(sumMeans, 1e-12);

        for (size_t i = 0; i < 3; ++i) {
            features.values[i + 6] =
                features.values[i] / sumMeans;
        }

        return features;
    }
};

struct TrainingSample {
    FeatureVector features;
    string label;
};

struct Prediction {
    string label;
    double confidence;
    map<string, double> probabilities;
};

class NearestCentroidClassifier {
private:
    map<string, array<double, 9>> centroids;
    map<string, size_t> counts;

public:
    void fit(const vector<TrainingSample>& samples) {
        if (samples.empty()) {
            throw PipelineError("Training dataset is empty.");
        }

        centroids.clear();
        counts.clear();

        for (const auto& sample : samples) {
            if (sample.label.empty()) {
                throw PipelineError("Training label cannot be empty.");
            }

            auto& centroid = centroids[sample.label];

            for (size_t i = 0; i < 9; ++i) {
                if (!isfinite(sample.features.values[i])) {
                    throw PipelineError("Non-finite training feature.");
                }

                centroid[i] += sample.features.values[i];
            }

            counts[sample.label]++;
        }

        for (auto& [label, centroid] : centroids) {
            for (double& value : centroid) {
                value /= counts[label];
            }
        }
    }

    Prediction predict(const FeatureVector& features) const {
        if (centroids.empty()) {
            throw PipelineError("Classifier has not been trained.");
        }

        for (double value : features.values) {
            if (!isfinite(value)) {
                throw PipelineError("Invalid prediction feature.");
            }
        }

        map<string, double> distances;
        double minimum = numeric_limits<double>::max();

        for (const auto& [label, centroid] : centroids) {
            double squaredDistance = 0.0;

            for (size_t i = 0; i < 9; ++i) {
                double difference =
                    features.values[i] - centroid[i];

                squaredDistance += difference * difference;
            }

            double distance = sqrt(squaredDistance);

            distances[label] = distance;
            minimum = min(minimum, distance);
        }

        map<string, double> weights;
        double totalWeight = 0.0;

        for (const auto& [label, distance] : distances) {
            double weight = exp(-(distance - minimum));

            weights[label] = weight;
            totalWeight += weight;
        }

        Prediction result;
        result.confidence = -1.0;

        for (const auto& [label, weight] : weights) {
            double probability = weight / totalWeight;

            result.probabilities[label] = probability;

            if (probability > result.confidence) {
                result.label = label;
                result.confidence = probability;
            }
        }

        return result;
    }

    void save(const string& filePath) const {
        if (centroids.empty()) {
            throw PipelineError("Cannot save an untrained classifier.");
        }

        ofstream output(filePath);

        if (!output) {
            throw PipelineError("Cannot create model file.");
        }

        output << setprecision(17);
        output << centroids.size() << "\n";

        for (const auto& [label, centroid] : centroids) {
            output << quoted(label) << "\n";

            for (double value : centroid) {
                output << value << " ";
            }

            output << "\n";
        }
    }

    void load(const string& filePath) {
        ifstream input(filePath);

        if (!input) {
            throw PipelineError("Cannot open model file.");
        }

        size_t count;

        if (!(input >> count) || count == 0 || count > 1000) {
            throw PipelineError("Invalid model class count.");
        }

        map<string, array<double, 9>> loaded;

        for (size_t i = 0; i < count; ++i) {
            string label;
            array<double, 9> centroid{};

            if (!(input >> quoted(label)) || label.empty()) {
                throw PipelineError("Invalid model label.");
            }

            for (double& value : centroid) {
                if (!(input >> value) || !isfinite(value)) {
                    throw PipelineError("Invalid model centroid.");
                }
            }

            if (!loaded.emplace(label, centroid).second) {
                throw PipelineError("Duplicate model class.");
            }
        }

        string extra;

        if (input >> extra) {
            throw PipelineError("Unexpected model data.");
        }

        centroids = move(loaded);
        counts.clear();
    }
};

class ComputerVisionPipeline {
private:
    int targetWidth;
    int targetHeight;

    ImageAugmentor augmentor;
    NearestCentroidClassifier classifier;

public:
    ComputerVisionPipeline(
        int width = 32,
        int height = 32,
        uint32_t seed = 42
    )
        : targetWidth(width),
          targetHeight(height),
          augmentor(seed) {
        if (width <= 0 || height <= 0 ||
            width > 4096 || height > 4096) {
            throw PipelineError("Invalid target dimensions.");
        }
    }

    Image prepare(const Image& image) const {
        return ImagePreprocessor::resize(
            image,
            targetWidth,
            targetHeight
        );
    }

    void train(
        const vector<pair<Image, string>>& dataset,
        bool useAugmentation = true
    ) {
        if (dataset.empty()) {
            throw PipelineError("Training dataset is empty.");
        }

        vector<TrainingSample> samples;

        for (const auto& [image, label] : dataset) {
            if (label.empty()) {
                throw PipelineError("Empty dataset label.");
            }

            Image processed = prepare(image);

            samples.push_back({
                FeatureExtractor::extract(processed),
                label
            });

            if (useAugmentation) {
                Image augmented = augmentor.randomTransform(processed);

                samples.push_back({
                    FeatureExtractor::extract(augmented),
                    label
                });
            }
        }

        classifier.fit(samples);
    }

    Prediction predict(const Image& image) const {
        Image processed = prepare(image);

        return classifier.predict(
            FeatureExtractor::extract(processed)
        );
    }

    void saveModel(const string& filePath) const {
        classifier.save(filePath);
    }

    void loadModel(const string& filePath) {
        classifier.load(filePath);
    }

    static Image syntheticImage(
        Pixel color,
        int width = 16,
        int height = 16
    ) {
        vector<Pixel> pixels;

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int variation = (x + y) % 12;

                auto adjust = [&](uint16_t channel) {
                    return static_cast<uint16_t>(
                        min(255, static_cast<int>(channel) + variation)
                    );
                };

                pixels.push_back({
                    adjust(color.r),
                    adjust(color.g),
                    adjust(color.b)
                });
            }
        }

        return Image(width, height, 255, move(pixels));
    }
};

int main(int argc, char* argv[]) {
    try {
        if (argc >= 3 && string(argv[1]) == "--predict") {
            if (argc < 4) {
                throw PipelineError(
                    "Usage: vision --predict image.ppm model.txt"
                );
            }

            Image image = PPMCodec::read(argv[2]);

            ComputerVisionPipeline pipeline;
            pipeline.loadModel(argv[3]);

            Prediction result = pipeline.predict(image);

            cout << "\nPredicted class: " << result.label << "\n";
            cout << "Confidence: "
                 << fixed << setprecision(4)
                 << result.confidence << "\n";

            for (const auto& [label, probability] :
                 result.probabilities) {
                cout << label << ": " << probability << "\n";
            }

            return 0;
        }

        ComputerVisionPipeline pipeline(16, 16, 42);

        vector<pair<Image, string>> trainingData = {
            {ComputerVisionPipeline::syntheticImage({230, 20, 20}), "red"},
            {ComputerVisionPipeline::syntheticImage({210, 30, 25}), "red"},
            {ComputerVisionPipeline::syntheticImage({20, 220, 20}), "green"},
            {ComputerVisionPipeline::syntheticImage({25, 200, 30}), "green"},
            {ComputerVisionPipeline::syntheticImage({20, 20, 230}), "blue"},
            {ComputerVisionPipeline::syntheticImage({25, 30, 210}), "blue"}
        };

        pipeline.train(trainingData, true);

        vector<pair<Image, string>> tests = {
            {ComputerVisionPipeline::syntheticImage({225, 25, 25}), "red"},
            {ComputerVisionPipeline::syntheticImage({25, 215, 25}), "green"},
            {ComputerVisionPipeline::syntheticImage({25, 25, 220}), "blue"}
        };

        cout << "\nCOMPUTER VISION CLASSIFICATION\n";

        for (const auto& [image, expected] : tests) {
            Prediction result = pipeline.predict(image);

            cout << "\nExpected: " << expected << "\n";
            cout << "Predicted: " << result.label << "\n";
            cout << "Confidence: "
                 << fixed << setprecision(4)
                 << result.confidence << "\n";

            if (result.label != expected) {
                throw PipelineError("Classification test failed.");
            }
        }

        Image original =
            ComputerVisionPipeline::syntheticImage({200, 80, 40});

        Image resized = pipeline.prepare(original);

        ImageAugmentor augmentor(42);

        Image flipped = augmentor.horizontalFlip(resized);
        Image rotated = augmentor.rotate90(resized);

        if (flipped.width != resized.width ||
            rotated.width != resized.height ||
            rotated.height != resized.width) {
            throw PipelineError("Image transformation test failed.");
        }

        PPMCodec::write(resized, "processed_image.ppm");

        pipeline.saveModel("vision_model.txt");

        ComputerVisionPipeline restored;
        restored.loadModel("vision_model.txt");

        Prediction restoredResult = restored.predict(
            ComputerVisionPipeline::syntheticImage({225, 25, 25})
        );

        if (restoredResult.label != "red") {
            throw PipelineError("Model persistence test failed.");
        }

        cout << "\nPreprocessing tests: PASSED\n";
        cout << "Augmentation tests: PASSED\n";
        cout << "Classification tests: PASSED\n";
        cout << "Model persistence tests: PASSED\n";

        cout << "\nOutput files:\n";
        cout << "processed_image.ppm\n";
        cout << "vision_model.txt\n";

        return 0;

    } catch (const exception& error) {
        cerr << "Pipeline error: " << error.what() << "\n";
        return 1;
    }
}
