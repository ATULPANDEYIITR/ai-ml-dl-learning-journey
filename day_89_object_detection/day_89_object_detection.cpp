#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

class DetectionError : public runtime_error {
public:
    explicit DetectionError(const string& message)
        : runtime_error(message) {}
};

struct BoundingBox {
    double x1, y1, x2, y2;

    BoundingBox(double ax, double ay, double bx, double by)
        : x1(ax), y1(ay), x2(bx), y2(by) {

        if (!isfinite(x1) || !isfinite(y1) ||
            !isfinite(x2) || !isfinite(y2)) {
            throw DetectionError("Coordinates must be finite.");
        }

        if (x2 <= x1 || y2 <= y1) {
            throw DetectionError("Invalid bounding box dimensions.");
        }
    }

    double width() const {
        return x2 - x1;
    }

    double height() const {
        return y2 - y1;
    }

    double area() const {
        return width() * height();
    }

    double centerX() const {
        return (x1 + x2) / 2.0;
    }

    double centerY() const {
        return (y1 + y2) / 2.0;
    }

    static BoundingBox fromXYWH(
        double x,
        double y,
        double width,
        double height
    ) {
        return BoundingBox(x, y, x + width, y + height);
    }

    bool clip(double imageWidth, double imageHeight) {
        if (imageWidth <= 0 || imageHeight <= 0) {
            throw DetectionError("Invalid image dimensions.");
        }

        x1 = max(0.0, min(imageWidth, x1));
        y1 = max(0.0, min(imageHeight, y1));
        x2 = max(0.0, min(imageWidth, x2));
        y2 = max(0.0, min(imageHeight, y2));

        return x2 > x1 && y2 > y1;
    }
};

struct Detection {
    BoundingBox box;
    double confidence;
    int classId;
    string label;
    string imageId;

    Detection(
        BoundingBox b,
        double score,
        int category,
        string name,
        string image
    )
        : box(b),
          confidence(score),
          classId(category),
          label(move(name)),
          imageId(move(image)) {

        if (!isfinite(confidence) ||
            confidence < 0 || confidence > 1) {
            throw DetectionError("Confidence must be in [0, 1].");
        }

        if (classId < 0 || label.empty() || imageId.empty()) {
            throw DetectionError("Invalid detection metadata.");
        }
    }
};

enum class SuppressionMode {
    HardNMS,
    SoftNMS
};

struct DetectionConfig {
    double confidenceThreshold = 0.25;
    double iouThreshold = 0.45;
    double softSigma = 0.5;
    size_t maxDetections = 100;
    bool classAgnostic = false;
    SuppressionMode mode = SuppressionMode::HardNMS;

    void validate() const {
        if (confidenceThreshold < 0 || confidenceThreshold > 1 ||
            iouThreshold < 0 || iouThreshold > 1 ||
            !isfinite(confidenceThreshold) ||
            !isfinite(iouThreshold) ||
            softSigma <= 0 || !isfinite(softSigma) ||
            maxDetections == 0) {
            throw DetectionError("Invalid pipeline configuration.");
        }
    }
};

double calculateIoU(
    const BoundingBox& a,
    const BoundingBox& b
) {
    const double ix1 = max(a.x1, b.x1);
    const double iy1 = max(a.y1, b.y1);
    const double ix2 = min(a.x2, b.x2);
    const double iy2 = min(a.y2, b.y2);

    const double intersection =
        max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1);

    const double unionArea =
        a.area() + b.area() - intersection;

    return unionArea > 0 ? intersection / unionArea : 0.0;
}

double calculateGIoU(
    const BoundingBox& a,
    const BoundingBox& b
) {
    const double overlap = calculateIoU(a, b);

    const double x1 = min(a.x1, b.x1);
    const double y1 = min(a.y1, b.y1);
    const double x2 = max(a.x2, b.x2);
    const double y2 = max(a.y2, b.y2);

    const double enclosing = (x2 - x1) * (y2 - y1);

    const double ix1 = max(a.x1, b.x1);
    const double iy1 = max(a.y1, b.y1);
    const double ix2 = min(a.x2, b.x2);
    const double iy2 = min(a.y2, b.y2);

    const double intersection =
        max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1);

    const double unionArea =
        a.area() + b.area() - intersection;

    return enclosing > 0
        ? overlap - (enclosing - unionArea) / enclosing
        : overlap;
}

vector<Detection> confidenceFilter(
    const vector<Detection>& detections,
    double threshold
) {
    if (threshold < 0 || threshold > 1 || !isfinite(threshold)) {
        throw DetectionError("Invalid confidence threshold.");
    }

    vector<Detection> result;

    copy_if(
        detections.begin(),
        detections.end(),
        back_inserter(result),
        [threshold](const Detection& d) {
            return d.confidence >= threshold;
        }
    );

    return result;
}

class SuppressionEngine {
private:
    DetectionConfig config;

public:
    explicit SuppressionEngine(DetectionConfig cfg)
        : config(cfg) {
        config.validate();
    }

    vector<Detection> hardNMS(
        vector<Detection> candidates
    ) const {
        sort(
            candidates.begin(),
            candidates.end(),
            [](const Detection& a, const Detection& b) {
                return a.confidence > b.confidence;
            }
        );

        vector<Detection> selected;

        while (!candidates.empty() &&
               selected.size() < config.maxDetections) {

            Detection best = candidates.front();
            candidates.erase(candidates.begin());

            selected.push_back(best);

            vector<Detection> remaining;

            for (const auto& candidate : candidates) {
                if (!config.classAgnostic &&
                    candidate.classId != best.classId) {
                    remaining.push_back(candidate);
                    continue;
                }

                if (calculateIoU(best.box, candidate.box)
                    <= config.iouThreshold) {
                    remaining.push_back(candidate);
                }
            }

            candidates = move(remaining);
        }

        return selected;
    }

    vector<Detection> softNMS(
        vector<Detection> candidates
    ) const {
        vector<double> scores;

        for (const auto& d : candidates) {
            scores.push_back(d.confidence);
        }

        vector<Detection> selected;

        while (!candidates.empty() &&
               selected.size() < config.maxDetections) {

            size_t bestIndex = 0;

            for (size_t i = 1; i < scores.size(); ++i) {
                if (scores[i] > scores[bestIndex]) {
                    bestIndex = i;
                }
            }

            Detection best = candidates[bestIndex];
            double bestScore = scores[bestIndex];

            candidates.erase(candidates.begin() + bestIndex);
            scores.erase(scores.begin() + bestIndex);

            if (bestScore < 0.001) break;

            best.confidence = bestScore;
            selected.push_back(best);

            for (size_t i = 0; i < candidates.size(); ++i) {
                if (candidates[i].classId != best.classId) {
                    continue;
                }

                double overlap = calculateIoU(
                    best.box,
                    candidates[i].box
                );

                if (overlap > config.iouThreshold) {
                    scores[i] *= exp(
                        -(overlap * overlap) / config.softSigma
                    );
                }
            }

            vector<Detection> remaining;
            vector<double> remainingScores;

            for (size_t i = 0; i < candidates.size(); ++i) {
                if (scores[i] >= 0.001) {
                    remaining.push_back(candidates[i]);
                    remainingScores.push_back(scores[i]);
                }
            }

            candidates = move(remaining);
            scores = move(remainingScores);
        }

        return selected;
    }

    vector<Detection> run(
        const vector<Detection>& detections
    ) const {
        if (config.mode == SuppressionMode::SoftNMS) {
            return softNMS(detections);
        }

        return hardNMS(detections);
    }
};

struct EvaluationResult {
    int truePositives = 0;
    int falsePositives = 0;
    int falseNegatives = 0;

    double precision() const {
        int denominator = truePositives + falsePositives;

        return denominator == 0
            ? 0.0
            : static_cast<double>(truePositives) / denominator;
    }

    double recall() const {
        int denominator = truePositives + falseNegatives;

        return denominator == 0
            ? 0.0
            : static_cast<double>(truePositives) / denominator;
    }

    double f1() const {
        double p = precision();
        double r = recall();

        return p + r == 0 ? 0 : 2 * p * r / (p + r);
    }
};

EvaluationResult evaluate(
    const vector<Detection>& predictions,
    const vector<Detection>& targets,
    double threshold = 0.5
) {
    if (threshold < 0 || threshold > 1) {
        throw DetectionError("Invalid evaluation IoU threshold.");
    }

    vector<Detection> ordered = predictions;

    sort(
        ordered.begin(),
        ordered.end(),
        [](const Detection& a, const Detection& b) {
            return a.confidence > b.confidence;
        }
    );

    set<size_t> matched;
    EvaluationResult result;

    for (const auto& prediction : ordered) {
        double bestIoU = threshold;
        size_t bestIndex = targets.size();

        for (size_t i = 0; i < targets.size(); ++i) {
            if (matched.count(i)) continue;

            if (prediction.imageId != targets[i].imageId ||
                prediction.classId != targets[i].classId) {
                continue;
            }

            double overlap = calculateIoU(
                prediction.box,
                targets[i].box
            );

            if (overlap >= bestIoU) {
                bestIoU = overlap;
                bestIndex = i;
            }
        }

        if (bestIndex != targets.size()) {
            matched.insert(bestIndex);
            result.truePositives++;
        } else {
            result.falsePositives++;
        }
    }

    result.falseNegatives =
        static_cast<int>(targets.size() - matched.size());

    return result;
}

class RepositoryGovernanceEngine {
private:
    DetectionConfig config;
    SuppressionEngine suppression;

    size_t imagesProcessed = 0;
    size_t totalInput = 0;
    size_t totalOutput = 0;

public:
    explicit RepositoryGovernanceEngine(DetectionConfig cfg)
        : config(cfg), suppression(cfg) {}

    vector<Detection> process(
        const string& imageId,
        vector<Detection> detections,
        int imageWidth,
        int imageHeight
    ) {
        if (imageId.empty() || imageWidth <= 0 || imageHeight <= 0) {
            throw DetectionError("Invalid image metadata.");
        }

        for (const auto& d : detections) {
            if (d.imageId != imageId) {
                throw DetectionError("Mixed image IDs.");
            }
        }

        totalInput += detections.size();

        vector<Detection> clipped;

        for (auto& detection : detections) {
            if (detection.box.clip(imageWidth, imageHeight)) {
                clipped.push_back(detection);
            }
        }

        auto filtered = confidenceFilter(
            clipped,
            config.confidenceThreshold
        );

        auto output = suppression.run(filtered);

        imagesProcessed++;
        totalOutput += output.size();

        return output;
    }

    void report() const {
        cout << "\nPIPELINE REPORT\n";
        cout << "Images processed: " << imagesProcessed << '\n';
        cout << "Input detections: " << totalInput << '\n';
        cout << "Output detections: " << totalOutput << '\n';

        double reduction = totalInput == 0
            ? 0.0
            : 100.0 * (totalInput - totalOutput) / totalInput;

        cout << "Detection reduction: "
             << fixed << setprecision(2)
             << reduction << "%\n";
    }
};

int main() {
    try {
        BoundingBox a(50, 40, 200, 220);
        BoundingBox b(55, 45, 198, 218);

        cout << fixed << setprecision(4);

        cout << "IoU: " << calculateIoU(a, b) << '\n';
        cout << "GIoU: " << calculateGIoU(a, b) << '\n';

        if (abs(calculateIoU(a, a) - 1.0) > 1e-9) {
            throw DetectionError("IoU identity test failed.");
        }

        if (calculateIoU(
            BoundingBox(0, 0, 10, 10),
            BoundingBox(10, 0, 20, 10)
        ) != 0.0) {
            throw DetectionError("Boundary test failed.");
        }

        DetectionConfig config;
        config.confidenceThreshold = 0.25;
        config.iouThreshold = 0.45;
        config.maxDetections = 100;
        config.classAgnostic = false;
        config.mode = SuppressionMode::HardNMS;

        RepositoryGovernanceEngine pipeline(config);

        vector<Detection> detections = {
            {
                BoundingBox(50, 40, 200, 220),
                0.96, 0, "person", "image-001"
            },
            {
                BoundingBox(55, 45, 198, 218),
                0.89, 0, "person", "image-001"
            },
            {
                BoundingBox(300, 100, 430, 260),
                0.91, 2, "car", "image-001"
            },
            {
                BoundingBox(500, 50, 580, 130),
                0.18, 0, "person", "image-001"
            },
            {
                BoundingBox(610, 100, 710, 220),
                0.82, 0, "person", "image-001"
            }
        };

        auto results = pipeline.process(
            "image-001",
            detections,
            640,
            480
        );

        cout << "\nFINAL DETECTIONS\n";

        for (const auto& d : results) {
            cout << d.label
                 << " confidence=" << d.confidence
                 << " box=("
                 << d.box.x1 << ","
                 << d.box.y1 << ","
                 << d.box.x2 << ","
                 << d.box.y2 << ")\n";
        }

        vector<Detection> targets = {
            {
                BoundingBox(52, 42, 202, 222),
                1.0, 0, "person", "image-001"
            },
            {
                BoundingBox(305, 105, 435, 265),
                1.0, 2, "car", "image-001"
            }
        };

        auto evaluation = evaluate(results, targets, 0.5);

        cout << "\nEVALUATION\n";
        cout << "TP: " << evaluation.truePositives << '\n';
        cout << "FP: " << evaluation.falsePositives << '\n';
        cout << "FN: " << evaluation.falseNegatives << '\n';
        cout << "Precision: " << evaluation.precision() << '\n';
        cout << "Recall: " << evaluation.recall() << '\n';
        cout << "F1: " << evaluation.f1() << '\n';

        pipeline.report();

    } catch (const exception& error) {
        cerr << "Detection pipeline failure: "
             << error.what() << '\n';

        return 1;
    }

    return 0;
}
