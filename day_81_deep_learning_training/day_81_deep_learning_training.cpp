#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;
namespace fs = std::filesystem;

enum class TrainingState {
    Initialized,
    Running,
    Completed,
    EarlyStopped,
    Failed
};

string stateName(TrainingState state) {
    switch (state) {
        case TrainingState::Initialized: return "Initialized";
        case TrainingState::Running: return "Running";
        case TrainingState::Completed: return "Completed";
        case TrainingState::EarlyStopped: return "EarlyStopped";
        case TrainingState::Failed: return "Failed";
    }
    return "Unknown";
}

struct TrainingConfig {
    int epochs = 40;
    int batchSize = 16;
    double learningRate = 0.05;
    double minimumLearningRate = 0.00001;
    double decay = 0.98;
    int patience = 8;
    int checkpointInterval = 5;
    double minDelta = 1e-8;
    unsigned int seed = 42;
    string checkpointDirectory = "cpp_checkpoints";

    void validate() const {
        if (epochs < 1 || batchSize < 1) {
            throw invalid_argument("Epochs and batch size must be positive.");
        }

        if (!isfinite(learningRate) || learningRate <= 0 ||
            !isfinite(minimumLearningRate) || minimumLearningRate <= 0) {
            throw invalid_argument("Invalid learning rate.");
        }

        if (!isfinite(decay) || decay <= 0 || decay > 1) {
            throw invalid_argument("Decay must be in (0, 1].");
        }

        if (patience < 1 || checkpointInterval < 1) {
            throw invalid_argument("Invalid training intervals.");
        }
    }
};

struct Sample {
    vector<double> features;
    double target;
};

struct Metrics {
    int epoch;
    int batches;
    double learningRate;
    double trainingLoss;
    double validationLoss;
};

class Dataset {
private:
    vector<Sample> samples;
    size_t featureCount = 0;

public:
    explicit Dataset(vector<Sample> input) : samples(move(input)) {
        if (samples.empty()) {
            throw invalid_argument("Dataset cannot be empty.");
        }

        featureCount = samples.front().features.size();

        if (featureCount == 0) {
            throw invalid_argument("At least one feature is required.");
        }

        for (const auto& sample : samples) {
            if (sample.features.size() != featureCount ||
                !isfinite(sample.target)) {
                throw invalid_argument("Invalid sample dimensions or target.");
            }

            for (double value : sample.features) {
                if (!isfinite(value)) {
                    throw invalid_argument("Non-finite feature detected.");
                }
            }
        }
    }

    size_t size() const {
        return samples.size();
    }

    size_t dimensions() const {
        return featureCount;
    }

    const Sample& operator[](size_t index) const {
        return samples.at(index);
    }
};

class LinearModel {
private:
    vector<double> weights;
    double bias = 0;

public:
    LinearModel(size_t dimensions, unsigned int seed) {
        if (dimensions == 0) {
            throw invalid_argument("Model requires at least one feature.");
        }

        mt19937 generator(seed);
        uniform_real_distribution<double> distribution(-0.1, 0.1);

        weights.resize(dimensions);

        for (double& weight : weights) {
            weight = distribution(generator);
        }
    }

    double predict(const vector<double>& features) const {
        if (features.size() != weights.size()) {
            throw invalid_argument("Model feature dimension mismatch.");
        }

        double result = bias;

        for (size_t i = 0; i < weights.size(); ++i) {
            result += weights[i] * features[i];
        }

        return result;
    }

    const vector<double>& getWeights() const {
        return weights;
    }

    double getBias() const {
        return bias;
    }

    void update(
        const vector<double>& gradients,
        double biasGradient,
        double learningRate
    ) {
        if (gradients.size() != weights.size()) {
            throw invalid_argument("Gradient dimension mismatch.");
        }

        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] -= learningRate * gradients[i];
        }

        bias -= learningRate * biasGradient;

        for (double weight : weights) {
            if (!isfinite(weight)) {
                throw runtime_error("Numerical instability in weights.");
            }
        }

        if (!isfinite(bias)) {
            throw runtime_error("Numerical instability in bias.");
        }
    }

    void restore(const vector<double>& restoredWeights, double restoredBias) {
        if (restoredWeights.size() != weights.size() ||
            !isfinite(restoredBias)) {
            throw invalid_argument("Invalid restored model.");
        }

        for (double value : restoredWeights) {
            if (!isfinite(value)) {
                throw invalid_argument("Non-finite checkpoint weight.");
            }
        }

        weights = restoredWeights;
        bias = restoredBias;
    }
};

class Scheduler {
private:
    double initial;
    double decay;
    double minimum;

public:
    Scheduler(double rate, double decayFactor, double minRate)
        : initial(rate), decay(decayFactor), minimum(minRate) {}

    double getRate(int epoch) const {
        return max(
            minimum,
            initial * pow(decay, epoch - 1)
        );
    }
};

class CheckpointManager {
private:
    fs::path directory;

public:
    explicit CheckpointManager(const string& location)
        : directory(location) {
        fs::create_directories(directory);
    }

    void save(
        const LinearModel& model,
        int epoch,
        double learningRate,
        double bestLoss,
        int bestEpoch
    ) const {
        fs::path destination =
            directory / ("epoch_" + to_string(epoch) + ".txt");

        fs::path temporary = destination;
        temporary += ".tmp";

        {
            ofstream output(temporary, ios::trunc);

            if (!output) {
                throw runtime_error("Cannot create checkpoint.");
            }

            output << setprecision(17);
            output << "TRAINING_CHECKPOINT_V1\n";
            output << epoch << '\n';
            output << learningRate << '\n';
            output << bestLoss << '\n';
            output << bestEpoch << '\n';
            output << model.getWeights().size() << '\n';

            for (double weight : model.getWeights()) {
                output << weight << '\n';
            }

            output << model.getBias() << '\n';
            output.flush();

            if (!output) {
                throw runtime_error("Checkpoint write failed.");
            }
        }

        std::error_code error;
        fs::rename(temporary, destination, error);

        if (error) {
            fs::remove(temporary);
            throw runtime_error("Checkpoint replacement failed.");
        }
    }

    struct Checkpoint {
        int epoch;
        double learningRate;
        double bestLoss;
        int bestEpoch;
        vector<double> weights;
        double bias;
    };

    Checkpoint load(const string& filename, size_t expectedDimensions) const {
        fs::path filePath = fs::weakly_canonical(filename);
        fs::path root = fs::weakly_canonical(directory);

        if (filePath.parent_path() != root) {
            throw runtime_error("Checkpoint path outside storage directory.");
        }

        ifstream input(filePath);

        if (!input) {
            throw runtime_error("Checkpoint cannot be opened.");
        }

        string signature;
        Checkpoint checkpoint{};
        size_t dimensions = 0;

        if (!(input >> signature) || signature != "TRAINING_CHECKPOINT_V1") {
            throw runtime_error("Invalid checkpoint signature.");
        }

        if (!(input >> checkpoint.epoch
                    >> checkpoint.learningRate
                    >> checkpoint.bestLoss
                    >> checkpoint.bestEpoch
                    >> dimensions)) {
            throw runtime_error("Corrupted checkpoint metadata.");
        }

        if (checkpoint.epoch < 0 || checkpoint.bestEpoch < 0 ||
            !isfinite(checkpoint.learningRate) ||
            checkpoint.learningRate <= 0 ||
            !isfinite(checkpoint.bestLoss) ||
            dimensions != expectedDimensions) {
            throw runtime_error("Checkpoint validation failed.");
        }

        checkpoint.weights.resize(dimensions);

        for (double& weight : checkpoint.weights) {
            if (!(input >> weight) || !isfinite(weight)) {
                throw runtime_error("Invalid checkpoint weight.");
            }
        }

        if (!(input >> checkpoint.bias) || !isfinite(checkpoint.bias)) {
            throw runtime_error("Invalid checkpoint bias.");
        }

        string unexpected;

        if (input >> unexpected) {
            throw runtime_error("Unexpected checkpoint content.");
        }

        return checkpoint;
    }
};

class Trainer {
private:
    TrainingConfig config;
    Dataset training;
    Dataset validation;
    LinearModel model;
    Scheduler scheduler;
    CheckpointManager checkpoints;

    TrainingState state = TrainingState::Initialized;
    vector<Metrics> history;

    double bestLoss = numeric_limits<double>::infinity();
    int bestEpoch = 0;
    int staleEpochs = 0;

    LinearModel bestModel;

    double calculateLoss(const Dataset& dataset) const {
        double total = 0;

        for (size_t i = 0; i < dataset.size(); ++i) {
            double error =
                model.predict(dataset[i].features) - dataset[i].target;

            total += error * error;
        }

        return total / static_cast<double>(dataset.size());
    }

    void trainBatch(
        const vector<size_t>& indices,
        double learningRate
    ) {
        vector<double> gradients(training.dimensions(), 0.0);
        double biasGradient = 0;

        for (size_t index : indices) {
            const Sample& sample = training[index];

            double error =
                model.predict(sample.features) - sample.target;

            for (size_t j = 0; j < gradients.size(); ++j) {
                gradients[j] +=
                    2.0 * error * sample.features[j] /
                    static_cast<double>(indices.size());
            }

            biasGradient +=
                2.0 * error / static_cast<double>(indices.size());
        }

        model.update(gradients, biasGradient, learningRate);
    }

    void restoreBestModel() {
        model.restore(bestModel.getWeights(), bestModel.getBias());
    }

public:
    Trainer(
        TrainingConfig trainingConfig,
        Dataset trainingData,
        Dataset validationData
    )
        : config(move(trainingConfig)),
          training(move(trainingData)),
          validation(move(validationData)),
          model(training.dimensions(), config.seed),
          scheduler(
              config.learningRate,
              config.decay,
              config.minimumLearningRate
          ),
          checkpoints(config.checkpointDirectory),
          bestModel(model) {

        config.validate();

        if (training.dimensions() != validation.dimensions()) {
            throw invalid_argument("Training dimensions do not match.");
        }
    }

    void resume(const string& filename) {
        if (state != TrainingState::Initialized) {
            throw runtime_error("Cannot restore after training starts.");
        }

        auto checkpoint =
            checkpoints.load(filename, training.dimensions());

        model.restore(checkpoint.weights, checkpoint.bias);
        bestModel = model;
        bestLoss = checkpoint.bestLoss;
        bestEpoch = checkpoint.bestEpoch;
        startEpoch = checkpoint.epoch + 1;
    }

    void fit() {
        if (state != TrainingState::Initialized) {
            throw runtime_error("Trainer can only run once.");
        }

        state = TrainingState::Running;

        int firstEpoch = startEpoch;

        try {
            mt19937 generator(config.seed + firstEpoch);

            for (int epoch = firstEpoch;
                 epoch <= config.epochs;
                 ++epoch) {

                vector<size_t> indices(training.size());

                for (size_t i = 0; i < indices.size(); ++i) {
                    indices[i] = i;
                }

                shuffle(indices.begin(), indices.end(), generator);

                double learningRate = scheduler.getRate(epoch);
                int batches = 0;

                for (size_t start = 0;
                     start < indices.size();
                     start += static_cast<size_t>(config.batchSize)) {

                    size_t end = min(
                        start + static_cast<size_t>(config.batchSize),
                        indices.size()
                    );

                    vector<size_t> batch(
                        indices.begin() + static_cast<ptrdiff_t>(start),
                        indices.begin() + static_cast<ptrdiff_t>(end)
                    );

                    trainBatch(batch, learningRate);
                    ++batches;
                }

                double trainingLoss = calculateLoss(training);
                double validationLoss = calculateLoss(validation);

                bool improved =
                    validationLoss < bestLoss - config.minDelta;

                if (improved) {
                    bestLoss = validationLoss;
                    bestEpoch = epoch;
                    bestModel = model;
                    staleEpochs = 0;
                } else {
                    ++staleEpochs;
                }

                Metrics metrics{
                    epoch,
                    batches,
                    learningRate,
                    trainingLoss,
                    validationLoss
                };

                history.push_back(metrics);

                cout << fixed << setprecision(6)
                     << "Epoch " << setw(3) << epoch
                     << " | batches=" << batches
                     << " | lr=" << learningRate
                     << " | train=" << trainingLoss
                     << " | validation=" << validationLoss
                     << '\n';

                if (epoch % config.checkpointInterval == 0 || improved) {
                    checkpoints.save(
                        model,
                        epoch,
                        learningRate,
                        bestLoss,
                        bestEpoch
                    );
                }

                if (staleEpochs >= config.patience) {
                    state = TrainingState::EarlyStopped;
                    break;
                }
            }

            restoreBestModel();

            if (state == TrainingState::Running) {
                state = TrainingState::Completed;
            }

        } catch (...) {
            state = TrainingState::Failed;
            throw;
        }
    }

    int getBestEpoch() const {
        return bestEpoch;
    }

    double getBestLoss() const {
        return bestLoss;
    }

    TrainingState getState() const {
        return state;
    }

    const LinearModel& getModel() const {
        return model;
    }

private:
    int startEpoch = 1;
};

pair<Dataset, Dataset> generateDataset(size_t count, unsigned int seed) {
    mt19937 generator(seed);
    uniform_real_distribution<double> distribution(-2.0, 2.0);
    normal_distribution<double> noise(0.0, 0.05);

    vector<Sample> trainingSamples;
    vector<Sample> validationSamples;

    size_t split = static_cast<size_t>(count * 0.8);

    if (split == 0 || split == count) {
        throw invalid_argument("Invalid dataset split.");
    }

    for (size_t i = 0; i < count; ++i) {
        double x1 = distribution(generator);
        double x2 = distribution(generator);

        double target =
            3.0 * x1 - 2.0 * x2 + 1.5 + noise(generator);

        Sample sample{{x1, x2}, target};

        if (i < split) {
            trainingSamples.push_back(sample);
        } else {
            validationSamples.push_back(sample);
        }
    }

    return {
        Dataset(move(trainingSamples)),
        Dataset(move(validationSamples))
    };
}

int main() {
    try {
        TrainingConfig config;

        auto datasets = generateDataset(200, config.seed);

        Trainer trainer(
            config,
            move(datasets.first),
            move(datasets.second)
        );

        trainer.fit();

        cout << "\nTraining report\n";
        cout << "---------------\n";
        cout << "State: " << stateName(trainer.getState()) << '\n';
        cout << "Best epoch: " << trainer.getBestEpoch() << '\n';
        cout << "Best validation loss: "
             << fixed << setprecision(8)
             << trainer.getBestLoss() << '\n';

        cout << "Weights: ";

        for (double weight : trainer.getModel().getWeights()) {
            cout << weight << ' ';
        }

        cout << "\nBias: " << trainer.getModel().getBias() << '\n';

        return 0;

    } catch (const exception& error) {
        cerr << "Training failure: " << error.what() << '\n';
        return 1;
    }
}
