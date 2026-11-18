#include "core.h"
#include "tensor.h"
#include "model.h"
#include "utils.h"
#include "image.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <filesystem>


int main(const int argc, const char *argv[]) {
    if (argc != 4) {
        throw std::runtime_error("main: provide model dir, data dir, and optimization-run flag.");
    }

    std::string model_dir{argv[1]};
    std::string data_dir{argv[2]};
    const bool optimize{std::string{argv[3]} == "true"};

    if (model_dir.back() != '/') {
        model_dir = model_dir + "/";
    }

    if (data_dir.back() != '/') {
        data_dir = data_dir + "/";
    }

    // load data
    std::vector<inference_engine::Tensor> dataset;
    std::vector<std::string> sample_names;
    for (const auto &fn: std::filesystem::directory_iterator(data_dir)) {
        if (fn.is_regular_file() && fn.path().extension() == ".jpg") {
            const auto img = inference_engine::load_image(fn.path().string());
            if (img.shape != inference_engine::Shape{3, 224, 224}) {
                throw std::runtime_error("Expected 224x224 RGB image.");
            }
            dataset.push_back(img);
            sample_names.push_back(fn.path().stem().string());
        }
    }

    // load model
    auto start = std::chrono::steady_clock::now();
    inference_engine::Model model(model_dir);
    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Model loading: " << elapsed.count() << "ms\n";

    // load labels
    std::ifstream file(model_dir + "/label.txt");
    if (!file.is_open()) {
        throw std::runtime_error("imagenet: label.txt can not be opened.");
    }

    std::string line;
    getline(file, line);
    const auto labels = inference_engine::tokenize(line);

    if (optimize) {
        std::cout << "--------------------\n";
        std::cout << "Optimize model\n";
        model.optimize(dataset[0]);
        return 0;
    }

    // warmup
    std::cout << "--------------------\n";
    for (size_t i = 0; i < 25; i++) {
        std::cout << "Warmup run, inference: class=" << inference_engine::argmax(model.compute(dataset[0])) << "\n";
    }

    // measure inference time
    std::cout << "Measuring inference time (output delayed)\n";
    std::cout << "--------------------\n";
    std::stringstream cached_outputs;
    size_t total_runs = 0;
    start = std::chrono::steady_clock::now();
    for (size_t j = 0; j < 100; j++) {
        for (size_t i = 0; i < dataset.size(); i++) {
            const auto prob_tensor = inference_engine::softmax(model.compute(dataset[i]));
            const auto pred_class = inference_engine::argmax(prob_tensor);

            cached_outputs << "Sample: " << sample_names[i] << "\n";
            cached_outputs << "Predicted: class=" << pred_class << " label=" << labels[pred_class] << "\n";
            cached_outputs << "Prob: " << prob_tensor.get(pred_class) << "\n";
            cached_outputs << "--------------------\n";
            total_runs += 1;
        }
    }

    end = std::chrono::steady_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << cached_outputs.str();
    std::cout << "Runtime per sample: " << elapsed.count() / total_runs << "ms\n";

    return 0;
}
