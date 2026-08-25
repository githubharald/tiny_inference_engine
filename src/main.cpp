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
    if (argc != 3) {
        throw std::runtime_error("main: model and data dirs not provided.");
    }

    std::string model_dir{argv[1]};
    std::string data_dir{argv[2]};

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
    auto elapsed = duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Model loading: " << elapsed.count() << "ms\n";

    // load labels
    std::ifstream file(model_dir + "/label.txt");
    if (!file.is_open()) {
        throw std::runtime_error("imagenet: label.txt can not be opened.");
    }

    std::string line;
    getline(file, line);
    const auto labels = inference_engine::tokenize(line);

    // warmup
    std::cout << "--------------------\n";
    std::cout << "Warmup run, inference: class=" << inference_engine::argmax(model.compute(dataset[0])) << "\n";

    // measure inference time
    std::cout << "Measuring inference time\n";
    std::cout << "--------------------\n";
    size_t total_runs = 0;
    start = std::chrono::steady_clock::now();
    for (size_t j = 0; j < 10; j++) {
        for (size_t i = 0; i < dataset.size(); i++) {
            const auto prob_tensor = inference_engine::softmax(model.compute(dataset[i]));
            const auto pred_class = inference_engine::argmax(prob_tensor);

            std::cout << "Sample: " << sample_names[i] << "\n";
            std::cout << "Predicted: class=" << pred_class << " label=" << labels[pred_class] << "\n";
            std::cout << "Prob: " << prob_tensor.get(pred_class) << "\n";
            std::cout << "--------------------\n";
            total_runs += 1;
        }
    }

    end = std::chrono::steady_clock::now();
    elapsed = duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Runtime per sample: " << elapsed.count() / total_runs << "ms\n";

    return 0;
}
