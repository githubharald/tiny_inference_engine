#include "model.h"

#include "utils.h"

#include <Eigen/Dense>

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>


namespace inference_engine {
    Model::Model(const std::string &model_dir) : m_model_dir(model_dir) {
        std::ifstream model_file(m_model_dir + "/model.txt");
        if (!model_file.is_open()) {
            throw std::runtime_error("Model: model.txt can not be opened.");
        }

        std::string line;
        while (getline(model_file, line)) {
            size_t i = line.find('=');
            std::string var_name_out = line.substr(0, i);
            std::string command = line.substr(i + 1);

            i = command.find('(');
            size_t j = command.find(')');
            std::string operation_name = command.substr(0, i);
            std::string arguments_str = command.substr(i + 1, j - i - 1);
            std::vector<std::string> arguments = tokenize(arguments_str);

            // select operation
            std::shared_ptr<IOperation> operation;
            if (operation_name == "Identity") {
                // we optimize two Identity calls away: input and output
                if (var_name_out == "__OUTPUT__") {
                    m_model_output = arguments[0];
                } else if (arguments[0] == "__INPUT__") {
                    m_model_input = var_name_out;
                } else {
                    operation = std::make_shared<Identity>(var_name_out, arguments[0]);
                }
            } else if (operation_name == "Gemm") {
                operation = std::make_shared<Linear>(var_name_out,
                                                     arguments[0],
                                                     arguments[1],
                                                     arguments[2]);
            } else if (operation_name == "Add") {
                operation = std::make_shared<Add>(var_name_out,
                                                  arguments[0],
                                                  arguments[1],
                                                  arguments[2] == "Relu");
            } else if (operation_name == "Relu") {
                operation = std::make_shared<Relu>(var_name_out, arguments[0]);
            } else if (operation_name == "LogSoftmax") {
                operation = std::make_shared<LogSoftmax>(var_name_out, arguments[0]);
            } else if (operation_name == "Reshape") {
                operation = std::make_shared<Reshape>(var_name_out,
                                                      arguments[0],
                                                      arguments[1]);
            } else if (operation_name == "Conv") {
                const auto conv_op = std::make_shared<Conv>(var_name_out,
                                                            arguments[0],
                                                            arguments[1],
                                                            arguments[2],
                                                            std::stoi(arguments[3]),
                                                            std::stoi(arguments[4]),
                                                            arguments[5] == "Relu");;
                m_conv_ops.push_back(conv_op);
                operation = conv_op;
            } else if (operation_name == "MaxPool") {
                operation = std::make_shared<MaxPool>(var_name_out,
                                                      arguments[0],
                                                      std::stoi(arguments[1]),
                                                      std::stoi(arguments[2]),
                                                      std::stoi(arguments[3]));
            } else if (operation_name == "AveragePool") {
                operation = std::make_shared<AveragePool>(var_name_out, arguments[0]);
            } else if (operation_name == "ReduceMean") {
                operation = std::make_shared<ReduceMean>(var_name_out, arguments[0]);
            } else {
                throw std::runtime_error("Model: unknown operation: " + operation_name);
            }

            if (operation) {
                operation->identifier = "OP=" + operation_name + "|OUT=" + var_name_out;
                m_operations.push_back(operation);
                const auto initializer_names = operation->get_initializer_names();
                for (const auto &initializer_name: initializer_names) {
                    m_variable_store[initializer_name] = load_tensor(m_model_dir + initializer_name + ".bin");
                }
            }

            std::cout << "OP=" << operation_name << "|OUT=" << var_name_out << "|ARG=";
            for (const auto &v: arguments) {
                std::cout << v << " ";
            }
            if (!operation) {
                std::cout << "[DEL]";
            }
            std::cout << "\n";
        }


        std::ifstream optimization_file(m_model_dir + "/optimization.txt");
        if (optimization_file.is_open()) {
            std::cout << "Loading optimization file\n";
            getline(optimization_file, line);

            std::vector<int> optimal_shards;
            for (const auto &s: tokenize(line)) {
                optimal_shards.push_back(std::stoi(s));
            }

            if (optimal_shards.size() != m_conv_ops.size()) {
                throw std::runtime_error("Optimization file does not match model");
            }

            for (int i = 0; i < optimal_shards.size(); i++) {
                m_conv_ops[i]->set_shards(optimal_shards[i]);
            }
        }
    }

    std::string Model::get_layer_runtime(const Tensor &input) {
        m_variable_store[m_model_input] = input;

        std::string csv_data;
        for (const auto &operation: m_operations) {
            auto start = std::chrono::steady_clock::now();
            operation->compute(m_variable_store);
            auto end = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            auto elapsed_us = elapsed.count();
            csv_data += operation->identifier + "," + std::to_string(elapsed_us) + ",\n";
        }

        return csv_data;
    }


    Tensor Model::compute(const Tensor &input) {
        m_variable_store[m_model_input] = input;
        for (const auto &operation: m_operations) {
            operation->compute(m_variable_store);
        }


#ifdef CHECK_NAN_INF
        for (auto const &[k, t]: m_variable_store) {
            if (t.has_invalid_elements()) {
                std::cout << "NAN/INF values in variable: " << k << "\n";
            }
        }
#endif

        return m_variable_store.at(m_model_output);
    }

    float Model::measure_model_runtime(const Tensor &input) {
        auto start = std::chrono::steady_clock::now();
        int num_runs = 7;
        for (int i = 0; i < num_runs; i++) {
            compute(input);
        }
        auto end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        return static_cast<float>(elapsed.count()) / static_cast<float>(num_runs);
    }


    void Model::optimize(const Tensor &input) {
        // warmup
        for (int i = 0; i < 5; i++) {
            compute(input);
        }

        // global optimization
        std::string optimal_shards;
        float best_runtime = std::numeric_limits<float>::max();
        int best_num_shards = 1;
        for (int s = 32; s > 0; s--) {
            for (const auto &conv_op: m_conv_ops) {
                conv_op->set_shards(s);
            }

            const float runtime = measure_model_runtime(input);
            if (runtime < best_runtime) {
                best_runtime = runtime;
                best_num_shards = s;
            }

            std::cout << "Shards: " << s << ". Runtime: " << runtime << "ms\n";
        }

        for (const auto &conv_op: m_conv_ops) {
            conv_op->set_shards(best_num_shards);
            optimal_shards += std::to_string(best_num_shards) + ",";
        }

        std::ofstream optimization_file(m_model_dir + "/optimization.txt", std::ofstream::out);
        optimization_file << optimal_shards;
    }
}
