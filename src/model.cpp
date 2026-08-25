#include "model.h"

#include "utils.h"

#include <Eigen/Dense>

#include <fstream>
#include <iostream>
#include <algorithm>
#include <stdexcept>

//#define CHECK_NAN_INF


namespace inference_engine {
    Model::Model(const std::string &model_dir) {
        std::ifstream file(model_dir + "/model.txt");
        if (!file.is_open()) {
            throw std::runtime_error("Model: model.txt can not be opened.");
        }

        std::string line;
        while (getline(file, line)) {
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
                                                  arguments[1]);
            } else if (operation_name == "Relu") {
                operation = std::make_shared<Relu>(var_name_out, arguments[0]);
            } else if (operation_name == "LogSoftmax") {
                operation = std::make_shared<LogSoftmax>(var_name_out, arguments[0]);
            } else if (operation_name == "Reshape") {
                operation = std::make_shared<Reshape>(var_name_out,
                                                      arguments[0],
                                                      arguments[1]);
            } else if (operation_name == "Conv") {
                operation = std::make_shared<Conv>(var_name_out,
                                                   arguments[0],
                                                   arguments[1],
                                                   arguments[2],
                                                   std::stoi(arguments[3]),
                                                   std::stoi(arguments[4]));
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
                m_operations.push_back(operation);
                const auto initializer_names = operation->get_initializer_names();
                for (const auto &initializer_name: initializer_names) {
                    m_variable_store[initializer_name] = load_tensor(model_dir + initializer_name + ".bin");
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
}
