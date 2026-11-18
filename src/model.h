#ifndef CPP_MODEL_H
#define CPP_MODEL_H

#include "operation.h"

#include <string>
#include <memory>
#include "tensor.h"
#include <vector>

namespace inference_engine {
    /// a neural network model loaded for inference
    class Model {
    public:
        /// load model from a directory
        explicit Model(const std::string &model_dir);

        /// feed input tensor through model
        Tensor compute(const Tensor &input);

        /// get runtime per layer as a csv string
        std::string get_layer_runtime(const Tensor &input);

        /// find optimal sharding size
        void optimize(const Tensor &input);

    private:
        std::string m_model_dir;
        std::vector<std::shared_ptr<IOperation> > m_operations;
        std::vector<std::shared_ptr<Conv> > m_conv_ops;
        VariableStore m_variable_store;
        std::string m_model_input;
        std::string m_model_output;

        float measure_model_runtime(const Tensor &input);
    };
}
#endif //CPP_MODEL_H
