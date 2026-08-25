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

    private:
        std::vector<std::shared_ptr<IOperation> > m_operations;
        VariableStore m_variable_store;
        std::string m_model_input;
        std::string m_model_output;
    };
}
#endif //CPP_MODEL_H
