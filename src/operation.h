#ifndef CPP_OPERATION_H
#define CPP_OPERATION_H

#include "tensor.h"

#include <map>
#include <string>

#define GEMM_BASED_CONV

namespace inference_engine {
    /// all named tensors required for the model inference (including temporary results)
    using VariableStore = std::map<std::string, Tensor>;

    /// create tensor filled with zeros, use cached tensor instance if possible
    Tensor &zeros_cached(VariableStore &variable_store, const std::string &var_name, const Shape &shape);

    /// abstract interface of an operation
    class IOperation {
    public:
        virtual void compute(VariableStore &variable_store) = 0;

        virtual std::vector<std::string> get_initializer_names() { return {}; }

        virtual ~IOperation() = default;
    };

    class Identity : public IOperation {
    public:
        Identity(const std::string &var_name_out, const std::string &var_name_in);

        void compute(VariableStore &variable_store) override;

    private:
        std::string m_var_name_out;
        std::string m_var_name_in;
    };

    class LogSoftmax : public IOperation {
    public:
        LogSoftmax(const std::string &var_name_out, const std::string &var_name_in);

        void compute(VariableStore &variable_store) override;

    private:
        std::string m_var_name_out;
        std::string m_var_name_in;
    };


    class Reshape : public IOperation {
    public:
        Reshape(const std::string &var_name_out, const std::string &var_name_in, const std::string &shape_name);

        void compute(VariableStore &variable_store) override;

        std::vector<std::string> get_initializer_names() override;

    private:
        std::string m_var_name_out;
        std::string m_var_name_in;
        std::string m_shape_name;
    };

    class Relu : public IOperation {
    public:
        Relu(const std::string &var_name_out, const std::string &var_name_in);

        void compute(VariableStore &variable_store) override;

    private:
        std::string m_var_name_out;
        std::string m_var_name_in;
    };

    class MaxPool : public IOperation {
    public:
        MaxPool(const std::string &var_name_out,
                const std::string &var_name_in,
                int kernel,
                int pad,
                int stride);

        void compute(VariableStore &variable_store) override;

    private:
        std::string m_var_name_out;
        std::string m_var_name_in;
        int m_kernel;
        int m_pad;
        int m_stride;
    };


    class ReduceMean : public IOperation {
    public:
        ReduceMean(const std::string &var_name_out, const std::string &var_name_in);

        void compute(VariableStore &variable_store) override;

    private:
        std::string m_var_name_out;
        std::string m_var_name_in;
    };

    class AveragePool : public IOperation {
    public:
        AveragePool(const std::string &var_name_out, const std::string &var_name_in);

        void compute(VariableStore &variable_store) override;

    private:
        std::string m_var_name_out;
        std::string m_var_name_in;
    };

    class Linear : public IOperation {
    public:
        Linear(const std::string &var_name_out,
               const std::string &var_name_in,
               const std::string &weight_name,
               const std::string &bias_name);

        std::vector<std::string> get_initializer_names() override;

        void compute(VariableStore &variable_store) override;

    private:
        std::string m_var_name_out;
        std::string m_var_name_in;
        std::string m_weight_name;
        std::string m_bias_name;
    };

    class Conv : public IOperation {
    public:
        Conv(const std::string &var_name_out,
             const std::string &var_name_in,
             const std::string &weight_name,
             const std::string &bias_name,
             int pad,
             int stride);

        std::vector<std::string> get_initializer_names() override;

        void compute(VariableStore &variable_store) override;

    private:
        std::string m_var_name_out;
        std::string m_var_name_in;
        std::string m_weight_name;
        std::string m_bias_name;
        int m_pad;
        int m_stride;
#ifdef GEMM_BASED_CONV
        // cache tensors for gemm-based conv
        Tensor m_img_patch_mat;
        Tensor m_kernel_mat;
#endif
    };


    class Add : public IOperation {
    public:
        Add(const std::string &var_name_out, const std::string &var_name_in_1, const std::string &var_name_in_2);

        void compute(VariableStore &variable_store) override;

    private:
        std::string m_var_name_out;
        std::string m_var_name_in_1;
        std::string m_var_name_in_2;
    };
}


#endif //CPP_OPERATION_H
