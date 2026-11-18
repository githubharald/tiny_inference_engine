#include "operation.h"

#include "utils.h"
#include "core.h"

#include <Eigen/Dense>

#include <iostream>
#include <algorithm>
#include <stdexcept>

#include "conv2d.h"
#include "maxpool2d.h"

namespace inference_engine {
    Tensor &zeros_cached(VariableStore &variable_store, const std::string &var_name, const Shape &shape) {
        auto it = variable_store.find(var_name);
        if (it != variable_store.end() && it->second.shape == shape) {
            it->second.clear();
            return it->second;
        }

        it = variable_store.insert(std::make_pair(var_name, zeros(shape))).first;
        return it->second;
    }

    Identity::Identity(const std::string &var_name_out, const std::string &var_name_in)
        : m_var_name_out(var_name_out),
          m_var_name_in(var_name_in) {
    }

    void Identity::compute(VariableStore &variable_store) {
        variable_store[m_var_name_out] = variable_store.at(m_var_name_in);
    }


    LogSoftmax::LogSoftmax(const std::string &var_name_out, const std::string &var_name_in)
        : m_var_name_out(var_name_out),
          m_var_name_in(var_name_in) {
    }

    void LogSoftmax::compute(VariableStore &variable_store) {
        const auto &in_tensor = variable_store.at(m_var_name_in);
        if (in_tensor.shape.size() != 1) {
            throw std::runtime_error("LogSoftmax: requires 1d tensor.");
        }
        variable_store[m_var_name_out] = softmax(in_tensor, true);
    }


    Reshape::Reshape(const std::string &var_name_out, const std::string &var_name_in, const std::string &shape_name)
        : m_var_name_out(var_name_out),
          m_var_name_in(var_name_in),
          m_shape_name(shape_name) {
    }

    void Reshape::compute(VariableStore &variable_store) {
        Shape target_shape;
        const auto &shape_tensor = variable_store.at(m_shape_name);
        // ONNX: first shape entry is batch dim and fixed at 1
        for (int i = 1; i < shape_tensor.num_elements(); i++) {
            target_shape.push_back(static_cast<int>(shape_tensor.get(i)));
        }

        auto out_tensor = variable_store.at(m_var_name_in);
        if (out_tensor.num_elements() != num_elements(target_shape)) {
            throw std::runtime_error("Reshape: different number of elements.");
        }
        out_tensor.shape = target_shape;
        variable_store[m_var_name_out] = out_tensor;
    }

    std::vector<std::string> Reshape::get_initializer_names() {
        return {m_shape_name};
    }


    Relu::Relu(const std::string &var_name_out, const std::string &var_name_in)
        : m_var_name_out(var_name_out),
          m_var_name_in(var_name_in) {
    }

    void Relu::compute(VariableStore &variable_store) {
        // input
        const auto &in_tensor = variable_store.at(m_var_name_in);

        // output
        auto &out_tensor = zeros_cached(variable_store, m_var_name_out, in_tensor.shape);

        // apply operation
        for (int i = 0; i < out_tensor.num_elements(); i++) {
            out_tensor.get(i) = std::max(in_tensor.get(i), static_cast<float>(0));
        }
    }


    MaxPool::MaxPool(const std::string &var_name_out,
                     const std::string &var_name_in,
                     const int kernel,
                     const int pad,
                     const int stride)
        : m_var_name_out(var_name_out),
          m_var_name_in(var_name_in),
          m_kernel(kernel),
          m_pad(pad),
          m_stride(stride) {
    }

    void MaxPool::compute(VariableStore &variable_store) {
        // input
        const auto &in_tensor = variable_store.at(m_var_name_in);

        // output
        const int height_out = (in_tensor.shape[1] + 2 * m_pad - m_kernel) / m_stride + 1;
        const int width_out = (in_tensor.shape[2] + 2 * m_pad - m_kernel) / m_stride + 1;
        auto &out_tensor = zeros_cached(variable_store,
                                        m_var_name_out, {
                                            in_tensor.shape[0],
                                            height_out,
                                            width_out
                                        });

        // compute
        maxpool2d(in_tensor, out_tensor, m_kernel, m_pad, m_stride);
    }


    ReduceMean::ReduceMean(const std::string &var_name_out, const std::string &var_name_in)
        : m_var_name_out(var_name_out),
          m_var_name_in(var_name_in) {
    }

    void ReduceMean::compute(VariableStore &variable_store) {
        // input
        const auto &in_tensor = variable_store.at(m_var_name_in);

        // output
        auto &out_tensor = zeros_cached(variable_store,
                                        m_var_name_out, {
                                            in_tensor.shape[0],
                                            1,
                                            1
                                        });
        for (int c = 0; c < in_tensor.shape[0]; c++) {
            float acc = 0;
            for (int y = 0; y < in_tensor.shape[1]; y++) {
                for (int x = 0; x < in_tensor.shape[2]; x++) {
                    acc += in_tensor.get_3d(c, y, x);
                }
            }
            out_tensor.get_3d(c, 0, 0) = acc / static_cast<float>(in_tensor.shape[1] * in_tensor.shape[2]);
        }
    }


    AveragePool::AveragePool(const std::string &var_name_out, const std::string &var_name_in)
        : m_var_name_out(var_name_out),
          m_var_name_in(var_name_in) {
    }

    void AveragePool::compute(VariableStore &variable_store) {
        // TODO this is just a fake implementation to support the op in its most basic form (pass-through)
        // TODO for arbitrary input sizes, a proper implementation would be required
        variable_store[m_var_name_out] = variable_store.at(m_var_name_in);
    }


    Linear::Linear(const std::string &var_name_out,
                   const std::string &var_name_in,
                   const std::string &weight_name,
                   const std::string &bias_name) : m_var_name_out(var_name_out),
                                                   m_var_name_in(var_name_in),
                                                   m_weight_name(weight_name),
                                                   m_bias_name(bias_name) {
    }

    std::vector<std::string> Linear::get_initializer_names() {
        return {m_weight_name, m_bias_name};
    }

    void Linear::compute(VariableStore &variable_store) {
        const auto &weight = variable_store.at(m_weight_name);
        const auto &bias = variable_store.at(m_bias_name);
        const auto &in_tensor = variable_store.at(m_var_name_in);

        auto &out_tensor = zeros_cached(variable_store, m_var_name_out, bias.shape);
        mat_vec_mul(out_tensor, weight, in_tensor, bias);
    }


    Conv::Conv(const std::string &var_name_out,
               const std::string &var_name_in,
               const std::string &weight_name,
               const std::string &bias_name,
               const int pad,
               const int stride,
               const bool apply_relu) : m_var_name_out(var_name_out),
                                        m_var_name_in(var_name_in),
                                        m_weight_name(weight_name),
                                        m_bias_name(bias_name),
                                        m_pad(pad),
                                        m_stride(stride),
                                        m_apply_relu(apply_relu) {
    }

    std::vector<std::string> Conv::get_initializer_names() {
        return {m_weight_name, m_bias_name};
    }

    void Conv::compute(VariableStore &variable_store) {
        const auto &kernel = variable_store.at(m_weight_name);
        const auto &bias = variable_store.at(m_bias_name);
        const auto &in_tensor = variable_store.at(m_var_name_in);

        const int channel_out = kernel.shape[0];
        const int height = in_tensor.shape[1];
        const int width = in_tensor.shape[2];
        const int kernel_height = kernel.shape[2];
        const int kernel_width = kernel.shape[3];

        if (kernel_height != kernel_width && kernel_height / 2 != m_pad) {
            throw std::runtime_error("Conv: kernel must be square, and padding must be 'same'.");
        }

        auto &out_tensor = zeros_cached(variable_store, m_var_name_out, {
                                            channel_out,
                                            height / m_stride,
                                            width / m_stride
                                        });

        m_possible_max_shards = conv2d(out_tensor,
                                       m_img_patch_mat,
                                       m_kernel_mat,
                                       kernel,
                                       in_tensor,
                                       bias,
                                       m_stride,
                                       m_apply_relu,
                                       m_configured_shards);
    }


    Add::Add(const std::string &var_name_out,
             const std::string &var_name_in_1,
             const std::string &var_name_in_2,
             const bool apply_relu)
        : m_var_name_out(var_name_out),
          m_var_name_in_1(var_name_in_1),
          m_var_name_in_2(var_name_in_2),
          m_apply_relu(apply_relu) {
    }

    void Add::compute(VariableStore &variable_store) {
        const auto &in_tensor_1 = variable_store[m_var_name_in_1];
        const auto &in_tensor_2 = variable_store[m_var_name_in_2];
        auto &out_tensor = zeros_cached(variable_store, m_var_name_out, in_tensor_1.shape);
        if (in_tensor_1.shape != in_tensor_2.shape) {
            throw std::runtime_error("Add: shapes do not match.");
        }

        // data pointers
        auto *p_out = out_tensor.data_ptr();
        auto *p_in1 = in_tensor_1.data_ptr();
        auto *p_in2 = in_tensor_2.data_ptr();

        // iterate through tensor data
        const int num_elements = out_tensor.num_elements();
        for (int i = 0; i < num_elements; i++) {
            DataElement res = *p_in1 + *p_in2;
            if (m_apply_relu) {
                res = std::max(res, 0.0f);
            }
            *p_out = res;

            p_in1++;
            p_in2++;
            p_out++;
        }
    }
}
