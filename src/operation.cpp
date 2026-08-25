#include "operation.h"

#include "utils.h"
#include "core.h"

#include <Eigen/Dense>

#include <iostream>
#include <algorithm>
#include <thread>
#include <stdexcept>

#include "conv2d.h"

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
        auto &out_tensor = zeros_cached(variable_store,
                                       m_var_name_out, {
                                           in_tensor.shape[0],
                                           (in_tensor.shape[1] + 2 * m_pad - m_kernel) / m_stride + 1,
                                           (in_tensor.shape[2] + 2 * m_pad - m_kernel) / m_stride + 1
                                       });
        for (int c = 0; c < out_tensor.shape[0]; c++) {
            for (int y = 0; y < out_tensor.shape[1]; y++) {
                for (int x = 0; x < out_tensor.shape[2]; x++) {
                    float max_val = std::numeric_limits<float>::lowest();
                    for (int ky = 0; ky < m_kernel; ky++) {
                        for (int kx = 0; kx < m_kernel; kx++) {
                            const int y_in = m_stride * y + ky - m_pad;
                            const int x_in = m_stride * x + kx - m_pad;

                            if (y_in < 0 || y_in >= in_tensor.shape[1] || x_in < 0 || x_in >= in_tensor.shape[2]) {
                                continue;
                            }
                            max_val = std::max(max_val, in_tensor.get_3d(c, y_in, x_in));
                        }
                    }
                    out_tensor.get_3d(c, y, x) = max_val;
                }
            }
        }
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
        // TODO to support any input size, this op would need to be properly implemented
        // we just pass through the input tensor
        // this op is used in VGGNet, but not needed with the proper input size
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
               const int stride) : m_var_name_out(var_name_out),
                                   m_var_name_in(var_name_in),
                                   m_weight_name(weight_name),
                                   m_bias_name(bias_name),
                                   m_pad(pad),
                                   m_stride(stride) {
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

#ifdef GEMM_BASED_CONV
        gemm_based_conv(out_tensor, m_img_patch_mat, m_kernel_mat, kernel, in_tensor, bias, m_stride);
#else
        // start worker threads
        std::vector<std::thread> workers;
        const int num_shards = std::min({12, channel_out});
        const int shard_size = channel_out % num_shards == 0
                                   ? channel_out / num_shards
                                   : channel_out / num_shards + 1;
        for (int shard = 0; shard < num_shards; shard++) {
            const int c_out_begin = shard * shard_size;
            const int c_out_end = std::min({c_out_begin + shard_size, channel_out});
            if (m_stride == 1) {
                workers.emplace_back(sharded_conv,
                                     std::ref(out_tensor),
                                     std::cref(in_tensor),
                                     std::cref(kernel),
                                     std::cref(bias),
                                     c_out_begin,
                                     c_out_end
                );
            } else {
                workers.emplace_back(sharded_strided_conv,
                                     std::ref(out_tensor),
                                     std::cref(in_tensor),
                                     std::cref(kernel),
                                     std::cref(bias),
                                     c_out_begin,
                                     c_out_end,
                                     m_stride
                );
            }
        }

        // wait for the to finish
        for (auto &t: workers) {
            t.join();
        }
#endif
    }


    Add::Add(const std::string &var_name_out, const std::string &var_name_in_1, const std::string &var_name_in_2)
        : m_var_name_out(var_name_out),
          m_var_name_in_1(var_name_in_1),
          m_var_name_in_2(var_name_in_2) {
    }

    void Add::compute(VariableStore &variable_store) {
        const auto &in_tensor_1 = variable_store[m_var_name_in_1];
        const auto &in_tensor_2 = variable_store[m_var_name_in_2];
        if (in_tensor_1.shape != in_tensor_2.shape) {
            throw std::runtime_error("Add: shapes do not match.");
        }

        auto &out_tensor = zeros_cached(variable_store, m_var_name_out, in_tensor_1.shape);

        for (int i = 0; i < out_tensor.num_elements(); i++) {
            out_tensor.get(i) = in_tensor_1.get(i) + in_tensor_2.get(i);
        }
    }
}
