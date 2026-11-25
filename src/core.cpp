#include "tensor.h"

#include <Eigen/Dense>

namespace inference_engine {
    Tensor softmax(const Tensor &t, const bool apply_log) {
        Tensor res = zeros(t.shape);

        // find min value to subtract to avoid overflow
        float min_val = std::numeric_limits<float>::max();
        for (const auto val: t.data) {
            if (val < min_val) {
                min_val = val;
            }
        }

        // denominator
        float exp_sum = 0;
        for (const auto v: t.data) {
            exp_sum += std::exp(v - min_val);
        }

        // compute numerator, and divide by denominator
        const int num_elements = res.num_elements();
        for (int i = 0; i < num_elements; i++) {
            auto val = std::exp(t.get(i) - min_val) / exp_sum;
            if (apply_log) {
                val = std::log(val);
            }
            res.get(i) = val;
        }
        return res;
    }

    int argmax(const Tensor &t) {
        int res = 0;
        float max_val = t.get(0);
        const int num_elements = t.num_elements();
        for (int i = 0; i < num_elements; i++) {
            if (t.get(i) > max_val) {
                max_val = t.get(i);
                res = i;
            }
        }
        return res;
    }

    Tensor &mat_vec_mul(Tensor &out_tensor, const Tensor &mat, const Tensor &in_tensor, const Tensor &bias) {
        const int rows1 = mat.shape[0];
        const int cols1 = mat.shape[1];
        const int rows2 = in_tensor.shape[0];

        using Matrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        using Vector = Eigen::Matrix<float, Eigen::Dynamic, 1>;

        const Eigen::Map<const Matrix> emat1(mat.data_ptr(), rows1, cols1);
        const Eigen::Map<const Matrix> emat2(in_tensor.data_ptr(), rows2, 1);
        const Eigen::Map<const Vector> ebias(bias.data_ptr(), rows1);
        [[maybe_unused]] Eigen::Map<Matrix> eout(out_tensor.data_ptr(), rows1, 1);
        eout.noalias() = emat1 * emat2;
        eout += ebias;
        return out_tensor;
    }

    Tensor &mat_mat_mul(Tensor &out_tensor,
                        const Tensor &mat1,
                        const bool trans1,
                        const Tensor &mat2,
                        const bool trans2,
                        const Tensor &bias,
                        const bool apply_relu) {
        const int rows1 = mat1.shape[0];
        const int cols1 = mat1.shape[1];
        const int rows2 = mat2.shape[0];
        const int cols2 = mat2.shape[1];

        using Matrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        using Vector = Eigen::Matrix<float, Eigen::Dynamic, 1>;

        Eigen::Map<const Matrix> emat1(mat1.data_ptr(), rows1, cols1);
        Eigen::Map<const Matrix> emat2(mat2.data_ptr(), rows2, cols2);
        const Eigen::Map<const Vector> ebias(bias.data_ptr(), bias.shape[0]);

        // explicitly write out transpose combinations as this gives fast execution
        if (!trans1 && !trans2) {
            [[maybe_unused]] Eigen::Map<Matrix> eout(out_tensor.data_ptr(), rows1, cols2);
            eout.noalias() = emat1 * emat2;
            eout.colwise() += ebias;
            if (apply_relu) {
                eout = eout.cwiseMax(0.0);
            }
            return out_tensor;
        }
        if (!trans1 && trans2) {
            [[maybe_unused]] Eigen::Map<Matrix> eout(out_tensor.data_ptr(), rows1, rows2);
            eout.noalias() = emat1 * emat2.transpose();
            eout.colwise() += ebias;
            if (apply_relu) {
                eout = eout.cwiseMax(0.0);
            }
            return out_tensor;
        }

        // not needed (for now)
        throw std::runtime_error("mat_mat_mul: transpose of mat1 not implemented");
    }
}
