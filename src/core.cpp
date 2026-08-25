#include "tensor.h"

#include <Eigen/Dense>

namespace inference_engine {
    Tensor softmax(const Tensor &t, const bool apply_log) {
        Tensor res = zeros(t.shape);
        float exp_sum = 0;
        for (int i = 0; i < t.num_elements(); i++) {
            exp_sum += std::exp(t.get(i));
        }
        for (int i = 0; i < res.num_elements(); i++) {
            auto val = std::exp(t.get(i)) / exp_sum;
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
        for (int i = 0; i < t.num_elements(); i++) {
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

        const Eigen::Map<const Matrix> emat1(mat.data.data(), rows1, cols1);
        const Eigen::Map<const Matrix> emat2(in_tensor.data.data(), rows2, 1);
        const Eigen::Map<const Vector> ebias(bias.data.data(), rows1);
        [[maybe_unused]] Eigen::Map<Matrix> eout(out_tensor.data.data(), rows1, 1);
        eout.noalias() = emat1 * emat2;
        eout += ebias;
        return out_tensor;
    }

    Tensor &mat_mat_mul(Tensor &out_tensor,
                        const Tensor &mat1,
                        const bool trans1,
                        const Tensor &mat2,
                        const bool trans2,
                        const Tensor &bias) {
        const int rows1 = mat1.shape[0];
        const int cols1 = mat1.shape[1];
        const int rows2 = mat2.shape[0];
        const int cols2 = mat2.shape[1];

        using Matrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        using Vector = Eigen::Matrix<float, Eigen::Dynamic, 1>;

        const Eigen::Map<const Matrix> emat1(mat1.data.data(), rows1, cols1);
        const Eigen::Map<const Matrix> emat2(mat2.data.data(), rows2, cols2);
        const Eigen::Map<const Vector> ebias(bias.data.data(), bias.shape[0]);

        // explicitly write out transpose combinations as this gives fast execution
        if (!trans1 && !trans2) {
            [[maybe_unused]] Eigen::Map<Matrix> eout(out_tensor.data.data(), rows1, cols2);
            eout.noalias() = emat1 * emat2;
            eout.colwise() += ebias;
            return out_tensor;
        }
        if (!trans1 && trans2) {
            [[maybe_unused]] Eigen::Map<Matrix> eout(out_tensor.data.data(), rows1, rows2);
            eout.noalias() = emat1 * emat2.transpose();
            eout.colwise() += ebias;
            return out_tensor;
        }

        // not needed (for now)
        throw std::runtime_error("mat_mat_mul: transpose of mat1 not implemented");
    }
}
