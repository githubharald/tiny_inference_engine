#ifndef CPP_CORE_H
#define CPP_CORE_H

#include "tensor.h"


namespace inference_engine {
    /// softmax for a 1d tensor
    Tensor softmax(const Tensor &t, bool apply_log = false);

    /// 1d index of maximum element
    int argmax(const Tensor &t);

    /// matrix vector multiplication
    Tensor &mat_vec_mul(Tensor &out_tensor, const Tensor &mat, const Tensor &in_tensor, const Tensor &bias);

    /// matrix matrix multiplication, optionally with transposed matrices
    Tensor &mat_mat_mul(Tensor &out_tensor,
                        const Tensor &mat1,
                        bool trans1,
                        const Tensor &mat2,
                        bool trans2,
                        const Tensor &bias);
}

#endif //CPP_CORE_H
