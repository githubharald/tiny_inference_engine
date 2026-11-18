#ifndef CPP_CONV2D_H
#define CPP_CONV2D_H

#include "tensor.h"

namespace inference_engine {
    /// convolution gets converted to matrix multiplication, fastest option here
    int conv2d(Tensor &out_tensor,
               Tensor &img_patch_mat,
               Tensor &kernel_mat,
               const Tensor &kernel,
               const Tensor &in_tensor,
               const Tensor &bias,
               int stride,
               bool apply_relu,
               int max_num_shards
    );
}

#endif //CPP_CONV2D_H
