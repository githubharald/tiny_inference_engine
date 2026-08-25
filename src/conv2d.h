#ifndef CPP_CONV2D_H
#define CPP_CONV2D_H

#include "tensor.h"

namespace inference_engine {
    /// convolution gets converted to matrix multiplication, fastest option here
    void gemm_based_conv(Tensor &out_tensor,
                         Tensor &img_patch_mat,
                         Tensor &kernel_mat,
                         const Tensor &kernel,
                         const Tensor &in_tensor,
                         const Tensor &bias,
                         int stride
    );

    /// classic implementation of convolution, supports all strides
    void sharded_strided_conv(Tensor &out_tensor,
                              const Tensor &in_tensor,
                              const Tensor &kernel,
                              const Tensor &bias,
                              int c_out_begin,
                              int c_out_end,
                              int stride);

    /// optimized classic implementation of convolution, but only supports unit strides
    void sharded_conv(Tensor &out_tensor,
                      const Tensor &in_tensor,
                      const Tensor &kernel,
                      const Tensor &bias,
                      int c_out_begin,
                      int c_out_end
    );
}

#endif //CPP_CONV2D_H
