#ifndef CPP_MAXPOOL2D_H
#define CPP_MAXPOOL2D_H

#include "tensor.h"

namespace inference_engine {
    void maxpool2d(const Tensor &in_tensor,
                   Tensor &out_tensor,
                   int kernel,
                   int pad,
                   int stride);
}

#endif //CPP_MAXPOOL2D_H
