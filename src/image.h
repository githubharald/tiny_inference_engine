#ifndef CPP_IMAGE_H
#define CPP_IMAGE_H

#include "tensor.h"

namespace inference_engine {
    /// load RGB image
    Tensor load_image(const std::string &fn);
}


#endif //CPP_IMAGE_H
