#include "conv2d.h"
#include "tensor.h"
#include "check.hpp"

using namespace inference_engine;

void test_conv2d_3x3_same() {
    Tensor out_tensor{zeros({1, 3, 3})};
    Tensor kernel{ones({1, 1, 3, 3})};
    Tensor img{zeros({1, 3, 3})};
    img.get_3d(0, 1, 1) = 1;
    Tensor bias{zeros({1})};

    Tensor img_patch_mat;
    Tensor kernel_mat;
    conv2d(out_tensor,
           img_patch_mat,
           kernel_mat,
           kernel,
           img,
           bias,
           1,
           false,
           1);

    CHECK(out_tensor.data == std::vector<float>(9, 1));
}

void test_conv2d_3x3_same_relu() {
    Tensor out_tensor{zeros({1, 3, 3})};
    Tensor kernel{ones({1, 1, 3, 3})};
    Tensor img{zeros({1, 3, 3})};
    img.get_3d(0, 1, 1) = -1;
    Tensor bias{zeros({1})};

    Tensor img_patch_mat;
    Tensor kernel_mat;
    conv2d(out_tensor,
           img_patch_mat,
           kernel_mat,
           kernel,
           img,
           bias,
           1,
           true,
           1);

    CHECK(out_tensor.data == std::vector<float>(9, 0));
}


int main() {
    test_conv2d_3x3_same();
    test_conv2d_3x3_same_relu();
    return 0;
}
