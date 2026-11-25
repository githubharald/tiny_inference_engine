#include "maxpool2d.h"
#include "tensor.h"
#include "check.hpp"

using namespace inference_engine;


void test_maxpool2d_2x2() {
    Tensor input_tensor{zeros({1, 2, 2})};
    input_tensor.get_3d(0, 0, 0) = 0;
    input_tensor.get_3d(0, 1, 1) = 3;
    Tensor output_tensor{zeros({1, 1, 1})};

    maxpool2d(input_tensor, output_tensor, 2, 0, 2);

    CHECK(output_tensor.get_3d(0, 0, 0) == 3);
}


void test_maxpool2d_2x2_pad_1() {
    Tensor input_tensor{zeros({1, 2, 2})};
    input_tensor.get_3d(0, 0, 0) = -1;
    input_tensor.get_3d(0, 0, 1) = 1;
    input_tensor.get_3d(0, 1, 1) = 3;
    Tensor output_tensor{zeros({1, 2, 2})};

    maxpool2d(input_tensor, output_tensor, 2, 1, 2);

    CHECK(output_tensor.get_3d(0, 0, 0) == 0);
    CHECK(output_tensor.get_3d(0, 0, 1) == 1);
    CHECK(output_tensor.get_3d(0, 1, 0) == 0);
    CHECK(output_tensor.get_3d(0, 1, 1) == 3);
}

void test_maxpool2d_2x2_pad_1_stride_1() {
    Tensor input_tensor{zeros({1, 2, 2})};
    input_tensor.get_3d(0, 0, 0) = -1;
    input_tensor.get_3d(0, 0, 1) = 1;
    input_tensor.get_3d(0, 1, 1) = 3;
    Tensor output_tensor{zeros({1, 3, 3})};

    maxpool2d(input_tensor, output_tensor, 2, 1, 1);

    CHECK(output_tensor.get_3d(0, 0, 0) == 0);
    CHECK(output_tensor.get_3d(0, 0, 1) == 1);
    CHECK(output_tensor.get_3d(0, 0, 2) == 1);

    CHECK(output_tensor.get_3d(0, 1, 0) == 0);
    CHECK(output_tensor.get_3d(0, 1, 1) == 3);
    CHECK(output_tensor.get_3d(0, 1, 2) == 3);

    CHECK(output_tensor.get_3d(0, 2, 0) == 0);
    CHECK(output_tensor.get_3d(0, 2, 1) == 3);
    CHECK(output_tensor.get_3d(0, 2, 2) == 3);
}

int main() {
    test_maxpool2d_2x2();
    test_maxpool2d_2x2_pad_1();
    test_maxpool2d_2x2_pad_1_stride_1();
    return 0;
}
