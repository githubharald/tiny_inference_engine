#include "check.hpp"
#include "core.h"
#include "tensor.h"
#include <cmath>


using namespace inference_engine;

void test_softmax() {
    Tensor in_tensor{zeros({2})};
    in_tensor.data[0] = 0;
    in_tensor.data[1] = std::log(3.0f);
    auto res = softmax(in_tensor);

    CHECK(is_close(res.data[0],0.25));
    CHECK(is_close(res.data[1],0.75));
}

void test_softmax_overflow() {
    Tensor in_tensor{zeros({2})};
    float bias = 100; // this overflows exp when not taking care of
    in_tensor.data[0] = bias;
    in_tensor.data[1] = bias + std::log(3.0f);
    auto res = softmax(in_tensor);

    CHECK(is_close(res.data[0],0.25));
    CHECK(is_close(res.data[1],0.75));
}

int main() {
    test_softmax();
    test_softmax_overflow();
    return 0;
}
