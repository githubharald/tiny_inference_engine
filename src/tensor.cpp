#include "tensor.h"
#include "utils.h"
#include <fstream>
#include <iostream>

namespace inference_engine {
    Tensor load_tensor(const std::string &fn) {
        Tensor res;

        std::ifstream f(fn, std::ios::binary);
        if (!f) {
            throw std::runtime_error("load_tensor: file not found.");
        }

        int dims = 0;
        int ival = 0;

        // read num of dimensions
        f.read(reinterpret_cast<char *>(&dims), 4);

        // read shape (element-wise, could be >4byte)
        for (int i = 0; i < dims; i++) {
            f.read(reinterpret_cast<char *>(&ival), 4);
            res.shape.push_back(ival);
        }

        // read data (all at once, requires same float type)
        const int data_num_elements = num_elements(res.shape);
        res.data.resize(data_num_elements);
        f.read(reinterpret_cast<char *>(res.data.data()), sizeof(float) * data_num_elements);
        return res;
    }


    int num_elements(const Shape &shape) {
        if (shape.empty()) {
            throw std::runtime_error("num_elements: shape empty.");
        }
        int res = 1;
        for (const auto s: shape) {
            res *= s;
        }
        return res;
    }

    Tensor zeros(const Shape &shape) {
        return Tensor(shape, Data(num_elements(shape), 0));
    }

    std::string to_string(const Tensor &t) {
        std::string res("shape: ");
        for (const auto s: t.shape) {
            res += std::to_string(s) + " ";
        }
        res += "\ndata: ";
        for (const auto v: t.data) {
            res += std::to_string(v) + " ";
        }
        return res;
    }
}
