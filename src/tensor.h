#ifndef CPP_TENSOR_H
#define CPP_TENSOR_H

#include <vector>
#include <stdexcept>
#include <cmath>

//#define CHECK_TENSOR_SHAPE

namespace inference_engine {
    using Shape = std::vector<int>;
    using DataElement = float;
    using Data = std::vector<DataElement>;
    struct Tensor;

    /// number of elements (product of dimensions)
    int num_elements(const Shape &shape);

    /// string representation of tensor
    std::string to_string(const Tensor &t);

    /// create tensor filled with zeros
    Tensor zeros(const Shape &shape);

    /// Load tensor from file
    Tensor load_tensor(const std::string &fn);

    /// Tensor class that supports 1d to 4d indexing
    struct Tensor {
        Shape shape;
        Data data;

        void clear() {
            std::fill(data.begin(), data.end(), 0);
        }

        [[nodiscard]] bool empty() const {
            return data.empty();
        }

        [[nodiscard]] int num_elements() const {
            return inference_engine::num_elements(shape);
        }

        [[nodiscard]] bool has_invalid_elements() const {
            for (const auto v: data) {
                if (std::isnan(v) || std::isinf(v)) {
                    return true;
                }
            }
            return false;
        }

        [[nodiscard]] DataElement get(const int i0) const {
#ifdef CHECK_TENSOR_SHAPE
            if (i0 >= num_elements()) {
                throw std::runtime_error("CHECK_TENSOR_SHAPE");
            }
#endif
            return data[i0];
        }

        DataElement &get(const int i0) {
            return data[i0];
        }

        [[nodiscard]] DataElement get_1d(const int i0) const {
            return get(i0);
        }

        DataElement &get_1d(const int i0) {
            return get(i0);
        }

        [[nodiscard]] DataElement get_2d(const int i0, const int i1) const {
            return get(i0 * shape[1] + i1);
        }

        DataElement &get_2d(const int i0, const int i1) {
            return get(i0 * shape[1] + i1);
        }

        [[nodiscard]] DataElement get_3d(const int i0, const int i1, const int i2) const {
            return get((i0 * shape[1] + i1) * shape[2] + i2);
        }

        DataElement &get_3d(const int i0, const int i1, const int i2) {
            return get((i0 * shape[1] + i1) * shape[2] + i2);
        }

        [[nodiscard]] DataElement get_4d(const int i0, const int i1, const int i2, const int i3) const {
            return get(((i0 * shape[1] + i1) * shape[2] + i2) * shape[3] + i3);
        }

        DataElement &get_4d(const int i0, const int i1, const int i2, const int i3) {
            return get(((i0 * shape[1] + i1) * shape[2] + i2) * shape[3] + i3);
        }
    };
}
#endif //CPP_TENSOR_H
