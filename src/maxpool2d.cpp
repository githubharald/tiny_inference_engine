#include "maxpool2d.h"

#include "utils.h"

#include <thread>

namespace inference_engine {
    void sharded_maxpool(const Tensor &in_tensor,
                         Tensor &out_tensor,
                         const int kernel,
                         const int pad,
                         const int stride,
                         const int c_begin,
                         const int c_end) {
        const int height_in = in_tensor.shape[1];
        const int width_in = in_tensor.shape[2];
        const int height_out = out_tensor.shape[1];
        const int width_out = out_tensor.shape[2];

        auto *p_out = &out_tensor.get_3d(c_begin, 0, 0);
        for (int c = c_begin; c < c_end; c++) {
            for (int y = 0; y < height_out; y++) {
                for (int x = 0; x < width_out; x++) {
                    float max_val = std::numeric_limits<float>::lowest();
                    for (int ky = 0; ky < kernel; ky++) {
                        for (int kx = 0; kx < kernel; kx++) {
                            const int y_in = stride * y + ky - pad;
                            const int x_in = stride * x + kx - pad;

                            if (y_in < 0 || y_in >= height_in || x_in < 0 || x_in >= width_in) {
                                continue;
                            }
                            max_val = std::max(max_val, in_tensor.get_3d(c, y_in, x_in));
                        }
                    }
                    *p_out = max_val;
                    p_out++;
                }
            }
        }
    }

    void maxpool2d(const Tensor &in_tensor,
                   Tensor &out_tensor,
                   const int kernel,
                   const int pad,
                   const int stride) {
        const int channel = in_tensor.shape[0];
        const int height_in = in_tensor.shape[1];
        const int width_in = in_tensor.shape[2];
        const int height_out = (height_in + 2 * pad - kernel) / stride + 1;
        const int width_out = (width_in + 2 * pad - kernel) / stride + 1;

        if (out_tensor.shape != std::vector<int>({channel, height_out, width_out})) {
            throw std::runtime_error("Invalid output shape");
        }

        // start worker threads
        std::vector<std::thread> workers;
        for (const auto &[c_begin, c_end]: compute_shards(channel, 12)) {
            workers.emplace_back(sharded_maxpool,
                                 std::cref(in_tensor),
                                 std::ref(out_tensor),
                                 kernel,
                                 pad,
                                 stride,
                                 c_begin,
                                 c_end);
        }

        // wait for the workers to finish
        for (auto &t: workers) {
            t.join();
        }
    }
}
