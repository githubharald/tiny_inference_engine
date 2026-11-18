#include "conv2d.h"

#include "core.h"
#include "utils.h"

#include <algorithm>
#include <thread>

namespace inference_engine {
    void sharded_img_to_matrix(
        Tensor &img_patch_mat,
        const Tensor &out_tensor,
        const Tensor &kernel,
        const Tensor &in_tensor,
        const int stride,
        const int y_begin,
        const int y_end
    ) {
        const int channel_in = kernel.shape[1];
        const int kernel_height = kernel.shape[2];
        const int kernel_width = kernel.shape[3];
        const int kernel_offset_y = kernel_height / 2;
        const int kernel_offset_x = kernel_width / 2;
        const int height_in = in_tensor.shape[1];
        const int width_in = in_tensor.shape[2];
        const int width_out = out_tensor.shape[2];

        for (int y = y_begin; y < y_end; y++) {
            for (int x = 0; x < width_out; x++) {
                int element_idx = 0;
                for (int c_in = 0; c_in < channel_in; c_in++) {
                    for (int ky = 0; ky < kernel_height; ky++) {
                        for (int kx = 0; kx < kernel_width; kx++, element_idx++) {
                            const int y_in = stride * y + ky - kernel_offset_y;
                            const int x_in = stride * x + kx - kernel_offset_x;

                            float val = 0;
                            if (y_in >= 0 && y_in < height_in && x_in >= 0 && x_in < width_in)[[likely]] {
                                val = in_tensor.get_3d(c_in, y_in, x_in);
                            }
                            img_patch_mat.get_2d(y * width_out + x, element_idx) = val;
                        }
                    }
                }
            }
        }
    }

    int conv2d(Tensor &out_tensor,
               Tensor &img_patch_mat,
               Tensor &kernel_mat,
               const Tensor &kernel,
               const Tensor &in_tensor,
               const Tensor &bias,
               const int stride,
               const bool apply_relu,
               const int max_num_shards
    ) {
        // see https://sahnimanas.github.io/post/anatomy-of-a-high-performance-convolution/
        const int channel_out = kernel.shape[0];
        const int channel_in = kernel.shape[1];
        const int kernel_height = kernel.shape[2];
        const int kernel_width = kernel.shape[3];
        const int height_out = out_tensor.shape[1];
        const int width_out = out_tensor.shape[2];

        if (kernel_mat.empty()) {
            kernel_mat.data = kernel.data;
            kernel_mat.shape = {channel_out, kernel.shape[1] * kernel.shape[2] * kernel.shape[3]};
        }

        // image as matrix
        if (img_patch_mat.empty()) {
            img_patch_mat = std::move(zeros({height_out * width_out, channel_in * kernel_height * kernel_width}));
        }

        // start worker threads
        std::vector<std::thread> workers;
        for (const auto &[y_begin, y_end]: compute_shards(height_out, max_num_shards)) {
            workers.emplace_back(sharded_img_to_matrix,
                                 std::ref(img_patch_mat),
                                 std::cref(out_tensor),
                                 std::cref(kernel),
                                 std::cref(in_tensor),
                                 stride,
                                 y_begin,
                                 y_end);
        }

        // wait for the workers to finish
        for (auto &t: workers) {
            t.join();
        }

        const auto orig_shape = out_tensor.shape;
        out_tensor.shape = {channel_out, height_out * width_out};
        mat_mat_mul(out_tensor, kernel_mat, false, img_patch_mat, true, bias, apply_relu);
        out_tensor.shape = orig_shape;

        // return maximum possible number of shards that can be set
        return height_out;
    }
}
