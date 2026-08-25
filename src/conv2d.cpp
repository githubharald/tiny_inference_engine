#include "conv2d.h"

#include "core.h"

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

    void gemm_based_conv(Tensor &out_tensor,
                         Tensor &img_patch_mat,
                         Tensor &kernel_mat,
                         const Tensor &kernel,
                         const Tensor &in_tensor,
                         const Tensor &bias,
                         const int stride
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
        const int num_shards = std::min({12, height_out});
        const int shard_size = height_out % num_shards == 0
                                   ? height_out / num_shards
                                   : height_out / num_shards + 1;
        for (int shard = 0; shard < num_shards; shard++) {
            const int y_begin = shard * shard_size;
            const int y_end = std::min({y_begin + shard_size, height_out});

            workers.emplace_back(sharded_img_to_matrix,
                                 std::ref(img_patch_mat),
                                 std::cref(out_tensor),
                                 std::cref(kernel),
                                 std::cref(in_tensor),
                                 stride,
                                 y_begin,
                                 y_end);
        }

        // wait for the to finish
        for (auto &t: workers) {
            t.join();
        }

        //Tensor out_mat = zeros({channel_out, height_out * width_out});
        const auto orig_shape = out_tensor.shape;
        out_tensor.shape = {channel_out, height_out * width_out};
        mat_mat_mul(out_tensor, kernel_mat, false, img_patch_mat, true, bias);
        out_tensor.shape = orig_shape;
    }

    void sharded_strided_conv(Tensor &out_tensor,
                              const Tensor &in_tensor,
                              const Tensor &kernel,
                              const Tensor &bias,
                              const int c_out_begin,
                              const int c_out_end,
                              const int stride) {
        // less optimized convolution, but supports non-unit strides
        const int channel_in = kernel.shape[1];
        const int kernel_height = kernel.shape[2];
        const int kernel_width = kernel.shape[3];
        const int kernel_offset_y = kernel_height / 2;
        const int kernel_offset_x = kernel_width / 2;
        const int height_in = in_tensor.shape[1];
        const int width_in = in_tensor.shape[2];
        const int height_out = out_tensor.shape[1];
        const int width_out = out_tensor.shape[2];

        // apply bias
        for (int c_out = c_out_begin; c_out < c_out_end; c_out++) {
            for (int y = 0; y < height_out; y++) {
                for (int x = 0; x < width_out; x++) {
                    out_tensor.get_3d(c_out, y, x) = bias.get_1d(c_out);
                }
            }
        }

        // apply kernel
        for (int c_out = c_out_begin; c_out < c_out_end; c_out++) {
            for (int c_in = 0; c_in < channel_in; c_in++) {
                for (int y = 0; y < height_out; y++) {
                    for (int x = 0; x < width_out; x++) {
                        for (int ky = 0; ky < kernel_height; ky++) {
                            for (int kx = 0; kx < kernel_width; kx++) {
                                const int y_in = stride * y + ky - kernel_offset_y;
                                const int x_in = stride * x + kx - kernel_offset_x;

                                if (0 <= y_in && y_in < height_in && 0 <= x_in && x_in < width_in) {
                                    float val = in_tensor.get_3d(c_in, y_in, x_in);
                                    val *= kernel.get_4d(c_out, c_in, ky, kx);
                                    out_tensor.get_3d(c_out, y, x) += val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    void sharded_conv(Tensor &out_tensor,
                      const Tensor &in_tensor,
                      const Tensor &kernel,
                      const Tensor &bias,
                      const int c_out_begin,
                      const int c_out_end
    ) {
        // highly optimized convolution, but only supports unit strides
        const int channel_in = kernel.shape[1];
        const int kernel_height = kernel.shape[2];
        const int kernel_width = kernel.shape[3];
        const int offset_y = kernel_height / 2;
        const int offset_x = kernel_width / 2;
        const int height = in_tensor.shape[1];
        const int width = in_tensor.shape[2];

        // apply bias
        for (int c_out = c_out_begin; c_out < c_out_end; c_out++) {
            const float bias_val = bias.get(c_out);
            for (int i = c_out * height * width; i < c_out * height * width + height * width; i++) {
                out_tensor.get(i) = bias_val;
            }
        }

        // apply kernel
        auto *out_data = out_tensor.data.data();
        const auto *in_data = in_tensor.data.data();
        for (int c_out = c_out_begin; c_out < c_out_end; c_out++) {
            for (int c_in = 0; c_in < channel_in; c_in++) {
                for (int ky = 0; ky < kernel_height; ky++) {
                    const int ky_offset = ky - offset_y;
                    for (int kx = 0; kx < kernel_width; kx++) {
                        const int kx_offset = kx - offset_x;
                        const float kernel_val = kernel.get_4d(c_out, c_in, ky, kx);
                        auto *p_out = out_data + c_out * height * width;
                        const auto *p_in = in_data + (c_in * height + ky_offset) * width + kx_offset;
                        for (int y = 0; y < height; y++) {
                            const int y_in = y + ky_offset;
                            if (y_in < 0 || y_in >= height) [[unlikely]] {
                                p_out += width;
                                p_in += width;
                                continue;
                            }
                            for (int x = 0; x < width; x++, p_out++, p_in++) {
                                const auto x_in = x + kx_offset;
                                if (x_in < 0 || x_in >= width) [[unlikely]]{
                                    continue;
                                }

                                *p_out += kernel_val * *p_in;
                            }
                        }
                    }
                }
            }
        }
    }
}
