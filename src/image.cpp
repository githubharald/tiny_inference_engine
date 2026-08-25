#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "image.h"

namespace inference_engine {
    Tensor load_image(const std::string &fn) {
        int width = 0, height = 0, num_channels = 0;
        unsigned char *data = stbi_load(fn.c_str(), &width, &height, &num_channels, 0);
        if (!(num_channels == 3 || num_channels == 1) || !data) {
            stbi_image_free(data);
            throw std::runtime_error("Could not load image");
        }

        Tensor res{zeros({num_channels, height, width})};
        auto p_data = data; // byte format: RGBRGBRGB...
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                for (int c = 0; c < num_channels; c++) {
                    res.get_3d(c, y, x) = static_cast<float>(*p_data) / 255.0f - 0.5f;
                    p_data++;
                }
            }
        }

        stbi_image_free(data);
        return res;
    }
}
