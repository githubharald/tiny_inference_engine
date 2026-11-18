#ifndef CPP_UTILS_H
#define CPP_UTILS_H

#include <string>
#include <vector>
#include <utility>

namespace inference_engine {
    /// split a comma separated string
    std::vector<std::string> tokenize(const std::string &s);

    /// split work into shards
    std::vector<std::pair<int, int> > compute_shards(int num_elements, int max_shards);
}
#endif //CPP_UTILS_H
