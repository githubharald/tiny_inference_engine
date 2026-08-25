#ifndef CPP_UTILS_H
#define CPP_UTILS_H

#include <string>
#include <vector>

namespace inference_engine {
    /// split a comma separated string
    std::vector<std::string> tokenize(const std::string &s);
}
#endif //CPP_UTILS_H
