#include "utils.h"

namespace inference_engine {
    std::vector<std::string> tokenize(const std::string &s) {
        std::vector<std::string> res;
        std::string tmp;

        for (const char c: s) {
            if (c == ',') {
                res.push_back(tmp);
                tmp.clear();
            } else {
                tmp.push_back(c);
            }
        }

        if (!tmp.empty()) {
            res.push_back(tmp);
        }
        return res;
    }
}
