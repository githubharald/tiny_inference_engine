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

    std::vector<std::pair<int, int> > compute_shards(int num_elements, int max_shards) {
        const int num_shards = std::min(num_elements, max_shards);
        const int shard_size = num_elements % num_shards == 0
                                   ? num_elements / num_shards
                                   : num_elements / num_shards + 1;

        std::vector<std::pair<int, int> > shards;
        for (int shard = 0; shard < num_shards; shard++) {
            const int begin = shard * shard_size;
            const int end = std::min(begin + shard_size, num_elements);
            shards.emplace_back(begin, end);
        }
        return shards;
    }
}
