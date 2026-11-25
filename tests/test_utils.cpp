#include "check.hpp"
#include "utils.h"


using namespace inference_engine;


void test_utils_tokenize() {
    auto res = tokenize("1,2,3"); // no comma at end
    CHECK(res==std::vector<std::string>{"1", "2", "3"});

    res = tokenize("1,2,3,"); // comma at end
    CHECK(res==std::vector<std::string>{"1", "2", "3"});
}

void test_sharding() {
    auto res = compute_shards(1, 1);
    CHECK(res.size()==1);
    CHECK(res[0].first==0);
    CHECK(res[0].second==1);

    res = compute_shards(5, 1);
    CHECK(res.size()==1);
    CHECK(res[0].first==0);
    CHECK(res[0].second==5);

    res = compute_shards(1, 5);
    CHECK(res.size()==1);
    CHECK(res[0].first==0);
    CHECK(res[0].second==1);

    res = compute_shards(10, 2);
    CHECK(res.size()==2);
    CHECK(res[0].first==0);
    CHECK(res[0].second==5);
    CHECK(res[1].first==5);
    CHECK(res[1].second==10);

    res = compute_shards(11, 2);
    CHECK(res.size()==2);
    CHECK(res[0].first==0);
    CHECK(res[0].second==6);
    CHECK(res[1].first==6);
    CHECK(res[1].second==11);
}


int main() {
    test_utils_tokenize();
    test_sharding();
    return 0;
}
