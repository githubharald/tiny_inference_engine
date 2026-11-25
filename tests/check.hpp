#ifndef CPP_CHECK_H
#define CPP_CHECK_H

#include <iostream>
#include <cmath>

#define CHECK(...) \
    do { \
        if (!(__VA_ARGS__)) { \
            std::cerr << "FAIL: " #__VA_ARGS__ " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
            std::exit(1); \
        } \
    } while (0)

bool is_close(float a, float b) {
    return std::abs(a-b)<0.0001;
}

#endif
