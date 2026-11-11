// KIMBYU_17 must throw on selection (its coefficients were a copy of
// KIMBYU_3); neighboring schemes must still build.
#include <cstdio>
#include <stdexcept>

#include "derivatives/derivs_factory.h"

using namespace dendroderivs;

static bool builds(const char *name) {
    try {
        return DerivsFactory::create_first_order(name, 8, {}, 0, "none", {}) !=
               nullptr;
    } catch (const std::exception &e) {
        std::printf("%s: %s\n", name, e.what());
        return false;
    }
}

int main() {
    int bad = 0;
    if (builds("KIMBYU_17")) { std::printf("FAIL: KIMBYU_17 built\n"); bad++; }
    for (const char *n : {"KIMBYU_3", "KIMBYU_16", "KIMBYU_18"})
        if (!builds(n)) { std::printf("FAIL: %s did not build\n", n); bad++; }
    std::printf(bad ? "FAIL\n" : "PASS\n");
    return bad ? 1 : 0;
}
