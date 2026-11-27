#!/usr/bin/env bash
# Pre-merge tier: before a merge to experimental or a production campaign (CI
# runs only `ctest -LE premerge`). Stages: Release full ctest (premerge label
# included), ASan+UBSan per-push ctest, C++17 build. Exits nonzero on any fail.
#     scripts/premerge.sh [build-root=build-premerge] [jobs=8]

set -uo pipefail

root="${1:-build-premerge}"
jobs="${2:-8}"
src="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$root"
root="$(cd "$root" && pwd)"
san="-O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined"
fail=0

# Open MPI's memory patcher conflicts with ASan; also needed at build time,
# because doctest discovery runs the test binaries
export OMPI_MCA_memory=^patcher
# oversubscribe when a test asks for more ranks than cores (Open MPI 5 form)
export PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe

stage() {  # stage <name> <build-dir> <ctest args> -- <cmake args>
    local name=$1 dir=$2 ctest_args=$3
    shift 4
    echo "== $name"
    if ! cmake -S "$src" -B "$dir" -D BUILD_DENDRO_EXAMPLES=ON "$@" \
        > "$dir.cfg.log" 2>&1; then
        echo "   configure failed: $dir.cfg.log"; fail=1; return
    fi
    if ! cmake --build "$dir" --parallel "$jobs" > "$dir.build.log" 2>&1; then
        echo "   build failed: $dir.build.log"; fail=1; return
    fi
    [ "$ctest_args" = none ] && { echo "   built"; return; }
    # shellcheck disable=SC2086
    if (cd "$dir" && ctest --output-on-failure $ctest_args) > "$dir.ctest.log" 2>&1; then
        echo "   $(grep -E 'tests passed' "$dir.ctest.log")"
    else
        echo "   FAILED: $(grep -E 'tests passed|tests failed' "$dir.ctest.log")"
        grep -E '\*\*\*|Not Run' "$dir.ctest.log" | sed 's/^/     /'
        fail=1
    fi
}

stage "Release, full ctest" "$root/release" "--timeout 900" -- \
    -D CMAKE_BUILD_TYPE=Release

ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1:suppressions=$src/scripts/ubsan.supp \
stage "ASan+UBSan, per-push tests" "$root/sanitize" "-LE premerge --timeout 1800" -- \
    -D CMAKE_BUILD_TYPE=None "-DCMAKE_CXX_FLAGS=$san" "-DCMAKE_C_FLAGS=$san" \
    "-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address,undefined"

stage "C++17 build" "$root/cxx17" none -- \
    -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_STANDARD=17

[ $fail -eq 0 ] && echo "premerge: PASS" || echo "premerge: FAIL"
exit $fail
