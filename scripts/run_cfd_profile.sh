#!/usr/bin/env bash
# Compact-FD kernel profiler (methods paper, BSSN-free).
#
# Builds testCompactDerivs at up to three SIMD levels -- no-SIMD (scalar SSE2),
# AVX2, AVX-512 -- each with identical config (only DENDRO_CPU_ARCH differs),
# then sweeps block size / fusion / nvar and tabulates per-call time for the
# LIBXSMM matrix-form "compact" path vs the production explicit kernels.
# Variants the host CPU can't run are skipped (never fatal).
#
# Run `./scripts/run_cfd_profile.sh --help` for usage.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

usage() {
  cat <<EOF
compact-FD kernel profiler -- builds + runs the no-SIMD / AVX2 / AVX-512 sweep.

USAGE
  ./scripts/run_cfd_profile.sh [--help]

  All options are environment variables (defaults in brackets):
    VARIANTS    ISA labels to build/run     [nosimd avx2 avx512]
    ELEORDERS   element orders (comma)       [4,6,8]  (E4/E6/E8 have explicit
                schemes; higher orders have no explicit baseline to compare to)
    NVARS       field-var counts (comma)     [1]   e.g. NVARS=1,4,12
    JOBS        build parallelism            [nproc]
    TOML        base param file              [scripts/cfd_perf_params.toml]
    CSV         output data csv              [scripts/cfd_perf_data.csv]
    OUTDIR      plot output dir              [scripts/cfd_perf_plots]
    BUILD_TYPE  CMake build type             [Release]
    FORCE_ISA   build a variant even if host lacks the ISA  [0]
    CMAKE_EXTRA extra args for every cmake configure         []

  Low-noise timing -- every run is pinned to one core, single-threaded:
    PIN_CORE    core id for taskset  [last core in the process's affinity set]
    CFD_LAUNCH  full launcher override (HPC), e.g.
                "numactl --physcpubind=2 --membind=0"  or  "srun --cpu-bind=cores -n1"
                (empty string disables pinning)

  HPC:
    MODULES     modules to 'module load' before building/running, e.g.
                MODULES="gcc/15.1.0 intel-oneapi-mkl/2025.3.1"
    For offline compute nodes (no internet for libxsmm FetchContent), pass
    CMAKE_EXTRA="-DUSE_LOCAL_XSMM=ON -DLOCAL_XSMM_PATH=/path/to/libxsmm".

EXAMPLES
  ./scripts/run_cfd_profile.sh                      # full default sweep
  VARIANTS="nosimd avx2" ELEORDERS=4,6,8 ./scripts/run_cfd_profile.sh
  NVARS=1,4,12 ./scripts/run_cfd_profile.sh         # add the nvar axis
  ELEORDERS=6 ./scripts/run_cfd_profile.sh          # quick single-order check
  # HPC (inside an salloc/sbatch allocation, exclusive node recommended):
  MODULES="gcc/15.1.0 intel-oneapi-mkl/2025.3.1" PIN_CORE=2 ./scripts/run_cfd_profile.sh

OUTPUT
  <CSV>             tidy per-(isa,eleorder,nvar,fusion,impl) timings
  <OUTDIR>/*.png    isa_bars, block_size_scaling, block_fusion[, nvar_throughput]
  a per-call speedup summary table (compact vs production explicit) on stdout
EOF
}
[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && { usage; exit 0; }

VARIANTS=${VARIANTS:-"nosimd avx2 avx512"}
ELEORDERS=${ELEORDERS:-"4,6,8"}
NVARS=${NVARS:-"1"}
JOBS=${JOBS:-$(nproc)}
TOML=${TOML:-"$HERE/cfd_perf_params.toml"}
CSV=${CSV:-"$HERE/cfd_perf_data.csv"}
OUTDIR=${OUTDIR:-"$HERE/cfd_perf_plots"}
BUILD_TYPE=${BUILD_TYPE:-Release}
FORCE_ISA=${FORCE_ISA:-0}
CMAKE_EXTRA=${CMAKE_EXTRA:-}
MODULES=${MODULES:-}
PYTHON=${PYTHON:-python3}

# HPC: load environment modules (compiler, MKL, ...) if requested.
if [[ -n "$MODULES" ]] && command -v module >/dev/null 2>&1; then
  echo "## module load $MODULES"
  # shellcheck disable=SC2086
  module load $MODULES || { echo "ERROR: module load failed" >&2; exit 1; }
fi

declare -A ARCH=([nosimd]="nosimd" [avx2]="generic_avx2" [avx512]="icelake-server")
declare -A NEEDS=([nosimd]="" [avx2]="avx2" [avx512]="avx512f")  # /proc/cpuinfo flag

command -v cmake >/dev/null || { echo "ERROR: cmake not found" >&2; exit 1; }
[[ -f "$TOML" ]] || { echo "ERROR: param file $TOML missing" >&2; exit 1; }

CPUFLAGS="$(grep -m1 '^flags' /proc/cpuinfo 2>/dev/null || true)"
host_has() { [[ -z "$1" ]] && return 0; grep -qw "$1" <<<"$CPUFLAGS"; }

echo "compact-FD profiler | host: avx2=$(host_has avx2 && echo y || echo n) avx512=$(host_has avx512f && echo y || echo n) | variants: $VARIANTS"

built=()
for v in $VARIANTS; do
  arch="${ARCH[$v]-}"
  [[ -z "$arch" ]] && { echo "  $v: unknown variant -- skipping"; continue; }
  if [[ "$FORCE_ISA" != "1" ]] && ! host_has "${NEEDS[$v]}"; then
    echo "  $v: skipped (host lacks '${NEEDS[$v]}'; set FORCE_ISA=1 to cross-build)"
    continue
  fi
  bdir="$REPO/build_$v"; bin="$bdir/testCompactDerivs"
  if [[ -x "$bin" ]]; then
    echo "  $v: reusing $bdir"
  else
    printf "  %s: building (logs -> %s) ... " "$v" "$(basename "$bdir").build.log"
    # shellcheck disable=SC2086
    if cmake -S "$REPO" -B "$bdir" -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
             -DDENDRO_CPU_ARCH="$arch" -DBUILD_DENDRO_EXAMPLES=ON $CMAKE_EXTRA \
             >"$bdir.cfg.log" 2>&1 \
       && cmake --build "$bdir" --target testCompactDerivs -j"$JOBS" \
             >"$bdir.build.log" 2>&1; then
      echo "ok"
    else
      echo "FAILED"; tail -8 "$bdir.build.log" >&2; continue
    fi
  fi
  built+=("$v")
done

[[ ${#built[@]} -eq 0 ]] && { echo "no variants built -- nothing to profile." >&2; exit 1; }

# cfd_perf_compare.py auto-detects build_<isa>/testCompactDerivs, skips absent
# ones, and pins LIBXSMM_TARGET per ISA so the matrix-form JIT matches the build.
"$PYTHON" "$HERE/cfd_perf_compare.py" all \
    --toml "$TOML" --csv "$CSV" --out "$OUTDIR" \
    --eleorders "$ELEORDERS" --nvars "$NVARS"

echo ""
echo "done. data -> $CSV"
echo "      plots -> $OUTDIR/"
echo "re-run / tune: ./scripts/run_cfd_profile.sh --help"
