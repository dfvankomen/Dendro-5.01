#!/usr/bin/env bash
# build the pinned sources on a cluster login node. edit the module block for the site.
set -euo pipefail
ROOT=${ROOT:-$PWD/repart_bundle}
FORK_URL=${FORK_URL:-git@github.com:dfvankomen/Dendro-5.01.git}
FORK_REV=${FORK_REV:-93c4b526828012f19278ff329b9c2ad2f19a07d8}
FASTPART_URL=${FASTPART_URL:-git@github.com:paralab/fastpart.git}
FASTPART_REV=${FASTPART_REV:-dendro-refine}
EM4_URL=${EM4_URL:?set EM4_URL}     # em4_base_cpy remote, branch repartitioning (+ sandwich fixes)
NLSM_URL=${NLSM_URL:?set NLSM_URL}  # nlsm-gr remote, branch lts-graph-fix
JOBS=${JOBS:-16}

# --- site modules -------------------------------------------------------
# module load gcc openmpi cmake        # BYU / CHPC: fill in
# ------------------------------------------------------------------------
mkdir -p "$ROOT" && cd "$ROOT"
[ -d fastpart ] || git clone -q "$FASTPART_URL" fastpart
git -C fastpart checkout -q "$FASTPART_REV"
cmake -S fastpart -B fastpart/build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build fastpart/build --target install -j"$JOBS" >/dev/null
FP=$ROOT/fastpart/build/install

[ -d dendrolib ] || git clone -q "$FORK_URL" dendrolib
git -C dendrolib checkout -q "$FORK_REV"

for s in em4 nlsm; do
  url=$([ $s = em4 ] && echo "$EM4_URL" || echo "$NLSM_URL")
  br=$([ $s = em4 ] && echo repartitioning || echo lts-graph-fix)
  [ -d $s ] || git clone -q "$url" $s
  git -C $s checkout -q "$br"
  cmake -S $s -B $s/build -DCMAKE_BUILD_TYPE=Release \
    -DDENDRO_dendrolib_DIR="$ROOT/dendrolib" \
    -DFASTPART_INCLUDE_DIR="$FP/include" -DFASTPART_LIB_DIR="$FP/lib" \
    $([ $s = nlsm ] && echo "-DNLSM_COMPARE_WITH_ANALYTICAL_SOL=ON -DNLSM_WITH_GRAPH_PARTITIONING=ON") \
    $([ $s = em4 ] && echo "-DEM4_WITH_GRAPH_PARTITIONING=ON") >/dev/null
done
cmake --build em4/build  -j"$JOBS" --target em4Solver
cmake --build nlsm/build -j"$JOBS" --target nlsmNUTS
echo "built: em4/build/solver/em4Solver nlsm/build/solver/nlsmNUTS  (fastpart $FASTPART_REV, fork $FORK_REV)"
