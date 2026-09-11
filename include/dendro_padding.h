#pragma once
/**
 * @file dendro_padding.h
 * @brief Single source of truth for the block padding width.
 *
 * DENDRO_WIDE_PADDING (CMake option, OFF by default) is the prototype flag for
 * "one extra ghost ring per block face". With it ON, every unzipped block has
 * pw = eleOrder/2 + DENDRO_WIDE_PADDING_EXTRA padding points per face instead
 * of eleOrder/2. The extra ring is filled from data that is already on-rank
 * (same-level neighbour elements carry eleOrder+1 nodes, coarser neighbours are
 * prolongated to 2*eleOrder+1), so it costs no additional communication. The
 * only face that cannot supply the extra ring is one abutting a FINER octant
 * (a finer element only has eleOrder/2 nodes at block spacing); those faces are
 * marked with a per-block "fine face" flag and the compact derivative operator
 * for that face falls back to the eleOrder/2-deep closure (its first row is
 * trimmed). See Block::getBlkFineFaceFlag and DerivMatrixStorage.
 *
 * Revert: build with -DDENDRO_WIDE_PADDING=OFF (default). Every code site is
 * wrapped in `#ifdef DENDRO_WIDE_PADDING` so `grep -rn DENDRO_WIDE_PADDING`
 * finds all of it.
 */

#ifdef DENDRO_WIDE_PADDING

#ifndef DENDRO_WIDE_PADDING_EXTRA
#define DENDRO_WIDE_PADDING_EXTRA 1u
#endif

/// padding width used for block allocation and every consumer of it
#define DENDRO_PAD_WIDTH_FOR_ORDER(eo) (((eo) >> 1u) + DENDRO_WIDE_PADDING_EXTRA)

#else

#define DENDRO_WIDE_PADDING_EXTRA 0u
#define DENDRO_PAD_WIDTH_FOR_ORDER(eo) ((eo) >> 1u)

#endif

/// the depth a FINER neighbour can fill at block spacing (and the depth the
/// wavelet criterion is written for): always eleOrder/2, independent of the flag
#define DENDRO_NATIVE_PAD_WIDTH(eo) ((eo) >> 1u)

/// The block boundary flag (bflag) uses bits [0,6) for the six physical
/// boundary faces (OCT_DIR_LEFT..OCT_DIR_FRONT). The "fine face" flag reuses
/// the same direction indices shifted by DENDRO_FINE_FACE_SHIFT so both can be
/// OR-ed into one word for the derivative dispatch. Solvers must mask with
/// DENDRO_BFLAG_PHYS_MASK before any physical-boundary logic.
#define DENDRO_FINE_FACE_SHIFT 6u
#define DENDRO_BFLAG_PHYS_MASK 0x3Fu
#define DENDRO_FINE_FACE_BIT(dir) (1u << ((dir) + DENDRO_FINE_FACE_SHIFT))
