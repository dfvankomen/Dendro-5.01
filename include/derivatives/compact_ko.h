#pragma once

namespace dendroderivs {

/// Supported Compact Kreiss-Oliger schemes. No accuracy order is implied.
enum class CompactKOScheme { Radius1 };

/// Returns the stencil radius; throws std::invalid_argument for unknown
/// schemes.
unsigned int compact_ko_radius(CompactKOScheme scheme);

/** Add Compact KO to one unzipped field's RHS, without allocating or modifying
 * the field/gradients. Arrays contain sz[0]*sz[1]*sz[2] doubles, x fastest.
 * rhs must not overlap any input. Spacings must be finite and positive.
 *
 * Field interior is [padding_width, sz[d]-padding_width). All three supplied
 * gradients must be valid in [gradient_padding_width,
 * sz[d]-gradient_padding_width) in every dimension. The latter width must be
 * <= padding_width. Each directional term is added only where its entire
 * stencil lies in that valid region. Field ghosts alone do NOT imply valid
 * gradient ghosts: callers must compute/exchange gradients before claiming
 * a smaller gradient_padding_width. This also applies at coarse/fine AMR
 * interfaces; this routine performs no interpolation or communication.
 *
 * Physical boundary planes (Dendro OCT_DIR_* bits in bflag) remain untouched;
 * no one-sided Compact KO closure is claimed. Internal faces have no special
 * exclusion: with radius-one gradient halos they receive all three terms.
 * With interior-only gradients their normal term is omitted on one layer,
 * while valid tangential terms are retained. Empty interiors are no-ops.
 */
void add_compact_ko(double* rhs, const double* u, const double* du_x,
                    const double* du_y, const double* du_z,
                    const unsigned int* sz, double hx, double hy, double hz,
                    unsigned int padding_width,
                    unsigned int gradient_padding_width, unsigned int bflag,
                    double sigma, CompactKOScheme scheme);

/** Same contract as add_compact_ko, but requires padding_width >= 2.
 * Where a directional gradient stencil is unavailable, add centered radius-two
 * KO2 / 3 using valid field ghosts. Caller guarantees nonphysical field ghosts
 * are populated, including AMR transfer. Physical planes remain untouched and
 * fallback stencils never cross a physical face (including on tiny blocks).
 * A direction with neither valid stencil is left unchanged. No gradient is
 * overwritten; no element-order-dependent explicit filter is used.
 */
void add_hybrid_ko(double* rhs, const double* u, const double* du_x,
                    const double* du_y, const double* du_z,
                    const unsigned int* sz, double hx, double hy, double hz,
                    unsigned int padding_width,
                    unsigned int gradient_padding_width, unsigned int bflag,
                    double sigma, CompactKOScheme scheme);

}  // namespace dendroderivs
