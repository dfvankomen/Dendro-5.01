Continue the repartitioning work on the dendrolib_dfvk fork (jj on git).

Start by reading docs/handoff_2026-09-26_sandwich_next.md, then docs/plan_2026-09-26_sandwich_cuts.md
and docs/plan_2026-09-25_kingspeak.md, then /recall the Ontos vault (Repartitioning: Index, Conventions
and Constraints, Open Questions, Next Session Handoff). Do not re-derive anything those say.

Standing rules: local runs only, and Kingspeak (CHPC) only under the setup in the handoff — no BYU, no
new cluster without my OK. jj commits allowed in dendrolib_dfvk only, one logical change per commit,
subject-only messages (feat/fix/perf/test/docs(scope):), never push. Other repos: edit only, I commit.
New behaviour goes behind a DENDRO_* knob with the default byte-identical; gate with the knob on, then
flip the default and gate the defaults path; write a plan to docs/plan_*.md before executing; /save at
milestones. Every gate must include the ragged arm and the SFC ENUTS arm. Never edit sources while a
build compiles them; never put a process pattern on a command line; never edit a running script; kill
by PID from a script file. Report deterministic metrics (bytes, counters, phase splits, checksums);
wall-clock only from Kingspeak, arms interleaved inside one job, ratio within a round. When a gate
fails, revert the default and keep the change opt-in with the evidence rather than narrowing the gate.
State plainly what every ratio compares (fork-before vs fork-after is not "vs SFC").

Task 1 — price a case where repartitioning can win (the symmetric EM4 case can only tie). (a) Size an
LTS case locally: nlsmNUTS, depth 10–11, tolerance tightened until the first LTS partition has ≥ 100
elements per rank at np=48, gated to 18 digits SFC vs levelwise+refine (DENDRO_GRAPH_REFINE=2) at np=4
and np=8; then the Kingspeak ladder (48/96/192) reading lts_substep_eff, dump_est_speedup, LD Bal and
wall. (b) An asymmetric global-RK case: the puncture mesh (PARTEVOLVE_INIT=puncture or BSSN q1 with
BH-driven refinement, ≥ 500 elements per rank) with DENDRO_GRAPH_REFINE=3 + DENDRO_GRAPH_VTX_WEIGHTED=2
+ DENDRO_GRAPH_BALANCE_ROUNDS vs SFC, reading padded_vol_imbalance and wall. Both need the one-block
floor per rank in the block vote first if the ragged arm hits the zero-element guard at scale.

Task 2 — the sandwich, in the handoff's order: the transport's directory rendezvous and orphan
Allgatherv in redistributeVec (both g2s and s2g), then the scatter-map double build, then inherited
labels. Each behind a knob, each gated as above, each re-priced on Kingspeak with the EXP=S-style
profile pair.

Leave Design A (remesh without the twin) alone unless Task 2 leaves more than ~0.5 s per remesh on
the table at 192 ranks. Keep the vault current as you go (log, roadmap, index, the relevant concept
or experiment note), and write the next handoff before you stop.
