#!/usr/bin/env python3
"""Parse + compare partition-quality records emitted by edgeCutCompare.

Reads JSON Lines (one record per config x partitioner x adjacency-graph) from
files or directories and prints comparison tables. Stdlib only -- cluster nodes
often have no pandas.

The headline question this answers
----------------------------------
fastpart's `dgraph_from_octree` builds edges from e2e[6] -- FACE neighbours only.
But Dendro's real ghost traffic follows block padding, which reaches EDGE and
CORNER neighbours too. So we score every labelling under BOTH graphs:

    graph=face6      the objective fastpart actually minimises
    graph=stencil26  the objective that (approximately) costs real money

If fastpart WINS on face6 but LOSES on stencil26, it is optimising the wrong
graph -- and DENDRO_GRAPH_FULL_STENCIL already exists to fix it. That is the
smoking gun; `--verdict` calls it explicitly.

Usage
-----
    ./analyze_partition_quality.py results/                 # summary tables
    ./analyze_partition_quality.py results/ --verdict        # + the 2x2 call
    ./analyze_partition_quality.py results/ --csv out.csv    # flat CSV
    ./analyze_partition_quality.py results/ --group max_depth npes
    ./analyze_partition_quality.py a.jsonl b.jsonl --baseline sfc
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

# metrics where LOWER is better -> ratio < 1 means the partitioner beat baseline
LOWER_IS_BETTER = {
    "edge_cut",
    "edge_cut_frac",
    "total_comm_volume",
    "boundary_vertices",
    "ghost_surface_nodes",
    "max_rank_cut",
    "ele_imbalance",
    "work_imbalance",
}

# what we print, in order, when the record has them
DEFAULT_METRICS = [
    "edge_cut",
    "edge_cut_frac",
    "total_comm_volume",
    "boundary_vertices",
    "ghost_surface_nodes",
    "ele_imbalance",
    "work_imbalance",
]

CONFIG_KEYS = ["npes", "max_depth", "wavelet_tol", "grain_sz", "partition_tol", "label"]


# ---------------------------------------------------------------- loading


def iter_jsonl_files(paths):
    for p in paths:
        if os.path.isdir(p):
            for root, _dirs, files in os.walk(p):
                for fn in sorted(files):
                    if fn.endswith((".jsonl", ".json", ".ndjson")):
                        yield os.path.join(root, fn)
        else:
            yield p


def load_records(paths, strict=False):
    """Return (records, problems). Tolerates junk lines -- MPI stdout gets mixed
    into logs constantly, so a stray line must not kill the analysis."""
    records, problems = [], []
    for path in iter_jsonl_files(paths):
        try:
            with open(path, "r") as fh:
                for lineno, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    # tolerate a prefix before the JSON (rank tags etc.)
                    if not line.startswith("{"):
                        brace = line.find("{")
                        if brace < 0:
                            continue
                        line = line[brace:]
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError as e:
                        problems.append(f"{path}:{lineno}: {e}")
                        continue
                    if not isinstance(rec, dict) or "metrics" not in rec:
                        continue
                    rec["_src"] = f"{path}:{lineno}"
                    records.append(rec)
        except OSError as e:
            problems.append(f"{path}: {e}")
    if strict and problems:
        for p in problems:
            print(f"ERROR {p}", file=sys.stderr)
        sys.exit(2)
    return records, problems


def cfg_of(rec):
    c = rec.get("config", {}) or {}
    return {k: c.get(k) for k in CONFIG_KEYS if c.get(k) is not None}


def cfg_key(rec, group_by):
    c = rec.get("config", {}) or {}
    return tuple((k, c.get(k)) for k in group_by)


def cfg_str(key):
    return " ".join(f"{k}={v}" for k, v in key if v is not None) or "(default)"


# ---------------------------------------------------------------- formatting


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        if v != v or math.isinf(v):
            return "nan"
        if v == 0:
            return "0"
        if abs(v) < 0.01 or abs(v) >= 1e7:
            return f"{v:.3e}"
        return f"{v:.4g}"
    if isinstance(v, int) and abs(v) >= 10000:
        return f"{v:,}"
    return str(v)


def table(headers, rows):
    cols = [list(map(str, col)) for col in zip(*([headers] + rows))] if rows else []
    if not cols:
        return "  (no rows)"
    w = [max(len(x) for x in col) for col in cols]
    out = []
    out.append("  " + "  ".join(h.ljust(w[i]) for i, h in enumerate(headers)))
    out.append("  " + "  ".join("-" * w[i] for i in range(len(headers))))
    for r in rows:
        out.append("  " + "  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))
    return "\n".join(out)


# ---------------------------------------------------------------- analysis


def metrics_present(records):
    seen = set()
    for r in records:
        seen.update(r.get("metrics", {}).keys())
    ordered = [m for m in DEFAULT_METRICS if m in seen]
    ordered += sorted(m for m in seen if m not in DEFAULT_METRICS and not m.startswith("per_rank"))
    return ordered


def summarize(records, group_by, baseline, metrics=None):
    """Print, per config group and per adjacency graph, a partitioner x metric
    table plus ratios against the baseline partitioner."""
    metrics = metrics or metrics_present(records)
    groups = defaultdict(list)
    for r in records:
        groups[cfg_key(r, group_by)].append(r)

    for key in sorted(groups, key=lambda k: [(str(v) if v is not None else "") for _, v in k]):
        recs = groups[key]
        print(f"\n=== {cfg_str(key)} ===")
        by_graph = defaultdict(dict)
        for r in recs:
            g = r.get("graph", "?")
            p = r.get("partitioner", "?")
            # last one wins; edge-cut is deterministic so reps should agree
            by_graph[g][p] = r.get("metrics", {})

        for graph in sorted(by_graph):
            parts = by_graph[graph]
            print(f"\n  [graph = {graph}]")
            headers = ["partitioner"] + metrics
            rows = []
            for p in sorted(parts):
                rows.append([p] + [fmt(parts[p].get(m)) for m in metrics])
            print(table(headers, rows))

            base = parts.get(baseline)
            if not base:
                print(f"    (no '{baseline}' baseline in this group -- ratios skipped)")
                continue
            rrows = []
            for p in sorted(parts):
                if p == baseline:
                    continue
                cells = []
                for m in metrics:
                    a, b = parts[p].get(m), base.get(m)
                    if a is None or b in (None, 0):
                        cells.append("-")
                        continue
                    ratio = a / b
                    mark = ""
                    if m in LOWER_IS_BETTER:
                        mark = " WIN" if ratio < 0.995 else (" LOSS" if ratio > 1.005 else " tie")
                    cells.append(f"{ratio:.3f}x{mark}")
                rrows.append([f"{p} / {baseline}"] + cells)
            if rrows:
                print()
                print(table(["ratio"] + metrics, rrows))


def verdict(records, group_by, baseline, face="face6", stencil="stencil26"):
    """The 2x2: does fastpart win the graph it optimises but lose the one that costs?"""
    print("\n" + "=" * 78)
    print("VERDICT -- is fastpart optimising the wrong graph?")
    print("=" * 78)
    print(
        "\n  fastpart minimises cut on FACE adjacency (dgraph_from_octree uses e2e[6]).\n"
        "  Real ghost traffic follows block padding -> edge/corner neighbours too.\n"
        "  So: win on %s but loss on %s => wrong objective (DENDRO_GRAPH_FULL_STENCIL\n"
        "  is the existing lever). Loss on BOTH => the partitioner is simply weaker\n"
        "  (fastpart has no multilevel coarsening) and dendrolib cannot fix it.\n"
        % (face, stencil)
    )
    groups = defaultdict(list)
    for r in records:
        groups[cfg_key(r, group_by)].append(r)

    any_called = False
    for key in sorted(groups, key=lambda k: [(str(v) if v is not None else "") for _, v in k]):
        recs = groups[key]
        idx = defaultdict(dict)
        for r in recs:
            idx[r.get("graph", "?")][r.get("partitioner", "?")] = r.get("metrics", {})
        parts = {p for g in idx for p in idx[g]} - {baseline}
        for p in sorted(parts):
            got = {}
            for g in (face, stencil):
                a = idx.get(g, {}).get(p, {}).get("edge_cut")
                b = idx.get(g, {}).get(baseline, {}).get("edge_cut")
                got[g] = (a / b) if (a is not None and b) else None
            if got[face] is None and got[stencil] is None:
                continue
            any_called = True
            print(f"\n  {cfg_str(key)}  --  {p} vs {baseline}  (edge_cut ratio, <1 = {p} wins)")
            print(f"    {face:<12} {fmt(got[face])}")
            print(f"    {stencil:<12} {fmt(got[stencil])}")
            f_, s_ = got[face], got[stencil]
            if f_ is None or s_ is None:
                print("    -> incomplete: need both graphs scored for this partitioner")
            elif f_ < 1.0 and s_ > 1.0:
                print("    -> SMOKING GUN: wins its own objective, loses the real one.")
                print("       fastpart is minimising FACE cut while cost follows the full")
                print("       stencil. Fix is DENDRO_GRAPH_FULL_STENCIL (feed it the right")
                print("       graph), not more dendrolib plumbing.")
            elif f_ > 1.0 and s_ > 1.0:
                print("    -> LOSES BOTH: not an objective-mismatch. The partitioner itself is")
                print("       weaker than the curve (no multilevel coarsening). Full-stencil")
                print("       will NOT rescue it; needs a better partitioner / upstream work.")
            elif f_ < 1.0 and s_ < 1.0:
                print("    -> WINS BOTH: cut quality is fine. The comms deficit must come from")
                print("       elsewhere (nodal surface != element cut) -- re-open the plumbing.")
            else:
                print("    -> loses its own objective but wins the real one: surprising;")
                print("       check the graph construction before trusting this.")
    if not any_called:
        print("\n  (nothing to call: need edge_cut for both graphs and a baseline)")


def write_csv(records, path):
    rows, cols = [], []
    for r in records:
        row = dict(cfg_of(r))
        row["partitioner"] = r.get("partitioner")
        row["graph"] = r.get("graph")
        for k, v in (r.get("metrics") or {}).items():
            if isinstance(v, (int, float, str)) or v is None:
                row[k] = v
        rows.append(row)
        for k in row:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"wrote {path}  ({len(rows)} rows x {len(cols)} cols)")


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(
        description="Compare SFC vs fastpart partition quality from edgeCutCompare JSONL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("paths", nargs="+", help="JSONL files or directories to scan")
    ap.add_argument("--baseline", default="sfc", help="partitioner to ratio against (default: sfc)")
    ap.add_argument(
        "--group", nargs="*", default=["npes", "max_depth"], metavar="KEY",
        help="config keys to group by (default: npes max_depth)",
    )
    ap.add_argument("--metrics", nargs="*", default=None, help="restrict to these metrics")
    ap.add_argument("--verdict", action="store_true", help="print the face6-vs-stencil26 call")
    ap.add_argument("--csv", metavar="PATH", help="also write a flat CSV")
    ap.add_argument("--strict", action="store_true", help="fail on any unparseable line")
    args = ap.parse_args()

    records, problems = load_records(args.paths, strict=args.strict)
    if not records:
        print("no records found. Did edgeCutCompare write JSONL to these paths?", file=sys.stderr)
        sys.exit(1)

    parts = sorted({r.get("partitioner", "?") for r in records})
    graphs = sorted({r.get("graph", "?") for r in records})
    print(f"loaded {len(records)} records | partitioners: {', '.join(parts)} | graphs: {', '.join(graphs)}")
    if problems:
        print(f"({len(problems)} unparseable lines skipped; --strict to fail instead)", file=sys.stderr)
    if args.baseline not in parts:
        print(f"WARNING: baseline '{args.baseline}' not present; ratios will be skipped", file=sys.stderr)

    summarize(records, args.group, args.baseline, args.metrics)
    if args.verdict:
        verdict(records, args.group, args.baseline)
    if args.csv:
        write_csv(records, args.csv)


if __name__ == "__main__":
    main()
