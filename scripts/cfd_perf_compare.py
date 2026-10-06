#!/usr/bin/env python3
"""Profile testCompactDerivs across {no-SIMD, AVX2, AVX-512} x eleorder sweep and
plot per-block-size cost for the LIBXSMM matrix-form compact (CFD) path vs the
production explicit stencils ("original") and the class-based explicit path.

This is the methods-paper (BSSN-free) kernel profiler. Three things matter:

  * ISA of the COMPILED explicit kernels is set at build time by
    DENDRO_CPU_ARCH (build_nosimd / build_avx2 / build_avx512).
  * ISA of the LIBXSMM matrix-form (compact) path is chosen by LIBXSMM's JIT
    at RUNTIME from CPUID, *not* the build flags -- so we force it per-run with
    the env var LIBXSMM_TARGET (nosimd->sse, avx2->hsw, avx512->skx).
  * A build whose CPU level the host can't run is simply absent (the runner
    skips building it); collection/plotting tolerate any subset of ISAs.

Usage:
    python cfd_perf_compare.py collect        # run available builds, write csv
    python cfd_perf_compare.py plot           # read csv, write *.png
    python cfd_perf_compare.py all            # both
    python cfd_perf_compare.py all --nvars 1,4,12   # also sweep the nvar axis

Pre-built binaries are expected at build_<isa>/testCompactDerivs.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ISA rungs in increasing width. nosimd is the SSE2 scalar baseline (x86-64
# mandates SSE2; LIBXSMM's lowest JIT target is "sse").
ISA_ORDER = ["nosimd", "avx2", "avx512"]
ISA_BUILD = {isa: REPO_ROOT / f"build_{isa}" / "testCompactDerivs" for isa in ISA_ORDER}
# env LIBXSMM_TARGET that forces the matrix-form JIT to the matching width.
ISA_LIBXSMM_TARGET = {"nosimd": "sse", "avx2": "hsw", "avx512": "skx"}

# Only 4/6/8 have a matching production EXPLICIT scheme (E4/E6/E8, padding 2-4);
# higher orders fall back to E4 and abort (pw out of range). The compact path
# itself handles any even order, but this runner is a compact-vs-explicit compare.
ELEORDERS = [4, 6, 8]
DEFAULT_TOML = REPO_ROOT / "scripts" / "cfd_perf_params.toml"
DATA_CSV = REPO_ROOT / "scripts" / "cfd_perf_data.csv"
OUT_DIR = REPO_ROOT / "scripts" / "cfd_perf_plots"

IMPL_LABEL = {"original": "Explicit (production)", "compact": "JTT6 CFD", "class_based": "Explicit (class)"}
# colors keyed by impl; ISA distinguished by shade/linestyle/marker.
IMPL_COLOR = {"original": "#1f77b4", "class_based": "#7f7f7f", "compact": "#d62728"}
ISA_SHADE = {"nosimd": 0.45, "avx2": 0.72, "avx512": 1.0}  # darker = wider ISA
ISA_MARKER = {"nosimd": "^", "avx2": "o", "avx512": "s"}
ISA_LS = {"nosimd": ":", "avx2": "--", "avx512": "-"}


def available_isas() -> list[str]:
    """ISA rungs whose testCompactDerivs binary actually exists, in width order."""
    return [isa for isa in ISA_ORDER if ISA_BUILD[isa].exists()]


# ---------- cpu pinning (low-noise timing) -------------------------------- #
# Every run is pinned to ONE dedicated core, single-threaded, so the timing
# noise floor stays low and runs are comparable. Controls (env):
#   CFD_LAUNCH  full launcher override, e.g. "numactl --physcpubind=2 --membind=0"
#               or "srun --cpu-bind=cores -n1" on HPC. Empty string disables pinning.
#   PIN_CORE    core id for the default taskset launcher [last core in the
#               process's allowed affinity set -- respects a SLURM cpuset].

def _pin_core() -> str | None:
    core = os.environ.get("PIN_CORE")
    if core:
        return core
    try:  # last allowed core (avoids core 0; honors SLURM/cgroup affinity)
        allowed = sorted(os.sched_getaffinity(0))
        return str(allowed[-1]) if allowed else None
    except (AttributeError, OSError):
        return None


def launch_prefix() -> list[str]:
    custom = os.environ.get("CFD_LAUNCH")
    if custom is not None:          # explicit override (may be empty = no pin)
        return custom.split()
    core = _pin_core()
    if core and shutil.which("taskset"):
        return ["taskset", "-c", core]
    return []


def run_env(isa: str) -> dict:
    """env for a run: pin LIBXSMM ISA + force single-thread, bound OpenMP."""
    env = dict(os.environ)
    env["LIBXSMM_TARGET"] = ISA_LIBXSMM_TARGET[isa]
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OMP_PROC_BIND", "true")
    env.setdefault("OMP_PLACES", "cores")
    return env


# ---------- collect ------------------------------------------------------- #

def make_params(base_toml: Path, eleorder: int, nvar: int) -> Path:
    """drop a temp toml overriding eleorder and nvar."""
    src = base_toml.read_text()
    src = re.sub(r'^"?eleorder"?\s*=\s*\d+', f'"eleorder" = {eleorder}', src, count=1, flags=re.M)
    if re.search(r'^"?nvar"?\s*=\s*\d+', src, flags=re.M):
        src = re.sub(r'^"?nvar"?\s*=\s*\d+', f'"nvar" = {nvar}', src, count=1, flags=re.M)
    else:
        src += f'\n"nvar" = {nvar}\n'
    fd, name = tempfile.mkstemp(suffix=".toml", prefix=f"cfd_eo{eleorder}_nv{nvar}_")
    os.write(fd, src.encode())
    os.close(fd)
    return Path(name)


# the printAbbr=True output looks like:
#   :::xfused, yfused, zfused: 2, 1, 3:::
#   :::nvar: 4:::
#    - Size: 13, 9, 17
#    \ttype,nruns,x_total,x_avg,y_total,y_avg,z_total,z_avg
#    \toriginal   ,2000,...
HEADER_RE = re.compile(r":::xfused,\s*yfused,\s*zfused:\s*(\d+),\s*(\d+),\s*(\d+):::")
NVAR_RE   = re.compile(r":::nvar:\s*(\d+):::")
SIZE_RE   = re.compile(r"-\s*Size:\s*(\d+),\s*(\d+),\s*(\d+)")
ROW_RE    = re.compile(
    r"^\s*(original|class_based|compact)\s*,\s*(\d+),"
    r"\s*([0-9.eE+-]+),\s*([0-9.eE+-]+),"
    r"\s*([0-9.eE+-]+),\s*([0-9.eE+-]+),"
    r"\s*([0-9.eE+-]+),\s*([0-9.eE+-]+)"
)


def parse_sweep(stdout: str, nvar_default: int = 1):
    """yield dicts, one per (xfused,yfused,zfused) x impl. timings are seconds/call."""
    cur = None
    nvar = nvar_default
    for line in stdout.splitlines():
        m = HEADER_RE.search(line)
        if m:
            cur = dict(xfused=int(m.group(1)), yfused=int(m.group(2)), zfused=int(m.group(3)))
            continue
        m = NVAR_RE.search(line)
        if m:
            nvar = int(m.group(1))
            if cur is not None:
                cur["nvar"] = nvar
            continue
        m = SIZE_RE.search(line)
        if m and cur is not None:
            cur["nx"], cur["ny"], cur["nz"] = (int(m.group(i)) for i in (1, 2, 3))
            continue
        stripped = line.replace("\t", "").strip() if line.lstrip().startswith(
            ("original", "class_based", "compact")) else line
        m = ROW_RE.match(stripped)
        if m and cur is not None:
            yield {
                "nvar": cur.get("nvar", nvar),
                **{k: cur[k] for k in cur if k != "nvar"},
                "impl":  m.group(1).strip(),
                "nruns": int(m.group(2)),
                "x_avg": float(m.group(4)),
                "y_avg": float(m.group(6)),
                "z_avg": float(m.group(8)),
            }


def run_one(binary: Path, params_path: Path, isa: str) -> str:
    """run the test pinned to a core, single-threaded, with the ISA's LIBXSMM
    target, return stdout."""
    cmd = [*launch_prefix(), str(binary), str(params_path)]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False, env=run_env(isa))
    if res.returncode != 0:
        sys.stderr.write(res.stdout[-2000:])
        sys.stderr.write(res.stderr[-2000:])
        raise RuntimeError(f"{' '.join(cmd)} failed (rc={res.returncode})")
    return res.stdout


def collect(base_toml: Path, out_csv: Path, eleorders: list[int], nvars: list[int]) -> None:
    isas = available_isas()
    if not isas:
        raise FileNotFoundError(
            f"no testCompactDerivs binaries found under {REPO_ROOT}/build_<isa>/ "
            f"(expected one of: {', '.join(ISA_ORDER)}). Build them first "
            f"(scripts/run_cfd_profile.sh).")
    skipped = [isa for isa in ISA_ORDER if isa not in isas]
    if skipped:
        print(f"## note: no build for {', '.join(skipped)} -- omitting from this report",
              flush=True)

    total = len(isas) * len(eleorders) * len(nvars)
    pin = launch_prefix()
    pin_desc = " ".join(pin) if pin else "(none -- set PIN_CORE/CFD_LAUNCH)"
    print(f"collecting {total} runs: ISAs={','.join(isas)}  "
          f"E={','.join(map(str, eleorders))}  nvar={','.join(map(str, nvars))}", flush=True)
    print(f"  pin: {pin_desc}   OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '1')}", flush=True)
    rows = []
    i = 0
    for isa in isas:
        binary = ISA_BUILD[isa]
        for eo in eleorders:
            for nv in nvars:
                i += 1
                params = make_params(base_toml, eo, nv)
                try:
                    out = run_one(binary, params, isa)
                except RuntimeError as e:
                    print(f"  [{i}/{total}] {isa:<6} E{eo} nvar{nv} -> SKIPPED ({e})", flush=True)
                    continue
                finally:
                    params.unlink(missing_ok=True)
                n_before = len(rows)
                for r in parse_sweep(out, nvar_default=nv):
                    r["isa"] = isa
                    r["eleorder"] = eo
                    rows.append(r)
                print(f"  [{i}/{total}] {isa:<6} E{eo} nvar{nv} "
                      f"(LIBXSMM_TARGET={ISA_LIBXSMM_TARGET[isa]}) -> "
                      f"{len(rows) - n_before} rows", flush=True)

    fields = ["isa", "eleorder", "nvar", "xfused", "yfused", "zfused",
              "nx", "ny", "nz", "impl", "nruns", "x_avg", "y_avg", "z_avg"]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {out_csv}  (ISAs: {', '.join(isas)})")


# ---------- plot ---------------------------------------------------------- #

def load_csv(path: Path):
    out = []
    with path.open() as f:
        for row in csv.DictReader(f):
            for k in ("xfused", "yfused", "zfused", "nx", "ny", "nz", "nruns", "eleorder"):
                row[k] = int(row[k])
            row["nvar"] = int(row.get("nvar", 1) or 1)
            for k in ("x_avg", "y_avg", "z_avg"):
                row[k] = float(row[k])
            row["points"]  = row["nx"] * row["ny"] * row["nz"]
            row["mean_us"] = (row["x_avg"] + row["y_avg"] + row["z_avg"]) / 3.0 * 1e6
            row["ns_per_pt"] = (row["x_avg"] + row["y_avg"] + row["z_avg"]) / 3.0 / row["points"] * 1e9
            out.append(row)
    return out


def _shade(hex_color: str, factor: float) -> str:
    """lighten a hex color toward white as factor decreases (1.0 = unchanged)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * (1 - factor))
    g = int(g + (255 - g) * (1 - factor))
    b = int(b + (255 - b) * (1 - factor))
    return f"#{r:02x}{g:02x}{b:02x}"


def _isas_in(rows) -> list[str]:
    present = {r["isa"] for r in rows}
    return [isa for isa in ISA_ORDER if isa in present]


def _best_isa(isas: list[str]) -> str:
    """highest-width ISA available, used as the normalization baseline."""
    return isas[-1] if isas else "nosimd"


def plot_all(csv_path: Path, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_csv(csv_path)
    if not rows:
        print("no rows to plot")
        return

    isas = _isas_in(rows)
    base_isa = _best_isa(isas)
    eleorders = sorted({r["eleorder"] for r in rows})
    impls = [i for i in ("original", "compact") if any(r["impl"] == i for r in rows)]

    # single-block, single-variable slice for the headline ISA/eleorder figures.
    single = [r for r in rows
              if r["xfused"] == 1 and r["yfused"] == 1 and r["zfused"] == 1 and r["nvar"] == 1]

    def get(rowset, impl, isa, eo, key, default=0.0):
        rec = [r for r in rowset
               if r["impl"] == impl and r["isa"] == isa and r["eleorder"] == eo]
        return rec[0][key] if rec else default

    # ---- plot 1: grouped bars per eleorder; height = per-call time normalized
    # to production-explicit at the best available ISA & that element order ----
    if single:
        fig, ax = plt.subplots(figsize=(9.5, 4.8))
        groups = [(impl, isa) for impl in impls for isa in isas]
        n = len(groups)
        width = 0.8 / max(n, 1)
        x = list(range(len(eleorders)))
        for gi, (impl, isa) in enumerate(groups):
            off = (gi - (n - 1) / 2) * width
            ratios = []
            for eo in eleorders:
                base = get(single, "original", base_isa, eo, "mean_us")
                ratios.append(get(single, impl, isa, eo, "mean_us") / base if base else 0.0)
            xs = [xi + off for xi in x]
            color = _shade(IMPL_COLOR[impl], ISA_SHADE[isa])
            ax.bar(xs, ratios, width=width, color=color, edgecolor="black", linewidth=0.4,
                   label=f"{IMPL_LABEL[impl]} / {isa}")
        ax.axhline(1.0, color="black", lw=0.8, ls="--", alpha=0.5)
        ax.set_xticks(x, [f"E{eo}\n(N={2*eo+1}³)" for eo in eleorders])
        ax.set_xlabel("element order (single block)")
        ax.set_ylabel(f"per-call time / Explicit {base_isa} (same E)")
        ax.set_title("Compact (LIBXSMM) vs production explicit across SIMD levels")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.20),
                  ncol=min(n, 3), frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / "isa_bars.png", dpi=140)
        plt.close(fig)

    # ---- plot 2: block-size SCALING (log-log per-call time vs N=2E+1) ----
    if single:
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        for impl in impls:
            for isa in isas:
                pts = sorted(
                    ((2 * eo + 1, get(single, impl, isa, eo, "mean_us"))
                     for eo in eleorders), key=lambda p: p[0])
                pts = [(nx, t) for nx, t in pts if t > 0]
                if not pts:
                    continue
                xs, ys = zip(*pts)
                color = _shade(IMPL_COLOR[impl], ISA_SHADE[isa])
                ax.plot(xs, ys, marker=ISA_MARKER[isa], ls=ISA_LS[isa], color=color,
                        label=f"{IMPL_LABEL[impl]} / {isa}")
        # reference slopes N^3 (explicit, work ~ points) and N^4 (matrix-form)
        if single:
            n0 = 2 * eleorders[0] + 1
            base_t = get(single, "compact", base_isa, eleorders[0], "mean_us") or 1.0
            ref_x = [2 * eo + 1 for eo in eleorders]
            for p, style in ((3, (0, (1, 1))), (4, (0, (4, 2)))):
                ax.plot(ref_x, [base_t * (nx / n0) ** p for nx in ref_x],
                        color="gray", lw=0.9, ls=style, alpha=0.6,
                        label=f"$N^{p}$ ref")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("block size N (= 2·eleorder + 1)")
        ax.set_ylabel("per-call wall time (µs)")
        ax.set_title("Per-block kernel scaling vs block size")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, ncol=2, frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / "block_size_scaling.png", dpi=140)
        plt.close(fig)

    # ---- plot 3: multi-block FUSION sweep (diagonal xfused==yfused==zfused) ----
    # pick a representative eleorder + the best ISA; show compact vs original.
    eo_fix = 6 if 6 in eleorders else eleorders[len(eleorders) // 2]
    fused = [r for r in rows
             if r["eleorder"] == eo_fix and r["isa"] == base_isa and r["nvar"] == 1
             and r["xfused"] == r["yfused"] == r["zfused"]]
    if fused:
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        for impl in impls:
            pts = sorted(((r["xfused"], r["mean_us"]) for r in fused if r["impl"] == impl))
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", color=IMPL_COLOR[impl], label=IMPL_LABEL[impl])
        ax.set_xlabel(f"fusion level (xfused=yfused=zfused), E{eo_fix} / {base_isa}")
        ax.set_ylabel("per-call wall time (µs)")
        ax.set_title("Multi-block (fused) throughput")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / "block_fusion.png", dpi=140)
        plt.close(fig)

    # ---- plot 4: nvar axis (only if swept) -- per-field cost vs nvar ----
    nvars = sorted({r["nvar"] for r in rows})
    if len(nvars) > 1:
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        sb1 = [r for r in rows if r["xfused"] == 1 and r["yfused"] == 1 and r["zfused"] == 1
               and r["eleorder"] == eo_fix and r["isa"] == base_isa]
        for impl in impls:
            pts = sorted(((r["nvar"], r["mean_us"]) for r in sb1 if r["impl"] == impl))
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", color=IMPL_COLOR[impl], label=IMPL_LABEL[impl])
        ax.set_xlabel(f"nvar (field variables), E{eo_fix} / {base_isa}, single block")
        ax.set_ylabel("per-field per-call wall time (µs)")
        ax.set_title("Per-field cost vs variable count (cache-reuse)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / "nvar_throughput.png", dpi=140)
        plt.close(fig)

    print(f"\nplots -> {out_dir}/  (ISAs plotted: {', '.join(isas)})")
    for p in sorted(out_dir.glob("*.png")):
        print(f"  {p.name}")


# ---------- summary ------------------------------------------------------- #

def print_summary(csv_path: Path) -> None:
    rows = load_csv(csv_path)
    single = [r for r in rows
              if r["xfused"] == 1 and r["yfused"] == 1 and r["zfused"] == 1 and r["nvar"] == 1]
    if not single:
        return
    isas = _isas_in(single)
    eleorders = sorted({r["eleorder"] for r in single})
    print("\n== single-block per-call summary (µs) : compact vs production-explicit ==")
    print(f"{'isa':8} {'E':>3} {'compact_us':>12} {'orig_us':>12} {'speedup':>9}")
    for isa in isas:
        for eo in eleorders:
            def g(impl):
                rec = [r for r in single if r["impl"] == impl and r["isa"] == isa and r["eleorder"] == eo]
                return rec[0]["mean_us"] if rec else 0.0
            c, o = g("compact"), g("original")
            sp = f"{o / c:.2f}x" if c else "-"
            print(f"{isa:8} {eo:>3} {c:>12.4f} {o:>12.4f} {sp:>9}")


# ---------- main ---------------------------------------------------------- #

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["collect", "plot", "all"])
    p.add_argument("--toml", type=Path, default=DEFAULT_TOML)
    p.add_argument("--csv",  type=Path, default=DATA_CSV)
    p.add_argument("--out",  type=Path, default=OUT_DIR)
    p.add_argument("--eleorders", default=",".join(str(e) for e in ELEORDERS),
                   help="comma list of element orders to sweep")
    p.add_argument("--nvars", default="1",
                   help="comma list of field-variable counts to sweep (nvar axis)")
    args = p.parse_args()

    eleorders = [int(x) for x in args.eleorders.split(",") if x.strip()]
    nvars = [int(x) for x in args.nvars.split(",") if x.strip()]

    if args.mode in ("collect", "all"):
        collect(args.toml, args.csv, eleorders, nvars)
    if args.mode in ("plot", "all"):
        plot_all(args.csv, args.out)
        print_summary(args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
