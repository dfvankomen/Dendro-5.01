#!/usr/bin/env python3
"""Aggregate per-rank breadcrumb / record files (DENDRO_BREADCRUMB_DIR, or the
first_xchg/ record dir) written by dendro::bc.

usage: breadcrumbs.py <dir> [--stall] [--tail N]

Default: max/sum over ranks of the [first-xchg] records per (mesh, call), the
[first-xchg-probe] max per mesh, and every [first-xchg-stall] line.
--stall: hang diagnosis. Per rank: last line and still-open scopes ("tag+"
without "tag-"); ranks grouped by their last tag, so the odd ones out stand
out; mesh dtor counts per mesh id (a mesh destroyed on only some ranks).
"""
import collections
import glob
import os
import re
import sys

KV = re.compile(r"(\w+)=(-?[0-9.]+(?:e[-+]?\d+)?)")
PHASES = "sync alloc post pack isend wait unpack mirror free total lazy_before".split()


def load(d):
    out = {}
    for f in glob.glob(os.path.join(d, "r*.txt")):
        m = re.search(r"r(\d+)\.txt$", f)
        if not m:
            continue
        with open(f, errors="replace") as fh:
            out[int(m.group(1))] = [ln.rstrip("\n") for ln in fh if ln.strip()]
    return out


def body(ln):
    return ln.split(" ", 1)[1] if ln.startswith("t=") and " " in ln else ln


def stamp(ln):
    m = re.match(r"t=([0-9.]+)", ln)
    return float(m.group(1)) if m else 0.0


def kv(ln):
    return {k: float(v) for k, v in KV.findall(body(ln))}


def report(files):
    xchg = collections.defaultdict(list)
    probe = collections.defaultdict(list)
    stalls = []
    for r, lines in files.items():
        for ln in lines:
            b = body(ln)
            if b.startswith("[first-xchg] "):
                v = kv(ln)
                v["variant"] = "ctx" if " ctx " in b else "plain"
                xchg[(int(v["mesh"]), int(v["call"]))].append(v)
            elif b.startswith("[first-xchg-probe]"):
                probe[int(kv(ln)["mesh"])].append(kv(ln))
            elif b.startswith("[first-xchg-stall]"):
                stalls.append("r%d %s" % (r, b))
    for (mesh, call), recs in sorted(xchg.items()):
        mx = {k: max(x.get(k, 0.0) for x in recs) for k in PHASES + ["send", "recv", "majflt", "minflt"]}
        vs = {x["variant"] for x in recs}
        print("[first-xchg] mesh=%d call=%d %s dof=%d labelled=%d ranks=%d/%d %s ms (max) "
              "minflt=%d/%d majflt=%d (max/sum) maxsend=%d maxrecv=%d"
              % (mesh, call, vs.pop() if len(vs) == 1 else "mixed", recs[0]["dof"],
                 recs[0]["labelled"], len(recs), recs[0]["np"],
                 " ".join("%s=%.2f" % (k, mx[k]) for k in PHASES),
                 mx["minflt"], sum(x["minflt"] for x in recs), mx["majflt"], mx["send"], mx["recv"]))
    for mesh, recs in sorted(probe.items()):
        print("[first-xchg-probe] mesh=%d labelled=%d ranks=%d/%d dof=%d %s ms (max)"
              % (mesh, recs[0]["labelled"], len(recs), recs[0]["np"], recs[0]["dof"],
                 " ".join("%s=%.2f" % (k, max(x[k] for x in recs))
                          for k in ("hs_comm", "warm_xchg", "warm_xchg2"))))
    for s in stalls:
        print(s)


def stall(files, tail):
    groups = collections.defaultdict(list)
    dtors = collections.defaultdict(set)
    ctors = collections.defaultdict(set)
    for r, lines in sorted(files.items()):
        open_ = []
        for ln in lines:
            b = body(ln)
            tok = b.split(" ")[0]
            mid = re.search(r"\bid=(\d+)", b)
            if tok == "mesh.dtor" and mid:
                dtors[int(mid.group(1))].add(r)
            if tok == "mesh.ctor-" and mid:
                ctors[int(mid.group(1))].add(r)
            if tok.endswith("+"):
                open_.append(b)
            elif tok.endswith("-") and open_:
                for i in range(len(open_) - 1, -1, -1):
                    if open_[i].split(" ")[0][:-1] == tok[:-1]:
                        del open_[i]
                        break
        last = lines[-1] if lines else ""
        key = re.sub(r"t=[0-9.]+ ", "", last)
        groups[key].append(r)
        if tail:
            print("r%d last=%.3f open=[%s]" % (r, stamp(last), " | ".join(open_)))
            for ln in lines[-tail:]:
                print("    " + ln)
    print("-- ranks grouped by last line (%d ranks)" % len(files))
    for key, rs in sorted(groups.items(), key=lambda kv_: len(kv_[1])):
        shown = rs if len(rs) <= 16 else rs[:16] + ["..."]
        print("%5d  %s  ranks=%s" % (len(rs), key, ",".join(map(str, shown))))
    nr = len(files)
    bad = [(i, len(dtors.get(i, ())), len(ctors.get(i, ()))) for i in sorted(set(dtors) | set(ctors))
           if len(dtors.get(i, ())) not in (0, nr) or len(ctors.get(i, ())) not in (0, nr)]
    print("-- meshes built/destroyed on a rank subset: %s" % (bad if bad else "none"))


def main():
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    files = load(a[0])
    if not files:
        sys.exit("no r*.txt files in %s" % a[0])
    if "--stall" in a:
        tail = int(a[a.index("--tail") + 1]) if "--tail" in a else 0
        stall(files, tail)
    else:
        report(files)


if __name__ == "__main__":
    main()
