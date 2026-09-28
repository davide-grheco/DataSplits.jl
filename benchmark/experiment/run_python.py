import json
import os
import tomllib
import platform
import random
import sys
import time
import timeit
import warnings

warnings.filterwarnings("ignore")

import datetime

import numpy as np
from astartes import train_test_split as astartes_split

DATADIR = sys.argv[1] if len(sys.argv) > 1 else ""
OUTFILE = sys.argv[2] if len(sys.argv) > 2 else ""

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifest.toml"), "rb") as fh:
    CFG = tomllib.load(fh)
TIMING_N = CFG["data"]["timing_n"]
_OVERRIDE = (int(os.environ["DATASPLITS_BENCH_SIZES"].split(",")[-1])
             if "DATASPLITS_BENCH_SIZES" in os.environ else None)

SECONDS_PER_CELL = 2.0

def load(structure, D, N, datadir=None):
    root = datadir or DATADIR

    def rd(suffix, dtype, count):
        return np.fromfile(f"{root}/{structure}_D{D}_N{N}_{suffix}.bin", dtype=dtype, count=count)

    times = rd("times", np.int64, N)
    epoch = datetime.date(1970, 1, 1)
    dates = [epoch + datetime.timedelta(days=int(t)) for t in times]

    return dict(
        dates=dates,
        X=rd("X", np.float64, D * N).reshape(N, D),
        y=rd("y", np.float64, N),
        times=times,
    )

def _ast(sampler, hopts=None, **kw):
    return lambda d: astartes_split(
        d["X"], train_size=0.8, test_size=0.2, sampler=sampler,
        return_indices=True, hopts=dict(hopts or {}),
        **({k: d[v] for k, v in kw.items()}),
    )

CASES = {
    "KennardStoneSplit":  ("astartes", "distance_eager", _ast("kennard_stone")),
    "SPXYSplit":          ("astartes", "distance_eager", _ast("spxy", y="y")),
    "OptiSimSplit":       ("astartes", "distance_eager", _ast("optisim")),
    "sphere_exclusion":   ("astartes", "clustering", _ast("sphere_exclusion")),
    "RandomSplit":        ("astartes", "simple", _ast("random")),
    "TargetPropertyHigh": ("astartes", "simple", _ast("target_property", {"descending": True}, y="y")),
    "TargetPropertyLow":  ("astartes", "simple", _ast("target_property", y="y")),
    "TimeSplitOldest":    ("astartes", "simple", _ast("time_based", labels="dates")),
}

_unknown = sorted({fam for _, fam, _ in CASES.values()} - set(TIMING_N))
if _unknown:
    raise SystemExit(
        f"family {_unknown} is in CASES but not in manifest timing_n; "
        "the Python family names must match those in cases.jl")

DETERMINISTIC = {"KennardStoneSplit", "SPXYSplit", "TargetPropertyHigh",
                 "TargetPropertyLow", "TimeSplitOldest"}


def digest(result):
    """Digest of a split, so the timing rows carry their own evidence that the
    two implementations computed the same thing."""
    if isinstance(result, tuple):
        tr = np.asarray(result[-2]).tolist()
        return dict(kind="traintest", n_train=len(tr),
                    checksum=int(sum(tr)), first20=tr[:20])
    te = sorted((sorted(int(i) for i in t) for _, t in result),
                key=lambda t: t[0] if t else 1 << 62)
    return dict(kind="cv", fold_sizes=[len(t) for t in te],
                checksum=int(sum((k + 1) * sum(t) for k, t in enumerate(te))))


def cells():
    D = CFG["data"]["features"]
    return [(n, fam, "isotropic", D, _OVERRIDE or TIMING_N[fam])
            for n, (_, fam, _) in CASES.items()]

def main():
    grid = cells()
    random.Random(11).shuffle(grid)
    cache, results, broken = {}, [], []

    print(f"cells={len(grid)}  strategies={len(CASES)}")
    print("\n-- measurement (shuffled) -----------------------------------")

    for i, (name, family, structure, D, N) in enumerate(grid, 1):
        fn = CASES[name][2]
        try:
            d = cache.setdefault((structure, D, N), load(structure, D, N))
            warm = fn(d)
            dig = digest(warm) if name in DETERMINISTIC else None
            reps, elapsed, times = 0, 0.0, []
            while elapsed < SECONDS_PER_CELL and reps < 10_000:
                t = timeit.repeat(lambda: fn(d), number=1, repeat=1)[0]
                times.append(t); elapsed += t; reps += 1
                if reps >= 5 and elapsed > SECONDS_PER_CELL:
                    break
            times.sort()
        except Exception as e:
            broken.append(dict(strategy=name, N=N, error=f"{type(e).__name__}: {str(e)[:180]}"))
            print(f"  [{i:2d}/{len(grid):2d}] {name:<26} N={N:<7} ERROR {type(e).__name__}")
            continue

        results.append(dict(
            strategy=name, library=CASES[name][0], family=family, structure=structure,
            D=D, N=N,
            time_median_ns=float(np.median(times) * 1e9),
            time_q25_ns=float(np.quantile(times, 0.25) * 1e9),
            time_q75_ns=float(np.quantile(times, 0.75) * 1e9),
            time_min_ns=float(times[0] * 1e9),
            samples=len(times), digest=dig,
        ))
        print(f"  [{i:2d}/{len(grid):2d}] {name:<26} N={N:<7} "
              f"{results[-1]['time_median_ns']/1e6:9.3f} ms")

    import sklearn, astartes
    out = dict(
        runtime="python",
        python_version=platform.python_version(),
        numpy_version=np.__version__, sklearn_version=sklearn.__version__,
        astartes_version=getattr(astartes, "__version__", "1.3.3"),
        threads=os.environ.get("OMP_NUM_THREADS", "unset"),
        machine=platform.machine(),
        results=results, broken=broken,
    )
    os.makedirs(os.path.dirname(OUTFILE) or ".", exist_ok=True)
    json.dump(out, open(OUTFILE, "w"), indent=2)
    print(f"\nwrote {OUTFILE}  ({len(results)} rows)")

    if broken:
        print(f"\n{len(broken)} case(s) could not run:")
        for b in broken:
            print(f"  {b['strategy']} at N={b['N']}: {b['error']}")
        raise SystemExit(1)

if __name__ == "__main__":
    main()
