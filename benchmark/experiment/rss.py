import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_python

DATADIR, NAME, STRUCTURE = sys.argv[1], sys.argv[2], sys.argv[3]
D, N = int(sys.argv[4]), int(sys.argv[5])

def vmhwm_gib():
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 2 ** 20
    return float("nan")

_, _, run = run_python.CASES[NAME]

try:
    run(run_python.load("isotropic", 20, 100, datadir=DATADIR))
except Exception:
    pass

baseline = vmhwm_gib()

try:
    d = run_python.load(STRUCTURE, D, N, datadir=DATADIR)
    t0 = time.perf_counter()
    run(d)
    elapsed = time.perf_counter() - t0
    peak = vmhwm_gib()
    print(f"{NAME}\t{STRUCTURE}\t{D}\t{N}\tOK\t{elapsed:.4f}\t"
          f"{peak:.4f}\t{max(peak - baseline, 0.0):.4f}")
except MemoryError:
    print(f"{NAME}\t{STRUCTURE}\t{D}\t{N}\tOOM\t0\t0\t0\tMemoryError")
except Exception as e:
    print(f"{NAME}\t{STRUCTURE}\t{D}\t{N}\tFAIL\t0\t0\t0\t{str(e)[:60]}")
