# Benchmark experiment

Runtime and peak memory of every DataSplits strategy with an astartes counterpart, measured head to head on identical
inputs.

Linux only: peak memory is read from `/proc/self/status`, and the per-stage memory cap uses a systemd user scope.

## Setup

```sh
julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
python -m venv .venv && .venv/bin/pip install astartes numpy
```

## Run

```sh
benchmark/experiment/run_all.sh
```

## How it works

| Stage      | Does                                                            |
| ---------- | --------------------------------------------------------------- |
| `generate` | Writes raw arrays plus a checksum index, read by both languages |
| `*-main`   | Times one cell per strategy, shuffled order, and digests the split |
| `rss`      | Peak memory, one process per cell under a cgroup cap            |
| `analyse`  | Writes `comparison.csv` and `capacity.csv` from the raw JSON     |

## Output

`comparison.csv`, one row per sampler both packages implement:

- **`verdict`.** `identical` means both produce the same split, so the ratio compares like with like. `unchecked` means
  the sampler is stochastic and has no canonical split, so the ratio is indicative only. `divergent` means it is not the
  same computation and no ratio is reported.
- **`ratio`** is astartes over DataSplits: above 1 means DataSplits is faster. Within 1.15x is comparable, not a win.
- **`*_iqr`** is the interquartile range as a fraction of the median, so a ratio can be read against the noise.
- **`*_peak_gib`** is a high-water mark, so small values are a floor rather than a measurement. Use
  `datasplits_alloc_bytes` for the exact figure.

`capacity.csv`, peak memory for every cell. `OOM` and `SKIPPED` are results, not errors: they locate where a strategy
stops fitting in the budget. `FAIL` is an error and fails the stage.
