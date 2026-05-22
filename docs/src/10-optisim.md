```@meta
CurrentModule = DataSplits
```

# OptiSim and the Dissimilarity Family

OptiSim (Clark 1997) is a **tunable diversity-selection** algorithm. Where
Kennard–Stone always evaluates all remaining candidates at each step (expensive but
optimal), OptiSim samples a random subsample of candidates and picks the most
dissimilar one — trading some optimality for speed.

## How it works

1. Maintain a pool of unselected candidates (initially all N points).
2. At each step, draw a random subsample of `max_subsample_size` candidates from the
   pool. Remove any candidate whose distance to the closest already-selected point
   falls below `distance_cutoff` (these are "too similar" to what is already selected).
3. From the remaining subsample, add the candidate most dissimilar to the current
   training set.
4. Repeat until `n_train` samples have been selected.

The `distance_cutoff` parameter filters out candidates that are too close to existing
training points, preventing redundant selection.

## The dissimilarity family

 | Strategy | Effectively | When to use |
 | ---------- | ------------- | ------------- |
 | [`MinimumDissimilaritySplit`](@ref) | `max_subsample_size = 1` | Fastest greedy option |
 | [`OptiSimSplit`](@ref) | `max_subsample_size` tunable | Balance speed vs. quality |
 | [`MaximumDissimilaritySplit`](@ref) | `max_subsample_size = N` | Maximum spread (slower) |

All three follow the same interface. Lazy variants avoid storing the full N×N matrix
(O(N) peak memory), but recompute distances on-the-fly — making them 3–11× slower
at N=1000 (MinimumDissimilarity: ~3×, OptiSim: ~5×, MaximumDissimilarity: ~11×).
Prefer lazy only when the full matrix does not fit in RAM.

## Usage

```julia
using DataSplits

# Greedy minimum dissimilarity (fastest).
res = partition(X, MinimumDissimilaritySplit(); train = 0.8, test = 0.2)
X_train, X_test = splitdata(res, X)

# OptiSim with a custom subsample size.
res = partition(X, OptiSimSplit(; max_subsample_size = 20); train = 0.8, test = 0.2)

# Maximum dissimilarity (best spread, slowest).
res = partition(X, MaximumDissimilaritySplit(); train = 0.8, test = 0.2)

# Large datasets — lazy variants.
res = partition(X, LazyOptiSimSplit(; max_subsample_size = 10); train = 0.8, test = 0.2)
res = partition(X, LazyMinimumDissimilaritySplit(); train = 0.8, test = 0.2)
res = partition(X, LazyMaximumDissimilaritySplit(); train = 0.8, test = 0.2)

# Custom metric.
using Distances
res = partition(X, OptiSimSplit(; metric = Cityblock()); train = 0.8, test = 0.2)
```

## Tuning `distance_cutoff`

The `distance_cutoff` (default `0.35`) filters out candidates too close to existing
training points. If it is too large relative to the data's spread, the candidate
pool empties before `n_train` samples are selected; a `@warn` is emitted and the
training set is returned smaller than requested.

To silence the warning for a large batch of splits:

```julia
using Logging, LoggingExtras

silent = EarlyFilteredLogger(
    log -> log.id !== :datasplits_optisim_undershoot,
    current_logger()
)
with_logger(silent) do
    results = [partition(X, OptiSimSplit(); train = 0.8, test = 0.2) for _ in 1:100]
end
```

Reduce `distance_cutoff` if you are consistently getting smaller-than-requested
training sets.

## OptiSim vs. Kennard–Stone

Kennard–Stone is deterministic and globally optimal under the maximin criterion —
it always selects the best next point. OptiSim is stochastic (due to random
subsampling) and greedy, but it is faster for large N and gives you a tunable knob
between speed and quality.

For small datasets where you can afford it, prefer Kennard–Stone. For large datasets
or when you want multiple diverse splits, prefer OptiSim. The lazy variant is only
needed when the N×N matrix does not fit in RAM — it is ~5× slower at N=1000.

## Limitation: outliers

`MaximumDissimilaritySplit` greedily selects the most extreme point at each step.
This means outliers are almost always included in the training set. If your data
contains measurement errors or genuine outliers, remove them before splitting.

## API reference

- [`OptiSimSplit`](@ref), [`LazyOptiSimSplit`](@ref)
- [`MinimumDissimilaritySplit`](@ref), [`LazyMinimumDissimilaritySplit`](@ref)
- [`MaximumDissimilaritySplit`](@ref), [`LazyMaximumDissimilaritySplit`](@ref)

## References

Clark, R. D. OptiSim: An Extended Dissimilarity Selection Method for Finding Diverse
Representative Subsets. *J. Chem. Inf. Comput. Sci.* 1997, 37(6), 1181–1188.
<https://doi.org/10.1021/ci970282v>.
