```@meta
CurrentModule = DataSplits
```

# Morais–Lima–Martin Split

`MoraisLimaMartinSplit` starts from a Kennard–Stone selection and then **randomly
swaps a fraction of samples** between the training and test sets. This introduces
controlled stochasticity into an otherwise deterministic diversity-based split,
useful when you need multiple independent splits from a small dataset.

## How it works

1. Run Kennard–Stone to produce an initial train/test split.
2. Randomly select `swap_frac × min(n_train, n_test)` samples from each cohort.
3. Swap those samples between train and test.

The result retains most of the spatial coverage of Kennard–Stone while adding
randomness that enables ensemble evaluation.

## When to use it

- You need **multiple different splits** from the same small dataset (e.g. to
  estimate the variance of a model evaluation metric), but pure random splits lose
  the coverage guarantee.
- You want **Kennard–Stone as a baseline** with some stochastic perturbation to
  assess split sensitivity.
- You are building a **bootstrap or ensemble** of calibration models and need
  diverse but roughly coverage-preserving training sets.

## Usage

```julia
using DataSplits

# Default: swap 10% of samples.
res = partition(X, MoraisLimaMartinSplit(); train = 0.8, test = 0.2,
                rng = MersenneTwister(42))
X_train, X_test = splitdata(res, X)

# Swap 5% — closer to pure Kennard–Stone.
res = partition(X, MoraisLimaMartinSplit(; swap_frac = 0.05);
                train = 0.8, test = 0.2, rng = MersenneTwister(1))

# Swap 20% — more random, less deterministic.
res = partition(X, MoraisLimaMartinSplit(; swap_frac = 0.20);
                train = 0.8, test = 0.2)

# Custom metric.
using Distances
res = partition(X, MoraisLimaMartinSplit(; metric = Cityblock());
                train = 0.8, test = 0.2)
```

The `rng` keyword controls the random swap — fix it for reproducibility, vary it to
get different realisations.

## Parameters

| Parameter | Default | Effect |
| --- | --- | --- |
| `swap_frac` | `0.1` | Fraction of samples swapped; must be in (0, 1) |
| `metric` | `Euclidean()` | Distance metric for the Kennard–Stone initialisation |

## API reference

- [`MoraisLimaMartinSplit`](@ref)
