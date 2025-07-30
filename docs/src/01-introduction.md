```@meta
CurrentModule = DataSplits
```

# Introduction

## Motivation and Features

DataSplits provides rational train/test splitting algorithms and clustering-based pre-processing for reproducible model evaluation. Unlike random splits, these methods ensure that the training and test sets are representative and diverse, which is especially important for small or structured datasets.

**Key Features:**

- Multiple splitting strategies: maximin, clustering-based, group-aware, and more
- Extensible API for custom strategies
- Works with arrays, tuples, and custom data types

## Installation

```julia
] add https://github.com/davide-grheco/DataSplits.jl
```

## When to Use DataSplits

- When random splits are not enough (e.g., small datasets, strong structure)
- When you need reproducible, rational splits for benchmarking
- When you want to preserve group or cluster structure in your splits

## Quickstart Example

```julia
using DataSplits, Distances

# Simple Kennard–Stone split
splitter = KennardStoneSplit(0.8)
result = split(X, splitter)
X_train, X_test = splitdata(result, X)

# SPXY split on features and target
splitter = SPXYSplit(0.7; metric=Cityblock())
result = split((X, y), splitter)
X_train, X_test = splitdata(result, X)
```

## Glossary

- **train/test**: The two sets into which data is split for model training and evaluation.
- **fraction**: The proportion of samples to assign to the training set (between 0 and 1).
- **indices**: Integer positions of samples in the data array.
- **samples**: Individual data points (columns in a matrix, or elements in a vector).
- **splitter/strategy**: An object describing how to split the data (e.g., `KennardStoneSplit`).
- **SplitResult**: The object returned by `split`, containing train/test indices.
- **splitdata**: Function to extract the actual data splits from a `SplitResult`.
- **ClusteringResult**: Object representing cluster assignments for samples.

---

For more details, see the [Getting Started](02-getting-started.md) and [Core API Reference](03-core-api-reference.md).
