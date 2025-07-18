```@meta
CurrentModule = DataSplits
```

# 01. Introduction

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
train, test = split(X, KennardStoneSplit(0.8))

# SPXY split on features and target
train, test = split((X, y), SPXYSplit(0.7; metric=Cityblock()))
```
