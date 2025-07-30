```@meta
CurrentModule = DataSplits
```

# Getting Started

## Basic API

- `split(data, strategy)`: main entry point.
- `data`: array, tuple `(X,)` or `(X, y)`.
- `strategy`: subtype of `SplitStrategy`.

## Data Formats

- Accepts matrices, arrays, tables, or custom types.
- **Important:** DataSplits expects matrices to be in the Julia ML convention: columns are samples, rows are features. If your data uses rows as samples, transpose it before splitting (e.g., use `X'`).
- For custom data types, implement `Base.length` (number of samples) and `Base.getindex(data, i)` (returning the i-th sample) as described in the [MLUtils documentation](https://juliaml.github.io/MLUtils.jl).

## Robust Index Handling

All splitting strategies in DataSplits are robust to arrays with arbitrary axes (e.g., OffsetArrays, SubArrays, etc.). The library automatically handles mapping between user-facing indices and internal positions, so you can use any AbstractArray as input.

## Randomness Control

Pass `rng` keyword to strategies supporting it, e.g. `split(X, RandomSplit(0.7); rng=123)`.

## Example: Custom Data Type

To use your own data type, implement `sample_indices(data)` and `get_sample(data, i)`.
