```@meta
CurrentModule = DataSplits
```

# 02. Getting Started

## Basic API

- `split(data, strategy)`: main entry point.
- `data`: array, tuple `(X,)` or `(X, y)`.
- `strategy`: subtype of `SplitStrategy`.

## Data Formats

- Accepts matrices, tables, or custom types implementing `sample_indices`.

## Randomness Control

Pass `rng` keyword to strategies supporting it, e.g. `split(X, RandomSplit(0.7); rng=123)`.

## Example: Custom Data Type

To use your own data type, implement `sample_indices(data)` and `get_sample(data, i)`.
