```@meta
CurrentModule = DataSplits
```

# 03. Core API Reference

## split

```julia
split(data, strategy; rng)
```

Dispatches to `strategy._split`. Returns `(train, test)` index vectors.

## SplitStrategy Interface

To add a new strategy, subtype `SplitStrategy` and implement:

```julia
_split(data, s::YourStrategy; rng)
```

## Utility Functions

- `sample_indices(data)`: returns iterable indices of samples (default: `1:size(data,1)`).
- `ValidFraction`: bounds-checked fraction type for split ratios.

## Example: Custom Splitter

```julia
struct MySplit <: SplitStrategy; frac; end
function _split(data, s::MySplit; rng)
  # ...
  return train_idx, test_idx
end
```
