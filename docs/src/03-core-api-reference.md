```@meta
CurrentModule = DataSplits
```

# 03. Core API Reference

## split

```julia
split(data, strategy; rng)
```

Dispatches to `strategy._split`. Returns `(train, test)` index vectors.

## split_with_positions

```julia
split_with_positions(data, s, core_algorithm; rng=Random.default_rng())
```

A utility function that handles mapping between user-facing indices (which may be non-1-based or non-contiguous) and internal 1:N positions. The `core_algorithm` should operate on positions `1:N` and return `(train_pos, test_pos)`, which are then mapped back to the correct indices for the user.

## SplitStrategy Interface

To add a new strategy, subtype `SplitStrategy` and implement:

```julia
_split(data, s::YourStrategy; rng)
```

## Utility Functions

- `sample_indices(data)`: returns iterable indices of samples (default: `1:size(data,1)`, but now robust to any AbstractArray axes).
- `ValidFraction`: bounds-checked fraction type for split ratios.

## Example: Custom Splitter (new pattern)

```julia
struct MySplit <: SplitStrategy; frac; end

function mysplit(N, s, rng, data)
    cut = floor(Int, s.frac * N)
    train_pos = 1:cut
    test_pos = (cut+1):N
    return train_pos, test_pos
end

function _split(data, s::MySplit; rng=Random.default_rng())
    split_with_positions(data, s, mysplit; rng=rng)
end
```
