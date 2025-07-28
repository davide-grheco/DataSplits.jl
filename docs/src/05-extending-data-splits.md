```@meta
CurrentModule = DataSplits
```

# 05. Extending DataSplits

## Checklist for Adding a New Splitter

1. Subtype `SplitStrategy`.
2. Implement `_split(data, s; rng)`.
3. Use `ValidFraction` for fraction validation.
4. Add a docstring and example usage.
5. Add a test in `test/`.

**Best Practices:**

- Always validate input shapes and types.
- Use `getobs` and `numobs` for compatibility with custom data types.
- Return a `TrainTestSplit` or other `SplitResult` subtype.
- Document edge cases and limitations.

**Example:**

```julia
struct MySplit <: SplitStrategy; frac; end

function mysplit(N, s, rng, data)
    cut = floor(Int, s.frac * N)
    train_pos = 1:cut
    test_pos = (cut+1):N
    return train_pos, test_pos
end

function _split(data, s::MySplit; rng=Random.default_rng())
    N = numobs(data)
    train_pos, test_pos = mysplit(N, s, rng, data)
    return TrainTestSplit(train_pos, test_pos)
end
```
