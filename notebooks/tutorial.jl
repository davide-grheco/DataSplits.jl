### A Pluto.jl notebook ###
# v0.20.25

using Markdown
using InteractiveUtils

# ╔═╡ a2b3c4d5-e6f7-4901-2345-bcdef0123456
begin
  import Pkg
  Pkg.develop(path = joinpath(@__DIR__, ".."))
  Pkg.add(["MLJ", "MLJDecisionTreeInterface", "Flux", "DataFrames", "CategoricalArrays"])

  using DataSplits
  import DataSplits: partition     # MLJ also exports partition; pin to DataSplits
  using MLJ
  import MLJDecisionTreeInterface   # registers DecisionTreeClassifier with MLJ
  using Flux
  using DataFrames, CategoricalArrays
  using Random, Statistics
end

# ╔═╡ f1e2d3c4-b5a6-4890-1234-abcdef012345
md"""
# DataSplits.jl — Integration with MLJ and Flux

**DataSplits.jl** provides data-splitting and cross-validation strategies that return plain
index vectors, composable with any Julia ML framework.

| Section | Strategy | Framework | Task |
|---------|----------|-----------|------|
| 1 — Cross-validation | `StratifiedKFold` | **MLJ** | Classification |
| 2 — Train / test split | `RandomSplit` | **Flux** | Regression |

Swap any strategy without touching the training code.

> **Run this notebook** by opening it with [Pluto.jl](https://plutojl.org/).
> Pluto will install all required packages automatically.
"""

# ╔═╡ c4d5e6f7-a8b9-4123-4567-def012345678
md"""
---
## 1 — Cross-validation with MLJ

We build a **binary classification** dataset with an 80 / 20 class imbalance and use
`StratifiedKFold` to guarantee every fold mirrors the global class ratio.

Plain `KFold` shuffles observations uniformly and can produce folds where the minority
class is under- or over-represented, masking the true generalisation error.
Replacing `StratifiedKFold` with `KFold` below is a one-word change.
"""

# ╔═╡ d5e6f7a8-b9c0-4234-5678-ef0123456789
begin
  rng_cls = Xoshiro(42)
  n_a, n_b = 320, 80

  X_cls = DataFrame(
    x1 = vcat(randn(rng_cls, n_a), randn(rng_cls, n_b) .+ 3.0),
    x2 = vcat(randn(rng_cls, n_a), randn(rng_cls, n_b) .+ 3.0),
  )
  y_cls = categorical(vcat(fill("A", n_a), fill("B", n_b)))

  "$(nrow(X_cls)) observations  |  class A: $n_a ($(n_a*100÷(n_a+n_b))%)  |  class B: $n_b ($(n_b*100÷(n_a+n_b))%)"
end

# ╔═╡ f7a8b9c0-d1e2-4456-789a-012345678901
begin
  DecisionTreeClassifier = @load DecisionTreeClassifier pkg = DecisionTree verbosity = 0
  tree = DecisionTreeClassifier(max_depth = 4)

  # rowpairs converts the CrossValidationSplit into the (train, test) tuple format
  # that MLJ's evaluate! accepts natively via the resampling= keyword.
  mach = machine(tree, X_cls, y_cls)
  cv_result = evaluate!(
    mach;
    resampling = rowpairs(y_cls, StratifiedKFold(5); rng = rng_cls),
    measure = accuracy,
    verbosity = 0,
  )
  fold_accuracies = cv_result.per_fold[1]
end

# ╔═╡ a8b9c0d1-e2f3-4567-89ab-123456789012
md"""
**Per-fold accuracy:** $(join(string.(round.(fold_accuracies .* 100; digits=1)) .* "%", "  ·  "))

**Cross-validated accuracy:** $(round(mean(fold_accuracies) * 100; digits=1))%
± $(round(std(fold_accuracies) * 100; digits=1))%

Each test fold contains ≈ $(round(Int, n_b / (n_a + n_b) * 100))% class-B observations —
matching the global ratio, guaranteed by `StratifiedKFold`.
"""

# ╔═╡ b9c0d1e2-f3a4-4678-9abc-234567890123
md"""
---
## 2 — Train / test split with Flux

A **regression** task: approximate a noisy nonlinear function of four features with
a small MLP. `RandomSplit` gives an 80 / 20 split; you could replace it with
`KennardStoneSplit` (maximally diverse training set) or `StratifiedShuffleSplit`
(repeated random splits with target stratification).

DataSplits uses the **features × samples** convention — columns are observations —
which matches Flux's `Dense` layer directly.
"""

# ╔═╡ c0d1e2f3-a4b5-4789-abcd-345678901234
begin
  rng_reg = Xoshiro(123)
  n_reg, n_features = 600, 4

  X_reg = randn(rng_reg, Float32, n_features, n_reg)
  y_reg =
    sin.(X_reg[1, :]) .+ 0.5f0 .* X_reg[2, :] .+ 0.1f0 .* randn(rng_reg, Float32, n_reg)

  "X: $(size(X_reg))  |  y: $(length(y_reg)) values"
end

# ╔═╡ d1e2f3a4-b5c6-4890-bcde-456789012345
begin
  # train=80, test=20 means 80% / 20% (integers summing to 100 are percentages).
  split_reg = partition(X_reg, RandomSplit(); train = 80, test = 20, rng = rng_reg)

  # trainview / testview extract a cohort from multiple sources in one call.
  # The tuple returned by trainview passes directly to Flux.DataLoader.
  X_train, y_train = trainview(split_reg, X_reg, y_reg)
  X_test, y_test = testview(split_reg, X_reg, y_reg)

  "Train: $(size(X_train))  |  Test: $(size(X_test))"
end

# ╔═╡ e2f3a4b5-c6d7-4901-cdef-567890123456
begin
  model = Chain(Dense(n_features => 64, relu), Dense(64 => 32, relu), Dense(32 => 1))
  # Dense(32 => 1) outputs shape (1, batch); vec collapses it to a vector to match y.
  mse(m, x, y) = Flux.mse(vec(m(x)), y)

  opt_state = Flux.setup(Adam(1.0f-3), model)
  # trainview returns a tuple — pass it directly without unpacking.
  loader =
    Flux.DataLoader(trainview(split_reg, X_reg, y_reg); batchsize = 64, shuffle = true)

  for _ = 1:300
    for (xb, yb) in loader
      grads = Flux.gradient(model) do m
        Flux.mse(vec(m(xb)), yb)
      end
      Flux.update!(opt_state, model, grads[1])
    end
  end

  train_mse = mse(model, X_train, y_train)
  test_mse = mse(model, X_test, y_test)

  (; train_mse = round(train_mse; digits = 4), test_mse = round(test_mse; digits = 4))
end

# ╔═╡ f3a4b5c6-d7e8-4012-def0-678901234567
md"""
| Cohort | MSE |
|--------|-----|
| Train  | $(round(train_mse; digits = 4)) |
| Test   | $(round(test_mse;  digits = 4)) |
"""

# ╔═╡ Cell order:
# ╟─f1e2d3c4-b5a6-4890-1234-abcdef012345
# ╠═a2b3c4d5-e6f7-4901-2345-bcdef0123456
# ╟─c4d5e6f7-a8b9-4123-4567-def012345678
# ╠═d5e6f7a8-b9c0-4234-5678-ef0123456789
# ╠═f7a8b9c0-d1e2-4456-789a-012345678901
# ╟─a8b9c0d1-e2f3-4567-89ab-123456789012
# ╟─b9c0d1e2-f3a4-4678-9abc-234567890123
# ╠═c0d1e2f3-a4b5-4789-abcd-345678901234
# ╠═d1e2f3a4-b5c6-4890-bcde-456789012345
# ╠═e2f3a4b5-c6d7-4901-cdef-567890123456
# ╟─f3a4b5c6-d7e8-4012-def0-678901234567
