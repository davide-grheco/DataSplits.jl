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
# DataSplits.jl — Tutorial Notebook

**DataSplits.jl** provides data-splitting and cross-validation strategies that return plain
index vectors, composable with any Julia ML framework.

| Section | Strategy | Task |
|---------|----------|------|
| 1 — Cross-validation | `StratifiedKFold` | Classification with MLJ |
| 2 — Train / test split | `RandomSplit` | Regression with Flux |
| 3 — Feature-space coverage | `KennardStoneSplit` vs `RandomSplit` | Why random splits mislead |
| 4 — Joint coverage | `SPXYSplit` | Covering features and target together |

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

# ╔═╡ 1a2b3c4d-5e6f-4701-8234-aabbccddeeff
md"""
---
## 3 — Kennard–Stone: covering the feature space

Random splitting is blind to where points sit in feature space. By chance, test points
may land close to training points — a model only needs to interpolate, and the error
looks artificially low.

`KennardStoneSplit` uses the **maximin criterion**: each new training point is the one
furthest from all already-selected training points. The result is a training set that
spans the convex hull of the data, pushing test points into genuine gaps.

**Key metric:** average minimum distance from each test point to its nearest training
neighbour. Larger → test points are further from training → the error estimate reflects
extrapolation, not interpolation.
"""

# ╔═╡ 2b3c4d5e-6f70-4812-9345-bbccddeeff00
begin
  rng_ks = Xoshiro(7)
  n_ks = 300

  # 4-dimensional feature matrix (features × samples)
  X_ks = rand(rng_ks, Float32, 4, n_ks) .* Float32(2π)
  y_ks = sin.(X_ks[1, :]) .* cos.(X_ks[2, :]) .+ 0.05f0 .* randn(rng_ks, Float32, n_ks)

  "$(n_ks) samples  |  4 features  |  target: sin(x₁)·cos(x₂) + noise"
end

# ╔═╡ 3c4d5e6f-7081-4923-a456-ccddeeff0011
begin
  split_rand_ks = partition(X_ks, RandomSplit(); train = 80, test = 20, rng = rng_ks)
  split_ks = partition(X_ks, KennardStoneSplit(); train = 80, test = 20)

  # Average minimum distance from each test point to its nearest training point.
  function avg_min_dist(X, tr_idx, te_idx)
    X_tr = X[:, tr_idx]
    X_te = X[:, te_idx]
    mean(
      minimum(sqrt(sum((X_te[:, i] .- X_tr[:, j]) .^ 2)) for j = 1:size(X_tr, 2)) for
      i = 1:size(X_te, 2)
    )
  end

  d_rand = avg_min_dist(X_ks, trainindices(split_rand_ks), testindices(split_rand_ks))
  d_ks = avg_min_dist(X_ks, trainindices(split_ks), testindices(split_ks))

  (;
    random_avg_min_dist = round(d_rand; digits = 3),
    ks_avg_min_dist = round(d_ks; digits = 3),
    ratio = round(d_ks / d_rand; digits = 2),
  )
end

# ╔═╡ 4d5e6f70-8192-4034-b567-ddeeff001122
md"""
| Split | Avg min distance (test → train) |
|-------|--------------------------------|
| `RandomSplit` | $(round(d_rand; digits = 3)) |
| `KennardStoneSplit` | $(round(d_ks; digits = 3)) |

Kennard–Stone test points are **$(round(d_ks / d_rand; digits = 1))× further** from
their nearest training neighbour. A model evaluated on this split must genuinely
generalise — it cannot rely on nearby training examples for easy interpolation.

Use `LazyKennardStoneSplit` for datasets with tens of thousands of samples to avoid
the O(N²) memory cost of precomputing the full distance matrix.
"""

# ╔═╡ 5e6f7081-9203-4145-c678-eeff00112233
md"""
---
## 4 — SPXY: covering features and target jointly

`KennardStoneSplit` maximises diversity in feature space only. When the target variable
has a wide range or skewed distribution, feature-space coverage alone may leave part of
the response range under-represented in training. A model trained on a narrow target
range will extrapolate — and fail — at inference time for values outside that range.

`SPXYSplit` standardises X and y into a single combined distance, so the maximin
criterion operates on features **and** target simultaneously. The training set covers
both the feature space and the full response range.
"""

# ╔═╡ 6f708192-0314-4256-d789-ff0011223344
begin
  split_spxy = partition(X_ks, SPXYSplit(); target = y_ks, train = 80, test = 20)

  function target_stats(y, tr_idx)
    y_tr = y[tr_idx]
    lo, hi = extrema(y)
    lo_tr, hi_tr = extrema(y_tr)
    coverage = (hi_tr - lo_tr) / (hi - lo) * 100
    (;
      min = round(lo_tr; digits = 3),
      max = round(hi_tr; digits = 3),
      coverage_pct = round(coverage; digits = 1),
    )
  end

  stats_rand = target_stats(y_ks, trainindices(split_rand_ks))
  stats_ks = target_stats(y_ks, trainindices(split_ks))
  stats_spxy = target_stats(y_ks, trainindices(split_spxy))

  (; random = stats_rand, kennard_stone = stats_ks, spxy = stats_spxy)
end

# ╔═╡ 7081920a-1425-4367-e890-001122334455
md"""
| Split | Train y range | Coverage of full y range |
|-------|--------------|--------------------------|
| `RandomSplit` | [$(stats_rand.min), $(stats_rand.max)] | $(stats_rand.coverage_pct)% |
| `KennardStoneSplit` | [$(stats_ks.min), $(stats_ks.max)] | $(stats_ks.coverage_pct)% |
| `SPXYSplit` | [$(stats_spxy.min), $(stats_spxy.max)] | $(stats_spxy.coverage_pct)% |

`SPXYSplit` achieves the broadest target coverage by design. Use it — rather than
`KennardStoneSplit` — whenever the response range matters for calibration quality:
NIR spectroscopy, physical property prediction, or any regression task where a
narrow training y-range would produce a poorly calibrated model.

`MDKSSplit` is a variant that uses Mahalanobis distance for features (accounting for
inter-feature correlations) while keeping Euclidean distance for the target.
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
# ╟─1a2b3c4d-5e6f-4701-8234-aabbccddeeff
# ╠═2b3c4d5e-6f70-4812-9345-bbccddeeff00
# ╠═3c4d5e6f-7081-4923-a456-ccddeeff0011
# ╟─4d5e6f70-8192-4034-b567-ddeeff001122
# ╟─5e6f7081-9203-4145-c678-eeff00112233
# ╠═6f708192-0314-4256-d789-ff0011223344
# ╟─7081920a-1425-4367-e890-001122334455
