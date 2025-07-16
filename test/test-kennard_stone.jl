using Random
using Distances
using DataSplits
using Test
using NPZ

function check_split(train, test, N; n_test_expected = nothing)
  @test sort(vcat(train, test)) == 1:N
  @test isempty(intersect(train, test))
  if n_test_expected !== nothing
    @test length(test) == n_test_expected
    @test length(train) == N - length(test)
  end
end

function standard_kennard_tests(split_fn, X, Xv)
  Random.seed!(42)

  # Test 1: small numeric dataset (matrix)
  tr, te = split_fn(X, 0.8, Euclidean())
  check_split(tr, te, 50; n_test_expected = 10)

  # Test 2: vector-of-vectors input
  tr2, te2 = split_fn(Xv, 0.3, CosineDist())
  check_split(tr2, te2, 50; n_test_expected = 35)

  # Test 3: deterministic behavior
  rng1 = MersenneTwister(123)
  tr1, te1 = split_fn(X, 0.25, Euclidean(); rng = rng1)
  rng2 = MersenneTwister(123)
  tr2b, te2b = split_fn(X, 0.25, Euclidean(); rng = rng2)
  @test tr1 == tr2b && te1 == te2b

  # Test 4: boundary handling
  @test_throws ArgumentError split_fn(rand(10, 2), 0.02, Euclidean())
  @test_throws ArgumentError split_fn(rand(10, 2), 0.98, Euclidean())

  # Test 5: edge cases
  @test_throws ArgumentError split_fn(rand(5, 5), 0.0, Euclidean())
  @test_throws ArgumentError split_fn(rand(5, 5), 1.0, Euclidean())

  # Test 6: max-min property
  X2 = vcat(randn(5, 2) .+ 5, randn(5, 2) .- 5)
  tr_small, te_small = split_fn(X2, 0.2, Euclidean())
  labels = [x[1] > 0 ? 1 : -1 for x in eachrow(X2)]
  test_labels = labels[te_small]
  @test in(1, test_labels) && in(-1, test_labels)

  data_dir = joinpath(@__DIR__, "data")

  X = npzread(joinpath(data_dir, "kennard-stone-data.npy"))
  train_idx_py = npzread(joinpath(data_dir, "kennard-stone-train-id.npy")) .+ 1

  # Test against astartes KS implementation
  train, test = split_fn(X, 0.75, Euclidean())
  @test length(train) == length(train_idx_py)
  @test train == train_idx_py
end


@testset "LazyKennardStone (CADEX)" begin
  Random.seed!(42)
  X = rand(50, 3)
  Xv = [X[i, :] for i = 1:size(X, 1)]

  split_fn(X, frac, metric; rng = Random.GLOBAL_RNG) =
    DataSplits.split(X, LazyKennardStoneSplit(frac, metric); rng = rng)

  standard_kennard_tests(split_fn, X, Xv)
end

@testset "In Memory Kennard Stone" begin
  Random.seed!(42)
  X = rand(50, 3)
  Xv = [X[i, :] for i = 1:size(X, 1)]

  split_fn(X, frac, metric; rng = Random.GLOBAL_RNG) =
    DataSplits.split(X, KennardStoneSplit(frac, metric); rng = rng)

  standard_kennard_tests(split_fn, X, Xv)
end

@testset "Kennard Stone Consistency" begin
  Random.seed!(42)
  X = rand(100, 4)

  for frac in [0.2, 0.5, 0.8]
    for metric in [Euclidean(), CosineDist()]
      strat1 = KennardStoneSplit(frac, metric)
      strat2 = LazyKennardStoneSplit(frac, metric)

      tr1, te1 = DataSplits.split(X, strat1)
      tr2, te2 = DataSplits.split(X, strat2)

      @test Set(tr1) == Set(tr2)
      @test Set(te1) == Set(te2)

      @test length(te1) == length(te2)
    end
  end
end
