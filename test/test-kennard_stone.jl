using Random
using Distances
using DataSplits
using Test
using NPZ

function check_split(result, N; n_test_expected = nothing)
  train, test = result.train, result.test
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
  result = split_fn(X, 0.8, Euclidean())
  check_split(result, 50; n_test_expected = 10)

  # Test 2: vector-of-vectors input
  result2 = split_fn(Xv, 0.3, CosineDist())
  check_split(result2, 50; n_test_expected = 35)

  # Test 3: deterministic behavior
  rng1 = MersenneTwister(123)
  result1 = split_fn(X, 0.25, Euclidean(); rng = rng1)
  tr1, te1 = result1.train, result1.test
  rng2 = MersenneTwister(123)
  result2b = split_fn(X, 0.25, Euclidean(); rng = rng2)
  tr2b, te2b = result2b.train, result2b.test
  @test tr1 == tr2b && te1 == te2b

  # Test 4: boundary handling
  @test_throws ArgumentError split_fn(rand(2, 10), 0.02, Euclidean())
  @test_throws ArgumentError split_fn(rand(2, 10), 0.98, Euclidean())

  # Test 5: edge cases
  @test_throws ArgumentError split_fn(rand(5, 5), 0.0, Euclidean())
  @test_throws ArgumentError split_fn(rand(5, 5), 1.0, Euclidean())

  # Test 6: max-min property
  X2 = vcat(randn(5, 2) .+ 5, randn(5, 2) .- 5)'
  labels = [x[1] > 0 ? 1 : -1 for x in getobs(X2)]
  result_small = split_fn(X2, 0.2, Euclidean())
  test_labels = getobs(labels, result_small.test)
  @test 1 in test_labels
  @test -1 in test_labels

  data_dir = joinpath(@__DIR__, "data")

  X = npzread(joinpath(data_dir, "kennard-stone-data.npy"))'
  train_idx_py = npzread(joinpath(data_dir, "kennard-stone-train-id.npy")) .+ 1

  # Test against astartes KS implementation
  result_py = split_fn(X, 0.75, Euclidean())
  train, test = result_py.train, result_py.test
  @test length(train) == length(train_idx_py)
  @test sort(train) == train_idx_py
end


@testset "LazyKennardStone (CADEX)" begin
  Random.seed!(42)
  X = rand(3, 50)
  Xv = [X[:, i] for i = 1:size(X, 2)]

  split_fn(X, frac, metric; rng = Random.GLOBAL_RNG) =
    DataSplits.split(X, LazyKennardStoneSplit(frac, metric); rng = rng)

  standard_kennard_tests(split_fn, X, Xv)
end

@testset "In Memory Kennard Stone" begin
  Random.seed!(42)
  X = rand(3, 50)
  Xv = [X[:, i] for i = 1:size(X, 2)]

  split_fn(X, frac, metric; rng = Random.GLOBAL_RNG) =
    DataSplits.split(X, KennardStoneSplit(frac, metric); rng = rng)

  standard_kennard_tests(split_fn, X, Xv)
end

@testset "Kennard Stone Consistency" begin
  Random.seed!(42)
  X = rand(4, 100)

  for frac in [0.2, 0.5, 0.8]
    for metric in [Euclidean(), CosineDist()]
      strat1 = KennardStoneSplit(frac, metric)
      strat2 = LazyKennardStoneSplit(frac, metric)

      result1 = DataSplits.split(X, strat1)
      result2 = DataSplits.split(X, strat2)

      @test Set(result1.train) == Set(result2.train)
      @test Set(result1.test) == Set(result2.test)

      @test length(result1.test) == length(result2.test)
    end
  end
end
