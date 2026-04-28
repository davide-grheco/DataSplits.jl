using Random
using Distances
using DataSplits
using Test
using NPZ
using MLUtils

function check_split(result, N; n_test_expected = nothing)
  train, test = result.train, result.test
  @test sort(vcat(train, test)) == 1:N
  @test isempty(intersect(train, test))
  if n_test_expected !== nothing
    @test length(test) == n_test_expected
    @test length(train) == N - length(test)
  end
end

# Helper: convert (frac, metric) into the new (train_pct, test_pct) form.
function _pct(frac)
  train = round(Int, frac * 100)
  test = 100 - train
  return train, test
end

function standard_kennard_tests(strategy_fn, X, Xv)
  Random.seed!(42)

  function call(X, frac, metric; rng = Random.GLOBAL_RNG)
    train, test = _pct(frac)
    return DataSplits.partition(X, strategy_fn(metric); train, test, rng)
  end

  result = call(X, 0.8, Euclidean())
  check_split(result, 50; n_test_expected = 10)

  result2 = call(Xv, 0.3, CosineDist())
  check_split(result2, 50; n_test_expected = 35)

  rng1 = MersenneTwister(123)
  result1 = call(X, 0.25, Euclidean(); rng = rng1)
  tr1, te1 = result1.train, result1.test
  rng2 = MersenneTwister(123)
  result2b = call(X, 0.25, Euclidean(); rng = rng2)
  tr2b, te2b = result2b.train, result2b.test
  @test tr1 == tr2b && te1 == te2b

  # Resolved counts < 1 must be rejected
  @test_throws SplitParameterError call(rand(2, 10), 0.02, Euclidean())
  @test_throws SplitParameterError call(rand(2, 10), 0.98, Euclidean())
  @test_throws SplitParameterError call(rand(5, 5), 0.0, Euclidean())
  @test_throws SplitParameterError call(rand(5, 5), 1.0, Euclidean())

  X2 = vcat(randn(5, 2) .+ 5, randn(5, 2) .- 5)'
  labels = [x[1] > 0 ? 1 : -1 for x in getobs(X2)]
  result_small = call(X2, 0.2, Euclidean())
  test_labels = getobs(labels, result_small.test)
  @test 1 in test_labels
  @test -1 in test_labels

  data_dir = joinpath(@__DIR__, "data")
  X = npzread(joinpath(data_dir, "kennard-stone-data.npy"))'
  train_idx_py = npzread(joinpath(data_dir, "kennard-stone-train-id.npy")) .+ 1

  N = numobs(X)
  n_train_py = length(train_idx_py)
  result_py = DataSplits.partition(
    X,
    strategy_fn(Euclidean());
    train = n_train_py,
    test = N - n_train_py,
  )
  train, test = result_py.train, result_py.test
  @test length(train) == n_train_py
  @test sort(train) == train_idx_py
end


@testset "LazyKennardStone (CADEX)" begin
  Random.seed!(42)
  X = rand(3, 50)
  Xv = [X[:, i] for i = 1:size(X, 2)]

  standard_kennard_tests(metric -> LazyKennardStoneSplit(metric), X, Xv)
end

@testset "In Memory Kennard Stone" begin
  Random.seed!(42)
  X = rand(3, 50)
  Xv = [X[:, i] for i = 1:size(X, 2)]

  standard_kennard_tests(metric -> KennardStoneSplit(metric), X, Xv)
end

@testset "Kennard Stone Consistency" begin
  Random.seed!(42)
  X = rand(4, 100)

  for frac in [0.2, 0.5, 0.8]
    for metric in [Euclidean(), CosineDist()]
      strat1 = KennardStoneSplit(metric)
      strat2 = LazyKennardStoneSplit(metric)
      train_pct, test_pct = _pct(frac)

      result1 = DataSplits.partition(X, strat1; train = train_pct, test = test_pct)
      result2 = DataSplits.partition(X, strat2; train = train_pct, test = test_pct)

      @test Set(result1.train) == Set(result2.train)
      @test Set(result1.test) == Set(result2.test)
      @test length(result1.test) == length(result2.test)
    end
  end
end
