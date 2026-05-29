using Test
using DataSplits
using Distances
using Random
import DataSplits: SplitInputError, duplex_from_distance_matrix

rng_data = MersenneTwister(42)
X50 = randn(rng_data, 4, 50)

@testset "DuplexSplit basic properties" begin
  res = partition(X50, DuplexSplit(); train = 70, test = 30)
  train_idx, test_idx = res.train, res.test

  @test length(train_idx) + length(test_idx) == 50
  @test isempty(intersect(Set(train_idx), Set(test_idx)))
  @test Set(vcat(train_idx, test_idx)) == Set(1:50)
  @test length(train_idx) == 35
  @test length(test_idx) == 15
end

@testset "DuplexSplit dual coverage (1D)" begin
  # Key property: BOTH train and test should span the full range.
  # KennardStone only guarantees this for train; Duplex guarantees it for both.
  X1d = reshape(collect(1.0:20.0), 1, 20)
  res = partition(X1d, DuplexSplit(); train = 10, test = 10)
  train_vals = vec(X1d[:, res.train])
  test_vals = vec(X1d[:, res.test])

  @test minimum(train_vals) ≤ 3.0   # train covers low end
  @test maximum(train_vals) ≥ 18.0  # train covers high end
  @test minimum(test_vals) ≤ 3.0   # test covers low end
  @test maximum(test_vals) ≥ 18.0  # test covers high end
end

@testset "DuplexSplit vs KennardStone (test coverage)" begin
  # For KS, only train is spread across the range; test is "the rest".
  # For Duplex, both are spread.
  X1d = reshape(collect(1.0:30.0), 1, 30)
  res_ks = partition(X1d, KennardStoneSplit(); train = 15, test = 15)
  res_dp = partition(X1d, DuplexSplit(); train = 15, test = 15)

  ks_test_range = maximum(X1d[:, res_ks.test]) - minimum(X1d[:, res_ks.test])
  dp_test_range = maximum(X1d[:, res_dp.test]) - minimum(X1d[:, res_dp.test])

  # Duplex test set should span a wider range than KS's leftover test set
  @test dp_test_range ≥ ks_test_range
end

@testset "LazyDuplexSplit basic properties" begin
  res = partition(X50, LazyDuplexSplit(); train = 70, test = 30)
  @test length(res.train) + length(res.test) == 50
  @test Set(vcat(res.train, res.test)) == Set(1:50)
  @test length(res.train) == 35
  @test length(res.test) == 15
end

@testset "Duplex lazy/eager agreement" begin
  X1d = reshape(collect(1.0:20.0), 1, 20)
  r_eager = partition(X1d, DuplexSplit(); train = 10, test = 10)
  r_lazy = partition(X1d, LazyDuplexSplit(); train = 10, test = 10)

  @test Set(r_eager.train) == Set(r_lazy.train)
  @test Set(r_eager.test) == Set(r_lazy.test)
end

@testset "DuplexSplit target keyword rejected" begin
  y = randn(50)
  @test_throws SplitInputError partition(
    X50,
    DuplexSplit();
    target = y,
    train = 70,
    test = 30,
  )
end

@testset "DuplexSplit vector-of-vectors input" begin
  Xvov = [X50[:, i] for i = 1:50]
  res = partition(Xvov, DuplexSplit(); train = 70, test = 30)
  @test length(res.train) + length(res.test) == 50
  @test Set(vcat(res.train, res.test)) == Set(1:50)
end

@testset "DuplexSplit asymmetric sizes" begin
  res = partition(X50, DuplexSplit(); train = 40, test = 10)
  @test length(res.train) == 40
  @test length(res.test) == 10
  @test Set(vcat(res.train, res.test)) == Set(1:50)
end
