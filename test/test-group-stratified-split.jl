using Test
using DataSplits
import DataSplits: SplitInputError, SplitParameterError

@testset "GroupStratifiedSplit" begin
  X = rand(4, 15)
  groups = vcat(fill(1, 5), fill(2, 5), fill(3, 5))

  @testset "Proportional allocation" begin
    s = GroupStratifiedSplit(:proportional)
    result = partition(X, s; groups = groups, train = 50, test = 50)
    train, test = result.train, result.test
    @test total_size(result) == 15
    @test is_disjoint(result)
    # Each group contributes to both train and test
    for gid in unique(groups)
      idxs = findall(==(gid), groups)
      @test any(i -> i in train, idxs)
    end
  end

  @testset "Equal allocation" begin
    s = GroupStratifiedSplit(:equal; n = 4)
    result = partition(X, s; groups = groups, train = 50, test = 50)
    train, test = result.train, result.test
    @test total_size(result) <= 12   # at most 4 per cluster × 3
    @test is_disjoint(result)
  end

  @testset "Neyman allocation" begin
    s = GroupStratifiedSplit(:neyman; n = 4)
    result = partition(X, s; groups = groups, train = 50, test = 50)
    train, test = result.train, result.test
    @test total_size(result) <= 15
    @test is_disjoint(result)
  end

  @testset "Unknown allocation raises error" begin
    s = GroupStratifiedSplit(:bogus)
    @test_throws SplitParameterError partition(X, s; groups = groups, train = 50, test = 50)
  end

  @testset "Fallback: groups as both data and groups" begin
    result = partition(groups, GroupStratifiedSplit(:proportional); train = 60, test = 40)
    @test total_size(result) == 15
    @test is_disjoint(result)
  end

  @testset "Neyman allocation on non-matrix data (issue #21)" begin
    # Tables.jl input (NamedTuple of vectors): used to crash with
    # `MethodError: no method matching std(::NamedTuple; dims=2)`.
    nt = (a = randn(15), b = randn(15), c = randn(15), d = randn(15))
    res = partition(
      nt,
      GroupStratifiedSplit(:neyman; n = 4);
      groups = groups,
      train = 50,
      test = 50,
    )
    @test is_disjoint(res)
    @test total_size(res) <= 15

    # Vector input: used to crash with `InexactError: Int64(NaN)` because
    # `std(::Vector; dims=2)` returned `NaN`.
    v = randn(15)
    res = partition(
      v,
      GroupStratifiedSplit(:neyman; n = 4);
      groups = groups,
      train = 50,
      test = 50,
    )
    @test is_disjoint(res)
    @test total_size(res) <= 15

    # Singleton group: within-group std falls back to 0 instead of NaN.
    groups_singleton = vcat(fill(1, 7), fill(2, 7), [3])
    Xs = rand(4, 15)
    res = partition(
      Xs,
      GroupStratifiedSplit(:neyman; n = 2);
      groups = groups_singleton,
      train = 50,
      test = 50,
    )
    @test is_disjoint(res)
  end

  @testset "Neyman allocation handles all-zero within-group dispersion" begin
    X = ones(4, 15)
    groups_constant = vcat(fill(1, 5), fill(2, 5), fill(3, 5))

    res = partition(
      X,
      GroupStratifiedSplit(:neyman; n = 2);
      groups = groups_constant,
      train = 50,
      test = 50,
    )

    @test is_disjoint(res)
    @test total_size(res) == 2 * length(unique(groups_constant))
  end
end
