using Test
using DataSplits
import DataSplits: SplitInputError, SplitParameterError

@testset "GroupStratifiedSplit" begin
  X = rand(4, 15)
  groups = vcat(fill(1, 5), fill(2, 5), fill(3, 5))

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
    @test_throws SplitParameterError GroupStratifiedSplit(:bogus)
  end

  @testset "Equal and Neyman require n" begin
    @test_throws SplitParameterError GroupStratifiedSplit(:equal)
    @test_throws SplitParameterError GroupStratifiedSplit(:neyman)
    @test_throws SplitParameterError GroupStratifiedSplit(:equal; n = 0)
    @test_throws SplitParameterError GroupStratifiedSplit(:neyman; n = -1)
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

const gss_prop_case_gen =
  @composed function make_gss_prop_case(n_groups = Data.Integers(2, 8))
    groups = Int[]
    for g = 1:n_groups
      group_size = Data.produce!(Data.Integers(2, 6))
      append!(groups, fill(g, group_size))
    end
    return groups
  end

@testset "GroupStratifiedSplit proportional properties" begin
  @check max_examples = 300 rng = Xoshiro(93) function gss_proportional_full_partition(
    groups = gss_prop_case_gen,
  )
    N = length(groups)
    X = reshape(collect(1.0:N), 1, N)
    result = partition(
      X,
      GroupStratifiedSplit(:proportional);
      groups = groups,
      train = 50,
      test = 50,
    )
    train = trainindices(result)
    is_full_partition(result, N) &&
      cohorts_are_complements(result, N) &&
      all(g -> any(i -> i in train, findall(==(g), groups)), unique(groups))
  end
end
