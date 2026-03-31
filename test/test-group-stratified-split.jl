using Test
using DataSplits
import DataSplits: SplitInputError, SplitParameterError

@testset "GroupStratifiedSplit" begin
  X = rand(4, 15)
  groups = vcat(fill(1, 5), fill(2, 5), fill(3, 5))

  @testset "Proportional allocation" begin
    s = GroupStratifiedSplit(:proportional; frac = 0.5)
    result = partition(X, s; groups = groups)
    train, test = result.train, result.test
    @test length(train) + length(test) == 15
    @test isempty(intersect(train, test))
    # Each group contributes to both train and test
    for gid in unique(groups)
      idxs = findall(==(gid), groups)
      @test any(i -> i in train, idxs)
    end
  end

  @testset "Equal allocation" begin
    s = GroupStratifiedSplit(:equal; n = 4, frac = 0.5)
    result = partition(X, s; groups = groups)
    train, test = result.train, result.test
    @test length(train) + length(test) <= 12   # at most 4 per cluster × 3
    @test isempty(intersect(train, test))
  end

  @testset "Neyman allocation" begin
    s = GroupStratifiedSplit(:neyman; n = 4, frac = 0.5)
    result = partition(X, s; groups = groups)
    train, test = result.train, result.test
    @test length(train) + length(test) <= 15
    @test isempty(intersect(train, test))
  end

  @testset "Unknown allocation raises error" begin
    s = GroupStratifiedSplit(:bogus; frac = 0.5)
    @test_throws SplitParameterError partition(X, s; groups = groups)
  end

  @testset "Fallback: groups as both data and groups" begin
    result = partition(groups, GroupStratifiedSplit(:proportional; frac = 0.6))
    @test length(result.train) + length(result.test) == 15
    @test isempty(intersect(result.train, result.test))
  end
end
