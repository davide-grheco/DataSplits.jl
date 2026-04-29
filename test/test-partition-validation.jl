using Test
using Random
using Distances
using DataSplits
import DataSplits: SplitInputError, SplitParameterError, SplitNotImplementedError

@testset "partition: train/validation/test" begin
  Random.seed!(42)
  N = 100
  X = rand(3, N)
  y = randn(N)
  groups = repeat(1:10, inner = 10)
  dates = repeat(1:10, inner = 10)

  @testset "two RandomSplit strategies, percentages" begin
    res = partition(
      X,
      RandomSplit(),
      RandomSplit();
      train = 70,
      validation = 10,
      test = 20,
      rng = MersenneTwister(1),
    )
    @test res isa DataSplits.TrainValTestSplit
    train, val, test = res.train, res.val, res.test
    @test length(train) == 70
    @test length(val) == 10
    @test length(test) == 20
    @test isempty(intersect(train, val))
    @test isempty(intersect(train, test))
    @test isempty(intersect(val, test))
    @test sort(vcat(train, val, test)) == 1:N
  end

  @testset "two RandomSplit strategies, absolute counts" begin
    res = partition(
      X,
      RandomSplit(),
      RandomSplit();
      train = 65,
      validation = 15,
      test = 20,
      rng = MersenneTwister(2),
    )
    @test length(res.train) == 65
    @test length(res.val) == 15
    @test length(res.test) == 20
  end

  @testset "splitdata returns 3-tuple" begin
    res = partition(
      X,
      RandomSplit(),
      RandomSplit();
      train = 70,
      validation = 10,
      test = 20,
      rng = MersenneTwister(3),
    )
    Xtr, Xv, Xte = splitdata(res, X)
    @test size(Xtr, 2) == 70
    @test size(Xv, 2) == 10
    @test size(Xte, 2) == 20
  end

  @testset "mixed slot consumers" begin
    # outer consumes :time, inner consumes :target
    res = partition(
      X,
      TimeSplitOldest(),
      TargetPropertyHigh();
      time = dates,
      target = y,
      train = 60,
      validation = 20,
      test = 20,
      rng = MersenneTwister(4),
    )
    @test res isa DataSplits.TrainValTestSplit
    @test length(res.train) + length(res.val) + length(res.test) == N
    @test isempty(intersect(res.train, res.val))
    @test isempty(intersect(res.train, res.test))
    @test isempty(intersect(res.val, res.test))
    # Test cohort is the most-recent dates (TimeSplitOldest pushes oldest to train pool)
    pool = vcat(res.train, res.val)
    @test maximum(dates[res.test]) >= minimum(dates[pool])
  end

  @testset "group-aware composition keeps groups whole" begin
    res = partition(
      X,
      GroupShuffleSplit(),
      GroupShuffleSplit();
      groups = groups,
      train = 60,
      validation = 20,
      test = 20,
      rng = MersenneTwister(5),
    )
    for gid in unique(groups)
      idxs = findall(==(gid), groups)
      hits = (
        any(i -> i in res.train, idxs),
        any(i -> i in res.val, idxs),
        any(i -> i in res.test, idxs),
      )
      @test count(hits) == 1
    end
  end

  @testset "reproducibility under fixed rng" begin
    res1 = partition(
      X,
      RandomSplit(),
      KennardStoneSplit();
      train = 70,
      validation = 10,
      test = 20,
      rng = MersenneTwister(123),
    )
    res2 = partition(
      X,
      RandomSplit(),
      KennardStoneSplit();
      train = 70,
      validation = 10,
      test = 20,
      rng = MersenneTwister(123),
    )
    @test res1.train == res2.train
    @test res1.val == res2.val
    @test res1.test == res2.test
  end

  @testset "input validation" begin
    # bad sum (three-cohort)
    @test_throws SplitParameterError partition(
      X,
      RandomSplit(),
      RandomSplit();
      train = 70,
      validation = 10,
      test = 30,
    )
    # empty data
    @test_throws SplitInputError partition(
      zeros(3, 0),
      RandomSplit();
      train = 70,
      test = 30,
    )
  end

  @testset "float fractions" begin
    res = partition(
      X,
      RandomSplit();
      train = 0.7,
      test = 0.3,
      rng = MersenneTwister(10),
    )
    @test length(res.train) == 70
    @test length(res.test) == 30

    res3 = partition(
      X,
      RandomSplit(),
      RandomSplit();
      train = 0.7,
      validation = 0.1,
      test = 0.2,
      rng = MersenneTwister(11),
    )
    @test length(res3.train) == 70
    @test length(res3.val) == 10
    @test length(res3.test) == 20
  end
end
