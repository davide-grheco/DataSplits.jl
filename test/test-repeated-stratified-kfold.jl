using Test
using Random
using DataSplits
import DataSplits: SplitParameterError

@testset "RepeatedStratifiedKFold" begin
  N = 40
  labels = vcat(fill(:a, 20), fill(:b, 20))

  @testset "Produces k * n_repeats folds" begin
    cvs = partition(rand(2, N), RepeatedStratifiedKFold(5; n_repeats = 3);
                    target = labels)
    @test length(folds(cvs)) == 15
  end

  @testset "Each repeat is a full partition" begin
    cvs = partition(rand(2, N), RepeatedStratifiedKFold(4; n_repeats = 2);
                    target = labels, rng = MersenneTwister(0))
    fs = folds(cvs)
    for r = 0:1
      slice = fs[(r*4+1):(r*4+4)]
      test_concat = sort(reduce(vcat, [f.test for f in slice]))
      @test test_concat == 1:N
    end
  end

  @testset "Class balance preserved within each fold" begin
    cvs = partition(rand(2, N), RepeatedStratifiedKFold(4; n_repeats = 3);
                    target = labels, rng = MersenneTwister(1))
    for f in folds(cvs)
      n_a = count(==(:a), labels[f.test])
      n_b = count(==(:b), labels[f.test])
      @test 3 <= n_a <= 7
      @test 3 <= n_b <= 7
    end
  end

  @testset "Reproducible with seeded rng" begin
    a = partition(rand(2, N), RepeatedStratifiedKFold(5; n_repeats = 3);
                  target = labels, rng = MersenneTwister(7))
    b = partition(rand(2, N), RepeatedStratifiedKFold(5; n_repeats = 3);
                  target = labels, rng = MersenneTwister(7))
    for (x, y) in zip(folds(a), folds(b))
      @test x.train == y.train
      @test x.test == y.test
    end
  end

  @testset "Continuous target with quantile bins" begin
    rng = MersenneTwister(2)
    y = randn(rng, 60)
    cvs = partition(rand(2, 60), RepeatedStratifiedKFold(5; n_repeats = 2, bins = 4);
                    target = y)
    @test length(folds(cvs)) == 10
  end

  @testset "Parameter validation" begin
    @test_throws SplitParameterError partition(rand(2, N),
                                                RepeatedStratifiedKFold(1; n_repeats = 3);
                                                target = labels)
    @test_throws SplitParameterError partition(rand(2, N),
                                                RepeatedStratifiedKFold(5; n_repeats = 0);
                                                target = labels)
    @test_throws SplitParameterError partition(rand(2, N),
                                                RepeatedStratifiedKFold(5; bins = 1);
                                                target = randn(N))
  end
end
