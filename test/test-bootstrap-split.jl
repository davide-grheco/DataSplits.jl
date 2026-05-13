using Test, Random, DataSplits
import DataSplits: SplitParameterError

@testset "BootstrapSplit" begin
  N = 60
  X = randn(2, N)

  @testset "Produces n_splits folds" begin
    cvs = partition(X, BootstrapSplit(20); rng = MersenneTwister(0))
    @test length(folds(cvs)) == 20
  end

  @testset "Train has exactly N indices (with duplicates)" begin
    cvs = partition(X, BootstrapSplit(10); rng = MersenneTwister(0))
    for f in folds(cvs)
      @test length(f.train) == N
      @test length(unique(f.train)) < N  # duplicates almost sure for moderate N
    end
  end

  @testset "Test is the out-of-bag (no overlap with train, unique)" begin
    cvs = partition(X, BootstrapSplit(10); rng = MersenneTwister(0))
    for f in folds(cvs)
      @test length(unique(f.test)) == length(f.test)
      @test isempty(intersect(Set(f.train), Set(f.test)))
      @test sort(collect(union(Set(f.train), Set(f.test)))) == 1:N
    end
  end

  @testset "OOB share is roughly 1 - 1/e on average" begin
    cvs = partition(X, BootstrapSplit(200); rng = MersenneTwister(42))
    oob_fractions = [length(f.test) / N for f in folds(cvs)]
    avg = sum(oob_fractions) / length(oob_fractions)
    # Theoretical limit ≈ 0.368; allow a generous tolerance.
    @test 0.30 <= avg <= 0.44
  end

  @testset "Resamples differ from each other" begin
    cvs = partition(X, BootstrapSplit(5); rng = MersenneTwister(0))
    fs = folds(cvs)
    @test fs[1].train != fs[2].train
    @test fs[1].test != fs[2].test
  end

  @testset "Reproducible with seeded rng" begin
    a = partition(X, BootstrapSplit(5); rng = MersenneTwister(7))
    b = partition(X, BootstrapSplit(5); rng = MersenneTwister(7))
    for (x, y) in zip(folds(a), folds(b))
      @test x.train == y.train
      @test x.test == y.test
    end
  end

  @testset "Parameter validation" begin
    @test_throws SplitParameterError partition(X, BootstrapSplit(0))
  end
end
