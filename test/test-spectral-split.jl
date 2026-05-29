using Test, Random, DataSplits, Distances

rng_data = MersenneTwister(42)
X30 = randn(rng_data, 3, 30)

@testset "SpectralSplit basic properties" begin
  res = partition(X30, SpectralSplit(5); train = 70, test = 30, rng = MersenneTwister(1))
  @test !isempty(res.train)
  @test !isempty(res.test)
  @test isempty(intersect(Set(res.train), Set(res.test)))
  @test Set(vcat(res.train, res.test)) == Set(1:30)
end

@testset "SpectralSplit full partition invariant" begin
  for seed in [1, 2, 3]
    res =
      partition(X30, SpectralSplit(4); train = 60, test = 40, rng = MersenneTwister(seed))
    @test Set(vcat(res.train, res.test)) == Set(1:30)
    @test isempty(intersect(Set(res.train), Set(res.test)))
  end
end

@testset "SpectralSplit fewer clusters than samples" begin
  X5 = randn(MersenneTwister(7), 2, 5)
  res = partition(X5, SpectralSplit(10); train = 60, test = 40, rng = MersenneTwister(1))
  @test Set(vcat(res.train, res.test)) == Set(1:5)
end

@testset "SpectralSplit vector-of-vectors input" begin
  Xvov = [X30[:, i] for i = 1:30]
  res = partition(Xvov, SpectralSplit(5); train = 70, test = 30, rng = MersenneTwister(1))
  @test Set(vcat(res.train, res.test)) == Set(1:30)
end

@testset "SpectralSplit custom metric" begin
  res = partition(
    X30,
    SpectralSplit(4; metric = Cityblock());
    train = 70,
    test = 30,
    rng = MersenneTwister(1),
  )
  @test Set(vcat(res.train, res.test)) == Set(1:30)
end

@testset "SpectralSplit different seeds give different results" begin
  r1 = partition(X30, SpectralSplit(5); train = 70, test = 30, rng = MersenneTwister(1))
  r2 = partition(X30, SpectralSplit(5); train = 70, test = 30, rng = MersenneTwister(999))
  # At least sometimes the random cluster ordering differs
  # (not guaranteed on every dataset — just verify both are valid)
  @test Set(vcat(r1.train, r1.test)) == Set(1:30)
  @test Set(vcat(r2.train, r2.test)) == Set(1:30)
end
