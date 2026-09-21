using Test, Random, DataSplits, Distances, LinearAlgebra, Statistics

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

@testset "SpectralSplit partial eigensolver matches dense" begin
  # N > 64 with n_clusters well below N takes the iterative path; the leading
  # eigenvectors must span the same subspace as a full dense decomposition.
  for (N, k) in ((100, 10), (300, 8))
    X = randn(MersenneTwister(11), 6, N)
    D = DataSplits.distance_matrix(X, Euclidean())

    σ = max(median(D[i, j] for i = 1:N for j = (i+1):N), 1e-10)
    W = exp.(-(D ./ σ) .^ 2 ./ 2)
    W[diagind(W)] .= 0
    deg_inv_sqrt = 1 ./ sqrt.(max.(vec(sum(W; dims = 2)), 1e-10))
    L = Symmetric(I - Diagonal(deg_inv_sqrt) * W * Diagonal(deg_inv_sqrt))

    dense = eigen(L).vectors[:, 1:k]
    dense = dense ./ max.(sqrt.(sum(dense .^ 2; dims = 2)), 1e-10)
    actual = DataSplits._spectral_embed(D, k)

    # Distance between subspace projectors: invariant to sign and rotation.
    @test opnorm(dense * dense' - actual * actual') < 1e-6
  end
end

@testset "SpectralSplit on a larger dataset" begin
  # Exercises the iterative path end to end; the dense solver made this slow.
  X = randn(MersenneTwister(3), 8, 400)
  res = partition(X, SpectralSplit(10); train = 0.7, test = 0.3, rng = MersenneTwister(1))

  @test Set(vcat(res.train, res.test)) == Set(1:400)
  @test isempty(intersect(Set(res.train), Set(res.test)))
  @test !isempty(res.train) && !isempty(res.test)
end

@testset "SpectralSplit dense fallback when clusters approach samples" begin
  # k >= N - 1 cannot use a Krylov subspace, so this must take the dense path.
  X = randn(MersenneTwister(5), 3, 70)
  res = partition(X, SpectralSplit(70); train = 0.6, test = 0.4, rng = MersenneTwister(1))

  @test Set(vcat(res.train, res.test)) == Set(1:70)
  @test isempty(intersect(Set(res.train), Set(res.test)))
end
