using Test, DataSplits, Distances, Clustering

@testset "SphereExclusion edge cases" begin
  # Empty data
  X0 = zeros(2, 0)
  res0 = sphere_exclusion(X0; radius = 1.0)
  @test nclusters(res0) == 0
  @test counts(res0) == Int[]

  # Single point
  X1 = [1.0 2.0]'
  res1 = sphere_exclusion(X1; radius = 0.5)
  @test nclusters(res1) == 1
  @test counts(res1) == [1]

  # Identical points
  X2 = repeat([0.0; 0.0], inner = (1, 5))
  res2 = sphere_exclusion(X2; radius = 0.1)
  @test nclusters(res2) == 1
  @test counts(res2) == [5]

  # Zero radius → each point its own cluster
  X3 = [0.0 0.0; 1.0 0.0; 0.0 1.0]'
  res3 = sphere_exclusion(X3; radius = 0.0)
  @test nclusters(res3) == 3
  @test sort(counts(res3)) == [1, 1, 1]

  # Large radius → single cluster
  res4 = sphere_exclusion(X3; radius = 10.0)
  @test nclusters(res4) == 1
  @test counts(res4) == [3]
end

@testset "SphereExclusion properties" begin
  @check max_examples = 300 rng = Xoshiro(90) function sphere_exclusion_partition_invariant(
    N = Data.Integers(2, 30),
    r_int = Data.Integers(0, 10),
  )
    X = reshape(collect(1.0:N), 1, N)
    radius = Float64(r_int) * 0.5
    res = sphere_exclusion(X; radius = radius)
    a = assignments(res)
    length(a) == N && all(1 .<= a .<= nclusters(res)) && sum(counts(res)) == N
  end

  @check max_examples = 300 rng = Xoshiro(91) function sphere_exclusion_zero_radius_n_clusters(
    N = Data.Integers(2, 30),
  )
    X = reshape(collect(1.0:N), 1, N)
    res = sphere_exclusion(X; radius = 0.0)
    nclusters(res) == N
  end

  @check max_examples = 300 rng = Xoshiro(92) function sphere_exclusion_large_radius_one_cluster(
    N = Data.Integers(2, 30),
  )
    X = reshape(collect(1.0:N), 1, N)
    # max pairwise distance for 1:N is N-1; using N as radius is always ≥ that
    res = sphere_exclusion(X; radius = Float64(N))
    nclusters(res) == 1
  end
end
