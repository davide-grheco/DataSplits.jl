using Test, DataSplits, Distances, Clustering, StableRNGs

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

@testset "SphereExclusion is deterministic and seeds in index order" begin
  X = randn(StableRNG(11), 4, 300)

  a = sphere_exclusion(X; radius = 0.25).assignments
  b = sphere_exclusion(X; radius = 0.25).assignments
  @test a == b

  # Every sample lands in exactly one cluster.
  @test all(>(0), a)
  @test sort(unique(a)) == 1:maximum(a)

  # Sample 1 can never be anything but the first center
  @test a[1] == 1

  firsts = [findfirst(==(c), a) for c = 1:maximum(a)]
  @test issorted(firsts)
end
