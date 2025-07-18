using Test, DataSplits, Distances

@testset "SphereExclusion edge cases" begin
  # Empty data
  X0 = zeros(0, 2)
  res0 = sphere_exclusion(X0; radius = 1.0)
  @test nclusters(res0) == 0
  @test counts(res0) == Int[]

  # Single point
  X1 = [1.0 2.0]
  res1 = sphere_exclusion(X1; radius = 0.5)
  @test nclusters(res1) == 1
  @test counts(res1) == [1]

  # Identical points
  X2 = repeat([0.0 0.0], inner = (5, 1))
  res2 = sphere_exclusion(X2; radius = 0.1)
  @test nclusters(res2) == 1
  @test counts(res2) == [5]

  # Zero radius → each point its own cluster
  X3 = [0.0 0.0; 1.0 0.0; 0.0 1.0]
  res3 = sphere_exclusion(X3; radius = 0.0)
  @test nclusters(res3) == 3
  @test sort(counts(res3)) == [1, 1, 1]

  # Large radius → single cluster
  res4 = sphere_exclusion(X3; radius = 10.0)
  @test nclusters(res4) == 1
  @test counts(res4) == [3]
end
