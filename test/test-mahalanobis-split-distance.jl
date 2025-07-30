using MLUtils

@testset "Small matrix sanity test for mahalanobis_split_distance" begin
  train = [
    1.0 3.0
    3.0 3.0
  ]

  test = [
    4.0 4.0
    4.0 6.0
  ]

  Λ = mahalanobis_split_distance(train, test)

  @test isa(Λ, Float64)
  @test Λ > 0.0
  @test isapprox(Λ, 9, atol = 1e-5)
end

@testset "Iris dataset" begin
  X, y, names = MLUtils.load_iris()
  train = getobs(X, 1:100)
  test = getobs(X, 101:150)
  @test mahalanobis_split_distance(train, test) ≈ 11.287 atol = 1e-3
end

@testset "Error on too few samples (singular covariance)" begin
  # Fewer samples than features: should throw ArgumentError
  train = [1.0 1.0; 2.0 2.0; 3.0 3.0]  # 3 features, 2 samples
  test = [1.0 1.0; 2.0 2.0; 3.0 3.0]
  @test_throws ArgumentError mahalanobis_split_distance(train, test)
end
