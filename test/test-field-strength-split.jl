using Test, Random, DataSplits, Distances, StableRNGs
import DataSplits: field_strength_from_distance_matrix

X50 = randn(StableRNG(42), 4, 50)

@testset "FieldStrengthSplit basic properties" begin
  res = partition(X50, FieldStrengthSplit(); train = 70, test = 30)
  @test length(res.train) + length(res.test) == 50
  @test isempty(intersect(Set(res.train), Set(res.test)))
  @test Set(vcat(res.train, res.test)) == Set(1:50)
  @test length(res.train) == 35
  @test length(res.test) == 15
end

@testset "FieldStrengthSplit covers extremes (1D)" begin
  X1d = reshape(collect(1.0:20.0), 1, 20)
  res = partition(X1d, FieldStrengthSplit(); train = 10, test = 10)
  train_vals = vec(X1d[:, res.train])
  @test minimum(train_vals) <= 2.0
  @test maximum(train_vals) >= 19.0
end

@testset "FieldStrengthSplit asymmetric sizes" begin
  res = partition(X50, FieldStrengthSplit(); train = 40, test = 10)
  @test length(res.train) == 40
  @test length(res.test) == 10
  @test Set(vcat(res.train, res.test)) == Set(1:50)
end

@testset "FieldStrengthSplit deterministic (no rng dependency)" begin
  r1 = partition(X50, FieldStrengthSplit(); train = 70, test = 30, rng = StableRNG(1))
  r2 = partition(X50, FieldStrengthSplit(); train = 70, test = 30, rng = StableRNG(99))
  @test Set(r1.train) == Set(r2.train)
end

@testset "FieldStrengthSplit custom metric" begin
  res = partition(X50, FieldStrengthSplit(Cityblock()); train = 70, test = 30)
  @test length(res.train) == 35
  @test Set(vcat(res.train, res.test)) == Set(1:50)
end

@testset "FieldStrengthSplit vector-of-vectors input" begin
  Xvov = [X50[:, i] for i = 1:50]
  res = partition(Xvov, FieldStrengthSplit(); train = 70, test = 30)
  @test length(res.train) + length(res.test) == 50
  @test Set(vcat(res.train, res.test)) == Set(1:50)
end

@testset "field_strength_from_distance_matrix public API" begin
  D = zeros(5, 5)
  for i = 1:5, j = 1:5
    D[i, j] = abs(i - j)
  end
  train, test = field_strength_from_distance_matrix(D, 3)
  @test length(train) == 3
  @test length(test) == 2
  @test isempty(intersect(Set(train), Set(test)))
  @test Set(vcat(train, test)) == Set(1:5)
end

@testset "FieldStrengthSplit golden output" begin
  # 1D equally-spaced grid — easy to audit by hand against the algorithm
  X1d = reshape(collect(1.0:15.0), 1, 15)
  res = partition(X1d, FieldStrengthSplit(); train = 8, test = 7)
  @test sort(res.train) == [1, 2, 4, 6, 8, 10, 12, 15]
  @test sort(res.test) == [3, 5, 7, 9, 11, 13, 14]

  X4d = randn(StableRNG(123), 4, 20)
  res = partition(X4d, FieldStrengthSplit(); train = 12, test = 8)
  @test sort(res.train) == [2, 3, 4, 6, 8, 9, 10, 11, 13, 15, 18, 20]
  @test sort(res.test) == [1, 5, 7, 12, 14, 16, 17, 19]
end
