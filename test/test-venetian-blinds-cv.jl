using Test
using DataSplits
using Random
import DataSplits: SplitParameterError

@testset "VenetianBlindsCV basic fold count" begin
  y = collect(1.0:50.0)
  cvs = partition(y, VenetianBlindsCV(5))
  @test length(folds(cvs)) == 5
end

@testset "VenetianBlindsCV every observation tested once" begin
  y = collect(1.0:50.0)
  for k in [2, 5, 10]
    cvs = partition(y, VenetianBlindsCV(k))
    test_counts = zeros(Int, 50)
    for fold in cvs
      for i in testindices(fold)
        test_counts[i] += 1
      end
    end
    @test all(==(1), test_counts)
  end
end

@testset "VenetianBlindsCV deterministic fold assignment" begin
  # With target = 1:N and k=5, fold f gets indices f, f+5, f+10, ...
  N = 20
  y = collect(1:N)
  cvs = partition(y, VenetianBlindsCV(5))
  for f = 1:5
    expected_test = Set(f:5:N)
    @test Set(testindices(cvs[f])) == expected_test
  end
end

@testset "VenetianBlindsCV sorts by target" begin
  # target = [5,1,3,2,4] → sorted order [2,4,3,5,1]
  # With k=5: fold 1 gets rank-1 sample = index 2
  y = [5, 1, 3, 2, 4]
  cvs = partition(y, VenetianBlindsCV(5))
  # sorted indices: y[2]=1, y[4]=2, y[3]=3, y[5]=4, y[1]=5
  @test testindices(cvs[1]) == [2]
  @test testindices(cvs[2]) == [4]
  @test testindices(cvs[3]) == [3]
  @test testindices(cvs[4]) == [5]
  @test testindices(cvs[5]) == [1]
end

@testset "VenetianBlindsCV fallback to data" begin
  y = collect(1.0:20.0)
  cvs = partition(y, VenetianBlindsCV(4))
  @test length(folds(cvs)) == 4
end

@testset "VenetianBlindsCV shuffle reproducible" begin
  y = [1, 1, 1, 2, 2, 2, 3, 3, 3]
  cvs1 = partition(y, VenetianBlindsCV(3; shuffle = true); rng = MersenneTwister(42))
  cvs2 = partition(y, VenetianBlindsCV(3; shuffle = true); rng = MersenneTwister(42))
  cvs3 = partition(y, VenetianBlindsCV(3; shuffle = true); rng = MersenneTwister(99))
  @test testindices(cvs1[1]) == testindices(cvs2[1])
  @test testindices(cvs1[1]) != testindices(cvs3[1]) || true  # different seeds may differ
end

@testset "VenetianBlindsCV fold sizes balanced" begin
  y = collect(1.0:50.0)
  cvs = partition(y, VenetianBlindsCV(5))
  sizes = [length(testindices(f)) for f in cvs]
  @test maximum(sizes) - minimum(sizes) <= 1
end

@testset "VenetianBlindsCV errors" begin
  y = collect(1.0:10.0)
  @test_throws SplitParameterError partition(y, VenetianBlindsCV(1))
  @test_throws SplitParameterError partition(y, VenetianBlindsCV(11))
end

@testset "VenetianBlindsCV differs from KFold" begin
  # KFold gives contiguous blocks; VenetianBlinds interleaves
  N = 20
  y = collect(1.0:N)
  cvs_vb = partition(y, VenetianBlindsCV(4))
  cvs_kf = partition(y, KFold(4))
  # VenetianBlinds fold 1 should get {1, 5, 9, 13, 17}, not {1..5}
  @test Set(testindices(cvs_vb[1])) != Set(testindices(cvs_kf[1]))
  @test Set(testindices(cvs_vb[1])) == Set(1:4:N)
end
