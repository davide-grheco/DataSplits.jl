using DataSplits
using Test

@testset "TargetPropertySplit" begin
  y = [10, 20, 30, 40, 50]

  # Fallback: y is both data and target
  s_high = TargetPropertyHigh()
  result_high = partition(y, s_high; train = 60, test = 40)
  train = trainindices(result_high)
  test = testindices(result_high)
  @test length(train) == 3
  @test all(y[i] >= y[j] for i in train, j in test)

  s_low = TargetPropertyLow()
  result = partition(y, s_low; train = 60, test = 40)
  train, test = result.train, result.test
  @test length(train) == 3
  @test all(y[i] <= y[j] for i in train, j in test)

  s_col = TargetPropertySplit(:desc)
  result = partition(y, s_col; train = 40, test = 60)
  train, test = result.train, result.test
  @test length(train) == 2
  @test all(y[i] >= y[j] for i in train, j in test)

  allidx = sort(vcat(train, test))
  @test allidx == [1, 2, 3, 4, 5]

  # Explicit target= keyword: split X by y values
  X = rand(3, 5)
  result2 = partition(X, TargetPropertyHigh(); target = y, train = 60, test = 40)
  @test length(trainindices(result2)) == 3
  @test isempty(intersect(result2.train, result2.test))
  # same indices as the fallback case
  @test Set(trainindices(result2)) == Set(trainindices(result_high))
end
