using DataSplits
using Test

@testset "TargetPropertySplit" begin
  y = [10, 20, 30, 40, 50]

  s_high = TargetPropertyHigh(0.6)
  train, test = split(y, s_high)
  @test length(train) == 3
  @test all(y[i] >= y[j] for i in train, j in test)

  s_low = TargetPropertyLow(0.6)
  train, test = split(y, s_low)
  @test length(train) == 3
  @test all(y[i] <= y[j] for i in train, j in test)

  s_col = TargetPropertySplit(0.4, :desc)
  train, test = split(y, s_col)
  @test length(train) == 2
  @test all(y[i] >= y[j] for i in train, j in test)

  allidx = sort(vcat(train, test))
  @test allidx == [1, 2, 3, 4, 5]
end
