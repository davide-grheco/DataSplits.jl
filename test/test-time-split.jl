using Dates
using DataSplits
using Test

@testset "TimeSplit" begin
  d = collect(Date(2020, 1, 1):Day(1):Date(2020, 1, 10))

  # Fallback: dates is both data and time ordering
  s = TimeSplitOldest()
  result = partition(d, s; train = 60, test = 40)
  train, test = result.train, result.test
  @test all(d[i] <= d[j] for i in train, j in test)
  @test length(train) + length(test) == 10
  @test length(train) >= 6

  # Explicit time= keyword
  X = rand(3, 10)
  result2 = partition(X, s; time = d, train = 60, test = 40)
  @test length(result2.train) + length(result2.test) == 10
  @test isempty(intersect(result2.train, result2.test))

  # All same date
  d2 = fill(Date(2020, 1, 1), 10)
  result = partition(d2, s; train = 60, test = 40)
  train, test = result.train, result.test
  @test (length(train) == 10 && isempty(test)) || (isempty(train) && length(test) == 10)

  # Ties: two groups
  d3 = vcat(fill(Date(2020, 1, 1), 5), fill(Date(2020, 1, 2), 5))
  result = partition(d3, s; train = 60, test = 40)
  train, test = result.train, result.test
  @test (length(train) == 5 || length(train) == 10)
  @test length(train) + length(test) == 10

  # Newest in train
  s2 = TimeSplitNewest()
  result = partition(d, s2; train = 50, test = 50)
  train, test = result.train, result.test
  @test all(d[i] >= d[j] for i in train, j in test)
end
