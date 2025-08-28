using BenchmarkTools
using DataSplits
using Random

println("Benchmarking SPXYSplit memory and time usage...")

Random.seed!(1234)

N = 2000
D = 100
X = randn(D, N)
y = randn(N)

splitter = SPXYSplit(0.7)

# Warmup
DataSplits.split((X, y), splitter)

# Benchmark
results = @benchmark DataSplits.split(($X, $y), $splitter)

mem = minimum(results).memory
med_time = median(results).time

println("Memory used: $(mem/1024^2) MB")
println("Median time: $(med_time/1e6) ms")

mem_threshold = 130 * 1024^2  # 130 MB (Current implementation consumes 122 MB)
if mem > mem_threshold
  println("WARNING: Memory usage above threshold!")
else
  println("Memory usage is within expected limits.")
end
