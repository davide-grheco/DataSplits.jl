using BenchmarkTools, Random, Statistics, JSON, TOML, LinearAlgebra, Printf, Dates

include(joinpath(@__DIR__, "data.jl"))
include(joinpath(@__DIR__, "cases.jl"))

const DATADIR = ARGS[1]
const OUTFILE = ARGS[2]
const CFG = TOML.parsefile(joinpath(@__DIR__, "manifest.toml"))

BLAS.set_num_threads(1)

function cells()
  override =
    haskey(ENV, "DATASPLITS_BENCH_SIZES") ?
    parse.(Int, split(ENV["DATASPLITS_BENCH_SIZES"], ","))[end] : nothing
  per_family = CFG["data"]["timing_n"]
  D = Int(CFG["data"]["features"])
  return [
    (
      case = c,
      structure = "isotropic",
      D = D,
      N = override === nothing ? Int(per_family[String(c.family)]) : override,
    ) for c in CASES
  ]
end

const _CACHE = Dict{Tuple{String,Int,Int},Any}()
dataset(structure, D, N) = get!(_CACHE, (structure, D, N)) do
  load_dataset(DATADIR, structure, D, N)
end

# Digest of a split, so the timing rows carry their own evidence that the two
# implementations computed the same thing. Only meaningful for strategies
# whose output is a deterministic function of the data.
const DETERMINISTIC = Set([
  "KennardStoneSplit",
  "SPXYSplit",
  "TargetPropertyHigh",
  "TargetPropertyLow",
  "TimeSplitOldest",
])

function digest(r)
  if r isa TrainTestSplit
    tr = trainindices(r) .- 1
    return Dict(
      "kind" => "traintest",
      "n_train" => length(tr),
      "checksum" => sum(tr),
      "first20" => tr[1:min(20, end)],
    )
  elseif r isa CrossValidationSplit
    te = sort(
      [sort(testindices(f) .- 1) for f in folds(r)];
      by = t -> (isempty(t) ? typemax(Int) : first(t)),
    )
    return Dict(
      "kind" => "cv",
      "fold_sizes" => length.(te),
      "checksum" => sum(k * sum(t) for (k, t) in enumerate(te); init = 0),
    )
  end
  return Dict("kind" => "other")
end

function measure(cell)
  d = dataset(cell.structure, cell.D, cell.N)
  run_case = cell.case.run

  dig = cell.case.name in DETERMINISTIC ? digest(run_case(d)) : nothing

  t = run(
    @benchmarkable($run_case($d)),
    seconds = Float64(CFG["budget"]["seconds_per_cell"]),
    samples = 10_000,
  )
  times = sort(t.times)

  return Dict(
    "strategy" => cell.case.name,
    "family" => String(cell.case.family),
    "eager_pair" => cell.case.eager_pair,
    "structure" => cell.structure,
    "D" => cell.D,
    "N" => cell.N,
    "time_median_ns" => median(times),
    "time_q25_ns" => quantile(times, 0.25),
    "time_q75_ns" => quantile(times, 0.75),
    "time_min_ns" => first(times),
    "samples" => length(times),
    "digest" => dig,
    "alloc_bytes" => t.memory,
    "alloc_count" => t.allocs,
  )
end

function main()
  grid = shuffle(Xoshiro(Int(CFG["seeds"]["strategy"])), cells())
  println("cells=$(length(grid))  strategies=$(length(CASES))")
  println("\n── measurement (shuffled) ────────────────────────────────────")

  rows = Dict{String,Any}[]
  broken = NamedTuple[]
  for (i, cell) in enumerate(grid)
    try
      r = measure(cell)
      push!(rows, r)
      @printf(
        "  [%2d/%2d] %-26s N=%-7d %9.3f ms  %9s\n",
        i,
        length(grid),
        cell.case.name,
        cell.N,
        r["time_median_ns"] / 1e6,
        Base.format_bytes(r["alloc_bytes"])
      )
    catch e
      msg = first(sprint(showerror, e), 200)
      push!(broken, (strategy = cell.case.name, N = cell.N, error = msg))
      @printf(
        "  [%2d/%2d] %-26s N=%-7d ERROR %s\n",
        i,
        length(grid),
        cell.case.name,
        cell.N,
        first(msg, 60)
      )
    end
  end

  out = Dict(
    "runtime" => "julia",
    "julia_version" => string(VERSION),
    "cpu" => Sys.cpu_info()[1].model,
    "blas_threads" => BLAS.get_num_threads(),
    "results" => rows,
    "broken" =>
      [Dict("strategy" => b.strategy, "N" => b.N, "error" => b.error) for b in broken],
  )

  mkpath(dirname(OUTFILE))
  open(io -> JSON.print(io, out, 2), OUTFILE, "w")
  println("\nwrote ", OUTFILE, "  (", length(rows), " rows)")

  if !isempty(broken)
    println("\n", length(broken), " case(s) could not run:")
    for b in broken
      println("  ", b.strategy, " at N=", b.N, ": ", first(b.error, 120))
    end
    exit(1)
  end
end

main()
