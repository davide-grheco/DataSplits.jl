using JSON, TOML, Printf, Dates

const DATADIR = ARGS[1]
const OUTFILE = ARGS[2]
const PYTHON = length(ARGS) >= 3 ? ARGS[3] : nothing
const CFG = TOML.parsefile(joinpath(@__DIR__, "manifest.toml"))
include(joinpath(@__DIR__, "cases.jl"))

const CAP_GIB = Int(CFG["budget"]["memory_cap_gib"])
# Probe rather than test for the binary: systemd-run can be installed but
# unable to reach the user bus, and then every cell fails instead of running
# uncapped.
cap_works() =
  success(`which systemd-run`) && success(
    pipeline(
      `systemd-run --user --scope -q -p MemoryMax=64M -- true`,
      stdout = devnull,
      stderr = devnull,
    ),
  )
const CAPPED = !haskey(ENV, "DATASPLITS_BENCH_NOCAP") && cap_works()
const D = Int(CFG["data"]["features"])

sizes() =
  haskey(ENV, "DATASPLITS_BENCH_SIZES") ?
  parse.(Int, split(ENV["DATASPLITS_BENCH_SIZES"], ",")) : Int.(CFG["data"]["sizes"])

const JULIA_CASES = [c.name for c in CASES]

const PYTHON_CASES = [
  "KennardStoneSplit",
  "SPXYSplit",
  "OptiSimSplit",
  "sphere_exclusion",
  "RandomSplit",
  "TargetPropertyHigh",
  "TargetPropertyLow",
  "TimeSplitOldest",
]
for n in PYTHON_CASES
  haskey(CASE_BY_NAME, n) ||
    error("rss_sweep: `$n` is in PYTHON_CASES but not in CASES; the two have drifted")
end

function measure(runtime::String, name, structure, D, N)
  inner =
    runtime == "julia" ?
    `julia --project=benchmark $(joinpath(@__DIR__, "rss.jl")) $DATADIR $name $structure $D $N` :
    `$PYTHON $(joinpath(@__DIR__, "rss.py")) $DATADIR $name $structure $D $N`
  cmd =
    CAPPED ?
    `systemd-run --user --scope -q -p MemoryMax=$(CAP_GIB)G -p MemorySwapMax=0 $inner` :
    inner

  env = copy(ENV)
  for v in
      ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "JULIA_NUM_THREADS")
    env[v] = "1"
  end

  out, err = IOBuffer(), IOBuffer()
  t0 = time()
  signal, code = 0, 0
  try
    proc = run(pipeline(setenv(cmd, env), stdout = out, stderr = err), wait = false)
    wait(proc)
    signal, code = proc.termsignal, proc.exitcode
  catch
  end
  elapsed = time() - t0

  line = strip(String(take!(out)))
  stderr_txt = String(take!(err))
  fields = split(line, '\t')

  if length(fields) >= 8 && fields[5] == "OK"
    return (
      runtime = runtime,
      strategy = name,
      structure = structure,
      D = D,
      N = N,
      status = "OK",
      seconds = parse(Float64, fields[6]),
      peak_gib = parse(Float64, fields[7]),
      net_gib = parse(Float64, fields[8]),
      wall_seconds = elapsed,
      detail = "",
    )
  end

  detail = isempty(line) ? first(replace(stderr_txt, '\n' => ' '), 300) : line
  killed = signal == 9 || code == 137
  oom =
    occursin(
      r"OutOfMemory|MemoryError|cannot allocate|bad_alloc|oom-kill"i,
      detail * stderr_txt,
    ) || (CAPPED && killed)
  status = oom ? "OOM" : "FAIL"
  if isempty(detail) && killed
    detail = "killed by signal 9 under the $(CAP_GIB) GiB cap"
  end
  return (
    runtime = runtime,
    strategy = name,
    structure = structure,
    D = D,
    N = N,
    status = status,
    seconds = NaN,
    peak_gib = NaN,
    net_gib = NaN,
    wall_seconds = elapsed,
    detail = detail,
  )
end

function main()
  grid = NamedTuple[]
  for N in sizes()
    for name in JULIA_CASES
      push!(grid, (runtime = "julia", name = name, N = N))
    end
    if PYTHON !== nothing
      for name in PYTHON_CASES
        push!(grid, (runtime = "python", name = name, N = N))
      end
    end
  end

  @printf(
    "cells=%d  cap=%s\n",
    length(grid),
    CAPPED ? string(CAP_GIB, " GiB (cgroup)") : "none"
  )
  println("-"^78)

  results = Any[]
  dead = Set{Tuple{String,String}}()

  for (i, g) in enumerate(grid)
    key = (g.runtime, g.name)
    if key in dead
      push!(
        results,
        Dict(
          "runtime" => g.runtime,
          "strategy" => g.name,
          "structure" => "isotropic",
          "D" => D,
          "N" => g.N,
          "status" => "SKIPPED",
          "detail" => "exceeded the cap at a smaller N",
        ),
      )
      @printf(
        "[%3d/%3d] %-8s %-28s N=%-7d  skipped (already over cap)\n",
        i,
        length(grid),
        g.runtime,
        g.name,
        g.N
      )
      continue
    end

    r = measure(g.runtime, g.name, "isotropic", D, g.N)
    push!(
      results,
      Dict(string(k) => (v isa Float64 && isnan(v)) ? nothing : v for (k, v) in pairs(r)),
    )
    if r.status == "OK"
      @printf(
        "[%3d/%3d] %-8s %-28s N=%-7d  %8.3f s  %7.4f GiB net\n",
        i,
        length(grid),
        g.runtime,
        g.name,
        g.N,
        r.seconds,
        r.net_gib
      )
    else
      @printf(
        "[%3d/%3d] %-8s %-28s N=%-7d  %s  %s\n",
        i,
        length(grid),
        g.runtime,
        g.name,
        g.N,
        r.status,
        first(r.detail, 60)
      )
      r.status == "OOM" && push!(dead, key)
    end
  end

  open(OUTFILE, "w") do io
    JSON.print(
      io,
      Dict(
        "rows" => results,
        "cap_gib" => CAP_GIB,
        "features" => D,
        "sizes" => sizes(),
        "julia_cases" => JULIA_CASES,
        "python_cases" => PYTHON === nothing ? String[] : PYTHON_CASES,
      ),
      2,
    )
  end
  @printf("\nwrote %s  (%d rows)\n", OUTFILE, length(results))

  nfail = count(r -> r["status"] == "FAIL", results)
  if nfail > 0
    @printf("%d cell(s) failed for reasons other than the memory cap\n", nfail)
    exit(1)
  end
end

main()
