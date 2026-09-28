using LinearAlgebra, Printf

include(joinpath(@__DIR__, "data.jl"))
include(joinpath(@__DIR__, "cases.jl"))

BLAS.set_num_threads(1)

function vmhwm_gib()
  for l in eachline("/proc/self/status")
    startswith(l, "VmHWM:") && return parse(Int, split(l)[2]) / 2^20
  end
  return NaN
end

const DATADIR, NAME, STRUCTURE = ARGS[1], ARGS[2], ARGS[3]
const D, N = parse(Int, ARGS[4]), parse(Int, ARGS[5])

case = CASE_BY_NAME[NAME]

try
  case.run(load_dataset(DATADIR, "isotropic", 20, 100))
catch
end
GC.gc(true)
baseline = vmhwm_gib()

d = load_dataset(DATADIR, STRUCTURE, D, N)
try
  t = @elapsed case.run(d)
  peak = vmhwm_gib()
  @printf(
    "%s\t%s\t%d\t%d\tOK\t%.4f\t%.4f\t%.4f\n",
    NAME,
    STRUCTURE,
    D,
    N,
    t,
    peak,
    max(peak - baseline, 0.0)
  )
catch e
  @printf(
    "%s\t%s\t%d\t%d\tFAIL\t0\t0\t0\t%s\n",
    NAME,
    STRUCTURE,
    D,
    N,
    first(sprint(showerror, e), 60)
  )
end
