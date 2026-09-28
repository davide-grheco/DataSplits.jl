using JSON, Printf

const RESULTS = ARGS[1]
const OUTDIR = length(ARGS) >= 2 ? ARGS[2] : RESULTS

load(name) = (p = joinpath(RESULTS, name); isfile(p) ? JSON.parsefile(p) : nothing)

function verdict(a, b)
  (get(a, "kind", "") == "error" || get(b, "kind", "") == "error") && return "divergent"
  a["kind"] != b["kind"] && return "divergent"
  if a["kind"] == "traintest"
    a["checksum"] == b["checksum"] && a["first20"] == b["first20"] && return "identical"
    a["n_train"] != b["n_train"] && return "divergent"
    a["checksum"] == b["checksum"] && return "equivalent"
    return "divergent"
  end
  a["checksum"] == b["checksum"] && a["fold_sizes"] == b["fold_sizes"] && return "identical"
  sort(a["fold_sizes"]) == sort(b["fold_sizes"]) && return "equivalent"
  return "divergent"
end

function gate(jl, py)
  ji = Dict(r["strategy"] => r["digest"] for r in jl["results"] if r["digest"] !== nothing)
  pi = Dict(r["strategy"] => r["digest"] for r in py["results"] if r["digest"] !== nothing)
  return Dict(n => verdict(ji[n], pi[n]) for n in intersect(keys(ji), keys(pi)))
end

index(rows) = Dict((r["strategy"], r["N"]) => r for r in rows)
iqr(r) = (r["time_q75_ns"] - r["time_q25_ns"]) / r["time_median_ns"]

function rss_index(rs)
  rs === nothing && return Dict()
  return Dict((r["runtime"], r["strategy"], r["N"]) => r for r in rs["rows"])
end

csv(x::Nothing) = ""
csv(x::AbstractFloat) = isnan(x) ? "" : @sprintf("%.6g", x)
csv(x) = string(x)
row(io, xs...) = println(io, join(csv.(xs), ","))

"""
comparison.csv: one row per sampler both packages implement, at the size it was
timed. The head-to-head the manuscript quotes.
"""
function write_comparison(jl, py, rs)
  verdicts = gate(jl, py)
  ji, pi = index(jl["results"]), index(py["results"])
  ri = rss_index(rs)
  open(joinpath(OUTDIR, "comparison.csv"), "w") do io
    row(
      io,
      "strategy",
      "verdict",
      "N",
      "datasplits_ms",
      "datasplits_iqr",
      "astartes_ms",
      "astartes_iqr",
      "ratio",
      "datasplits_peak_gib",
      "astartes_peak_gib",
      "datasplits_alloc_bytes",
    )
    for (s, n) in sort(collect(keys(pi)))
      haskey(ji, (s, n)) || continue
      j, p = ji[(s, n)], pi[(s, n)]
      v = get(verdicts, s, "unchecked")
      jr = get(ri, ("julia", s, n), nothing)
      pr = get(ri, ("python", s, n), nothing)
      peak(x) = (x === nothing || x["status"] != "OK") ? nothing : x["net_gib"]
      row(
        io,
        s,
        v,
        n,
        j["time_median_ns"] / 1e6,
        iqr(j),
        p["time_median_ns"] / 1e6,
        iqr(p),
        v == "divergent" ? nothing : p["time_median_ns"] / j["time_median_ns"],
        peak(jr),
        peak(pr),
        get(j, "alloc_bytes", nothing),
      )
    end
  end
end

"""
capacity.csv: peak memory for every cell, long format. OOM and SKIPPED are
results, not errors: they locate the point at which a strategy stops fitting.
"""
function write_capacity(rs)
  rs === nothing && return
  open(joinpath(OUTDIR, "capacity.csv"), "w") do io
    row(io, "runtime", "strategy", "N", "status", "peak_gib", "seconds")
    for r in sort(rs["rows"]; by = r -> (r["runtime"], r["strategy"], r["N"]))
      ok = r["status"] == "OK"
      row(
        io,
        r["runtime"],
        r["strategy"],
        r["N"],
        r["status"],
        ok ? r["net_gib"] : nothing,
        ok ? r["seconds"] : nothing,
      )
    end
  end
end

function main()
  jl = load("julia_main.json")
  jl === nothing && error("julia_main.json not found in $RESULTS")
  py, rs = load("python_main.json"), load("rss.json")
  mkpath(OUTDIR)
  py === nothing || write_comparison(jl, py, rs)
  write_capacity(rs)
  for f in ("comparison.csv", "capacity.csv")
    p = joinpath(OUTDIR, f)
    isfile(p) && println("wrote ", p, "  (", countlines(p) - 1, " rows)")
  end
end

main()
