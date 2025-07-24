#!/usr/bin/env julia
using TOML
using Downloads
using Pkg
using Printf

function usage()
  println("Usage: julia test_all_versions_generic.jl <PackageName> [YourPackageName]")
  exit(1)
end

if length(ARGS) < 1
  usage()
end

const DEP_NAME = ARGS[1]
const YOUR_PKG_NAME = length(ARGS) > 1 ? ARGS[2] : "DataSplits"
const YOUR_PKG_PATH = pwd()
const VERSIONS_URL = "https://raw.githubusercontent.com/JuliaRegistries/General/master/$(uppercase(first(DEP_NAME)))/$DEP_NAME/Versions.toml"

function fetch_all_versions()
  tmpfile = tempname()
  Downloads.download(VERSIONS_URL, tmpfile)
  toml = TOML.parsefile(tmpfile)
  return [Pkg.Types.VersionNumber(v) for v in keys(toml)]
end

function filter_versions(versions)
  keep_versions = Pkg.Types.VersionNumber[]
  by_major = Dict{Int,Vector{Pkg.Types.VersionNumber}}()
  for v in versions
    push!(get!(by_major, v.major, Pkg.Types.VersionNumber[]), v)
  end
  if haskey(by_major, 0)
    append!(keep_versions, sort(by_major[0]))
  end
  majors = sort([k for k in keys(by_major) if k > 0], rev = true)
  for maj in majors
    latest = maximum(by_major[maj])
    push!(keep_versions, latest)
  end
  return sort(keep_versions, rev = true)
end

function get_installed_version(dep)
  for (uuid, pkg) in Pkg.dependencies()
    if pkg.name == dep
      return pkg.version
    end
  end
  return nothing
end

# Run tests for a given version in a temp environment using Pkg.activate(; temp=true)
function run_in_temp_env(dep, majmin, patchver, your_pkg_name, your_pkg_path, julia_cmd)
  try
    Pkg.activate(; temp = true)
    Pkg.develop(PackageSpec(path = your_pkg_path))
    Pkg.add(PackageSpec(name = dep, version = string(patchver)))
    Pkg.resolve()
    actual_version = get_installed_version(dep)
    if actual_version != patchver
      println(
        "Could not install $dep $patchver, got $actual_version instead. Marking as incompatible.",
      )
      return false
    else
      println("Running tests on version $patchver")
    end
    Pkg.test(your_pkg_name)
    return true
  catch e
    println("Test run failed for $dep $patchver: ", e)
    return false
  end
end

function main()
  all_versions = sort(fetch_all_versions())
  test_versions = filter_versions(all_versions)
  julia_cmd = get(ENV, "JULIA_CMD", "julia")
  println("Will test versions ", test_versions)
  results = Dict{Pkg.Types.VersionNumber,Bool}()
  passed = []
  failure_stop = nothing
  for v in test_versions
    majmin = "$(v.major).$(v.minor)"
    println("  Testing $DEP_NAME = $majmin (patch $(string(v)))")
    ok = run_in_temp_env(DEP_NAME, majmin, v, YOUR_PKG_NAME, YOUR_PKG_PATH, julia_cmd)
    results[v] = ok
    @printf("    Result: %s\n", ok ? "PASS" : "FAIL")
    if ok
      append!(passed, [v])
    else
      failure_stop = v
      break
    end
  end
  passing_versions = [v for (v, ok) in results if ok]
  if !isempty(passing_versions)
    println("\nAll passing versions: ", join(string.(passing_versions), ", "))
    println("First Failure: $failure_stop")
    println(
      "Suggested compat entry: $DEP_NAME = \"$(join(["$(v.major).$(v.minor)" for v in passing_versions], ", "))\"",
    )
  else
    println("No passing versions found!")
  end
end

main()
