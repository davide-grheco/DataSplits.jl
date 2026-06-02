# Contributing to DataSplits.jl

Thank you for your interest in contributing. This document explains how to report issues, request features, and submit code changes.

## Reporting issues and requesting features

Please open a [GitHub issue](https://github.com/davide-grheco/DataSplits.jl/issues) for:

- **Bug reports** — include a minimal reproducible example, the Julia version (`julia --version`), and the DataSplits.jl version (`] status DataSplits`).
- **Feature requests** — describe the splitting strategy or API improvement, ideally with a reference to the algorithm's original publication.
- **Documentation gaps** — point to the section that is unclear or missing.

## Getting support

For usage questions, open a GitHub issue with the `question` label or start a thread on the [Julia Discourse](https://discourse.julialang.org/) under the Machine Learning category.

## Contributing code

### Setting up the development environment

```julia
# Clone and activate the project
git clone https://github.com/davide-grheco/DataSplits.jl
cd DataSplits.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Install the pre-commit hooks (requires Python and [pre-commit](https://pre-commit.com/)):

```bash
pip install pre-commit
pre-commit install
```

### Running tests

```julia
julia --project=. test/runtests.jl
```

Or from the Julia REPL:

```julia
] test
```

### Running the linter

```bash
pre-commit run --all-files
```

### Code style

- **Indentation**: 2 spaces; Unix LF line endings; UTF-8 encoding.
- **Naming**: types in `CamelCase`, functions and variables in `snake_case`.
- **File names** match the primary type or function they define (e.g. `KennardStone.jl`).
- **Exports**: only in `src/DataSplits.jl`; never `export` from strategy files.
- **Imports**: `using` for public APIs, `import` for internals.
- **Matrix convention**: columns are samples, rows are features.
- **Errors**: use specific exception types (`ArgumentError`, `ErrorException`); never fail silently; provide actionable messages.
- **Docstrings**: Markdown triple-quoted strings with a description, fields, and a `# Examples` block.

### Adding a new splitting strategy

1. Create `src/strategies/MyStrategySplit.jl` with the `struct`, `consumes` trait, and `_partition` method.
2. Add `include("strategies/MyStrategySplit.jl")` and the `export` line in `src/DataSplits.jl`.
3. Add a test file `test/test-MyStrategySplit.jl` covering size resolution, non-overlap, and reproducibility.
4. Add a docstring entry in `docs/src/index.md`.
5. Run `pre-commit run --all-files` and the test suite before opening a pull request.

### Pull request checklist

- [ ] Tests pass locally
- [ ] Pre-commit hooks pass
- [ ] New strategy has a docstring with a `# Examples` block
- [ ] New strategy is exported in `src/DataSplits.jl`
- [ ] Documentation updated in `docs/src/index.md`
- [ ] `CHANGELOG.md` entry added under `[Unreleased]`

## Governance and maintenance

DataSplits.jl is maintained by the [GrHeCo-Xen group](https://github.com/davide-grheco). Pull requests are reviewed by the core maintainers. New splitting strategies are accepted when they have a published algorithmic reference and clear scientific motivation.
