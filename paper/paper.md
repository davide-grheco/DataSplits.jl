---
title: 'DataSplits.jl: Data splitting strategies for machine learning in Julia'
tags:
  - Julia
  - machine learning
  - chemometrics
  - spectroscopy
  - cross-validation
  - resampling
  - data splitting
authors:
  - given-names: Davide
    surname: Crucitti
    orcid: 0009-0006-9458-9793
    corresponding: true
    affiliation: "1, 2"
  - given-names: Carlos
    surname: Pérez Míguez
    orcid: 0009-0004-7972-9431
    affiliation: 1
  - given-names: Adrián
    surname: Mosquera Orgueira
    orcid: 0000-0003-4838-6750
    affiliation: "1, 3"
affiliations:
  - name: Group of Computational Genomics and Hematology (GrHeCoXen), Health Research Institute of Santiago de Compostela (IDIS), Santiago de Compostela, Spain
    index: 1
  - name: Departamento de Farmacología, Farmacia y Tecnología Farmacéutica, Universidade de Santiago de Compostela, Santiago de Compostela, Spain
    index: 2
  - name: Division of Hematology, Complexo Hospitalario Universitario de Santiago de Compostela, SERGAS, Santiago de Compostela, Spain
    index: 3
date: 2 June 2026
bibliography: paper.bib
repository: https://github.com/davide-grheco/DataSplits.jl
---

# Summary

DataSplits.jl is a Julia package [@Bezanson2017Julia] for constructing training, validation, test, and cross-validation partitions in scientific machine learning workflows. While random train/test splits and standard k-fold cross-validation are widely available in machine learning libraries, many scientific applications require more controlled splitting procedures. The package exposes a single entry point:

```julia
res = partition(data, strategy; kwargs...)
```

The returned split objects expose train, validation, and test indices, lazy observation views, materialised subsets, and row-pair representations suitable for external model-evaluation frameworks. Supported strategy families include representative sample selection for calibration modelling, distance-based selection in spectral or molecular descriptor spaces, group-aware validation to avoid leakage between related observations, and time-aware validation for temporally ordered data. The package is designed to be lightweight and interoperable with the Julia machine learning ecosystem, including MLUtils.jl, Flux.jl, and MLJ.jl. It can be installed from the Julia General Registry with `] add DataSplits`. Source code is available at <https://github.com/davide-grheco/DataSplits.jl> and documentation at <https://davide-grheco.github.io/DataSplits.jl>.

# Statement of need

The choice of data partitioning strategy can strongly affect the reliability of model evaluation, particularly in small-data scientific domains such as chemometrics, spectroscopy, metabolomics, materials informatics, and quantitative structure--property (QSPR) or structure--activity (QSAR) modelling. In these settings, independent and identically distributed (i.i.d.) random splits may be inappropriate: samples can be highly structured, class distributions may be imbalanced, repeated measurements may induce group leakage, and calibration sets may need to span the relevant feature and response spaces [@kohavi1995study].
Representative sample selection algorithms such as Kennard--Stone/CADEX [@Kennard1969Computer] and SPXY/MDKS
[@Galvao2005method; @Saptoro2012Modified] are commonly used in chemometrics to construct calibration and validation
sets with broad coverage of the experimental domain. Related approaches include the Duplex algorithm
[@sneeValidation77a], which selects both training and test sets simultaneously using the maximin criterion;
OptiSim and dissimilarity-based selection methods [@Clark1997OptiSim]; the Onion method [@gallagherSelection20]
and its joint X--y extension, XY-Onion [@ezenarro2025xy]; electrostatic field-strength selection [@heField26];
spectral clustering-based splits; and classification-oriented adaptations such as the Morais--Lima--Martin
algorithm [@Lelis2019Improving]. Time-series and financial machine learning workflows call for chronological
splitting strategies such as purged k-fold cross-validation and combinatorial purged cross-validation
[@lopezdepradoAdvances18], and blocked cross-validation [@bergmeirUse12; @Roberts2017Cross].
Target-aware cross-validation strategies such as Venetian blinds [@naes2002multivariate] assign samples sorted
by response value across folds in round-robin order, guaranteeing uniform target coverage without binning. These methods are useful in scientific modelling but, to our knowledge, are not
currently available together in any general-purpose Julia splitting library.

DataSplits.jl addresses this gap by collecting these splitting strategies in a single Julia package with a consistent interface. It is intended for researchers who need reproducible, inspectable, and domain-appropriate partitions while remaining compatible with existing Julia modelling tools.

# State of the field

The closest Python equivalents are scikit-learn [@pedregosaScikitlearn18] and astartes [@Burns2023Machine]. Scikit-learn provides group-aware and time-series CV strategies (`GroupKFold`, `TimeSeriesSplit`, `StratifiedGroupKFold`) but does not implement distance-based selection methods such as Kennard--Stone, SPXY, or OptiSim. Astartes fills that gap for molecular and materials data, offering Kennard--Stone and several dissimilarity-based splitters, but does not cover group-aware CV, time-series CV, or nested cross-validation. DataSplits.jl covers both families — distance-based selection and a broad catalogue of standard CV strategies — in a consistent interface.

Within Julia, MLUtils.jl provides an observation interface based on `numobs`, `getobs`, and lazy observation views, together with data loaders and basic splitting utilities [@mlutils]. Flux.jl re-exports MLUtils data-loading functionality for neural network training [@Innes2018Flux]. MLJ.jl provides a comprehensive model-evaluation framework with built-in holdout and cross-validation resampling strategies, and also accepts explicit train/test row-pair vectors for custom resampling workflows [@Blaom2020MLJ]. To our knowledge, none of these packages implement distance-based, group-stratified, or purged time-series splitting strategies.

DataSplits.jl is not intended to replace these packages. Instead, it complements them by focusing on splitting strategies that are especially useful in scientific datasets and calibration/validation design. The output of these strategies can be consumed by other packages: lazy views can be passed to Flux/MLUtils data loaders, while explicit row pairs can be supplied to MLJ evaluation routines without requiring DataSplits.jl to depend on MLJ.jl.

DataSplits.jl was developed as a separate package rather than an extension to MLJ.jl or MLUtils.jl because its core contribution is split construction rather than model evaluation or data loading. Keeping the package lightweight allows the same split definitions to be reused across MLJ, Flux, MLUtils, and custom scientific workflows without imposing a dependency on any specific modelling framework.

# Software design

The central abstraction in DataSplits.jl is a splitting strategy. Each strategy is represented by a concrete subtype of `AbstractSplitStrategy`, while cross-validation strategies subtype `AbstractCVStrategy`. Users call `partition(data, strategy; kwargs...)` to produce an `AbstractSplitResult`. A single entry point is preferable to separate per-algorithm functions because it lets user code switch strategies without changing call sites. Result types include two-way train/test splits, three-way train/validation/test splits, and cross-validation splits containing one split per fold.

The package exposes stable accessors such as `trainindices`, `testindices`, `valindices`, and `folds`, so downstream code does not need to depend on internal result fields. Exposing indices rather than only data subsets matters in practice: the same split can be applied to multiple aligned arrays (features, labels, metadata, auxiliary covariates) without re-running the selection algorithm. For data extraction, DataSplits.jl provides both lazy and materialised interfaces. Functions such as `trainview`, `testview`, `valview`, and `splitview` use MLUtils-compatible observation views, while `traindata`, `testdata`, `valdata`, and `splitdata` materialise subsets using `getobs`.

All distance-based strategies have a lazy counterpart (e.g. `LazyKennardStoneSplit`, `LazyOptiSimSplit`) that computes distances on-the-fly in O(N) peak memory instead of precomputing the full O(N²) distance matrix. This makes the strategies applicable to datasets where the full matrix would not fit in RAM, at the cost of increased runtime — a trade-off that is important for the high-dimensional descriptor matrices common in chemometrics and molecular modelling.

The package is designed to be extensible. Adding a custom strategy requires subtyping `AbstractSplitStrategy`, declaring which auxiliary slots (`:data`, `:target`, `:time`, `:groups`) the strategy reads via the `consumes` trait, and implementing a single `_partition` method. The `partition` entry point handles cohort-size resolution, slot validation, and feature-matrix conversion automatically.

The time-aware strategies include `BlockedCV`, which implements blocked cross-validation [@bergmeirUse12] by partitioning the time axis into contiguous blocks to avoid look-ahead bias, and `PurgedKFold`, which implements purged k-fold cross-validation [@lopezdepradoAdvances18] by additionally removing observations from the training fold whose labels overlap in time with the test fold.

A key design choice is to keep ecosystem integration lightweight. DataSplits.jl provides `rowpairs`, which converts any split result into the vector of `(train_indices, test_indices)` pairs accepted by MLJ's `evaluate!` `resampling=` keyword, without requiring a dependency on MLJ.jl or MLJBase.jl. This avoids coupling split construction to any particular modelling framework.

The test suite checks split-size resolution, non-overlap between cohorts, complete fold coverage where applicable, reproducibility under fixed random number generators, stratification and group constraints, and consistency of lazy and materialised data extraction.

# Example usage

A representative train/test split can be constructed using the Kennard--Stone strategy:

```julia
using DataSplits

res = partition(X, KennardStoneSplit(); train = 0.8, test = 0.2)

X_train, X_test = splitview(res, X)
train_idx = trainindices(res)
test_idx  = testindices(res)
```

For stratified cross-validation with MLJ, the result can be converted to row pairs directly:

```julia
using DataSplits, MLJ, Random

cvs  = partition(X, StratifiedKFold(5); target = y, rng = Xoshiro(42))
mach = machine(model, X, y)
evaluate!(mach; resampling = rowpairs(cvs), measure = accuracy)
```

The same split indices can also be applied to multiple aligned data sources, enabling integration with Flux data loaders:

```julia
using DataSplits, Flux

res    = partition(X, KennardStoneSplit(); train = 0.8, test = 0.2)
loader = Flux.DataLoader(trainview(res, X, y); batchsize = 64, shuffle = true)
```

In chemometric calibration, Kennard–Stone-style selection constructs a training set that spans the observed descriptor space, reducing the risk that validation samples fall outside the calibration domain — a problem that random splits cannot guard against by design. The practical difference is illustrated in \autoref{fig:ks}.

![Comparison of random and Kennard–Stone training-set selection on a synthetic dataset with a dense central region and a sparse outer boundary. Gray points denote all samples (N=120) and blue points the selected training set (n=35). Random selection follows the empirical density and concentrates training samples in the center, whereas Kennard–Stone produces a more space-filling selection that better covers the boundary of the experimental domain.\label{fig:ks}](figures/ks_vs_random.png)

| Strategy family               | Examples                                                                                                                              | Typical use                                                   |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| Distance-based | KennardStoneSplit, SPXYSplit, OptiSimSplit, DuplexSplit, OnionSplit, FieldStrengthSplit, SpectralSplit | Calibration/test design in chemometrics and descriptor spaces |
| Target-aware                  | TargetPropertySplit, StratifiedKFold, StratifiedShuffleSplit, XYOnionSplit, VenetianBlindsCV                                          | Class-balanced or response-aware validation                   |
| Group-aware                   | GroupShuffleSplit, GroupKFold, StratifiedGroupKFold                                                                                   | Avoiding leakage across related observations                  |
| Time-aware                    | TimeSplit, BlockedCV, PurgedKFold, CombinatorialPurgedKFold                                                                           | Temporally ordered validation                                 |
| Resampling | BootstrapSplit, ShuffleSplit, RepeatedKFold, NestedCV                                                                                 | Repeated performance estimation and hyperparameter tuning     |

# Research impact statement

DataSplits.jl has been developed to support benchmarking and validation workflows in scientific machine learning
where split construction directly affects reported performance estimates. The package underpins a companion study
currently under review [@crucitti2025splits], which demonstrates empirically that split strategy choice can
dominate differences between modelling algorithms in structured chemical datasets, with random splits
systematically producing more optimistic performance estimates than distance-based and group-aware alternatives.
A reproducible tutorial (`notebooks/tutorial.jl`) included in the repository provides side-by-side comparisons
of random, Kennard–Stone, SPXY, and group-aware splits on controlled datasets, illustrating how coverage of
feature and response spaces differs across strategies and quantifying group leakage under random versus
group-aware partitioning. The tutorial also generates \autoref{fig:ks}. A benchmark
suite (`benchmark/benchmarks.jl`) tracks runtime scaling of all strategies, including lazy variants, to support
informed strategy selection on datasets of varying size. DataSplits.jl will be maintained by the GrHeCo-Xen
group through the public GitHub repository, where users can report issues, request features, and contribute new
splitting strategies.

# AI usage disclosure

Generative AI tools, including Claude Code (Anthropic), were used to assist with software development — specifically in the integration of some algorithms — documentation drafting, and drafting and language editing of this paper. AI tools were not used to make autonomous design, implementation, or scientific decisions. All AI-assisted code was thoroughly verified against independent reference implementations and covered by the test suite. All AI-assisted text was reviewed and edited by the authors, and technical claims were checked against the implementation, tests, documentation, and cited literature. The authors are responsible for the final content of the manuscript, code, and documentation.

# Acknowledgements

This research project was made possible through the access granted by the Galician Supercomputing Center (CESGA) to its supercomputing infrastructure. The supercomputer FinisTerrae III and its permanent data storage system have been funded by the NextGeneration EU 2021 Recovery, Transformation and Resilience Plan, ICT2021-006904, and also from the Pluriregional Operational Programme of Spain 2014-2020 of the European Regional Development Fund (ERDF), ICTS-2019-02-CESGA-3, and from the State Programme for the Promotion of Scientific and Technical Research of Excellence of the State Plan for Scientific and Technical Research and Innovation 2013-2016 State subprogramme for scientific and technical infrastructures and equipment of ERDF, CESG15-DE-3114.

# References
