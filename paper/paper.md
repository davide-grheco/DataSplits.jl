---
title: "DataSplits.jl: Data splitting strategies for machine learning in Julia"
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
  - name:
      Group of Computational Genomics and Hematology (GrHeCoXen), Health Research Institute of Santiago de Compostela
      (IDIS), Santiago de Compostela, Spain
    index: 1
  - name:
      Departamento de Farmacología, Farmacia y Tecnología Farmacéutica, Universidade de Santiago de Compostela, Santiago
      de Compostela, Spain
    index: 2
  - name:
      Division of Hematology, Complexo Hospitalario Universitario de Santiago de Compostela, SERGAS, Santiago de
      Compostela, Spain
    index: 3
date: 2 June 2026
bibliography: paper.bib
repository: https://github.com/davide-grheco/DataSplits.jl
---

# Summary

DataSplits.jl is a Julia package [@Bezanson2017Julia] for constructing training, validation, test, and cross-validation
partitions in scientific machine learning workflows. While random train/test splits and standard k-fold cross-validation
are widely available in machine learning libraries, many scientific applications require more controlled splitting
procedures. The package exposes a single entry point:

```julia
res = partition(data, strategy; kwargs...)
```

Supported strategies include distance-based representative sample selection for calibration modelling, group-aware
validation to avoid leakage between related observations, and time-aware validation for temporally ordered data. The
package is lightweight and interoperable with MLUtils.jl, Flux.jl, and MLJ.jl. It can be installed from the Julia
General Registry with `] add DataSplits`.

# Statement of need

The choice of data partitioning strategy can strongly affect the reliability of model evaluation, particularly in
small-data scientific domains such as chemometrics, spectroscopy, metabolomics, materials informatics, and quantitative
structure--property (QSPR) or structure--activity (QSAR) modelling. In these settings, independent and identically
distributed (i.i.d.) random splits may be inappropriate: samples can be highly structured, class distributions may be
imbalanced, repeated measurements may induce group leakage, and calibration sets may need to span the relevant feature
and response spaces [@kohavi1995study]. Representative sample selection algorithms such as Kennard--Stone/CADEX
[@Kennard1969Computer] and SPXY/MDKS [@Galvao2005method; @Saptoro2012Modified] are commonly used in chemometrics to
construct calibration and validation sets with broad coverage of the experimental domain. Related approaches include the
Duplex algorithm [@sneeValidation77a], which selects both training and test sets simultaneously using the maximin
criterion; OptiSim and dissimilarity-based selection methods [@Clark1997OptiSim]; the Onion method
[@gallagherSelection20] and its joint X--y extension, XYOnion [@ezenarro2025xy]; electrostatic field-strength selection
[@heField26]; spectral clustering-based splits [@ngSpectral01]; and classification-oriented adaptations such as the
Morais--Lima--Martin algorithm [@Lelis2019Improving]. Time-series and financial machine learning workflows call for
chronological splitting strategies such as purged k-fold cross-validation and combinatorial purged cross-validation
[@lopezdepradoAdvances18], and blocked cross-validation [@bergmeirUse12; @Roberts2017Cross]. Target-aware
cross-validation strategies such as Venetian blinds [@naes2002multivariate] assign samples sorted by response value
across folds in round-robin order, guaranteeing uniform target coverage without binning. To our knowledge, these methods
are not available together in any general-purpose Julia library.

DataSplits.jl collects them in a single package with a consistent interface, for researchers who need reproducible,
inspectable, and domain-appropriate partitions. For example, Kennard--Stone selection constructs a training set that
spans the descriptor space, which random splits cannot guarantee (\autoref{fig:ks}).

![Random and Kennard–Stone training-set selection on a synthetic dataset with a dense centre and a sparse boundary. Gray points denote all samples (N=120) and blue points the training set (n=35). Kennard–Stone covers the boundary of the experimental domain, whereas random selection concentrates in the centre.\label{fig:ks}](figures/ks_vs_random.png)

# State of the field

The closest Python equivalents are scikit-learn [@pedregosaScikitlearn18] and astartes [@Burns2023Machine]. Scikit-learn
provides group-aware and time-series CV strategies (`GroupKFold`, `TimeSeriesSplit`, `StratifiedGroupKFold`) but does
not implement distance-based selection methods such as Kennard--Stone, SPXY, or OptiSim. Astartes fills that gap for
molecular and materials data, offering Kennard--Stone and several dissimilarity-based splitters, but does not cover
group-aware CV, time-series CV, or nested cross-validation. DataSplits.jl covers both families: distance-based selection
and a broad catalogue of standard CV strategies. On the samplers the two packages share, DataSplits.jl is 1.2 to 27
times faster than astartes and uses less peak memory (\autoref{tab:astartes}). Both packages precompute the full
distance matrix for distance-based samplers, which limits the dataset size that fits in memory; DataSplits.jl also
provides lazy variants that allow working with data of any size.

<!-- prettier-ignore -->
| Sampler                |       N | DataSplits.jl (ms) | astartes (ms) | Speed-up | DataSplits.jl (GiB) | astartes (GiB) |
| :--------------------------- | ------: | -----------: | ---------: | --------: | -----------: | ---------: |
| Kennard--Stone         |  10,000 |                389 |         1,126 |      2.9 |                0.75 |           1.58 |
| SPXY                   |  10,000 |                987 |         1,555 |      1.6 |                1.48 |           2.33 |
| OptiSim                |  10,000 |                329 |         1,098 |      3.3 |                0.75 |           1.12 |
| Sphere exclusion       |  10,000 |                545 |           676 |      1.2 |                0.75 |           1.49 |
| Random                 | 100,000 |               0.45 |          12.1 |       27 |                0.02 |           0.04 |
| Target property (high) | 100,000 |                2.9 |          56.2 |       19 |                0.02 |           0.05 |
| Target property (low)  | 100,000 |                3.0 |          51.1 |       17 |                0.02 |           0.05 |
| Time-based             | 100,000 |                6.7 |          21.1 |      3.2 |                0.02 |           0.04 |

Table: Median runtime and peak memory of DataSplits.jl and astartes on identical synthetic inputs with 20 features.
Scripts and results are in `benchmark/experiment/`. \label{tab:astartes}

Within Julia, MLUtils.jl provides an observation interface based on `numobs`, `getobs`, and lazy observation views,
together with data loaders and basic splitting utilities [@mlutils]. Flux.jl re-exports MLUtils data-loading
functionality for neural network training [@Innes2018Flux]. MLJ.jl provides a model-evaluation framework with holdout
and cross-validation resampling, and also accepts explicit train/test row pairs [@Blaom2020MLJ]. None of these three
implement distance-based, group-stratified, or purged time-series splitting strategies. The closest Julia package in
scope is ChemometricsTools.jl [@chemometricstools], a broad chemometrics toolkit offering Kennard--Stone selection,
two-way `SplitByProportion` and `VenetianBlinds` splits, and `KFoldsValidation`/`LeaveOneOut` iterators, all returning
materialised arrays, but no group-stratified or purged time-series splitting.

DataSplits.jl complements these packages, its core contribution is split construction rather than model evaluation or
data loading; keeping it lightweight lets the same split definitions be reused across MLJ, Flux, MLUtils, and custom
workflows.

# Software design

The central abstraction in DataSplits.jl is a splitting strategy. Each strategy is represented by a concrete subtype of
`AbstractSplitStrategy`, while cross-validation strategies subtype `AbstractCVStrategy`. Users call
`partition(data, strategy; kwargs...)` to produce an `AbstractSplitResult` (a two-way, three-way, or per-fold
cross-validation split). A single entry point lets user code switch strategies without changing call sites.
\autoref{tab:families} lists the strategy families.

<!-- prettier-ignore -->
| Strategy family | Examples                                                                                               | Typical use                                                   |
| :----------------------------- | :--------------------------------------------------------------------------------------- | :----------------------------------------------------------- |
| Distance-based  | KennardStoneSplit, SPXYSplit, OptiSimSplit, DuplexSplit, OnionSplit, FieldStrengthSplit, SpectralSplit | Calibration/test design in chemometrics and descriptor spaces |
| Target-aware    | TargetPropertySplit, StratifiedKFold, StratifiedShuffleSplit, XYOnionSplit, VenetianBlindsCV           | Class-balanced or response-aware validation                   |
| Group-aware     | GroupShuffleSplit, GroupKFold, StratifiedGroupKFold                                                    | Avoiding leakage across related observations                  |
| Time-aware      | TimeSplit, BlockedCV, PurgedKFold, CombinatorialPurgedKFold                                            | Temporally ordered validation                                 |
| Resampling      | BootstrapSplit, ShuffleSplit, RepeatedKFold, NestedCV                                                  | Repeated performance estimation and hyperparameter tuning     |

Table: Splitting strategy families implemented in DataSplits.jl. \label{tab:families}

The package exposes stable accessors (e.g. `trainindices`), so downstream code does not need to depend on internal
result fields. Exposing indices rather than only data subsets lets the same split be applied to multiple aligned arrays
(features, labels, metadata) without re-running the selection algorithm. Data can be extracted either as lazy
MLUtils-compatible observation views or as materialised subsets.

All distance-based strategies have a lazy counterpart that computes distances on-the-fly in O(N) peak memory instead of
precomputing the full O(N²) distance matrix. This makes the strategies applicable to datasets where the full matrix
would not fit in RAM, at the cost of increased runtime. At N = 10,000 the lazy variants allocate 236 KiB where the eager
ones allocate 763 MiB, for a 1.4 to 2.4 times increase in runtime. Under a 4 GiB budget this raises the largest feasible
sample count from 10,000 to at least 100,000.

The package is designed to be extensible. Adding a custom strategy requires subtyping `AbstractSplitStrategy`, declaring
which auxiliary slots (`:data`, `:target`, `:time`, `:groups`) the strategy reads via the `consumes` trait, and
implementing a single `_partition` method. The `partition` entry point handles cohort-size resolution, slot validation,
and feature-matrix conversion automatically.

Ecosystem integration is kept lightweight: lazy views can be passed directly to Flux/MLUtils data loaders, and
`rowpairs` converts any split result into the `(train_indices, test_indices)` pairs accepted by MLJ's `evaluate!`,
without a dependency on MLJ.jl.

# Research impact statement

DataSplits.jl has been developed to support benchmarking and validation workflows in scientific machine learning where
split construction directly affects reported performance estimates. The package underpins a published companion study
[@crucitti2025splits], which demonstrates empirically that split strategy choice can dominate differences between
modelling algorithms in structured chemical datasets, with random splits systematically producing more optimistic
performance estimates than distance-based and group-aware alternatives. A reproducible tutorial
(`notebooks/tutorial.jl`) compares random, Kennard–Stone, SPXY, and group-aware splits on controlled datasets,
quantifying feature-space coverage and group leakage, and generates \autoref{fig:ks}. A benchmark suite
(`benchmark/benchmarks.jl`) tracks runtime scaling of all strategies. DataSplits.jl will be maintained by the GrHeCo-Xen
group through the public GitHub repository, where users can report issues, request features, and contribute new
splitting strategies.

# AI usage disclosure

Generative AI tools, including Claude Code (Anthropic), were used to assist with software development — specifically in
the integration of some algorithms — documentation drafting, and drafting and language editing of this paper. AI tools
were not used to make autonomous design, implementation, or scientific decisions. All AI-assisted code was thoroughly
verified against independent reference implementations and covered by the test suite. All AI-assisted text was reviewed
and edited by the authors, and technical claims were checked against the implementation, tests, documentation, and cited
literature. The authors are responsible for the final content of the manuscript, code, and documentation.

# Acknowledgements

This research project was made possible through the access granted by the Galician Supercomputing Center (CESGA) to its
supercomputing infrastructure. The supercomputer FinisTerrae III and its permanent data storage system have been funded
by the NextGeneration EU 2021 Recovery, Transformation and Resilience Plan, ICT2021-006904, and also from the
Pluriregional Operational Programme of Spain 2014-2020 of the European Regional Development Fund (ERDF),
ICTS-2019-02-CESGA-3, and from the State Programme for the Promotion of Scientific and Technical Research of Excellence
of the State Plan for Scientific and Technical Research and Innovation 2013-2016 State subprogramme for scientific and
technical infrastructures and equipment of ERDF, CESG15-DE-3114.

# References
