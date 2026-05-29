using Clustering: kmeans, assignments
using Distances
using LinearAlgebra
using Statistics: median

"""
    SpectralSplit <: AbstractSplitStrategy

Spectral graph-partitioning train/test split.

Builds an affinity graph over the samples, computes the normalised graph
Laplacian, embeds the data in the space spanned by its `n_clusters` smallest
eigenvectors, clusters with k-means, then assigns clusters to the training and
test cohorts (see Notes below).

Spectral splitting tends to produce train and test sets that are more
structurally separated than random or Kennard–Stone splits: samples within the
same cluster are similar; samples in different clusters are dissimilar.  This
creates a harder, more realistic evaluation scenario.

# Fields
- `n_clusters::Int`: Number of spectral clusters / k-means centres (default `10`).
- `metric::Distances.SemiMetric`: Distance metric for building the affinity
  matrix (default: `Euclidean()`).

# Notes
- **Approximate sizes.** Clusters are added to the training cohort in random
  order until `n_train` is reached; the actual count may exceed `n_train` by up
  to one cluster's worth of samples.
- **Bandwidth.** The RBF affinity kernel `W[i,j] = exp(−d²/(2σ²))` uses the
  median pairwise distance as the bandwidth `σ` (the *median heuristic*).
- Precomputes the full N×N distance matrix (O(N²) memory).

# Examples
```julia
res = partition(X, SpectralSplit(); train = 70, test = 30)
X_train, X_test = splitdata(res, X)

res = partition(X, SpectralSplit(20); train = 0.8, test = 0.2)
```

# References
Klarner et al. A tougher molecular data split: spectral split.
Oxford Protein Informatics Group Blog (2024).
<https://www.blopig.com/blog/2024/12/a-tougher-molecular-data-split-spectral-split/>.

Ng, A. Y.; Jordan, M. I.; Weiss, Y. On Spectral Clustering: Analysis and an
Algorithm. *NeurIPS* 2001.
"""
struct SpectralSplit <: AbstractSplitStrategy
  n_clusters::Int
  metric::Distances.SemiMetric
end

SpectralSplit(n_clusters::Integer = 10; metric = Euclidean()) =
  SpectralSplit(Int(n_clusters), metric)

consumes(::SpectralSplit) = (:data,)
fallback_from_data(::SpectralSplit) = ()

function _spectral_embed(D::AbstractMatrix, n_clusters::Integer)
  N = size(D, 1)

  # RBF affinity matrix using median-heuristic bandwidth
  σ = median(D[i, j] for i = 1:N for j = (i+1):N)
  σ = max(σ, 1e-10)
  W = exp.(-(D ./ σ) .^ 2 ./ 2)
  W[diagind(W)] .= 0  # no self-loops

  # Normalised graph Laplacian: L = I − D^{-1/2} W D^{-1/2}
  deg = vec(sum(W; dims = 2))
  deg_inv_sqrt = 1 ./ sqrt.(max.(deg, 1e-10))
  L = I - Diagonal(deg_inv_sqrt) * W * Diagonal(deg_inv_sqrt)

  # Smallest n_clusters eigenvectors (Symmetric → sorted ascending)
  vals, vecs = eigen(Symmetric(L))
  embedding = vecs[:, 1:n_clusters]  # N × n_clusters

  # Row-normalise (standard spectral clustering step)
  row_norms = sqrt.(sum(embedding .^ 2; dims = 2))
  embedding ./= max.(row_norms, 1e-10)

  return embedding
end

function _partition(
  X,
  s::SpectralSplit;
  n_train,
  n_test,
  rng = Random.default_rng(),
  kwargs...,
)
  N = numobs(X)
  n_clusters = min(s.n_clusters, N)

  D = distance_matrix(X, s.metric)
  embedding = _spectral_embed(D, n_clusters)

  # k-means on the spectral embedding (Clustering.jl: columns = observations)
  km = kmeans(Matrix(embedding'), n_clusters; maxiter = 300, rng = rng)
  labels = assignments(km)

  # Assign clusters to train/test (GroupShuffleSplit-style greedy)
  sorted_keys, perm = groupsortperm(labels)
  off = group_offsets(sorted_keys, perm, labels)
  n_groups = length(sorted_keys)

  block_order = shuffle(rng, collect(1:n_groups))
  train_idx = Int[]
  test_idx = Int[]
  for b in block_order
    idxs = perm[(off[b]+1):off[b+1]]
    if length(train_idx) < n_train
      append!(train_idx, idxs)
    else
      append!(test_idx, idxs)
    end
  end

  return TrainTestSplit(train_idx, test_idx)
end

function _partition(
  X::AbstractVector{<:AbstractVector},
  s::SpectralSplit;
  n_train,
  n_test,
  rng = Random.default_rng(),
  kwargs...,
)
  _partition(stack(X), s; n_train, n_test, rng)
end
