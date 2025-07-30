using LinearAlgebra
using MLUtils: numobs, getobs
"""
    mahalanobis_split_distance(train::AbstractMatrix, test::AbstractMatrix) -> Float64

Computes the symmetric Mahalanobis-based distributional distance (Λ)
between training and test sets, as proposed by Jain et al. (2022). This
quantifies whether a given train/test split is statistically representative
of the overall dataset.

Assumes both datasets are continuous and have the same number of features.

!!! warning
    The number of samples in both train and test must be greater than the number of features. Otherwise, the covariance matrix will be singular and the Mahalanobis metric is not defined. Check your input data before using this metric.

# Arguments
- `train::AbstractMatrix`: Matrix of training samples.
- `test::AbstractMatrix`: Matrix of test samples.

# Returns
- `Float64`: Symmetric Mahalanobis distance Λ.

# Reference

Jain, E.; Neeraja, J.; Banerjee, B.; Ghosh, P. A Diagnostic Approach to Assess the Quality of Data Splitting in Machine Learning. arXiv 2022. https://doi.org/10.48550/ARXIV.2206.11721.

"""
function mahalanobis_split_distance(train::AbstractMatrix, test::AbstractMatrix)::Float64
  n_train, n_test = numobs(train), numobs(test)
  n_features = length(getobs(train, 1))
  if n_features != length(getobs(test, 1))
    throw(ArgumentError("Train and test must have the same number of features."))
  end

  if n_train < 2 || n_test < 2
    throw(
      ArgumentError("Each split must have at least 2 samples for covariance calculation."),
    )
  end

  if n_train + n_test < n_features
    throw(
      ArgumentError(
        "The Mahalanobis Split Distance metric is undefined when the number of samples is less than the number of features (It requires a matrix inversion on the covariance matrix). Check your input: you need more samples than features.",
      ),
    )
  end

  μ_train = mean(train, dims = 2)
  μ_test = mean(test, dims = 2)
  Σ_pooled = pooled_covariance([train, test])
  Σ_inv = inv(Σ_pooled)
  train_c = train .- μ_test
  test_c = test .- μ_train
  Δ²_train_to_test = sum(1:n_train) do i
    x = view(train_c, :, i)
    dot(x, Σ_inv * x)
  end / n_train
  Δ²_test_to_train = sum(1:n_test) do i
    x = view(test_c, :, i)
    dot(x, Σ_inv * x)
  end / n_test
  Λ = 0.5 * (Δ²_train_to_test + Δ²_test_to_train)
  return Λ
end
