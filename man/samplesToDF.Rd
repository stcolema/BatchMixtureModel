% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/samplesToDF.R
\name{samplesToDF}
\alias{samplesToDF}
\title{Samples to data frame}
\usage{
samplesToDF(
  samples,
  type,
  R = nrow(samples$samples),
  thin = 1,
  keep_allocation = TRUE
)
}
\arguments{
\item{samples}{Output from the ``batchSemiSupervisedMixtureModel`` or
``batchMixtureModel``.}

\item{type}{The type of mixture model used; this changes which parameters
the function expects to find.}

\item{R}{The number of iterations run. Defaults to the number of slices in
the sampled batch mean array.}

\item{thin}{The thinning factor of the sampler. Defaults to 1.}

\item{keep_allocation}{A logical indicating if the final data frame should
include the sampled class/cluster membership variables.}
}
\value{
A wide data.frame of all the sampled parameters and the iteration.
}
\description{
Turns the output from the mixture models into a data.frame
}
\examples{
# Data in matrix format
X <- matrix(c(rnorm(100, 0, 1), rnorm(100, 3, 1)), ncol = 2, byrow = TRUE)

# Observed batches represented by integers
batch_vec <- sample(1:5, size = 100, replace = TRUE)

# MCMC iterations (this is too low for real use)
R <- 100
thin <- 5

# MCMC samples
samples <- batchMixtureModel(X, R, thin, batch_vec, "MVN")

samples_df <- samplesToDF(samples, "MVN", R = R, thin = thin)
}
