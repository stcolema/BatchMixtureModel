\name{BatchMixtureModel-package}
\alias{BatchMixtureModel-package}
\alias{BatchMixtureModel}
\docType{package}

\title{Bayesian Mixture Modelling for Model-Based Clustering, Classification, and Batch Correction}

\description{

Bayesian multivariate normal and multivaraite t mixture models for joint clustering or classification and batch correction. Models estimated via Gibbs-within-Metropolis algorithm implemented in C++.
}


\author{
Stephen Coleman, Paul D.W. Kirk, Chris Wallace

Maintainer: Stephen Coleman \email{stcolema@tcd.ie}
}

\examples{

# Data in a matrix format
X <- matrix(c(rnorm(100, 0, 1), rnorm(100, 3, 1)), ncol = 2, byrow = TRUE)
# Initial labelling
labels <- c(rep(1, 10), 
  sample(c(1,2), size = 40, replace = TRUE), 
  rep(2, 10), 
  sample(c(1,2), size = 40, replace = TRUE)
)
fixed <- c(rep(1, 10), rep(0, 40), rep(1, 10), rep(0, 40))
# Batch
batch_vec <- sample(seq(1, 5), replace = TRUE, size = 100)
# Sampling parameters
R <- 1000
thin <- 50

# Classification
samples <- batchSemiSupervisedMixtureModel(X, R, thin, labels, fixed, batch_vec, "MVN")

# Clustering
samples <- batchUnsupervisedMixtureModel(X, R, thin, batch_vec, "MVT")

}

\keyword{package}