#!/usr/bin/Rscript
#' @title Caclulate allocation probabilities
#' @description Calculate the empirical allocation probability for each class
#' based on the sampled allocations.
#' @param samples Sampled allocation probabilities in output from the
#' ``batchSemiSupervisedMixtureModel`` or ``batchMixtureModel``.
#' @param burn The number of samples to discard.
#' @return An N x K matrix of class probabilities.
#' @examples
#' # Data in matrix format
#' X <- matrix(c(rnorm(100, 0, 1), rnorm(100, 3, 1)), ncol = 2, byrow = TRUE)
#' 
#' # Initial labelling
#' labels <- c(rep(1, 10), 
#'   sample(c(1,2), size = 40, replace = TRUE), 
#'   rep(2, 10), 
#'   sample(c(1,2), size = 40, replace = TRUE)
#' )
#' 
#' fixed <- c(rep(1, 10), rep(0, 40), rep(1, 10), rep(0, 40))
#' 
#' # Batch
#' batch_vec <- sample(seq(1, 5), replace = TRUE, size = 100)
#' 
#' # Sampling parameters
#' R <- 1000
#' thin <- 50
#'
#' # MCMC samples and BIC vector
#' samples <- batchSemiSupervisedMixtureModel(X, R, thin, labels, fixed, batch_vec, "MVN")
#' 
#' # Burn in
#' burn <- 20
#' eff_burn <- burn / thin
#' 
#' # Probability across classes
#' probs <- calcAllocProb(samples$alloc, burn = eff_burn)
#' @export
calcAllocProb <- function(samples, burn){
  if (burn > 0) {
    dropped_samples <- seq(1, burn)
    samples <- samples[, , -dropped_samples]
  }
  prob <- rowSums(samples, dims = 2) / dim(samples)[3]
  prob
}
