#!/usr/bin/Rscript
#' @title Caclulate allocation probabilities
#' @description Calculate the empirical allocation probability for each class
#' based on the sampled allocations.
#' @param samples Output from the ``batchSemiSupervisedMixtureModel`` or
#' ``batchMixtureModel``.
#' @param burn The number of samples to discard.
#' @return An N x K matrix of class probabilities.
#' @examples
#' # Convert data to matrix format
#' X <- as.matrix(my_data)
#'
#' # Sampling parameters
#' R <- 10000
#' thin <- 50
#'
#' # MCMC samples
#' samples <- batchMixtureModel(X, R, thin, type = "MVN")
#'
#' burn <- 2000
#' eff_burn <- burn / thin
#' probs <- allocProb(samples, burn = eff_burn)
#' @export
calcAllocProb <- function(samples, burn){
  if (burn > 0) {
    samples <- samples[, , -c(1:burn)]
  }
  prob <- rowSums(samples, dims = 2) / dim(samples)[3]
  prob
}
