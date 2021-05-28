#!/usr/bin/Rscript
#' @title Predict class
#' @description Predicts a final class for each item given a matrix of 
#' allocation probabilities.
#' @param prob Output from the ``calcAllocProb`` function, a N x K matrix of 
#' allocation probabilities.
#' @return An N vecotr of class allocations.
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
#' preds <- predictClass(probs)
#' @export
predictClass <- function(prob) {
  
  pred_cl <- apply(prob, 1, which.max)
  
  pred_cl
}