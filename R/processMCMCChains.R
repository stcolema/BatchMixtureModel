#!/usr/bin/Rscript
#' @title Process MCMC chains
#' @description Applies a burn in to and finds a point estimate for each of the 
#' chains outputted from ``runMCMCChains``. 
#' @param mcmc_lst Output from ``runMCMCChains``
#' @param burn The number of MCMC samples to drop as part of a burn in.
#' @returns A named list similar to the output of 
#' ``batchSemiSupervisedMixtureModel`` with some additional entries:
#' \describe{
#' 
#'  \item {``mean_est``}{$(P x K)$ matrix. The point estimate of the cluster 
#'  means with columns  corresponding to clusters.}
#'  
#'  \item {``cov_est``}{$(P x P x K)$ array. The point estimate of the 
#'  cluster covariance matrices with slices corresponding to clusters.}
#'  
#'  \item {``shift_est``} {$(P x B)$ matrix. The point estimate of the batch 
#'  shift effect with columns  corresponding to batches.}
#'  
#'  \item {``scale_est``} {$(P x B)$ matrix. The point estimate of the batch
#'  scale effects. The $bth$ column contains the diagonal entries of the scaling 
#'  matrix for the $bth£ batch.}
#'  
#'  \item {``mean_sum_est``} {$(P x K x B)$ array. The point estimate of the
#'  sum of the cluster  means and the batch shift effect with columns 
#'  corresponding to clusters and slices to batches.}
#'  
#'  \item {``cov_comb_est``} {List of length $B$, with each entry being a 
#'  $(P x P x K)$ array. The point estimate of the combination of the 
#'  cluster covariance matrices and the batch scale effect with list entries
#'  corresponding to batches and slices of each array corresponding to clusters.}
#'  
#'  \item {``inferred_dataset``} {$(N x P)$ matrix. The inferred ``batch-free''
#'  dataset.}
#'  
#'  \item {``allocation_probability``} {$(N x K)$ matrix. The point estimate of 
#'  the allocation probabilities for each data point to each class.}
#'  
#'  \item {``prob``} {$N$ vector. The point estimate of the probability of being 
#'  allocated to the class with the highest probability.}
#'  
#'  \item {``pred``} {$N$ vector. The predicted class for each sample.}
#'  }
#' @export
#' @examples
#' 
#' # Data in a matrix format
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
#' burn <- 250
#' thin <- 50
#' n_chains <- 4
#'
#' # MCMC samples and BIC vector
#' samples <- runMCMCChains(X, n_chains, R, thin, labels, fixed, batch_vec, "MVN")
#' 
#' # Process the MCMC samples 
#' processed_samples <- processMCMCChains(samples, burn)
#' 
processMCMCChains <- function(mcmc_lst, burn) {
  
  new_output <- lapply(mcmc_lst, processMCMCChain, burn)
  
  # Return the MCMC object with burn in applied and point estimates found
  new_output
}
