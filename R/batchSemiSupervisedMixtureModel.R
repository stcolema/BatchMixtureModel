#!/usr/bin/Rscript
#' @title Batch semi-supervised mixture model
#' @description A Bayesian mixture model with batch effects.
#' @param X Data to cluster as a matrix with the items to cluster held in rows.
#' @param initial_labels Initial clustering.
#' @param fixed Which items are fixed in their initial label.
#' @param batch_vec Labels identifying which batch each item being clustered is from.
#' @param R The number of iterations in the sampler.
#' @param thin The factor by which the samples generated are thinned, e.g. if ``thin=50`` only every 50th sample is kept.
#' @param K_max The number of components to include (the upper bound on the number of clusters found). Defaults to the number of unique labels in ``initial_labels``.
#' @param alpha The concentration parameter for the stick-breaking prior and the weights in the model.
#' @param verbose The random seed for reproducibility.
#' @param mu_proposal_window The proposal window for the cluster mean proposal kernel.
#' @param cov_proposal_window The proposal window for the cluster covariance proposal kernel.
#' @param m_proposal_window The proposal window for the batch mean proposal kernel.
#' @param S_proposal_window The proposal window for the batch standard deviation proposal kernel.
#' @param verbose A bool indicating if the acceptance count for each parameter should be printed or not.
#' @return Named list of the matrix of MCMC samples generated (each row
#' corresponds to a different sample) and BIC for each saved iteration.
#' @examples
#' # Convert data to matrix format
#' X <- as.matrix(my_data)
#'
#' # Sampling parameters
#' R <- 1000
#' thin <- 50
#'
#' # MCMC samples and BIC vector
#' samples <- mixtureModel(X, R, thin)
#'
#' # Predicted clustering and PSM
#' pred_cl <- mcclust::maxpear(samples$samples)$cl
#' psm <- createSimilarityMatrix(pred_cl)
#' @export
batchSemiSupervisedMixtureModel <- function(X, R, thin, initial_labels, fixed, batch_vec,
                              K_max = length(unique(initial_labels)),
                              alpha = 1,
                              mu_proposal_window = 0.5**2,
                              cov_proposal_window = 100,
                              m_proposal_window = 0.3**2,
                              S_proposal_window = 100,
                              rho = 41.0,
                              theta = 40.0,
                              lambda = 1.0,
                              verbose = FALSE,
                              doCombinations = FALSE,
                              printCovariance = FALSE) {
  if (!is.matrix(X)) {
    stop("X is not a matrix. Data should be in matrix format.")
  }
  
  if(length(batch_vec) != nrow(X)){
    stop("The number of rows in X and the number of batch labels are not equal.")
  }
  
  if(rho < 2.0) {
    stop("rho parameter must be a whole number greater than or equal to 2.")
  }
  
  if(theta < 1.0) {
    stop("rho parameter must be a positive whole.")
  }
  
  if(lambda <= 0.0) {
    stop("lambda must be a positive real number.")
  }
  
  # if(theta != (rho - 1)) {
  #   warning("The prior on the batch mean parameters is no longer expected to be standard normal.")
  # }
  
  if (R < thin) {
    warning("Iterations to run less than thinning factor. No samples recorded.")
  }
  
  # Check that the initial labels starts at 0, if not remedy this.
  if (!any(initial_labels == 0)) {
    initial_labels <- initial_labels %>%
      as.factor() %>%
      as.numeric() - 1
  }
  
  if(max(initial_labels) != (length(unique(initial_labels)) - 1)){
    stop("initial labels are not all contiguous integers.")
  }
  
  # Check that the batch labels starts at 0, if not remedy this.
  if (!any(batch_vec == 0)) {
    batch_vec <- batch_vec %>%
      as.factor() %>%
      as.numeric() - 1
  }
  
  if(max(batch_vec) != (length(unique(batch_vec)) - 1)){
    stop("batch labels are not all contiguous integers.")
  }
  
  # The number of batches present
  B <- length(unique(batch_vec))
  
  # The concentration parameter for the prior Dirichlet distirbution of the
  # component weights.
  concentration <- rep(alpha, K_max)
  
  # Pull samples from the mixture model
  samples <- sampleSemisupervisedMixtureModel(
    X,
    K_max,
    B,
    initial_labels,
    batch_vec,
    fixed,
    mu_proposal_window,
    cov_proposal_window,
    m_proposal_window,
    S_proposal_window,
    rho,
    theta,
    lambda,
    R,
    thin,
    concentration,
    verbose,
    doCombinations,
    printCovariance
  )
  
  samples
}