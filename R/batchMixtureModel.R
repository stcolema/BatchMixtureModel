#!/usr/bin/Rscript
#' @title Batch mixture model
#' @description A Bayesian mixture model with batch effects.
#' @param X Data to cluster as a matrix with the items to cluster held in rows.
#' @param R The number of iterations in the sampler.
#' @param thin The factor by which the samples generated are thinned, e.g. if
#' ``thin=50`` only every 50th sample is kept.
#' @param batch_vec Labels identifying which batch each item being clustered is
#' from.
#' @param type Character indicating density type to use. One of 'MVN' 
#' (multivariate normal distribution), 'MVT' (multivariate t distribution) or 
#' 'MSN' (multivariate skew normal distribution).
#' @param initial_labels Labels to begin from (if ``NULL`` defaults to a 
#' stick-breaking prior).
#' @param K_max The number of components to include (the upper bound on the 
#' number of clusters found in each sample).
#' @param alpha The concentration parameter for the stick-breaking prior and the 
#' weights in the model.
#' @param verbose The random seed for reproducibility.
#' @param mu_proposal_window The proposal window for the cluster mean proposal
#' kernel.
#' @param cov_proposal_window The proposal window for the cluster covariance 
#' proposal kernel.
#' @param m_proposal_window The proposal window for the batch mean proposal
#'  kernel.
#' @param S_proposal_window The proposal window for the batch standard deviation
#'  proposal kernel.
#' @param t_df_proposal_window The proposal window for the degrees of freedom 
#' for the multivariate t distribution (not used if type is not 'MVT').
#' @param phi_proposal_window The proposal window for the shape parameter for
#' the multivariate skew normal distribution (not used if type is not 'MSN').
#' @param verbose A bool indicating if the acceptance count for each parameter 
#' should be printed or not.
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
batchMixtureModel <- function(X, R, thin, batch_vec, type,
                              initial_labels = NULL,
                              K_max = 50,
                              alpha = 1,
                              mu_proposal_window = 0.5**2,
                              cov_proposal_window = 100,
                              m_proposal_window = 0.3**2,
                              S_proposal_window = 100,
                              t_df_proposal_window = 100,
                              phi_proposal_window = 1.2**2,
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
    stop("rho parameter must be a positive whole number.")
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
  
  if (is.null(initial_labels)) {
    # Sample the stick breaking prior
    initial_labels <- priorLabels(alpha, K_max, nrow(X))
  } else {
    if(length(initial_labels) != nrow(X)){
      stop("Number of membership labels does not equal the number of items in X.")
    }
  }
  
  # Check that the initial labels starts at 0, if not remedy this.
  if (!any(initial_labels == 0)) {
    initial_labels <- initial_labels %>%
      as.factor() %>%
      as.numeric() - 1
  }
  
  # Check that the batch labels starts at 0, if not remedy this.
  if (!any(batch_vec == 0)) {
    batch_vec <- batch_vec %>%
      as.factor() %>%
      as.numeric() - 1
  }
  
  # The number of batches present
  B <- length(unique(batch_vec))
  
  # The concentration parameter for the prior Dirichlet distribution of the
  # component weights.
  concentration <- rep(alpha, K_max)
  
  # Pull samples from the mixture model
  if(type == "MVN") {
    samples <- sampleMVN(
      X,
      K_max,
      B,
      initial_labels,
      batch_vec,
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
  }
  
  if(type == "MVT") {
    samples <- sampleMVT(
      X,
      K_max,
      B,
      initial_labels,
      batch_vec,
      mu_proposal_window,
      cov_proposal_window,
      m_proposal_window,
      S_proposal_window,
      t_df_proposal_window,
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
  }
  
  if(type == "MSN") {
    samples <- sampleMSN(
      X,
      K_max,
      B,
      initial_labels,
      batch_vec,
      mu_proposal_window,
      cov_proposal_window,
      m_proposal_window,
      S_proposal_window,
      phi_proposal_window,
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
  }
  
  if (! type %in% c("MVN", "MSN", "MVT")) {
    stop("Type not recognised. Please use one of 'MVN', 'MVT' or 'MSN'.")
  }
  samples
}
