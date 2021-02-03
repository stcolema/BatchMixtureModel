#!/usr/bin/Rscript
#' @title Mixture model
#' @description A Bayesian mixture model.
#' @param X Data to cluster as a matrix with the items to cluster held in rows.
#' @param R The number of iterations in the sampler.
#' @param thin The factor by which the samples generated are thinned, e.g. if ``thin=50`` only every 50th sample is kept.
#' @param batch_vec Labels identifying which batch each item being clustered is from.
#' @param initial_labels Labels to begin from (if ``NULL`` defaults to a stick-breaking prior).
#' @param K_max The number of components to include (the upper bound on the number of clusters found).
#' @param alpha The concentration parameter for the stick-breaking prior and the weights in the model.
#' @param verbose The random seed for reproducibility.
#' @param mean_proposal_windows The proposal window for the mean proposal kernel.
#' @param cov_proposal_windows The proposal window for the cluster covariance proposal kernel.
#' @param S_proposal_windows The proposal window for the batch standard deviation proposal kernel.
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
mixtureModel <- function(X, R, thin, batch_vec,
                         initial_labels = NULL,
                         K_max = 50,
                         alpha = 1,
                         mean_proposal_window = 0.5**2,
                         cov_proposal_window = 100,
                         S_proposal_window = 100,
                         verbose = FALSE) {
  if (!is.matrix(X)) {
    stop("X is not a matrix. Data should be in matrix format.")
  }

  if (R < thin) {
    warning("Iterations to run less than thinning factor. No samples recorded.")
  }

  if (is.null(initial_labels)) {
    # Sample the stick breaking prior
    initial_labels <- priorLabels(alpha, K_max, nrow(X))
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

  # The concentration parameter for the prior Dirichlet distirbution of the
  # component weights.
  concentration <- rep(alpha, K_max)

  # Pull samples from the mixture model
  samples <- sampleMixtureModel(
    X,
    K_max,
    B,
    initial_labels,
    batch_vec,
    mean_proposal_window,
    cov_proposal_window,
    S_proposal_window,
    R,
    thin,
    concentration,
    seed,
    verbose
  )

  samples
}
