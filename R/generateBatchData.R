#!/usr/bin/Rscript
#
#' @title Generate batch data
#' @description Generate data from clusters across batches. Assumes independence
#' across columns. In each column the parameters are randomly permuted for both
#' the clusters and batches.
#' @param N The number of items (rows) to generate.
#' @param P The number of columns in the generated dataset.
#' @param cluster_means A vector of the cluster means for a column.
#' @param std_dev A vector of cluster standard deviations for a column.
#' @param batch_shift A vector of batch means in a column.
#' @param batch_var A vector of batch standard deviations within a column.
#' @param cluster_weights A vector of the expected proportion of N in each cluster.
#' @param batch_weights A vector of the expected proportion of N in each batch.
#' @param frac_known The expected fraction of observed labels. Used to generate
#' a ``fixed`` vector to feed into the ``batchSemiSupervisedMixtureModel`` function.
#' @return A list of 4 objects; the data generated from the clusters with and
#' without batch effects, the label indicating the generating cluster and the
#' batch label.
#' @importFrom stats rnorm
#' @examples
#' N <- 500
#' P <- 2
#' K <- 2
#' B <- 5
#' mean_dist <- 4
#' batch_dist <- 0.3
#' cluster_means <- 1:K * mean_dist
#' batch_shift <- stats::rnorm(B, sd = batch_dist)
#' std_dev <- rep(2, K)
#' batch_var <- rep(1.2, B)
#' cluster_weights <- rep(1 / K, K)
#' batch_weights <- rep(1 / B, B)
#'
#' my_data <- generateBatchData(
#'   N,
#'   P,
#'   cluster_means,
#'   std_dev,
#'   batch_shift,
#'   batch_var,
#'   cluster_weights,
#'   batch_weights
#' )
generateBatchData <- function(N, P,
                              cluster_means,
                              std_dev,
                              batch_shift,
                              batch_var,
                              cluster_weights,
                              batch_weights,
                              frac_known = 0.2) {

  # The number of clusters to generate
  K <- length(cluster_means)

  # The number of batches to generate
  B <- length(batch_shift)

  # The membership vector for the N points
  cluster_IDs <- sample(K, N, replace = T, prob = cluster_weights)

  # The batch labels for the N points
  batch_IDs <- sample(1:B, N, replace = T, prob = batch_weights)

  # The fixed labels for the semi-supervised case
  fixed <- sample(0:1, N, replace = T, prob = c(1 - frac_known, frac_known))
  
  # The data matrices
  my_data <- my_corrected_data <- matrix(nrow = N, ncol = P)

  # Iterate over each of the columns permuting the means associated with each
  # label.
  for (p in 1:P)
  {
    reordered_cluster_means <- sample(cluster_means)
    reordered_std_devs <- sample(std_dev)

    reordered_batch_shift <- sample(batch_shift)
    reordered_batch_var <- sample(batch_var)

    # Draw n points from the K univariate Gaussians defined by the permuted means.
    for (n in 1:N) {

      # Draw a point from a standard normal
      x <- stats::rnorm(1)

      # Adjust to the cluster distribution
      my_corrected_data[n, p] <- x * reordered_std_devs[cluster_IDs[n]] + reordered_cluster_means[cluster_IDs[n]]

      # Adjust to the batched cluster distribution
      my_data[n, p] <- x * reordered_std_devs[cluster_IDs[n]] * reordered_batch_var[batch_IDs[n]] + reordered_cluster_means[cluster_IDs[n]] + reordered_batch_shift[batch_IDs[n]]
    }
  }

  # Return the data, the data without batch effects, the allocation labels and
  # the batch labels.
  list(
    observed_data = my_data,
    corrected_data = my_corrected_data,
    cluster_IDs = cluster_IDs,
    batch_IDs = batch_IDs,
    fixed = fixed
  )
}
