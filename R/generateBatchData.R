#!/usr/bin/Rscript
#
#' @title Generate batch data
#' @description Generate data from clusters across batches.
#' @param N The number of items (rows) to generate.
#' @param P The number of columns in the generated dataset.
#' @param cluster_means A vector of the cluster means for a column.
#' @param std_dev A vector of cluster standard deviations for a column.
#' @param batch_shift A vector of batch means in a column.
#' @param batch_var A vector of batch standard deviations within a column.
#' @param cluster_weights A vector of the expected proportion of N in each cluster.
#' @param batch_weights A vector of the expected proportion of N in each batch.
#' @param row_names The row names of the final data matrix. Defaults to ``paste0("Person_", 1:N)``.
#' @param row_names The column names of the final data matrix. Defaults to ``paste0("Gene_", 1:P)``.
#' @return A list of 4 objects; the data generated from the clusters with and
#' without batch effects, the label indicating the generating cluster and the
#' batch label.
#' @examples
#' N <- 200
#' P <- 2
#' K <- 3
#' B <- 10
#' mean_dist <- 7
#' batch_dist <- 2
#' cluster_means <- 1:K * mean_dist
#' batch_shift <- rnorm(B, mean = batch_dist)
#' std_dev <- rep(2, K)
#' batch_var <- rep(1, B)
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
                              row_names = paste0("Person_", 1:N),
                              col_names = paste0("Gene_", 1:P)) {

  # The number of clusters to generate
  K <- length(cluster_means)

  # The number of batches to generate
  B <- length(batch_shift)

  # The membership vector for the N points
  cluster_IDs <- sample(K, N, replace = T, prob = cluster_weights)

  # The batch labels for the N points
  batch_IDs <- sample(1:B, N, replace = T, prob = batch_weights)

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
      x <- rnorm(1)

      # Adjust to the cluster distribution
      my_corrected_data[n, p] <- x * reordered_std_devs[cluster_IDs[n]] + reordered_cluster_means[cluster_IDs[n]]

      # Adjust to the batched cluster distribution
      my_data[n, p] <- x * reordered_std_devs[cluster_IDs[n]] * reordered_batch_var[batch_IDs[n]] + reordered_cluster_means[cluster_IDs[n]] + reordered_batch_shift[batch_IDs[n]]
    }
  }

  # Order based upon allocation label
  row_order <- order(cluster_IDs)

  # Assign rownames and column names
  rownames(my_data) <- row_names
  colnames(my_data) <- col_names

  rownames(my_corrected_data) <- row_names
  colnames(my_corrected_data) <- col_names

  # Return the data, the data without batch effects, the allocation labels and
  # the batch labels.
  list(
    data = my_data[row_order, ],
    corrected_data = my_corrected_data[row_order, ],
    cluster_IDs = cluster_IDs[row_order],
    batch_IDs = batch_IDs[row_order]
  )
}
