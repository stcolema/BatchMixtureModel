#!/usr/bin/Rscript
#' @title Generate batch data
#' @description Generate data from groups across batches. Assumes independence
#' across columns. In each column the parameters are randomly permuted for both
#' the groups and batches.
#' @param N The number of items (rows) to generate.
#' @param P The number of columns in the generated dataset.
#' @param group_means A vector of the group means for a column.
#' @param std_dev A vector of group standard deviations for a column.
#' @param batch_shift A vector of batch means in a column.
#' @param batch_var A vector of batch standard deviations within a column.
#' @param group_weights A vector of the expected proportion of N in each group.
#' @param batch_weights A vector of the expected proportion of N in each batch.
#' @param frac_known The expected fraction of observed labels. Used to generate
#' a ``fixed`` vector to feed into the ``batchSemiSupervisedMixtureModel`` function.
#' @return A list of 4 objects; the data generated from the groups with and
#' without batch effects, the label indicating the generating group and the
#' batch label.
#' @importFrom stats rnorm
#' @export
#' @examples
#' N <- 500
#' P <- 2
#' K <- 2
#' B <- 5
#' mean_dist <- 4
#' batch_dist <- 0.3
#' group_means <- seq(1, K) * mean_dist
#' batch_shift <- stats::rnorm(B, sd = batch_dist)
#' std_dev <- rep(2, K)
#' batch_var <- rep(1.2, B)
#' group_weights <- rep(1 / K, K)
#' batch_weights <- rep(1 / B, B)
#'
#' my_data <- generateBatchData(
#'   N,
#'   P,
#'   group_means,
#'   std_dev,
#'   batch_shift,
#'   batch_var,
#'   group_weights,
#'   batch_weights
#' )
generateBatchData <- function(N, P,
                              group_means,
                              std_dev,
                              batch_shift,
                              batch_var,
                              group_weights,
                              batch_weights,
                              frac_known = 0.2) {

  # The number of groups to generate
  K <- length(group_means)

  # The number of batches to generate
  B <- length(batch_shift)

  # The membership vector for the N points
  group_IDs <- sample(seq(1, K), N, replace = TRUE, prob = group_weights)

  # The batch labels for the N points
  batch_IDs <- sample(seq(1, B), N, replace = TRUE, prob = batch_weights)

  # The fixed labels for the semi-supervised case
  fixed <- sample(seq(0, 1), N, 
    replace = TRUE, 
    prob = c(1 - frac_known, frac_known)
  )
  
  # The data matrices
  observed_data <- true_data <- matrix(nrow = N, ncol = P)

  # Iterate over each of the columns permuting the means associated with each
  # label.
  for (p in seq(1,P))
  {
    reordered_group_means <- sample(group_means)
    reordered_std_devs <- sample(std_dev)

    reordered_batch_shift <- sample(batch_shift)
    reordered_batch_var <- sample(batch_var)

    # Draw n points from the K univariate Gaussians defined by the permuted means.
    for (n in seq(1, N)) {

      # Draw a point from a standard normal
      x <- stats::rnorm(1)

      # For ease of reading the following lines, create group and batch parameters
      k <- group_IDs[n]
      b <- batch_IDs[n]
      
      .mu <- reordered_group_means[k]
      .sd <- reordered_std_devs[k]
      .m <- reordered_batch_shift[b]
      .s <- reordered_batch_var[b]
      
      # Adjust to the group distribution
      true_data[n, p] <- x * .sd + .mu

      # Adjust to the batched group distribution
      observed_data[n, p] <- x * .sd * .s + .mu + .m
    }
  }

  # Return the data, the data without batch effects, the allocation labels and
  # the batch labels.
  list(
    observed_data = observed_data,
    corrected_data = true_data,
    group_IDs = group_IDs,
    batch_IDs = batch_IDs,
    fixed = fixed
  )
}
