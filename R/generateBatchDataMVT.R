#!/usr/bin/Rscript
#
#' @title Generate batch data from a multivariate t distribution
#' @description Generate data from K multivariate t distributions with 
#' additional noise from batches. Assumes independence across columns. In each
#' column the parameters are randomly permuted for both the clusters and batches.
#' @param N The number of items (rows) to generate.
#' @param P The number of columns in the generated dataset.
#' @param cluster_means A vector of the class means for a column.
#' @param std_dev A vector of class standard deviations for a column.
#' @param batch_shift A vector of batch means in a column.
#' @param batch_var A vector of batch standard deviations within a column.
#' @param cluster_weights A K x B matrix of the expected proportion of N in each class in each batch.
#' @param batch_weights A vector of the expected proportion of N in each batch.
#' @param frac_known The number of items with known labels.
#' @param dfs A K-vector of the class specific degrees of freedom.
#' @return A list of 5 objects; the data generated from the clusters with and
#' without batch effects, the label indicating the generating cluster, the
#' batch label and the vector indicating training versus test.
#' @examples
#' N <- 500
#' P <- 2
#' K <- 2
#' B <- 5
#' mean_dist <- 4
#' batch_dist <- 0.3
#' cluster_means <- 1:K * mean_dist
#' batch_shift <- rnorm(B, mean = batch_dist, sd = batch_dist)
#' std_dev <- rep(2, K)
#' batch_var <- rep(1.2, B)
#' cluster_weights <- rep(1 / K, K)
#' batch_weights <- rep(1 / B, B)
#' dfs <- c(4, 7)
#' my_data <- generateBatchDataMVT(
#'   N,
#'   P,
#'   cluster_means,
#'   std_dev,
#'   batch_shift,
#'   batch_var,
#'   cluster_weights,
#'   batch_weights,
#'   dfs
#' )
#' @export
generateBatchDataMVT <- function(N,
                                 P,
                                 cluster_means,
                                 std_dev,
                                 batch_shift,
                                 batch_var,
                                 cluster_weights,
                                 batch_weights,
                                 dfs,
                                 frac_known = 0.2) {
  
  # The number of clusters to generate
  K <- length(cluster_means)
  
  # The number of batches to generate
  B <- length(batch_shift)
  
  # The membership vector for the N points
  cluster_IDs <- sample(K, N, replace = T, prob = cluster_weights)
  
  # The batch labels for the N points
  batch_IDs <- sample(1:B, N, prob = batch_weights, replace = T)
  
  # Fixed labels
  fixed <- sample(0:1, N, replace = T, prob = c(1 - frac_known, frac_known))
  
  # The data matrices
  observed_data <- true_data <- matrix(nrow = N, ncol = P)
  
  # Iterate over each of the columns permuting the means associated with each
  # label.
  for (p in 1:P)
  {
    
    # To provide different information in each column, randomly sample the 
    # parameters with each class and batch
    reordered_cluster_means <- sample(cluster_means)
    reordered_std_devs <- sample(std_dev)
    
    reordered_batch_shift <- sample(batch_shift)
    reordered_batch_var <- sample(batch_var)
    
    # Draw n points from the K univariate Gaussians defined by the permuted means.
    for (n in 1:N) {
      
      # Draw a point from a standard normal
      x <- rnorm(1)
      k <- cluster_IDs[n]
      b <- batch_IDs[n]
      
      chi_draw <- rchisq(1, dfs[k])
      
      # For ease of reading the following lines, create class and batch parameters
      .mu <- reordered_cluster_means[k]
      .sd <- reordered_std_devs[k]
      .m <- reordered_batch_shift[b]
      .s <- reordered_batch_var[b]
      
      # Adjust to the cluster distribution
      true_data[n, p] <- x * .sd * sqrt(dfs[k] / chi_draw) + .mu
      
      # Adjust to the batched cluster distribution
      observed_data[n, p] <- x * .sd * .s * sqrt(dfs[k] / chi_draw) + .mu + .m
    }
  }

  # Return the data, the data without batch effects, the allocation labels and
  # the batch labels.
  list(
    data = observed_data,
    corrected_data = true_data,
    cluster_IDs = cluster_IDs,
    batch_IDs = batch_IDs,
    fixed = fixed
  )
}

