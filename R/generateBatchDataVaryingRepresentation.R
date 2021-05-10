#!/usr/bin/Rscript
#
#' @title Generate batch data
#' @description Generate data from clusters across batches. Assumes independence
#' across columns. In each column the parameters are randomly permuted for both
#' the clusters and batches.
#' @param N The number of items (rows) to generate.
#' @param P The number of columns in the generated dataset.
#' @param cluster_means A vector of the class means for a column.
#' @param std_dev A vector of class standard deviations for a column.
#' @param batch_shift A vector of batch means in a column.
#' @param batch_var A vector of batch standard deviations within a column.
#' @param cluster_weights A K x B matrix of the expected proportion of N in each class in each batch.
#' @param batch_weights A vector of the expected proportion of N in each batch.
#' @param row_names The row names of the final data matrix. Defaults to ``paste0("Person_", 1:N)``.
#' @param row_names The column names of the final data matrix. Defaults to ``paste0("Gene_", 1:P)``.
#' @return A list of 4 objects; the data generated from the clusters with and
#' without batch effects, the label indicating the generating cluster and the
#' batch label.
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
#' cluster_weights <- matrix(c(
#'   0.8, 0.6, 0.4, 0.2, 0.2,
#'   0.2, 0.4, 0.6, 0.8, 0.8),
#'   nrow = K, ncol = B, byrow = T
#' )
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
#' @export
generateBatchDataVaryingRepresentation <- function(N, 
                                                   P,
                                                   cluster_means,
                                                   std_dev,
                                                   batch_shift,
                                                   batch_var,
                                                   class_weights,
                                                   batch_weights,
                                                   frac_known = 0.2,
                                                   row_names = paste0("Person_", 1:N),
                                                   col_names = paste0("Gene_", 1:P)) {
  
  # The number of clusters to generate
  K <- length(cluster_means)
  
  # The number of batches to generate
  B <- length(batch_shift)
  
  if(ncol(class_weights) != B) {
    stop("Number of columns in class weight matrix does not match the number of batches.")
  }
  if(nrow(class_weights) != K) {
    stop("Number of rows in class weight matrix does not match the number of classes.")
  }
  
  # The membership vector for the N points, currently empty
  labels <- rep(0, N)
  
  # The batch labels for the N points
  batches <- sample(1:B, N, replace = T, prob = batch_weights)
  
  # The fixed labels for the semi-supervised case
  fixed <- sample(0:1, N, replace = T, prob = c(1 - frac_known, frac_known))
  
  # The data matrices
  corrected_data <- observed_data <- matrix(0, nrow = N, ncol = P)
  
  # Iterate over the batches to sample appropriate labels
  for(b in 1:B) {
    batch_ind <- which(batches == b)
    N_b <- length(batch_ind)
    labels[batch_ind] <- sample(1:K, N_b, replace = T, prob = class_weights[, b])
  }
  
  # Generate the data
  for(p in 1:P) {
    
    # To provide different information in each column, randomly sample the 
    # parameters with each class and batch
    reordered_cluster_means <- sample(cluster_means)
    reordered_std_devs <- sample(std_dev)
    
    reordered_batch_shift <- sample(batch_shift)
    reordered_batch_var <- sample(batch_var)
    
    
    for(n in 1:N) {
      
      # Find the class and batch
      b <- batches[n]
      k <- labels[n]
      
      # Random data
      x <- rnorm(1)
      
      # Corrected and observed data point
      corrected_data[n, p] <- x * reordered_std_devs[k] + reordered_cluster_means[k]
      observed_data[n, p] <- x * reordered_std_devs[k] * reordered_batch_var[b] + reordered_cluster_means[k] + reordered_batch_shift[b]
      
    }  
  }
  
  # Return a list of the class labels, batch labels, fixed points and the 
  # observed and true datasets.
  list(
    labels = labels,
    batches = batches,
    fixed  = fixed,
    observed_data = observed_data,
    corrected_data = corrected_data
  )
  
}