#!/usr/bin/Rscript
#
#' @name Generate batch data
#' @description Generate data from clusters across batches
generateBatchData <- function(N, P, 
                              cluster_means, 
                              std_dev, 
                              batch_shift, 
                              batch_var, 
                              cluster_weights, 
                              batch_weights,
                              row_names = paste0("Person_", 1:n),
                              col_names = paste0("Gene_", 1:p)) {

  # The number of clusters and batches to generate
  K <- length(cluster_means)
  B <- length(batch_shift)

  # The cluster and batch membership vector for the N points
  cluster_IDs <- sample(K, N, replace = T, prob = cluster_weights)
  batch_IDs <- sample(1:B, N, replace = T, prob = batch_weights)

  # The data matrix
  my_data <- my_corrected_data <- matrix(nrow = N, ncol = P)

  # Iterate over each of the columns permuting the parameters associated with
  # each label.
  for (p in 1:P)
  {
    reordered_cluster_means <- sample(cluster_means)
    reordered_std_devs <- sample(std_dev)
    
    reordered_batch_shift <- sample(batch_shift)
    reordered_batch_var <- sample(batch_var)

    for (n in 1:N) {
      
      # Draw the cluster defined univariate Gaussian.
      my_corrected_data[n, p] <- x <- rnorm(1,
        mean = reordered_cluster_means[cluster_IDs[n]],
        sd = reordered_std_devs[cluster_IDs[n]]
      )
      
      # Add batch specific noise to the data
      my_data[n, p] <- x + rnorm(1,
        mean = reordered_batch_shift[batch_IDs[n]],
        sd = reordered_batch_var[batch_IDs[n]]
      )
    }
  }

  # Order based upon allocation label
  row_order <- order(cluster_IDs)

  # Assign rownames and column names
  rownames(my_data) <- row_names
  colnames(my_data) <- col_names

  rownames(my_corrected_data) <- row_names
  colnames(my_corrected_data) <- col_names

  # Return the data, the data without batch effects, and the allocation and 
  # batch labels
  list(
    data = my_data[row_order, ],
    corrected_data = my_corrected_data[row_order, ],
    cluster_IDs = cluster_IDs[row_order],
    batch_IDs = batch_IDs[row_order]
  )
}
