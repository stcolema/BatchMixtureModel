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
  
  # The number of distirbutions to sample from
  K <- length(cluster_means)
  
  B <- length(batch_shift)
  
  # The membership vector for the n points
  cluster_IDs <- sample(K, N, replace = T, prob = cluster_weights)
  
  batch_IDs <- sample(1:B, N, replace = T, prob = batch_weights)
  
  # The data matrix
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
  
  # Return the data and the allocation labels
  list(
    data = my_data[row_order, ],
    corrected_data = my_corrected_data[row_order, ],
    cluster_IDs = cluster_IDs[row_order],
    batch_IDs = batch_IDs[row_order]
  )
}
