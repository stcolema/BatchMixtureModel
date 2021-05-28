#!/usr/bin/Rscript
#' @title Get sampled cluster means
#' @description Given an array of sampled cluster means from the 
#' ``mixtureModel`` function, acquire a tidy version ready for ``ggplot2`` use.
#' @param sampled_cluster_means A 3D array of sampled cluster means. 
#' @param K The number of clusters present. Defaults to the number of columns in
#' the batch mean matrix from the first sample.
#' @param P The dimension of the batch mean shifts. Defaults to the number of 
#' rows in the batch mean matrix from the first sample.
#' @param R The number of iterations run. Defaults to the number of slices in
#' the sampled batch mean array.
#' @param thin The thinning factor of the sampler. Defaults to 1. 
#' @return A data.frame of three columns; the parameter, the sampled value and the iteration.
#' @examples
#' # Convert data to matrix format
#' X <- as.matrix(my_data)
#'
#' # Sampling parameters
#' R <- 1000
#' thin <- 50
#'
#' # MCMC samples
#' samples <- mixtureModel(X, R, thin)
#'
#' batch_shift_df <- getSampledClusterMeans(samples$means, R = R, thin = thin)
#' @importFrom tidyr pivot_longer 
#' @export
getSampledClusterMeans <- function(sampled_cluster_means, 
                                   K = dim(sampled_cluster_means)[2], 
                                   P = dim(sampled_cluster_means)[1],
                                   R = dim(sampled_cluster_means)[3],
                                   thin = 1) {
  
  # Check that the values of R and thin make sense
  if(floor(R/thin) != dim(sampled_cluster_means)[3]){
    stop("The ratio of R to thin does not match the number of samples present.")
  }
  
  # Stack the sampled matrices on top of each other
  mean_df <- data.frame(t(apply(sampled_cluster_means, 3L, rbind)))
  
  # Give sensible column names
  colnames(mean_df) <- suppressWarnings(paste0("Mu_", sort(as.numeric(levels(interaction(1:K, 1:P, sep = ""))))))
  
  # Add a variable for the iteration the sample comes from
  mean_df$Iteration <- c(1:(R / thin)) * thin
  
  # Pivot to a long format ready for ``ggplot2``
  mean_long_df <- tidyr::pivot_longer(mean_df, contains("Mu_"))
  mean_long_df$Dimension <- rep(1:P, nrow(mean_long_df) / P)
  mean_long_df$Cluster <- rep(1:K, nrow(mean_long_df) / (K * P), each = P)
  
  mean_long_df
}
