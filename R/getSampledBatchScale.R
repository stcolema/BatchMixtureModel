#!/usr/bin/Rscript
#' @title Get sampled batch shift
#' @description Given an array of sampled batch scales from the 
#' ``mixtureModel`` function, acquire a tidy version ready for ``ggplot2`` use.
#' @param sampled_batch_shift A 3D array of sampled batch mean shifts. 
#' @param B The number of batches present. Defaults to the number of columns in
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
#' batch_shift_df <- getSampledBatchShift(samples$batch_shift, R = R, thin = thin)
#' @importFrom tidyr pivot_longer 
#' @export
getSampledBatchScale <- function(sampled_batch_scale, 
                                 B = dim(sampled_batch_scale)[2], 
                                 P = dim(sampled_batch_scale)[1],
                                 R = dim(sampled_batch_scale)[3],
                                 thin = 1) {
  
  # Check that the values of R and thin make sense
  if(floor(R/thin) != dim(sampled_batch_scale)[3]){
    stop("The ratio of R to thin does not match the number of samples present.")
  }
  
  # Stack the sampled matrices on top of each other
  sample_df <- data.frame(t(apply(sampled_batch_scale, 3L, rbind)))
  
  # Give sensible column names
  colnames(sample_df) <- suppressWarnings(paste0("S_", sort(as.numeric(levels(interaction(1:B, 1:P, sep = ""))))))
  
  # Add a variable for the iteration the sample comes from
  sample_df$Iteration <- c(1:(R / thin)) * thin
  
  # Pivot to a long format ready for ``ggplot2``
  long_sample_df <- tidyr::pivot_longer(sample_df, contains("S_"))
  long_sample_df$Dimension <- rep(1:P, nrow(long_sample_df) / P)
  long_sample_df$Batch <- rep(1:B, nrow(long_sample_df) / (B * P), each = P)
  
  long_sample_df
}
