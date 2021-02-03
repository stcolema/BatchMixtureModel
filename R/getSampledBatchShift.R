#!/usr/bin/Rscript
#' @title Get sampled batch shift
#' @description Given an array of sampled batch mean shifts from the 
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
getSampledBatchShift <- function(sampled_batch_shift, 
                                 B = dim(sampled_batch_shift)[2], 
                                 P = dim(sampled_batch_shift)[1],
                                 R = dim(sampled_batch_shift)[3],
                                 thin = 1) {
  
  # Check that the values of R and thin make sense
  if(floor(R/thin) != dim(sampled_batch_shift)[3]){
    stop("The ratio of R to thin does not match the number of samples present.")
  }
  
  # Stack the sampled matrices on top of each other
  sampled_batch_shift <- as.data.frame(t(apply(sampled_batch_shift, 3L, rbind)))
  
  # Give sensible column names
  colnames(sampled_batch_shift) <- suppressWarnings(paste0("M_", sort(levels(interaction(1:B, 1:P, sep = "")))))
  
  # Add a variable for the iteration the sample comes from
  sampled_batch_shift$Iteration <- c(1:(R / thin)) * thin

  # Pivot to a long format ready for ``ggplot2``
  long_format_sampled_batch_shift <- tidyr::pivot_longer(sampled_batch_shift, contains("M_"))
  long_format_sampled_batch_shift
}
