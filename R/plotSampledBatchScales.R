#!/usr/bin/Rscript
#' @title Plot sampled batch scales
#' @description Plot the sampled values for the batch scale in each 
#' dimension from the output of the ``mixtureModel`` function. Not recommended
#' for large B or P.
#' @param samples The output of the ``mixtureModel`` function.
#' @param R The number of iterations run. Defaults to the number of samples for
#' the cluster membership.
#' @param thin The thinning factor of the sampler. Defaults to 1. 
#' @return A ggplot object of the values in each sampled batch mean per iteration.
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
#' plotSampledBatchScales(samples, R, thin)
#' @importFrom ggplot2 ggplot aes geom_point facet_wrap labs
#' @export
plotSampledBatchScales <- function(samples, R = NULL, thin = 1, burn_in = 0) {
  B <- dim(samples$batch_shift)[2]
  P <- dim(samples$batch_shift)[1]
  
  if (is.null(R)) {
    R <- nrow(samples$samples)
  }
  
  # Check that the values of R and thin make sense
  if(floor(R/thin) != nrow(samples$samples)){
    stop("The ratio of R to thin does not match the number of samples present.")
  }
  
  sampled_batch_scale <- getSampledBatchShift(samples$batch_scale, B, P, R = R, thin = thin)
  
  sampled_batch_scale <- sampled_batch_scale[sampled_batch_scale$Iteration > burn_in, ]
  
  p <- ggplot2::ggplot(sampled_batch_scale, ggplot2::aes(x = Iteration, y = value)) +
    ggplot2::geom_point() +
    ggplot2::facet_wrap(~name, ncol = P) +
    ggplot2::labs(
      title = "Batch scale",
      x = "MCMC iteration",
      y = "Sampled value"
    )
  
  p
}