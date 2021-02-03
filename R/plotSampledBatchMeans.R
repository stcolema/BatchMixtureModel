#!/usr/bin/Rscript
#' @title Plot sampled batch means
#' @description Plot the sampled values for the batch mean shifts in each 
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
#' plotSampledBatchMeans(samples, R, thin)
#' @importFrom ggplot2 ggplot aes geom_point facet_wrap labs
#' @export
plotSampledBatchMeans <- function(samples, R = NULL, thin = 1) {
  B <- ncol(samples$batch_shift[, , 1])
  P <- nrow(samples$batch_shift[, , 1])
  
  if (is.null(R)) {
    R <- nrow(samples$samples)
  }
  
  # Check that the values of R and thin make sense
  if(floor(R/thin) != nrow(samples$samples)){
    stop("The ratio of R to thin does not match the number of samples present.")
  }
  
  sampled_batch_shift <- getSampledBatchShift(samples$batch_shift, B, P, R = R, thin = thin)
  
  p <- ggplot2::ggplot(sampled_batch_shift, ggplot2::aes(x = Iteration, y = value)) +
    ggplot2::geom_point() +
    ggplot2::facet_wrap(~name, ncol = P) +
    ggplot2::labs(
      title = "Batch mean shift",
      x = "MCMC iteration",
      y = "Sampled value"
    )
  
  p
}
