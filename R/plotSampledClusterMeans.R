#!/usr/bin/Rscript
#' @title Plot sampled cluster means
#' @description Plot the sampled values for the cluster means in each 
#' dimension from the output of the ``mixtureModel`` function. Not recommended 
#' for large K or P.
#' @param samples The output of the ``mixtureModel`` function.
#' @param R The number of iterations run. Defaults to the number of samples for
#' the cluster membership.
#' @param thin The thinning factor of the sampler. Defaults to 1. 
#' @return A ggplot object of the values in each sampled cluster mean per iteration.
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
#' plotSampledClusterMeans(samples, R, thin)
#' @importFrom ggplot2 ggplot aes geom_point facet_wrap labs
#' @export
plotSampledClusterMeans <- function(samples, R = NULL, thin = 1, burn_in = 0) {
  
  K <- ncol(samples$means[, , 1])
  P <- nrow(samples$means[, , 1])
  
  if (is.null(R)) {
    R <- nrow(samples$samples)
  }
  
  # Check that the values of R and thin make sense
  if(floor(R/thin) != nrow(samples$samples)){
    stop("The ratio of R to thin does not match the number of samples present.")
  }
  
  sampled_cluster_means <- getSampledClusterMeans(samples$means, K, P, R = R, thin = thin)
  
  sampled_cluster_means <- sampled_cluster_means[sampled_cluster_means$Iteration > burn_in, ]
                                             
  p <- ggplot(sampled_cluster_means, aes(x = Iteration, y = value)) +
    geom_point() +
    facet_wrap(~name, ncol = P) +
    labs(
      title = "Cluster means",
      x = "MCMC iteration",
      y = "Sampled value"
    )
  
  p
}
