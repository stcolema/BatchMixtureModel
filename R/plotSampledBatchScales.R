#!/usr/bin/Rscript
#' @title Plot sampled batch scales
#' @description Plot the sampled values for the batch scale in each
#' dimension from the output of the ``mixtureModel`` function. Not recommended
#' for large B or P.
#' @param samples The output of the ``mixtureModel`` function.
#' @param R The number of iterations run. Defaults to the number of samples for
#' the cluster membership.
#' @param thin The thinning factor of the sampler. Defaults to 1.
#' @param burn_in The samples at the beginning of the chain to drop. Defaults to 0.
#' @return A ggplot object of the values in each sampled batch mean per iteration.
#' @examples
#' # Data in matrix format
#' X <- matrix(c(rnorm(100, 0, 1), rnorm(100, 3, 1)), ncol = 2, byrow = TRUE)
#' 
#' # Observed batches represented by integers
#' batch_vec <- sample(seq(1, 5), size = 100, replace = TRUE)
#' 
#' # MCMC iterations (this is too low for real use)
#' R <- 100
#' thin <- 5
#'
#' # MCMC samples and BIC vector
#' samples <- batchMixtureModel(X, R, thin, batch_vec, "MVN")
#' 
#' # Plot the sampled value of the batch scales against MCMC iteration 
#' plotSampledBatchScales(samples, R, thin)
#' @importFrom ggplot2 ggplot aes_string geom_point facet_grid labs labeller label_both
#' @export
plotSampledBatchScales <- function(samples, R = NULL, thin = 1, burn_in = 0) {
  B <- dim(samples$batch_shift)[2]
  P <- dim(samples$batch_shift)[1]

  if (is.null(R)) {
    R <- nrow(samples$samples)
  }

  # Check that the values of R and thin make sense
  if (floor(R / thin) != nrow(samples$samples)) {
    stop("The ratio of R to thin does not match the number of samples present.")
  }

  sampled_batch_scale <- getSampledBatchScale(samples$batch_scale, B, P, 
    R = R, 
    thin = thin
  )

  # Remove the warm-up samples
  sampled_batch_scale <- sampled_batch_scale[
    sampled_batch_scale$Iteration > burn_in, 
  ]

  p <- ggplot2::ggplot(sampled_batch_scale, 
      ggplot2::aes_string(x = "Iteration", y = "value")
    ) +
    ggplot2::geom_point() +
    ggplot2::facet_grid(Batch ~ Dimension,
      labeller = ggplot2::labeller(
        Batch = ggplot2::label_both,
        Dimension = ggplot2::label_both
      )
    ) +
    ggplot2::labs(
      title = "Batch scale",
      x = "MCMC iteration",
      y = "Sampled value"
    )

  p
}
