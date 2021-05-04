#!/usr/bin/Rscript
#' @title Samples to data frame
#' @description Turns the output from the mixture models into a data.frame
#' @param samples Output from the ``batchSemiSupervisedMixtureModel`` or
#' ``batchMixtureModel``.
#' @param type The type of mixture model used; this changes which parameters
#' the function expects to find.
#' @param R The number of iterations run. Defaults to the number of slices in
#' the sampled batch mean array.
#' @param thin The thinning factor of the sampler. Defaults to 1.
#' @param keep_allocation A logical indicating if the final data frame should
#' include the sampled class/cluster membership variables.
#' @return A wide data.frame of all the sampled parameters and the iteration.
#' @examples
#' # Convert data to matrix format
#' X <- as.matrix(my_data)
#'
#' # Sampling parameters
#' R <- 1000
#' thin <- 50
#'
#' # MCMC samples
#' samples <- batchMixtureModel(X, R, thin, type = "MVN")
#'
#' samples_df <- samplesToDF(samples, "MVN", R = R, thin = thin)
#' @importFrom stringr str_match
#' @export
samplesToDF <- function(samples, type,
                        R = nrow(samples$samples),
                        thin = 1,
                        keep_allocation = TRUE) {

  # Number of classes and batches
  K <- ncol(samples$means[, , 1])
  B <- ncol(samples$batch_shift[, , 1])
  P <- nrow(samples$means[, , 1])
  N <- ncol(samples$samples)

  # Stack the sampled matrices on top of each other
  means_df <- data.frame(t(apply(samples$means, 3L, rbind)))
  batch_shift_df <- data.frame(t(apply(samples$batch_shift, 3L, rbind)))
  batch_scale_df <- data.frame(t(apply(samples$batch_scale, 3L, rbind)))
  mean_sums_df <- data.frame(t(apply(samples$mean_sum, 3L, rbind)))

  # Give sensible column names
  colnames(means_df) <- suppressWarnings(paste0("Mu_", sort(as.numeric(levels(interaction(1:K, 1:P, sep = ""))))))
  colnames(batch_shift_df) <- suppressWarnings(paste0("m_", sort(as.numeric(levels(interaction(1:B, 1:P, sep = ""))))))
  colnames(batch_scale_df) <- suppressWarnings(paste0("S_", sort(as.numeric(levels(interaction(1:B, 1:P, sep = ""))))))

  # The combination objects are awkward to name correctly
  mean_sum_names <- suppressWarnings(levels(interaction(colnames(means_df), colnames(batch_shift_df))))
  mean_sum_names <- as.data.frame(stringr::str_match(mean_sum_names, "Mu_([:digit:]*).m_([:digit:]*)"))
  colnames(mean_sum_names) <- c("Comb", "Mu", "m")
  mean_sum_names$Mu <- as.numeric(mean_sum_names$Mu)
  mean_sum_names$m <- as.numeric(mean_sum_names$m)

  correct_comb <- which(mean_sum_names$Mu %% 10 == mean_sum_names$m %% 10)
  mean_sum_names <- mean_sum_names[correct_comb, ]

  inds <- matrix(1:(P * B * K), nrow = 4, byrow = T)
  colnames(mean_sums_df) <- mean_sum_names$Comb[order(mean_sum_names$Mu)][c(inds)]

  # The covariance is slightly more awkward
  cov_df <- data.frame(t(apply(samples$covariance, 3L, rbind)))
  colnames(cov_df) <- suppressWarnings(paste0("Sigma_", sort(as.numeric(levels(interaction(list(1:K, 1:P, 1:P), sep = ""))))))

  # The combined batch and cluster covariances - this only effects the diagonal
  # entries, so let's keep only them
  cov_comb_df <- data.frame(t(apply(samples$cov_comb, 3L, rbind)))
  cov_comb_df <- cov_comb_df[, 1:ncol(cov_comb_df) %% 4 %in% c(0, 1)]

  comb_cov_names <- c(
    suppressWarnings(levels(interaction(colnames(cov_df), colnames(batch_scale_df))))[(P**2) * (0:((2 * K * B) - 1)) + 1][(1:(2 * K * B) %% 4) %in% c(1, 2)],
    suppressWarnings(levels(interaction(colnames(cov_df), colnames(batch_scale_df))))[(P**2) * (0:((2 * K * B) - 1)) + 4][((1:(2 * K * B) %% 4) %% 4) %in% c(0, 3)]
  )

  inds <- matrix(c(1:(B * K), (B * K + 1):(2 * B * K)), nrow = 2, byrow = T)

  colnames(cov_comb_df) <- comb_cov_names[c(inds)]

  # The sampled weights
  weights_df <- as.data.frame(samples$weights)
  colnames(weights_df) <- c(paste0("pi_", 1:K))

  # Add a variable for the iteration the sample comes from
  Iteration <- c(1:(R / thin)) * thin

  output_df <- as.data.frame(
    cbind(
      Iteration,
      weights_df,
      means_df,
      cov_df,
      batch_shift_df,
      batch_scale_df,
      mean_sums_df,
      cov_comb_df
    )
  )

  # Type dependent parameters
  if (type == "MVT") {
    df_df <- samples$t_df
    colnames(df_df) <- paste0("t_df_", 1:K)
    output_df <- cbind(output_df, df_df)
  }

  if (type == "MSN") {
    shape_df <- data.frame(t(apply(samples$shapes, 3L, rbind)))
    colnames(shape_df) <- suppressWarnings(paste0("phi_", sort(as.numeric(levels(interaction(1:K, 1:P, sep = ""))))))
    output_df <- cbind(output_df, shape_df)
  }
  
  # Sampled observed likelihood and BIC
  output_df$Likelihood <- samples$likelihood
  output_df$BIC <- samples$BIC
  
  # The sampled allocations
  if (keep_allocation) {
    samples_df <- data.frame(samples$samples)
    colnames(samples_df) <- paste0("c_", 1:N)
    output_df <- cbind(output_df, samples_df)
  }
  
  output_df
}
