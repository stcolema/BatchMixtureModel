#!/usr/bin/Rscript
#' @title Samples to data frame
#' @description Turns the output from the mixture models into a data.frame
#' @param samples Output from the ``batchSemiSupervisedMixtureModel`` or
#' ``batchMixtureModel``.
#' @param type The type of mixture model used; this changes which parameters
#' the function expects to find.
#' @param keep_allocation A logical indicating if the final data frame should
#' include the sampled class/cluster membership variables.
#' @return A wide data.frame of all the sampled parameters and the iteration.
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
#' # MCMC samples
#' samples <- batchMixtureModel(X, R, thin, batch_vec, "MVN")
#'
#' samples_df <- samplesToDF(samples, "MVN", R = R, thin = thin)
#' @importFrom stringr str_match
#' @export
samplesToDF <- function(samples,
                        keep_allocation = TRUE) {

  R <- samples$R
  thin <- samples$thin
  
  # Number of classes and batches
  K <- samples$K_max  # ncol(samples$means[, , 1, drop = F])
  B <- samples$B      # ncol(samples$batch_shift[, , 1, drop = F])
  P <- samples$P      # nrow(samples$means[, , 1, drop = F])
  N <- samples$N      # ncol(samples$samples)

  type <- samples$type
  
  # Stack the sampled matrices on top of each other
  means_df <- data.frame(t(apply(samples$means, 3L, rbind)))
  batch_shift_df <- data.frame(t(apply(samples$batch_shift, 3L, rbind)))
  batch_scale_df <- data.frame(t(apply(samples$batch_scale, 3L, rbind)))
  mean_sums_df <- data.frame(t(apply(samples$mean_sum, 3L, rbind)))

  # Indices over columns and batches
  col_inds <- seq(1, P)
  batch_inds <- seq(1, B)
  group_inds <- seq(1, K)

  # Give sensible column names
  colnames(means_df) <- suppressWarnings(
    paste0(
      "Mu_",
      sort(as.numeric(levels(interaction(group_inds, col_inds, sep = ""))))
    )
  )

  colnames(batch_shift_df) <- suppressWarnings(
    paste0(
      "m_",
      sort(as.numeric(levels(interaction(batch_inds, col_inds, sep = ""))))
    )
  )

  colnames(batch_scale_df) <- suppressWarnings(
    paste0(
      "S_",
      sort(as.numeric(levels(interaction(batch_inds, col_inds, sep = ""))))
    )
  )

  # The combination objects are awkward to name correctly
  mean_sum_names <- suppressWarnings(levels(interaction(
    colnames(means_df),
    colnames(batch_shift_df)
  )))

  mean_sum_names <- as.data.frame(stringr::str_match(
    mean_sum_names,
    "Mu_([:digit:]*).m_([:digit:]*)"
  ))

  colnames(mean_sum_names) <- c("Comb", "Mu", "m")
  mean_sum_names$Mu <- as.numeric(mean_sum_names$Mu)
  mean_sum_names$m <- as.numeric(mean_sum_names$m)

  correct_comb <- which(mean_sum_names$Mu %% 10 == mean_sum_names$m %% 10)
  mean_sum_names <- mean_sum_names[correct_comb, ]

  inds <- matrix(seq(1, (P * B * K)), nrow = P * K, byrow = T)
  colnames(mean_sums_df) <- mean_sum_names$Comb[order(mean_sum_names$Mu)][c(inds)]

  # The covariance is slightly more awkward
  cov_df <- data.frame(t(apply(samples$covariance, 3L, rbind)))
  colnames(cov_df) <- suppressWarnings(
    paste0(
      "Sigma_",
      sort(
        as.numeric(
          levels(
            interaction(
              list(group_inds, col_inds, col_inds),
              sep = ""
            )
          )
        )
      )
    )
  )

  # The combined batch and cluster covariances
  cov_comb_df <- data.frame(t(apply(samples$cov_comb, 3L, rbind)))
  
  if(P == 2) {
    # this only effects the diagonal entries of the group covariance, so let's 
    # drop the rest
    cov_comb_df <- cov_comb_df[, seq(1, ncol(cov_comb_df)) %% 4 %in% c(0, 1)]
    
  comb_cov_names <- c(
    suppressWarnings(
      levels(
        interaction(
          colnames(cov_df),
          colnames(batch_scale_df)
        )
      )
    )[(P**2) * seq(0, ((2 * K * B) - 1)) + 1][(seq(1, (2 * K * B)) %% 4) %in% c(1, 2)],
    suppressWarnings(
      levels(
        interaction(
          colnames(cov_df),
          colnames(batch_scale_df)
        )
      )
    )[(P**2) * seq(0, ((2 * K * B) - 1)) + 4][((seq(1, (2 * K * B)) %% 4) %% 4) %in% c(0, 3)]
  )

  inds <- matrix(c(
    seq(1, (B * K)),
    seq((B * K + 1), (2 * B * K))
  ),
  nrow = 2,
  byrow = TRUE
  )
  colnames(cov_comb_df) <- comb_cov_names[c(inds)]
  }
  if(P == 1) {
    comb_cov_names <- c(
      suppressWarnings(
        levels(
          interaction(
            colnames(cov_df),
            colnames(batch_scale_df)
          )
        )
      )[(seq(1, (P * K * B)) %% 2) %in% 1],
      suppressWarnings(
        levels(
          interaction(
            colnames(cov_df),
            colnames(batch_scale_df)
          )
        )
      )[(seq(1, (P * K * B)) %% 2) %in% 0]
    )
    colnames(cov_comb_df) <- comb_cov_names
  }

  # The sampled weights
  weights_df <- as.data.frame(samples$weights)
  colnames(weights_df) <- c(paste0("pi_", group_inds))

  # Add a variable for the iteration the sample comes from
  Iteration <- seq(1, (R / thin)) * thin

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
    colnames(df_df) <- paste0("t_df_", group_inds)
    output_df <- cbind(output_df, df_df)
  }

  if (type == "MSN") {
    shape_df <- data.frame(t(apply(samples$shapes, 3L, rbind)))
    colnames(shape_df) <- suppressWarnings(
      paste0(
        "phi_",
        sort(as.numeric(levels(interaction(group_inds, col_inds, sep = ""))))
      )
    )
    output_df <- cbind(output_df, shape_df)
  }

  # Sampled observed likelihood and BIC
  output_df$Complete_likelihood <- samples$complete_likelihood
  output_df$Observed_likelihood <- samples$observed_likelihood
  output_df$BIC <- samples$BIC

  # The sampled allocations
  if (keep_allocation) {
    samples_df <- data.frame(samples$samples)
    colnames(samples_df) <- paste0("c_", seq(1, N))
    output_df <- cbind(output_df, samples_df)
  }

  output_df
}
