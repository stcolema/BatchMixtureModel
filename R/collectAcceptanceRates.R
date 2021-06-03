#!/usr/bin/Rscript
#' @title Collect acceptance rate
#' @description Collects the acceptance rates for each parameter into a data.frame
#' @param samples The output of either the ``batchMixtureModel`` or 
#' ``batchSemiSupervisedMixtureModel`` function.
#' @param type The type of mixture model used; this changes which parameters
#' the function expects to find.
#' @return A wide data.frame of all the sampled parameters and the iteration.
#' @export
collectAcceptanceRates <- function(samples, type) {
  
  # Number of classes and batches
  K <- ncol(samples$means[, , 1])
  B <- ncol(samples$batch_shift[, , 1])
  P <- nrow(samples$means[, , 1])
  N <- ncol(samples$samples)
  
  # Stack the sampled matrices on top of each other
  mean_rate <- t(samples$mu_acceptance_rate)
  colnames(mean_rate) <- paste0("Mu_", 1:K)
  cov_rate <- t(samples$cov_acceptance_rate)
  colnames(cov_rate) <- paste0("Sigma_", 1:K)
  
  batch_shift_rate <- t(samples$m_acceptance_rate)
  colnames(batch_shift_rate) <- paste0("m_", 1:B)
  batch_scale_rate <- t(samples$S_acceptance_rate)
  colnames(batch_scale_rate) <- paste0("S_", 1:B)
  
  output_df <- as.data.frame(
    cbind(
      mean_rate,
      cov_rate,
      batch_shift_rate,
      batch_scale_rate
    )
  )
  
  if(type =="MVT") {
    t_df_rate <- t(samples$t_df_acceptance_rate)
    colnames(t_df_rate) <- paste0("nu_", 1:K)
    output_df <- cbind(output_df, t_df_rate)
  }
  
  if(type =="MSN") {
    phi_rate <- t(samples$phi_acceptance_rate)
    colnames(phi_rate) <- paste0("phi_", 1:K)
    output_df <- cbind(output_df, phi_rate)
  }
  
  output_df
}
