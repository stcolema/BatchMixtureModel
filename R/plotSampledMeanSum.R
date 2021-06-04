# 
# 
# plotSampledMeanSum <- function(samples, R = NULL, thin = 1, burn_in = 0) {
#   
#   K <- dim(samples$means)[2]
#   P <- dim(samples$means)[1]
#   B <- dim(samples$batch_shift)[2]
#   
#   if (is.null(R)) {
#     R <- nrow(samples$samples)
#   }
#   
#   # Check that the values of R and thin make sense
#   if(floor(R/thin) != nrow(samples$samples)){
#     stop("The ratio of R to thin does not match the number of samples present.")
#   }
#   
#   # Stack the sampled matrices on top of each other
#   mean_df <- data.frame(t(apply(samples$mean_sum, 3L, rbind)))
#   
#   # Suppress warnings to avoid warnings if 1:K and 1:P or 1:B and 1:P are not of
#   # matching lengths
#   mus <- suppressWarnings(paste0("Mu_", sort(levels(interaction(1:K, 1:P, sep = "")))))
#   ms <- suppressWarnings(paste0("m_", sort(levels(interaction(1:B, 1:P, sep = "")))))
#   
#   mean_sum_names <- suppressWarnings(sort(levels(interaction(mus, ms, sep = "."))))
#   
#   k_inds <- do.call(`+`, expand.grid(c(0, B * P), 1:(B * P)))
#   k_inds_wanted <- sort(c(seq(1, B*P*K, 4), seq(P * K, B*P*K, P**2)))
#   
#   used_names <- c()
#   
#   for(k in 1:K){
#     used_names <- c(used_names, mean_sum_names[k_inds][k_inds_wanted])
#     mean_sum_names <- mean_sum_names[-k_inds]
#   }
#   
#   # Give sensible column names
#   colnames(mean_df) <- used_names
#   
#   # Add a variable for the iteration the sample comes from
#   mean_df$Iteration <- c(1:(R / thin)) * thin
#   
#   # Pivot to a long format ready for ``ggplot2``
#   mean_long_df <- tidyr::pivot_longer(mean_df, contains("Mu_"))
#   
#   mean_long_df <- mean_long_df[mean_long_df$Iteration > burn_in, ]
#   
#   p <- ggplot(mean_long_df, aes(x = Iteration, y = value)) +
#     geom_point() +
#     facet_wrap(~name, ncol = P) +
#     labs(
#       title = "Cluster means",
#       x = "MCMC iteration",
#       y = "Sampled value"
#     )
#   
#   p
# }
