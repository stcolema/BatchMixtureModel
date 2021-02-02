#!/usr/bin/Rscript
#
# Example Metropolis-Hastings algorithm
#
# Define some transition kernel to propose new values
# Calculate the likelihood
# Compare to old likelihood
# Sample u from Unif(0,1)
# Accept new value with probability u

library(mdiHelpR)
library(magrittr)
library(pheatmap)
library(mcclust)
library(mclust)
# library(BatchMixtureModel)
library(ggplot2)
library(tidyr)
library(dplyr)
library(tibble)
library(MASS)
library(ggridges)
library(ggExtra)


setMyTheme()
set.seed(1)

# Rcpp::sourceCpp('Model ideas/model_metropolis_hastings.cpp')

# transition_kernel <- function(curr_theta, proposed_width = 0.5) {
#   n_param <- length(curr_theta)
#   new_theta <- rep(0, n_param)
#   for (i in 1:n_param) {
#     new_theta[i] <- rnorm(1, curr_theta[i], sd = proposed_width)
#   }
#   new_theta
# }
#
# transition_std_dev <- function(curr_sd, proposed_width = 0.5) {
#   n_param <- length(curr_sd)
#   new_sd <- rep(0, n_param)
#   for (i in 1:n_param) {
#     new_sd[i] <- max(0.1, rnorm(1, curr_sd[i], sd = proposed_width))
#   }
#   new_sd
# }
#
# normal_ll <- function(x, mean = 0, sd = 1) {
#   -0.5 * (2 * pi * sd**2 + (x - mean)**2 / sd**2)
# }
#
# my_kernel <- function(X, c, b, mu, sigma, K, a, s, B, N_k, N_b,
#                       mu0 = 0,
#                       gamma = 1,
#                       xi = 1,
#                       lambda = 1,
#                       delta = 1,
#                       epsilon = 1,
#                       theta = 1) {
#   N <- length(X)
#
#   part_1 <- sum((N_k + gamma - 0.5) * log(sigma)) +
#     sum((N_b + gamma - 0.5) * log(s))
#
#   exp1 <- 0
#   for (n in 1:N) {
#     c_n <- c[n]
#     b_n <- b[n]
#     exp1 <- exp1 + 1 / (sigma[c_n] * s[b_n]) * (X[n] - (mu[c_n] + a[b_n]))**2
#   }
#
#   exp2 <- lambda * sum(((mu - mu0)**2 + 2 * epsilon) / sigma)
#   exp3 <- delta * sum(((a - xi)**2 + 2 * theta) / s)
#
#   part_2 <- -0.5 * (exp1 + exp2 + exp3)
#
#   score <- part_1 + part_2
#
#   score
# }
#
#
# # my_sampler <- function(X, R, thin, K, c_init, b){
# #
# #   N <- length(X)
# #   # P <- ncol(X)
# #   B <- length(unique(b))
# #
# #   mu0 <- mean(X)
# #   lambda <- 0.2
# #   gamma <- 20
# #   epsilon <- 4
# #   xi <- 0
# #   delta <- 1
# #   rho <- 8
# #   theta <- 4
# #
# #   std_dev <- rgamma(K, gamma, epsilon)
# #   mu <- rep(0, K)
# #   for(k in 1:K){
# #     mu[k] <- rnorm(1, mu0, (1/lambda) * std_dev[k])
# #   }
# #   s <- rgamma(B, rho, theta)
# #
# #   a <- rep(0, K)
# #   for(i in 1:B){
# #     a[i] <- rnorm(1, xi, (1/delta) * s[i])
# #   }
# #
# #   N_k <- table(c_init)
# #   N_b <- table(b)
# #   c <- c_init
# #
# #   curr_score <- my_kernel(X, c, b, mu, std_dev, K, a, s, B, N_k, N_b,
# #             mu0 = mu0,
# #             gamma = gamma,
# #             xi = xi,
# #             lambda = lambda,
# #             delta = delta,
# #             epsilon = epsilon,
# #             theta = theta)
# #
# #   mu_saved <- vector("list", R)
# #   a_saved <- vector("list", R)
# #   std_dev_saved <- vector("list", R)
# #   s_saved <- vector("list", R)
# #   c_saved <- vector("list", R)
# #   weights_saved <- vector("list", R)
# #   accepted <- rep(0, R)
# #   acceptance_probs <- rep(0, R)
# #
# #   for(r in 1:R){
# #     new_mu <- transition_kernel(mu)
# #     new_a <- transition_kernel(a)
# #     new_std_dev <- transition_std_dev(std_dev)
# #     new_s <- transition_std_dev(s)
# #
# #     # detailed_balance_part <- normal_ll(mu, mean = new_mu, sd = 0.5) +
# #     #   normal_ll(a, mean = new_a, sd = 0.5) +
# #     #   normal_ll(std_dev, mean = new_std_dev, sd = 0.5) +
# #     #   normal_ll(s, mean = new_s, sd = 0.5) -
# #     #   (normal_ll(new_mu, mean = mu, sd = 0.5) +
# #     #      normal_ll(new_a, mean = a, sd = 0.5) +
# #     #      normal_ll(new_std_dev, mean = std_dev, sd = 0.5) +
# #     #      normal_ll(new_s, mean = s, sd = 0.5))
# #
# #     print(new_mu)
# #     print(new_std_dev)
# #     print(new_a)
# #     print(new_s)
# #
# #     new_score <- my_kernel(X, c, b, new_mu, new_std_dev, K, new_a, new_s, B, N_k, N_b,
# #        mu0 = mu0,
# #        gamma = gamma,
# #        xi = xi,
# #        lambda = lambda,
# #        delta = delta,
# #        epsilon = epsilon,
# #        theta = theta
# #     )
# #
# #     # detailed_balance_part <- 0
# #     # for(k in 1:K){
# #     #   detailed_balance_part <- detailed_balance_part +
# #     #     normal_ll(mu[k], mean = new_mu[k], sd = 0.5) +
# #     #     normal_ll(std_dev[k], mean = new_std_dev[k], sd = 0.5) -
# #     #     normal_ll(new_mu[k], mean = mu[k], sd = 0.5) -
# #     #     normal_ll(new_std_dev[k], mean = std_dev[k], sd = 0.5)
# #     # }
# #     #
# #     # for(b in 1:B){
# #     #   detailed_balance_part <- detailed_balance_part +
# #     #     normal_ll(a[b], mean = new_a[b], sd = 0.5) +
# #     #     normal_ll(s[b], mean = new_s[b], sd = 0.5) -
# #     #     (normal_ll(new_a[b], mean = a[b], sd = 0.5) +
# #     #        normal_ll(new_s[b], mean = s[b], sd = 0.5))
# #     # }
# #     #
# #     # print(new_score)
# #     # print(curr_score)
# #     # print(detailed_balance_part)
# #
# #     acceptance_score <- new_score - curr_score #+ detailed_balance_part
# #     acceptance_prob <- exp(acceptance_score)
# #
# #     print(acceptance_prob)
# #
# #     u <- runif(1, 0, 1)
# #     if(u < acceptance_prob){
# #       mu <- new_mu
# #       a <- new_a
# #       std_dev <- new_std_dev
# #       s  <- new_s
# #       accepted[r] <- 1
# #     }
# #
# #     weights <- N_k / sum(N_k)
# #     N_k <- rep(0, K)
# #     for(n in 1:N){
# #       x <- X[n]
# #       b_n <- b[n]
# #       probs <- rep(0,K)
# #       for(k in 1:K){
# #         probs[k] <- weights[k] * (dnorm(x, mean = mu[k] + a[b_n], sd = std_dev[k] * s[b_n]) + 1e-12)
# #       }
# #
# #       probs <- probs/sum(probs)
# #
# #       u <- runif(1, 0, 1)
# #       c[n] <- c_n <- 1 + sum(u > cumsum(probs))
# #       N_k[c_n] <- N_k[c_n] + 1
# #
# #     }
# #
# #     mu_saved[[r]] <- mu
# #     a_saved[[r]] <- a
# #     std_dev_saved[[r]] <- std_dev
# #     s_saved[[r]] <- s
# #     c_saved[[r]] <- c
# #     weights_saved[[r]] <- weights
# #     acceptance_probs[r] <- acceptance_prob
# #   }
# #
# #   list(
# #     "mu" = mu_saved,
# #     "a" = a_saved,
# #     "std_dev" = std_dev_saved,
# #     "s" = s_saved,
# #     "c" = c_saved,
# #     "weights" = weights_saved,
# #     "accepted" = accepted,
# #     "acceptance_probs" = acceptance_probs
# #   )
# #
# # }
#
#
# generateBatchData <- function(N, P, cluster_means, std_dev, batch_shift, batch_var, cluster_weights, batch_weights,
#                               row_names = paste0("Person_", 1:n),
#                               col_names = paste0("Gene_", 1:p)) {
#
#   # The number of distirbutions to sample from
#   K <- length(cluster_means)
#
#   B <- length(batch_shift)
#
#   # The membership vector for the n points
#   cluster_IDs <- sample(K, N, replace = T, prob = cluster_weights)
#
#   batch_IDs <- sample(1:B, N, replace = T, prob = batch_weights)
#
#   # The data matrix
#   my_data <- my_corrected_data <- matrix(nrow = N, ncol = P)
#
#   # Iterate over each of the columns permuting the means associated with each
#   # label.
#   for (p in 1:P)
#   {
#     reordered_cluster_means <- sample(cluster_means)
#     reordered_std_devs <- sample(std_dev)
#
#     reordered_batch_shift <- sample(batch_shift)
#     reordered_batch_var <- sample(batch_var)
#
#     # Draw n points from the K univariate Gaussians defined by the permuted means.
#     for (n in 1:N) {
#
#       # point_mean <- reordered_cluster_means[cluster_IDs[n]] + batch_shift[batch_IDs[n]]
#       # point_sd <- std_dev * batch_var
#
#       x <- rnorm(1)
#
#       my_corrected_data[n, p] <- x * reordered_std_devs[cluster_IDs[n]] + reordered_cluster_means[cluster_IDs[n]]
#       my_data[n, p] <- x * reordered_std_devs[cluster_IDs[n]] * reordered_batch_var[batch_IDs[n]] + reordered_cluster_means[cluster_IDs[n]] + reordered_batch_shift[batch_IDs[n]]
#     }
#   }
#
#   # Order based upon allocation label
#   row_order <- order(cluster_IDs)
#
#   # Assign rownames and column names
#   rownames(my_data) <- row_names
#   colnames(my_data) <- col_names
#
#   rownames(my_corrected_data) <- row_names
#   colnames(my_corrected_data) <- col_names
#
#   # Return the data and the allocation labels
#   list(
#     data = my_data[row_order, ],
#     corrected_data = my_corrected_data[row_order, ],
#     cluster_IDs = cluster_IDs[row_order],
#     batch_IDs = batch_IDs[row_order]
#   )
# }

# === Data generation ==========================================================
set.seed(1)

N <- 400
P <- 2
K <- 3
B <- 5
mean_dist <- 7
batch_dist <- 1
cluster_means <- 1:K * mean_dist
batch_shift <- rnorm(B) #   1:B * batch_dist
std_dev <- rep(2, K)
batch_var <- rep(1, B) #  rgamma(n = B, shape = 1, scale = 1)
cluster_weights <- rep(1 / K, K)
batch_weights <- rep(1 / B, B)

my_data <- generateBatchData(
  N,
  P,
  cluster_means,
  std_dev,
  batch_shift,
  batch_var,
  cluster_weights,
  batch_weights
)

# Separate out the data
X <- my_data$data
X_true <- my_data$corrected_data
b <- my_data$batch_IDs

my_df <- data.frame(
  x = X[, 1],
  y = X[, 2],
  x_true = X_true[, 1],
  y_true = X_true[, 2],
  cluster = factor(my_data$cluster_IDs),
  batch = factor(my_data$batch_IDs)
)

# Look at the data with and without batch effects
my_df %>%
  ggplot(aes(x = x, y = y, shape = cluster, colour = batch)) +
  geom_point() +
  labs(
    title = "Generated data",
    subtitle = "Batch effects included"
  )

# ggsave("../Model ideas/gen_data.png")

my_df %>%
  ggplot(aes(x = x_true, y = y_true, shape = cluster, colour = batch)) +
  geom_point() +
  labs(
    title = "Generated data",
    subtitle = "Batch effects removed"
  )

my_df %>%
  ggplot(aes(x = x, fill = cluster)) +
  geom_histogram() +
  labs(
    title = "Generated data",
    subtitle = "Batch effects present"
  )

my_df %>%
  ggplot(aes(x = x_true, fill = cluster)) +
  geom_histogram() +
  labs(
    title = "Generated data",
    subtitle = "Batch effects removed"
  )

# === Priors ===================================================================

# Hyperparameters
xi <- colMeans(X)
kappa <- 0.01
psi <- cov(X) / (K**(2 / P))
psi_inv <- solve(psi)
nu <- P + 2
delta <- 0
lambda <- 1.0
rho <- 11.0
theta <- 10.0

# Number of draws
n_draw <- 10000

# Sample covariance matrices
sigmas <- rWishart(n_draw, nu, psi_inv) %>%
  apply(3, solve) %>%
  `*`(1 / kappa) %>%
  as.data.frame() %>%
  lapply(matrix, nrow = P, ncol = P)

# Sample cluster means
mus <- sigmas %>%
  lapply(function(x) {
    mvrnorm(1, xi, x)
  }) %>%
  unlist() %>%
  matrix(ncol = 2, byrow = T) %>%
  as_tibble() %>%
  set_colnames(c("x", "y"))

# Visualise
mus %>%
  ggplot(aes(x = x, y = y)) +
  geom_point(alpha = 0.2) +
  geom_density_2d() +
  labs(
    title = "Prior distribution of cluster means",
    subtitle = paste0(n_draw, " draws")
  )

ggsave("../prior_mu_k.png")

# Batch variables
S_samples <- rgamma(n_draw, rho, theta)

m_samples <- S_samples %>%
  lapply(function(x) {
    rnorm(1, delta, x)
  }) %>%
  unlist() %>%
  as_tibble() %>%
  cbind(S_samples) %>%
  set_colnames(c("m", "S"))


p1 <- m %>%
  ggplot(aes(x = m, y = S)) +
  geom_point() +
  labs(
    title = "Prior distribution of batch parameters",
    subtitle = paste0(n_draw, " draws")
  )

# Marginal histograms
ggMarginal(p1, type = "histogram")

# === Proposal densities =======================================================

mean_proposal_window <- 0.5**2
cov_proposal_window <- 100
S_proposal_window <- 100

n_draw <- 10000

mu_k <- c(0, 1, 2, 3, 4, 5)
mean_samples <- mu_k %>%
  lapply(function(x) {
    rnorm(n_draw, x, mean_proposal_window)
  }) %>%
  list2DF() %>%
  set_colnames(mu_k) %>%
  as_tibble() %>%
  pivot_longer(everything(), names_to = "Mu", values_to = "Value")

mean_samples %>%
  ggplot(aes(x = Value, y = Mu)) +
  geom_density_ridges(alpha = 0.4)

s_b <- 10**seq(-3, 2)
S_samples <- s_b %>%
  lapply(function(x) {
    rgamma(n_draw, x * S_proposal_window, rate = S_proposal_window)
  }) %>%
  list2DF() %>%
  set_colnames(paste0(s_b)) %>%
  as_tibble() %>%
  pivot_longer(everything(), names_to = "S", values_to = "Value")


S_samples %>%
  ggplot(aes(x = Value, y = S)) +
  geom_density_ridges(alpha = 0.4) +
  xlim(c(-1.0, 15))

S_samples %>%
  ggplot(aes(x = Value, y = ..count..)) +
  geom_histogram(alpha = 0.4, bins = 100) +
  xlim(c(-1.0, 3)) +
  facet_wrap(~S)

S_samples_2 <- s_b %>%
  lapply(function(x) {
    exp(rnorm(n_draw, mean = x, sd = S_proposal_window))
  }) %>%
  list2DF() %>%
  set_colnames(paste0(s_b)) %>%
  as_tibble() %>%
  pivot_longer(everything(), names_to = "S", values_to = "Value")

S_samples_2 %>%
  ggplot(aes(x = Value, y = S)) +
  geom_density_ridges(alpha = 0.4) +
  xlim(c(-1.0, 5))

S_samples_2 %>%
  ggplot(aes(x = Value)) +
  geom_histogram(alpha = 0.4, bins = 100) +
  xlim(c(-1.0, 15)) +
  facet_wrap(~S)

# === Clustering ===============================================================

# X <- matrix(c(1, 1, 1, 1, -1, -1, -1, -1), ncol = 2, byrow = T)
# N <- nrow(X)
# c_init <- sample(1:K, N,replace = T)
R <- 10000
thin <- 100

K_max <- K
# Some random initialisation
c_init <- sample(1:K_max, N, replace = T)

samples <- sampleMixtureModel(
  matrix(X[,1], ncol = 1),
  # X_true,
  # X,
  # K,
  K_max,
  # 1,
  B,
  # my_data$cluster_IDs - 1,
  c_init - 1,
  # rep(0, N), #
  my_data$batch_IDs - 1,
  mean_proposal_window,
  cov_proposal_window,
  S_proposal_window,
  R,
  thin,
  rep(1, K_max),
  1
)

# samples
hist(samples$weights[, 1])

psm <- createSimilarityMat(samples$samples) %>%
  set_rownames(row.names(my_data$data)) %>%
  set_colnames(row.names(my_data$data))

annotatedHeatmap(psm, my_data$cluster_IDs,
  col_pal = simColPal(),
  main = "PSM",
  breaks = mdiHelpR::defineBreaks(simColPal(), lb = 0)
)

# annotatedHeatmap(psm, c_init, col_pal = simColPal(), main = "PSM")
annotatedHeatmap(psm, my_data$batch_IDs, col_pal = simColPal())

annotatedHeatmap(my_data$data, my_data$cluster_IDs)

pred_cl <- mcclust::maxpear(psm)
annotatedHeatmap(my_data$data, pred_cl$cl)
annotatedHeatmap(my_data$corrected_data, pred_cl$cl)

mcclust::arandi(pred_cl$cl, my_data$cluster_IDs)

sampled_batch_shift <- (apply(samples$batch_shift, 3L, rbind)) %>%
  t() %>%
  as_tibble() %>%
  set_colnames(paste0("M_", sort(levels(interaction(1:B, 1:P, sep = ""))))) %>%
  add_column(Iteration = c(1:(R / thin)) * thin)

sampled_batch_shift %>%
  pivot_longer(contains("M_")) %>%
  ggplot(aes(x = Iteration, y = value)) +
  geom_point() +
  geom_hline(yintercept = c(-1 * batch_shift, batch_shift), lty = 2, colour = "orange") +
  geom_hline(yintercept = 0.0, lty = 3, colour = "navy") +
  facet_wrap(~name, ncol = P) +
  # geom_smooth() +
  labs(
    title = "Batch mean shift",
    subtitle = "Batch covariance fixed",
    x = "MCMC iteration",
    y = "Sampled value"
  )
# ggsave("./Sampled_batch_mean_batch_cov_fixed.png")

sampled_cluster_mean <- (apply(samples$means, 3L, rbind)) %>%
  t() %>%
  as_tibble() %>%
  set_colnames(paste0("Mu_", sort(levels(interaction(1:K_max, 1:P, sep = ""))))) %>%
  add_column(Iteration = c(1:(R / thin)) * thin)

sampled_cluster_mean %>%
  pivot_longer(contains("Mu_")) %>%
  ggplot(aes(x = Iteration, y = value)) +
  geom_point() +
  # geom_hline(yintercept = c(-1*cluster_means, cluster_means), lty = 2, colour = "orange") +
  geom_hline(yintercept = 0.0, lty = 3, colour = "navy") +
  facet_wrap(~name, ncol = P) +
  # geom_smooth()  +
  # coord_cartesian(
  #   xlim = NULL,
  #   ylim = c(-30, 30)
  # ) +

  labs(
    title = "Cluster mean",
    subtitle = "Batch covariance fixed",
    x = "MCMC iteration",
    y = "Sampled value"
  )

# ggsave("./Sampled_cluster_mean_batch_cov_fixed.png")


plot(t(samples$means[, 1, ]))
plot(t(samples$means[, 2, ]))

plot(t(samples$batch_shift[, 1, ]))

sample_df %>%
  tidyr::pivot_longer(dplyr::contains("Mu_")) %>%
  ggplot(aes(x = Iteration, y = value)) +
  geom_point() +
  facet_wrap(~name) +
  geom_smooth()

sample_df %>%
  tidyr::pivot_longer(dplyr::contains("m_")) %>%
  ggplot(aes(x = Iteration, y = value)) +
  geom_point() +
  facet_wrap(~name) +
  geom_smooth()

sample_df %>%
  tidyr::pivot_longer(dplyr::contains("Mu_")) %>%
  ggplot(aes(x = value, fill = name)) +
  geom_histogram(alpha = 0.3) +
  geom_vline(xintercept = c(2, -2))



# samples$acceptance %>%
#   max()
#
samples$means[, 1, ]
hist(samples$means[, 1, -c(1:100)])

samples$means[, 2, ]
hist(samples$means[, 2, -c(1:100)])

hist(samples$batch_shift[, 1, -c(1:100)])
hist(samples$batch_shift[, 2, -c(1:100)])


samples$precisions[, 1, ] %>% unique()
hist(samples$precisions[, 1, ])
samples$precisions[, 2, ] %>% unique()
hist(samples$precisions[, 2, ])

plot(t(samples$means[, 1, ]))
plot(t(samples$means[, 2, ]))

plot(t(samples$precisions[, 1, ]))
plot(t(samples$precisions[, 2, ]))


pred_cl <- maxpear(psm, max.k = 5)$cl

plot(samples$acceptance, type = "l")

means <- samples$mu %>%
  unlist() %>%
  matrix(ncol = K, byrow = T)

std_devs <- samples$std_dev %>%
  unlist() %>%
  matrix(ncol = K, byrow = T)

plot(means)
plot(std_devs)

mcl1 <- Mclust(X, G = 1:10)
mcl1$G

my_data$cluster_IDs
mcl1$classification

labels <- data.frame(
  truth = my_data$cluster_IDs,
  batch = my_data$batch_IDs,
  data = my_data$data,
  corrected_data = my_data$corrected_data,
  mclust = mcl1$classification
)

pheatmap(labels, cluster_rows = F)
