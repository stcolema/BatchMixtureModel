// # include <RcppArmadillo.h>
// # include <math.h> 
// # include <string>
// # include <iostream>
// 
// // [[Rcpp::depends(RcppArmadillo)]]
// 
// using namespace Rcpp ;
// 
// // double logGamma(double x){
// //   if( x == 1 ) {
// //     return 0;
// //   } 
// //   return (std::log(x - 1) + logGamma(x - 1));
// // }
// 
// //' @title Calculate sample covariance
// //' @description Returns the unnormalised sample covariance. Required as 
// //' arma::cov() does not work for singletons.
// //' @param data Data in matrix format
// //' @param sample_mean Sample mean for data
// //' @param n The number of samples in data
// //' @param n_col The number of columns in data
// //' @return One of the parameters required to calculate the posterior of the
// //'  Multivariate normal with uknown mean and covariance (the unnormalised 
// //'  sample covariance).
// arma::mat calcSampleCov(arma::mat data,
//                         arma::vec sample_mean,
//                         arma::uword N,
//                         arma::uword P
// ) {
//   
//   arma::mat sample_covariance = arma::zeros<arma::mat>(P, P);
//   
//   // If n > 0 (as this would crash for empty clusters), and for n = 1 the 
//   // sample covariance is 0
//   if(N > 1){
//     data.each_row() -= sample_mean.t();
//     sample_covariance = data.t() * data;
//   }
//   return sample_covariance;
// };
// 
// class paulSampler {
//   
// private:
//   
// public:
//   
//   arma::uword K, B, N, P, K_occ, accepted = 0;
//   double model_likelihood = 0.0, BIC = 0.0, model_score = 0.0;
//   arma::uvec labels, N_k, batch_vec, N_b, KB_inds, B_inds;
//   arma::vec concentration, w, ll, likelihood;
//   arma::umat members;
//   arma::mat X, X_t, Y, Y_t, alloc;
//   arma::field<arma::uvec> batch_ind; //, cluster_ind;
// ;
//   
//   // Parametrised class
//   paulSampler(
//     arma::uword _K,
//     arma::uword _B,
//     arma::uvec _labels,
//     arma::uvec _batch_vec,
//     arma::vec _concentration,
//     arma::mat _X)
//   {
//     
//     K = _K;
//     B = _B;
//     labels = _labels;
//     batch_vec = _batch_vec;
//     concentration = _concentration;
// 
//     // Observed data
//     X = _X;
//     X_t = X.t();
//     
//     // Latent, batch corrected data. Initialise to observed data.
//     Y = X;
//     Y_t = X_t;
//     
//     // // Plausibly belongs in the MVN sampler. Used for selecting slices / columns 
//     // // in the metropolis steps.
//     // KB_inds = arma::linspace<arma::uvec>(0, K - 1, K) * B;
//     // B_inds = arma::linspace<arma::uvec>(0, B - 1, B);
//     // 
//     
//     // Dimensions
//     N = X.n_rows;
//     P = X.n_cols;
//     
//     // std::cout << "\nN: " << N << "\nP: " << P << "\n\n";
//     
//     // Class populations
//     N_k = arma::zeros<arma::uvec>(K);
//     N_b = arma::zeros<arma::uvec>(B);
//     
//     // The batch numbers won't ever change, so let's count them now
//     for(arma::uword b = 0; b < B; b++){
//       N_b(b) = arma::sum(batch_vec == b);
//     }
//     
//     // Weights
//     // double x, y;
//     w = arma::zeros<arma::vec>(K);
//     
//     // Log likelihood (individual and model)
//     ll = arma::zeros<arma::vec>(K);
//     likelihood = arma::zeros<arma::vec>(N);
//     
//     // Class members
//     members.set_size(N, K);
//     members.zeros();
//     
//     // Allocation probability matrix (only makes sense in predictive models)
//     alloc.set_size(N, K);
//     alloc.zeros();
//     
//     // Latent true data
//     // Y.set_size(N, P);
//     // Y.zeros();
//     // 
//     // // And the transpose
//     // Y_t.set_size(N, P);
//     // Y_t.zeros();
//     
//     // The indices of the members of each batch in the dataset
//     batch_ind.set_size(B);
//     for(arma::uword b = 0; b < B; b++) {
//       batch_ind(b) = arma::find(batch_vec == b);
//     }
//     
//     // // The indices of the members of each batch in the dataset
//     // cluster_ind.set_size(K);
//     // for(arma::uword k = 0; k < K; k++){
//     //   cluster_ind(k) = arma::find(labels == k);
//     // }
//     
//   };
//   
//   // Destructor
//   virtual ~paulSampler() { };
//   
//   // Virtual functions are those that should actual point to the sub-class
//   // version of the function.
//   // Print the sampler type.
//   virtual void printType() {
//     std::cout << "\nType: NULL.\n";
//   };
//   
//   // Functions required of all mixture models
//   virtual void updateWeights(){
//     
//     double a = 0.0;
//     
//     for (arma::uword k = 0; k < K; k++) {
//       
//       // Find how many labels have the value
//       members.col(k) = labels == k;
//       N_k(k) = arma::sum(members.col(k));
//       
//       // Update weights by sampling from a Gamma distribution
//       a  = concentration(k) + N_k(k);
//       w(k) = arma::randg( arma::distr_param(a, 1.0) );
//     }
//     
//     // Convert the cluster weights (previously gamma distributed) to Beta
//     // distributed by normalising
//     w = w / arma::sum(w);
//     
//   };
//   
//   virtual void updateAllocation() {
//     
//     double u = 0.0;
//     arma::uvec uniqueK;
//     arma::vec comp_prob(K);
//     
//     for(arma::uword n = 0; n < N; n++){
//       
//       ll = itemLogLikelihood(Y.row(n).t());
//       
//       // std::cout << "\n\nAllocation log likelihood: " << ll;
//       // Update with weights
//       comp_prob = ll + log(w);
//       
//       likelihood(n) = arma::accu(comp_prob);
//       
//       // std::cout << "\n\nWeights: " << w;
//       // std::cout << "\n\nAllocation log probability: " << comp_prob;
//       
//       // Normalise and overflow
//       comp_prob = exp(comp_prob - max(comp_prob));
//       comp_prob = comp_prob / sum(comp_prob);
//       
//       // Prediction and update
//       u = arma::randu<double>( );
//       
//       labels(n) = sum(u > cumsum(comp_prob));
//       alloc.row(n) = comp_prob.t();
//       
//       // Record the likelihood of the item in it's allocated component
//       // likelihood(n) = ll(labels(n));
//     }
//     
//     // The model log likelihood
//     model_likelihood = arma::accu(likelihood);
//     
//     // Number of occupied components (used in BIC calculation)
//     uniqueK = arma::unique(labels);
//     K_occ = uniqueK.n_elem;
//   };
//   
//   // The virtual functions that will be defined in any subclasses
//   virtual void metropolisStep(){};
//   virtual void sampleFromPriors() {};
//   virtual void sampleParameters(){};
//   virtual void calcBIC(){};
//   virtual arma::vec itemLogLikelihood(arma::vec x) { return arma::vec(); };
//   
// };
// 
// 
// 
// 
// //' @name mvnSampler
// //' @title Multivariate Normal mixture type
// //' @description The sampler for the Multivariate Normal mixture model for batch effects.
// //' @field new Constructor \itemize{
// //' \item Parameter: K - the number of components to model
// //' \item Parameter: B - the number of batches present
// //' \item Parameter: labels - the initial clustering of the data
// //' \item Parameter: concentration - the vector for the prior concentration of 
// //' the Dirichlet distribution of the component weights
// //' \item Parameter: X - the data to model
// //' }
// //' @field printType Print the sampler type called.
// //' @field updateWeights Update the weights of each component based on current 
// //' clustering.
// //' @field updateAllocation Sample a new clustering. 
// //' @field sampleFromPrior Sample from the priors for the multivariate normal
// //' density.
// //' @field calcBIC Calculate the BIC of the model.
// //' @field logLikelihood Calculate the likelihood of a given data point in each
// //' component. \itemize{
// //' \item Parameter: point - a data point.
// //' }
// class paulMVNSampler: virtual public paulSampler {
//   
// public:
//   
//   arma::uword n_param_cluster = 1 + P + P * (P + 1) * 0.5, n_param_batch = 2 * P;
//   double kappa, nu, lambda, rho, theta, y_proposal_window;
//   arma::uvec y_count;
//   arma::vec xi, delta, cov_log_det;
//   arma::mat scale, mu, m, S;
//   arma::cube cov, cov_inv;
//   
//   using paulSampler::paulSampler;
//   
//   paulMVNSampler(                           
//     arma::uword _K,
//     arma::uword _B,
//     double _y_proposal_window,
//     double _rho,
//     double _theta,
//     double _lambda,
//     arma::uvec _labels,
//     arma::uvec _batch_vec,
//     arma::vec _concentration,
//     arma::mat _X
//   ) : paulSampler(_K,
//   _B,
//   _labels,
//   _batch_vec,
//   _concentration,
//   _X) {
//     
//     arma::rowvec X_min = arma::min(X), X_max = arma::max(X);
// 
//     // Default values for hyperparameters
//     // Cluster hyperparameters for the Normal-inverse Wishart
//     // Prior shrinkage
//     kappa = 0.01;
//     // Degrees of freedom
//     nu = P + 2;
//     
//     // Mean
//     arma::mat mean_mat = arma::mean(_X, 0).t();
//     xi = mean_mat.col(0);
//     
//     // Empirical Bayes for a diagonal covariance matrix
//     arma::mat scale_param = _X.each_row() - xi.t();
//     arma::rowvec diag_entries = arma::sum(scale_param % scale_param, 0) / (N * std::pow(K, 1.0 / (double) P));
//     scale = arma::diagmat( diag_entries );
//     
//     // The mean of the prior distribution for the batch shift, m, parameter
//     delta = arma::zeros<arma::vec>(P);
//     lambda = _lambda; // 1.0;
//     
//     // The shape and scale of the prior for the batch scale, S
//     rho = _rho; // 41.0; // 3.0 / 2.0;
//     theta = _theta; // 40.0; // arma::stddev(X.as_col()) / std::pow(B, 2.0 / B ); // 2.0;
//     
//     // Set the size of the objects to hold the component specific parameters
//     mu.set_size(P, K);
//     mu.zeros();
//     
//     cov.set_size(P, P, K);
//     cov.zeros();
//     
//     // Set the size of the objects to hold the batch specific parameters
//     m.set_size(P, B);
//     m.zeros();
//     
//     // We are assuming a diagonal structure in the batch scale
//     S.set_size(P, B);
//     S.zeros();
//     
//     // Count the number of times proposed values are accepted
//     y_count = arma::zeros<arma::uvec>(N);
//     
//     // These will hold vertain matrix operations to avoid computational burden
//     // The log determinant of each cluster covariance
//     cov_log_det = arma::zeros<arma::vec>(K);
//     
//     // Inverse of the cluster covariance
//     cov_inv.set_size(P, P, K);
//     cov_inv.zeros();
//     
//     // // The log determinant of the covariance combination
//     // cov_comb_log_det.set_size(K, B);
//     // cov_comb_log_det.zeros();
//     // 
//     // // The possible combinations for the sum of the cluster and batch means
//     // mean_sum.set_size(P, K * B);
//     // mean_sum.zeros();
//     // 
//     // // The combination of each possible cluster and batch covariance
//     // cov_comb.set_size(P, P, K * B);
//     // cov_comb.zeros();
//     // 
//     // 
//     // // The inverse of the covariance combination
//     // cov_comb_inv.set_size(P, P, K * B);
//     // cov_comb_inv.zeros();
//     
//     // The proposal windows for the cluster and batch parameters
//     y_proposal_window = _y_proposal_window;
//   };
//   
//   
//   // Destructor
//   virtual ~paulMVNSampler() { };
//   
//   // Print the sampler type.
//   virtual void printType() {
//     std::cout << "\nType: Paul's MVN.\n";
//   }
//   
//   virtual void sampleFromPriors() {
//     
//     for(arma::uword k = 0; k < K; k++){
//       cov.slice(k) = arma::iwishrnd(scale, nu);
//       mu.col(k) = arma::mvnrnd(xi, (1.0/kappa) * cov.slice(k), 1);
//     }
//     for(arma::uword b = 0; b < B; b++){
//       for(arma::uword p = 0; p < P; p++){
//         
//         // Fix the 0th batch at no effect; all other batches have an effect
//         // relative to this
//         if(b == 0){
//         //   S(p, b) = 1.0;
//           m(p, b) = 0.0;
//         } else {
//          S(p, b) = 1.0 / arma::randg<double>( arma::distr_param(0.5 * rho, 1.0 / theta ) );
//         // S(p, b) = 1.0;
//           m(p, b) = arma::randn<double>() * S(p, b) / lambda + delta(p);
//         // m(p, b) = arma::randn<double>() / lambda + delta(p);
//         }
//       }
//     }
//     // for(arma::uword n = 0; n < N; n++) {
//     //   
//     //   Y.row(n) = arma::mvnrnd( mu.col(labels(n)), cov.slice(labels(n)) ).t();
//     //   
//     // }
//     
//     // std::cout << "\n\nPrior covariance:\n" << cov << "\n\nPrior mean:\n" << mu << "\n\nPrior S:\n" << S << "\n\nPrior m:\n" << m;
//   };
//   
//   // Update the common matrix manipulations to avoid recalculating N times
//   virtual void matrixCombinations() {
//     
//     for(arma::uword k = 0; k < K; k++) {
//       cov_inv.slice(k) = arma::inv_sympd(cov.slice(k));
//       cov_log_det(k) = arma::log_det(cov.slice(k)).real();
//     }
//     Y_t = Y.t();
//   };
//   
//   void sampleClusterParameters(){
//     arma::vec mu_n(P);
//     
//     arma::uword n_k = 0;
//     arma::vec mu_k(P), sample_mean(P);
//     arma::mat sample_cov(P, P), dist_from_prior(P, P), scale_n(P, P);
//     
//     for (arma::uword k = 0; k < K; k++) {
//       
//       // Find how many labels have the value
//       n_k = N_k(k);
//       if(n_k > 0){
//         
//         // Component data
//         arma::mat component_data = Y.rows( arma::find(labels == k) );
//         
//         // Sample mean in the component data
//         sample_mean = arma::mean(component_data).t();
//         
//         // The weighted average of the prior mean and sample mean
//         mu_k = (kappa * xi + n_k * sample_mean) / (double)(kappa + n_k);
//         
//         sample_cov = calcSampleCov(component_data, sample_mean, n_k, P);
//         
//         // Calculate the distance of the sample mean from the prior
//         dist_from_prior = (sample_mean - xi) * (sample_mean - xi).t();
//         
//         // Update the scale hyperparameter
//         scale_n = scale + sample_cov + ((kappa * n_k) / (double) (kappa + n_k)) * dist_from_prior;
//         
//         cov.slice(k) = arma::iwishrnd(scale_n, nu + n_k);
//         
//         mu.col(k) = arma::mvnrnd(mu_k, (1.0 / (double) (kappa + n_k)) * cov.slice(k), 1);
//         
//       } else{
//         
//         // If no members in the component, draw from the prior distribution
//         cov.slice(k) = arma::iwishrnd(scale, nu);
//         mu.col(k) = arma::mvnrnd(xi, (1.0 / (double) kappa) * cov.slice(k), 1);
//         
//       }
//       
//       cov_inv.slice(k) = arma::inv_sympd(cov.slice(k));
//       cov_log_det(k) = arma::log_det(cov.slice(k)).real();
//     }
//   }
// 
//   void sampleBatchParameters(){
//     
//     arma::uword n_b = 0;
//     double batch_sd = 0.0, theta_n = 0.0, m_bp = 0.0;
//     arma::vec batch_mean(P), dist_from_prior(P);
//     arma::mat Z(N, P);
//     
//     Z = X - Y;
//     
//     for (arma::uword b = 0; b < B; b++) {
//       
//       // Find how many labels have the value
//       n_b = N_b(b);
// 
//       // Component data
//       arma::mat batch_data = Z.rows( batch_ind(b) );
//         
//       // Sample mean in the component data
//       batch_mean = arma::mean(batch_data).t();
//       
//       // Calculate the distance of the sample mean from the prior
//       dist_from_prior = arma::pow(batch_mean - delta, 2.0);
//         
//       for(arma::uword p = 0; p < P; p++) {
//         // The weighted average of the prior mean and sample mean
//         m_bp = (lambda * delta(p) + n_b * batch_mean(p)) / (double)(lambda + n_b);
//         
//         batch_sd = arma::accu(arma::pow(batch_data.col(p) - batch_mean(p), 2.0));
//         
//   
//         // Update the scale hyperparameter
//         theta_n = theta + 0.5 * (batch_sd + ((lambda * n_b) / (double) (lambda + n_b)) * dist_from_prior(p));
//         
//         S(p, b) = arma::randg(arma::distr_param(rho * 0.5 + n_b, theta_n));
//         
//         if(b > 0) {
//           m(p, b) = arma::randn<double>() * S(p, b) / (lambda + n_b) + m_bp;
//         }
//       }
//     }
//   };
//   
//   // The log likelihood of a item belonging to each cluster given the batch label.
//   virtual arma::vec itemLogLikelihood(arma::vec item) {
//     
//     double exponent = 0.0;
//     arma::vec ll(K), dist_to_mean(P);
//     ll.zeros();
//     dist_to_mean.zeros();
//     
//     for(arma::uword k = 0; k < K; k++){
//       
//       // The exponent part of the MVN pdf
//       dist_to_mean = item - mu.col(k);
//       exponent = arma::as_scalar(dist_to_mean.t() * cov_inv.slice(k) * dist_to_mean);
//       
//       // Normal log likelihood
//       ll(k) = -0.5 *(cov_log_det(k) + exponent + (double) P * log(2.0 * M_PI));
//     }
//     
//     return(ll);
//   };
//   
//   // The cluster score is density specific
//   double yClusterScore(arma::uword k, arma::vec y) {
//     return -0.5*arma::as_scalar((y - mu.col(k)).t() * cov_inv.slice(k) * (y - mu.col(k)));
//   }
//   
//   // The unnormalised posterior density for the latent y's. Given the 
//   // ``yClusterScore`` function, this should be the same across mixture types.
//   double yLogKernel(arma::uword b, arma::uword k, arma::vec y, arma::uword n) {
//     
//     double score = yClusterScore(k, y);
// 
//     for(arma::uword p = 0; p < P; p++) {
//       score += -0.5*std::pow(y(p) - X(n, p) - m(p, b), 2.0) / S(p, b);
//     }
//     
//     return score;
//   };
//   
//   virtual void calcBIC(){
//     
//     // Each component has a weight, a mean vector and a symmetric covariance matrix. 
//     // Each batch has a mean and standard deviations vector.
//     // arma::uword n_param = (P + P * (P + 1) * 0.5) * K_occ + (2 * P) * B;
//     // BIC = n_param * std::log(N) - 2 * model_likelihood;
//     
//     // arma::uword n_param_cluster = 1 + P + P * (P + 1) * 0.5;
//     // arma::uword n_param_batch = 2 * P;
//     
//     // BIC = 2 * model_likelihood;
//     
//     BIC = 2 * model_likelihood - (n_param_batch + n_param_batch) * std::log(N);
//     
//     // for(arma::uword k = 0; k < K; k++) {
//     //   BIC -= n_param_cluster * std::log(N_k(k) + 1);
//     // }
//     // for(arma::uword b = 0; b < B; b++) {
//     //   BIC -= n_param_batch * std::log(N_b(b) + 1);
//     // }
//     
//   };
//   
//   virtual void yMetropolis() {
//     
//     arma::uword k = 0, b = 0;
//     double u = 0.0, proposed_model_score = 0.0, acceptance_prob = 0.0, current_model_score = 0.0;
//     arma::vec y_proposed(P);
//     
//     y_proposed.zeros();
//     
//     for(arma::uword n = 0; n < N; n++) {
//       
//       acceptance_prob = 0.0, proposed_model_score = 0.0, current_model_score = 0.0;
//       k = labels(n);
//       b = batch_vec(n);
//       
//       for(arma::uword p = 0; p < P; p++) {
//         
//         // Propose a new value for y_np
//         y_proposed(p) = arma::randn() * y_proposal_window + Y(n, p);
//   
//       }
//       
//       proposed_model_score = yLogKernel(b, k, y_proposed, n);
//       current_model_score = yLogKernel(b, k, Y_t.col(n), n);
//       
// 
//       u = arma::randu();
//       acceptance_prob = std::min(1.0, std::exp(proposed_model_score - current_model_score));
//       
//       if(u < acceptance_prob){
//         Y.row(n) = y_proposed.t();
//         y_count(n)++;
//         
//       }
//     }
//     
//     Y_t = Y.t();
//     
//   };
//   
//   virtual void sampleParameters() {
//     
//     // std::cout << "\n\nCluster.";
//     
//     // Update the cluster parameters
//     sampleClusterParameters();
//     
//     // std::cout << "\nBatch.\n";
//     
//     // Update the batch parameters
//     sampleBatchParameters();
//     
//     // std::cout << "\nY.";
//     
//     // Metropolis step for the y's
//     yMetropolis();
//     
//   };
//   
// };
// 
// 
// class paulSemiSupervisedSampler : public virtual paulSampler {
// private:
//   
// public:
//   
//   arma::uword N_fixed = 0;
//   arma::uvec fixed, unfixed_ind;
//   arma::mat alloc_prob;
//   
//   using paulSampler::paulSampler;
//   
//   paulSemiSupervisedSampler(
//     arma::uword _K,
//     arma::uword _B,
//     arma::uvec _labels,
//     arma::uvec _batch_vec,
//     arma::vec _concentration,
//     arma::mat _X,
//     arma::uvec _fixed
//   ) : 
//     paulSampler(_K, _B, _labels, _batch_vec, _concentration, _X) {
//     
//     arma::uvec fixed_ind(N);
//     
//     fixed = _fixed;
//     N_fixed = arma::sum(fixed);
//     fixed_ind = arma::find(_fixed == 1);
//     unfixed_ind = find(fixed == 0);
//     
//     alloc_prob.set_size(N, K);
//     alloc_prob.zeros();
//     
//     for (auto& n : fixed_ind) {
//       alloc_prob(n, labels(n)) = 1.0;
//     }
//   };
//   
//   // Destructor
//   virtual ~paulSemiSupervisedSampler() { };
//   
//   virtual void updateAllocation() {
//     
//     double u = 0.0;
//     arma::uvec uniqueK;
//     arma::vec comp_prob(K);
//     
//     for (auto& n : unfixed_ind) {
//       
//       ll = itemLogLikelihood(Y.row(n).t());
//       
//       // Update with weights
//       comp_prob = ll + log(w);
//       
//       likelihood(n) = arma::accu(comp_prob);
//       
//       // Normalise and overflow
//       comp_prob = exp(comp_prob - max(comp_prob));
//       comp_prob = comp_prob / sum(comp_prob);
//       
//       // Save the allocation probabilities
//       alloc_prob.row(n) = comp_prob.t();
//       
//       // Prediction and update
//       u = arma::randu<double>( );
//       
//       labels(n) = sum(u > cumsum(comp_prob));
//       alloc.row(n) = comp_prob.t();
//       
//       // Record the log likelihood of the item in it's allocated component
//       // likelihood(n) = ll(labels(n));
//     }
//     
//     // The model log likelihood
//     model_likelihood = arma::accu(likelihood);
//     
//     // Number of occupied components (used in BIC calculation)
//     uniqueK = arma::unique(labels);
//     K_occ = uniqueK.n_elem;
//   };
//   
// };
// 
// 
// class paulMVNPredictive : public paulMVNSampler, public paulSemiSupervisedSampler {
//   
// private:
//   
// public:
//   
//   using paulMVNSampler::paulMVNSampler;
//   
//   paulMVNPredictive(
//     arma::uword _K,
//     arma::uword _B,
//     double _y_proposal_window,
//     double _rho,
//     double _theta,
//     double _lambda,
//     arma::uvec _labels,
//     arma::uvec _batch_vec,
//     arma::vec _concentration,
//     arma::mat _X,
//     arma::uvec _fixed
//   ) : 
//     paulSampler(_K, _B, _labels, _batch_vec, _concentration, _X),
//     paulMVNSampler(_K,
//                    _B,
//                    _y_proposal_window,
//                    _rho,
//                    _theta,
//                    _lambda,
//                    _labels,
//                    _batch_vec,
//                    _concentration,
//                    _X),
//                paulSemiSupervisedSampler(_K, _B, _labels, _batch_vec, _concentration, _X, _fixed)
//   {
//   };
//   
//   virtual ~paulMVNPredictive() { };
//   
//   // virtual void sampleFromPriors() {
//   //   
//   //   arma::mat X_k;
//   //   
//   //   for(arma::uword k = 0; k < K; k++){
//   //     X_k = X.rows(arma::find(labels == k && fixed == 1));
//   //     cov.slice(k) = arma::diagmat(arma::stddev(X_k).t());
//   //     mu.col(k) = arma::mean(X_k).t();
//   //   }
//   //   for(arma::uword b = 0; b < B; b++){
//   //     for(arma::uword p = 0; p < P; p++){
//   //       
//   //       // Fix the 0th batch at no effect; all other batches have an effect
//   //       // relative to this
//   //       // if(b == 0){
//   //       S(p, b) = 1.0;
//   //       m(p, b) = 0.0;
//   //       // } else {
//   //       // S(p, b) = 1.0 / arma::randg<double>( arma::distr_param(rho, 1.0 / theta ) );
//   //       // m(p, b) = arma::randn<double>() * S(p, b) / lambda + delta(p);
//   //       // }
//   //     }
//   //   }
//   // };
//   
// };
// 
// 
// //' @title Sample batch mixture model
// //' @description Performs MCMC sampling for a mixture model with batch effects.
// //' @param X The data matrix to perform clustering upon (items to cluster in rows).
// //' @param K The number of components to model (upper limit on the number of clusters found).
// //' @param labels Vector item labels to initialise from.
// //' @param dataType Int, 0: independent Gaussians, 1: Multivariate normal, or 2: Categorical distributions.
// //' @param R The number of iterations to run for.
// //' @param thin thinning factor for samples recorded.
// //' @param concentration Vector of concentrations for mixture weights (recommended to be symmetric).
// //' @return Named list of the matrix of MCMC samples generated (each row 
// //' corresponds to a different sample) and BIC for each saved iteration.
// Rcpp::List samplePaulModel (
//     arma::mat X,
//     arma::uword K,
//     arma::uword B,
//     arma::uvec labels,
//     arma::uvec batch_vec,
//     double y_proposal_window,
//     double rho,
//     double theta,
//     double lambda,
//     arma::uword R,
//     arma::uword thin,
//     arma::vec concentration,
//     bool verbose = true,
//     bool doCombinations = false,
//     bool printCovariance = false
// ) {
//   
//   // The random seed is set at the R level via set.seed() apparently.
//   // std::default_random_engine generator(seed);
//   // arma::arma_rng::set_seed(seed);
//   
//   
//   paulMVNSampler my_sampler(K,
//                         B,
//                         y_proposal_window,
//                         rho,
//                         theta,
//                         lambda,
//                         labels,
//                         batch_vec,
//                         concentration,
//                         X
//   );
//   
//   // // Declare the factory
//   // samplerFactory my_factory;
//   // 
//   // // Convert from an int to the samplerType variable for our Factory
//   // samplerFactory::samplerType val = static_cast<samplerFactory::samplerType>(dataType);
//   // 
//   // // Make a pointer to the correct type of sampler
//   // std::unique_ptr<sampler> sampler_ptr = my_factory.createSampler(val,
//   //                                                                 K,
//   //                                                                 labels,
//   //                                                                 concentration,
//   //                                                                 X);
//   
//   // We use this enough that declaring it is worthwhile
//   arma::uword P = X.n_cols;
//   
//   // The output matrix
//   arma::umat class_record(floor(R / thin), X.n_rows);
//   class_record.zeros();
//   
//   // We save the BIC at each iteration
//   arma::vec BIC_record = arma::zeros<arma::vec>(floor(R / thin));
//   arma::vec model_likelihood = arma::zeros<arma::vec>(floor(R / thin));
//   arma::uvec acceptance_vec = arma::zeros<arma::uvec>(floor(R / thin));
//   arma::mat weights_saved = arma::zeros<arma::mat>(floor(R / thin), K);
//   
//   arma::cube mean_sum_saved(P, K * B, floor(R / thin)), mu_saved(P, K, floor(R / thin)), m_saved(P, B, floor(R / thin)), cov_saved(P, K * P, floor(R / thin)), t_saved(P, B, floor(R / thin)), cov_comb_saved(P, P * K * B, floor(R / thin));
//   // arma::field<arma::cube> cov_saved(my_sampler.P, my_sampler.P, K, floor(R / thin));
//   mu_saved.zeros();
//   cov_saved.zeros();
//   cov_comb_saved.zeros();
//   m_saved.zeros();
//   t_saved.zeros();
//   
//   arma::uword save_int = 0;
//   
//   // Sampler from priors
//   my_sampler.sampleFromPriors();
//   my_sampler.matrixCombinations();
//   // my_sampler.modelScore();
//   // sampler_ptr->sampleFromPriors();
//   
//   // my_sampler.model_score = my_sampler.modelLogLikelihood(
//   //   my_sampler.mu,
//   //   my_sampler.tau,
//   //   my_sampler.m,
//   //   my_sampler.t
//   // ) + my_sampler.priorLogProbability(
//   //     my_sampler.mu,
//   //     my_sampler.tau,
//   //     my_sampler.m,
//   //     my_sampler.t
//   // );
//   
//   // sample_prt.model_score->sampler_ptr.modelLo
//   
//   // Iterate over MCMC moves
//   for(arma::uword r = 0; r < R; r++){
//     
//     my_sampler.updateWeights();
//     
//     // Metropolis step for batch parameters
//     my_sampler.sampleParameters(); 
//     
//     my_sampler.updateAllocation();
//     
//     
//     // sampler_ptr->updateWeights();
//     // sampler_ptr->proposeNewParameters();
//     // sampler_ptr->updateAllocation();
//     
//     // Record results
//     if((r + 1) % thin == 0){
//       
//       // Update the BIC for the current model fit
//       // sampler_ptr->calcBIC();
//       // BIC_record( save_int ) = sampler_ptr->BIC; 
//       // 
//       // // Save the current clustering
//       // class_record.row( save_int ) = sampler_ptr->labels.t();
//       
//       my_sampler.calcBIC();
//       BIC_record( save_int ) = my_sampler.BIC;
//       model_likelihood( save_int ) = my_sampler.model_likelihood;
//       class_record.row( save_int ) = my_sampler.labels.t();
//       acceptance_vec( save_int ) = my_sampler.accepted;
//       weights_saved.row( save_int ) = my_sampler.w.t();
//       mu_saved.slice( save_int ) = my_sampler.mu;
//       // tau_saved.slice( save_int ) = my_sampler.tau;
//       // cov_saved( save_int ) = my_sampler.cov;
//       m_saved.slice( save_int ) = my_sampler.m;
//       t_saved.slice( save_int ) = my_sampler.S;
//       
//       
//       cov_saved.slice ( save_int ) = arma::reshape(arma::mat(my_sampler.cov.memptr(), my_sampler.cov.n_elem, 1, false), P, P * K);
//       
//       if(printCovariance) {  
//         std::cout << "\n\nCovariance cube:\n" << my_sampler.cov;
//         std::cout << "\n\nBatch covariance matrix:\n" << my_sampler.S;
//       }
//       
//       save_int++;
//     }
//   }
//   
//   if(verbose) {
//     std::cout << "\nY acceptance rate:\n" << arma::conv_to< arma::vec >::from(my_sampler.y_count) / R;
//   }
//   
//   return(List::create(Named("samples") = class_record, 
//                       Named("means") = mu_saved,
//                       Named("covariance") = cov_saved,
//                       Named("batch_shift") = m_saved,
//                       Named("batch_scale") = t_saved,
//                       Named("mean_sum") = mean_sum_saved,
//                       Named("cov_comb") = cov_comb_saved,
//                       Named("weights") = weights_saved,
//                       Named("y_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.y_count) / R,                      
//                       Named("likelihood") = model_likelihood,
//                       Named("BIC") = BIC_record));
//   
// };
// 
// 
// 
// 
// 
// 
// 
// //' @title Mixture model
// //' @description Performs MCMC sampling for a mixture model.
// //' @param X The data matrix to perform clustering upon (items to cluster in rows).
// //' @param K The number of components to model (upper limit on the number of clusters found).
// //' @param labels Vector item labels to initialise from.
// //' @param fixed Binary vector of the items that are fixed in their initial label.
// //' @param dataType Int, 0: independent Gaussians, 1: Multivariate normal, or 2: Categorical distributions.
// //' @param R The number of iterations to run for.
// //' @param thin thinning factor for samples recorded.
// //' @param concentration Vector of concentrations for mixture weights (recommended to be symmetric).
// //' @return Named list of the matrix of MCMC samples generated (each row 
// //' corresponds to a different sample) and BIC for each saved iteration.
// Rcpp::List sampleSemiSupervisedPaulModel (
//     arma::mat X,
//     arma::uword K,
//     arma::uword B,
//     arma::uvec labels,
//     arma::uvec batch_vec,
//     arma::uvec fixed,
//     double y_proposal_window,
//     double rho,
//     double theta,
//     double lambda,
//     arma::uword R,
//     arma::uword thin,
//     arma::vec concentration,
//     bool verbose = true,
//     bool doCombinations = false,
//     bool printCovariance = false
// ) {
//   
//   // // Set the random number
//   // std::default_random_engine generator(seed);
//   // 
//   // // Declare the factory
//   // semisupervisedSamplerFactory my_factory;
//   // 
//   // // Convert from an int to the samplerType variable for our Factory
//   // semisupervisedSamplerFactory::samplerType val = static_cast<semisupervisedSamplerFactory::samplerType>(dataType);
//   // 
//   // // Make a pointer to the correct type of sampler
//   // std::unique_ptr<sampler> sampler_ptr = my_factory.createSemisupervisedSampler(val,
//   //                                                                               K,
//   //                                                                               labels,
//   //                                                                               concentration,
//   //                                                                               X,
//   //                                                                               fixed);
//   
//   // std::cout << "\nDeclare sampler.";
//   
//   paulMVNPredictive my_sampler(K,
//                            B,
//                            y_proposal_window,
//                            rho,
//                            theta,
//                            lambda,
//                            labels,
//                            batch_vec,
//                            concentration,
//                            X,
//                            fixed
//   );
//   
//   // // Declare the factory
//   // samplerFactory my_factory;
//   // 
//   // // Convert from an int to the samplerType variable for our Factory
//   // samplerFactory::samplerType val = static_cast<samplerFactory::samplerType>(dataType);
//   // 
//   // // Make a pointer to the correct type of sampler
//   // std::unique_ptr<sampler> sampler_ptr = my_factory.createSampler(val,
//   //                                                                 K,
//   //                                                                 labels,
//   //                                                                 concentration,
//   //                                                                 X);
//   
//   arma::uword P = X.n_cols, N = X.n_rows;
//   
//   // The output matrix
//   arma::umat class_record(floor(R / thin), X.n_rows);
//   class_record.zeros();
//   
//   // We save the BIC at each iteration
//   arma::vec BIC_record = arma::zeros<arma::vec>(floor(R / thin));
//   arma::vec model_likelihood = arma::zeros<arma::vec>(floor(R / thin));
//   arma::uvec acceptance_vec = arma::zeros<arma::uvec>(floor(R / thin));
//   arma::mat weights_saved = arma::zeros<arma::mat>(floor(R / thin), K);
//   
//   arma::cube Y_saved(N, P, floor(R / thin)), mu_saved(P, K, floor(R / thin)), m_saved(P, B, floor(R / thin)), S_saved(P, B, floor(R / thin)), cov_saved(P, K * P, floor(R / thin)), alloc_prob(my_sampler.N, K, floor(R / thin));
//   // arma::field<arma::cube> cov_saved(my_sampler.P, my_sampler.P, K, floor(R / thin));
//   mu_saved.zeros();
//   cov_saved.zeros();
//   m_saved.zeros();
//   S_saved.zeros();
// 
//   Y_saved.zeros();
//   
//   arma::uword save_int = 0;
//   
//   // Sampler from priors
//   
//   // std::cout << "\nSample from prior.";
//   
//   my_sampler.sampleFromPriors();
//   
//   // std::cout << "\nMatrix combinations.";
//   
//   my_sampler.matrixCombinations();
//   // my_sampler.modelScore();
//   // sampler_ptr->sampleFromPriors();
//   
//   // my_sampler.model_score = my_sampler.modelLogLikelihood(
//   //   my_sampler.mu,
//   //   my_sampler.tau,
//   //   my_sampler.m,
//   //   my_sampler.t
//   // ) + my_sampler.priorLogProbability(
//   //     my_sampler.mu,
//   //     my_sampler.tau,
//   //     my_sampler.m,
//   //     my_sampler.t
//   // );
//   
//   // sample_prt.model_score->sampler_ptr.modelLo
//   
//   // Iterate over MCMC moves
//   for(arma::uword r = 0; r < R; r++){
//     
//     // std::cout << "\nWeights.";
//     my_sampler.updateWeights();
//     
//     // std::cout << "\nParameters.";
//     // Metropolis step for batch parameters
//     my_sampler.sampleParameters(); 
//     
//     // std::cout << "\nAllocation.";
//     my_sampler.updateAllocation();
//     
//     
//     // sampler_ptr->updateWeights();
//     // sampler_ptr->proposeNewParameters();
//     // sampler_ptr->updateAllocation();
//     
//     // Record results
//     if((r + 1) % thin == 0){
//       
//       // Update the BIC for the current model fit
//       // sampler_ptr->calcBIC();
//       // BIC_record( save_int ) = sampler_ptr->BIC; 
//       // 
//       // // Save the current clustering
//       // class_record.row( save_int ) = sampler_ptr->labels.t();
//       
//       my_sampler.calcBIC();
//       BIC_record( save_int ) = my_sampler.BIC;
//       model_likelihood( save_int ) = my_sampler.model_likelihood;
//       class_record.row( save_int ) = my_sampler.labels.t();
//       acceptance_vec( save_int ) = my_sampler.accepted;
//       weights_saved.row( save_int ) = my_sampler.w.t();
//       mu_saved.slice( save_int ) = my_sampler.mu;
//       // tau_saved.slice( save_int ) = my_sampler.tau;
//       // cov_saved( save_int ) = my_sampler.cov;
//       m_saved.slice( save_int ) = my_sampler.m;
//       S_saved.slice( save_int ) = my_sampler.S;
// 
//       Y_saved.slice( save_int ) = my_sampler.Y;
//       
//       alloc_prob.slice( save_int ) = my_sampler.alloc_prob;
//       cov_saved.slice ( save_int ) = arma::reshape(arma::mat(my_sampler.cov.memptr(), my_sampler.cov.n_elem, 1, false), P, P * K);
// 
//       if(printCovariance) {  
//         std::cout << "\n\nCovariance cube:\n" << my_sampler.cov;
//         std::cout << "\n\nBatch covariance matrix:\n" << my_sampler.S;
//       }
//       
//       save_int++;
//     }
//   }
//   
//   if(verbose) {
//     std::cout << "\nY acceptance rate:\n" << arma::conv_to< arma::vec >::from(my_sampler.y_count) / R;
//   }
//   
//   return(
//     List::create(Named("samples") = class_record, 
//       Named("Y") = Y_saved,
//       Named("means") = mu_saved,
//       Named("covariance") = cov_saved,
//       Named("batch_shift") = m_saved,
//       Named("batch_scale") = S_saved,
//       Named("weights") = weights_saved,
//       Named("y_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.y_count) / R,
//       Named("alloc_prob") = alloc_prob,
//       Named("likelihood") = model_likelihood,
//       Named("BIC") = BIC_record
//     )
//   );
//   
// };
