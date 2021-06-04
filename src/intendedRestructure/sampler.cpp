// # include <RcppArmadillo.h>
// # include <math.h>
// # include <string>
// # include <iostream>
// 
// # include "sampler.h"
// # include "mixture.h"
// 
// // [[Rcpp::depends(RcppArmadillo)]]
// 
// using namespace Rcpp ;
// using namespace arma ;
// 
// 
// // Parametrised class
// sampler::sampler(
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
//     X = _X;
//     X_t = X.t();
// 
//     // Plausibly belongs in the MVN sampler. Used for selecting slices / columns
//     // in the metropolis steps.
//     KB_inds = linspace<uvec>(0, K - 1, K) * B;
//     B_inds = linspace<uvec>(0, B - 1, B);
// 
//     // Dimensions
//     N = X.n_rows;
//     P = X.n_cols;
// 
//     // std::cout << "\nN: " << N << "\nP: " << P << "\n\n";
// 
//     // Class populations
//     N_k = zeros<uvec>(K);
//     N_b = zeros<uvec>(B);
// 
//     // The batch numbers won't ever change, so let's count them now
//     for(uword b = 0; b < B; b++){
//       N_b(b) = sum(batch_vec == b);
//     }
// 
//     // Weights
//     // double x, y;
//     w = zeros<vec>(K);
// 
//     // Log likelihood (individual and model)
//     ll = zeros<vec>(K);
//     likelihood = zeros<vec>(N);
// 
//     // Class members
//     members.set_size(N, K);
//     members.zeros();
// 
//     // Allocation probability matrix (only makes sense in predictive models)
//     alloc.set_size(N, K);
//     alloc.zeros();
// 
//     // The indices of the members of each batch in the dataset
//     batch_ind.set_size(B);
//     for(uword b = 0; b < B; b++) {
//       batch_ind(b) = find(batch_vec == b);
//     }
//   };
// 
// // Destructor
// sampler::~sampler() { };
// 
// // Functions required of all mixture models
// void sampler::updateWeights(){
// 
//   double a = 0.0;
// 
//   for (uword k = 0; k < K; k++) {
// 
//     // Find how many labels have the value
//     members.col(k) = labels == k;
//     N_k(k) = sum(members.col(k));
// 
//     // Update weights by sampling from a Gamma distribution
//     a  = concentration(k) + N_k(k);
//     w(k) = randg( distr_param(a, 1.0) );
//   }
// 
//   // Convert the cluster weights (previously gamma distributed) to Beta
//   // distributed by normalising
//   w = w / accu(w);
// 
// };
// 
// void sampler::updateAllocation() {
// 
//   double u = 0.0;
//   uvec uniqueK;
//   vec comp_prob(K);
// 
//   complete_likelihood = 0.0;
//   for(uword n = 0; n < N; n++){
// 
//     ll = itemLogLikelihood(X_t.col(n), batch_vec(n));
// 
//     // std::cout << "\n\nAllocation log likelihood: " << ll;
//     // Update with weights
//     comp_prob = ll + log(w);
// 
//     likelihood(n) = accu(comp_prob);
// 
//     // std::cout << "\n\nWeights: " << w;
//     // std::cout << "\n\nAllocation log probability: " << comp_prob;
// 
//     // Normalise and overflow
//     comp_prob = exp(comp_prob - max(comp_prob));
//     comp_prob = comp_prob / sum(comp_prob);
// 
//     // Prediction and update
//     u = randu<double>( );
// 
//     labels(n) = sum(u > cumsum(comp_prob));
//     alloc.row(n) = comp_prob.t();
// 
//     complete_likelihood += ll(labels(n));
// 
//     // Record the likelihood of the item in it's allocated component
//     // likelihood(n) = ll(labels(n));
//   }
// 
//   // The model log likelihood
//   observed_likelihood = accu(likelihood);
// 
//   // Number of occupied components (used in BIC calculation)
//   uniqueK = unique(labels);
//   K_occ = uniqueK.n_elem;
// };
// 
// // // The virtual functions that will be defined in any subclasses
// void sampler::metropolisStep() = 0;
// void sampler::sampleFromPriors() = 0;
// void sampler::calcBIC() = 0;
// arma::vec sampler::itemLogLikelihood(arma::vec x, arma::uword b) = 0;
// 
