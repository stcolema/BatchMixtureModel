// sampleSemisupervisedMVN.cpp
// =============================================================================
// included dependencies
# include <RcppArmadillo.h>
# include "sampleSemisupervisedMVN.h"

// =============================================================================
// namespace
using namespace Rcpp ;
using namespace arma ;

// =============================================================================
// sampleSemisupervisedMVN function implementation

Rcpp::List sampleSemisupervisedMVN (
    arma::mat X,
    arma::uword K,
    arma::uword B,
    arma::uvec labels,
    arma::uvec batch_vec,
    arma::uvec fixed,
    double mu_proposal_window,
    double cov_proposal_window,
    double m_proposal_window,
    double S_proposal_window,
    arma::uword R,
    arma::uword thin,
    arma::vec concentration,
    double m_scale,
    double rho,
    double theta,
    arma::mat initial_mu,
    arma::cube initial_cov,
    arma::mat initial_m,
    arma::mat initial_S,
    bool mu_initialised,
    bool cov_initialised,
    bool m_initialised,
    bool S_initialised
) {
  
  mvnPredictive my_sampler(K,
    B,
    mu_proposal_window,
    cov_proposal_window,
    m_proposal_window,
    S_proposal_window,
    labels,
    batch_vec,
    concentration,
    X,
    fixed,
    m_scale,
    rho,
    theta
  );
  
  uword P = X.n_cols, N = X.n_rows;

  // The output matrix
  umat class_record(std::floor(R / thin), X.n_rows);
  class_record.zeros();
  
  // We save the BIC at each iteration
  vec BIC_record = zeros<vec>(std::floor(R / thin)),
    observed_likelihood = zeros<vec>(std::floor(R / thin)),
    complete_likelihood = zeros<vec>(std::floor(R / thin));
  
  mat weights_saved = zeros<mat>(std::floor(R / thin), K);
  
  cube mean_sum_saved(P, K * B, std::floor(R / thin)), 
    mu_saved(P, K, std::floor(R / thin)),
    m_saved(P, B, std::floor(R / thin)), 
    cov_saved(P, K * P, std::floor(R / thin)),
    S_saved(P, B, std::floor(R / thin)), 
    cov_comb_saved(P, P * K * B, std::floor(R / thin)), 
    alloc(N, K, std::floor(R / thin)), 
    batch_corrected_data(N, P, std::floor(R / thin));

  mu_saved.zeros();
  cov_saved.zeros();
  cov_comb_saved.zeros();
  m_saved.zeros();
  S_saved.zeros();

  uword save_int = 0;
  
  // Sampler from priors
  my_sampler.sampleFromPriors();
  
  // Pass initial values if any are given
  if(mu_initialised) {
    my_sampler.mu = initial_mu;
  }
   if(cov_initialised) {
     my_sampler.cov = initial_cov;
   }
   if(m_initialised) {
     my_sampler.m = initial_m;
   }
   if(S_initialised) {
     my_sampler.S = initial_S;
   }

  my_sampler.matrixCombinations();
 
  // Iterate over MCMC moves
  for(uword r = 0; r < R; r++){
    
    my_sampler.updateWeights();

    // Metropolis step for batch parameters
    my_sampler.metropolisStep();
    
    my_sampler.updateAllocation();

    // Record results
    if((r + 1) % thin == 0){
      
      // Update the BIC for the current model fit
      my_sampler.calcBIC();
      BIC_record( save_int ) = my_sampler.BIC;
      observed_likelihood( save_int ) = my_sampler.observed_likelihood;
      complete_likelihood( save_int ) = my_sampler.complete_likelihood;
      
      class_record.row( save_int ) = my_sampler.labels.t();
      alloc.slice( save_int ) = my_sampler.alloc;
      
      weights_saved.row( save_int ) = my_sampler.w.t();
      mu_saved.slice( save_int ) = my_sampler.mu;
      m_saved.slice( save_int ) = my_sampler.m;
      S_saved.slice( save_int ) = my_sampler.S;
      mean_sum_saved.slice( save_int ) = my_sampler.mean_sum;
      
      cov_saved.slice ( save_int ) = reshape(mat(my_sampler.cov.memptr(), my_sampler.cov.n_elem, 1, false), P, P * K);
      cov_comb_saved.slice( save_int) = reshape(mat(my_sampler.cov_comb.memptr(), my_sampler.cov_comb.n_elem, 1, false), P, P * K * B); 
      
      my_sampler.updateBatchCorrectedData();
      batch_corrected_data.slice( save_int ) =  my_sampler.Y;


      save_int++;
    }
  }

  return(
    List::create(Named("samples") = class_record, 
      Named("means") = mu_saved,
      Named("covariance") = cov_saved,
      Named("batch_shift") = m_saved,
      Named("batch_scale") = S_saved,
      Named("mean_sum") = mean_sum_saved,
      Named("cov_comb") = cov_comb_saved,
      Named("weights") = weights_saved,
      Named("cov_acceptance_rate") = conv_to< vec >::from(my_sampler.cov_count) / R,
      Named("mu_acceptance_rate") = conv_to< vec >::from(my_sampler.mu_count) / R,
      Named("S_acceptance_rate") = conv_to< vec >::from(my_sampler.S_count) / R,
      Named("m_acceptance_rate") = conv_to< vec >::from(my_sampler.m_count) / R,
      Named("alloc") = alloc,
      Named("observed_likelihood") = observed_likelihood,
      Named("complete_likelihood") = complete_likelihood,
      Named("BIC") = BIC_record,
      Named("batch_corrected_data") = batch_corrected_data
    )
  );
  
};
