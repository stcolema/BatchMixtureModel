// sampleMVN.cpp
// =============================================================================
// included dependencies
# include <RcppArmadillo.h>
# include "sampleMVN.h"

// =============================================================================
// namespace
using namespace Rcpp ;
using namespace arma ;

// =============================================================================
// sampleMVN function implementation

Rcpp::List sampleMVN (
    arma::mat X,
    arma::uword K,
    arma::uword B,
    arma::uvec labels,
    arma::uvec batch_vec,
    double mu_proposal_window,
    double cov_proposal_window,
    double m_proposal_window,
    double S_proposal_window,
    arma::uword R,
    arma::uword thin,
    arma::vec concentration
) {
  
  mvnSampler my_sampler(K,
    B,
    mu_proposal_window,
    cov_proposal_window,
    m_proposal_window,
    S_proposal_window,
    labels,
    batch_vec,
    concentration,
    X
  );
  
  // We use this enough that declaring it is worthwhile
  arma::uword P = X.n_cols;
  
  // The output matrix
  arma::umat class_record(std::floor(R / thin), X.n_rows);
  class_record.zeros();
  
  // We save the BIC and model likelihoods
  arma::vec BIC_record = arma::zeros<arma::vec>(std::floor(R / thin)),
    observed_likelihood = arma::zeros<arma::vec>(std::floor(R / thin)),
    complete_likelihood = arma::zeros<arma::vec>(std::floor(R / thin));
  
  // Sampled weights
  arma::mat weights_saved = arma::zeros<arma::mat>(std::floor(R / thin), K);
  
  // Various sampled parameters of interest
  arma::cube mean_sum_saved(P, K * B, std::floor(R / thin)),
    mu_saved(P, K, std::floor(R / thin)), 
    m_saved(P, B, std::floor(R / thin)),
    cov_saved(P, K * P, std::floor(R / thin)), 
    S_saved(P, B, std::floor(R / thin)),
    cov_comb_saved(P, P * K * B, std::floor(R / thin)), 
    batch_corrected_data(my_sampler.N, P, std::floor(R / thin));
  
  mu_saved.zeros();
  cov_saved.zeros();
  cov_comb_saved.zeros();
  m_saved.zeros();
  
  // Counter for saved objects
  arma::uword save_int = 0;
  
  // Sampler from priors
  my_sampler.sampleFromPriors();
  my_sampler.matrixCombinations();
  
  // Iterate over MCMC moves
  for(arma::uword r = 0; r < R; r++){
    
    // Sample component weights
    my_sampler.updateWeights();
    
    // Metropolis step for batch parameters
    my_sampler.metropolisStep(); 
    
    // Sample clustering
    my_sampler.updateAllocation();
    
    // Record results
    if((r + 1) % thin == 0){
      
      // Update the BIC for the current model fit
      my_sampler.calcBIC();
      BIC_record( save_int ) = my_sampler.BIC;
      
      // Save the observed and model likelihoods
      observed_likelihood( save_int ) = my_sampler.observed_likelihood;
      complete_likelihood( save_int ) = my_sampler.complete_likelihood;
      
      // Vaious inferred objects of interest
      class_record.row( save_int ) = my_sampler.labels.t();
      weights_saved.row( save_int ) = my_sampler.w.t();
      mu_saved.slice( save_int ) = my_sampler.mu;
      m_saved.slice( save_int ) = my_sampler.m;
      S_saved.slice( save_int ) = my_sampler.S;
      mean_sum_saved.slice( save_int ) = my_sampler.mean_sum;
      
      // The covariance is saved in a non-ideal way
      cov_saved.slice ( save_int ) = arma::reshape(arma::mat(my_sampler.cov.memptr(), my_sampler.cov.n_elem, 1, false), P, P * K);
      cov_comb_saved.slice( save_int) = arma::reshape(arma::mat(my_sampler.cov_comb.memptr(), my_sampler.cov_comb.n_elem, 1, false), P, P * K * B); 
      
      // Update the ``batch-corrected'' data based on current batch and class
      // parameters
      my_sampler.updateBatchCorrectedData();
      batch_corrected_data.slice( save_int ) =  my_sampler.Y;
      
      save_int++;
    }
  }
  
  return(
    List::create(
      Named("samples") = class_record, 
      Named("means") = mu_saved,
      Named("covariance") = cov_saved,
      Named("batch_shift") = m_saved,
      Named("batch_scale") = S_saved,
      Named("mean_sum") = mean_sum_saved,
      Named("cov_comb") = cov_comb_saved,
      Named("weights") = weights_saved,
      Named("cov_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.cov_count) / R,
      Named("mu_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.mu_count) / R,
      Named("S_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.S_count) / R,
      Named("m_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.m_count) / R,                      
      Named("complete_likelihood") = complete_likelihood,
      Named("observed_likelihood") = observed_likelihood,
      Named("BIC") = BIC_record
    )
  );
};
