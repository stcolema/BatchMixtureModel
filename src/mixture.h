// mixture.h
// =============================================================================
// include guard
#ifndef MIXTURE_H
#define MIXTURE_H

// =============================================================================
// included dependencies
# include <RcppArmadillo.h>

// =============================================================================
// mixture class
class mixture {
  
public:
  
  arma::uword K, B, N, P, K_occ, n_param;
  double complete_likelihood = 0.0,
    observed_likelihood = 0.0, 
    BIC = 0.0;
  
  arma::uvec labels, N_k, batch_vec, N_b, KB_inds, B_inds;
  arma::vec ll, likelihood;
  arma::umat members;
  arma::mat X, X_t, alloc;
  arma::field < arma::uvec > batch_ind;
  
  
  mixture(
    arma::uword _K,
    arma::uword _B,
    arma::uvec _labels,
    arma::uvec _batch_vec,
    arma::mat _X);
  
  // Destructor
  virtual ~mixture();
  
  // Functions required of all mixture models
  virtual void metropolisStep() = 0;
  virtual void sampleFromPriors() = 0;
  virtual void calcBIC() = 0;
  virtual arma::vec itemLogLikelihood(arma::vec x, arma::uword b) = 0;
  virtual void updateBatchCorrectedData() = 0;
};

#endif /* MIXTURE_H */