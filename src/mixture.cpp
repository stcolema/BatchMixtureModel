# include <RcppArmadillo.h>
# include "mixture.h"

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp ;
using namespace arma ;

// Parametrised class
mixture::mixture(
  arma::uword _K,
  arma::uword _B,
  arma::uvec _labels,
  arma::uvec _batch_vec,
  arma::mat _X)
{

  K = _K;
  B = _B;
  labels = _labels;
  batch_vec = _batch_vec;
  X = _X;
  X_t = X.t();

  // Plausibly belongs in the MVN mixture. Used for selecting slices / columns
  // in the metropolis steps.
  KB_inds = linspace<uvec>(0, K - 1, K) * B;
  B_inds = linspace<uvec>(0, B - 1, B);

  // Dimensions
  N = X.n_rows;
  P = X.n_cols;

  // Class populations
  N_k = zeros<uvec>(K);
  N_b = zeros<uvec>(B);

  // The batch numbers won't ever change, so let's count them now
  for(uword b = 0; b < B; b++){
    N_b(b) = sum(batch_vec == b);
  }

  // Class members
  members.set_size(N, K);
  members.zeros();

  // Allocation probability matrix (only makes sense in predictive models)
  alloc.set_size(N, K);
  alloc.zeros();

  // The indices of the members of each batch in the dataset
  batch_ind.set_size(B);
  for(uword b = 0; b < B; b++) {
    batch_ind(b) = find(batch_vec == b);
  }
};

// Destructor
mixture::~mixture() { };

// The virtual functions that will be defined in any subclasses
// mixture::metropolisStep() = 0;
// mixture::sampleFromPriors() = 0;
// mixture::calcBIC() = 0;
// mixture::itemLogLikelihood(arma::vec x, arma::uword b) = 0;
// mixture::updateBatchCorrectedData = 0;