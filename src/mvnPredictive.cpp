// mvnPredictive.cpp
// =============================================================================
// included dependencies
# include <RcppArmadillo.h>
# include "pdfs.h"
# include "sampler.h"
# include "semisupervisedSampler.h"
# include "mvnSampler.h"
# include "mvnPredictive.h"

// =============================================================================
// namespace
using namespace Rcpp ;
using namespace arma ;

// =============================================================================
// mvnPredictive class
mvnPredictive::mvnPredictive(
  arma::uword _K,
  arma::uword _B,
  double _mu_proposal_window,
  double _cov_proposal_window,
  double _m_proposal_window,
  double _S_proposal_window,
  arma::uvec _labels,
  arma::uvec _batch_vec,
  arma::vec _concentration,
  arma::mat _X,
  arma::uvec _fixed
) :
  sampler(_K, _B, _labels, _batch_vec, _concentration, _X),
  mvnSampler(                           
    _K,
    _B,
    _mu_proposal_window,
    _cov_proposal_window,
    _m_proposal_window,
    _S_proposal_window,
    _labels,
    _batch_vec,
    _concentration,
    _X
  ),
  semisupervisedSampler(_K, _B, _labels, _batch_vec, _concentration, _X, _fixed)
{
};