// sampleSemisupervisedMVT.h
// =============================================================================
// include guard
#ifndef SAMPLESEMISUPERVISEDMVT_H
#define SAMPLESEMISUPERVISEDMVT_H

// =============================================================================
// included dependencies
# include <RcppArmadillo.h>
# include "mvtPredictive.h"

// =============================================================================
// sampleSemisupervisedMVT function header


// [[Rcpp::export]]
Rcpp::List sampleSemisupervisedMVT (
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
  double t_df_proposal_window,
  arma::uword R,
  arma::uword thin,
  arma::vec concentration
);

#endif /* SAMPLESEMISUPERVISEDMVT_H */