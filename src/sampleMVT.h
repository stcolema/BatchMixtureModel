// sampleMVT.h
// =============================================================================
// include guard
#ifndef SAMPLEMVT_H
#define SAMPLEMVT_H

// =============================================================================
// included dependencies
# include <RcppArmadillo.h>
# include "mvtSampler.h"

// =============================================================================
// sampleMVT function header
// [[Rcpp::export]]
Rcpp::List sampleMVT (
    arma::mat X,
    arma::uword K,
    arma::uword B,
    arma::uvec labels,
    arma::uvec batch_vec,
    double mu_proposal_window,
    double cov_proposal_window,
    double m_proposal_window,
    double S_proposal_window,
    double t_df_proposal_window,
    arma::uword R,
    arma::uword thin,
    arma::vec concentration
);

#endif /* SAMPLEMVT_H */