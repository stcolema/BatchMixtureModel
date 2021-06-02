// samplerFactory.h
// =============================================================================
// include guard
#ifndef SAMPLEMVN_H
#define SAMPLEMVN_H

// =============================================================================
// included dependencies
# include <RcppArmadillo.h>
# include "mvnSampler.h"

// =============================================================================
// sampleMVN function header

//' @title Sample batch mixture model
//' @description Performs MCMC sampling for a mixture model with batch effects.
//' @param X The data matrix to perform clustering upon (items to cluster in rows).
//' @param K The number of components to model (upper limit on the number of clusters found).
//' @param labels Vector item labels to initialise from.
//' @param dataType Int, 0: independent Gaussians, 1: Multivariate normal, or 2: Categorical distributions.
//' @param R The number of iterations to run for.
//' @param thin thinning factor for samples recorded.
//' @param concentration Vector of concentrations for mixture weights (recommended to be symmetric).
//' @return Named list of the matrix of MCMC samples generated (each row 
//' corresponds to a different sample) and BIC for each saved iteration.
// [[Rcpp::export]]
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
) ;

#endif /* SAMPLEMVN_H */