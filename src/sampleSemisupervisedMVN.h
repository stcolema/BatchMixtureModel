// sampleSemisupervisedMVN.h
// =============================================================================
// include guard
#ifndef SAMPLESEMISUPERVISEDMVN_H
#define SAMPLESEMISUPERVISEDMVN_H

// =============================================================================
// included dependencies
# include <RcppArmadillo.h>
# include "mvnPredictive.h"

// =============================================================================
// sampleSemisupervisedMVN function header

//' @title Sample semi-supervised MVN Mixture model
//' @description Performs MCMC sampling for a mixture model.
//' @param X The data matrix to perform clustering upon (items to cluster in rows).
//' @param K The number of components to model (upper limit on the number of clusters found).
//' @param B The number of batches to model.
//' @param labels Vector item labels to initialise from.
//' @param batch_vec Observed batch labels.
//' @param fixed Binary vector of the items that are fixed in their initial label.
//' @param mu_proposal_window The standard deviation for the Gaussian proposal density of the cluster means.
//' @param cov_proposal_window The degrees of freedom for the Wishart proposal density of the cluster covariances.
//' @param m_proposal_window The standard deviation for the Gaussian proposal density of the batch mean effects.
//' @param S_proposal_window The rate for the Gamma proposal density of the batch scale.
//' @param R The number of iterations to run for.
//' @param thin thinning factor for samples recorded.
//' @param concentration Vector of concentrations for mixture weights (recommended to be symmetric).
//' @return Named list of the matrix of MCMC samples generated (each row 
//' corresponds to a different sample) and BIC for each saved iteration.
// [[Rcpp::export]]
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
    double theta
) ;

#endif /* SAMPLESEMISUPERVISEDMVN_H */