// pdfs.h
// =============================================================================
// include guard
#ifndef PDFS_H
#define PDFS_H

// =============================================================================
// included dependencies
# include <RcppArmadillo.h>

// =============================================================================
// Log-likelihood functions used in Metropolis-hastings



double gammaLogLikelihood(double x, double shape, double rate);
double invGammaLogLikelihood(double x, double shape, double scale);
double wishartLogLikelihood(arma::mat X, arma::mat V, double n, arma::uword P);
double invWishartLogLikelihood(arma::mat X, arma::mat Psi, double nu, arma::uword P);

#endif /* PDFS_H */