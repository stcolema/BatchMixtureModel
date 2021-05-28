// mvnMixture.h
// =============================================================================
// include guard
#ifndef MVNMIXTURE_H
#define MVNMIXTURE_H

// =============================================================================
// included dependencies
# include "mixture.h"

// =============================================================================
// mvnMixture class

//' @name mvnMixture
//' @title Multivariate Normal mixture type
//' @description The sampler for the Multivariate Normal mixture model for batch effects.
//' @field new Constructor \itemize{
//' \item Parameter: K - the number of components to model
//' \item Parameter: B - the number of batches present
//' \item Parameter: labels - the initial clustering of the data
//' \item Parameter: X - the data to model
//' }
//' @field printType Print the sampler type called.
//' @field updateWeights Update the weights of each component based on current
//' clustering.
//' @field updateAllocation Sample a new clustering.
//' @field sampleFromPrior Sample from the priors for the multivariate normal
//' density.
//' @field calcBIC Calculate the BIC of the model.
//' @field logLikelihood Calculate the likelihood of a given data point in each
//' component. \itemize{
//' \item Parameter: point - a data point.
//' }
class mvnMixture: virtual public mixture {

public:

  arma::uword n_param_cluster = 1 + P + P * (P + 1) * 0.5,
    n_param_batch = 2 * P;

  // Prior hyperparameters and proposal parameters
  double kappa = 0.01,
    nu = P + 2,

    // Hyperparameters for the batch mean
    delta = 0.0,
    t,

    // Hyperparameters for the batch scale. These choices give > 99% of sampled
    // values in the range of 1.2 to 2.0 which seems a sensible prior belief.
    // Posisbly a little too informative; if S = 2.0 we're saying the batch
    // provides as much variation as the biology. However, as our batch scales
    // are strictly greater than 1.0 some shared global scaling is collected
    // here.
    rho = 21,
    theta = 10,
    S_loc = 1.0, // this gives the batch scale a support of (1.0, \infty)

    // Proposal windows
    mu_proposal_window,
    cov_proposal_window,
    m_proposal_window,
    S_proposal_window;


  arma::uvec mu_count, cov_count, m_count, S_count, phi_count, rcond_count;
  arma::vec xi, cov_log_det, global_mean;
  arma::mat scale, mu, m, S, phi, cov_comb_log_det, mean_sum, global_cov, Y;
  arma::cube cov, cov_inv, cov_comb, cov_comb_inv;

  using mixture::mixture;

  mvnMixture(
    arma::uword _K,
    arma::uword _B,
    double _mu_proposal_window,
    double _cov_proposal_window,
    double _m_proposal_window,
    double _S_proposal_window,
    arma::uvec _labels,
    arma::uvec _batch_vec,
    arma::mat _X
  ) ;


  // Destructor
  virtual ~mvnMixture() { };

  // Print the sampler type.
  virtual void printType();

  // Sample from priors
  virtual void sampleCovPrior();
  virtual void sampleMuPrior();
  virtual void sampleSPrior();
  virtual void sampleMPrior();
  virtual void sampleFromPriors();

  // Update the common matrix manipulations to avoid recalculating N times
  virtual void matrixCombinations();

  // The log likelihood of a item belonging to each cluster given the batch label.
  virtual arma::vec itemLogLikelihood(arma::vec item, arma::uword b);
  virtual void calcBIC();

  // Likelihood of class/batch
  virtual double groupLikelihood(arma::uvec inds,
                                 arma::uvec group_inds,
                                 arma::vec cov_det,
                                 arma::mat mean_sum,
                                 arma::cube cov_inv);

  // Posterior kernels for parameters
  virtual double mLogKernel(arma::uword b, arma::vec m_b, arma::mat mean_sum);
  virtual double sLogKernel(arma::uword b,
                            arma::vec S_b,
                            arma::vec cov_comb_log_det,
                            arma::cube cov_comb_inv);

  virtual double muLogKernel(arma::uword k, arma::vec mu_k, arma::mat mean_sum);
  virtual double covLogKernel(arma::uword k,
                              arma::mat cov_k,
                              double cov_log_det,
                              arma::mat cov_inv,
                              arma::vec cov_comb_log_det,
                              arma::cube cov_comb_inv);

  // Metropolis steps
  virtual void batchScaleMetropolis();
  virtual void batchShiftMetorpolis();
  virtual void clusterCovarianceMetropolis();
  virtual void clusterMeanMetropolis();
  virtual void metropolisStep();

  // Infer batch corrected data
  virtual void updateBatchCorrectedData();

  // Check matrices are positive semi-definite
  void checkPositiveDefinite(arma::uword r);
};

#endif /* MVNMIXTURE_H */