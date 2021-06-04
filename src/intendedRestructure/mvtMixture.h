// mixture.h
// =============================================================================
// include guard
#ifndef MVTMIXTURE_H
#define MVTMIXTURE_H

// =============================================================================
// included dependencies
# include "mixture.h"
# include "mvnMixture.h"

// =============================================================================
// mvtMixture class
class mvtMixture: virtual public mvnMixture {
  
public:
  
  arma::uword n_param_cluster = 2 + P + P * (P + 1) * 0.5, n_param_batch = 2 * P;
  
  // t degree of freedom hyperparameters (decision from
  // https://statmodeling.stat.columbia.edu/2015/05/17/do-we-have-any-recommendations-for-priors-for-student_ts-degrees-of-freedom-parameter/)
  // This gives a very wide range and a support of [2.0, infty).
  double psi = 2.0, 
    chi = 0.01, 
    t_loc = 2.0,
    
    // Our proposal window
    t_df_proposal_window = 0.0, 
    
    // A value in the pdf defined by the degrees of freedom which we save to // [[Rcpp::export]]
    // avoid recomputing
    pdf_const = 0.0;
  
  arma::uvec t_df_count;
  arma::vec t_df, pdf_coef;
  
  // Use the mxnMixture constructor
  using mvnMixture::mvnMixture;
  
  // Our paramtrised constructor
  mvtMixture(                           
    arma::uword _K,
    arma::uword _B,
    double _mu_proposal_window,
    double _cov_proposal_window,
    double _m_proposal_window,
    double _S_proposal_window,
    double _t_df_proposal_window,
    arma::uvec _labels,
    arma::uvec _batch_vec,
    arma::mat _X
  );
  
  
  // Destructor
  virtual ~mvtMixture() { };
  
  // Print the sampler type.
  virtual void printType();
  
  double calcPDFCoef(double t_df);
  
  virtual void sampleDFPrior();
  
  virtual void sampleFromPriors();
  
  // Update the common matrix manipulations to avoid recalculating N times
  virtual void matrixCombinations();
  
  // The log likelihood of a item belonging to each cluster given the batch label.
  arma::vec itemLogLikelihood(arma::vec item, arma::uword b);
  
  void calcBIC();
  
  double clusterLikelihood(
      double t_df,
      arma::uvec cluster_ind,
      arma::vec cov_det,
      arma::mat mean_sum,
      arma::cube cov_inv
  );
  
  double batchLikelihood(
      arma::uvec batch_inds,
      arma::uvec labels,
      arma::vec cov_det,
      arma::vec t_df,
      arma::mat mean_sum,
      arma::cube cov_inv);
  
  double mLogKernel(arma::uword b, arma::vec m_b, arma::mat mean_sum);
  
  double sLogKernel(arma::uword b,
                    arma::vec S_b,
                    arma::vec cov_comb_log_det,
                    arma::cube cov_comb_inv);
  
  double muLogKernel(arma::uword k, arma::vec mu_k, arma::mat mean_sum);
  
  double covLogKernel(arma::uword k, 
                      arma::mat cov_k,
                      double cov_log_det,
                      arma::mat cov_inv,
                      arma::vec cov_comb_log_det,
                      arma::cube cov_comb_inv);
  
  double dfLogKernel(arma::uword k, 
                     double t_df,
                     double pdf_coef);
  
  void clusterDFMetropolis();
  
  virtual void metropolisStep();
  
};

#endif /* MVTMIXTURE_H */