// mvtMixture.cpp
// =============================================================================
// included dependencies
# include <RcppArmadillo.h>

# include "mvtMixture.h"
# include "pdfs.h"

// Choice of namespace
using namespace Rcpp ;
using namespace arma ;

// =============================================================================
// mvtMixture functions

mvtMixture::mvtMixture(                           
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
  ) : mixture(_K,
  _B,
  _labels,
  _batch_vec,
  _X), mvnMixture(
      _K,
      _B,
      _mu_proposal_window,
      _cov_proposal_window,
      _m_proposal_window,
      _S_proposal_window,
      _labels,
      _batch_vec,
      _X
  ) {

  t_df.set_size(K);
  t_df.zeros();
  
  pdf_coef.set_size(K);
  pdf_coef.zeros();
  
  t_df_count.set_size(K);
  t_df_count.zeros();
  
  // The proposal windows for the cluster degrees of freedom
  t_df_proposal_window = _t_df_proposal_window;
};

// Print statement (useful for checks)
void mvtMixture::printType() {
  std::cout << "\nType: Multivariate T.\n";
};
  
double mvtMixture::calcPDFCoef(double t_df){
  return std::lgamma(0.5 * (t_df + P)) - std::lgamma(0.5 * t_df) - 0.5 * P * log(t_df);
};
  
void mvtMixture::sampleDFPrior() {
    for(arma::uword k = 0; k < K; k++){
      // Draw from a shifted gamma distribution (i.e. gamma with location parameter)
      t_df(k) = t_loc + arma::randg<double>( arma::distr_param(psi, 1.0 / chi));
    }
  };
  
void mvtMixture::sampleFromPriors() {
  sampleCovPrior();
  sampleMuPrior();
  sampleDFPrior();
  sampleSPrior();
  sampleMPrior();
};
  

// Update the common matrix manipulations to avoid recalculating N times
void mvtMixture::matrixCombinations() {
  
  for(arma::uword k = 0; k < K; k++) {
    pdf_coef(k) = calcPDFCoef(t_df(k));
    cov_inv.slice(k) = arma::inv_sympd(cov.slice(k));
    cov_log_det(k) = arma::log_det(cov.slice(k)).real();
    
    for(arma::uword b = 0; b < B; b++) {
      cov_comb.slice(k * B + b) = cov.slice(k); // + arma::diagmat(S.col(b))
      for(arma::uword p = 0; p < P; p++) {
        cov_comb.slice(k * B + b)(p, p) *= S(p, b);
      }
      cov_comb_log_det(k, b) = arma::log_det(cov_comb.slice(k * B + b)).real();
      cov_comb_inv.slice(k * B + b) = arma::inv_sympd(cov_comb.slice(k * B + b));
      
      mean_sum.col(k * B + b) = mu.col(k) + m.col(b);
    }
  }
};
  
  
// The log likelihood of a item belonging to each cluster given the batch label.
arma::vec mvtMixture::itemLogLikelihood(arma::vec item, arma::uword b) {
  
  double x = 0.0, y = 0.0;
  arma::vec ll(K), dist_to_mean(P);
  ll.zeros();
  dist_to_mean.zeros();
  
  for(arma::uword k = 0; k < K; k++){
  
    // The different compontents of the likelihood
    dist_to_mean = item - mean_sum.col(k * B + b);
    x = arma::as_scalar(dist_to_mean.t() * cov_comb_inv.slice(k * B + b) * dist_to_mean);
    y = (t_df(k) + P) * log(1.0 + (1/t_df(k)) * x);
    
    ll(k) = pdf_coef(k) - 0.5 * (cov_comb_log_det(k, b) + y + P * log(PI));
  }
  
  return(ll);
};

void mvtMixture::calcBIC(){
  
  // Each component has a weight, a mean vector, a symmetric covariance matrix and a
  // degree of freedom parameter. Each batch has a mean and standard
  // deviations vector.
  // arma::uword n_param = (P + P * (P + 1) * 0.5 + 1) * K_occ + (2 * P) * B;
  // BIC = n_param * std::log(N) - 2 * observed_likelihood;
  
  // arma::uword n_param_cluster = 2 + P + P * (P + 1) * 0.5;
  // arma::uword n_param_batch = 2 * P;
  
  BIC = 2 * observed_likelihood - (n_param_batch + n_param_batch) * std::log(N);
  
  // for(arma::uword k = 0; k < K; k++) {
  //   BIC -= n_param_cluster * std::log(N_k(k)+ 1);
  // }
  // for(arma::uword b = 0; b < B; b++) {
  //   BIC -= n_param_batch * std::log(N_b(b)+ 1);
  // }
  
};
  
double mvtMixture::clusterLikelihood(
    double t_df,
    arma::uvec cluster_ind,
    arma::vec cov_det,
    arma::mat mean_sum,
    arma::cube cov_inv
) {
  
  arma::uword b = 0;
  double score = 0.0;
  arma::vec dist_from_mean(P);
  
  for (auto& n : cluster_ind) {
    b = batch_vec(n);
    dist_from_mean = X_t.col(n) - mean_sum.col(b);
    score += cov_det(b) + (t_df + P) * log(1 + (1/t_df) * arma::as_scalar(dist_from_mean.t() * cov_inv.slice(b) * dist_from_mean));
  }
  return (-0.5 * score);
}
  
double mvtMixture::batchLikelihood(
    arma::uvec batch_inds,
    arma::uvec labels,
    arma::vec cov_det,
    arma::vec t_df,
    arma::mat mean_sum,
    arma::cube cov_inv){
  
  arma::uword k = 0;
  double score = 0.0;
  arma::vec dist_from_mean(P);
  
  for (auto& n : batch_inds) {
    k = labels(n);
    dist_from_mean = X_t.col(n) - mean_sum.col(k);
    score += cov_det(k) + (t_df(k) + P) * log(1 + (1/t_df(k)) * arma::as_scalar(dist_from_mean.t() * cov_inv.slice(k) * dist_from_mean));
  }
  return (-0.5 * score);
}
  
double mvtMixture::mLogKernel(arma::uword b, arma::vec m_b, arma::mat mean_sum) {
  
  double score = 0.0;
  score = batchLikelihood(batch_ind(b), 
    labels, 
    cov_comb_log_det.col(b),
    t_df,
    mean_sum,
    cov_comb_inv.slices(KB_inds + b)
  );
  
  for(arma::uword p = 0; p < P; p++) {
    score += -0.5 * t * std::pow(m_b(p) - delta, 2.0);
  }
  
  return score;
};

double mvtMixture::sLogKernel(arma::uword b,
                  arma::vec S_b,
                  arma::vec cov_comb_log_det,
                  arma::cube cov_comb_inv) {
  
  double score = 0.0;
  
  score = batchLikelihood(batch_ind(b), 
    labels, 
    cov_comb_log_det,
    t_df,
    mean_sum.cols(KB_inds + b),
    cov_comb_inv
  );
  
  for(arma::uword p = 0; p < P; p++) {
    score += -((rho + 1) * std::log(S_b(p) - S_loc) + theta / (S_b(p) - S_loc));
  }
  
  return score;
};

double mvtMixture::muLogKernel(arma::uword k, arma::vec mu_k, arma::mat mean_sum) {
  
  double score = 0.0;
  arma::uvec cluster_ind = arma::find(labels == k);
  
  score = clusterLikelihood(
    t_df(k),
    cluster_ind,
    cov_comb_log_det.row(k).t(),
    mean_sum,
    cov_comb_inv.slices(k * B + B_inds)
  );
  
  score += -0.5 * arma::as_scalar(kappa * ((mu_k - xi).t() *  cov_inv.slice(k) * (mu_k - xi)));
  
  return score;
};

double mvtMixture::covLogKernel(arma::uword k, 
                    arma::mat cov_k,
                    double cov_log_det,
                    arma::mat cov_inv,
                    arma::vec cov_comb_log_det,
                    arma::cube cov_comb_inv) {
  
  double score = 0.0;
  arma::uvec cluster_ind = arma::find(labels == k);
  
  score = clusterLikelihood(
    t_df(k),
    cluster_ind,
    cov_comb_log_det,
    mean_sum.cols(k * B + B_inds),
    cov_comb_inv
  );
  
  score += -0.5 *( arma::as_scalar((nu + P + 2) * cov_log_det 
                                     + kappa * ((mu.col(k) - xi).t() * cov_inv * (mu.col(k) - xi)) 
                                     + arma::trace(scale * cov_inv)));
                                     
  return score;
};

double mvtMixture::dfLogKernel(arma::uword k, 
                   double t_df,
                   double pdf_coef) {
  
  arma::uword b = 0;
  double score = 0.0;
  arma::uvec cluster_ind = arma::find(labels == k);
  arma::vec dist_from_mean(P);
  
  for (auto& n : cluster_ind) {
    b = batch_vec(n);
    dist_from_mean = X_t.col(n) - mean_sum.col(k * B + b);
    score += pdf_coef - 0.5 * (t_df + P) * log(1 + (1/t_df) * arma::as_scalar(dist_from_mean.t() * cov_comb_inv.slice(k * B + b) * dist_from_mean));
  }
  score += (psi - 1) * log(t_df - t_loc) - chi * (t_df - t_loc);
  return score;
};

void mvtMixture::clusterDFMetropolis() {
  double u = 0.0,
    proposed_model_score = 0.0, 
    acceptance_prob = 0.0, 
    current_model_score = 0.0, 
    t_df_proposed = 0.0, 
    proposed_pdf_coef = 0.0;
  
  for(arma::uword k = 0; k < K ; k++) {
    
    proposed_model_score = 0.0, 
      acceptance_prob = 0.0, 
      current_model_score = 0.0, 
      t_df_proposed = 0.0, 
      proposed_pdf_coef = 0.0;
    
    if(N_k(k) == 0){
      t_df_proposed = t_loc + arma::randg<double>( arma::distr_param(psi, 1.0 / chi));
      proposed_pdf_coef = calcPDFCoef(t_df_proposed);
    } else {
      
      // Proposed value
      t_df_proposed = t_loc + arma::randg( arma::distr_param( (t_df(k) - t_loc) * t_df_proposal_window, 1.0 / t_df_proposal_window) );
      
      proposed_pdf_coef = calcPDFCoef(t_df_proposed);
      
      // Asymmetric proposal density
      proposed_model_score = gammaLogLikelihood(t_df(k) - t_loc, (t_df_proposed - t_loc) * t_df_proposal_window, t_df_proposal_window);
      current_model_score = gammaLogLikelihood(t_df_proposed - t_loc, (t_df(k) - t_loc) * t_df_proposal_window, t_df_proposal_window);
      
      // The prior is included in the kernel
      proposed_model_score = dfLogKernel(k, t_df_proposed, proposed_pdf_coef);
      current_model_score = dfLogKernel(k, t_df(k), pdf_coef(k));
      
      // Check if we accept
      u = arma::randu();
      acceptance_prob = std::min(1.0, std::exp(proposed_model_score - current_model_score));
      
    }
    
    if((u < acceptance_prob) || (N_k(k) == 0)) {
      t_df(k) = t_df_proposed;
      t_df_count(k)++;
      pdf_coef(k) = proposed_pdf_coef;
    }
  }
}

void mvtMixture::metropolisStep() {
  
  // Metropolis step for cluster parameters
  clusterCovarianceMetropolis();
  clusterMeanMetropolis();
  clusterDFMetropolis();
  
  // Metropolis step for batch parameters if more than 1 batch
  batchScaleMetropolis();
  batchShiftMetorpolis();
};
