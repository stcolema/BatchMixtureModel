// # include <RcppArmadillo.h>
// // # include <math.h> 
// // # include <string>
// # include <iostream>
// 
// // Header file containing generic pdfs (not needed here)
// // # include "pdfs.h"
// 
// // Different sampler types
// // # include "sampler.h"
// // # include "semisupervisedSampler.h"
// # include "mvnSampler.h"
// # include "msnSampler.h"
// # include "mvtSampler.h"
// 
// // Different semi-supervised samplers
// # include "mvnPredictive.h"
// # include "msnPredictive.h"
// # include "mvtPredictive.h"
// 
// // [[Rcpp::depends(RcppArmadillo)]]
// 
// using namespace Rcpp ;
// using namespace arma ;  
// 
// 
// // 
// // class mvtSampler: virtual public mvnSampler {
// //   
// // public:
// // 
// //   // arma::uword t_df = 4;
// //   arma::uword n_param_cluster = 2 + P + P * (P + 1) * 0.5, n_param_batch = 2 * P;
// //   
// //   // t degree of freedom hyperparameters (decision from
// //   // https://statmodeling.stat.columbia.edu/2015/05/17/do-we-have-any-recommendations-for-priors-for-student_ts-degrees-of-freedom-parameter/)
// //   // This gives a very wide range and a support of [2.0, infty).
// //   double psi = 2.0, 
// //     chi = 0.01, 
// //     t_loc = 2.0,
// //     
// //     // Our proposal window
// //     t_df_proposal_window = 0.0, 
// //     
// //     // A value in the pdf defined by the degrees of freedom which we save to // [[Rcpp::export]]
// //     // avoid recomputing
// //     pdf_const = 0.0;
// //   
// //   arma::uvec t_df_count;
// //   arma::vec t_df, pdf_coef;
// //   
// //   
// //   using mvnSampler::mvnSampler;
// //   
// //   mvtSampler(                           
// //     arma::uword _K,
// //     arma::uword _B,
// //     double _mu_proposal_window,
// //     double _cov_proposal_window,
// //     double _m_proposal_window,
// //     double _S_proposal_window,
// //     double _t_df_proposal_window,
// //     // double _rho,
// //     // double _theta,
// //     arma::uvec _labels,
// //     arma::uvec _batch_vec,
// //     arma::vec _concentration,
// //     arma::mat _X
// //   ) : sampler(_K,
// //     _B,
// //     _labels,
// //     _batch_vec,
// //     _concentration,
// //     _X), 
// //   mvnSampler(                           
// //     _K,
// //     _B,
// //     _mu_proposal_window,
// //     _cov_proposal_window,
// //     _m_proposal_window,
// //     _S_proposal_window,
// //     _labels,
// //     _batch_vec,
// //     _concentration,
// //     _X
// //   ) {
// //     
// //     // Hyperparameter for the d.o.f for the t-distn
// //     // psi = 0.5;
// //     // chi = 0.5;
// //     
// //     t_df.set_size(K);
// //     t_df.zeros();
// //     
// //     pdf_coef.set_size(K);
// //     pdf_coef.zeros();
// //     
// //     t_df_count.set_size(K);
// //     t_df_count.zeros();
// //     
// //     // The shape of the skew normal
// //     // phi.set_size(P, K);
// //     // phi.zeros();
// //     
// //     // Count the number of times proposed values are accepted
// //     // phi_count = arma::zeros<arma::uvec>(K);
// //     
// //     // The proposal windows for the cluster and batch parameters
// //     t_df_proposal_window = _t_df_proposal_window;
// //     
// //     // The constant for the item likelihood (changes if t_df != const)
// //     // pdf_const = std::lgamma(0.5 * (t_df + P)) - std::lgamma(0.5 * t_df) - 0.5 * P * log(t_df);
// //   };
// //   
// //   
// //   // Destructor
// //   virtual ~mvtSampler() { };
// //   
// //   // Print the sampler type.
// //   virtual void printType() {
// //     std::cout << "\nType: Multivariate T.\n";
// //   };
// //   
// //   double calcPDFCoef(double t_df){
// //     double x = std::lgamma(0.5 * (t_df + P)) - std::lgamma(0.5 * t_df) - 0.5 * P * log(t_df);
// //     return x;
// //   };
// //   
// //   virtual void sampleDFPrior() {
// //     for(arma::uword k = 0; k < K; k++){
// //       // Draw from a shifted gamma distribution (i.e. gamma with location parameter)
// //       t_df(k) = t_loc + arma::randg<double>( arma::distr_param(psi, 1.0 / chi));
// //     }
// //   };
// //   
// //   virtual void sampleFromPriors() {
// //     
// //     sampleCovPrior();
// //     sampleMuPrior();
// //     sampleDFPrior();
// //     sampleSPrior();
// //     sampleMPrior();
// //   };
// //   
// //   // virtual void sampleFromPriors() {
// //   //   
// //   //   for(arma::uword k = 0; k < K; k++){
// //   //     cov.slice(k) = arma::iwishrnd(scale, nu);
// //   //     mu.col(k) = arma::mvnrnd(xi, (1.0/kappa) * cov.slice(k), 1);
// //   //     
// //   //     // Draw from a shifted gamma distribution (i.e. gamma with location parameter)
// //   //     t_df(k) = t_loc + arma::randg<double>( arma::distr_param(psi, 1.0 / chi));
// //   //   }
// //   //   for(arma::uword b = 0; b < B; b++){
// //   //     for(arma::uword p = 0; p < P; p++){
// //   //       S(p, b) = S_loc + 1.0 / arma::randg<double>( arma::distr_param(rho, 1.0 / theta ) );
// //   //       m(p, b) = arma::randn<double>() * S(p, b) / lambda + delta(p);
// //   //     }
// //   //   }
// //   // };
// //   
// //   // Update the common matrix manipulations to avoid recalculating N times
// //   virtual void matrixCombinations() {
// //     
// //     for(arma::uword k = 0; k < K; k++) {
// //       pdf_coef(k) = calcPDFCoef(t_df(k));
// //       cov_inv.slice(k) = arma::inv_sympd(cov.slice(k));
// //       cov_log_det(k) = arma::log_det(cov.slice(k)).real();
// //       
// //       for(arma::uword b = 0; b < B; b++) {
// //         cov_comb.slice(k * B + b) = cov.slice(k); // + arma::diagmat(S.col(b))
// //         for(arma::uword p = 0; p < P; p++) {
// //           cov_comb.slice(k * B + b)(p, p) *= S(p, b);
// //         }
// //         cov_comb_log_det(k, b) = arma::log_det(cov_comb.slice(k * B + b)).real();
// //         cov_comb_inv.slice(k * B + b) = arma::inv_sympd(cov_comb.slice(k * B + b));
// //  
// //         mean_sum.col(k * B + b) = mu.col(k) + m.col(b);
// //       }
// //     }
// //   };
// //     
// //   
// //   // The log likelihood of a item belonging to each cluster given the batch label.
// //   arma::vec itemLogLikelihood(arma::vec item, arma::uword b) {
// //     
// //     double x = 0.0, y = 0.0, my_det = 0.0;
// //     arma::vec ll(K), dist_to_mean(P);
// //     ll.zeros();
// //     dist_to_mean.zeros();
// //     arma::mat my_cov_comv_inv(P, P), my_inv(P, P), my_cov_comb(P, P);
// //     my_cov_comv_inv.zeros();
// //     my_inv.zeros();
// //     my_cov_comb.zeros();
// //     
// //     double cov_correction = 0.0;
// //     
// //     for(arma::uword k = 0; k < K; k++){
// //     
// //       // gamma(0.5 * (nu + P)) / (gamma(0.5 * nu) * nu ^ (0.5 * P) * pi ^ (0.5 * P)  * det(cov) ^ 0.5) * (1 + (1 / nu) * (x - mu)^t * inv(cov) * (x - mu)) ^ (-0.5 * (nu + P))
// //       // std::lgamma(0.5 * (nu + P)) - std::lgamma(0.5 * nu) - (0.5 * P) * log(nu) - 0.5 * P * log(pi) - 0.5 * logDet(cov) -0.5 * (nu + P) * log(1 + (1 / nu) * (x - mu)^t * inv(cov) * (x - mu))
// //       
// //       // my_cov_comv_inv = cov.slice(k);
// //       // for(arma::uword p = 0; p < P; p++) {
// //       //   my_cov_comv_inv(p, p) *= S(p, b);
// //       // }
// //       
// //       // cov_correction = t_df(k) / (t_df(k) - 2.0);
// //       
// //       
// //       // my_cov_comb = cov.slice(k);
// //       // 
// //       // for(arma::uword p = 0; p < P; p++) {
// //       //   my_cov_comb(p, p) = my_cov_comb(p, p) * S(p, b);
// //       // }
// //       // 
// //       // // my_cov_comb = my_cov_comb / cov_correction;
// //       // 
// //       // // std::cout << "\nThe invariance.";
// //       // 
// //       // my_inv = arma::inv_sympd(my_cov_comb);
// //       // 
// //       // // std::cout << "\nDeterminant.";
// //       // my_det = arma::log_det(my_cov_comb).real();
// //       
// //       // The exponent part of the MVN pdf
// //       dist_to_mean = item - mean_sum.col(k * B + b);
// //       x = arma::as_scalar(dist_to_mean.t() * cov_comb_inv.slice(k * B + b) * dist_to_mean);
// //       // x = arma::as_scalar(dist_to_mean.t() * my_inv * dist_to_mean);
// //       y = (t_df(k) + P) * log(1.0 + (1/t_df(k)) * x);
// //       
// //       ll(k) = pdf_coef(k) - 0.5 * (cov_comb_log_det(k, b) + y + P * log(PI));
// //       // ll(k) = pdf_coef(k) - 0.5 * (my_det + y + P * log(PI)); 
// //       
// //       // std::cout << "\nCheck.";
// //       
// //       // if(! arma::approx_equal(mean_sum.col(k * B + b), (mu.col(k) + m.col(b)), "absdiff", 0.001)) {
// //       //   std::cout << "\n\nMean sum has deviated from expected.";
// //       // }
// //       // 
// //       // if(! arma::approx_equal(cov_comb_inv.slice(k * B + b), my_inv, "absdiff", 0.001)) {
// //       //   std::cout << "\n\nCovariance inverse has deviated from expected.";
// //       //   std::cout << "\n\nExpected:\n" << cov_comb_inv.slice(k * B + b) <<
// //       //     "\n\nCalculated:\n" << my_inv;
// //       // 
// //       //   throw std::invalid_argument( "\nMy inverses diverged." );
// //       // }
// //       // 
// //       // if(isnan(ll(k))) {
// //       //   std::cout << "\nNaN!\n";
// //       //   
// //       //   double new_x = (1/t_df(k)) * arma::as_scalar((item - mu.col(k) - m.col(b)).t() * my_inv * (item - mu.col(k) - m.col(b)));
// //       //   
// //       //   std::cout << "\n\nItem likelihood:\n" << ll(k) << 
// //       //     "\nPDF coefficient: " << pdf_coef(k) << "\nLog determinant: " <<
// //       //       cov_comb_log_det(k, b) << "\nX: " << x << "\nY: " << y <<
// //       //         "\nLog comp of y: " << 1.0 + (1/t_df(k)) * x <<
// //       //           "\nLogged: " << log(1.0 + (1/t_df(k)) * x) <<
// //       //             "\nt_df(k): " << t_df(k) << "\n" << 
// //       //               "\nMy new x" << new_x << "\nLL alt: " << 
// //       //                 pdf_coef(k) - 0.5 * (my_det + (t_df(k) + P) * log(1.0 + new_x) + P * log(PI)) <<
// //       //                   "\n\nCov combined expected:\n" << cov_comb_inv.slice(k * B + b) <<
// //       //                     "\n\nCov combined real:\n" << my_inv;
// //       //                 
// //       //   throw std::invalid_argument( "\nNaN returned from likelihood." );
// //       //   
// //       // }
// //       
// //       
// //     }
// //     
// //     return(ll);
// //   };
// //   
// //   void calcBIC(){
// //     
// //     // Each component has a weight, a mean vector, a symmetric covariance matrix and a
// //     // degree of freedom parameter. Each batch has a mean and standard
// //     // deviations vector.
// //     // arma::uword n_param = (P + P * (P + 1) * 0.5 + 1) * K_occ + (2 * P) * B;
// //     // BIC = n_param * std::log(N) - 2 * observed_likelihood;
// //     
// //     // arma::uword n_param_cluster = 2 + P + P * (P + 1) * 0.5;
// //     // arma::uword n_param_batch = 2 * P;
// // 
// //     BIC = 2 * observed_likelihood - (n_param_batch + n_param_batch) * std::log(N);
// //     
// //     // for(arma::uword k = 0; k < K; k++) {
// //     //   BIC -= n_param_cluster * std::log(N_k(k)+ 1);
// //     // }
// //     // for(arma::uword b = 0; b < B; b++) {
// //     //   BIC -= n_param_batch * std::log(N_b(b)+ 1);
// //     // }
// //     
// //   };
// //   
// //   double clusterLikelihood(
// //       double t_df,
// //       arma::uvec cluster_ind,
// //       arma::vec cov_det,
// //       arma::mat mean_sum,
// //       arma::cube cov_inv
// //     ) {
// //     
// //     arma::uword b = 0;
// //     double score = 0.0;
// //     arma::vec dist_from_mean(P);
// //     
// //     for (auto& n : cluster_ind) {
// //       b = batch_vec(n);
// //       dist_from_mean = X_t.col(n) - mean_sum.col(b);
// //       score += cov_det(b) + (t_df + P) * log(1 + (1/t_df) * arma::as_scalar(dist_from_mean.t() * cov_inv.slice(b) * dist_from_mean));
// //     }
// //     // 
// //     // std::cout << "\nScore before halving: " << score << "\nT DF: " << t_df <<
// //     //   "\n\nCov log det:\n" << cov_det << "\n\nCov inverse:\n " << cov_inv;
// //     // 
// //     // 
// //     
// //     return (-0.5 * score);
// //   }
// //   
// //   double batchLikelihood(
// //       arma::uvec batch_inds,
// //       arma::uvec labels,
// //       arma::vec cov_det,
// //       arma::vec t_df,
// //       arma::mat mean_sum,
// //       arma::cube cov_inv){
// //     
// //     arma::uword k = 0;
// //     double score = 0.0;
// //     arma::vec dist_from_mean(P);
// //     
// //     for (auto& n : batch_inds) {
// //       k = labels(n);
// //       dist_from_mean = X_t.col(n) - mean_sum.col(k);
// //       score += cov_det(k) + (t_df(k) + P) * log(1 + (1/t_df(k)) * arma::as_scalar(dist_from_mean.t() * cov_inv.slice(k) * dist_from_mean));
// //     }
// //     return (-0.5 * score);
// //   }
// // 
// //   double mLogKernel(arma::uword b, arma::vec m_b, arma::mat mean_sum) {
// // 
// //     arma::uword k = 0;
// //     double score = 0.0, score_alt = 0.0;
// //     arma::vec dist_from_mean(P);
// //     dist_from_mean.zeros();
// //     
// //     score = batchLikelihood(batch_ind(b), 
// //       labels, 
// //       cov_comb_log_det.col(b),
// //       t_df,
// //       mean_sum,
// //       cov_comb_inv.slices(KB_inds + b)
// //     );
// //     
// //     // for (auto& n : batch_ind(b)) {
// //     //   k = labels(n);
// //     //   dist_from_mean = X_t.col(n) - mean_sum.col(k);
// //     //   score_alt += cov_comb_log_det(k, b) + (t_df(k) + P) * log(1 + (1/t_df(k)) * arma::as_scalar(dist_from_mean.t() * cov_comb_inv.slice(k * B + b) * dist_from_mean));
// //     // }
// //     // 
// //     // score_alt *= -0.5;
// //     // 
// //     // if(std::abs(score_alt - score) > 1e-6) {
// //     //   std::cout << "\nProblem in m kernel function.\nOld score: " << 
// //     //     score << "\nAlternative score: " << score_alt;
// //     //   throw std::invalid_argument( "\n" );
// //     // }
// //     
// //     for(arma::uword p = 0; p < P; p++) {
// //       score += -0.5 * t * std::pow(m_b(p) - delta, 2.0);
// //     }
// //     
// //     // score *= -0.5;
// //     
// //     return score;
// //   };
// // 
// //   double sLogKernel(arma::uword b,
// //                     arma::vec S_b,
// //                     arma::vec cov_comb_log_det,
// //                     arma::cube cov_comb_inv) {
// // 
// //     arma::uword k = 0;
// //     double score = 0.0, score_alt = 0.0;
// //     arma::vec dist_from_mean(P);
// //     dist_from_mean.zeros();
// //     arma::mat curr_sum(P, P);
// // 
// //     score = batchLikelihood(batch_ind(b), 
// //       labels, 
// //       cov_comb_log_det,
// //       t_df,
// //       mean_sum.cols(KB_inds + b),
// //       cov_comb_inv
// //     );
// //     
// //     // for (auto& n : batch_ind(b)) {
// //     //   k = labels(n);
// //     //   dist_from_mean = X_t.col(n) - mean_sum.col(k * B + b);
// //     //   score_alt += (cov_comb_log_det(k) + (t_df(k) + P) * log(1 + (1/t_df(k)) * arma::as_scalar(dist_from_mean.t() * cov_comb_inv.slice(k) * dist_from_mean)));
// //     // }
// //     // 
// //     // score_alt *= -0.5;
// //     // 
// //     // if(std::abs(score_alt - score) > 1e-6) {
// //     //   std::cout << "\nProblem in S kernel function.\nOld score: " << 
// //     //     score << "\nAlternative score: " << score_alt;
// //     //   throw std::invalid_argument( "\n" );
// //     // }
// //     
// //     for(arma::uword p = 0; p < P; p++) {
// //       score += -((rho + 1) * std::log(S_b(p) - S_loc) + theta / (S_b(p) - S_loc));
// // 
// //      // score +=   (0.5 - 1) * std::log(S(p,b) - S_loc) 
// //      //   - 0.5 * (S(p, b) - S_loc)
// //      //   - 0.5 * lambda * std::pow(m(p,b) - delta(p), 2.0) / S_b(p);
// //     }
// //     
// //     // score *= -0.5;
// //     return score;
// //   };
// // 
// //   double muLogKernel(arma::uword k, arma::vec mu_k, arma::mat mean_sum) {
// // 
// //     arma::uword b = 0;
// //     double score = 0.0, score_alt = 0.0;
// //     arma::uvec cluster_ind = arma::find(labels == k);
// //     arma::vec dist_from_mean(P);
// // 
// //     score = clusterLikelihood(
// //       t_df(k),
// //       cluster_ind,
// //       cov_comb_log_det.row(k).t(),
// //       mean_sum,
// //       cov_comb_inv.slices(k * B + B_inds)
// //     );
// //     
// //     // for (auto& n : cluster_ind) {
// //     //   b = batch_vec(n);
// //     //   dist_from_mean = X_t.col(n) - mean_sum.col(b);
// //     //   score_alt += cov_comb_log_det(k, b) +  (t_df(k) + P) * log(1 + (1/t_df(k)) * arma::as_scalar(dist_from_mean.t() * cov_comb_inv.slice(k * B + b) * dist_from_mean));
// //     // }
// //     // 
// //     // score_alt *= -0.5;
// //     // 
// //     // if(std::abs(score_alt - score) > 1e-6) {
// //     //   std::cout << "\nProblem in mu kernel function.\nOld score: " << 
// //     //     score << "\nAlternative score: " << score_alt;
// //     //   throw std::invalid_argument( "\n" );
// //     // }
// //     
// //     score += -0.5 * arma::as_scalar(kappa * ((mu_k - xi).t() *  cov_inv.slice(k) * (mu_k - xi)));
// //     // score *= -0.5;
// //     
// //     return score;
// //   };
// // 
// //   double covLogKernel(arma::uword k, 
// //                       arma::mat cov_k,
// //                       double cov_log_det,
// //                       arma::mat cov_inv,
// //                       arma::vec cov_comb_log_det,
// //                       arma::cube cov_comb_inv) {
// // 
// //     arma::uword b = 0;
// //     double score = 0.0, score_alt = 0.0;
// //     arma::uvec cluster_ind = arma::find(labels == k);
// //     arma::vec dist_from_mean(P);
// //     
// //     score = clusterLikelihood(
// //       t_df(k),
// //       cluster_ind,
// //       cov_comb_log_det,
// //       mean_sum.cols(k * B + B_inds),
// //       cov_comb_inv
// //     );
// // 
// //     
// //     // for (auto& n : cluster_ind) {
// //     //   b = batch_vec(n);
// //     //   dist_from_mean = X_t.col(n) - mean_sum.col(k * B + b);
// //     //   score_alt += cov_comb_log_det(b) + (t_df(k) + P) * log(1 + (1/t_df(k)) * arma::as_scalar(dist_from_mean.t() * cov_comb_inv.slice(b) * dist_from_mean));
// //     // }
// //     // 
// //     // score_alt *= -0.5;
// //     // 
// //     // if(std::abs(score_alt - score) > 1e-6) {
// //     //   std::cout << "\nProblem in cov kernel function.\nOld score: " << 
// //     //     score << "\nAlternative score: " << score_alt;
// //     //   
// //     //   std::cout << "\nT DF: " << t_df(k) << "\n\nCov log det:\n" << cov_comb_log_det <<
// //     //     "\n\nCov inverse:\n " << cov_comb_inv;
// //     //   
// //     //   // std::cout << "\n\nMean sums:\n";
// //     //   // for(arma::uword b = 0;b < B; b++){
// //     //   //   std::cout << mean_sum.col(k * B + b) << "\n\n";
// //     //   // }
// //     //   // std::cout << "\n\nMean sums:\n" << mean_sum.cols(k * B + B_inds);
// //     //   throw std::invalid_argument( "\n" );
// //     // }
// //     
// //     score += -0.5 *( arma::as_scalar((nu + P + 2) * cov_log_det 
// //                     + kappa * ((mu.col(k) - xi).t() * cov_inv * (mu.col(k) - xi)) 
// //                     + arma::trace(scale * cov_inv)));
// //     // score *= -0.5;
// //     
// //     return score;
// //   };
// //   
// //   double dfLogKernel(arma::uword k, 
// //                      double t_df,
// //                      double pdf_coef) {
// //     
// //     arma::uword b = 0;
// //     double score = 0.0;
// //     arma::uvec cluster_ind = arma::find(labels == k);
// //     arma::vec dist_from_mean(P);
// //     for (auto& n : cluster_ind) {
// //       b = batch_vec(n);
// //       dist_from_mean = X_t.col(n) - mean_sum.col(k * B + b);
// //       score += pdf_coef - 0.5 * (t_df + P) * log(1 + (1/t_df) * arma::as_scalar(dist_from_mean.t() * cov_comb_inv.slice(k * B + b) * dist_from_mean));
// //     }
// //     // score += (psi - 1) * log(t_df - t_loc) - (t_df - t_loc) / chi;
// //     score += (psi - 1) * log(t_df - t_loc) - chi * (t_df - t_loc);
// //     return score;
// //   };
// //   
// //   void clusterDFMetropolis() {
// //     double u = 0.0, proposed_model_score = 0.0, acceptance_prob = 0.0, current_model_score = 0.0, t_df_proposed = 0.0, proposed_pdf_coef = 0.0;
// //     
// //     for(arma::uword k = 0; k < K ; k++) {
// //       proposed_model_score = 0.0, acceptance_prob = 0.0, current_model_score = 0.0, t_df_proposed = 0.0, proposed_pdf_coef = 0.0;
// //       if(N_k(k) == 0){
// //         t_df_proposed = t_loc + arma::randg<double>( arma::distr_param(psi, 1.0 / chi));
// //         proposed_pdf_coef = calcPDFCoef(t_df_proposed);
// //       } else {
// //         
// //         // std::cout << "\n\nT df.\nPsi: " << psi << "\nChi: " << chi
// //         // << "\nWindow: " << t_df_proposal_window << "\nCurrent: " << t_df(k);
// //         
// //         t_df_proposed = t_loc + arma::randg( arma::distr_param( (t_df(k) - t_loc) * t_df_proposal_window, 1.0 / t_df_proposal_window) );
// //         
// //         // t_df_proposed = t_loc + std::exp(arma::randn() * t_df_proposal_window + log(t_df(k) - t_loc) );
// //         
// //         // proposed_model_score = logNormalLogProbability(t_df(k) - t_loc, t_df_proposed - t_loc, t_df_proposal_window);
// //         // current_model_score = logNormalLogProbability(t_df_proposed - t_loc, t_df(k) - t_loc, t_df_proposal_window);
// //         // 
// //         // std::cout  << "\nProposed score: " << proposed_model_score << "\nCurrent score: " << current_model_score;
// //         
// //         // t_df_proposed = t_loc + std::exp((arma::randn() * t_df_proposal_window) + t_df(k) - t_loc);
// // 
// //         // // Log probability under the proposal density
// //         // proposed_model_score = logNormalLogProbability(t_df(k) - t_loc, (t_df_proposed - t_loc), t_df_proposal_window);
// //         // current_model_score = logNormalLogProbability(t_df_proposed - t_loc, (t_df(k) - t_loc), t_df_proposal_window);
// //         
// //         // Proposed value
// //         // t_df_proposed = t_loc + arma::randg( arma::distr_param( (t_df(k) - t_loc) * t_df_proposal_window, 1.0 / t_df_proposal_window) );
// //         proposed_pdf_coef = calcPDFCoef(t_df_proposed);
// // 
// //         // std::cout << "\n\nDF: " << t_df(k) << "\nProposed DF: " << t_df_proposed;
// //         
// //         // Asymmetric proposal density
// //         proposed_model_score = gammaLogLikelihood(t_df(k) - t_loc, (t_df_proposed - t_loc) * t_df_proposal_window, t_df_proposal_window);
// //         current_model_score = gammaLogLikelihood(t_df_proposed - t_loc, (t_df(k) - t_loc) * t_df_proposal_window, t_df_proposal_window);
// // 
// //         // The prior is included in the kernel
// //         proposed_model_score = dfLogKernel(k, t_df_proposed, proposed_pdf_coef);
// //         current_model_score = dfLogKernel(k, t_df(k), pdf_coef(k));
// //         
// //         u = arma::randu();
// //         acceptance_prob = std::min(1.0, std::exp(proposed_model_score - current_model_score));
// //         
// //       }
// //       
// //       if((u < acceptance_prob) || (N_k(k) == 0)) {
// //         t_df(k) = t_df_proposed;
// //         t_df_count(k)++;
// //         pdf_coef(k) = proposed_pdf_coef;
// //       }
// //     }
// //   }
// //   
// //   virtual void metropolisStep() {
// //     
// //     // Metropolis step for cluster parameters
// //     clusterCovarianceMetropolis();
// //     
// //     // std::cout << "\n\nCluster covariance.";
// //     
// //     // matrixCombinations();
// //     
// //     clusterMeanMetropolis();
// //     
// //     // std::cout << "\n\nCluster mean.";
// //     
// //     // matrixCombinations();
// // 
// //     // Update the shape parameter of the skew normal
// //     clusterDFMetropolis();
// //     
// //     // std::cout << "\n\nCluster df.";
// //     
// //     // matrixCombinations();
// // 
// //     // Metropolis step for batch parameters if more than 1 batch
// //     // if(B > 1){
// //     batchScaleMetropolis();
// //     
// //     // std::cout << "\n\nBatch scale.";
// //     
// //     // matrixCombinations(); 
// //     
// //     batchShiftMetorpolis();
// //     
// //     // std::cout << "\n\nBatch mean.";
// //     
// //     // matrixCombinations();
// //     
// //     // }
// //   };
// //   
// // };
// 
// // class mvtPredictive : public mvtSampler, public semisupervisedSampler {
// //   
// // private:
// //   
// // public:
// //   
// //   using mvtSampler::mvtSampler;
// //   
// //   mvtPredictive(
// //     arma::uword _K,
// //     arma::uword _B,
// //     double _mu_proposal_window,
// //     double _cov_proposal_window,
// //     double _m_proposal_window,
// //     double _S_proposal_window,
// //     double _t_df_proposal_window,
// //     // double _rho,
// //     // double _theta,
// //     arma::uvec _labels,
// //     arma::uvec _batch_vec,
// //     arma::vec _concentration,
// //     arma::mat _X,
// //     arma::uvec _fixed
// //   ) : 
// //     sampler(_K, _B, _labels, _batch_vec, _concentration, _X),
// //     mvnSampler(                           
// //       _K,
// //       _B,
// //       _mu_proposal_window,
// //       _cov_proposal_window,
// //       _m_proposal_window,
// //       _S_proposal_window,
// //       _labels,
// //       _batch_vec,
// //       _concentration,
// //       _X
// //     ), mvtSampler(                           
// //       _K,
// //       _B,
// //       _mu_proposal_window,
// //       _cov_proposal_window,
// //       _m_proposal_window,
// //       _S_proposal_window,
// //       _t_df_proposal_window,
// //       // _rho,
// //       // _theta,
// //       _labels,
// //       _batch_vec,
// //       _concentration,
// //       _X
// //     ), semisupervisedSampler(_K, _B, _labels, _batch_vec, _concentration, _X, _fixed)
// //     {
// //     };
// //  
// //   virtual ~mvtPredictive() { };
// //   
// //   // virtual void sampleFromPriors() {
// //   //   
// //   //   arma::mat X_k;
// //   //   
// //   //   for(arma::uword k = 0; k < K; k++){
// //   //     X_k = X.rows(arma::find(labels == k && fixed == 1));
// //   //     cov.slice(k) = arma::diagmat(arma::stddev(X_k).t());
// //   //     mu.col(k) = arma::mean(X_k).t();
// //   //     
// //   //     // Draw from a shifted gamma distribution (i.e. gamma with location parameter)
// //   //     t_df(k) = t_loc + arma::randg<double>( arma::distr_param(psi, 1.0 / chi));
// //   //     
// //   //   }
// //   //   for(arma::uword b = 0; b < B; b++){
// //   //     for(arma::uword p = 0; p < P; p++){
// //   //       
// //   //       // Fix the 0th batch at no effect; all other batches have an effect
// //   //       // relative to this
// //   //       // if(b == 0){
// //   //       S(p, b) = 1.0;
// //   //       m(p, b) = 0.0;
// //   //       // } else {
// //   //       // S(p, b) = 1.0 / arma::randg<double>( arma::distr_param(rho, 1.0 / theta ) );
// //   //       // m(p, b) = arma::randn<double>() * S(p, b) / lambda + delta(p);
// //   //       // }
// //   //     }
// //   //   }
// //   // };
// //   
// // };
// 
// 
// // // Factory for creating instances of samplers
// // //' @name semisupervisedSamplerFactory
// // //' @title Factory for different sampler subtypes.
// // //' @description The factory allows the type of mixture implemented to change 
// // //' based upon the user input.
// // //' @field new Constructor \itemize{
// // //' \item Parameter: samplerType - the density type to be modelled
// // //' \item Parameter: K - the number of components to model
// // //' \item Parameter: labels - the initial clustering of the data
// // //' \item Parameter: concentration - the vector for the prior concentration of 
// // //' the Dirichlet distribution of the component weights
// // //' \item Parameter: X - the data to model
// // //' }
// // class semisupervisedSamplerFactory
// // {
// // public:
// //   enum samplerType {
// //     // G = 0,
// //     MVN = 1,
// //     MVT = 2,
// //     MSN = 3
// //   };
// //   
// //   static std::unique_ptr<semisupervisedSampler> createSemisupervisedSampler(samplerType type,
// //     arma::uword K,
// //     arma::uword B,
// //     double mu_proposal_window,
// //     double cov_proposal_window,
// //     double m_proposal_window,
// //     double S_proposal_window,
// //     double t_df_proposal_window,
// //     double phi_proposal_window,
// //     double rho,
// //     double theta,
// //     arma::uvec labels,
// //     arma::uvec batch_vec,
// //     arma::vec concentration,
// //     arma::mat X,
// //     arma::uvec fixed
// //     ) {
// //       switch (type) {
// //       // case G: return std::make_unique<gaussianSampler>(K, labels, concentration, X);
// //         
// //       case MVN: return std::make_unique<mvnPredictive>(K,
// //                                                     B,
// //                                                     mu_proposal_window,
// //                                                     cov_proposal_window,
// //                                                     m_proposal_window,
// //                                                     S_proposal_window,
// //                                                     rho,
// //                                                     theta,
// //                                                     labels,
// //                                                     batch_vec,
// //                                                     concentration,
// //                                                     X,
// //                                                     fixed);
// //       case MVT: return std::make_unique<mvtPredictive>(K,
// //                                                     B,
// //                                                     mu_proposal_window,
// //                                                     cov_proposal_window,
// //                                                     m_proposal_window,
// //                                                     S_proposal_window,
// //                                                     t_df_proposal_window,
// //                                                     rho,
// //                                                     theta,
// //                                                     labels,
// //                                                     batch_vec,
// //                                                     concentration,
// //                                                     X,
// //                                                     fixed);
// //       case MSN: return std::make_unique<msnPredictive>(K,
// //                                                     B,
// //                                                     mu_proposal_window,
// //                                                     cov_proposal_window,
// //                                                     m_proposal_window,
// //                                                     S_proposal_window,
// //                                                     phi_proposal_window,
// //                                                     rho,
// //                                                     theta,
// //                                                     labels,
// //                                                     batch_vec,
// //                                                     concentration,
// //                                                     X,
// //                                                     fixed);
// //       default: throw "invalid sampler type.";
// //       }
// //       
// //     }
// //   
// // };
// 
// 
// 
// // //' @title Sample batch mixture model
// // //' @description Performs MCMC sampling for a mixture model with batch effects.
// // //' @param X The data matrix to perform clustering upon (items to cluster in rows).
// // //' @param K The number of components to model (upper limit on the number of clusters found).
// // //' @param labels Vector item labels to initialise from.
// // //' @param dataType Int, 0: independent Gaussians, 1: Multivariate normal, or 2: Categorical distributions.
// // //' @param R The number of iterations to run for.
// // //' @param thin thinning factor for samples recorded.
// // //' @param concentration Vector of concentrations for mixture weights (recommended to be symmetric).
// // //' @return Named list of the matrix of MCMC samples generated (each row 
// // //' corresponds to a different sample) and BIC for each saved iteration.
// // // [[Rcpp::export]]
// // Rcpp::List sampleMVN (
// //     arma::mat X,
// //     arma::uword K,
// //     arma::uword B,
// //     arma::uvec labels,
// //     arma::uvec batch_vec,
// //     double mu_proposal_window,
// //     double cov_proposal_window,
// //     double m_proposal_window,
// //     double S_proposal_window,
// //     double rho,
// //     double theta,
// //     arma::uword R,
// //     arma::uword thin,
// //     arma::vec concentration,
// //     bool verbose = true,
// //     bool doCombinations = false,
// //     bool printCovariance = false
// // ) {
// //   
// //   // The random seed is set at the R level via set.seed() apparently.
// //   // std::default_random_engine generator(seed);
// //   // arma::arma_rng::set_seed(seed);
// //   
// // 
// //   mvnSampler my_sampler(K,
// //                         B,
// //                         mu_proposal_window,
// //                         cov_proposal_window,
// //                         m_proposal_window,
// //                         S_proposal_window,
// //                         // rho,
// //                         // theta,
// //                         labels,
// //                         batch_vec,
// //                         concentration,
// //                         X
// //   );
// //   
// //   // // Declare the factory
// //   // samplerFactory my_factory;
// //   // 
// //   // // Convert from an int to the samplerType variable for our Factory
// //   // samplerFactory::samplerType val = static_cast<samplerFactory::samplerType>(dataType);
// //   // 
// //   // // Make a pointer to the correct type of sampler
// //   // std::unique_ptr<sampler> sampler_ptr = my_factory.createSampler(val,
// //   //                                                                 K,
// //   //                                                                 labels,
// //   //                                                                 concentration,
// //   //                                                                 X);
// //   
// //   // We use this enough that declaring it is worthwhile
// //   arma::uword P = X.n_cols;
// //   
// //   // The output matrix
// //   arma::umat class_record(floor(R / thin), X.n_rows);
// //   class_record.zeros();
// //   
// //   // We save the BIC at each iteration
// //   arma::vec BIC_record = arma::zeros<arma::vec>(floor(R / thin)),
// //     observed_likelihood = arma::zeros<arma::vec>(floor(R / thin)),
// //     complete_likelihood = arma::zeros<arma::vec>(floor(R / thin));
// //   
// //   arma::uvec acceptance_vec = arma::zeros<arma::uvec>(floor(R / thin));
// //   arma::mat weights_saved = arma::zeros<arma::mat>(floor(R / thin), K);
// //   
// //   arma::cube mean_sum_saved(P, K * B, floor(R / thin)), mu_saved(P, K, floor(R / thin)), m_saved(P, B, floor(R / thin)), cov_saved(P, K * P, floor(R / thin)), t_saved(P, B, floor(R / thin)), cov_comb_saved(P, P * K * B, floor(R / thin)), batch_corrected_data(my_sampler.N, P, floor(R / thin));
// // 
// //   // arma::field<arma::cube> cov_saved(my_sampler.P, my_sampler.P, K, floor(R / thin));
// //   mu_saved.zeros();
// //   cov_saved.zeros();
// //   cov_comb_saved.zeros();
// //   m_saved.zeros();
// //   t_saved.zeros();
// //   
// //   arma::uword save_int = 0;
// //   
// //   // Sampler from priors
// //   my_sampler.sampleFromPriors();
// //   my_sampler.matrixCombinations();
// //   // my_sampler.modelScore();
// //   // sampler_ptr->sampleFromPriors();
// //   
// //   // my_sampler.model_score = my_sampler.modelLogLikelihood(
// //   //   my_sampler.mu,
// //   //   my_sampler.tau,
// //   //   my_sampler.m,
// //   //   my_sampler.t
// //   // ) + my_sampler.priorLogProbability(
// //   //     my_sampler.mu,
// //   //     my_sampler.tau,
// //   //     my_sampler.m,
// //   //     my_sampler.t
// //   // );
// //   
// //   // sample_prt.model_score->sampler_ptr.modelLo
// //   
// //   // Iterate over MCMC moves
// //   for(arma::uword r = 0; r < R; r++){
// //     
// //     my_sampler.updateWeights();
// //     
// //     // Metropolis step for batch parameters
// //     my_sampler.metropolisStep(); 
// //     
// //     my_sampler.updateAllocation();
// //     
// //     
// //     // sampler_ptr->updateWeights();
// //     // sampler_ptr->proposeNewParameters();
// //     // sampler_ptr->updateAllocation();
// //     
// //     // Record results
// //     if((r + 1) % thin == 0){
// //       
// //       // Update the BIC for the current model fit
// //       // sampler_ptr->calcBIC();
// //       // BIC_record( save_int ) = sampler_ptr->BIC; 
// //       // 
// //       // // Save the current clustering
// //       // class_record.row( save_int ) = sampler_ptr->labels.t();
// //       
// //       my_sampler.calcBIC();
// //       BIC_record( save_int ) = my_sampler.BIC;
// //       observed_likelihood( save_int ) = my_sampler.observed_likelihood;
// //       class_record.row( save_int ) = my_sampler.labels.t();
// //       acceptance_vec( save_int ) = my_sampler.accepted;
// //       weights_saved.row( save_int ) = my_sampler.w.t();
// //       mu_saved.slice( save_int ) = my_sampler.mu;
// //       // tau_saved.slice( save_int ) = my_sampler.tau;
// //       // cov_saved( save_int ) = my_sampler.cov;
// //       m_saved.slice( save_int ) = my_sampler.m;
// //       t_saved.slice( save_int ) = my_sampler.S;
// //       mean_sum_saved.slice( save_int ) = my_sampler.mean_sum;
// //       
// //       
// //       cov_saved.slice ( save_int ) = arma::reshape(arma::mat(my_sampler.cov.memptr(), my_sampler.cov.n_elem, 1, false), P, P * K);
// //       cov_comb_saved.slice( save_int) = arma::reshape(arma::mat(my_sampler.cov_comb.memptr(), my_sampler.cov_comb.n_elem, 1, false), P, P * K * B); 
// //       
// //       my_sampler.updateBatchCorrectedData();
// //       batch_corrected_data.slice( save_int ) =  my_sampler.Y;
// //       
// //       complete_likelihood( save_int ) = my_sampler.complete_likelihood;
// //       
// //       if(printCovariance) {  
// //         std::cout << "\n\nCovariance cube:\n" << my_sampler.cov;
// //         std::cout << "\n\nBatch covariance matrix:\n" << my_sampler.S;
// //       }
// //       
// //       save_int++;
// //     }
// //   }
// //   
// //   if(verbose) {
// //     std::cout << "\n\nCovariance acceptance rate:\n" << arma::conv_to< arma::vec >::from(my_sampler.cov_count) / R;
// //     std::cout << "\n\ncluster mean acceptance rate:\n" << arma::conv_to< arma::vec >::from(my_sampler.mu_count) / R;
// //     std::cout << "\n\nBatch covariance acceptance rate:\n" << arma::conv_to< arma::vec >::from(my_sampler.S_count) / R;
// //     std::cout << "\n\nBatch mean acceptance rate:\n" << arma::conv_to< arma::vec >::from(my_sampler.m_count) / R;
// //   }
// //   
// //   return(List::create(Named("samples") = class_record, 
// //                       Named("means") = mu_saved,
// //                       Named("covariance") = cov_saved,
// //                       Named("batch_shift") = m_saved,
// //                       Named("batch_scale") = t_saved,
// //                       Named("mean_sum") = mean_sum_saved,
// //                       Named("cov_comb") = cov_comb_saved,
// //                       Named("weights") = weights_saved,
// //                       Named("cov_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.cov_count) / R,
// //                       Named("mu_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.mu_count) / R,
// //                       Named("S_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.S_count) / R,
// //                       Named("m_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.m_count) / R,                      
// //                       Named("observed_likelihood") = observed_likelihood,
// //                       Named("BIC") = BIC_record));
// //   
// // };
// // 
// 
// 
// 
// 
// 
// 
// // [[Rcpp::export]]
// Rcpp::List sampleMSN (
//     arma::mat X,
//     arma::uword K,
//     arma::uword B,
//     arma::uvec labels,
//     arma::uvec batch_vec,
//     double mu_proposal_window,
//     double cov_proposal_window,
//     double m_proposal_window,
//     double S_proposal_window,
//     double phi_proposal_window,
//     double rho,
//     double theta,
//     arma::uword R,
//     arma::uword thin,
//     arma::vec concentration,
//     bool verbose = true,
//     bool doCombinations = false,
//     bool printCovariance = false
// ) {
//   
//   // The random seed is set at the R level via set.seed() apparently.
//   // std::default_random_engine generator(seed);
//   // arma::arma_rng::set_seed(seed);
//   
//   msnSampler my_sampler(K,
//                           B,
//                           mu_proposal_window,
//                           cov_proposal_window,
//                           m_proposal_window,
//                           S_proposal_window,
//                           phi_proposal_window,
//                           // rho,
//                           // theta,
//                           labels,
//                           batch_vec,
//                           concentration,
//                           X
//   );
//   
//   // We use this enough that declaring it is worthwhile
//   arma::uword P = X.n_cols;
//   
//   // The output matrix
//   arma::umat class_record(floor(R / thin), X.n_rows);
//   class_record.zeros();
//   
//   // We save the BIC at each iteration
//   arma::vec BIC_record = arma::zeros<arma::vec>(floor(R / thin)),
//     observed_likelihood = arma::zeros<arma::vec>(floor(R / thin)),
//     complete_likelihood = arma::zeros<arma::vec>(floor(R / thin));
//   
//   arma::uvec acceptance_vec = arma::zeros<arma::uvec>(floor(R / thin));
//   arma::mat weights_saved = arma::zeros<arma::mat>(floor(R / thin), K);
//   
//   arma::cube mean_sum_save(my_sampler.P, K * B, floor(R / thin)), mu_saved(my_sampler.P, K, floor(R / thin)), m_saved(my_sampler.P, B, floor(R / thin)), cov_saved(P, K * P, floor(R / thin)), t_saved(P, B, floor(R / thin)), cov_comb_saved(P, P * K * B, floor(R / thin)), phi_saved(my_sampler.P, K, floor(R / thin)), batch_corrected_data(my_sampler.N, P, floor(R / thin));
// 
//   // arma::field<arma::cube> cov_saved(my_sampler.P, my_sampler.P, K, floor(R / thin));
//   mu_saved.zeros();
//   cov_saved.zeros();
//   cov_comb_saved.zeros();
//   m_saved.zeros();
//   t_saved.zeros();
//   
//   arma::uword save_int = 0;
//   
//   // Sampler from priors
//   my_sampler.sampleFromPriors();
//   my_sampler.matrixCombinations();
//   
//   // Iterate over MCMC moves
//   for(arma::uword r = 0; r < R; r++){
//     
//     my_sampler.updateWeights();
//     
//     // Metropolis step for batch parameters
//     my_sampler.metropolisStep(); 
//     
//     my_sampler.updateAllocation();
//     
//     // Record results
//     if((r + 1) % thin == 0){
//       
//       my_sampler.calcBIC();
//       BIC_record( save_int ) = my_sampler.BIC;
//       observed_likelihood( save_int ) = my_sampler.observed_likelihood;
//       class_record.row( save_int ) = my_sampler.labels.t();
//       acceptance_vec( save_int ) = my_sampler.accepted;
//       weights_saved.row( save_int ) = my_sampler.w.t();
//       mu_saved.slice( save_int ) = my_sampler.mu;
//       // tau_saved.slice( save_int ) = my_sampler.tau;
//       // cov_saved( save_int ) = my_sampler.cov;
//       m_saved.slice( save_int ) = my_sampler.m;
//       t_saved.slice( save_int ) = my_sampler.S;
//       mean_sum_save.slice( save_int ) = my_sampler.mean_sum;
//       phi_saved.slice( save_int ) = my_sampler.phi;
//       
//       cov_saved.slice( save_int ) = arma::reshape(arma::mat(my_sampler.cov.memptr(), my_sampler.cov.n_elem, 1, false), P, P * K);
//       cov_comb_saved.slice( save_int) = arma::reshape(arma::mat(my_sampler.cov_comb.memptr(), my_sampler.cov_comb.n_elem, 1, false), P, P * K * B);
//       
//       my_sampler.updateBatchCorrectedData();
//       batch_corrected_data.slice( save_int ) =  my_sampler.Y;
//       
//       complete_likelihood( save_int ) = my_sampler.complete_likelihood;
//       
//       if(printCovariance) {  
//         std::cout << "\n\nCovariance cube:\n" << my_sampler.cov;
//         std::cout << "\n\nBatch covariance matrix:\n" << my_sampler.S;
//       }
//       
//       save_int++;
//     }
//   }
//   
//   if(verbose) {
//     std::cout << "\n\nCovariance acceptance rate:\n" << my_sampler.cov_count;
//     std::cout << "\n\ncluster mean acceptance rate:\n" << my_sampler.mu_count;
//     std::cout << "\n\nCluster shape acceptance rate:\n" << my_sampler.phi_count;
//     std::cout << "\n\nBatch covariance acceptance rate:\n" << my_sampler.S_count;
//     std::cout << "\n\nBatch mean acceptance rate:\n" << my_sampler.m_count;
//   }
//   
//   return(List::create(Named("samples") = class_record, 
//                       Named("means") = mu_saved,
//                       Named("covariance") = cov_saved,
//                       Named("cov_comb") = cov_comb_saved,
//                       Named("shapes") = phi_saved,
//                       Named("batch_shift") = m_saved,
//                       Named("batch_scale") = t_saved,
//                       Named("mean_sum") = mean_sum_save,
//                       Named("weights") = weights_saved,
//                       Named("cov_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.cov_count) / R,
//                       Named("mu_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.mu_count) / R,
//                       Named("S_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.S_count) / R,
//                       Named("m_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.m_count) / R,
//                       Named("phi_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.phi_count) / R,
//                       Named("observed_likelihood") = observed_likelihood,
//                       Named("BIC") = BIC_record));
//   
// };
// // 
// // // [[Rcpp::export]]
// // Rcpp::List sampleMVT (
// //     arma::mat X,
// //     arma::uword K,
// //     arma::uword B,
// //     arma::uvec labels,
// //     arma::uvec batch_vec,
// //     double mu_proposal_window,
// //     double cov_proposal_window,
// //     double m_proposal_window,
// //     double S_proposal_window,
// //     double t_df_proposal_window,
// //     double rho,
// //     double theta,
// //     arma::uword R,
// //     arma::uword thin,
// //     arma::vec concentration,
// //     bool verbose = true,
// //     bool doCombinations = false,
// //     bool printCovariance = false
// // ) {
// //   
// //   // The random seed is set at the R level via set.seed() apparently.
// //   // std::default_random_engine generator(seed);
// //   // arma::arma_rng::set_seed(seed);
// //   
// //   mvtSampler my_sampler(K,
// //     B,
// //     mu_proposal_window,
// //     cov_proposal_window,
// //     m_proposal_window,
// //     S_proposal_window,
// //     t_df_proposal_window,
// //     // rho,
// //     // theta,
// //     labels,
// //     batch_vec,
// //     concentration,
// //     X
// //   );
// //   
// //   // We use this enough that declaring it is worthwhile
// //   arma::uword P = X.n_cols;
// //   
// //   // The output matrix
// //   arma::umat class_record(floor(R / thin), X.n_rows);
// //   class_record.zeros();
// //   
// //   // We save the BIC at each iteration
// //   arma::vec BIC_record = arma::zeros<arma::vec>(floor(R / thin)),
// //     observed_likelihood = arma::zeros<arma::vec>(floor(R / thin)),
// //     complete_likelihood = arma::zeros<arma::vec>(floor(R / thin));
// //   
// //   arma::uvec acceptance_vec = arma::zeros<arma::uvec>(floor(R / thin));
// //   arma::mat weights_saved(floor(R / thin), K), t_df_saved(floor(R / thin), K);
// //   weights_saved.zeros();
// //   t_df_saved.zeros();
// //   
// //   arma::cube mean_sum_saved(my_sampler.P, K * B, floor(R / thin)), 
// //     mu_saved(my_sampler.P, K, floor(R / thin)), 
// //     m_saved(my_sampler.P, B, floor(R / thin)), 
// //     cov_saved(P, K * P, floor(R / thin)), 
// //     t_saved(P, B, floor(R / thin)), 
// //     cov_comb_saved(P, P * K * B, floor(R / thin)),
// //     batch_corrected_data(my_sampler.N, P, floor(R / thin));
// //   
// //   mu_saved.zeros();
// //   cov_saved.zeros();
// //   cov_comb_saved.zeros();
// //   m_saved.zeros();
// //   t_saved.zeros();
// //   
// //   arma::uword save_int = 0;
// //   
// //   // Sampler from priors
// //   my_sampler.sampleFromPriors();
// //   my_sampler.matrixCombinations();
// //   
// //   // Iterate over MCMC moves
// //   for(arma::uword r = 0; r < R; r++){
// //     
// //     my_sampler.updateWeights();
// //     
// //     // std::cout << "\nWeights.\n";
// //     
// //     // Metropolis step for batch parameters
// //     my_sampler.metropolisStep(); 
// //     
// //     // std::cout << "\nMetropolis.\n";
// //     
// //     my_sampler.updateAllocation();
// //     
// //     // std::cout << "\nAllocation.\n";
// //     
// //     // Record results
// //     if((r + 1) % thin == 0){
// //       
// //       // Update the BIC for the current model fit
// //       // sampler_ptr->calcBIC();
// //       // BIC_record( save_int ) = sampler_ptr->BIC; 
// //       // 
// //       // // Save the current clustering
// //       // class_record.row( save_int ) = sampler_ptr->labels.t();
// //       
// //       my_sampler.calcBIC();
// //       BIC_record( save_int ) = my_sampler.BIC;
// //       observed_likelihood( save_int ) = my_sampler.observed_likelihood;
// //       class_record.row( save_int ) = my_sampler.labels.t();
// //       acceptance_vec( save_int ) = my_sampler.accepted;
// //       weights_saved.row( save_int ) = my_sampler.w.t();
// //       mu_saved.slice( save_int ) = my_sampler.mu;
// //       m_saved.slice( save_int ) = my_sampler.m;
// //       t_saved.slice( save_int ) = my_sampler.S;
// //       mean_sum_saved.slice( save_int ) = my_sampler.mean_sum;
// //       t_df_saved.row( save_int ) = my_sampler.t_df.t();
// //       
// //       cov_saved.slice ( save_int ) = arma::reshape(arma::mat(my_sampler.cov.memptr(), my_sampler.cov.n_elem, 1, false), P, P * K);
// //       cov_comb_saved.slice( save_int) = arma::reshape(arma::mat(my_sampler.cov_comb.memptr(), my_sampler.cov_comb.n_elem, 1, false), P, P * K * B); 
// //       
// //       my_sampler.updateBatchCorrectedData();
// //       batch_corrected_data.slice( save_int ) =  my_sampler.Y;
// //       
// //       complete_likelihood( save_int ) = my_sampler.complete_likelihood;
// //       
// //       
// //       if(printCovariance) {  
// //         std::cout << "\n\nCovariance cube:\n" << my_sampler.cov;
// //         std::cout << "\n\nBatch covariance matrix:\n" << my_sampler.S;
// //       }
// //       
// //       save_int++;
// //     }
// //   }
// //   
// //   if(verbose) {
// // 
// //     std::cout << "\n\nCovariance acceptance rate:\n" << arma::conv_to< arma::vec >::from(my_sampler.cov_count) / R;
// //     std::cout << "\n\ncluster mean acceptance rate:\n" << arma::conv_to< arma::vec >::from(my_sampler.mu_count) / R;
// //     std::cout << "\n\nBatch covariance acceptance rate:\n" << arma::conv_to< arma::vec >::from(my_sampler.S_count) / R;
// //     std::cout << "\n\nBatch mean acceptance rate:\n" << arma::conv_to< arma::vec >::from(my_sampler.m_count) / R;
// //     std::cout << "\n\nCluster t d.f. acceptance rate:\n" << arma::conv_to< arma::vec >::from(my_sampler.t_df_count) / R;
// //     
// //   }
// //   
// //   return(List::create(Named("samples") = class_record, 
// //     Named("means") = mu_saved,
// //     Named("covariance") = cov_saved,
// //     Named("batch_shift") = m_saved,
// //     Named("batch_scale") = t_saved,
// //     Named("mean_sum") = mean_sum_saved,
// //     Named("cov_comb") = cov_comb_saved,
// //     Named("t_df") = t_df_saved,
// //     Named("weights") = weights_saved,
// //     Named("cov_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.cov_count) / R,
// //     Named("mu_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.mu_count) / R,
// //     Named("S_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.S_count) / R,
// //     Named("m_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.m_count) / R,
// //     Named("t_df_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.t_df_count) / R,
// //     Named("observed_likelihood") = observed_likelihood,
// //     Named("BIC") = BIC_record)
// //   );
// //   
// // };
// 
// // 
// // //' @title Mixture model
// // //' @description Performs MCMC sampling for a mixture model.
// // //' @param X The data matrix to perform clustering upon (items to cluster in rows).
// // //' @param K The number of components to model (upper limit on the number of clusters found).
// // //' @param labels Vector item labels to initialise from.
// // //' @param fixed Binary vector of the items that are fixed in their initial label.
// // //' @param dataType Int, 0: independent Gaussians, 1: Multivariate normal, or 2: Categorical distributions.
// // //' @param R The number of iterations to run for.
// // //' @param thin thinning factor for samples recorded.
// // //' @param concentration Vector of concentrations for mixture weights (recommended to be symmetric).
// // //' @return Named list of the matrix of MCMC samples generated (each row 
// // //' corresponds to a different sample) and BIC for each saved iteration.
// // // [[Rcpp::export]]
// // Rcpp::List sampleSemisupervisedMVN (
// //     arma::mat X,
// //     arma::uword K,
// //     arma::uword B,
// //     arma::uvec labels,
// //     arma::uvec batch_vec,
// //     arma::uvec fixed,
// //     double mu_proposal_window,
// //     double cov_proposal_window,
// //     double m_proposal_window,
// //     double S_proposal_window,
// //     double rho,
// //     double theta,
// //     arma::uword R,
// //     arma::uword thin,
// //     arma::vec concentration,
// //     bool verbose = true,
// //     bool doCombinations = false,
// //     bool printCovariance = false
// // ) {
// //   
// //   // // Set the random number
// //   // std::default_random_engine generator(seed);
// //   // 
// //   // // Declare the factory
// //   // semisupervisedSamplerFactory my_factory;
// //   // 
// //   // // Convert from an int to the samplerType variable for our Factory
// //   // semisupervisedSamplerFactory::samplerType val = static_cast<semisupervisedSamplerFactory::samplerType>(dataType);
// //   // 
// //   // // Make a pointer to the correct type of sampler
// //   // std::unique_ptr<sampler> sampler_ptr = my_factory.createSemisupervisedSampler(val,
// //   //                                                                               K,
// //   //                                                                               labels,
// //   //                                                                               concentration,
// //   //                                                                               X,
// //   //                                                                               fixed);
// //   
// //   
// //   mvnPredictive my_sampler(K,
// //                            B,
// //                            mu_proposal_window,
// //                            cov_proposal_window,
// //                            m_proposal_window,
// //                            S_proposal_window,
// //                            // rho,
// //                            // theta,
// //                            labels,
// //                            batch_vec,
// //                            concentration,
// //                            X,
// //                            fixed
// //   );
// //   
// //   // // Declare the factory
// //   // samplerFactory my_factory;
// //   // 
// //   // // Convert from an int to the samplerType variable for our Factory
// //   // samplerFactory::samplerType val = static_cast<samplerFactory::samplerType>(dataType);
// //   // 
// //   // // Make a pointer to the correct type of sampler
// //   // std::unique_ptr<sampler> sampler_ptr = my_factory.createSampler(val,
// //   //                                                                 K,
// //   //                                                                 labels,
// //   //                                                                 concentration,
// //   //                                                                 X);
// //   
// //   uword P = X.n_cols, N = X.n_rows;
// //   
// //   // uword restart_count = 0, n_restarts = 3, check_iter = 250;
// //   // double min_acceptance = 0.15;
// //   // 
// //   // restart:
// //   
// //   // The output matrix
// //   umat class_record(floor(R / thin), X.n_rows);
// //   class_record.zeros();
// //   
// //   // We save the BIC at each iteration
// //   vec BIC_record = zeros<vec>(floor(R / thin)),
// //     observed_likelihood = zeros<vec>(floor(R / thin)),
// //     complete_likelihood = zeros<vec>(floor(R / thin));
// //   
// //   uvec acceptance_vec = zeros<uvec>(floor(R / thin));
// //   mat weights_saved = zeros<mat>(floor(R / thin), K);
// //   
// //   cube mean_sum_saved(P, K * B, floor(R / thin)), 
// //     mu_saved(P, K, floor(R / thin)),
// //     m_saved(P, B, floor(R / thin)), 
// //     cov_saved(P, K * P, floor(R / thin)),
// //     S_saved(P, B, floor(R / thin)), 
// //     cov_comb_saved(P, P * K * B, floor(R / thin)), 
// //     alloc(N, K, floor(R / thin)), 
// //     batch_corrected_data(N, P, floor(R / thin));
// //     // weights_saved(K, B, floor(R / thin));
// //   
// //   // field<cube> cov_saved(my_sampler.P, my_sampler.P, K, floor(R / thin));
// //   mu_saved.zeros();
// //   cov_saved.zeros();
// //   cov_comb_saved.zeros();
// //   m_saved.zeros();
// //   S_saved.zeros();
// //   // weights_saved.zeros();
// //   
// //   uword save_int = 0;
// //   
// //   // Sampler from priors
// //   my_sampler.sampleFromPriors();
// //   
// //   my_sampler.matrixCombinations();
// //   // my_sampler.modelScore();
// //   // sampler_ptr->sampleFromPriors();
// //   
// //   // my_sampler.model_score = my_sampler.modelLogLikelihood(
// //   //   my_sampler.mu,
// //   //   my_sampler.tau,
// //   //   my_sampler.m,
// //   //   my_sampler.t
// //   // ) + my_sampler.priorLogProbability(
// //   //     my_sampler.mu,
// //   //     my_sampler.tau,
// //   //     my_sampler.m,
// //   //     my_sampler.t
// //   // );
// //   
// //   // sample_prt.model_score->sampler_ptr.modelLo
// //   
// //   // Iterate over MCMC moves
// //   for(uword r = 0; r < R; r++){
// //     
// //     // my_sampler.checkPositiveDefinite(r);
// //     // 
// //     // if(r == check_iter) {
// //     // 
// //     //   if(any((conv_to< vec >::from(my_sampler.cov_count) / R) < min_acceptance)){
// //     //     if(restart_count == n_restarts) {
// //     //       std::cout << "Cluster covariance acceptance rates too low.\nPlease restart with a different proposal window and/or random seed.";
// //     //       throw;
// //     //     }
// //     //     restart_count++;
// //     //     goto restart;
// //     //   }
// //     //   if(any((conv_to< vec >::from(my_sampler.mu_count) / R) < min_acceptance)){
// //     //     if(restart_count == n_restarts) {
// //     //       std::cout << "Cluster mean acceptance rates too low.\nPlease restart with a different proposal window and/or random seed.";
// //     //       throw;
// //     //     }
// //     //     restart_count++;
// //     //     goto restart;
// //     //   }
// //     //   if(any((conv_to< vec >::from(my_sampler.m_count) / R) < min_acceptance)){
// //     //     if(restart_count == n_restarts) {
// //     //       std::cout << "Batch shift acceptance rates too low.\nPlease restart with a different proposal window and/or random seed.";
// //     //       throw;
// //     //     }
// //     //     restart_count++;
// //     //     goto restart;
// //     //   }
// //     //   
// //     //   if(any((conv_to< vec >::from(my_sampler.S_count) / R) < min_acceptance)){
// //     //     if(restart_count == n_restarts) {
// //     //       std::cout << "Batch scale acceptance rates too low.\nPlease restart with a different proposal window and/or random seed.";
// //     //       throw;
// //     //     }
// //     //     restart_count++;
// //     //     goto restart;
// //     //   }
// //     // }
// //     
// //     // std::cout << "\n\nUpdate weights.";
// //     
// //     my_sampler.updateWeights();
// //     
// //     // std::cout << "\n\nUpdate parameters.";
// //     
// //     // Metropolis step for batch parameters
// //     my_sampler.metropolisStep();
// // 
// //     // std::cout << "\n\nUpdate allocation.";
// //     my_sampler.updateAllocation();
// //     
// //     
// //     // sampler_ptr->updateWeights();
// //     // sampler_ptr->proposeNewParameters();
// //     // sampler_ptr->updateAllocation();
// //     
// //     // Record results
// //     if((r + 1) % thin == 0){
// //       
// //       // Update the BIC for the current model fit
// //       // sampler_ptr->calcBIC();
// //       // BIC_record( save_int ) = sampler_ptr->BIC; 
// //       // 
// //       // // Save the current clustering
// //       // class_record.row( save_int ) = sampler_ptr->labels.t();
// //       
// //       my_sampler.calcBIC();
// //       BIC_record( save_int ) = my_sampler.BIC;
// //       observed_likelihood( save_int ) = my_sampler.observed_likelihood;
// //       class_record.row( save_int ) = my_sampler.labels.t();
// //       acceptance_vec( save_int ) = my_sampler.accepted;
// //       weights_saved.row( save_int ) = my_sampler.w.t();
// //       // weights_saved.slice( save_int ) = my_sampler.w;
// //       mu_saved.slice( save_int ) = my_sampler.mu;
// //       // tau_saved.slice( save_int ) = my_sampler.tau;
// //       // cov_saved( save_int ) = my_sampler.cov;
// //       m_saved.slice( save_int ) = my_sampler.m;
// //       S_saved.slice( save_int ) = my_sampler.S;
// //       mean_sum_saved.slice( save_int ) = my_sampler.mean_sum;
// //       
// //       alloc.slice( save_int ) = my_sampler.alloc;
// //       cov_saved.slice ( save_int ) = reshape(mat(my_sampler.cov.memptr(), my_sampler.cov.n_elem, 1, false), P, P * K);
// //       cov_comb_saved.slice( save_int) = reshape(mat(my_sampler.cov_comb.memptr(), my_sampler.cov_comb.n_elem, 1, false), P, P * K * B); 
// //       
// //       my_sampler.updateBatchCorrectedData();
// //       batch_corrected_data.slice( save_int ) =  my_sampler.Y;
// //       
// //       complete_likelihood( save_int ) = my_sampler.complete_likelihood;
// //       
// //       if(printCovariance) {  
// //         std::cout << "\n\nCovariance cube:\n" << my_sampler.cov;
// //         std::cout << "\n\nBatch covariance matrix:\n" << my_sampler.S;
// //       }
// //       
// //       save_int++;
// //     }
// //   }
// //   
// //   if(verbose) {
// //     std::cout << "\n\nCovariance acceptance rate:\n" << conv_to< vec >::from(my_sampler.cov_count) / R;
// //     std::cout << "\n\ncluster mean acceptance rate:\n" << conv_to< vec >::from(my_sampler.mu_count) / R;
// //     std::cout << "\n\nBatch covariance acceptance rate:\n" << conv_to< vec >::from(my_sampler.S_count) / R;
// //     std::cout << "\n\nBatch mean acceptance rate:\n" << conv_to< vec >::from(my_sampler.m_count) / R;
// //   }
// //   
// //   // std::cout << "\nReciprocal condition number\n" << my_sampler.rcond_count;
// //   
// //   return(List::create(Named("samples") = class_record, 
// //                       Named("means") = mu_saved,
// //                       Named("covariance") = cov_saved,
// //                       Named("batch_shift") = m_saved,
// //                       Named("batch_scale") = S_saved,
// //                       Named("mean_sum") = mean_sum_saved,
// //                       Named("cov_comb") = cov_comb_saved,
// //                       Named("weights") = weights_saved,
// //                       Named("cov_acceptance_rate") = conv_to< vec >::from(my_sampler.cov_count) / R,
// //                       Named("mu_acceptance_rate") = conv_to< vec >::from(my_sampler.mu_count) / R,
// //                       Named("S_acceptance_rate") = conv_to< vec >::from(my_sampler.S_count) / R,
// //                       Named("m_acceptance_rate") = conv_to< vec >::from(my_sampler.m_count) / R,
// //                       Named("alloc") = alloc,
// //                       Named("observed_likelihood") = observed_likelihood,
// //                       Named("complete_likelihood") = complete_likelihood,
// //                       Named("BIC") = BIC_record,
// //                       Named("batch_corrected_data") = batch_corrected_data
// //   )
// //   );
// //   
// // };
// 
// 
// 
// 
// 
// 
// 
// // [[Rcpp::export]]
// Rcpp::List sampleSemisupervisedMSN (
//     arma::mat X,
//     arma::uword K,
//     arma::uword B,
//     arma::uvec labels,
//     arma::uvec batch_vec,
//     arma::uvec fixed,
//     double mu_proposal_window,
//     double cov_proposal_window,
//     double m_proposal_window,
//     double S_proposal_window,
//     double phi_proposal_window,
//     double rho,
//     double theta,
//     arma::uword R,
//     arma::uword thin,
//     arma::vec concentration,
//     bool verbose = true,
//     bool doCombinations = false,
//     bool printCovariance = false
// ) {
//   
//   // The random seed is set at the R level via set.seed() apparently.
//   // std::default_random_engine generator(seed);
//   // arma::arma_rng::set_seed(seed);
//   
//   msnPredictive my_sampler(K,
//                            B,
//                            mu_proposal_window,
//                            cov_proposal_window,
//                            m_proposal_window,
//                            S_proposal_window,
//                            phi_proposal_window,
//                            // rho,
//                            // theta,
//                            labels,
//                            batch_vec,
//                            concentration,
//                            X,
//                            fixed
//   );
//   
//   uword P = X.n_cols;
//   
//   // The output matrix
//   umat class_record(floor(R / thin), X.n_rows);
//   class_record.zeros();
//   
//   // We save the BIC at each iteration
//   vec BIC_record = zeros<vec>(floor(R / thin)),
//     observed_likelihood = zeros<vec>(floor(R / thin)),
//     complete_likelihood = zeros<vec>(floor(R / thin));
//   
//   uvec acceptance_vec = zeros<uvec>(floor(R / thin));
//   mat weights_saved = zeros<mat>(floor(R / thin), K);
//   
//   cube mean_sum_saved(P, K * B, floor(R / thin)), mu_saved(P, K, floor(R / thin)), m_saved(P, B, floor(R / thin)), cov_saved(P, K * P, floor(R / thin)), S_saved(P, B, floor(R / thin)), cov_comb_saved(P, P * K * B, floor(R / thin)), alloc(my_sampler.N, K, floor(R / thin)), phi_saved(my_sampler.P, K, floor(R / thin));
//   // field<cube> cov_saved(my_sampler.P, my_sampler.P, K, floor(R / thin));
//   mu_saved.zeros();
//   cov_saved.zeros();
//   cov_comb_saved.zeros();
//   m_saved.zeros();
//   S_saved.zeros();
//   
//   uword save_int = 0;
//   
//   // Sampler from priors
//   my_sampler.sampleFromPriors();
//   my_sampler.matrixCombinations();
//   
//   // Iterate over MCMC moves
//   for(uword r = 0; r < R; r++){
//     
//     my_sampler.updateWeights();
//     
//     // Metropolis step for batch parameters
//     my_sampler.metropolisStep(); 
//     
//     my_sampler.updateAllocation();
//     
//     // Record results
//     if((r + 1) % thin == 0){
//       
//       my_sampler.calcBIC();
//       BIC_record( save_int ) = my_sampler.BIC;
//       observed_likelihood( save_int ) = my_sampler.observed_likelihood;
//       class_record.row( save_int ) = my_sampler.labels.t();
//       acceptance_vec( save_int ) = my_sampler.accepted;
//       weights_saved.row( save_int ) = my_sampler.w.t();
//       mu_saved.slice( save_int ) = my_sampler.mu;
//       // tau_saved.slice( save_int ) = my_sampler.tau;
//       // cov_saved( save_int ) = my_sampler.cov;
//       m_saved.slice( save_int ) = my_sampler.m;
//       S_saved.slice( save_int ) = my_sampler.S;
//       mean_sum_saved.slice( save_int ) = my_sampler.mean_sum;
//       phi_saved.slice( save_int ) = my_sampler.phi;
//       
//       alloc.slice( save_int ) = my_sampler.alloc;
//       cov_saved.slice ( save_int ) = reshape(mat(my_sampler.cov.memptr(), my_sampler.cov.n_elem, 1, false), P, P * K);
//       cov_comb_saved.slice( save_int) = reshape(mat(my_sampler.cov_comb.memptr(), my_sampler.cov_comb.n_elem, 1, false), P, P * K * B); 
//       
//       complete_likelihood( save_int ) = my_sampler.complete_likelihood;
//       
//       if(printCovariance) {  
//         std::cout << "\n\nCovariance cube:\n" << my_sampler.cov;
//         std::cout << "\n\nBatch covariance matrix:\n" << my_sampler.S;
//       }
//       
//       save_int++;
//     }
//   }
//   
//   if(verbose) {
//     
//     std::cout << "\n\nCovariance acceptance rate:\n" << conv_to< vec >::from(my_sampler.cov_count) / R;
//     std::cout << "\n\nCluster mean acceptance rate:\n" << conv_to< vec >::from(my_sampler.mu_count) / R;
//     std::cout << "\n\nCluster shape acceptance rate:\n" << conv_to< vec >::from(my_sampler.phi_count) / R;
//     std::cout << "\n\nBatch covariance acceptance rate:\n" << conv_to< vec >::from(my_sampler.S_count) / R;
//     std::cout << "\n\nBatch mean acceptance rate:\n" << conv_to< vec >::from(my_sampler.m_count) / R;
//     
//   }
//   
//   
//   return(List::create(Named("samples") = class_record, 
//                       Named("means") = mu_saved,
//                       Named("covariance") = cov_saved,
//                       Named("shapes") = phi_saved,
//                       Named("batch_shift") = m_saved,
//                       Named("batch_scale") = S_saved,
//                       Named("mean_sum") = mean_sum_saved,
//                       Named("cov_comb") = cov_comb_saved,
//                       Named("weights") = weights_saved,
//                       Named("cov_acceptance_rate") = conv_to< vec >::from(my_sampler.cov_count) / R,
//                       Named("mu_acceptance_rate") = conv_to< vec >::from(my_sampler.mu_count) / R,
//                       Named("S_acceptance_rate") = conv_to< vec >::from(my_sampler.S_count) / R,
//                       Named("m_acceptance_rate") = conv_to< vec >::from(my_sampler.m_count) / R,
//                       Named("phi_acceptance_rate") = conv_to< vec >::from(my_sampler.phi_count) / R,
//                       Named("alloc") = alloc,
//                       Named("observed_likelihood") = observed_likelihood,
//                       Named("complete_likelihood") = complete_likelihood,
//                       Named("BIC") = BIC_record
//   )
//   );
//   
// };
// 
// // // [[Rcpp::export]]
// // Rcpp::List sampleSemisupervisedMVT (
// //     arma::mat X,
// //     arma::uword K,
// //     arma::uword B,
// //     arma::uvec labels,
// //     arma::uvec batch_vec,
// //     arma::uvec fixed,
// //     double mu_proposal_window,
// //     double cov_proposal_window,
// //     double m_proposal_window,
// //     double S_proposal_window,
// //     double t_df_proposal_window,
// //     double rho,
// //     double theta,
// //     arma::uword R,
// //     arma::uword thin,
// //     arma::vec concentration,
// //     bool verbose = true,
// //     bool doCombinations = false,
// //     bool printCovariance = false
// // ) {
// //   
// //   // The random seed is set at the R level via set.seed() apparently.
// //   // std::default_random_engine generator(seed);
// //   // arma::arma_rng::set_seed(seed);
// //   
// //   
// //   mvtPredictive my_sampler(K,
// //     B,
// //     mu_proposal_window,
// //     cov_proposal_window,
// //     m_proposal_window,
// //     S_proposal_window,
// //     t_df_proposal_window,
// //     // rho,
// //     // theta,
// //     labels,
// //     batch_vec,
// //     concentration,
// //     X,
// //     fixed
// //   );
// //   
// //   uword P = X.n_cols, N = X.n_rows;
// //   
// //   // The output matrix
// //   umat class_record(floor(R / thin), X.n_rows);
// //   class_record.zeros();
// //   
// //   // We save the BIC at each iteration
// //   vec BIC_record = zeros<vec>(floor(R / thin)),
// //     observed_likelihood = zeros<vec>(floor(R / thin)),
// //     complete_likelihood = zeros<vec>(floor(R / thin));
// //   
// //   uvec acceptance_vec = zeros<uvec>(floor(R / thin));
// //   mat weights_saved(floor(R / thin), K), t_df_saved(floor(R / thin), K);
// //   weights_saved.zeros();
// //   t_df_saved.zeros();
// //   
// //   cube mean_sum_saved(P, K * B, floor(R / thin)), 
// //     mu_saved(P, K, floor(R / thin)),
// //     m_saved(P, B, floor(R / thin)), 
// //     cov_saved(P, K * P, floor(R / thin)),
// //     S_saved(P, B, floor(R / thin)), 
// //     cov_comb_saved(P, P * K * B, floor(R / thin)),
// //     alloc(N, K, floor(R / thin)),
// //     batch_corrected_data(N, P, floor(R / thin));
// // 
// //   mu_saved.zeros();
// //   cov_saved.zeros();
// //   cov_comb_saved.zeros();
// //   m_saved.zeros();
// //   S_saved.zeros();
// //   alloc.zeros();
// //   batch_corrected_data.zeros();
// //   
// //   uword save_int = 0;
// //   
// //   // Sampler from priors
// //   my_sampler.sampleFromPriors();
// //   my_sampler.matrixCombinations();
// //   
// //   // Iterate over MCMC moves
// //   for(uword r = 0; r < R; r++){
// //     
// //     my_sampler.updateWeights();
// //     
// //     // std::cout << "\nWeights.\n";
// //     
// //     // Metropolis step for batch parameters
// //     my_sampler.metropolisStep(); 
// //     
// //     // std::cout << "\nMetropolis.\n";
// //     
// //     my_sampler.updateAllocation();
// //     
// //     // std::cout << "\nAllocation.\n";
// //     
// //     // Record results
// //     if((r + 1) % thin == 0){
// //       
// //       // Update the BIC for the current model fit
// //       // sampler_ptr->calcBIC();
// //       // BIC_record( save_int ) = sampler_ptr->BIC; 
// //       // 
// //       // // Save the current clustering
// //       // class_record.row( save_int ) = sampler_ptr->labels.t();
// //       
// //       my_sampler.calcBIC();
// //       BIC_record( save_int ) = my_sampler.BIC;
// //       observed_likelihood( save_int ) = my_sampler.observed_likelihood;
// //       class_record.row( save_int ) = my_sampler.labels.t();
// //       acceptance_vec( save_int ) = my_sampler.accepted;
// //       weights_saved.row( save_int ) = my_sampler.w.t();
// //       mu_saved.slice( save_int ) = my_sampler.mu;
// //       // tau_saved.slice( save_int ) = my_sampler.tau;
// //       // cov_saved( save_int ) = my_sampler.cov;
// //       m_saved.slice( save_int ) = my_sampler.m;
// //       S_saved.slice( save_int ) = my_sampler.S;
// //       mean_sum_saved.slice( save_int ) = my_sampler.mean_sum;
// //       t_df_saved.row( save_int ) = my_sampler.t_df.t();
// //       
// //       alloc.slice( save_int ) = my_sampler.alloc;
// //       cov_saved.slice ( save_int ) = reshape(mat(my_sampler.cov.memptr(), my_sampler.cov.n_elem, 1, false), P, P * K);
// //       cov_comb_saved.slice( save_int) = reshape(mat(my_sampler.cov_comb.memptr(), my_sampler.cov_comb.n_elem, 1, false), P, P * K * B); 
// //       
// //       my_sampler.updateBatchCorrectedData();
// //       batch_corrected_data.slice( save_int ) =  my_sampler.Y;
// //       
// //       complete_likelihood( save_int ) = my_sampler.complete_likelihood;
// //       
// //       if(printCovariance) {  
// //         std::cout << "\n\nCovariance cube:\n" << my_sampler.cov;
// //         std::cout << "\n\nBatch covariance matrix:\n" << my_sampler.S;
// //       }
// //       
// //       save_int++;
// //     }
// //   }
// //   
// //   if(verbose) {
// //     
// //     std::cout << "\n\nCovariance acceptance rate:\n" << conv_to< vec >::from(my_sampler.cov_count) / R;
// //     std::cout << "\n\ncluster mean acceptance rate:\n" << conv_to< vec >::from(my_sampler.mu_count) / R;
// //     std::cout << "\n\nBatch covariance acceptance rate:\n" << conv_to< vec >::from(my_sampler.S_count) / R;
// //     std::cout << "\n\nBatch mean acceptance rate:\n" << conv_to< vec >::from(my_sampler.m_count) / R;
// //     std::cout << "\n\nCluster t d.f. acceptance rate:\n" << conv_to< vec >::from(my_sampler.t_df_count) / R;
// //     
// //   }
// //   
// //   return(
// //     List::create(Named("samples") = class_record, 
// //       Named("means") = mu_saved,
// //       Named("covariance") = cov_saved,
// //       Named("batch_shift") = m_saved,
// //       Named("batch_scale") = S_saved,
// //       Named("mean_sum") = mean_sum_saved,
// //       Named("cov_comb") = cov_comb_saved,
// //       Named("t_df") = t_df_saved,
// //       Named("weights") = weights_saved,
// //       Named("cov_acceptance_rate") = conv_to< vec >::from(my_sampler.cov_count) / R,
// //       Named("mu_acceptance_rate") = conv_to< vec >::from(my_sampler.mu_count) / R,
// //       Named("S_acceptance_rate") = conv_to< vec >::from(my_sampler.S_count) / R,
// //       Named("m_acceptance_rate") = conv_to< vec >::from(my_sampler.m_count) / R,
// //       Named("t_df_acceptance_rate") = conv_to< vec >::from(my_sampler.t_df_count) / R,
// //       Named("alloc") = alloc,
// //       Named("observed_likelihood") = observed_likelihood,
// //       Named("complete_likelihood") = complete_likelihood,
// //       Named("BIC") = BIC_record,
// //       Named("batch_corrected_data") = batch_corrected_data
// //     )
// //   );
// //   
// // };
// // 
// 
// 
// 
// 
// 
// // //' @title Mixture model
// // //' @description Performs MCMC sampling for a mixture model.
// // //' @param X The data matrix to perform clustering upon (items to cluster in rows).
// // //' @param K The number of components to model (upper limit on the number of clusters found).
// // //' @param labels Vector item labels to initialise from.
// // //' @param dataType Int, 0: independent Gaussians, 1: Multivariate normal, or 2: Categorical distributions.
// // //' @param R The number of iterations to run for.
// // //' @param thin thinning factor for samples recorded.
// // //' @param concentration Vector of concentrations for mixture weights (recommended to be symmetric).
// // //' @return Named list of the matrix of MCMC samples generated (each row 
// // //' corresponds to a different sample) and BIC for each saved iteration.
// // // [[Rcpp::export]]
// // Rcpp::List sampleMixtureModel (
// //   arma::mat X,
// //   arma::uword K,
// //   arma::uword B,
// //   int dataType,
// //   arma::uvec labels,
// //   arma::uvec batch_vec,
// //   double mu_proposal_window,
// //   double cov_proposal_window,
// //   double m_proposal_window,
// //   double S_proposal_window,
// //   double t_df_proposal_window,
// //   double phi_proposal_window,
// //   double rho,
// //   double theta,
// //   double lambda,
// //   arma::uword R,
// //   arma::uword thin,
// //   arma::vec concentration,
// //   bool verbose = true,
// //   bool doCombinations = false,
// //   bool printCovariance = false
// // ) {
// //   
// //   // Set the random number
// //   std::default_random_engine generator(seed);
// //   
// //   // Declare the factory
// //   samplerFactory my_factory;
// //   
// //   // Convert from an int to the samplerType variable for our Factory
// //   samplerFactory::samplerType val = static_cast<samplerFactory::samplerType>(dataType);
// //   
// //   // Make a pointer to the correct type of sampler
// //   std::unique_ptr<sampler> sampler_ptr = my_factory.createSampler(val,
// //     K,
// //     B,
// //     mu_proposal_window,
// //     cov_proposal_window,
// //     m_proposal_window,
// //     S_proposal_window,
// //     t_df_proposal_window,
// //     phi_proposal_window,
// //     rho,
// //     theta,
// //     lambda,
// //     labels,
// //     batch_vec,
// //     concentration,
// //     X
// //   );
// //   
// //   // The output matrix
// //   arma::umat class_record(floor(R / thin), X.n_rows);
// //   class_record.zeros();
// //   
// //   // We save the BIC at each iteration
// //   arma::vec BIC_record = arma::zeros<arma::vec>(floor(R / thin));
// //   
// //   arma::uword save_int=0;
// //   
// //   // Sampler from priors (this is unnecessary)
// //   sampler_ptr->sampleFromPriors();
// //   
// //   // Iterate over MCMC moves
// //   for(arma::uword r = 0; r < R; r++){
// //     
// //     sampler_ptr->updateWeights();
// //     sampler_ptr->sampleParameters();
// //     sampler_ptr->updateAllocation();
// //     
// //     // Record results
// //     if((r + 1) % thin == 0){
// //       
// //       // Update the BIC for the current model fit
// //       sampler_ptr->calcBIC();
// //       BIC_record( save_int ) = sampler_ptr->BIC; 
// //       
// //       // Save the current clustering
// //       class_record.row( save_int ) = sampler_ptr->labels.t();
// //       save_int++;
// //     }
// //   }
// //   return(List::create(Named("samples") = class_record, Named("BIC") = BIC_record));
// // };
// // 
// // 
// // //' @title Mixture model
// // //' @description Performs MCMC sampling for a mixture model.
// // //' @param X The data matrix to perform clustering upon (items to cluster in rows).
// // //' @param K The number of components to model (upper limit on the number of clusters found).
// // //' @param labels Vector item labels to initialise from.
// // //' @param fixed Binary vector of the items that are fixed in their initial label.
// // //' @param dataType Int, 0: independent Gaussians, 1: Multivariate normal, or 2: Categorical distributions.
// // //' @param R The number of iterations to run for.
// // //' @param thin thinning factor for samples recorded.
// // //' @param concentration Vector of concentrations for mixture weights (recommended to be symmetric).
// // //' @return Named list of the matrix of MCMC samples generated (each row 
// // //' corresponds to a different sample) and BIC for each saved iteration.
// // // [[Rcpp::export]]
// // Rcpp::List sampleSemisupervisedMixtureModel (
// //     arma::mat X,
// //     arma::uword K,
// //     arma::uvec labels,
// //     arma::uvec fixed,
// //     int dataType,
// //     arma::uword R,
// //     arma::uword thin,
// //     arma::vec concentration,
// //     arma::uword seed
// // ) {
// //   
// //   // Set the random number
// //   std::default_random_engine generator(seed);
// //   
// //   // Declare the factory
// //   semisupervisedSamplerFactory my_factory;
// //   
// //   // Convert from an int to the samplerType variable for our Factory
// //   semisupervisedSamplerFactory::samplerType val = static_cast<semisupervisedSamplerFactory::samplerType>(dataType);
// //   
// //   // Make a pointer to the correct type of sampler
// //   std::unique_ptr<sampler> sampler_ptr = my_factory.createSemisupervisedSampler(val,
// //                                                                                 K,
// //                                                                                 labels,
// //                                                                                 concentration,
// //                                                                                 X,
// //                                                                                 fixed);
// //   
// //   // The output matrix
// //   arma::umat class_record(floor(R / thin), X.n_rows);
// //   class_record.zeros();
// //   
// //   // We save the BIC at each iteration
// //   arma::vec BIC_record = arma::zeros<arma::vec>(floor(R / thin));
// //   
// //   arma::uword save_int=0;
// //   
// //   // Sampler from priors (this is unnecessary)
// //   sampler_ptr->sampleFromPriors();
// //   
// //   // Iterate over MCMC moves
// //   for(arma::uword r = 0; r < R; r++){
// //     
// //     sampler_ptr->updateWeights();
// //     sampler_ptr->sampleParameters();
// //     sampler_ptr->updateAllocation();
// //     
// //     // Record results
// //     if((r + 1) % thin == 0){
// //       
// //       // Update the BIC for the current model fit
// //       sampler_ptr->calcBIC();
// //       BIC_record( save_int ) = sampler_ptr->BIC; 
// //       
// //       // Save the current clustering
// //       class_record.row( save_int ) = sampler_ptr->labels.t();
// //       save_int++;
// //     }
// //   }
// //   return(List::create(Named("samples") = class_record, Named("BIC") = BIC_record));
// // };
