// mvnSampler.cpp
// =============================================================================
//
// [[Rcpp::depends(BH)]]
//
// included dependencies
# include <RcppArmadillo.h>
# include <testthat.h>
# include "mvtSampler.h"
# include <iostream>
#include <boost/math/distributions/normal.hpp>

// =============================================================================
// namespace
using namespace Rcpp ;
using namespace arma ;

// =============================================================================
// mvnSampler unit tests

// [[Rcpp::export]]
arma::vec my_main() {
  uword K = 2, B = 3;
  double
    mu_proposal_window = 0.5,
      cov_proposal_window = 200,
      m_proposal_window = 0.4,
      S_proposal_window = 100,
      t_df_proposal_window = 35;

  uvec labels(10), batch_vec(10);
  vec concentration(K);
  // mat X(10, 1);

  labels = {0, 1, 1, 0, 1, 1, 1, 0, 0, 1};
  batch_vec = {0, 1, 0, 1, 0, 1, 1, 2, 2, 2};
  concentration = {1.0, 1.0};


  // mat A = { 1, 2 };
  //
  // std::cout << A;

  arma::mat Y = { 7.2,
    3.1,
    2.2,
    9.8,
    2.3,
    3.8,
    3.3,
    5.2,
    6.8,
    1.3
  }, X = Y.t();

  mvtSampler toy_sampler(
      K,
      B,
      mu_proposal_window,
      cov_proposal_window,
      m_proposal_window,
      S_proposal_window,
      t_df_proposal_window,
      labels,
      batch_vec,
      concentration,
      X
  );

  double val1 = 0.0,
    val2 = 0.0,
    val3 = 0.0,
    val4 = 0.0,
    val5 = 0.0,
    val6 = 0.0,
    val7 = 0.0,
    val8 = 0.0,
    val9 = 0.0,
    val10 = 0.0,
    val11 = 0.0,
    val12 = 0.0;

  vec mu_0 = {3.0}, mu_1 = {7.2};
  mat m = { {0.0, 1.0, -1.0} },
    mean_sum_0 = m + 3.0,
    mean_sum_1 = m + 7.2;
  
  // Initialise some mean vectors
  toy_sampler.mu.col(0) = {3.0};
  toy_sampler.mu.col(1) = {7.2};
  
  toy_sampler.m.col(0) = {0.0};
  toy_sampler.m.col(1) = {1.0};
  toy_sampler.m.col(2) = {-1.0};
  
  // Initialise some covariance matrices;
  toy_sampler.cov.slice(0) = {1.0};
  toy_sampler.cov.slice(1) = {0.7};
  
  toy_sampler.S.col(0) = {1.2};
  toy_sampler.S.col(1) = {1.3};
  toy_sampler.S.col(2) = {1.5};
  
  toy_sampler.t_df(0) = 13;
  toy_sampler.t_df(1) = 49;

  // Matrix combinations
  toy_sampler.matrixCombinations();

  val1 = toy_sampler.muLogKernel(0, mu_0, mean_sum_0);
  val2 = toy_sampler.muLogKernel(1, mu_1, mean_sum_1);

  // boost::math::normal boost_norm_b0c0(
  //     as_scalar(mean_sum_0.col(0)),
  //     as_scalar(toy_sampler.cov_comb.slice(0))
  //   ), boost_norm_b1c0(
  //       as_scalar(mean_sum_0.col(1)),
  //       as_scalar(toy_sampler.cov_comb.slice(1))
  //   ), boost_norm_b2c0(
  //       as_scalar(mean_sum_0.col(2)),
  //       as_scalar(toy_sampler.cov_comb.slice(2))
  //   ), boost_norm_b0c1(
  //       as_scalar(mean_sum_1.col(0)),
  //       as_scalar(toy_sampler.cov_comb.slice(3))
  //   ), boost_norm_b1c1(
  //       as_scalar(mean_sum_1.col(1)),
  //       as_scalar(toy_sampler.cov_comb.slice(4))
  //   ), boost_norm_b2c1(
  //       as_scalar(mean_sum_1.col(2)),
  //       as_scalar(toy_sampler.cov_comb.slice(5))
  //   );
  //
  // boost::math::normal prior_norm_0(as_scalar(toy_sampler.xi),
  //   as_scalar(toy_sampler.cov.slice(0))
  // ), prior_norm_1(as_scalar(toy_sampler.xi),
  //   as_scalar(toy_sampler.cov.slice(1))
  // );
  //
  // double prior_contribution_0
// 
//   std::cout << val1 << "\n\n" << val2;

  // double val1_expect= -22.02085, val2_expect = -78.48748;

  // For the covariance combination
  // toy_sampler.matrixCombinations();

  // val1 = toy_sampler.muLogKernel(0, mu_0, mean_sum_0);
  // val2 = toy_sampler.muLogKernel(1, mu_1, mean_sum_1);

  // double val1_expect= -22.02085, val2_expect = -78.48748;

  // test_that("mu log posterior kernel") {
  //   expect_true(compareDoubles(val1, -22.02085, 1e-5));
  //   expect_true(compareDoubles(val2, -78.48748, 1e-5));
  // }

  val3 = toy_sampler.mLogKernel(0, m.col(0), toy_sampler.mean_sum.cols(0, 1) );
  val4 = toy_sampler.mLogKernel(1, m.col(1), toy_sampler.mean_sum.cols(2, 3) );
  val5 = toy_sampler.mLogKernel(2, m.col(2), toy_sampler.mean_sum.cols(4, 5) );

  // val3 = -10.91562
  // val4 = -54.2134
  // val5 = -22.39516
  // return val1;
  
  val6 = toy_sampler.covLogKernel(0, 
                                  toy_sampler.cov.slice(0), 
                                  toy_sampler.cov_log_det(0),
                                  toy_sampler.cov_inv.slice(0),
                                  toy_sampler.cov_comb_log_det.row(0).t(),
                                  toy_sampler.cov_comb_inv.slices(0, 2)
                                  );
  val7 = toy_sampler.covLogKernel(1, 
                                  toy_sampler.cov.slice(1), 
                                  toy_sampler.cov_log_det(1),
                                  toy_sampler.cov_inv.slice(1),
                                  toy_sampler.cov_comb_log_det.row(1).t(),
                                  toy_sampler.cov_comb_inv.slices(3, 5)
  );
  
  // -32.92946
  // -78.71547

  // vec my_out = { val1, val2 };
  // 
  // sLogKernel(arma::uword b, 
  //            arma::vec S_b, 
  //            arma::vec cov_comb_log_det,
  //            arma::cube cov_comb_inv)
  
  val8 = toy_sampler.sLogKernel(0,
                                  toy_sampler.S.col(0),
                                  toy_sampler.cov_comb_log_det.col(0),
                                  toy_sampler.cov_comb_inv.slices(0, 3)
  );

  val9 = toy_sampler.sLogKernel(1,
                                  toy_sampler.S.col(1),
                                  toy_sampler.cov_comb_log_det.col(1),
                                  toy_sampler.cov_comb_inv.slices(1, 4)
  );

  val10 = toy_sampler.sLogKernel(2,
                                  toy_sampler.S.col(2),
                                  toy_sampler.cov_comb_log_det.col(2),
                                  toy_sampler.cov_comb_inv.slices(2, 5)
  );
  
  
  val11 = toy_sampler.dfLogKernel(0, toy_sampler.t_df(0), toy_sampler.pdf_coef(0));
  val12 = toy_sampler.dfLogKernel(1, toy_sampler.t_df(1), toy_sampler.pdf_coef(1));
  
  
  
  vec my_out = { val1, val2, val3, val4, val5, val6, val7, val8, val9, val10, val11, val12 };
  
  // -22.017362
  // -63.961374
  // -8.802964
  // -38.648574
  // -20.210158
  // -22.925973
  // -64.189365
  // -36.077722
  // -34.990054
  // -25.083692
  // -20.553550
  // -62.930631
  
  return my_out;
};