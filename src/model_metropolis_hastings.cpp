# include <RcppArmadillo.h>
# include <math.h> 
# include <string>
# include <iostream>

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp ;

// double logGamma(double x){
//   if( x == 1 ) {
//     return 0;
//   } 
//   return (std::log(x - 1) + logGamma(x - 1));
// }

double gammaLogLikelihood(double x, double shape, double scale){
  return ((shape - 1) * x - scale * x);
};

double invGammaLogLikelihood(double x, double shape, double scale) {
  return ( (-shape - 1) * x - scale / x );
};

double logNormalLogProbability(double x, double mu, double sigma2) {
  return(-std::log(sigma2) - std::log(x) - (1 / sigma2) * std::pow((std::log(x) - mu), 2.0));
};

class sampler {
  
private:
  // virtual arma::vec itemLogLikelihood(arma::vec x) { return 0.0; }
  // void updateAllocation() { return; }
  
public:
  arma::uword K, B, N, P, K_occ, accepted = 0;
  double model_likelihood = 0.0, BIC = 0.0, model_score = 0.0;
  arma::uvec labels, N_k, batch_vec, N_b;
  arma::vec concentration, w, ll, likelihood;
  arma::umat members;
  arma::mat X, alloc;
  
  // Parametrised class
  sampler(
    arma::uword _K,
    arma::uword _B,
    arma::uvec _labels,
    arma::uvec _batch_vec,
    arma::vec _concentration,
    arma::mat _X)
  {
    
    K = _K;
    B = _B;
    labels = _labels;
    batch_vec = _batch_vec;
    concentration = _concentration;
    X = _X;
    
    // Dimensions
    N = X.n_rows;
    P = X.n_cols;
    
    // std::cout << "\nN: " << N << "\nP: " << P << "\n\n";
    
    // Class populations
    N_k = arma::zeros<arma::uvec>(_K);
    N_b = arma::zeros<arma::uvec>(_B);
    
    // The batch numbers won't ever change, so let's count them now
    for(arma::uword b = 0; b < B; b++){
      N_b(b) = arma::sum(batch_vec == b);
    }
    
    // Weights
    // double x, y;
    w = arma::zeros<arma::vec>(_K);
    
    // for(arma::uword k = 0; k < K; k++){
    //   x = arma::randg(1, 1.0 / concentration(k));
    //   y = arma::randg(1, 1.0 / concentration(k));
    //   w(k) = (1 - sum(w)) * x/(x + y);
    // }
    
    // Log likelihood (individual and model)
    ll = arma::zeros<arma::vec>(_K);
    likelihood = arma::zeros<arma::vec>(N);
    
    // Class members
    members.set_size(N, _K);
    members.zeros();
    
    // Allocation probability matrix (only makes sense in predictive models)
    alloc.set_size(N, _K);
    alloc.zeros();
  };
  
  // Destructor
  virtual ~sampler() { };
  
  // Virtual functions are those that should actual point to the sub-class
  // version of the function.
  // Print the sampler type.
  virtual void printType() {
    std::cout << "\nType: NULL.\n";
  }
  
  // Functions required of all mixture models
  virtual void updateWeights(){
    
    double a = 0.0;
    
    for (arma::uword k = 0; k < K; k++) {
      
      // Find how many labels have the value
      members.col(k) = labels == k;
      N_k(k) = arma::sum(members.col(k));
      
      // Update weights by sampling from a Gamma distribution
      a  = concentration(k) + N_k(k);
      w(k) = arma::randg( arma::distr_param(a, 1.0) );
    }
    
    // Convert the cluster weights (previously gamma distributed) to Beta
    // distributed by normalising
    w = w / arma::sum(w);
    
  };
  
  virtual void updateAllocation() {
    
    double u = 0.0;
    arma::uvec uniqueK;
    arma::vec comp_prob(K);
    
    for(arma::uword n = 0; n < N; n++){
      
      ll = itemLogLikelihood(X.row(n).t(), batch_vec(n));
    
      // Update with weights
      comp_prob = ll + log(w);
      
      // Normalise and overflow
      comp_prob = exp(comp_prob - max(comp_prob));
      comp_prob = comp_prob / sum(comp_prob);
      
      // Prediction and update
      u = arma::randu<double>( );
      
      labels(n) = sum(u > cumsum(comp_prob));
      alloc.row(n) = comp_prob.t();
      
      // Record the likelihood of the item in it's allocated component
      likelihood(n) = ll(labels(n));
    }
    
    // The model log likelihood
    model_likelihood = arma::accu(likelihood);
    
    // Number of occupied components (used in BIC calculation)
    uniqueK = arma::unique(labels);
    K_occ = uniqueK.n_elem;
  };
  
  // void predictLabel(arma::uword n, arma::vec ll) {
  //
  //   arma::uword u = 0;
  //   arma::vec comp_prob = ll + log(w);
  //
  //   // Normalise and overflow
  //   comp_prob = exp(comp_prob - max(comp_prob));
  //   comp_prob = comp_prob / sum(comp_prob);
  //
  //   // Prediction and update
  //   u = arma::randu<double>( );
  //   labels(n) = sum(u > cumsum(comp_prob));
  //   alloc.row(n) = comp_prob.t();
  //
  // };
  
  virtual void metropolisStep(){};
  virtual void sampleFromPriors() {};
  virtual void sampleParameters(){};
  virtual void calcBIC(){};
  virtual arma::vec itemLogLikelihood(arma::vec x, arma::uword b) { return arma::vec(); }
  
};


//' @name gaussianSampler
//' @title Gaussian mixture type
//' @description The sampler for a mixture of Gaussians, where each feature is
//' assumed to be independent (i.e. a multivariate Normal with a diagonal 
//' covariance matrix).
//' @field new Constructor \itemize{
//' \item Parameter: K - the number of components to model
//' \item Parameter: labels - the initial clustering of the data
//' \item Parameter: concentration - the vector for the prior concentration of 
//' the Dirichlet distribution of the component weights
//' \item Parameter: X - the data to model
//' }
//' @field printType Print the sampler type called.
//' @field updateWeights Update the weights of each component based on current 
//' clustering.
//' @field updateAllocation Sample a new clustering. 
//' @field sampleFromPrior Sample from the priors for the Gaussian density.
//' @field calcBIC Calculate the BIC of the model.
//' @field itemLogLikelihood Calculate the likelihood of a given data point in each
//' component. \itemize{
//' \item Parameter: point - a data point.
//' }
class gaussianSampler: virtual public sampler {
  
public:
  
  double xi, kappa, alpha, beta, g, h, a, delta, lambda, rho, theta, proposal_window, proposal_window_for_logs;
  // arma::vec beta;
  arma::mat mu, mu_proposed, m, m_proposed, tau, tau_proposed, t, t_proposed;
  
  using sampler::sampler;
  
  // Parametrised
  gaussianSampler(
    arma::uword _K,
    arma::uword _B,
    double _proposal_window,
    double _proposal_window_for_logs,
    arma::uvec _labels,
    arma::uvec _batch_vec,
    arma::vec _concentration,
    arma::mat _X
  ) : sampler(_K,
      _B,
      _labels,
      _batch_vec,
      _concentration,
      _X) {
    
    double data_range = X.max() - X.min(), data_range_inv = std::pow(1.0 / data_range, 2);
    
    xi = arma::accu(X)/(N * P);
    kappa = 0.01;
    alpha = 0.5 * (2 + P);
    beta = arma::stddev(arma::vectorise(X)) / std::pow(K, 2);
    
    g = 0.2;
    a = 10;
    
    delta = 0.0;
    lambda = 0.01;
    // lambda = 1.0 / data_range;
    rho = 2.0; // 0.5 * (2 + P);
    theta = 2.0; // arma::stddev(arma::vectorise(X)) / std::pow(K, 2);
    
    proposal_window = _proposal_window;
    proposal_window_for_logs = _proposal_window_for_logs;
    
    // kappa =  data_range_inv;
    h = a * data_range_inv;
    
    // beta.set_size(P);
    // beta.zeros(); 
    
    mu.set_size(P, K);
    mu.zeros();
    
    mu_proposed.set_size(P, K);
    mu_proposed.zeros();
    
    tau.set_size(P, K);
    tau.zeros();
    
    tau_proposed.set_size(P, K);
    tau_proposed.zeros();
    
    m.set_size(P, B);
    m.zeros();
    
    m_proposed.set_size(P, B);
    m_proposed.zeros();
    
    t.set_size(P, B);
    t.zeros();
    
    t_proposed.set_size(P, B);
    t_proposed.zeros();
    
    
  }
  
  // Destructor
  virtual ~gaussianSampler() { };
  
  // Print the sampler type.
  virtual void printType() {
    std::cout << "\nType: Gaussian.\n";
  }
  
  // Parameters for the mixture model. The priors are empirical and follow the
  // suggestions of Richardson and Green <https://doi.org/10.1111/1467-9868.00095>.
  void sampleFromPriors() {
    for(arma::uword p = 0; p < P; p++){
      // beta(p) = arma::randg<double>( arma::distr_param(g, 1.0 / h) );
      for(arma::uword b = 0; b < B; b++){
        t(p, b) = arma::randg<double>( arma::distr_param(rho, 1.0 / theta ) );
        m(p, b) = arma::randn<double>() / (t(p, b) * lambda ) + delta;
      }
      for(arma::uword k = 0; k < K; k++){
        // tau(p, k) = 1.0 / arma::randg<double>( arma::distr_param(alpha, 1.0 / arma::as_scalar(beta(p))) );
        tau(p, k) = arma::randg<double>( arma::distr_param(alpha, 1.0 / beta) );
        mu(p, k) = arma::randn<double>() / ( tau(p, k) * kappa ) + xi;
      }
      
    }
  }
  
  
  // Sample beta
  // void updateBeta(){
  //   
  //   double a = g + K * alpha;
  //   double b = 0.0;
  //   
  //   for(arma::uword p = 0; p < P; p++){
  //     b = h + arma::accu(tau.row(p));
  //     beta(p) = arma::randg<double>( arma::distr_param(a, 1.0 / b) );
  //   }
  // }
  // 
  // Metropolis prosposal
  void proposeNewParameters() {
    
    // std::cout << "\nProposing.\n";
    
    for(arma::uword p = 0; p < P; p++){
      
      for(arma::uword b = 0; b < B; b++){
        // t_proposed(p, b) = std::exp((arma::randn() / proposal_window_for_logs) + t(p, b));
        t_proposed(p, b) = arma::randg(arma::distr_param(proposal_window * t(p,b), 1.0 / proposal_window));
        m_proposed(p, b) = (arma::randn() * proposal_window) + m(p, b);
      }
      
      // std::cout << "\nProposing to components.\n";
      
      for(arma::uword k = 0; k < K; k++){
        // tau_proposed(p, k) = std::exp((arma::randn() / proposal_window_for_logs) + tau(p, k));
        tau_proposed(p, k) = arma::randg(arma::distr_param(proposal_window * tau(p, k), 1.0 / proposal_window));
        mu_proposed(p, k) = (arma::randn() * proposal_window) + mu(p, k);
      }
      
    }
    // std::cout << "\nProposed.\n";
  }
  
  // double modelLogLikelihood(arma::mat mu, 
  //                        arma::mat tau,
  //                        arma::mat m,
  //                        arma::mat t) {
  //   
  //   double model_log_likelihood = 0;
  //   arma::uword c_n, b_n;
  //   arma::rowvec x_n;
  // 
  //   for(arma::uword n = 0; n < N; n++){
  //     c_n = labels(n);
  //     b_n = batch_vec(n);
  //     x_n = X.row(n);
  //     for(arma::uword p = 0; p < P; p++){
  // 
  //       model_log_likelihood += -0.5 * (std::log(2) + std::log(PI)
  //                                   + std::log(arma::as_scalar(tau(p, c_n)))
  //                                   + std::log(arma::as_scalar(t(p, b_n)))
  //                                   + arma::as_scalar(tau(p, c_n) 
  //                                     * t(p, b_n)
  //                                     * pow((x_n(p) - (mu(p, c_n) + m(p, b_n))), 2.0)
  //                                   )
  //                                 );
  //                                   
  //     }
  //     
  //   }
  //   
  //   return model_log_likelihood;
  //   
  // };
  
  double priorLogProbability(arma::mat mu, 
                          arma::mat tau,
                          arma::mat m,
                          arma::mat t){
    
    double prior_score = 0.0;
    
    for(arma::uword p = 0; p < P; p++){
      
      for(arma::uword b = 0; b < B; b++){
        prior_score += invGammaLogLikelihood(t(p, b), rho, 1.0 / theta);
        prior_score += arma::log_normpdf(m(p, b), delta, lambda * t(p, b));
      }
      for(arma::uword k = 0; k < K; k++){
        // tau(p, k) = 1.0 / arma::randg<double>( arma::distr_param(alpha, 1.0 / arma::as_scalar(beta(p))) );
        prior_score += invGammaLogLikelihood(tau(p, k), alpha, 1.0 / beta);
        prior_score += arma::log_normpdf(mu(p, k), xi, kappa * tau(p, k));
      }
      
    }
    return prior_score;
  };
  
  double proposalScore(arma::mat x, arma::mat y, double window, arma::uword dim){
    
    double score = 0.0;
    
    for(arma::uword p = 0; p < P; p++) {
      
      for(arma::uword j = 0; j < dim; j++){
        // score += logNormalLogProbability(x(p, j), y(p, j), window);
        score += invGammaLogLikelihood(x(p, j), window, y(p, j)  * (window - 1.0));
      }
    }
    return score;
  }
  
  arma::vec itemLogLikelihood(arma::vec item, arma::uword b) {
    
    arma::vec ll(K);
    ll.zeros();

    for(arma::uword k = 0; k < K; k++){
      for (arma::uword p = 0; p < P; p++){
        ll(k) += -0.5*(std::log(2) + std::log(PI) - std::log(arma::as_scalar(tau(p, k))) - std::log(arma::as_scalar(t(p, b)))+ arma::as_scalar((tau(p, k) * t(p, b)) *  std::pow(item(p) - (mu(p, k) + m(p, b) ), 2.0))); 
      }
    }
    return ll;
  };
  
  void calcBIC(){
    
    arma::uword n_param = (P + P) * K_occ;
    BIC = n_param * std::log(N) - 2 * model_likelihood;
    
  };
  
  double mKernel(arma::uword b, arma::vec m_b) {
    
    double score = 0.0, score_p = 0.0;
    
    arma::uvec batch_ind = arma::find(batch_vec == b);
    for (auto& n : batch_ind) {
      for(arma::uword p = 0; p < P; p++) {
        score_p += tau(p, labels(n)) * std::pow((X(n, p) - (mu(p, labels(n)) + m_b(p))), 2.0);
      }
    }
    for(arma::uword p = 0; p < P; p++) {
      score_p += lambda * std::pow(m_b(p) - delta, 2.0);
      score += -0.5 * t(p, b) *  score_p;
    }
  return score;
  };
  
  double tKernel(arma::uword b, arma::vec t_b) {
    
    double score = 0.0, score_p = 0.0;
    arma::uvec batch_ind = find(batch_vec == b);
    for (auto& n : batch_ind) {
      for(arma::uword p = 0; p < P; p++) {
        score_p += tau(p, labels(n)) * std::pow((X(n, p) - (mu(p, labels(n)) + m(p, b))), 2.0);
      }
    }
    for(arma::uword p = 0; p < P; p++) {
    score_p += lambda * std::pow(m(p, b) - delta, 2.0) + 2 * theta;
    score +=  0.5 * ((N_b(b) + 2 * rho - 1) * std::log(t_b(p)) - t_b(p) * score_p);
    }
    return score;
  };
  
  double muLogKernel(arma::uword k, arma::vec mu_k) {
    
    double score = 0.0, score_p = 0.0;
    arma::uvec cluster_ind = arma::find(labels == k);
    for (auto& n : cluster_ind) {
      for(arma::uword p = 0; p < P; p++) {
        score_p +=  t(p, batch_vec(n))* std::pow((X(n, p) - (mu_k(p) + m(p, batch_vec(n)))), 2.0);
      }
    }
    for(arma::uword p = 0; p < P; p++) {
      score_p += kappa * std::pow(mu_k(p) - xi, 2.0);
      score += -0.5 * tau(p, k) *  score_p;
    }
    return score;
  };
  
  double tauKernel(arma::uword k, arma::vec tau_k) {
    
    double score = 0.0, score_p = 0.0;
    arma::uvec cluster_ind = arma::find(labels == k);
    for (auto& n : cluster_ind) {
      for(arma::uword p = 0; p < P; p++) {
        score_p += t(p, batch_vec(n)) * std::pow((X(n, p) - (mu(p, k) + m(p, batch_vec(n)))), 2.0);
      }
    }
    for(arma::uword p = 0; p < P; p++) {
      score_p += kappa * std::pow(mu(p, k) - xi, 2.0) + 2 * beta;
      score +=  0.5 * ((N_k(k) + 2 * alpha - 1) * std::log(tau_k(p)) - tau_k(p) * score_p);
    }
    return score;
  };
  
  void batchScaleMetorpolis() {
    
    double u = 0.0, proposed_model_score = 0.0, acceptance_prob = 0.0, current_model_score = 0.0;
    
    for(arma::uword b = 0; b < B ; b++) {
      
      proposed_model_score = 0.0, acceptance_prob = 0.0, current_model_score = 0.0;
      
      for(arma::uword p = 0; p < P; p++){
        t_proposed(p, b) = std::exp((arma::randn() * proposal_window_for_logs) + t(p, b));
        // t_proposed(p, b) = arma::randg(arma::distr_param(proposal_window_for_logs * t(p, b),  1.0 / proposal_window_for_logs));
        
        // Prior log probability is included in the kernel
        // proposed_model_score += gammaLogLikelihood(t_proposed(p, b), rho, theta);
        // current_model_score += gammaLogLikelihood(t(p, b), rho, theta);
        
        // Log probability under the proposal density
        proposed_model_score += logNormalLogProbability(t(p, b), t_proposed(p, b), proposal_window_for_logs);
        current_model_score += logNormalLogProbability(t_proposed(p, b), t(p, b), proposal_window_for_logs);
        
        // Assymetric proposal density
        // proposed_model_score += gammaLogLikelihood(t(p, b), proposal_window_for_logs * t_proposed(p, b), proposal_window_for_logs);
        // current_model_score += gammaLogLikelihood(t_proposed(p, b), proposal_window_for_logs * t(p, b), proposal_window_for_logs);

      }
      proposed_model_score += tKernel(b, t_proposed.col(b));
      current_model_score += tKernel(b, t.col(b));
      
      u = arma::randu();
      
      acceptance_prob = std::min(1.0, std::exp(proposed_model_score - current_model_score));
      
      if(u < acceptance_prob){
        t.col(b) = t_proposed.col(b);
        // t_score = proposed_model_score;
      }
      
    }

    
  };
  
  arma::vec batchScaleScore(arma::mat t) {
    
    arma::vec score(B);
    score.zeros();
    
    for(arma::uword b = 0; b < B ; b++) {
      // for(arma::uword p = 0; p < P; p++){
      //   score(b) += gammaLogLikelihood(t(p, b), rho, theta);
      // }
      score(b) += tKernel(b, t.col(b));
    }
    return score;
  };
  
  void batchShiftMetorpolis() {
    
    double u = 0.0, proposed_model_score = 0.0, acceptance_prob = 0.0, current_model_score = 0.0;
    
    current_model_score += model_score;
    
    for(arma::uword b = 0; b < B ; b++) {
      for(arma::uword p = 0; p < P; p++){
        m_proposed(p, b) = (arma::randn() / proposal_window) + m(p, b);
        // m_proposed(p, b) = 0.0; // 
        
        // Prior included in kernel
        // proposed_model_score += arma::log_normpdf(m_proposed(p, b), delta, t(p, b) / lambda );
        // current_model_score += arma::log_normpdf(m(p, b), delta, t(p, b) / lambda ); 
        
      }
      proposed_model_score = mKernel(b, m_proposed.col(b));
      current_model_score = mKernel(b, m.col(b));
      
      u = arma::randu();
      
      acceptance_prob = std::min(1.0, std::exp(proposed_model_score - current_model_score));
      
      if(u < acceptance_prob){
        m.col(b) = m_proposed.col(b);
        // m_score = proposed_model_score;
      }
      
    }
    
    
    
  };
  
  arma::vec batchShiftScore(arma::mat m) {
    
    arma::vec score(B);
    score.zeros();
    
    for(arma::uword b = 0; b < B ; b++) {
      // for(arma::uword p = 0; p < P; p++){
      //   score(b) += arma::log_normpdf(m(p, b), delta,  t(p, b) / lambda );
      // }
      score(b) += mKernel(b, m.col(b));
    }
    return score;
  };
  
  void clusterPrecisionMetropolis() {
    
    double u = 0.0, proposed_model_score = 0.0, acceptance_prob = 0.0, current_model_score = 0.0;
    
    // current_model_score += model_score;
    
    for(arma::uword k = 0; k < K ; k++) {
      
      acceptance_prob = 0.0, proposed_model_score = 0.0, current_model_score = 0.0;

      for(arma::uword p = 0; p < P; p++){
        if(N_k(k) == 0){
          tau(p, k) = arma::randg<double>( arma::distr_param(alpha, 1.0 / beta) );
        } else {
          tau_proposed(p, k) = std::exp((arma::randn() * proposal_window_for_logs) + tau(p, k));
          // tau_proposed(p, k) = arma::randg( arma::distr_param(proposal_window_for_logs * tau(p, k), proposal_window_for_logs));
          // tau_proposed(p, k) = 1.0;
          
          // Log probability under the proposal density
          proposed_model_score += logNormalLogProbability(tau(p, k), tau_proposed(p, k), proposal_window_for_logs);
          current_model_score += logNormalLogProbability(tau_proposed(p, k), tau(p, k), proposal_window_for_logs);
          
          // Asymmetric proposal density
          // proposed_model_score += gammaLogLikelihood(tau(p, k), proposal_window_for_logs * tau_proposed(p, k), proposal_window_for_logs);
          // current_model_score += gammaLogLikelihood(tau_proposed(p, k), proposal_window_for_logs * tau(p, k), proposal_window_for_logs);
        }
        
        // Prior log probability included in kernel
        // proposed_model_score += invGammaLogLikelihood(tau_proposed(p, k), alpha, 1.0 / beta);
        // current_model_score += invGammaLogLikelihood(tau(p, k), alpha, 1.0 / beta);
        
        // proposed_model_score += invGammaLogLikelihood(t(p, b), t_proposed(p, b) * window, window);
        // current_model_score += invGammaLogLikelihood(t_proposed(p, b), t(p, b) * window, window);
        
      }
      proposed_model_score += tauKernel(k, tau_proposed.col(k));
      current_model_score += tauKernel(k, tau.col(k));
      
      u = arma::randu();
      
      acceptance_prob = std::min(1.0, std::exp(proposed_model_score - current_model_score));
      
      if(u < acceptance_prob){
        tau.col(k) = tau_proposed.col(k);
        // t_score = proposed_model_score;
      }
    }
    
    
    
  };
  
  arma::vec clusterPrecisionScore(arma::mat tau) {
    
    arma::vec score(K);
    score.zeros();
    
    for(arma::uword k = 0; k < K ; k++) {
      // for(arma::uword p = 0; p < P; p++){
      //   score(k) += gammaLogLikelihood(tau(p, k), alpha, beta);
      // }
      score(k) += tauKernel(k, tau.col(k));
    }
    return score;
  };
  
  void clusterMeanMetropolis() {
    
    double u = 0.0, proposed_model_score = 0.0, acceptance_prob = 0.0, current_model_score = 0.0;
    
    // current_model_score += model_score;
    
    for(arma::uword k = 0; k < K ; k++) {
      
      // proposed_model_score = 0.0, acceptance_prob = 0.0, current_model_score = 0.0;
      
      for(arma::uword p = 0; p < P; p++){
        if(N_k(k) == 0){
            mu_proposed(p, k) = arma::randn<double>() / (tau(p, k) * kappa) + xi;
        } else {
          mu_proposed(p, k) = (arma::randn() * proposal_window) + mu(p, k);
        }
        
        // Prior log probability
        // proposed_model_score += arma::log_normpdf(mu_proposed(p, k), delta, tau(p, k) / kappa);
        // current_model_score += arma::log_normpdf(mu(p, k), delta, tau(p, k) / kappa ); 
          
      }
      // The prior is included in the kernel
      proposed_model_score = muLogKernel(k, mu_proposed.col(k));
      current_model_score = muLogKernel(k, mu.col(k));
      
      u = arma::randu();
      
      acceptance_prob = std::min(1.0, std::exp(proposed_model_score - current_model_score));
      
      if(u < acceptance_prob){
        mu.col(k) = mu_proposed.col(k);
        // mu_score = proposed_model_score;
      }
    }
      
    
    
  };
  
  arma::vec clusterMeanScore(arma::mat mu) {
    
    arma::vec score(K);
    score.zeros();
    
    for(arma::uword k = 0; k < K ; k++) {
      // for(arma::uword p = 0; p < P; p++){
      //   score(k) += arma::log_normpdf(mu(p, k), delta, tau(p, k) / kappa); 
      // }
      score(k) += muLogKernel(k, mu.col(k));
    }
    return score;
  };
  
  void metropolisStep() {
    
    // Metropolis step for cluster parameters
    clusterPrecisionMetropolis();
    clusterMeanMetropolis();
    
    // Metropolis step for batch parameters
    batchScaleMetorpolis();
    batchShiftMetorpolis();
    
  };
  
};

//' @name mvnSampler
//' @title Multivariate Normal mixture type
//' @description The sampler for the Multivariate Normal mixture model.
//' @field new Constructor \itemize{
//' \item Parameter: K - the number of components to model
//' \item Parameter: labels - the initial clustering of the data
//' \item Parameter: concentration - the vector for the prior concentration of 
//' the Dirichlet distribution of the component weights
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
class mvnSampler: virtual public sampler {
  
public:
  double kappa, nu, lambda, rho, theta, proposal_window_for_logs;
  arma::uvec mu_acceptance, cov_acceptance, m_acceptance;
  arma::vec xi, delta, mu_proposed, m_proposed, S_proposed, cov_log_det, mu_score, cov_score, m_score;
  arma::umat S_acceptance;
  arma::mat scale, mu, m, S, cov_proposed, cov_sum_log_det, mean_sum, S_score, proposal_window;
  arma::cube cov, cov_inv, cov_sum, cov_sum_inv;
  
  using sampler::sampler;
  
  mvnSampler(
    arma::uword _K,
    arma::uword _B,
    double _proposal_window,
    double _proposal_window_for_logs,
    arma::uvec _labels,
    arma::uvec _batch_vec,
    arma::vec _concentration,
    arma::mat _X
  ) : sampler(_K,
  _B,
  _labels,
  _batch_vec,
  _concentration,
  _X) {
    
    // Default values for hyperparameters
    kappa = 0.01;
    nu = P + 2;
    
    arma::mat mean_mat = arma::mean(_X, 0).t();
    xi = mean_mat.col(0);
    
    arma::mat scale_param = _X.each_row() - xi.t(); 
    arma::rowvec diag_entries = arma::sum(scale_param % scale_param, 0) / (N * pow(_K, 1.0 / (double)P));
    scale = arma::diagmat( diag_entries );
    
    // The mean of the prior distribution for the batch shift, m, parameter
    delta = arma::zeros<arma::vec>(P);
    lambda = 0.01;
    
    // The shape and scale of the prior for the batch scale, S
    rho = 2.0;
    theta = 2.0;
    
    // Set the size of the objects to hold the component specific parameters
    mu.set_size(P, K);
    mu.zeros();
    
    cov.set_size(P, P, K);
    cov.zeros();
    
    // Set the size of the objects to hold the batch specific parameters
    m.set_size(P, B);
    m.zeros();
    
    // We are assuming a diagonal structure in the batch scale
    S.set_size(P, B);
    S.zeros();
    
    // The variables to hold the batch/cluster specific proposed values
    mu_proposed = arma::zeros<arma::vec>(P);
    m_proposed = arma::zeros<arma::vec>(P);
    S_proposed = arma::zeros<arma::vec>(P);
    
    cov_proposed.set_size(K, K);
    cov_proposed.zeros();
    
    // Acceptance objects recording if the current values are accepted
    // Used to decide to update matrix products
    mu_acceptance = arma::ones<arma::uvec>(K);
    cov_acceptance = arma::ones<arma::uvec>(K);
    m_acceptance = arma::ones<arma::uvec>(B);
    S_acceptance.set_size(B, P);
    S_acceptance.ones();
    
    mu_score = arma::zeros<arma::vec>(K);
    cov_score = arma::zeros<arma::vec>(K);
    m_score = arma::zeros<arma::vec>(B);
    S_score.set_size(B, P);
    S_score.zeros();
    
    cov_log_det = arma::zeros<arma::vec>(K);
    
    cov_sum_log_det.set_size(K, B);
    cov_sum_log_det.zeros();
    
    mean_sum.set_size(P, K * B);
    mean_sum.zeros();
    
    cov_sum.set_size(P, P, K * B);
    cov_sum.zeros();
    
    cov_inv.set_size(P, P, K);
    cov_inv.zeros();
    
    cov_sum_inv.set_size(P, P, K * B);
    cov_sum_inv.zeros();
    
    // The proposal covariance
    proposal_window.set_size(P, P);
    proposal_window.eye();
    proposal_window.diag() *= _proposal_window;
  }
  
  
  // Destructor
  virtual ~mvnSampler() { };
  
  // Print the sampler type.
  virtual void printType() {
    std::cout << "\nType: MVN.\n";
  }
  
  void sampleFromPriors() {
    
    for(arma::uword k = 0; k < K; k++){
      // cov.slice(k) = arma::iwishrnd(scale, nu);
      cov.slice(k).eye(P, P);
      mu.col(k) = arma::mvnrnd(xi, (1.0/kappa) * cov.slice(k), 1);
    }
    for(arma::uword b = 0; b < B; b++){
      for(arma::uword p = 0; p < P; p++){
        // S(p, b) = 1.0 / arma::randg<double>( arma::distr_param(rho, 1.0 / theta ) );
        S(p, b) = 1.0;
        // m(p, b) = arma::randn<double>() * S(p, b) / lambda + delta(p);
        m(p, b) = 0.0;
      }
    }
    
    // std::cout << "\nCovariance cube:\n" << cov;
    // std::cout << "\n\nMean matrix:\n" << mu;
    // std::cout << "\n\nBatch mean matrix:\n" << m;
    // std::cout << "\n\nBatch covariance matrix:\n" << S;
    
  };
  
  // Update the common matrix manipulations to avoid recalculating N times
  void matrixCombinations() {
    for(arma::uword k = 0; k < K; k++) {
      if(cov_acceptance(k) == 1) {
        cov_inv.slice(k) = arma::inv(cov.slice(k));
        cov_log_det(k) = arma::log_det(cov.slice(k)).real();
      }
      for(arma::uword b = 0; b < B; b++) {
        if(cov_acceptance(k) == 1 || arma::any(S_acceptance.row(b))) {
          cov_sum.slice(k * B + b) = cov.slice(k) + arma::diagmat(S.col(b));
          cov_sum_log_det(k, b) = arma::log_det(cov_sum.slice(k * B + b)).real();
          cov_sum_inv.slice(k * B + b) = arma::inv(cov_sum.slice(k * B + b));
        }
        if(mu_acceptance(k) == 1 || m_acceptance(b)) {
          mean_sum.col(k * B + b) = mu.col(k) + m.col(b);
        }
      }
    }
    // std::cout << "\nCovariance inverse\n" << cov_inv;
    // std::cout << "\n\nCovariance log determinant\n" << cov_log_det;
    // std::cout << "\n\nCovariance sum\n" << cov_sum;
    // std::cout << "\n\nCovariance sum log determinant\n" << cov_sum_log_det;
    // std::cout << "\n\nCovariance sum inverse\n" << cov_sum_inv;
    // std::cout << "\n\nMean sum\n" << mean_sum;
  };
  
  arma::vec itemLogLikelihood(arma::vec item, arma::uword b) {
    
    double exponent = 0.0;
    
    arma::vec ll(K), dist_to_mean(P);
    ll.zeros();
    dist_to_mean.zeros();
    
    for(arma::uword k = 0; k < K; k++){
      
      // The exponent part of the MVN pdf
      dist_to_mean = item - mean_sum.col(k * B + b);
      exponent = arma::as_scalar(dist_to_mean.t() * cov_sum_inv.slice(k * B + b) * dist_to_mean);
      
      // Normal log likelihood
      ll(k) = -0.5 *(cov_sum_log_det(k * B + b) + exponent + (double) P * log(2.0 * M_PI)); 
      
    }
    
    return(ll);
  };
  
  void calcBIC(){
    
    arma::uword n_param = (P + P * (P + 1) * 0.5) * K_occ;
    BIC = n_param * std::log(N) - 2 * model_likelihood;
    
  };
  
  double mLogKernel(arma::uword b, arma::vec m_b) {
    
    arma::uword k = 0;
    double score = 0.0;
    
    arma::uvec batch_ind = arma::find(batch_vec == b);
    for (auto& n : batch_ind) {
      k = labels(n);
      score += arma::as_scalar((X.row(n).t() - mean_sum.col(k * B + b)).t() * cov_sum_inv.slice(k * B + b) * (X.row(n).t() - mean_sum.col(k * B + b)));
    }
    for(arma::uword p = 0; p < P; p++) {
      score += lambda * std::pow(m_b(p) - delta(p), 2.0) / S(p, b);
    }
    return score;
  };
  
  // double sLogKernel(arma::uword b, arma::vec t_b) {
  //   
  //   double score = 0.0, score_p = 0.0;
  //   arma::uvec batch_ind = find(batch_vec == b);
  //   for (auto& n : batch_ind) {
  //     for(arma::uword p = 0; p < P; p++) {
  //       score_p += tau(p, labels(n)) * std::pow((X(n, p) - (mu(p, labels(n)) + m(p, b))), 2.0);
  //     }
  //   }
  //   for(arma::uword p = 0; p < P; p++) {
  //     score_p += lambda * std::pow(m(p, b) - delta, 2.0) + 2 * theta;
  //     score +=  0.5 * ((N_b(b) + 2 * rho - 1) * std::log(t_b(p)) - t_b(p) * score_p);
  //   }
  //   return score;
  // };
  
  double muLogKernel(arma::uword k, arma::vec mu_k) {
    
    arma::uword b = 0;
    double score = 0.0;
    arma::uvec cluster_ind = arma::find(labels == k);
    
    for (auto& n : cluster_ind) {
      b = batch_vec(n);
      score +=  arma::as_scalar((X.row(n).t() - mean_sum.col(k * B + b)).t() * cov_sum_inv.slice(k * B + b) * (X.row(n).t() - mean_sum.col(k * B + b)));
    }
    score += arma::as_scalar(kappa * (mu_k - xi).t() *  cov_inv.slice(k) * (mu_k - xi));
    return score;
  };
  
  double covLogKernel(arma::uword k, arma::mat cov_k) {
    
    arma::uword b = 0;
    double score = 0.0;
    arma::uvec cluster_ind = arma::find(labels == k);
    
    for (auto& n : cluster_ind) {
      b = batch_vec(n);
      score += arma::as_scalar(cov_sum_log_det(k, b) + (X.row(n).t() - mean_sum.col(k * B + b)).t() * cov_sum_inv.slice(k * B + b) * (X.row(n).t() - mean_sum.col(k * B + b)));
    }
    for(arma::uword p = 0; p < P; p++) {
      score +=  arma::as_scalar((nu + P + 2) * cov_log_det(k) + kappa * (mu.col(k) - xi).t() * cov_inv.slice(k) * (mu.col(k) - xi) + arma::trace(scale * cov_inv.slice(k)));
    }
    return score;
  };
  
  // void batchScaleMetorpolis() {
  //   
  //   double u = 0.0, proposed_model_score = 0.0, acceptance_prob = 0.0, current_model_score = 0.0;
  //   
  //   for(arma::uword b = 0; b < B ; b++) {
  //     
  //     proposed_model_score = 0.0, acceptance_prob = 0.0, current_model_score = 0.0;
  //     
  //     for(arma::uword p = 0; p < P; p++){
  //       t_proposed(p, b) = std::exp((arma::randn() * proposal_window_for_logs) + t(p, b));
  //       // t_proposed(p, b) = arma::randg(arma::distr_param(proposal_window_for_logs * t(p, b),  1.0 / proposal_window_for_logs));
  //       
  //       // Prior log probability is included in the kernel
  //       // proposed_model_score += gammaLogLikelihood(t_proposed(p, b), rho, theta);
  //       // current_model_score += gammaLogLikelihood(t(p, b), rho, theta);
  //       
  //       // Log probability under the proposal density
  //       proposed_model_score += logNormalLogProbability(t(p, b), t_proposed(p, b), proposal_window_for_logs);
  //       current_model_score += logNormalLogProbability(t_proposed(p, b), t(p, b), proposal_window_for_logs);
  //       
  //       // Assymetric proposal density
  //       // proposed_model_score += gammaLogLikelihood(t(p, b), proposal_window_for_logs * t_proposed(p, b), proposal_window_for_logs);
  //       // current_model_score += gammaLogLikelihood(t_proposed(p, b), proposal_window_for_logs * t(p, b), proposal_window_for_logs);
  //       
  //     }
  //     proposed_model_score += tKernel(b, t_proposed.col(b));
  //     current_model_score += tKernel(b, t.col(b));
  //     
  //     u = arma::randu();
  //     
  //     acceptance_prob = std::min(1.0, std::exp(proposed_model_score - current_model_score));
  //     
  //     if(u < acceptance_prob){
  //       t.col(b) = t_proposed.col(b);
  //       // t_score = proposed_model_score;
  //     }
  //     
  //   }
  //   
  //   
  // };
  // 
  // arma::vec batchScaleScore(arma::mat t) {
  //   
  //   arma::vec score(B);
  //   score.zeros();
  //   
  //   for(arma::uword b = 0; b < B ; b++) {
  //     // for(arma::uword p = 0; p < P; p++){
  //     //   score(b) += gammaLogLikelihood(t(p, b), rho, theta);
  //     // }
  //     score(b) += tKernel(b, t.col(b));
  //   }
  //   return score;
  // };
  // 
  // void batchShiftMetorpolis() {
  //   
  //   double u = 0.0, proposed_model_score = 0.0, acceptance_prob = 0.0, current_model_score = 0.0;
  //   
  //   current_model_score += model_score;
  //   
  //   for(arma::uword b = 0; b < B ; b++) {
  //     for(arma::uword p = 0; p < P; p++){
  //       m_proposed(p, b) = (arma::randn() / proposal_window) + m(p, b);
  //       // m_proposed(p, b) = 0.0; // 
  //       
  //       // Prior included in kernel
  //       // proposed_model_score += arma::log_normpdf(m_proposed(p, b), delta, t(p, b) / lambda );
  //       // current_model_score += arma::log_normpdf(m(p, b), delta, t(p, b) / lambda ); 
  //       
  //     }
  //     proposed_model_score = mKernel(b, m_proposed.col(b));
  //     current_model_score = mKernel(b, m.col(b));
  //     
  //     u = arma::randu();
  //     
  //     acceptance_prob = std::min(1.0, std::exp(proposed_model_score - current_model_score));
  //     
  //     if(u < acceptance_prob){
  //       m.col(b) = m_proposed.col(b);
  //       // m_score = proposed_model_score;
  //     }
  //     
  //   }
  //   
  //   
  //   
  // };
  // 
  // arma::vec batchShiftScore(arma::mat m) {
  //   
  //   arma::vec score(B);
  //   score.zeros();
  //   
  //   for(arma::uword b = 0; b < B ; b++) {
  //     // for(arma::uword p = 0; p < P; p++){
  //     //   score(b) += arma::log_normpdf(m(p, b), delta,  t(p, b) / lambda );
  //     // }
  //     score(b) += mKernel(b, m.col(b));
  //   }
  //   return score;
  // };
  
  // void clusterPrecisionMetropolis() {
  //   
  //   double u = 0.0, proposed_model_score = 0.0, acceptance_prob = 0.0, current_model_score = 0.0;
  //   
  //   // current_model_score += model_score;
  //   
  //   for(arma::uword k = 0; k < K ; k++) {
  //     
  //     acceptance_prob = 0.0, proposed_model_score = 0.0, current_model_score = 0.0;
  //     
  //     for(arma::uword p = 0; p < P; p++){
  //       if(N_k(k) == 0){
  //         tau(p, k) = arma::randg<double>( arma::distr_param(alpha, 1.0 / beta) );
  //       } else {
  //         tau_proposed(p, k) = std::exp((arma::randn() * proposal_window_for_logs) + tau(p, k));
  //         // tau_proposed(p, k) = arma::randg( arma::distr_param(proposal_window_for_logs * tau(p, k), proposal_window_for_logs));
  //         // tau_proposed(p, k) = 1.0;
  //         
  //         // Log probability under the proposal density
  //         proposed_model_score += logNormalLogProbability(tau(p, k), tau_proposed(p, k), proposal_window_for_logs);
  //         current_model_score += logNormalLogProbability(tau_proposed(p, k), tau(p, k), proposal_window_for_logs);
  //         
  //         // Asymmetric proposal density
  //         // proposed_model_score += gammaLogLikelihood(tau(p, k), proposal_window_for_logs * tau_proposed(p, k), proposal_window_for_logs);
  //         // current_model_score += gammaLogLikelihood(tau_proposed(p, k), proposal_window_for_logs * tau(p, k), proposal_window_for_logs);
  //       }
  //       
  //       // Prior log probability included in kernel
  //       // proposed_model_score += invGammaLogLikelihood(tau_proposed(p, k), alpha, 1.0 / beta);
  //       // current_model_score += invGammaLogLikelihood(tau(p, k), alpha, 1.0 / beta);
  //       
  //       // proposed_model_score += invGammaLogLikelihood(t(p, b), t_proposed(p, b) * window, window);
  //       // current_model_score += invGammaLogLikelihood(t_proposed(p, b), t(p, b) * window, window);
  //       
  //     }
  //     proposed_model_score += tauKernel(k, tau_proposed.col(k));
  //     current_model_score += tauKernel(k, tau.col(k));
  //     
  //     u = arma::randu();
  //     
  //     acceptance_prob = std::min(1.0, std::exp(proposed_model_score - current_model_score));
  //     
  //     if(u < acceptance_prob){
  //       tau.col(k) = tau_proposed.col(k);
  //       // t_score = proposed_model_score;
  //     }
  //   }
  //   
  //   
  //   
  // };
  // 
  // arma::vec clusterPrecisionScore(arma::mat tau) {
  //   
  //   arma::vec score(K);
  //   score.zeros();
  //   
  //   for(arma::uword k = 0; k < K ; k++) {
  //     // for(arma::uword p = 0; p < P; p++){
  //     //   score(k) += gammaLogLikelihood(tau(p, k), alpha, beta);
  //     // }
  //     score(k) += tauKernel(k, tau.col(k));
  //   }
  //   return score;
  // };
  
  void clusterMeanMetropolis() {
    
    double u = 0.0, proposed_model_score = 0.0, acceptance_prob = 0.0, current_score = 0.0;
    
    mu_acceptance.zeros();
    
    // std::cout << "\nMu score:\n" << mu_score << "\n\n";
    
    for(arma::uword k = 0; k < K ; k++) {
      
      current_score = mu_score(k);
      
      if(N_k(k) == 0){
        mu_proposed = arma::mvnrnd(xi, (1.0/kappa) * cov.slice(k), 1);
      } else {
        mu_proposed = arma::mvnrnd(mu.col(k), proposal_window, 1);
      }
      // The prior is included in the kernel
      proposed_model_score = muLogKernel(k, mu_proposed);
      // current_score = muLogKernel(k, mu.col(k));
      
      u = arma::randu();
      
      acceptance_prob = std::min(1.0, std::exp(proposed_model_score - current_score));
      
      mu_acceptance(k) = 0;
      if(u < acceptance_prob){
        mu.col(k) = mu_proposed;
        mu_acceptance(k) = 1;
        mu_score(k) = proposed_model_score;
      }
    };
    
    
    
  };
  
  arma::vec clusterMeanScore(arma::mat mu) {
    
    arma::vec score(K);
    score.zeros();
    
    for(arma::uword k = 0; k < K ; k++) {
      score(k) = muLogKernel(k, mu.col(k));
    }
    return score;
  };
  
  void modelScore() {
    
    for(arma::uword k = 0; k < K; k++) {
      mu_score(k) = muLogKernel(k, mu.col(k));
    }
    
  }
  
  void metropolisStep() {
    
    // Metropolis step for cluster parameters
    // clusterPrecisionMetropolis();
    clusterMeanMetropolis();
    
    // Metropolis step for batch parameters
    // batchScaleMetorpolis();
    // batchShiftMetorpolis();
    
    // Update the matrix combinations
    matrixCombinations();
  };
  
};


//' @title Mixture model
//' @description Performs MCMC sampling for a mixture model.
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
Rcpp::List sampleMixtureModel (
    arma::mat X,
    arma::uword K,
    arma::uword B,
    arma::uvec labels,
    arma::uvec batch_vec,
    double proposal_window,
    double proposal_window_for_logs,
    arma::uword R,
    arma::uword thin,
    arma::vec concentration,
    arma::uword seed
) {
  
  // Set the random number
  std::default_random_engine generator(seed);
  
  mvnSampler my_sampler(K,
    B,
    proposal_window,
    proposal_window_for_logs,
    labels,
    batch_vec,
    concentration,
    X
  );
  
  // // Declare the factory
  // samplerFactory my_factory;
  // 
  // // Convert from an int to the samplerType variable for our Factory
  // samplerFactory::samplerType val = static_cast<samplerFactory::samplerType>(dataType);
  // 
  // // Make a pointer to the correct type of sampler
  // std::unique_ptr<sampler> sampler_ptr = my_factory.createSampler(val,
  //                                                                 K,
  //                                                                 labels,
  //                                                                 concentration,
  //                                                                 X);
  
  // The output matrix
  arma::umat class_record(floor(R / thin), X.n_rows);
  class_record.zeros();
  
  // We save the BIC at each iteration
  arma::vec BIC_record = arma::zeros<arma::vec>(floor(R / thin));
  arma::uvec acceptance_vec = arma::zeros<arma::uvec>(floor(R / thin));
  arma::mat weights_saved = arma::zeros<arma::mat>(floor(R / thin), K);
  
  arma::cube mu_saved(my_sampler.P, K, floor(R / thin)), tau_saved(my_sampler.P, K, floor(R / thin)), m_saved(my_sampler.P, B, floor(R / thin)), t_saved(my_sampler.P, B, floor(R / thin));
  mu_saved.zeros();
  tau_saved.zeros();
  m_saved.zeros();
  t_saved.zeros();
  
  arma::uword save_int = 0;
  
  // Sampler from priors
  my_sampler.sampleFromPriors();
  my_sampler.matrixCombinations();
  
  my_sampler.modelScore();
  // sampler_ptr->sampleFromPriors();

  // my_sampler.model_score = my_sampler.modelLogLikelihood(
  //   my_sampler.mu,
  //   my_sampler.tau,
  //   my_sampler.m,
  //   my_sampler.t
  // ) + my_sampler.priorLogProbability(
  //     my_sampler.mu,
  //     my_sampler.tau,
  //     my_sampler.m,
  //     my_sampler.t
  // );
  
    // sample_prt.model_score->sampler_ptr.modelLo
  
  // Iterate over MCMC moves
  for(arma::uword r = 0; r < R; r++){
    
    my_sampler.updateWeights();

    // Metropolis step for batch parameters
    my_sampler.metropolisStep(); 

    my_sampler.updateAllocation();

    
    // sampler_ptr->updateWeights();
    // sampler_ptr->proposeNewParameters();
    // sampler_ptr->updateAllocation();
    
    // Record results
    if((r + 1) % thin == 0){
      
      // Update the BIC for the current model fit
      // sampler_ptr->calcBIC();
      // BIC_record( save_int ) = sampler_ptr->BIC; 
      // 
      // // Save the current clustering
      // class_record.row( save_int ) = sampler_ptr->labels.t();
      
      my_sampler.calcBIC();
      BIC_record( save_int ) = my_sampler.BIC;
      class_record.row( save_int ) = my_sampler.labels.t();
      acceptance_vec( save_int ) = my_sampler.accepted;
      weights_saved.row( save_int ) = my_sampler.w.t();
      mu_saved.slice( save_int ) = my_sampler.mu;
      // tau_saved.slice( save_int ) = my_sampler.tau;
      // m_saved.slice( save_int ) = my_sampler.m;
      // t_saved.slice( save_int ) = my_sampler.t;
      
      save_int++;
    }
  }
  return(List::create(Named("samples") = class_record, 
                      Named("means") = mu_saved,
                      Named("precisions") = tau_saved,
                      Named("batch_shift") = m_saved,
                      Named("batch_scale") = t_saved,
                      Named("weights") = weights_saved,
                      Named("acceptance") = acceptance_vec,
                      Named("BIC") = BIC_record));
  
};
