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

double logGamma(double x){
  double out = 0.0;
  arma::mat A(1, 1);
  A(0, 0) = x;
  out = arma::as_scalar(arma::lgamma(A));
  return out;
}

double gammaLogLikelihood(double x, double shape, double rate){
  double out = 0.0;
  out = shape * std::log(rate) - logGamma(shape) + (shape - 1) * std::log(x) - rate * x;
  return out;
};

double invGammaLogLikelihood(double x, double shape, double scale) {
  double out = 0.0;
  out = shape * std::log(scale) - logGamma(shape) + (-shape - 1) * std::log(x) - scale / x;
  return out;
};

double logNormalLogProbability(double x, double mu, double s) {
  double out = 0.0;
  arma::vec X(1), Y(1);
  arma::mat S(1,1);
  X(0) = std::log(x);
  Y(0) = std::log(mu);
  S(0, 0) = s;
  out = arma::as_scalar(arma::log_normpdf(X, Y, S));
  return out;
};

// double logNormalLogProbability(double x, double mu, double sigma2) {
//   double out = 0.0;
//   out = -std::log(x) - (1 / sigma2) * std::pow((std::log(x) - mu), 2.0);
//   return out;
// };

// double logWishartProbability(arma::mat X, arma::mat V, double n, arma::uword P){
//   return (-0.5*(n * arma::log_det(V).real() - (n - P - 1) * arma::log_det(X).real() + arma::trace(arma::inv(V) * X)));
// }

double logWishartProbability(arma::mat X, arma::mat V, double n, arma::uword P){
  double out = 0.5*((n - P - 1) * arma::log_det(X).real() - arma::trace(arma::inv_sympd(V) * X) - n * arma::log_det(V).real());
  return out;
}


double logInverseWishartProbability(arma::mat X, arma::mat Psi, double nu, arma::uword P){
  double out =  -0.5*(nu*arma::log_det(Psi).real()+(nu+P+1)*arma::log_det(X).real()+arma::trace(Psi * arma::inv_sympd(X)));
  return out;
}

class sampler {
  
private:
  
public:
  
  arma::uword K, B, N, P, K_occ, accepted = 0;
  double model_likelihood = 0.0, BIC = 0.0, model_score = 0.0;
  arma::uvec labels, N_k, batch_vec, N_b, KB_inds, B_inds;
  arma::vec concentration, w, ll, likelihood;
  arma::umat members;
  arma::mat X, X_t, alloc;
  arma::field<arma::uvec> batch_ind;
  
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
    X_t = X.t();
    
    // Plausibly belongs in the MVN sampler. Used for selecting slices / columns 
    // in the metropolis steps.
    KB_inds = arma::linspace<arma::uvec>(0, K - 1, K) * B;
    B_inds = arma::linspace<arma::uvec>(0, B - 1, B);
    
    // Dimensions
    N = X.n_rows;
    P = X.n_cols;
    
    // std::cout << "\nN: " << N << "\nP: " << P << "\n\n";
    
    // Class populations
    N_k = arma::zeros<arma::uvec>(K);
    N_b = arma::zeros<arma::uvec>(B);
    
    // The batch numbers won't ever change, so let's count them now
    for(arma::uword b = 0; b < B; b++){
      N_b(b) = arma::sum(batch_vec == b);
    }
    
    // Weights
    // double x, y;
    w = arma::zeros<arma::vec>(K);
    
    // Log likelihood (individual and model)
    ll = arma::zeros<arma::vec>(K);
    likelihood = arma::zeros<arma::vec>(N);
    
    // Class members
    members.set_size(N, K);
    members.zeros();
    
    // Allocation probability matrix (only makes sense in predictive models)
    alloc.set_size(N, K);
    alloc.zeros();
    
    // The indices of the members of each batch in the dataset
    batch_ind.set_size(B);
    for(arma::uword b = 0; b < B; b++) {
      batch_ind(b) = arma::find(batch_vec == b);
    }
  };
  
  // Destructor
  virtual ~sampler() { };
  
  // Virtual functions are those that should actual point to the sub-class
  // version of the function.
  // Print the sampler type.
  virtual void printType() {
    std::cout << "\nType: NULL.\n";
  };
  
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
    w = w / arma::accu(w);
    
  };
  
  virtual void updateAllocation() {
    
    double u = 0.0;
    arma::uvec uniqueK;
    arma::vec comp_prob(K);
    
    for(arma::uword n = 0; n < N; n++){
      
      ll = itemLogLikelihood(X_t.col(n), batch_vec(n));
      
      // std::cout << "\n\nAllocation log likelihood: " << ll;
      // Update with weights
      comp_prob = ll + log(w);
      
      likelihood(n) = arma::accu(comp_prob);
      
      // std::cout << "\n\nWeights: " << w;
      // std::cout << "\n\nAllocation log probability: " << comp_prob;
      
      // Normalise and overflow
      comp_prob = exp(comp_prob - max(comp_prob));
      comp_prob = comp_prob / sum(comp_prob);
      
      // Prediction and update
      u = arma::randu<double>( );
      
      labels(n) = sum(u > cumsum(comp_prob));
      alloc.row(n) = comp_prob.t();
      
      // Record the likelihood of the item in it's allocated component
      // likelihood(n) = ll(labels(n));
    }
    
    // The model log likelihood
    model_likelihood = arma::accu(likelihood);
    
    // Number of occupied components (used in BIC calculation)
    uniqueK = arma::unique(labels);
    K_occ = uniqueK.n_elem;
  };
  
  // The virtual functions that will be defined in any subclasses
  virtual void metropolisStep(){};
  virtual void sampleFromPriors() {};
  virtual void sampleParameters(){};
  virtual void calcBIC(){};
  virtual arma::vec itemLogLikelihood(arma::vec x, arma::uword b) { return arma::vec(); };
  
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
    
    // arma::uvec batch_ind = arma::find(batch_vec == b);
    for (auto& n : batch_ind(b)) {
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
    // arma::uvec batch_ind = find(batch_vec == b);
    for (auto& n : batch_ind(b)) {
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
//' @description The sampler for the Multivariate Normal mixture model for batch effects.
//' @field new Constructor \itemize{
//' \item Parameter: K - the number of components to model
//' \item Parameter: B - the number of batches present
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
  
  arma::uword n_param_cluster = 1 + P + P * (P + 1) * 0.5, 
    n_param_batch = 2 * P;
  
  double kappa, 
    nu, 
    lambda, 
    rho, 
    theta, 
    mu_proposal_window, 
    cov_proposal_window, 
    m_proposal_window, 
    S_proposal_window, 
    S_loc = 1.0;
  
  
  arma::uvec mu_count, cov_count, m_count, S_count, phi_count, rcond_count;
  arma::vec xi, delta, cov_log_det, global_mean;
  arma::mat scale, mu, m, S, phi, cov_comb_log_det, mean_sum, global_cov, Y;
  arma::cube cov, cov_inv, cov_comb, cov_comb_inv;
  
  using sampler::sampler;
  
  mvnSampler(                           
    arma::uword _K,
    arma::uword _B,
    double _mu_proposal_window,
    double _cov_proposal_window,
    double _m_proposal_window,
    double _S_proposal_window,
    double _rho,
    double _theta,
    double _lambda,
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
    
    arma::rowvec X_min = arma::min(X), X_max = arma::max(X);
    arma::mat global_cov = arma::cov(X);
    
    // Default values for hyperparameters
    // Cluster hyperparameters for the Normal-inverse Wishart
    // Prior shrinkage
    kappa = 0.01;
    // Degrees of freedom
    nu = P + 2;
    
    // Mean
    arma::mat mean_mat = arma::mean(_X, 0).t();
    xi = mean_mat.col(0);
    
    // mu_upper = xi + 10 * arma::abs(xi - X_max.t());
    // mu_lower = xi - 10 * arma::abs(xi - X_min.t());
    
    // std::cout << "\n\nMu upper:\n" << mu_upper << "\n\nMu lower:\n" << mu_lower;
    
    // cov_upper = 5 * global_cov;
    // std::cout << "\n\nCov upper:\n" << cov_upper;
    
    // Empirical Bayes for a diagonal covariance matrix
    arma::mat scale_param = _X.each_row() - xi.t();
    arma::vec diag_entries(P);
    // double scale_entry = arma::accu(scale_param % scale_param, 0) / (N * std::pow(K, 1.0 / (double) P));
    
    double scale_entry = (arma::accu(global_cov.diag()) / P) / std::pow(K, 2.0 / (double) P);
      
    diag_entries.fill(scale_entry);
    scale = arma::diagmat( diag_entries );
    
    // std::cout << "\nPrior scale:\n" << scale;
    
    // scale = arma::inv( arma::cov(X) / std::pow(K, 2.0 / P) );
    // scale = arma::cov(X) / std::pow(K, 2.0 / P);
    
    // std::cout << "\n\nPrior scale:\n" << scale;
    
    // The mean of the prior distribution for the batch shift, m, parameter
    delta = arma::zeros<arma::vec>(P);
    lambda = _lambda; // 1.0;
    
    // The shape and scale of the prior for the batch scale, S
    rho = _rho; // 41.0; // 3.0 / 2.0;
    theta = _theta; // 40.0; // arma::stddev(X.as_col()) / std::pow(B, 2.0 / B ); // 2.0;
    
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
    
    // Count the number of times proposed values are accepted
    cov_count = arma::zeros<arma::uvec>(K);
    mu_count = arma::zeros<arma::uvec>(K);
    m_count = arma::zeros<arma::uvec>(B);
    S_count = arma::zeros<arma::uvec>(B);
    
    // These will hold vertain matrix operations to avoid computational burden
    // The log determinant of each cluster covariance
    cov_log_det = arma::zeros<arma::vec>(K);
    
    // The log determinant of the covariance combination
    cov_comb_log_det.set_size(K, B);
    cov_comb_log_det.zeros();
    
    // The possible combinations for the sum of the cluster and batch means
    mean_sum.set_size(P, K * B);
    mean_sum.zeros();
    
    // The combination of each possible cluster and batch covariance
    cov_comb.set_size(P, P, K * B);
    cov_comb.zeros();
    
    // Inverse of the cluster covariance
    cov_inv.set_size(P, P, K);
    cov_inv.zeros();
    
    // The inverse of the covariance combination
    cov_comb_inv.set_size(P, P, K * B);
    cov_comb_inv.zeros();
    
    // The batch corrected data
    Y.set_size(N, P);
    Y.zeros();
    
    // The proposal windows for the cluster and batch parameters
    mu_proposal_window = _mu_proposal_window;
    cov_proposal_window = _cov_proposal_window;
    m_proposal_window = _m_proposal_window;
    S_proposal_window = _S_proposal_window;
    
    rcond_count.set_size(4);
    rcond_count.zeros();
  };
  
  
  // Destructor
  virtual ~mvnSampler() { };
  
  // Print the sampler type.
  virtual void printType() {
    std::cout << "\nType: MVN.\n";
  };
  
  virtual void sampleCovPrior() {
    for(arma::uword k = 0; k < K; k++){
      cov.slice(k) = arma::iwishrnd(scale, nu);
    }
  };
  
  virtual void sampleMuPrior() {
    for(arma::uword k = 0; k < K; k++){
      mu.col(k) = arma::mvnrnd(xi, (1.0/kappa) * cov.slice(k), 1);
    }
  }
  
  virtual void sampleSPrior() {
    for(arma::uword b = 0; b < B; b++){
      // S.col(b) = S_loc + arma::chi2rnd( 1.0, P );
      for(arma::uword p = 0; p < P; p++){
        S(p, b) = S_loc + 1.0 / arma::randg<double>( arma::distr_param(rho, 1.0 / theta ) );
      }
    }
  };
  
  virtual void sampleMPrior() {
    for(arma::uword b = 0; b < B; b++){
      for(arma::uword p = 0; p < P; p++){
        m(p, b) = arma::randn<double>() * (S(p, b)) / lambda + delta(p);
      }
    }
  };
  
  virtual void sampleFromPriors() {
    
    sampleCovPrior();
    sampleMuPrior();
    sampleSPrior();
    sampleMPrior();
    // 
    // for(arma::uword k = 0; k < K; k++){
    //   cov.slice(k) = arma::iwishrnd(scale, nu);
    //   mu.col(k) = arma::mvnrnd(xi, (1.0/kappa) * cov.slice(k), 1);
    //   
    //   // while(arma::all(mu.col(k) < mu_lower) || arma::all(mu.col(k) > mu_upper) ) {
    //   //   mu.col(k) = arma::mvnrnd(xi, (1.0/kappa) * cov.slice(k), 1);
    //   // }
    //   
    //   
    // }
    // for(arma::uword b = 0; b < B; b++){
    //   for(arma::uword p = 0; p < P; p++){
    //     
    //     // Fix the 0th batch at no effect; all other batches have an effect
    //     // relative to this
    //     // if(b == 0){
    //     //   S(p, b) = 1.0;
    //     //   m(p, b) = 0.0;
    //     // } else {
    //         S(p, b) = 1.0; // S_loc + 1.0 / arma::randg<double>( arma::distr_param(rho, 1.0 / theta ) );
    //     // S(p, b) = 1.0;
    //       m(p, b) = arma::randn<double>() * S(p, b) / lambda + delta(p);
    //     // m(p, b) = arma::randn<double>() / lambda + delta(p);
    //     // }
    //   }
    // }
    
    // std::cout << "\n\nPrior covariance:\n" << cov << "\n\nPrior mean:\n" << mu << "\n\nPrior S:\n" << S << "\n\nPrior m:\n" << m;
  };
  
  // Update the common matrix manipulations to avoid recalculating N times
  virtual void matrixCombinations() {
    
    for(arma::uword k = 0; k < K; k++) {
      cov_inv.slice(k) = arma::inv_sympd(cov.slice(k));
      cov_log_det(k) = arma::log_det(cov.slice(k)).real();
      for(arma::uword b = 0; b < B; b++) {
        cov_comb.slice(k * B + b) = cov.slice(k); // + arma::diagmat(S.col(b))
        for(arma::uword p = 0; p < P; p++) {
          cov_comb.slice(k * B + b)(p, p) *= S(p, b);
          // cov_comb.slice(k * B + b)(p, p) += S(p, b);
          
        }
        cov_comb_log_det(k, b) = arma::log_det(cov_comb.slice(k * B + b)).real();
        cov_comb_inv.slice(k * B + b) = arma::inv_sympd(cov_comb.slice(k * B + b));
        
        mean_sum.col(k * B + b) = mu.col(k) + m.col(b);
      }
    }
  };
  
  // The log likelihood of a item belonging to each cluster given the batch label.
  virtual arma::vec itemLogLikelihood(arma::vec item, arma::uword b) {
    
    double exponent = 0.0, my_det = 0.0;
    arma::vec ll(K), dist_to_mean(P), m_b(B);
    ll.zeros();
    dist_to_mean.zeros();
    m_b = m.col(b);
    
    arma::mat my_cov_comb(P, P), my_inv(P, P);
    
    for(arma::uword k = 0; k < K; k++){
      
      // The exponent part of the MVN pdf
      dist_to_mean = item - mean_sum.col(k * B + b);
      exponent = arma::as_scalar(dist_to_mean.t() * cov_comb_inv.slice(k * B + b) * dist_to_mean);
      
      // Normal log likelihood
      ll(k) = -0.5 *(cov_comb_log_det(k, b) + exponent + (double) P * log(2.0 * M_PI));
    }
    
    return(ll);
  };
  
  virtual void calcBIC(){
    
    // Each component has a weight, a mean vector and a symmetric covariance matrix. 
    // Each batch has a mean and standard deviations vector.
    // arma::uword n_param = (P + P * (P + 1) * 0.5) * K_occ + (2 * P) * B;
    // BIC = n_param * std::log(N) - 2 * model_likelihood;
    
    // arma::uword n_param_cluster = 1 + P + P * (P + 1) * 0.5;
    // arma::uword n_param_batch = 2 * P;
    
    // BIC = 2 * model_likelihood;
    
    BIC = 2 * model_likelihood - (n_param_batch + n_param_batch) * std::log(N);
    
    // for(arma::uword k = 0; k < K; k++) {
    //   BIC -= n_param_cluster * std::log(N_k(k) + 1);
    // }
    // for(arma::uword b = 0; b < B; b++) {
    //   BIC -= n_param_batch * std::log(N_b(b) + 1);
    // }
    
  };
  
  // virtual double clusterLikelihood(arma::uword k,
  //                          arma::vec cov_det,
  //                          arma::mat mean_sum,
  //                          arma::cube cov_inv) {
  //   
  //   arma::uword b = 0;
  //   double score = 0.0;
  //   arma::uvec cluster_ind = arma::find(labels == k);
  //   arma::vec dist_from_mean(P);
  //   
  //   for (auto& n : cluster_ind) {
  //     b = batch_vec(n);
  //     dist_from_mean = X_t.col(n) - mean_sum.col(b);
  //     score += arma::as_scalar(cov_det(b) + (dist_from_mean.t() * cov_inv.slice(b) * dist_from_mean));
  //   }
  //   return (-0.5 * score);
  // }
  
  
  
  virtual double groupLikelihood(arma::uvec inds,
                         arma::uvec group_inds,
                         arma::vec cov_det,
                         arma::mat mean_sum,
                         arma::cube cov_inv){
    
    arma::uword c = 0;
    double score = 0.0;
    arma::vec dist_from_mean(P);
    
    for (auto& n : inds) {
      c = group_inds(n);
      dist_from_mean = X_t.col(n) - mean_sum.col(c);
      score += arma::as_scalar(cov_det(c) + (dist_from_mean.t() * cov_inv.slice(c) * dist_from_mean));
    }
    return (-0.5 * score);
  }
  
  
  virtual double mLogKernel(arma::uword b, arma::vec m_b, arma::mat mean_sum) {
    
    // arma::uword k = 0;
    double score = 0.0;
    arma::vec dist_from_mean(P);
    dist_from_mean.zeros();
    
    score = groupLikelihood(batch_ind(b),
                            labels,
                            cov_comb_log_det.col(b),
                            mean_sum,
                            cov_comb_inv.slices(KB_inds + b));
    
  
    // for (auto& n : batch_ind(b)) {
    //   k = labels(n);
    //   dist_from_mean = X_t.col(n) - mean_sum.col(k);
    //   score +=  arma::as_scalar(cov_comb_log_det(k, b) + dist_from_mean.t() * cov_comb_inv.slice(k * B + b) * dist_from_mean);
    // }
    // 
    // if(std::abs(-0.5*score - score_alt) > 1e-6 ) {
    //   std::cout << "\n\nOriginal: " << -0.5*score << "\nMy new: " << score_alt;
    //   throw std::invalid_argument( "\nScores different in m Kernel." );
    // }
    
    
    for(arma::uword p = 0; p < P; p++) {
      score += -0.5 * (lambda * std::pow(m_b(p) - delta(p), 2.0) / (S(p, b)));
    }
    
    // score *= -0.5;
    return score;
  };
  
  virtual double sLogKernel(arma::uword b, 
                            arma::vec S_b, 
                            arma::vec cov_comb_log_det,
                            arma::cube cov_comb_inv) {
    
    arma::uword k = 0;
    double score = 0.0, score_alt = 0.0, my_det = 0.0;
    arma::vec dist_from_mean(P);
    dist_from_mean.zeros();
    
    arma::mat my_cov_comb(P, P), my_inv(P, P);
    
    score = groupLikelihood(batch_ind(b),
                            labels,
                            cov_comb_log_det,
                            mean_sum.cols(KB_inds + b),
                            cov_comb_inv);
    
    // for (auto& n : batch_ind(b)) {
    //   k = labels(n);
    //   
    //   my_cov_comb = cov.slice(k);
    //   
    //   for(arma::uword p = 0; p < P; p++) {
    //     my_cov_comb(p, p) = my_cov_comb(p, p) * S_b(p);
    //   }
    //   
    //   // std::cout << "\nThe invariance.";
    //   
    //   my_inv = arma::inv_sympd(my_cov_comb);
    //   
    //   // std::cout << "\nDeterminant.";
    //   my_det = arma::log_det(my_cov_comb).real();
    //   
    //   dist_from_mean = X_t.col(n) - mu.col(k) - m.col(b);
    //   score_alt += -0.5 * arma::as_scalar(my_det + (dist_from_mean.t() * my_inv * dist_from_mean));
    // }

    // if(std::abs(score - score_alt) > 1e-6 ) {
    //   std::cout << "\n\nOriginal: " << score << "\nMy new: " << score_alt <<
    //     "\n\nMean sums:\n" << mean_sum.cols(KB_inds + b) << 
    //       "\n\nCalculated:\n" << mu.col(k) + m.col(b);
    //   throw std::invalid_argument( "\nScores different in S kernels." );
    // }
    
    
    for(arma::uword p = 0; p < P; p++) {
      // score +=  (2 * rho + 3) * std::log(S_b(p)) + 2 * theta / S_b(p);
      
      score +=  -0.5 * ((2 * rho + 3) * std::log(S_b(p) - S_loc)
                        + (2 * theta) / (S_b(p) - S_loc)
                        + lambda * std::pow(m(p,b) - delta(p), 2.0) / S_b(p));

      // score +=   (0.5 - 1) * std::log(S(p,b) - S_loc) 
      //   - 0.5 * (S(p, b) - S_loc)
      //   - 0.5 * lambda * std::pow(m(p,b) - delta(p), 2.0) / S_b(p);
    }
    // score *= -0.5;
    
    return score;
  };
  
  virtual double muLogKernel(arma::uword k, arma::vec mu_k, arma::mat mean_sum) {
    
    // arma::uword b = 0;
    double score = 0.0, score_alt = 0.0;
    arma::uvec cluster_ind = arma::find(labels == k);
    arma::vec dist_from_mean(P);
    
    score = groupLikelihood(cluster_ind,
                            batch_vec,
                            cov_comb_log_det.row(k).t(),
                            mean_sum,
                            cov_comb_inv.slices(k * B + B_inds));
  
    // for (auto& n : cluster_ind) {
    //   b = batch_vec(n);
    //   dist_from_mean = X_t.col(n) - mean_sum.col(b);
    //   score += arma::as_scalar(cov_comb_log_det(k, b) + dist_from_mean.t() * cov_comb_inv.slice(k * B + b) * dist_from_mean);
    // }
    // 
    // if(std::abs(-0.5*score - score_alt) > 1e-6 ) {
    //   std::cout << "\n\nOriginal: " << -0.5*score << "\nMy new: " << score_alt;
    //   throw std::invalid_argument( "\nScores different in mu." );
    // }
    
    score += -0.5 * arma::as_scalar(kappa * ((mu_k - xi).t() *  cov_inv.slice(k) * (mu_k - xi)));
    // score *= -0.5;
    
    return score;
  };
  
  
  virtual double covLogKernel(arma::uword k, arma::mat cov_k, 
                              double cov_log_det,
                              arma::mat cov_inv,
                              arma::vec cov_comb_log_det,
                              arma::cube cov_comb_inv) {
    
    arma::uword b = 0;
    double score = 0.0, score_alt = 0.0, my_det = 0.0;
    arma::vec dist_from_mean(P);
    arma::mat my_cov_comb(P, P), my_inv(P, P);
    
    
    // double score_alt = clusterLikelihood(k,
    //                                      cov_comb_log_det,
    //                                      mean_sum.cols(k * B + B_inds),
    //                                      cov_comb_inv);
    

    arma::uvec cluster_ind = arma::find(labels == k);
    
    score = groupLikelihood(cluster_ind,
                            batch_vec,
                            cov_comb_log_det,
                            mean_sum.cols(k * B + B_inds),
                            cov_comb_inv);
    
    // for (auto& n : cluster_ind) {
    //   b = batch_vec(n);
    //   
    //   my_cov_comb = cov_k;
    //   
    //   for(arma::uword p = 0; p < P; p++) {
    //     my_cov_comb(p, p) = my_cov_comb(p, p) * S( p, b );
    //   }
    //   
    //   // std::cout << "\nThe invariance.";
    //   
    //   my_inv = arma::inv_sympd(my_cov_comb);
    //   
    //   // std::cout << "\nDeterminant.";
    //   my_det = arma::log_det(my_cov_comb).real();
    //   
    //   dist_from_mean = X_t.col(n) - mu.col(k) - m.col(b);
    //   score_alt += -0.5 * arma::as_scalar(my_det + (dist_from_mean.t() * my_inv * dist_from_mean));
    // }
    
    // if(std::abs(score - score_alt) > 1e-6 ) {
    //   std::cout << "\n\nOriginal: " << score << "\nMy new: " << score_alt <<
    //     "\n\nMean sums:\n" << mean_sum.cols(KB_inds + b) << 
    //       "\n\nCalculated:\n" << mu.col(k) + m.col(b);
    //   throw std::invalid_argument( "\nScores different in S kernels." );
    // }
    
    //   if(std::abs(score_alter - score_alt) > 1e-6 ) {
    //     std::cout << "\n\nNewest: " << score_alter << "\nOld new: " << score_alt;
    //     throw std::invalid_argument( "\nScores different." );
    //   }
    // 
    score += -0.5 * ( arma::as_scalar((nu + P + 2) * cov_log_det + kappa * ((mu.col(k) - xi).t() * cov_inv * (mu.col(k) - xi)) + arma::trace(scale * cov_inv)) );
    
    // arma::vec dist_from_mean(P);
    // 
    // for (auto& n : cluster_ind) {
    //   b = batch_vec(n);
    //   dist_from_mean = X_t.col(n) - mean_sum.col(k * B + b);
    //   score += arma::as_scalar(cov_comb_log_det(b) + (dist_from_mean.t() * cov_comb_inv.slice(b) * dist_from_mean));
    // }
    // 
    // score += arma::as_scalar((nu + P + 2) * cov_log_det + kappa * ((mu.col(k) - xi).t() * cov_inv * (mu.col(k) - xi)) + arma::trace(scale * cov_inv));
    // score *= -0.5;
    // 
    // if(std::abs(score - score_alt) > 1e-6 ) {
    //   std::cout << "\n\nOriginal: " << score << "\nMy new: " << score_alt;
    //   throw std::invalid_argument( "\nScores different in covariance kernel." );
    // }
    
    return score;
  };
  
  virtual void batchScaleMetropolis() {
    
    bool next = false;
    double u = 0.0, proposed_model_score = 0.0, acceptance_prob = 0.0, current_model_score = 0.0;
    arma::vec S_proposed(P), proposed_cov_comb_log_det(K);
    arma::cube proposed_cov_comb(P, P, K), proposed_cov_comb_inv(P, P, K);
    
    S_proposed.zeros();
    proposed_cov_comb_log_det.zeros();
    proposed_cov_comb.zeros();
    proposed_cov_comb_inv.zeros();
    
    for(arma::uword b = 0; b < B; b++) {
      
      next = false;
      acceptance_prob = 0.0, proposed_model_score = 0.0, current_model_score = 0.0;
      proposed_cov_comb.zeros();
      
      for(arma::uword p = 0; p < P; p++) {
        
        // (arma::randn() * m_proposal_window) + m(p, b);
        
        // S_proposed(p) = S_loc + (S(p, b) - S_loc) * std::exp(arma::randn() * S_proposal_window);
        // S_proposed(p) = S_loc + std::exp(arma::randn() * S_proposal_window + log(S(p, b) - S_loc) );
        // S_proposed(p) = std::exp(arma::randn() * S_proposal_window + log(S(p, b)));

        // proposed_model_score += logNormalLogProbability(S(p, b), S_proposed(p), S_proposal_window);
        // current_model_score += logNormalLogProbability(S_proposed(p), S(p, b), S_proposal_window);
  
        // if(std::abs(S(p, b) - S_loc) < 1e-6){
        //   std::cout << "S(p,b): " << S(p,b) << "\nS_loc: " << S_loc << "\n\nToo close.";
        // }
        // 
        // std::cout << "S(p,b): " << S(p,b) << "\nS_loc: " << S_loc << "\n\nToo close.";
        // 
        // 
        S_proposed(p) = S_loc + arma::randg( arma::distr_param( (S(p, b) - S_loc) * S_proposal_window, 1.0 / S_proposal_window) );
        
        if(S_proposed(p) <= 0.0) {
          next = true;
        }
        //
        // // Asymmetric proposal density
        proposed_model_score += gammaLogLikelihood(S(p, b) - S_loc, (S_proposed(p) - S_loc) * S_proposal_window, S_proposal_window);
        current_model_score += gammaLogLikelihood(S_proposed(p) - S_loc, (S(p, b) - S_loc) * S_proposal_window, S_proposal_window);
      }
      
      if(next) {
        continue;
      }
      
      // std::cout << "\n\nS proposed:n" << S_proposed << "\n\nS(b):\n" << S.col(b);
      
      proposed_cov_comb = cov;
      for(arma::uword k = 0; k < K; k++) {
        // proposed_batch_cov_comb.slice(k) = cov.slice(k); // + arma::diagmat(S.col(b))
        for(arma::uword p = 0; p < P; p++) {
          proposed_cov_comb.slice(k)(p, p) *= S_proposed(p);
          // proposed_cov_comb.slice(k)(p, p) += S_proposed(p);
          
        }
        proposed_cov_comb_log_det(k) = arma::log_det(proposed_cov_comb.slice(k)).real();
        // proposed_cov_comb_inv.slice(k) = arma::inv(proposed_cov_comb.slice(k));
        proposed_cov_comb_inv.slice(k) = arma::inv_sympd(proposed_cov_comb.slice(k));
      }
      
      // std::cout << "\n\nProposed S:\n" << S_proposed <<
      //   "\nProposed model score: " << proposed_model_score <<
      //     "\n\nCurrent S:\n" << S.col(b) << "\nCurrent model score: " <<
      //       current_model_score;
      // 
      // std::cout << "\n\nMean\n:" << mu << "\n\nCovariance:\n" << cov << "\n\nBatch shift:\n" << m;
      
      proposed_model_score += sLogKernel(b, 
        S_proposed, 
        proposed_cov_comb_log_det,
        proposed_cov_comb_inv
      );
      
      current_model_score += sLogKernel(b, 
        S.col(b), 
        cov_comb_log_det.col(b),
        cov_comb_inv.slices(KB_inds + b)
      );
      
      // std::cout << "\n\nProposed model score: " << proposed_model_score <<
      //     "\nCurrent model score: " << current_model_score;
      
      // if(proposed_model_score > 1) {
      //   std::cout << "\nProposed S score is too high. Score: " << proposed_model_score <<
      //     "\nProposed value\n" << S_proposed;
      //   throw std::invalid_argument( "\nBad score." );
      // }
      // 
      // 
      // if(current_model_score > 1) {
      //   std::cout << "\nCurrent S score is too high. Score: " << current_model_score <<
      //     "\nProposed value\n" << S.col(b);
      //   throw std::invalid_argument( "\nBad score." );
      // }
      
      // if((proposed_model_score - current_model_score) > 20000) {
      //     std::cout << "\nCurrent S scores are odd. Current score: " << current_model_score <<
      //       "\n Proposed score: " << proposed_model_score << 
      //         "\n\nProposed value\n" << S_proposed << "\nCurrent value\n" << 
      //           S.col(b) << "\n\nCov:\n" << cov <<  
      //             "\n\nCov combined inverse:\n" <<
      //             cov_comb_inv.slices(KB_inds + b) << 
      //               "\n\nProposed cov combined inverse:\n" << proposed_cov_comb_inv;
      //     throw std::invalid_argument( "\nBad score." );
      // }
      
      u = arma::randu();
      acceptance_prob = std::min(1.0, std::exp(proposed_model_score - current_model_score));
      
      // std::cout << "\n\nProposed S:\n"<< S_proposed << "\n\nCurrent S:\n" << S.col(b) << "\n\nProposed score: " << proposed_model_score << "\nCurrent score: " << current_model_score;
      
      if(u < acceptance_prob){
        S.col(b) = S_proposed;
        S_count(b)++;
        
        for(arma::uword k = 0; k < K; k++) {
          cov_comb.slice(k * B + b) = proposed_cov_comb.slice(k);
          cov_comb_log_det(k, b) = proposed_cov_comb_log_det(k);
          cov_comb_inv.slice(k * B + b) = proposed_cov_comb_inv.slice(k);
        }
      }
    }
  };
  
  virtual void batchShiftMetorpolis() {
    
    double u = 0.0, proposed_model_score = 0.0, acceptance_prob = 0.0, current_model_score = 0.0;
    arma::vec m_proposed(P);
    arma::mat proposed_mean_sum(P, K);
    m_proposed.zeros();
    
    for(arma::uword b = 0; b < B; b++) {
      // std::cout << "\nIn here.\n";
      for(arma::uword p = 0; p < P; p++){
        // The proposal window is now a diagonal matrix of common entries.
        m_proposed(p) = (arma::randn() * m_proposal_window) + m(p, b);
      }
      
      for(arma::uword k = 0; k < K; k++) {
        proposed_mean_sum.col(k) = mu.col(k) + m_proposed;
      }
      
      proposed_model_score = mLogKernel(b, m_proposed, proposed_mean_sum);
      current_model_score = mLogKernel(b, m.col(b), mean_sum.cols(KB_inds + b));
      
      // std::cout << "\nProposed value:\n\n" << m_proposed << 
      //   "\n\nProposed model score: " <<  proposed_model_score << 
      //     "nCurrent model score: " <<  current_model_score;
      
      u = arma::randu();
      acceptance_prob = std::min(1.0, std::exp(proposed_model_score - current_model_score));
      
      // std::cout << "\n\nProposed m:\n"<< m_proposed << "\n\nCurrent m:\n" << m.col(b) << "\n\nProposed score: " << proposed_model_score << "\nCurrent score: " << current_model_score;
      
      if(u < acceptance_prob){
        m.col(b) = m_proposed;
        m_count(b)++;
        
        for(arma::uword k = 0; k < K; k++) {
          mean_sum.col(k * B + b) = proposed_mean_sum.col(k);
        }
      }
    }
  };
  
  virtual void clusterCovarianceMetropolis() {
    
    bool cov_range_acceptable = true;
    double u = 0.0, proposed_model_score = 0.0, acceptance_prob = 0.0, current_model_score = 0.0, proposed_cov_log_det = 0.0;
    arma::vec proposed_cov_comb_log_det(B);
    arma::mat cov_proposed(P, P), proposed_cov_inv(P, P);
    arma::cube proposed_cov_comb(P, P, B), proposed_cov_comb_inv(P, P, B);
    
    cov_proposed.zeros();
    proposed_cov_inv.zeros();
    proposed_cov_comb_log_det.zeros();
    proposed_cov_comb.zeros();
    proposed_cov_comb_inv.zeros();
    
    for(arma::uword k = 0; k < K ; k++) {
      
      cov_range_acceptable = true;
      
      proposed_cov_comb.zeros();
      acceptance_prob = 0.0, proposed_model_score = 0.0, current_model_score = 0.0;
      
      if(N_k(k) == 0){
        cov_proposed = arma::iwishrnd(scale, nu);
        // proposed_cov_inv = arma::inv(cov_proposed);
        proposed_cov_inv = arma::inv_sympd(cov_proposed);
        proposed_cov_log_det = arma::log_det(cov_proposed).real();
        for(arma::uword b = 0; b < B; b++) {
          proposed_cov_comb.slice(b) = cov_proposed; // + arma::diagmat(S.col(b))
          for(arma::uword p = 0; p < P; p++) {
            proposed_cov_comb.slice(b)(p, p) *= S(p, b);
            // proposed_cov_comb.slice(b)(p, p) += S(p, b);
            
          }
          proposed_cov_comb_log_det(b) = arma::log_det(proposed_cov_comb.slice(b)).real();
          proposed_cov_comb_inv.slice(b) = arma::inv_sympd(proposed_cov_comb.slice(b));
        }
      } else {
        
        // std::cout << "\n\n\nCovariance " << k << cov.slice(k) <<
        //   "\n\nProposal covariance " << k << cov.slice(k) / cov_proposal_window;

        cov_proposed = arma::wishrnd(cov.slice(k) / cov_proposal_window, cov_proposal_window);

        // 
        // if(arma::rcond(cov_proposed) <= 0.1) {
        //   rcond_count(0)++;
        // } 
        // if(arma::rcond(cov_proposed) <= 0.01) {
        //   rcond_count(1)++;
        // } 
        // if(arma::rcond(cov_proposed) <= 0.001) {
        //   rcond_count(2)++;
        // } 
        // if(arma::rcond(cov_proposed) <= 0.0001) {
        //   rcond_count(3)++;
        // } 
        
        // for(arma::uword p = 0; p < P; p++) {
        //   for(arma::uword p = 0; p < P; p++) {
        //     if(cov_proposed(p, p) > cov_upper(p,p)) {
        //       cov_range_acceptable = false;
        //       
        //       std::cout << "\n\nCov proposed:\n" << cov_proposed << 
        //         "\n\nComparison between this and upper bound:\n" <<
        //           arma::any(cov_proposed > cov_upper) << "\n\nUpper bound\n" << 
        //             cov_upper;
        //     }
        //   }
        // }
        
        // if(arma::sum(arma::any(cov_proposed > cov_upper)) >= 1) {
        //   std::cout << "\n\nCov proposed:\n" << cov_proposed << 
        //     "\n\nComparison between this and upper bound:\n" <<
        //       arma::any(cov_proposed > cov_upper) << "\n\nUpper bound\n" << 
        //         cov_upper;
        //   
        //   // cov_range_acceptable = false;
        // }
        // std::cout << "\n\nProposed covariance " << cov_proposed;
        
        
        // Log probability under the proposal density
        proposed_model_score = logWishartProbability(cov.slice(k), cov_proposed / cov_proposal_window, cov_proposal_window, P);
        current_model_score = logWishartProbability(cov_proposed, cov.slice(k) / cov_proposal_window, cov_proposal_window, P);
        
        proposed_cov_inv = arma::inv_sympd(cov_proposed);
        proposed_cov_log_det = arma::log_det(cov_proposed).real();
        
        for(arma::uword b = 0; b < B; b++) {
          proposed_cov_comb.slice(b) = cov_proposed; // + arma::diagmat(S.col(b))
          for(arma::uword p = 0; p < P; p++) {
            proposed_cov_comb.slice(b)(p, p) *= S(p, b);
            // proposed_cov_comb.slice(b)(p, p) += S(p, b);
          }
          proposed_cov_comb_log_det(b) = arma::log_det(proposed_cov_comb.slice(b)).real();
          proposed_cov_comb_inv.slice(b) = arma::inv_sympd(proposed_cov_comb.slice(b));
        }
        
        // std::cout << "\n\nProposed cov:\n" << cov_proposed << 
        //   "\nProposed model score: " << proposed_model_score <<
        //     "\n\nCurrent cov:\n" << cov.slice(k) << "\nCurrent model score: " << 
        //       current_model_score;
        // 
        // std::cout << "\n\nMean\n:" << mu <<  "\n\nBatch shift:\n" << m << "\n\nBatch scale:\n" << S;
        
        // The boolean variables indicate use of the old manipulated matrix or the 
        // proposed.
        proposed_model_score += covLogKernel(k, 
          cov_proposed,
          proposed_cov_log_det,
          proposed_cov_inv,
          proposed_cov_comb_log_det,
          proposed_cov_comb_inv
        );
        
        current_model_score += covLogKernel(k, 
          cov.slice(k), 
          cov_log_det(k),
          cov_inv.slice(k),
          cov_comb_log_det.row(k).t(),
          cov_comb_inv.slices(k * B + B_inds)
        );
        
        // std::cout << "\n\nProposed model score: " << proposed_model_score <<
        //   "\nCurrent model score: " << current_model_score;
        
        // if((proposed_model_score - current_model_score) > 10000) {
        //   std::cout << "\nCurrent Cov scores are odd. Current score: " << current_model_score <<
        //     "\n Proposed score: " << proposed_model_score << 
        //       "\n\nProposed value\n" << cov_proposed << "\nCurrent value\n" << 
        //         cov.slice(k) << "\n\nCov combined inverse:\n" << 
        //         cov_comb_inv.slices(k * B + B_inds) << 
        //           "\n\nProposed cov combined inverse:\n" << proposed_cov_comb_inv;
        //   throw std::invalid_argument( "\nBad score." );
        // }
        
        // std::cout << "\n\nProposed Cov:\n"<< cov_proposed << "\n\nCurrent cov:\n" << cov.slice(k) << "\n\nProposed score: " << proposed_model_score << "\nCurrent score: " << current_model_score;
        
        // Accept or reject
        u = arma::randu();
        acceptance_prob = std::min(1.0, std::exp(proposed_model_score - current_model_score));
        
        // std::cout << "\n\nProposed value:\n" << cov_proposed <<
        //   "\n\nCurrent:\n" << cov.slice(k) << "\nProposed score: " <<
        //     proposed_model_score << "\nCurrent score: " << current_model_score <<
        //       "\nAcceptance probability: " << acceptance_prob;
      }
      if( (u < acceptance_prob) || (N_k(k) == 0) ){
        cov.slice(k) = cov_proposed;
        cov_count(k)++;
        
        cov_inv.slice(k) = proposed_cov_inv;
        cov_log_det(k) = proposed_cov_log_det;
        for(arma::uword b = 0; b < B; b++) {
          cov_comb.slice(k * B + b) = proposed_cov_comb.slice(b);
          cov_comb_log_det(k, b) = proposed_cov_comb_log_det(b);
          cov_comb_inv.slice(k * B + b) = proposed_cov_comb_inv.slice(b);
        }
      }
    }
  };
  
  virtual void clusterMeanMetropolis() {
    
    bool mu_range_acceptable = true;
    double u = 0.0, proposed_model_score = 0.0, acceptance_prob = 0.0, current_model_score = 0.0;
    arma::vec mu_proposed(P);
    arma::mat proposed_mean_sum(P, B);
    
    mu_proposed.zeros();
    proposed_mean_sum.zeros();
    
    for(arma::uword k = 0; k < K ; k++) {
      
      mu_range_acceptable = true;
      
      if(N_k(k) == 0){
        mu_proposed = arma::mvnrnd(xi, (1.0/kappa) * cov.slice(k), 1);
        for(arma::uword b = 0; b < B; b++) {
          proposed_mean_sum.col(b) = mu_proposed + m.col(b);
        }
      } else {
        for(arma::uword p = 0; p < P; p++){
          // The proposal window is now a diagonal matrix of common entries.
          mu_proposed(p) = (arma::randn() * mu_proposal_window) + mu(p, k);
        }
        
        // if(arma::all(mu_proposed < mu_lower) || arma::all(mu_proposed > mu_upper) ) {
        //   std::cout << "\n\nMu proposed:\n" << mu_proposed <<
        //     "\n\nComparison lower:\n" << arma::any(mu_proposed < mu_lower) <<
        //       "\n\nComparison upper:\n" << arma::any(mu_proposed > mu_upper) <<
        //         "\n\nJoint comparison:\n" << arma::any(mu_proposed < mu_lower) || arma::any(mu_proposed > mu_upper);
        //   
        //   mu_range_acceptable = false;
        // }
        // 
        
        for(arma::uword b = 0; b < B; b++) {
          proposed_mean_sum.col(b) = mu_proposed + m.col(b);
        }
        
        // The prior is included in the kernel
        proposed_model_score = muLogKernel(k, mu_proposed, proposed_mean_sum);
        current_model_score = muLogKernel(k, mu.col(k), mean_sum.cols(k * B + B_inds));
        
        u = arma::randu();
        acceptance_prob = std::min(1.0, std::exp(proposed_model_score - current_model_score));
        
      }
      
      if(((u < acceptance_prob) || (N_k(k) == 0)) && mu_range_acceptable) {
        mu.col(k) = mu_proposed;
        mu_count(k)++;
        
        for(arma::uword b = 0; b < B; b++) {
          mean_sum.col(k * B + b) = proposed_mean_sum.col(b);
        }
        
      }
    }
  };
  
  virtual void updateBatchCorrectedData() {
    
    arma::uword b = 0, k = 0;
    arma::vec unscaled_data(P);
    arma::mat mu_mat = mu.cols(labels); //, Y_alt(N, P);
    
    Y = ((X_t - mu_mat - m.cols(batch_vec)) / arma::sqrt(S.cols(batch_vec)) + mu_mat).t();
    
    // for(arma::uword n = 0; n < N; n++) {
    //   b = batch_vec(n);
    //   k = labels(n);
    //   Y.row(n) = (((X_t.col(n) - mu.col(k) - m.col(b)) / arma::sqrt(S.col(b))) + mu.col(k)).t();
    // }
    // if(! approx_equal(Y, Y_alt, "absdiff", 0.002)) {
    //   std::cout << "\n\nY's disagree.";
    //   throw;
    // }
    // Y = Y_alt;
  }
  
  virtual void checkPositiveDefinite(arma::uword r) {
    for(arma::uword k = 0; k < K; k++) {
      if(! cov.slice(k).is_sympd()) {
        
        std::cout << "\nIteration " << r;
        std::cout << "\n\nCovariance " << k << " is not positive definite.\n\n" 
        << cov.slice(k);
        
        throw;
      }
      if(! cov_inv.slice(k).is_sympd()) {
        
        std::cout << "\nIteration " << r;
        std::cout << "\n\nCovariance inverse " << k << 
          " is not positive definite.\n\n" << cov_inv.slice(k);
        throw;
      }
      for(arma::uword b = 0; b < B; b++) {
      if(! cov_comb.slice(k * B + b).is_sympd()) {
        std::cout << "\nIteration " << r;
        std::cout << "\n\nCombined covariance for cluster " << k << " and batch " 
        << b << " is not positive definite.\n\n" << cov_comb.slice(k * B + b);
        std::cout << "\n\nS(b):\n" << S.col(b) << "\n\nCov(k):\n" << cov.slice(k);
        throw;
      }
      if(! cov_comb_inv.slice(k * B + b).is_sympd()) {
        std::cout << "\nIteration " << r;
        std::cout << "\n\nCombined covariance inverse for cluster " << k << " and batch " 
                  << b << " is not positive definite.\n\n" << cov_comb_inv.slice(k * B + b);
        throw;
      }
      }
    }
  }
  
  virtual void metropolisStep() {
    
    // Metropolis step for cluster parameters
    clusterCovarianceMetropolis();
    clusterMeanMetropolis();
  
    // Metropolis step for batch parameters
    batchScaleMetropolis();
    batchShiftMetorpolis();
    
    // }
  };
  
};


class semisupervisedSampler : public virtual sampler {
private:
  
public:
  
  arma::uword N_fixed = 0;
  arma::uvec fixed, unfixed_ind;
  arma::mat alloc_prob;
  
  using sampler::sampler;
  
  semisupervisedSampler(
    arma::uword _K,
    arma::uword _B,
    arma::uvec _labels,
    arma::uvec _batch_vec,
    arma::vec _concentration,
    arma::mat _X,
    arma::uvec _fixed
  ) : 
    sampler(_K, _B, _labels, _batch_vec, _concentration, _X) {
    
    arma::uvec fixed_ind(N);
    
    fixed = _fixed;
    N_fixed = arma::sum(fixed);
    fixed_ind = arma::find(_fixed == 1);
    unfixed_ind = find(fixed == 0);
    
    alloc_prob.set_size(N, K);
    alloc_prob.zeros();
    
    for (auto& n : fixed_ind) {
      alloc_prob(n, labels(n)) = 1.0;
    }
  };
  
  // Destructor
  virtual ~semisupervisedSampler() { };
  
  virtual void updateAllocation() {
    
    double u = 0.0;
    arma::uvec uniqueK;
    arma::vec comp_prob(K);
    
    for (auto& n : unfixed_ind) {
      
      ll = itemLogLikelihood(X_t.col(n), batch_vec(n));
      
      // Update with weights
      comp_prob = ll + log(w);

      likelihood(n) = arma::accu(comp_prob);
      
      // Normalise and overflow
      comp_prob = exp(comp_prob - max(comp_prob));
      comp_prob = comp_prob / sum(comp_prob);
      
      // Save the allocation probabilities
      alloc_prob.row(n) = comp_prob.t();
      
      // Prediction and update
      u = arma::randu<double>( );
      
      labels(n) = sum(u > cumsum(comp_prob));
      alloc.row(n) = comp_prob.t();
      
      // Record the log likelihood of the item in it's allocated component
      // likelihood(n) = ll(labels(n));
    }
    
    // The model log likelihood
    model_likelihood = arma::accu(likelihood);
    
    // Number of occupied components (used in BIC calculation)
    uniqueK = arma::unique(labels);
    K_occ = uniqueK.n_elem;
  };
  
};


class mvnPredictive : public mvnSampler, public semisupervisedSampler {
  
private:
  
public:
  
  using mvnSampler::mvnSampler;
  
  mvnPredictive(
    arma::uword _K,
    arma::uword _B,
    double _mu_proposal_window,
    double _cov_proposal_window,
    double _m_proposal_window,
    double _S_proposal_window,
    double _rho,
    double _theta,
    double _lambda,
    arma::uvec _labels,
    arma::uvec _batch_vec,
    arma::vec _concentration,
    arma::mat _X,
    arma::uvec _fixed
  ) : 
    sampler(_K, _B, _labels, _batch_vec, _concentration, _X),
    mvnSampler(_K,
               _B,
               _mu_proposal_window,
               _cov_proposal_window,
               _m_proposal_window,
               _S_proposal_window,
               _rho,
               _theta,
               _lambda,
               _labels,
               _batch_vec,
               _concentration,
               _X),
    semisupervisedSampler(_K, _B, _labels, _batch_vec, _concentration, _X, _fixed)
    {
  };
  
  virtual ~mvnPredictive() { };
  
  // virtual void sampleFromPriors() {
  //   
  //   arma::mat X_k;
  //   
  //   for(arma::uword k = 0; k < K; k++){
  //     X_k = X.rows(arma::find(labels == k && fixed == 1));
  //     cov.slice(k) = arma::diagmat(arma::stddev(X_k).t());
  //     mu.col(k) = arma::mean(X_k).t();
  //   }
  //   for(arma::uword b = 0; b < B; b++){
  //     for(arma::uword p = 0; p < P; p++){
  //       
  //       // Fix the 0th batch at no effect; all other batches have an effect
  //       // relative to this
  //       // if(b == 0){
  //       S(p, b) = 1.0;
  //       m(p, b) = 0.0;
  //       // } else {
  //       // S(p, b) = 1.0 / arma::randg<double>( arma::distr_param(rho, 1.0 / theta ) );
  //       // m(p, b) = arma::randn<double>() * S(p, b) / lambda + delta(p);
  //       // }
  //     }
  //   }
  // };
  
};


//' @name msnSampler
//' @title Multivariate Skew Normal mixture type
//' @description The sampler for the Multivariate Normal mixture model for batch effects.
//' @field new Constructor \itemize{
//' \item Parameter: K - the number of components to model
//' \item Parameter: B - the number of batches present
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
class msnSampler: virtual public mvnSampler {
  
public:
  
  bool nan_found = false;
  arma::uword n_param_cluster = 1 + 2 * P + P * (P + 1) * 0.5, n_param_batch = 2 * P;
  double omega, phi_proposal_window;
  arma::uvec phi_count;
  arma::mat cov_comb_inv_diag_sqrt;
  
  using mvnSampler::mvnSampler;
  
  msnSampler(                           
    arma::uword _K,
    arma::uword _B,
    double _mu_proposal_window,
    double _cov_proposal_window,
    double _m_proposal_window,
    double _S_proposal_window,
    double _phi_proposal_window,
    double _rho,
    double _theta,
    double _lambda,
    arma::uvec _labels,
    arma::uvec _batch_vec,
    arma::vec _concentration,
    arma::mat _X
  ) : sampler(_K,
  _B,
  _labels,
  _batch_vec,
  _concentration,
  _X), mvnSampler(                           
      _K,
      _B,
      _mu_proposal_window,
      _cov_proposal_window,
      _m_proposal_window,
      _S_proposal_window,
      _rho,
      _theta,
      _lambda,
      _labels,
      _batch_vec,
      _concentration,
      _X
  ) {
    
    // Hyperparameter for the prior on the shape of the skew normal
    omega = 0.5;
    
    // The shape of the skew normal
    phi.set_size(P, K);
    phi.zeros();
    
    // Count the number of times proposed values are accepted
    phi_count = arma::zeros<arma::uvec>(K);
    
    // These will hold vertain matrix operations to avoid computational burden
    // The standard deviations of the data
    cov_comb_inv_diag_sqrt.set_size(P, K * B);
    cov_comb_inv_diag_sqrt.zeros();
  
    // The proposal windows for the cluster and batch parameters
    phi_proposal_window = _phi_proposal_window;
  };
  
  
  // Destructor
  virtual ~msnSampler() { };
  
  // Print the sampler type.
  virtual void printType() {
    std::cout << "\nType: Multivariate Skew Normal.\n";
  }
  
  virtual void sampleFromPriors() {
    
    for(arma::uword k = 0; k < K; k++){
      cov.slice(k) = arma::iwishrnd(scale, nu);
      mu.col(k) = arma::mvnrnd(xi, (1.0/kappa) * cov.slice(k), 1);
      for(arma::uword p = 0; p < P; p++){
        phi(p, k) =  arma::randn<double>() * omega;
      }
    }
    for(arma::uword b = 0; b < B; b++){
      for(arma::uword p = 0; p < P; p++){
        
        // Fix the 0th batch at no effect; all other batches have an effect
        // relative to this
        // if(b == 0){
        //   S(p, b) = 1.0;
        //   m(p, b) = 0.0;
        // } else {
          S(p, b) = 1.0 / arma::randg<double>( arma::distr_param(rho, 1.0 / theta ) );
          m(p, b) = arma::randn<double>() * S(p, b) / lambda + delta(p);
        // }
      }
    }
  };
  
  // Update the common matrix manipulations to avoid recalculating N times
  virtual void matrixCombinations() {
    
    for(arma::uword k = 0; k < K; k++) {
      cov_inv.slice(k) = arma::inv_sympd(cov.slice(k));
      cov_log_det(k) = arma::log_det(cov.slice(k)).real();
      for(arma::uword b = 0; b < B; b++) {
        cov_comb.slice(k * B + b) = cov.slice(k);
        for(arma::uword p = 0; p < P; p++) {
          cov_comb.slice(k * B + b)(p, p) *= S(p, b);
        }
        cov_comb_log_det(k, b) = arma::log_det(cov_comb.slice(k * B + b)).real();
        cov_comb_inv.slice(k * B + b) = arma::inv_sympd(cov_comb.slice(k * B + b));
        cov_comb_inv_diag_sqrt.col(k * B + b) = arma::sqrt(cov_comb_inv.slice(k * B + b).diag());
        mean_sum.col(k * B + b) = mu.col(k) + m.col(b);
      }
    }
  };
  
  // The log likelihood of a item belonging to each cluster given the batch label.
  virtual arma::vec itemLogLikelihood(arma::vec item, arma::uword b) {
    
    double exponent = 0.0;
    arma::vec ll(K), dist_to_mean(P);
    ll.zeros();
    dist_to_mean.zeros();
    
    for(arma::uword k = 0; k < K; k++){
      
      // The exponent part of the MVN pdf
      dist_to_mean = item - mean_sum.col(k * B + b);

      exponent = arma::as_scalar(dist_to_mean.t() * cov_comb_inv.slice(k * B + b) * dist_to_mean);
      
      // Normal log likelihood
      ll(k) = log(2.0) + -0.5 * (cov_comb_log_det(k, b) + exponent + (double) P * log(2.0 * M_PI)); 
      ll(k) += log(arma::normcdf(arma::as_scalar(phi.col(k).t() * cov_comb_inv_diag_sqrt(k * B + b) * dist_to_mean)));
      
    }
    
    return(ll);
  };
  
  virtual void calcBIC(){
    
    // Each component has a weight, a mean and shape vector and a symmetric covariance 
    // matrix. Each batch has a mean and standard deviations vector.
    // arma::uword n_param = (2 * P + P * (P + 1) * 0.5) * K_occ + (2 * P) * B;
    // BIC = n_param * std::log(N) - 2 * model_likelihood;
    // 
    // arma::uword n_param_cluster = 1 + 2 * P + P * (P + 1) * 0.5;
    // arma::uword n_param_batch = 2 * P;
    
    // BIC = 2 * model_likelihood;
    
    BIC = 2 * model_likelihood - (n_param_batch + n_param_batch) * std::log(N);
    
    // for(arma::uword k = 0; k < K; k++) {
    //   BIC -= n_param_cluster * std::log(N_k(k) + 1);
    // }
    // for(arma::uword b = 0; b < B; b++) {
    //   BIC -= n_param_batch * std::log(N_b(b)+ 1);
    // }
    
  };
  
  double mLogKernel(arma::uword b, arma::vec m_b, arma::mat mean_sum) {
    
    arma::uword k = 0;
    double score = 0.0;
    arma::vec dist_from_mean(P);
    dist_from_mean.zeros();
    
    for (auto& n : batch_ind(b)) {
      k = labels(n);
      dist_from_mean = X_t.col(n) - mean_sum.col(k);
      score -= 0.5 * arma::as_scalar(dist_from_mean.t() * cov_comb_inv.slice(k * B + b) * dist_from_mean);
      score += log(arma::normcdf(arma::as_scalar(phi.col(k).t() * arma::diagmat(cov_comb_inv_diag_sqrt.col(k * B + b)) * dist_from_mean)));
    }
    for(arma::uword p = 0; p < P; p++) {
      score -= 0.5 * lambda * std::pow(m_b(p) - delta(p), 2.0) / S(p, b);
    }
    
    std::cout << "\nm score: " << score;
    
    return score;
  };
  
  double sLogKernel(arma::uword b, arma::vec S_b, 
                    arma::vec cov_comb_log_det,
                    arma::mat cov_comb_inv_diag_sqrt,
                    arma::cube cov_comb_inv) {
    
    arma::uword k = 0;
    double score = 0.0;
    arma::vec dist_from_mean(P);
    dist_from_mean.zeros();
    arma::mat curr_sum(P, P);
    
    for (auto& n : batch_ind(b)) {
      k = labels(n);
      dist_from_mean = X_t.col(n) - mean_sum.col(k * B + b);
      score -= 0.5 * arma::as_scalar(cov_comb_log_det(k) + (dist_from_mean.t() * cov_comb_inv.slice(k) * dist_from_mean));
      score += log(arma::normcdf(arma::as_scalar(phi.col(k).t() * arma::diagmat(cov_comb_inv_diag_sqrt.col(k)) * dist_from_mean)));
    }
    for(arma::uword p = 0; p < P; p++) {
      // score +=  (2 * rho + 3) * std::log(S_b(p)) + 2 * theta / S_b(p);
      score -=  0.5 * (2 * rho + 3) * std::log(S_b(p)) + (lambda * std::pow(m(p,b) - delta(p), 2.0) + 2 * theta) / S_b(p);
    }
    std::cout << "\nS score: " << score;
    
    return score;
  };
  
  double muLogKernel(arma::uword k, arma::vec mu_k, arma::mat mean_sum) {
    
    arma::uword b = 0;
    double score = 0.0;
    arma::uvec cluster_ind = arma::find(labels == k);
    arma::vec dist_from_mean(P);
    
    for (auto& n : cluster_ind) {
      b = batch_vec(n);
      dist_from_mean = X_t.col(n) - mean_sum.col(b);
      score -= 0.5 * arma::as_scalar(dist_from_mean.t() * cov_comb_inv.slice(k * B + b) * dist_from_mean);
      score += log(arma::normcdf(arma::as_scalar(phi.col(k).t() * arma::diagmat(cov_comb_inv_diag_sqrt.col(k * B + b)) * dist_from_mean)));
    }
    score -= 0.5 * arma::as_scalar(kappa * ((mu_k - xi).t() *  cov_inv.slice(k) * (mu_k - xi)));
    
    std::cout << "\nMu score: " << score;
    
    return score;
  };
  
  double covLogKernel(arma::uword k, 
                      arma::mat cov_k, 
                      double cov_log_det,
                      arma::mat cov_inv,
                      arma::vec cov_comb_log_det,
                      arma::mat cov_comb_inv_diag_sqrt,
                      arma::cube cov_comb_inv) {
    
    arma::uword b = 0;
    double score = 0.0, cdf = 0.0;
    arma::uvec cluster_ind = arma::find(labels == k);
    arma::vec dist_from_mean(P);
    arma::mat curr_sum(P, P);
    for (auto& n : cluster_ind) {
      b = batch_vec(n);
      dist_from_mean = X_t.col(n) - mean_sum.col(k * B + b);
      score -= 0.5 * arma::as_scalar(cov_comb_log_det(b) + (dist_from_mean.t() * cov_comb_inv.slice(b) * dist_from_mean));
      score += log(arma::normcdf(arma::as_scalar(phi.col(k).t() * arma::diagmat(cov_comb_inv_diag_sqrt.col(b)) * dist_from_mean)));
      
      // cdf = log(arma::normcdf(arma::as_scalar(phi.col(k).t() * arma::diagmat(cov_comb_inv_diag_sqrt.col(b)) * dist_from_mean)));
      // 
      // if(isnan(cdf) && ! nan_found){
      //   nan_found = true;
      //   std::cout << "\nLog norm cdf is a NaN.";
      // }
    }
    score -= 0.5 * arma::as_scalar((nu + P + 2) * cov_log_det + kappa * ((mu.col(k) - xi).t() * cov_inv * (mu.col(k) - xi)) + arma::trace(scale * cov_inv));
    // std::cout << "\nCov score: " << score;
    
    return score;
  };
  
  
  double phiLogKernel(arma::uword k, arma::vec phi_k) {
    
    arma::uword b = 0;
    double score = 0.0;
    arma::uvec cluster_ind = arma::find(labels == k);
    arma::vec dist_from_mean(P);
    
    for (auto& n : cluster_ind) {
      b = batch_vec(n);
      dist_from_mean = X_t.col(n) - mean_sum.col(b);
      score += log(arma::normcdf(arma::as_scalar(phi.col(k).t() * arma::diagmat(cov_comb_inv_diag_sqrt.col(k * B + b)) * dist_from_mean)));
      
      // if(isnan(score) && ! nan_found){
      //   nan_found = true;
      //   std::cout << "\nLog norm cdf is a NaN in the dof kernel.";
      // }
      
    }
    
    for(arma::uword p = 0; p < P; p++) {
      score += -0.5 * std::pow(phi_k(p), 2.0) / omega;
    }
    
    // std::cout << "\nPhi score: " << score;
    
    return score;
  };
  
  void batchScaleMetropolis() {

    double u = 0.0, proposed_model_score = 0.0, acceptance_prob = 0.0, current_model_score = 0.0;
    arma::vec S_proposed(P), proposed_cov_comb_log_det(K);
    arma::mat proposed_cov_comb_inv_diag_sqrt(P, K);
    arma::cube proposed_cov_comb(P, P, K), proposed_cov_comb_inv(P, P, K);

    S_proposed.zeros();
    proposed_cov_comb_log_det.zeros();
    proposed_cov_comb_inv_diag_sqrt.zeros();
    proposed_cov_comb.zeros();
    proposed_cov_comb_inv.zeros();

    for(arma::uword b = 0; b < B ; b++) {

      acceptance_prob = 0.0, proposed_model_score = 0.0, current_model_score = 0.0;

      for(arma::uword p = 0; p < P; p++) {
        
        // S_proposed(p) = std::exp(arma::randn() * S_proposal_window + log(S(p, b)));
        // std::cout << "\n\nS:" << S(p,b);

        S_proposed(p) = arma::randg( arma::distr_param( S(p, b) * S_proposal_window, 1.0 / S_proposal_window) );

        // std::cout << "\nS proposed:" << S_proposed(p);

        // Asymmetric proposal density
        proposed_model_score += gammaLogLikelihood(S(p, b), S_proposed(p) * S_proposal_window, S_proposal_window);
        current_model_score += gammaLogLikelihood(S_proposed(p), S(p, b) * S_proposal_window, S_proposal_window);
        
        
      }

      proposed_cov_comb = cov;
      for(arma::uword k = 0; k < K; k++) {
        // proposed_batch_cov_comb.slice(k) = cov.slice(k); // + arma::diagmat(S.col(b))
        for(arma::uword p = 0; p < P; p++) {
          proposed_cov_comb.slice(k)(p, p) *= S_proposed(p);
        }
        proposed_cov_comb_log_det(k) = arma::log_det(proposed_cov_comb.slice(k)).real();
        proposed_cov_comb_inv.slice(k) = arma::inv_sympd(proposed_cov_comb.slice(k));
        proposed_cov_comb_inv_diag_sqrt.col(k) = arma::sqrt(proposed_cov_comb_inv.slice(k).diag());
      }

      proposed_model_score += sLogKernel(b,
        S_proposed,
        proposed_cov_comb_log_det,
        proposed_cov_comb_inv_diag_sqrt,
        proposed_cov_comb_inv
      );

      current_model_score += sLogKernel(b,
        S.col(b),
        cov_comb_log_det.col(b),
        cov_comb_inv_diag_sqrt.cols(KB_inds + b),
        cov_comb_inv.slices(KB_inds + b)
      );
      
      // std::cout << "\nModel scores, current: " << current_model_score << "\nproposed: " << proposed_model_score;

      u = arma::randu();
      acceptance_prob = std::min(1.0, std::exp(proposed_model_score - current_model_score));

      // std::cout << "\n\nProposed S:\n"<< S_proposed << "\n\nCurrent S:\n" << S.col(b) << "\n\nProposed score: " << proposed_model_score << "\nCurrent score: " << current_model_score;

      if(u < acceptance_prob){
        S.col(b) = S_proposed;
        S_count(b)++;

        for(arma::uword k = 0; k < K; k++) {
          cov_comb.slice(k * B + b) = proposed_cov_comb.slice(k);
          cov_comb_log_det(k, b) = proposed_cov_comb_log_det(k);
          cov_comb_inv.slice(k * B + b) = proposed_cov_comb_inv.slice(k);
          cov_comb_inv_diag_sqrt.col(k * B + b) = proposed_cov_comb_inv_diag_sqrt.col(k);
        }
      }
    }
  };

  // void batchShiftMetorpolis() {
  //   
  //   double u = 0.0, proposed_model_score = 0.0, acceptance_prob = 0.0, current_model_score = 0.0;
  //   arma::vec m_proposed(P);
  //   arma::mat proposed_mean_sum(P, K);
  //   m_proposed.zeros();
  //   
  //   for(arma::uword b = 1; b < B ; b++) {
  //     for(arma::uword p = 0; p < P; p++){
  //       // if((m(p, b) < X_min(p)) || (m(p, b) > X_max(p))) {
  //       //   m(p, b) = arma::randn<double>() * S(p, b) / lambda + delta(p);
  //       // } else {
  //       // The proposal window is now a diagonal matrix of common entries.
  //       m_proposed(p) = (arma::randn() * m_proposal_window) + m(p, b);
  //       // }
  //     }
  //     
  //     for(arma::uword k = 0; k < K; k++) {
  //       proposed_mean_sum.col(k) = mu.col(k) + m_proposed;
  //     }
  //     
  //     proposed_model_score = mLogKernel(b, m_proposed, proposed_mean_sum);
  //     
  //     current_model_score = mLogKernel(b, m.col(b), mean_sum.cols(KB_inds + b));
  //     
  //     u = arma::randu();
  //     acceptance_prob = std::min(1.0, std::exp(proposed_model_score - current_model_score));
  //     
  //     // std::cout << "\n\nProposed m:\n"<< m_proposed << "\n\nCurrent m:\n" << m.col(b) << "\n\nProposed score: " << proposed_model_score << "\nCurrent score: " << current_model_score;
  //     
  //     if(u < acceptance_prob){
  //       m.col(b) = m_proposed;
  //       m_count(b)++;
  //       
  //       for(arma::uword k = 0; k < K; k++) {
  //         mean_sum.col(k * B + b) = proposed_mean_sum.col(k);
  //       }
  //     }
  //   }
  // };

  virtual void clusterCovarianceMetropolis() {

    double u = 0.0, proposed_model_score = 0.0, acceptance_prob = 0.0, current_model_score = 0.0, proposed_cov_log_det = 0.0;
    arma::vec proposed_cov_comb_log_det(B);
    arma::mat cov_proposed(P, P), proposed_cov_inv(P, P), proposed_cov_comb_inv_diag_sqrt(P, B);
    arma::cube proposed_cov_comb(P, P, B), proposed_cov_comb_inv(P, P, B);

    cov_proposed.zeros();
    proposed_cov_inv.zeros();
    proposed_cov_comb_log_det.zeros();
    proposed_cov_comb_inv_diag_sqrt.zeros();
    proposed_cov_comb.zeros();
    proposed_cov_comb_inv.zeros();

    for(arma::uword k = 0; k < K ; k++) {

      proposed_cov_comb.zeros();
      acceptance_prob = 0.0, proposed_model_score = 0.0, current_model_score = 0.0;

      if(N_k(k) == 0){
        cov_proposed = arma::iwishrnd(scale, nu);
        proposed_cov_inv = arma::inv_sympd(cov_proposed);
        proposed_cov_log_det = arma::log_det(cov_proposed).real();
        for(arma::uword b = 0; b < B; b++) {
          proposed_cov_comb.slice(b) = cov_proposed; // + arma::diagmat(S.col(b))
          for(arma::uword p = 0; p < P; p++) {
            proposed_cov_comb.slice(b)(p, p) *= S(p, b);
          }
          proposed_cov_comb_log_det(b) = arma::log_det(proposed_cov_comb.slice(b)).real();
          proposed_cov_comb_inv.slice(b) = arma::inv_sympd(proposed_cov_comb.slice(b));
          proposed_cov_comb_inv_diag_sqrt.col(b) = arma::sqrt(proposed_cov_comb.slice(b).diag());
        }
      } else {

        cov_proposed = arma::wishrnd(cov.slice(k) / cov_proposal_window, cov_proposal_window);

        // Log probability under the proposal density
        proposed_model_score = logWishartProbability(cov.slice(k), cov_proposed / cov_proposal_window, cov_proposal_window, P);
        current_model_score = logWishartProbability(cov_proposed, cov.slice(k) / cov_proposal_window, cov_proposal_window, P);

        proposed_cov_inv = arma::inv_sympd(cov_proposed);
        proposed_cov_log_det = arma::log_det(cov_proposed).real();
        for(arma::uword b = 0; b < B; b++) {
          proposed_cov_comb.slice(b) = cov_proposed; // + arma::diagmat(S.col(b))
          for(arma::uword p = 0; p < P; p++) {
            proposed_cov_comb.slice(b)(p, p) *= S(p, b);
          }
          proposed_cov_comb_log_det(b) = arma::log_det(proposed_cov_comb.slice(b)).real();
          proposed_cov_comb_inv.slice(b) = arma::inv_sympd(proposed_cov_comb.slice(b));
          proposed_cov_comb_inv_diag_sqrt.col(b) = arma::sqrt(proposed_cov_comb.slice(b).diag());
        }

        // The boolean variables indicate use of the old manipulated matrix or the
        // proposed.
        proposed_model_score += covLogKernel(k,
          cov_proposed,
          proposed_cov_log_det,
          proposed_cov_inv,
          proposed_cov_comb_log_det,
          proposed_cov_comb_inv_diag_sqrt,
          proposed_cov_comb_inv
        );

        current_model_score += covLogKernel(k,
          cov.slice(k),
          cov_log_det(k),
          cov_inv.slice(k),
          cov_comb_log_det.row(k).t(),
          cov_comb_inv_diag_sqrt.cols(k * B + B_inds),
          cov_comb_inv.slices(k * B + B_inds)
        );

        // std::cout << "\n\nProposed Cov:\n"<< cov_proposed << "\n\nCurrent cov:\n" << cov.slice(k) << "\n\nProposed score: " << proposed_model_score << "\nCurrent score: " << current_model_score;


        // Accept or reject
        u = arma::randu();
        acceptance_prob = std::min(1.0, std::exp(proposed_model_score - current_model_score));
      }
      if((u < acceptance_prob) || (N_k(k) == 0)){
        cov.slice(k) = cov_proposed;
        cov_count(k)++;

        cov_inv.slice(k) = proposed_cov_inv;
        cov_log_det(k) = proposed_cov_log_det;
        for(arma::uword b = 0; b < B; b++) {
          cov_comb.slice(k * B + b) = proposed_cov_comb.slice(b);
          cov_comb_log_det(k, b) = proposed_cov_comb_log_det(b);
          cov_comb_inv.slice(k * B + b) = proposed_cov_comb_inv.slice(b);
          cov_comb_inv_diag_sqrt.col(k * B + b) = proposed_cov_comb_inv_diag_sqrt.col(b);
        }
      }
    }
  };

  // void clusterMeanMetropolis() {
  //   
  //   double u = 0.0, proposed_model_score = 0.0, acceptance_prob = 0.0, current_model_score = 0.0;
  //   arma::vec mu_proposed(P);
  //   arma::mat proposed_mean_sum(P, B);
  //   
  //   mu_proposed.zeros();
  //   proposed_mean_sum.zeros();
  //   
  //   for(arma::uword k = 0; k < K ; k++) {
  //     if(N_k(k) == 0){
  //       mu_proposed = arma::mvnrnd(xi, (1.0/kappa) * cov.slice(k), 1);
  //       for(arma::uword b = 0; b < B; b++) {
  //         proposed_mean_sum.col(b) = mu_proposed + m.col(b);
  //       }
  //     } else {
  //       for(arma::uword p = 0; p < P; p++){
  //         // The proposal window is now a diagonal matrix of common entries.
  //         mu_proposed(p) = (arma::randn() * mu_proposal_window) + mu(p, k);
  //       }
  //       
  //       for(arma::uword b = 0; b < B; b++) {
  //         proposed_mean_sum.col(b) = mu_proposed + m.col(b);
  //       }
  //       
  //       // The prior is included in the kernel
  //       proposed_model_score = muLogKernel(k, mu_proposed, proposed_mean_sum);
  //       current_model_score = muLogKernel(k, mu.col(k), mean_sum.cols(k * B + B_inds));
  //       
  //       u = arma::randu();
  //       acceptance_prob = std::min(1.0, std::exp(proposed_model_score - current_model_score));
  //       
  //     }
  //     
  //     if((u < acceptance_prob) || (N_k(k) == 0)) {
  //       mu.col(k) = mu_proposed;
  //       mu_count(k)++;
  //       
  //       for(arma::uword b = 0; b < B; b++) {
  //         mean_sum.col(k * B + b) = proposed_mean_sum.col(b);
  //       }
  //       
  //     }
  //   }
  // };
  // 
  virtual void clusterShapeMetropolis() {
    
    double u = 0.0, proposed_model_score = 0.0, acceptance_prob = 0.0, current_model_score = 0.0;
    arma::vec phi_proposed(P);
    phi_proposed.zeros();
    
    for(arma::uword k = 0; k < K ; k++) {
      if(N_k(k) == 0){
        for(arma::uword p = 0; p < P; p++) {
          phi_proposed(p) = arma::randn<double>() * omega;
        }
      } else {
        for(arma::uword p = 0; p < P; p++){
          // The proposal window is now a diagonal matrix of common entries.
          phi_proposed(p) = (arma::randn() * phi_proposal_window) + phi(p, k);
        }
        
        
        // The prior is included in the kernel
        proposed_model_score = phiLogKernel(k, phi_proposed);
        current_model_score = phiLogKernel(k, phi.col(k));
        
        u = arma::randu();
        acceptance_prob = std::min(1.0, std::exp(proposed_model_score - current_model_score));
        
      }
      
      if((u < acceptance_prob) || (N_k(k) == 0)) {
        phi.col(k) = phi_proposed;
        phi_count(k)++;
      }
    }
    
  };
  
  virtual void metropolisStep() {
    
    // Metropolis step for cluster parameters
    clusterCovarianceMetropolis();
    clusterMeanMetropolis();
    
    // Update the shape parameter of the skew normal
    clusterShapeMetropolis();
    
    // Metropolis step for batch parameters if more than 1 batch
    // if(B > 1){
    batchScaleMetropolis();
    batchShiftMetorpolis();
    // }
  };
  
};

class msnPredictive : public msnSampler, public semisupervisedSampler {
  
private:
  
public:
  
  using msnSampler::msnSampler;
  
  msnPredictive(
    arma::uword _K,
    arma::uword _B,
    double _mu_proposal_window,
    double _cov_proposal_window,
    double _m_proposal_window,
    double _S_proposal_window,
    double _phi_proposal_window,
    double _rho,
    double _theta,
    double _lambda,
    arma::uvec _labels,
    arma::uvec _batch_vec,
    arma::vec _concentration,
    arma::mat _X,
    arma::uvec _fixed
  ) : 
    sampler(_K, _B, _labels, _batch_vec, _concentration, _X),
    mvnSampler(                           
      _K,
      _B,
      _mu_proposal_window,
      _cov_proposal_window,
      _m_proposal_window,
      _S_proposal_window,
      _rho,
      _theta,
      _lambda,
      _labels,
      _batch_vec,
      _concentration,
      _X
    ),
    msnSampler(_K,
     _B,
     _mu_proposal_window,
     _cov_proposal_window,
     _m_proposal_window,
     _S_proposal_window,
     _phi_proposal_window,
     _rho,
     _theta,
     _lambda,
     _labels,
     _batch_vec,
     _concentration,
     _X),
  semisupervisedSampler(_K, _B, _labels, _batch_vec, _concentration, _X, _fixed)
  {
  };
  
  virtual ~msnPredictive() { };
  
  // virtual void sampleFromPriors() {
  // 
  //   arma::mat X_k;
  // 
  //   for(arma::uword k = 0; k < K; k++){
  //     X_k = X.rows(arma::find(labels == k && fixed == 1));
  //     cov.slice(k) = arma::diagmat(arma::stddev(X_k).t());
  //     mu.col(k) = arma::mean(X_k).t();
  // 
  //     for(arma::uword p = 0; p < P; p++){
  //       phi(p, k) =  arma::randn<double>() * omega;
  //     }
  //   }
  //   for(arma::uword b = 0; b < B; b++){
  //     for(arma::uword p = 0; p < P; p++){
  // 
  //       // Fix the 0th batch at no effect; all other batches have an effect
  //       // relative to this
  //       // if(b == 0){
  //       S(p, b) = 1.0;
  //       m(p, b) = 0.0;
  //       // } else {
  //       // S(p, b) = 1.0 / arma::randg<double>( arma::distr_param(rho, 1.0 / theta ) );
  //       // m(p, b) = arma::randn<double>() * S(p, b) / lambda + delta(p);
  //       // }
  //     }
  //   }
  // };
  
};


class mvtSampler: virtual public mvnSampler {
  
public:

  // arma::uword t_df = 4;
  arma::uword n_param_cluster = 2 + P + P * (P + 1) * 0.5, n_param_batch = 2 * P;
  double psi = 2.0, chi = 0.01, t_df_proposal_window = 0.0, pdf_const = 0.0, t_loc = 2.0;
  arma::uvec t_df_count;
  arma::vec t_df, pdf_coef;
  
  
  using mvnSampler::mvnSampler;
  
  mvtSampler(                           
    arma::uword _K,
    arma::uword _B,
    double _mu_proposal_window,
    double _cov_proposal_window,
    double _m_proposal_window,
    double _S_proposal_window,
    double _t_df_proposal_window,
    double _rho,
    double _theta,
    double _lambda,
    arma::uvec _labels,
    arma::uvec _batch_vec,
    arma::vec _concentration,
    arma::mat _X
  ) : sampler(_K,
  _B,
  _labels,
  _batch_vec,
  _concentration,
  _X), mvnSampler(
      _K,
      _B,
      _mu_proposal_window,
      _cov_proposal_window,
      _m_proposal_window,
      _S_proposal_window,
      _rho,
      _theta,
      _lambda,
      _labels,
      _batch_vec,
      _concentration,
      _X
  ) {
    
    // Hyperparameter for the d.o.f for the t-distn
    // psi = 0.5;
    // chi = 0.5;
    
    t_df.set_size(K);
    t_df.zeros();
    
    pdf_coef.set_size(K);
    pdf_coef.zeros();
    
    t_df_count.set_size(K);
    t_df_count.zeros();
    
    // The shape of the skew normal
    // phi.set_size(P, K);
    // phi.zeros();
    
    // Count the number of times proposed values are accepted
    // phi_count = arma::zeros<arma::uvec>(K);
    
    // The proposal windows for the cluster and batch parameters
    t_df_proposal_window = _t_df_proposal_window;
    
    // The constant for the item likelihood (changes if t_df != const)
    // pdf_const = logGamma(0.5 * (t_df + P)) - logGamma(0.5 * t_df) - 0.5 * P * log(t_df);
  };
  
  
  // Destructor
  virtual ~mvtSampler() { };
  
  // Print the sampler type.
  virtual void printType() {
    std::cout << "\nType: Multivariate T.\n";
  };
  
  double calcPDFCoef(double t_df){
    double x = logGamma(0.5 * (t_df + P)) - logGamma(0.5 * t_df) - 0.5 * P * log(t_df);
    return x;
  };
  
  virtual void sampleDFPrior() {
    for(arma::uword k = 0; k < K; k++){
      // Draw from a shifted gamma distribution (i.e. gamma with location parameter)
      t_df(k) = t_loc + arma::randg<double>( arma::distr_param(psi, 1.0 / chi));
    }
  };
  
  virtual void sampleFromPriors() {
    
    sampleCovPrior();
    sampleMuPrior();
    sampleDFPrior();
    sampleSPrior();
    sampleMPrior();
  };
  
  // virtual void sampleFromPriors() {
  //   
  //   for(arma::uword k = 0; k < K; k++){
  //     cov.slice(k) = arma::iwishrnd(scale, nu);
  //     mu.col(k) = arma::mvnrnd(xi, (1.0/kappa) * cov.slice(k), 1);
  //     
  //     // Draw from a shifted gamma distribution (i.e. gamma with location parameter)
  //     t_df(k) = t_loc + arma::randg<double>( arma::distr_param(psi, 1.0 / chi));
  //   }
  //   for(arma::uword b = 0; b < B; b++){
  //     for(arma::uword p = 0; p < P; p++){
  //       S(p, b) = S_loc + 1.0 / arma::randg<double>( arma::distr_param(rho, 1.0 / theta ) );
  //       m(p, b) = arma::randn<double>() * S(p, b) / lambda + delta(p);
  //     }
  //   }
  // };
  
  // Update the common matrix manipulations to avoid recalculating N times
  virtual void matrixCombinations() {
    
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
  arma::vec itemLogLikelihood(arma::vec item, arma::uword b) {
    
    double x = 0.0, y = 0.0, my_det = 0.0;
    arma::vec ll(K), dist_to_mean(P);
    ll.zeros();
    dist_to_mean.zeros();
    arma::mat my_cov_comv_inv(P, P), my_inv(P, P), my_cov_comb(P, P);
    my_cov_comv_inv.zeros();
    my_inv.zeros();
    my_cov_comb.zeros();
    
    double cov_correction = 0.0;
    
    for(arma::uword k = 0; k < K; k++){
    
      // gamma(0.5 * (nu + P)) / (gamma(0.5 * nu) * nu ^ (0.5 * P) * pi ^ (0.5 * P)  * det(cov) ^ 0.5) * (1 + (1 / nu) * (x - mu)^t * inv(cov) * (x - mu)) ^ (-0.5 * (nu + P))
      // logGamma(0.5 * (nu + P)) - logGamma(0.5 * nu) - (0.5 * P) * log(nu) - 0.5 * P * log(pi) - 0.5 * logDet(cov) -0.5 * (nu + P) * log(1 + (1 / nu) * (x - mu)^t * inv(cov) * (x - mu))
      
      // my_cov_comv_inv = cov.slice(k);
      // for(arma::uword p = 0; p < P; p++) {
      //   my_cov_comv_inv(p, p) *= S(p, b);
      // }
      
      // cov_correction = t_df(k) / (t_df(k) - 2.0);
      
      
      // my_cov_comb = cov.slice(k);
      // 
      // for(arma::uword p = 0; p < P; p++) {
      //   my_cov_comb(p, p) = my_cov_comb(p, p) * S(p, b);
      // }
      // 
      // // my_cov_comb = my_cov_comb / cov_correction;
      // 
      // // std::cout << "\nThe invariance.";
      // 
      // my_inv = arma::inv_sympd(my_cov_comb);
      // 
      // // std::cout << "\nDeterminant.";
      // my_det = arma::log_det(my_cov_comb).real();
      
      // The exponent part of the MVN pdf
      dist_to_mean = item - mean_sum.col(k * B + b);
      x = arma::as_scalar(dist_to_mean.t() * cov_comb_inv.slice(k * B + b) * dist_to_mean);
      // x = arma::as_scalar(dist_to_mean.t() * my_inv * dist_to_mean);
      y = (t_df(k) + P) * log(1.0 + (1/t_df(k)) * x);
      
      ll(k) = pdf_coef(k) - 0.5 * (cov_comb_log_det(k, b) + y + P * log(PI));
      // ll(k) = pdf_coef(k) - 0.5 * (my_det + y + P * log(PI)); 
      
      // std::cout << "\nCheck.";
      
      // if(! arma::approx_equal(mean_sum.col(k * B + b), (mu.col(k) + m.col(b)), "absdiff", 0.001)) {
      //   std::cout << "\n\nMean sum has deviated from expected.";
      // }
      // 
      // if(! arma::approx_equal(cov_comb_inv.slice(k * B + b), my_inv, "absdiff", 0.001)) {
      //   std::cout << "\n\nCovariance inverse has deviated from expected.";
      //   std::cout << "\n\nExpected:\n" << cov_comb_inv.slice(k * B + b) <<
      //     "\n\nCalculated:\n" << my_inv;
      // 
      //   throw std::invalid_argument( "\nMy inverses diverged." );
      // }
      // 
      // if(isnan(ll(k))) {
      //   std::cout << "\nNaN!\n";
      //   
      //   double new_x = (1/t_df(k)) * arma::as_scalar((item - mu.col(k) - m.col(b)).t() * my_inv * (item - mu.col(k) - m.col(b)));
      //   
      //   std::cout << "\n\nItem likelihood:\n" << ll(k) << 
      //     "\nPDF coefficient: " << pdf_coef(k) << "\nLog determinant: " <<
      //       cov_comb_log_det(k, b) << "\nX: " << x << "\nY: " << y <<
      //         "\nLog comp of y: " << 1.0 + (1/t_df(k)) * x <<
      //           "\nLogged: " << log(1.0 + (1/t_df(k)) * x) <<
      //             "\nt_df(k): " << t_df(k) << "\n" << 
      //               "\nMy new x" << new_x << "\nLL alt: " << 
      //                 pdf_coef(k) - 0.5 * (my_det + (t_df(k) + P) * log(1.0 + new_x) + P * log(PI)) <<
      //                   "\n\nCov combined expected:\n" << cov_comb_inv.slice(k * B + b) <<
      //                     "\n\nCov combined real:\n" << my_inv;
      //                 
      //   throw std::invalid_argument( "\nNaN returned from likelihood." );
      //   
      // }
      
      
    }
    
    return(ll);
  };
  
  void calcBIC(){
    
    // Each component has a weight, a mean vector, a symmetric covariance matrix and a
    // degree of freedom parameter. Each batch has a mean and standard
    // deviations vector.
    // arma::uword n_param = (P + P * (P + 1) * 0.5 + 1) * K_occ + (2 * P) * B;
    // BIC = n_param * std::log(N) - 2 * model_likelihood;
    
    // arma::uword n_param_cluster = 2 + P + P * (P + 1) * 0.5;
    // arma::uword n_param_batch = 2 * P;

    BIC = 2 * model_likelihood - (n_param_batch + n_param_batch) * std::log(N);
    
    // for(arma::uword k = 0; k < K; k++) {
    //   BIC -= n_param_cluster * std::log(N_k(k)+ 1);
    // }
    // for(arma::uword b = 0; b < B; b++) {
    //   BIC -= n_param_batch * std::log(N_b(b)+ 1);
    // }
    
  };
  
  double clusterLikelihood(
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
    // 
    // std::cout << "\nScore before halving: " << score << "\nT DF: " << t_df <<
    //   "\n\nCov log det:\n" << cov_det << "\n\nCov inverse:\n " << cov_inv;
    // 
    // 
    
    return (-0.5 * score);
  }
  
  double batchLikelihood(
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

  double mLogKernel(arma::uword b, arma::vec m_b, arma::mat mean_sum) {

    arma::uword k = 0;
    double score = 0.0, score_alt = 0.0;
    arma::vec dist_from_mean(P);
    dist_from_mean.zeros();
    
    score = batchLikelihood(batch_ind(b), 
      labels, 
      cov_comb_log_det.col(b),
      t_df,
      mean_sum,
      cov_comb_inv.slices(KB_inds + b)
    );
    
    // for (auto& n : batch_ind(b)) {
    //   k = labels(n);
    //   dist_from_mean = X_t.col(n) - mean_sum.col(k);
    //   score_alt += cov_comb_log_det(k, b) + (t_df(k) + P) * log(1 + (1/t_df(k)) * arma::as_scalar(dist_from_mean.t() * cov_comb_inv.slice(k * B + b) * dist_from_mean));
    // }
    // 
    // score_alt *= -0.5;
    // 
    // if(std::abs(score_alt - score) > 1e-6) {
    //   std::cout << "\nProblem in m kernel function.\nOld score: " << 
    //     score << "\nAlternative score: " << score_alt;
    //   throw std::invalid_argument( "\n" );
    // }
    
    for(arma::uword p = 0; p < P; p++) {
      score += -0.5 * lambda * std::pow(m_b(p) - delta(p), 2.0) / S(p, b);
    }
    
    // score *= -0.5;
    
    return score;
  };

  double sLogKernel(arma::uword b,
                    arma::vec S_b,
                    arma::vec cov_comb_log_det,
                    arma::cube cov_comb_inv) {

    arma::uword k = 0;
    double score = 0.0, score_alt = 0.0;
    arma::vec dist_from_mean(P);
    dist_from_mean.zeros();
    arma::mat curr_sum(P, P);

    score = batchLikelihood(batch_ind(b), 
      labels, 
      cov_comb_log_det,
      t_df,
      mean_sum.cols(KB_inds + b),
      cov_comb_inv
    );
    
    // for (auto& n : batch_ind(b)) {
    //   k = labels(n);
    //   dist_from_mean = X_t.col(n) - mean_sum.col(k * B + b);
    //   score_alt += (cov_comb_log_det(k) + (t_df(k) + P) * log(1 + (1/t_df(k)) * arma::as_scalar(dist_from_mean.t() * cov_comb_inv.slice(k) * dist_from_mean)));
    // }
    // 
    // score_alt *= -0.5;
    // 
    // if(std::abs(score_alt - score) > 1e-6) {
    //   std::cout << "\nProblem in S kernel function.\nOld score: " << 
    //     score << "\nAlternative score: " << score_alt;
    //   throw std::invalid_argument( "\n" );
    // }
    
    for(arma::uword p = 0; p < P; p++) {
      score += -0.5 * ((2 * rho + 3) * std::log(S_b(p) - S_loc)
                       + 2 * theta / (S_b(p) - S_loc)
                       + lambda * std::pow(m(p,b) - delta(p), 2.0) / S_b(p));

     // score +=   (0.5 - 1) * std::log(S(p,b) - S_loc) 
     //   - 0.5 * (S(p, b) - S_loc)
     //   - 0.5 * lambda * std::pow(m(p,b) - delta(p), 2.0) / S_b(p);
    }
    
    // score *= -0.5;
    return score;
  };

  double muLogKernel(arma::uword k, arma::vec mu_k, arma::mat mean_sum) {

    arma::uword b = 0;
    double score = 0.0, score_alt = 0.0;
    arma::uvec cluster_ind = arma::find(labels == k);
    arma::vec dist_from_mean(P);

    score = clusterLikelihood(
      t_df(k),
      cluster_ind,
      cov_comb_log_det.row(k).t(),
      mean_sum,
      cov_comb_inv.slices(k * B + B_inds)
    );
    
    // for (auto& n : cluster_ind) {
    //   b = batch_vec(n);
    //   dist_from_mean = X_t.col(n) - mean_sum.col(b);
    //   score_alt += cov_comb_log_det(k, b) +  (t_df(k) + P) * log(1 + (1/t_df(k)) * arma::as_scalar(dist_from_mean.t() * cov_comb_inv.slice(k * B + b) * dist_from_mean));
    // }
    // 
    // score_alt *= -0.5;
    // 
    // if(std::abs(score_alt - score) > 1e-6) {
    //   std::cout << "\nProblem in mu kernel function.\nOld score: " << 
    //     score << "\nAlternative score: " << score_alt;
    //   throw std::invalid_argument( "\n" );
    // }
    
    score += -0.5 * arma::as_scalar(kappa * ((mu_k - xi).t() *  cov_inv.slice(k) * (mu_k - xi)));
    // score *= -0.5;
    
    return score;
  };

  double covLogKernel(arma::uword k, 
                      arma::mat cov_k,
                      double cov_log_det,
                      arma::mat cov_inv,
                      arma::vec cov_comb_log_det,
                      arma::cube cov_comb_inv) {

    arma::uword b = 0;
    double score = 0.0, score_alt = 0.0;
    arma::uvec cluster_ind = arma::find(labels == k);
    arma::vec dist_from_mean(P);
    
    score = clusterLikelihood(
      t_df(k),
      cluster_ind,
      cov_comb_log_det,
      mean_sum.cols(k * B + B_inds),
      cov_comb_inv
    );

    
    // for (auto& n : cluster_ind) {
    //   b = batch_vec(n);
    //   dist_from_mean = X_t.col(n) - mean_sum.col(k * B + b);
    //   score_alt += cov_comb_log_det(b) + (t_df(k) + P) * log(1 + (1/t_df(k)) * arma::as_scalar(dist_from_mean.t() * cov_comb_inv.slice(b) * dist_from_mean));
    // }
    // 
    // score_alt *= -0.5;
    // 
    // if(std::abs(score_alt - score) > 1e-6) {
    //   std::cout << "\nProblem in cov kernel function.\nOld score: " << 
    //     score << "\nAlternative score: " << score_alt;
    //   
    //   std::cout << "\nT DF: " << t_df(k) << "\n\nCov log det:\n" << cov_comb_log_det <<
    //     "\n\nCov inverse:\n " << cov_comb_inv;
    //   
    //   // std::cout << "\n\nMean sums:\n";
    //   // for(arma::uword b = 0;b < B; b++){
    //   //   std::cout << mean_sum.col(k * B + b) << "\n\n";
    //   // }
    //   // std::cout << "\n\nMean sums:\n" << mean_sum.cols(k * B + B_inds);
    //   throw std::invalid_argument( "\n" );
    // }
    
    score += -0.5 *( arma::as_scalar((nu + P + 2) * cov_log_det 
                    + kappa * ((mu.col(k) - xi).t() * cov_inv * (mu.col(k) - xi)) 
                    + arma::trace(scale * cov_inv)));
    // score *= -0.5;
    
    return score;
  };
  
  double dfLogKernel(arma::uword k, 
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
    // score += (psi - 1) * log(t_df - t_loc) - (t_df - t_loc) / chi;
    score += (psi - 1) * log(t_df - t_loc) - chi * (t_df - t_loc);
    return score;
  };
  
  void clusterDFMetropolis() {
    double u = 0.0, proposed_model_score = 0.0, acceptance_prob = 0.0, current_model_score = 0.0, t_df_proposed = 0.0, proposed_pdf_coef = 0.0;
    
    for(arma::uword k = 0; k < K ; k++) {
      proposed_model_score = 0.0, acceptance_prob = 0.0, current_model_score = 0.0, t_df_proposed = 0.0, proposed_pdf_coef = 0.0;
      if(N_k(k) == 0){
        t_df_proposed = t_loc + arma::randg<double>( arma::distr_param(psi, 1.0 / chi));
        proposed_pdf_coef = calcPDFCoef(t_df_proposed);
      } else {
        
        // std::cout << "\n\nT df.\nPsi: " << psi << "\nChi: " << chi
        // << "\nWindow: " << t_df_proposal_window << "\nCurrent: " << t_df(k);
        
        t_df_proposed = t_loc + arma::randg( arma::distr_param( (t_df(k) - t_loc) * t_df_proposal_window, 1.0 / t_df_proposal_window) );
        
        // t_df_proposed = t_loc + std::exp(arma::randn() * t_df_proposal_window + log(t_df(k) - t_loc) );
        
        // proposed_model_score = logNormalLogProbability(t_df(k) - t_loc, t_df_proposed - t_loc, t_df_proposal_window);
        // current_model_score = logNormalLogProbability(t_df_proposed - t_loc, t_df(k) - t_loc, t_df_proposal_window);
        // 
        // std::cout  << "\nProposed score: " << proposed_model_score << "\nCurrent score: " << current_model_score;
        
        // t_df_proposed = t_loc + std::exp((arma::randn() * t_df_proposal_window) + t_df(k) - t_loc);

        // // Log probability under the proposal density
        // proposed_model_score = logNormalLogProbability(t_df(k) - t_loc, (t_df_proposed - t_loc), t_df_proposal_window);
        // current_model_score = logNormalLogProbability(t_df_proposed - t_loc, (t_df(k) - t_loc), t_df_proposal_window);
        
        // Proposed value
        // t_df_proposed = t_loc + arma::randg( arma::distr_param( (t_df(k) - t_loc) * t_df_proposal_window, 1.0 / t_df_proposal_window) );
        proposed_pdf_coef = calcPDFCoef(t_df_proposed);

        // std::cout << "\n\nDF: " << t_df(k) << "\nProposed DF: " << t_df_proposed;
        
        // Asymmetric proposal density
        proposed_model_score = gammaLogLikelihood(t_df(k) - t_loc, (t_df_proposed - t_loc) * t_df_proposal_window, t_df_proposal_window);
        current_model_score = gammaLogLikelihood(t_df_proposed - t_loc, (t_df(k) - t_loc) * t_df_proposal_window, t_df_proposal_window);

        // The prior is included in the kernel
        proposed_model_score = dfLogKernel(k, t_df_proposed, proposed_pdf_coef);
        current_model_score = dfLogKernel(k, t_df(k), pdf_coef(k));
        
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
  
  virtual void metropolisStep() {
    
    // Metropolis step for cluster parameters
    clusterCovarianceMetropolis();
    
    // std::cout << "\n\nCluster covariance.";
    
    // matrixCombinations();
    
    clusterMeanMetropolis();
    
    // std::cout << "\n\nCluster mean.";
    
    // matrixCombinations();

    // Update the shape parameter of the skew normal
    clusterDFMetropolis();
    
    // std::cout << "\n\nCluster df.";
    
    // matrixCombinations();

    // Metropolis step for batch parameters if more than 1 batch
    // if(B > 1){
    batchScaleMetropolis();
    
    // std::cout << "\n\nBatch scale.";
    
    // matrixCombinations(); 
    
    batchShiftMetorpolis();
    
    // std::cout << "\n\nBatch mean.";
    
    // matrixCombinations();
    
    // }
  };
  
};

class mvtPredictive : public mvtSampler, public semisupervisedSampler {
  
private:
  
public:
  
  using mvtSampler::mvtSampler;
  
  mvtPredictive(
    arma::uword _K,
    arma::uword _B,
    double _mu_proposal_window,
    double _cov_proposal_window,
    double _m_proposal_window,
    double _S_proposal_window,
    double _t_df_proposal_window,
    double _rho,
    double _theta,
    double _lambda,
    arma::uvec _labels,
    arma::uvec _batch_vec,
    arma::vec _concentration,
    arma::mat _X,
    arma::uvec _fixed
  ) : 
    sampler(_K, _B, _labels, _batch_vec, _concentration, _X),
    mvnSampler(_K,
      _B,
      _mu_proposal_window,
      _cov_proposal_window,
      _m_proposal_window,
      _S_proposal_window,
      _rho,
      _theta,
      _lambda,
      _labels,
      _batch_vec,
      _concentration,
      _X
    ), mvtSampler(                           
      _K,
      _B,
      _mu_proposal_window,
      _cov_proposal_window,
      _m_proposal_window,
      _S_proposal_window,
      _t_df_proposal_window,
      _rho,
      _theta,
      _lambda,
      _labels,
      _batch_vec,
      _concentration,
      _X
    ), semisupervisedSampler(_K, _B, _labels, _batch_vec, _concentration, _X, _fixed)
    {
    };
 
  virtual ~mvtPredictive() { };
  
  // virtual void sampleFromPriors() {
  //   
  //   arma::mat X_k;
  //   
  //   for(arma::uword k = 0; k < K; k++){
  //     X_k = X.rows(arma::find(labels == k && fixed == 1));
  //     cov.slice(k) = arma::diagmat(arma::stddev(X_k).t());
  //     mu.col(k) = arma::mean(X_k).t();
  //     
  //     // Draw from a shifted gamma distribution (i.e. gamma with location parameter)
  //     t_df(k) = t_loc + arma::randg<double>( arma::distr_param(psi, 1.0 / chi));
  //     
  //   }
  //   for(arma::uword b = 0; b < B; b++){
  //     for(arma::uword p = 0; p < P; p++){
  //       
  //       // Fix the 0th batch at no effect; all other batches have an effect
  //       // relative to this
  //       // if(b == 0){
  //       S(p, b) = 1.0;
  //       m(p, b) = 0.0;
  //       // } else {
  //       // S(p, b) = 1.0 / arma::randg<double>( arma::distr_param(rho, 1.0 / theta ) );
  //       // m(p, b) = arma::randn<double>() * S(p, b) / lambda + delta(p);
  //       // }
  //     }
  //   }
  // };
  
};



// Factory for creating instances of samplers
//' @name samplerFactory
//' @title Factory for different sampler subtypes.
//' @description The factory allows the type of mixture implemented to change 
//' based upon the user input.
//' @field new Constructor \itemize{
//' \item Parameter: samplerType - the density type to be modelled
//' \item Parameter: K - the number of components to model
//' \item Parameter: labels - the initial clustering of the data
//' \item Parameter: concentration - the vector for the prior concentration of 
//' the Dirichlet distribution of the component weights
//' \item Parameter: X - the data to model
//' }
class samplerFactory
{
public:
  enum samplerType {
    // G = 0,
    MVN = 1,
    MVT = 2,
    MSN = 3
  };
  
  static std::unique_ptr<sampler> createSampler(samplerType type,
    arma::uword K,
    arma::uword B,
    double mu_proposal_window,
    double cov_proposal_window,
    double m_proposal_window,
    double S_proposal_window,
    double t_df_proposal_window,
    double phi_proposal_window,
    double rho,
    double theta,
    double lambda,
    arma::uvec labels,
    arma::uvec batch_vec,
    arma::vec concentration,
    arma::mat X
  ) {
    switch (type) {
    // case G: return std::make_unique<gaussianSampler>(K, labels, concentration, X);
      
    case MVN: return std::make_unique<mvnSampler>(K,
                                                  B,
                                                  mu_proposal_window,
                                                  cov_proposal_window,
                                                  m_proposal_window,
                                                  S_proposal_window,
                                                  rho,
                                                  theta,
                                                  lambda,
                                                  labels,
                                                  batch_vec,
                                                  concentration,
                                                  X);
    case MVT: return std::make_unique<mvtSampler>(K,
                                                  B,
                                                  mu_proposal_window,
                                                  cov_proposal_window,
                                                  m_proposal_window,
                                                  S_proposal_window,
                                                  t_df_proposal_window,
                                                  rho,
                                                  theta,
                                                  lambda,
                                                  labels,
                                                  batch_vec,
                                                  concentration,
                                                  X);
    case MSN: return std::make_unique<msnSampler>(K,
                                                  B,
                                                  mu_proposal_window,
                                                  cov_proposal_window,
                                                  m_proposal_window,
                                                  S_proposal_window,
                                                  phi_proposal_window,
                                                  rho,
                                                  theta,
                                                  lambda,
                                                  labels,
                                                  batch_vec,
                                                  concentration,
                                                  X);
    default: throw "invalid sampler type.";
    }
    
  }
  
};


// Factory for creating instances of samplers
//' @name semisupervisedSamplerFactory
//' @title Factory for different sampler subtypes.
//' @description The factory allows the type of mixture implemented to change 
//' based upon the user input.
//' @field new Constructor \itemize{
//' \item Parameter: samplerType - the density type to be modelled
//' \item Parameter: K - the number of components to model
//' \item Parameter: labels - the initial clustering of the data
//' \item Parameter: concentration - the vector for the prior concentration of 
//' the Dirichlet distribution of the component weights
//' \item Parameter: X - the data to model
//' }
class semisupervisedSamplerFactory
{
public:
  enum samplerType {
    // G = 0,
    MVN = 1,
    MVT = 2,
    MSN = 3
  };
  
  static std::unique_ptr<semisupervisedSampler> createSemisupervisedSampler(samplerType type,
    arma::uword K,
    arma::uword B,
    double mu_proposal_window,
    double cov_proposal_window,
    double m_proposal_window,
    double S_proposal_window,
    double t_df_proposal_window,
    double phi_proposal_window,
    double rho,
    double theta,
    double lambda,
    arma::uvec labels,
    arma::uvec batch_vec,
    arma::vec concentration,
    arma::mat X,
    arma::uvec fixed
    ) {
      switch (type) {
      // case G: return std::make_unique<gaussianSampler>(K, labels, concentration, X);
        
      case MVN: return std::make_unique<mvnPredictive>(K,
                                                    B,
                                                    mu_proposal_window,
                                                    cov_proposal_window,
                                                    m_proposal_window,
                                                    S_proposal_window,
                                                    rho,
                                                    theta,
                                                    lambda,
                                                    labels,
                                                    batch_vec,
                                                    concentration,
                                                    X,
                                                    fixed);
      case MVT: return std::make_unique<mvtPredictive>(K,
                                                    B,
                                                    mu_proposal_window,
                                                    cov_proposal_window,
                                                    m_proposal_window,
                                                    S_proposal_window,
                                                    t_df_proposal_window,
                                                    rho,
                                                    theta,
                                                    lambda,
                                                    labels,
                                                    batch_vec,
                                                    concentration,
                                                    X,
                                                    fixed);
      case MSN: return std::make_unique<msnPredictive>(K,
                                                    B,
                                                    mu_proposal_window,
                                                    cov_proposal_window,
                                                    m_proposal_window,
                                                    S_proposal_window,
                                                    phi_proposal_window,
                                                    rho,
                                                    theta,
                                                    lambda,
                                                    labels,
                                                    batch_vec,
                                                    concentration,
                                                    X,
                                                    fixed);
      default: throw "invalid sampler type.";
      }
      
    }
  
};



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
    double rho,
    double theta,
    double lambda,
    arma::uword R,
    arma::uword thin,
    arma::vec concentration,
    bool verbose = true,
    bool doCombinations = false,
    bool printCovariance = false
) {
  
  // The random seed is set at the R level via set.seed() apparently.
  // std::default_random_engine generator(seed);
  // arma::arma_rng::set_seed(seed);
  

  mvnSampler my_sampler(K,
                        B,
                        mu_proposal_window,
                        cov_proposal_window,
                        m_proposal_window,
                        S_proposal_window,
                        rho,
                        theta,
                        lambda,
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
  
  // We use this enough that declaring it is worthwhile
  arma::uword P = X.n_cols;
  
  // The output matrix
  arma::umat class_record(floor(R / thin), X.n_rows);
  class_record.zeros();
  
  // We save the BIC at each iteration
  arma::vec BIC_record = arma::zeros<arma::vec>(floor(R / thin));
  arma::vec model_likelihood = arma::zeros<arma::vec>(floor(R / thin));
  arma::uvec acceptance_vec = arma::zeros<arma::uvec>(floor(R / thin));
  arma::mat weights_saved = arma::zeros<arma::mat>(floor(R / thin), K);
  
  arma::cube mean_sum_saved(P, K * B, floor(R / thin)), mu_saved(P, K, floor(R / thin)), m_saved(P, B, floor(R / thin)), cov_saved(P, K * P, floor(R / thin)), t_saved(P, B, floor(R / thin)), cov_comb_saved(P, P * K * B, floor(R / thin));
  // arma::field<arma::cube> cov_saved(my_sampler.P, my_sampler.P, K, floor(R / thin));
  mu_saved.zeros();
  cov_saved.zeros();
  cov_comb_saved.zeros();
  m_saved.zeros();
  t_saved.zeros();
  
  arma::uword save_int = 0;
  
  // Sampler from priors
  my_sampler.sampleFromPriors();
  my_sampler.matrixCombinations();
  // my_sampler.modelScore();
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
      model_likelihood( save_int ) = my_sampler.model_likelihood;
      class_record.row( save_int ) = my_sampler.labels.t();
      acceptance_vec( save_int ) = my_sampler.accepted;
      weights_saved.row( save_int ) = my_sampler.w.t();
      mu_saved.slice( save_int ) = my_sampler.mu;
      // tau_saved.slice( save_int ) = my_sampler.tau;
      // cov_saved( save_int ) = my_sampler.cov;
      m_saved.slice( save_int ) = my_sampler.m;
      t_saved.slice( save_int ) = my_sampler.S;
      mean_sum_saved.slice( save_int ) = my_sampler.mean_sum;
      
      
      cov_saved.slice ( save_int ) = arma::reshape(arma::mat(my_sampler.cov.memptr(), my_sampler.cov.n_elem, 1, false), P, P * K);
      cov_comb_saved.slice( save_int) = arma::reshape(arma::mat(my_sampler.cov_comb.memptr(), my_sampler.cov_comb.n_elem, 1, false), P, P * K * B); 
      
      if(printCovariance) {  
        std::cout << "\n\nCovariance cube:\n" << my_sampler.cov;
        std::cout << "\n\nBatch covariance matrix:\n" << my_sampler.S;
      }
      
      save_int++;
    }
  }
  
  if(verbose) {
    std::cout << "\n\nCovariance acceptance rate:\n" << arma::conv_to< arma::vec >::from(my_sampler.cov_count) / R;
    std::cout << "\n\ncluster mean acceptance rate:\n" << arma::conv_to< arma::vec >::from(my_sampler.mu_count) / R;
    std::cout << "\n\nBatch covariance acceptance rate:\n" << arma::conv_to< arma::vec >::from(my_sampler.S_count) / R;
    std::cout << "\n\nBatch mean acceptance rate:\n" << arma::conv_to< arma::vec >::from(my_sampler.m_count) / R;
  }
  
  return(List::create(Named("samples") = class_record, 
                      Named("means") = mu_saved,
                      Named("covariance") = cov_saved,
                      Named("batch_shift") = m_saved,
                      Named("batch_scale") = t_saved,
                      Named("mean_sum") = mean_sum_saved,
                      Named("cov_comb") = cov_comb_saved,
                      Named("weights") = weights_saved,
                      Named("cov_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.cov_count) / R,
                      Named("mu_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.mu_count) / R,
                      Named("S_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.S_count) / R,
                      Named("m_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.m_count) / R,                      
                      Named("likelihood") = model_likelihood,
                      Named("BIC") = BIC_record));
  
};







// [[Rcpp::export]]
Rcpp::List sampleMSN (
    arma::mat X,
    arma::uword K,
    arma::uword B,
    arma::uvec labels,
    arma::uvec batch_vec,
    double mu_proposal_window,
    double cov_proposal_window,
    double m_proposal_window,
    double S_proposal_window,
    double phi_proposal_window,
    double rho,
    double theta,
    double lambda,
    arma::uword R,
    arma::uword thin,
    arma::vec concentration,
    bool verbose = true,
    bool doCombinations = false,
    bool printCovariance = false
) {
  
  // The random seed is set at the R level via set.seed() apparently.
  // std::default_random_engine generator(seed);
  // arma::arma_rng::set_seed(seed);
  
  msnSampler my_sampler(K,
                          B,
                          mu_proposal_window,
                          cov_proposal_window,
                          m_proposal_window,
                          S_proposal_window,
                          phi_proposal_window,
                          rho,
                          theta,
                          lambda,
                          labels,
                          batch_vec,
                          concentration,
                          X
  );
  
  // We use this enough that declaring it is worthwhile
  arma::uword P = X.n_cols;
  
  // The output matrix
  arma::umat class_record(floor(R / thin), X.n_rows);
  class_record.zeros();
  
  // We save the BIC at each iteration
  arma::vec BIC_record = arma::zeros<arma::vec>(floor(R / thin));
  arma::vec model_likelihood = arma::zeros<arma::vec>(floor(R / thin));
  arma::uvec acceptance_vec = arma::zeros<arma::uvec>(floor(R / thin));
  arma::mat weights_saved = arma::zeros<arma::mat>(floor(R / thin), K);
  
  arma::cube mean_sum_save(my_sampler.P, K * B, floor(R / thin)), mu_saved(my_sampler.P, K, floor(R / thin)), m_saved(my_sampler.P, B, floor(R / thin)), cov_saved(P, K * P, floor(R / thin)), t_saved(P, B, floor(R / thin)), cov_comb_saved(P, P * K * B, floor(R / thin)), phi_saved(my_sampler.P, K, floor(R / thin));
  // arma::field<arma::cube> cov_saved(my_sampler.P, my_sampler.P, K, floor(R / thin));
  mu_saved.zeros();
  cov_saved.zeros();
  cov_comb_saved.zeros();
  m_saved.zeros();
  t_saved.zeros();
  
  arma::uword save_int = 0;
  
  // Sampler from priors
  my_sampler.sampleFromPriors();
  my_sampler.matrixCombinations();
  
  // Iterate over MCMC moves
  for(arma::uword r = 0; r < R; r++){
    
    my_sampler.updateWeights();
    
    // Metropolis step for batch parameters
    my_sampler.metropolisStep(); 
    
    my_sampler.updateAllocation();
    
    // Record results
    if((r + 1) % thin == 0){
      
      my_sampler.calcBIC();
      BIC_record( save_int ) = my_sampler.BIC;
      model_likelihood( save_int ) = my_sampler.model_likelihood;
      class_record.row( save_int ) = my_sampler.labels.t();
      acceptance_vec( save_int ) = my_sampler.accepted;
      weights_saved.row( save_int ) = my_sampler.w.t();
      mu_saved.slice( save_int ) = my_sampler.mu;
      // tau_saved.slice( save_int ) = my_sampler.tau;
      // cov_saved( save_int ) = my_sampler.cov;
      m_saved.slice( save_int ) = my_sampler.m;
      t_saved.slice( save_int ) = my_sampler.S;
      mean_sum_save.slice( save_int ) = my_sampler.mean_sum;
      phi_saved.slice( save_int ) = my_sampler.phi;
      
      cov_saved.slice( save_int ) = arma::reshape(arma::mat(my_sampler.cov.memptr(), my_sampler.cov.n_elem, 1, false), P, P * K);
      cov_comb_saved.slice( save_int) = arma::reshape(arma::mat(my_sampler.cov_comb.memptr(), my_sampler.cov_comb.n_elem, 1, false), P, P * K * B);
      
      if(printCovariance) {  
        std::cout << "\n\nCovariance cube:\n" << my_sampler.cov;
        std::cout << "\n\nBatch covariance matrix:\n" << my_sampler.S;
      }
      
      save_int++;
    }
  }
  
  if(verbose) {
    std::cout << "\n\nCovariance acceptance rate:\n" << my_sampler.cov_count;
    std::cout << "\n\ncluster mean acceptance rate:\n" << my_sampler.mu_count;
    std::cout << "\n\nCluster shape acceptance rate:\n" << my_sampler.phi_count;
    std::cout << "\n\nBatch covariance acceptance rate:\n" << my_sampler.S_count;
    std::cout << "\n\nBatch mean acceptance rate:\n" << my_sampler.m_count;
  }
  
  return(List::create(Named("samples") = class_record, 
                      Named("means") = mu_saved,
                      Named("covariance") = cov_saved,
                      Named("cov_comb") = cov_comb_saved,
                      Named("shapes") = phi_saved,
                      Named("batch_shift") = m_saved,
                      Named("batch_scale") = t_saved,
                      Named("mean_sum") = mean_sum_save,
                      Named("weights") = weights_saved,
                      Named("cov_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.cov_count) / R,
                      Named("mu_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.mu_count) / R,
                      Named("S_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.S_count) / R,
                      Named("m_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.m_count) / R,
                      Named("phi_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.phi_count) / R,                      Named("likelihood") = model_likelihood,
                      Named("BIC") = BIC_record));
  
};

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
    double rho,
    double theta,
    double lambda,
    arma::uword R,
    arma::uword thin,
    arma::vec concentration,
    bool verbose = true,
    bool doCombinations = false,
    bool printCovariance = false
) {
  
  // The random seed is set at the R level via set.seed() apparently.
  // std::default_random_engine generator(seed);
  // arma::arma_rng::set_seed(seed);
  
  mvtSampler my_sampler(K,
    B,
    mu_proposal_window,
    cov_proposal_window,
    m_proposal_window,
    S_proposal_window,
    t_df_proposal_window,
    rho,
    theta,
    lambda,
    labels,
    batch_vec,
    concentration,
    X
  );
  
  // We use this enough that declaring it is worthwhile
  arma::uword P = X.n_cols;
  
  // The output matrix
  arma::umat class_record(floor(R / thin), X.n_rows);
  class_record.zeros();
  
  // We save the BIC at each iteration
  arma::vec BIC_record = arma::zeros<arma::vec>(floor(R / thin));
  arma::vec model_likelihood = arma::zeros<arma::vec>(floor(R / thin));
  arma::uvec acceptance_vec = arma::zeros<arma::uvec>(floor(R / thin));
  arma::mat weights_saved(floor(R / thin), K), t_df_saved(floor(R / thin), K);
  weights_saved.zeros();
  t_df_saved.zeros();
  
  arma::cube mean_sum_saved(my_sampler.P, K * B, floor(R / thin)), mu_saved(my_sampler.P, K, floor(R / thin)), m_saved(my_sampler.P, B, floor(R / thin)), cov_saved(P, K * P, floor(R / thin)), t_saved(P, B, floor(R / thin)), cov_comb_saved(P, P * K * B, floor(R / thin));
  mu_saved.zeros();
  cov_saved.zeros();
  cov_comb_saved.zeros();
  m_saved.zeros();
  t_saved.zeros();
  
  arma::uword save_int = 0;
  
  // Sampler from priors
  my_sampler.sampleFromPriors();
  my_sampler.matrixCombinations();
  
  // Iterate over MCMC moves
  for(arma::uword r = 0; r < R; r++){
    
    my_sampler.updateWeights();
    
    // std::cout << "\nWeights.\n";
    
    // Metropolis step for batch parameters
    my_sampler.metropolisStep(); 
    
    // std::cout << "\nMetropolis.\n";
    
    my_sampler.updateAllocation();
    
    // std::cout << "\nAllocation.\n";
    
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
      model_likelihood( save_int ) = my_sampler.model_likelihood;
      class_record.row( save_int ) = my_sampler.labels.t();
      acceptance_vec( save_int ) = my_sampler.accepted;
      weights_saved.row( save_int ) = my_sampler.w.t();
      mu_saved.slice( save_int ) = my_sampler.mu;
      m_saved.slice( save_int ) = my_sampler.m;
      t_saved.slice( save_int ) = my_sampler.S;
      mean_sum_saved.slice( save_int ) = my_sampler.mean_sum;
      t_df_saved.row( save_int ) = my_sampler.t_df.t();
      
      cov_saved.slice ( save_int ) = arma::reshape(arma::mat(my_sampler.cov.memptr(), my_sampler.cov.n_elem, 1, false), P, P * K);
      cov_comb_saved.slice( save_int) = arma::reshape(arma::mat(my_sampler.cov_comb.memptr(), my_sampler.cov_comb.n_elem, 1, false), P, P * K * B); 
      if(printCovariance) {  
        std::cout << "\n\nCovariance cube:\n" << my_sampler.cov;
        std::cout << "\n\nBatch covariance matrix:\n" << my_sampler.S;
      }
      
      save_int++;
    }
  }
  
  if(verbose) {

    std::cout << "\n\nCovariance acceptance rate:\n" << arma::conv_to< arma::vec >::from(my_sampler.cov_count) / R;
    std::cout << "\n\ncluster mean acceptance rate:\n" << arma::conv_to< arma::vec >::from(my_sampler.mu_count) / R;
    std::cout << "\n\nBatch covariance acceptance rate:\n" << arma::conv_to< arma::vec >::from(my_sampler.S_count) / R;
    std::cout << "\n\nBatch mean acceptance rate:\n" << arma::conv_to< arma::vec >::from(my_sampler.m_count) / R;
    std::cout << "\n\nCluster t d.f. acceptance rate:\n" << arma::conv_to< arma::vec >::from(my_sampler.t_df_count) / R;
    
  }
  
  return(List::create(Named("samples") = class_record, 
    Named("means") = mu_saved,
    Named("covariance") = cov_saved,
    Named("batch_shift") = m_saved,
    Named("batch_scale") = t_saved,
    Named("mean_sum") = mean_sum_saved,
    Named("cov_comb") = cov_comb_saved,
    Named("t_df") = t_df_saved,
    Named("weights") = weights_saved,
    Named("cov_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.cov_count) / R,
    Named("mu_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.mu_count) / R,
    Named("S_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.S_count) / R,
    Named("m_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.m_count) / R,
    Named("t_df_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.t_df_count) / R,
    Named("likelihood") = model_likelihood,
    Named("BIC") = BIC_record)
  );
  
};


//' @title Mixture model
//' @description Performs MCMC sampling for a mixture model.
//' @param X The data matrix to perform clustering upon (items to cluster in rows).
//' @param K The number of components to model (upper limit on the number of clusters found).
//' @param labels Vector item labels to initialise from.
//' @param fixed Binary vector of the items that are fixed in their initial label.
//' @param dataType Int, 0: independent Gaussians, 1: Multivariate normal, or 2: Categorical distributions.
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
    double rho,
    double theta,
    double lambda,
    arma::uword R,
    arma::uword thin,
    arma::vec concentration,
    bool verbose = true,
    bool doCombinations = false,
    bool printCovariance = false
) {
  
  // // Set the random number
  // std::default_random_engine generator(seed);
  // 
  // // Declare the factory
  // semisupervisedSamplerFactory my_factory;
  // 
  // // Convert from an int to the samplerType variable for our Factory
  // semisupervisedSamplerFactory::samplerType val = static_cast<semisupervisedSamplerFactory::samplerType>(dataType);
  // 
  // // Make a pointer to the correct type of sampler
  // std::unique_ptr<sampler> sampler_ptr = my_factory.createSemisupervisedSampler(val,
  //                                                                               K,
  //                                                                               labels,
  //                                                                               concentration,
  //                                                                               X,
  //                                                                               fixed);
  
  
  mvnPredictive my_sampler(K,
                           B,
                           mu_proposal_window,
                           cov_proposal_window,
                           m_proposal_window,
                           S_proposal_window,
                           rho,
                           theta,
                           lambda,
                           labels,
                           batch_vec,
                           concentration,
                           X,
                           fixed
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
  
  arma::uword P = X.n_cols, N = X.n_rows;
  
  // arma::uword restart_count = 0, n_restarts = 3, check_iter = 250;
  // double min_acceptance = 0.15;
  // 
  // restart:
  
  // The output matrix
  arma::umat class_record(floor(R / thin), X.n_rows);
  class_record.zeros();
  
  // We save the BIC at each iteration
  arma::vec BIC_record = arma::zeros<arma::vec>(floor(R / thin));
  arma::vec model_likelihood = arma::zeros<arma::vec>(floor(R / thin));
  arma::uvec acceptance_vec = arma::zeros<arma::uvec>(floor(R / thin));
  arma::mat weights_saved = arma::zeros<arma::mat>(floor(R / thin), K);
  
  arma::cube mean_sum_saved(P, K * B, floor(R / thin)), 
    mu_saved(P, K, floor(R / thin)),
    m_saved(P, B, floor(R / thin)), 
    cov_saved(P, K * P, floor(R / thin)),
    S_saved(P, B, floor(R / thin)), 
    cov_comb_saved(P, P * K * B, floor(R / thin)), 
    alloc_prob(N, K, floor(R / thin)), 
    batch_corrected_data(N, P, floor(R / thin));
  
  // arma::field<arma::cube> cov_saved(my_sampler.P, my_sampler.P, K, floor(R / thin));
  mu_saved.zeros();
  cov_saved.zeros();
  cov_comb_saved.zeros();
  m_saved.zeros();
  S_saved.zeros();
  
  arma::uword save_int = 0;
  
  // Sampler from priors
  my_sampler.sampleFromPriors();
  
  my_sampler.matrixCombinations();
  // my_sampler.modelScore();
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
    
    // my_sampler.checkPositiveDefinite(r);
    // 
    // if(r == check_iter) {
    // 
    //   if(any((arma::conv_to< arma::vec >::from(my_sampler.cov_count) / R) < min_acceptance)){
    //     if(restart_count == n_restarts) {
    //       std::cout << "Cluster covariance acceptance rates too low.\nPlease restart with a different proposal window and/or random seed.";
    //       throw;
    //     }
    //     restart_count++;
    //     goto restart;
    //   }
    //   if(any((arma::conv_to< arma::vec >::from(my_sampler.mu_count) / R) < min_acceptance)){
    //     if(restart_count == n_restarts) {
    //       std::cout << "Cluster mean acceptance rates too low.\nPlease restart with a different proposal window and/or random seed.";
    //       throw;
    //     }
    //     restart_count++;
    //     goto restart;
    //   }
    //   if(any((arma::conv_to< arma::vec >::from(my_sampler.m_count) / R) < min_acceptance)){
    //     if(restart_count == n_restarts) {
    //       std::cout << "Batch shift acceptance rates too low.\nPlease restart with a different proposal window and/or random seed.";
    //       throw;
    //     }
    //     restart_count++;
    //     goto restart;
    //   }
    //   
    //   if(any((arma::conv_to< arma::vec >::from(my_sampler.S_count) / R) < min_acceptance)){
    //     if(restart_count == n_restarts) {
    //       std::cout << "Batch scale acceptance rates too low.\nPlease restart with a different proposal window and/or random seed.";
    //       throw;
    //     }
    //     restart_count++;
    //     goto restart;
    //   }
    // }
    
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
      model_likelihood( save_int ) = my_sampler.model_likelihood;
      class_record.row( save_int ) = my_sampler.labels.t();
      acceptance_vec( save_int ) = my_sampler.accepted;
      weights_saved.row( save_int ) = my_sampler.w.t();
      mu_saved.slice( save_int ) = my_sampler.mu;
      // tau_saved.slice( save_int ) = my_sampler.tau;
      // cov_saved( save_int ) = my_sampler.cov;
      m_saved.slice( save_int ) = my_sampler.m;
      S_saved.slice( save_int ) = my_sampler.S;
      mean_sum_saved.slice( save_int ) = my_sampler.mean_sum;
      
      alloc_prob.slice( save_int ) = my_sampler.alloc_prob;
      cov_saved.slice ( save_int ) = arma::reshape(arma::mat(my_sampler.cov.memptr(), my_sampler.cov.n_elem, 1, false), P, P * K);
      cov_comb_saved.slice( save_int) = arma::reshape(arma::mat(my_sampler.cov_comb.memptr(), my_sampler.cov_comb.n_elem, 1, false), P, P * K * B); 
      
      my_sampler.updateBatchCorrectedData();
      batch_corrected_data.slice( save_int ) =  my_sampler.Y;
      
      if(printCovariance) {  
        std::cout << "\n\nCovariance cube:\n" << my_sampler.cov;
        std::cout << "\n\nBatch covariance matrix:\n" << my_sampler.S;
      }
      
      save_int++;
    }
  }
  
  if(verbose) {
    std::cout << "\n\nCovariance acceptance rate:\n" << arma::conv_to< arma::vec >::from(my_sampler.cov_count) / R;
    std::cout << "\n\ncluster mean acceptance rate:\n" << arma::conv_to< arma::vec >::from(my_sampler.mu_count) / R;
    std::cout << "\n\nBatch covariance acceptance rate:\n" << arma::conv_to< arma::vec >::from(my_sampler.S_count) / R;
    std::cout << "\n\nBatch mean acceptance rate:\n" << arma::conv_to< arma::vec >::from(my_sampler.m_count) / R;
  }
  
  // std::cout << "\nReciprocal condition number\n" << my_sampler.rcond_count;
  
  return(List::create(Named("samples") = class_record, 
                      Named("means") = mu_saved,
                      Named("covariance") = cov_saved,
                      Named("batch_shift") = m_saved,
                      Named("batch_scale") = S_saved,
                      Named("mean_sum") = mean_sum_saved,
                      Named("cov_comb") = cov_comb_saved,
                      Named("weights") = weights_saved,
                      Named("cov_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.cov_count) / R,
                      Named("mu_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.mu_count) / R,
                      Named("S_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.S_count) / R,
                      Named("m_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.m_count) / R,
                      Named("alloc_prob") = alloc_prob,
                      Named("likelihood") = model_likelihood,
                      Named("BIC") = BIC_record,
                      Named("batch_corrected_data") = batch_corrected_data
  )
  );
  
};







// [[Rcpp::export]]
Rcpp::List sampleSemisupervisedMSN (
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
    double phi_proposal_window,
    double rho,
    double theta,
    double lambda,
    arma::uword R,
    arma::uword thin,
    arma::vec concentration,
    bool verbose = true,
    bool doCombinations = false,
    bool printCovariance = false
) {
  
  // The random seed is set at the R level via set.seed() apparently.
  // std::default_random_engine generator(seed);
  // arma::arma_rng::set_seed(seed);
  
  msnPredictive my_sampler(K,
                           B,
                           mu_proposal_window,
                           cov_proposal_window,
                           m_proposal_window,
                           S_proposal_window,
                           phi_proposal_window,
                           rho,
                           theta,
                           lambda,
                           labels,
                           batch_vec,
                           concentration,
                           X,
                           fixed
  );
  
  arma::uword P = X.n_cols;
  
  // The output matrix
  arma::umat class_record(floor(R / thin), X.n_rows);
  class_record.zeros();
  
  // We save the BIC at each iteration
  arma::vec BIC_record = arma::zeros<arma::vec>(floor(R / thin));
  arma::vec model_likelihood = arma::zeros<arma::vec>(floor(R / thin));
  arma::uvec acceptance_vec = arma::zeros<arma::uvec>(floor(R / thin));
  arma::mat weights_saved = arma::zeros<arma::mat>(floor(R / thin), K);
  
  arma::cube mean_sum_saved(P, K * B, floor(R / thin)), mu_saved(P, K, floor(R / thin)), m_saved(P, B, floor(R / thin)), cov_saved(P, K * P, floor(R / thin)), S_saved(P, B, floor(R / thin)), cov_comb_saved(P, P * K * B, floor(R / thin)), alloc_prob(my_sampler.N, K, floor(R / thin)), phi_saved(my_sampler.P, K, floor(R / thin));
  // arma::field<arma::cube> cov_saved(my_sampler.P, my_sampler.P, K, floor(R / thin));
  mu_saved.zeros();
  cov_saved.zeros();
  cov_comb_saved.zeros();
  m_saved.zeros();
  S_saved.zeros();
  
  arma::uword save_int = 0;
  
  // Sampler from priors
  my_sampler.sampleFromPriors();
  my_sampler.matrixCombinations();
  
  // Iterate over MCMC moves
  for(arma::uword r = 0; r < R; r++){
    
    my_sampler.updateWeights();
    
    // Metropolis step for batch parameters
    my_sampler.metropolisStep(); 
    
    my_sampler.updateAllocation();
    
    // Record results
    if((r + 1) % thin == 0){
      
      my_sampler.calcBIC();
      BIC_record( save_int ) = my_sampler.BIC;
      model_likelihood( save_int ) = my_sampler.model_likelihood;
      class_record.row( save_int ) = my_sampler.labels.t();
      acceptance_vec( save_int ) = my_sampler.accepted;
      weights_saved.row( save_int ) = my_sampler.w.t();
      mu_saved.slice( save_int ) = my_sampler.mu;
      // tau_saved.slice( save_int ) = my_sampler.tau;
      // cov_saved( save_int ) = my_sampler.cov;
      m_saved.slice( save_int ) = my_sampler.m;
      S_saved.slice( save_int ) = my_sampler.S;
      mean_sum_saved.slice( save_int ) = my_sampler.mean_sum;
      phi_saved.slice( save_int ) = my_sampler.phi;
      
      alloc_prob.slice( save_int ) = my_sampler.alloc_prob;
      cov_saved.slice ( save_int ) = arma::reshape(arma::mat(my_sampler.cov.memptr(), my_sampler.cov.n_elem, 1, false), P, P * K);
      cov_comb_saved.slice( save_int) = arma::reshape(arma::mat(my_sampler.cov_comb.memptr(), my_sampler.cov_comb.n_elem, 1, false), P, P * K * B); 
      
      if(printCovariance) {  
        std::cout << "\n\nCovariance cube:\n" << my_sampler.cov;
        std::cout << "\n\nBatch covariance matrix:\n" << my_sampler.S;
      }
      
      save_int++;
    }
  }
  
  if(verbose) {
    
    std::cout << "\n\nCovariance acceptance rate:\n" << arma::conv_to< arma::vec >::from(my_sampler.cov_count) / R;
    std::cout << "\n\nCluster mean acceptance rate:\n" << arma::conv_to< arma::vec >::from(my_sampler.mu_count) / R;
    std::cout << "\n\nCluster shape acceptance rate:\n" << arma::conv_to< arma::vec >::from(my_sampler.phi_count) / R;
    std::cout << "\n\nBatch covariance acceptance rate:\n" << arma::conv_to< arma::vec >::from(my_sampler.S_count) / R;
    std::cout << "\n\nBatch mean acceptance rate:\n" << arma::conv_to< arma::vec >::from(my_sampler.m_count) / R;
    
  }
  
  
  return(List::create(Named("samples") = class_record, 
                      Named("means") = mu_saved,
                      Named("covariance") = cov_saved,
                      Named("shapes") = phi_saved,
                      Named("batch_shift") = m_saved,
                      Named("batch_scale") = S_saved,
                      Named("mean_sum") = mean_sum_saved,
                      Named("cov_comb") = cov_comb_saved,
                      Named("weights") = weights_saved,
                      Named("cov_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.cov_count) / R,
                      Named("mu_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.mu_count) / R,
                      Named("S_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.S_count) / R,
                      Named("m_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.m_count) / R,
                      Named("phi_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.phi_count) / R,
                      Named("alloc_prob") = alloc_prob,
                      Named("likelihood") = model_likelihood,
                      Named("BIC") = BIC_record
  )
  );
  
};

// [[Rcpp::export]]
Rcpp::List sampleSemisupervisedMVT (
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
    double t_df_proposal_window,
    double rho,
    double theta,
    double lambda,
    arma::uword R,
    arma::uword thin,
    arma::vec concentration,
    bool verbose = true,
    bool doCombinations = false,
    bool printCovariance = false
) {
  
  // The random seed is set at the R level via set.seed() apparently.
  // std::default_random_engine generator(seed);
  // arma::arma_rng::set_seed(seed);
  
  
  mvtPredictive my_sampler(K,
    B,
    mu_proposal_window,
    cov_proposal_window,
    m_proposal_window,
    S_proposal_window,
    t_df_proposal_window,
    rho,
    theta,
    lambda,
    labels,
    batch_vec,
    concentration,
    X,
    fixed
  );
  
  arma::uword P = X.n_cols, N = X.n_rows;
  
  // The output matrix
  arma::umat class_record(floor(R / thin), X.n_rows);
  class_record.zeros();
  
  // We save the BIC at each iteration
  arma::vec BIC_record = arma::zeros<arma::vec>(floor(R / thin));
  arma::vec model_likelihood = arma::zeros<arma::vec>(floor(R / thin));
  arma::uvec acceptance_vec = arma::zeros<arma::uvec>(floor(R / thin));
  arma::mat weights_saved(floor(R / thin), K), t_df_saved(floor(R / thin), K);
  weights_saved.zeros();
  t_df_saved.zeros();
  
  arma::cube mean_sum_saved(P, K * B, floor(R / thin)), 
    mu_saved(P, K, floor(R / thin)),
    m_saved(P, B, floor(R / thin)), 
    cov_saved(P, K * P, floor(R / thin)),
    S_saved(P, B, floor(R / thin)), 
    cov_comb_saved(P, P * K * B, floor(R / thin)),
    alloc_prob(N, K, floor(R / thin)),
    batch_corrected_data(N, P, floor(R / thin));

  mu_saved.zeros();
  cov_saved.zeros();
  cov_comb_saved.zeros();
  m_saved.zeros();
  S_saved.zeros();
  alloc_prob.zeros();
  batch_corrected_data.zeros();
  
  arma::uword save_int = 0;
  
  // Sampler from priors
  my_sampler.sampleFromPriors();
  my_sampler.matrixCombinations();
  
  // Iterate over MCMC moves
  for(arma::uword r = 0; r < R; r++){
    
    my_sampler.updateWeights();
    
    // std::cout << "\nWeights.\n";
    
    // Metropolis step for batch parameters
    my_sampler.metropolisStep(); 
    
    // std::cout << "\nMetropolis.\n";
    
    my_sampler.updateAllocation();
    
    // std::cout << "\nAllocation.\n";
    
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
      model_likelihood( save_int ) = my_sampler.model_likelihood;
      class_record.row( save_int ) = my_sampler.labels.t();
      acceptance_vec( save_int ) = my_sampler.accepted;
      weights_saved.row( save_int ) = my_sampler.w.t();
      mu_saved.slice( save_int ) = my_sampler.mu;
      // tau_saved.slice( save_int ) = my_sampler.tau;
      // cov_saved( save_int ) = my_sampler.cov;
      m_saved.slice( save_int ) = my_sampler.m;
      S_saved.slice( save_int ) = my_sampler.S;
      mean_sum_saved.slice( save_int ) = my_sampler.mean_sum;
      t_df_saved.row( save_int ) = my_sampler.t_df.t();
      
      alloc_prob.slice( save_int ) = my_sampler.alloc_prob;
      cov_saved.slice ( save_int ) = arma::reshape(arma::mat(my_sampler.cov.memptr(), my_sampler.cov.n_elem, 1, false), P, P * K);
      cov_comb_saved.slice( save_int) = arma::reshape(arma::mat(my_sampler.cov_comb.memptr(), my_sampler.cov_comb.n_elem, 1, false), P, P * K * B); 
      
      my_sampler.updateBatchCorrectedData();
      batch_corrected_data.slice( save_int ) =  my_sampler.Y;
      
      if(printCovariance) {  
        std::cout << "\n\nCovariance cube:\n" << my_sampler.cov;
        std::cout << "\n\nBatch covariance matrix:\n" << my_sampler.S;
      }
      
      save_int++;
    }
  }
  
  if(verbose) {
    
    std::cout << "\n\nCovariance acceptance rate:\n" << arma::conv_to< arma::vec >::from(my_sampler.cov_count) / R;
    std::cout << "\n\ncluster mean acceptance rate:\n" << arma::conv_to< arma::vec >::from(my_sampler.mu_count) / R;
    std::cout << "\n\nBatch covariance acceptance rate:\n" << arma::conv_to< arma::vec >::from(my_sampler.S_count) / R;
    std::cout << "\n\nBatch mean acceptance rate:\n" << arma::conv_to< arma::vec >::from(my_sampler.m_count) / R;
    std::cout << "\n\nCluster t d.f. acceptance rate:\n" << arma::conv_to< arma::vec >::from(my_sampler.t_df_count) / R;
    
  }
  
  return(
    List::create(Named("samples") = class_record, 
      Named("means") = mu_saved,
      Named("covariance") = cov_saved,
      Named("batch_shift") = m_saved,
      Named("batch_scale") = S_saved,
      Named("mean_sum") = mean_sum_saved,
      Named("cov_comb") = cov_comb_saved,
      Named("t_df") = t_df_saved,
      Named("weights") = weights_saved,
      Named("cov_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.cov_count) / R,
      Named("mu_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.mu_count) / R,
      Named("S_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.S_count) / R,
      Named("m_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.m_count) / R,
      Named("t_df_acceptance_rate") = arma::conv_to< arma::vec >::from(my_sampler.t_df_count) / R,
      Named("alloc_prob") = alloc_prob,
      Named("likelihood") = model_likelihood,
      Named("BIC") = BIC_record,
      Named("batch_corrected_data") = batch_corrected_data
    )
  );
  
};






// //' @title Mixture model
// //' @description Performs MCMC sampling for a mixture model.
// //' @param X The data matrix to perform clustering upon (items to cluster in rows).
// //' @param K The number of components to model (upper limit on the number of clusters found).
// //' @param labels Vector item labels to initialise from.
// //' @param dataType Int, 0: independent Gaussians, 1: Multivariate normal, or 2: Categorical distributions.
// //' @param R The number of iterations to run for.
// //' @param thin thinning factor for samples recorded.
// //' @param concentration Vector of concentrations for mixture weights (recommended to be symmetric).
// //' @return Named list of the matrix of MCMC samples generated (each row 
// //' corresponds to a different sample) and BIC for each saved iteration.
// // [[Rcpp::export]]
// Rcpp::List sampleMixtureModel (
//   arma::mat X,
//   arma::uword K,
//   arma::uword B,
//   int dataType,
//   arma::uvec labels,
//   arma::uvec batch_vec,
//   double mu_proposal_window,
//   double cov_proposal_window,
//   double m_proposal_window,
//   double S_proposal_window,
//   double t_df_proposal_window,
//   double phi_proposal_window,
//   double rho,
//   double theta,
//   double lambda,
//   arma::uword R,
//   arma::uword thin,
//   arma::vec concentration,
//   bool verbose = true,
//   bool doCombinations = false,
//   bool printCovariance = false
// ) {
//   
//   // Set the random number
//   std::default_random_engine generator(seed);
//   
//   // Declare the factory
//   samplerFactory my_factory;
//   
//   // Convert from an int to the samplerType variable for our Factory
//   samplerFactory::samplerType val = static_cast<samplerFactory::samplerType>(dataType);
//   
//   // Make a pointer to the correct type of sampler
//   std::unique_ptr<sampler> sampler_ptr = my_factory.createSampler(val,
//     K,
//     B,
//     mu_proposal_window,
//     cov_proposal_window,
//     m_proposal_window,
//     S_proposal_window,
//     t_df_proposal_window,
//     phi_proposal_window,
//     rho,
//     theta,
//     lambda,
//     labels,
//     batch_vec,
//     concentration,
//     X
//   );
//   
//   // The output matrix
//   arma::umat class_record(floor(R / thin), X.n_rows);
//   class_record.zeros();
//   
//   // We save the BIC at each iteration
//   arma::vec BIC_record = arma::zeros<arma::vec>(floor(R / thin));
//   
//   arma::uword save_int=0;
//   
//   // Sampler from priors (this is unnecessary)
//   sampler_ptr->sampleFromPriors();
//   
//   // Iterate over MCMC moves
//   for(arma::uword r = 0; r < R; r++){
//     
//     sampler_ptr->updateWeights();
//     sampler_ptr->sampleParameters();
//     sampler_ptr->updateAllocation();
//     
//     // Record results
//     if((r + 1) % thin == 0){
//       
//       // Update the BIC for the current model fit
//       sampler_ptr->calcBIC();
//       BIC_record( save_int ) = sampler_ptr->BIC; 
//       
//       // Save the current clustering
//       class_record.row( save_int ) = sampler_ptr->labels.t();
//       save_int++;
//     }
//   }
//   return(List::create(Named("samples") = class_record, Named("BIC") = BIC_record));
// };
// 
// 
// //' @title Mixture model
// //' @description Performs MCMC sampling for a mixture model.
// //' @param X The data matrix to perform clustering upon (items to cluster in rows).
// //' @param K The number of components to model (upper limit on the number of clusters found).
// //' @param labels Vector item labels to initialise from.
// //' @param fixed Binary vector of the items that are fixed in their initial label.
// //' @param dataType Int, 0: independent Gaussians, 1: Multivariate normal, or 2: Categorical distributions.
// //' @param R The number of iterations to run for.
// //' @param thin thinning factor for samples recorded.
// //' @param concentration Vector of concentrations for mixture weights (recommended to be symmetric).
// //' @return Named list of the matrix of MCMC samples generated (each row 
// //' corresponds to a different sample) and BIC for each saved iteration.
// // [[Rcpp::export]]
// Rcpp::List sampleSemisupervisedMixtureModel (
//     arma::mat X,
//     arma::uword K,
//     arma::uvec labels,
//     arma::uvec fixed,
//     int dataType,
//     arma::uword R,
//     arma::uword thin,
//     arma::vec concentration,
//     arma::uword seed
// ) {
//   
//   // Set the random number
//   std::default_random_engine generator(seed);
//   
//   // Declare the factory
//   semisupervisedSamplerFactory my_factory;
//   
//   // Convert from an int to the samplerType variable for our Factory
//   semisupervisedSamplerFactory::samplerType val = static_cast<semisupervisedSamplerFactory::samplerType>(dataType);
//   
//   // Make a pointer to the correct type of sampler
//   std::unique_ptr<sampler> sampler_ptr = my_factory.createSemisupervisedSampler(val,
//                                                                                 K,
//                                                                                 labels,
//                                                                                 concentration,
//                                                                                 X,
//                                                                                 fixed);
//   
//   // The output matrix
//   arma::umat class_record(floor(R / thin), X.n_rows);
//   class_record.zeros();
//   
//   // We save the BIC at each iteration
//   arma::vec BIC_record = arma::zeros<arma::vec>(floor(R / thin));
//   
//   arma::uword save_int=0;
//   
//   // Sampler from priors (this is unnecessary)
//   sampler_ptr->sampleFromPriors();
//   
//   // Iterate over MCMC moves
//   for(arma::uword r = 0; r < R; r++){
//     
//     sampler_ptr->updateWeights();
//     sampler_ptr->sampleParameters();
//     sampler_ptr->updateAllocation();
//     
//     // Record results
//     if((r + 1) % thin == 0){
//       
//       // Update the BIC for the current model fit
//       sampler_ptr->calcBIC();
//       BIC_record( save_int ) = sampler_ptr->BIC; 
//       
//       // Save the current clustering
//       class_record.row( save_int ) = sampler_ptr->labels.t();
//       save_int++;
//     }
//   }
//   return(List::create(Named("samples") = class_record, Named("BIC") = BIC_record));
// };
