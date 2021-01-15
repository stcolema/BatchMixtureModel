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

double logNormalLogProbability(double x, double mu, double tau) {
  return(std::log(tau) - std::log(x) - tau * std::pow((std::log(x) - mu), 2.0));
}

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
    
    // std::cout << "N_k: \n" << N_k << "\n\n";
    
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
  arma::mat mu, mu_proposed, batch_shift, batch_shift_proposed, tau, tau_proposed, batch_scale, batch_scale_proposed;
  
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
    
    xi = 0.0; // arma::accu(X)/(N * P);
    kappa = 0.01;
    alpha = 2.0; // 0.5 * (2 + P);
    beta = 2.0; // arma::stddev(arma::vectorise(X)) / std::pow(K, 2);
    
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
    
    batch_shift.set_size(P, B);
    batch_shift.zeros();
    
    batch_shift_proposed.set_size(P, B);
    batch_shift_proposed.zeros();
    
    batch_scale.set_size(P, B);
    batch_scale.zeros();
    
    batch_scale_proposed.set_size(P, B);
    batch_scale_proposed.zeros();
    
    
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
        batch_scale(p, b) =  1.0 / arma::randg<double>( arma::distr_param(rho, 1.0 / theta ) );
        batch_shift(p, b) = (arma::randn<double>() * batch_scale(p, b) / lambda ) + delta;
      }
      for(arma::uword k = 0; k < K; k++){
        // tau(p, k) = 1.0 / arma::randg<double>( arma::distr_param(alpha, 1.0 / arma::as_scalar(beta(p))) );
        tau(p, k) = 1.0 / arma::randg<double>( arma::distr_param(alpha, 1.0 / beta) );
        mu(p, k) = (arma::randn<double>() * tau(p, k) / kappa ) + xi;
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
        // batch_scale_proposed(p, b) = std::exp((arma::randn() / proposal_window_for_logs) + batch_scale(p, b));
        batch_scale_proposed(p, b) = 1.0 / arma::randg(arma::distr_param(batch_scale(p, b) * proposal_window, 1.0 / proposal_window));
        batch_shift_proposed(p, b) = (arma::randn() / proposal_window) + batch_shift(p, b);
      }
      
      // std::cout << "\nProposing to components.\n";
      
      for(arma::uword k = 0; k < K; k++){
        // tau_proposed(p, k) = std::exp((arma::randn() / proposal_window_for_logs) + tau(p, k));
        tau_proposed(p, k) = 1.0 / arma::randg(arma::distr_param(tau(p, k) * proposal_window, 1.0 / proposal_window));
        mu_proposed(p, k) = (arma::randn() / proposal_window) + mu(p, k);
      }
      
    }
    // std::cout << "\nProposed.\n";
  }
  
  double modelLogLikelihood(arma::mat mu, 
                         arma::mat tau,
                         arma::mat batch_shift,
                         arma::mat batch_scale) {
    
    double model_log_likelihood = 0;
    arma::uword c_n, b_n;
    arma::rowvec x_n;

    for(arma::uword n = 0; n < N; n++){
      c_n = labels(n);
      b_n = batch_vec(n);
      x_n = X.row(n);
      for(arma::uword p = 0; p < P; p++){

        model_log_likelihood += -0.5 * (std::log(2) + std::log(PI)
                                    + std::log(arma::as_scalar(tau(p, c_n)))
                                    + std::log(arma::as_scalar(batch_scale(p, b_n)))
                                    + arma::as_scalar(tau(p, c_n) 
                                      * batch_scale(p, b_n)
                                      * pow((x_n(p) - (mu(p, c_n) + batch_shift(p, b_n))), 2.0)
                                    )
                                  );
                                    
      }
      
    }
    
    return model_log_likelihood;
    
  };
  
  double priorLogProbability(arma::mat mu, 
                          arma::mat tau,
                          arma::mat batch_shift,
                          arma::mat batch_scale){
    
    double prior_score = 0.0;
    
    for(arma::uword p = 0; p < P; p++){
      
      for(arma::uword b = 0; b < B; b++){
        prior_score += invGammaLogLikelihood(batch_scale(p, b), rho, 1.0 / theta);
        prior_score += arma::log_normpdf(batch_shift(p, b), delta, lambda * batch_scale(p, b));
      }
      for(arma::uword k = 0; k < K; k++){
        // tau(p, k) = 1.0 / arma::randg<double>( arma::distr_param(alpha, 1.0 / arma::as_scalar(beta(p))) );
        prior_score += invGammaLogLikelihood(tau(p, k), alpha, 1.0 / beta);
        prior_score += arma::log_normpdf(mu(p, k), xi, kappa * tau(p, k));
      }
      
    }
    return prior_score;
  };
  
  // // Sample mu
  // void updateMuTau() {
  //   
  //   arma::uword n_k = 0;
  //   double _sd = 0, _mean = 0;
  //   
  //   double a, b;
  //   arma::vec mu_k(P);
  //   
  //   for (arma::uword k = 0; k < K; k++) {
  //     
  //     // Find how many labels have the value
  //     n_k = N_k(k);
  //     if(n_k > 0){
  //       
  //       arma::mat component_data = X.rows( arma::find(labels == k) );
  //       
  //       for (arma::uword p = 0; p < P; p++){
  //         
  //         // The updated parameters for mu
  //         _sd = 1.0/(tau(p, k) * n_k + kappa);
  //         _mean = (tau(p, k) * arma::sum(component_data.col(p)) + kappa * xi) / (1.0/_sd) ;
  //         
  //         
  //         // Sample a new value
  //         mu(p, k) = arma::randn<double>() * _sd + _mean;
  //         
  //         // Parameters of the distribution for tau
  //         a = alpha + 0.5 * n_k;
  //         
  //         arma::vec b_star = component_data.col(p) - mu(p, k);
  //         b = beta(p) + 0.5 * arma::accu(b_star % b_star);
  //         
  //         // The updated parameters
  //         tau(p, k) = 1.0 / arma::randg<double>(arma::distr_param(a, 1.0 / b) );
  //       }
  //     } else {
  //       for (arma::uword p = 0; p < P; p++){
  //         // Sample a new value from the priors
  //         mu(p, k) = arma::randn<double>() * (1.0/kappa) + xi;
  //         tau(p, k) = 1.0 / arma::randg<double>(arma::distr_param(alpha, 1.0 / beta(p)) );
  //       }
  //     }
  //   }
  // };
  // 
  // void sampleParameters() {
  //   updateMuTau();
  //   updateBeta();
  // }
  
  double proposalScore(arma::mat x, arma::mat y, double window, arma::uword dim){
    
    double score = 0.0;
    
    for(arma::uword p = 0; p < P; p++) {
      
      for(arma::uword j = 0; j < dim; j++){
        // score += logNormalLogProbability(x(p, j), y(p, j), window);
        score += invGammaLogLikelihood(x(p, j), y(p, j) * window, window);
      }
    }
    return score;
  }
  
  void metropolisStep(){
      
    double u = 0.0, proposed_model_score = 0.0, acceptance_prob = 0.0, current_model_score = 0.0;
  
    u = arma::randu();
    proposed_model_score = modelLogLikelihood(mu_proposed, 
        tau_proposed, 
        batch_shift_proposed, 
        batch_scale_proposed) +
      priorLogProbability(mu_proposed, 
        tau_proposed, 
        batch_shift_proposed, 
        batch_scale_proposed);
    
    proposed_model_score += proposalScore(tau, tau_proposed, proposal_window_for_logs, K);
    proposed_model_score += proposalScore(batch_scale, batch_scale_proposed, proposal_window_for_logs, B);

    // Set upt the value for the denominator
    current_model_score += model_score;
    current_model_score += proposalScore(tau_proposed, tau, proposal_window_for_logs, K);
    current_model_score += proposalScore(batch_scale_proposed, batch_scale, proposal_window_for_logs, B);
    
    acceptance_prob = std::min(1.0, std::exp(proposed_model_score - current_model_score));
    
    if(u < acceptance_prob){
      mu = mu_proposed;
      tau = tau_proposed;
      batch_shift = batch_shift_proposed;
      batch_scale = batch_scale_proposed;
      model_score = proposed_model_score;
      accepted++;
    }
  };
  
  arma::vec itemLogLikelihood(arma::vec item, arma::uword batch) {
    
    // arma::vec my_ll(K);
    // ll.zeros();
    // 
    for(arma::uword k = 0; k < K; k++){
      for (arma::uword p = 0; p < P; p++){
        ll(k) += -0.5*(std::log(2) + std::log(PI) - std::log(arma::as_scalar(tau(p, k))) - std::log(arma::as_scalar(batch_scale(p, batch)))+ arma::as_scalar(tau(p, k) * batch_scale(p, batch) *  std::pow(item(p) - (mu(p, k) + batch_shift(p, batch) ), 2))); 
      }
    }
    return ll;
  };
  
  void calcBIC(){
    
    arma::uword n_param = (P + P) * K_occ;
    BIC = n_param * std::log(N) - 2 * model_likelihood;
    
  };
  
  double batchShiftKernel(arma::uword b, arma::vec a_b) {
    
    double score = 0.0;
    arma::uvec batch_ind = arma::find(batch_vec == b);
    for(arma::uword p = 0; p < P; p++) {
      for (auto& n : batch_ind) {
        score += (1.0 / tau(p, labels(n))) * std::pow((X(n, p) - (mu(p, labels(n)) + a_b(p))), 2.0);
      }
      score += lambda * (a_b(p) - delta);
      score = -0.5 * (1.0 / batch_scale(p, b)) *  score;
    }
  return score;
  };
  
  double batchScaleKernel(arma::uword b, arma::vec s_b) {
    
    double score = 0.0;
    arma::uvec batch_ind = find(batch_vec == b);
    for(arma::uword p = 0; p < P; p++) {
      for (auto& n : batch_ind) {
        score += (1.0 / tau(p, labels(n))) * std::pow((X(n, p) - (mu(p, labels(n)) + batch_shift(p, b))), 2.0);
      }
      score += lambda * (batch_shift(p, b) - delta) + 2 * theta;
      score = (1.0 / s_b(p)) * (-0.5 * score + (N_b(b) + rho - 0.5));
    }
    return score;
  };
  
  double clusterMeanKernel(arma::uword k, arma::vec mu_k) {
    
    double score = 0.0;
    arma::uvec cluster_ind = arma::find(labels == k);
    for(arma::uword p = 0; p < P; p++) {
      for (auto& n : cluster_ind) {
        score +=  (1.0 / batch_scale(p, batch_vec(n)))* std::pow((X(n, p) - (mu_k(p) + batch_shift(p, batch_vec(n)))), 2.0);
      }
      score += kappa * (mu_k(p) - xi);
      score = -0.5 *( 1.0 / tau(p, k)) *  score;
    }
    return score;
  };
  
  double clusterPrecisionKernel(arma::uword k, arma::vec tau_k) {
    
    double score = 0.0;
    arma::uvec cluster_ind = arma::find(labels == k);
    for(arma::uword p = 0; p < P; p++) {
      for (auto& n : cluster_ind) {
        score += (1.0 /  batch_scale(p, batch_vec(n))) * std::pow((X(n, p) - (mu(p, k) + batch_shift(p, batch_vec(n)))), 2.0);
      }
      score += kappa * (mu(p, k) - xi) + 2 * beta;
      score = (1.0 / tau_k(p)) * (-0.5 * score + N_k(k) + alpha - 0.5);
    }
    return score;
  };
  
  void batchScaleMetorpolis() {
    
    double u = 0.0, proposed_model_score = 0.0, acceptance_prob = 0.0, current_model_score = 0.0;
    
    current_model_score += model_score;
    
    for(arma::uword b = 0; b < B ; b++) {
      for(arma::uword p = 0; p < P; p++){
        // batch_scale_proposed(p, b) = std::exp((arma::randn() / proposal_window_for_logs) + batch_scale(p, b));
        batch_scale_proposed(p, b) = 1.0 / arma::randg(arma::distr_param(proposal_window_for_logs, 1.0 /(batch_scale(p, b) * (proposal_window_for_logs - 1))));
        
        // Prior log probability
        proposed_model_score += invGammaLogLikelihood(batch_scale_proposed(p, b), rho, 1.0 / theta);
        current_model_score += invGammaLogLikelihood(batch_scale(p, b), rho, 1.0 / theta);
        
        // Log probability under the proposal density
        // proposed_model_score += logNormalLogProbability(batch_scale(p, b), batch_scale_proposed(p, b), proposal_window_for_logs);
        // current_model_score += logNormalLogProbability(batch_scale_proposed(p, b), batch_scale(p, b), proposal_window_for_logs);
        
        proposed_model_score += invGammaLogLikelihood(batch_scale(p, b), proposal_window_for_logs, 1.0 /(batch_scale_proposed(p, b) * (proposal_window_for_logs - 1)));
        current_model_score += invGammaLogLikelihood(batch_scale_proposed(p, b), proposal_window_for_logs, 1.0 /(batch_scale(p, b) * (proposal_window_for_logs - 1)));

      }
      proposed_model_score += batchScaleKernel(b, batch_scale_proposed.col(b));
      current_model_score += batchScaleKernel(b, batch_scale.col(b));
    }

    u = arma::randu();
    
    acceptance_prob = std::min(1.0, std::exp(proposed_model_score - current_model_score));
    
    if(u < acceptance_prob){
      batch_scale = batch_scale_proposed;
      // batch_scale_score = proposed_model_score;
    }
    
  };
  
  arma::vec batchScaleScore(arma::mat batch_scale) {
    
    arma::vec score(B);
    score.zeros();
    
    for(arma::uword b = 0; b < B ; b++) {
      for(arma::uword p = 0; p < P; p++){
        score(b) += invGammaLogLikelihood(batch_scale(p, b), rho, 1.0 / theta);
      }
      score(b) += batchScaleKernel(b, batch_scale.col(b));
    }
    return score;
  }
  
  void batchShiftMetorpolis() {
    
    double u = 0.0, proposed_model_score = 0.0, acceptance_prob = 0.0, current_model_score = 0.0;
    
    current_model_score += model_score;
    
    for(arma::uword b = 0; b < B ; b++) {
      for(arma::uword p = 0; p < P; p++){
        batch_shift_proposed(p, b) = (arma::randn() / proposal_window) + batch_shift(p, b);
        proposed_model_score += arma::log_normpdf(batch_shift_proposed(p, b), delta, lambda * batch_scale(p, b) );
        current_model_score += arma::log_normpdf(batch_shift(p, b), delta, lambda * batch_scale(p, b) ); 
        
      }
      proposed_model_score += batchShiftKernel(b, batch_shift_proposed.col(b));
      current_model_score += batchShiftKernel(b, batch_shift.col(b));
    }
    
    u = arma::randu();
    
    acceptance_prob = std::min(1.0, std::exp(proposed_model_score - current_model_score));
    
    if(u < acceptance_prob){
      batch_shift = batch_shift_proposed;
      // batch_shift_score = proposed_model_score;
    }
    
  };
  
  arma::vec batchShiftScore(arma::mat batch_shift) {
    
    arma::vec score(B);
    score.zeros();
    
    for(arma::uword b = 0; b < B ; b++) {
      for(arma::uword p = 0; p < P; p++){
        score(b) += arma::log_normpdf(batch_shift(p, b), delta, lambda * batch_scale(p, b) );
      }
      score(b) += batchShiftKernel(b, batch_shift.col(b));
    }
    return score;
  };
  
  void clusterPrecisionMetropolis() {
    
    double u = 0.0, proposed_model_score = 0.0, acceptance_prob = 0.0, current_model_score = 0.0;
    
    current_model_score += model_score;
    
    for(arma::uword k = 0; k < K ; k++) {

      for(arma::uword p = 0; p < P; p++){
        if(N_k(k) == 0){
          tau(p, k) = 1.0 / arma::randg<double>( arma::distr_param(alpha, 1.0 / beta) );
        } else {
          // tau_proposed(p, k) = std::exp((arma::randn() / proposal_window_for_logs) + tau(p, k));
          tau_proposed(p, k) = 1.0 / arma::randg( arma::distr_param(proposal_window_for_logs, 1.0 / (tau(p, k) * (proposal_window_for_logs - 1))));
          
          // Log probability under the proposal density
          // proposed_model_score += logNormalLogProbability(tau(p, k), tau_proposed(p, k), proposal_window_for_logs);
          // current_model_score += logNormalLogProbability(tau_proposed(p, k), tau(p, k), proposal_window_for_logs);
          
          proposed_model_score += invGammaLogLikelihood(tau(p, k), proposal_window_for_logs, 1.0 / (tau_proposed(p, k) * (proposal_window_for_logs - 1)));
          current_model_score += invGammaLogLikelihood(tau_proposed(p, k), proposal_window_for_logs, 1.0 / (tau(p, k) * (proposal_window_for_logs - 1)));
        }
        
        // Prior log probability
        proposed_model_score += invGammaLogLikelihood(tau_proposed(p, k), alpha, 1.0 / beta);
        current_model_score += invGammaLogLikelihood(tau(p, k), alpha, 1.0 / beta);
        
        // proposed_model_score += invGammaLogLikelihood(batch_scale(p, b), batch_scale_proposed(p, b) * window, window);
        // current_model_score += invGammaLogLikelihood(batch_scale_proposed(p, b), batch_scale(p, b) * window, window);
        
      }
      proposed_model_score += clusterPrecisionKernel(k, tau_proposed.col(k));
      current_model_score += clusterPrecisionKernel(k, tau.col(k));
    }
    
    u = arma::randu();
    
    acceptance_prob = std::min(1.0, std::exp(proposed_model_score - current_model_score));
    
    if(u < acceptance_prob){
      tau = tau_proposed;
      // batch_scale_score = proposed_model_score;
    }
    
  };
  
  arma::vec clusterPrecisionScore(arma::mat tau) {
    
    arma::vec score(K);
    score.zeros();
    
    for(arma::uword k = 0; k < K ; k++) {
      for(arma::uword p = 0; p < P; p++){
        score(k) += invGammaLogLikelihood(tau(p, k), alpha, 1.0 / beta);
      }
      score(k) += clusterPrecisionKernel(k, tau.col(k));
    }
    return score;
  };
  
  void clusterMeanMetropolis() {
    
    double u = 0.0, proposed_model_score = 0.0, acceptance_prob = 0.0, current_model_score = 0.0;
    
    current_model_score += model_score;
    
    for(arma::uword k = 0; k < K ; k++) {
      for(arma::uword p = 0; p < P; p++){
        if(N_k(k) == 0){
            mu_proposed(p, k) = (arma::randn<double>() * tau(p, k)  / kappa) + xi;
        } else {
          mu_proposed(p, k) = (arma::randn() / proposal_window) + mu(p, k);
        }
        
        // Prior log probability
        proposed_model_score += arma::log_normpdf(mu_proposed(p, k), delta, kappa * 1.0 / tau(p, k) );
        current_model_score += arma::log_normpdf(mu(p, k), delta, kappa * 1.0 / tau(p, k) ); 
          
      }
      proposed_model_score += clusterMeanKernel(k, mu_proposed.col(k));
      current_model_score += clusterMeanKernel(k, mu.col(k));
    }
      
    u = arma::randu();
    
    acceptance_prob = std::min(1.0, std::exp(proposed_model_score - current_model_score));
    
    if(u < acceptance_prob){
      mu = mu_proposed;
      // mu_score = proposed_model_score;
    }
    
  };
  
  arma::vec clusterMeanScore(arma::mat mu) {
    
    arma::vec score(K);
    score.zeros();
    
    for(arma::uword k = 0; k < K ; k++) {
      for(arma::uword p = 0; p < P; p++){
        score(k) += arma::log_normpdf(mu(p, k), delta, kappa / tau(p, k) ); 
      }
      score(k) += clusterMeanKernel(k, mu.col(k));
    }
    return score;
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
  
  gaussianSampler my_sampler(K,
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
  
  arma::cube mu_saved(my_sampler.P, K, floor(R / thin)), tau_saved(my_sampler.P, K, floor(R / thin));
  mu_saved.zeros();
  tau_saved.zeros();
  
  arma::uword save_int = 0;
  
  // Sampler from priors
  my_sampler.sampleFromPriors();
  // sampler_ptr->sampleFromPriors();

  // my_sampler.model_score = my_sampler.modelLogLikelihood(
  //   my_sampler.mu,
  //   my_sampler.tau,
  //   my_sampler.batch_shift,
  //   my_sampler.batch_scale
  // ) + my_sampler.priorLogProbability(
  //     my_sampler.mu,
  //     my_sampler.tau,
  //     my_sampler.batch_shift,
  //     my_sampler.batch_scale
  // );
  
    // sample_prt.model_score->sampler_ptr.modelLo
  
  // Iterate over MCMC moves
  for(arma::uword r = 0; r < R; r++){
    
    my_sampler.updateWeights();
    
    // Metropolis step for cluster parameters
    my_sampler.clusterPrecisionMetropolis();
    my_sampler.clusterMeanMetropolis();
    
    // Metropolis step for batch parameters
    my_sampler.batchScaleMetorpolis();
    my_sampler.batchShiftMetorpolis();
    
    // my_sampler.proposeNewParameters();
    // my_sampler.metropolisStep();
    
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
      mu_saved.slice( save_int ) = my_sampler.mu;
      tau_saved.slice( save_int ) = my_sampler.tau;
      
      save_int++;
    }
  }
  return(List::create(Named("samples") = class_record, 
                      Named("means") = mu_saved,
                      Named("precisions") = tau_saved,
                      Named("acceptance") = acceptance_vec,
                      Named("BIC") = BIC_record));
  
};

