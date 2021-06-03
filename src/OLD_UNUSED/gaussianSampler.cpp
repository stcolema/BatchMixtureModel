





// //' @name gaussianSampler
// //' @title Gaussian mixture type
// //' @description The sampler for a mixture of Gaussians, where each feature is
// //' assumed to be independent (i.e. a multivariate Normal with a diagonal 
// //' covariance matrix).
// //' @field new Constructor \itemize{
// //' \item Parameter: K - the number of components to model
// //' \item Parameter: labels - the initial clustering of the data
// //' \item Parameter: concentration - the vector for the prior concentration of 
// //' the Dirichlet distribution of the component weights
// //' \item Parameter: X - the data to model
// //' }
// //' @field printType Print the sampler type called.
// //' @field updateWeights Update the weights of each component based on current 
// //' clustering.
// //' @field updateAllocation Sample a new clustering. 
// //' @field sampleFromPrior Sample from the priors for the Gaussian density.
// //' @field calcBIC Calculate the BIC of the model.
// //' @field itemLogLikelihood Calculate the likelihood of a given data point in each
// //' component. \itemize{
// //' \item Parameter: point - a data point.
// //' }
// class gaussianSampler: virtual public sampler {
//   
// public:
//   
//   double xi, kappa, alpha, beta, g, h, a, delta, lambda, rho, theta, proposal_window, proposal_window_for_logs;
//   // arma::vec beta;
//   arma::mat mu, mu_proposed, m, m_proposed, tau, tau_proposed, t, t_proposed;
//   
//   using sampler::sampler;
//   
//   // Parametrised
//   gaussianSampler(
//     arma::uword _K,
//     arma::uword _B,
//     double _proposal_window,
//     double _proposal_window_for_logs,
//     arma::uvec _labels,
//     arma::uvec _batch_vec,
//     arma::vec _concentration,
//     arma::mat _X
//   ) : sampler(_K,
//   _B,
//   _labels,
//   _batch_vec,
//   _concentration,
//   _X) {
//     
//     double data_range = X.max() - X.min(), data_range_inv = std::pow(1.0 / data_range, 2);
//     
//     xi = arma::accu(X)/(N * P);
//     kappa = 0.01;
//     alpha = 0.5 * (2 + P);
//     beta = arma::stddev(arma::vectorise(X)) / std::pow(K, 2);
//     
//     g = 0.2;
//     a = 10;
//     
//     delta = 0.0;
//     lambda = 0.01;
//     // lambda = 1.0 / data_range;
//     rho = 2.0; // 0.5 * (2 + P);
//     theta = 2.0; // stddev(vectorise(X)) / std::pow(K, 2);
//     
//     proposal_window = _proposal_window;
//     proposal_window_for_logs = _proposal_window_for_logs;
//     
//     // kappa =  data_range_inv;
//     h = a * data_range_inv;
//     
//     // beta.set_size(P);
//     // beta.zeros(); 
//     
//     mu.set_size(P, K);
//     mu.zeros();
//     
//     mu_proposed.set_size(P, K);
//     mu_proposed.zeros();
//     
//     tau.set_size(P, K);
//     tau.zeros();
//     
//     tau_proposed.set_size(P, K);
//     tau_proposed.zeros();
//     
//     m.set_size(P, B);
//     m.zeros();
//     
//     m_proposed.set_size(P, B);
//     m_proposed.zeros();
//     
//     t.set_size(P, B);
//     t.zeros();
//     
//     t_proposed.set_size(P, B);
//     t_proposed.zeros();
//     
//     
//   }
//   
//   // Destructor
//   virtual ~gaussianSampler() { };
//   
//   // Print the sampler type.
//   virtual void printType() {
//     std::cout << "\nType: Gaussian.\n";
//   }
//   
//   // Parameters for the mixture model. The priors are empirical and follow the
//   // suggestions of Richardson and Green <https://doi.org/10.1111/1467-9868.00095>.
//   void sampleFromPriors() {
//     for(uword p = 0; p < P; p++){
//       // beta(p) = randg<double>( distr_param(g, 1.0 / h) );
//       for(uword b = 0; b < B; b++){
//         t(p, b) = randg<double>( distr_param(rho, 1.0 / theta ) );
//         m(p, b) = randn<double>() / (t(p, b) * lambda ) + delta;
//       }
//       for(uword k = 0; k < K; k++){
//         // tau(p, k) = 1.0 / randg<double>( distr_param(alpha, 1.0 / as_scalar(beta(p))) );
//         tau(p, k) = randg<double>( distr_param(alpha, 1.0 / beta) );
//         mu(p, k) = randn<double>() / ( tau(p, k) * kappa ) + xi;
//       }
//       
//     }
//   }
//   
//   
//   // Sample beta
//   // void updateBeta(){
//   //   
//   //   double a = g + K * alpha;
//   //   double b = 0.0;
//   //   
//   //   for(uword p = 0; p < P; p++){
//   //     b = h + accu(tau.row(p));
//   //     beta(p) = randg<double>( distr_param(a, 1.0 / b) );
//   //   }
//   // }
//   // 
//   // Metropolis prosposal
//   void proposeNewParameters() {
//     
//     // std::cout << "\nProposing.\n";
//     
//     for(uword p = 0; p < P; p++){
//       
//       for(uword b = 0; b < B; b++){
//         // t_proposed(p, b) = std::exp((randn() / proposal_window_for_logs) + t(p, b));
//         t_proposed(p, b) = randg(distr_param(proposal_window * t(p,b), 1.0 / proposal_window));
//         m_proposed(p, b) = (randn() * proposal_window) + m(p, b);
//       }
//       
//       // std::cout << "\nProposing to components.\n";
//       
//       for(uword k = 0; k < K; k++){
//         // tau_proposed(p, k) = std::exp((randn() / proposal_window_for_logs) + tau(p, k));
//         tau_proposed(p, k) = randg(distr_param(proposal_window * tau(p, k), 1.0 / proposal_window));
//         mu_proposed(p, k) = (randn() * proposal_window) + mu(p, k);
//       }
//       
//     }
//     // std::cout << "\nProposed.\n";
//   }
//   
//   // double modelLogLikelihood(arma::mat mu, 
//   //                        arma::mat tau,
//   //                        arma::mat m,
//   //                        arma::mat t) {
//   //   
//   //   double model_log_likelihood = 0;
//   //   uword c_n, b_n;
//   //   rowvec x_n;
//   // 
//   //   for(uword n = 0; n < N; n++){
//   //     c_n = labels(n);
//   //     b_n = batch_vec(n);
//   //     x_n = X.row(n);
//   //     for(uword p = 0; p < P; p++){
//   // 
//   //       model_log_likelihood += -0.5 * (std::log(2) + std::log(PI)
//   //                                   + std::log(as_scalar(tau(p, c_n)))
//   //                                   + std::log(as_scalar(t(p, b_n)))
//   //                                   + as_scalar(tau(p, c_n) 
//   //                                     * t(p, b_n)
//   //                                     * pow((x_n(p) - (mu(p, c_n) + m(p, b_n))), 2.0)
//   //                                   )
//   //                                 );
//   //                                   
//   //     }
//   //     
//   //   }
//   //   
//   //   return model_log_likelihood;
//   //   
//   // };
//   
//   double priorLogProbability(arma::mat mu, 
//                              arma::mat tau,
//                              arma::mat m,
//                              arma::mat t){
//     
//     double prior_score = 0.0;
//     
//     for(uword p = 0; p < P; p++){
//       
//       for(uword b = 0; b < B; b++){
//         prior_score += invGammaLogLikelihood(t(p, b), rho, 1.0 / theta);
//         prior_score += log_normpdf(m(p, b), delta, lambda * t(p, b));
//       }
//       for(uword k = 0; k < K; k++){
//         // tau(p, k) = 1.0 / randg<double>( distr_param(alpha, 1.0 / as_scalar(beta(p))) );
//         prior_score += invGammaLogLikelihood(tau(p, k), alpha, 1.0 / beta);
//         prior_score += log_normpdf(mu(p, k), xi, kappa * tau(p, k));
//       }
//       
//     }
//     return prior_score;
//   };
//   
//   double proposalScore(arma::mat x, arma::mat y, double window, arma::uword dim){
//     
//     double score = 0.0;
//     
//     for(arma::uword p = 0; p < P; p++) {
//       
//       for(arma::uword j = 0; j < dim; j++){
//         // score += logNormalLogProbability(x(p, j), y(p, j), window);
//         score += invGammaLogLikelihood(x(p, j), window, y(p, j)  * (window - 1.0));
//       }
//     }
//     return score;
//   }
//   
//   arma::vec itemLogLikelihood(arma::vec item, arma::uword b) {
//     
//     vec ll(K);
//     ll.zeros();
//     
//     for(uword k = 0; k < K; k++){
//       for (uword p = 0; p < P; p++){
//         ll(k) += -0.5*(std::log(2) + std::log(PI) - std::log(as_scalar(tau(p, k))) - std::log(as_scalar(t(p, b)))+ as_scalar((tau(p, k) * t(p, b)) *  std::pow(item(p) - (mu(p, k) + m(p, b) ), 2.0))); 
//       }
//     }
//     return ll;
//   };
//   
//   void calcBIC(){
//     
//     uword n_param = (P + P) * K_occ;
//     BIC = n_param * std::log(N) - 2 * observed_likelihood;
//     
//   };
//   
//   double mKernel(arma::uword b, arma::vec m_b) {
//     
//     double score = 0.0, score_p = 0.0;
//     
//     // uvec batch_ind = find(batch_vec == b);
//     for (auto& n : batch_ind(b)) {
//       for(uword p = 0; p < P; p++) {
//         score_p += tau(p, labels(n)) * std::pow((X(n, p) - (mu(p, labels(n)) + m_b(p))), 2.0);
//       }
//     }
//     for(uword p = 0; p < P; p++) {
//       score_p += lambda * std::pow(m_b(p) - delta, 2.0);
//       score += -0.5 * t(p, b) *  score_p;
//     }
//     return score;
//   };
//   
//   double tKernel(arma::uword b, arma::vec t_b) {
//     
//     double score = 0.0, score_p = 0.0;
//     // uvec batch_ind = find(batch_vec == b);
//     for (auto& n : batch_ind(b)) {
//       for(uword p = 0; p < P; p++) {
//         score_p += tau(p, labels(n)) * std::pow((X(n, p) - (mu(p, labels(n)) + m(p, b))), 2.0);
//       }
//     }
//     for(uword p = 0; p < P; p++) {
//       score_p += lambda * std::pow(m(p, b) - delta, 2.0) + 2 * theta;
//       score +=  0.5 * ((N_b(b) + 2 * rho - 1) * std::log(t_b(p)) - t_b(p) * score_p);
//     }
//     return score;
//   };
//   
//   double muLogKernel(arma::uword k, arma::vec mu_k) {
//     
//     double score = 0.0, score_p = 0.0;
//     arma::uvec cluster_ind = arma::find(labels == k);
//     for (auto& n : cluster_ind) {
//       for(arma::uword p = 0; p < P; p++) {
//         score_p +=  t(p, batch_vec(n))* std::pow((X(n, p) - (mu_k(p) + m(p, batch_vec(n)))), 2.0);
//       }
//     }
//     for(arma::uword p = 0; p < P; p++) {
//       score_p += kappa * std::pow(mu_k(p) - xi, 2.0);
//       score += -0.5 * tau(p, k) *  score_p;
//     }
//     return score;
//   };
//   
//   double tauKernel(arma::uword k, arma::vec tau_k) {
//     
//     double score = 0.0, score_p = 0.0;
//     arma::uvec cluster_ind = arma::find(labels == k);
//     for (auto& n : cluster_ind) {
//       for(arma::uword p = 0; p < P; p++) {
//         score_p += t(p, batch_vec(n)) * std::pow((X(n, p) - (mu(p, k) + m(p, batch_vec(n)))), 2.0);
//       }
//     }
//     for(arma::uword p = 0; p < P; p++) {
//       score_p += kappa * std::pow(mu(p, k) - xi, 2.0) + 2 * beta;
//       score +=  0.5 * ((N_k(k) + 2 * alpha - 1) * std::log(tau_k(p)) - tau_k(p) * score_p);
//     }
//     return score;
//   };
//   
//   void batchScaleMetorpolis() {
//     
//     double u = 0.0, proposed_model_score = 0.0, acceptance_prob = 0.0, current_model_score = 0.0;
//     
//     for(arma::uword b = 0; b < B ; b++) {
//       
//       proposed_model_score = 0.0, acceptance_prob = 0.0, current_model_score = 0.0;
//       
//       for(arma::uword p = 0; p < P; p++){
//         t_proposed(p, b) = std::exp((arma::randn() * proposal_window_for_logs) + t(p, b));
//         // t_proposed(p, b) = arma::randg(arma::distr_param(proposal_window_for_logs * t(p, b),  1.0 / proposal_window_for_logs));
//         
//         // Prior log probability is included in the kernel
//         // proposed_model_score += gammaLogLikelihood(t_proposed(p, b), rho, theta);
//         // current_model_score += gammaLogLikelihood(t(p, b), rho, theta);
//         
//         // Log probability under the proposal density
//         proposed_model_score += logNormalLogProbability(t(p, b), t_proposed(p, b), proposal_window_for_logs);
//         current_model_score += logNormalLogProbability(t_proposed(p, b), t(p, b), proposal_window_for_logs);
//         
//         // Assymetric proposal density
//         // proposed_model_score += gammaLogLikelihood(t(p, b), proposal_window_for_logs * t_proposed(p, b), proposal_window_for_logs);
//         // current_model_score += gammaLogLikelihood(t_proposed(p, b), proposal_window_for_logs * t(p, b), proposal_window_for_logs);
//         
//       }
//       proposed_model_score += tKernel(b, t_proposed.col(b));
//       current_model_score += tKernel(b, t.col(b));
//       
//       u = arma::randu();
//       
//       acceptance_prob = std::min(1.0, std::exp(proposed_model_score - current_model_score));
//       
//       if(u < acceptance_prob){
//         t.col(b) = t_proposed.col(b);
//         // t_score = proposed_model_score;
//       }
//       
//     }
//     
//     
//   };
//   
//   arma::vec batchScaleScore(arma::mat t) {
//     
//     arma::vec score(B);
//     score.zeros();
//     
//     for(arma::uword b = 0; b < B ; b++) {
//       // for(arma::uword p = 0; p < P; p++){
//       //   score(b) += gammaLogLikelihood(t(p, b), rho, theta);
//       // }
//       score(b) += tKernel(b, t.col(b));
//     }
//     return score;
//   };
//   
//   void batchShiftMetorpolis() {
//     
//     double u = 0.0, proposed_model_score = 0.0, acceptance_prob = 0.0, current_model_score = 0.0;
//     
//     current_model_score += model_score;
//     
//     for(arma::uword b = 0; b < B ; b++) {
//       for(arma::uword p = 0; p < P; p++){
//         m_proposed(p, b) = (arma::randn() / proposal_window) + m(p, b);
//         // m_proposed(p, b) = 0.0; // 
//         
//         // Prior included in kernel
//         // proposed_model_score += arma::log_normpdf(m_proposed(p, b), delta, t(p, b) / lambda );
//         // current_model_score += arma::log_normpdf(m(p, b), delta, t(p, b) / lambda ); 
//         
//       }
//       proposed_model_score = mKernel(b, m_proposed.col(b));
//       current_model_score = mKernel(b, m.col(b));
//       
//       u = arma::randu();
//       
//       acceptance_prob = std::min(1.0, std::exp(proposed_model_score - current_model_score));
//       
//       if(u < acceptance_prob){
//         m.col(b) = m_proposed.col(b);
//         // m_score = proposed_model_score;
//       }
//       
//     }
//     
//     
//     
//   };
//   
//   arma::vec batchShiftScore(arma::mat m) {
//     
//     arma::vec score(B);
//     score.zeros();
//     
//     for(arma::uword b = 0; b < B ; b++) {
//       // for(arma::uword p = 0; p < P; p++){
//       //   score(b) += arma::log_normpdf(m(p, b), delta,  t(p, b) / lambda );
//       // }
//       score(b) += mKernel(b, m.col(b));
//     }
//     return score;
//   };
//   
//   void clusterPrecisionMetropolis() {
//     
//     double u = 0.0, proposed_model_score = 0.0, acceptance_prob = 0.0, current_model_score = 0.0;
//     
//     // current_model_score += model_score;
//     
//     for(arma::uword k = 0; k < K ; k++) {
//       
//       acceptance_prob = 0.0, proposed_model_score = 0.0, current_model_score = 0.0;
//       
//       for(arma::uword p = 0; p < P; p++){
//         if(N_k(k) == 0){
//           tau(p, k) = arma::randg<double>( arma::distr_param(alpha, 1.0 / beta) );
//         } else {
//           tau_proposed(p, k) = std::exp((arma::randn() * proposal_window_for_logs) + tau(p, k));
//           // tau_proposed(p, k) = arma::randg( arma::distr_param(proposal_window_for_logs * tau(p, k), proposal_window_for_logs));
//           // tau_proposed(p, k) = 1.0;
//           
//           // Log probability under the proposal density
//           proposed_model_score += logNormalLogProbability(tau(p, k), tau_proposed(p, k), proposal_window_for_logs);
//           current_model_score += logNormalLogProbability(tau_proposed(p, k), tau(p, k), proposal_window_for_logs);
//           
//           // Asymmetric proposal density
//           // proposed_model_score += gammaLogLikelihood(tau(p, k), proposal_window_for_logs * tau_proposed(p, k), proposal_window_for_logs);
//           // current_model_score += gammaLogLikelihood(tau_proposed(p, k), proposal_window_for_logs * tau(p, k), proposal_window_for_logs);
//         }
//         
//         // Prior log probability included in kernel
//         // proposed_model_score += invGammaLogLikelihood(tau_proposed(p, k), alpha, 1.0 / beta);
//         // current_model_score += invGammaLogLikelihood(tau(p, k), alpha, 1.0 / beta);
//         
//         // proposed_model_score += invGammaLogLikelihood(t(p, b), t_proposed(p, b) * window, window);
//         // current_model_score += invGammaLogLikelihood(t_proposed(p, b), t(p, b) * window, window);
//         
//       }
//       proposed_model_score += tauKernel(k, tau_proposed.col(k));
//       current_model_score += tauKernel(k, tau.col(k));
//       
//       u = arma::randu();
//       
//       acceptance_prob = std::min(1.0, std::exp(proposed_model_score - current_model_score));
//       
//       if(u < acceptance_prob){
//         tau.col(k) = tau_proposed.col(k);
//         // t_score = proposed_model_score;
//       }
//     }
//     
//     
//     
//   };
//   
//   arma::vec clusterPrecisionScore(arma::mat tau) {
//     
//     arma::vec score(K);
//     score.zeros();
//     
//     for(arma::uword k = 0; k < K ; k++) {
//       // for(arma::uword p = 0; p < P; p++){
//       //   score(k) += gammaLogLikelihood(tau(p, k), alpha, beta);
//       // }
//       score(k) += tauKernel(k, tau.col(k));
//     }
//     return score;
//   };
//   
//   void clusterMeanMetropolis() {
//     
//     double u = 0.0, proposed_model_score = 0.0, acceptance_prob = 0.0, current_model_score = 0.0;
//     
//     // current_model_score += model_score;
//     
//     for(arma::uword k = 0; k < K ; k++) {
//       
//       // proposed_model_score = 0.0, acceptance_prob = 0.0, current_model_score = 0.0;
//       
//       for(arma::uword p = 0; p < P; p++){
//         if(N_k(k) == 0){
//           mu_proposed(p, k) = arma::randn<double>() / (tau(p, k) * kappa) + xi;
//         } else {
//           mu_proposed(p, k) = (arma::randn() * proposal_window) + mu(p, k);
//         }
//         
//         // Prior log probability
//         // proposed_model_score += arma::log_normpdf(mu_proposed(p, k), delta, tau(p, k) / kappa);
//         // current_model_score += arma::log_normpdf(mu(p, k), delta, tau(p, k) / kappa ); 
//         
//       }
//       // The prior is included in the kernel
//       proposed_model_score = muLogKernel(k, mu_proposed.col(k));
//       current_model_score = muLogKernel(k, mu.col(k));
//       
//       u = arma::randu();
//       
//       acceptance_prob = std::min(1.0, std::exp(proposed_model_score - current_model_score));
//       
//       if(u < acceptance_prob){
//         mu.col(k) = mu_proposed.col(k);
//         // mu_score = proposed_model_score;
//       }
//     }
//     
//     
//     
//   };
//   
//   arma::vec clusterMeanScore(arma::mat mu) {
//     
//     arma::vec score(K);
//     score.zeros();
//     
//     for(arma::uword k = 0; k < K ; k++) {
//       // for(arma::uword p = 0; p < P; p++){
//       //   score(k) += arma::log_normpdf(mu(p, k), delta, tau(p, k) / kappa); 
//       // }
//       score(k) += muLogKernel(k, mu.col(k));
//     }
//     return score;
//   };
//   
//   void metropolisStep() {
//     
//     // Metropolis step for cluster parameters
//     clusterPrecisionMetropolis();
//     clusterMeanMetropolis();
//     
//     // Metropolis step for batch parameters
//     batchScaleMetorpolis();
//     batchShiftMetorpolis();
//     
//   };
//   
// };