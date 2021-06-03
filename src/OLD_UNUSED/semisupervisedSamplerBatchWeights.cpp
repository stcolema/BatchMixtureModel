// 
// 
// 
// 
// 
// class semisupervisedSamplerBatchWeights : public virtual samplerBatchWeights, 
//                                           public virtual sampler {
// private:
//   
// public:
//   
//   arma::uword N_fixed = 0;
//   arma::uvec fixed, unfixed_ind;
//   arma::mat alloc;
//   
//   using sampler::sampler;
//   
//   semisupervisedSamplerBatchWeights(
//     arma::uword _K,
//     arma::uword _B,
//     arma::uvec _labels,
//     arma::uvec _batch_vec,
//     arma::vec _concentration,
//     arma::mat _X,
//     arma::uvec _fixed
//   ) : 
//     samplerBatchWeights(_K, _B, _labels, _batch_vec, _concentration, _X),
//     sampler(_K, _B, _labels, _batch_vec, _concentration, _X) {
//     
//     arma::uvec fixed_ind(N);
//     
//     fixed = _fixed;
//     N_fixed = arma::sum(fixed);
//     fixed_ind = arma::find(_fixed == 1);
//     unfixed_ind = find(fixed == 0);
//     
//     alloc.set_size(N, K);
//     alloc.zeros();
//     
//     for (auto& n : fixed_ind) {
//       alloc(n, labels(n)) = 1.0;
//     }
//     
//     std::cout << "\n\nIn semi no 2.";
//   };
//   
//   // Destructor
//   virtual ~semisupervisedSamplerBatchWeights() { };
//   
//   virtual void updateAllocation() {
//     
//     uword b = 0;
//     double u = 0.0;
//     arma::uvec uniqueK;
//     arma::vec comp_prob(K);
//     
//     complete_likelihood = 0.0;
//     for (auto& n : unfixed_ind) {
//       
//       b = batch_vec(n);
//       ll = itemLogLikelihood(X_t.col(n), b);
//       
//       // Update with weights
//       comp_prob = ll + log(w.col(b));
//       
//       likelihood(n) = arma::accu(comp_prob);
//       
//       // Normalise and overflow
//       comp_prob = exp(comp_prob - max(comp_prob));
//       comp_prob = comp_prob / sum(comp_prob);
//       
//       // Save the allocation probabilities
//       alloc.row(n) = comp_prob.t();
//       
//       // Prediction and update
//       u = arma::randu<double>( );
//       
//       labels(n) = sum(u > cumsum(comp_prob));
//       alloc.row(n) = comp_prob.t();
//       
//       complete_likelihood += ll(labels(n));
//       
//       // Record the log likelihood of the item in it's allocated component
//       likelihood(n) = ll(labels(n));
//     }
//     
//     // The model log likelihood
//     observed_likelihood = arma::accu(likelihood);
//     
//     // Number of occupied components (used in BIC calculation)
//     uniqueK = arma::unique(labels);
//     K_occ = uniqueK.n_elem;
//   };
//   
// };
