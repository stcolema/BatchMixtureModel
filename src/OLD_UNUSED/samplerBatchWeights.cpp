// 
// 
// 
// 
// class samplerBatchWeights : virtual public sampler {
//   
// private:
//   
// public:
//   
//   // double x = 0.0, y = 0.0;
//   umat N_kb;
//   mat w;
//   
//   // Parametrised class
//   samplerBatchWeights(
//     arma::uword _K,
//     arma::uword _B,
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
//     // // Weights on hyperparameters for the weights.
//     // x = _x;
//     // y = _y;
//     
//     // Weights
//     // double x, y;
//     w = zeros<mat>(K, B);
//     
//     // Count of individuals in class k in batch b
//     N_kb = zeros<umat>(K, B);
//     
//     std::cout << "\n\nIn new smapler.";
//     
//   };
//   
//   // Destructor
//   virtual ~samplerBatchWeights() { };
//   
//   // Functions required of all mixture models
//   void updateWeights(){
//     
//     
//     w.zeros();
//     
//     double a = 0.0;
//     
//     double x = 0.0, y = 1.0, z = 0.0;
//     //
//     for (uword b = 0; b < B; b++) {
//       for (uword k = 0; k < K; k++) {
//         
//         // Find how many labels have the value
//         members.col(k) = labels == k;
//         N_k(k) = sum(members.col(k));
//         // N_kb(k, b) = sum(members.col(k));
//         N_kb(k, b) = sum( (labels == k) && (batch_vec == b) );
//         
//         // // Hyperparameter is a weighted sum of the previous batch's class count
//         // // and the current batch's class count
//         // if(b > 0) {
//         //   if(b < B - 1) {
//         //     a  = concentration(k) + x * N_kb(k, b - 1) + y * N_kb(k, b) + z * N_kb(k, b + 1);
//         //     a = a / (1 + x + y + z);
//         //   } else {
//         //     a  = concentration(k) + x * N_kb(k, b - 1) + y * N_kb(k, b);
//         //     a = a / (1 + x + y);
//         //   }
//         //   // a = a / (1 + x  + y);
//         // } else {
//         //   a  = concentration(k) + y * N_kb(k, b) + z * N_kb(k, b + 1);
//         //   a = a / (1 + x + z);
//         //   // a = (1 + x);
//         // }
//         
//         a  = concentration(k) + N_kb(k, b);
//         
//         // Update weights by sampling from a Gamma distribution
//         w(k, b) = randg( distr_param(a, 1.0) );
//         
//       }
//       
//       // std::cout << "\n\nLast a: " << a;
//       //
//       // std::cout << "\n\nw.col(b):\n" << w.col(b);
//       
//       x = accu(w.col(b));
//       
//       // Convert the cluster weights (previously gamma distributed) to Beta
//       // distributed by normalising
//       w.col(b) = w.col(b) / x;
//       
//       // std::cout << "\n\nw.col(b) (after normalisation):\n" << w.col(b);
//     }
//     
//     // std::cout << "\n\nWeights:\n" << w;
//     // std::cout << "\n\nN_kb:\n" << N_kb;
//     // std::cout << "\n\nN_k:\n" << N_k;
//     
//   };
//   
//   virtual void updateAllocation() {
//     
//     uword b = 0;
//     double u = 0.0;
//     uvec uniqueK;
//     vec comp_prob(K);
//     
//     complete_likelihood = 0.0;
//     for(uword n = 0; n < N; n++){
//       
//       b = batch_vec(n);
//       ll = itemLogLikelihood(X_t.col(n), b);
//       
//       // std::cout << "\n\nAllocation log likelihood: " << ll;
//       // Update with weights
//       comp_prob = ll + log(w.col(b));
//       
//       likelihood(n) = accu(comp_prob);
//       
//       // std::cout << "\n\nWeights: " << w;
//       // std::cout << "\n\nAllocation log probability: " << comp_prob;
//       
//       // Normalise and overflow
//       comp_prob = exp(comp_prob - max(comp_prob));
//       comp_prob = comp_prob / sum(comp_prob);
//       
//       // Prediction and update
//       u = randu<double>( );
//       
//       labels(n) = sum(u > cumsum(comp_prob));
//       // alloc.row(n) = comp_prob.t();
//       
//       complete_likelihood += ll(labels(n));
//       
//       // Record the likelihood of the item in it's allocated component
//       // likelihood(n) = ll(labels(n));
//     }
//     
//     // The model log likelihood
//     observed_likelihood = accu(likelihood);
//     
//     // Number of occupied components (used in BIC calculation)
//     uniqueK = unique(labels);
//     K_occ = uniqueK.n_elem;
//   };
//   
// };
