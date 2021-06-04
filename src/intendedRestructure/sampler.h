// // sampler.h
// #ifndef SAMPLER_H
// #define SAMPLER_H
// 
// # include "mixture.h"
// 
// class sampler {
// 
// private:
// 
// public:
// 
//   arma::uword K, B, N, P, K_occ, accepted = 0;
//   double observed_likelihood = 0.0, BIC = 0.0, model_score = 0.0, complete_likelihood = 0.0;
//   arma::uvec labels, N_k, batch_vec, N_b, KB_inds, B_inds;
//   arma::vec concentration, w, ll, likelihood;
//   arma::umat members;
//   arma::mat X, X_t, alloc;
//   arma::field<arma::uvec> batch_ind;
// 
//   // This will have to be a smart pointer and use a factory.
//   // mixture mix;
// 
//   // Parametrised class
//   sampler(
//     arma::uword _K,
//     arma::uword _B,
//     arma::uvec _labels,
//     arma::uvec _batch_vec,
//     arma::vec _concentration,
//     arma::mat _X);
// 
//   // Destructor
//   virtual ~sampler();
// 
//   // Functions required of all mixture models
//   virtual void updateWeights();
//   virtual void updateAllocation();
//   virtual void metropolisStep();
//   virtual void sampleFromPriors();
//   virtual void calcBIC();
//   virtual arma::vec itemLogLikelihood(arma::vec x, arma::uword b);
// 
// };
// 
// #endif /* SAMPLER_H */