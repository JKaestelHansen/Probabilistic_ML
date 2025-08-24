# Probabilistic_ML
This repo explores uncertainty estimation and evalutaion of predicted uncertainties using a plethora of metrics
This repo explores probabilistic ML in the context of predictions of diffusional properties with a specific interest in predicting Cohesin extrusion speeds.

Prediction with uncertainty for various modelling approaches.
    * Predict extrusion speeds from DNA loci pairs movement using Mirny lab simulations keeping residence time constant
    * Do this at different extrusion speeds and also for the null model with no extrusion but only an attraction between loci pairs to prove we can differentiate that
    * Show accuracy versus extrusion speed - the idea is that at low extrusion speeds the signal is hidden in the bigger polymer movement and very high become obvious
    * Using a well calibrated model on real data we can see what extrusion speed it predicts and if it even has a uncertainty to trust and do this w/wo cohesion
    
* Different models are explored and compared using uncertainty calibration and quantification
  * The Laplace at last layer (must), Monte-Carlo dropout+Deep ensembles+concrete dropout, SWAG+Deep ensembles
* Model both epistermic and aleatoric uncertainty
* Effect of loss functions, weight decay, variance regularization (+0.5*log variance or similar), training with epistermic and/or aleatoric uncertainty, batch size and batching (Nicky’s work)
* Explore approaches to obtain more calibrated uncertainties
* Use certainties to identify data examples model is uncertain about
* evaluate and calibrate uncertainty
* (add-on 1) do some mix-up strategy for generalization
* (add-on 2) data augmentation
* evaluate accuracy w/wo add-on 1 and/or 2
* test on real data w/wo cohesin depletion
* use model uncertainty to evaluate if any real data examples are far from training distribution + mitigate
