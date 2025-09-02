# Probabilistic_ML
This repo explores uncertainty estimation and evalutaion of predicted uncertainties using a plethora of metrics
This repo explores probabilistic ML in the context of predictions of diffusional properties with a specific interest in predicting Cohesin extrusion speeds.

# metrics

1 
histogram of uncertainties to assess spread

2 
scatter plot of uncertainties vs errors to assess calibration
higher uncertainties pair with higher errors

3 
scatter plot of uncertainties vs target to assess patterns or bias

4 
quantile_and_oracle_errors and ranking_confidence_curve are the same
but quantile_and_oracle_errors normalizes to case of all errors included (quantile=1)
Quantile Error is the average error in each quantile of uncertainty (error in bins of uncertainty)
Oracle Error is the average error in each quantile of oracle errors (error in bins of error) corresponding 
if quantile errors follow the oracle errors, the model is well calibrated

5
error_based_calibration is a plot of the average error versus the average uncertainty
when binning by uncertainty being on the diagonal is good calibration

6
Reliability Diagram
Reliability Diagram is a plot of the predicted confidence against the true confidence binning the predictions by confidence and plotting against accuracy in each bin if predicting probability of 10%, we want accuracy to be 10% etc. The diagonal is the line of perfect calibration. The closer the plot is to the diagonal, the better the calibration. Empirical coverage is the observed proportion of true values that fall within the expected intervals. If we observe the true values falling within, say, a 95% confidence interval 95% of the time, then the empirical coverage matches the expected coverage. If empirical coverage deviates from expected coverage, the model may be over- or under-confident. If the model is overconfident, it predicts narrow uncertainty intervals (i.e., low uncertainties). This leads to intervals that are too tight to  capture the true values often enough, resulting in low empirical coverage  compared to the expected coverage. Thus, empirical < expected suggests overconfidence the line will then be below the diagonal line vice versa for underconfidence

7
plotting number of standard deviations away predictions are from the true value

8
Area Under Confidence Oracle Error: {auco}
measures differences in quantile error curves and oracle error curves

9
Error Drop
Difference between first uncertainty quantile and last uncertainty quantile

10
Decreasing Ratio
fractions of uncertainties larger than the next quantiles uncertainties, to cover monotonicity

11
Reduced chi squared statistic. A method would thus be over-confident if the empirical error is larger than the uncertainties it predicts.

12
Expected Calibration Error
average error between bins of the reliability diagram showing the average deviation from the true value in each bin

13 
Max Calibration Error
max difference in reliability diagram showing worst case deviation from the true value in each bin

14
expected normalized calibration error
measures the mean of differences between the predicted root mean variance and the RMSE per bin normalized by root mean variance of the error-based calibration diagram

15
Sharpness
np.std(uncertainties, ddof=1) / np.mean(uncertainties)
measures diversity in the uncertainty estimates because outputting constant uncertainty is not useful



# goals
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
