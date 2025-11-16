# Localized Entropy Cost Function for Neural Network Training in Multi-Class Event Probability Estimation

## FIELD OF INVENTION

The present invention relates to the field of machine learning, specifically
to improved training methodologies for neural networks used in multi-class
event probability estimation. The invention pertains to novel cost functions
for training neural networks to accurately estimate probability distributions
across multiple events belonging to many classes with widely varying
probability ranges.

The invention addresses current limitations in training approaches for neural
networks designed to estimate probabilities of diverse events, such as
medical condition diagnoses, user interaction predictions, and risk
assessment.

The field encompasses computational methods for optimizing neural network
training when the target probability distributions follow log-normal
characteristics and range from very low probabilities (e.g., 10<sup>-6</sup>) to
relatively high probabilities (e.g., 10<sup>-1</sup>).

The invention enhances conventional cost functions, particularly cross
entropy and its variants, which are commonly employed in training neural
networks for classification and probability estimation tasks, but exhibit
diminished effectiveness when applied to multi-class event probability
estimation across disparate probability ranges.

## BACKGROUND

### Problem Being Solved

Neural networks have become ubiquitous tools for estimating probabilities
across various domains, including medical diagnosis, advertising interaction
prediction, and risk assessment. A significant challenge arises when
training these networks to simultaneously estimate probabilities for events
from multiple classes whose natural distributions vary by several orders of
magnitude. The probability distributions of such events typically follow
log-normal patterns, with some events occurring at extremely low
probabilities (10<sup>-6</sup> to 10<sup>-5</sup>) while others manifest at much higher rates
(10<sup>-2</sup> to 10<sup>-1</sup>). When a single neural network is tasked with predicting
this spectrum of events, traditional training approaches fail to optimize
equally across the entire probability range, resulting in suboptimal
performance for certain events, particularly those with lower probabilities.

### Limitations of Existing Approaches

The current state-of-the-art approach for training neural networks in
probability estimation relies predominantly on cross entropy (or its variants
such as focal loss). These cost functions, while mathematically sound for
classification tasks, exhibit significant limitations when applied to
multi-class event probability estimation:

1.  **Disproportional Learning Signal:** Cross entropy inherently provides
    stronger learning signals for higher probability events compared to lower
    probability events, creating an imbalance in the training process. This
    results in neural networks that perform well for common events but poorly
    for rare ones.

2.  **Probability Range Insensitivity:** The standard implementations of
    cross entropy lack sensitivity to the vast differences in probability
    scales that naturally occur across different events, treating a
    prediction error at probability 10<sup>-1</sup> similarly to an error of the same
    magnitude at probability 10<sup>-6</sup>.

3.  **Suboptimal Convergence:** Due to the imbalanced learning signal,
    networks trained with standard cross entropy demonstrate inconsistent
    convergence rates across the probability spectrum, often plateauing
    prematurely for low-probability events.

## TECHNICAL CHALLENGES ADDRESSED

1.  **Scale-Appropriate Penalization:** Creating a cost function that equally
    penalizes prediction errors across distinct events with different multiple
    orders of magnitude in the base probability rate.

2.  **Distribution Recreation:** Enabling neural networks to accurately
    recreate the underlying probability distributions for all events,
    regardless of their position in the probability spectrum.

3.  **Balanced Learning:** Establishing training dynamics that provide equal
    learning signals for all events, ensuring that learning to predict events
    from classes with high and low probability provides sufficient gradient value
    during back propagation during the optimization process.

## DATA AND ACCURACY ISSUES

1.  **Distribution Mismatch:** Real-world data often follows log-normal
    distributions governed by power laws and the outcome of the central limit
    theorem, but standard cost functions are not optimized for such
    distributions.

2.  **Accuracy Disparities:** Networks trained with conventional methods show
    significant disparities in prediction accuracy across the probability
    spectrum, with typically poorer performance at probability extremes.

3.  **Calibration Problems:** Probability estimates for rare events tend to be
    poorly calibrated when trained with standard cross entropy, leading to
    systematic overestimation or underestimation.

## SUMMARY

This invention introduces a novel cost function for training neural networks
in multi-class event probability estimation that overcomes limitations of
conventional approaches. It normalizes each event's cross entropy loss by
the cross-entropy of the mean probability for that class, creating a balanced
training objective that equalizes learning effects across all probability
ranges. This normalization technique is a significant departure from
traditional approaches, which use cross entropy as a cost function, with
a difference error (penalty) attributed to the prediction of a positive event
(label 1) or a negative event (label 0).

This formula effectively increases the penalty when the prediction diverges
from the true label, making it a critical component in training binary
classifiers.

```
         N
(1 / N) ∑ [ - y_i * log(ŷ_i) - (1 - y_i) * log(1 - ŷ_i) ]
        i=1
```

or

```
        N
        ∑ [ -y_i * log(ŷ_i) - (1-y_i) * log(1-ŷ_i) ]
       i=1
NCE = ─────────────────────────────────────────────
        N
        ∑ [ -y_i * log(p) - (1-y_i) * log(1-p) ]
       i=1
```

Where *p* is the mean probability for all events.

NCE is a metric that scales standard cross-entropy loss relative to a
baseline, making model performance more interpretable and comparable across
different datasets.

It is calculated as NCE = CE/H, where CE is the standard cross-entropy loss
and H is a normalization factor, typically the entropy of the ground truth
distribution:

```
        N
        ∑ [ -y_i * log(ŷ_i) - (1-y_i) * log(1-ŷ_i) ]
       i=1
NCE = ─────────────────────────────────────────────
        N
        ∑ [ -y_i * log(p) - (1-y_i) * log(1-p) ]
       i=1
```

Where *p* is the mean probability for all events.

NE is a good metric to assess a model's prediction quality (power) and
compare one model to another, using NE vs BCE does not change anything in the
neural network's training as the H denominator is constant for a given
training data set.

One important way to interpret NE is that if NE is close to 1.0, then all
predictions are close to the base rate and the prediction model isn't providing
much value. Such a model, while very simple, is also not useful as it just
generates a mean value for each prediction estimation.

When NE is decreasing and approaching 0, predictions become more precise, and
more diverse events are predicted with different probabilities.

Therefore, NE as a cost metric would be efficient and optimal if we would be
predicting a single event. However, when a neural network is trained to
predict a set of events from multiple classes with significantly different
means (e.g., 10<sup>-6</sup> vs 10<sup>-2</sup>), NE is no longer accurate.

The core of this invention is to extend the properties that work for single
class events to multi-class event prediction and retain similar properties
of NE. This is achieved by normalizing the binary cross entropy numerator
for predictions of each event by the cross-entropy of the base probability
for that class. The resulting cost function is called Localized Entropy, and
expressed by the following equation:

```
               ⎛ N_j                                                      ⎞
               ⎜  ∑  [ -y_i * log(ŷ_i) - (1-y_i) * log(1-ŷ_i) ]           ⎟
          M    ⎜ i=1                                                      ⎟
          ∑    ⎜ ──────────────────────────────────────────────────────── ⎟
LE =     j=1   ⎜ N_j                                                      ⎟
               ⎜  ∑  [ -y_i * log(p_j) - (1-y_i) * log(1-p_j) ]           ⎟
               ⎝ i=1                                                      ⎠
    ────────────────────────────────────────────────────────────────────────
                                    M
                                    ∑ N_j
                                   j=1
```

```tex
\[
\frac{\displaystyle \sum_{j=1}^{M}
\frac{\displaystyle \sum_{i=1}^{N_j}\!\left[-\,y_i\log(\hat{y}_i)-(1-y_i)\log\!\bigl(1-\hat{y}_i\bigr)\right]}
{\displaystyle \sum_{i=1}^{N_j}\!\left[-\,y_i\log(p_j)-(1-y_i)\log\!\bigl(1-p_j\bigr)\right]}}
{\displaystyle \sum_{j=1}^{M} N_j}
\]
```

- `M` – total number of different events, where j iterates from 1 to M
- `p_j` – mean probability for a given event j
- `N_j` – total number of examples for given event j
- `ŷ_i` – predicted probability for example i for a given event
- `y_i` – label for a given example i for given event

As a result, the cost function produces an equal effect on samples from
different events even if they have different base probabilities.

## CLAIMS

1.  A localized entropy cost function for training neural networks in
    multi-class event prediction, comprising:
    a. calculating a cross-entropy loss for each event class.
    b. normalizing the cross-entropy loss for each event class by the
       cross-entropy of the mean probability for that class.
    c. aggregating the normalized losses to form the overall cost function
       wherein the localized entropy cost function achieves balanced learning
       and accurate recreation of the event probability distribution across
       all event classes.

2.  The localized entropy cost function of Claim 1, wherein the normalization
    ensures that the contribution of each event class to the total cost is
    independent of its mean probability, thereby achieving balanced learning
    across all classes.

3.  The localized entropy cost function of Claim 1, wherein the cross-entropy
    of the mean probability for each event class is computed as the
    cross-entropy between the true labels and a constant prediction equal to
    the mean probability for that class.

4.  A method for implementing the localized entropy cost function of Claim 1,
    comprising:
    a. performing an initial training pass to estimate the mean probability
       for each event class.
    b. using the estimated mean probabilities to compute the normalization
       factors.
    c. performing subsequent training passes using the normalized cost
       function.

5.  A method for implementing the localized entropy cost function of Claim 1,
    comprising:
    a. initializing the normalization factors using a global mean probability.
    b. during training, accumulating statistical data for each event class
       with each batch.
    c. updating normalization factors for each class as sufficient statistical
       data is accumulated.
    d. using the updated normalization factors in the cost function for
       subsequent training.

6.  A system for training a neural network, comprising:
    a. a neural network model configured to predict probabilities for multiple
       event classes.
    b. a training module that uses the localized entropy cost function of
       Claim 1 to optimize the neural network's parameters.
    c. a data input module for receiving training data with labels for the
       multiple event classes.