## Evaluating a Hypothesis

Split the data into a training set of $70\%$ and a test set of $30\%$. Fit $\Theta$ and minimize $J_{train}(\Theta)$ on the training set, then compute the test set error $J_{test}(\Theta)$.

**Test Set Error on Linear Regression**
$$
J_{test}(\Theta) = \dfrac{1}{2m_{test}} \sum_{i=1}^{m_{test}}(h_\Theta(x^{(i)}_{test}) - y^{(i)}_{test})^2
$$
**Test Set Error on Classification**
$$
err(h_\Theta(x),y) = \begin{matrix} 1 & \mbox{if } h_\Theta(x) \geq 0.5\ and\ y = 0\ or\ h_\Theta(x) < 0.5\ and\ y = 1\newline 0 & \mbox otherwise \end{matrix}
$$
The average test error can be computed as
$$
\text{Test Error} = \dfrac{1}{m_{test}} \sum^{m_{test}}_{i=1} err(h_\Theta(x^{(i)}_{test}), y^{(i)}_{test})
$$
is the proportion of the misclassified test data.

## Model Selection

When choosing a model, if the criteria to chose the best model is the lowest test set error, this means that the test set is no longer viable to evaluate the data, since the chosen model already has the lowest test set error.

Partition the data into 3 sets: training set, cross validation set, and test set.

1. Fit the model, $\Theta$ to the data in the training set
2. Pick the model with the least error in the cross validation set
3. Estimate the generalization error using the test set error

**Diagnosing High Variance or High Bias**

Let $d$ represent the degree of the polynomial being fit or some measure of variance in the model. The training error will decrease with the increase of $d$, eventually leading to overfitting. The cross validation error will decrease with the increase of $d$ to some minimum, then will increase when the model starts to overfit.

High bias (underfitting) can be identified by a high training error $J_{train}(\Theta)$ and high cross validation error $J_{CV}(\Theta)$. $J_{CV}(\Theta) \approx J_{train}(\Theta)$ 

High variance (overfitting) can be identified by a low training error  $J_{train}(\Theta)$ and a high cross validation error $J_{CV}(\Theta)$.

**Choosing a Value for $\lambda$**

Where $\lambda$ determines the magnitude of the $\Theta$ values in the cost function.

As $\lambda$ approaches 0, the model tends to overfit the data since there is penalty for a high magnitude of $\Theta$ values. As $\lambda$ increases, the fit becomes more rigid and will underfit the data at some point.

Using various $\lambda$ values, train the data on the training set, then choose the $\lambda$ values that correspond to the smallest error on the cross validation set. Then evaluate the model the test set. 

**Learning Curves**

Learning curves are a plot of the error as a function of the number of training observations.

High bias (underfitting) can be identified by a $J_{train}$ curve that starts low with a small number of training observations since it can fit that perfectly, but will increase when there are more training observations are cannot model them well. The $J_{CV}$ error will begin high since the model does not understand the structure at all, but will decrease with more training observations, and will converge near the training error $J_{train}(\Theta) \approx J_{CV}(\Theta)$. The value they converge on will be a high error. More training data will not help.

High variance (overfitting) can be identified by a $J_{train}$ curve that starts low, then increases with more observations since it will not be able to fit the data perfectly. The $J_{CV}$ curve will start high and then decrease without leveling off. $J_{train}(\Theta) < J_{CV}(\Theta)$ with the difference between them being significant. With more training observations the value between them will become less. More training data is likely to help.

**Basic troubleshooting includes**

* Adding more training examples $\rightarrow$ fixes high variance
* Smaller sets of features $\rightarrow$ fixes high variance
* Additional features  $\rightarrow$ fixes high bias
* Polynomial features $\rightarrow$ fixes high bias
* Decrease $\lambda$ $\rightarrow$ fixes high bias
* Increase $\lambda$  $\rightarrow$ fixes high variance

**Diagnosing Neural Networks**

* A neural network with fewer parameters is prone to underfitting and is computationally cheaper. This can be addressed by adding more layers or nodes.
* A neural network with more parameters is prone to overfitting and is computationally expensive. Regularization can be used to fix this problem.

One single hidden layer is a good starting point. Train the neural network on different hidden layers and choose the one with the best cross validation error.

**Error Analysis**

Recommended approach to solving machine learning problems.

1. Create a simple algorithm, implement it quickly ( 1 day ), and test it early on the cross validation data.
2. Plot learning curves to decide if more data, features, etc.. are needed.
3. Manually examine the errors on examples in the cross validation set and try to spot trends where errors were made.

**Error Metrics**

For some cases, ratio of true positives / false positives is more important than the total error of a data set.

Data with skewed classes can give false confidence when they have a low error rate since the errors remain in the class with fewer observations.

Confusion Matrix or Precision / Recall measure the number of misclassifications in the test data. The top row corresponds to the true observation results, and the left column is the predicted classifications.

|      | 1              | 0              |
| ---- | -------------- | -------------- |
| 1    | True positive  | False positive |
| 0    | False negative | True negative  |

$$
\begin{align}
\text{Precision} &= \frac{\text{True positives}}{ \text{# Predicted positive}} = \frac{\text{True positives}}{ \text{True positives + False positives}}\\
\text{Recall} &= \frac{\text{True positives}}{ \text{# Actual positive}} = \frac{\text{True positives}}{ \text{True positives + False negatives}}
\end{align}
$$

( Cancer prediction example )

Precision : Of all patients where predicted $y=1$, what fraction actually has cancer? Measure of false positives.

Recall : Of all patients that actually have cancer, what fraction was correctly detected? Measure of false negatives.

1 should be assigned to the more rare class.

Optimally, both values should be high.

Precision / Recall Tradeoff: 
$$
h_\theta(x) \ge \text{Threshold}
$$
A higher threshold corresponds to higher precision and lower recall. Reduces the number of false positives.

A lower threshold corresponds to a higher recall and lower precision. Reduces false negatives.
$$
F_1 Score = 2 \frac{PR}{P+R}
$$
Good measure of the average of Precision and Recall in a test set without favoring one over the other.