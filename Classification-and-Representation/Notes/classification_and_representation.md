## Logistic Regression Model

The hypothesis function can be written as the logistic or the sigmoid function.
$$
h_\theta(x) = \frac{1}{1 + e^{-\theta^T_x}}
$$
The hypothesis function represents the probability of $x$ being in a class. $h_\theta(x) = P(y=1|x;\theta)$ - probability that $y=1$, given $x$, parameterized by $\theta$.

**Cost Function**

Using the regular MSE cost function would not work for linear regression since it would create a non-convex function, which would be difficult to find the global minimum instead of the local minimum.
$$
\begin{align*}& J(\theta) = \dfrac{1}{m} \sum_{i=1}^m \mathrm{Cost}(h_\theta(x^{(i)}),y^{(i)}) \newline & \mathrm{Cost}(h_\theta(x),y) = -\log(h_\theta(x)) \; & \text{if y = 1} \newline & \mathrm{Cost}(h_\theta(x),y) = -\log(1-h_\theta(x)) \; & \text{if y = 0}\end{align*}
$$

If the hypothesis correctly predicts that $y$ is equal to 1, the cost will be 0, but will increasingly large the further the hypothesis is from 0, approaching infinity. Similarly, if the hypothesis function predicts 0 correctly, the cost is 0, otherwise it will be increasingly large the further it is from 0.

This is equivalent to
$$
\mathrm{Cost}(h_\theta(x),y) = - y \; \log(h_\theta(x)) - (1 - y) \log(1 - h_\theta(x))
$$
Can be derived from the principle of maximum likelihood estimation, and is convex.

**Gradient Descent**
$$
\begin{align*}& Repeat \; \lbrace \newline & \; \theta_j := \theta_j - \alpha \dfrac{\partial}{\partial \theta_j}J(\theta) \newline & \rbrace\end{align*}
$$
 The partial derivative of the logistic regression cost function derivates to
$$
\begin{align*} & Repeat \; \lbrace \newline & \; \theta_j := \theta_j - \frac{\alpha}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)} \newline & \rbrace \end{align*}
$$
This is identical to the linear regression gradient descent rule, but the definition of the hypothesis function has changed.

**Optimization Algorithms**

There are more complex ways to minimize the cost function, when they are given the cost function and the partial derivative.

* Conjugate Gradient
* BFGS
* L-BFGS

They are often faster than gradient descent and do not require to manually pick a learning rate.

**Multiclass Classification: One-vs-all**

For each of the classes, create a dummy representation where the negative case is all of the other classes, and predict the probability of the observation being in the current class over it being in one of the others. Choose the class where this probability value is largest among all of the classes.

## Overfitting

There are two main options to address overfitting

1. Reduce the number of features
   * Manually select the important features
   * Use a model selection algorithm
2. Regularization
   * Keep all features, but reduce the magnitude of the parameter coefficients
   * Works well with a lot of slightly useful features

**Regularization**

To apply regularization, use a modified cost function.
$$
\frac{1}{2m}\sum^m_{i=1}(h_\theta(x^{(i)})-y^{(i)})^2+\lambda\sum ^n_{j=1}\theta^2_j
$$
Where $\lambda$ is some large number. In order to minimize this cost function, the magnitude of the coefficients must be reduced, which would allow for smoother models and less overfitting. $\lambda$ must be chosen so that it is large enough to reduce the coefficients, but not too large where the coefficients are set near 0 and result in underfitting.

**Regularized Linear Regression**

Gradient Descent will now take the form of
$$
\begin{align*} & \text{Repeat}\ \lbrace \newline & \ \ \ \ \theta_0 := \theta_0 - \alpha\ \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_0^{(i)} \newline & \ \ \ \ \theta_j := \theta_j - \alpha\ \left[ \left( \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)} \right) + \frac{\lambda}{m}\theta_j \right] &\ \ \ \ \ \ \ \ \ \ j \in \lbrace 1,2...n\rbrace\newline & \rbrace \end{align*}
$$
 The partial derivative now takes into account the regularization term. Note that regularization is not applied to $\theta_0$ since the intercept coefficient does not need to be penalized.

The Normal Equations Method will now look like
$$
\begin{align*}& \theta = \left( X^TX + \lambda \cdot L \right)^{-1} X^Ty \newline& \text{where}\ \ L = \begin{bmatrix} 0 & & & & \newline & 1 & & & \newline & & 1 & & \newline & & & \ddots & \newline & & & & 1 \newline\end{bmatrix}\end{align*}
$$
Here $L$ is a $(n+1) \times (n+1)$ matrix. 

Previously, $X^TX$ would not be invertible if $m < n$, however, when the term $\lambda \cdot L$ is added, $X^TX + \lambda \cdot L$ will always be invertible.

**Regularized Logistic Regression**

With regularization, the cost function will now look like
$$
J(\theta) = -\frac{1}{m}\sum^m_{i=1}[y^{(i)}log(h_\theta(x^{(i)}))+(1-y^{(i)})log(1-h_\theta(x^{(i)})] + \frac{\lambda}{2m}\sum^n_{j=1}\theta^2_j
$$
Gradient Descent stays identical to the implementation for linear regression with a sigmoid hypothesis function.
$$
\begin{align*} & \text{Repeat}\ \lbrace \newline & \ \ \ \ \theta_0 := \theta_0 - \alpha\ \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_0^{(i)} \newline & \ \ \ \ \theta_j := \theta_j - \alpha\ \left[ \left( \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)} \right) + \frac{\lambda}{m}\theta_j \right] &\ \ \ \ \ \ \ \ \ \ j \in \lbrace 1,2...n\rbrace\newline & \rbrace \end{align*}
$$
