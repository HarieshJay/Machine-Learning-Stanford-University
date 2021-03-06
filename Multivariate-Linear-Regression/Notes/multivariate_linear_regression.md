## Hypothesis Function

$$
h_\theta(x) = \theta_0 + \theta_1x_1 + ... + \theta_nx_n
$$

Where $n$ is the number of features.

By including $x^{(i)}_0 = 1$, the intercept can be seen as another feature. Which allows the hypothesis function to be written as the transpose of the coefficients multiplied by the parameters.
$$
\begin{align}
h_\theta(x) &= \theta_0x_0 + \theta_1x_1 + ... + \theta_nx_n \\
&= \theta^Tx
\end{align}
$$


## Multivariable Gradient Descent

$$
\begin{align*} & \text{repeat until convergence:} \; \lbrace \newline \; & \theta_0 := \theta_0 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_0^{(i)}\newline \; & \theta_1 := \theta_1 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_1^{(i)} \newline \; & \theta_2 := \theta_2 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_2^{(i)} \newline & \cdots \newline \rbrace \end{align*}
$$

Identical to Gradient Descent for univariate linear regression, but needs to be repeated for $n$ features.

**Feature Scaling**

If there is a large range of values that a feature can take, the contour graph will be extremely elliptical, causing gradient descent to take more iterations and a complicated trajectory to find the global minimum. By scaling the features, there will be a much more direct path to the global minimum and gradient descent will converge in fewer iterations.

Scale the features such that $-1 \le x_i \le 1$ for every feature $i$. $[-1,1]$ are rough estimates, values slightly greater or less are acceptable. For example, $x_1 = \frac{size (feet^2) }{2000}$, where the range of the sizes (max - min) is 2000.

The new range after feature scaling is 1.

**Mean normalization**

Subtract the mean for the input from each value, creating a new mean of 0. 

Applying both mean normalization and feature scaling will look like
$$
x_i := \frac{x_i-\mu_i}{s_i}
$$
Where $s_i$ is the range of values or the standard deviation, which will have different results.

**Debugging**

Plotting the cost function and the number of iterations will display how well gradient descent is working, and how many iterations are needed for gradient descent to converge. If the cost function increases as the number of iterations increase, this means gradient descent is not working, and usually means the learning rate is too high. If the learning rate is too high, the global minimum can be overshot. If the cost function increases then decreases repeatedly, the learning rate is also too high.

If the cost function decreases by a extremely small value, for example less than $10^{-3}$ in one iteration, declare convergence, this is called an automatic convergence test.

Try a sequence of $\alpha$ similar to $..,0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1,...$.

## Normal Equation Method

An alternative to using Gradient Descent to minimize the cost function.

Create a $m \times (n+1)$ design matrix of the observations, where $m$ is the number of observations, and $n+1$ is the number of parameters and the column of 1s for the intercept. Also, create a $m-$dimensional vector for the responses. Then $\theta$ can be computed with
$$
\theta = (X^TX)^{-1}X^Ty
$$
Feature scaling is not necessary when using the normal equation method.

Normal equation is slow if $n$ is large, unlike with gradient descent. Because the cost of computing an inverse is cubic in terms of the parameters. Avoid using the normal equation method around $10^4+$ features.

It is possible that $(X^TX)$ is non-invertible. One reason may be because of redundant features, $x_i =$ size in feet and $x_2 =$ size in meters will be linearly dependent. $(X^TX)$ may not be invertible if $m \le n$, the solution would be to delete some features or use regularization.