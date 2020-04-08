## Cost Function

Cost functions are used to measure the accuracy of the hypothesis. Minimizing the cost function guarantees the best fit to the data with the given model.
$$
J(\theta_0, \theta_1) = \frac{1}{2m}\sum^m_{i=1}(\hat{y}-y_i)^2 = \frac{1}{2m}\sum^m_{i=1}(h_\theta(x_i)-y_i)^2
$$

For linear regression the cost function is the mean squared error, or the residual sum of squares divided by the number of degrees of freedom. The $\frac{1}{2}$ is added so the derivative of gradient descent is simpler.

The cost function can be seen as a function of the coefficients. Minimize the cost function by finding the coefficients that when applied to the regression line minimizes the mean squared error.

## Gradient Descent ( Batch )

Can be used to minimize the cost function, there are other applications as well.
$$
\theta_j := \theta_j - \alpha\frac{\partial}{\partial\theta_j}J(\theta_0, \theta_1)
$$
where $j=0,1$. This is done until convergence. Take the partial derivative of the cost function with respect to the coefficient being adjusted for and choose the direction with the steepest descent. The size of the steps are determined by $\alpha$, which is the learning rate.

Both coefficients should be updated simultaneously, otherwise the updated value of one will be used to calculate the value of the other, leading to a wrong implementation. Regardless if the partial derivative is positive or negative, $\theta_j$ moves closer to the minima, and will eventually converge. If the partial derivative is negative, $\theta_j$ increases, and decreases if the partial derivative is positive. It will converge at the minimum since the partial derivative will be 0.

If $\alpha$ is too small, gradient descent will be too slow and computationally intensive. However, if $\alpha$ is too large, the minimum can be overshot, fail to converge and may diverge.

This is batch gradient descent, due to the fact that at every step the entire training set is used.

Finding the coefficients for simple linear regression can be seen as using the mean squared error as the cost function
$$
\begin{align*} \text{repeat until convergence: } \lbrace & \newline \theta_0 := & \theta_0 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m}(h_\theta(x_{i}) - y_{i}) \newline \theta_1 := & \theta_1 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m}\left((h_\theta(x_{i}) - y_{i}) x_{i}\right) \newline \rbrace& \end{align*}
$$
