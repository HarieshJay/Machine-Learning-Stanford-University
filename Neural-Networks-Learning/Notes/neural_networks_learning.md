## Cost Function and Backpropagation

* $L =$ total number of layers in network
* $s_l =$ number of units ( not counting bias unit ) in layer $l$
* $K =$ number of output units/ classes

$$
\begin{gather*} J(\Theta) = - \frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K \left[y^{(i)}_k \log ((h_\Theta (x^{(i)}))_k) + (1 - y^{(i)}_k)\log (1 - (h_\Theta(x^{(i)}))_k)\right] + \frac{\lambda}{2m}\sum_{l=1}^{L-1} \sum_{i=1}^{s_l} \sum_{j=1}^{s_{l+1}} ( \Theta_{j,i}^{(l)})^2\end{gather*}
$$

The cost function is similar to the regularized logistic regression cost function but iterated on all the output nodes and $\Theta$ values.

**Backpropagation**

**Calculating partial derivatives**

1. Set $a^{(1)} := x^{(t)}$

2. Perform forward propagation to compute $a^{(l)}$ for $l=2,3,...L$ 

3. Using $y^{(t)}$, compute $\delta^{(L)} = a^{(L)} - y^{(t)}$

   ​	 $\delta^{(L)}$ is a vector of the error values in the last layer. They can be computed by comparing the output to the true response values.

4. Compute $\delta^{(L-1)}, \delta^{(L-2)},...,\delta^{(2)}$ using 
$$
   \delta^{(l)} = ((\Theta^{(l)})^T \delta^{(l+1)})\ \cdot g'(z^{(l)}) = ((\Theta^{(l)})^T \delta^{(l+1)})\ \cdot a^{(l)}\ \cdot (1 - a^{(l)}))
$$
   ​	delta values of layer $l$ are calculated by multiplying the delta values of the next layer with the theta matrix of layer $l$. Element-wise multiply that with the derivative of the activation function $g$ with input values of $z^{(l)}$.

The delta values are the partial derivative of the cost function with respect to the weighted sum of inputs, $z^{(l)}_j$.
$$
\delta^{(l)}_j = \frac{\part}{\part z^{(l)}_j}\text{cost(i)}
$$
(for $j\ge0$), where
$$
\text{cost}(i) = y^{(i)}log(h_\Theta(x^{(i)}))+(1-y^{(i)})log(h_\Theta(x^{(i)}))
$$
They are a measure of how much the neural networks weights need to be changed so that these intermediate values of the computation are changed, so that the final output and the overall cost is affected.

5. $\Delta^{(l)}_{i,j} := \Delta^{l}_{i,j} +a^{(l)}_j\delta^{l+1}_i$ with vectorization, $\Delta^{(l)} := \Delta^{(l)} + \delta^{(l+1)}(a^{(l)})^T$ 

   Update the matrix of $\Delta$

   When $j \neq 1$
   $$
   D^{(l)}_{i,j} := \frac{1}{m}(\Delta_{i,j}^{(l)} + \lambda\Theta^{(l)}_{i,j})
   $$
   when $j=0$ ( Bias term )
   $$
   D^{(l)}_{i,j} := \frac{1}{m}(\Delta_{i,j}^{(l)} )
   $$
   The matrix of $\Delta$ is an accumulator to add values so it can be used to compute the partial derivative.

$$
\frac \partial {\partial \Theta_{i,j}^{(l)}} J(\Theta) = D^{(l)}_{i,j}
$$

The cost function $J(\Theta)$ is non-convex, so it is possible that gradient descent or a more advanced optimization algorithm finds the local minima instead of the global minima, but this is not a huge problem since the cost function will still be relatively low. 

**Verifying Gradients**

The derivative of the cost function can be approximated with
$$
\dfrac{\partial}{\partial\Theta}J(\Theta) \approx \dfrac{J(\Theta + \epsilon) - J(\Theta - \epsilon)}{2\epsilon}
$$
where $\epsilon=10^{-4}$ or some small value.

With multiple parameters, the partial derivative with respect to $\Theta_j$ can be approximated with
$$
\dfrac{\partial}{\partial\Theta_j}J(\Theta) \approx \dfrac{J(\Theta_1, \dots, \Theta_j + \epsilon, \dots, \Theta_n) - J(\Theta_1, \dots, \Theta_j - \epsilon, \dots, \Theta_n)}{2\epsilon}
$$
The gradients computed by the $\delta^{(l)}_j$ can be checked using this approximation.

**Random Initialization**

Initial values of the weights $\Theta$ cannot be set to all zeros, since this mean the activation $a^{(l)}_j$ units will have the same value and the delta values $\delta^{(l)}_j$ will be the same. Causing the weights to update by the same value repeatedly.

Randomly initial each weight $\Theta^{(l)}_{i,j}$ to avoid this situation.