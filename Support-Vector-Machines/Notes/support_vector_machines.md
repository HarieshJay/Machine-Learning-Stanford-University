## Cost Function

$$
\mbox{min}_\theta\ C\left[\sum_{i=1}^{m}y^{(i)}\text{cost}_1(\theta^Tx^{(i)}) + (1-y^{(i)}) \text{cost}_0(\theta^Tx^{(i)})\right]+\frac{\lambda}{2}\sum_{j=1}^n\theta^2_j
$$

Approximates the logistic regression cost function with a piecewise linear function. For $y=1$, the $cost(\theta^Tx^{(i)})$ function returns $0$ when $\theta^Tx^{(i)}$ is greater or equal to $1$ and a positive value on a linear function the lower it is from $1$.

Due to notation, the $\frac{1}{m}$ has been omitted from both terms and $\lambda$ on the regularization has been replaced with $C$ on the cost term. A small $C$ value corresponds to a large $\lambda$ value. Large $C$ value corresponds to lower bias and high variance, a small $C$ value corresponds to higher bias and low variance.

To minimize the cost function, then $\theta$ values must be chosen such that
$$
\text{when } y=1, \theta^Tx \ge 1 \\\text{when } y=1, \theta^Tx \lt -1
$$
because only then is the cost function equal to zero.

Support vector machines is a large margin classifier, so it creates a decision boundary with the greatest separation between the classes. This makes it more robust, but makes it susceptible to outliers causing the decision boundary to vary greatly to fit the outlier. This usually occurs when $C$ is very large.

## Hypothesis

Rather than predict the probability of class $1$, SVM classifies directly.
$$
h_\theta(x) = 
\left\{
\begin{array}{ll}
      1 & \theta^Tx \ge 0 \\
      0 & \theta^Tx \lt 0 \\
\end{array} 
\right.
$$
For multi-class classification, use the one vs all method to train $K$ SVMs, where $K$ is the number of classes. Predict the class with the largest $(\theta^{(i)})^Tx$. 

## Kernels

Create landmarks, $l^{(i)}$, in the feature space, then add the $f_i$ to the features where
$$
f_i = similarity(x, l^{(i)})
$$
one representation of a similarity function is the Gaussian kernel
$$
exp(-\frac{||x-l^{(i)}||}{2\sigma^2})
$$
which gives a value of 1 when $x$ is near the landmark $l^{(i)}$ and a value near $0$ when it is far.

A large $\sigma^2$ corresponds to higher bias and lower variance since features vary more smoothly, and a smaller $\sigma^2$ means lower bias and higher variance.

A kernel can be created for every training observation, so that each observation becomes a feature in the model. SVM goes well with kernels due to the computational tricks associated with SVM.

Linear kernel means standard linear classifier with no kernels, $\theta^Tx \ge 0$, can be used when there are already many features for a small number of observations. Use kernels if the training set is large and the decision boundary is complex.

When using the Gaussian kernel, perform feature scaling, otherwise features with larger values will be dominant and have more impact.

To be a valid kernel the function needs to satisfy Mercer's Theorem making sure that that SVM packages optimizations run correctly and do not diverge.

## Choice of Classifier

If the number of features is large relative to the training set size, use logistic regression or SVM with a linear kernel. There is not enough data to fit a complicated decision boundary.

If the number of features is small and the training set size is intermediate, use SVM with the Gaussian Kernel.

If the number of features is small and the training set size is very large, then use logistic regression or SVM with a linear kernel, since this will be hard to compute.

Neural networks will work well for most of these settings, but may be slower to train.