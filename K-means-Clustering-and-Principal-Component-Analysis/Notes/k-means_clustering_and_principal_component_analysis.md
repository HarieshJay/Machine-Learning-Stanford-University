## K-Means Algorithm

Given a number of clusters $K$ and a training set $\{x^{(1)}, x^{(2)}, ... x^{(n)}\}$, randomly initialize $K$ cluster centroids $\mu_1,\mu_2...\mu_n$, then repeatedly assign each point to the cluster centroid that has the closest Euclidean distance in the feature space then assign new centroids to the mean of the points in each cluster.

Randomly initialize cluster centroids by choosing $K$ observations in the training set. This may lead to converging at local optima, so running K-Means multiple times may be necessary to find the global optima. Running multiple random initializations will have an effect when $K$ is small, but will not have a large impact when there are many clusters.

**Optimization Objective - Distortion**

The cost function can be represented as the average squared distance between the points and the cluster they have been assigned to.
$$
J(c^{(1)}, ..., c^{(1)},\mu_1,...,\mu_k) = \frac{1}{m}\sum^m_{i=1}||x^{(i)}-\mu_{c^{(i)}}||^2
$$

## Dimensionality Reduction

If 2 features have a high correlation, they can be reduced to a new feature that plots the location of the observations on the line mapping their relationship. This will reduce the feature space from 2 dimensions to 1 dimension.

Similarly, if the feature space is in 3 dimensions, the points can be projected onto a plane and represented in 2D as their coordinates on the plane.

**Principal Component Analysis**

Finds a hyperplane in the feature space to project the observations such that the projection error to that hyperplane is minimized. Project the observations to the linear subspace spanned by a set of $K$ vectors, where $K$ is the new dimension.

Before applying PCA, perform mean normalization by subtracting each feature by it's mean, $x_j - \mu_j$, where the mean is calculated with $\mu_j = \frac{1}{m}\sum^m_{i=1}x_j^{(i)}$, so that each feature will now have $0$ mean. Use feature scaling so they have a comparable range of values by dividing each feature by it's range or standard deviation.

Principal Component Analysis Algorithm:

compute the covariance matrix $\Sigma$
$$
\Sigma = \frac{1}{m}(X\times X^T)
$$
compute the eigenvectors of the matrix $\Sigma$. To reduce to $K$ dimensions, take the first $K$ columns of this matrix, transpose it and multiply it by the observations $X$, so the new $K$ dimensional feature space will be 
$$
Z= U_{reduced}^T \times X
$$
To reconstruct the original feature space
$$
X_{approx} = U_{reduced} \times Z
$$
PCA minimizes the average squared projection error where $x^{(i)}_{approx}$ is the point on the projected plane.
$$
\frac{1}{m}\sum^m_{i=1}||x^{(i)}-x^{(i)}_{approx}||^2
$$
The total variation in the data is
$$
\frac{1}{m}\sum^m_{i=1}||x^{(i)}||^2
$$
which is the measure of how far away are the training observations from the origin.

Choose the smallest value of $K$ such that the ratio between these is less than $0.01$
$$
{\frac{\frac{1}{m}\sum^m_{i=1}||x^{(i)}-x^{(i)}_{approx}||^2}{\frac{1}{m}\sum^m_{i=1}||x^{(i)}||^2}} \le 0.01
$$
A ratio of $0.01$ shows that $99\%$ of the variance has been retained, $95\%$ is also acceptable.

Mapping from $x^{(i)} \rightarrow z^{(i)}$ should be defined by running PCA only on the training set. Apply the mapping from the training set to the cross validation and test sets.

Before implementing PCA, try using the original data without PCA, and only use when needed.