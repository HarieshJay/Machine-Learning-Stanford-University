## Model Representation

An artificial neural network consists of the input layer, hidden layer and the output layer. 

Activations units are represented as $a^{(j)}_i$, where $i$ is the unit in layer $j$. $\Theta^{(j)}$ is the matrix of weights controlling function mapping from layer $j$ to layer $j+1$. For example, in $\Theta^{(1)}_{10}$, the superscript $1$ represents the first layer, the $1$ in the $10 $ value represents activation unit 1 ($a^{(2)}_1$), and the $0$ shows that its from $x_0$, which is the bias term.
$$
\begin{bmatrix}x_0 \newline x_1 \newline x_2 \newline x_3\end{bmatrix}\rightarrow\begin{bmatrix}a_1^{(2)} \newline a_2^{(2)} \newline a_3^{(2)} \newline \end{bmatrix}\rightarrow h_\theta(x)
$$
In this neural network there is only one hidden layer. The activation units are represented as.
$$
\begin{align*} a_1^{(2)} = g(\Theta_{10}^{(1)}x_0 + \Theta_{11}^{(1)}x_1 + \Theta_{12}^{(1)}x_2 + \Theta_{13}^{(1)}x_3) \newline a_2^{(2)} = g(\Theta_{20}^{(1)}x_0 + \Theta_{21}^{(1)}x_1 + \Theta_{22}^{(1)}x_2 + \Theta_{23}^{(1)}x_3) \newline a_3^{(2)} = g(\Theta_{30}^{(1)}x_0 + \Theta_{31}^{(1)}x_1 + \Theta_{32}^{(1)}x_2 + \Theta_{33}^{(1)}x_3) \newline h_\Theta(x) = a_1^{(3)} = g(\Theta_{10}^{(2)}a_0^{(2)} + \Theta_{11}^{(2)}a_1^{(2)} + \Theta_{12}^{(2)}a_2^{(2)} + \Theta_{13}^{(2)}a_3^{(2)}) \newline \end{align*}
$$
$g(x)$ is the sigmoid activation function, $\frac{1}{1 + e^{-\Theta^Tx}}$.

If the network has $s_j$ units in layer $j$ and $s_{j+1}$ units in layer $j+1$, then $\Theta^{(j)}$ will have the dimensions $s_{j+1} \times (s_j + 1)$. The $+1$ comes from the bias node.

Define $z_k^{(j)}$ as the parameters inside the $k$th activation unit for the $j$th layer.
$$
\begin{align*}a_1^{(2)} = g(z_1^{(2)}) \newline a_2^{(2)} = g(z_2^{(2)}) \newline a_3^{(2)} = g(z_3^{(2)}) \newline \end{align*}
$$
For layer $2$ and node $k$, 
$$

z_k^{(2)} = \Theta_{k,0}^{(1)}x_0 + \Theta_{k,1}^{(1)}x_1 + \cdots + \Theta_{k,n}^{(1)}x_n
$$
Now in vector notation, the parameters can be calculated as
$$
z^{(j)} = \Theta^{(j-1)}a^{(j-1)}
$$
and the activations can be computed with this value as their parameters
$$
a^{(j)} = g(z^{j})
$$
The last step of computing the output is exactly the same as logistic regress. Adding multiple intermediate layers in the neural network allows for more complex non-linear hypotheses. Rather than feeding features into logistic regression, the neural network learns it's own features that are used in logistic regression.

**Multiclass Classification**

Create a separate output node for each category, where a 1 represents if it fits in that category, and set the output to be a vector of those categories. If the first index represent the class of motorcycle, then $[1,0,0,0]$ would be considered a motorcycle. Likewise, the response variable $y$ will also have to be of this form.