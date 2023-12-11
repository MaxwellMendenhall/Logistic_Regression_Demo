# Trading Algorithm Machine Learning

- Needed to optimize the trading algorithm without spending too much time on number crunching and when I am not at the computer. Machine learning can do this easily; a few models can help, but the first iteration of this model would be a logistic regression model.
- Could use [sklearn.linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), but I wanted to build this from the ground up so I know how it works and I can expand it in the future. To understand the logistic regression model and how it functions, you need two main algorithms/functions: Cost Function and Gradient Descent. A simple explaination of these algorithms/functions -> Cost Function is the heart of the logistic regression model, Gradient Descent is the algorithm used to optimize the Cost Function (find the best w and b values that work best), and derivatives are used in gradient descent.

### Cost Function: Measure of losses over a training set
$$J(\vec{w},b)=\frac{1}{m}\sum_{i=1}^{m}(\mathrm{f}\vec{w},b(\vec{x}^{i}),y^{i})$$
#### Lost function(function inside the cost function): Measue of differnce of a single example to its target value 

$$
L(\mathrm{f}\vec{w},b(\vec{x}^{i}),y^{i}) = 
\begin{cases}
    -y^i*log(\mathrm{f}\vec{w},b(\vec{x}^{i}) & \text{if } y^i = 1, \\
    -(1-y^i)log(1-\mathrm{f}\vec{w},b(\vec{x}^{i}) & \text{if } y^i = 0.
\end{cases}
$$

### Gradient Descent: Algorithm used to optimize the cost function and find best value for w and b, where a is learning rate.
$$w_j = w_j - a[\frac{1}{m}\sum_{i=1}^{m}(\mathrm{f}\vec{w},b(\vec{x}^{i})-y^i)x_j^i]$$

$$b = b - a[\frac{1}{m}\sum_{i=1}^{m}(\mathrm{f}\vec{w},b(\vec{x}^i)-y^i)]$$

- Now that we have the functions to produce the cost function we need to use regluariztion so the cost function does not overfit or underfit the data.
- Regularization is reducing features sizes so the data does not overfit.

### Regularization Cost Function: This is just the cost function with one additional element

$$
J_{reg}(\vec{w},b) = J(\vec{w},b) + \frac{\lambda}{2m}\sum_{i=1}^{m}w_j^2 = 
\begin{cases}
    \text{if } \lambda=0 & \text{overfit}, \\
    \text{if } \lambda=\lambda \to \infty & \text{underfit}
\end{cases}
$$

### Regularization Gradient Descent: This is just the gradient descent algorithm with one additional element
- Formula for w changes while the formula for b stays the same.

$$w_j = w_j - a[\frac{1}{m}\sum_{i=1}^{m}(\mathrm{f}\vec{w},b(\vec{x}^{i})-y^i)x_j^i] + \frac{\lambda}{m}w_j$$

$$b = b - a[\frac{1}{m}\sum_{i=1}^{m}(\mathrm{f}\vec{w},b(\vec{x}^i)-y^i)]$$

# Getting Angle Data
- The data that will be fed into this machine learning model will be angles can getting the probability they win or do not win.
- In order to get angles of a line you need to find the slope of point of interest. To get slope you need t-1 and t+1 where t is the value of intersection. Then you would need to plug the two points into the slope formula to get it:
$$m = \frac{y_{2} - y_{1}}{x_{2} - x_{1}}$$
- After finding slope to get angle of intersection you can use this formula:
$$\theta = arctan(|\frac{m_{1} - m_{2}}{1 + m_{1}m_{2}}|)$$

$$\theta \text{ in degrees} = \theta * \frac{180}{\pi}$$
