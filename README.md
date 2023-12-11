# Trading Algorithm using Machine Learning
## Introduction
This project aims to optimize a trading algorithm leveraging machine learning techniques. The primary goal is to automate the process and enhance performance without the need for constant manual intervention. The initial model employed is based on logistic regression.
## Implementation Details

### Logistic Regression Model
nstead of using pre-built libraries like sklearn.linear_model.LogisticRegression, this project involves building the logistic regression model from scratch. This approach aids in understanding the fundamental mechanisms and allows for future expansions. The model is built around two key components:
- Cost Function: The core of the logistic regression model.
- Gradient Descent: An optimization algorithm used to refine the Cost Function.

#### Cost Function: Measure of losses over a training set
$$J(\vec{w},b)=\frac{1}{m}\sum_{i=1}^{m}(\mathrm{f}\vec{w},b(\vec{x}^{i}),y^{i})$$

#### Lost function(function inside the cost function): Measue of differnce of a single example to its target value 

$$
L(\mathrm{f}\vec{w},b(\vec{x}^{i}),y^{i}) = 
\begin{cases}
    -y^i*log(\mathrm{f}\vec{w},b(\vec{x}^{i}) & \text{if } y^i = 1, \\
    -(1-y^i)log(1-\mathrm{f}\vec{w},b(\vec{x}^{i}) & \text{if } y^i = 0.
\end{cases}
$$

#### Gradient Descent: Algorithm used to optimize the cost function and find best value for w and b, where a is learning rate.
$$w_j = w_j - a[\frac{1}{m}\sum_{i=1}^{m}(\mathrm{f}\vec{w},b(\vec{x}^{i})-y^i)x_j^i]$$

$$b = b - a[\frac{1}{m}\sum_{i=1}^{m}(\mathrm{f}\vec{w},b(\vec{x}^i)-y^i)]$$

### Regularization
To prevent overfitting or underfitting, regularization is applied. It involves reducing feature sizes.
#### Regularization Cost Function: This is just the cost function with one additional element

$$
J_{reg}(\vec{w},b) = J(\vec{w},b) + \frac{\lambda}{2m}\sum_{i=1}^{m}w_j^2 = 
\begin{cases}
    \text{if } \lambda=0 & \text{overfit}, \\
    \text{if } \lambda=\lambda \to \infty & \text{underfit}
\end{cases}
$$

#### Regularization Gradient Descent: This is just the gradient descent algorithm with one additional element
- Formula for w changes while the formula for b stays the same.

$$w_j = w_j - a[\frac{1}{m}\sum_{i=1}^{m}(\mathrm{f}\vec{w},b(\vec{x}^{i})-y^i)x_j^i] + \frac{\lambda}{m}w_j$$

$$b = b - a[\frac{1}{m}\sum_{i=1}^{m}(\mathrm{f}\vec{w},b(\vec{x}^i)-y^i)]$$

## Angle Data for the Model
The model uses angles as input data to predict probabilities.

### Calculating Slope
To determine the slope at a point of interest:
$$m = \frac{y_{2} - y_{1}}{x_{2} - x_{1}}$$

### Calculating Angle of Intersection
Once the slope is obtained, the angle of intersection is calculated using:
$$\theta = arctan(|\frac{m_{1} - m_{2}}{1 + m_{1}m_{2}}|)$$

$$\theta \text{ in degrees} = \theta * \frac{180}{\pi}$$
