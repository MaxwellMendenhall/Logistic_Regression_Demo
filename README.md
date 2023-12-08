# Trading Algorithm Machine Learning

- Needed to optimize the trading algorithm without spending too much time on number crunching and when I am not at the computer. Machine learning can do this easily; a few models can help, but the first iteration of this model would be a logistic regression model.
- Could use [sklearn.linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), but I wanted to build this from the ground up so I know how it works and I can expand it in the future. To understand the logistic regression model and how it functions, you need three main algorithms: Cost Function, Gradient Descent, and Gradient. The Cost Function is the heart of the logistic regression model, Gradient Descent is the algorithm used to optimize the Cost Function (find the best w and b values that work best), and Gradient is used within the Gradient Descent to find the derivative of the current iteration.
- 

### Cost Function: 
$$J(\vec{w},b)=\frac{1}{m}\sum_{i=1}^{m}(\mathrm{f}\vec{w},b(\vec{x}^{i}),y^{i})$$
