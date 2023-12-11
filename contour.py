# Adjusting the range for weights and biases around the trained parameters
# Since the weight is small, we choose a small range around it to see the variation in cost
w_range = np.linspace(w - 0.005, w + 0.005, 100).flatten()
b_range = np.linspace(b - 0.5, b + 0.5, 100).flatten()

# Recreate a grid of w and b values
W, B = np.meshgrid(w_range, b_range)

# Recompute the cost for each combination of w and b on the grid
Cost = np.zeros(W.shape)

for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        Cost[i, j] = cost_function(X, y, W[i, j], B[i, j])

# Create the contour plot again with adjusted ranges
plt.figure(figsize=(10, 8))
cp = plt.contour(W, B, Cost, levels=50, cmap='viridis')
plt.colorbar(cp)

# Label the axes and the plot
plt.title('Logistic Regression Cost Function Contour')
plt.xlabel('Weight (w)')
plt.ylabel('Bias (b)')
plt.plot(w, b, 'ro')  # Mark the trained weight and bias

plt.show()
