import numpy as np

# Define the input data
x = np.array([1, 3.5, 6])
y = np.array([4, 5.5, 9])

# Define the learning rate, number of iterations, and momentum factor
learning_rate = 0.01
num_iterations = 1000
momentum_factor = 0.9

# Initialize the parameters
theta = np.random.rand(2, 1)
velocity = np.zeros_like(theta)

# Define the cost function
def cost_function(theta, x, y):
    m = len(y)
    predictions = x.dot(theta)
    cost = (1/(2*m)) * np.sum(np.square(predictions-y))
    return cost

# Define the gradient function
def gradient_function(theta, x, y):
    m = len(y)
    predictions = x.dot(theta)
    gradient = (1/m) * x.T.dot(predictions-y)
    return gradient

# Vanilla gradient descent
for i in range(num_iterations):
    
    gradient = gradient_function(theta, x, y)
    theta = theta - learning_rate * gradient

    # Print the cost function every 100 iterations
    if i % 100 == 0:
        cost = cost_function(theta, x, y)
        print(f"Vanilla GD - Iteration {i}: Cost={cost}")

# Momentum-based gradient descent
for i in range(num_iterations):
    gradient = gradient_function(theta, x, y)
    velocity = momentum_factor * velocity - learning_rate * gradient
    theta = theta + velocity

    # Print the cost function every 100 iterations
    if i % 100 == 0:
        cost = cost_function(theta, x, y)
        print(f"Momentum GD - Iteration {i}: Cost={cost}")

# Print the final parameters
print("Final Parameters:")
print(theta)
