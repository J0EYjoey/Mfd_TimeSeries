import numpy as np
import matplotlib.pyplot as plt

def simulate_time_series(T, d):
    # Generate time points
    i = np.arange(1, T + 1)
    
    # Initialize the output matrix
    X = np.zeros((T, d))
    
    # For each row i, generate d independent realizations
    for t in range(T):
        # Generate uniform random variables a_i on [-0.5, 0.5]
        a_i = np.random.uniform(-0.5, 0.5, d)
        
        # Generate Rademacher variables b_i (values -1 or 1 with equal probability)
        b_i = np.random.choice([-1, 1], d)
        
        # Calculate the time series for this row
        X[t, :] = (i[t]/T)**0.5 * a_i + (1 - i[t]/T)**0.5 * b_i
    
    return X

# Set the length of the time series and number of columns
T = 1000
d = 5  # number of columns

# Simulate the time series matrix
X = simulate_time_series(T, d)

# Plot the first column of the time series
plt.figure(figsize=(12, 6))
plt.plot(X[:, 0])
plt.title('Simulated Time Series (First Column)')
plt.xlabel('Time')
plt.ylabel('Value')
plt.grid(True)
plt.show()

# Print some basic statistics
print(f"Matrix shape: {X.shape}")
print(f"Mean: {np.mean(X):.4f}")
print(f"Standard Deviation: {np.std(X):.4f}")
print(f"Minimum: {np.min(X):.4f}")
print(f"Maximum: {np.max(X):.4f}")

# Plot all columns to show they are different
plt.figure(figsize=(12, 6))
for j in range(d):
    plt.plot(X[:, j], label=f'Column {j+1}')
plt.title('All Columns of Simulated Time Series')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show() 