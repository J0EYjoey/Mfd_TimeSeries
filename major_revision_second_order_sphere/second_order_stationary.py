import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def generate_second_order_stationary(T):
    """
    Generate a time series that is second-order stationary but not strongly stationary.
    The process alternates between normal and uniform distributions while maintaining
    constant mean and variance.
    """
    # Initialize the time series
    X = np.zeros(T)
    
    # Parameters to maintain constant mean and variance
    mean = 0
    variance = 1
    
    # Generate the time series
    for t in range(T):
        if t % 2 == 0:  # Even time points: normal distribution
            X[t] = np.random.normal(mean, np.sqrt(variance))
        else:  # Odd time points: uniform distribution
            # For uniform distribution, we need to adjust the parameters
            # to maintain the same mean and variance
            a = mean - np.sqrt(3 * variance)
            b = mean + np.sqrt(3 * variance)
            X[t] = np.random.uniform(a, b)
    
    return X

# Set the length of the time series
T = 1000

# Generate the time series
X = generate_second_order_stationary(T)

# Plot the time series
plt.figure(figsize=(12, 6))
plt.plot(X)
plt.title('Second-Order Stationary but Not Strongly Stationary Time Series')
plt.xlabel('Time')
plt.ylabel('Value')
plt.grid(True)
plt.show()

# Plot histograms for even and odd time points to show different distributions
plt.figure(figsize=(12, 6))
plt.hist(X[::2], bins=30, alpha=0.5, label='Even time points (Normal)')
plt.hist(X[1::2], bins=30, alpha=0.5, label='Odd time points (Uniform)')
plt.title('Distribution of Values at Even vs Odd Time Points')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()

# Print statistics to verify second-order stationarity
print("Statistics for the entire series:")
print(f"Mean: {np.mean(X):.4f}")
print(f"Variance: {np.var(X):.4f}")

print("\nStatistics for even time points:")
print(f"Mean: {np.mean(X[::2]):.4f}")
print(f"Variance: {np.var(X[::2]):.4f}")

print("\nStatistics for odd time points:")
print(f"Mean: {np.mean(X[1::2]):.4f}")
print(f"Variance: {np.var(X[1::2]):.4f}")

# Plot autocorrelation to show stationarity in second moments
plt.figure(figsize=(12, 6))
plt.acorr(X, maxlags=50)
plt.title('Autocorrelation Function')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.grid(True)
plt.show() 