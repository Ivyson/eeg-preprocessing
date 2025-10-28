import numpy as np
import matplotlib.pyplot as plt

# Define the noisy unit step function
def noisy_unit_step(t, noise_level=0.05):
    """
    Returns a unit step function with added Gaussian noise.
    
    Parameters:
        t (array-like or float): Input time(s)
        noise_level (float): Standard deviation of the Gaussian noise to add
        
    Returns:
        np.ndarray: Noisy unit step values
    """
    # Convert input to numpy array for vectorized operations
    t = np.array(t)
    
    # Standard unit step function
    step = np.where(t >= 0, 1.0, 0.0)
    
    # Add Gaussian noise
    noise = np.random.normal(loc=0.0, scale=noise_level, size=t.shape)
    
    # Combine step and noise
    noisy_step = step + noise
    
    # Ensure values stay within [0, 1] for realistic step approximation
    noisy_step = np.clip(noisy_step, 0, 1)
    
    return noisy_step

# Example usage
if __name__ == "__main__":
    # Define time range
    t = np.linspace(-2, 2, 400)
    
    # Generate noisy unit step
    y = noisy_unit_step(t, noise_level=0.1)
    
    # Plot results
    plt.figure(figsize=(8, 4))
    plt.plot(t, y, label="Noisy Unit Step", color='blue')
    plt.plot(t, np.where(t >= 0, 1, 0), '--', label="Ideal Unit Step", color='red')
    plt.xlabel("t")
    plt.ylabel("u(t)")
    plt.title("Noisy Unit Step Function")
    plt.legend()
    plt.grid(True)
    plt.show()