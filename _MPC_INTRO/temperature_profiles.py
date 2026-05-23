# in this file we generate the temperature profile for the MPC controller
# first, the real temperature profile is generated, then we add some noise to it to simulate the real world conditions

import numpy as np
import matplotlib.pyplot as plt
# generate the real temperature profile
def generate_temperature_profile(time_steps, amplitude=5, offset=20):
    # we will use a simple sinusoidal function to generate the temperature profile
    # the temperature will vary between 20 and 30 degrees Celsius
    time = np.arange(time_steps)
    temperature_profile = offset + amplitude * np.sin(2 * np.pi * time / 24)
    return temperature_profile


# add some noise to the temperature profile
def add_noise(temperature_profile, noise_level):
    rng = np.random.default_rng(seed=42)
    noise = rng.normal(0, noise_level, size=temperature_profile.shape)
    noisy_temperature_profile = temperature_profile + noise
    return noisy_temperature_profile

# Example usage
if __name__ == "__main__":
    time_steps = 100
    temperature_profile = generate_temperature_profile(time_steps)
    noisy_temperature_profile = add_noise(temperature_profile, noise_level=1.0)

    # Plot the real and noisy temperature profiles
    plt.figure(figsize=(10, 5))
    plt.plot(temperature_profile, label='Real Temperature Profile')
    plt.plot(noisy_temperature_profile, label='Noisy Temperature Profile', alpha=0.7)
    plt.xlabel('Time Steps')
    plt.ylabel('Temperature (°C)')
    plt.title('Temperature Profiles')
    plt.legend()
    plt.grid()
    plt.show()