# Milestone 1
# Step 1: Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.fftpack import fft

# Step 2: Duration 3 seconds with 12*1024 samples
samples = 12 * 1024
t = np.linspace(0, 3, samples)

# Step 3: Details of the song
pairs = 5  # pairs

F = np.array([3135, 583, 2800, 44, 1392])  # Frequencies of the left hand
f = np.array([523, 587, 659, 698, 784])  # Frequencies of the right hand
t_i = np.array([0, 0.6, 1.2, 1.8, 2.4])  # Starting times
T_i = np.array([1.5, 0.5, 0.5, 0.5, 0.5])  # Durations each press

# Step 4: Signal Equation
x = np.zeros_like(t)
for i in range(pairs):
    x += (np.sin(2 * np.pi * F[i] * t) + np.sin(2 * np.pi * f[i] * t)) * (
            np.heaviside(t - t_i[i], 0.5) - np.heaviside(t - t_i[i] - T_i[i], 0.5))

# Playing the song
sd.play(x, 3 * 1024)
sd.wait()
# Step 5: Plotting
plt.plot(t, x)
plt.title("Original Song in time domain")
plt.show()

# Milestone 2
# Samples count
N = 3 * 1024
# Frequency axis range
fNoise = np.linspace(0, 512, int(N / 2))
# Converting to the frequency domain
x_f = fft(x)
x_f = 2 / N * np.abs(x_f[0:int(N / 2)])
plt.plot(fNoise, x_f)
plt.title("Original song in frequency domain")
plt.show()
# Noise generation from two random frequencies
fn1 = np.random.randint(0, 512)
fn2 = np.random.randint(0, 512)
print(f"real noise frequencies: {fn1} {fn2}")
#Noise signal
noise = np.sin(2 * np.pi * fn1 * t) + np.sin(2 * np.pi * fn2 * t)

# Step 4 Adding Noise
xn = x + noise
sd.play(xn, 3 * 1024)
sd.wait()
plt.plot(t, xn)
plt.title("Song after noise in time domain")
plt.show()

# Step 5 Converting the noise to the frequency domain
xn_f = fft(xn)
xn_f = 2 / N * np.abs(xn_f[0:int(N / 2)])
plt.plot(fNoise, xn_f)
plt.title("Song after noise in frequency domain")
plt.show()
# Step6 searching for the peak of the current noise frequency diagram and then saving the amplitude and the frequency of this noise
x_filtered = xn
for j in range(0, 2):
    maxvalue = 0
    f = 0
    for i in range(0, len(fNoise)):
        if round(xn_f[i]) > round(maxvalue):
            # print(f"Noise at frequency {round(fNoise[i])} has amplitude {xn_f[i]}")
            # Step8 filtering
            maxvalue = xn_f[i]
            f = fNoise[i]
	# we reset the frequency signal to find the other peak after removing the first one
    x_filtered -= np.sin(2 * np.pi * round(f) * t)
    xn_f = fft(x_filtered)
    xn_f = 2 / N * np.abs(xn_f[0:int(N / 2)])

# Playing after filtering and then getting the graphs
sd.play(x_filtered, 3 * 1024)
sd.wait()

x_filtered_f = fft(x_filtered)
x_filtered_f = 2 / N * np.abs(x_filtered_f[0:int(N / 2)])
plt.plot(t, x_filtered)
plt.title("Song after noise cancellation in time domain")
plt.show()
plt.plot(fNoise, x_filtered_f)
plt.title("Song after noise cancellation in frequency domain")
plt.show()
