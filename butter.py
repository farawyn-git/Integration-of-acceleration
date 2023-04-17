import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy import signal
import pandas as pd

### Signal creation
start = 0
n_samples = 5000
fs = 10240
dx = 1 / fs

x = np.linspace(start, n_samples * dx, n_samples, endpoint=True)

a1 = 2
f1 = 10
a2 = 0.2
f2 = 1500
a3 = 0.1
f3 = 2000

y = a1 * np.sin(2*np.pi*f1*x) + a2 * np.sin(2*np.pi*f2*x) + a3 * np.sin(2*np.pi*f3*x)
print(np.size(y))

print(type(x))

# arr = np.concatenate((x, y), axis = 0)
# df = pd.DataFrame(arr)
# print(arr)


### Hanning  Window
han_win = signal.windows.hann(n_samples, sym=False)

### FFT
yf = rfft(y)
yf = yf * 2 / n_samples
xf = rfftfreq(n_samples, dx)


### Butterworth Filter
sos = signal.butter(6, 100, 'lp', fs=fs, output='sos')
filtered = signal.sosfilt(sos, y)

### FFT
yf_filtered = rfft(filtered)
yf_filtered = yf_filtered * 2 / n_samples
xf_filtered = rfftfreq(n_samples, dx)


### Plotting
fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1)
plt.style.context('ggplot')

ax0.set_title('Input Signal')
ax0.plot(x, y)
ax0.axis(xmin=0, xmax=0.5)

ax1.set_title('FFT Input Signal')
ax1.plot(xf, np.abs(yf))
ax1.axis(xmin=0, xmax=fs//2)

ax2.set_title('Filtered Signal')
ax2.plot(x, filtered)
ax2.axis(xmin=0, xmax=0.5)

ax3.set_title('FFT Filtered Signal')
ax3.plot(xf_filtered, np.abs(yf_filtered))
ax3.axis(xmin=0, xmax=fs//2)

plt.subplots_adjust(left=0.06,
                    bottom=0.06,
                    right=0.97,
                    top=0.95,
                    wspace=0.3,
                    hspace=0.3)

plt.show()
