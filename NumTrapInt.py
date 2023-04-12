import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from scipy import signal
from scipy.fft import rfft, rfftfreq

'''
Weg-Zeit Gesetze:
s(t) = -(A/w²) * sin(w * t) + 1
v(t) = -(A/w) * cos(w * t) + 1 
a(t) = A * sin(w * t)

Kreisfrequenz w:
w = 2 * Pi * f 
'''

##### Annahmen #####

# Weg Amplitude in m/s²
A = 9.81

# Frequenz in Hz
f = 50

# Zeit Daten
start = 0
end = 1
samplerate = 100000
step = 1 / samplerate
t = np.arange(start, end, step)

# Kreisfrequenz:
w = 2 * np.pi * f


##### Berechneung #####

# Beschleunigung (exemplarisch):
a = A * np.sin(w * t)

# FFt
N = samplerate * end
yf = rfft(a)
xf = rfftfreq(N, 1 / samplerate)


# Butterworth Filter
# sos = signal.butter(2, 5, 'hp', fs=samplerate, output='sos')
# a = signal.sosfilt(sos, a)

# Geschwindigkeit (analytisch):
fv = -1 * (A / w)
v = fv * np.cos(w*t)

# Geschwindigkeit (numerisch integriert):
v_int = cumtrapz(a, t) + fv

# Weg (analytisch):
s = -1 * (A / w**2) * np.sin(w*t)

# Weg (numerisch integriert):
s_int = cumtrapz(v_int, t[0:-1])


##### Plotten #####

# Plot object:
fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1)

# plot Data:t
ax0.plot(t, a)
ax0.set_xlabel('Time [s]')
ax1.plot(t, v / 1000)
ax1.plot(t[0:-1], v_int / 1000)
ax1.set_xlabel('Time [s]')
ax2.plot(t, s / 1000)
ax2.plot(t[0:-2] + 2*step, s_int / 1000)
ax2.set_xlabel('Time [s]')
ax3.plot(xf, np.abs(yf))
ax3.axis(xmin=0, xmax=200)
ax3.set_xlabel('Frequenz [Hz]')

# Plot Labels
ax0.set_ylabel('Beschleunigung [m/s²]')
ax1.set_ylabel('Gschwindigkeit [mm/s]')
ax2.set_ylabel('Weg [m]')
ax3.set_ylabel('FFT')

plt.show()


