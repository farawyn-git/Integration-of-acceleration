import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz

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
f = 160

# Zeit Daten
start = 0
end = 0.015
samplerate = 100000
step = 1 / samplerate
t = np.arange(start, end, step)


##### Berechneung #####

# Kreisfrequenz:
w = 2 * np.pi * f

# Beschleunigung (nach Annahmen:
a = A * np.sin(w * t)

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

# PLot object:
fig, (ax0, ax1, ax2) = plt.subplots(3, 1)

# plot Data:t
ax0.plot(t, a)
ax1.plot(t, v / 1000)
ax1.plot(t[0:-1] + step, v_int / 1000)
ax2.plot(t, s / 1000)
ax2.plot(t[0:-2] + 2*step, s_int / 1000)

# Plot Labels
plt.xlabel('Time [s]')
ax0.set_ylabel('Beschleunigung [m/s²]')
ax1.set_ylabel('Gschwindigkeit [mm/s]')
ax2.set_ylabel('Weg [m]')

plt.show()


