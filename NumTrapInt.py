import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from scipy.fft import rfft, rfftfreq
from scipy import signal
import pandas as pd

'''
Vergleich der analytischen Interation eines theoretischen Beschleunigungssignals 
mit dem gemessenen Geschwindigkeitssignals eines Laserinteferometers.

Vergleich der Integration eines gemessenen Beschleunigungssignals mit dem
mit dem gemessenen Geschwindigkeitssignals eines Laserinteferometers.

Hintergrundinformationen:

Weg-Zeit Gesetze:
s(t) = -(A/w²) * sin(w * t) + ct + c
v(t) = -(A/w) * cos(w * t) + C 
a(t) = A * sin(w * t)

Kreisfrequenz w:
w = 2 * Pi * f 
'''

'''
Integration analytisch:
Berechnung der Geschwindigkeit und des Weges mittels analytischer Verfahren.
'''

##### Annahmen #####

# Weg Amplitude in m/s²
A = 9.81

# Frequenz in Hz
f = 159.2

# Zeit Daten
start = 0
end_ana = 1
samplerate_ana = 100000
step = 1 / samplerate_ana
t = np.arange(start, end_ana, step)

# Kreisfrequenz:
w = 2 * np.pi * f


##### Berechnung der therorethischen Beschleunigung #####

# Beschleunigung (analytisch):
a = A * np.sin(w * t)

# FFT
N_ana = samplerate_ana * end_ana
a_norm_ana= a / 50000
yf_ana = rfft(a_norm_ana)
xf_ana = rfftfreq(N_ana, 1 / samplerate_ana)

# Geschwindigkeit (analytisch):
fv = -1 * (A / w)
v = fv * np.cos(w*t)

# Weg (analytisch):
s = -1 * (A / w**2) * np.sin(w*t)




'''
Integration numerisch:
Berechnung der Geschwindigkeit und es Wegs mit Hilfe des Trapezverfahrens.
'''

### Daten Einlesen
df = pd.read_csv('test1.csv', sep=';', header=None, names=['time', 'acc', 'speed'])

df = df.iloc[0:5996]
# 5996 = 1sec

print('Eingelesene Daten:')
print(df)


### Butterworth Filter
samplerate = 1 / df['time'].iloc[1]
# sos = signal.butter(2, 150, 'hp', fs=samplerate, output='sos')
# acc_fil = signal.sosfilt(sos, df['acc'])


### Kennwerte Beschleunigungssignal
mean_acc = np.mean(df['acc'])
min_acc = np.min(df['acc'])
max_acc = np.max(df['acc'])
amp_acc = (max_acc - min_acc) // 2

print('')
print('Kennwerte Beschleunigungsignal:')
print(f'fs:  {samplerate}')
print(f'Mean: {mean_acc}')
print(f'Min: {min_acc}')
print(f'Max: {max_acc}')
print(f'Amp: {amp_acc}')


### Kennwerte Geschwindigkeitssignal:
mean_vel = np.mean(df['speed'])
min_vel = np.min(df['speed'])
max_vel = np.max(df['speed'])
amp_vel = (max_vel - min_vel) // 2

print('')
print('Kennwerte Geschwindigkeitssignal:')
print(f'fs:  {samplerate}')
print(f'Mean: {mean_vel}')
print(f'Min: {min_vel}')
print(f'Max: {max_vel}')
print(f'Amp: {amp_vel}')


#### Geschwindigkeit (numerisch integriert):
acc = df['acc'] - np.mean(df['acc'])
# acc = acc_fil - np.mean(acc_fil)
# acc = df['acc']
first_value = df['time'].iloc[0]
ts = df['time'] - first_value
v_int = cumtrapz(acc, ts) * 1000
v_int = v_int - np.mean(v_int)

mean_v = np.mean(v_int)
min_v = np.min(v_int)
max_v = np.max(v_int)
amp_v = (max_v + np.abs(min_v)) // 2

print('')
print('Kennwerte Geschwindigkeit integriert:')
print(f'Mean: {mean_v}')
print(f'Min: {min_v}')
print(f'Max: {max_v}')
print(f'Amp: {amp_v}')


#### Weg (numerisch integriert):
s_int = cumtrapz(v_int, ts.iloc[:-1])


#### FFT gemessenes Beschlenigungssignal:
dx = ts.iloc[1]
a_num = acc.to_numpy()
# a_num = acc
N_num = a_num.size
yf_num = rfft(a_num)
xf_num = rfftfreq(N_num, dx)[0:N_num//2]

# amp_fft = signal.find_peaks(np.abs(yf_num), height=5)
# print(f'amp_fft: {amp_fft}')



'''
Plotten der Daten:
Plotten der Daten mit Matplotlib
'''

# Plot object erzeugen
fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1)

# 
x_min = 0
x_max = 1

# Plot Beschleunigung
ax0.plot(t, a)                                          # analytisch
ax0.plot(ts, df['acc'])                                 # numerisch
ax0.axis(xmin = x_min, xmax = x_max)
ax0.set_xlabel('Time [s]')
ax0.set_ylabel('Beschleunigung [m/s²]')

# Plot Geschwindigkeit
ax1.plot(t, v * 1000)                                   # analytisch
# ax1.plot(ts, df['speed'])                             # gemessen
ax1.plot(ts.iloc[:-1], v_int)                           # numerisch
ax1.axis(xmin = x_min, xmax = x_max)
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Gschwindigkeit [mm/s]')

# Plot Weg
# ax2.plot(t, s * 1000)                                 # analytisch
# ax2.plot(ts.iloc[:-2], s_int * 1000)                  # numerisch
ax2.axis(xmin = x_min, xmax = x_max)
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Weg [mm]')

# Plot FFT
ax3.plot(xf_ana, np.abs(yf_ana))
ax3.plot(xf_num, 2.0 / N_num * np.abs(yf_num[0:N_num//2]))
ax3.axis(xmin=0, xmax=200)
ax3.set_xlabel('Frequenz [Hz]')
ax3.set_ylabel('FFT [m/s²]')

plt.subplots_adjust(left=0.06,
                    bottom=0.06,
                    right=0.97,
                    top=0.98,
                    wspace=0.3,
                    hspace=0.3)

# plt.legend()

plt.show()

