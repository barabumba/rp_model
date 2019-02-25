import numpy as np
import matplotlib.pyplot as plt

N = 2**14
M = 2**5
x = np.random.normal(size=N)
fft = np.fft.fft(x)
frequencies = np.fft.fftfreq(N, 1./N)
_f = lambda x: 1./(1+(np.pi*x)**2)
psd_samples = np.vectorize(_f)(frequencies/M)
transfer_function_samples = psd_samples / (psd_samples + 1)
# plt.plot(frequencies, transfer_function_samples)
# plt.plot(frequencies, np.log(1+psd_samples))

y = np.real(np.fft.ifft(fft*np.sqrt(transfer_function_samples)))
fft2 = np.real(fft*fft.conjugate()*transfer_function_samples)

print np.var(y), np.sum(y**2)/N, np.sum(fft2)/N/N
print 1./(2*np.sqrt(2))*M/N * 2
# plt.plot(y)

l = 1000
psd = 0
d1 = 0
d2 = 0
for i in range(l):
    x = np.random.normal(size=N)
    fft = np.fft.fft(x)
    fft2 = np.real(fft * fft.conjugate() * transfer_function_samples)
    psd += fft2
    y = np.real(np.fft.ifft(fft * np.sqrt(transfer_function_samples)))
    d1 += np.var(y)
    d2 += np.sum(fft2)/N/N
d1 /= l
d2 /= l
psd /= l
print d1, d2
plt.plot(frequencies, psd/N)
plt.plot(frequencies, transfer_function_samples)

plt.grid(1)
plt.show()