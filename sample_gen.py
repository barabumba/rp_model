import numpy as np
import matplotlib.pyplot as plt


class RandomSignalGenerator(object):
    def __init__(self, normalized_psd_fun, size, bandwidth):
        self.normalized_psd_fun = np.vectorize(lambda x: normalized_psd_fun(x))
        self.fft_coefficients = None
        self.n = size
        self.m = bandwidth
        self.frequencies = np.fft.fftfreq(size, 1./size)
        self.normalized_psd_samples = self.normalized_psd_fun(self.frequencies/self.m)

    def gen(self):
        x = np.random.normal(size=self.n)
        fft_x = np.fft.fft(x)
        self.fft_coefficients = fft_x * np.sqrt(self.normalized_psd_samples)

    def get_fft_coefficients(self):
        return self.fft_coefficients

    def get_samples(self):
        return np.real(np.fft.ifft(self.fft_coefficients))


class PSDAveraging(object):
    def __init__(self, random_signal_generator, size=1000):
        self.rsg = random_signal_generator
        self.n = size
        self.psd_average = None

    def average(self):
        psd_average = 0
        for i in range(self.n):
            self.rsg.gen()
            psd_average += np.real(self.rsg.fft_coefficients*self.rsg.fft_coefficients.conj()) / self.n
        self.psd_average = psd_average

    def plot(self):
        new_fig = plt.figure().add_subplot(111)
        new_fig.plot(self.rsg.frequencies, self.psd_average / self.rsg.n)
        new_fig.plot(self.rsg.frequencies, self.rsg.normalized_psd_samples)
        new_fig.grid()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    _f = lambda x: 1./(1+(np.pi*x)**2)
    rsg = RandomSignalGenerator(_f, 2**10, 2**6)
    rsg.gen()
    plt.plot(rsg.get_samples())
    test = PSDAveraging(rsg, size=2**30)
    test.average()
    test.plot()
    plt.show()