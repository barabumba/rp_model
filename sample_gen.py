import numpy as np
import matplotlib.pyplot as plt


class RandomSignalGenerator(object):
    def __init__(self, normalized_psd_fun, size, bandwidth):
        self.normalized_psd_fun = np.vectorize(lambda x: normalized_psd_fun(x))
        self.squared_fft_coefficients = None
        self.fft = None
        self.n = size
        self.m = bandwidth
        self.frequencies = np.fft.fftfreq(size, 1./size)
        self.normalized_psd_samples = self.normalized_psd_fun(self.frequencies/self.m)

    def gen(self):
        xi, n = np.random.normal(size=self.n), np.random.normal(size=self.n)
        fft_xi, fft_n = np.fft.fft(xi), np.fft.fft(n)
        fft_xi *= np.sqrt(self.normalized_psd_samples)
        fft = 0*fft_n + fft_xi
        self.fft = fft
        self.squared_fft_coefficients = np.real(fft * fft.conjugate())

    def get(self):
        return self.squared_fft_coefficients

    def get_sample(self):
        return np.real(np.fft.ifft(self.fft))


class RandomSignalGeneratorTest(object):
    def __init__(self, random_signal_generator, size=1000):
        self.rsg = random_signal_generator
        self.n = size
        self.psd_average = None
        self.autocorrelation = None

    def autocorrelation_function(self):
        sample = np.real(np.fft.ifft(self.rsg.fft))
        n = len(sample)
        partial_sample = sample[:int(0.95*n)]

        self.autocorrelation = np.correlate(sample, partial_sample) / np.var(partial_sample)
        plt.plot(self.autocorrelation)
        plt.grid()

    def average(self):
        psd_average = 0
        for i in range(self.n):
            self.rsg.gen()
            psd_average += np.real(self.rsg.squared_fft_coefficients) / self.n
        self.psd_average = psd_average

    def plot(self):
        new_fig = plt.figure().add_subplot(111)
        new_fig.plot(self.rsg.frequencies, self.psd_average / self.rsg.n)
        new_fig.plot(self.rsg.frequencies, self.rsg.normalized_psd_samples)
        new_fig.grid()


if __name__ == '__main__':
    _f = lambda x: 1. / (1 + (np.pi * x) ** 2)
    rsg = RandomSignalGenerator(_f, 2 ** 18, 2 ** 12)
    rsg.gen()
    tester = RandomSignalGeneratorTest(rsg)
    tester.autocorrelation_function()
    # plt.plot(rsg.get_sample())
    plt.show()