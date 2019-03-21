import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


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
        fft = fft_n + fft_xi
        self.fft = fft_xi
        self.squared_fft_coefficients = np.real(fft * fft.conjugate())

    def get_squared_fft_coefficients(self):
        return self.squared_fft_coefficients

    def get_sample(self):
        return np.real(np.fft.ifft(self.fft))


class RandomSignalGeneratorTest(object):
    def __init__(self, random_signal_generator, size=1000):
        self.rsg = random_signal_generator
        self.n = size

    def show_realisation(self, external_axis=False):
        sample = self.rsg.get_sample()[:int(8. * self.rsg.n / self.rsg.m / 2)]
        if not external_axis:
            fig_, ax = plt.subplots()
            ax.plot(sample, label='realisation')
            ax.grid(1)
            ax.legend()
        return sample

    def acf_test(self, external_axis=False):
        sample = self.rsg.get_sample()
        partial_sample = sample[:int(self.rsg.n*(self.rsg.m-8./2)/self.rsg.m)+1]

        acf = np.correlate(sample, partial_sample) / np.sum(partial_sample**2)
        acf_expected = np.real(np.fft.ifft(self.rsg.normalized_psd_samples))[:len(acf)]
        acf_expected /= max(acf_expected)

        if not external_axis:
            fig_, ax = plt.subplots()
            ax.plot(acf_expected, label='Expected ACF')
            ax.plot(acf, label='Actual ACF')
            ax.legend()
            ax.grid(1)
        return acf, acf_expected

    def dispersion_test(self, external_print=False):
        var_actual = np.sum(self.rsg.get_sample()**2)/self.rsg.n
        _const = 0.5 * self.rsg.n / self.rsg.m
        var_expected = quad(lambda x: self.rsg.normalized_psd_fun(x), 0, _const)[0]/_const

        if not external_print:
            print("#" * 30)
            print("Actual dispersion: {0:f}".format(var_actual))
            print("Expected dispersion: {0:f}".format(var_expected))
            print("#" * 30)
        return var_actual, var_expected


if __name__ == '__main__':
    delta = 0.75

    def _f(x):
        if abs(x) <= (1-delta)/2:
            return 1.
        elif x < -(1-delta)/2:
            return 1. / (1 + (np.pi / delta * (x + (1-delta)/2)) ** 2)
        else:
            return 1. / (1 + (np.pi / delta * (x - (1-delta)/2)) ** 2)

    rsg = RandomSignalGenerator(_f, 2 ** 22, 2 ** 12)
    rsg.gen()

    tester = RandomSignalGeneratorTest(rsg)
    tester.show_realisation()
    tester.dispersion_test()
    tester.acf_test()

    plt.show()
