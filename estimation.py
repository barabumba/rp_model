import numpy as np
from scipy.integrate import quad
from scipy.stats import norm, skew, kurtosis
from sample_gen import RandomSignalGenerator
import matplotlib
print(matplotlib.matplotlib_fname(), matplotlib.get_backend(), matplotlib.__version__)


class BasicMLE(object):
    def __init__(self, random_signal):
        self.random_signal = random_signal
        self.n = self.random_signal.n
        self.m0 = self.random_signal.m
        self.psd_fun = lambda x: self.random_signal.normalized_psd_fun(x)
        self.weight_fun = lambda x, q=1: 1+q*self.random_signal.normalized_psd_fun(x)

        self.filter_matrix = self._init_filter_matrix()
        self.correction = self._init_correction()

        self.temp = np.sqrt(self.m0 * quad(lambda x: self.psd_fun(x)**2, 0, np.inf)[0])
        print(quad(lambda x: self.psd_fun(x)**2, -np.inf, np.inf))

    def _init_filter_matrix(self):
        psd_samples = self.random_signal.normalized_psd_fun(self.random_signal.frequencies/self.random_signal.m)
        filter_matrix = psd_samples / (psd_samples + 1)
        return np.array(filter_matrix)

    def _init_correction(self, q=1):
        return self.m0 * quad(lambda x: np.log(self.weight_fun(x, q=q)), -np.inf, np.inf)[0]

    def run(self):
        fft_squared = self.random_signal.get_squared_fft_coefficients()
        out = np.matmul(self.filter_matrix, fft_squared)
        out -= 0.5 * (self.filter_matrix[0] * fft_squared[0] + self.filter_matrix[-1] * fft_squared[-1])
        out /= self.n
        # out -= self.correction
        out -= self.m0 * quad(lambda x: self.psd_fun(x), 0, 0.5*self.n/self.m0)[0]
        out /= self.temp
        return out


if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt

    _f = lambda x: 1./(1+(np.pi*x)**2)
    rsg = RandomSignalGenerator(_f, 2**15, 2**8)
    estimator = BasicMLE(rsg)

    rsg.gen()
    a = []
    t0 = time.time()
    for i in range(10**5):
        if i%1000==0:
            print(i)
        rsg.gen()
        a.append(estimator.run())
    print(time.time() - t0)
    print(np.mean(a), np.var(a), skew(np.array(a)), kurtosis(np.array(a)))
    plt.hist(a, bins=30, density=True)
    plt.plot(np.linspace(-5, 5, 100), norm.pdf(np.linspace(-5, 5, 100)))
    plt.show()
