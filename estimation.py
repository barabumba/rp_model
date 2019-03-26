import numpy as np
from scipy.integrate import quad
from scipy.stats import norm, skew
from sample_gen import RandomSignalGenerator


class BasicMLE(object):
    def __init__(self, random_signal):
        self.random_signal = random_signal
        self.n = self.random_signal.n
        self.m0 = self.random_signal.m
        self.psd_fun = lambda x: self.random_signal.normalized_psd_fun(x)
        self.weight_fun = lambda x, q=1: 1+q*self.random_signal.normalized_psd_fun(x)

        self.filter_matrix = self._init_filter_matrix()
        self.correction = self._init_correction()

        self.temp = np.sqrt(self.m0 /2 * quad(lambda x: self.psd_fun(x)**2, -np.inf, np.inf)[0])
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
        out -= self.m0 / 2
        out /= self.temp
        return out


class LikelihoodEstimation(object):
    def __init__(self, random_signal, auto_init=True):
        self.random_signal = random_signal
        self.search_range = np.arange(int(3 * random_signal.m / 4), int(5 * random_signal.m / 4), step=1)

        self.filter_matrix = None
        self.second_component = None
        if auto_init:
            self.init_filter_matrix()
            self.init_second_component()

    def init_filter_matrix(self):
        filter_matrix = []
        for m in self.search_range:
            psd_samples = self.random_signal.normalized_psd_fun(self.random_signal.frequencies/m)
            transfer_function_samples = psd_samples / (psd_samples + 1)
            filter_matrix.append(transfer_function_samples)
        self.filter_matrix = np.array(filter_matrix)
        print('Filter is initialized')

    def init_second_component(self):
        second_component = []

        def integrand(x, q=1):
            return np.log(1+q*self.random_signal.normalized_psd_fun(x))
        i = quad(lambda x: integrand(x), -np.inf, np.inf)[0]
        print(i)
        for m in self.search_range:
            second_component.append(m*i)
        self.second_component = np.array(second_component)
        print('Second component is initialized')

    def run(self):
        filter_output = np.matmul(self.filter_matrix, self.random_signal.get_squared_fft_coefficients()) / (self.random_signal.n / 2)
        self.out = filter_output - self.second_component
        return self.search_range[np.argmax(filter_output)]
        # print filter_output, self.random_signal.m/2*0.707*np.pi


if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt

    _f = lambda x: 1./(1+(np.pi*x)**2)
    rsg = RandomSignalGenerator(_f, 2**17, 2**10)
    estimator = BasicMLE(rsg)

    rsg.gen()
    a = []
    t0 = time.time()
    for i in range(10**5):
        print(i)
        rsg.gen()
        a.append(estimator.run())
    print(time.time() - t0)
    print(np.mean(a), np.var(a), skew(np.array(a)))
    plt.hist(a, bins=50, density=True)
    plt.plot(np.linspace(-5,5,100), norm.pdf(np.linspace(-5,5,100)))
    plt.show()
