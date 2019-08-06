import numpy as np
import numexpr as ne
import scipy.optimize as opt
from scipy.integrate import quad
from scipy.stats import norm, skew, kurtosis
from sample_gen import RandomSignalGenerator
import matplotlib
print(matplotlib.matplotlib_fname(), matplotlib.get_backend(), matplotlib.__version__)


class BasicMLE(object):
    def __init__(self, random_signal):
        self.random_signal = random_signal
        self.n = self.random_signal.n
        self.m_array = np.linspace(0.5*self.random_signal.m, 1.5*self.random_signal.m, 128, endpoint=False)
        assert self.random_signal.m in self.m_array

        self.psd_fun = lambda x: self.random_signal.normalized_psd_fun(x)
        self.weight_fun = lambda x, q=1: 1+q*self.random_signal.normalized_psd_fun(x)
        self.lc_base = quad(lambda x: np.log(self.weight_fun(x)), -np.inf, np.inf)[0]

    def run(self):
        fft_squared = self.random_signal.get_squared_fft_coefficients()
        l_fop = np.zeros(len(self.m_array))
        m = opt.minimize_scalar(lambda x: -self.calc(x, fft_squared),
                                bounds=(self.m_array[0], self.m_array[-1]), method="bounded")
        return m

    def calc(self, m, fft_squared):
        m = int(m)
        psd_samples = self.random_signal.normalized_psd_fun(self.random_signal.frequencies / m)
        factor = ne.evaluate("psd_samples/(psd_samples + 1)")
        lx = ne.evaluate("sum(fft_squared*factor)")
        lx -= 0.5 * (fft_squared[0] * factor[0] + fft_squared[-1] * factor[-1])
        l_fop = lx / self.n
        l_fop -= 0.5 * m * self.lc_base
        return l_fop


if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt

    _f = lambda x: np.exp(-np.pi*x**2)
    rsg = RandomSignalGenerator(_f, 2**14, 2**11)
    estimator = BasicMLE(rsg)

    t0 = time.time()
    a = []
    b = 0
    for i in range(1000):
        rsg.gen()
        print i
        temp = estimator.run()
        a.append(temp.x)
        b+= temp.nfev/1000.

    print b
    plt.hist(a, bins=30)
    plt.axvline(estimator.random_signal.m)
    print("TOTAL:", time.time() - t0)
    plt.show()

