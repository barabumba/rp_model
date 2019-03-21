import numpy as np
from scipy.integrate import quad
from sample_gen import RandomSignalGenerator, PSDAveraging


class LikelihoodEstimation(object):
    def __init__(self, random_signal, auto_init=True):
        self.random_signal = random_signal
        self.bandwidth_search_range = random_signal.m + np.arange(-random_signal.m/2**2, random_signal.m/2**2, step=1)
        # print self.bandwidth_search_range
        self.filter_matrix = None
        self.second_component = None
        if auto_init:
            self.init_filter_matrix()
            self.init_second_component()

    def init_filter_matrix(self):
        filter_matrix = []
        for m in self.bandwidth_search_range:
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
        for m in self.bandwidth_search_range:
            second_component.append(m*i)
        self.second_component = np.array(second_component)
        print('Second component is initialized')

    def run(self):
        filter_output = np.matmul(self.filter_matrix, self.random_signal.get_squared_fft_coefficients()) / self.random_signal.n
        self.out = filter_output - self.second_component
        return self.bandwidth_search_range[np.argmax(filter_output)]
        # print filter_output, self.random_signal.m/2*0.707*np.pi


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    _f = lambda x: 1./(1+(np.pi*x)**2)
    rsg = RandomSignalGenerator(_f, 2**18, 2**11)
    estimator = LikelihoodEstimation(rsg)

    static = []
    l = 100
    for i in range(l):
        print(i)
        rsg.gen()
        estimator.run()
        static.append(estimator.out)
    static = np.array(static)
    print(np.mean(static, axis=0), np.var(static, axis=0))
    plt.show()
