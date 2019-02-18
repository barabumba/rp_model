import numpy as np
from scipy.integrate import quad
from sample_gen import RandomSignalGenerator, PSDAveraging


class LikelihoodEstimation(object):
    def __init__(self, random_signal):
        self.random_signal = random_signal
        self.bandwidth_search_range = range(random_signal.m/2, 3*random_signal.m/2)
        self.filter_matrix = None
        self.second_component = None

    def init_filter_matrix(self):
        filter_matrix = []
        for m in self.bandwidth_search_range:
            psd_samples = self.random_signal.normalized_psd_fun(self.random_signal.frequencies/m)
            transfer_function_samples = psd_samples / (psd_samples + 1)
            filter_matrix.append(transfer_function_samples)
        self.filter_matrix = np.array(filter_matrix)

    def init_second_component(self):
        second_component = []

        def integrand(x, q=1):
            return np.log(1+q*self.random_signal.normalized_psd_fun(x))
        i = quad(lambda x: integrand(x), -np.inf, np.inf)[0]
        print i
        for m in self.bandwidth_search_range:
            second_component.append(m*i/2)
        self.second_component = np.array(second_component)
        print second_component

    def run(self):
        filter_output = np.matmul(self.filter_matrix, self.random_signal.get()) / self.random_signal.n
        self.out = filter_output - self.second_component
        # print filter_output, self.random_signal.m/2*0.707*np.pi


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    _f = lambda x: 1./(1+(np.pi*x)**2)
    rsg = RandomSignalGenerator(_f, 2**15, 2**5)
    rsg.gen()
    # plt.plot(rsg.get_samples())
    estimator = LikelihoodEstimation(rsg)
    estimator.init_filter_matrix()
    estimator.init_second_component()

    average_out = 0
    n = 1000
    for i in range(n):
        estimator.run()
        average_out += estimator.out / n
        rsg.gen()
    print average_out, estimator.random_signal.m/2*0.707
    # test = PSDAveraging(rsg, size=2**12)
    # test.average()
    # test.plot()
    plt.show()