import numpy as np
from sample_gen import RandomSignalGenerator, PSDAveraging


class LikelihoodEstimation(object):
    def __init__(self, random_signal):
        self.random_signal = random_signal
        self.bandwidth_search_range = range(random_signal.m/2, 3*random_signal.m/2)
        self.filter_matrix = None

    def init_filter_matrix(self):
        filter_matrix = []
        for m in self.bandwidth_search_range:
            psd_samples = self.random_signal.normalized_psd_fun(self.random_signal.frequencies/m)
            transfer_function_samples = psd_samples / (psd_samples + 1)
            filter_matrix.append(transfer_function_samples)
        self.filter_matrix = np.array(filter_matrix)

    def run(self):
        filter_output = np.matmul(np.sqrt(self.filter_matrix), self.random_signal.get()) / self.random_signal.n
        print filter_output


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    _f = lambda x: 1./(1+(np.pi*x)**2)
    rsg = RandomSignalGenerator(_f, 2**13, 2**5)
    rsg.gen()
    # plt.plot(rsg.get_samples())
    estimator = LikelihoodEstimation(rsg)
    estimator.init_filter_matrix()
    estimator.run()
    # test = PSDAveraging(rsg, size=2**13)
    # test.average()
    # test.plot()
    plt.show()