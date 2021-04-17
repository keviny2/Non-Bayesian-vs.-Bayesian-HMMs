import os
import numpy as np
from SimulateData import SimulateData
from numba_functions import backward_robust, sample_states_numba
import matplotlib.pyplot as plt


class BayesianHMM():

    def __init__(self, observations=None, state_path=None, num_states=6):
        """

        :param observations: observation vector
        :param states: state sequence
        :param num_states: number of states
        """
        self.observations = np.array(observations)
        self.state_path = np.array(state_path)
        self.num_obs = len(self.observations)
        self.num_states = num_states
        self.chain = []

        # model parameters initially set to None
        self.initial_dist = None
        self.A = None
        self.mu = None
        self.sigma_invsq = None

        # model hyperparameters initially set to None
        self.xi = None
        self.kappa = None
        self.alpha = None
        self.beta = None


    def generate_priors(self):
        """
        generate priors according to the paper and set class variables
        :return:
        """
        dir_param = np.ones(self.num_states)
        initial_dist = np.random.dirichlet(dir_param)

        # state transition matrix
        A = np.random.dirichlet(dir_param, self.num_states)

        # generate mus from prior dist.
        max = self.observations.max()
        min = self.observations.min()
        R = max - min

        xi = (min + max) / 2
        kappa = 1 / R**2

        mu = np.random.normal(loc=xi, scale=1/kappa, size=self.num_states)

        # generate sigmas from prior dist.
        alpha = 2
        g = 0.2
        h = 10/R**2
        beta = np.random.gamma(shape=g, scale=1/h)
        sigma_invsq = np.random.gamma(shape=alpha, scale=1/beta)

        # set hmm model parameter class variables
        self.initial_dist = initial_dist
        self.A = A
        self.mu = mu
        self.sigma_invsq = sigma_invsq

        # set hmm model hyperparameter class variables
        self.xi = xi
        self.kappa = kappa
        self.g = g
        self.h = h
        self.alpha = alpha
        self.beta = beta


    def sample_parameters(self, num_iter=int(1e5), num_burnin=int(1e2)):
        """
        inference step
        :return:
        """

        # burnin period
        print("=" * 20, 'Performing Burn-in', '=' * 20)
        for i in range(num_burnin):
            print('(B) Iteration:', i+1)

            self.sample_mu()
            self.sample_sigma_invsq()
            self.sample_beta()
            self.sample_A()
            self.sample_initial_dist()
            self.sample_states()

        # inference
        print("=" * 20, 'Performing Inference', '=' * 20)
        for i in range(num_iter):
            print('(I) Iteration:', i+1)

            self.sample_mu()
            self.sample_sigma_invsq()
            self.sample_beta()
            self.sample_A()
            self.sample_initial_dist()
            self.sample_states()

            # if i % 100 == 0:
            self.chain.append({'mu': self.mu,
                               'sigma_invsq': self.sigma_invsq,
                               'beta': self.beta,
                               'A': self.A,
                               'initial_dist': self.initial_dist,
                               'sample_states': self.state_path})


    def sample_mu(self):
        for i in range(self.num_states):
            index = np.where(self.state_path == i)[0]
            S_i = np.sum(self.observations[index])
            n_i = len(index)
            sigma_sq = 1 / self.sigma_invsq
            self.mu[i] = np.random.normal((S_i + self.kappa*self.xi*sigma_sq) / (n_i + self.kappa*sigma_sq),
                                           sigma_sq / (n_i + self.kappa*sigma_sq))


    def sample_sigma_invsq(self):
        mus = [self.mu[int(state)] for state in self.state_path]
        self.sigma_invsq = np.random.gamma(self.alpha + 0.5*self.num_obs,
                                           1/(self.beta + 0.5*np.sum((self.observations-mus)**2)))


    def sample_beta(self):
        self.beta = np.random.gamma(self.g + self.alpha,
                                    1/(self.h + self.sigma_invsq))

    def sample_A(self):
        for i in range(self.num_states):
            # find indices of states that come right after state i
            indices = np.where(self.state_path == i)[0] + 1  # indices of X_k

            # need to address the case for the last state in the sequence
            if self.num_obs in indices:
                indices = np.delete(indices, np.where(indices == self.num_obs))

            states = self.state_path[indices]
            n_i = np.zeros(self.num_states)
            for j in range(self.num_states):
                n_i[j] = np.count_nonzero(states == j)
            self.A[i, :] = np.random.dirichlet(n_i + 1)


    def sample_initial_dist(self):
        alpha = np.zeros(self.num_states)
        for i in range(self.num_states):
            alpha[i] = np.count_nonzero(self.state_path == i)
        self.initial_dist = np.random.dirichlet(alpha + 1)


    def sample_states(self):
        B = [[self.mu[i], np.sqrt(1 / self.sigma_invsq)] for i in range(self.num_states)]

        # beta[i, j] = p(x_{i+1,...,n} | z_i = j)
        # beta[i, j] represents the probability that at time i,
        # you observe everything after (x_{i+1},...,x_n) and given that you are at state j (z_i = j)
        beta = backward_robust(self.A, np.asmatrix(B), self.observations)

        new_state_path = sample_states_numba(beta,
                                             self.initial_dist,
                                             self.observations,
                                             self.mu,
                                             self.sigma_invsq,
                                             self.A,
                                             self.num_obs)

        self.state_path = new_state_path


    def plot_results(self, num_iter, max=None, name = "Bayes_plot"):
        """
        plot the path of mus in MCMC chain
        :param num_iter:
        :param max:
        :return:
        """

        if max is None:
            max = num_iter

        mus = [entry['mu'] for entry in self.chain]
        mus = mus[-max:]
        mus = np.asmatrix(mus)  # row i corresponds to the mus in the chain at iteration i

        x = np.linspace(1, max, num=max)
        plt.figure()

        for i in range(self.num_states):
            plt.plot(x, mus[:, i])

        fname = os.path.join("..", "plots", name)
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)


if __name__ == '__main__':
    np.random.seed(123)

    simulate = SimulateData()
    observations, state_path, A, B, initial = simulate.simulate_continuous(num_obs=int(1e4))

    HMM = BayesianHMM(observations=observations, state_path=state_path, num_states=A.shape[0])
    HMM.generate_priors()

    num_iter = int(1e3)
    HMM.sample_parameters(num_iter=num_iter, num_burnin=int(1e2))

    max = 500
    HMM.plot_results(num_iter=num_iter, max=max)