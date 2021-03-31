import numpy as np
from SimulateData import SimulateData
from Distribution import normal_pdf

class BayesianHMM:

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


    def sample_parameters(self, num_iter=int(1e5)):
        """
        inference step
        :return:
        """
        for i in range(num_iter):
            self.sample_mu()
            self.sample_sigma_invsq()
            self.sample_beta()
            self.sample_A()
            self.sample_initial_dist()
            self.sample_states()


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
            indices = np.where(self.state_path == i)[0] + 1 # indices of X_k

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
        self.initial_dist = np.random.dirichlet(alpha)

    def sample_states(self):
        # sample X1
        beta = self.backward_continuous()



    def backward_continuous(self):
        beta = np.zeros((self.num_obs, self.num_states))

        # setting beta(T) = 1
        beta[self.observations.shape[0] - 1] = np.ones((self.num_states))

        # Loop in backward way from T-1 to
        # Due to python indexing the actual loop will be T-2 to 0
        for t in range(self.num_obs - 2, -1, -1):
            for j in range(self.num_states):
                beta[t, j] = (beta[t + 1] * normal_pdf(self.observations[t + 1], self.mu[j], np.sqrt(1/self.sigma_invsq))).dot(self.A[j, :])
        return beta


if __name__ == '__main__':
    # TODO: work on simulating data and initializing state path
    simulate = SimulateData()
    observations, state_path, A, B, initial = simulate.simulate_data(num_obs=int(1e4), continuous=True)

    HMM = BayesianHMM(observations=observations, state_path=state_path, num_states=3)
    HMM.generate_priors()
    HMM.sample_parameters()