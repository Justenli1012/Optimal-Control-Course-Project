import numpy as np


class MPPI:

    def __init__(
        self,
        problem,
        K=1000,
        num_iterations=50,
        temperature=1.0,
        noise_sigma=0.1
    ):

        self.problem = problem

        self.K = K

        self.num_iterations = num_iterations

        self.temperature = temperature

        self.noise_sigma = noise_sigma



def solve(self):

    problem = self.problem

    U = np.zeros((problem.N,2))

    for iteration in range(self.num_iterations):

        perturbations = np.random.normal(
            0,
            self.noise_sigma,
            size=(self.K,
                  problem.N,
                  2)
        )

        U_samples = U + perturbations

        costs = np.zeros(self.K)

        for k in range(self.K):

            X = problem.rollout(
                U_samples[k]
            )

            costs[k] = (
                problem.trajectory_cost(
                    X,
                    U_samples[k]
                )
            )

        beta = np.min(costs)

        weights = np.exp(
            -(costs-beta)
            /
            self.temperature
        )

        weights /= np.sum(weights)

        weighted_noise = np.sum(
            weights[:,None,None]
            *
            perturbations,
            axis=0
        )

        U += weighted_noise

    X_final = problem.rollout(U)

    return X_final, U