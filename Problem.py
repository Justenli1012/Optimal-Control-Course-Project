
import numpy as np
from scipy.linalg import expm


class Problem:

    def __init__(self, x0, goal, N, dt, orbital_radius, mu=3.986004418e14):

        self.x0 = np.asarray(x0, dtype=float)
        self.goal = np.asarray(goal, dtype=float)

        self.N = N
        self.dt = dt

        # Reference orbit parameters (SI units)
        self.mu = mu
        self.orbital_radius = orbital_radius

        # Orbital mean motion
        self.n = np.sqrt(mu / orbital_radius**3)

        # Continuous CWH dynamics
        n = self.n

        self.Ac = np.array([
            [0,      0,   1,    0],
            [0,      0,   0,    1],
            [3*n**2, 0,   0,  2*n],
            [0,      0, -2*n,   0]
        ])

        # Discrete CWH dynamics
        self.Ad = expm(self.Ac * self.dt)

        # Impulsive control matrix
        self.G = np.array([
            [0, 0],
            [0, 0],
            [1, 0],
            [0, 1]
        ])

    def rollout(self, U):
        """Simulate the spacecraft trajectory."""

        X = np.zeros((self.N + 1, 4))
        X[0] = self.x0

        for k in range(self.N):

            # Apply delta-v impulse
            state_after_burn = X[k] + self.G @ U[k]

            # Propagate spacecraft motion
            X[k + 1] = self.Ad @ state_after_burn

        return X

    def cost(self, U):
        """Calculate total delta-v expenditure."""

        total_delta_v = 0.0

        for k in range(self.N):
            total_delta_v += np.linalg.norm(U[k])

        return total_delta_v
