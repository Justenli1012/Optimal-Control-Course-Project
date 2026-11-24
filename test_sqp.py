import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from dataclasses import dataclass
from SQP import SQPSolver


@dataclass
class DoubleIntegrator:

    nx = 2
    nu = 1

    N = 30
    dt = 0.1

    x0 = jnp.array([0.0, 0.0])
    x_goal = jnp.array([5.0, 0.0])

    u_min = -3.0
    u_max = 3.0

    inequality = None
    equality = None

    terminal_inequality = None
    terminal_equality = None

    def dynamics(self, x, u, k):

        p, v = x
        a = u[0]

        return jnp.array([
            p + v * self.dt + 0.5 * a * self.dt**2,
            v + a * self.dt
        ])

    def stage_cost(self, x, u, k):

        return 0.1 * jnp.sum(u**2)

    def terminal_cost(self, x):

        error = x - self.x_goal

        return 1000 * jnp.sum(error**2)


problem = DoubleIntegrator()

solver = SQPSolver(
    max_iter=100,
    tol=1e-5,
    verbose=True
)

result = solver.solve(problem)

X = result["X"]
U = result["U"]

print("Final state:", X[-1])
print("Cost:", result["cost"])
print("Runtime:", result["runtime"])
print("Success:", result["success"])
print("Message:", result["message"])

t = np.arange(problem.N + 1) * problem.dt

plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(t, X[:, 0])
plt.axhline(5.0, linestyle="--")
plt.ylabel("Position (m)")
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(t, X[:, 1])
plt.ylabel("Velocity (m/s)")
plt.grid()

plt.subplot(3, 1, 3)
plt.step(t[:-1], U[:, 0], where="post")
plt.ylabel("Acceleration (m/s²)")
plt.xlabel("Time (s)")
plt.grid()

plt.tight_layout()
plt.show()