#   NONLINEAR OPTIMIZATION PROBLEM
#   min_z J(z)
#   s.t. g(z) <= 0
#        h(z) = 0
#        zmin <= z <= zmax

#   QUADRATIC SUBPROBLEM
#   min_d 1/2 * d.T * B_k * d + grad(J(z_k)).T * d
#   s.t grad(g(z_k)) * d <= -g(z_k)
#       grad(h(z_k)) * d = -h(z_k)
#       zmin - z_k <= d <= zmax - z_k

import numpy as np
import jax
import jax.numpy as jnp
import scipy.sparse as sp
import osqp

from time import perf_counter

jax.config.update("jax_enable_x64", True)

class SQPSolver:

    def __init__(
            self,
            max_iter = 500,
            tol = 1e-5,
            rho = 10.0,
            beta = 0.5,
            c1 = 1e-4,
            max_line_search = 25,
            verbose = False
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.rho = rho
        self.beta = beta
        self.c1 = c1
        self.max_line_search = max_line_search
        self.verbose = verbose

    def solve_qp(self, H, grad, g, G, h, E, z, lb, ub):

        n = len(z)

        A = np.vstack([G, E, np.eye(n)])

        lower = np.concatenate([np.full(len(g), -np.inf), -h, lb - z])

        upper = np.concatenate([-g, -h, ub - z])

        qp = osqp.OSQP()

        qp.setup(
            P = sp.csc_matrix(np.triu(H)),
            q = grad,
            A = sp.csc_matrix(A),
            l = lower,
            u = upper,
            verbose = False,
            eps_abs = 1e-8,
            eps_rel = 1e-8,
            max_iter = 10000
        )

        result = qp.solve()

        return result.x, result.y, result.info.status

    def violation(self, g, h):

        ineq_violation = np.sum(np.maximum(g, 0.0))

        eq_violation = np.sum(np.abs(h))

        return ineq_violation + eq_violation

    def merit(self, z, objective, inequality, equality, rho):

        cost = float(objective(z))
        g = np.asarray(inequality(z))
        h = np.asarray(equality(z))

        return cost + rho * self.violation(g, h)

    def line_search(self, z, d, objective, inequality, equality, grad, g, G, h, E, rho):

        alpha = 1.0

        merit_current = self.merit(z, objective, inequality, equality, rho)

        predicted_change = (grad @ d + rho * (self.violation(g + G @ d, h+ E @ d) - self.violation(g, h)))

        if predicted_change >= 0:
            return 0.0

        for _ in range(self.max_line_search):

            z_trial = z + alpha * d

            merit_trial = self.merit(z_trial, objective, inequality, equality, rho)

            if merit_trial <= (merit_current + self.c1 * alpha * predicted_change):
                return alpha

            alpha *= self.beta

        return 0.0

    def update_hessian(self, H, s, y):

        if np.linalg.norm(s) < 1e-12:
            return H

        Hs = H @ s

        sHs = s @ Hs
        sy = s @ y

        if sHs <= 1e-12:
            return np.eye(len(s))

        if sy >= 0.2 * sHs:
            theta = 1.0
        else:
            theta = 0.8 * sHs / (sHs - sy)

        r = theta * y + (1.0 - theta) * Hs

        sr = s @ r

        H_new = (H - np.outer(Hs, Hs) / sHs + np.outer(r, r) / sr)

        H_new = 0.5 * (H_new + H_new.T)

        eig_min = np.min(np.linalg.eigvalsh(H_new))

        if eig_min < 1e-8:
            H_new += (1e-8 - eig_min) * np.eye(len(s))

        return H_new

    def solve_nlp(self, objective, z_init, inequality = None, equality = None, lb = None, ub = None):

        start = perf_counter()

        runtime = perf_counter() - start

        z = np.asarray(z_init, dtype=float).copy()
        n = len(z)

        if inequality is None:
            inequality = lambda z: jnp.zeros(0)

        if equality is None:
            equality = lambda z: jnp.zeros(0)

        if lb is None:
            lb = np.full(n, -np.inf)
        else:
            lb = np.broadcast_to(lb, (n,)).astype(float)

        if ub is None:
            ub = np.full(n, np.inf)
        else:
            ub = np.broadcast_to(ub, (n,)).astype(float)

        if np.any(lb > ub):
            raise ValueError("Invalid Variable Bounds.")

        z = np.clip(z, lb, ub)

        grad_f = jax.grad(objective)
        jac_g = jax.jacfwd(inequality)
        jac_h = jax.jacfwd(equality)

        H = np.eye(n)

        rho = self.rho

        history = {
            "cost": [],
            "violation": [],
            "step_norm": [],
            "stationarity": [],
            "alpha": []
            }

        success = False
        message = "Maximum Iterations Reached."
        iterations = 0

        for k in range(self.max_iter):

            iterations = k+1

            grad = np.asarray(grad_f(z))
            g = np.asarray(inequality(z))
            h = np.asarray(equality(z))

            G = np.asarray(jac_g(z))
            E = np.asarray(jac_h(z))

            d, multipliers, status = self.solve_qp(H, grad, g, G, h, E, z, lb, ub)

            lambda_g = multipliers[:len(g)]
            lambda_h = multipliers[len(g):len(g) + len(h)]
            lambda_bounds = multipliers[len(g) + len(h)]

            grad_L = (grad + G.T @ lambda_g + E.T @ lambda_h + lambda_bounds)

            stationarity = np.linalg.norm(grad_L, ord=np.inf)

            violation = self.violation(g, h)
            step_norm = np.linalg.norm(d, ord=np.inf)

            complementarity = np.max(np.abs(lambda_g * g), initial=0.0)

            history["cost"].append(float(objective(z)))
            history["violation"].append(violation)
            history["step_norm"].append(step_norm)
            history["stationarity"].append(stationarity)

            if (step_norm <= self.tol and violation <= self.tol and stationarity <= self.tol and complementarity <= self.tol):
                success = True
                message = "SQP Convergence."
                break

            constraint_multipliers = np.concatenate([lambda_g, lambda_h])

            if constraint_multipliers.size > 0:
                rho = max(rho, 1.5 * np.max(np.abs(constraint_multipliers)))

            alpha = self.line_search(z, d, objective, inequality, equality, grad, g, G, h, E, rho)

            history["alpha"].append(alpha)

            if alpha == 0.0:
                message = "Line Search Failed."
                break

            z_new = z + alpha * d

            grad_new = np.asarray(grad_f(z_new))
            G_new = np.asarray(jac_g(z_new))
            E_new = np.asarray(jac_h(z_new))

            grad_L_new = (grad_new + G_new.T @ lambda_g + E_new.T @ lambda_h + lambda_bounds)

            s = z_new - z
            y = grad_L_new - grad_L

            H = self.update_hessian(H, s, y)

            z = z_new

        g_final = np.asarray(inequality(z))
        h_final = np.asarray(equality(z))

        final_violation = self.violation(g_final, h_final)

        bound_violation = max(np.max(lb - z, initial = 0.0), np.max( z - ub, initial = 0.0))

        feasible = (final_violation <= self.tol and bound_violation <= self.tol)

        return {
            'z': z,
            'cost': float(objective(z)),
            'runtime': runtime,
            'iterations': iterations,
            'success': success and feasible,
            'feasible': feasible,
            'violation': final_violation,
            'message': message,
            'history': history
        }

    def rollout(self, problem, U):

        x = jnp.asarray(problem.x0)
        states = [x]

        for k in range(problem.N):

            x = problem.dynamics(x, U[k], k)

            states.append(x)

        return jnp.stack(states)

    def solve(self, problem, U_init=None):

        N = problem.N
        nu = problem.nu

        if U_init is None:
            U_init = np.zeros((N, nu))

        def objective(z):

            U = z.reshape(N, nu)
            X = self.rollout(problem, U)

            cost = 0.0

            for k in range(N):
                cost ++ problem.stage_cost(X[k], U[k], k)

            cost += problem.terminal_cost(X[-1])

            return cost

        def inequality(z):

            U = z.reshape(N, nu)
            X = self.rollout(problem, U)

            constraints = []

            if problem.inequality is not None:

                for k in range(N):

                    g = problem.inequality(X[k], U[k], k)

                    constraints.append(jnp.ravel(g))

            if problem.terminal_inequality is not None:

                g = problem.terminal_inequality(X[-1])

                constraints.append(jnp.ravel(g))

            if len(constraints) == 0:
                return jnp.zeros(0)

            return jnp.concatenate(constraints)

        def equality(z):

            U = z.reshape(N, nu)
            X = self.rollout(problem, U)

            constraints = []

            if problem.equality is not None:

                for k in range(N):

                    h = problem.equality(X[l], U[k], k)

                    constraints.append(jnp.ravel(h))

            if problem.terminal_equality is not None:

                h = problem.terminal_inequality(X[-1])

                constraints.append(jnp.ravel(h))

            if len(constraints) == 0:
                return jnp.zeros(0)

            return jnp.concatenate(constraints)

        lb = np.tile(np.broadcast_to(problem.u_min, (nu,)), N)

        ub = np.tile(np.broadcast_to(problem.u_max, (nu,)), N)

        result = self.solve_nlp(
            objective = objective,
            z_init = np.asarray(U_init).reshape(-1),
            inequality = inequality,
            equality = equality,
            lb = lb,
            ub = ub
        )

        U_opt = result['z'].reshape(N, nu)

        X_opt = np.asarray(self.rollout(problem, U_opt))

        result['X'] = X_opt
        result['U'] = U_opt

        return result
    




