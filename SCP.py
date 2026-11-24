#Justen Part
# Take in nonconvex problem: approximate nonconvex constraints, repeatedely solve convex optimization problem using cvxpy+clarabel

# Initialize trajectroy: X_ref, U_ref
# Linearize obstacle constraints: build convex approximations near X_ref
# solve with cvxpy+clarabel: X*, U*
# evaluate original cost and constraints: accept/reject candidates, adjust trust region
# Repeat until convergence


# CVXPY - inputs: objective function, dynamics, constraints, variables 
# CLARABEL - solves the SCP formualted by cvxpy

# own algo: repeatedeltly updates trajectory and calls cvxpy+clarabel

# Clohessy-Wiltshire (CW) dynamics: 
# x_k+1 = A * x_k + B * u_k

# Ad: discrete spacecraft dynamics
# Bd: discete Control input matrix
# x_0, x_goal, N (timesteps), u_max 
# obstacle center [3x1 vector], obstacle radius = r
# reference trajectory: 6 x (N+1): current SCP trajectory estimate
# trust region radius (scalar): max change from reference trajectory

# x_k = [x y z vx vy vz ].T
# u_k = [ux uy uz].T

import time
import numpy as np
import cvxpy as cp

from problem_definition import Result

# An impulse changes velocity, but not position.
G = np.zeros((6, 3))
G[3:6, :] = np.eye(3)


def rollout(problem, controls):
    """
    Simulate the spacecraft using discrete CWH dynamics.
    """

    X = np.zeros((problem.N + 1, 6))
    X[0] = problem.x0

    for k in range(problem.N):

        x_current_state = X[k]
        delta_v = controls[k]

        #x_k+1 = A* (x_k + G*delta_v)
        # Apply the impulse at the start of the interval: 
        x_next_state = problem.Ad @ ( x_current_state + G @ delta_v )
        X[k + 1] = next_state

    return X


def initialize_reference(problem):
    """
    Find an initial trajectory connecting the start and goal.
    This initial trajectory does not need to avoid obstacles.
    """

    N = problem.N

    X = cp.Variable((N + 1, 6))
    U = cp.Variable((N, 3))

    constraints = []

    # Initial and final states
    constraints.append(X[0] == problem.x0)
    constraints.append(X[N] == problem.goal)

    # Spacecraft dynamics and impulse limits
    for k in range(N):

        next_state = problem.Ad @ (X[k] + G @ U[k])

        constraints.append(X[k + 1] == next_state)
        constraints.append(cp.norm(U[k], 2) <= problem.u_max)

    # Find a reasonable starting control sequence
    cost = cp.sum_squares(U)

    optimization = cp.Problem(
        cp.Minimize(cost),
        constraints
    )

    optimization.solve(solver=cp.CLARABEL)

    if optimization.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError("Could not find an initial trajectory.")

    controls_ref = np.array(U.value, copy=True)
    states_ref = rollout(problem, controls_ref)

    return states_ref, controls_ref


def evaluate_trajectory(problem, controls, penalty):
    """
    Evaluate the original fuel cost and obstacle violations.
    """

    X = rollout(problem, controls)

    # Total delta-v required
    fuel_cost = 0.0

    for k in range(problem.N):
        fuel_cost += np.linalg.norm(controls[k])

    # Check each obstacle at each discrete state
    total_violation = 0.0
    max_violation = 0.0

    for obstacle in problem.obstacles:

        for k in range(problem.N + 1):

            position = X[k, :3]

            distance = np.linalg.norm(
                position - obstacle.center
            )

            violation = max(
                0.0,
                obstacle.radius - distance
            )

            total_violation += violation
            max_violation = max(max_violation, violation)

    # Merit function for accepting/rejecting SCP steps
    merit = fuel_cost + penalty * total_violation

    return merit, max_violation


def build_subproblem(problem, X_ref, trust_radius, penalty):
    """
    Build one convex approximation of the trajectory problem.
    """

    N = problem.N

    # Decision variables
    X = cp.Variable((N + 1, 6))
    U = cp.Variable((N, 3))

    constraints = []

    # Initial and final state constraints
    constraints.append(X[0] == problem.x0)
    constraints.append(X[N] == problem.goal)

    # Dynamics and maximum impulse constraints
    for k in range(N):

        next_state = problem.Ad @ (X[k] + G @ U[k])

        constraints.append(X[k + 1] == next_state)
        constraints.append(cp.norm(U[k], 2) <= problem.u_max)

    # Trust region: restrict changes from reference positions
    for k in range(N + 1):

        position_change = X[k, :3] - X_ref[k, :3]

        constraints.append(
            cp.norm(position_change, 2) <= trust_radius
        )

    # Objective: minimize the sum of delta-v magnitudes
    fuel_cost = 0

    for k in range(N):
        fuel_cost += cp.norm(U[k], 2)

    cost = fuel_cost

    # Linearize the obstacle constraints
    num_obstacles = len(problem.obstacles)

    if num_obstacles > 0:

        # Slack allows temporary obstacle violations
        slack = cp.Variable(
            (num_obstacles, N + 1),
            nonneg=True
        )

        for j, obstacle in enumerate(problem.obstacles):

            for k in range(N + 1):

                reference_position = X_ref[k, :3]
                obstacle_center = obstacle.center

                # Direction from obstacle to reference point
                direction = reference_position - obstacle_center
                distance = np.linalg.norm(direction)

                if distance > 1e-9:
                    normal = direction / distance
                else:
                    normal = np.array([1.0, 0.0, 0.0])

                # Linearized obstacle avoidance constraint
                new_position = X[k, :3]

                clearance = normal @ (
                    new_position - obstacle_center
                )

                constraints.append(
                    clearance + slack[j, k] >= obstacle.radius
                )

        # Penalize obstacle violations
        obstacle_cost = penalty * cp.sum(slack)

        cost += obstacle_cost

    optimization = cp.Problem(
        cp.Minimize(cost),
        constraints
    )

    return optimization, X, U


def solve(
    problem,
    max_iterations=30,
    trust_radius=10.0,
    penalty=1000.0,
    tolerance=1e-4,
    verbose=False
):
    """
    Solve the spacecraft trajectory problem using SCP.
    """

    start_time = time.perf_counter()

    # Step 1: Find the starting trajectory
    X_ref, U_ref = initialize_reference(problem)

    current_merit, current_violation = evaluate_trajectory(
        problem,
        U_ref,
        penalty
    )

    status = "max_iterations"
    iterations = 0

    # Step 2: Repeatedly solve convex approximations
    for iteration in range(max_iterations):

        iterations = iteration + 1

        # Build a convex problem around the current trajectory
        optimization, X, U = build_subproblem(
            problem,
            X_ref,
            trust_radius,
            penalty
        )

        # Step 3: Solve using Clarabel
        try:
            optimization.solve(solver=cp.CLARABEL)

        except cp.error.SolverError:
            status = "solver_failed"
            break

        if optimization.status not in (
            cp.OPTIMAL,
            cp.OPTIMAL_INACCURATE
        ):
            status = "solver_failed"
            break

        # Extract the optimized control sequence
        U_new = np.array(U.value, copy=True)

        # Step 4: Evaluate the original problem
        X_new = rollout(problem, U_new)

        new_merit, new_violation = evaluate_trajectory(
            problem,
            U_new,
            penalty
        )

        improvement = current_merit - new_merit

        if verbose:
            print(f"Iteration: {iterations}")
            print(f"Merit cost: {new_merit:.6f}")
            print(f"Obstacle violation: {new_violation:.6f}")
            print(f"Trust radius: {trust_radius:.4f}")
            print()

        # Step 5: Decide whether to accept the new trajectory
        if improvement < -1e-7:

            # Reject: original merit cost increased
            trust_radius = trust_radius * 0.5

        else:

            # Measure how much the trajectory changed
            position_changes = X_new[:, :3] - X_ref[:, :3]

            step_size = np.max(
                np.linalg.norm(position_changes, axis=1)
            )

            # Accept the new trajectory
            X_ref = X_new
            U_ref = U_new

            current_merit = new_merit
            current_violation = new_violation

            # Check convergence
            if abs(improvement) < tolerance and step_size < tolerance:

                if current_violation < tolerance:
                    status = "converged"
                else:
                    status = "stalled_with_obstacle_violation"

                break

            # Expand trust region after accepting a step
            trust_radius = min(
                trust_radius * 1.2,
                100.0
            )

        if trust_radius < 1e-5:

            if current_violation < tolerance:
                status = "stalled"
            else:
                status = "stalled_with_obstacle_violation"

            break

    # Check the final obstacle feasibility
    if current_violation >= tolerance and status == "max_iterations":
        status = "obstacle_violation"

    # Step 6: Return the solution
    runtime = time.perf_counter() - start_time

    result = Result(
        name="SCP",
        X=X_ref,
        U=U_ref,
        runtime_s=runtime,
        iterations=iterations,
        status=status
    )

    return result
