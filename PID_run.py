import numpy as np
from DroneEnv import DroneLandingEnv, plot_xy_z, plot_trajectory_3d
from PIDCon import PIDController


def run_pid_simulation(
    env: DroneLandingEnv,
    pid: PIDController,
    init_mode: str,
    init_pos: list[float] = None,
    plot=True,
    verbose=False,
):
    """
    Run a PID-controlled landing simulation.

    Args:
        env.dt: simulation step (s)
        controller_dt: control update step (s)
        Kp, Ki, Kd: PID gains (lists of 4 values for x, y, z, yaw)
        integral_limit: anti-windup limit
        z_fixed_speed: fixed descent speed (m/s)
        init_mode: randomize initial position if True
        init_pos
        plot: show trajectory plots if True

    Returns:
        states (np.ndarray), target (np.ndarray)
    """

    steps_per_ctrl = int(pid.dt / env.dt)

    obs, info = env.reset(init_mode=init_mode, init_pos=init_pos)
    done = False
    states = []

    last_action = np.zeros(4)
    step_count = 0

    while not done:
        pos = obs[:4]  # [x,y,z,theta]
        error = env.target - pos
        if verbose:
            print("error", error)

        # Controller update
        if step_count % steps_per_ctrl == 0:
            theta = pos[3]
            if verbose:
                print("theta", theta)
            world_cmd = pid.compute(error)
            last_action = pid.rotate(world_cmd, theta)

        # Step simulation
        if verbose:
            print("action", last_action)
        obs, reward, terminated, truncated, info = env.step(last_action)
        states.append(obs.copy())
        step_count += 1
        done = terminated or truncated

    env.close()

    states = np.array(states)
    x, y, z, theta = states[:, 0], states[:, 1], states[:, 2], states[:, 3]
    target = env.target

    if plot:
        plot_trajectory_3d(x, y, z, theta, target)
        plot_xy_z(x, y, z, theta, target, env.dt)

    return states, target


def run_noisy_pid_simulation(
    env: DroneLandingEnv,
    pid: PIDController,
    init_mode: str,
    init_pos: list[float] = None,
    random_init=False,
    plot=True,
    max_delay_steps=2,  # 1 or 2 sim steps additional delay
    p_loss=0.1,  # packet loss probability
    wind_type="off",
    wind_std=[0.1, 0.1, 0.02, 0.05],
    noise_std=0.01,
    verbose=False,
):

    steps_per_ctrl = int(pid.dt / env.dt)
    obs, info = env.reset(init_mode=init_mode, init_pos=init_pos)
    done = False
    states = []

    # Initialize control
    last_action = np.zeros(4)  # last applied action
    pending_actions = []  # queue of (time_to_apply, action)

    step_count = 0

    while not done:
        pos = obs[:4]
        vel = obs[4:]
        error = env.target - pos

        current_time = step_count * env.dt

        # --- Controller update ---
        if step_count % steps_per_ctrl == 0:
            # PID computes control in WORLD frame
            noisy_error = error * (1 + np.random.randn(4) * noise_std)
            theta = pos[3]
            world_cmd = pid.compute(noisy_error)
            body_cmd = pid.rotate(world_cmd, theta)
            if verbose :
                print("body_cmd",body_cmd)

            # Packet loss check
            if np.random.rand() > p_loss:
                additional_delay = (
                    np.random.randint(1, max_delay_steps + 1)
                    if max_delay_steps > 0
                    else 0
                )
                apply_time = current_time + additional_delay * env.dt
                pending_actions.append((apply_time, body_cmd.copy()))

        # --- Apply action according to delay queue ---
        # Find any actions whose apply_time <= current_time
        if pending_actions and pending_actions[0][0] <= current_time:
            _, last_action = pending_actions.pop(0)

        # Step simulation with wind
        obs, reward, terminated, truncated, info = env.step(
            last_action, wind_std=wind_std
        )
        states.append(obs.copy())
        step_count += 1
        done = terminated or truncated

    env.close()

    states = np.array(states)
    x, y, z, theta = states[:, 0], states[:, 1], states[:, 2], states[:, 3]
    target = env.target

    # --- Plot results ---
    if plot:
        plot_trajectory_3d(x, y, z, theta, target)
        plot_xy_z(x, y, z, theta, target, env.dt)

    return states, target
