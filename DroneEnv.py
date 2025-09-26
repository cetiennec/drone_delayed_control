import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.patches as patches
import plotly.graph_objects as go
import plotly.graph_objects as go
import matplotlib.pyplot as plt



class DroneLandingEnv(gym.Env):
    """
    Drone landing environment:
    - State: [x, y, z, vx, vy, vz]
    - Action: [ax, ay, az] (acceleration commands)
    - Dynamics: simple double integrator
    - Goal: reach target position
    """

    metadata = {"render_modes": ["human", "3d"], "render_fps": 30}

    def __init__(self, render_mode=None, max_steps=200, dt=0.1):
        super().__init__()

        # Drone state: position + velocity
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )

        # Action: acceleration in 3D (bounded)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Target position
        self.target = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Simulation parameters
        self.dt = dt
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.step_count = 0
        self.auto_landing = False

        # Internal state
        self.state = None

    def reset(self, seed=None, options=None, init_mode='fixed', init_pos = None):
        super().reset(seed=seed)

        if init_pos is not None:
            pos = np.array(init_pos, dtype=np.float32)
        elif init_mode == 'random':
            # Uniform sampling within bounds
            x = self.np_random.uniform(-2.0, 2.0)  # X bounds
            y = self.np_random.uniform(-2.0, 2.0)  # Y bounds
            z = self.np_random.uniform(1.0, 5.0)  # Z bounds (above ground)
            theta = self.np_random.uniform(-np.pi, np.pi)  # Yaw
            pos = np.array([x, y, z, theta], dtype=np.float32)
        elif init_mode == 'fixed':
            # Fixed deterministic start
            pos = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)


        vel = np.zeros(4, dtype=np.float32)  # velocities [vx, vy, vz, omega]
        self.state = np.concatenate([pos, vel])
        self.step_count = 0

        return self.state, {}

    def step(self, action, wind_std=[0.0, 0.0, 0.0, 0.0]):
        """
        Step the drone dynamics with optional wind disturbance.
        Actions are given in the drone body frame (vx, vy, vz, yaw_rate).
        State is in world frame (x, y, z, theta).
        """
        # Clip action
        action = np.clip(action, self.action_space.low, self.action_space.high)

        pos = self.state[:4]  # [x, y, z, theta]
        vel = self.state[4:]  # [vx, vy, vz, omega]

        # Stop if drone hits the ground
        if pos[2] <= 0.4:
            self.auto_landing = True
        if self.auto_landing:
            action = np.array([0.0, 0.0, -1.0, 0.0])

        # --- Transform action from body frame to world frame ---
        theta = pos[3]
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
        action_world = np.zeros_like(action)
        action_world[:3] = R @ action[:3]  # rotate (x,y,z) part
        action_world[3] = action[3]  # yaw rate stays the same

        # --- Velocity update with wind ---
        wind_effect = np.random.randn(4) * wind_std
        new_vel = action_world * (1 + wind_effect)

        # --- Position update ---
        new_pos = pos.copy()
        new_pos[:3] += new_vel[:3] * self.dt
        new_pos[3] += new_vel[3] * self.dt  # update yaw

        # --- Add integrated wind push ---
        wind_push = wind_effect * self.dt
        new_pos += wind_push

        # --- Save new state ---
        self.state = np.concatenate([new_pos, new_vel])
        self.step_count += 1

        # Distance to target
        dist = np.linalg.norm(new_pos[:3] - self.target[:3])

        # Reward: -distance - small penalty on action
        reward = -dist - 0.01 * np.linalg.norm(action)

        # Termination conditions
        terminated = dist < 0.1
        truncated = self.step_count >= self.max_steps
        if new_pos[2] <= 0.1:
            terminated = True
            truncated = True

        return self.state, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "human":
            pos = self.state[:3]
            print(f"Step {self.step_count}: pos={pos}, target={self.target}")
        elif self.render_mode == "3d":
            # Could plug the Plotly visualization here later
            pass

    def close(self):
        pass


def drone_shape(x, y, z, theta=0.0, size=0.15):
    # Rotation matrix in XY plane
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])

    arms = []
    props = []

    # Define arms in body frame (each as 2 endpoints)
    arms_body = [
        ([-size, size], [0, 0]),  # X-axis arm
        ([0, 0], [-size, size])   # Y-axis arm
    ]

    for xb, yb in arms_body:
        # Rotate each endpoint
        pts = np.vstack([xb, yb]).T  # shape (2,2)
        rotated = (R @ pts.T).T      # shape (2,2)
        arms.append((
            [x + rotated[0,0], x + rotated[1,0]],  # X coords
            [y + rotated[0,1], y + rotated[1,1]],  # Y coords
            [z, z]                                 # Z coords (same for both ends)
        ))

    # Propeller positions
    props_body = [
        [size, 0],
        [-size, 0],
        [0, size],
        [0, -size],
    ]
    for xb, yb in props_body:
        xy = R @ np.array([xb, yb])
        props.append((x + xy[0], y + xy[1], z))

    return arms, props



def plot_trajectory_3d(x, y, z, theta, target, show_drone=True, box_size=0.3):
    fig = go.Figure()

    # Trajectory
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines",
            line=dict(color="blue", width=3),
            name="Trajectory",
        )
    )

    # Target
    fig.add_trace(
        go.Scatter3d(
            x=[target[0]],
            y=[target[1]],
            z=[target[2]],
            mode="markers",
            marker=dict(color="red", size=4, symbol="x"),
            name="Target",
        )
    )

    # Drone final position
    fig.add_trace(
        go.Scatter3d(
            x=[x[-1]],
            y=[y[-1]],
            z=[z[-1]],
            mode="markers",
            marker=dict(color="green", size=3, symbol="x"),
            name="Drone final position",
        )
    )

    # Drone at final position
    if show_drone:
        arms, props = drone_shape(x[-1], y[-1], z[-1], theta[-1])
        for arm in arms:
            fig.add_trace(
                go.Scatter3d(
                    x=arm[0],
                    y=arm[1],
                    z=arm[2],
                    mode="lines",
                    line=dict(color="black", width=5),
                    showlegend=False,
                )
            )
        for px, py, pz in props:
            fig.add_trace(
                go.Scatter3d(
                    x=[px],
                    y=[py],
                    z=[pz],
                    mode="markers",
                    marker=dict(size=5, color="gray", symbol="circle"),
                    showlegend=False,
                )
            )

    # --- Draw orange square on the ground ---
    s = box_size
    # corners in XY plane at z=0
    corners = np.array(
        [
            [target[0] - s, target[1] - s, 0],
            [target[0] + s, target[1] - s, 0],
            [target[0] + s, target[1] + s, 0],
            [target[0] - s, target[1] + s, 0],
            [target[0] - s, target[1] - s, 0],  # close the loop
        ]
    )
    fig.add_trace(
        go.Scatter3d(
            x=corners[:, 0],
            y=corners[:, 1],
            z=corners[:, 2],
            mode="lines",
            line=dict(color="orange", width=4),
            name="Target area",
        )
    )

    fig.update_layout(
        title="Drone Landing Trajectory",
        scene=dict(
            xaxis_title="X [m]",
            yaxis_title="Y [m]",
            zaxis_title="Z [m]",
            aspectmode="data",
        ),
        width=800,
        height=600,
    )
    fig.show()


def plot_xy_z(x, y, z, theta, target, dt, box_size=0.3, arrow_every=5):
    """
    Plot XY trajectory (large) on left, Z and Theta vs time stacked on right.
    """
    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1], hspace=0.5, wspace=0.2)

    # ---------------- XY view (large, spans both rows left) ----------------
    ax_xy = fig.add_subplot(gs[:, 0])
    ax_xy.plot(x, y, "b-", label="Trajectory", linewidth=1.5)

    # Target area
    lower_left = (target[0] - box_size, target[1] - box_size)
    rect = patches.Rectangle(lower_left, 2*box_size, 2*box_size,
                             linewidth=2, edgecolor='orange', facecolor='none', label="Target box")
    ax_xy.add_patch(rect)

    # Drone final box rotated
    drone_size = 0.15
    cx, cy = x[-1], y[-1]
    yaw = theta[-1]
    corners = np.array([[-drone_size, -drone_size],
                        [ drone_size, -drone_size],
                        [ drone_size,  drone_size],
                        [-drone_size,  drone_size]])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    rotated = (R @ corners.T).T + np.array([cx, cy])
    poly = patches.Polygon(rotated, closed=True, linewidth=2, edgecolor='black', facecolor='none', label="Drone")
    ax_xy.add_patch(poly)

    # Markers
    ax_xy.scatter(target[0], target[1], color="r", marker="x", s=90, label="Target")
    ax_xy.scatter(cx, cy, color="green", marker="x", s=90, label="Final position")

    # Orientation arrows
    arrow_len = 0.1
    for i in range(0, len(x), arrow_every):
        dx = arrow_len * np.cos(theta[i])
        dy = arrow_len * np.sin(theta[i])
        ax_xy.arrow(x[i], y[i], dx, dy, head_width=0.05, head_length=0.06,
                    fc='m', ec='m', alpha=0.9)
    ax_xy.plot([], [], color='m', marker=r'$\rightarrow$', linestyle='None', label="Orientation")

    ax_xy.set_xlabel("X [m]")
    ax_xy.set_ylabel("Y [m]")
    ax_xy.set_title("Top-down view (X/Y)")
    ax_xy.legend()
    ax_xy.grid(True)
    ax_xy.set_aspect('equal', adjustable='box')

    # ---------------- Z vs time (top right) ----------------
    ax_z = fig.add_subplot(gs[0, 1])
    t = np.arange(len(z)) * dt
    ax_z.plot(t, z, "b-", label="Drone altitude")
    ax_z.axhline(target[2], color="r", linestyle="--", label="Target Z")
    ax_z.set_xlabel("Time [s]")
    ax_z.set_ylabel("Z [m]")
    ax_z.set_title("Altitude vs time")
    ax_z.legend()
    ax_z.grid(True)

    # ---------------- Theta vs time (bottom right) ----------------
    ax_theta = fig.add_subplot(gs[1, 1])
    ax_theta.plot(t, theta, "b-", label="Drone orientation")
    ax_theta.set_xlabel("Time [s]")
    ax_theta.set_ylabel("$\theta$ [rad]")
    ax_theta.set_title("Orientation vs time")
    ax_theta.legend()
    ax_theta.grid(True)

    # plt.tight_layout()
    plt.show()

