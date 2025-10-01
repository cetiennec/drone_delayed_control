import numpy as np
import matplotlib.pyplot as plt
from PID_run import run_noisy_pid_simulation
from PIDCon import PIDController
from DroneEnv import DroneLandingEnv
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def evaluate_controller(
    p_loss, max_delay_steps, cdt, wind_type, noise_std, n_runs=15, tol=0.3
):
    """
    Run multiple simulations and evaluate stability.
    Returns average MAE, average final distance, stability fraction.
    """
    maes, final_dists, traj_lengths, weakly_stables, strongly_stables = (
        [],
        [],
        [],
        [],
        [],
    )

    for _ in range(n_runs):
        sim_dt = 0.1  # simulation step
        env = DroneLandingEnv(render_mode=None, dt=sim_dt, consider_wind=wind_type)

        pid = PIDController(Kp=[1.0,1.0,0.0, 2.0], Ki=[0.1,0.1,0.0,0.05], Kd=[0.0,0.0,0.0,0.0], dt=cdt, integral_limit=0.5, z_fixed_speed= -0.2, yaw_3_steps =True, finer_gains=True)

        states, target = run_noisy_pid_simulation(
            env=env,
            pid=pid,
            init_mode="random",
            plot=False,
            max_delay_steps=max_delay_steps,
            p_loss=p_loss,
            wind_type=wind_type,
            noise_std=noise_std,
        )

        x, y, z, theta = states[:, 0], states[:, 1], states[:, 2], states[:, 3]

        # Compute errors
        errors = np.linalg.norm(states[:, :3] - target[:3], axis=1)  # distance in XYZ
        mae = np.mean(errors)

        # Final horizontal error
        dx = abs(x[-1] - target[0])
        dy = abs(y[-1] - target[1])

        weak_stability = (dx < tol) and (dy < tol)

        drone_size = 0.15  # as in your plot function
        target_box_size = tol  # assuming tol is half the side length of the target box

        # Final position and orientation
        cx, cy = x[-1], y[-1]
        yaw = theta[-1]

        # Define drone corners in body frame
        corners = np.array(
            [
                [-drone_size, -drone_size],
                [drone_size, -drone_size],
                [drone_size, drone_size],
                [-drone_size, drone_size],
            ]
        )

        # Rotation matrix
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s], [s, c]])

        # Rotate and translate corners to world frame
        rotated = (R @ corners.T).T + np.array([cx, cy])

        # Target box boundaries
        target_min = target[:2] - target_box_size
        target_max = target[:2] + target_box_size

        # Check if all corners are inside the target box
        inside = (
            (rotated[:, 0] >= target_min[0])
            & (rotated[:, 0] <= target_max[0])
            & (rotated[:, 1] >= target_min[1])
            & (rotated[:, 1] <= target_max[1])
        )
        strong_stability = np.all(inside)

        # Store
        maes.append(mae)
        final_dists.append(errors[-1])
        traj_lengths.append(len(states))

        strongly_stables.append(strong_stability)
        weakly_stables.append(weak_stability)

    return {
        "mae": np.mean(maes),
        "final_dist": np.mean(final_dists),
        "traj_len": np.mean(traj_lengths),
        "weakly_stable_fraction": np.mean(weakly_stables),
        "strongly_stable_fraction": np.mean(strongly_stables),
    }


def sweep_parameters(wind_type, noise_std):
    """
    Sweep p_loss, max_delay_steps, controller_dt and collect stability data.
    """
    p_loss_vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    delay_vals = [0, 1, 2, 3, 4, 5]
    ctrl_dt_vals = [0.2, 0.4, 0.8, 1.0, 1.2]

    results = []
    for p in p_loss_vals:
        for d in delay_vals:
            for cdt in ctrl_dt_vals:
                # if d/10.0 <=cdt : # just check that we cant have more variable delay than frequency
                res = evaluate_controller(p, d, cdt, wind_type, noise_std)
                results.append(
                    (
                        p,
                        d,
                        cdt,
                        res["weakly_stable_fraction"],
                        res["strongly_stable_fraction"],
                    )
                )

    return results


def plot_stability_volume(results, title=None, ax=None, weak_stability=True):
    """
    Plot stable region in 3D space (p_loss, delay, controller_dt).
    """
    p_vals, d_vals, cdt_vals, weakly_stable_fracs, strongly_stable_fracs = zip(*results)

    if weak_stability:
        stable_fracs = weakly_stable_fracs
    else:
        stable_fracs = strongly_stable_fracs

    sc = ax.scatter(
        p_vals, d_vals, cdt_vals, c=stable_fracs, cmap="viridis", s=60, edgecolor="k"
    )

    ax.set_xlabel("Packet loss probability")
    ax.set_ylabel("Max delay steps")
    ax.set_zlabel("Controller dt [s]")

    if title is None:
        ax.set_title("Stability volume")
    else:
        ax.set_title(title)

    return sc


def show_stability_volume(results, title=None, ax=None, fig=None, weak_stability=True):
    """
    Plot stable region in 3D space (p_loss, delay, controller_dt).
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

    p_vals, d_vals, cdt_vals, weakly_stable_fracs, strongly_stable_fracs = zip(*results)

    if weak_stability:
        stable_fracs = weakly_stable_fracs
    else:
        stable_fracs = strongly_stable_fracs

    sc = ax.scatter(
        p_vals, d_vals, cdt_vals, c=stable_fracs, cmap="viridis", s=100, edgecolor="k"
    )

    ax.set_xlabel("Packet dropout probability")
    ax.set_ylabel("Max delay steps")
    ax.set_zlabel("Controller dt [s]")
    if title is None:
        ax.set_title("Stability volume (fraction of stable runs)")
    else:
        ax.set_title(title)
    if weak_stability:
        fig.colorbar(sc, label="Weakly stable fraction (0–1)")
    else:
        fig.colorbar(sc, label="Strongly stable fraction (0–1)")
    plt.savefig("images/stability.png")
    plt.show()


def plot_stability_volume_3D(results, title=None, threshold=0.9, fig=None):
    """
    Plot stable region in 3D space (p_loss, delay, controller_dt),
    filling the convex hull of points where stable fraction > threshold.
    """
    if fig is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

    p_vals, d_vals, cdt_vals, stable_fracs, _ = zip(*results)

    p_vals = np.array(p_vals)
    d_vals = np.array(d_vals)
    cdt_vals = np.array(cdt_vals)
    stable_fracs = np.array(stable_fracs)

    # Keep only stable points
    mask = stable_fracs >= threshold
    p_stable = p_vals[mask]
    d_stable = d_vals[mask]
    cdt_stable = cdt_vals[mask]

    # Scatter stable points
    ax.scatter(
        p_stable,
        d_stable,
        cdt_stable,
        c="green",
        s=60,
        edgecolor="k",
        alpha=0.8,
        label=f"Stable > {threshold}",
    )

    # Build convex hull (only if enough points)
    if len(p_stable) >= 4:
        pts = np.vstack([p_stable, d_stable, cdt_stable]).T
        hull = ConvexHull(pts)

        # Add each face as a polygon
        for simplex in hull.simplices:
            poly3d = [pts[simplex]]
            ax.add_collection3d(Poly3DCollection(poly3d, alpha=0.2, facecolor="green"))

    ax.set_xlabel("Packet dropout probability")
    ax.set_ylabel("Max delay steps")
    ax.set_zlabel("Controller dt [s]")

    if title is None:
        ax.set_title(f"Stable region (fraction > {threshold})")
    else:
        ax.set_title(title)

    ax.legend()
    plt.show()
