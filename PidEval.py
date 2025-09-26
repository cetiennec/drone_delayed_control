import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PID_run import run_noisy_pid_simulation
from PIDCon import PIDController
from DroneEnv import DroneLandingEnv


def evaluate_controller( p_loss, max_delay_steps, cdt, wind_std, noise_std, n_runs=15, tol=0.3):
    """
    Run multiple simulations and evaluate stability.
    Returns average MAE, average final distance, stability fraction.
    """
    maes, final_dists, traj_lengths, stables = [], [], [], []

    for _ in range(n_runs):
        sim_dt = 0.1  # simulation step
        env = DroneLandingEnv(render_mode=None, dt=sim_dt)

        pid = PIDController(
            Kp=[2.0, 2.0, 2.0, 2.0],
            Ki=[0.1, 0.1, 0.0, 0.2],
            Kd=[0.0, 0.0, 0.0, 0.0],
            dt=cdt,  # initial value, will be overwritten in sweep
            integral_limit=0.5,
            z_fixed_speed=-0.2,
            yaw_2_steps=True,
            finer_gains=True
        )

        states, target = run_noisy_pid_simulation(
            env=env,
            pid=pid,
            init_mode='random',
            plot=False,
            max_delay_steps=max_delay_steps,
            p_loss=p_loss,
            wind_std=wind_std,
            noise_std=noise_std
        )

        x, y, z, theta = states[:,0], states[:,1], states[:,2], states[:,3]

        # Compute errors
        errors = np.linalg.norm(states[:,:3] - target[:3], axis=1)  # distance in XYZ
        mae = np.mean(errors)

        # Final horizontal error
        dx = abs(x[-1] - target[0])
        dy = abs(y[-1] - target[1])
        stable = (dx < tol) and (dy < tol)

        # Store
        maes.append(mae)
        final_dists.append(errors[-1])
        traj_lengths.append(len(states))
        stables.append(stable)

    return {
        "mae": np.mean(maes),
        "final_dist": np.mean(final_dists),
        "traj_len": np.mean(traj_lengths),
        "stable_fraction": np.mean(stables)
    }

def sweep_parameters(wind_std, noise_std):
    """
    Sweep p_loss, max_delay_steps, controller_dt and collect stability data.
    """
    p_loss_vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    delay_vals = [0, 1, 2, 3, 4, 5]
    ctrl_dt_vals = [0.1, 0.2, 0.4, 0.8, 1.0, 1.2, 1.6]

    results = []
    for p in p_loss_vals:
        for d in delay_vals:
            for cdt in ctrl_dt_vals:
                res = evaluate_controller( p, d, cdt, wind_std, noise_std)
                results.append((p, d, cdt, res["stable_fraction"], res["mae"]))


    return results

def plot_stability_volume(results, title = None):
    """
    Plot stable region in 3D space (p_loss, delay, controller_dt).
    """
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection="3d")

    p_vals, d_vals, cdt_vals, stable_fracs, _ = zip(*results)

    sc = ax.scatter(
        p_vals, d_vals, cdt_vals,
        c=stable_fracs, cmap="viridis", s=100, edgecolor="k"
    )

    ax.set_xlabel("Packet loss probability")
    ax.set_ylabel("Max delay steps")
    ax.set_zlabel("Controller dt [s]")
    if title is None:
        ax.set_title("Stability volume (fraction of stable runs)")
    else:
        ax.set_title(title)
    fig.colorbar(sc, label="Stable fraction (0–1)")
    plt.show()

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_stability_volume_3D(results, title=None, threshold=0.9):
    """
    Plot stable region in 3D space (p_loss, delay, controller_dt),
    filling the convex hull of points where stable fraction > threshold.
    """
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
        p_stable, d_stable, cdt_stable,
        c="green", s=60, edgecolor="k", alpha=0.8, label=f"Stable > {threshold}"
    )

    # Build convex hull (only if enough points)
    if len(p_stable) >= 4:
        pts = np.vstack([p_stable, d_stable, cdt_stable]).T
        hull = ConvexHull(pts)

        # Add each face as a polygon
        for simplex in hull.simplices:
            poly3d = [pts[simplex]]
            ax.add_collection3d(
                Poly3DCollection(poly3d, alpha=0.2, facecolor="green")
            )

    ax.set_xlabel("Packet loss probability")
    ax.set_ylabel("Max delay steps")
    ax.set_zlabel("Controller dt [s]")

    if title is None:
        ax.set_title(f"Stable region (fraction > {threshold})")
    else:
        ax.set_title(title)

    ax.legend()
    plt.show()
