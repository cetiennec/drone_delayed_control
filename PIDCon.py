import numpy as np


# --- PID Controller Class with optional fixed z descent ---
class PIDController:
    def __init__(self, Kp, Ki, Kd, dt, integral_limit=None, z_fixed_speed=None, yaw_2_steps = False, finer_gains = False):
        """
        PID controller for 3D drone control
        Kp, Ki, Kd: gains for proportional, integral, derivative
        dt: timestep
        integral_limit: optional max absolute value for integral term
        z_fixed_speed: optional constant speed for Z axis (overrides PID on Z)
        """
        self.Kp = np.array(Kp)
        self.Ki = np.array(Ki)
        self.Kd = np.array(Kd)
        self.dt = dt
        self.integral = np.zeros(4)
        self.prev_error = np.zeros(4)
        self.integral_limit = integral_limit
        self.z_fixed_speed = z_fixed_speed

        self.yaw_condition = yaw_2_steps
        self.finer_gains = finer_gains

    def update_dt(self,dt):
        self.dt = dt

    def reset(self):
        self.integral[:] = 0
        self.prev_error[:] = 0

    def compute(self, error):
        """
        Compute PID output given current error (3D vector: [x, y, theta])
        """
        # Integral term
        self.integral += error * self.dt
        if self.integral_limit is not None:
            self.integral = np.clip(
                self.integral, -self.integral_limit, self.integral_limit
            )

        # Derivative term
        derivative = (error - self.prev_error) / self.dt

        # PID output
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        # Option: control theta only if x,y error < 0.3m
        if abs(error[0]) >= 0.3 or abs(error[1]) >= 0.3:
            if self.yaw_condition:
                output[3] = 0.0  # no yaw movement
            if self.finer_gains :
                output = output/2

        # Override Z component if fixed speed is enabled
        if self.z_fixed_speed is not None:
            output[2] = self.z_fixed_speed

        # Save previous error
        self.prev_error = error.copy()

        return output

    def rotate(self, command, theta):

        c, s = np.cos(theta), np.sin(theta)
        R = np.array([
            [c, s, 0],
            [-s, c, 0],
            [0, 0, 1]
        ])  # world-to-body rotation (transpose of body-to-world)

        body_output = np.zeros_like(command)
        body_output[:3] = R @ command[:3]   # rotate linear part
        body_output[3] = command[3]         # yaw rate unchanged

        return body_output


    def inverse_rotate(self, command, theta):

        c, s = np.cos(theta), np.sin(theta)
        R = np.array([
            [c, s, 0],
            [-s, c, 0],
            [0, 0, 1]
        ])  # world-to-body rotation (transpose of body-to-world)

        body_output = np.zeros_like(command)
        body_output[:3] = R @ command[:3]   # rotate linear part
        body_output[3] = command[3]         # yaw rate unchanged

        return body_output
