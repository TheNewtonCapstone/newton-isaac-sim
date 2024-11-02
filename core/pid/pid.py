import torch


class PidController(object):
    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        min_output: float,
        max_output: float,
        max_integral: float,
        setpoint: float,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.min_output = min_output
        self.max_output = max_output
        self.max_integral = max_integral
        self.integral = 0
        self.prev_error = 0

    def reset(self):
        self.integral = 0
        self.prev_error = 0

    def predict(self, current: float, dt: float) -> torch.Tensor:
        # self.integral += error * dt
        # self.integral = np.clip(self.integral, -self.max_integral, self.max_integral)
        # derivative = (error - self.prev_error) / dt
        proportional = self.kp * (self.setpoint - current)
        pid = proportional
        output = torch.tensor([pid, pid], dtype=torch.float32)
        output = torch.clamp(output, self.min_output, self.max_output)

        # + self.ki * self.integral + self.kd * derivative)
        # output = np.clip(output, self.min_output, self.max_output)
        # self.prev_error = error
        return output
