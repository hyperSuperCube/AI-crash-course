import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

class CartPoleEnv(gym.Env):
    def __init__(self):
        # self.model = mujoco.MjModel.from_xml_string(CARTPOLE_XML)
        self.model = mujoco.MjModel.from_xml_path("cartpole.xml")
        self.data = mujoco.MjData(self.model)

        # 观测：4 维状态
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        # 动作：1 维连续力
        self.action_space = spaces.Box(
            low=-20, high=20, shape=(1,), dtype=np.float32)

        self.max_steps = 1000
        self.step_count = 0

    def reset(self, seed=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # 随机初始状态（课程学习或随机化）
        self.data.qpos[:] = [
            np.random.uniform(-0.5, 0.5),           # 小车位置
            np.random.uniform(-0.5, 0.5) + np.pi,   # 摆杆角度（π 附近）
        ]
        self.data.qvel[:] = np.random.uniform(-0.1, 0.1, 2)
        self.step_count = 0

        return self._obs(), {}

    def step(self, action):
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        obs = self._obs()
        reward = self._reward(action)
        terminated = self._terminated()
        truncated = self.step_count >= self.max_steps

        return obs, reward, terminated, truncated, {}

    def _obs(self):
        return np.array([
            self.data.qpos[0], self.data.qvel[0],
            self.data.qpos[1], self.data.qvel[1],
        ], dtype=np.float32)

    def _reward(self, action):
        x, x_dot, theta, theta_dot = self._obs()
        upright = np.cos(theta - np.pi)
        pos_penalty = -0.1 * x**2
        vel_penalty = -0.01 * (x_dot**2 + theta_dot**2)
        ctrl_penalty = -0.001 * action[0]**2
        return upright + pos_penalty + vel_penalty + ctrl_penalty

    def _terminated(self):
        x, _, theta, _ = self._obs()
        if abs(x) > 10.0:
            return True
        # 摆杆偏离竖直太远？（可选）
        # if abs(theta - np.pi) > np.pi / 2:
        #     return True
        return False