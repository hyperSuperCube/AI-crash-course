from stable_baselines3 import SAC
from cartpole_env import CartPoleEnv
import numpy as np

env = CartPoleEnv()
model = SAC.load("cartpole_sac")

obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break

print("Final θ:", obs[2], "target π =", np.pi)