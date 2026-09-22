from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from cartpole_env import CartPoleEnv

env = CartPoleEnv()
eval_env = CartPoleEnv()

model = SAC(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    buffer_size=1_000_000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    verbose=1,
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/best_model",
    log_path="./logs",
    eval_freq=5000,
    n_eval_episodes=5,
)

model.learn(total_timesteps=500_000, callback=eval_callback)
model.save("cartpole_sac")