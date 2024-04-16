import gym_super_mario_bros
from setup import setup_env
from constants import CHECKPOINT_DIR
from stable_baselines3 import PPO
import time

env = setup_env(gym_super_mario_bros.make("SuperMarioBros-v0"))

model_name = "ppo_model_50000_steps"
model = PPO.load(f"{CHECKPOINT_DIR}{model_name}")
state = env.reset()

while True:
    action, _state = model.predict(state)
    state, reward, done, info = env.step(action)
    time.sleep(0.02)
    env.render()
