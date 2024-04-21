from setup import setup_env
from constants import CHECKPOINT_DIR
from stable_baselines3 import PPO
import time

if __name__ == "__main__":
    env = setup_env("SuperMarioBrosRandomStages-v0", 1)

    model_name = "ppo_model_7800000_steps"
    model = PPO.load(f"{CHECKPOINT_DIR}{model_name}")
    state = env.reset()

    total_reward = 0

    while True:
        action, _state = model.predict(state)
        state, reward, done, info = env.step(action)
        time.sleep(0.02)
        total_reward += reward
        print(total_reward)
        if done:
            total_reward = 0
        env.render()
