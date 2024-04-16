import gym_super_mario_bros
from setup import setup_env
from constants import CHECKPOINT_DIR, LOG_DIR
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import os

# Setup game
env = setup_env(gym_super_mario_bros.make("SuperMarioBros-v0"))

state = env.reset()
# state, reward, done, info = env.step([5])

# plt.figure(figsize=(20, 16))
# for idx in range(state.shape[3]):
#     plt.subplot(1, 4, idx + 1)
#     plt.imshow(state[0][:, :, idx])
# plt.show()


class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(
                self.save_path, "ppo_model_{}_steps".format(self.num_timesteps)
            )
            self.model.save(model_path)

        return True


callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

# Initial training
model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    learning_rate=0.00001,
    n_steps=2048,
    device="mps",
)

# Continue
# model_name = "ppo_model_4000_steps"
# model = PPO.load(path=f"{CHECKPOINT_DIR}{model_name}", device="mps")
# model.set_env(env)

model.learn(
    total_timesteps=50000,
    callback=callback,
    reset_num_timesteps=False,
)
