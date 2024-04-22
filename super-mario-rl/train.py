from setup import setup_env
from constants import CHECKPOINT_DIR, LOG_DIR
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import os
from datetime import datetime


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
                self.save_path, f"ppo_model_{self.num_timesteps}_steps"
            )
            self.model.save(model_path)
        if self.num_timesteps % 20000 == 0:
            print(f"timesteps: {self.num_timesteps} ({datetime.now()})")

        return True


if __name__ == "__main__":
    # Setup game
    env = setup_env("SuperMarioBros-1-2-v0")

    state = env.reset()
    # state, reward, done, info = env.step([5])

    # plt.figure(figsize=(20, 16))
    # for idx in range(state.shape[3]):
    #     plt.subplot(1, 4, idx + 1)
    #     plt.imshow(state[0][:, :, idx])
    # plt.show()

    callback = TrainAndLoggingCallback(check_freq=25000, save_path=CHECKPOINT_DIR)

    # Initial training
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        learning_rate=0.00001,
        device="mps",
        ent_coef=0.01,
    )

    # Continue
    # model_name = "original-reward/ppo_model_4800000_steps"
    # model = PPO.load(path=f"{CHECKPOINT_DIR}{model_name}", device="mps")
    # model.set_env(env)

    model.learn(
        total_timesteps=20000000,
        callback=callback,
        reset_num_timesteps=False,
        tb_log_name="score-1-2",
    )
