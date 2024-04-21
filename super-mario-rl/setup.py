import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym.wrappers import GrayScaleObservation, ResizeObservation
from gym import Wrapper
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv, VecMonitor
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.utils import set_random_seed

NUM_CPU_CORE = 8


class MarioEnv(Wrapper):
    def __init__(self, env):
        super(MarioEnv, self).__init__(env)

        self.last_info = None

    def step(self, action):
        def get_status_reward(old_status, new_status):
            status_order = ["small", "tall", "fireball"]
            old_index = status_order.index(old_status)
            new_index = status_order.index(new_status)
            if new_index > old_index:
                return 1
            elif new_index < old_index:
                return -1
            else:
                return 0

        state, originalReward, done, info = self.env.step(action)

        if self.last_info is None:
            reward = 0
        else:
            moving_distance = info["x_pos"] - self.last_info["x_pos"]
            time = info["time"] - self.last_info["time"]
            death = (
                -30
                if (info["life"] == 255 or self.last_info["life"] > info["life"])
                else 0
            )
            score = (info["score"] - self.last_info["score"]) / 100
            flag = 200 if info["flag_get"] is True else 0
            status = get_status_reward(self.last_info["status"], info["status"])
            reward = (
                moving_distance * 0 + time + score * 10 + status * 100 + flag + death
            )

        if done:
            self.last_info = None
        else:
            self.last_info = info

        return state, originalReward, done, info


def make_env(env_id, rank, seed=0):
    def _init():
        env = MarioEnv(gym_super_mario_bros.make(env_id))
        env = JoypadSpace(env, COMPLEX_MOVEMENT)
        env = MaxAndSkipEnv(env, 4)
        env = ResizeObservation(env, shape=84)
        env = GrayScaleObservation(env, keep_dim=True)
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


def setup_env(env_id, num_process=NUM_CPU_CORE):
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_process)])
    env = VecMonitor(env)
    env = VecFrameStack(env, 4, channels_order="last")

    return env
