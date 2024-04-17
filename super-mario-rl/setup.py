import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym.wrappers import GrayScaleObservation, ResizeObservation
from gym import Env
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv, VecMonitor
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.utils import set_random_seed

NUM_CPU_CORE = 4


class MarioEnv(Env):
    def __init__(self, env_id):
        super(MarioEnv, self).__init__()

        self.gym_mario_env = gym_super_mario_bros.make(env_id)
        self.action_space = self.gym_mario_env.action_space
        self.observation_space = self.gym_mario_env.observation_space

    def step(self, action):
        return self.gym_mario_env.step(action)

    def reset(self):
        return self.gym_mario_env.reset()

    def render(self, mode="human"):
        return self.gym_mario_env.render(mode)

    def close(self):
        return self.gym_mario_env.close()


def make_env(env_id, rank, seed=0):
    def _init():
        env = MarioEnv(env_id)
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
