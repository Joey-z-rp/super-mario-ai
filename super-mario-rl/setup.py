import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym.wrappers import GrayScaleObservation, ResizeObservation
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv, VecMonitor
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.utils import set_random_seed

NUM_CPU_CORE = 4


def make_env(env_id, rank, seed=0):
    def _init():
        env = gym_super_mario_bros.make(env_id)
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
