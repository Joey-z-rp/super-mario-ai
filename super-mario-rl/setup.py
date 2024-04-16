from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym.wrappers import GrayScaleObservation, ResizeObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecMonitor
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv


def setup_env(init_env):
    env = JoypadSpace(init_env, COMPLEX_MOVEMENT)
    env = MaxAndSkipEnv(env, 4)
    env = ResizeObservation(env, shape=84)
    env = GrayScaleObservation(env, keep_dim=True)
    env = DummyVecEnv([lambda: env])
    env = VecMonitor(env)
    env = VecFrameStack(env, 4, channels_order="last")

    return env
