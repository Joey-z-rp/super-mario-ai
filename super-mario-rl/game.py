import pygame
from setup import setup_env
import gym_super_mario_bros
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from constants import CHECKPOINT_DIR
from stable_baselines3 import PPO

pygame.init()

FRAME_WIDTH = 256
FRAME_HEIGHT = 240
SCALING_FACTOR = 3
WINDOW_WIDTH = FRAME_WIDTH * 2 * SCALING_FACTOR
WINDOW_HEIGHT = FRAME_HEIGHT * SCALING_FACTOR

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Super Mario Bros")

# Create a clock object to control frame rate
clock = pygame.time.Clock()


def transform_surface(surface):
    transformed = pygame.transform.rotate(surface, -90)
    transformed = pygame.transform.flip(transformed, True, False)
    transformed = pygame.transform.scale(
        transformed, (FRAME_WIDTH * SCALING_FACTOR, FRAME_HEIGHT * SCALING_FACTOR)
    )
    return transformed


def map_keyboard_input(key):
    if key == pygame.K_w:
        return 1
    elif key == pygame.K_s:
        return 2
    elif key == pygame.K_a:
        return 3
    elif key == pygame.K_d:
        return 4
    elif key == pygame.K_SPACE:
        return 5
    elif key == pygame.K_m:
        return 6
    else:
        return 0


if __name__ == "__main__":
    human_env = MaxAndSkipEnv(gym_super_mario_bros.make("SuperMarioBros-1-1-v0"), 4)
    ai_env = setup_env("SuperMarioBros-1-1-v0", 1, True)
    model_name = "original-reward/ppo_model_4800000_steps"
    model = PPO.load(f"{CHECKPOINT_DIR}{model_name}")
    human_env.reset()
    ai_state = ai_env.reset()

    running = True
    while running:
        # Event handling
        human_action = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                human_action = map_keyboard_input(event.key)

        human_frame, reward, done, info = human_env.step([human_action])
        ai_action, _state = model.predict(ai_state)
        ai_state, _reward, _done, _info = ai_env.step(ai_action)
        ai_frame = ai_env.render()

        human_surface = pygame.surfarray.make_surface(human_frame)
        ai_surface = pygame.surfarray.make_surface(ai_frame)
        screen.blit(
            transform_surface(human_surface),
            (0, 0),
        )
        screen.blit(
            transform_surface(ai_surface),
            (WINDOW_WIDTH // 2, 0),
        )
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(20)

    pygame.quit()
