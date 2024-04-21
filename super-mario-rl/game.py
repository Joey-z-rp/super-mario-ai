import pygame
from setup import setup_env
import gym_super_mario_bros

pygame.init()

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Super Mario Bros")

# Create a clock object to control frame rate
clock = pygame.time.Clock()

if __name__ == "__main__":
    human_env = gym_super_mario_bros.make("SuperMarioBrosRandomStages-v0")
    ai_env = setup_env("SuperMarioBrosRandomStages-v0", 1, True)
    human_env.reset()
    ai_env.reset()

    running = True
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        human_frame, reward, done, info = human_env.step([3])
        ai_env.step([2])
        ai_frame = ai_env.render()

        human_surface = pygame.surfarray.make_surface(human_frame)
        ai_surface = pygame.surfarray.make_surface(ai_frame)
        screen.blit(
            pygame.transform.flip(
                pygame.transform.rotate(human_surface, -90), True, False
            ),
            (0, 0),
        )
        screen.blit(
            pygame.transform.flip(
                pygame.transform.rotate(ai_surface, -90), True, False
            ),
            (WINDOW_WIDTH / 2, 0),
        )
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(60)

    pygame.quit()
